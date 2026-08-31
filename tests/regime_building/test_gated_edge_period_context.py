"""Gated-edge callables receive the context of the period they price."""

from types import MappingProxyType

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarFloat,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

_BETA = 0.5
_AGES = AgeGrid(start=40, stop=50, step="5Y")
_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=2)
_N_SUBJECTS = 2


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    fallback: ScalarInt


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_source(*, x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


def _utility_target(x: ContinuousState) -> FloatND:
    return 10.0 + jnp.zeros_like(x)


def _utility_reference(x: ContinuousState) -> FloatND:
    return x


def _utility_fallback(x: ContinuousState) -> FloatND:
    return 100.0 + x


def _utility_target_by_x(x: ContinuousState) -> FloatND:
    return x


def _utility_fallback_prefers_low_x(x: ContinuousState) -> FloatND:
    return 2.0 - x


def _next_x_from_work(work: DiscreteAction) -> FloatND:
    return 2.0 * work


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _context_projection(*, period: ScalarInt, age: ScalarInt | ScalarFloat) -> FloatND:
    """Project to x=2 only in the edge's target period, otherwise x=0."""
    return jnp.where((period == 1) & (age == 45), 2.0, 0.0)


def _age_projection(age: ScalarInt | ScalarFloat) -> FloatND:
    """Project to x=2 only at the target age."""
    return jnp.where(age == 45, 2.0, 0.0)


def _context_gate_open(
    *,
    V_reference: FloatND,
    period: ScalarInt,
    age: ScalarInt | ScalarFloat,
) -> BoolND:
    """Open only when context and the gate-reference projection are right."""
    return (period == 1) & (age == 45) & (V_reference > 1.5)


def _context_gate_closed(
    *,
    V_reference: FloatND,
    period: ScalarInt,
    age: ScalarInt | ScalarFloat,
) -> BoolND:
    """Stay closed at the right context so the fallback projection is read."""
    return (period == 1) & (age == 45) & (V_reference < 0.0)


def _period_gate_open(*, V_reference: FloatND, period: ScalarInt) -> BoolND:
    """Open only in the target period after the age-only projection."""
    return (period == 1) & (V_reference > 1.5)


def _context_gate_open_without_reference(
    *,
    period: ScalarInt,
    age: ScalarInt | ScalarFloat,
) -> BoolND:
    """Open exactly in the target period."""
    return (period == 1) & (age == 45)


def _make_model(
    *,
    gate_open: bool,
    n_subjects: int | None,
    action_sensitive: bool = False,
    split_context: bool = False,
) -> Model:
    gate = (
        _context_gate_open_without_reference
        if action_sensitive
        else _period_gate_open
        if split_context
        else _context_gate_open
        if gate_open
        else _context_gate_closed
    )
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "target": ValueDependentTransition(
                        probability=MarkovTransition(_prob_one),
                        gate=gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={
                                        "x": (
                                            _identity_x
                                            if action_sensitive
                                            else _age_projection
                                            if split_context
                                            else _context_projection
                                        )
                                    },
                                )
                            )
                        },
                        gate_references={}
                        if action_sensitive
                        else {
                            "V_reference": ProjectedRegimeValue(
                                regime="reference",
                                projection={
                                    "x": (
                                        _age_projection
                                        if split_context
                                        else _context_projection
                                    )
                                },
                            )
                        },
                    )
                },
                active=lambda age: age < 45,
                states={"x": _X},
                state_transitions={
                    "x": (
                        _next_x_from_work if action_sensitive else fixed_transition("x")
                    )
                },
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _utility_source},
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 45,
                states={"x": _X},
                functions={
                    "utility": (
                        _utility_target_by_x if action_sensitive else _utility_target
                    )
                },
            ),
            "reference": Regime(
                transition=None,
                active=lambda age: age >= 45,
                states={"x": _X},
                functions={"utility": _utility_reference},
            ),
            "fallback": Regime(
                transition=None,
                active=lambda age: age >= 45,
                states={"x": _X},
                functions={
                    "utility": (
                        _utility_fallback_prefers_low_x
                        if action_sensitive
                        else _utility_fallback
                    )
                },
            ),
        },
        ages=_AGES,
        regime_id_class=RegimeId,
        n_subjects=n_subjects,
    )


def _initial_conditions(model: Model) -> MappingProxyType:
    return MappingProxyType(
        {
            "x": jnp.asarray([0.0, 2.0]),
            "age": jnp.asarray([40.0, 40.0]),
            "regime_id": jnp.full(
                _N_SUBJECTS,
                model.regime_names_to_ids["source"],
                dtype=jnp.int32,
            ),
        }
    )


def _solve_and_simulate(
    *,
    gate_open: bool,
    n_subjects: int | None,
    action_sensitive: bool = False,
    split_context: bool = False,
):
    model = _make_model(
        gate_open=gate_open,
        n_subjects=n_subjects,
        action_sensitive=action_sensitive,
        split_context=split_context,
    )
    params = {"discount_factor": _BETA}
    solution = model.solve(params=params, log_level="off")
    simulation = model.simulate(
        params=params,
        initial_conditions=_initial_conditions(model),
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=0,
    )
    return solution, simulation


def _leaf_paths(*, node: object, prefix: tuple[str, ...] = ()) -> set[str]:
    if isinstance(node, dict):
        return {
            path
            for key, value in node.items()
            for path in _leaf_paths(node=value, prefix=(*prefix, key))
        }
    return {"__".join(prefix)}


def test_period_and_age_are_engine_context_not_gated_edge_parameters():
    """Gated-edge context names do not create user parameters."""
    model = _make_model(gate_open=True, n_subjects=None)

    assert _leaf_paths(node=model.get_params_template()) == {
        "source__koopmans_aggregator__discount_factor"
    }


def test_gate_and_gate_reference_use_target_period_context():
    """Solve and simulation evaluate an open gate in the target period."""
    solution, simulation = _solve_and_simulate(gate_open=True, n_subjects=_N_SUBJECTS)

    aaae(
        solution[0]["source"],
        [5.0, 5.0],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(simulation.raw_results["target"][1].in_regime.all())


def test_period_only_gate_and_age_only_projections_receive_context():
    """Each context name is bound when edge callables declare it alone."""
    solution, simulation = _solve_and_simulate(
        gate_open=True,
        n_subjects=_N_SUBJECTS,
        split_context=True,
    )

    aaae(
        solution[0]["source"],
        [5.0, 5.0],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(simulation.raw_results["target"][1].in_regime.all())


def test_fallback_projection_uses_target_period_context():
    """A closed gate projects fallback states in the target period."""
    solution, simulation = _solve_and_simulate(gate_open=False, n_subjects=None)

    aaae(
        solution[0]["source"],
        [51.0, 51.0],
        decimal=DECIMAL_PRECISION,
    )
    fallback = simulation.raw_results["fallback"][1]
    assert bool(fallback.in_regime.all())
    aaae(
        fallback.states["x"],
        [2.0, 2.0],
        decimal=DECIMAL_PRECISION,
    )


def test_simulation_prices_source_actions_with_target_period_context():
    """Simulation's source action responds to the target-period gated fold."""
    solution, simulation = _solve_and_simulate(
        gate_open=True,
        n_subjects=None,
        action_sensitive=True,
    )

    aaae(
        solution[0]["source"],
        [1.0, 1.0],
        decimal=DECIMAL_PRECISION,
    )
    source_actions = simulation.raw_results["source"][0].actions["work"]
    assert bool((source_actions == Work.work).all())
