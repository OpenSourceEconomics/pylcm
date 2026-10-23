"""A dissolution gate is decided at the state a household realises.

A `~D_target` gate asks whether the target regime has a feasible action at the
state the household lands on. The solve tabulates that answer at the target's
grid nodes, but a landing between two nodes can have no feasible action even
though both nodes do, and a landing next to an empty node can have one. Forward
simulation therefore answers the question at the realised landing itself: the
household enters the target, the target's own decision evaluates every action
there, and a household left with no feasible action takes its gate's dissolution
route in that same period instead of aborting the simulation.

The model is the smallest one with that geometry. A source regime carries the
household's wage unchanged into a two-stakeholder `pair` regime whose wage grid
has nodes 1 and 3. The wife participates only when her pair payoff beats her
outside option of 0.5, and her pair payoff is a function of the wage alone, so
whether `pair` has a feasible action at a wage is a scalar inequality that the
reference below evaluates by enumerating both actions directly.
"""

import logging
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    Gate,
    IrregSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
    categorical,
)
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

_OUTSIDE_OPTION_F = 0.5


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    pair: ScalarInt
    alone_f: ScalarInt
    alone_m: ScalarInt


def _bowl(wage: FloatND | np.floating) -> FloatND | np.floating:
    """Feasible at both nodes (payoff 1), infeasible around wage 2."""
    return (wage - 2.0) ** 2


def _ramp(wage: FloatND | np.floating) -> FloatND | np.floating:
    """Infeasible at node 1 (payoff 0), feasible at node 3 (payoff 2)."""
    return wage - 1.0


_PAYOFFS = {"bowl": _bowl, "ramp": _ramp}


def _to_pair(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _u_source(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _keep_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _pair_f_bowl(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return _bowl(wage) + 0.0 * work


def _pair_f_ramp(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return _ramp(wage) + 0.0 * work


_PAIR_F = {"bowl": _pair_f_bowl, "ramp": _pair_f_ramp}


def _pair_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage + 0.0 * work


def _outside_f(wage: ContinuousState) -> FloatND:
    return _OUTSIDE_OPTION_F + 0.0 * wage


def _outside_m(wage: ContinuousState) -> FloatND:
    return -100.0 + 0.0 * wage


def _identity(wage: ContinuousState) -> ContinuousState:
    return wage


def _participation_f(*, Q_f: FloatND, V_alone_f: FloatND) -> BoolND:
    return Q_f > V_alone_f


def _no_dissolution(D_target: BoolND) -> BoolND:
    return ~D_target


def _double(wage: ContinuousState) -> ContinuousState:
    return 2.0 * wage


def _build_model(
    *,
    payoff_name: str,
    enable_jit: bool = True,
    execution_config: ExecutionConfig | None = None,
    fallback_grid: IrregSpacedGrid | None = None,
    fallback_projection: Callable[[ContinuousState], ContinuousState] = _identity,
) -> Model:
    wage_grid = IrregSpacedGrid(points=np.array([1.0, 3.0]))
    return Model(
        edges={
            "source": Transition(
                targets={"pair": 0, "alone_m": 0},
                law=ByAge(cases={0: {"pair": StochasticTransition(func=_to_pair)}}),
                gates={
                    "pair": Gate(
                        predicate=_no_dissolution,
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="f",
                                fallback=ProjectedRegimeValue(
                                    regime="alone_m",
                                    projection={"wage": fallback_projection},
                                ),
                            )
                        },
                    )
                },
            )
        },
        regimes={
            "source": Regime(
                states={"wage": wage_grid},
                state_transitions={"wage": _keep_wage},
                functions={"utility": _u_source},
            ),
            "pair": Regime(
                states={"wage": wage_grid},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _PAIR_F[payoff_name], "m": _pair_m}
                    )
                },
                constraints={
                    "participation_f": ValueDependentConstraint(
                        predicate=_participation_f,
                        references={
                            "V_alone_f": ProjectedRegimeValue(
                                regime="alone_f", projection={"wage": _identity}
                            )
                        },
                    )
                },
            ),
            "alone_f": Regime(
                states={"wage": wage_grid}, functions={"utility": _outside_f}
            ),
            "alone_m": Regime(
                states={"wage": wage_grid if fallback_grid is None else fallback_grid},
                functions={"utility": _outside_m},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "source"},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig()
        if execution_config is None
        else execution_config,
    )


_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "pair": {},
    "alone_f": {},
    "alone_m": {},
}


def _reference_regime(*, payoff_name: str, wage: np.floating) -> str:
    """Enumerate `pair`'s actions at one realised wage, in the working format.

    The wife's payoff does not depend on the action, so each action is feasible
    exactly when that payoff beats her outside option. The arithmetic runs in
    the dtype of `wage`, the format the model evaluates the constraint in.
    """
    payoff = _PAYOFFS[payoff_name]
    outside = wage.dtype.type(_OUTSIDE_OPTION_F)
    for _work in (0, 1):
        if payoff(wage) > outside:
            return "pair"
    return "alone_m"


def _working_dtype() -> type[np.floating]:
    return np.float64 if jnp.zeros(1).dtype == jnp.float64 else np.float32


def _landings(*, payoff_name: str) -> np.ndarray:
    """Landings at, next to, and between the nodes, and around the boundary.

    The participation boundary is where the wife's payoff equals 0.5. Each
    boundary point is represented in the working format and flanked by its
    three representable neighbours on either side, so the reference and the
    simulation are compared on both sides of a structural decision without any
    tolerance. The interior points come from a fixed seed.
    """
    dtype = _working_dtype()
    boundaries = (
        (2.0 - np.sqrt(0.5), 2.0 + np.sqrt(0.5)) if payoff_name == "bowl" else (1.5,)
    )
    points = [dtype(1.0), dtype(3.0), dtype(2.0)]
    for node in (dtype(1.0), dtype(3.0)):
        toward = dtype(2.0)
        neighbour = node
        for _ in range(3):
            neighbour = np.nextafter(neighbour, toward)
            points.append(neighbour)
    for boundary in boundaries:
        below = above = dtype(boundary)
        points.append(below)
        for _ in range(3):
            below = np.nextafter(below, dtype(-np.inf))
            above = np.nextafter(above, dtype(np.inf))
            points.extend((below, above))
    rng = np.random.default_rng(seed=588)
    points.extend(rng.uniform(1.0, 3.0, size=16).astype(dtype))
    return np.asarray(points, dtype=dtype)


def _simulate(*, payoff_name: str, wages: np.ndarray, enable_jit: bool = True):
    model = _build_model(payoff_name=payoff_name, enable_jit=enable_jit)
    solution = model.solve(params=_PARAMS, log_level="off")
    n_subjects = wages.shape[0]
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wage": jnp.asarray(wages),
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.full(
                n_subjects, model.regime_names_to_ids["source"], jnp.int32
            ),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    return solution, result.to_dataframe().query("period == 1").sort_values(
        "subject_id"
    )


def test_bowl_nodes_are_undissolved_while_the_middle_is_empty():
    """Both wage nodes have a feasible action, so the solved flag is clear there."""
    solution = _build_model(payoff_name="bowl").solve(params=_PARAMS, log_level="off")
    flags = np.asarray(solution.replay_artifacts.project(DISSOLUTION_FLAG)[1]["pair"])
    assert flags.tolist() == [False, False]


def test_simulate_routes_an_empty_landing_between_feasible_nodes_to_the_fallback():
    """A household landing at wage 2 has no feasible pair action and takes its route.

    Both nodes of the cell around wage 2 are feasible, but at wage 2 the wife's
    payoff is 0 and no action satisfies her participation constraint.
    """
    _, panel = _simulate(payoff_name="bowl", wages=np.array([2.0]))
    assert panel["regime_name"].tolist() == ["alone_m"]


def test_simulate_carries_the_projected_state_into_the_fallback():
    """The fallback leg's identity projection keeps the realised wage."""
    _, panel = _simulate(payoff_name="bowl", wages=np.array([2.0]))
    assert panel["wage"].tolist() == [2.0]


@pytest.mark.parametrize("payoff_name", ["bowl", "ramp"])
@pytest.mark.parametrize("enable_jit", [True, False], ids=["jit", "eager"])
def test_simulated_regime_matches_the_enumerated_reference(
    *, payoff_name: str, enable_jit: bool
):
    """Every landing goes where enumerating `pair`'s actions at it says it goes."""
    wages = _landings(payoff_name=payoff_name)
    _, panel = _simulate(payoff_name=payoff_name, wages=wages, enable_jit=enable_jit)
    expected = [
        (float(w), _reference_regime(payoff_name=payoff_name, wage=w)) for w in wages
    ]
    got = list(zip(panel["wage"].tolist(), panel["regime_name"].tolist(), strict=True))
    assert got == expected


@pytest.mark.parametrize("payoff_name", ["bowl", "ramp"])
def test_generated_landings_reach_both_sides_of_the_boundary(*, payoff_name: str):
    """The generated class holds feasible and empty landings alike."""
    wages = _landings(payoff_name=payoff_name)
    assert {_reference_regime(payoff_name=payoff_name, wage=w) for w in wages} == {
        "pair",
        "alone_m",
    }


def test_every_dissolution_route_is_recorded_with_its_state_and_route():
    """The rerouted household is recorded with subject, period, state and route."""
    model = _build_model(payoff_name="bowl")
    solution = model.solve(params=_PARAMS, log_level="off")
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wage": jnp.array([3.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["source"], jnp.int32),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    assert [
        (r.subject_id, r.period, r.regime, dict(r.state), r.source, r.fallback)
        for r in result.dissolution_fallbacks
    ] == [(1, 1, "pair", {"wage": 2.0}, "source", "alone_m")]


def _clip_state_fallback_model() -> Model:
    """`alone_m` clips its wage to `[1, 3]`; the route doubles the realised wage."""
    return _build_model(
        payoff_name="bowl",
        fallback_grid=IrregSpacedGrid(
            points=np.array([1.0, 2.0, 3.0]), off_grid="clip_state"
        ),
        fallback_projection=_double,
    )


def _simulate_at_wage_two(*, model: Model, log_level: LogLevel = "off"):
    solution = model.solve(params=_PARAMS, log_level="off")
    return model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wage": jnp.array([2.0]),
            "age": jnp.zeros(1),
            "regime_id": jnp.full(1, model.regime_names_to_ids["source"], jnp.int32),
        },
        solution=solution,
        log_level=log_level,
        seed=0,
    )


def test_a_rerouted_state_follows_the_fallback_grid_clip_rule():
    """Wage 2 is empty in `pair`; doubled to 4, `alone_m` clips it to its top node."""
    result = _simulate_at_wage_two(model=_clip_state_fallback_model())
    panel = result.to_dataframe().query("period == 1")
    assert list(zip(panel["regime_name"], panel["wage"], strict=True)) == [
        ("alone_m", 3.0)
    ]


def test_debug_transition_counts_name_the_regime_a_rerouted_household_acted_in(
    caplog: pytest.LogCaptureFixture,
):
    """The household rerouted at period 1 acted in `alone_m`, then stays there."""
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        _simulate_at_wage_two(model=_clip_state_fallback_model(), log_level="debug")
    transition_lines = [
        line.strip()
        for record in caplog.records
        if record.getMessage().startswith("  transitions:")
        for line in record.getMessage().splitlines()[1:]
    ]
    assert transition_lines == [
        "- source → pair = 1",
        "- alone_m → alone_m = 1",
        "- alone_m → alone_m = 1",
    ]
