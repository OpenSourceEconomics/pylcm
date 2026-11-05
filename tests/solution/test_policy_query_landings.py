"""Canonical typed policy queries retain reached-landing diagnostics."""

from collections.abc import Mapping
from fractions import Fraction
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.current_iid import CurrentIIDDrawPlan
from _lcm.execution.core_program import CoreBuildContext
from _lcm.policy_queries import ScalarPolicyQueries, _PolicyQueryKernel
from _lcm.typing import PytreeValue
from lcm import (
    ByPeriod,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    Model,
    ParticipationCheck,
    ParticipationRead,
    Periods,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
    categorical,
)
from lcm.exceptions import OffGridStateError
from lcm.shocks import Normal, PhysicalValues, ShockGroup, UserNodes
from lcm.typing import BoolND, FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

type LandingCase = Literal[
    "on_grid",
    "infeasible_off_grid",
    "positive_mass_off_grid",
    "zero_mass_off_grid",
    "nonwinning_off_grid",
]

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


class ReachedLandingError(ValueError):
    """A positive-mass admissible action reaches outside the declared grid."""


def reference_values(  # noqa: C901 - independent literal finite oracle keeps the original cases together
    *,
    case: LandingCase = "on_grid",
    nodes: tuple[int, ...] = (-1, 1),
    masses: tuple[Fraction, ...] = (Fraction(1, 2), Fraction(1, 2)),
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Return the two source-wealth cells' common-policy stakeholder tuples.

    ``infeasible_off_grid`` drops action 1 before the reached-landing check.
    ``zero_mass_off_grid`` must be paired with zero mass on the positive shock.
    ``nonwinning_off_grid`` makes action 0 off-grid but value-dominated: it still
    counts because it is admissible, even though it is not the maximizing action.
    """
    if len(nodes) != len(masses) or sum(masses) != 1 or min(masses) < 0:
        raise ValueError("The oracle needs a normalized nonnegative finite law.")
    if case not in {
        "on_grid",
        "infeasible_off_grid",
        "positive_mass_off_grid",
        "zero_mass_off_grid",
        "nonwinning_off_grid",
    }:
        raise ValueError(case)
    output = []
    for wealth in (0, 1):
        total_f, total_m = Fraction(0), Fraction(0)
        for eps, mass in zip(nodes, masses, strict=True):
            if mass == 0:
                continue
            candidates = []
            for action in (0, 1):
                if case in {"infeasible_off_grid", "positive_mass_off_grid"}:
                    landing = 2 * action
                elif case == "zero_mass_off_grid":
                    landing = action + 2 * (eps > 0)
                elif case == "nonwinning_off_grid":
                    landing = -1 if action == 0 else 1
                else:
                    landing = action
                # beta=1/2, terminal stakeholder values=2*wealth.
                q_f = Fraction(wealth + action + eps + landing)
                q_m = Fraction(wealth + action - eps + landing)
                feasible = q_f >= -1000 and (
                    case != "infeasible_off_grid" or action == 0
                )
                if not feasible:
                    continue
                if not 0 <= landing <= 1:
                    raise ReachedLandingError((wealth, eps, action, landing))
                candidates.append((q_f + q_m, action, q_f, q_m))
            # Stable lowest original action ID at an exact objective tie.
            _, _, value_f, value_m = max(candidates, key=lambda row: (row[0], -row[1]))
            total_f += mass * value_f
            total_m += mass * value_m
        output.append((total_f, total_m))
    return tuple(output)


@categorical(ordered=False)
class _Regimes:
    source: ScalarInt
    destination: ScalarInt


def _female(*, wealth: FloatND, action: IntND, eps: FloatND) -> FloatND:
    return wealth + action + eps


def _male(*, wealth: FloatND, action: IntND, eps: FloatND) -> FloatND:
    return wealth + action - eps


def _terminal(wealth: FloatND) -> FloatND:
    return 2.0 * wealth


def _admit_all(Q_f: FloatND) -> BoolND:
    return Q_f >= -1000.0


def _admit_zero(*, Q_f: FloatND, action: IntND) -> BoolND:
    return (Q_f >= -1000.0) & (action == 0)


def _on_grid(action: IntND) -> FloatND:
    return 1.0 * action


def _double(action: IntND) -> FloatND:
    return 2.0 * action


def _shock_tail(*, action: IntND, eps: FloatND) -> FloatND:
    return 1.0 * action + 2.0 * (eps > 0)


def _bad_loser(action: IntND) -> FloatND:
    return jnp.where(action == 0, -1.0, 1.0)


def _model(
    *,
    case: LandingCase,
    enable_jit: bool,
    reverse: bool = False,
    split_negative_mass: bool = False,
    checked: bool = True,
) -> Model:
    nodes = [-1.0, -1.0, 1.0] if split_negative_mass else [-1.0, 1.0]
    masses = [0.25, 0.25, 0.5] if split_negative_mass else [0.5, 0.5]
    if case == "zero_mass_off_grid":
        masses = [0.5, 0.5, 0.0] if split_negative_mass else [1.0, 0.0]
    if reverse:
        nodes.reverse()
        masses.reverse()
    landing = {
        "on_grid": _on_grid,
        "infeasible_off_grid": _double,
        "positive_mass_off_grid": _double,
        "zero_mass_off_grid": _shock_tail,
        "nonwinning_off_grid": _bad_loser,
    }[case]
    return Model(
        regimes={
            "source": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _female, "m": _male})
                },
                constraints={
                    "admissible_value": ValueDependentConstraint(
                        predicate=_admit_zero
                        if case == "infeasible_off_grid"
                        else _admit_all
                    )
                },
                state_transitions={"wealth": landing},
                shocks=("innovation",),
            ),
            "destination": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=0,
                        stop=1,
                        n_points=2,
                        off_grid="raise" if checked else "extrapolate",
                    )
                },
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _terminal, "m": _terminal}
                    )
                },
            ),
        },
        regime_id_class=_Regimes,
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={"source": {"destination": Periods(values=(0,))}},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray(nodes)[:, None], weights=np.asarray(masses)
                ),
            )
        },
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _solve(*, model: Model) -> np.ndarray:
    solution = model.solve(
        params={"discount_factor": 0.5}, log_level="off", runtime_checks=False
    )
    return np.asarray(solution.values[0]["source"])


@pytest.mark.parametrize("enable_jit", [False, True])
def test_checked_on_grid_collective_iid_query_matches_affine_oracle(
    *, enable_jit: bool
) -> None:
    """The only change from the control is a raise-rule grid; all landings are valid."""
    expected = np.asarray(reference_values(), dtype=float)
    control = _solve(model=_model(case="on_grid", enable_jit=enable_jit, checked=False))
    np.testing.assert_array_almost_equal(control, expected, decimal=DECIMAL_PRECISION)
    checked = _solve(model=_model(case="on_grid", enable_jit=enable_jit))
    np.testing.assert_array_almost_equal(checked, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("case", ["infeasible_off_grid", "zero_mass_off_grid"])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_unreached_landings_do_not_poison_the_query(
    *, case: LandingCase, enable_jit: bool
) -> None:
    masses = (
        (Fraction(1), Fraction(0))
        if case == "zero_mass_off_grid"
        else (Fraction(1, 2), Fraction(1, 2))
    )
    expected = np.asarray(reference_values(case=case, masses=masses), dtype=float)
    actual = _solve(model=_model(case=case, enable_jit=enable_jit))
    np.testing.assert_array_almost_equal(actual, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("case", ["positive_mass_off_grid", "nonwinning_off_grid"])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_reached_landings_remain_mandatory_even_on_a_nonwinning_action(
    *, case: LandingCase, enable_jit: bool
) -> None:
    with pytest.raises(ReachedLandingError):
        reference_values(case=case)
    with pytest.raises(OffGridStateError):
        _solve(model=_model(case=case, enable_jit=enable_jit))


@pytest.mark.parametrize(
    "case", ["on_grid", "infeasible_off_grid", "zero_mass_off_grid"]
)
@pytest.mark.parametrize(
    ("reverse", "split_negative_mass"), [(True, False), (False, True)]
)
def test_original_row_reordering_and_equal_mass_splitting_preserve_checked_values(
    *, case: LandingCase, reverse: bool, split_negative_mass: bool
) -> None:
    masses = (
        (Fraction(1), Fraction(0))
        if case == "zero_mass_off_grid"
        else (Fraction(1, 2), Fraction(1, 2))
    )
    expected = np.asarray(reference_values(case=case, masses=masses), dtype=float)
    actual = _solve(
        model=_model(
            case=case,
            enable_jit=True,
            reverse=reverse,
            split_negative_mass=split_negative_mass,
        )
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=DECIMAL_PRECISION)


@categorical(ordered=False)
class _ParticipationRegimes:
    parent: ScalarInt
    source: ScalarInt
    destination: ScalarInt
    fallback: ScalarInt


def _zero() -> float:
    return 0.0


def _wealth(wealth: FloatND) -> FloatND:
    return wealth


def _square_wealth(wealth: FloatND) -> FloatND:
    return wealth * wealth


def _keep_participation(V_source: FloatND) -> BoolND:
    return V_source >= -1000.0


@pytest.mark.parametrize("off_grid", [False, True])
def test_participation_pricing_preserves_nested_actual_landing_identity(
    *, off_grid: bool
) -> None:
    """Safe source nodes do not excuse an off-grid actual parent participation check."""
    model = Model(
        regimes={
            "parent": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=2 if off_grid else 0,
                        stop=3 if off_grid else 1,
                        n_points=2,
                    )
                },
                functions={"utility": _zero},
                state_transitions={"wealth": _wealth},
            ),
            "source": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=0, stop=1, n_points=2, off_grid="extrapolate"
                    )
                },
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _female},
                state_transitions={"wealth": _square_wealth},
                shocks=("innovation",),
            ),
            "destination": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=0, stop=1, n_points=2, off_grid="raise"
                    )
                },
                functions={"utility": _terminal},
            ),
            "fallback": Regime(functions={"utility": _zero}),
        },
        regime_id_class=_ParticipationRegimes,
        n_periods=3,
        initial_nodes=InitialNodes(by_period={0: "parent"}),
        edges={
            "parent": Transition(
                targets={
                    "source": Periods(values=(0,)),
                    "fallback": Periods(values=(0,)),
                },
                law=ByPeriod(cases={0: "source"}),
                participation={
                    "source": ParticipationCheck(
                        predicate=_keep_participation,
                        reads={
                            "V_source": ParticipationRead(
                                kind="Raw",
                                regime="source",
                                projection={"wealth": _wealth},
                            )
                        },
                        routes={
                            "self": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                )
                            )
                        },
                    )
                },
            ),
            "source": {"destination": Periods(values=(1,))},
        },
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                ),
            )
        },
    )
    if off_grid:
        with pytest.raises(OffGridStateError) as raised:
            model.solve(
                params={"discount_factor": 0.5}, log_level="off", runtime_checks=False
            )
        # Report the nested producer at t=1 rather than its parent t=0.
        message = str(raised.value)
        assert "regime='source'" in message
        assert "target='destination'" in message
        assert "state='wealth'" in message
        assert "period=1" in message
    else:
        solution = model.solve(
            params={"discount_factor": 0.5}, log_level="off", runtime_checks=False
        )
        # Source Q=w+1+E eps+(.5)(2w²); parent pays half that same native query.
        np.testing.assert_array_almost_equal(
            solution.values[0]["parent"], [0.5, 1.5], decimal=DECIMAL_PRECISION
        )


@pytest.mark.parametrize("case", ["on_grid", "zero_mass_off_grid"])
def test_physical_replay_keeps_checked_landing_counts_after_query_reduction(
    *, case: LandingCase
) -> None:
    """Null quadrature mass does not excuse a realized positive physical landing."""
    model = _model(case=case, enable_jit=True)
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    initial = {
        "period": jnp.zeros(64, dtype=jnp.int32),
        "regime_id": jnp.full(64, _Regimes.source, dtype=jnp.int32),
        "wealth": jnp.tile(jnp.asarray([0.0, 1.0]), 32),
    }
    if case == "zero_mass_off_grid":
        with pytest.raises(OffGridStateError, match="phase='realized'"):
            model.simulate(
                params=params,
                solution=solution,
                initial_conditions=initial,
                seed=3,
                log_level="off",
                runtime_checks=False,
            )
    else:
        frame = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            seed=3,
            log_level="off",
            runtime_checks=False,
        ).to_dataframe()
        source = frame.loc[frame["period"] == 0].sort_values("subject_id")
        destination = frame.loc[frame["period"] == 1].sort_values("subject_id")
        np.testing.assert_array_equal(source["action"], 1)
        np.testing.assert_array_equal(destination["wealth"], 1)
        np.testing.assert_array_almost_equal(
            source["value_f"] + source["value_m"],
            2 * source["wealth"] + 4,
            decimal=DECIMAL_PRECISION,
        )


def test_scalar_physical_query_retains_nonempty_actual_landing_lane() -> None:
    """The real physical frame reaches the count lane before any transition guard."""
    model = _model(case="zero_mass_off_grid", enable_jit=True)
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    flat_params, _generation = model._prepare_call(params=params, runtime_checks=False)
    kernel = model._regimes["source"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    scalar = ScalarPolicyQueries(
        definitions=kernel.definitions, root=kernel.root, period=0
    )
    assert scalar.landing_checks == (("source", 0, "destination", "wealth"),)
    plan = kernel.definitions[kernel.root].context.current_iid_plans[0]
    assert plan is not None
    draw = CurrentIIDDrawPlan(
        bindings={name: binding for name, binding, _ in plan.bindings}
    )
    arguments = cast(
        "Mapping[str, Mapping[str, PytreeValue]]",
        scalar.arguments(
            context=CoreBuildContext(
                state_action_space=kernel.root_context.state_action_space,
                next_regime_to_V_arr=solution.values[1],
                next_regime_to_continuation={},
                flat_params=flat_params,
                period=0,
                ages=kernel.root_context.ages,
            ),
            fixed_flat_params=kernel.fixed_flat_params,
        ),
    )
    evaluate = jax.jit(scalar)
    for eps, expected_value, expected_count in (
        (-0.5, (2.0, 3.0), 0),
        (0.5, (5.0, 4.0), 2),
    ):
        frame = draw.realise(
            keys={"eps": jax.random.key(123)},
            group_params={"innovation": {}},
            arguments={
                "wealth": jnp.asarray(0.5),
                "period": jnp.int32(0),
                "age": jnp.int32(0),
            },
            overrides={"eps": PhysicalValues(values=jnp.asarray(eps))},
        )
        answer = evaluate(
            states={"wealth": jnp.asarray(0.5)}, arguments=arguments, frame=frame
        )
        # At a=1: landing=1 or3, Q=(.5+1±eps)+(.5)(2*landing).
        # Both original admissible actions count at the positive physical draw.
        np.testing.assert_array_almost_equal(
            answer[0], expected_value, decimal=DECIMAL_PRECISION
        )
        assert int(answer[1]) == 1
        assert bool(answer[2])
        assert bool(answer[3])
        assert answer[4].dtype == jnp.int32
        np.testing.assert_array_equal(answer[4], [expected_count])
