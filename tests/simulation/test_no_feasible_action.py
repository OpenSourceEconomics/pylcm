"""Report empty action sets without confusing them with feasible infinite values."""

import ast
from contextlib import nullcontext
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    InvariantBlockSchedule,
    JointNormalIIDProcess,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    exceptions,
    fixed_transition,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserInitialConditions,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class DecisionId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class PreferenceType:
    empty: ScalarInt
    first: ScalarInt
    second: ScalarInt


def _type_affordable(*, wealth: ContinuousState, pref_type: DiscreteState) -> BoolND:
    return (wealth >= 1) & (pref_type != PreferenceType.empty)


def _type_zero(*, pref_type: DiscreteState) -> FloatND:
    return 0.0 * pref_type


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return consumption


def _infinite_utility(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return jnp.where(wealth == 1, -jnp.inf, consumption)


def _affordable(*, wealth: ContinuousState) -> BoolND:
    return wealth >= 1


def _next_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth - 1


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _two_state_utility(*, wealth: ContinuousState, ability: ContinuousState) -> FloatND:
    return wealth + ability


@pytest.mark.parametrize(("extent", "overflows"), [(46_340, False), (46_341, True)])
def test_native_counter_domain_is_bounded_before_cartesian_allocation(
    *, extent: int, overflows: bool
) -> None:
    """Real node vectors can declare the boundary without solving their product."""
    outcome = (
        pytest.raises(exceptions.ExecutionPlanningError, match="int32 feasibility")
        if overflows
        else nullcontext()
    )
    with outcome:
        Model(
            regimes={
                "working": Regime(
                    states={
                        "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=extent),
                        "ability": LinSpacedGrid(start=0.0, stop=1.0, n_points=extent),
                    },
                    state_transitions={
                        "wealth": fixed_transition("wealth"),
                        "ability": fixed_transition("ability"),
                    },
                    functions={"utility": _two_state_utility},
                ),
                "dead": Regime(functions={"utility": _zero}),
            },
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            initial_nodes={0: "working"},
            edges={"working": {"dead": (0,)}},
            regime_id_class=DecisionId,
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )


def _model(
    *,
    enable_jit: bool,
    infinite_utility: bool = False,
    constrained: bool = True,
    collective: bool = False,
    axis_widths: dict[str, int] | None = None,
) -> Model:
    utility = _infinite_utility if infinite_utility else _utility
    return Model(
        regimes={
            "working": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=2, n_points=3)},
                actions={"consumption": LinSpacedGrid(start=1, stop=2, n_points=2)},
                functions={
                    "utility": CollectiveUtility(utilities={"f": utility, "m": utility})
                    if collective
                    else utility
                },
                constraints={"affordable": _affordable} if constrained else {},
                state_transitions={"wealth": _next_wealth},
            ),
            "dead": Regime(
                functions={
                    "utility": CollectiveUtility(utilities={"f": _zero, "m": _zero})
                    if collective
                    else _zero
                }
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: ("working", "dead"), 1: "working"},
        edges={"working": {"working": (0,), "dead": (1,)}},
        regime_id_class=DecisionId,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            device_memory_bytes=None, axis_widths=axis_widths or {}
        ),
    )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_simulation_reports_the_reached_empty_action_set(
    *,
    enable_jit: bool,
    runtime_checks: bool,
    log_level: Literal["off", "warning", "progress", "debug"],
) -> None:
    """An affordable initial state reaches wealth zero and reports that decision."""
    model = _model(enable_jit=enable_jit)
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    error_class = getattr(exceptions, "NoFeasibleActionError", exceptions.PyLCMError)
    with pytest.raises(error_class, match=r"working.*period 1.*wealth.*0"):
        model.simulate(
            params={"discount_factor": 0.5},
            solution=solution,
            initial_conditions={
                "age": jnp.asarray([0.0]),
                "wealth": jnp.asarray([1.0]),
                "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
            },
            log_level=log_level,
            runtime_checks=runtime_checks,
            seed=0,
        )


def test_a_reached_collective_empty_set_without_a_fallback_is_refused() -> None:
    """An owned household cannot execute an arbitrary action after losing support."""
    model = _model(enable_jit=True, collective=True)
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    with pytest.raises(
        exceptions.NoFeasibleActionError, match=r"working.*period 1.*wealth.*0"
    ):
        model.simulate(
            params={"discount_factor": 0.5},
            solution=solution,
            initial_conditions={
                "age": jnp.asarray([0.0]),
                "wealth": jnp.asarray([1.0]),
                "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
            },
            log_level="off",
            runtime_checks=False,
            seed=0,
        )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("axis_widths", [None, {"action_product": 1, "cell": 2}])
@pytest.mark.parametrize("check", ["values", "empty_count", "other_inf_count"])
def test_solve_counts_empty_sets_separately_from_feasible_negative_infinity(
    *,
    enable_jit: bool,
    axis_widths: dict[str, int] | None,
    check: Literal["values", "empty_count", "other_inf_count"],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wealth zero has no action; wealth one has feasible utility negative infinity."""
    model = _model(
        enable_jit=enable_jit, infinite_utility=True, axis_widths=axis_widths
    )
    result = model.solve(params={"discount_factor": 0.5}, log_level="warning")
    if check == "values":
        np.testing.assert_array_equal(
            np.asarray(result.values[1]["working"]), [-np.inf, -np.inf, 2.0]
        )
        return
    messages = [record.getMessage() for record in caplog.records]
    expected = (
        "1 state(s) with no feasible action"
        if check == "empty_count"
        else "Inf in V_arr for 1 other state(s)"
    )
    assert any(
        expected in message and "working" in message and "period 1" in message
        for message in messages
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_simulation_reports_an_empty_initial_state(*, enable_jit: bool) -> None:
    """Initial-condition preflight reports the same public failure context."""
    model = _model(enable_jit=enable_jit)
    with pytest.raises(
        exceptions.NoFeasibleActionError, match=r"working.*period 0.*wealth.*0"
    ):
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions={
                "age": jnp.asarray([0.0]),
                "wealth": jnp.asarray([0.0]),
                "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
            },
            log_level="off",
            seed=0,
        )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_a_fully_feasible_solve_has_no_warning(
    *, enable_jit: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """A fully feasible finite grid leaves the warning log empty."""
    model = _model(enable_jit=enable_jit, constrained=False)
    model.solve(params={"discount_factor": 0.5}, log_level="warning")
    assert [record.getMessage() for record in caplog.records] == []


@pytest.mark.parametrize("enable_jit", [False, True])
def test_empty_other_regime_and_padded_rows_do_not_refuse_owned_decisions(
    *, enable_jit: bool
) -> None:
    """Mixed regimes and a short subject tile keep the working subjects' actions."""
    model = _model(enable_jit=enable_jit, axis_widths={"subject": 3})
    result = model.simulate(
        params={"discount_factor": 0.5},
        initial_conditions={
            "age": jnp.zeros(4),
            "wealth": jnp.asarray([0.0, 2.0, 0.0, 2.0]),
            "regime_id": jnp.asarray(
                [
                    DecisionId.dead,
                    DecisionId.working,
                    DecisionId.dead,
                    DecisionId.working,
                ],
                dtype=jnp.int32,
            ),
        },
        log_level="off",
        seed=0,
    )
    frame = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        frame.loc[frame["regime_name"] == "working", "consumption"], [2.0] * 4
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_the_owned_row_in_a_short_subject_tile_still_refuses_empty_actions(
    *, enable_jit: bool
) -> None:
    """The last real subject reaches an empty set despite its tile's padding."""
    model = _model(enable_jit=enable_jit, axis_widths={"subject": 3})
    with pytest.raises(
        exceptions.NoFeasibleActionError, match=r"working.*period 1.*wealth.*0"
    ):
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions={
                "age": jnp.zeros(4),
                "wealth": jnp.asarray([0.0, 2.0, 0.0, 1.0]),
                "regime_id": jnp.asarray(
                    [
                        DecisionId.dead,
                        DecisionId.working,
                        DecisionId.dead,
                        DecisionId.working,
                    ],
                    dtype=jnp.int32,
                ),
            },
            log_level="off",
            seed=0,
        )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_feasible_negative_infinity_does_not_refuse_the_action(
    *, enable_jit: bool
) -> None:
    """A feasible negative-infinity decision retains its normal winning identity."""
    model = _model(enable_jit=enable_jit, infinite_utility=True)
    result = model.simulate(
        params={"discount_factor": 0.5},
        initial_conditions={
            "age": jnp.asarray([1.0]),
            "wealth": jnp.asarray([1.0]),
            "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
        },
        log_level="off",
        runtime_checks=False,
        seed=0,
    )
    frame = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        frame.loc[frame["regime_name"] == "working", "consumption"], [1.0]
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_feasible_simulation_keeps_the_optimal_action(*, enable_jit: bool) -> None:
    """Wealth two reaches wealth one; both decisions consume the best grid action."""
    model = _model(enable_jit=enable_jit)
    result = model.simulate(
        params={"discount_factor": 0.5},
        initial_conditions={
            "age": jnp.asarray([0.0]),
            "wealth": jnp.asarray([2.0]),
            "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
        },
        log_level="off",
        seed=0,
    )
    frame = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        frame.loc[frame["regime_name"] == "working", "consumption"], [2.0, 2.0]
    )


@pytest.mark.parametrize(
    "schedule",
    [InvariantBlockSchedule.PERIOD_MAJOR, InvariantBlockSchedule.BLOCK_MAJOR],
)
def test_blocked_solve_counts_the_complete_regime_period(
    *, schedule: InvariantBlockSchedule, caplog: pytest.LogCaptureFixture
) -> None:
    """Empty sets are counted across invariant components, rather than per block."""
    types = DiscreteGrid(category_class=PreferenceType)
    model = Model(
        regimes={
            "working": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=2, n_points=3),
                    "pref_type": types,
                },
                actions={"consumption": LinSpacedGrid(start=1, stop=2, n_points=2)},
                functions={"utility": _utility},
                constraints={"affordable": _type_affordable},
                state_transitions={
                    "wealth": _next_wealth,
                    "pref_type": fixed_transition("pref_type"),
                },
            ),
            "dead": Regime(
                states={"pref_type": types}, functions={"utility": _type_zero}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "working"},
        edges={"working": {"working": (0,), "dead": (1,)}},
        regime_id_class=DecisionId,
        execution_config=ExecutionConfig(
            device_memory_bytes=None,
            invariant_block_widths={"pref_type": 1},
            invariant_block_schedule=schedule,
        ),
    )
    model.solve(params={"discount_factor": 0.5}, log_level="warning")
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "5 state(s) with no feasible action" in message
        and "working" in message
        and "period 1" in message
        for message in messages
    )


def _joint_utility(
    *, consumption: ContinuousAction, x: ContinuousState, y: ContinuousState
) -> FloatND:
    return consumption + 0.0 * x + 0.0 * y


def _joint_empty_model(
    *,
    constrained: bool,
    runtime: bool,
    fold: bool,
    memory_budget: int | None,
    fixed_law: bool = False,
) -> Model:
    fixed_params: UserParams = {
        "working": {"koopmans_aggregator": {"discount_factor": 0.5}}
    }
    if fixed_law:
        fixed_params = {
            "working": {
                "koopmans_aggregator": {"discount_factor": 0.5},
                "shocks": {
                    "mean": jnp.asarray((2.0, -1.0)),
                    "cov": jnp.asarray(((1.0, 0.4), (0.4, 2.0))),
                },
            }
        }
    return Model(
        regimes={
            "working": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=2, n_points=3),
                    "shocks": JointNormalIIDProcess(
                        states=("x", "y"),
                        n_points=(3, 3),
                        gauss_hermite=True,
                        mean=None if runtime else (2.0, -1.0),
                        cov=None if runtime else ((1.0, 0.4), (0.4, 2.0)),
                        fold=fold,
                    ),
                },
                actions={"consumption": LinSpacedGrid(start=1, stop=2, n_points=2)},
                functions={"utility": _joint_utility},
                constraints={"affordable": _affordable} if constrained else {},
                state_transitions={"wealth": _next_wealth},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "working"},
        edges={"working": {"working": (0,), "dead": (1,)}},
        regime_id_class=DecisionId,
        fixed_params=fixed_params,
        execution_config=ExecutionConfig(device_memory_bytes=memory_budget),
    )


def _assert_joint_empty_matches_native_control(
    *,
    control: Model,
    model: Model,
    params: UserParams,
    initial: UserInitialConditions,
) -> dict[str, float]:
    """Run the native positive oracle before checking the matched empty-state error."""
    positive = control.simulate(
        params=params,
        initial_conditions=initial,
        seed=531,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe()
    reached = positive.loc[positive["period"] == 1].iloc[0]
    assert reached["wealth"] == 0.0
    assert reached["regime_name"] == "working"
    assert np.isfinite(reached["value"])
    assert reached["consumption"] == 2.0
    assert np.isfinite(reached[["x", "y"]].to_numpy(dtype=float)).all()
    with pytest.raises(exceptions.NoFeasibleActionError) as caught:
        model.simulate(
            params=params,
            initial_conditions=initial,
            seed=531,
            log_level="off",
            runtime_checks=False,
        )
    message = str(caught.value)
    assert "working" in message
    assert "period 1" in message
    state = ast.literal_eval(
        message.split("state ", 1)[1].rsplit(" has no feasible", 1)[0]
    )
    assert set(state) == {"wealth", "x", "y"}
    np.testing.assert_array_almost_equal(
        [state[name] for name in ("wealth", "x", "y")],
        reached[["wealth", "x", "y"]].to_numpy(dtype=float),
        decimal=DECIMAL_PRECISION,
    )
    return {name: float(reached[name]) for name in ("wealth", "x", "y")}


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("memory_budget", [None, 64 * 1024**2])
def test_reached_joint_empty_state_reports_physical_values(
    *, runtime: bool, fold: bool, memory_budget: int | None
) -> None:
    """Error state agrees with the actual reached state of a matching native run."""
    params = (
        {
            "working": {
                "shocks": {
                    "mean": jnp.asarray((2.0, -1.0)),
                    "cov": jnp.asarray(((1.0, 0.4), (0.4, 2.0))),
                }
            }
        }
        if runtime
        else {}
    )
    initial = {
        "age": jnp.asarray([0.0]),
        "wealth": jnp.asarray([1.0]),
        "x": jnp.asarray([2.0]),
        "y": jnp.asarray([-1.0]),
        "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
    }
    control = _joint_empty_model(
        constrained=False, runtime=runtime, fold=fold, memory_budget=memory_budget
    )
    model = _joint_empty_model(
        constrained=True, runtime=runtime, fold=fold, memory_budget=memory_budget
    )
    _assert_joint_empty_matches_native_control(
        control=control, model=model, params=params, initial=initial
    )


@pytest.mark.parametrize("fold", [False, True])
def test_reached_joint_empty_state_uses_fixed_runtime_law(*, fold: bool) -> None:
    """A runtime-declared law supplied by Model.fixed_params stays physical."""
    control = _joint_empty_model(
        constrained=False,
        runtime=True,
        fold=fold,
        memory_budget=64 * 1024**2,
        fixed_law=True,
    )
    model = _joint_empty_model(
        constrained=True,
        runtime=True,
        fold=fold,
        memory_budget=64 * 1024**2,
        fixed_law=True,
    )
    initial = {
        "age": jnp.asarray([0.0]),
        "wealth": jnp.asarray([1.0]),
        "x": jnp.asarray([2.0]),
        "y": jnp.asarray([-1.0]),
        "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
    }
    _assert_joint_empty_matches_native_control(
        control=control, model=model, params={}, initial=initial
    )


@pytest.mark.parametrize("fold", [False, True])
def test_reached_joint_empty_state_uses_each_current_runtime_law(*, fold: bool) -> None:
    """Changed then original laws on the same Models cannot bind a stale law."""
    control = _joint_empty_model(
        constrained=False,
        runtime=True,
        fold=fold,
        memory_budget=64 * 1024**2,
    )
    model = _joint_empty_model(
        constrained=True,
        runtime=True,
        fold=fold,
        memory_budget=64 * 1024**2,
    )
    original_mean = (2.0, -1.0)
    original_cov = ((1.0, 0.4), (0.4, 2.0))
    reached_states = []
    for mean, cov in (
        (original_mean, original_cov),
        ((4.0, 3.0), ((2.0, 0.2), (0.2, 1.0))),
        (original_mean, original_cov),
    ):
        params = {
            "working": {
                "shocks": {
                    "mean": jnp.asarray(mean),
                    "cov": jnp.asarray(cov),
                }
            }
        }
        initial = {
            "age": jnp.asarray([0.0]),
            "wealth": jnp.asarray([1.0]),
            "x": jnp.asarray([mean[0]]),
            "y": jnp.asarray([mean[1]]),
            "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
        }
        reached = _assert_joint_empty_matches_native_control(
            control=control, model=model, params=params, initial=initial
        )
        reached_states.append([reached[name] for name in ("wealth", "x", "y")])
    # The native control proves the changed law reaches a different physical state.
    assert not np.allclose(reached_states[0][1:], reached_states[1][1:])
    np.testing.assert_array_almost_equal(
        reached_states[2], reached_states[0], decimal=DECIMAL_PRECISION
    )
