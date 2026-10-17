"""Report empty action sets without confusing them with feasible infinite values."""

import ast
import re
from contextlib import nullcontext
from pathlib import Path
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
from lcm.component_jobs import plan_component_jobs, run_component_job
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
def test_a_fully_feasible_solve_has_only_off_grid_warning(
    *, enable_jit: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Feasible actions can leave the hull without empty-action/value warnings."""
    model = _model(enable_jit=enable_jit, constrained=False)
    model.solve(params={"discount_factor": 0.5}, log_level="warning")
    assert [record.getMessage() for record in caplog.records] == [
        (
            "Off-grid next states: regime='working', target='working', "
            "period=0, state='wealth', phase='solve', count=2"
        )
    ]


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


def _second_type_infinite_utility(
    *,
    consumption: ContinuousAction,
    wealth: ContinuousState,
    pref_type: DiscreteState,
) -> FloatND:
    return jnp.where(
        (wealth == 1) & (pref_type == PreferenceType.second), -jnp.inf, consumption
    )


def _blocked_model(
    *,
    schedule: InvariantBlockSchedule = InvariantBlockSchedule.BLOCK_MAJOR,
    constrained: bool = True,
    infinite_utility: bool = False,
) -> Model:
    """Block the preference type; working period 1 has 3 + 1 + 1 empty cells.

    Every wealth cell of the empty type and wealth zero of the other two types
    have no feasible action. With the infinite utility, wealth one of the
    second type is feasible with value negative infinity.
    """
    types = DiscreteGrid(category_class=PreferenceType)
    return Model(
        regimes={
            "working": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=2, n_points=3),
                    "pref_type": types,
                },
                actions={"consumption": LinSpacedGrid(start=1, stop=2, n_points=2)},
                functions={
                    "utility": _second_type_infinite_utility
                    if infinite_utility
                    else _utility
                },
                constraints={"affordable": _type_affordable} if constrained else {},
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


_BLOCKED_PARAMS: UserParams = {"discount_factor": 0.5}
_BLOCKED_SUBJECT: UserInitialConditions = {
    "age": jnp.asarray([0.0]),
    "wealth": jnp.asarray([2.0]),
    "pref_type": jnp.asarray([PreferenceType.first], dtype=jnp.int32),
    "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
}
_COUNT_PATTERNS = {
    "empty": re.compile(
        r"(\d+) state\(s\) with no feasible action for regime 'working', period 1 "
    ),
    "other_inf": re.compile(
        r"Inf in V_arr for (\d+) other state\(s\) for regime 'working', period 1 "
    ),
}


def _period_one_counts(*, messages: list[str], kind: str) -> list[int]:
    """Return the counts of every working period-1 warning of one kind."""
    return [
        int(match.group(1))
        for message in messages
        if (match := _COUNT_PATTERNS[kind].match(message)) is not None
    ]


@pytest.mark.parametrize(
    "schedule",
    [InvariantBlockSchedule.PERIOD_MAJOR, InvariantBlockSchedule.BLOCK_MAJOR],
)
def test_blocked_solve_counts_the_complete_regime_period(
    *, schedule: InvariantBlockSchedule, caplog: pytest.LogCaptureFixture
) -> None:
    """Empty sets are counted once across invariant components, not per block."""
    _blocked_model(schedule=schedule).solve(params=_BLOCKED_PARAMS, log_level="warning")
    messages = [record.getMessage() for record in caplog.records]

    assert _period_one_counts(messages=messages, kind="empty") == [5]


@pytest.mark.parametrize(("kind", "total"), [("empty", 5), ("other_inf", 1)])
@pytest.mark.parametrize("n_jobs", [1, 3])
def test_component_solve_jobs_report_each_count_once(
    *,
    kind: str,
    total: int,
    n_jobs: int,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each job warns once for its codes; the jobs' counts add up to the solve's."""
    model = _blocked_model(infinite_utility=True)
    directory = tmp_path / "jobs"
    plan_component_jobs(
        model=model, params=_BLOCKED_PARAMS, directory=directory, n_jobs=n_jobs
    )
    counts = []
    for job in range(n_jobs):
        caplog.clear()
        run_component_job(
            model=model,
            params=_BLOCKED_PARAMS,
            directory=directory,
            job=job,
            log_level="warning",
        )
        counts.append(
            _period_one_counts(
                messages=[record.getMessage() for record in caplog.records], kind=kind
            )
        )

    assert (
        sum(sum(job_counts) for job_counts in counts),
        max(len(job_counts) for job_counts in counts),
    ) == (total, 1)


def _period_one_messages(*, messages: list[str]) -> list[str]:
    """Return the working period-1 feasibility warnings, in logging order."""
    return [
        message
        for message in messages
        if any(pattern.match(message) for pattern in _COUNT_PATTERNS.values())
    ]


def test_blocked_solve_warning_names_the_whole_regime_period(
    *, caplog: pytest.LogCaptureFixture
) -> None:
    """A whole-model solve reports its counts without a scope label."""
    _blocked_model(infinite_utility=True).solve(
        params=_BLOCKED_PARAMS, log_level="warning"
    )

    assert _period_one_messages(
        messages=[record.getMessage() for record in caplog.records]
    ) == [
        "5 state(s) with no feasible action for regime 'working', period 1 (age 1.0)",
        "Inf in V_arr for 1 other state(s) for regime 'working', period 1 (age 1.0)",
    ]


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (
            0,
            [
                (
                    "3 state(s) with no feasible action for regime 'working', "
                    "period 1 (age 1.0), pref_type codes (0,) only"
                ),
            ],
        ),
        (
            2,
            [
                (
                    "1 state(s) with no feasible action for regime 'working', "
                    "period 1 (age 1.0), pref_type codes (2,) only"
                ),
                (
                    "Inf in V_arr for 1 other state(s) for regime 'working', "
                    "period 1 (age 1.0), pref_type codes (2,) only"
                ),
            ],
        ),
    ],
)
def test_component_job_warning_names_the_codes_it_covers(
    *,
    job: int,
    expected: list[str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A job's counts are labelled with the blocked-state codes they cover."""
    model = _blocked_model(infinite_utility=True)
    directory = tmp_path / "jobs"
    plan_component_jobs(
        model=model,
        params=_BLOCKED_PARAMS,
        directory=directory,
        assignment=((0,), (1,), (2,)),
    )
    caplog.clear()
    run_component_job(
        model=model,
        params=_BLOCKED_PARAMS,
        directory=directory,
        job=job,
        log_level="warning",
    )

    assert (
        _period_one_messages(
            messages=[record.getMessage() for record in caplog.records]
        )
        == expected
    )


def test_component_simulation_job_reports_each_count_once(
    *, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A simulating job warns once with the counts of the codes it solved."""
    model = _blocked_model()
    directory = tmp_path / "jobs"
    plan_component_jobs(
        model=model,
        params=_BLOCKED_PARAMS,
        directory=directory,
        n_jobs=1,
        initial_conditions=_BLOCKED_SUBJECT,
        seed=0,
    )
    caplog.clear()
    run_component_job(
        model=model,
        params=_BLOCKED_PARAMS,
        directory=directory,
        job=0,
        log_level="warning",
        initial_conditions=_BLOCKED_SUBJECT,
    )
    messages = [record.getMessage() for record in caplog.records]

    assert _period_one_counts(messages=messages, kind="empty") == [5]


def test_block_major_simulation_reports_each_count_once(
    *, caplog: pytest.LogCaptureFixture
) -> None:
    """Solving every component while simulating warns once per regime-period."""
    _blocked_model().simulate(
        params=_BLOCKED_PARAMS,
        initial_conditions=_BLOCKED_SUBJECT,
        seed=0,
        log_level="warning",
    )
    messages = [record.getMessage() for record in caplog.records]

    assert _period_one_counts(messages=messages, kind="empty") == [5]


def test_feasible_component_job_has_no_feasibility_warning(
    *, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fully feasible job logs no feasibility warning."""
    model = _blocked_model(constrained=False)
    directory = tmp_path / "jobs"
    plan_component_jobs(
        model=model, params=_BLOCKED_PARAMS, directory=directory, n_jobs=1
    )
    caplog.clear()
    run_component_job(
        model=model,
        params=_BLOCKED_PARAMS,
        directory=directory,
        job=0,
        log_level="warning",
    )

    assert [
        message
        for message in (record.getMessage() for record in caplog.records)
        if "no feasible action" in message or message.startswith("Inf in V_arr")
    ] == []


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


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_initial_joint_error_preserves_public_physical_states(
    *,
    runtime: bool,
    fold: bool,
    log_level: Literal["off", "warning", "progress", "debug"],
) -> None:
    """An adopted initial error must describe physical inputs, not lowered slots."""
    params: UserParams = (
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
        "wealth": jnp.asarray([0.5]),
        "x": jnp.asarray([3.1]),
        "y": jnp.asarray([-1.3]),
        "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
    }
    control = _joint_empty_model(
        constrained=False, runtime=runtime, fold=fold, memory_budget=None
    )
    positive = control.simulate(
        params=params,
        initial_conditions=initial,
        seed=534,
        log_level="off",
        runtime_checks=True,
    ).to_dataframe()
    first = positive.loc[positive["period"] == 0].iloc[0]
    assert np.isfinite(first["value"])
    assert first["consumption"] == 2.0
    expected = np.asarray([0.5, 3.1, -1.3])
    names = ("wealth", "x", "y")
    np.testing.assert_array_almost_equal(
        first[list(names)].to_numpy(dtype=float), expected, decimal=DECIMAL_PRECISION
    )
    model = _joint_empty_model(
        constrained=True, runtime=runtime, fold=fold, memory_budget=None
    )
    with pytest.raises(
        exceptions.NoFeasibleActionError, match=r"working.*period 0"
    ) as caught:
        model.simulate(
            params=params,
            initial_conditions=initial,
            seed=534,
            log_level=log_level,
            # False would reach a different guard and miss initial formatting.
            runtime_checks=True,
        )
    message = str(caught.value)
    state = ast.literal_eval(
        message.split("state ", 1)[1].split(" has no feasible action.", 1)[0]
    )
    assert set(state) == set(names)
    np.testing.assert_array_almost_equal(
        [state[name] for name in names], expected, decimal=DECIMAL_PRECISION
    )
    table = message.split("Per-constraint admissibility", 1)[1]
    header = next(line for line in table.splitlines() if "affordable" in line)
    assert set(names).issubset(header.split())
    assert "shocks_normal_" not in message


@pytest.mark.parametrize(
    ("runtime", "fixed_law"), [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("memory_budget", [None, 64 * 1024**2])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_initial_joint_empty_state_reports_physical_values(
    *,
    runtime: bool,
    fixed_law: bool,
    fold: bool,
    memory_budget: int | None,
    log_level: Literal["off", "warning", "progress", "debug"],
) -> None:
    """Initial error state and constraint table expose the supplied physical law."""
    model = _joint_empty_model(
        constrained=True,
        runtime=runtime,
        fixed_law=fixed_law,
        fold=fold,
        memory_budget=memory_budget,
    )
    params = (
        {
            "working": {
                "shocks": {
                    "mean": jnp.asarray((2.0, -1.0)),
                    "cov": jnp.asarray(((1.0, 0.4), (0.4, 2.0))),
                }
            }
        }
        if runtime and not fixed_law
        else {}
    )
    initial = {
        "age": jnp.asarray([0.0, 0.0, 0.0]),
        "wealth": jnp.asarray([1.0, 0.0, 0.0]),
        "x": jnp.asarray([2.0, 3.0, 1.5]),
        "y": jnp.asarray([-1.0, -2.0, -0.5]),
        "regime_id": jnp.asarray([DecisionId.working] * 3, dtype=jnp.int32),
    }
    parameter_calls = [params]
    if runtime and not fixed_law:
        changed: UserParams = {
            "working": {
                "shocks": {
                    "mean": jnp.asarray((2.25, -0.75)),
                    "cov": jnp.asarray(((1.5, -0.2), (-0.2, 2.5))),
                }
            }
        }
        parameter_calls = [params, changed, params]
    control = _joint_empty_model(
        constrained=False,
        runtime=runtime,
        fixed_law=fixed_law,
        fold=fold,
        memory_budget=memory_budget,
    )
    for current_params in parameter_calls:
        positive = control.simulate(
            params=current_params,
            initial_conditions=initial,
            seed=531,
            log_level="off",
            runtime_checks=True,
        ).to_dataframe()
        first_period = positive.loc[positive["period"] == 0]
        assert np.isfinite(first_period["value"]).all()
        np.testing.assert_array_almost_equal(
            first_period[["wealth", "x", "y"]].to_numpy(),
            [[1.0, 2.0, -1.0], [0.0, 3.0, -2.0], [0.0, 1.5, -0.5]],
            decimal=DECIMAL_PRECISION,
        )
        with pytest.raises(exceptions.NoFeasibleActionError) as caught:
            model.simulate(
                params=current_params,
                initial_conditions=initial,
                seed=531,
                log_level=log_level,
                runtime_checks=True,
            )
        message = str(caught.value)
        assert "period 0" in message
        state = ast.literal_eval(
            message.split("state ", 1)[1].split(" has no feasible", 1)[0]
        )
        assert set(state) == {"wealth", "x", "y"}
        np.testing.assert_array_almost_equal(
            [state[name] for name in ("wealth", "x", "y")],
            [0.0, 3.0, -2.0],
            decimal=DECIMAL_PRECISION,
        )
        assert "shocks_normal_" not in message
        table = message.split("):\n", 1)[1].split("\n\nNo action", 1)[0]
        assert table.splitlines()[0].split() == ["wealth", "x", "y", "affordable"]
        rows = [line.split() for line in table.splitlines()[-2:]]
        assert [row[0] for row in rows] == ["1", "2"]
        assert all(row[-1] == "False" for row in rows)
        np.testing.assert_array_almost_equal(
            np.asarray([row[1:-1] for row in rows], dtype=float),
            [[0.0, 3.0, -2.0], [0.0, 1.5, -0.5]],
            decimal=5,
        )


_ORIGINAL_JOINT_LAW = {
    "mean": jnp.asarray((2.0, -1.0)),
    "cov": jnp.asarray(((1.0, 0.4), (0.4, 2.0))),
}


def _initial_joint_states(
    *,
    control: Model,
    model: Model,
    params: UserParams,
    x: float,
    y: float,
    checks: bool,
) -> tuple[dict[str, float | str], dict[str, float]]:
    """Return the empty-set error's state and the control's entered state.

    Both models start one subject at wealth zero; only the constrained model
    rejects every action there, so the unconstrained control enters the same
    physical state and records it.
    """
    initial = {
        "age": jnp.asarray([0.0]),
        "wealth": jnp.asarray([0.0]),
        "x": jnp.asarray([x]),
        "y": jnp.asarray([y]),
        "regime_id": jnp.asarray([DecisionId.working], dtype=jnp.int32),
    }
    entered = (
        control.simulate(
            params=params,
            initial_conditions=initial,
            seed=531,
            log_level="off",
            runtime_checks=checks,
        )
        .to_dataframe()
        .loc[lambda frame: frame["period"] == 0]
        .iloc[0]
    )
    with pytest.raises(exceptions.NoFeasibleActionError) as caught:
        model.simulate(
            params=params,
            initial_conditions=initial,
            seed=531,
            log_level="off",
            runtime_checks=checks,
        )
    message = str(caught.value)
    prefix = "Regime 'working', period 0, state "
    state = ast.literal_eval(
        message.split(prefix, 1)[1].split(" has no feasible action.", 1)[0]
    )
    return state, {name: float(entered[name]) for name in ("wealth", "x", "y")}


@pytest.mark.parametrize("checks", [True, False])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("memory_budget", [None, 64 * 1024**2])
def test_initial_joint_empty_state_reports_entered_physical_values(
    *, checks: bool, fold: bool, memory_budget: int | None
) -> None:
    """An initial empty-set error names the physical state the subject entered."""
    state, entered = _initial_joint_states(
        control=_joint_empty_model(
            constrained=False, runtime=False, fold=fold, memory_budget=memory_budget
        ),
        model=_joint_empty_model(
            constrained=True, runtime=False, fold=fold, memory_budget=memory_budget
        ),
        params={},
        x=2.0,
        y=-1.0,
        checks=checks,
    )

    assert state == pytest.approx(entered, abs=10.0**-DECIMAL_PRECISION)


@pytest.mark.parametrize("law", ["declared", "runtime", "fixed"])
@pytest.mark.parametrize(("x", "y"), [(2.0, -1.0), (3.0, 0.5)])
@pytest.mark.parametrize("fold", [False, True])
def test_initial_joint_empty_state_uses_the_current_law(
    *, law: Literal["declared", "runtime", "fixed"], x: float, y: float, fold: bool
) -> None:
    """Declared, runtime and fixed laws all report the entered physical state."""
    models = [
        _joint_empty_model(
            constrained=constrained,
            runtime=law != "declared",
            fold=fold,
            memory_budget=64 * 1024**2,
            fixed_law=law == "fixed",
        )
        for constrained in (False, True)
    ]
    state, entered = _initial_joint_states(
        control=models[0],
        model=models[1],
        params={"working": {"shocks": _ORIGINAL_JOINT_LAW}} if law == "runtime" else {},
        x=x,
        y=y,
        checks=True,
    )

    assert state == pytest.approx(entered, abs=10.0**-DECIMAL_PRECISION)


@pytest.mark.parametrize("fold", [False, True])
def test_initial_joint_empty_state_uses_each_current_runtime_law(*, fold: bool) -> None:
    """Changed then original laws on the same Models report each entered state."""
    control, model = (
        _joint_empty_model(
            constrained=constrained, runtime=True, fold=fold, memory_budget=None
        )
        for constrained in (False, True)
    )
    changed = {
        "mean": jnp.asarray((4.0, 3.0)),
        "cov": jnp.asarray(((2.0, 0.2), (0.2, 1.0))),
    }
    observed = [
        _initial_joint_states(
            control=control,
            model=model,
            params={"working": {"shocks": law}},
            x=3.0,
            y=0.5,
            checks=True,
        )
        for law in (_ORIGINAL_JOINT_LAW, changed, _ORIGINAL_JOINT_LAW)
    ]

    assert [state for state, _ in observed] == [
        pytest.approx(entered, abs=10.0**-DECIMAL_PRECISION) for _, entered in observed
    ]


@pytest.mark.parametrize("keep_intermediates", [(), ("F",)])
def test_lookup_policy_refuses_a_state_without_a_feasible_action(
    *, keep_intermediates: tuple[Literal["F"], ...]
) -> None:
    """A queried state with an empty action set is refused, with or without parts."""
    model = _model(enable_jit=True)
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    with pytest.raises(
        exceptions.NoFeasibleActionError, match=r"working.*period 1.*wealth.*0"
    ):
        model.lookup_policy(
            params={"discount_factor": 0.5},
            solution=solution,
            period=1,
            regime_name="working",
            states={"wealth": jnp.asarray([0.0, 2.0])},
            keep_intermediates=keep_intermediates,
        )


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (InvariantBlockSchedule.PERIOD_MAJOR, [(5, "")]),
        (InvariantBlockSchedule.BLOCK_MAJOR, [(4, ", pref_type codes (0, 1) only")]),
    ],
)
def test_simulation_logs_the_solve_counts_before_refusing_an_empty_set(
    *,
    schedule: InvariantBlockSchedule,
    expected: list[tuple[int, str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The solve's empty-set counts are logged even when the simulation refuses.

    Block-major has solved the empty and the first type when the first-type
    subject reaches wealth zero, so its count covers those two codes and says so.
    """
    with pytest.raises(exceptions.NoFeasibleActionError):
        _blocked_model(schedule=schedule).simulate(
            params=_BLOCKED_PARAMS,
            initial_conditions={**_BLOCKED_SUBJECT, "wealth": jnp.asarray([1.0])},
            seed=0,
            log_level="warning",
        )
    messages = [record.getMessage() for record in caplog.records]

    assert [
        (int(match.group(1)), _scope_of(message))
        for message in messages
        if (match := _COUNT_PATTERNS["empty"].match(message)) is not None
    ] == expected


def _scope_of(message: str) -> str:
    """Return the codes label a component count carries, or `""`."""
    match = re.search(r", pref_type codes \(.*\) only", message)
    return match.group(0) if match else ""
