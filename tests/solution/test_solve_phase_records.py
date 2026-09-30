"""A public solve names each host phase it passes through, with a call id and status."""

import logging
import re
from typing import Any, cast

import jax.numpy as jnp
import pytest

from _lcm.solution import backward_induction
from benchmarks.warm_solve_phases import CallPhases, parse_phase_records
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND, ScalarInt

_PHASES = (
    "params_validation",
    "authority_fingerprint",
    "solver_param_checks",
    "state_action_spaces",
    "continuation_templates",
    "program_graphs",
    "structural_resolution",
    "residency_inventory",
    "compilation_waves",
    "workspace_selection",
    "backward_induction",
    "result_assembly",
    "result_readiness",
)
_RECORD = re.compile(
    r"^solve call (?P<call>[0-9a-f]+) phase (?P<name>[a-z_]+) "
    r"(?P<edge>begin|end)(?: status=(?P<status>\w+) seconds=(?P<seconds>[0-9.]+))?$"
)

# Points of the wealth grid, so one regime value is this many elements.
_N_WEALTH = 16


@categorical(ordered=False)
class RegimeId:
    """Regime identities of the two-regime lifecycle."""

    acting: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class Work:
    """Binary labour-supply action."""

    leisure: ScalarInt
    working: ScalarInt


def _next_regime(*, age: int) -> ScalarInt:
    """Leave the acting regime after the last acting age."""
    return jnp.where(age < 2, RegimeId.acting, RegimeId.done)


def _utility(*, consumption: float, work: ScalarInt, wealth: float) -> FloatND:
    """Value one action cell at one wealth node."""
    return jnp.log(consumption) - 0.1 * work + 0.01 * wealth


def _terminal_utility(*, wealth: float) -> float:
    """Value the terminal regime by its wealth."""
    return wealth


def get_model(*, budget_bytes: int | None = None) -> Model:
    """Build one acting regime over three periods into a terminal regime."""
    acting = Regime(
        transition=_next_regime,
        active=lambda age: age < 3,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=_N_WEALTH)},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=1.0, stop=3.0, n_points=3),
        },
        functions={"utility": _utility},
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= 3,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=_N_WEALTH)},
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"acting": acting, "done": done},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=budget_bytes),
    )


def get_params(*, model: Model) -> dict[str, Any]:
    """Return solvable parameters of the fixture model."""
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    return params


def _records(caplog: pytest.LogCaptureFixture) -> list[re.Match[str]]:
    return [m for r in caplog.records if (m := _RECORD.match(r.getMessage()))]


def _solve(
    *,
    caplog: pytest.LogCaptureFixture,
    budget_bytes: int | None = None,
    n_calls: int = 1,
) -> list[re.Match[str]]:
    model = get_model(budget_bytes=budget_bytes)
    with caplog.at_level(logging.INFO, logger="lcm"):
        for _ in range(n_calls):
            model.solve(params=get_params(model=model), log_level="progress")
    return _records(caplog)


@pytest.mark.parametrize("budget_bytes", [None, 2**30])
def test_children_open_and_close_in_order_inside_public_solve(
    *, caplog: pytest.LogCaptureFixture, budget_bytes: int | None
) -> None:
    """Every phase opens and closes once, in order, between the public brackets."""
    edges = [
        (m["name"], m["edge"]) for m in _solve(caplog=caplog, budget_bytes=budget_bytes)
    ]
    expected = (
        [("public_solve", "begin")]
        + [(n, e) for n in _PHASES for e in ("begin", "end")]
        + [("public_solve", "end")]
    )
    assert edges == expected


def test_every_phase_end_reports_ok_and_a_nonnegative_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A solve that returns closes every phase with `ok` and a duration.

    The guard is that every phase of the vocabulary, plus the public bracket,
    is present: `all()` over an empty list would otherwise hold whether or not
    a single record was written.
    """
    ends = [m for m in _solve(caplog=caplog) if m["edge"] == "end"]
    assert [
        m["name"] for m in ends if m["status"] == "ok" and float(m["seconds"]) >= 0.0
    ] == [
        *_PHASES,
        "public_solve",
    ]


def test_two_calls_carry_two_distinct_call_ids(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each public call stamps its records with an id of its own."""
    ids = {m["call"] for m in _solve(caplog=caplog, n_calls=2)}
    assert len(ids) == 2


def test_children_reconcile_to_the_public_call_within_tolerance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The children of a public call sum to it up to a small, non-negative residual."""
    ends = {
        m["name"]: float(m["seconds"])
        for m in _solve(caplog=caplog)
        if m["edge"] == "end"
    }
    children = sum(ends[n] for n in _PHASES)
    assert 0.0 <= ends["public_solve"] - children <= 0.05 * ends["public_solve"] + 0.05


def test_a_raising_phase_ends_with_status_error(
    *, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The phase whose body raised closes with `error`."""

    def boom(**_kwargs: object) -> None:
        raise ExecutionPlanningError("forced")

    monkeypatch.setattr(backward_induction, "_build_continuation_templates", boom)
    model = get_model()
    with (
        caplog.at_level(logging.INFO, logger="lcm"),
        pytest.raises(ExecutionPlanningError),
    ):
        model.solve(params=get_params(model=model), log_level="progress")
    statuses = {m["name"]: m["status"] for m in _records(caplog) if m["edge"] == "end"}
    assert statuses["continuation_templates"] == "error"


def test_parse_phase_records_reports_a_phase_without_an_end_as_incomplete() -> None:
    """A phase whose `begin` has no matching `end` comes back as `incomplete`."""
    call_id = "0123456789ab"
    lines = [
        f"solve call {call_id} phase public_solve begin",
        f"solve call {call_id} phase params_validation begin",
        f"solve call {call_id} phase params_validation end status=ok seconds=0.500000",
        f"solve call {call_id} phase backward_induction begin",
    ]
    (call,) = parse_phase_records(lines=lines)
    assert [(p.name, p.status, p.seconds) for p in call.phases] == [
        ("public_solve", "incomplete", None),
        ("params_validation", "ok", 0.5),
        ("backward_induction", "incomplete", None),
    ]


def test_off_carries_no_phase_records(caplog: pytest.LogCaptureFixture) -> None:
    """A solve with logging off emits no phase records at all."""
    model = get_model()
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        model.solve(params=get_params(model=model), log_level="off")
    assert _records(caplog) == []


def test_parse_phase_records_reads_a_stamped_line() -> None:
    """A record behind a `<ISO stamp> <LEVEL> ` prefix parses as the bare one does."""
    call_id = "0123456789ab"
    lines = [
        f"2026-09-18T11:22:33.123 INFO solve call {call_id} phase public_solve begin",
        (
            f"2026-09-18T11:22:33.456 INFO solve call {call_id} phase public_solve "
            "end status=ok seconds=0.250000"
        ),
    ]
    (call,) = parse_phase_records(lines=lines)
    assert [(p.name, p.status, p.seconds) for p in call.phases] == [
        ("public_solve", "ok", 0.25)
    ]


# The phases a simulate call passes through after its internal solve returns,
# in emission order. A chunked run repeats `simulation_chunk` once per chunk.
_POST_SOLVE_PHASES = (
    "result_assembly",
    "solution_resolution",
    "solution_handover",
    "chunk_planning",
    "simulation_setup",
    "simulation_chunk",
    "simulation_completion",
    "simulation_result",
    "result_finalization",
)


def _simulate_calls(
    *,
    caplog: pytest.LogCaptureFixture,
    log_level: str,
    budget_bytes: int | None = None,
) -> tuple[CallPhases, ...]:
    model = get_model(budget_bytes=budget_bytes)
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        model.simulate(
            params=get_params(model=model),
            initial_conditions={
                "wealth": jnp.array([1.0, 1.5, 2.0]),
                "age": jnp.zeros(3),
                "regime_id": jnp.full(3, RegimeId.acting, dtype=jnp.int32),
            },
            log_level=log_level,  # ty: ignore[invalid-argument-type]
            seed=0,
        )
    return parse_phase_records(lines=[r.getMessage() for r in caplog.records])


@pytest.mark.parametrize("budget_bytes", [None, 2**30])
@pytest.mark.parametrize("log_level", ["progress", "debug"])
def test_simulate_names_its_post_solve_phases_under_one_call(
    *, caplog: pytest.LogCaptureFixture, log_level: str, budget_bytes: int | None
) -> None:
    """One simulate call writes one call id whose top-level phases end the solve,
    resolve and hand over the solution, plan the chunks, simulate and assemble
    the result."""
    (call,) = _simulate_calls(
        caplog=caplog, log_level=log_level, budget_bytes=budget_bytes
    )
    top_level = [p.name for p in call.phases if p.depth == 1]
    after_solve = top_level[top_level.index("backward_induction") + 1 :]
    assert list(dict.fromkeys(after_solve)) == list(_POST_SOLVE_PHASES)


def test_simulate_opens_with_the_public_bracket_and_the_solve_phases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The call's outermost phase is `public_simulate`, and its internal solve
    contributes the solve's own phases from the fingerprint through backward
    induction."""
    (call,) = _simulate_calls(caplog=caplog, log_level="progress")
    top_level = [p.name for p in call.phases if p.depth == 1]
    solve_phases = _PHASES[_PHASES.index("authority_fingerprint") : -2]
    start = top_level.index("authority_fingerprint")
    assert (
        call.phases[0].name,
        tuple(top_level[start : start + len(solve_phases)]),
    ) == ("public_simulate", solve_phases)


def test_simulate_phases_reconcile_to_the_public_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Top-level phases plus the explicit residual equal the public call's time.

    Each duration is written with six decimals, so the sum is exact to half a
    microsecond per phase; the residual itself is non-negative and small.
    """
    (call,) = _simulate_calls(caplog=caplog, log_level="progress")
    public = call.seconds(name="public_simulate")
    top_level = [p.seconds for p in call.phases if p.depth == 1]
    residual = call.residual_seconds()
    assert public is not None
    assert residual is not None
    tolerance = 1e-6 * (len(top_level) + 1)
    assert abs(sum(s for s in top_level if s is not None) + residual - public) <= (
        tolerance
    )
    assert -tolerance <= residual <= 0.05 * public + 0.05


def test_budgeted_simulate_times_each_simulation_compilation_inside_a_phase(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every simulation executable compiled during a budgeted call is bracketed
    as a `simulation_compilation` phase nested inside a top-level phase."""
    (call,) = _simulate_calls(caplog=caplog, log_level="progress", budget_bytes=2**30)
    depths = {p.depth for p in call.phases if p.name == "simulation_compilation"}
    assert depths == {2}


def test_simulate_at_off_carries_no_phase_records(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A simulate call with logging off emits no phase records at all."""
    assert _simulate_calls(caplog=caplog, log_level="off") == ()


def test_parse_phase_records_reports_each_phase_depth() -> None:
    """A phase opened inside another is one level deeper than its parent."""
    call_id = "0123456789ab"
    lines = [
        f"solve call {call_id} phase public_simulate begin",
        f"solve call {call_id} phase chunk_planning begin",
        f"solve call {call_id} phase simulation_compilation begin",
        (
            f"solve call {call_id} phase simulation_compilation end status=ok "
            "seconds=0.100000"
        ),
        f"solve call {call_id} phase chunk_planning end status=ok seconds=0.300000",
        f"solve call {call_id} phase public_simulate end status=ok seconds=0.500000",
    ]
    (call,) = parse_phase_records(lines=lines)
    assert [(p.name, p.depth) for p in call.phases] == [
        ("public_simulate", 0),
        ("chunk_planning", 1),
        ("simulation_compilation", 2),
    ]


def test_residual_counts_only_top_level_phases() -> None:
    """A nested phase's time is already inside its parent and is not subtracted."""
    call_id = "0123456789ab"
    lines = [
        f"solve call {call_id} phase public_simulate begin",
        f"solve call {call_id} phase chunk_planning begin",
        f"solve call {call_id} phase simulation_compilation begin",
        (
            f"solve call {call_id} phase simulation_compilation end status=ok "
            "seconds=0.100000"
        ),
        f"solve call {call_id} phase chunk_planning end status=ok seconds=0.300000",
        f"solve call {call_id} phase public_simulate end status=ok seconds=0.500000",
    ]
    (call,) = parse_phase_records(lines=lines)
    assert call.residual_seconds() == pytest.approx(0.2, abs=1e-9)


def test_parse_phase_records_reads_overlapping_same_named_phases_as_siblings() -> None:
    """Same-named phases open at once (compiles on a worker pool) share one depth,
    and each keeps its own duration."""
    call_id = "0123456789ab"
    lines = [
        f"solve call {call_id} phase public_simulate begin",
        f"solve call {call_id} phase chunk_planning begin",
        f"solve call {call_id} phase simulation_compilation begin",
        f"solve call {call_id} phase simulation_compilation begin",
        (
            f"solve call {call_id} phase simulation_compilation end status=ok "
            "seconds=0.100000"
        ),
        (
            f"solve call {call_id} phase simulation_compilation end status=ok "
            "seconds=0.200000"
        ),
        f"solve call {call_id} phase chunk_planning end status=ok seconds=0.300000",
        f"solve call {call_id} phase public_simulate end status=ok seconds=0.500000",
    ]
    (call,) = parse_phase_records(lines=lines)
    assert [(p.name, p.depth, p.seconds) for p in call.phases] == [
        ("public_simulate", 0, 0.5),
        ("chunk_planning", 1, 0.3),
        ("simulation_compilation", 2, 0.1),
        ("simulation_compilation", 2, 0.2),
    ]
