"""A public solve names each host phase it passes through, with a call id and status."""

import logging
import re
from typing import Any, cast

import jax.numpy as jnp
import pytest

from _lcm.solution import backward_induction
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
    "state_action_spaces",
    "continuation_templates",
    "program_graphs",
    "structural_resolution",
    "residency_inventory",
    "compilation_waves",
    "workspace_selection",
    "backward_induction",
    "result_assembly",
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
    """A solve that returns closes every phase with `ok` and a duration."""
    ends = [m for m in _solve(caplog=caplog) if m["edge"] == "end"]
    assert all(m["status"] == "ok" and float(m["seconds"]) >= 0.0 for m in ends)


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


def test_off_carries_no_phase_records(caplog: pytest.LogCaptureFixture) -> None:
    """A solve with logging off emits no phase records at all."""
    model = get_model()
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        model.solve(params=get_params(model=model), log_level="off")
    assert _records(caplog) == []
