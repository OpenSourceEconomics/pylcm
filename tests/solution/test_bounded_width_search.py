"""The bounded width search, driven through a real budgeted solve.

Every width here belongs to the donor-pair fixture of
`tests/solution/_candidate_census.py`, whose `cell` axis is declared with
alignment 1 and minimum width 1: extent 2 in the `alive` regime and extent 8 in
the `dead` one. The ranked frontier therefore offers the powers of two at or
below the extent, and every other width in between is off-frontier.
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import pytest

from _lcm.solution import backward_induction
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import WidthSearch, WidthSearchPolicy
from lcm.solver_api import ResultRetention
from tests.solution._candidate_census import (
    CensusRecorder,
    Triple,
    donor_pair_model,
    solve_donor_pair,
)
from tests.test_models import nbegm_ride_along_toy

# Every byte budget below is a census constant read at fp64: reservations and
# residencies halve at fp32, so the budgets would refuse nothing there and the
# tests would measure the precision rather than the search.
pytestmark = pytest.mark.usefixtures("x64_enabled")

ALIVE_0 = ("alive", 0, "main")
DEAD_3 = ("dead", 3, "main")


def _captured_frontier(*, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Solve the donor-pair fixture and hand back the frontier it resolved."""
    captured: dict[str, Any] = {}
    resolve = backward_induction._resolve_output_layouts_and_lowering_keys

    def observe(**kwargs: Any) -> Any:
        result = resolve(**kwargs)
        captured.setdefault("frontier", result[-1])
        return result

    monkeypatch.setattr(
        backward_induction, "_resolve_output_layouts_and_lowering_keys", observe
    )
    solve_donor_pair(model=donor_pair_model(device_memory_bytes=10**8))
    return captured


def test_bind_widths_binds_a_width_the_ranked_frontier_never_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A width between two ranked candidates resolves into a bound candidate."""
    frontier = _captured_frontier(monkeypatch=monkeypatch)["frontier"]
    candidate = frontier.bind_widths(triple=DEAD_3, widths={"cell": 3})
    assert candidate[1] == (("cell", 3),)


def test_bind_widths_returns_the_bound_candidate_for_a_width_already_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking twice for one width binds one candidate, not two."""
    frontier = _captured_frontier(monkeypatch=monkeypatch)["frontier"]
    frontier.bind_widths(triple=DEAD_3, widths={"cell": 5})
    frontier.bind_widths(triple=DEAD_3, widths={"cell": 5})
    assert [candidate[1] for candidate in frontier.candidates_by_triple[DEAD_3]] == [
        (("cell", 8),),
        (("cell", 5),),
    ]


def test_bind_widths_refuses_a_width_the_axis_declaration_excludes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A width above the declared extent is refused in the declaration's words."""
    frontier = _captured_frontier(monkeypatch=monkeypatch)["frontier"]
    with pytest.raises(ValueError, match="exceeds its product extent 8"):
        frontier.bind_widths(triple=DEAD_3, widths={"cell": 9})


def _bounded(
    *,
    seed: str = "conservative",
    max_evaluations: int = 24,
    refinement_share: int = 8,
    device_memory_bytes: int = 10**8,
) -> ExecutionConfig:
    """Build the donor-pair fixture's execution config under a bounded search."""
    return ExecutionConfig(
        device_memory_bytes=device_memory_bytes,
        width_search=WidthSearchPolicy(
            kind=WidthSearch.BOUNDED,
            seed=cast('Literal["conservative", "widest"]', seed),
            max_evaluations=max_evaluations,
            refinement_share=refinement_share,
        ),
    )


def _solve_with(*, execution_config: ExecutionConfig, log_level: str = "off") -> Any:
    """Solve the donor-pair fixture under one execution config."""
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_liquid=8,
        n_savings=10,
        n_consumption=12,
        execution_config=execution_config,
    )
    return model.solve(
        params=nbegm_ride_along_toy.build_params(),
        retention=ResultRetention.VALUES,
        log_level=cast("Any", log_level),
    )


def _selected_widths(
    *, monkeypatch: pytest.MonkeyPatch, execution_config: ExecutionConfig
) -> Mapping[Triple, Any]:
    """Return the widths a solve under one execution config dispatched."""
    recorder = CensusRecorder()
    recorder.install(monkeypatch=monkeypatch)
    _solve_with(execution_config=execution_config)
    return recorder.census().selected_widths


def test_bounded_search_admits_its_conservative_seed_without_shrinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget the bootstrap anchor fits selects that anchor, four below extent."""
    widths = _selected_widths(monkeypatch=monkeypatch, execution_config=_bounded())
    assert widths[DEAD_3] == (("cell", 4),)


def test_bounded_search_shrinks_the_width_the_paired_admission_re_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A width its donating variant fits but its pair does not is shrunk, not kept."""
    widths = _selected_widths(
        monkeypatch=monkeypatch,
        execution_config=_bounded(seed="widest", device_memory_bytes=64171),
    )
    assert widths[ALIVE_0] == (("cell", 1),)


def test_bounded_search_leaves_the_other_cores_at_the_width_their_pair_admits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the core whose pair exceeds the budget is shrunk."""
    widths = _selected_widths(
        monkeypatch=monkeypatch,
        execution_config=_bounded(seed="widest", device_memory_bytes=64171),
    )
    assert widths[("alive", 1, "main")] == (("cell", 2),)


def test_bounded_search_at_the_widest_seed_selects_the_exhaustive_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded at the full extent, the bounded search keeps the ranked winner."""
    bounded = _selected_widths(
        monkeypatch=monkeypatch, execution_config=_bounded(seed="widest")
    )
    exhaustive = _selected_widths(
        monkeypatch=monkeypatch,
        execution_config=ExecutionConfig(device_memory_bytes=10**8),
    )
    assert bounded == exhaustive


def test_bounded_search_at_the_widest_seed_computes_the_exhaustive_values() -> None:
    """Values agree with the ranked walk's wherever both select the same width."""
    bounded = _solve_with(execution_config=_bounded(seed="widest"))
    exhaustive = _solve_with(
        execution_config=ExecutionConfig(device_memory_bytes=10**8)
    )
    np.testing.assert_array_equal(
        np.asarray(bounded.values[0]["alive"]),
        np.asarray(exhaustive.values[0]["alive"]),
    )


def test_bounded_search_announces_its_policy_and_seed_for_every_core(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The opening record names the policy, the seed rule and the first widths."""
    with caplog.at_level(logging.INFO, logger="lcm"):
        _solve_with(execution_config=_bounded(seed="widest"), log_level="progress")

    assert (
        "bounded width search regime 'dead', core 'main', period 3: policy bounded, "
        "seed rule 'widest', at most 24 evaluations of which 8 refine; "
        "first widths {'cell': 8}" in caplog.text
    )


def test_bounded_search_logs_each_evaluation_with_its_decision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every evaluation records its reservation, its residency and its verdict."""
    with caplog.at_level(logging.INFO, logger="lcm"):
        _solve_with(execution_config=_bounded(seed="widest"), log_level="progress")

    assert (
        "bounded width search regime 'dead', core 'main', period 3: evaluation 1 at "
        "{'cell': 8} — 136 reservation plus 1232 resident bytes, admitted"
        in caplog.text
    )


def test_bounded_search_closes_by_counting_the_variants_its_evaluations_compiled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The closing record counts variants, not evaluations.

    A core admitted at its seed spends one evaluation and compiles two
    variants, because a donating survivor is paired with its fallback.
    """
    with caplog.at_level(logging.INFO, logger="lcm"):
        _solve_with(execution_config=_bounded(seed="widest"), log_level="progress")

    assert (
        "bounded width search regime 'alive', core 'main', period 0: 1 admission "
        "evaluations, 2 unique variants compiled, 2 compiler requests, 0 cache "
        "hits; selected {'cell': 2} before the fusion check" in caplog.text
    )


def test_bounded_search_refuses_by_naming_the_evaluation_budget_it_spent() -> None:
    """An exhausted search says the frontier, not the model, went untested."""
    with pytest.raises(
        ExecutionPlanningError, match="not an exhaustive test of the frontier"
    ):
        _solve_with(
            execution_config=_bounded(
                seed="widest",
                max_evaluations=1,
                refinement_share=0,
                device_memory_bytes=64171,
            )
        )


def test_bounded_search_names_a_core_whose_residency_already_fills_the_budget() -> None:
    """A core with no room for any workspace is refused by name, not searched."""
    with pytest.raises(ExecutionPlanningError, match="reach the budget"):
        _solve_with(execution_config=_bounded(device_memory_bytes=1500))
