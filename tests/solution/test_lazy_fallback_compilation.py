"""A candidate's donation-free fallback is compiled only once it is still in play.

Every budget here is read off the census of the donor-pair fixture at `cell=2`
under `ResultRetention.VALUES`, recorded by
`tests/solution/test_candidate_census.py`:

| triple           | primary pair         | fallback pair        |
| ---------------- | -------------------- | -------------------- |
| `alive` period 0 | 62220 + 1888 = 64108 | 62348 + 1888 = 64236 |
| `alive` period 1 | 62220 + 1760 = 63980 | 62348 + 1760 = 64108 |
| `alive` period 2 | 62156 + 1496 = 63652 | no fallback          |
| `dead` period 3  | 521 + 1232 = 1753    | no fallback          |

A pair is the variant's compiler reservation plus the bytes it leaves resident.

The two `alive` candidates that donate share one primary key and one fallback
key, so the budgets below separate them through their residencies rather than
through their reservations.

Both donating triples have a frontier of length one, because `cell` reaches its
full extent at 2. So the "a rejected candidate's successor wins" observation is
not realisable on this fixture: a refusal exhausts the triple instead of opening
another wave. What still holds, and is what the eager reference checks here, is
that the ranked winner is the same either way, because `max(P, F) <= B` requires
`P <= B`: a candidate the lazy pass refuses on its donating variant alone is a
candidate the paired admission would have refused too.
"""

import re
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from _lcm.solution import backward_induction
from lcm.exceptions import ExecutionPlanningError
from tests.solution._candidate_census import (
    FALLBACK,
    PRIMARY,
    Census,
    CensusRecorder,
    Triple,
    WidthKey,
    donor_pair_model,
    solve_donor_pair,
)

# Every byte budget below is a census constant read at fp64: reservations and
# residencies halve at fp32, so the budgets would refuse nothing there and the
# tests would measure the precision rather than the search.
pytestmark = pytest.mark.usefixtures("x64_enabled")

ALIVE_0 = ("alive", 0, "main")
ALIVE_1 = ("alive", 1, "main")
ALIVE_2 = ("alive", 2, "main")
DEAD_3 = ("dead", 3, "main")
CELL_2 = (("cell", 2),)

# Below `alive` period 1's primary pair, so both donating primaries are refused.
BOTH_DONATING_PRIMARIES_REFUSED = 63979

# Between the two donating primaries' pairs: period 1 survives, period 0 does not.
ONLY_THE_FIRST_DONATING_PRIMARY_REFUSED = 64107

# Above every primary pair and above period 1's fallback pair, below period 0's.
ONLY_THE_FIRST_FALLBACK_REFUSED = 64164

# Far above every measured pair, so the widest candidate of each triple wins.
EVERY_CANDIDATE_ADMITTED = 10**8


def observed_census(
    *,
    monkeypatch: pytest.MonkeyPatch,
    device_memory_bytes: int | None,
    axis_widths: Mapping[str, int] = MappingProxyType({"cell": 2}),
) -> Census:
    """Solve the donor-pair fixture under one budget and return its census."""
    recorder = CensusRecorder()
    recorder.install(monkeypatch=monkeypatch)
    solve_donor_pair(
        model=donor_pair_model(
            device_memory_bytes=device_memory_bytes, axis_widths=axis_widths
        )
    )
    return recorder.census()


def refused_census(
    *, monkeypatch: pytest.MonkeyPatch, device_memory_bytes: int
) -> tuple[Census, str]:
    """Solve under a budget that refuses a candidate; return census and message."""
    recorder = CensusRecorder()
    recorder.install(monkeypatch=monkeypatch)
    with pytest.raises(ExecutionPlanningError) as refusal:
        solve_donor_pair(
            model=donor_pair_model(
                device_memory_bytes=device_memory_bytes, axis_widths={"cell": 2}
            )
        )
    return recorder.census(), str(refusal.value)


def eager_first_feasible(
    *, census: Census, budget_bytes: int
) -> dict[Triple, WidthKey]:
    """Pick each triple's widest candidate whose every variant fits the budget.

    The reference walks the measured rows in the order they were taken, which is
    the ranked frontier order, and keeps each variant paired with its own
    reservation and its own residency.
    """
    pairs: dict[tuple[Triple, WidthKey], int] = {}
    order: list[tuple[Triple, WidthKey]] = []
    for row in census.rows:
        candidate = (row.triple, row.widths)
        if candidate not in pairs:
            order.append(candidate)
        pairs[candidate] = max(pairs.get(candidate, 0), row.reservation + row.residency)
    selected: dict[Triple, WidthKey] = {}
    for triple, widths in order:
        if triple not in selected and pairs[(triple, widths)] <= budget_bytes:
            selected[triple] = widths
    return selected


def test_primary_rejection_skips_that_candidates_fallback_request(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate refused on its donating variant asks for no fallback at all."""
    census, _ = refused_census(
        monkeypatch=monkeypatch,
        device_memory_bytes=BOTH_DONATING_PRIMARIES_REFUSED,
    )

    assert census.rows_for(role=PRIMARY), "The solve measured no donating variant."
    assert census.keys_in(role=FALLBACK) == ()


def test_fallback_rejection_uses_the_fallbacks_own_pairing(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback refusal reads the fallback's own reservation and residency."""
    census, _ = refused_census(
        monkeypatch=monkeypatch, device_memory_bytes=ONLY_THE_FIRST_FALLBACK_REFUSED
    )

    rows = [row for row in census.rows_for(role=FALLBACK) if row.triple == ALIVE_0]
    assert rows, "The refused candidate's fallback was never measured."
    assert [(row.reservation, row.residency) for row in rows] == [(62348, 1888)]


def test_selected_donating_winner_has_an_admitted_fallback(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each selected donating winner carries its own measured, donation-free twin."""
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=EVERY_CANDIDATE_ADMITTED
    )

    assert census.donation_fallbacks, "The solve selected no donation-free fallback."
    assert all(
        fallback.compiled is not census.selected_cores[triple].compiled
        and fallback.donated_arguments == ()
        and any(
            row.triple == triple
            and row.widths == census.selected_widths[triple]
            and row.role == FALLBACK
            for row in census.rows
        )
        for triple, fallback in census.donation_fallbacks.items()
    )


def test_shared_fallback_compiles_once_for_the_surviving_triple(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback key two triples share is lowered for the triple still in play."""
    census, _ = refused_census(
        monkeypatch=monkeypatch,
        device_memory_bytes=ONLY_THE_FIRST_DONATING_PRIMARY_REFUSED,
    )

    calls = [call for call in census.calls if call.role == FALLBACK]
    assert calls, "The solve issued no fallback wave call."
    assert [call.candidates for call in calls] == [((ALIVE_1, CELL_2),)]


@pytest.mark.skipif(
    not hasattr(backward_induction, "_solve_executable_cache"),
    reason="solve executable cache not implemented",
)
def test_cached_fallback_variant_is_reused_without_skipping_residency(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second solve reuses the cached fallback and still measures its residency."""
    observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=EVERY_CANDIDATE_ADMITTED
    )
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=EVERY_CANDIDATE_ADMITTED
    )

    assert census.rows_for(role=FALLBACK), "The second solve measured no fallback."
    assert census.keys_in(role=FALLBACK) == ()


def test_required_primary_compile_error_propagates(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure lowering a donating variant reaches the caller unchanged."""

    def fail(**_kwargs: object) -> None:
        raise RuntimeError("primary lowering refused")

    monkeypatch.setattr(backward_induction, "_lower_and_compile_wave", fail)

    with pytest.raises(RuntimeError, match="primary lowering refused"):
        solve_donor_pair(
            model=donor_pair_model(
                device_memory_bytes=EVERY_CANDIDATE_ADMITTED, axis_widths={"cell": 2}
            )
        )


def test_required_fallback_compile_error_propagates(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure lowering a donation-free variant reaches the caller unchanged."""
    wave = backward_induction._lower_and_compile_wave
    calls: list[int] = []

    def fail_on_the_fallback_call(**kwargs: object) -> None:
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("fallback lowering refused")
        wave(**kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(
        backward_induction, "_lower_and_compile_wave", fail_on_the_fallback_call
    )

    with pytest.raises(RuntimeError, match="fallback lowering refused"):
        solve_donor_pair(
            model=donor_pair_model(
                device_memory_bytes=EVERY_CANDIDATE_ADMITTED, axis_widths={"cell": 2}
            )
        )


def test_malformed_required_report_propagates(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reservation report the engine cannot read reaches the caller unchanged."""

    def malformed(**_kwargs: object) -> object:
        raise ValueError("reservation report is malformed")

    monkeypatch.setattr(backward_induction, "compiler_memory_reservation", malformed)

    with pytest.raises(ValueError, match="reservation report is malformed"):
        solve_donor_pair(
            model=donor_pair_model(
                device_memory_bytes=EVERY_CANDIDATE_ADMITTED, axis_widths={"cell": 2}
            )
        )


def test_unbudgeted_wave_zero_compiles_both_variants(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a budget one wave lowers every triple's primary and its fallback."""
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=None, axis_widths={}
    )

    assert census.keys_in(role=PRIMARY), "The unbudgeted solve lowered no primary."
    assert len(census.keys_in(role=FALLBACK)) == 1


def test_no_donation_route_unchanged(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate naming no fallback is measured exactly once, as a primary."""
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=EVERY_CANDIDATE_ADMITTED
    )

    without = [row for row in census.rows if not row.has_fallback]
    assert without, "Every measured candidate named a fallback."
    assert [row.role for row in without] == [PRIMARY] * len(without)


def test_lazy_selection_matches_the_eager_reference(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatched widths are the ones an eager pass over the census picks."""
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=EVERY_CANDIDATE_ADMITTED
    )

    assert census.selected_widths, "The solve dispatched no core."
    assert dict(census.selected_widths) == eager_first_feasible(
        census=census, budget_bytes=EVERY_CANDIDATE_ADMITTED
    )


@pytest.mark.parametrize(
    ("budget_bytes", "expected"),
    [
        (
            ONLY_THE_FIRST_FALLBACK_REFUSED,
            {ALIVE_1: CELL_2, ALIVE_2: CELL_2, DEAD_3: CELL_2},
        ),
        (ONLY_THE_FIRST_DONATING_PRIMARY_REFUSED, {ALIVE_2: CELL_2, DEAD_3: CELL_2}),
    ],
)
def test_eager_reference_drops_the_triples_a_refusing_budget_exhausts(
    *,
    monkeypatch: pytest.MonkeyPatch,
    budget_bytes: int,
    expected: Mapping[Triple, WidthKey],
) -> None:
    """A budget that exhausts a triple leaves it out of the eager reference.

    Under `ONLY_THE_FIRST_FALLBACK_REFUSED` the `alive` period 0 candidate is the
    only one whose fallback pair exceeds the budget; under
    `ONLY_THE_FIRST_DONATING_PRIMARY_REFUSED` both donating candidates exceed it.
    Every surviving triple keeps the widths it was measured at. The guard assert
    is that the engine refused a triple the reference also dropped, which is what
    makes the two verdicts comparable at all.
    """
    census, message = refused_census(
        monkeypatch=monkeypatch, device_memory_bytes=budget_bytes
    )

    assert f"Regime {ALIVE_0[0]!r} at period {ALIVE_0[1]}" in message
    assert eager_first_feasible(census=census, budget_bytes=budget_bytes) == expected


def test_exhaustion_message_counts_skipped_evaluations(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal names how many candidates skipped their fallback evaluation."""
    _, message = refused_census(
        monkeypatch=monkeypatch,
        device_memory_bytes=BOTH_DONATING_PRIMARIES_REFUSED,
    )

    assert "never compiled" not in message
    assert re.search(
        r"fallback evaluation skipped for \d+ rejected candidate",
        message,
        flags=re.IGNORECASE,
    )
