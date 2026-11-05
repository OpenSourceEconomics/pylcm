"""A candidate's donation-free fallback is compiled only once it is still in play.

Every budget here is derived at test time from the census of the donor-pair
fixture at `cell=2` under `ResultRetention.VALUES`, the census
`tests/solution/test_candidate_census.py` records. At the time of writing it
reads, in bytes:

| triple           | primary pair         | fallback pair        |
| ---------------- | -------------------- | -------------------- |
| `alive` period 0 | 38020 + 1704 = 39724 | 38148 + 1704 = 39852 |
| `alive` period 1 | 38020 + 1576 = 39596 | 38148 + 1576 = 39724 |
| `alive` period 2 | 37508 + 1312 = 38820 | no fallback          |
| `dead` period 3  | 608 + 1048 = 1656    | no fallback          |

A pair is the variant's compiler reservation plus the bytes it leaves resident.
Each budget sits a fixed distance from the pair whose refusal it tests, and the
census must still order the pairs the way that budget needs, so a change in
program size moves the budgets with it instead of turning a refusal test into
one that refuses nothing.

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

import dataclasses
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Never, Unpack

import pytest

from _lcm.solution import backward_induction
from lcm.exceptions import ExecutionPlanningError
from tests.solution._callback_types import LowerWaveKwargs
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

# Every byte budget below is derived from a census taken at fp64: the distances
# between pairs were chosen there, and reservations and residencies halve at
# fp32, so the tests would measure the precision rather than the search.
pytestmark = pytest.mark.usefixtures("x64_enabled")

ALIVE_0 = ("alive", 0, "main")
ALIVE_1 = ("alive", 1, "main")
ALIVE_2 = ("alive", 2, "main")
DEAD_3 = ("dead", 3, "main")
CELL_2 = (("cell", 2),)

# Far above every measured pair, so the widest candidate of each triple wins.
EVERY_CANDIDATE_ADMITTED = 10**8


@dataclasses.dataclass(frozen=True, kw_only=True)
class Budgets:
    """Byte budgets placed around the donor-pair census's pairs."""

    both_donating_primaries_refused: int
    """Below `alive` period 1's primary pair, above period 2's."""
    only_the_first_donating_primary_refused: int
    """Between the two donating primaries' pairs: period 1 survives, 0 does not."""
    only_the_first_fallback_refused: int
    """Above every primary pair and period 1's fallback pair, below period 0's."""
    first_fallback_variant: tuple[int, int]
    """Reservation and residency of `alive` period 0's fallback variant."""


@pytest.fixture(name="budgets")
def _fixture_budgets(*, request: pytest.FixtureRequest) -> Budgets:
    """Place the budgets around the pairs of an unrefused census at fp64."""
    request.getfixturevalue("x64_enabled")
    with pytest.MonkeyPatch.context() as patch:
        census = observed_census(
            monkeypatch=patch, device_memory_bytes=EVERY_CANDIDATE_ADMITTED
        )
    rows = {(row.triple, row.role): row for row in census.rows}

    def pair(*, triple: Triple, role: str) -> int:
        row = rows[(triple, role)]
        return row.reservation + row.residency

    primary_0 = pair(triple=ALIVE_0, role=PRIMARY)
    primary_1 = pair(triple=ALIVE_1, role=PRIMARY)
    fallback_0 = pair(triple=ALIVE_0, role=FALLBACK)
    fallback_1 = pair(triple=ALIVE_1, role=FALLBACK)
    budgets = Budgets(
        both_donating_primaries_refused=primary_1 - 21,
        only_the_first_donating_primary_refused=primary_0 - 21,
        only_the_first_fallback_refused=fallback_1 + 56,
        first_fallback_variant=(
            rows[(ALIVE_0, FALLBACK)].reservation,
            rows[(ALIVE_0, FALLBACK)].residency,
        ),
    )
    undonating = max(
        pair(triple=ALIVE_2, role=PRIMARY), pair(triple=DEAD_3, role=PRIMARY)
    )
    assert undonating <= budgets.both_donating_primaries_refused < primary_1
    assert primary_1 <= budgets.only_the_first_donating_primary_refused < primary_0
    assert max(primary_0, fallback_1) <= budgets.only_the_first_fallback_refused
    assert budgets.only_the_first_fallback_refused < fallback_0
    return budgets


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
    *, monkeypatch: pytest.MonkeyPatch, budgets: Budgets
) -> None:
    """A candidate refused on its donating variant asks for no fallback at all."""
    census, _ = refused_census(
        monkeypatch=monkeypatch,
        device_memory_bytes=budgets.both_donating_primaries_refused,
    )

    assert census.rows_for(role=PRIMARY), "The solve measured no donating variant."
    assert census.keys_in(role=FALLBACK) == ()


def test_fallback_rejection_uses_the_fallbacks_own_pairing(
    *, monkeypatch: pytest.MonkeyPatch, budgets: Budgets
) -> None:
    """A fallback refusal reads the fallback's own reservation and residency."""
    census, _ = refused_census(
        monkeypatch=monkeypatch,
        device_memory_bytes=budgets.only_the_first_fallback_refused,
    )

    rows = [row for row in census.rows_for(role=FALLBACK) if row.triple == ALIVE_0]
    assert rows, "The refused candidate's fallback was never measured."
    assert [(row.reservation, row.residency) for row in rows] == [
        budgets.first_fallback_variant
    ]


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
    *, monkeypatch: pytest.MonkeyPatch, budgets: Budgets
) -> None:
    """A fallback key two triples share is lowered for the triple still in play."""
    census, _ = refused_census(
        monkeypatch=monkeypatch,
        device_memory_bytes=budgets.only_the_first_donating_primary_refused,
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

    def fail[Ignored](**_kwargs: Ignored) -> Never:
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

    def fail_on_the_fallback_call(**kwargs: Unpack[LowerWaveKwargs]) -> None:
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("fallback lowering refused")
        wave(**kwargs)

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

    def malformed[Ignored](**_kwargs: Ignored) -> Never:
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
    ("budget_name", "expected"),
    [
        (
            "only_the_first_fallback_refused",
            {ALIVE_1: CELL_2, ALIVE_2: CELL_2, DEAD_3: CELL_2},
        ),
        ("only_the_first_donating_primary_refused", {ALIVE_2: CELL_2, DEAD_3: CELL_2}),
    ],
)
def test_eager_reference_drops_the_triples_a_refusing_budget_exhausts(
    *,
    monkeypatch: pytest.MonkeyPatch,
    budgets: Budgets,
    budget_name: str,
    expected: Mapping[Triple, WidthKey],
) -> None:
    """A budget that exhausts a triple leaves it out of the eager reference.

    Under `only_the_first_fallback_refused` the `alive` period 0 candidate is the
    only one whose fallback pair exceeds the budget; under
    `only_the_first_donating_primary_refused` both donating candidates exceed it.
    Every surviving triple keeps the widths it was measured at. The guard assert
    is that the engine refused a triple the reference also dropped, which is what
    makes the two verdicts comparable at all.
    """
    budget_bytes = getattr(budgets, budget_name)
    census, message = refused_census(
        monkeypatch=monkeypatch, device_memory_bytes=budget_bytes
    )

    assert f"Regime {ALIVE_0[0]!r} at period {ALIVE_0[1]}" in message
    assert eager_first_feasible(census=census, budget_bytes=budget_bytes) == expected


def test_exhaustion_message_counts_skipped_evaluations(
    *, monkeypatch: pytest.MonkeyPatch, budgets: Budgets
) -> None:
    """A refusal names how many candidates skipped their fallback evaluation."""
    _, message = refused_census(
        monkeypatch=monkeypatch,
        device_memory_bytes=budgets.both_donating_primaries_refused,
    )

    assert "never compiled" not in message
    assert re.search(
        r"fallback evaluation skipped for \d+ rejected candidate",
        message,
        flags=re.IGNORECASE,
    )
