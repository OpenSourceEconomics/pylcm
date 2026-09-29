"""The donor-pair fixture offers a frontier the census can be read off."""

import json
from collections.abc import Callable, Mapping

import pytest

from tests.solution._candidate_census import (
    FALLBACK,
    PRIMARY,
    Census,
    CensusRecorder,
    donor_pair_model,
    solve_donor_pair,
)


def observed_census(
    *,
    monkeypatch: pytest.MonkeyPatch,
    device_memory_bytes: int,
    axis_widths: Mapping[str, int],
) -> Census:
    """Solve the donor-pair fixture under a budget and return its census."""
    recorder = CensusRecorder()
    recorder.install(monkeypatch=monkeypatch)
    solve_donor_pair(
        model=donor_pair_model(
            device_memory_bytes=device_memory_bytes, axis_widths=axis_widths
        )
    )
    return recorder.census()


def census_as_json(*, census: Census) -> str:
    """Render a census as the flat record a test suite property carries."""
    return json.dumps(
        {
            "waves": [
                {
                    "wave": call.wave,
                    "role": call.role,
                    "n_keys": len(call.keys),
                    "candidates": [
                        [list(triple), [list(item) for item in widths]]
                        for triple, widths in call.candidates
                    ],
                }
                for call in census.calls
            ],
            "rows": [
                {
                    "triple": list(row.triple),
                    "widths": [list(item) for item in row.widths],
                    "role": row.role,
                    "reservation": row.reservation,
                    "residency": row.residency,
                    "has_fallback": row.has_fallback,
                    "admitted": row.admitted,
                }
                for row in census.rows
            ],
            "n_unique_keys": len(census.unique_keys),
            "n_donation_fallbacks": len(census.donation_fallbacks),
        }
    )


def test_census_of_the_donor_pair_fixture_is_nonempty(
    *,
    monkeypatch: pytest.MonkeyPatch,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    """Some triple of the donor-pair fixture names a donation-free fallback.

    The fixture is `nbegm_ride_along_toy` at `cell=2` under a 10**8-byte budget
    with `ResultRetention.VALUES`. One wave measures four triples: `alive` at
    periods 0, 1 and 2 and `dead` at period 3, each at `cell=2`. The `alive`
    triples of periods 0 and 1 name a fallback and share one fallback key; their
    primary reservation is 62220 bytes against residencies of 1888 and 1760
    bytes, and their fallback reservation is 62348 bytes against the same
    residencies. The recorded census is what the scheduling tests set their
    budgets from.
    """
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=10**8, axis_widths={"cell": 2}
    )
    record_testsuite_property("donor_pair_census", census_as_json(census=census))

    assert census.rows, "The solve measured no candidate."
    assert census.donation_fallbacks


def test_census_of_the_donor_pair_fixture_pairs_every_fallback_with_a_primary(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every measured fallback variant stands beside its candidate's primary."""
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=10**8, axis_widths={"cell": 2}
    )

    assert census.rows_for(role=FALLBACK), "The solve measured no fallback variant."
    assert {(row.triple, row.widths) for row in census.rows_for(role=FALLBACK)} == {
        (row.triple, row.widths)
        for row in census.rows_for(role=PRIMARY)
        if row.has_fallback
    }


def test_census_of_the_donor_pair_fixture_reads_the_cell_frontier(
    *,
    monkeypatch: pytest.MonkeyPatch,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    """Leaving `cell` unfixed offers more than one width on the `cell` axis.

    The scheduling tests that need a candidate after the rejected one read
    their widths from this census rather than assuming the frontier.
    """
    census = observed_census(
        monkeypatch=monkeypatch, device_memory_bytes=10**8, axis_widths={}
    )
    record_testsuite_property(
        "donor_pair_frontier_census", census_as_json(census=census)
    )

    assert census.rows, "The solve measured no candidate."
    assert census.widths_on_axis(axis="cell")[0] > 0
