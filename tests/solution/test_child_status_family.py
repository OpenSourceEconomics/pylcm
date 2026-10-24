"""Synthetic mutation suite for the independent child-slot law.

These mutations test oracle discrimination; the separate public-model witness
tests production.
"""

from dataclasses import replace

import pytest

from tests.solution._child_status_witness import (
    ExecutedRow,
    fold_executed,
    require_publication,
)


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_all",
        "keeper_only",
        "adjuster_only",
        "winner_only",
        "neutral",
        "reverse_slots",
        "drop_counts",
        "deduplicate_counts",
    ],
)
def test_executed_rows_cannot_be_laundered(*, mutation) -> None:
    rows = (
        ExecutedRow("keeper", ((True, False, False), (False, False, False)), (1, 3)),
        ExecutedRow(
            "winning-adjuster", ((False, False, False), (False, True, False)), (2, 4)
        ),
        ExecutedRow(
            "excluded-evaluated-adjuster",
            ((False, False, False), (False, False, True)),
            (2, 4),
        ),
    )
    expected = fold_executed(n_slots=2, rows=rows)
    if mutation == "drop_all":
        bad = ExecutedRow("published", ())
    elif mutation == "keeper_only":
        bad = fold_executed(n_slots=2, rows=rows[:1])
    elif mutation == "adjuster_only":
        bad = fold_executed(n_slots=2, rows=rows[1:])
    elif mutation == "winner_only":
        bad = fold_executed(n_slots=2, rows=rows[1:2])
    elif mutation == "neutral":
        bad = replace(expected, slots=((False, False, False),) * 2)
    elif mutation == "reverse_slots":
        bad = replace(expected, slots=expected.slots[::-1])
    elif mutation == "drop_counts":
        bad = replace(expected, counts=None)
    else:
        bad = replace(expected, counts=(3, 7))
    with pytest.raises(AssertionError):
        require_publication(expected=expected, actual=bad)


def test_all_false_child_slot_still_has_required_identity() -> None:
    expected = fold_executed(n_slots=1, rows=(ExecutedRow("keeper", ((False,) * 3,)),))
    with pytest.raises(AssertionError):
        require_publication(expected=expected, actual=ExecutedRow("missing", ()))


def test_optional_counts_and_zero_padding_remain_neutral() -> None:
    actual = ExecutedRow("keeper", ((False, False, True),), (2, 3))
    zero_padding = ExecutedRow("padding", ((False, False, False),), (0, 0))
    require_publication(
        expected=fold_executed(n_slots=1, rows=(actual,)),
        actual=fold_executed(n_slots=1, rows=(actual, zero_padding)),
    )
    without_counts = replace(actual, counts=None)
    assert fold_executed(n_slots=1, rows=(without_counts,)).counts is None
