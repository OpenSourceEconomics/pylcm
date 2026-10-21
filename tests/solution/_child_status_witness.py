"""Independent ordered-slot fold oracle for the child-status law; stdlib only.

This oracle has no model, solver, JAX, or production fold dependency.  A row is
an actually evaluated keeper or adjuster.  Economic exclusion and losing an
argmax do not make a row unexecuted; true padding is not an input row here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutedRow:
    name: str
    slots: tuple[tuple[bool, ...], ...]
    counts: tuple[int, ...] | None = None


def fold_executed(*, n_slots: int, rows: tuple[ExecutedRow, ...]) -> ExecutedRow:
    """OR each declared slot and add counts over every executed row once."""
    if n_slots < 0 or not rows:
        raise AssertionError("The declaration and executed-row set must be explicit.")
    if any(len(row.slots) != n_slots for row in rows):
        raise AssertionError("An executed row lost or invented a declaration slot.")
    if any(len(slot) != 3 for row in rows for slot in row.slots):
        raise AssertionError("Every query status has exactly three flags.")
    slots = tuple(
        tuple(any(row.slots[i][j] for row in rows) for j in range(3))
        for i in range(n_slots)
    )
    row_counts = tuple(row.counts for row in rows)
    present = tuple(counts for counts in row_counts if counts is not None)
    if present and len(present) != len(rows):
        raise AssertionError("Executed rows disagree on the optional count schema.")
    counts = None
    if present:
        counts = tuple(sum(row[j] for row in present) for j in range(2))
    return ExecutedRow(name="published", slots=slots, counts=counts)


def require_publication(*, expected: ExecutedRow, actual: ExecutedRow) -> None:
    """No neutral fabrication, slot reordering, winner filtering, or lost counts."""
    if actual.slots != expected.slots:
        raise AssertionError("Published ordered child statuses differ from execution.")
    if actual.counts != expected.counts:
        raise AssertionError(
            "Published continuation-read counts differ from execution."
        )


def _self_check() -> None:
    keeper = ExecutedRow(
        "keeper", ((True, False, False), (False, False, False)), (1, 3)
    )
    losing = ExecutedRow("loser", ((False, False, False), (False, True, True)), (2, 4))
    actual = fold_executed(n_slots=2, rows=(keeper, losing))
    assert actual.slots == ((True, False, False), (False, True, True))
    assert actual.counts == (3, 7)
    require_publication(expected=actual, actual=actual)
    neutral = fold_executed(n_slots=1, rows=(ExecutedRow("keeper", ((False,) * 3,)),))
    try:
        require_publication(expected=neutral, actual=ExecutedRow("missing", ()))
    except AssertionError:
        pass
    else:
        raise AssertionError("All-false slots must still be present.")


if __name__ == "__main__":
    _self_check()
