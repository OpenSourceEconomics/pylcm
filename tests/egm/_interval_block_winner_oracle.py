"""Independent exact oracle for partition-invariant interval winner reduction.

This module is deliberately standard-library only.  It states the total orders in
terms of exact ``Fraction`` fields and implements them twice: a one-shot selection
and a block fold that re-enters the standing winner at an arbitrary local slot.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Literal

Ordering = Literal["ordinary", "certified"]
Identity = Literal["stable", "position"]
CarrySlot = Literal["head", "middle", "tail"]


@dataclass(frozen=True)
class Candidate:
    """One represented affine candidate and its already-computed ordering fields."""

    stable_index: int
    value: Fraction
    right_available: bool
    slope_high: Fraction
    slope_low: Fraction = Fraction(0)
    brackets: bool = True
    policy: Fraction = Fraction(0)
    marginal: Fraction = Fraction(0)


def _primary_rank(*, candidate: Candidate, ordering: Ordering) -> tuple[object, ...]:
    """Return all rank fields except the final identity field."""
    if ordering == "ordinary":
        return (
            candidate.value,
            candidate.right_available,
            candidate.slope_high,
            candidate.slope_low,
        )
    return (candidate.value, candidate.right_available, candidate.slope_high)


def _rank(
    *,
    candidate: Candidate,
    ordering: Ordering,
    identity: Identity,
    position: int,
) -> tuple[object, ...]:
    """Return a largest-wins total-order key.

    The contract says the smaller stable global index wins the final tie, hence
    its negation is the final largest-wins field.  ``identity="position"`` is the
    deliberately defective positive control: it substitutes the candidate's
    current local slot, which changes when the standing winner re-enters a block.
    """
    final = candidate.stable_index if identity == "stable" else position
    return (*_primary_rank(candidate=candidate, ordering=ordering), -final)


def one_shot_winner(
    *, candidates: Sequence[Candidate], ordering: Ordering
) -> Candidate | None:
    """Select once over all represented candidates using stable global identity."""
    eligible = [candidate for candidate in candidates if candidate.brackets]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: _rank(
            candidate=candidate,
            ordering=ordering,
            identity="stable",
            position=0,
        ),
    )


def blocked_winner(
    *,
    blocks: Sequence[Sequence[Candidate]],
    ordering: Ordering,
    identity: Identity,
    carry_slots: Sequence[CarrySlot] = (),
) -> Candidate | None:
    """Fold arbitrary blocks while re-entering the standing winner.

    A block with no eligible candidate is an identity element.  ``carry_slots``
    controls where the standing winner appears in each later block, exposing any
    accidental dependence on local operand position.
    """
    held: Candidate | None = None
    for block_number, raw_block in enumerate(blocks):
        block = list(raw_block)
        if held is not None:
            slot = (
                carry_slots[block_number - 1]
                if block_number - 1 < len(carry_slots)
                else "head"
            )
            insertion = {
                "head": 0,
                "middle": len(block) // 2,
                "tail": len(block),
            }[slot]
            block.insert(insertion, held)
        eligible = [
            (position, candidate)
            for position, candidate in enumerate(block)
            if candidate.brackets
        ]
        if eligible:
            held = max(
                eligible,
                key=lambda item: _rank(
                    candidate=item[1],
                    ordering=ordering,
                    identity=identity,
                    position=item[0],
                ),
            )[1]
    return held


def compositions(total: int) -> Iterator[tuple[int, ...]]:
    """Yield every ordered positive composition of ``total``."""
    if total == 0:
        yield ()
        return
    for cuts in product((False, True), repeat=total - 1):
        sizes: list[int] = []
        run = 1
        for cut in cuts:
            if cut:
                sizes.append(run)
                run = 1
            else:
                run += 1
        sizes.append(run)
        yield tuple(sizes)


def partition_by_sizes(
    *, candidates: Sequence[Candidate], sizes: Sequence[int]
) -> tuple[tuple[Candidate, ...], ...]:
    """Split one candidate ordering according to an ordered composition."""
    blocks: list[tuple[Candidate, ...]] = []
    offset = 0
    for size in sizes:
        blocks.append(tuple(candidates[offset : offset + size]))
        offset += size
    if offset != len(candidates):
        raise ValueError("partition sizes do not consume the candidate sequence")
    return tuple(blocks)


def winner_signature(candidate: Candidate | None) -> tuple[object, ...] | None:
    """Identity plus all published channels, suitable for bit/exact comparison."""
    if candidate is None:
        return None
    return (
        candidate.stable_index,
        candidate.value,
        candidate.policy,
        candidate.marginal,
    )


def every_carry_slot(n_transitions: int) -> Iterable[tuple[CarrySlot, ...]]:
    """Yield head/middle/tail placements without an exponential test explosion."""
    if n_transitions <= 0:
        yield ()
        return
    slots: tuple[CarrySlot, ...] = ("head", "middle", "tail")
    for slot in slots:
        yield tuple(slot for _ in range(n_transitions))
    yield tuple(slots[index % 3] for index in range(n_transitions))
