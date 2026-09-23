"""Finite-set reference for legal tile widths, independent of pylcm.

Enumerates the legal integers directly instead of reproducing the production
rounding, so the planner and dispatch can be checked against it.
"""

from collections.abc import Iterator
from dataclasses import dataclass


class NoLegalWidthError(ValueError):
    """The stated ceiling excludes every legal width."""


@dataclass(frozen=True)
class Axis:
    """A tiled axis with its extent and width policy."""

    extent: int
    """Number of elements along the axis."""
    alignment: int = 1
    """Every partial width must be a multiple of this."""
    minimum: int = 1
    """Smallest admissible partial width."""


def legal_widths(*, axis: Axis, ceiling: int | None = None) -> tuple[int, ...]:
    """List legal widths; the full extent is always legal before the ceiling."""
    values = tuple(
        width
        for width in range(1, axis.extent + 1)
        if (
            width == axis.extent
            or (width >= axis.minimum and width % axis.alignment == 0)
        )
        and (ceiling is None or width <= ceiling)
    )
    if not values:
        raise NoLegalWidthError(f"subject: no legal width under ceiling {ceiling}")
    return values


def effective_pin(*, axis: Axis, pin: int, ceiling: int | None = None) -> int:
    """Select the largest legal width not above `pin`, else the smallest legal."""
    allowed = legal_widths(axis=axis, ceiling=ceiling)
    below = [value for value in allowed if value <= pin]
    return max(below) if below else min(allowed)


def cases() -> Iterator[tuple[Axis, int, int]]:
    """Yield a bounded neighbourhood of extents, floors, alignments, pins, caps."""
    for extent in (1, 2, 3, 5, 8, 9, 17):
        for alignment in (1, 2, 4):
            for minimum in sorted({1, min(3, extent), extent}):
                axis = Axis(extent=extent, alignment=alignment, minimum=minimum)
                values = sorted({1, 2, max(1, extent - 1), extent, extent + 3})
                for pin in values:
                    for ceiling in values:
                        yield axis, pin, ceiling
