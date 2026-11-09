"""A shock average over one value grid, composed from the production stencil.

Production composes `shock_reading_stencil` and `apply_averaging` itself, inside
the batched reader. The tests check the stencil and the averaging through this
one-shot form with unscaled weights.
"""

from collections.abc import Callable

from _lcm.regime_building.shock_average import apply_averaging, shock_reading_stencil
from lcm.typing import Float1D, FloatND


def average_over_shock(
    *,
    values: FloatND,
    points: Float1D,
    coordinate: Callable[[FloatND], FloatND],
    shocks: Float1D,
    weights: Float1D,
) -> tuple[Float1D, FloatND]:
    """Average values over a shock added to the last axis's coordinate.

    Args:
        values: Values on a grid, the shocked axis last.
        points: The grid points of the last axis, ascending.
        coordinate: Map from a value of the last axis to its fractional grid
            coordinate, as the value function's interpolator computes it.
        shocks: One shock per node of the averaged draws.
        weights: The nodes' probabilities.

    Returns:
        Tuple of the merged points, ascending, and the average at those points,
        the last axis of `values` replaced by them. A point that coincides with
        another is kept once; its copy moves above the largest point, where the
        average is linear, so there are always `len(points) * len(shocks)` merged
        points, plus one anchor below and one above them
        (`shock_average._merged_knots`).

    """
    knots, stencil = shock_reading_stencil(
        points=points, coordinate=coordinate, shocks=shocks
    )
    return knots, apply_averaging(
        values=values,
        stencil=stencil,
        coefficients=weights,
        shifts=None,
        subscripts="...n,nzm->...mz",
    )
