r"""Exact owner sequence of the affine envelope inside one node cell.

A node cell is covered by at most one affine link from each monotone run. The
upper envelope is therefore the maximum of a fixed, small set of rational lines
whose coefficients are determined by stored IEEE endpoint operands.

The owner sequence and the states at which it changes are structural predicates:
a rounded product may neither invent a tie nor move a right-continuous handover
by one representable state. They are consequently resolved together by the
fixed-width exact kernel in ``_exact_affine``. Every float is decoded as a signed
dyadic, comparisons are integer determinants, and a crossing is rounded directly
to the least IEEE state at or above its exact rational root.

Keeping the whole cell inside one FFI call is part of the contract. Expanding the
integer limbs, or calling one exact product at each arithmetic site, would make
the traced solve scale with the representation and recreate the compile failure
that motivated this boundary. The JAX program instead sees one operation per
cell regardless of limb count or owner-walk length.
"""

from _lcm.egm.upper_envelope._exact_affine.ffi import exact_cell_hull
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarBool


def hull_owners(
    *,
    left: FloatND,
    right: FloatND,
    live: BoolND,
    low: IntND,
    high: IntND,
    endog_grid: Float1D,
    value: Float1D,
    max_runs: int,
) -> tuple[Float1D, Int1D, ScalarBool]:
    """Return exact breakpoints and owners across one node cell.

    Args:
        left: Abscissa of the cell's left edge.
        right: Abscissa of the cell's right edge.
        live: Boolean ``(max_runs,)`` mask; whether each run covers the cell.
        low: Candidate index of each link's lower endpoint, ``(max_runs,)``.
        high: Candidate index of each link's upper endpoint, ``(max_runs,)``.
        endog_grid: Candidate endogenous grid points in producer order.
        value: Candidate value-correspondence points at ``endog_grid``.
        max_runs: Static capacity for the number of monotone runs.

    Returns:
        The ``max_runs + 1`` weakly ascending breakpoints, the ``max_runs``
        owner indices, and whether exact resolution failed. Slots after the last
        positive-width piece are parked at ``right`` and repeat the last owner.

    """
    bounds, owners, status = exact_cell_hull(
        left=left,
        right=right,
        live=live,
        low=low,
        high=high,
        endog_grid=endog_grid,
        value=value,
        max_runs=max_runs,
    )
    return bounds, owners, status != 0
