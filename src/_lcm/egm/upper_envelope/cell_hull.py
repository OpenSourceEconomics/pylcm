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

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope._exact_affine.ffi import (
    exact_affine_handover,
    exact_cell_hull,
)
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


def _place_handovers(
    *,
    bounds: Float1D,
    owners: Int1D,
    low: IntND,
    high: IntND,
    endog_grid: Float1D,
    value: Float1D,
    left: FloatND,
    right: FloatND,
) -> tuple[Float1D, ScalarBool]:
    """Place a proposed owner sequence on the exact representable event lattice.

    This helper remains separately testable because it states the publication
    convention independently of the owner walk: an incoming line starts at the
    smallest representable state at or above the exact crossing. Parallel lines
    and invalid operands have no unique handover and are reported unresolved.
    Breakpoints where the owner does not change stay where the caller put them.
    """
    outgoing, incoming = owners[:-1], owners[1:]
    a_x0, a_x1 = endog_grid[low[outgoing]], endog_grid[high[outgoing]]
    a_v0, a_v1 = value[low[outgoing]], value[high[outgoing]]
    b_x0, b_x1 = endog_grid[low[incoming]], endog_grid[high[incoming]]
    b_v0, b_v1 = value[low[incoming]], value[high[incoming]]

    candidate, status = exact_affine_handover(
        a_x0=a_x0,
        a_x1=a_x1,
        a_v0=a_v0,
        a_v1=a_v1,
        b_x0=b_x0,
        b_x1=b_x1,
        b_v0=b_v0,
        b_v1=b_v1,
        left=left,
        right=right,
    )
    interior = bounds[1:-1]
    hands_over = outgoing != incoming
    placed = jnp.where(hands_over, jnp.clip(candidate, left, right), interior)

    # A proposed walk can contain several zero-width owners at one approximate
    # event. Exact pairwise roots can round to neighbouring states, but those
    # middle owners still own no published state. Preserve the proposal's groups
    # and enforce monotonicity exactly as the old placement stage did.
    opens_group = jnp.concatenate(
        [jnp.ones_like(interior[:1], dtype=bool), interior[1:] != interior[:-1]]
    )
    placed = jax.lax.cummax(jnp.where(opens_group, placed, -jnp.inf))
    unresolved = jnp.any(hands_over & (status != 0))
    return jnp.concatenate([bounds[:1], placed, bounds[-1:]]), unresolved
