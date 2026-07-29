"""Exact upper envelope of an EGM candidate segment set.

The candidates form a chain of affine value links that folds back on itself in
non-concave regions, so several links cover the same resources and the envelope
is their pointwise maximum. Computing it exactly requires more than comparing
candidates at their own abscissae: a branch can own an interior subinterval while
winning at no candidate node, and it is then dropped entirely — value and policy
both wrong on a positive-width interval.

The construction here is therefore driven by *cells*, not by nodes:

1. the chain is normalized into maximal x-monotone runs, so "which links belong
   to one branch" is exact rather than inferred from a value heuristic;
2. sorting the live abscissae partitions resources into node cells. Inside a node
   cell no run starts or ends, so each run spans the whole cell, contributes one
   full line there, and the envelope is the maximum of at most `max_runs` lines —
   which makes it convex, with its owners in order of increasing slope;
3. `cell_hull.hull_owners` returns that cell's owner sequence and breakpoints,
   with every live line certified against the owners. A branch owning only an
   interior subinterval is one of those pieces;
4. each piece is an open interval on which one link owns the envelope.

Convexity is what keeps the cost linear in the branch count. Splitting a cell at
every pairwise crossing instead would give `max_runs**2` sub-cells and cost a
further `max_runs` to find each one's owner, which is what makes a capacity wide
enough for a repeatedly folding model unaffordable; certifying each owner against
every rival pairwise would cost as much again.

Emission then follows from ownership alone. At a sub-cell boundary the readings
from the owner on either side are compared: equal readings are one point of a
continuous envelope and emit a single record, different readings are a genuine
kink and emit exactly two (left owner, then right owner), which preserves the
policy discontinuity in a weakly ascending row. A crossing that coincides with a
node needs no special case — it is simply a boundary whose two sides have
different owners.

Equal published values buy nothing. A row carries a policy and a marginal as
well as a value, so which link owns the interval beside a boundary still matters
even where two of them round to the same value there. Ownership is therefore
settled by certified comparison throughout, never by a rounded reading. Where a
sign cannot be certified, where a live link escapes certification against the
owners, or where the realized run count exceeds the static capacity, the row is
NaN-poisoned and reported as an overflow so the solve loop's diagnostics name the
offending cell instead of publishing a guess.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.cell_hull import hull_owners
from _lcm.egm.upper_envelope.topology import count_linked_runs, monotone_run_ids
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarBool, ScalarInt

# Static capacity for the number of x-monotone runs a candidate chain may fold
# into. One discrete action can fold arbitrarily often, so this is a validated
# capacity, never an assumed bound: exceeding it poisons the row.
DEFAULT_MAX_RUNS: int = 24

# How many node cells to resolve at once. Cells are independent, so this changes
# only the working set, never a published value. Small enough to keep the peak
# off the whole cell axis, large enough that the per-batch fixed cost stays
# amortized; a chain with fewer cells than this resolves in a single batch and
# pays nothing for the knob.
DEFAULT_CELL_BATCH_SIZE: int = 32


def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_batch_size: int | None = DEFAULT_CELL_BATCH_SIZE,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Refine a candidate value correspondence to its exact upper envelope.

    Args:
        endog_grid: Candidate endogenous grid points (resources) in producer
            order; dead candidates are NaN.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        n_refined: Static length of the refined output arrays.
        segment_id: Optional per-candidate branch label. Runs are always split
            where resources stop increasing; a label change splits them further.
        max_runs: Static capacity for the number of x-monotone runs.
        cell_batch_size: How many node cells to resolve at once; `None` resolves
            the whole axis in one go. Cells are independent, so this partitions
            the work without changing any published value.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each of
        length `n_refined`, NaN-padded), and the number of envelope points
        `n_kept`. `n_kept > n_refined` signals overflow — either genuine capacity
        overflow or a decision that could not be certified — and the arrays are
        then not publishable.

    """
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)
    run_id = monotone_run_ids(endog_grid=endog_grid, dead=dead, segment_id=segment_id)
    n_runs = count_linked_runs(endog_grid=endog_grid, dead=dead, segment_id=segment_id)

    cell_left, cell_right, cell_live = _node_cells(endog_grid=endog_grid, dead=dead)
    active = _active_link_per_run(
        endog_grid=endog_grid,
        run_id=run_id,
        cell_left=cell_left,
        cell_live=cell_live,
        max_runs=max_runs,
    )

    sub_cells = _sub_cells_per_node_cell(
        cell_left=cell_left,
        cell_right=cell_right,
        active=active,
        endog_grid=endog_grid,
        value=value,
        max_runs=max_runs,
        cell_batch_size=cell_batch_size,
    )

    out_grid, out_policy, out_value, n_kept = _emit_envelope(
        sub_cells=sub_cells,
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
    )

    uncertain = sub_cells.poisoned | (n_runs > max_runs)
    overflow_marker = jnp.asarray(n_refined + 1, dtype=jnp.int32)
    n_kept = jnp.where(uncertain, overflow_marker, n_kept)
    poison = jnp.full((), jnp.nan, dtype=endog_grid.dtype)
    out_grid = jnp.where(uncertain, poison, out_grid)
    out_policy = jnp.where(uncertain, poison, out_policy)
    out_value = jnp.where(uncertain, poison, out_value)
    return out_grid, out_policy, out_value, n_kept


class _ActiveLinks:
    """Per node cell, the one link each run contributes to that cell."""

    def __init__(self, *, live: BoolND, left_index: IntND, right_index: IntND) -> None:
        self.live = live
        """Boolean `(max_runs, n_cells)`; whether the run covers the cell."""
        self.left_index = left_index
        """Candidate index of the link's lower endpoint, `(max_runs, n_cells)`."""
        self.right_index = right_index
        """Candidate index of the link's upper endpoint, `(max_runs, n_cells)`."""


class _SubCells:
    """The open intervals the envelope is piecewise affine on, with their owners."""

    def __init__(
        self,
        *,
        left: FloatND,
        right: FloatND,
        owner_left_index: IntND,
        owner_right_index: IntND,
        live: BoolND,
        poisoned: ScalarBool,
    ) -> None:
        self.left = left
        """Left boundary of each sub-cell, flattened in ascending order."""
        self.right = right
        """Right boundary of each sub-cell, flattened in ascending order."""
        self.owner_left_index = owner_left_index
        """Candidate index of the owning link's lower endpoint."""
        self.owner_right_index = owner_right_index
        """Candidate index of the owning link's upper endpoint."""
        self.live = live
        """Whether the sub-cell spans a positive-width interval with an owner."""
        self.poisoned = poisoned
        """Whether any ownership decision could not be certified."""


def _node_cells(
    *, endog_grid: Float1D, dead: BoolND
) -> tuple[Float1D, Float1D, BoolND]:
    """Partition resources at the live candidate abscissae.

    No run starts or ends strictly inside such a cell, so each run contributes at
    most one link to it.
    """
    sort_key = jnp.where(dead, jnp.inf, endog_grid)
    sorted_x = jnp.sort(sort_key)
    left = sorted_x[:-1]
    right = sorted_x[1:]
    live = jnp.isfinite(left) & jnp.isfinite(right) & (right > left)
    return left, right, live


def _active_link_per_run(
    *,
    endog_grid: Float1D,
    run_id: Int1D,
    cell_left: Float1D,
    cell_live: BoolND,
    max_runs: int,
) -> _ActiveLinks:
    """Locate, for every run and node cell, the link covering that cell."""
    n_candidates = endog_grid.shape[0]

    def locate(run: ScalarInt) -> tuple[BoolND, IntND, IntND]:
        in_run = run_id == run
        key = jnp.where(in_run, endog_grid, jnp.inf)
        order = jnp.argsort(key).astype(jnp.int32)
        run_x = key[order]
        n_nodes = jnp.sum(in_run, dtype=jnp.int32)
        # The cell's left boundary is itself a node abscissa, so a right-side
        # search lands one past the node opening the covering link.
        position = (
            jnp.searchsorted(run_x, cell_left, side="right").astype(jnp.int32) - 1
        )
        live = cell_live & (position >= 0) & (position <= n_nodes - 2)
        safe = jnp.clip(position, 0, n_candidates - 2)
        return live, order[safe].astype(jnp.int32), order[safe + 1].astype(jnp.int32)

    live, left_index, right_index = jax.vmap(locate)(
        jnp.arange(max_runs, dtype=jnp.int32)
    )
    return _ActiveLinks(live=live, left_index=left_index, right_index=right_index)


def _sub_cells_per_node_cell(
    *,
    cell_left: Float1D,
    cell_right: Float1D,
    active: _ActiveLinks,
    endog_grid: Float1D,
    value: Float1D,
    max_runs: int,
    cell_batch_size: int | None,
) -> _SubCells:
    """Resolve the envelope's owners across every node cell.

    Ownership is decided on value alone; the policy is read afterwards from the
    owning link, so it never influences which branch wins.

    Cells are independent, so `cell_batch_size` partitions them without changing
    anything published. What it does change is the working set: resolving the
    whole axis at once puts a `n_cells * max_runs` intermediate in flight per
    row, and rows are themselves mapped over, so the peak carries the product of
    all three. Chunking replaces the cell factor with the batch size.
    """

    def split(
        left: FloatND,
        right: FloatND,
        live: BoolND,
        low: IntND,
        high: IntND,
    ) -> tuple[FloatND, FloatND, IntND, IntND, BoolND, ScalarBool]:
        bounds, owner, unresolved = hull_owners(
            left=left,
            right=right,
            live=live,
            low=low,
            high=high,
            endog_grid=endog_grid,
            value=value,
            max_runs=max_runs,
        )
        sub_left = bounds[:-1]
        sub_right = bounds[1:]
        sub_live = (sub_right > sub_left) & jnp.any(live)
        return (
            sub_left,
            sub_right,
            low[owner],
            high[owner],
            sub_live,
            unresolved,
        )

    # `lax.map` maps over a leading axis, so the per-run arrays are transposed to
    # put cells first; `vmap` read them along axis 1 instead.
    per_cell = (
        cell_left,
        cell_right,
        active.live.T,
        active.left_index.T,
        active.right_index.T,
    )
    sub_left, sub_right, owner_low, owner_high, sub_live, unresolved = jax.lax.map(
        lambda cell: split(*cell), per_cell, batch_size=cell_batch_size
    )

    return _SubCells(
        left=sub_left.reshape(-1),
        right=sub_right.reshape(-1),
        owner_left_index=owner_low.reshape(-1),
        owner_right_index=owner_high.reshape(-1),
        live=sub_live.reshape(-1),
        poisoned=jnp.any(unresolved),
    )


def _emit_envelope(
    *,
    sub_cells: _SubCells,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Turn owned sub-cells into a weakly ascending, NaN-padded envelope row.

    Emission is driven purely by ownership: a boundary whose two sides differ in
    value *or* in policy emits two records, the outgoing owner's then the
    incoming owner's, so a kink that is invisible in the value but real in the
    policy still survives. Where several links meet at one abscissa the middle
    ones own no positive-width interval and correctly leave no record — the two
    one-sided records there belong to the links that own the ground on either
    side.
    """
    # Owned sub-cells already ascend — node cells ascend and each cell's
    # sub-cells ascend within it — so dropping the empty ones preserves the
    # order and needs no sort. Compacting straight into a row-sized workspace
    # keeps everything downstream independent of the fold capacity: a chain with
    # more owned sub-cells than the row has slots overflows regardless.
    n_slots = n_refined
    keep_at = jnp.cumsum(sub_cells.live.astype(jnp.int32)) - 1
    target = jnp.where(sub_cells.live, keep_at, n_slots)
    n_live = jnp.sum(sub_cells.live, dtype=jnp.int32)

    def compact(source: FloatND | IntND, empty: float) -> FloatND | IntND:
        return (
            jnp.full(n_slots, empty, dtype=source.dtype)
            .at[target]
            .set(source, mode="drop")
        )

    left = compact(sub_cells.left, jnp.nan)
    right = compact(sub_cells.right, jnp.nan)
    low = compact(sub_cells.owner_left_index, 0)
    high = compact(sub_cells.owner_right_index, 0)
    live = jnp.arange(n_slots) < n_live

    own_value = _line_value(low, high, left, endog_grid, value)
    own_policy = _line_value(low, high, left, endog_grid, policy)
    previous = jnp.clip(jnp.arange(n_slots) - 1, 0, n_slots - 1)
    prior_value = _line_value(low[previous], high[previous], left, endog_grid, value)
    prior_policy = _line_value(low[previous], high[previous], left, endog_grid, policy)

    is_first = jnp.arange(n_slots) == 0
    kinks = (
        live & ~is_first & ((prior_value != own_value) | (prior_policy != own_policy))
    )

    # Two rows per sub-cell boundary: the outgoing owner's reading where the
    # envelope kinks, then the incoming owner's reading.
    row_valid = jnp.stack([kinks, live], axis=1).ravel()
    row_grid = jnp.stack([left, left], axis=1).ravel()
    row_policy = jnp.stack([prior_policy, own_policy], axis=1).ravel()
    row_value = jnp.stack([prior_value, own_value], axis=1).ravel()

    # The final boundary closes the last owned sub-cell.
    last = jnp.clip(n_live - 1, 0, n_slots - 1)
    closing_valid = (n_live > 0)[None]
    closing_grid = right[last][None]
    closing_policy = _line_value(
        low[last], high[last], right[last], endog_grid, policy
    )[None]
    closing_value = _line_value(low[last], high[last], right[last], endog_grid, value)[
        None
    ]

    row_valid = jnp.concatenate([row_valid, closing_valid])
    row_grid = jnp.concatenate([row_grid, closing_grid])
    row_policy = jnp.concatenate([row_policy, closing_policy])
    row_value = jnp.concatenate([row_value, closing_value])

    position = jnp.cumsum(row_valid.astype(jnp.int32)) - 1
    slot = jnp.where(row_valid, position, n_refined)
    nan = jnp.full((), jnp.nan, dtype=endog_grid.dtype)
    out_grid = jnp.full(n_refined, jnp.nan, dtype=endog_grid.dtype)
    out_policy = jnp.full(n_refined, jnp.nan, dtype=policy.dtype)
    out_value = jnp.full(n_refined, jnp.nan, dtype=value.dtype)
    out_grid = out_grid.at[slot].set(jnp.where(row_valid, row_grid, nan), mode="drop")
    out_policy = out_policy.at[slot].set(
        jnp.where(row_valid, row_policy, nan), mode="drop"
    )
    out_value = out_value.at[slot].set(
        jnp.where(row_valid, row_value, nan), mode="drop"
    )
    return out_grid, out_policy, out_value, jnp.sum(row_valid, dtype=jnp.int32)


def _line_value(
    low: IntND,
    high: IntND,
    x_query: FloatND,
    endog_grid: Float1D,
    ordinate: Float1D,
) -> FloatND:
    """Read a link's affine interpolant, exactly at its own endpoints."""
    x0 = endog_grid[low]
    x1 = endog_grid[high]
    y0 = ordinate[low]
    y1 = ordinate[high]
    width = x1 - x0
    safe_width = jnp.where(width == 0.0, 1.0, width)
    interpolated = y0 + (x_query - x0) / safe_width * (y1 - y0)
    at_low = x_query == x0
    at_high = x_query == x1
    return jnp.where(at_low, y0, jnp.where(at_high, y1, interpolated))
