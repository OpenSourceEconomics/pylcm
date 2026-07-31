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

# How many node cells to resolve in parallel. Cells are independent, so this
# changes only the working set, never a published value. `None` scans them one
# at a time: the smallest working set, and on the models pylcm measures also the
# fastest, since one cell already carries enough work to occupy the device and a
# wider step only adds intermediates. Raise it for a model whose cells are small
# enough to leave a device idle — after measuring, not on principle.
DEFAULT_CELL_BATCH_SIZE: int | None = None


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
        cell_batch_size: How many node cells to resolve in parallel; `None`
            resolves them one at a time, which is the smallest working set
            available. Cells are independent, so this partitions the work
            without changing any published value.

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
    runs = _run_node_order(endog_grid=endog_grid, run_id=run_id, max_runs=max_runs)

    sub_cells = _sub_cells_per_node_cell(
        cell_left=cell_left,
        cell_right=cell_right,
        cell_live=cell_live,
        runs=runs,
        endog_grid=endog_grid,
        value=value,
        max_runs=max_runs,
        n_refined=n_refined,
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


class _RunNodes:
    """Every run's nodes in one ascending sequence, with each run's extent in it.

    Runs partition the candidates, so ordering by run and then by abscissa lays
    each run's own nodes out contiguously in a single candidate-length sequence.
    Holding the runs this way keeps the topology linear in the candidates rather
    than the product of the candidates and the run capacity.
    """

    def __init__(
        self, *, order: Int1D, node_x: Float1D, start: Int1D, n_nodes: Int1D
    ) -> None:
        self.order = order
        """Candidate indices ordered by run, then abscissa, `(n_candidates,)`."""
        self.node_x = node_x
        """The abscissae in that same order, `(n_candidates,)`."""
        self.start = start
        """Where each run's block begins in that order, `(max_runs,)`."""
        self.n_nodes = n_nodes
        """How many candidates each run owns, `(max_runs,)`."""


class _SubCells:
    """The open intervals the envelope is piecewise affine on, with their owners.

    Already compacted into a row-sized workspace: the owned sub-cells occupy the
    first `n_live` slots in ascending order, so nothing here scales with the
    number of node cells.
    """

    def __init__(
        self,
        *,
        left: FloatND,
        right: FloatND,
        owner_left_index: IntND,
        owner_right_index: IntND,
        n_live: ScalarInt,
        poisoned: ScalarBool,
    ) -> None:
        self.left = left
        """Left boundary of each owned sub-cell, `(n_refined,)`, NaN-padded."""
        self.right = right
        """Right boundary of each owned sub-cell, `(n_refined,)`, NaN-padded."""
        self.owner_left_index = owner_left_index
        """Candidate index of the owning link's lower endpoint, `(n_refined,)`."""
        self.owner_right_index = owner_right_index
        """Candidate index of the owning link's upper endpoint, `(n_refined,)`."""
        self.n_live = n_live
        """How many sub-cells were owned, counted before any were dropped."""
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


def _run_node_order(*, endog_grid: Float1D, run_id: Int1D, max_runs: int) -> _RunNodes:
    """Order the candidates by run, then by abscissa, and record each run's block.

    Locating the link a run contributes to a cell is a search in that run's own
    nodes. The order does not depend on the cell, so it is established once here
    rather than per cell. Ordering all runs together rather than each run over
    the whole candidate array is what keeps this linear in the candidates: a
    per-run order would carry a run axis alongside the candidate axis, and the
    row maps production adds would then multiply that product by the row count.
    """
    dead = ~jnp.isfinite(endog_grid)
    # Candidates belonging to no run get a block of their own past the last run,
    # so they are outside every run's extent and can never be located.
    block = jnp.where(dead, max_runs, jnp.clip(run_id, 0, max_runs))
    order = jnp.lexsort(
        (jnp.where(dead, jnp.inf, endog_grid), block.astype(jnp.float32))
    ).astype(jnp.int32)
    n_nodes = jax.ops.segment_sum(
        jnp.ones_like(block, dtype=jnp.int32), block, num_segments=max_runs + 1
    )[:max_runs]
    start = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(n_nodes)[:-1].astype(jnp.int32)]
    )
    return _RunNodes(
        order=order, node_x=endog_grid[order], start=start, n_nodes=n_nodes
    )


def _active_links_at_cell(
    *, runs: _RunNodes, cell_left: FloatND, cell_live: BoolND, n_candidates: int
) -> tuple[BoolND, IntND, IntND]:
    """Return, for one node cell, the link each run contributes to it."""
    # A bisection over a run's block, carried as scalars. Searching the block in
    # place keeps the cell body free of any candidate-length array, so what a
    # cell holds is one entry per run and nothing that grows with the row.
    n_steps = max(int(n_candidates).bit_length(), 1)

    def locate(start: ScalarInt, n_nodes: ScalarInt) -> tuple[BoolND, IntND, IntND]:
        low = jnp.zeros_like(n_nodes)
        high = n_nodes
        for _ in range(n_steps):
            searching = low < high
            middle = (low + high) // 2
            at_middle = runs.node_x[jnp.clip(start + middle, 0, n_candidates - 1)]
            below = at_middle <= cell_left
            low = jnp.where(searching & below, middle + 1, low)
            high = jnp.where(searching & ~below, middle, high)
        # The cell's left boundary is itself a node abscissa, so counting the
        # nodes at or below it lands one past the node opening the covering link.
        position = low - 1
        live = cell_live & (position >= 0) & (position <= n_nodes - 2)
        safe = start + jnp.clip(position, 0, jnp.maximum(n_nodes - 2, 0))
        safe = jnp.clip(safe, 0, n_candidates - 2)
        return live, runs.order[safe], runs.order[safe + 1]

    return jax.vmap(locate)(runs.start, runs.n_nodes)


def _sub_cells_per_node_cell(
    *,
    cell_left: Float1D,
    cell_right: Float1D,
    cell_live: BoolND,
    runs: _RunNodes,
    endog_grid: Float1D,
    value: Float1D,
    max_runs: int,
    n_refined: int,
    cell_batch_size: int | None,
) -> _SubCells:
    """Resolve the envelope's owners across every node cell.

    Ownership is decided on value alone; the policy is read afterwards from the
    owning link, so it never influences which branch wins.

    Cells are independent, so `cell_batch_size` partitions them without changing
    anything published. What it does change is the working set. Cells are visited
    in ascending order and each chunk's owned sub-cells are appended to the row
    as it goes, so what is live is one chunk's `cell_batch_size * max_runs`
    splits and the row itself — never the `n_cells * max_runs` product, which is
    what the row count then multiplies. `None` resolves one cell at a time and so
    holds a single cell's worth, the floor.

    Appending in cell order needs no sort: node cells ascend, and each cell's
    sub-cells ascend within it. Slots past the row's capacity are dropped while
    the count keeps rising, so an overflow is still visible to the caller.
    """
    n_slots = n_refined
    chunk = max(1, cell_batch_size or 1)
    n_cells = cell_left.shape[0]
    n_chunks = -(-n_cells // chunk)
    padded = n_chunks * chunk
    n_candidates = endog_grid.shape[0]

    def pad(array: FloatND | BoolND, *, fill: float | bool) -> FloatND | BoolND:
        return jnp.concatenate(
            [array, jnp.full(padded - n_cells, fill, dtype=array.dtype)]
        )

    # Padding cells are dead, so they contribute no sub-cell and cannot change
    # the row; they only make the cell count divide by the chunk.
    chunks = (
        pad(cell_left, fill=0.0).reshape(n_chunks, chunk),
        pad(cell_right, fill=0.0).reshape(n_chunks, chunk),
        pad(cell_live, fill=False).reshape(n_chunks, chunk),
    )

    def split(
        left: FloatND, right: FloatND, live_cell: BoolND
    ) -> tuple[FloatND, FloatND, IntND, IntND, BoolND, ScalarBool]:
        live, low, high = _active_links_at_cell(
            runs=runs, cell_left=left, cell_live=live_cell, n_candidates=n_candidates
        )
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
        return sub_left, sub_right, low[owner], high[owner], sub_live, unresolved

    type _RowCarry = tuple[ScalarInt, FloatND, FloatND, IntND, IntND, ScalarBool]

    def append_chunk(
        carry: _RowCarry, cells: tuple[FloatND, FloatND, BoolND]
    ) -> tuple[_RowCarry, None]:
        cursor, row_left, row_right, row_low, row_high, poisoned = carry
        sub_left, sub_right, owner_low, owner_high, sub_live, unresolved = jax.vmap(
            split
        )(*cells)
        keep = sub_live.reshape(-1)
        target = cursor + jnp.cumsum(keep.astype(jnp.int32)) - 1
        slot = jnp.where(keep, target, n_slots)

        def place(
            row: FloatND | IntND, source: FloatND | IntND, empty: float
        ) -> FloatND | IntND:
            return row.at[slot].set(
                jnp.where(keep, source.reshape(-1), empty), mode="drop"
            )

        return (
            cursor + jnp.sum(keep, dtype=jnp.int32),
            place(row_left, sub_left, jnp.nan),
            place(row_right, sub_right, jnp.nan),
            place(row_low, owner_low, 0),
            place(row_high, owner_high, 0),
            poisoned | jnp.any(unresolved),
        ), None

    init = (
        jnp.zeros((), dtype=jnp.int32),
        jnp.full(n_slots, jnp.nan, dtype=cell_left.dtype),
        jnp.full(n_slots, jnp.nan, dtype=cell_left.dtype),
        jnp.zeros(n_slots, dtype=jnp.int32),
        jnp.zeros(n_slots, dtype=jnp.int32),
        jnp.zeros((), dtype=bool),
    )
    (n_live, row_left, row_right, row_low, row_high, poisoned), _ = jax.lax.scan(
        append_chunk, init, chunks
    )

    return _SubCells(
        left=row_left,
        right=row_right,
        owner_left_index=row_low,
        owner_right_index=row_high,
        n_live=n_live,
        poisoned=poisoned,
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
    # The row arrives already compacted: owned sub-cells were appended in cell
    # order as the cells were resolved. `n_live` is the untruncated count, so a
    # chain with more owned sub-cells than the row has slots still reports the
    # overflow even though the surplus was dropped on the way in.
    n_slots = n_refined
    n_live = sub_cells.n_live
    left = sub_cells.left
    right = sub_cells.right
    low = sub_cells.owner_left_index
    high = sub_cells.owner_right_index
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
