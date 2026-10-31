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
owners, or where more runs cover one cell than the static capacity holds, the
row is NaN-poisoned and reported as an overflow so the solve loop's diagnostics
name the offending cell instead of publishing a guess.
"""

import functools
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, NamedTuple, cast, overload

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope._exact_affine import exact_affine_read
from _lcm.egm.upper_envelope.cell_hull import hull_owners
from _lcm.egm.upper_envelope.topology import monotone_run_ids
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    Int1D,
    IntND,
    LoopIndex,
    ScalarBool,
    ScalarInt,
)

if TYPE_CHECKING:
    from _lcm.typing import PytreeChild
else:
    # The internal type vocabulary imports EGM carries and their owned rows.
    type PytreeChild = object

# Static capacity for the number of x-monotone runs that may cover one node cell.
# One discrete action can fold arbitrarily often, so this is a validated capacity,
# never an assumed bound: exceeding it poisons the row. The chain as a whole may
# hold any number of runs.
DEFAULT_MAX_RUNS: int = 24


class OwnedEnvelope(NamedTuple):
    """An envelope associated with its original candidate saving and topology.

    Owners are monotone candidate runs; source indices identify their actual
    links. Sides are incoming `1`, outgoing `-1`, or unsided `0`. Links have
    length `n_refined - 1` and never join different owners or coverage gaps.
    Only emitted nodes are represented; singleton candidate runs spanning no
    interval do not acquire support.
    """

    grid: Float1D
    policy: Float1D
    value: Float1D
    n_kept: ScalarInt
    savings: Float1D
    owner_id: Int1D
    side_id: Int1D
    link_valid: BoolND
    source_left_index: Int1D
    source_right_index: Int1D

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], OwnedEnvelope],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Retain original candidate buffers through native artifact restoration."""
        return _unflatten_owned_envelope, (None, tuple(self))


class _NegmBindingEnvelope(NamedTuple):
    """Associate current publication queries with native floor-source ownership."""

    grid: Float1D
    policy: Float1D
    value: Float1D
    n_kept: ScalarInt
    binding_query: BoolND


class _OwnedNegmBindingEnvelope(NamedTuple):
    """Keep native publication flags beside the same envelope's original owners."""

    owned: OwnedEnvelope
    binding_query: BoolND


@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    savings: Float1D,
    _binding_source_count: int,
    _binding_queries: FloatND,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
) -> _OwnedNegmBindingEnvelope: ...


@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    _binding_source_count: int,
    _binding_queries: FloatND,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
    savings: None = None,
) -> _NegmBindingEnvelope: ...


@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
    savings: None = None,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]: ...


@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
    savings: Float1D,
) -> OwnedEnvelope: ...


def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
    savings: Float1D | None = None,
    _binding_source_count: int | None = None,
    _binding_queries: FloatND | None = None,
) -> (
    tuple[Float1D, Float1D, Float1D, ScalarInt]
    | OwnedEnvelope
    | _NegmBindingEnvelope
    | _OwnedNegmBindingEnvelope
):
    """Refine a candidate value correspondence to its exact upper envelope.

    Args:
        endog_grid: Candidate endogenous grid points (resources) in producer
            order; dead candidates are NaN.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        n_refined: Static length of the refined output arrays.
        segment_id: Optional per-candidate branch label. Runs are always split
            where resources stop increasing; a label change splits them further.
        max_runs: Static capacity for the number of x-monotone runs over one
            node cell.
        cell_width: How many node cells the execution plan resolves together.
            The default visits one cell at a time. Cells are independent, so
            this partitions the work without changing any published value.
        savings: Original candidate savings. When present, publish their
            associated ownership and support with an `OwnedEnvelope`.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each of
        length `n_refined`, NaN-padded), and the number of envelope points
        `n_kept`. `n_kept > n_refined` signals overflow — either genuine capacity
        overflow or a decision that could not be certified — and the arrays are
        then not publishable.

    """
    if (_binding_source_count is None) != (_binding_queries is None):
        raise ValueError("Binding queries require their source-prefix count.")
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)
    run_id = monotone_run_ids(endog_grid=endog_grid, dead=dead, segment_id=segment_id)

    cell_left, cell_right, cell_live = _node_cells(endog_grid=endog_grid, dead=dead)
    runs = _run_node_order(
        endog_grid=endog_grid, run_id=run_id, dead=dead, max_runs=max_runs
    )

    sub_cells = _sub_cells_per_node_cell(
        cell_left=cell_left,
        cell_right=cell_right,
        cell_live=cell_live,
        runs=runs,
        endog_grid=endog_grid,
        value=value,
        max_runs=max_runs,
        n_refined=n_refined,
        cell_width=cell_width,
    )

    emitted = _emit_envelope(
        sub_cells=sub_cells,
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
        savings=savings,
        run_id=run_id,
    )
    out_grid, out_policy, out_value, n_kept = emitted[:4]

    uncertain = sub_cells.poisoned | runs.crowded
    overflow_marker = jnp.asarray(n_refined + 1, dtype=jnp.int32)
    n_kept = jnp.where(uncertain, overflow_marker, n_kept)
    poison = jnp.full((), jnp.nan, dtype=endog_grid.dtype)
    out_grid = jnp.where(uncertain, poison, out_grid)
    out_policy = jnp.where(uncertain, poison, out_policy)
    out_value = jnp.where(uncertain, poison, out_value)
    owned = None
    if isinstance(emitted, OwnedEnvelope):
        owned = OwnedEnvelope(
            grid=out_grid,
            policy=out_policy,
            value=out_value,
            n_kept=n_kept,
            savings=jnp.where(uncertain, poison, emitted.savings),
            owner_id=jnp.where(uncertain, -1, emitted.owner_id),
            side_id=jnp.where(uncertain, 0, emitted.side_id),
            link_valid=emitted.link_valid & ~uncertain,
            source_left_index=jnp.where(uncertain, -1, emitted.source_left_index),
            source_right_index=jnp.where(uncertain, -1, emitted.source_right_index),
        )
    if _binding_queries is not None and _binding_source_count is not None:
        binding_query = _classify_binding_queries(
            sub_cells=sub_cells,
            endog_grid=endog_grid,
            queries=_binding_queries,
            source_count=_binding_source_count,
            emitted_grid=out_grid,
            emitted_policy=out_policy,
            emitted_value=out_value,
            n_kept=n_kept,
            uncertain=uncertain,
        )
        if owned is not None:
            return _OwnedNegmBindingEnvelope(owned, binding_query)
        return _NegmBindingEnvelope(
            out_grid, out_policy, out_value, n_kept, binding_query
        )
    if owned is not None:
        return owned
    return out_grid, out_policy, out_value, n_kept


class _RunNodes:
    """Every slot's nodes in one ascending sequence, with each slot's extent in it.

    Each run that spans an interval holds one of `max_runs` slots, and the runs
    sharing a slot never overlap, so ordering by slot and then by abscissa lays
    each slot's runs out side by side in a single candidate-length sequence.
    Holding the runs this way keeps the topology linear in the candidates rather
    than the product of the candidates and the run capacity.
    """

    def __init__(
        self,
        *,
        order: Int1D,
        node_x: Float1D,
        node_run: Int1D,
        start: Int1D,
        n_nodes: Int1D,
        crowded: ScalarBool,
    ) -> None:
        self.order = order
        """Candidate indices ordered by slot, then abscissa, `(n_candidates,)`."""
        self.node_x = node_x
        """The abscissae in that same order, `(n_candidates,)`."""
        self.node_run = node_run
        """The run of each candidate in that same order, `(n_candidates,)`."""
        self.start = start
        """Where each slot's block begins in that order, `(max_runs,)`."""
        self.n_nodes = n_nodes
        """How many candidates each slot holds, `(max_runs,)`."""
        self.crowded = crowded
        """Whether two runs sharing a slot overlap, so a cell lost a run."""


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


def _classify_binding_queries(
    *,
    sub_cells: _SubCells,
    endog_grid: Float1D,
    queries: FloatND,
    source_count: int,
    emitted_grid: Float1D,
    emitted_policy: Float1D,
    emitted_value: Float1D,
    n_kept: ScalarInt,
    uncertain: ScalarBool,
) -> BoolND:
    """Locate each query's native sub-cell without a query-by-cell workspace.

    At a shared boundary the incoming sub-cell owns the query. An exact original
    endpoint uses that endpoint's source; every other query needs both owning
    link endpoints in the producer's floor prefix. Unresolved ownership,
    capacity overflow and refused live emitted channels withhold all flags.
    """
    capacity = sub_cells.left.size
    n_live = jnp.clip(sub_cells.n_live, 0, capacity)
    lower = jnp.zeros(queries.shape, dtype=jnp.int32)
    upper = jnp.full(queries.shape, n_live, dtype=jnp.int32)
    safe_query = jnp.where(jnp.isfinite(queries), queries, 0.0)

    # keyword-only-exempt: library-callback=jax.lax.fori_loop
    def bisect(_: LoopIndex, bounds: tuple[IntND, IntND]) -> tuple[IntND, IntND]:
        low, high = bounds
        middle = low + (high - low) // 2
        searching = low < high
        go_right = (
            searching
            & (middle < n_live)
            & (sub_cells.left[jnp.minimum(middle, capacity - 1)] <= safe_query)
        )
        return (
            jnp.where(go_right, middle + 1, low),
            jnp.where(searching & ~go_right, middle, high),
        )

    lower, _ = jax.lax.fori_loop(0, capacity.bit_length(), bisect, (lower, upper))
    selected = jnp.clip(lower - 1, 0, capacity - 1)
    low = sub_cells.owner_left_index[selected]
    high = sub_cells.owner_right_index[selected]
    low_floor = (low >= 0) & (low < source_count)
    high_floor = (high >= 0) & (high < source_count)
    source_is_floor = jnp.where(
        safe_query == endog_grid[low],
        low_floor,
        jnp.where(safe_query == endog_grid[high], high_floor, low_floor & high_floor),
    )
    covered = (
        jnp.isfinite(queries)
        & (lower > 0)
        & (selected < n_live)
        & (safe_query >= sub_cells.left[selected])
        & (safe_query <= sub_cells.right[selected])
    )
    live = jnp.arange(emitted_grid.size) < n_kept
    readable = jnp.all(
        ~live
        | (
            jnp.isfinite(emitted_grid)
            & jnp.isfinite(emitted_policy)
            & jnp.isfinite(emitted_value)
        )
    )
    resolved = (
        ~uncertain
        & (sub_cells.n_live <= capacity)
        & (n_kept <= emitted_grid.size)
        & readable
    )
    return covered & source_is_floor & resolved


def _node_cells(
    *, endog_grid: Float1D, dead: BoolND
) -> tuple[Float1D, Float1D, BoolND]:
    """Partition resources at the live candidate abscissae.

    No run starts or ends strictly inside such a cell, so each run contributes at
    most one link to it.
    """
    sort_key = jnp.where(dead, jnp.inf, endog_grid)
    sorted_x = jnp.sort(a=sort_key)
    left = sorted_x[:-1]
    right = sorted_x[1:]
    live = jnp.isfinite(left) & jnp.isfinite(right) & (right > left)
    return left, right, live


def _run_node_order(
    *, endog_grid: Float1D, run_id: Int1D, dead: BoolND, max_runs: int
) -> _RunNodes:
    """Assign the runs to slots, order them by slot, then by abscissa.

    Locating the link a run contributes to a cell is a search in its slot's
    nodes. The order does not depend on the cell, so it is established once here
    rather than per cell. Ordering all slots together rather than each slot over
    the whole candidate array is what keeps this linear in the candidates: a
    per-slot order would carry a slot axis alongside the candidate axis, and the
    row maps production adds would then multiply that product by the row count.

    The capacity bounds the runs over one cell, not the runs in the chain: a fine
    savings grid resolves every small fold, so a chain can hold far more runs
    than ever overlap. The `k`-th run that spans an interval takes slot
    `k mod max_runs`, so runs sharing a slot lie `max_runs` folds apart along the
    chain, and the slot holds them side by side. Should two of them overlap, a
    cell would see only one of them; the row is then `crowded` and is poisoned.
    A run spanning no interval carries no line and takes no slot.

    A candidate is absent from every slot when the caller declared it dead, when
    the run labelling gave it no run or a run spanning no interval, or when its
    abscissa is not finite. These are asked here rather than re-deriving
    deadness from the abscissa alone: a candidate carrying genuine poison keeps a
    finite abscissa, so an abscissa test would read it as live and its NaN value
    would enter the node list of whichever run its abscissa falls in.
    """
    n_candidates = endog_grid.shape[0]
    absent = dead | (run_id < 0) | ~jnp.isfinite(endog_grid)
    segment = jnp.where(absent, n_candidates, run_id)
    lowest = jax.ops.segment_min(
        jnp.where(absent, jnp.inf, endog_grid), segment, num_segments=n_candidates + 1
    )[:n_candidates]
    highest = jax.ops.segment_max(
        jnp.where(absent, -jnp.inf, endog_grid), segment, num_segments=n_candidates + 1
    )[:n_candidates]
    spans = highest > lowest
    safe_run = jnp.clip(run_id, 0, n_candidates - 1)
    absent = absent | ~spans[safe_run]
    slot = (jnp.cumsum(spans) - 1)[safe_run] % max_runs
    # Candidates in no slot get a block of their own past the last slot, so they
    # are outside every slot's extent and can never be located.
    block = jnp.where(absent, max_runs, slot)
    order = jnp.lexsort(
        (
            run_id.astype(jnp.float32),
            jnp.where(absent, jnp.inf, endog_grid),
            block.astype(jnp.float32),
        )
    ).astype(jnp.int32)
    n_nodes = jax.ops.segment_sum(
        jnp.ones_like(block, dtype=jnp.int32), block, num_segments=max_runs + 1
    )[:max_runs]
    start = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(n_nodes)[:-1].astype(jnp.int32)]
    )
    # Runs sharing a slot are disjoint exactly when each one's nodes stay
    # contiguous in the slot's order: one run change per further run.
    node_run = run_id[order]
    node_block = block[order]
    run_changes = jnp.sum(
        (node_block[1:] == node_block[:-1])
        & (node_block[:-1] < max_runs)
        & (node_run[1:] != node_run[:-1])
    )
    n_spanning = jnp.sum(spans)
    crowded = run_changes != n_spanning - jnp.minimum(n_spanning, max_runs)
    return _RunNodes(
        order=order,
        node_x=endog_grid[order],
        node_run=node_run,
        start=start,
        n_nodes=n_nodes,
        crowded=crowded,
    )


def _active_links_at_cell(
    *, runs: _RunNodes, cell_left: FloatND, cell_live: BoolND, n_candidates: int
) -> tuple[BoolND, IntND, IntND]:
    """Return, for one node cell, the link each slot contributes to it."""
    locate = functools.partial(
        _locate_link_in_run,
        runs=runs,
        cell_left=cell_left,
        cell_live=cell_live,
        n_candidates=n_candidates,
        n_steps=max(int(n_candidates).bit_length(), 1),
    )
    return jax.vmap(locate)(start=runs.start, n_nodes=runs.n_nodes)


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
    cell_width: int,
) -> _SubCells:
    """Resolve the envelope's owners across every node cell.

    Ownership is decided on value alone; the policy is read afterwards from the
    owning link, so it never influences which branch wins.

    Cells are independent, so `cell_width` partitions them without changing
    anything published. What it does change is the working set. Cells are visited
    in ascending order and each chunk's owned sub-cells are appended to the row
    as it goes, so what is live is one chunk's `cell_width * max_runs`
    splits and the row itself — never the `n_cells * max_runs` product, which is
    what the row count then multiplies. Width one holds a single cell's worth.

    Appending in cell order needs no sort: node cells ascend, and each cell's
    sub-cells ascend within it. Slots past the row's capacity are dropped while
    the count keeps rising, so an overflow is still visible to the caller.
    """
    n_slots = n_refined
    n_cells = cell_left.shape[0]
    chunk = min(max(1, cell_width), max(1, n_cells))
    n_chunks = -(-n_cells // chunk)
    n_padding = n_chunks * chunk - n_cells

    # Padding cells are dead, so they contribute no sub-cell and cannot change
    # the row; they only make the cell count divide by the chunk.
    chunks = (
        _pad_cells(array=cell_left, fill=0.0, n_padding=n_padding).reshape(
            n_chunks, chunk
        ),
        _pad_cells(array=cell_right, fill=0.0, n_padding=n_padding).reshape(
            n_chunks, chunk
        ),
        _pad_cells(array=cell_live, fill=False, n_padding=n_padding).reshape(
            n_chunks, chunk
        ),
    )
    split = functools.partial(
        _split_cell,
        runs=runs,
        n_candidates=endog_grid.shape[0],
        endog_grid=endog_grid,
        value=value,
        max_runs=max_runs,
    )

    init = (
        jnp.zeros((), dtype=jnp.int32),
        jnp.full(n_slots, jnp.nan, dtype=cell_left.dtype),
        jnp.full(n_slots, jnp.nan, dtype=cell_left.dtype),
        jnp.zeros(n_slots, dtype=jnp.int32),
        jnp.zeros(n_slots, dtype=jnp.int32),
        jnp.zeros((), dtype=bool),
    )
    (n_live, row_left, row_right, row_low, row_high, poisoned), _ = jax.lax.scan(
        functools.partial(_append_chunk, split=split, n_slots=n_slots), init, chunks
    )

    return _SubCells(
        left=row_left,
        right=row_right,
        owner_left_index=row_low,
        owner_right_index=row_high,
        n_live=n_live,
        poisoned=poisoned,
    )


def _locate_link_in_run(
    *,
    start: ScalarInt,
    n_nodes: ScalarInt,
    runs: _RunNodes,
    cell_left: FloatND,
    cell_live: BoolND,
    n_candidates: int,
    n_steps: int,
) -> tuple[BoolND, IntND, IntND]:
    """Find the link one slot contributes to the node cell opening at `cell_left`.

    A bisection over the slot's block, carried as scalars. Searching the block in
    place keeps the cell body free of any candidate-length array, so what a cell
    holds is one entry per slot and nothing that grows with the row. A cell in
    the gap between two runs of the slot gets no link from it.
    """
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
    safe = start + jnp.clip(position, 0, jnp.maximum(n_nodes - 2, 0))
    safe = jnp.clip(safe, 0, n_candidates - 2)
    live = (
        cell_live
        & (position >= 0)
        & (position <= n_nodes - 2)
        & (runs.node_run[safe] == runs.node_run[safe + 1])
    )
    return live, runs.order[safe], runs.order[safe + 1]


def _pad_cells(
    *, array: FloatND | BoolND, fill: float | bool, n_padding: int
) -> FloatND | BoolND:
    """Append `n_padding` dead cells so the cell count divides by the chunk."""
    return jnp.concatenate([array, jnp.full(n_padding, fill, dtype=array.dtype)])


type _SplitCell = tuple[FloatND, FloatND, IntND, IntND, BoolND, ScalarBool]


def _split_cell(
    *,
    left: FloatND,
    right: FloatND,
    live_cell: BoolND,
    runs: _RunNodes,
    n_candidates: int,
    endog_grid: Float1D,
    value: Float1D,
    max_runs: int,
) -> _SplitCell:
    """Resolve one node cell into its owned sub-cells and their owning links."""
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


# keyword-only-exempt: library-callback=jax.lax.scan
def _append_chunk(
    carry: _RowCarry,
    cells: tuple[FloatND, FloatND, BoolND],
    *,
    split: Callable[..., _SplitCell],
    n_slots: int,
) -> tuple[_RowCarry, None]:
    """Resolve one chunk of node cells and append its owned sub-cells to the row."""
    cursor, row_left, row_right, row_low, row_high, poisoned = carry
    sub_left, sub_right, owner_low, owner_high, sub_live, unresolved = jax.vmap(split)(
        left=cells[0], right=cells[1], live_cell=cells[2]
    )
    keep = sub_live.reshape(-1)
    target = cursor + jnp.cumsum(keep.astype(jnp.int32)) - 1
    slot = jnp.where(keep, target, n_slots)
    return (
        cursor + jnp.sum(keep, dtype=jnp.int32),
        _place_in_row(
            row=row_left, source=sub_left, empty=jnp.nan, slot=slot, keep=keep
        ),
        _place_in_row(
            row=row_right, source=sub_right, empty=jnp.nan, slot=slot, keep=keep
        ),
        _place_in_row(row=row_low, source=owner_low, empty=0, slot=slot, keep=keep),
        _place_in_row(row=row_high, source=owner_high, empty=0, slot=slot, keep=keep),
        poisoned | jnp.any(unresolved),
    ), None


def _place_in_row(
    *,
    row: FloatND | IntND,
    source: FloatND | IntND,
    empty: float,
    slot: IntND,
    keep: BoolND,
) -> FloatND | IntND:
    """Scatter the kept sub-cell entries of `source` into their row slots."""
    return row.at[slot].set(jnp.where(keep, source.reshape(-1), empty), mode="drop")


def _emit_envelope(  # noqa: PLR0915 - channels share one emission and scatter topology
    *,
    sub_cells: _SubCells,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    savings: Float1D | None = None,
    run_id: Int1D,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt] | OwnedEnvelope:
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

    own_channels = _link_channels(
        low=low,
        high=high,
        x_query=left,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        savings=savings,
    )
    own_value, own_policy = own_channels[:2]
    previous = jnp.clip(jnp.arange(n_slots) - 1, 0, n_slots - 1)
    is_first = jnp.arange(n_slots) == 0
    gap = live & ~is_first & (right[previous] < left)
    prior_grid = jnp.where(gap, right[previous], left) if savings is not None else left
    prior_channels = _link_channels(
        low=low[previous],
        high=high[previous],
        x_query=prior_grid,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        savings=savings,
    )
    prior_value, prior_policy = prior_channels[:2]

    kinks = (
        live & ~is_first & ((prior_value != own_value) | (prior_policy != own_policy))
    )
    if savings is not None:
        kinks = (
            live
            & ~is_first
            & (
                gap
                | kinks
                | (own_channels[-1] != prior_channels[-1])
                | (run_id[low] != run_id[low[previous]])
            )
        )

    # Two rows per sub-cell boundary: the outgoing owner's reading where the
    # envelope kinks, then the incoming owner's reading.
    row_valid = jnp.stack([kinks, live], axis=1).ravel()
    row_grid = jnp.stack([prior_grid, left], axis=1).ravel()
    row_policy = jnp.stack([prior_policy, own_policy], axis=1).ravel()
    row_value = jnp.stack([prior_value, own_value], axis=1).ravel()

    # The final boundary closes the last owned sub-cell.
    last = jnp.clip(n_live - 1, 0, n_slots - 1)
    closing_valid = (n_live > 0)[None]
    closing_grid = right[last][None]
    closing_channels = _link_channels(
        low=low[last],
        high=high[last],
        x_query=right[last],
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        savings=savings,
    )
    closing_read_value, closing_read_policy = closing_channels[:2]
    closing_policy = closing_read_policy[None]
    closing_value = closing_read_value[None]

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
    n_kept = jnp.sum(row_valid, dtype=jnp.int32)
    if savings is None:
        return out_grid, out_policy, out_value, n_kept

    row_savings = jnp.concatenate(
        [
            jnp.stack([prior_channels[-1], own_channels[-1]], axis=1).ravel(),
            closing_channels[-1][None],
        ]
    )
    row_low = jnp.concatenate(
        [jnp.stack([low[previous], low], axis=1).ravel(), low[last][None]]
    )
    row_high = jnp.concatenate(
        [jnp.stack([high[previous], high], axis=1).ravel(), high[last][None]]
    )
    row_side = jnp.concatenate(
        [
            jnp.stack(
                [
                    -jnp.ones(n_slots, dtype=jnp.int32),
                    jnp.where(is_first | kinks, 1, 0),
                ],
                axis=1,
            ).ravel(),
            jnp.array([-1], dtype=jnp.int32),
        ]
    )
    row_end = jnp.concatenate(
        [jnp.stack([right[previous], right], axis=1).ravel(), right[last][None]]
    )
    out_savings = (
        jnp.full(n_refined, jnp.nan, dtype=savings.dtype)
        .at[slot]
        .set(jnp.where(row_valid, row_savings, nan), mode="drop")
    )
    out_low = (
        jnp.full(n_refined, -1, dtype=jnp.int32)
        .at[slot]
        .set(jnp.where(row_valid, row_low, -1), mode="drop")
    )
    out_high = (
        jnp.full(n_refined, -1, dtype=jnp.int32)
        .at[slot]
        .set(jnp.where(row_valid, row_high, -1), mode="drop")
    )
    out_owner = (
        jnp.full(n_refined, -1, dtype=jnp.int32)
        .at[slot]
        .set(jnp.where(row_valid, run_id[row_low], -1), mode="drop")
    )
    out_side = (
        jnp.zeros(n_refined, dtype=jnp.int32)
        .at[slot]
        .set(jnp.where(row_valid, row_side, 0), mode="drop")
    )
    out_end = (
        jnp.full(n_refined, jnp.nan, dtype=endog_grid.dtype)
        .at[slot]
        .set(jnp.where(row_valid, row_end, nan), mode="drop")
    )
    readable = (
        jnp.isfinite(out_policy) & jnp.isfinite(out_savings) & ~jnp.isnan(out_value)
    )
    link_valid = (
        readable[:-1]
        & readable[1:]
        & (out_grid[1:] > out_grid[:-1])
        & (out_grid[1:] <= out_end[:-1])
        & (out_owner[:-1] == out_owner[1:])
        & (out_owner[:-1] >= 0)
    )
    return OwnedEnvelope(
        grid=out_grid,
        policy=out_policy,
        value=out_value,
        n_kept=n_kept,
        savings=out_savings,
        owner_id=out_owner,
        side_id=out_side,
        link_valid=link_valid,
        source_left_index=out_low,
        source_right_index=out_high,
    )


def _line_read(
    *,
    low: IntND,
    high: IntND,
    x_query: FloatND,
    endog_grid: Float1D,
    ordinate: Float1D,
) -> tuple[FloatND, IntND]:
    """Read a link's affine value at `x_query`, with the status of that read.

    The quotient is formed as an exact rational from the four stored endpoint
    operands and rounded once. Publishing it through a floating interpolant
    instead loses results the format holds: two nearly opposite normal ordinates
    have a midpoint in the subnormal band, and the subtraction that forms it
    cancels the magnitude away before the division can see it.

    A link of zero width has no affine line, so it is not asked for one: it takes
    the value stored at its own node. That is the only reading available there,
    and it is exact.

    Returns:
        Tuple of the rounded value and its status, `0` where the value was
        published and `UNRESOLVED_STATUS` where the operands admit no answer.

    """
    x0 = endog_grid[low]
    x1 = endog_grid[high]
    y0 = ordinate[low]
    y1 = ordinate[high]

    published, status = exact_affine_read(x0=x0, x1=x1, v0=y0, v1=y1, x_query=x_query)
    degenerate = ~(x1 > x0)
    return (
        jnp.where(degenerate, y0, published),
        jnp.where(degenerate, jnp.int32(0), status),
    )


def _link_channels(
    *,
    low: IntND,
    high: IntND,
    x_query: FloatND,
    endog_grid: Float1D,
    value: Float1D,
    policy: Float1D,
    savings: Float1D | None = None,
) -> tuple[FloatND, FloatND] | tuple[FloatND, FloatND, FloatND]:
    """Read one link's value and policy, publishing both channels or neither.

    A row carrying a finite value beside a NaN policy would be read downstream as
    a real state with an unknown action, which is worse than no row at all. The
    two channels share their link geometry, so they can only disagree on whether
    the exact result overflows — and where either does, both are withheld.
    """
    read_value, value_status = _line_read(
        low=low, high=high, x_query=x_query, endog_grid=endog_grid, ordinate=value
    )
    read_policy, policy_status = _line_read(
        low=low, high=high, x_query=x_query, endog_grid=endog_grid, ordinate=policy
    )
    readable = (value_status == 0) & (policy_status == 0)
    if savings is not None:
        read_savings, savings_status = _line_read(
            low=low, high=high, x_query=x_query, endog_grid=endog_grid, ordinate=savings
        )
        readable = readable & (savings_status == 0)
        return (
            jnp.where(readable, read_value, jnp.asarray(jnp.nan, value.dtype)),
            jnp.where(readable, read_policy, jnp.asarray(jnp.nan, policy.dtype)),
            jnp.where(readable, read_savings, jnp.asarray(jnp.nan, savings.dtype)),
        )
    return (
        jnp.where(readable, read_value, jnp.asarray(jnp.nan, value.dtype)),
        jnp.where(readable, read_policy, jnp.asarray(jnp.nan, policy.dtype)),
    )


def _flatten_owned_envelope(
    record: OwnedEnvelope,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(record), None


def _flatten_owned_envelope_with_keys(
    record: OwnedEnvelope,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(OwnedEnvelope._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_owned_envelope(
    metadata: None,
    children: Iterable[PytreeChild],
) -> OwnedEnvelope:
    """Restore the unchanged candidate schema for abstract and role contexts."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(OwnedEnvelope._fields):
        raise TypeError("Owned envelope reconstruction requires its exact fields.")
    return cast("OwnedEnvelope", tuple.__new__(OwnedEnvelope, leaves))


jax.tree_util.register_pytree_with_keys(
    OwnedEnvelope,
    _flatten_owned_envelope_with_keys,
    _unflatten_owned_envelope,
    _flatten_owned_envelope,
)
