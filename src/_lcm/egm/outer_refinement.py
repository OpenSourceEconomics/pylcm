"""Globally safeguarded outer refinement: bracket-local search, adaptive mesh.

Two layers, both refusing to assume global unimodality:

**Safeguarded continuous argmax** (`safeguarded_continuous_argmax`) — given
exact candidate values on shared outer nodes and a continuous surrogate (an
interpolant read), it identifies *every* node-local maximum per state cell,
golden-refines a bracket around each, and selects the best of {all exact
nodes, all refined bracket optima} with deterministic tie rules (higher
value, then smaller abscissa, then earlier candidate). Because the exact
nodes always compete, a surrogate can add resolution but never lose an exact
finite-grid winner; the domain endpoints are nodes, so boundary optima are
kept exactly.

**Adaptive shared mesh** (`refine_outer_mesh`) — a host-side driver that
inserts interval midpoints where the interpolant disagrees with exact solves
beyond tolerance, until every interval validates or the budget is exhausted.
Inference-grade runs fail closed (`OuterSearchConvergenceError`); development
runs may return the partial mesh flagged `unresolved`.

One deliberate deviation from a literal reading of the design document: an
optimum-containing interval is *not* marked unconditionally (that could never
terminate) — it is held to a 10x tighter validation standard, and to the
margin rule (marked while the cell's best/second-best margin is within ten
times the interval's absolute validation error).
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_search import AdaptiveOuterMesh
from _lcm.optimization.golden_section import maximize_golden_section
from lcm.exceptions import OuterSearchConvergenceError
from lcm.typing import BoolND, Float1D, FloatND

# Tightening factor for optimum-containing intervals: they must validate to
# 1/10 of the ordinary normalized-error threshold before they stop being
# refined, and they stay marked while the best/second-best value margin is
# within `_MARGIN_SAFETY` times the interval's absolute validation error.
_OPTIMUM_TIGHTENING = 0.1
_MARGIN_SAFETY = 10.0

# Neighbor-closure fraction: refining one interval perturbs its neighbors'
# accuracy (their shared-node slope stencils change on the graded mesh), so
# a neighbor of a marked interval is co-marked (transitively) while its own
# normalized error exceeds this fraction of the threshold — a near-threshold
# region refines together instead of burning one round per interval. The
# interpolant's 4-point slopes keep the grading amplification bounded
# (with 3-point slopes it grows like 1/h and a front creeps indefinitely —
# measured before the upgrade); 1/16 covers one full order of headroom.
_NEIGHBOR_FRACTION = 1.0 / 16.0


@dataclass(frozen=True, kw_only=True)
class SafeguardedSearchResult:
    """Per-cell outcome of the globally safeguarded continuous argmax."""

    x: FloatND
    """Selected outer action per cell."""

    value: FloatND
    """Value at the selection (surrogate value if a refined point won,
    exact value if a node won); `-inf` on invalid cells."""

    second_best_x: FloatND
    """Runner-up abscissa (for branch-margin diagnostics)."""

    second_best_value: FloatND
    """Runner-up value; `-inf` where fewer than two finite candidates."""

    coarse_x: FloatND
    """Best exact node per cell — the finite-grid answer."""

    coarse_value: FloatND
    """Exact value at `coarse_x`."""

    refinement_gain: FloatND
    """`value - coarse_value`; how much continuous refinement added."""

    bracket_width: FloatND
    """Final bracket width of the winning refined candidate; `0.0` where an
    exact node won outright."""

    at_lower_bound: BoolND
    """Whether the selection sits at the domain's lower endpoint."""

    at_upper_bound: BoolND
    """Whether the selection sits at the domain's upper endpoint."""

    valid: BoolND
    """Whether the cell had any finite candidate at all."""


def safeguarded_continuous_argmax(
    objective: Callable[[FloatND], FloatND],
    *,
    nodes: FloatND,
    node_values: FloatND,
    golden_iterations: int,
    max_brackets: int = 8,
) -> SafeguardedSearchResult:
    """Continuous outer argmax, safeguarded by the exact candidate mesh.

    Args:
        objective: Continuous surrogate (typically an interpolant read):
            maps an abscissa array whose shape broadcasts against the state
            shape to values of the broadcast shape. NaN probes are treated
            as `-inf` by the refiner.
        nodes: Shared exact outer nodes, shape `(C,)`, strictly increasing.
        node_values: Exact candidate values, shape `(C, *S)`.
        golden_iterations: Static golden-section budget per bracket.
        max_brackets: Node-local maxima refined per cell (the top ones by
            exact value); further local maxima still compete as exact nodes,
            just without continuous refinement.

    Returns:
        The per-cell selection with margins and bound/bracket diagnostics.

    """
    nodes = jnp.asarray(nodes)
    node_values = jnp.asarray(node_values)
    n_nodes = nodes.shape[0]
    state_shape = node_values.shape[1:]
    finite_values = jnp.where(jnp.isfinite(node_values), node_values, -jnp.inf)
    valid = jnp.any(jnp.isfinite(node_values), axis=0)

    # --- Node-local maxima (boundary nodes compare one-sided). ---
    up = jnp.concatenate(
        [
            finite_values[:1] >= finite_values[1:2],
            finite_values[1:] >= finite_values[:-1],
        ],
        axis=0,
    )
    down = jnp.concatenate(
        [
            finite_values[:-1] >= finite_values[1:],
            finite_values[-1:] >= finite_values[-2:-1],
        ],
        axis=0,
    )
    is_local_max = up & down & jnp.isfinite(finite_values)

    # --- Top-K local maxima per cell; unused slots masked invalid. ---
    n_brackets = min(n_nodes, max_brackets)
    masked = jnp.where(is_local_max, finite_values, -jnp.inf)
    masked_last = jnp.moveaxis(masked, 0, -1)  # (*S, C)
    top_values, top_idx = jax.lax.top_k(masked_last, n_brackets)  # (*S, K)
    top_idx = jnp.moveaxis(top_idx, -1, 0)  # (K, *S)
    bracket_valid = jnp.moveaxis(jnp.isfinite(top_values), -1, 0)  # (K, *S)
    lower = nodes[jnp.clip(top_idx - 1, 0, n_nodes - 1)]
    upper = nodes[jnp.clip(top_idx + 1, 0, n_nodes - 1)]

    refined = maximize_golden_section(
        objective,
        lower=lower,
        upper=upper,
        iterations=golden_iterations,
        valid=bracket_valid,
    )

    # --- Coarse selection: fold the exact nodes in increasing order. ---
    best_x = jnp.broadcast_to(nodes[0], state_shape)
    best_v = finite_values[0]
    second_x = jnp.broadcast_to(nodes[0], state_shape)
    second_v = jnp.full(state_shape, -jnp.inf)
    width = jnp.zeros(state_shape)
    for index in range(1, n_nodes):
        best_x, best_v, second_x, second_v, width = _consider(
            cand_x=jnp.broadcast_to(nodes[index], state_shape),
            cand_v=finite_values[index],
            cand_width=jnp.zeros(state_shape),
            best_x=best_x,
            best_v=best_v,
            second_x=second_x,
            second_v=second_v,
            width=width,
        )
    coarse_x = best_x
    coarse_v = best_v

    # --- Refined bracket optima join the competition. ---
    refined_width = refined.upper - refined.lower
    for index in range(n_brackets):
        best_x, best_v, second_x, second_v, width = _consider(
            cand_x=refined.x[index],
            cand_v=refined.value[index],
            cand_width=refined_width[index],
            best_x=best_x,
            best_v=best_v,
            second_x=second_x,
            second_v=second_v,
            width=width,
        )

    return SafeguardedSearchResult(
        x=jnp.where(valid, best_x, nodes[0]),
        value=jnp.where(valid, best_v, -jnp.inf),
        second_best_x=jnp.where(valid, second_x, nodes[0]),
        second_best_value=jnp.where(valid, second_v, -jnp.inf),
        coarse_x=jnp.where(valid, coarse_x, nodes[0]),
        coarse_value=jnp.where(valid, coarse_v, -jnp.inf),
        refinement_gain=jnp.where(valid, best_v - coarse_v, 0.0),
        bracket_width=jnp.where(valid, width, 0.0),
        at_lower_bound=valid & (best_x == nodes[0]),
        at_upper_bound=valid & (best_x == nodes[-1]),
        valid=valid,
    )


def _consider(
    *,
    cand_x: FloatND,
    cand_v: FloatND,
    cand_width: FloatND,
    best_x: FloatND,
    best_v: FloatND,
    second_x: FloatND,
    second_v: FloatND,
    width: FloatND,
) -> tuple[FloatND, FloatND, FloatND, FloatND, FloatND]:
    """One step of the two-best fold with the deterministic tie rule.

    A candidate displaces the incumbent when its value is strictly higher,
    or exactly ties with a strictly smaller abscissa; the displaced
    incumbent becomes the runner-up. Otherwise the candidate competes for
    the runner-up slot under the same rule.
    """
    better = (cand_v > best_v) | ((cand_v == best_v) & (cand_x < best_x))
    runner_up = (~better) & (
        (cand_v > second_v) | ((cand_v == second_v) & (cand_x < second_x))
    )
    new_second_x = jnp.where(better, best_x, jnp.where(runner_up, cand_x, second_x))
    new_second_v = jnp.where(better, best_v, jnp.where(runner_up, cand_v, second_v))
    return (
        jnp.where(better, cand_x, best_x),
        jnp.where(better, cand_v, best_v),
        new_second_x,
        new_second_v,
        jnp.where(better, cand_width, width),
    )


@dataclass(frozen=True, kw_only=True)
class AdaptiveMeshResult:
    """Outcome of the adaptive shared-mesh refinement."""

    nodes: Float1D
    """The final shared mesh (sorted, unique)."""

    node_values: FloatND
    """Exact candidate values on the final mesh, shape `(C, *S)`."""

    rounds_used: int
    """Midpoint-insertion rounds actually run."""

    max_validation_error: float
    """Largest normalized exact-vs-interpolated midpoint error observed in
    the final validation pass (over all intervals and cells)."""

    n_cells_all_invalid: int
    """State cells with no finite candidate on any node. Not a failure at
    this layer — whether such cells are *reachable* is solver knowledge."""

    unresolved: bool
    """Whether marked intervals remained when the budget ran out (only
    reachable with `fail_closed=False`)."""


def refine_outer_mesh(
    *,
    initial_nodes: Float1D,
    solve_at: Callable[[Float1D], FloatND],
    config: AdaptiveOuterMesh,
    interpolant: LocalCubicOuterInterpolant | None = None,
    fail_closed: bool = True,
) -> AdaptiveMeshResult:
    """Adaptively refine the shared outer mesh until midpoints validate.

    Host-side driver: each round proposes every interval midpoint, solves
    it exactly, compares against the current interpolant, and inserts the
    midpoints of intervals that fail validation. Optimum-containing
    intervals are held to a 10x tighter standard and to the
    best/second-best margin rule (see module docstring).

    Args:
        initial_nodes: Starting mesh, shape `(C0,)`, strictly increasing.
        solve_at: Exact conditional solve: maps new nodes `(M,)` to their
            candidate values `(M, *S)`. Must be deterministic.
        config: The `AdaptiveOuterMesh` tolerances and budgets.
        interpolant: Surface reader; defaults to the local cubic Hermite.
        fail_closed: Raise on an exhausted budget (inference mode) instead
            of returning the partial mesh flagged `unresolved`.

    Returns:
        The refined mesh with its exact values and diagnostics.

    Raises:
        OuterSearchConvergenceError: If `fail_closed` and marked intervals
            remain at the node or round budget.

    """
    interp = interpolant if interpolant is not None else LocalCubicOuterInterpolant()
    nodes = jnp.asarray(initial_nodes)
    _fail_if_bad_initial_mesh(nodes=nodes, config=config)
    values = jnp.asarray(solve_at(nodes))

    max_err = 0.0
    rounds_used = 0
    unresolved = False
    for _round in range(config.max_refinement_rounds + 1):
        midpoints = 0.5 * (nodes[:-1] + nodes[1:])
        exact = jnp.asarray(solve_at(midpoints))
        marked, max_err = _mark_intervals(
            nodes=nodes,
            values=values,
            midpoints=midpoints,
            exact=exact,
            config=config,
            interp=interp,
        )
        if not bool(jnp.any(marked)):
            break
        n_insert = int(jnp.sum(marked))
        if (
            _round == config.max_refinement_rounds
            or nodes.shape[0] + n_insert > config.max_nodes
        ):
            budget = (
                f"round budget ({config.max_refinement_rounds})"
                if _round == config.max_refinement_rounds
                else f"node budget ({config.max_nodes})"
            )
            if fail_closed:
                msg = (
                    f"Adaptive outer mesh exhausted its {budget} with "
                    f"{n_insert} marked interval(s) remaining "
                    f"(max normalized validation error {max_err:.3e}). "
                    "Raise the budget, loosen the tolerances, or "
                    "investigate the surface."
                )
                raise OuterSearchConvergenceError(msg)
            unresolved = True
            break
        keep = np.flatnonzero(np.asarray(marked))
        nodes, values = _insert_midpoints(
            nodes=nodes,
            values=values,
            midpoints=midpoints[keep],
            midpoint_values=exact[keep],
        )
        rounds_used = _round + 1

    n_cells_all_invalid = int(jnp.sum(~jnp.any(jnp.isfinite(values), axis=0)))
    return AdaptiveMeshResult(
        nodes=nodes,
        node_values=values,
        rounds_used=rounds_used,
        max_validation_error=float(max_err),
        n_cells_all_invalid=n_cells_all_invalid,
        unresolved=unresolved,
    )


def _mark_intervals(
    *,
    nodes: Float1D,
    values: FloatND,
    midpoints: Float1D,
    exact: FloatND,
    config: AdaptiveOuterMesh,
    interp: LocalCubicOuterInterpolant,
) -> tuple[BoolND, float]:
    """Which intervals fail midpoint validation, and the largest error.

    The normalized error is `|exact - interp| / (atol + rtol * scale)`; a
    midpoint where exactly one of the two is nonfinite scores `inf` (the
    interpolant misjudged feasibility), both-nonfinite scores `0` (a
    genuinely infeasible region needs no resolution).
    """
    state_ndim = values.ndim - 1
    query = midpoints.reshape(midpoints.shape[0], *([1] * state_ndim))
    interpolated = interp.evaluate(nodes=nodes, values=values, query=query)

    exact_finite = jnp.isfinite(exact)
    interp_finite = jnp.isfinite(interpolated)
    scale = jnp.maximum(jnp.abs(exact), jnp.abs(interpolated))
    normalized = jnp.abs(exact - interpolated) / (
        config.value_atol + config.value_rtol * scale
    )
    error = jnp.where(
        exact_finite & interp_finite,
        normalized,
        jnp.where(exact_finite ^ interp_finite, jnp.inf, 0.0),
    )
    state_axes = tuple(range(1, error.ndim))
    interval_error = jnp.max(error, axis=state_axes) if state_axes else error  # (C-1,)

    # Optimum-containing intervals: the two intervals flanking each cell's
    # best node, held to a tighter standard plus the margin rule.
    finite_values = jnp.where(jnp.isfinite(values), values, -jnp.inf)
    best_idx = jnp.argmax(finite_values, axis=0)  # (*S,)
    n_intervals = nodes.shape[0] - 1
    interval_ids = jnp.arange(n_intervals).reshape(n_intervals, *([1] * state_ndim))
    contains_opt = (interval_ids == best_idx - 1) | (interval_ids == best_idx)

    sorted_values = jnp.sort(finite_values, axis=0)
    margin = sorted_values[-1] - sorted_values[-2]  # (*S,)
    abs_error = jnp.where(
        exact_finite & interp_finite, jnp.abs(exact - interpolated), jnp.inf
    )
    margin_at_risk = margin < _MARGIN_SAFETY * abs_error

    per_cell_marked = (
        (error > 1.0)
        | (contains_opt & (error > _OPTIMUM_TIGHTENING))
        | (contains_opt & margin_at_risk)
    )
    marked = (
        jnp.any(per_cell_marked, axis=state_axes) if state_axes else per_cell_marked
    )
    marked = _close_over_neighbors(
        marked=np.asarray(marked), interval_error=np.asarray(interval_error)
    )
    return jnp.asarray(marked), float(jnp.max(interval_error))


def _close_over_neighbors(
    *, marked: np.ndarray, interval_error: np.ndarray
) -> np.ndarray:
    """Transitively co-mark near-threshold neighbors of marked intervals.

    See `_NEIGHBOR_FRACTION`: refining one interval degrades its neighbors'
    accuracy (graded-mesh slope-error asymmetry), so neighbors already
    within the amplification factor of the threshold must refine in the
    same round or the front creeps one interval per round.
    """
    marked = marked.copy()
    at_risk = interval_error > _NEIGHBOR_FRACTION
    while True:
        neighbor = np.zeros_like(marked)
        neighbor[1:] |= marked[:-1]
        neighbor[:-1] |= marked[1:]
        grow = neighbor & ~marked & at_risk
        if not grow.any():
            return marked
        marked |= grow


def _insert_midpoints(
    *,
    nodes: Float1D,
    values: FloatND,
    midpoints: Float1D,
    midpoint_values: FloatND,
) -> tuple[Float1D, FloatND]:
    merged_nodes = jnp.concatenate([nodes, midpoints])
    merged_values = jnp.concatenate([values, midpoint_values], axis=0)
    order = jnp.argsort(merged_nodes)
    return merged_nodes[order], merged_values[order]


def _fail_if_bad_initial_mesh(*, nodes: Float1D, config: AdaptiveOuterMesh) -> None:
    if nodes.ndim != 1 or nodes.shape[0] < 2:  # noqa: PLR2004
        msg = f"initial mesh must be 1-D with >= 2 nodes, got shape {nodes.shape}."
        raise OuterSearchConvergenceError(msg)
    if not bool(jnp.all(nodes[1:] > nodes[:-1])):
        msg = "initial mesh nodes must be strictly increasing."
        raise OuterSearchConvergenceError(msg)
    if nodes.shape[0] > config.max_nodes:
        msg = (
            f"initial mesh already exceeds max_nodes: {nodes.shape[0]} > "
            f"{config.max_nodes}."
        )
        raise OuterSearchConvergenceError(msg)
