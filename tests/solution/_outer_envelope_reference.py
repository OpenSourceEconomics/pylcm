"""Ex-post reference for the NEGM outer envelope's query-side maximum.

The production read path (`continuation._collapse_stacked_candidates`) collapses
the stacked outer candidates of `build_stacked_outer_carry` itself and takes the
value `argmax` with the lowest candidate index on exact ties. This module reads
every candidate at the query through the parent's interpolation convention, takes
the pointwise maximum there and publishes the winner's marginal (Danskin), never an
average across a crossing. Exact value ties resolve by the candidates' value germs:
the right germ first, then the left germ, then the lowest index. It is a reference
for the query-side maximum with that exact tie rule, not a byte-for-byte oracle of
the production aggregation.

The germs are the complete local description of a Hermite value read immediately to
one side of a query; they are built on the production bracket derivatives
(`interp._hermite_bracket_derivatives`).
"""

import functools

import jax
import jax.numpy as jnp

from _lcm.egm.interp import (
    _hermite_bracket_derivatives,
    interp_left_record_on_prepared_grid,
    interp_on_padded_grid,
    prepare_padded_grid,
)
from lcm.typing import Bool1D, BoolND, Float1D, FloatND, IntND, ScalarInt


def outer_envelope_at_query(
    *,
    candidate_endog: FloatND,
    candidate_value: FloatND,
    candidate_marginal: FloatND,
    x_query: Float1D,
) -> tuple[Float1D, Float1D]:
    """Pointwise upper envelope `max_j V_j(q)` of the lifted outer candidates.

    Each candidate row is read at every query through the parent's own
    interpolation convention — edge-clamped Fritsch-Carlson-limited cubic Hermite
    value with the marginal row as node slopes, a separate linear marginal read,
    and the value masked to `-inf` below the candidate's first finite coh node
    (its borrowing-constrained support has not started, and its marginal is
    zeroed alongside, so an all-infeasible query publishes the `(-inf, 0)`
    infeasible pair). Strictly above a candidate's own last finite node the
    value read clamps to a constant, so both marginal payloads are re-pinned to
    zero there per candidate — a winner queried past its own support publishes
    the locally constant envelope's zero slope, never its terminal record.
    The published value is the pointwise maximum over
    candidates; the published marginal is the *winning* candidate's resource
    slope (Danskin), so it is winner-consistent and never averaged across a
    branch crossing. At an exact value tie the winner is right-continuous in
    the value read itself: the tied candidates are compared by the complete
    right germ of their own value interpolants (`right_germ_winner`) — each
    local piece is a limited cubic Hermite or a constant clamp, so
    right-finiteness plus the first three one-sided derivatives determine the
    read on a right neighborhood exactly, and the branch whose read actually
    wins immediately to the right of the query owns the published (economic)
    marginal, matching the one-sided convention the parent's Euler inversion
    expects. Candidates whose right germs coincide — at a shared terminal
    abscissa every candidate clamps, so the right germ cannot discriminate —
    are compared by their *left* germs: the branch that carries the envelope
    on the left neighborhood wins ownership. Only candidates whose local pieces
    literally coincide on both sides fall back to the lowest index, a
    deterministic choice among identical branches. The germ decides ownership;
    the published payload is the winner's economic marginal — the two stay
    separate objects. On left-owned cells that payload is the winner's
    *left-record* marginal (`interp_left_record_on_padded_grid`), so a winner
    whose terminal abscissa is duplicated publishes the left duplicate's record
    — the one that justified ownership — not the right one.

    Taking the maximum at the query — rather than at a shared node grid and
    republishing a single interpolated row — is exact for the finite candidate set
    at every query: a candidate that wins only on an interval strictly between two
    nodes is read at its true value there instead of being bridged upward by a
    shared-node reinterpolation.

    Args:
        candidate_endog: Lifted common-coh grids, `(n_candidates, n_pad)`, each
            NaN-padded in the tail.
        candidate_value: Conditional value rows, `(n_candidates, n_pad)`.
        candidate_marginal: Conditional resource-marginal rows,
            `(n_candidates, n_pad)`.
        x_query: Query cash-on-hand points, `(n_query,)`.

    Returns:
        Tuple of the envelope value and the winner's marginal, each `(n_query,)`.

    """
    values, marginals, left_marginals, right_germ, left_germ = jax.vmap(
        functools.partial(_read_candidate_at_query, x_query=x_query)
    )(endog=candidate_endog, value=candidate_value, marginal=candidate_marginal)
    winner, left_owned = right_germ_winner(
        value=values.T,
        right_germ=_transposed_germ(right_germ),
        left_germ=_transposed_germ(left_germ),
    )
    # The payload follows the ownership side: the winner's ordinary
    # (right-continuous) marginal on right-decided cells, its left record on
    # left-owned ones — the two differ exactly at duplicated abscissae.
    envelope_marginal = jnp.where(
        left_owned,
        jnp.take_along_axis(left_marginals.T, winner, axis=-1),
        jnp.take_along_axis(marginals.T, winner, axis=-1),
    )[..., 0]
    # The published value is the maximum itself: identical to the winner's read
    # at any tie, and NaN-propagating when a poisoned candidate row (whose NaN
    # empties the tie set) must surface fail-loud.
    return jnp.max(values, axis=0), envelope_marginal


def right_germ_winner(
    *,
    value: FloatND,
    right_germ: tuple[BoolND, FloatND, FloatND, FloatND],
    left_germ: tuple[BoolND, FloatND, FloatND, FloatND],
) -> tuple[IntND, BoolND]:
    """Select the tie-owning candidate index along the trailing candidate axis.

    Staged lexicographic comparison, each stage exact (no packing, no
    tolerance — the claim is exact ordering of the reads' own local pieces):

    - only candidates attaining the maximum value compete,
    - a right-finite read beats one that dies to `-inf` immediately right,
    - then the first, second, and third right derivatives in turn (the local
      pieces are cubics or constant clamps, so agreement through the third
      derivative means the pieces coincide on a right neighborhood),
    - candidates still tied are right-identical — at a shared terminal
      abscissa every candidate clamps right — so the left germ decides:
      left-finite first, then the branch maximizing the read at `q - ε`
      (lexicographically the *smallest* first, *largest* second, *smallest*
      third left derivative), so the published payload is that branch's own
      economic marginal at such a boundary,
    - `argmax` resolves what remains to the lowest index, a deterministic
      choice among branches identical on both sides.

    Ownership has a *side*, and the published payload must follow it: when
    the right stages fully separate the tie set, the winner owns a right
    neighborhood and the ordinary (right-continuous) marginal read is the
    consistent payload; when they cannot — the surviving candidates are
    right-identical and the left stages (or the index fallback among
    left-identical branches) decide — ownership is a statement about the left
    neighborhood, and the winner's *left-record* marginal is the payload that
    justified it (at a duplicated terminal abscissa the two differ).

    Args:
        value: Candidate value reads; the candidate axis is last.
        right_germ: Tuple of the right-finiteness flag and the first three
            right derivatives of the candidate value reads, same shape.
        left_germ: Tuple of the left-finiteness flag and the first three
            left derivatives of the candidate value reads, same shape.

    Returns:
        Tuple of the winning candidate index per query cell and the
        left-ownership flag (True where the right stages left the tie
        undecided, so the winner's left-record payload applies), each with
        the candidate axis kept as a trailing length-1 axis (for
        `take_along_axis`).

    """
    right_finite, first, second, third = right_germ
    left_finite, left_first, left_second, left_third = left_germ
    survivors = value >= jnp.max(value, axis=-1, keepdims=True)
    for stage_key in (right_finite.astype(value.dtype), first, second, third):
        stage = jnp.where(survivors, stage_key, -jnp.inf)
        survivors = survivors & (stage >= jnp.max(stage, axis=-1, keepdims=True))
    left_owned = jnp.sum(survivors, axis=-1, keepdims=True) > 1
    left_stage_keys = (
        left_finite.astype(value.dtype),
        -left_first,
        left_second,
        -left_third,
    )
    for stage_key in left_stage_keys:
        stage = jnp.where(survivors, stage_key, -jnp.inf)
        survivors = survivors & (stage >= jnp.max(stage, axis=-1, keepdims=True))
    # int32 winner indices: the candidate axis has at most a few hundred
    # entries, so the x64-default int64 only doubles the gather-index buffers.
    return jnp.argmax(survivors, axis=-1, keepdims=True).astype(jnp.int32), left_owned


def _read_candidate_at_query(
    *, endog: Float1D, value: Float1D, marginal: Float1D, x_query: Float1D
) -> tuple[
    Float1D,
    Float1D,
    Float1D,
    tuple[Bool1D, Float1D, Float1D, Float1D],
    tuple[Bool1D, Float1D, Float1D, Float1D],
]:
    """Read one lifted candidate row at every query, support-masked.

    Returns the value read, the ordinary and left-record marginal reads, and
    the right and left value germs; see `outer_envelope_at_query` for the
    conventions each channel follows.
    """
    cand_lower = jnp.min(jnp.where(jnp.isfinite(endog), endog, jnp.inf))
    value_at_query = interp_on_padded_grid(
        x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
    )
    marginal_at_query = interp_on_padded_grid(x_query=x_query, xp=endog, fp=marginal)
    left_marginal_at_query = interp_left_record_on_padded_grid(
        x_query=x_query, xp=endog, fp=marginal
    )
    right_germ_at_query = interp_right_germ_on_padded_grid(
        x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
    )
    left_germ_at_query = interp_left_germ_on_padded_grid(
        x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
    )
    # The support mask applies only where a finite first node exists:
    # `cand_lower` is `+inf` on an all-NaN (poisoned) row, whose NaN read
    # must reach the maximum fail-loud instead of becoming an ordinary
    # infeasible `(-inf, 0)` pair.
    below_support = (x_query < cand_lower) & jnp.isfinite(cand_lower)
    value_at_query = jnp.where(below_support, -jnp.inf, value_at_query)
    marginal_at_query = jnp.where(below_support, 0.0, marginal_at_query)
    left_marginal_at_query = jnp.where(below_support, 0.0, left_marginal_at_query)
    # Strictly above a candidate's own last finite node its value read is
    # a constant clamp, so its marginal payload is exactly zero there —
    # the separate linear marginal read would republish the terminal
    # record of a node strictly below the query. Re-pinned per candidate,
    # BEFORE the collapse, so an earlier-ending clamp winner cannot
    # publish a stale record; at exact equality the node's own record
    # stands. `cand_upper` is `-inf` on an all-NaN row (mask off — the
    # NaN read stays poisonous).
    cand_upper = jnp.max(jnp.where(jnp.isfinite(endog), endog, -jnp.inf))
    above_support = (x_query > cand_upper) & jnp.isfinite(cand_upper)
    marginal_at_query = jnp.where(above_support, 0.0, marginal_at_query)
    left_marginal_at_query = jnp.where(above_support, 0.0, left_marginal_at_query)
    return (
        value_at_query,
        marginal_at_query,
        left_marginal_at_query,
        right_germ_at_query,
        left_germ_at_query,
    )


def _transposed_germ(
    germ: tuple[BoolND, FloatND, FloatND, FloatND],
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Transpose every component of a value germ."""
    alive, first, second, third = germ
    return (alive.T, first.T, second.T, third.T)


def interp_left_record_on_padded_grid(
    *,
    x_query: FloatND,
    xp: Float1D,
    fp: Float1D,
) -> FloatND:
    """Linear read that publishes the *left* record at a duplicated abscissa.

    Same row contract as `interp_on_padded_grid`; see
    `interp_left_record_on_prepared_grid` for the read semantics.

    Args:
        x_query: Points at which to evaluate; any shape.
        xp: Weakly ascending grid row with NaNs only in the tail.
        fp: Function values on `xp`, NaN-padded in lockstep with `xp`.

    Returns:
        Interpolated values with the shape of `x_query`.

    """
    search_grid, valid_length = prepare_padded_grid(xp)
    return interp_left_record_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
    )


def interp_right_germ_on_padded_grid(
    *,
    x_query: FloatND,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Compute the right germ of the Hermite value read on a padded row.

    Same row contract as `interp_on_padded_grid`; see
    `interp_right_germ_on_prepared_grid` for the germ semantics.

    Args:
        x_query: Points at which to evaluate the right germ; any shape.
        xp: Weakly ascending grid row with NaNs only in the tail.
        fp: Function values on `xp`, NaN-padded in lockstep with `xp`.
        fp_slopes: Derivatives of `fp` with respect to `xp` at the `xp` nodes,
            NaN-padded in lockstep.

    Returns:
        Tuple of the right-finiteness flag and the first, second, and third
        right derivatives, each with the shape of `x_query`.

    """
    search_grid, valid_length = prepare_padded_grid(xp)
    return interp_right_germ_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


def interp_right_germ_on_prepared_grid(
    *,
    x_query: FloatND,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Compute the right germ of the Hermite value read at each query.

    The germ is the complete local description of the *value interpolant* that
    `interp_on_prepared_grid` evaluates with `fp_slopes` immediately to the
    right of the query — not a read of the slope row itself. Each local piece
    is a cubic (or a constant clamp), so the germ is finite-dimensional: a
    right-finiteness flag plus the first, second, and third one-sided
    derivatives determine the read on a right neighborhood exactly. The germ
    differs from the raw slope row exactly where a tie-owner decision needs the
    truth: the Fritsch-Carlson limiter may cap a node's raw slope, the edge
    clamps flatten the read outside the valid range, and a `-inf` bracket
    endpoint kills the read immediately right of a finite node. Semantics:

    - Strictly inside a bracket: the derivatives of that bracket's limited
      cubic Hermite (the secant and zero curvature where the correction is
      inapplicable — the linear fallback).
    - Exactly on a node: the derivatives at the left edge of the node's *right*
      bracket (the bracket search is right-continuous).
    - Strictly below the first node, and at or above the last valid node: the
      read clamps to a constant — right-finite with all derivatives zero.
    - A bracket with a non-finite endpoint value: not right-finite (the read
      is `-inf` on the bracket's interior), derivatives zero.

    Args:
        x_query: Points at which to evaluate the right germ; any shape.
        search_grid: The row's `+inf`-padded search key from
            `prepare_padded_grid`.
        valid_length: The row's non-NaN prefix length from
            `prepare_padded_grid`.
        xp: The original NaN-padded grid row (the abscissa gather source).
        fp: Function values on `xp`, NaN-padded in lockstep.
        fp_slopes: Node derivatives, NaN-padded in lockstep.

    Returns:
        Tuple of the right-finiteness flag and the first, second, and third
        right derivatives, each with the shape of `x_query`.

    The germ is the *tie selector* over the published interpolants, one of two
    derivative objects with a deliberately different terminal convention: the
    germ's first derivative is zero at and above the last valid node (the read
    clamps right of it), while the published query derivative of the value
    read (`interp_on_prepared_grid` under autodiff) is the node's own limited
    slope *at* the last node and zero only strictly above — the economics the
    parent Euler inversion consumes. The two objects agree everywhere else and
    share one bracket/limiter implementation.

    """
    first, second, third, endpoint_finite, _, _ = _hermite_bracket_derivatives(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
        side="right",
    )
    # The read clamps to a constant strictly below the first node and at or
    # above the last valid one; the germ there is right-finite with all
    # derivatives exactly zero. (`search_grid` is `+inf` on the pad, so a
    # poisoned all-NaN row lands on the lower clamp.)
    first_node = search_grid[0]
    last_node = search_grid[jnp.maximum(valid_length - 1, 0)]
    on_clamp_ray = (x_query < first_node) | (x_query >= last_node)
    right_finite = on_clamp_ray | endpoint_finite
    return (
        right_finite,
        jnp.where(on_clamp_ray, 0.0, first),
        jnp.where(on_clamp_ray, 0.0, second),
        jnp.where(on_clamp_ray, 0.0, third),
    )


def interp_left_germ_on_padded_grid(
    *,
    x_query: FloatND,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Compute the left germ of the Hermite value read on a padded row.

    Same row contract as `interp_on_padded_grid`; see
    `interp_left_germ_on_prepared_grid` for the germ semantics.

    Args:
        x_query: Points at which to evaluate the left germ; any shape.
        xp: Weakly ascending grid row with NaNs only in the tail.
        fp: Function values on `xp`, NaN-padded in lockstep with `xp`.
        fp_slopes: Derivatives of `fp` with respect to `xp` at the `xp` nodes,
            NaN-padded in lockstep.

    Returns:
        Tuple of the left-finiteness flag and the first, second, and third
        left derivatives, each with the shape of `x_query`.

    """
    search_grid, valid_length = prepare_padded_grid(xp)
    return interp_left_germ_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


def interp_left_germ_on_prepared_grid(
    *,
    x_query: FloatND,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Compute the left germ of the Hermite value read at each query.

    The mirror of `interp_right_germ_on_prepared_grid`: the complete local
    description of the value interpolant immediately to the *left* of the
    query, as a tie-selection object for the stacked candidate readers. A tie
    whose right germs are identical (both candidates clamp at a shared
    terminal abscissa) is owned by the branch that carries the envelope on the
    left neighborhood, so the published payload is that branch's own economic
    marginal — the meaningful one-sided value for a parent Euler inversion at
    the boundary. Semantics:

    - Strictly inside a bracket: the derivatives of that bracket's limited
      cubic Hermite (the secant and zero curvature where the correction is
      inapplicable — the linear fallback).
    - Exactly on a node: the derivatives at the right edge of the node's
      *left* bracket.
    - At or below the first valid node: not left-finite, derivatives zero —
      the stacked readers mask a candidate to `-inf` below its first finite
      node, so as an envelope candidate the branch is dead immediately left.
    - Strictly above the last valid node: the read clamps to a constant —
      left-finite with all derivatives zero.
    - A bracket with a non-finite endpoint value: not left-finite (the read
      is `-inf` on the bracket's interior), derivatives zero.
    - A zero-width located bracket (a duplicated first abscissa): derivatives
      zero — no piece is defined on it.

    Args:
        x_query: Points at which to evaluate the left germ; any shape.
        search_grid: The row's `+inf`-padded search key from
            `prepare_padded_grid`.
        valid_length: The row's non-NaN prefix length from
            `prepare_padded_grid`.
        xp: The original NaN-padded grid row (the abscissa gather source).
        fp: Function values on `xp`, NaN-padded in lockstep.
        fp_slopes: Node derivatives, NaN-padded in lockstep.

    Returns:
        Tuple of the left-finiteness flag and the first, second, and third
        left derivatives, each with the shape of `x_query`.

    """
    first, second, third, endpoint_finite, zero_width, _ = _hermite_bracket_derivatives(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
        side="left",
    )
    first_node = search_grid[0]
    last_node = search_grid[jnp.maximum(valid_length - 1, 0)]
    # (`search_grid` is `+inf` on the pad, so a poisoned all-NaN row is dead
    # left at every query: `first_node` is `+inf`.)
    dead_left = x_query <= first_node
    on_clamp_ray = x_query > last_node
    left_finite = (on_clamp_ray | endpoint_finite) & ~dead_left
    suppressed = dead_left | on_clamp_ray | zero_width
    return (
        left_finite,
        jnp.where(suppressed, 0.0, first),
        jnp.where(suppressed, 0.0, second),
        jnp.where(suppressed, 0.0, third),
    )
