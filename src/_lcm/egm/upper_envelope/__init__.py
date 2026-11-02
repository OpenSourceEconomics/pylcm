"""Upper-envelope refinement of EGM candidate solutions.

Discrete choices make the EGM candidate value correspondence multi-valued in
non-concave regions; the algorithms here select the upper envelope and drop
dominated candidates. The EGM step obtains its backend through
`get_upper_envelope`, so additional algorithms slot in without touching the
step itself. Currently implemented:

- `ExactEnvelope` (`_lcm.egm.upper_envelope.segment_envelope`), the default: a
  cell-driven construction that settles ownership by certified comparison, so
  it recovers a branch owning an interior subinterval even where that branch
  wins at no candidate abscissa,
- `FUESEnvelope` (`_lcm.egm.upper_envelope.fues`), the Fast Upper-Envelope Scan — a
  sequential scan that inserts crossing points between the segments its
  jump-threshold heuristic identifies,
- `RFCEnvelope` (`_lcm.egm.upper_envelope.rfc`), the Rooftop-Cut algorithm — a
  parallel dominance test that only deletes points (no crossing insertion) and
  generalizes to multidimensional endogenous grids, and
- `LTMEnvelope` (`_lcm.egm.upper_envelope.ltm`), the local-upper-bound brute method —
  an $O(K^2)$ dense segment scan that evaluates the envelope at every candidate
  abscissa (the quadratic baseline of Dobrescu & Shanker 2024).

All backends share one signature: they consume the candidate
`(endog_grid, policy, value)` rows plus the candidate supgradient
`marginal_utility` ($\\mu = \\partial v / \\partial R$, exact by the envelope
theorem) and return a NaN-padded weakly-ascending refined `(grid, policy,
value)` triple plus the kept-point count. Only RFC consumes the supgradient, to
build each point's tangent; the others recover slopes from the segments.
"""

import functools
from collections.abc import Callable
from typing import Literal, NamedTuple, Protocol, cast, overload, runtime_checkable

import jax.numpy as jnp

from _lcm.egm.interp import prepare_padded_grid
from _lcm.egm.upper_envelope.fues import (
    QueryBracket,
    refine_to_bracket,
)
from _lcm.egm.upper_envelope.fues import (
    refine_envelope as refine_envelope_fues,
)
from _lcm.egm.upper_envelope.ltm import refine_envelope as refine_envelope_ltm
from _lcm.egm.upper_envelope.rfc import refine_envelope as refine_envelope_rfc
from _lcm.egm.upper_envelope.segment_envelope import (
    OwnedEnvelope,
    _line_read,
    _NegmBindingEnvelope,
    _OwnedNegmBindingEnvelope,
    refine_envelope_exact,
)
from _lcm.solution.dcegm import (
    DCEGM,
    ExactEnvelope,
    FUESEnvelope,
    LTMEnvelope,
    RFCEnvelope,
)
from lcm.typing import Float1D, FloatND, ScalarBool, ScalarFloat, ScalarInt


class OwnedQueryBracket(NamedTuple):
    """A bracket and original saving selected with its complete owned support.

    Exact nodes select the right copy independently of adjacent links. Queries
    at or above the last node clamp to that endpoint; queries below the first
    node extend its valid owning secant. Interior gaps remain unsupported.
    """

    bracket: QueryBracket
    savings: ScalarFloat
    owner_id: ScalarInt
    side_id: ScalarInt
    read_supported: ScalarBool
    lower_savings: ScalarFloat
    upper_savings: ScalarFloat
    lower_owner_id: ScalarInt
    upper_owner_id: ScalarInt
    lower_side_id: ScalarInt
    upper_side_id: ScalarInt
    link_valid: ScalarBool
    construction_invalid: ScalarBool
    """The full emitted prefix has invalid associated channels or exceeds capacity."""


@runtime_checkable
class UpperEnvelopeBackend(Protocol):
    """Refine one row of EGM candidates to its upper envelope."""

    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal_utility: Float1D,
        savings: Float1D,
        cell_width: int = 1,
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
        """Return refined (grid, policy, value) rows and the kept-point count.

        The supgradient `marginal_utility` carries $\\mu = \\partial v /
        \\partial R$ per candidate — the exact value-row slope by the envelope
        theorem. `savings` carries each candidate's exogenous source savings
        (the savings node for Euler candidates, the borrowing limit for
        constrained ones); FUES uses it to judge savings monotonicity exactly,
        the other backends ignore it. The refined rows are NaN-padded to a
        static length; a kept-point count exceeding that length signals overflow
        (the rows then hold a truncated prefix of the envelope).
        """
        ...


@runtime_checkable
class _OwnedUpperEnvelopeBackend(Protocol):
    """Refine candidates while publishing their associated original provenance."""

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal_utility: Float1D,
        savings: Float1D,
        _binding_source_count: int,
        _binding_queries: FloatND,
        cell_width: int = 1,
    ) -> _OwnedNegmBindingEnvelope: ...

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal_utility: Float1D,
        savings: Float1D,
        cell_width: int = 1,
    ) -> OwnedEnvelope: ...


@overload
def get_upper_envelope(
    *,
    solver: DCEGM,
    n_refined: int,
    publish_provenance: Literal[False] = False,
) -> UpperEnvelopeBackend: ...


@overload
def get_upper_envelope(
    *,
    solver: DCEGM,
    n_refined: int,
    publish_provenance: Literal[True],
) -> _OwnedUpperEnvelopeBackend: ...


def get_upper_envelope(
    *, solver: DCEGM, n_refined: int, publish_provenance: bool = False
) -> UpperEnvelopeBackend | _OwnedUpperEnvelopeBackend:
    """Build the upper-envelope backend selected by the solver configuration.

    Args:
        solver: The regime's DC-EGM solver configuration; its typed `envelope`
            object selects and configures the backend.
        n_refined: Static length of the refined output rows.
        publish_provenance: Publish original saving and topology with ExactEnvelope.

    Returns:
        The configured backend.

    """
    if publish_provenance and not isinstance(solver.envelope, ExactEnvelope):
        raise ValueError("Owned provenance publication requires ExactEnvelope.")
    if isinstance(solver.envelope, FUESEnvelope):
        return functools.partial(
            _fues_backend,
            n_refined=n_refined,
            jump_thresh=solver.envelope.jump_thresh,
            n_points_to_scan=solver.envelope.n_points_to_scan,
        )

    if isinstance(solver.envelope, ExactEnvelope):
        backend = functools.partial(
            _exact_backend,
            max_runs=solver.envelope.max_runs,
            n_refined=n_refined,
            publish_provenance=publish_provenance,
        )
        if publish_provenance:
            return cast("_OwnedUpperEnvelopeBackend", backend)
        return cast("UpperEnvelopeBackend", backend)

    if isinstance(solver.envelope, RFCEnvelope):
        return functools.partial(
            _rfc_backend,
            n_refined=n_refined,
            search_radius=solver.envelope.search_radius,
            jump_thresh=solver.envelope.jump_thresh,
        )

    if isinstance(solver.envelope, LTMEnvelope):
        return functools.partial(_ltm_backend, n_refined=n_refined)

    msg = f"Unknown upper-envelope backend: {solver.envelope!r}."
    raise ValueError(msg)


@overload
def get_bracket_finder(
    *,
    solver: DCEGM,
    n_refined: int,
    publish_provenance: Literal[False] = False,
) -> Callable[..., QueryBracket]: ...


@overload
def get_bracket_finder(
    *,
    solver: DCEGM,
    n_refined: int,
    publish_provenance: Literal[True],
) -> Callable[..., OwnedQueryBracket]: ...


def get_bracket_finder(
    *, solver: DCEGM, n_refined: int, publish_provenance: bool = False
) -> Callable[..., QueryBracket | OwnedQueryBracket]:
    """Build the single-query bracket finder for the asset-row solve.

    The geometry-only counterpart of `get_upper_envelope` for asset-row mode,
    where the refined envelope is read at exactly one query per node: it returns
    the two bracketing envelope nodes (plus the first node and the kept count).
    It is deliberately not on the `UpperEnvelopeBackend` Protocol — the backend
    returns envelope geometry; the asset-row module owns the EGM economics
    (utility gradients, the borrowing limit, the constrained floor).

    Every backend reaches the bracket the same way: it materializes the full
    refined envelope row of static length `n_refined` and locates the
    `searchsorted(side="right")`-clamped bracketing node pair. `"fues"`'s
    `refine_to_bracket` builds that row via `refine_envelope` and slices it, so
    the published `(value, policy)` is identical to a
    full-envelope-then-interpolate by construction (it reads the same row). A
    genuinely streamed, sub-`n_refined`-memory bracket finder is future work for
    all backends.

    Args:
        solver: The regime's DC-EGM solver configuration; its typed `envelope`
            object selects and configures the scan.
        n_refined: Static length of the refined envelope row every finder
            materializes before locating the bracket. This is the `n_pad`
            overflow threshold the asset-row publish compares `n_kept` against.
        publish_provenance: Publish associated saving and support with ExactEnvelope.

    Returns:
        The configured bracket finder.

    """
    if publish_provenance and not isinstance(solver.envelope, ExactEnvelope):
        raise ValueError("Owned provenance publication requires ExactEnvelope.")
    if isinstance(solver.envelope, FUESEnvelope):
        return functools.partial(
            _fues_bracket_finder,
            n_refined=n_refined,
            jump_thresh=solver.envelope.jump_thresh,
            n_points_to_scan=solver.envelope.n_points_to_scan,
        )

    if isinstance(solver.envelope, RFCEnvelope):
        return functools.partial(
            _rfc_bracket_finder,
            n_refined=n_refined,
            search_radius=solver.envelope.search_radius,
            jump_thresh=solver.envelope.jump_thresh,
        )

    if isinstance(solver.envelope, LTMEnvelope):
        return functools.partial(_ltm_bracket_finder, n_refined=n_refined)

    if isinstance(solver.envelope, ExactEnvelope):
        return functools.partial(
            _exact_bracket_finder,
            max_runs=solver.envelope.max_runs,
            n_refined=n_refined,
            publish_provenance=publish_provenance,
        )

    msg = f"Unknown upper-envelope backend: {solver.envelope!r}."
    raise ValueError(msg)


def _fues_backend(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    n_refined: int,
    jump_thresh: float,
    n_points_to_scan: int | None,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Run the FUES scan with the solver's thresholds.

    FUES recovers segment slopes from its own forward scan, so the candidate
    supgradient is not consumed; the exogenous source savings resolve the
    savings-monotonicity test exactly.
    """
    del cell_width, marginal_utility
    return refine_envelope_fues(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
        savings=savings,
    )


def _exact_backend(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    max_runs: int,
    n_refined: int,
    publish_provenance: bool = False,
    _binding_source_count: int | None = None,
    _binding_queries: FloatND | None = None,
) -> (
    tuple[Float1D, Float1D, Float1D, ScalarInt]
    | OwnedEnvelope
    | _NegmBindingEnvelope
    | _OwnedNegmBindingEnvelope
):
    """Build the exact segment envelope of the candidate cloud.

    The construction reads its topology off the candidate order and its geometry
    off the value links. Original saving follows those same owning links when
    provenance is requested; the candidate supgradient is not consumed.
    """
    del marginal_utility
    if (_binding_source_count is None) != (_binding_queries is None):
        raise ValueError("Binding queries require their source-prefix count.")
    if _binding_queries is not None and _binding_source_count is not None:
        if publish_provenance:
            return refine_envelope_exact(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
                max_runs=max_runs,
                cell_width=cell_width,
                savings=savings,
                _binding_source_count=_binding_source_count,
                _binding_queries=_binding_queries,
            )
        return refine_envelope_exact(
            endog_grid=endog_grid,
            policy=policy,
            value=value,
            n_refined=n_refined,
            max_runs=max_runs,
            cell_width=cell_width,
            _binding_source_count=_binding_source_count,
            _binding_queries=_binding_queries,
        )
    return refine_envelope_exact(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
        max_runs=max_runs,
        cell_width=cell_width,
        savings=savings if publish_provenance else None,
    )


def _rfc_backend(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    n_refined: int,
    search_radius: int,
    jump_thresh: float,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Run the rooftop cut with the solver's thresholds.

    The exogenous source savings are a FUES-only refinement; RFC judges
    monotonicity from its own geometry.
    """
    del cell_width, savings
    return refine_envelope_rfc(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal_utility=marginal_utility,
        n_refined=n_refined,
        search_radius=search_radius,
        jump_thresh=jump_thresh,
    )


def _ltm_backend(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    n_refined: int,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Run the brute local-upper-bound scan.

    LTM recovers segment slopes from the candidate chain, so neither the
    candidate supgradient nor the exogenous source savings are consumed.
    """
    del cell_width, marginal_utility, savings
    return refine_envelope_ltm(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
    )


def _fues_bracket_finder(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    x_query: ScalarFloat,
    n_refined: int,
    jump_thresh: float,
    n_points_to_scan: int | None,
) -> QueryBracket:
    """Build FUES's full refined row and slice the query bracket.

    FUES recovers segment slopes from its own forward scan, so the candidate
    supgradient is not consumed; the exogenous source savings resolve the
    savings-monotonicity test exactly. Like every backend it materializes the
    full `n_refined` row (`refine_to_bracket` builds it via `refine_envelope`)
    and slices the bracketing pair — there is no O(1) streamed carry.
    """
    del cell_width, marginal_utility
    return refine_to_bracket(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        x_query=x_query,
        n_refined=n_refined,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
        savings=savings,
    )


def _rfc_bracket_finder(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    x_query: ScalarFloat,
    n_refined: int,
    search_radius: int,
    jump_thresh: float,
) -> QueryBracket:
    """Locate the query bracket from RFC's full refined envelope.

    Runs the rooftop cut to a full NaN-padded envelope row, then reads the
    bracket the row path would: the `searchsorted(side="right")` pair clamped to
    `[1, max(n_kept - 1, 1)]` (the `interp_on_prepared_grid` rule), so the
    published value cannot diverge from full-envelope-then-interpolate. The
    exogenous source savings are a FUES-only refinement.
    """
    del cell_width, savings
    refined_grid, refined_policy, refined_value, n_kept = refine_envelope_rfc(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal_utility=marginal_utility,
        n_refined=n_refined,
        search_radius=search_radius,
        jump_thresh=jump_thresh,
    )
    return _bracket_from_refined_row(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=n_kept,
        x_query=x_query,
    )


def _ltm_bracket_finder(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    x_query: ScalarFloat,
    n_refined: int,
) -> QueryBracket:
    """Locate the query bracket from LTM's full refined envelope.

    Runs the brute scan to a full NaN-padded envelope row, then reads the
    bracket the row path would: the `searchsorted(side="right")` pair clamped to
    `[1, max(n_kept - 1, 1)]` (the `interp_on_prepared_grid` rule), so the
    published value cannot diverge from full-envelope-then-interpolate. Like
    every backend — FUES's `refine_to_bracket` included — it materializes the
    full `n_pad` row and slices; no backend streams an O(1) carry. The exogenous
    source savings are a FUES-only refinement.
    """
    del cell_width, marginal_utility, savings
    refined_grid, refined_policy, refined_value, n_kept = refine_envelope_ltm(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
    )
    return _bracket_from_refined_row(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=n_kept,
        x_query=x_query,
    )


def _exact_bracket_finder(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    savings: Float1D,
    cell_width: int = 1,
    x_query: ScalarFloat,
    max_runs: int,
    n_refined: int,
    publish_provenance: bool = False,
) -> QueryBracket | OwnedQueryBracket:
    """Locate the query bracket from the exact segment envelope.

    Builds the same full NaN-padded row the full-envelope path publishes and
    slices the bracketing pair, so the two paths cannot diverge.
    """
    del marginal_utility
    refined = refine_envelope_exact(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
        max_runs=max_runs,
        cell_width=cell_width,
        savings=savings if publish_provenance else None,
    )
    if isinstance(refined, OwnedEnvelope):
        return _bracket_from_owned_row(refined=refined, x_query=x_query)
    refined_grid, refined_policy, refined_value, n_kept = refined
    return _bracket_from_refined_row(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=n_kept,
        x_query=x_query,
    )


def _bracket_from_owned_row(
    *, refined: OwnedEnvelope, x_query: ScalarFloat
) -> OwnedQueryBracket:
    """Read original saving from the exact bracket that owns the query."""
    active_prefix = jnp.arange(refined.grid.size) < refined.n_kept
    construction_invalid = (refined.n_kept > refined.grid.size) | jnp.any(
        active_prefix
        & (
            ~jnp.isfinite(refined.grid)
            | ~jnp.isfinite(refined.policy)
            | ~(jnp.isfinite(refined.value) | jnp.isneginf(refined.value))
            | ~jnp.isfinite(refined.savings)
        )
    )
    search_grid, valid_length = prepare_padded_grid(refined.grid)
    insertion = jnp.searchsorted(search_grid, x_query, side="right")
    exact_index = jnp.clip(insertion - 1, 0, jnp.maximum(valid_length - 1, 0)).astype(
        jnp.int32
    )
    exact = (valid_length > 0) & (refined.grid[exact_index] == x_query)
    last = jnp.maximum(valid_length - 1, 0)
    upper_clamp = (valid_length > 0) & (x_query >= refined.grid[last])
    endpoint = exact | upper_clamp
    endpoint_index = jnp.where(upper_clamp, last, exact_index)
    upper = jnp.clip(insertion, 1, jnp.maximum(valid_length - 1, 1)).astype(jnp.int32)
    lower = upper - 1
    pair_link = (
        refined.link_valid[jnp.clip(lower, 0, refined.link_valid.shape[0] - 1)]
        if refined.link_valid.shape[0]
        else jnp.zeros((), dtype=bool)
    )
    lower = jnp.where(endpoint, endpoint_index, lower)
    upper = jnp.where(endpoint, endpoint_index, upper)
    bracket = QueryBracket(
        lower_grid=refined.grid[lower],
        upper_grid=refined.grid[upper],
        lower_policy=refined.policy[lower],
        upper_policy=refined.policy[upper],
        lower_value=refined.value[lower],
        upper_value=refined.value[upper],
        first_grid=refined.grid[0],
        n_kept=refined.n_kept,
    )
    saving, status = _line_read(
        low=lower,
        high=upper,
        x_query=jnp.where(endpoint, refined.grid[endpoint_index], x_query),
        endog_grid=refined.grid,
        ordinate=refined.savings,
    )
    readable = (
        jnp.isfinite(bracket.lower_policy)
        & jnp.isfinite(bracket.upper_policy)
        & ~jnp.isnan(bracket.lower_value)
        & ~jnp.isnan(bracket.upper_value)
        & jnp.isfinite(saving)
        & (status == 0)
    )
    supported = (
        (endpoint | pair_link) & readable & (refined.n_kept <= refined.grid.shape[0])
    )
    return OwnedQueryBracket(
        bracket=bracket,
        savings=jnp.where(supported, saving, jnp.nan),
        owner_id=jnp.where(supported, refined.owner_id[lower], -1),
        side_id=jnp.where(supported & endpoint, refined.side_id[lower], 0),
        read_supported=supported,
        lower_savings=refined.savings[lower],
        upper_savings=refined.savings[upper],
        lower_owner_id=refined.owner_id[lower],
        upper_owner_id=refined.owner_id[upper],
        lower_side_id=refined.side_id[lower],
        upper_side_id=refined.side_id[upper],
        link_valid=~endpoint & pair_link,
        construction_invalid=construction_invalid,
    )


def _bracket_from_refined_row(
    *,
    refined_grid: Float1D,
    refined_policy: Float1D,
    refined_value: Float1D,
    n_kept: ScalarInt,
    x_query: ScalarFloat,
) -> QueryBracket:
    """Read the query bracket off a full NaN-padded refined envelope row.

    Reproduces the asset-row publish's bracket location node-for-node by reusing
    the `interp_on_prepared_grid` rule: `searchsorted(search_grid, x_query,
    side="right")` clamped to `[1, max(valid_length - 1, 1)]`, with the NaN tail
    treated as $+\\infty$. The gathered `(lower, upper)` pair, `first_grid`, and
    `n_kept` are exactly what `publish_node_from_bracket` consumes, so a
    full-envelope-then-interpolate and this bracket read publish identical
    `(value, policy)`.

    Args:
        refined_grid: Refined endogenous grid row (NaN-padded tail).
        refined_policy: Refined policy row, NaN-padded in lockstep.
        refined_value: Refined value row, NaN-padded in lockstep.
        n_kept: Number of envelope points kept; `> n_pad` signals overflow.
        x_query: The single point at which the envelope is read.

    Returns:
        The query bracket.

    """
    search_grid, valid_length = prepare_padded_grid(refined_grid)
    upper = jnp.clip(
        jnp.searchsorted(search_grid, x_query, side="right"),
        1,
        jnp.maximum(valid_length - 1, 1),
    ).astype(jnp.int32)
    lower = upper - 1
    dtype = refined_grid.dtype
    return QueryBracket(
        lower_grid=refined_grid[lower].astype(dtype),
        upper_grid=refined_grid[upper].astype(dtype),
        lower_policy=refined_policy[lower].astype(dtype),
        upper_policy=refined_policy[upper].astype(dtype),
        lower_value=refined_value[lower].astype(dtype),
        upper_value=refined_value[upper].astype(dtype),
        first_grid=refined_grid[0].astype(dtype),
        n_kept=n_kept,
    )
