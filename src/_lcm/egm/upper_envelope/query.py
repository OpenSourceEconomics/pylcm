"""Exact query-side upper envelope of an EGM candidate correspondence.

The query backend evaluates every live consecutive branch link, plus every live
candidate as a zero-width self-bracket, directly at the requested abscissae.
The certified mode delegates each ownership reduction to an opaque native
operation. Stored IEEE operands are decoded as exact dyadics and ordered
lexicographically by exact affine value, right extension, exact slope, and an
explicit stable stored-link index. A blocked interval fold may make several such
calls, carrying the selected link's original geometry and global identity between
calls. Only after the final winner and status are known does Python read value,
policy, and marginal. Those channels publish together or all become NaN.

This boundary is deliberate. Same-format double-double words cannot encode every
nonzero residual of normal operands, so neither a zero product tail nor two equal
rounded slope words is an equality certificate. The native winner avoids that
representability limit and keeps the integer arithmetic outside the JAX trace.
An array of queries is handled by one custom call; an outer ``vmap`` is lowered
sequentially around that opaque call. Operand position never settles a final tie,
so neither segment blocking nor interval blocking changes which candidate owns a
query in certified mode: every partition resolves the same ownership. The levels
those reads publish are formed by kernels the backend vectorizes per compiled
block width, so two partitions can name one level with adjacent bit patterns.

The ordinary mode remains a dense working-format interpolation and maximum. It
is cheaper, but it carries no exact ownership guarantee.
"""

import functools
from typing import Literal, NamedTuple, Protocol, cast, overload, runtime_checkable

import jax
import jax.numpy as jnp

from _lcm.axis_boundaries import (
    ResolvedAxisPartition,
    axis_interval_indices,
    feasibility_region_indices,
)
from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.upper_envelope._exact_affine.ffi import (
    exact_affine_read,
    exact_query_winner,
    exact_query_winner_batched,
)
from _lcm.egm.upper_envelope.certified_sign import (
    affine_numerator,
    backend_flushes_subnormals,
    is_subnormal,
)
from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_from_difference,
    dd_quotient,
    scale_by_power_of_two,
)
from lcm.typing import BoolND, Float1D, FloatND, IntND

# The owner published at a query that carries no finite envelope level.
NO_OWNER = -1

# Eight geometry/payload endpoint columns plus two associated-saving endpoints.
_LINK_COLUMNS_WITH_SAVINGS = 10


def _along_link(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
    arithmetic: ComparisonArithmetic = "certified",
) -> FloatND:
    """Read a channel along a link at `query`.

    Whether the query *is* an endpoint is decided against the stored abscissae. A
    normalized coordinate cannot answer that: `(query - left_grid) / width` is a
    rounded quotient that reaches exactly `1.0` for queries strictly inside the
    link, and a read that trusted it would publish the wrong endpoint's value and
    policy for a point that is not an endpoint at all.

    Strictly inside, the channel is `v0*(x1 - x) + v1*(x - x0)` over the width,
    evaluated in the double-double arithmetic of `double_double`. Working at
    roughly twice the format's precision is what a link whose endpoint values
    nearly cancel requires: there
    the value at the query is smaller than either endpoint by many orders of
    magnitude, so rounding the slope first spends the whole result's significance
    on the cancellation, and what survives is not enough to report, let alone to
    decide on.
    """
    if arithmetic == "ordinary":
        at_endpoint, endpoint, left_grid, divisor_grid = _link_geometry(
            query=query,
            left_grid=left_grid,
            right_grid=right_grid,
            left=left,
            right=right,
        )
        inside = ((divisor_grid - query) * left + (query - left_grid) * right) / (
            divisor_grid - left_grid
        )
        # On a node the primal is the stored endpoint, but the read is still a
        # point of the link's line: its derivative carries the line's slope times
        # the node's motion relative to the query, as the certified reader's JVP
        # does. `stop_gradient(inside) - inside` is a primal +0.0 whose tangent is
        # minus the line's, so subtracting it keeps every endpoint bit (-0.0
        # included) and swaps in the line's tangent. A zero-width self-bracket has
        # no line and keeps its endpoint's tangent.
        on_line = (right_grid != left_grid) & jnp.isfinite(inside)
        zero = jnp.where(on_line, jax.lax.stop_gradient(inside) - inside, 0.0)
        on_node = jnp.where(on_line, jax.lax.stop_gradient(endpoint), endpoint) - zero
        return jnp.where(at_endpoint, on_node, inside)
    numerator, divisor, endpoint, at_endpoint = _link_terms(
        left=left,
        right=right,
        query=query,
        left_grid=left_grid,
        right_grid=right_grid,
    )
    high, low = dd_quotient(numerator, divisor)
    return jnp.where(at_endpoint, endpoint, high + low)


def _on_the_links_own_scale(
    *, left_grid: FloatND, divisor_grid: FloatND, query: FloatND
) -> tuple[FloatND, FloatND, FloatND]:
    """Return the link's three abscissae scaled to where its differences are readable.

    A link read is a ratio of quantities built from the differences between these
    three, and it is homogeneous of degree one in them: scaling all three by one
    power of two multiplies numerator and divisor alike and leaves the ratio
    identical. Scaling by a power of two is itself exact, so nothing is traded
    for it.

    What it buys is an arithmetic that can represent what it is handed. A link at
    the bottom of the normal range has differences among the subnormals, where a
    product rounds to a whole ULP — the endpoint values it was carrying are gone,
    and the ratio reads one for a link of any value. Measuring the link on its own
    scale first puts those differences back among ordinary numbers.

    The scale is anchored on the *smallest* of the three, not the largest. What
    ruins a read is a difference falling under the smallest normal, and it is the
    smallest abscissa that a difference sits closest to; normalizing the largest
    instead would push the small end down and manufacture the same loss on a link
    whose endpoints span a wide range. The anchor is then backed off far enough to
    keep the largest of the three finite, since a link spanning the whole range
    cannot have both.
    """
    terms = (left_grid, divisor_grid, query)
    _mantissa, largest = jnp.frexp(
        jnp.maximum(
            jnp.maximum(jnp.abs(terms[0]), jnp.abs(terms[1])), jnp.abs(terms[2])
        )
    )
    smallest = _smallest_finite_exponent(*terms)
    # Scaling the smallest term to the binade around one costs the largest term
    # `smallest` binades of headroom; back the exponent off by whatever that
    # overruns the top of the range.
    headroom = jnp.asarray(jnp.finfo(left_grid.dtype).maxexp - 1, dtype=largest.dtype)
    exponent = jnp.maximum(smallest, largest - headroom)
    scaled = tuple(scale_by_power_of_two(term, -exponent) for term in terms)
    return scaled[0], scaled[1], scaled[2]


def _smallest_finite_exponent(*terms: FloatND) -> IntND:
    """Return the `frexp` exponent of the smallest finite nonzero term.

    Zero and non-finite terms carry no scale of their own and are ignored; a group
    holding nothing else scales by `2**0`, which leaves it alone.
    """
    magnitude = jnp.full_like(terms[0], jnp.inf)
    for term in terms:
        usable = jnp.isfinite(term) & (term != 0.0)
        magnitude = jnp.minimum(magnitude, jnp.where(usable, jnp.abs(term), jnp.inf))
    _mantissa, exponent = jnp.frexp(
        jnp.where(jnp.isfinite(magnitude), magnitude, 1.0),
    )
    return exponent


def _link_geometry(
    *,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
    left: FloatND,
    right: FloatND,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Whether the query is an endpoint, which one, and a divisor-safe right node."""
    at_left = query == left_grid
    at_right = query == right_grid
    # A zero-width self-bracket carries no affine line. Its own abscissa is the
    # only query it brackets, so the quotient is never read there and has only to
    # stay finite; displacing the divisor keeps it so.
    degenerate = right_grid == left_grid
    return (
        at_left | at_right | degenerate,
        jnp.where(at_left | degenerate, left, right),
        left_grid,
        jnp.where(degenerate, left_grid + 1.0, right_grid),
    )


def _link_terms(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
) -> tuple[DoubleDouble, DoubleDouble, FloatND, BoolND]:
    """Split a link read into its exact endpoint case and its affine quotient.

    The quotient is homogeneous of degree one in the link's three abscissae, so
    the link is measured on its own scale first.
    """
    at_endpoint, endpoint, left_grid, divisor_grid = _link_geometry(
        query=query, left_grid=left_grid, right_grid=right_grid, left=left, right=right
    )
    left_grid, divisor_grid, query = _on_the_links_own_scale(
        left_grid=left_grid, divisor_grid=divisor_grid, query=query
    )
    numerator = affine_numerator(
        x0=left_grid, x1=divisor_grid, v0=left, v1=right, x_query=query
    )
    return (
        numerator,
        dd_from_difference(divisor_grid, left_grid),
        endpoint,
        at_endpoint,
    )


class _SegmentLinks(NamedTuple):
    """Per-link endpoints of the candidate correspondence (length `n - 1`).

    A link is the consecutive-candidate pair `(i, i+1)`; it is a real envelope
    segment only where `live` (both endpoints finite and sharing a branch label).
    """

    left_grid: Float1D
    right_grid: Float1D
    left_value: Float1D
    right_value: Float1D
    left_policy: Float1D
    right_policy: Float1D
    left_marginal: Float1D
    right_marginal: Float1D
    live: BoolND
    stable_index: IntND
    """Global stored-link identity used only as the final total-order field."""


class _LinkRows(NamedTuple):
    """`_SegmentLinks` fields gathered per query: each `(n_query, m)`."""

    left_grid: FloatND
    right_grid: FloatND
    left_value: FloatND
    right_value: FloatND
    left_policy: FloatND
    right_policy: FloatND
    left_marginal: FloatND
    right_marginal: FloatND
    live: BoolND
    stable_index: IntND


def _segment_links_from_candidates(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None,
    feasibility_partition: ResolvedAxisPartition | None,
    feasible_interval_mask: BoolND | None,
) -> tuple[_SegmentLinks, tuple[Float1D, Float1D, Float1D, Float1D]]:
    """Build consecutive links and self-brackets from one candidate block.

    ``stable_index`` names the links in the equivalent one-shot layout.  It is
    intentionally independent of this block's local storage order: a standing
    winner may be reintroduced in a later reduction without acquiring a new
    final-order identity.
    """
    if (feasibility_partition is None) != (feasible_interval_mask is None):
        raise ValueError(
            "feasibility_partition and feasible_interval_mask must be supplied "
            "together."
        )

    if feasibility_partition is None:
        candidate_region = jnp.zeros_like(segment_id, dtype=jnp.int32)
    else:
        feasible_interval_mask = cast("BoolND", feasible_interval_mask)
        candidate_interval = axis_interval_indices(
            partition=feasibility_partition,
            values=endog_grid,
        )
        candidate_feasible = feasible_interval_mask[candidate_interval]
        candidate_region = feasibility_region_indices(
            partition=feasibility_partition,
            values=endog_grid,
        )
        endog_grid = jnp.where(candidate_feasible, endog_grid, jnp.nan)
        value = jnp.where(candidate_feasible, value, jnp.nan)
        policy = jnp.where(candidate_feasible, policy, jnp.nan)
        marginal = jnp.where(candidate_feasible, marginal, jnp.nan)

    dead = ~jnp.isfinite(endog_grid) | ~jnp.isfinite(value)
    n_candidate = endog_grid.shape[0]
    if n_candidate == 0:
        empty_float = jnp.empty((0,), dtype=endog_grid.dtype)
        empty_bool = jnp.empty((0,), dtype=bool)
        empty_int = jnp.empty((0,), dtype=jnp.int32)
        return (
            _SegmentLinks(
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_bool,
                empty_int,
            ),
            (endog_grid, value, policy, marginal),
        )

    n_link = 2 * n_candidate - 1
    if stable_index is None:
        if n_link >= jnp.iinfo(jnp.int32).max:
            msg = (
                f"A bank of {n_candidate} candidates has {n_link} link identities, "
                "more than int32 can name; read it in smaller banks."
            )
            raise ValueError(msg)
        link_stable_index = jnp.arange(n_link, dtype=jnp.int32)
    else:
        link_stable_index = jnp.asarray(stable_index, dtype=jnp.int32)
        if link_stable_index.shape != (n_link,):
            msg = (
                "stable_index must name every consecutive link and self-bracket "
                f"with shape {(n_link,)}, got {link_stable_index.shape}."
            )
            raise ValueError(msg)

    consecutive = _SegmentLinks(
        left_grid=endog_grid[:-1],
        right_grid=endog_grid[1:],
        left_value=value[:-1],
        right_value=value[1:],
        left_policy=policy[:-1],
        right_policy=policy[1:],
        left_marginal=marginal[:-1],
        right_marginal=marginal[1:],
        live=(
            ~dead[:-1]
            & ~dead[1:]
            & (segment_id[:-1] == segment_id[1:])
            & (candidate_region[:-1] == candidate_region[1:])
        ),
        stable_index=link_stable_index[: n_candidate - 1],
    )
    self_bracket = _SegmentLinks(
        left_grid=endog_grid,
        right_grid=endog_grid,
        left_value=value,
        right_value=value,
        left_policy=policy,
        right_policy=policy,
        left_marginal=marginal,
        right_marginal=marginal,
        live=~dead,
        stable_index=link_stable_index[n_candidate - 1 :],
    )
    links = _SegmentLinks(
        *(
            jnp.concatenate([pair, point])
            for pair, point in zip(consecutive, self_bracket, strict=True)
        )
    )
    return links, (endog_grid, value, policy, marginal)


@runtime_checkable
class EnvelopeQueryReader(Protocol):
    """The original query read ABI, including associated savings and owner."""

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal: Float1D,
        segment_id: Float1D,
        stable_index: IntND | None = ...,
        x_query: FloatND,
        segment_block_size: int = ...,
        arithmetic: ComparisonArithmetic = ...,
        feasibility_partition: ResolvedAxisPartition | None = ...,
        feasible_interval_mask: BoolND | None = ...,
        return_owner: Literal[False] = ...,
        candidate_savings: None = ...,
    ) -> tuple[FloatND, FloatND, FloatND]: ...

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal: Float1D,
        segment_id: Float1D,
        stable_index: IntND | None = ...,
        x_query: FloatND,
        segment_block_size: int = ...,
        arithmetic: ComparisonArithmetic = ...,
        feasibility_partition: ResolvedAxisPartition | None = ...,
        feasible_interval_mask: BoolND | None = ...,
        return_owner: Literal[True],
        candidate_savings: None = ...,
    ) -> tuple[FloatND, FloatND, FloatND, IntND]: ...

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal: Float1D,
        segment_id: Float1D,
        stable_index: IntND | None = ...,
        x_query: FloatND,
        segment_block_size: int = ...,
        arithmetic: ComparisonArithmetic = ...,
        feasibility_partition: ResolvedAxisPartition | None = ...,
        feasible_interval_mask: BoolND | None = ...,
        return_owner: bool,
        candidate_savings: None = ...,
    ) -> tuple[FloatND, FloatND, FloatND] | tuple[FloatND, FloatND, FloatND, IntND]: ...

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal: Float1D,
        segment_id: Float1D,
        stable_index: IntND | None = ...,
        x_query: FloatND,
        segment_block_size: int = ...,
        arithmetic: ComparisonArithmetic = ...,
        feasibility_partition: ResolvedAxisPartition | None = ...,
        feasible_interval_mask: BoolND | None = ...,
        return_owner: Literal[False] = ...,
        candidate_savings: Float1D,
    ) -> tuple[FloatND, FloatND, FloatND, FloatND]: ...

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal: Float1D,
        segment_id: Float1D,
        stable_index: IntND | None = ...,
        x_query: FloatND,
        segment_block_size: int = ...,
        arithmetic: ComparisonArithmetic = ...,
        feasibility_partition: ResolvedAxisPartition | None = ...,
        feasible_interval_mask: BoolND | None = ...,
        return_owner: Literal[True],
        candidate_savings: Float1D,
    ) -> tuple[FloatND, FloatND, FloatND, FloatND, IntND]: ...

    @overload
    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal: Float1D,
        segment_id: Float1D,
        stable_index: IntND | None = ...,
        x_query: FloatND,
        segment_block_size: int = ...,
        arithmetic: ComparisonArithmetic = ...,
        feasibility_partition: ResolvedAxisPartition | None = ...,
        feasible_interval_mask: BoolND | None = ...,
        return_owner: bool = ...,
        candidate_savings: Float1D,
    ) -> (
        tuple[FloatND, FloatND, FloatND, FloatND]
        | tuple[FloatND, FloatND, FloatND, FloatND, IntND]
    ): ...


@overload
def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = ...,
    x_query: FloatND,
    segment_block_size: int = ...,
    arithmetic: ComparisonArithmetic = ...,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    return_owner: Literal[False] = ...,
    candidate_savings: None = ...,
) -> tuple[FloatND, FloatND, FloatND]: ...


@overload
def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = ...,
    x_query: FloatND,
    segment_block_size: int = ...,
    arithmetic: ComparisonArithmetic = ...,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    return_owner: Literal[True],
    candidate_savings: None = ...,
) -> tuple[FloatND, FloatND, FloatND, IntND]: ...


@overload
def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = ...,
    x_query: FloatND,
    segment_block_size: int = ...,
    arithmetic: ComparisonArithmetic = ...,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    return_owner: bool,
    candidate_savings: None = ...,
) -> tuple[FloatND, FloatND, FloatND] | tuple[FloatND, FloatND, FloatND, IntND]: ...


@overload
def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = ...,
    x_query: FloatND,
    segment_block_size: int = ...,
    arithmetic: ComparisonArithmetic = ...,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    return_owner: Literal[False] = ...,
    candidate_savings: Float1D,
) -> tuple[FloatND, FloatND, FloatND, FloatND]: ...


@overload
def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = ...,
    x_query: FloatND,
    segment_block_size: int = ...,
    arithmetic: ComparisonArithmetic = ...,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    return_owner: Literal[True],
    candidate_savings: Float1D,
) -> tuple[FloatND, FloatND, FloatND, FloatND, IntND]: ...


@overload
def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = ...,
    x_query: FloatND,
    segment_block_size: int = ...,
    arithmetic: ComparisonArithmetic = ...,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    return_owner: bool = ...,
    candidate_savings: Float1D,
) -> (
    tuple[FloatND, FloatND, FloatND, FloatND]
    | tuple[FloatND, FloatND, FloatND, FloatND, IntND]
): ...


def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = None,
    x_query: FloatND,
    segment_block_size: int = 0,
    arithmetic: ComparisonArithmetic = "certified",
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
    return_owner: bool = False,
    candidate_savings: Float1D | None = None,
) -> (
    tuple[FloatND, FloatND, FloatND]
    | tuple[FloatND, FloatND, FloatND, IntND]
    | tuple[FloatND, FloatND, FloatND, FloatND]
    | tuple[FloatND, FloatND, FloatND, FloatND, IntND]
):
    """Evaluate the branch-aware upper envelope at each query abscissa.

    Args:
        endog_grid: Candidate endogenous grid points (resources), any order
            within a branch; a NaN entry is a dead/padding candidate.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        marginal: Candidate marginal values (the supgradient) at `endog_grid`.
        segment_id: Per-candidate branch label. A segment is a consecutive-pair
            link whose endpoints share a label, so unrelated branches never join.
        stable_index: Optional global identity for every produced link, ordered as
            all consecutive links followed by all zero-width self-brackets. Its
            required shape is ``(2 * n_candidate - 1,)``. When omitted, the local
            stored-link order is used. A streamed caller must provide the indices
            of the equivalent one-shot layout so re-entering a standing winner
            cannot acquire a new identity from its block-local position.
        x_query: Abscissae at which to evaluate the envelope.
        segment_block_size: How many candidates the ordinary read holds against
            every query at once. `0` lets the reduction pick a window; a positive
            value pins it. The certified path is one native exact reduction whose
            partition is internal, so the hint does not reach it and is
            bit-identical for every value either way.
        arithmetic: Which arithmetic decides ownership.
            - `"certified"` compares the stored affine candidates in native
              fixed-width integer arithmetic, using exact value, right extension,
              exact slope, and stable segment index. It performs three exact reads
              only after the winner is known and publishes all channels together.
            - `"ordinary"` reads each link in the working format and takes the
              largest. It decides every bracketed query — there is no certificate
              to abstain on — and costs roughly an order of magnitude less per
              read. Adequate where candidate values are separated by much more
              than the format's resolution at their own magnitude.
            The choice is made when the function is traced, so `"ordinary"`
            emits none of the error-free transforms rather than masking them.
            Implemented for the dense reduction only.
        feasibility_partition: Shared liquid-axis partition containing the
            feasibility boundaries that candidates and queries must respect.
            Must be supplied together with `feasible_interval_mask`.
        feasible_interval_mask: Whether each interval of
            `feasibility_partition` is feasible. Infeasible candidates are
            removed before links are built, and infeasible queries publish the
            carry contract: `-inf` value, NaN policy, and zero marginal.
        return_owner: Also publish the owner of each query: the stable identity
            of the link the three channels were read from (`stable_index` where
            supplied, the stored-link position otherwise), `NO_OWNER` wherever
            the value is not finite. Ownership is a structural decision, so a
            caller comparing two layouts of the same candidates asserts it
            exactly and reserves a tolerance for the levels.
        candidate_savings: Optional original saving ordinate at every candidate.
            Its selected affine read follows the same winner and is published
            before the optional owner. All retained channels share read status.

    Returns:
        Tuple of the envelope value, the winning segment's policy, and the
        winning segment's marginal at each query, each shaped like `x_query`,
        followed by original savings when supplied and then the owner when
        `return_owner` is set. A query no live
        segment brackets yields NaN in all three channels.

    Note:
        Under `arithmetic="certified"` the read is differentiable in forward
        mode only: `jax.jvp`, `jax.jacfwd`, and forward-over-forward all carry
        the exact affine slope, while `jax.grad` and `jax.vjp` raise. The
        certified rule fails closed on a non-finite direction, so its registered
        custom JVP is not automatically transposable even though the affine
        differential on finite directions is linear. No explicit reverse rule is
        registered. Reverse mode over a scalar objective therefore needs
        `arithmetic="ordinary"`, or `jax.jacfwd` in place of `jax.grad`.
    """
    return _envelope_at_query_selected(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        stable_index=stable_index,
        x_query=x_query,
        segment_block_size=segment_block_size,
        arithmetic=arithmetic,
        feasibility_partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
        return_owner=return_owner,
        candidate_savings=candidate_savings,
        reducer=_reduce_envelope_exact
        if arithmetic == "certified"
        else _reduce_envelope_ordinary,
    )


def published_owner(*, value: FloatND, stable_index: IntND) -> IntND:
    """Name the owner of every query that publishes a finite level.

    The owner is a structural fact about a reduction — which stored link the
    published channels were read from — so it is reported wherever a finite level
    is published and is `NO_OWNER` everywhere else: at a query no live link
    brackets, one a certificate refused, and one the feasibility carry contract
    fills with `-inf`.
    """
    return jnp.where(jnp.isfinite(value), stable_index, NO_OWNER).astype(jnp.int32)


def _mask_infeasible_queries(
    *,
    published: tuple[FloatND, FloatND, FloatND],
    query: FloatND,
    partition: ResolvedAxisPartition | None,
    feasible_interval_mask: BoolND | None,
) -> tuple[FloatND, FloatND, FloatND]:
    """Apply the value, policy, and carry contracts to infeasible query rows."""
    if partition is None or feasible_interval_mask is None:
        return published

    interval = axis_interval_indices(partition=partition, values=query)
    feasible = feasible_interval_mask[interval]
    value, policy, marginal = published
    return (
        jnp.where(feasible, value, -jnp.inf),
        jnp.where(feasible, policy, jnp.nan),
        jnp.where(feasible, marginal, 0.0),
    )


def _subnormal_operand_present(*, row: tuple[Float1D, ...], query: FloatND) -> BoolND:
    """Report, per query, whether an operand the backend cannot read is in play.

    Whether a subnormal operand is readable belongs to the backend, not to the
    format, so the predicate asks before it refuses. On a backend that reads the
    whole band every stored operand arrives intact, nothing is lost on the way
    into the read, and refusing would withhold a verdict the arithmetic reached
    correctly — so the refusal leaves the compiled program entirely.

    Where the backend flushes, a subnormal operand compares equal to zero and
    every arithmetic operation on it yields zero. Its magnitude is then not
    merely rounded on the way through the affine read — it is gone before the
    read begins, and no rearrangement of the arithmetic downstream can recover
    it. A link from `0` to the smallest normal, read at a subnormal query, has an
    exact affine value of `2**-23` at float32 and `2**-52` at float64; the
    compiled program publishes zero, and a rival at half the exact value then
    wins a comparison it loses.

    Only the bit pattern survives a flush, so that is what is inspected. A query
    carrying one is refused on its own; one anywhere in the row refuses every
    query, because which segment a query ends up owned by is not known here and
    the affected link may be any of them.

    This predicate makes the unsupported read explicit. Supporting it requires
    arithmetic that reads operands through their significands and exponents
    rather than as working-dtype floats.
    """
    if not backend_flushes_subnormals(row[0].dtype):
        return jnp.zeros_like(jnp.asarray(query), dtype=bool)
    in_row = jnp.any(jnp.stack([jnp.any(is_subnormal(term)) for term in row]))
    return in_row | is_subnormal(query)


def _as_link_rows(column: FloatND | BoolND) -> FloatND | BoolND:
    """Shape a link column as rows against the queries: one shared row, or per query."""
    return column if column.ndim == 2 else column.reshape(1, -1)  # noqa: PLR2004


def _derived_subnormal_possible(
    *,
    links: _SegmentLinks | _LinkRows,
    query: FloatND,
    associated_channels: tuple[tuple[FloatND, FloatND], ...] = (),
) -> BoolND:
    """Report, per query, where the read's own result lands below the range.

    Every stored operand can be normal and the exact result still be a positive
    subnormal: a link from `0` to a value near the top of the range, read close
    to its zero endpoint, weights the far value by a ratio small enough to carry
    the product out of the normal range. A backend that flushes hands back a
    finite zero — neither the represented value nor an abstention, which is the
    one outcome the contract forbids in all three channels at once.

    The predicate is on the **result**, not on the terms that build it. A term of
    the read is routinely subnormal while the result is an ordinary number the
    format holds exactly: reading a link whose two values agree weights one of
    them by a vanishing ratio, and the other by a ratio of essentially one, so
    the sum is that second value. Refusing there would withhold an answer the
    arithmetic reached correctly, and no bound over the terms can tell the two
    situations apart, because in both of them a term is subnormal.

    Two facts make the result's magnitude cheap to decide exactly:

    - The two ratios sum to one, so the endpoint nearer the query carries a ratio
      of at least a half and contributes at least half its own value. The result
      can therefore be subnormal only where that near value is exactly zero, and
      then the result *is* the far term.
    - `_subnormal_operand_present` has already refused every row carrying a
      subnormal stored operand, so each endpoint value reaching here is zero or
      normal — there is no third case for "exactly zero" to hide.

    So the question reduces to whether `|far| * |query - near| / |span|` is
    subnormal, which is settled from the three exponents rather than by forming
    the quantity: on a backend that flushes, forming it is precisely what
    destroys the answer being asked for.

    Evaluated per link, over every live link that brackets the query. Ownership
    is not consulted and does not need to be: a link whose near value is nonzero
    publishes a normal result whether or not it wins.

    A link whose span, distance or far value is not finite is left alone. The
    result there is not a flushed magnitude but an ordinary infinity or NaN,
    which the caller can already see.
    """
    dtype = links.left_grid.dtype
    query_arr = jnp.asarray(query)
    if not backend_flushes_subnormals(dtype):
        return jnp.zeros_like(query_arr, dtype=bool)

    flat = query_arr.reshape(-1, 1)
    left_grid = _as_link_rows(links.left_grid)
    right_grid = _as_link_rows(links.right_grid)
    span = right_grid - left_grid

    to_left = jnp.abs(flat - left_grid)
    to_right = jnp.abs(right_grid - flat)
    left_is_near = to_left <= to_right
    distance_to_near = jnp.minimum(to_left, to_right)

    brackets = (
        _as_link_rows(links.live)
        & (span != 0.0)
        & (jnp.minimum(left_grid, right_grid) <= flat)
        & (flat <= jnp.maximum(left_grid, right_grid))
    )
    readable = (
        jnp.isfinite(span) & jnp.isfinite(distance_to_near) & (distance_to_near > 0.0)
    )

    channels = (
        (links.left_value, links.right_value),
        (links.left_policy, links.right_policy),
        (links.left_marginal, links.right_marginal),
        *associated_channels,
    )
    refused = jnp.zeros_like(brackets)
    for left_channel, right_channel in channels:
        left_c = _as_link_rows(left_channel)
        right_c = _as_link_rows(right_channel)
        near = jnp.where(left_is_near, left_c, right_c)
        far = jnp.where(left_is_near, right_c, left_c)
        refused = refused | (
            brackets
            & readable
            & (near == 0.0)
            & (far != 0.0)
            & jnp.isfinite(far)
            & _quotient_is_subnormal(factors=(far, distance_to_near), divisor=span)
        )
    return jnp.any(refused, axis=-1).reshape(query_arr.shape)


def _quotient_is_subnormal(
    *, factors: tuple[FloatND, FloatND], divisor: FloatND
) -> BoolND:
    """Report where `|f0| * |f1| / |divisor|` is a nonzero subnormal.

    Decided from the exponents, because the product and the quotient are exactly
    the intermediates a flushing backend destroys — forming either of them would
    answer with the zero the caller is trying to detect.

    `frexp` writes each magnitude as `mantissa * 2**exponent` with the mantissa in
    `[0.5, 1)`, so the quotient's mantissa lies in `(0.25, 2)` and one halving or
    one doubling returns it to `[0.5, 1)`. Both mantissa and exponent are then
    directly comparable with the same decomposition of the smallest normal, and
    neither carries a magnitude that can leave the range on the way.
    """
    mantissa = jnp.ones_like(divisor)
    exponent = jnp.zeros_like(divisor, dtype=jnp.int32)
    for factor in factors:
        factor_mantissa, factor_exponent = jnp.frexp(jnp.abs(factor))
        mantissa = mantissa * factor_mantissa
        exponent = exponent + factor_exponent
    divisor_mantissa, divisor_exponent = jnp.frexp(jnp.abs(divisor))
    mantissa = mantissa / divisor_mantissa
    exponent = exponent - divisor_exponent

    # `frexp` writes a magnitude as `mantissa * 2**exponent` with the mantissa in
    # `[mantissa_floor, 1)`. The quotient of three such mantissas lies in
    # `(mantissa_floor / 2, 2)`, so one halving or one doubling restores the
    # normalization and makes exponent and mantissa comparable term by term.
    mantissa_floor = 0.5
    mantissa, exponent = (
        jnp.where(mantissa >= 1.0, mantissa * mantissa_floor, mantissa),
        jnp.where(mantissa >= 1.0, exponent + 1, exponent),
    )
    mantissa, exponent = (
        jnp.where(mantissa < mantissa_floor, mantissa / mantissa_floor, mantissa),
        jnp.where(mantissa < mantissa_floor, exponent - 1, exponent),
    )

    tiny_mantissa, tiny_exponent = jnp.frexp(
        jnp.asarray(jnp.finfo(divisor.dtype).tiny, dtype=divisor.dtype)
    )
    return (exponent < tiny_exponent) | (
        (exponent == tiny_exponent) & (mantissa < tiny_mantissa)
    )


class EnvelopeWinner(NamedTuple):
    """One fixed-shape standing link per query in an interval-block fold."""

    left_grid: FloatND
    right_grid: FloatND
    left_value: FloatND
    right_value: FloatND
    left_policy: FloatND
    right_policy: FloatND
    left_marginal: FloatND
    right_marginal: FloatND
    live: BoolND
    stable_index: IntND
    settled: BoolND
    unreadable: BoolND


def _savings_link_endpoints(
    *, candidate_savings: Float1D | None, endog_grid: Float1D
) -> tuple[Float1D, Float1D] | None:
    """Associate original savings with the unchanged consecutive/self link order."""
    if candidate_savings is None:
        return None
    if candidate_savings.shape != endog_grid.shape:
        raise ValueError("Candidate savings must have the candidate grid shape")
    if candidate_savings.dtype != endog_grid.dtype:
        raise ValueError("Candidate savings must have the candidate grid dtype")
    return (
        jnp.concatenate((candidate_savings[:-1], candidate_savings)),
        jnp.concatenate((candidate_savings[1:], candidate_savings)),
    )


class _SavingsEnvelopeWinner(NamedTuple):
    """A standing link with its original associated savings endpoints."""

    winner: EnvelopeWinner
    left_savings: FloatND
    right_savings: FloatND


@overload
def empty_envelope_winner(
    *, query: FloatND, retain_savings: Literal[False] = ...
) -> EnvelopeWinner: ...


@overload
def empty_envelope_winner(
    *, query: FloatND, retain_savings: Literal[True]
) -> _SavingsEnvelopeWinner: ...


def empty_envelope_winner(
    *, query: FloatND, retain_savings: bool = False
) -> EnvelopeWinner | _SavingsEnvelopeWinner:
    """Return the neutral standing winner for a streamed candidate reduction."""
    query = jnp.asarray(query)
    zeros = jnp.zeros_like(query)
    winner = EnvelopeWinner(
        left_grid=zeros,
        right_grid=zeros,
        left_value=zeros,
        right_value=zeros,
        left_policy=zeros,
        right_policy=zeros,
        left_marginal=zeros,
        right_marginal=zeros,
        live=jnp.zeros_like(query, dtype=bool),
        stable_index=jnp.full(query.shape, jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        settled=jnp.ones_like(query, dtype=bool),
        unreadable=jnp.zeros_like(query, dtype=bool),
    )
    if retain_savings:
        return _SavingsEnvelopeWinner(winner, zeros, zeros)
    return winner


def _batched_link_terms(
    *,
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
    live: BoolND,
    stable_index: IntND,
    query: FloatND,
) -> tuple[_OrdinaryRank, BoolND]:
    """Build the ordinary total-order fields for per-query segment rows."""
    q = query[:, None]
    lower = jnp.minimum(left_grid, right_grid)
    upper = jnp.maximum(left_grid, right_grid)
    brackets = live & (q >= lower) & (q <= upper)
    value_read = _along_link(
        left=left_value,
        right=right_value,
        query=q,
        left_grid=left_grid,
        right_grid=right_grid,
        arithmetic="ordinary",
    )
    slope_high, slope_low = _slope_words(
        left_value=left_value,
        right_value=right_value,
        left_grid=left_grid,
        right_grid=right_grid,
    )
    excluded = jnp.full_like(value_read, -jnp.inf)
    return (
        _OrdinaryRank(
            value=jnp.where(brackets, value_read, excluded),
            right_available=jnp.where(
                brackets, (q < upper).astype(value_read.dtype), excluded
            ),
            slope_high=jnp.where(brackets, slope_high, excluded),
            slope_low=jnp.where(brackets, slope_low, excluded),
            stable_index=jnp.where(
                brackets,
                stable_index,
                jnp.iinfo(stable_index.dtype).max,
            ),
        ),
        brackets,
    )


def _ordinary_rank_argmax(*, rank: _OrdinaryRank) -> IntND:
    """Select the largest ordinary rank, minimizing its final identity field."""
    tied = jnp.ones_like(rank.value, dtype=bool)
    for field in rank[:-1]:
        best = jnp.max(jnp.where(tied, field, -jnp.inf), axis=1, keepdims=True)
        tied = tied & (field == best)
    earliest = jnp.min(
        jnp.where(tied, rank.stable_index, jnp.iinfo(jnp.int32).max),
        axis=1,
        keepdims=True,
    )
    tied = tied & (rank.stable_index == earliest)
    return jnp.argmax(tied, axis=1)[:, None].astype(jnp.int32)


@overload
def merge_envelope_winner(
    *,
    held: EnvelopeWinner,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND,
    query: FloatND,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    candidate_savings: None = ...,
) -> EnvelopeWinner: ...


@overload
def merge_envelope_winner(
    *,
    held: _SavingsEnvelopeWinner,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND,
    query: FloatND,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    candidate_savings: Float1D,
) -> _SavingsEnvelopeWinner: ...


def merge_envelope_winner(
    *,
    held: EnvelopeWinner | _SavingsEnvelopeWinner,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND,
    query: FloatND,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
    candidate_savings: Float1D | None = None,
) -> EnvelopeWinner | _SavingsEnvelopeWinner:
    """Fold one candidate block into a standing, globally identified winner.

    The held link occupies a temporary column only.  Its explicit identity and
    every challenger's one-shot identity are passed as data, so neither ordinary
    nor certified ownership can depend on where a block happens to be stored.
    """
    savings_held = held if isinstance(held, _SavingsEnvelopeWinner) else None
    if (savings_held is None) != (candidate_savings is None):
        raise ValueError(
            "Standing savings and candidate savings must be supplied together"
        )
    savings_links = _savings_link_endpoints(
        candidate_savings=candidate_savings, endog_grid=endog_grid
    )
    if savings_held is not None:
        if any(endpoint.shape != query.shape for endpoint in savings_held[1:]):
            raise ValueError("Standing savings must have the query shape")
        if any(endpoint.dtype != endog_grid.dtype for endpoint in savings_held[1:]):
            raise ValueError("Standing savings must have the candidate grid dtype")
        held = savings_held.winner
    held = cast("EnvelopeWinner", held)
    links, candidate_row = _segment_links_from_candidates(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        stable_index=stable_index,
        feasibility_partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
    )
    flat_query = jnp.asarray(query).reshape(-1)
    n_query = flat_query.shape[0]
    columns = tuple(
        jnp.concatenate(
            [
                held_column.reshape(-1, 1),
                _spread_over_queries(column=block_column, n_query=n_query),
            ],
            axis=1,
        )
        for held_column, block_column in zip(held[:10], links, strict=True)
    )
    # The ordering reads geometry, value and identity only; the winner's policy
    # and marginal are read from `columns` after the winner is known.
    left_grid, right_grid, left_value, right_value = columns[:4]
    live, identities = columns[8], columns[9]

    if arithmetic == "certified":
        winner, status = exact_query_winner_batched(
            left_grid=left_grid,
            right_grid=right_grid,
            left_value=left_value,
            right_value=right_value,
            live=live,
            stable_index=identities,
            x_query=flat_query[:, None],
        )
        index = winner.reshape(-1, 1)
        resolved = status.reshape(-1) == 0
        block_lower = jnp.minimum(links.left_grid, links.right_grid)[None, :]
        block_upper = jnp.maximum(links.left_grid, links.right_grid)[None, :]
        block_brackets = (
            links.live[None, :]
            & (flat_query[:, None] >= block_lower)
            & (flat_query[:, None] <= block_upper)
        )
        settled = held.settled.reshape(-1) & (
            resolved | ~jnp.any(block_brackets, axis=1)
        )
    else:
        rank, brackets = _batched_link_terms(
            left_grid=left_grid,
            right_grid=right_grid,
            left_value=left_value,
            right_value=right_value,
            live=live,
            stable_index=identities,
            query=flat_query,
        )
        index = _ordinary_rank_argmax(rank=rank)
        resolved = jnp.any(brackets, axis=1)
        settled = held.settled.reshape(-1)

    picked = tuple(_pick_column(column=column, index=index) for column in columns[:8])
    picked_identity = _pick_column(column=identities, index=index)
    unreadable = held.unreadable.reshape(-1)
    if arithmetic == "ordinary":
        row: tuple[Float1D, ...] = candidate_row
        if candidate_savings is not None:
            row = (
                *row,
                jnp.where(jnp.isfinite(candidate_row[0]), candidate_savings, jnp.nan),
            )
        unreadable = (
            unreadable
            | _subnormal_operand_present(row=row, query=jnp.asarray(query)).reshape(-1)
            | _derived_subnormal_possible(
                links=links,
                query=jnp.asarray(query),
                associated_channels=() if savings_links is None else (savings_links,),
            ).reshape(-1)
        )
    shape = jnp.asarray(query).shape
    winner = EnvelopeWinner(
        *(jnp.where(resolved, field, 0).reshape(shape) for field in picked),
        live=resolved.reshape(shape),
        stable_index=jnp.where(
            resolved, picked_identity, jnp.iinfo(jnp.int32).max
        ).reshape(shape),
        settled=settled.reshape(shape),
        unreadable=unreadable.reshape(shape),
    )
    if savings_held is None:
        return winner
    if savings_links is None:
        raise ValueError("Standing savings require candidate savings endpoints")
    endpoints = tuple(
        _pick_column(
            column=jnp.concatenate(
                [old.reshape(-1, 1), _spread_over_queries(column=new, n_query=n_query)],
                axis=1,
            ),
            index=index,
        ).reshape(shape)
        for old, new in zip(savings_held[1:], savings_links, strict=True)
    )
    return _SavingsEnvelopeWinner(winner, endpoints[0], endpoints[1])


def fold_ascending_runs(
    *,
    held: EnvelopeWinner | _SavingsEnvelopeWinner,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND,
    query: Float1D,
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
    candidate_savings: Float1D | None = None,
) -> EnvelopeWinner | _SavingsEnvelopeWinner:
    """Fold a candidate path whose segments ascend, reading each query's own links.

    Requires that within every finite `segment_id` the `endog_grid` strictly
    ascends and that a segment's candidates are contiguous, as
    `segment_ids_from_folds` labels them. A segment's live links are then
    disjoint intervals, so at query `q` only the link starting at its last point
    `p` with `x_p <= q`, the link ending there, and `p`'s self-bracket can bracket
    `q`. The fold visits the segments in turn, locates `p` by one search, and
    folds those three links per query under their one-shot identities, in the
    ordinary arithmetic. The winner equals `merge_envelope_winner` over the whole
    path at `O(n_segments * (n + n_query * log n))` instead of `O(n * n_query)`.
    """
    savings_held = held if isinstance(held, _SavingsEnvelopeWinner) else None
    if (savings_held is None) != (candidate_savings is None):
        raise ValueError(
            "Standing savings and candidate savings must be supplied together"
        )
    links, candidate_row = _segment_links_from_candidates(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        stable_index=stable_index,
        feasibility_partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
    )
    savings_links = _savings_link_endpoints(
        candidate_savings=candidate_savings, endog_grid=endog_grid
    )
    row: tuple[Float1D, ...] = candidate_row
    if candidate_savings is not None:
        row = (
            *row,
            jnp.where(jnp.isfinite(candidate_row[0]), candidate_savings, jnp.nan),
        )
    row_refused = _subnormal_operand_present(row=row, query=query).reshape(-1)
    n = endog_grid.shape[0]
    finite = jnp.isfinite(segment_id)
    starts_segment = finite & jnp.concatenate(
        [jnp.ones((1,), dtype=bool), segment_id[1:] != segment_id[:-1]]
    )
    starts = jnp.nonzero(starts_segment, size=n, fill_value=0)[0].astype(jnp.int32)
    n_segments = jnp.sum(starts_segment).astype(jnp.int32)
    position = jnp.arange(n, dtype=jnp.int32)

    def visit(
        state: tuple[jax.Array, EnvelopeWinner | _SavingsEnvelopeWinner],
    ) -> tuple[jax.Array, EnvelopeWinner | _SavingsEnvelopeWinner]:
        k, standing = state
        start = starts[k]
        key = jnp.where(
            position < start,
            -jnp.inf,
            jnp.where(segment_id == segment_id[start], endog_grid, jnp.inf),
        )
        last = jnp.searchsorted(key, query, side="right").astype(jnp.int32) - 1
        consecutive = jnp.clip(jnp.stack([last - 1, last], axis=1), 0, max(n - 2, 0))
        own = jnp.clip(last, 0, n - 1)[:, None] + (n - 1)
        picked = jnp.concatenate([consecutive, own], axis=1)
        rows = _LinkRows(*(jnp.take(field, picked) for field in links))
        savings_rows = (
            None
            if savings_links is None
            else (
                jnp.take(savings_links[0], picked),
                jnp.take(savings_links[1], picked),
            )
        )
        return k + 1, _merge_link_rows_ordinary(
            held=standing,
            rows=rows,
            savings_rows=savings_rows,
            query=query,
            refused=row_refused,
        )

    _, folded = jax.lax.while_loop(
        lambda state: state[0] < n_segments, visit, (jnp.int32(0), held)
    )
    return folded


def _merge_link_rows_ordinary(
    *,
    held: EnvelopeWinner | _SavingsEnvelopeWinner,
    rows: _LinkRows,
    savings_rows: tuple[FloatND, FloatND] | None,
    query: Float1D,
    refused: BoolND,
) -> EnvelopeWinner | _SavingsEnvelopeWinner:
    """`merge_envelope_winner`'s ordinary fold with each query's own link rows.

    `rows` holds `(n_query, m)` links, the self-brackets' channel and identity
    columns included; `refused` is the bank's stored-subnormal verdict per query.
    """
    savings_held = held if isinstance(held, _SavingsEnvelopeWinner) else None
    standing = cast(
        "EnvelopeWinner", held if savings_held is None else savings_held.winner
    )
    columns = tuple(
        jnp.concatenate([held_column.reshape(-1, 1), row], axis=1)
        for held_column, row in zip(standing[:10], rows, strict=True)
    )
    left_grid, right_grid, left_value, right_value = columns[:4]
    live, identities = columns[8], columns[9]
    rank, brackets = _batched_link_terms(
        left_grid=left_grid,
        right_grid=right_grid,
        left_value=left_value,
        right_value=right_value,
        live=live,
        stable_index=identities,
        query=query,
    )
    index = _ordinary_rank_argmax(rank=rank)
    resolved = jnp.any(brackets, axis=1)
    picked = tuple(_pick_column(column=column, index=index) for column in columns[:8])
    picked_identity = _pick_column(column=identities, index=index)
    unreadable = (
        standing.unreadable.reshape(-1)
        | refused
        | _derived_subnormal_possible(
            links=rows,
            query=query,
            associated_channels=() if savings_rows is None else (savings_rows,),
        ).reshape(-1)
    )
    shape = query.shape
    winner = EnvelopeWinner(
        *(jnp.where(resolved, field, 0).reshape(shape) for field in picked),
        live=resolved.reshape(shape),
        stable_index=jnp.where(
            resolved, picked_identity, jnp.iinfo(jnp.int32).max
        ).reshape(shape),
        settled=standing.settled.reshape(shape),
        unreadable=unreadable.reshape(shape),
    )
    if savings_held is None:
        return winner
    if savings_rows is None:
        raise ValueError("Standing savings require candidate savings endpoints")
    endpoints = tuple(
        _pick_column(
            column=jnp.concatenate([old.reshape(-1, 1), new], axis=1), index=index
        ).reshape(shape)
        for old, new in zip(savings_held[1:], savings_rows, strict=True)
    )
    return _SavingsEnvelopeWinner(winner, endpoints[0], endpoints[1])


@overload
def merge_query_point_columns(
    *,
    held: EnvelopeWinner,
    endog_grid: FloatND,
    policy: FloatND,
    value: FloatND,
    marginal: FloatND,
    segment_id: FloatND,
    first_index: IntND,
    first_self_index: int,
    query: Float1D,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    candidate_savings: None = ...,
    bank: tuple[FloatND, ...] | None = ...,
) -> EnvelopeWinner: ...


@overload
def merge_query_point_columns(
    *,
    held: _SavingsEnvelopeWinner,
    endog_grid: FloatND,
    policy: FloatND,
    value: FloatND,
    marginal: FloatND,
    segment_id: FloatND,
    first_index: IntND,
    first_self_index: int,
    query: Float1D,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
    candidate_savings: FloatND,
    bank: tuple[FloatND, ...] | None = ...,
) -> _SavingsEnvelopeWinner: ...


def merge_query_point_columns(
    *,
    held: EnvelopeWinner | _SavingsEnvelopeWinner,
    endog_grid: FloatND,
    policy: FloatND,
    value: FloatND,
    marginal: FloatND,
    segment_id: FloatND,
    first_index: IntND,
    first_self_index: int,
    query: Float1D,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
    candidate_savings: FloatND | None = None,
    bank: tuple[FloatND, ...] | None = None,
) -> EnvelopeWinner | _SavingsEnvelopeWinner:
    """Fold zero-width candidate pairs that each sit on one query's abscissa.

    Column `j` of the `(n_points, n_query)` channels holds points stored as
    zero-width pairs at `query[j]`, or dead. In the one-shot layout point `(i, j)`
    is the candidate pair starting at `first_index[i, j]`: its pair link carries
    that identity and its two self-brackets `first_self_index` plus it and plus
    one. A zero-width link brackets only its own abscissa, so with strictly
    increasing queries a column is read by its own query alone, and folding each
    column into that query's standing winner equals the one-shot read over the
    whole bank at `O(n_points * n_query)` instead of `O(n_points * n_query**2)`.

    Within a column every live pair reads its stored value, extends nowhere
    right, and has slope zero, so the total order keeps the largest value and,
    among equal values, the smallest identity. That point alone is folded. A
    stored subnormal operand anywhere in the bank still refuses every query on a
    flushing backend, as the one-shot read does. Where the columns were already
    reduced from a larger bank, `bank` passes that bank's channels `(endog,
    policy, value, marginal[, savings])`, which the refusal then reads instead.
    """
    live = jnp.isfinite(endog_grid) & jnp.isfinite(value)
    top = jnp.max(jnp.where(live, value, -jnp.inf), axis=0)
    best = jnp.argmin(
        jnp.where(live & (value == top), first_index, jnp.iinfo(jnp.int32).max),
        axis=0,
    )[None, :]

    def pick(column: FloatND | IntND) -> FloatND | IntND:
        return jnp.take_along_axis(column, best, axis=0)[0]

    first = pick(first_index).astype(jnp.int32)
    identities = jnp.stack(
        [first, first_self_index + first, first_self_index + first + 1], axis=1
    )
    point = tuple(pick(c) for c in (endog_grid, policy, value, marginal, segment_id))
    saving = None if candidate_savings is None else pick(candidate_savings)

    # keyword-only-exempt: library-callback=jax.vmap
    def fold_one(
        held_q: EnvelopeWinner | _SavingsEnvelopeWinner,
        query_q: FloatND,
        point_q: tuple[FloatND, ...],
        identity: IntND,
        saving_q: FloatND | None,
    ) -> EnvelopeWinner | _SavingsEnvelopeWinner:
        endog_q, policy_q, value_q, marginal_q, segment_q = (
            jnp.stack([c, c]) for c in point_q
        )
        return merge_envelope_winner(  # ty: ignore[no-matching-overload]
            held=held_q,
            endog_grid=endog_q,
            policy=policy_q,
            value=value_q,
            marginal=marginal_q,
            segment_id=segment_q,
            stable_index=identity,
            query=query_q[None],
            arithmetic=arithmetic,
            feasibility_partition=feasibility_partition,
            feasible_interval_mask=feasible_interval_mask,
            candidate_savings=None
            if saving_q is None
            else jnp.stack([saving_q, saving_q]),
        )

    folded = jax.vmap(fold_one)(
        jax.tree.map(lambda leaf: leaf.reshape(-1, 1), held),
        query,
        point,
        identities,
        saving,
    )
    folded = jax.tree.map(lambda leaf: leaf.reshape(query.shape), folded)
    if arithmetic != "ordinary" or not backend_flushes_subnormals(endog_grid.dtype):
        return folded
    if bank is not None:
        endog_grid, policy, value, marginal = bank[:4]
        candidate_savings = bank[4] if len(bank) > 4 else None  # noqa: PLR2004
    if feasibility_partition is not None:
        feasible = cast("BoolND", feasible_interval_mask)[
            axis_interval_indices(partition=feasibility_partition, values=endog_grid)
        ]
        endog_grid, value, policy, marginal = (
            jnp.where(feasible, c, jnp.nan)
            for c in (endog_grid, value, policy, marginal)
        )
    row = (endog_grid, value, policy, marginal)
    if candidate_savings is not None:
        row = (*row, jnp.where(jnp.isfinite(endog_grid), candidate_savings, jnp.nan))
    refused = _subnormal_operand_present(
        row=tuple(c.reshape(-1) for c in row), query=query
    )
    if isinstance(folded, _SavingsEnvelopeWinner):
        return folded._replace(
            winner=folded.winner._replace(unreadable=folded.winner.unreadable | refused)
        )
    return folded._replace(unreadable=folded.unreadable | refused)


@overload
def finish_envelope_winner(
    *,
    winner: EnvelopeWinner,
    query: FloatND,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
) -> tuple[FloatND, FloatND, FloatND]: ...


@overload
def finish_envelope_winner(
    *,
    winner: _SavingsEnvelopeWinner,
    query: FloatND,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = ...,
    feasible_interval_mask: BoolND | None = ...,
) -> tuple[FloatND, FloatND, FloatND, FloatND, IntND]: ...


def finish_envelope_winner(
    *,
    winner: EnvelopeWinner | _SavingsEnvelopeWinner,
    query: FloatND,
    arithmetic: ComparisonArithmetic,
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
) -> (
    tuple[FloatND, FloatND, FloatND] | tuple[FloatND, FloatND, FloatND, FloatND, IntND]
):
    """Read a completed fold's coupled channels and optional original savings."""
    savings_winner = winner if isinstance(winner, _SavingsEnvelopeWinner) else None
    if savings_winner is not None:
        winner = savings_winner.winner
    winner = cast("EnvelopeWinner", winner)
    query = jnp.asarray(query)
    flat_query = query.reshape(-1)
    left_grid = winner.left_grid.reshape(-1)
    right_grid = winner.right_grid.reshape(-1)
    descending = left_grid > right_grid
    zero_width = left_grid == right_grid
    lower_grid = jnp.where(descending, right_grid, left_grid)
    upper_grid = jnp.where(descending, left_grid, right_grid)

    if arithmetic == "certified":
        read_exact = functools.partial(
            _read_exact_channel,
            descending=descending,
            zero_width=zero_width,
            read_x0=jnp.where(zero_width, jnp.zeros_like(lower_grid), lower_grid),
            read_x1=jnp.where(zero_width, jnp.ones_like(upper_grid), upper_grid),
            read_query=jnp.where(zero_width, jnp.zeros_like(flat_query), flat_query),
        )
        value, value_status = read_exact(
            left=winner.left_value, right=winner.right_value
        )
        policy, policy_status = read_exact(
            left=winner.left_policy, right=winner.right_policy
        )
        marginal, marginal_status = read_exact(
            left=winner.left_marginal, right=winner.right_marginal
        )
        readable = (value_status | policy_status | marginal_status) == 0
        if savings_winner is not None:
            savings, savings_status = read_exact(
                left=savings_winner.left_savings, right=savings_winner.right_savings
            )
            readable = readable & (savings_status == 0)
    else:
        read_ordinary = functools.partial(
            _read_ordinary_channel,
            descending=descending,
            zero_width=zero_width,
            flat_query=flat_query,
            lower_grid=lower_grid,
            upper_grid=upper_grid,
        )
        value = read_ordinary(left=winner.left_value, right=winner.right_value)
        policy = read_ordinary(left=winner.left_policy, right=winner.right_policy)
        marginal = read_ordinary(left=winner.left_marginal, right=winner.right_marginal)
        readable = ~winner.unreadable.reshape(-1)
        if savings_winner is not None:
            savings = read_ordinary(
                left=savings_winner.left_savings, right=savings_winner.right_savings
            )
            readable = readable & jnp.isfinite(savings)

    decided = winner.live.reshape(-1) & winner.settled.reshape(-1) & readable
    published = _mask_infeasible_queries(
        published=(
            _publish_decided(channel=value, decided=decided, shape=query.shape),
            _publish_decided(channel=policy, decided=decided, shape=query.shape),
            _publish_decided(channel=marginal, decided=decided, shape=query.shape),
        ),
        query=query,
        partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
    )
    if savings_winner is None:
        return published
    associated = _publish_decided(channel=savings, decided=decided, shape=query.shape)
    return (
        *published,
        jnp.where(jnp.isfinite(published[0]), associated, jnp.nan),
        published_owner(value=published[0], stable_index=winner.stable_index),
    )


def _spread_over_queries(
    *, column: FloatND | BoolND | IntND, n_query: int
) -> FloatND | BoolND | IntND:
    """Repeat one candidate column across every query row."""
    return jnp.broadcast_to(column[None, :], (n_query, column.shape[0]))


def _pick_column(
    *, column: FloatND | BoolND | IntND, index: IntND
) -> FloatND | BoolND | IntND:
    """Gather each query row's entry at its `(n_query, 1)` column index."""
    return jnp.take_along_axis(column, index, axis=1)[:, 0]


def _take_rows(*, column: FloatND | IntND, index: IntND) -> FloatND | IntND:
    """Gather the candidate rows of `column` selected per query."""
    return jnp.take(column, index, axis=0)


def _publish_decided(
    *, channel: FloatND, decided: BoolND, shape: tuple[int, ...]
) -> FloatND:
    """Shape one channel like the query, NaN where no winner was decided."""
    return jnp.where(decided, channel, jnp.nan).reshape(shape)


def _endpoint_order(
    *, left: FloatND, right: FloatND, descending: BoolND, zero_width: BoolND
) -> tuple[FloatND, FloatND]:
    """Order a winner's stored endpoint channel by ascending abscissa.

    A zero-width winner has no affine line and reads its stored left endpoint,
    so both ordered endpoints carry that reading there.
    """
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    lower = jnp.where(descending, flat_right, flat_left)
    upper = jnp.where(descending, flat_left, flat_right)
    return lower, jnp.where(zero_width, lower, upper)


def _read_exact_channel(
    *,
    left: FloatND,
    right: FloatND,
    descending: BoolND,
    zero_width: BoolND,
    read_x0: FloatND,
    read_x1: FloatND,
    read_query: FloatND,
) -> tuple[FloatND, IntND]:
    """Read one channel of the winner at the query with the exact affine reader.

    The exact reader keeps its positive-width contract, so a zero-width winner
    arrives canonicalized to a flat unit line (`read_x0`, `read_x1`) read at zero
    (`read_query`), and all three channels pass through the one native reader.
    """
    lower, upper = _endpoint_order(
        left=left, right=right, descending=descending, zero_width=zero_width
    )
    return exact_affine_read(
        x0=read_x0,
        x1=read_x1,
        v0=lower,
        v1=upper,
        x_query=read_query,
    )


def _read_ordinary_channel(
    *,
    left: FloatND,
    right: FloatND,
    descending: BoolND,
    zero_width: BoolND,
    flat_query: FloatND,
    lower_grid: FloatND,
    upper_grid: FloatND,
) -> FloatND:
    """Read one channel of the winner at the query in the ordinary arithmetic."""
    lower, upper = _endpoint_order(
        left=left, right=right, descending=descending, zero_width=zero_width
    )
    return _along_link(
        left=lower,
        right=upper,
        query=flat_query,
        left_grid=lower_grid,
        right_grid=upper_grid,
        arithmetic="ordinary",
    )


class _EnvelopeReduction(NamedTuple):
    """The envelope at every query."""

    published: tuple[FloatND, FloatND, FloatND]
    """Value, policy, and marginal of the winning candidate at each query."""
    owner: IntND | None = None
    """Stable identity of the winner at each query, `NO_OWNER` where undecided.

    `None` for a reduction that does not report identities.
    """


class _SavingsEnvelopeReduction(NamedTuple):
    """Coupled envelope channels and the original associated saving."""

    published: tuple[FloatND, FloatND, FloatND]
    owner: IntND
    savings: FloatND


@runtime_checkable
class _EnvelopeReducer(Protocol):
    """The selected original numerical reduction and its readability status."""

    def __call__(
        self,
        *,
        links: _SegmentLinks,
        query: FloatND,
        savings_endpoints: tuple[Float1D, Float1D] | None,
        candidate_row: tuple[Float1D, Float1D, Float1D, Float1D],
        candidate_savings: Float1D | None,
        n_segment: int,
        segment_block_size: int,
        feasibility_partition: ResolvedAxisPartition | None,
        feasible_interval_mask: BoolND | None,
    ) -> tuple[
        _EnvelopeReduction | _SavingsEnvelopeReduction,
        tuple[FloatND, FloatND, FloatND],
        BoolND | None,
    ]: ...


def _envelope_at_query_selected(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    stable_index: IntND | None = None,
    x_query: FloatND,
    segment_block_size: int = 0,
    arithmetic: ComparisonArithmetic = "certified",
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
    return_owner: bool = False,
    candidate_savings: Float1D | None = None,
    reducer: _EnvelopeReducer,
) -> (
    tuple[FloatND, FloatND, FloatND]
    | tuple[FloatND, FloatND, FloatND, IntND]
    | tuple[FloatND, FloatND, FloatND, FloatND]
    | tuple[FloatND, FloatND, FloatND, FloatND, IntND]
):
    """Use the original preparation/publication around its selected reducer."""
    savings_endpoints = _savings_link_endpoints(
        candidate_savings=candidate_savings, endog_grid=endog_grid
    )
    links, candidate_row = _segment_links_from_candidates(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        stable_index=stable_index,
        feasibility_partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
    )
    endog_grid, value, policy, marginal = candidate_row

    query = jnp.asarray(x_query)
    n_segment = links.left_grid.shape[0]
    reduction, published, unreadable = reducer(
        links=links,
        query=query,
        savings_endpoints=savings_endpoints,
        candidate_savings=candidate_savings,
        n_segment=n_segment,
        segment_block_size=segment_block_size,
        feasibility_partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
        candidate_row=(endog_grid, value, policy, marginal),
    )
    if isinstance(reduction, _SavingsEnvelopeReduction):
        savings = jnp.where(jnp.isfinite(published[0]), reduction.savings, jnp.nan)
        if arithmetic == "ordinary":
            if unreadable is None:
                raise TypeError("Ordinary query lost its original readability status.")
            savings = jnp.where(unreadable, jnp.nan, savings)
        if not return_owner:
            return (*published, savings)
        return (
            *published,
            savings,
            published_owner(value=published[0], stable_index=reduction.owner),
        )
    if not return_owner:
        return published
    owner = cast("IntND", reduction.owner)
    return (*published, published_owner(value=published[0], stable_index=owner))


def _reduce_envelope_exact(
    *,
    links: _SegmentLinks,
    query: FloatND,
    savings_endpoints: tuple[Float1D, Float1D] | None,
    candidate_row: tuple[Float1D, Float1D, Float1D, Float1D],
    candidate_savings: Float1D | None,
    n_segment: int,
    segment_block_size: int,
    feasibility_partition: ResolvedAxisPartition | None,
    feasible_interval_mask: BoolND | None,
) -> tuple[
    _EnvelopeReduction | _SavingsEnvelopeReduction,
    tuple[FloatND, FloatND, FloatND],
    BoolND | None,
]:
    """Run the original exact branch without an ordinary-backend dependency."""
    del candidate_row, candidate_savings, n_segment, segment_block_size
    # The exact kernel owns the whole segment reduction. A block size is now
    # a partition request with no numerical effect: the native operation
    # streams the same stored segments and returns one winner/status per
    # query, so dense and blocked calls are identical by construction.
    reduction = _envelope_exact(
        links=links, query=query, savings_endpoints=savings_endpoints
    )
    published = _mask_infeasible_queries(
        published=reduction.published,
        query=query,
        partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
    )
    return reduction, published, None


def _reduce_envelope_ordinary(
    *,
    links: _SegmentLinks,
    query: FloatND,
    savings_endpoints: tuple[Float1D, Float1D] | None,
    candidate_row: tuple[Float1D, Float1D, Float1D, Float1D],
    candidate_savings: Float1D | None,
    n_segment: int,
    segment_block_size: int,
    feasibility_partition: ResolvedAxisPartition | None,
    feasible_interval_mask: BoolND | None,
) -> tuple[
    _EnvelopeReduction | _SavingsEnvelopeReduction,
    tuple[FloatND, FloatND, FloatND],
    BoolND | None,
]:
    """Run the original ordinary read, including its original backend probe."""
    endog_grid, value, policy, marginal = candidate_row
    reduction = _envelope_blocked_ordinary(
        links=links,
        query=query,
        block_size=_ordinary_block_size(
            requested=segment_block_size, n_segment=n_segment
        ),
        savings_endpoints=savings_endpoints,
    )
    row: tuple[Float1D, ...] = (endog_grid, value, policy, marginal)
    if candidate_savings is not None:
        row = (
            *row,
            jnp.where(jnp.isfinite(endog_grid), candidate_savings, jnp.nan),
        )
    unreadable = _subnormal_operand_present(
        row=row, query=query
    ) | _derived_subnormal_possible(
        links=links,
        query=query,
        associated_channels=() if savings_endpoints is None else (savings_endpoints,),
    )
    if isinstance(reduction, _SavingsEnvelopeReduction):
        unreadable = unreadable | ~jnp.isfinite(reduction.savings)
    readable_value, readable_policy, readable_marginal = (
        jnp.where(unreadable, jnp.nan, channel) for channel in reduction.published
    )
    published = _mask_infeasible_queries(
        published=(readable_value, readable_policy, readable_marginal),
        query=query,
        partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
    )
    return reduction, published, unreadable


def _envelope_exact(
    *,
    links: _SegmentLinks,
    query: FloatND,
    savings_endpoints: tuple[Float1D, Float1D] | None = None,
) -> _EnvelopeReduction | _SavingsEnvelopeReduction:
    """Resolve one exact winner and read only its coupled published channels.

    The native winner operation compares exact stored affine values under the
    documented total order: value, right extension, exact slope, then stable
    segment index. Its status is coupled to each selected affine read;
    a query either publishes all channels from one winner or NaN in all of them.
    """
    if links.left_grid.shape[0] == 0:
        empty = jnp.full_like(query, jnp.nan)
        if savings_endpoints is not None:
            return _SavingsEnvelopeReduction(
                published=(empty, empty, empty),
                owner=jnp.full(query.shape, NO_OWNER, dtype=jnp.int32),
                savings=empty,
            )
        return _EnvelopeReduction(
            published=(empty, empty, empty),
            owner=jnp.full(query.shape, NO_OWNER, dtype=jnp.int32),
        )

    winner, winner_status = exact_query_winner(
        left_grid=links.left_grid,
        right_grid=links.right_grid,
        left_value=links.left_value,
        right_value=links.right_value,
        live=links.live,
        stable_index=links.stable_index,
        x_query=query,
    )
    flat_query = query.reshape(-1)
    flat_winner = winner.reshape(-1)
    take = functools.partial(_take_rows, index=flat_winner)

    stored_left_grid = take(column=links.left_grid)
    stored_right_grid = take(column=links.right_grid)
    descending = stored_left_grid > stored_right_grid
    zero_width = stored_left_grid == stored_right_grid
    left_grid = jnp.where(descending, stored_right_grid, stored_left_grid)
    right_grid = jnp.where(descending, stored_left_grid, stored_right_grid)
    # The exact reader intentionally retains its positive-width public contract.
    # A zero-width winner has no affine line; the query reduction defined it as
    # the stored left endpoint. Canonicalize that selected point to a flat unit
    # line and read it at zero, so all three channels still pass through exactly
    # one winner-only native reader without broadening the reader's API.
    read = functools.partial(
        _read_exact_channel,
        descending=descending,
        zero_width=zero_width,
        read_x0=jnp.where(zero_width, jnp.zeros_like(left_grid), left_grid),
        read_x1=jnp.where(zero_width, jnp.ones_like(right_grid), right_grid),
        read_query=jnp.where(zero_width, jnp.zeros_like(flat_query), flat_query),
    )
    value, value_status = read(
        left=take(column=links.left_value), right=take(column=links.right_value)
    )
    policy, policy_status = read(
        left=take(column=links.left_policy), right=take(column=links.right_policy)
    )
    marginal, marginal_status = read(
        left=take(column=links.left_marginal), right=take(column=links.right_marginal)
    )
    coupled_status = (
        winner_status.reshape(-1) | value_status | policy_status | marginal_status
    )
    if savings_endpoints is not None:
        savings, savings_status = read(
            left=take(column=savings_endpoints[0]),
            right=take(column=savings_endpoints[1]),
        )
        coupled_status = coupled_status | savings_status
    decided = coupled_status == 0
    reduction = _EnvelopeReduction(
        published=(
            _publish_decided(channel=value, decided=decided, shape=query.shape),
            _publish_decided(channel=policy, decided=decided, shape=query.shape),
            _publish_decided(channel=marginal, decided=decided, shape=query.shape),
        ),
        owner=jnp.where(
            decided, jnp.take(links.stable_index, flat_winner, axis=0), NO_OWNER
        ).reshape(query.shape),
    )
    if savings_endpoints is None:
        return reduction
    return _SavingsEnvelopeReduction(
        published=reduction.published,
        owner=cast("IntND", reduction.owner),
        savings=_publish_decided(channel=savings, decided=decided, shape=query.shape),
    )


def _slope_words(
    *,
    left_value: FloatND,
    right_value: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
) -> tuple[FloatND, FloatND]:
    """Return the value-slope as two words, from exact endpoint differences.

    Both differences are formed by `two_sum` and so are exact; only the division
    rounds, and it rounds into a pair rather than a single float. Two slopes are
    then ordered whenever the working precision separates them twice over, where
    one rounded float stops separating them once the slope is steep.

    A zero-width link has no slope. Its divisor is displaced to keep the
    quotient finite; the tie-break only ever reads it against another
    zero-width link, which the right-extension field has already ordered.
    """
    width = dd_from_difference(right_grid, left_grid)
    safe_width = (
        jnp.where(width[0] == 0.0, jnp.ones_like(width[0]), width[0]),
        jnp.where(width[0] == 0.0, jnp.zeros_like(width[1]), width[1]),
        width[2],
    )
    return dd_quotient(dd_from_difference(right_value, left_value), safe_width)


class _BlockTerms(NamedTuple):
    """What one segment block contributes at every query, all `(n_query, block)`."""

    brackets: BoolND
    """Whether the link is live and brackets the query."""
    value: FloatND
    """Value read along the link at the query."""
    policy: FloatND
    """Policy read along the link at the query."""
    marginal: FloatND
    """Marginal read along the link at the query."""
    slope_high: FloatND
    """Leading word of the link's value-slope, for the right-continuous tie-break."""
    slope_low: FloatND
    """Trailing word of the same slope, which orders what the leading word ties."""
    upper: FloatND
    """Upper endpoint of the link, for the same tie-break."""


def _block_query_terms(
    *,
    block: FloatND,
    live: BoolND,
    flat: Float1D,
    arithmetic: ComparisonArithmetic = "certified",
) -> _BlockTerms:
    """Bracket-and-read one segment block against every query.

    `block` is one `(block_size, 8)` slice of the stacked link endpoint columns
    and `live` its `(block_size,)` live-flag slice — the same quantities the dense
    path forms over all segments at once, but only for this block, so the peak
    working set is `(n_query, block_size)`.
    """
    left_grid, right_grid = block[:, 0], block[:, 1]
    left_value, right_value = block[:, 2], block[:, 3]
    left_policy, right_policy = block[:, 4], block[:, 5]
    left_marginal, right_marginal = block[:, 6], block[:, 7]

    q = flat[:, None]
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = live[None, :] & (q >= lower) & (q <= upper)

    spread_left_grid = left_grid[None, :]
    spread_right_grid = right_grid[None, :]
    read = functools.partial(
        _along_link,
        query=q,
        left_grid=spread_left_grid,
        right_grid=spread_right_grid,
        arithmetic=arithmetic,
    )
    value_interp = read(left=left_value[None, :], right=right_value[None, :])
    policy_interp = read(left=left_policy[None, :], right=right_policy[None, :])
    marginal_interp = read(left=left_marginal[None, :], right=right_marginal[None, :])
    slope_high, slope_low = _slope_words(
        left_value=left_value[None, :],
        right_value=right_value[None, :],
        left_grid=spread_left_grid,
        right_grid=spread_right_grid,
    )
    return _BlockTerms(
        brackets=brackets,
        value=value_interp,
        policy=policy_interp,
        marginal=marginal_interp,
        slope_high=slope_high,
        slope_low=slope_low,
        upper=upper,
    )


# Candidates the ordinary read holds against one query when nothing is asked for.
# A window in candidates, not a working-set budget in bytes: the reduction sees
# its own query count but not how many of these reductions the caller is running
# side by side, and under a vmapped solve that outer width multiplies every array
# here. A budget computed from what is visible would therefore be spent many
# times over, while a fixed window bounds the working set per query row whatever
# the caller wraps around it.
_ORDINARY_CANDIDATE_WINDOW = 1024


def _ordinary_block_size(*, requested: int, n_segment: int) -> int:
    """Resolve the candidate window the ordinary read streams in.

    Args:
        requested: The caller's `segment_block_size`; `0` asks for the default.
        n_segment: Candidate links available, the largest useful window.

    Returns:
        A positive window no larger than the candidate count.

    """
    window = requested if requested > 0 else _ORDINARY_CANDIDATE_WINDOW
    return max(1, min(window, n_segment))


def _link_blocks(
    *,
    links: _SegmentLinks,
    block_size: int,
    savings_endpoints: tuple[Float1D, Float1D] | None = None,
) -> tuple[FloatND, BoolND, IntND]:
    """Pad the links to a whole number of blocks and stack them for a scan.

    The padding is dead segments, which never bracket a query, so a padded block
    contributes nothing a live one would not. Its stable identities are the
    largest representable index so they cannot win a final identity comparison.

    Args:
        links: The candidate links.
        block_size: Candidates per block.

    Returns:
        Tuple of endpoint columns shaped `(n_block, block_size, 8)` without
        savings or `(n_block, block_size, 10)` with savings, their
        `(n_block, block_size)` live flags, and their equally shaped stable
        global link identities.

    """
    n_segment = links.live.shape[0]
    pad = (-n_segment) % block_size
    columns = jnp.stack(
        [
            _pad_column(column=links.left_grid, fill=0.0, n_padding=pad),
            _pad_column(column=links.right_grid, fill=0.0, n_padding=pad),
            _pad_column(column=links.left_value, fill=0.0, n_padding=pad),
            _pad_column(column=links.right_value, fill=0.0, n_padding=pad),
            _pad_column(column=links.left_policy, fill=0.0, n_padding=pad),
            _pad_column(column=links.right_policy, fill=0.0, n_padding=pad),
            _pad_column(column=links.left_marginal, fill=0.0, n_padding=pad),
            _pad_column(column=links.right_marginal, fill=0.0, n_padding=pad),
            *(
                []
                if savings_endpoints is None
                else [
                    _pad_column(column=endpoint, fill=0.0, n_padding=pad)
                    for endpoint in savings_endpoints
                ]
            ),
        ],
        axis=1,
    )
    live = (
        links.live
        if pad == 0
        else jnp.concatenate([links.live, jnp.zeros((pad,), dtype=bool)])
    )
    stable_index = (
        links.stable_index
        if pad == 0
        else jnp.concatenate(
            [
                links.stable_index,
                jnp.full(
                    (pad,),
                    jnp.iinfo(links.stable_index.dtype).max,
                    dtype=links.stable_index.dtype,
                ),
            ]
        )
    )
    return (
        columns.reshape(-1, block_size, columns.shape[1]),
        live.reshape(-1, block_size),
        stable_index.reshape(-1, block_size),
    )


def _pad_column(*, column: FloatND, fill: float, n_padding: int) -> FloatND:
    """Append `n_padding` dead entries so the column divides into whole blocks."""
    if n_padding == 0:
        return column
    return jnp.concatenate([column, jnp.full((n_padding,), fill, dtype=column.dtype)])


class _OrdinaryRank(NamedTuple):
    """The fields the ordinary read compares, in the order it compares them.

    The largest read wins outright; among candidates reading exactly level the
    right-continuous tie-break decides, preferring one that extends strictly to
    the right of the query and then the larger value-slope. A candidate that does
    not bracket the query carries `-inf` in every field, so it loses on the first
    comparison and never reaches the later ones.
    """

    value: FloatND
    """Value read along the link at the query."""
    right_available: FloatND
    """Whether the link extends strictly right of the query, as a number."""
    slope_high: FloatND
    """Leading word of the link's value-slope."""
    slope_low: FloatND
    """Trailing word of the same slope, which orders what the leading word ties."""
    stable_index: IntND
    """Global stored-link identity; the smaller identity wins the final tie."""


def _envelope_blocked_ordinary(
    *,
    links: _SegmentLinks,
    query: FloatND,
    block_size: int,
    savings_endpoints: tuple[Float1D, Float1D] | None = None,
) -> _EnvelopeReduction | _SavingsEnvelopeReduction:
    """Evaluate the ordinary envelope by streaming the candidates in blocks.

    Taking a maximum does not require the candidates to be present at once, so
    the scan holds one `(n_query, block_size)` window and a running winner. The
    winner is the largest read, ties broken right-continuously — one ordered
    comparison over the fields of `_OrdinaryRank`, applied first within a block
    and then between the block's best and the standing best. A challenger has to
    beat the standing winner outright to replace it, so an earlier candidate
    keeps a tie however the candidates are partitioned.

    Args:
        links: The candidate links.
        query: Abscissae to evaluate at.
        block_size: Candidates held against every query at once.

    Returns:
        The envelope's value, policy, and marginal at each query, NaN where no
        live link brackets it.

    """
    flat = query.reshape(-1)
    n_query = flat.shape[0]
    dtype = links.left_grid.dtype
    blocks, live_blocks, stable_blocks = _link_blocks(
        links=links, block_size=block_size, savings_endpoints=savings_endpoints
    )
    step = functools.partial(_ordinary_block_step, flat=flat, dtype=dtype)

    empty_rank = jnp.full((n_query,), -jnp.inf, dtype=dtype)
    empty_stable_index = jnp.full(
        (n_query,),
        jnp.iinfo(links.stable_index.dtype).max,
        dtype=links.stable_index.dtype,
    )
    empty_channel = jnp.full((n_query,), jnp.nan, dtype=dtype)
    carry, _ = jax.lax.scan(
        step,
        (
            *(empty_rank,) * 4,
            empty_stable_index,
            *(empty_channel,) * (3 if savings_endpoints is None else 4),
        ),
        (blocks, live_blocks, stable_blocks),
    )
    decided = jnp.isfinite(carry[0])
    value, policy, marginal = (
        jnp.where(decided, channel, jnp.nan).reshape(query.shape)
        for channel in carry[5:8]
    )
    reduction = _EnvelopeReduction(
        published=(value, policy, marginal),
        owner=jnp.where(decided, carry[4], NO_OWNER).reshape(query.shape),
    )
    if savings_endpoints is None:
        return reduction
    return _SavingsEnvelopeReduction(
        published=reduction.published,
        owner=cast("IntND", reduction.owner),
        savings=jnp.where(decided, carry[8], jnp.nan).reshape(query.shape),
    )


def _ordinary_block_best(
    *,
    block: FloatND,
    block_live: BoolND,
    block_stable_index: IntND,
    flat: Float1D,
    dtype: jnp.dtype,
) -> tuple[_OrdinaryRank, tuple[FloatND, ...]]:
    """The block's own winner per query, as its rank and its channels."""
    terms = _block_query_terms(
        block=block, live=block_live, flat=flat, arithmetic="ordinary"
    )
    excluded = jnp.full_like(terms.value, -jnp.inf)
    offered = _OrdinaryRank(
        value=jnp.where(terms.brackets, terms.value, excluded),
        right_available=jnp.where(
            terms.brackets,
            (flat[:, None] < terms.upper).astype(dtype),
            excluded,
        ),
        slope_high=jnp.where(terms.brackets, terms.slope_high, excluded),
        slope_low=jnp.where(terms.brackets, terms.slope_low, excluded),
        stable_index=jnp.where(
            terms.brackets,
            jnp.broadcast_to(block_stable_index[None, :], terms.brackets.shape),
            jnp.iinfo(block_stable_index.dtype).max,
        ),
    )
    still_tied = jnp.ones_like(offered.value, dtype=bool)
    for field in offered[:-1]:
        best = jnp.max(jnp.where(still_tied, field, -jnp.inf), axis=1, keepdims=True)
        still_tied = still_tied & (field == best)
    earliest = jnp.min(
        jnp.where(
            still_tied,
            offered.stable_index,
            jnp.iinfo(offered.stable_index.dtype).max,
        ),
        axis=1,
        keepdims=True,
    )
    still_tied = still_tied & (offered.stable_index == earliest)
    index = jnp.argmax(still_tied, axis=1)[:, None].astype(jnp.int32)
    associated = ()
    if block.shape[-1] == _LINK_COLUMNS_WITH_SAVINGS:
        associated = (
            _along_link(
                left=_pick_column(
                    column=jnp.broadcast_to(block[:, 8][None, :], terms.value.shape),
                    index=index,
                ),
                right=_pick_column(
                    column=jnp.broadcast_to(block[:, 9][None, :], terms.value.shape),
                    index=index,
                ),
                left_grid=_pick_column(
                    column=jnp.broadcast_to(block[:, 0][None, :], terms.value.shape),
                    index=index,
                ),
                right_grid=_pick_column(
                    column=jnp.broadcast_to(block[:, 1][None, :], terms.value.shape),
                    index=index,
                ),
                query=flat,
                arithmetic="ordinary",
            ),
        )
    return (
        _OrdinaryRank(*(_pick_column(column=field, index=index) for field in offered)),
        (
            _pick_column(column=terms.value, index=index),
            _pick_column(column=terms.policy, index=index),
            _pick_column(column=terms.marginal, index=index),
            *associated,
        ),
    )


# keyword-only-exempt: library-callback=jax.lax.scan
def _ordinary_block_step(
    carry: tuple[FloatND | IntND, ...],
    block_live_and_stable: tuple[FloatND, BoolND, IntND],
    *,
    flat: Float1D,
    dtype: jnp.dtype,
) -> tuple[tuple[FloatND | IntND, ...], None]:
    """Replace the standing ordinary winner where this block's best outranks it."""
    n_query = flat.shape[0]
    held = _OrdinaryRank(*carry[:5])
    standing = carry[5:]
    offered, channels = _ordinary_block_best(
        block=block_live_and_stable[0],
        block_live=block_live_and_stable[1],
        block_stable_index=block_live_and_stable[2],
        flat=flat,
        dtype=dtype,
    )
    decided = jnp.zeros((n_query,), dtype=bool)
    take = jnp.zeros((n_query,), dtype=bool)
    for challenging, holding in zip(offered[:-1], held[:-1], strict=True):
        take = take | (~decided & (challenging > holding))
        decided = decided | (challenging != holding)
    take = take | (~decided & (offered.stable_index < held.stable_index))
    return (
        *(
            jnp.where(take, challenging, holding)
            for challenging, holding in zip(offered, held, strict=True)
        ),
        *(
            jnp.where(take, offered_channel, standing_channel)
            for offered_channel, standing_channel in zip(
                channels, standing, strict=True
            )
        ),
    ), None
