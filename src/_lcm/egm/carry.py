"""Cross-period data channel of the DC-EGM solver.

Backward induction with DC-EGM threads more than the value-function array
between adjacent periods: the parent's Euler inversion needs the child's
value and marginal utility on the child's endogenous (resources-space) grid.
`EGMCarry` bundles these rows; the solve loop rolls a
`next_regime_to_continuation` mapping alongside `next_regime_to_V_arr`, with one
entry per carry-producing regime (DC-EGM regimes and terminal regimes a
DC-EGM regime can target).

The carry is also the shipped `ContinuationReader`: a parent asks it for a value
or a marginal in `EGM_ENDOGENOUS_COORDINATE` at a query instead of interpolating
its rows itself. `read_value_row`, `read_marginal_row`, and
`read_value_and_slope_row` are the per-row kernels behind those answers, shared
with the multi-target continuation aggregation, so the reader and the hot path
interpolate through one implementation.
"""

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.interp import (
    interp_and_derivative_on_prepared_grid,
    interp_on_prepared_grid,
    prepare_padded_grid,
)
from _lcm.egm.query_data import (
    EGMQueryData,
    NumericQueryLeaf,
    QueryNodeKind,
)
from _lcm.egm.query_data import (
    build_template_egm_query_data as build_template_egm_query_data,  # noqa: PLC0414
)
from _lcm.grids import Grid
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import (
    EGM_CONTINUATION,
    EGM_ENDOGENOUS_COORDINATE,
    ArtifactKey,
    ContinuationCapabilities,
)
from lcm.typing import (
    BoolND,
    Float1D,
    Float2D,
    FloatND,
    Int1D,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    StateName,
    StateOrActionName,
)

if TYPE_CHECKING:
    from _lcm.execution.output_layout import OutputRoleLeaf
    from _lcm.typing import PytreeChild
else:
    # `_lcm.typing` imports this module, and `_lcm.execution.output_layout` imports
    # `_lcm.typing`, so neither alias is importable here at runtime.
    type OutputRoleLeaf = object
    type PytreeChild = object


@dataclass(frozen=True, kw_only=True)
class EGMCarry:
    """Per-regime EGM solution rows threaded between adjacent periods.

    All rows share the trailing grid axis of static, per-regime length so the
    carry has a period-invariant pytree shape (periods sharing a compiled
    program never trigger retracing). Regimes with combo dimensions carry one
    row per combo: leading axes are the regime's discrete states (in V state
    order; process states are node-valued discrete dimensions), then its
    passive continuous states (in V continuous-state order, one node per
    combo), then its discrete actions. Floating arrays use the canonical float dtype.
    """

    endog_grid: FloatND
    """Endogenous grid in resources space, NaN-padded in the tail.

    Weakly ascending per row: envelope kink abscissae appear twice, carrying
    the left- and right-extrapolated policy values.
    """

    value: FloatND
    """Choice-specific value at `endog_grid`; `-inf` marks infeasible rows."""

    marginal_utility: FloatND
    """Marginal value of resources $\\partial v / \\partial R$ at `endog_grid`.

    Exactly `0.0` (never NaN) wherever `value` is `-inf`: infeasible rows get
    zero choice probability, and `0 \\cdot \\mu` must stay finite in the
    parent's probability-weighted expectation.
    """

    taste_shock_scale: ScalarFloat
    """EV1 taste-shock scale of the regime as a 0-d array; `0.0` = hard max."""

    breakpoints: FloatND | None = None
    """Per-row one-sided boundary locations in the child's liquid state.

    NaN-padded and published by solvers whose value rows carry a declared jump
    or a finite-to-infeasible boundary; `None` for smooth-valued regimes.
    One-sided values and marginals ride inside `endog_grid` as exact/adjacent
    or duplicated abscissae. This field marks the rows' topology for the
    stochastic-dimension fold, which must not average rows whose boundary
    locations differ across the folded nodes, and lets an EGM parent add the
    fixed-savings candidate where the child boundary binds.
    """

    @property
    def artifact_key(self) -> ArtifactKey:
        """Versioned identity under which EGM kernels publish this carry."""
        return EGM_CONTINUATION

    policy: FloatND | None = None
    """Exact consumption at `endog_grid`, or `None` when not published.

    Populated only by the continuous-only, jump-free ride-along NB-EGM core
    (`_assemble_ride_carry`), whose per-cell EGM step already computes the
    optimal consumption on the liquid grid. The continuous-outer simulation
    replay (`derive_inner_sim_policy`) reads this directly instead of
    re-inverting `marginal_utility`: the marginal carries the budget's
    cash-on-hand slope in liquid (`marginal = (d coh/d liquid) * u'(c)`), so
    inverting it recovers `c` only when that slope is unit — carrying the
    exact policy is correct for any affine budget slope.
    `None` for every other carry (cross-period continuations, discrete-branch
    or jump-schedule rows), whose downstream never reaches that replay.
    """

    query_data: EGMQueryData | None = None
    """Conditional candidate/support data interpreted only by model authority."""

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Return capabilities of the standalone carry reader.

        The rows publish value and their declared marginal in the endogenous
        coordinate. They name no owning candidate, and they locate their own
        boundaries only when a `breakpoints` row is published. Query payloads
        require a model-bound adapter, which supplies its own reader capabilities;
        their numeric metadata does not make this standalone reader queryable.
        """
        if self.query_data is not None:
            return ContinuationCapabilities()
        return ContinuationCapabilities(
            value=True,
            marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE}),
            exact_candidate_identity=False,
            discontinuities=self.breakpoints is not None,
        )

    def value_at(self, *, query: FloatND) -> FloatND:
        """Return the value of these rows at `query`, one query per row."""
        if self.query_data is not None:
            raise UnsupportedOperationError(
                "Feature rows require the model-bound query adapter."
            )
        rows = self._row_arguments(query=query)
        return jax.vmap(read_value_row)(
            search_grid=rows.search_grid,
            valid_length=rows.valid_length,
            xp=rows.xp,
            fp=self._flat_rows(array=self.value),
            fp_slopes=rows.fp_slopes,
            x_query=rows.x_query,
        ).reshape(self.endog_grid.shape[:-1])

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Return the marginal in `state` at `query`, one query per row."""
        if self.query_data is not None:
            raise UnsupportedOperationError(
                "Feature rows require the model-bound query adapter."
            )
        capabilities = self.capabilities
        if state not in capabilities.marginal_states:
            msg = (
                "An EGM carry publishes a marginal in "
                f"{sorted(capabilities.marginal_states)}, not in {state!r}."
            )
            raise ValueError(msg)
        rows = self._row_arguments(query=query)
        return jax.vmap(read_marginal_row)(
            search_grid=rows.search_grid,
            valid_length=rows.valid_length,
            xp=rows.xp,
            fp=self._flat_rows(array=self.marginal_utility),
            x_query=rows.x_query,
        ).reshape(self.endog_grid.shape[:-1])

    def query_data_for_transport(self) -> EGMQueryData | None:
        """Return the optional whole query record without copying its buffers."""
        return self.query_data

    def leaves(self) -> MappingProxyType[tuple[str, ...], NumericQueryLeaf]:
        """Return every published array of this carry by its field path."""
        leaves: dict[tuple[str, ...], NumericQueryLeaf] = {
            (name,): getattr(self, name)
            for name in _EGM_CARRY_FIELDS
            if name != "query_data" and getattr(self, name) is not None
        }
        if self.query_data is not None:
            leaves.update(
                {
                    ("query_data", *path): leaf
                    for path, leaf in self.query_data.leaves().items()
                }
            )
        return MappingProxyType(leaves)

    def _flat_rows(self, *, array: FloatND) -> Float2D:
        """Collapse every leading axis of one payload row set."""
        return array.reshape(-1, self.endog_grid.shape[-1])

    def _row_arguments(self, *, query: FloatND) -> _EGMRowArguments:
        """Build the per-row read arguments shared by both query methods."""
        grid_rows = self._flat_rows(array=self.endog_grid)
        search_rows, valid_rows = jax.vmap(prepare_padded_grid)(grid_rows)
        return _EGMRowArguments(
            search_grid=search_rows,
            valid_length=valid_rows,
            xp=grid_rows,
            fp_slopes=self._flat_rows(array=self.marginal_utility),
            x_query=jnp.broadcast_to(query, self.endog_grid.shape[:-1]).reshape(-1),
        )


# Pytree registration with an `__init__`-bypassing unflatten: JAX's transform
# and AOT-lowering machinery reconstructs pytrees with non-array leaves
# (`ArgInfo`, tracers, `None`), which the runtime-checked constructor would
# reject. Flatten order matches field declaration order.
_EGM_CARRY_FIELDS = (
    "endog_grid",
    "value",
    "marginal_utility",
    "taste_shock_scale",
    "breakpoints",
    "policy",
    "query_data",
)


def _flatten_egm_carry(
    carry: EGMCarry,
) -> tuple[tuple[PytreeChild | EGMQueryData, ...], None]:
    return tuple(getattr(carry, name) for name in _EGM_CARRY_FIELDS), None


def _flatten_egm_carry_with_keys(
    carry: EGMCarry,
) -> tuple[
    tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild | EGMQueryData], ...], None
]:
    """Flatten with field-named keys so a leaf path reads `.endog_grid`."""
    return (
        tuple(
            (jax.tree_util.GetAttrKey(name), getattr(carry, name))
            for name in _EGM_CARRY_FIELDS
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_egm_carry(
    _aux: None, children: Iterable[PytreeChild | EGMQueryData]
) -> EGMCarry:
    return _assemble_egm_carry(children=children)


def _assemble_egm_carry(
    *, children: Iterable[PytreeChild | OutputRoleLeaf | EGMQueryData]
) -> EGMCarry:
    """Set the carry's fields in flatten order, bypassing its checked constructor."""
    carry = object.__new__(EGMCarry)
    for name, child in zip(_EGM_CARRY_FIELDS, children, strict=True):
        object.__setattr__(carry, name, child)
    return carry


def stack_egm_carries(*, candidates: tuple[EGMCarry, ...]) -> EGMCarry:
    """Insert one outer axis while keeping fixed-cell floors and mesh data shared."""
    if not candidates:
        raise ValueError("At least one outer candidate is required.")
    structure = jax.tree.structure(candidates[0])
    if any(jax.tree.structure(candidate) != structure for candidate in candidates[1:]):
        raise TypeError("Outer candidates require identical carry/query structure.")
    return cast(
        "EGMCarry",
        jax.tree.map_with_path(_stack_carry_leaf, *candidates),
    )


def _stack_carry_leaf(
    path: tuple[jax.tree_util.GetAttrKey, ...], *leaves: NumericQueryLeaf
) -> NumericQueryLeaf:
    """Stack candidate-dependent axes without duplicating shared query banks."""
    names = tuple(key.name for key in path)
    if names in (("taste_shock_scale",), ("query_data", "floor_used")):
        return leaves[0]
    if "adaptive_outer_data" in names:
        return leaves[0]
    axis = -3 if "liquid_corner_data" in names else -2
    return jnp.stack(leaves, axis=axis)


jax.tree_util.register_pytree_with_keys(
    EGMCarry,
    _flatten_egm_carry_with_keys,
    _unflatten_egm_carry,
    _flatten_egm_carry,
)


class OwnedRowRead(NamedTuple):
    """Associated candidate with missing support separate from corrupt records."""

    value: ScalarFloat
    marginal: ScalarFloat
    savings: ScalarFloat
    supported: ScalarBool
    invalid_numeric: ScalarBool
    owner_id: ScalarInt
    side_id: ScalarInt


def _flatten_owned_row_read(
    result: OwnedRowRead,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(result), None


def _flatten_owned_row_read_with_keys(
    result: OwnedRowRead,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(OwnedRowRead._fields, result, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_owned_row_read(
    _aux: None, children: Iterable[PytreeChild]
) -> OwnedRowRead:
    """Restore temporary query batches and compiler roles without scalar checks."""
    leaves = tuple(children)
    if len(leaves) != len(OwnedRowRead._fields):
        raise TypeError("An owned row read requires all its associated fields.")
    return cast("OwnedRowRead", tuple.__new__(OwnedRowRead, leaves))


jax.tree_util.register_pytree_with_keys(
    OwnedRowRead,
    _flatten_owned_row_read_with_keys,
    _unflatten_owned_row_read,
    _flatten_owned_row_read,
)


def read_owned_row(
    *,
    query: ScalarFloat,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    owner_capacity: int,
    side_is_row_side: bool = True,
) -> OwnedRowRead:
    """Read V, marginal and original saving through one owned conditional bracket.

    The caller supplies the canonical run capacity and retains model authority.
    `side_is_row_side` distinguishes row-side labels from native conditional
    envelope labels retained on a sampled liquid row. The latter retain their
    provenance while the value derivative uses the published row's right piece.
    Exact nodes select the right copy without requiring a neighbouring link.
    Below support the first owning secant extends; above support the last
    original candidate clamps. No gap or foreign run becomes represented.
    """
    search, count = prepare_padded_grid(grid)
    last = jnp.maximum(count - 1, 0)
    insertion = jnp.searchsorted(search, query, side="right")
    exact_index = jnp.clip(insertion - 1, 0, last).astype(jnp.int32)
    endpoint = (count > 0) & ((grid[exact_index] == query) | (query >= grid[last]))
    endpoint_index = jnp.where(query >= grid[last], last, exact_index)
    upper = jnp.clip(insertion, 1, jnp.maximum(count - 1, 1)).astype(jnp.int32)
    lower = upper - 1
    pair_link = (
        data.link_valid[jnp.clip(lower, 0, data.link_valid.shape[0] - 1)]
        if data.link_valid.shape[0]
        else jnp.asarray(False)  # noqa: FBT003
    )
    lower = jnp.where(endpoint, endpoint_index, lower)
    upper = jnp.where(endpoint, endpoint_index, upper)
    arithmetic_grid = jnp.where(jnp.isfinite(grid), grid, 0.0)
    width = arithmetic_grid[upper] - arithmetic_grid[lower]
    weight_upper = jnp.where(
        endpoint,
        1.0,
        (query - arithmetic_grid[lower]) / jnp.where(width > 0.0, width, 1.0),
    )
    active_lower = (1.0 - weight_upper) != 0.0
    active_upper = weight_upper != 0.0
    finite = data.node_kind == QueryNodeKind.FINITE
    economic = data.node_kind == QueryNodeKind.ECONOMIC_MINUS_INF
    known = finite | economic
    numeric_valid = (
        jnp.isfinite(grid)
        & jnp.isfinite(data.savings)
        & (
            (finite & jnp.isfinite(value) & jnp.isfinite(marginal))
            | (economic & jnp.isneginf(value) & (marginal == 0.0))
        )
    )
    owner_valid = (data.owner_id >= 0) & (data.owner_id < owner_capacity)
    side_valid = (data.side_id >= -1) & (data.side_id <= 1)
    supported_node = known & numeric_valid & owner_valid & side_valid
    invalid_record = (data.node_kind != QueryNodeKind.PAD) & (
        ~known | ~owner_valid | ~side_valid
    )
    invalid_numeric = (
        active_lower & (invalid_record[lower] | (known[lower] & ~numeric_valid[lower]))
    ) | (
        active_upper & (invalid_record[upper] | (known[upper] & ~numeric_valid[upper]))
    )
    negative_economic = (economic[lower] & ((1.0 - weight_upper) < 0.0)) | (
        economic[upper] & (weight_upper < 0.0)
    )
    positive_economic = (economic[lower] & ((1.0 - weight_upper) > 0.0)) | (
        economic[upper] & (weight_upper > 0.0)
    )
    safe_value = jnp.where(numeric_valid, value, 0.0)
    safe_marginal = jnp.where(finite & numeric_valid, marginal, 0.0)
    safe_savings = jnp.where(numeric_valid, data.savings, 0.0)
    row_value, row_marginal, row_savings = _read_owned_channels(
        query=query,
        grid=arithmetic_grid,
        value=safe_value,
        marginal=safe_marginal,
        savings=safe_savings,
        data=data,
        supported_node=supported_node,
        lower=lower,
        upper=upper,
        endpoint=endpoint,
        node_index=endpoint_index,
        last=last,
        side_is_row_side=side_is_row_side,
    )
    complete_nodes = (~active_lower | supported_node[lower]) & (
        ~active_upper | supported_node[upper]
    )
    connected = endpoint | (pair_link & (data.owner_id[lower] == data.owner_id[upper]))
    invalid_numeric |= (
        complete_nodes
        & connected
        & (
            ~jnp.isfinite(row_savings)
            | (
                ~positive_economic
                & (~jnp.isfinite(row_value) | ~jnp.isfinite(row_marginal))
            )
        )
    )
    supported = (
        (count > 0) & complete_nodes & connected & ~negative_economic & ~invalid_numeric
    )
    return OwnedRowRead(
        value=jnp.where(
            supported, jnp.where(positive_economic, -jnp.inf, row_value), 0.0
        ),
        marginal=jnp.where(supported & ~positive_economic, row_marginal, 0.0),
        savings=jnp.where(supported, row_savings, 0.0),
        supported=supported,
        invalid_numeric=invalid_numeric,
        owner_id=jnp.where(supported, data.owner_id[lower], -1),
        side_id=jnp.where(supported & endpoint, data.side_id[lower], 0),
    )


def _read_owned_channels(
    *,
    query: ScalarFloat,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    savings: Float1D,
    data: EGMQueryData,
    supported_node: BoolND,
    lower: ScalarInt,
    upper: ScalarInt,
    endpoint: ScalarBool,
    node_index: ScalarInt,
    last: ScalarInt,
    side_is_row_side: bool,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat]:
    """Reuse the declared interpolants on one safe, already owned local piece.

    A represented exact node without an owning adjacent piece reads as a
    singleton. Its value derivative is zero while its ordinary marginal stays
    available. Outgoing endpoints use their left piece; incoming nodes use
    their right piece. Local scratch never adds represented row support.
    """
    use_left = (node_index == last) | (
        side_is_row_side & (data.side_id[node_index] == -1)
    )
    piece_lower = jnp.where(
        endpoint, jnp.where(use_left, jnp.maximum(node_index - 1, 0), node_index), lower
    )
    piece_upper = jnp.where(
        endpoint,
        jnp.where(use_left, node_index, jnp.minimum(node_index + 1, last)),
        upper,
    )
    link = (
        data.link_valid[jnp.clip(piece_lower, 0, data.link_valid.shape[0] - 1)]
        if data.link_valid.shape[0]
        else jnp.asarray(False)  # noqa: FBT003
    )
    has_piece = (
        (piece_upper > piece_lower)
        & link
        & supported_node[piece_lower]
        & supported_node[piece_upper]
        & (data.owner_id[piece_lower] == data.owner_id[piece_upper])
        & (grid[piece_upper] > grid[piece_lower])
    )
    fallback = jnp.where(endpoint, node_index, lower)
    piece_lower = jnp.where(has_piece, piece_lower, fallback)
    piece_upper = jnp.where(has_piece, piece_upper, fallback)
    indices = jnp.stack((piece_lower, piece_upper))
    local_grid = grid[indices]
    count = jnp.where(has_piece, jnp.int32(2), jnp.int32(1))
    search_grid = jnp.where(jnp.arange(2) < count, local_grid, jnp.inf)
    read = functools.partial(
        interp_on_prepared_grid,
        x_query=query,
        search_grid=search_grid,
        valid_length=count,
        xp=local_grid,
    )
    return (
        read(fp=value[indices], fp_slopes=marginal[indices]),
        read(fp=marginal[indices]),
        read(fp=savings[indices]),
    )


def read_value_row(
    *,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
    x_query: ScalarFloat,
) -> ScalarFloat:
    """Interpolate one carry value row at its query."""
    return interp_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


def read_marginal_row(
    *,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    x_query: ScalarFloat,
) -> ScalarFloat:
    """Interpolate one carry row at its own query."""
    return interp_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
    )


def read_value_and_slope_row(
    *,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
    x_query: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Value read and its analytic derivative.

    The closed-form derivative of the selected piece — not autodiff through
    the bracket-selection program, whose `searchsorted`/`clip` representation
    returns arbitrary subgradients at exact grid nodes (a routine alignment: a
    zero-savings corner on a child grid that starts at zero).
    """
    return interp_and_derivative_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


@dataclass(frozen=True, kw_only=True)
class _EGMRowArguments:
    """The per-row abscissae one query pass reads every carry row on."""

    search_grid: Float2D
    """Padded search grids, one row per flattened leading cell."""

    valid_length: Int1D
    """Number of valid nodes in each row's search grid."""

    xp: Float2D
    """The endogenous grid rows the values are tabulated on."""

    fp_slopes: Float2D
    """The marginal rows used as the value interpolant's slopes."""

    x_query: Float1D
    """One query per flattened leading cell."""


def egm_carry_role_tree(
    *,
    row: OutputRoleLeaf,
    scalar: OutputRoleLeaf,
    breakpoints: OutputRoleLeaf | None,
    policy: OutputRoleLeaf | None,
    query_data: EGMQueryData | None = None,
) -> EGMCarry:
    """Build an `EGMCarry`-shaped tree of output roles.

    A kernel declares its carry outputs with the same pytree structure as the
    carry it publishes: one role for each of the three grid rows, one for the
    0-d taste-shock scale, and `None` for a row it does not publish. The leaves
    are role declarations rather than arrays, so the tree is assembled through
    the field-by-field assembly rather than the runtime-checked constructor.
    """
    return _assemble_egm_carry(
        children=(row, row, row, scalar, breakpoints, policy, query_data)
    )


def egm_query_role_tree(
    *, query_data: EGMQueryData, state_names: tuple[StateName, ...]
) -> EGMQueryData:
    """Declare each numeric query buffer's exact dtype, shape and state axes."""
    return cast(
        "EGMQueryData",
        jax.tree.map_with_path(
            functools.partial(_query_output_role, state_names=state_names),
            query_data,
        ),
    )


# keyword-only-exempt: library-callback=jax.tree.map_with_path
def _query_output_role(
    path: tuple[jax.tree_util.GetAttrKey, ...],
    leaf: NumericQueryLeaf,
    *,
    state_names: tuple[StateName, ...],
) -> OutputRoleLeaf:
    """Replicate shared mesh metadata and place conditional banks by fixed states."""
    from _lcm.execution.output_layout import StateAxesLeading  # noqa: PLC0415

    names = tuple(key.name for key in path)
    shared_mesh = names[0] == "adaptive_outer_data" and names[-1] in (
        "outer_nodes",
        "outer_count",
        "outer_node_kind",
    )
    return StateAxesLeading(
        state_names=() if shared_mesh else state_names,
        dtype=leaf.dtype,
        shape=leaf.shape,
    )


def build_template_egm_carry(
    *,
    n_rows: int,
    leading_shape: tuple[int, ...] = (),
    query_data: EGMQueryData | None = None,
) -> EGMCarry:
    """Build a benign all-finite carry template with `n_rows` grid slots.

    Used to initialize the rolling `next_regime_to_continuation` mapping before
    a regime has been solved, and as the lowering argument when AOT-compiling
    EGM kernels. The endogenous grid is strictly ascending and every row is
    finite, so a parent kernel evaluated against the template produces finite
    (probability-zeroed) contributions rather than NaN.

    Args:
        n_rows: Static length of the carry rows.
        leading_shape: Sizes of the regime's combo dimensions (discrete
            states, then passive states, then discrete actions); empty for
            regimes without combo dimensions.

    Returns:
        Carry with an ascending unit-interval grid and all-zero value and
        marginal-utility rows, broadcast over `leading_shape`.

    """
    dtype = canonical_float_dtype()
    shape = (*leading_shape, n_rows)
    return EGMCarry(
        endog_grid=jnp.broadcast_to(jnp.linspace(0.0, 1.0, n_rows, dtype=dtype), shape),
        value=jnp.zeros(shape, dtype=dtype),
        marginal_utility=jnp.zeros(shape, dtype=dtype),
        taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        query_data=query_data,
    )


def shard_carry_template(
    *,
    template: EGMCarry,
    grids: Mapping[StateOrActionName, Grid],
    sharded_state_names: frozenset[StateName],
    leading_axis_names: tuple[StateName, ...],
    devices: tuple[jax.Device, ...],
) -> EGMCarry:
    """Place a carry template on the same device sharding as runtime carries.

    The compiled cores accept one carry pytree layout across all periods. A
    carry publisher computes its runtime rows alongside the regime's sharded
    value array, so the published carry inherits the state sharding on its
    leading axes; the template — the compile-time lowering sample and the
    first backward-induction input — must carry that sharding too, or the
    cores compile for replicated carries and reject every runtime period.

    Consumed by every carry-template producer with state-shaped leading axes:

    - the NB-EGM ride-along template (leading axes = ride-along states);
    - the living-brute child template (leading axes = discrete then passive
      continuous states);
    - the terminal-wealth template (same leading-axes layout).

    Leading axes follow `leading_axis_names` order; trailing row axes stay
    unsharded. Scalars replicate across the mesh. `devices` are the devices the
    placement assigned to the regime that publishes the carry, so the
    template's mesh is the one its value array runs on.

    A regime no state grid of which is distributed has no mesh, and its
    template is committed to the one device the placement gave it — the same
    rule its value array follows. Where that device is the one a single-device
    solve would have used anyway, the template keeps the default placement.
    """
    # Function-local: `_lcm.engine` imports `_lcm.continuation`, which imports
    # `EGMCarry` from this module, so a module-level import cycles.
    from _lcm.engine import _build_regime_sharding  # noqa: PLC0415

    plan = _build_regime_sharding(
        grids=MappingProxyType(dict(grids)),
        sharded_state_names=sharded_state_names,
        devices=devices,
    )
    if plan is None:
        return _place_carry_template_on_one_device(template=template, devices=devices)
    if not any(name in plan.distributed_state_names for name in leading_axis_names):
        return template
    leading_spec = jax.NamedSharding(
        plan.mesh,
        jax.P(
            *(
                name if name in plan.distributed_state_names else None
                for name in leading_axis_names
            )
        ),
    )
    scalar_spec = jax.NamedSharding(plan.mesh, jax.P())
    return jax.tree.map_with_path(
        functools.partial(
            _place_carry_leaf, scalar_spec=scalar_spec, leading_spec=leading_spec
        ),
        template,
    )


def _place_carry_template_on_one_device(
    *, template: EGMCarry, devices: tuple[jax.Device, ...]
) -> EGMCarry:
    """Commit an unsharded template to the single device its regime runs on.

    The default placement is kept where a single-device solve would have put
    the template there anyway, which is what `placed_V_sharding` does for the
    same regime's value array.
    """
    # Function-local: `_lcm.execution.execution_plan` reaches `_lcm.typing`,
    # which imports `EGMCarry` from this module, so a module-level import
    # cycles.
    from _lcm.execution.execution_plan import visible_devices  # noqa: PLC0415

    visible = visible_devices()
    if devices == (visible[0],) or len(devices) == len(visible):
        return template
    return jax.device_put(template, jax.sharding.SingleDeviceSharding(devices[0]))


# keyword-only-exempt: library-callback=jax.tree.map_with_path
def _place_carry_leaf(
    path: tuple[jax.tree_util.GetAttrKey, ...],
    leaf: NumericQueryLeaf,
    *,
    scalar_spec: jax.NamedSharding,
    leading_spec: jax.NamedSharding,
) -> NumericQueryLeaf:
    """Replicate scalar/shared mesh data and partition fixed-state banks."""
    names = tuple(key.name for key in path)
    shared_mesh = "adaptive_outer_data" in names and names[-1] in (
        "outer_nodes",
        "outer_count",
        "outer_node_kind",
    )
    return jax.device_put(
        leaf, scalar_spec if leaf.ndim == 0 or shared_mesh else leading_spec
    )
