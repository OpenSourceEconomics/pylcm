"""Private balanced source-region catalog and original-selection read guard.

The catalog axis is resources, independent of quadrature rows. It stores only
structural native-piece/branch identities and uncertainty, never nodal policy,
value or marginal statistics. Candidate scores always use the original reader.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, replace
from itertools import islice
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluator
from _lcm.egm.current_iid_log_utility import LogUtilityProgram
from _lcm.egm.current_iid_resource_query import (
    read_resources_candidate,
    read_resources_candidate_details,
)
from _lcm.egm.current_iid_resources_artifact import (
    ResourcesRowLeaf,
    resources_row_payload,
)
from _lcm.egm.interp import materialize_hermite_shadow
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.step_core import OwnedResourcesRow, _EgmKernelPieces
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.probability import is_represented_zero
from _lcm.solution.feasibility import (
    BoundInputStatus,
    ContinuationReadCounts,
    QueryInputStatus,
)
from _lcm.typing import EconFunctionArg, EconFunctionKwargs, PytreeChild
from lcm._solver_api.replay import ContinuationLeaf
from lcm.shocks import ShockName
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ReferenceName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


@dataclass(frozen=True, kw_only=True)
class ResourcesRegionCatalog:
    """Original resource-space pieces with bounded unresolved neighborhoods."""

    points: Float1D
    """Sorted represented resource coordinates; padding is outside length."""
    length: ScalarInt
    """Mandatory retained prefix, bounded by the existing capacity."""
    smooth: BoolND
    """Cells whose ends keep one owner/corner branch; no interior proof."""
    piece: IntND
    """Original local Hermite lower index for each proposed interval."""
    branch: IntND
    """Original row (zero) or borrowing corner (one) interval proposal."""
    reason: IntND
    """Geometry/legality/critical/crossing reason plus stop-condition bits."""
    valid: ScalarBool
    """Global original-source and mandatory-node integrity; not coverage."""
    exact_owner: IntND
    """Original source run identity at each exact resource record."""
    exact_side: IntND
    """Original represented source side at that exact resource."""
    exact_corner: BoolND
    """Original row/corner choice, without interpolation between records."""
    exact_feasible: BoolND
    """Original selected-candidate feasibility at the exact resource."""
    exact_piece: IntND
    """Original owning local-piece index at the exact resource."""
    exact_row_legal: BoolND
    """Original ordinary-row legality before corner comparison."""


@dataclass(frozen=True, kw_only=True)
class BalancedResourcesGuard:
    """Separately acquired source row, dated context and original law operands."""

    row_payload: tuple[ResourcesRowLeaf, ...]
    """Original canonical row buffers in numerical leaf order."""
    row_layout: tuple[bool, int | None, bool]
    """Presence of original floor, child and read-count records."""
    pool_values: tuple[EconFunctionArg, ...]
    pool_names: tuple[ReferenceName, ...]
    physical_values: tuple[FloatND | IntND, ...]
    physical_names: tuple[ShockName, ...]
    factors: FloatND
    """Original factored joint law weights and represented-live population."""
    law_valid: ScalarBool
    """Mandatory original law and utility-class numerical integrity."""
    resources_slope: ScalarFloat
    """Original admitted resource chain factor for the economic marginal."""
    catalog: ResourcesRegionCatalog
    """Resource-axis structural authority, never a current-node policy bank."""

    @property
    def row(self) -> OwnedResourcesRow:
        """Restore only the fixed native row layout over the acquired buffers."""
        floor, children, counts = self.row_layout
        expected = (
            len(OwnedEnvelope._fields)
            + len(OwnedResourcesRow._fields)
            - 4
            + (len(BoundInputStatus._fields) if floor else 0)
            + (0 if children is None else children * len(QueryInputStatus._fields))
            + (len(ContinuationReadCounts._fields) if counts else 0)
        )
        if len(self.row_payload) != expected:
            raise TypeError("Balanced guard lost its original row leaf layout.")
        leaves = iter(self.row_payload)
        envelope = tuple.__new__(
            OwnedEnvelope, islice(leaves, len(OwnedEnvelope._fields))
        )
        body = tuple(islice(leaves, len(OwnedResourcesRow._fields) - 4))
        floor_status = (
            tuple.__new__(
                BoundInputStatus, islice(leaves, len(BoundInputStatus._fields))
            )
            if floor
            else None
        )
        child_status = (
            None
            if children is None
            else tuple(
                tuple.__new__(
                    QueryInputStatus, islice(leaves, len(QueryInputStatus._fields))
                )
                for _ in range(children)
            )
        )
        read_counts = (
            tuple.__new__(
                ContinuationReadCounts,
                islice(leaves, len(ContinuationReadCounts._fields)),
            )
            if counts
            else None
        )
        return cast(
            "OwnedResourcesRow",
            tuple.__new__(
                OwnedResourcesRow,
                (envelope, *body, floor_status, child_status, read_counts),
            ),
        )

    @property
    def pool(self) -> EconFunctionKwargs:
        """Recreate the original names over separately retained numerical operands."""
        return MappingProxyType(
            dict(zip(self.pool_names, self.pool_values, strict=True))
        )

    @property
    def physical(self) -> Mapping[ShockName, FloatND | IntND]:
        """Recreate the original physical frame without copying its buffers."""
        return MappingProxyType(
            dict(zip(self.physical_names, self.physical_values, strict=True))
        )

    @classmethod
    def from_native(
        cls,
        *,
        row: OwnedResourcesRow,
        pool: EconFunctionKwargs,
        physical: Mapping[ShockName, FloatND | IntND],
        factors: FloatND,
        law_valid: ScalarBool,
        resources_slope: ScalarFloat,
        catalog: ResourcesRegionCatalog,
    ) -> BalancedResourcesGuard:
        """Pack the original row and frames at their closed artifact boundary."""
        if (
            type(row) is not OwnedResourcesRow
            or type(row.envelope) is not OwnedEnvelope
        ):
            raise TypeError("Balanced guard requires its original native row.")
        if (
            (
                row.floor_status is not None
                and type(row.floor_status) is not BoundInputStatus
            )
            or (
                row.child_status is not None
                and (
                    type(row.child_status) is not tuple
                    or any(
                        type(item) is not QueryInputStatus for item in row.child_status
                    )
                )
            )
            or (
                row.continuation_read_counts is not None
                and type(row.continuation_read_counts) is not ContinuationReadCounts
            )
        ):
            raise TypeError(
                "Balanced guard requires its original native status layout."
            )
        return cls(
            row_payload=resources_row_payload(row=row),
            row_layout=(
                row.floor_status is not None,
                None if row.child_status is None else len(row.child_status),
                row.continuation_read_counts is not None,
            ),
            pool_values=tuple(pool.values()),
            pool_names=tuple(pool),
            physical_values=tuple(physical.values()),
            physical_names=tuple(physical),
            factors=factors,
            law_valid=law_valid,
            resources_slope=resources_slope,
            catalog=catalog,
        )


@dataclass(frozen=True, kw_only=True)
class BalancedProjectedEGMCarry(ProjectedEGMCarry):
    """Integrated channels, value slopes and original-candidate guard custody."""

    value_slope: FloatND
    """Derivative of the reference value at each finite record; zero elsewhere."""
    guard: BalancedResourcesGuard
    """Original source/law/context arrays with real native reads and lifetime."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.value_slope.shape != self.grid.shape:
            raise ValueError("Case-R value slopes have a different row schema.")
        if self.value_slope.dtype != self.grid.dtype:
            raise TypeError("Case-R value slopes must share the grid dtype.")

    @property
    def guard_required(self) -> bool:
        return True

    @property
    def value_slope_channel(self) -> FloatND:
        return self.value_slope

    def leaves(self) -> Mapping[tuple[str, ...], ContinuationLeaf]:
        """Expose every original guard buffer to native reads and lifetime rules."""
        return MappingProxyType(
            {path: leaf for path, (_, leaf) in self._native_paths().items()}
        )

    def input_path(self, *, leaf_path: tuple[str, ...]) -> tuple[str | int, ...]:
        """Translate stored string identity into the actual tuple/dataclass path."""
        return self._native_paths()[leaf_path][0]

    def _native_paths(
        self,
    ) -> Mapping[tuple[str, ...], tuple[tuple[str | int, ...], ContinuationLeaf]]:
        result = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(self)[0]:
            node: _NativePathNode = self
            consumer: list[str | int] = []
            identity: list[str] = []
            for raw_key in path:
                key = _semantic_native_key(node=node, key=raw_key)
                identity.append(_path_name(key=key))
                if isinstance(key, jax.tree_util.GetAttrKey):
                    # The registered records publish attribute keys; native argument
                    # resolution traverses tuple positions before attributes.
                    segment = (
                        node._fields.index(key.name)
                        if isinstance(
                            node,
                            (
                                OwnedResourcesRow,
                                OwnedEnvelope,
                                BoundInputStatus,
                                QueryInputStatus,
                                ContinuationReadCounts,
                            ),
                        )
                        else key.name
                    )
                    consumer.append(segment)
                    node = getattr(node, key.name)
                elif isinstance(key, jax.tree_util.DictKey):
                    if not isinstance(node, Mapping) or not isinstance(key.key, str):
                        raise TypeError(
                            "Balanced guard mapping lost its actual string-key schema."
                        )
                    consumer.append(key.key)
                    node = node[key.key]
                elif isinstance(key, jax.tree_util.SequenceKey):
                    if not isinstance(node, tuple):
                        raise TypeError(
                            "Balanced guard sequence lost its original tuple schema."
                        )
                    consumer.append(key.idx)
                    node = node[key.idx]
                else:
                    raise TypeError(
                        "Balanced guard has an unsupported native path key."
                    )
            if node is not leaf:
                raise TypeError(
                    "Balanced guard locator lost its original leaf identity."
                )
            if tuple(identity) in result:
                raise TypeError("Balanced guard has colliding semantic leaf paths.")
            result[tuple(identity)] = (
                tuple(consumer),
                cast("ContinuationLeaf", leaf),
            )
        return MappingProxyType(result)


@dataclass(frozen=True, kw_only=True)
class BalancedResourcesProgram:
    """Static original callbacks; numerical guard operands never close over them."""

    utility: LogUtilityProgram
    """Full unbound authenticated utility and original score callback."""
    resources: Callable[..., FloatND]
    """The original structurally certified rounded resource callback."""
    resources_names: tuple[ReferenceName, ...]
    """Original resource signature used by the producer and consumed guard."""
    liquid_state: ReferenceName
    """Actual pre-shock liquid coordinate of the producer."""
    owner_capacity: int
    """Original admitted native run capacity."""

    @classmethod
    def authenticate(
        cls,
        *,
        pieces: _EgmKernelPieces,
        owner_capacity: int,
        abstract_pool: Mapping[ReferenceName, jax.ShapeDtypeStruct | EconFunctionArg]
        | None = None,
    ) -> BalancedResourcesProgram:
        """Authenticate the full unbound utility; preserve original resources."""
        import inspect  # noqa: PLC0415

        names = tuple(inspect.signature(pieces.utility_func).parameters)
        dtype = pieces.savings_nodes.dtype
        abstract = (
            {
                name: jax.ShapeDtypeStruct((), jnp.int32 if name == "period" else dtype)
                for name in names
            }
            if abstract_pool is None
            else {**abstract_pool, pieces.action_name: jax.ShapeDtypeStruct((), dtype)}
        )
        utility = LogUtilityProgram.authenticate(
            function=pieces.utility_func,
            action_name=pieces.action_name,
            abstract_pool=abstract,
        )
        return cls(
            utility=utility,
            resources=pieces.own_resources_func,
            resources_names=tuple(
                inspect.signature(pieces.own_resources_func).parameters
            ),
            liquid_state=pieces.euler_state_name,
            owner_capacity=owner_capacity,
        )

    def evaluator(self, *, guard: BalancedResourcesGuard) -> CurrentResourcesEvaluator:
        """Reconstruct original candidate reads from explicit acquired operands."""

        from _lcm.egm.current_iid_native import BoundResourcesMap  # noqa: PLC0415

        resources = BoundResourcesMap(
            function=self.resources,
            liquid_state=self.liquid_state,
            pool=guard.pool,
            names=self.resources_names,
        )
        return CurrentResourcesEvaluator(
            row=guard.row,
            resources=resources,
            resources_slope=guard.resources_slope,
            utility_of_action=BoundUtilityOfAction(
                utility_func=self.utility.function,
                action_name=self.utility.action_name,
                bound=guard.pool,
            ),
            owner_capacity=self.owner_capacity,
        )

    def validate(self, *, carry: ProjectedEGMCarry, query: ScalarFloat) -> ScalarBool:
        """Compare original rounded tuples with resource-region authority.

        The returned continuation channels remain Hermite-read values and
        separately interpolated economic marginals. Original reference levels are not
        substituted into the parent's Euler computation.
        """
        if not isinstance(carry, BalancedProjectedEGMCarry):
            raise TypeError(
                "Balanced read requires its original numeric guard payload."
            )
        return self.validate_guard(guard=carry.guard, query=query)

    def value_slope(
        self, *, guard: BalancedResourcesGuard, query: ScalarFloat
    ) -> ScalarFloat:
        """Derivative of the reference value, the Hermite slope of the level read."""
        return self.evaluator(guard=guard).value_slope(
            liquid=query, physical=guard.physical, factors=guard.factors
        )

    def validate_guard(
        self, *, guard: BalancedResourcesGuard, query: ScalarFloat
    ) -> ScalarBool:
        """Verify a transient original rounded roster against resource authority."""
        evaluator = self.evaluator(guard=guard)
        result, selected, row_legal, piece = evaluator.integrate_with_roster(
            liquid=query,
            physical=guard.physical,
            factors=guard.factors,
            law_valid=guard.law_valid,
        )
        live = jnp.all(~is_represented_zero(guard.factors), axis=0)

        def matches(
            inputs: tuple[
                Mapping[ShockName, ScalarFloat | ScalarInt],
                ScalarBool,
                ScalarInt,
                ScalarInt,
                ScalarBool,
                ScalarBool,
                ScalarBool,
                ScalarInt,
            ],
        ) -> ScalarBool:
            draw, active, owner, side, corner, feasible, legal, native_piece = inputs
            return jax.lax.cond(
                active,
                lambda: _matches_catalog(
                    catalog=guard.catalog,
                    row=guard.row,
                    resources=evaluator.resources(query, draw),
                    owner=owner,
                    side=side,
                    corner=corner,
                    feasible=feasible,
                    legal=legal,
                    native_piece=native_piece,
                ),
                lambda: jnp.ones((), dtype=jnp.bool_),
            )

        matched = jax.lax.map(
            matches,
            (
                guard.physical,
                live,
                selected.native_owner,
                selected.side_id,
                selected.corner,
                selected.feasible,
                row_legal,
                piece,
            ),
        )
        _, _, utility_valid = self.utility.coefficients(
            pool=guard.pool, dtype=query.dtype
        )
        return (
            result.valid
            & guard.catalog.valid
            & utility_valid
            & jnp.all(~live | matched)
        )


def build_resources_catalog(
    *,
    evaluator: CurrentResourcesEvaluator,
    scale: ScalarFloat,
    required: Float1D,
    capacity: int,
) -> ResourcesRegionCatalog:
    """Subdivide original resources pieces until cell ends share a selection.

    A cell links when both ends keep the piece owner or corner branch and are
    feasible. Owner/branch changes, contacts and unresolved geometry stay
    holes. No interior-selection proof is made; `validate_guard` checks
    consumed ownership ex post.
    """
    row = evaluator.row
    grid = row.envelope.grid
    lower, upper = row.resources_domain[0], row.resources_domain[-1]
    raw = jnp.concatenate(
        (grid, lower[None], upper[None], row.borrowing_limit[None], required)
    )
    active = jnp.concatenate(
        (
            jnp.arange(grid.size) < row.envelope.n_kept,
            jnp.ones((3 + required.size,), dtype=jnp.bool_),
        )
    )
    points, length, retained = _ordered_points(
        raw=raw, active=active & (raw >= lower) & (raw <= upper), capacity=capacity
    )
    initial = ResourcesRegionCatalog(
        points=points,
        length=length,
        smooth=jnp.zeros((capacity - 1,), dtype=jnp.bool_),
        piece=jnp.full((capacity - 1,), -1, dtype=jnp.int32),
        branch=jnp.full((capacity - 1,), -1, dtype=jnp.int32),
        reason=jnp.ones((capacity - 1,), dtype=jnp.int32),
        exact_owner=jnp.full((capacity,), -1, dtype=jnp.int32),
        exact_side=jnp.zeros((capacity,), dtype=jnp.int32),
        exact_corner=jnp.zeros((capacity,), dtype=jnp.bool_),
        exact_feasible=jnp.zeros((capacity,), dtype=jnp.bool_),
        exact_piece=jnp.zeros((capacity,), dtype=jnp.int32),
        exact_row_legal=jnp.zeros((capacity,), dtype=jnp.bool_),
        valid=retained
        & ~row.construction_invalid
        & jnp.all(~active | jnp.isfinite(raw))
        & jnp.all(jnp.isfinite(required))
        & jnp.all((required >= lower) & (required <= upper))
        & jnp.isfinite(scale)
        & (scale > 0.0)
        & jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower < upper),
    )

    def assess(catalog: ResourcesRegionCatalog) -> ResourcesRegionCatalog:
        smooth, piece, branch, reason = jax.vmap(
            lambda left, right: _classify_resource_cell(
                evaluator=evaluator, lower=left, upper=right
            )
        )(catalog.points[:-1], catalog.points[1:])
        live = jnp.arange(capacity - 1) < catalog.length - 1
        return replace(
            catalog,
            smooth=live & smooth,
            piece=piece,
            branch=branch,
            reason=jnp.where(live, reason, 0),
        )

    def step(
        state: tuple[ScalarInt, ResourcesRegionCatalog, ScalarBool],
    ) -> tuple[ScalarInt, ResourcesRegionCatalog, ScalarBool]:
        iteration, catalog, _ = state
        assessed = assess(catalog)
        live = jnp.arange(capacity - 1) < assessed.length - 1
        width = assessed.points[1:] - assessed.points[:-1]
        middle = assessed.points[:-1] + width / 2.0
        candidates = (
            live
            & ~assessed.smooth
            & (assessed.reason != 1)
            & (middle > assessed.points[:-1])
            & (middle < assessed.points[1:])
        )
        # Split one widest unresolved interval per bounded iteration. Existing
        # declared/native cuts are never dropped when capacity is exhausted.
        index = jnp.argmax(jnp.where(candidates, width, -jnp.inf))
        insert = jnp.any(candidates) & (assessed.length < capacity)
        raw = jnp.concatenate((assessed.points, middle[index][None]))
        active = jnp.concatenate((jnp.arange(capacity) < assessed.length, insert[None]))
        points, length, valid = _ordered_points(
            raw=raw, active=active, capacity=capacity
        )
        return (
            iteration + 1,
            replace(
                assessed, points=points, length=length, valid=assessed.valid & valid
            ),
            insert,
        )

    # A fixed numerical capacity bounds locator work and scratch. Exhausted or
    # nonrepresentable neighborhoods retain false links and explicit reasons.
    # A step that inserts nothing is the identity, so stop at the first one.
    _, result, _ = jax.lax.while_loop(
        lambda state: (state[0] < min(capacity, 16)) & state[2],
        step,
        (jnp.int32(0), initial, jnp.ones((), dtype=jnp.bool_)),
    )
    result = assess(result)
    live_cells = jnp.arange(capacity - 1) < result.length - 1
    unresolved = live_cells & ~result.smooth
    middle = result.points[:-1] + (result.points[1:] - result.points[:-1]) / 2.0
    stagnant = unresolved & ~(
        (middle > result.points[:-1]) & (middle < result.points[1:])
    )
    reason = result.reason | jnp.where(stagnant, 16, 0).astype(jnp.int32)
    reason = reason | jnp.where(unresolved & (result.length == capacity), 32, 0).astype(
        jnp.int32
    )
    reason = reason | jnp.where(unresolved, 64, 0).astype(jnp.int32)
    result = replace(result, reason=reason)
    active_points = jnp.arange(capacity) < result.length
    points = jnp.where(active_points, result.points, lower)
    selected, legal, piece = jax.vmap(
        lambda resources: read_resources_candidate_details(
            row=row,
            resources=resources,
            utility_of_action=evaluator.utility_of_action,
            owner_capacity=evaluator.owner_capacity,
        )
    )(points)
    return replace(
        result,
        exact_owner=selected.native_owner,
        exact_side=selected.side_id,
        exact_corner=selected.corner,
        exact_feasible=selected.feasible,
        exact_piece=piece,
        exact_row_legal=legal,
        valid=result.valid & jnp.all(~active_points | selected.valid),
    )


def _classify_resource_cell(
    *,
    evaluator: CurrentResourcesEvaluator,
    lower: ScalarFloat,
    upper: ScalarFloat,
) -> tuple[ScalarBool, ScalarInt, ScalarInt, ScalarInt]:
    """Link a cell whose ends share the piece owner/corner branch, both feasible.

    An ordinary rule: no proof that the cell interior keeps that selection.
    Known discontinuities stay cut by the catalog's native points.
    """
    row = evaluator.row
    grid = row.envelope.grid
    count = row.envelope.n_kept
    search = jnp.where(jnp.arange(grid.size) < count, grid, jnp.inf)
    piece = jnp.clip(
        jnp.searchsorted(search, lower, side="right") - 1, 0, grid.size - 2
    )
    x0, x1 = grid[piece], grid[piece + 1]
    *_, applicable = materialize_hermite_shadow(
        xp_lower=x0,
        xp_upper=x1,
        fp_lower=row.envelope.value[piece],
        fp_upper=row.envelope.value[piece + 1],
        slope_lower=row.marginal[piece],
        slope_upper=row.marginal[piece + 1],
    )
    left = read_resources_candidate(
        row=row,
        resources=lower,
        utility_of_action=evaluator.utility_of_action,
        owner_capacity=evaluator.owner_capacity,
    )
    right = read_resources_candidate(
        row=row,
        resources=upper,
        utility_of_action=evaluator.utility_of_action,
        owner_capacity=evaluator.owner_capacity,
    )
    geometry = (
        jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower < upper)
        & (lower >= x0)
        & (upper <= x1)
        & (piece + 1 < count)
        & applicable
        & row.envelope.link_valid[piece]
        & (row.envelope.owner_id[piece] == row.envelope.owner_id[piece + 1])
        & (lower > row.borrowing_limit)
    )
    # On the unclamped normalized piece savings is a convex blend of S0/S1.
    # Requiring the entire savings range below x0 is sufficient for positive
    # consumption without substituting exact endpoint subtraction for rounded h.
    # Interior legality cuts remain holes; actual rounded legality is rechecked.
    s0, s1 = row.envelope.savings[piece], row.envelope.savings[piece + 1]
    legal = (
        (jnp.minimum(s0, s1) >= row.borrowing_limit)
        & (jnp.maximum(s0, s1) <= row.upper_savings)
        & (x0 > jnp.maximum(s0, s1))
    )
    same = (left.corner == right.corner) & (
        left.corner
        | (
            (left.native_owner == row.envelope.owner_id[piece])
            & (right.native_owner == row.envelope.owner_id[piece])
        )
    )
    branch = left.corner.astype(jnp.int32)
    smooth = (
        geometry
        & legal
        & same
        & left.valid
        & right.valid
        & left.feasible
        & right.feasible
    )
    reason = jnp.where(
        ~geometry,
        1,
        jnp.where(~legal, 2, jnp.where(~same, 4, 0)),
    ).astype(jnp.int32)
    return smooth, piece.astype(jnp.int32), branch, reason


def _matches_catalog(
    *,
    catalog: ResourcesRegionCatalog,
    row: OwnedResourcesRow,
    resources: ScalarFloat,
    owner: ScalarInt,
    side: ScalarInt,
    corner: ScalarBool,
    feasible: ScalarBool,
    legal: ScalarBool,
    native_piece: ScalarInt,
) -> ScalarBool:
    search = jnp.where(
        jnp.arange(catalog.points.size) < catalog.length, catalog.points, jnp.inf
    )
    index = jnp.clip(
        jnp.searchsorted(search, resources, side="right") - 1,
        0,
        catalog.points.size - 2,
    )
    interior = (resources > catalog.points[index]) & (
        resources < catalog.points[index + 1]
    )
    piece = catalog.piece[index]
    expected_owner = jnp.where(
        catalog.branch[index] == 1, -1, row.envelope.owner_id[jnp.maximum(piece, 0)]
    )
    matched = (
        interior
        & catalog.smooth[index]
        & (corner == (catalog.branch[index] == 1))
        & (owner == expected_owner)
        & (side == 0)
        & feasible
        & legal
        & (native_piece == piece)
    )
    exact = (jnp.arange(catalog.points.size) < catalog.length) & (
        catalog.points == resources
    )
    node = jnp.argmax(exact)
    exact_match = (
        jnp.any(exact)
        & (owner == catalog.exact_owner[node])
        & (side == catalog.exact_side[node])
        & (corner == catalog.exact_corner[node])
        & (feasible == catalog.exact_feasible[node])
        & (legal == catalog.exact_row_legal[node])
        & (native_piece == catalog.exact_piece[node])
    )
    return catalog.valid & (matched | exact_match)


def _ordered_points(
    *, raw: Float1D, active: BoolND, capacity: int
) -> tuple[Float1D, ScalarInt, ScalarBool]:
    ordered = jnp.sort(jnp.where(active, raw, jnp.inf))
    unique = jnp.isfinite(ordered) & jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), ordered[1:] != ordered[:-1])
    )
    count = jnp.sum(unique, dtype=jnp.int32)
    slots = jnp.where(unique, jnp.cumsum(unique, dtype=jnp.int32) - 1, capacity)
    points = (
        jnp.full((capacity,), jnp.inf, dtype=raw.dtype)
        .at[slots]
        .set(ordered, mode="drop")
    )
    return points, jnp.minimum(count, capacity), count <= capacity


type _NativePathNode = (
    PytreeChild
    | BalancedProjectedEGMCarry
    | BalancedResourcesGuard
    | ResourcesRegionCatalog
    | OwnedResourcesRow
    | OwnedEnvelope
    | BoundInputStatus
    | QueryInputStatus
    | ContinuationReadCounts
)


def _semantic_native_key(
    *,
    node: _NativePathNode,
    key: (
        jax.tree_util.GetAttrKey
        | jax.tree_util.DictKey
        | jax.tree_util.SequenceKey
        | jax.tree_util.FlattenedIndexKey
    ),
) -> jax.tree_util.GetAttrKey | jax.tree_util.DictKey | jax.tree_util.SequenceKey:
    """Normalize only the actual registered unkeyed immutable mapping schema."""
    if not isinstance(key, jax.tree_util.FlattenedIndexKey):
        return key
    if not isinstance(node, MappingProxyType):
        raise TypeError(
            "Balanced guard flattened key has no registered mapping schema."
        )
    return _mapping_path_key(mapping=node, index=key.key)


def _mapping_path_key(
    *, mapping: MappingProxyType[ReferenceName | int, PytreeChild], index: int
) -> jax.tree_util.DictKey:
    """Recover the actual ordered key of lcm's registered immutable mapping."""
    keys = tuple(mapping)
    if type(index) is not int or not 0 <= index < len(keys):
        raise TypeError("Balanced guard flattened mapping index is invalid.")
    name = keys[index]
    if not isinstance(name, str):
        raise TypeError("Balanced guard mapping lost its actual string-key schema.")
    return jax.tree_util.DictKey(name)


def _path_name(
    *, key: jax.tree_util.GetAttrKey | jax.tree_util.DictKey | jax.tree_util.SequenceKey
) -> str:
    if isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    if isinstance(key, jax.tree_util.DictKey):
        return str(key.key)
    if isinstance(key, jax.tree_util.SequenceKey):
        return str(key.idx)
    raise TypeError("Balanced guard has an unsupported native leaf path.")


type _BalancedChild = PytreeChild | BalancedResourcesGuard
type _GuardChild = PytreeChild | ResourcesRegionCatalog
type _GuardMetadata = tuple[
    tuple[bool, int | None, bool], tuple[ReferenceName, ...], tuple[ShockName, ...]
]
_GUARD_METADATA_FIELDS = ("row_layout", "pool_names", "physical_names")


def _flatten_balanced(
    carry: BalancedProjectedEGMCarry,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _BalancedChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(field.name), getattr(carry, field.name))
        for field in fields(carry)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_balanced(
    metadata: None, leaves: Iterable[_BalancedChild]
) -> BalancedProjectedEGMCarry:
    del metadata
    carry = object.__new__(BalancedProjectedEGMCarry)
    for field, leaf in zip(fields(BalancedProjectedEGMCarry), leaves, strict=True):
        object.__setattr__(carry, field.name, leaf)
    return carry


def _flatten_guard(
    guard: BalancedResourcesGuard,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _GuardChild], ...], _GuardMetadata]:
    return tuple(
        (jax.tree_util.GetAttrKey(field.name), getattr(guard, field.name))
        for field in fields(guard)
        if field.name not in _GUARD_METADATA_FIELDS
    ), (guard.row_layout, guard.pool_names, guard.physical_names)


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_guard(
    metadata: _GuardMetadata, leaves: Iterable[_GuardChild]
) -> BalancedResourcesGuard:
    guard = object.__new__(BalancedResourcesGuard)
    for name, value in zip(_GUARD_METADATA_FIELDS, metadata, strict=True):
        object.__setattr__(guard, name, value)
    dynamic = (
        field
        for field in fields(BalancedResourcesGuard)
        if field.name not in _GUARD_METADATA_FIELDS
    )
    for field, leaf in zip(dynamic, leaves, strict=True):
        object.__setattr__(guard, field.name, leaf)
    return guard


def _flatten_catalog(
    catalog: ResourcesRegionCatalog,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(field.name), getattr(catalog, field.name))
        for field in fields(catalog)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_catalog(
    metadata: None, leaves: Iterable[PytreeChild]
) -> ResourcesRegionCatalog:
    del metadata
    catalog = object.__new__(ResourcesRegionCatalog)
    for field, leaf in zip(fields(ResourcesRegionCatalog), leaves, strict=True):
        object.__setattr__(catalog, field.name, leaf)
    return catalog


jax.tree_util.register_pytree_with_keys(
    ResourcesRegionCatalog, _flatten_catalog, _unflatten_catalog
)
jax.tree_util.register_pytree_with_keys(
    BalancedResourcesGuard, _flatten_guard, _unflatten_guard
)
jax.tree_util.register_pytree_with_keys(
    BalancedProjectedEGMCarry, _flatten_balanced, _unflatten_balanced
)
