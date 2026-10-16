"""Model-bound regular-row queries for scalar EGM borrowing floors."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.continuation import EGMQuerySpec
from _lcm.egm.carry import EGM_ENDOGENOUS_COORDINATE
from _lcm.egm.interp import locate_on_grid
from _lcm.egm.preferences import (
    NEWTON_ACTION_FLOOR,
    Preferences,
    get_discount_factor_reader,
    get_preferences_builder,
    newton_action_ceiling,
)
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution.feasibility import (
    BoundInputStatus,
    ContinuationReadCounts,
    shift_savings_grid,
)
from _lcm.typing import EconFunction, EconFunctionKwargs, EconFunctionsMapping
from _lcm.utils.counts import sum_counts
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.solver_api import ContinuationCapabilities
from lcm.typing import BoolND, Float1D, FloatND, IntND, ReferenceName, ScalarFloat


@dataclass(frozen=True, kw_only=True)
class ScalarEGMQueryModel:
    """Canonical scalar evaluator pool prepared before any numerical kernel build."""

    authority: EGMQuerySpec
    """Independent model descriptor shared by publication and consuming adapters."""
    functions: EconFunctionsMapping
    """Processed functions reused by the canonical solution phase."""
    koopmans_aggregator: EconFunction
    """Processed scalar discounting function reused by the phase."""
    offsets: Float1D
    """One source saving-offset vector, without cell or period expansion."""


class ScalarEGMQueryResult(NamedTuple):
    """Associated scalar-row candidates and mandatory query summaries."""

    value: FloatND
    marginal: FloatND
    savings: FloatND
    coverage_missing: BoolND
    bound_status: BoundInputStatus


class PreparedScalarEGMQuery(NamedTuple):
    """Canonical target inputs shared by dependent reads in one compiled cell."""

    floor: ScalarFloat
    nodes: Float1D
    status: BoundInputStatus
    preferences: Preferences
    beta: ScalarFloat


def build_scalar_egm_query(
    *,
    model: ScalarEGMQueryModel,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
) -> ScalarEGMQuery:
    """Bind an actual-query adapter from the independently prepared model pool."""
    bound = model.authority.bound
    if bound.lower_target is None:
        raise TypeError("A scalar feature query requires the installed floor target.")
    if len(model.authority.interval_ids) != 1:
        raise TypeError("A scalar feature query requires one model-owned interval.")
    lower = concatenate_functions(functions, targets=bound.lower_target)
    return ScalarEGMQuery(
        lower=lower,
        lower_inputs=tuple(inspect.signature(lower).parameters),
        offsets=model.offsets,
        expected_owner_id=model.authority.interval_ids[0],
        expected_side_id=model.authority.interval_side_ids[0],
        build_preferences=get_preferences_builder(
            functions=functions,
            action_name=bound.action,
            action_lower=NEWTON_ACTION_FLOOR,
            action_upper=newton_action_ceiling(model.offsets),
        ),
        read_discount_factor=get_discount_factor_reader(
            functions=functions,
            koopmans_aggregator=koopmans_aggregator,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class ScalarEGMQuery:
    """Read a supported linear row and compare its associated exact lower corner."""

    lower: EconFunction
    """Composed canonical lower target, evaluated once per actual query cell."""
    lower_inputs: tuple[ReferenceName, ...]
    """Actual processed floor leaves, including their parameter namespaces."""
    offsets: Float1D
    """Saving offsets whose represented order remains mandatory at a query."""
    expected_owner_id: int
    """Canonical interval owner supplied by the independent model descriptor."""
    expected_side_id: int
    """Canonical one-sided owner supplied by the independent model descriptor."""
    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    """Target flow utility and marginal bound to actual target inputs."""
    read_discount_factor: Callable[[EconFunctionKwargs], ScalarFloat]
    """Target discounting with the target's canonical parameter namespace."""

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Offer value and resource marginal reads from the installed adapter."""
        return ContinuationCapabilities(
            value=True, marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE})
        )

    def evaluate_floor(
        self,
        *,
        pool: EconFunctionKwargs,
    ) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
        """Check the actual floor and shifted order before safe replacements."""
        floor = self.lower(**{name: pool[name] for name in self.lower_inputs})
        nodes, status = shift_savings_grid(floor=floor, offsets=self.offsets)
        invalid = status.nonfinite_floor | status.invalid_shifted_grid
        return (
            jnp.where(invalid, 0.0, floor),
            jnp.where(invalid, self.offsets, nodes),
            status,
        )

    def __call__(
        self,
        *,
        query: Float1D,
        grid: Float1D,
        value: Float1D,
        marginal: Float1D,
        data: EGMQueryData,
        pool: EconFunctionKwargs,
    ) -> ScalarEGMQueryResult:
        """Keep exact actual domains separate from represented numerical coverage."""
        return self.read_prepared(
            query=query,
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            prepared=self.prepare(pool=pool),
        )

    def prepare(self, *, pool: EconFunctionKwargs) -> PreparedScalarEGMQuery:
        """Evaluate the canonical floor and preferences once for dependent reads."""
        floor, nodes, status = self.evaluate_floor(pool=pool)
        return PreparedScalarEGMQuery(
            floor=floor,
            nodes=nodes,
            status=status,
            preferences=self.build_preferences(pool),
            beta=self.read_discount_factor(pool),
        )

    def read_prepared(
        self,
        *,
        query: Float1D,
        grid: Float1D,
        value: Float1D,
        marginal: Float1D,
        data: EGMQueryData,
        prepared: PreparedScalarEGMQuery,
    ) -> ScalarEGMQueryResult:
        """Read with this cell's already checked canonical target context."""
        result, _ = self._read_prepared_with_summary(
            query=query,
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            prepared=prepared,
            collect_continuation_read_counts=False,
            active=None,
        )
        return result

    def read_prepared_with_counts(
        self,
        *,
        query: Float1D,
        grid: Float1D,
        value: Float1D,
        marginal: Float1D,
        data: EGMQueryData,
        prepared: PreparedScalarEGMQuery,
        active: BoolND | None = None,
    ) -> tuple[ScalarEGMQueryResult, ContinuationReadCounts]:
        """Count paired raw scalar-row attempts before its actual corner recovery."""
        result, counts = self._read_prepared_with_summary(
            query=query,
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            prepared=prepared,
            collect_continuation_read_counts=True,
            active=active,
        )
        if counts is None:
            raise TypeError("A checked scalar read requires its paired population.")
        return result, counts

    def _read_prepared_with_summary(
        self,
        *,
        query: Float1D,
        grid: Float1D,
        value: Float1D,
        marginal: Float1D,
        data: EGMQueryData,
        prepared: PreparedScalarEGMQuery,
        collect_continuation_read_counts: bool,
        active: BoolND | None,
    ) -> tuple[ScalarEGMQueryResult, ContinuationReadCounts | None]:
        """Share one original support/owner calculation across static variants."""
        floor, nodes, status, preferences, beta = prepared
        domain = query > floor
        provenance_matches = data.floor_used == floor
        lower, upper, weight_upper = locate_on_grid(x_query=query, grid=grid)
        weight_lower = 1.0 - weight_upper
        active_lower, active_upper = weight_lower != 0.0, weight_upper != 0.0
        live = data.node_kind != QueryNodeKind.PAD
        owned = (data.owner_id == self.expected_owner_id) & (
            data.side_id == self.expected_side_id
        )
        supported = (
            (~active_lower | (live[lower] & owned[lower]))
            & (~active_upper | (live[upper] & owned[upper]))
            & (~(active_lower & active_upper) | data.link_valid[lower])
        )
        finite = data.node_kind == QueryNodeKind.FINITE
        economic = data.node_kind == QueryNodeKind.ECONOMIC_MINUS_INF
        invalid_node = (
            live
            & owned
            & (
                ~jnp.isfinite(data.savings)
                | ~(finite | economic)
                | (finite & (~jnp.isfinite(value) | ~jnp.isfinite(marginal)))
                | (economic & (~jnp.isneginf(value) | (marginal != 0.0)))
            )
        )
        invalid_row = (active_lower & invalid_node[lower]) | (
            active_upper & invalid_node[upper]
        )
        negative_economic = (economic[lower] & (weight_lower < 0.0)) | (
            economic[upper] & (weight_upper < 0.0)
        )
        positive_economic = (economic[lower] & (weight_lower > 0.0)) | (
            economic[upper] & (weight_upper > 0.0)
        )
        row_savings = _weighted_pair(
            values=jnp.where(live & owned, data.savings, 0.0),
            lower=lower,
            upper=upper,
            weight_upper=weight_upper,
        )
        row_value = _weighted_pair(
            values=jnp.where(finite & owned, value, 0.0),
            lower=lower,
            upper=upper,
            weight_upper=weight_upper,
        )
        row_marginal = _weighted_pair(
            values=jnp.where(finite & owned, marginal, 0.0),
            lower=lower,
            upper=upper,
            weight_upper=weight_upper,
        )
        invalid_row |= (
            supported
            & ~negative_economic
            & (
                ~jnp.isfinite(row_value)
                | ~jnp.isfinite(row_marginal)
                | ~jnp.isfinite(row_savings)
            )
        )
        row_value = jnp.where(positive_economic, -jnp.inf, row_value)
        row_marginal = jnp.where(positive_economic, 0.0, row_marginal)
        row_legal = (
            domain
            & provenance_matches
            & supported
            & ~negative_economic
            & ~invalid_row
            & (row_savings >= floor)
            & (row_savings <= nodes[-1])
            & (query - row_savings > 0.0)
            & (jnp.isfinite(row_value) | jnp.isneginf(row_value))
            & jnp.isfinite(row_marginal)
        )
        if (
            data.corner_continuation is None
            or data.corner_kind is None
            or data.corner_interval_id is None
            or data.corner_side_id is None
        ):
            raise TypeError("A scalar query requires its declared compact corner.")
        owned_corner = (data.corner_interval_id[0] == self.expected_owner_id) & (
            data.corner_side_id[0] == self.expected_side_id
        )
        safe_consumption = jnp.where(domain, query - floor, 1.0)
        corner_continuation = jnp.where(
            owned_corner & (data.corner_kind[0] != QueryNodeKind.PAD),
            data.corner_continuation[0],
            0.0,
        )
        corner_value = (
            preferences.utility(safe_consumption) + beta * corner_continuation
        )
        corner_marginal = preferences.marginal_utility(safe_consumption)
        finite_corner = data.corner_kind[0] == QueryNodeKind.FINITE
        economic_corner = data.corner_kind[0] == QueryNodeKind.ECONOMIC_MINUS_INF
        live_corner = (data.corner_kind[0] != QueryNodeKind.PAD) & owned_corner
        invalid_corner = live_corner & (
            ~(finite_corner | economic_corner)
            | (finite_corner & ~jnp.isfinite(data.corner_continuation[0]))
            | (economic_corner & ~jnp.isneginf(data.corner_continuation[0]))
            | (domain & finite_corner & ~jnp.isfinite(corner_value))
            | (domain & ~jnp.isfinite(corner_marginal))
        )
        corner_finite = (
            domain
            & provenance_matches
            & finite_corner
            & owned_corner
            & ~invalid_corner
            & jnp.isfinite(corner_value)
        )
        take_row = row_legal & (~corner_finite | (row_value >= corner_value))
        represented = row_legal | corner_finite
        result = ScalarEGMQueryResult(
            value=jnp.where(
                domain, jnp.where(take_row, row_value, corner_value), -jnp.inf
            ),
            marginal=jnp.where(
                domain & represented,
                jnp.where(take_row, row_marginal, corner_marginal),
                0.0,
            ),
            savings=jnp.where(take_row, row_savings, floor),
            coverage_missing=~provenance_matches
            | invalid_row
            | invalid_corner
            | (domain & ~represented),
            bound_status=status,
        )
        counts = None
        if collect_continuation_read_counts:
            supported_node = live & owned & ~invalid_node & jnp.isfinite(grid)
            first = grid[jnp.argmax(supported_node)]
            activity = (
                jnp.ones(query.shape, dtype=jnp.bool_) if active is None else active
            )
            eligible = activity & jnp.any(supported_node)
            counts = ContinuationReadCounts(
                below_first_live=sum_counts(counts=eligible & (query < first)),
                eligible=sum_counts(counts=eligible),
            )
        return result, counts


def _weighted_pair(
    *, values: Float1D, lower: IntND, upper: IntND, weight_upper: FloatND
) -> FloatND:
    """Read one shared bracket without letting a zero-weight endpoint contribute."""
    return zero_safe_weighted_term(
        weight=1.0 - weight_upper,
        value=values[lower],
        subnormal_is_accounted_for=True,
    ) + zero_safe_weighted_term(
        weight=weight_upper,
        value=values[upper],
        subnormal_is_accounted_for=True,
    )
