"""Read native liquid NB tuples and matched held-interval borrowing corners."""

import functools
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple, overload

import jax
import jax.numpy as jnp

from _lcm.egm.carry import OwnedRowRead, _read_owned_row_with_support, read_owned_row
from _lcm.egm.dcegm_query import (
    FiniteNEGMQueryResult,
    _blend_owned_rows,
    _BlendedRow,
    _evaluate,
    _passive_weights,
    _read_owned_derivative,
    _read_owned_derivative_with_support,
    _weighted_sum,
)
from _lcm.egm.ez_kernel import ez_marginal_of_resource, ez_period_value
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.nbegm_query_model import NBEGMIntervalRecipe, NBEGMQueryModel
from _lcm.egm.preferences import (
    NEWTON_ACTION_FLOOR,
    Preferences,
    concatenate_regime_function,
    get_discount_factor_reader,
    get_preferences_builder,
    newton_action_ceiling,
)
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind, select_query_data
from _lcm.grids.discrete import read_at_codes
from _lcm.solution.feasibility import (
    BoundInputStatus,
    ContinuationReadCounts,
    shift_savings_grid,
)
from _lcm.typing import EconFunction, EconFunctionKwargs, EconFunctionsMapping
from _lcm.utils.counts import sum_counts
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.solver_api import EGM_ENDOGENOUS_COORDINATE, ContinuationCapabilities
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ReferenceName,
    ScalarFloat,
    ScalarInt,
)


class NBEGMQueryResult(NamedTuple):
    """One associated conditional NB tuple and its mandatory query integrity."""

    value: FloatND
    marginal: FloatND
    savings: FloatND
    coverage_missing: BoolND
    bound_status: BoundInputStatus


class AdaptiveNBEGMQueryResult(NamedTuple):
    """Conditional adaptive values after nodewise outer folding and passive blending."""

    value: FloatND
    marginal: FloatND
    coverage_missing: BoolND
    bound_status: BoundInputStatus


type _NBChoiceResult = (
    NBEGMQueryResult | AdaptiveNBEGMQueryResult | FiniteNEGMQueryResult
)


class _NBCornerRead(NamedTuple):
    """Matched held-interval continuation on an exact signed passive stencil."""

    value: FloatND
    supported: BoolND
    invalid_numeric: BoolND


def _validate_nbegm_query_recipe(*, model: NBEGMQueryModel) -> None:
    """Require the declared native interval topology and compact corner ownership."""
    authority = model.authority
    topology = authority.row_topology
    if (
        authority.coordinate_kind != "liquid"
        or topology is None
        or topology.kind != "nb_budget_interval_run"
    ):
        raise TypeError(
            "An NB query requires its declared native-liquid budget-run topology."
        )
    if authority.corner_coordinate_kind != "compact_interval":
        raise TypeError(
            "Interval-held NB reads require their compact corner declaration."
        )
    if len(model.intervals.source_identity) + 1 != len(authority.interval_ids):
        raise TypeError(
            "NB interval capacity differs from its canonical source recipe."
        )
    if model.intervals.source_identity != authority.interval_source_identity:
        raise TypeError(
            "NB interval sources differ from independently declared model authority."
        )
    if len(model.intervals.equal_above) != len(model.intervals.source_identity):
        raise TypeError("NB source equality ownership differs from its declaration.")


def build_nbegm_query(
    *,
    model: NBEGMQueryModel,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
    target_period: int,
) -> NBEGMQuery:
    """Bind an independently declared liquid NB adapter, including its source recipe."""
    authority = model.authority
    if (
        target_period not in authority.target_periods
        or target_period not in model.intervals_by_period
    ):
        raise TypeError(
            "An NB query requires its canonical target-period interval recipe."
        )
    recipe = model.intervals_by_period[target_period]
    if recipe.source_identity != model.intervals.source_identity:
        raise TypeError(
            "An NB target-period recipe changes its declared source identities."
        )
    model = replace(
        model,
        intervals=recipe,
        functions=functions,
        koopmans_aggregator=koopmans_aggregator,
    )
    _validate_nbegm_query_recipe(model=model)
    bound = authority.bound
    if bound.lower_target is None:
        lower = None
    else:
        lower = concatenate_regime_function(
            functions=functions, target=bound.lower_target
        )
    query = NBEGMQuery(
        model=model,
        lower=lower,
        resources=concatenate_regime_function(
            functions=functions, target=bound.resources_target
        ),
        predicates=tuple(
            concatenate_regime_function(functions=functions, target=target)
            for purpose, target in authority.evaluator_targets
            if purpose.startswith("predicate:")
            and target not in bound.declaration_names
        ),
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

    models = (
        model.adaptive_inner_models
        if model.adaptive_inner_models is not None
        else model.finite_inner_models
    )
    if models is not None:
        bindings = []
        for role in ("keeper", "adjuster"):
            inner_model = models[role]
            if inner_model.functions_by_period is None:
                raise TypeError(
                    "A nested role lacks its canonical period function pool."
                )
            inner_functions = inner_model.functions_by_period[target_period]
            bindings.append(
                build_nbegm_query(
                    model=inner_model,
                    functions=inner_functions,
                    koopmans_aggregator=inner_functions["koopmans_aggregator"],
                    target_period=target_period,
                )
            )
        query = (
            replace(query, adaptive_queries=(bindings[0], bindings[1]))
            if model.adaptive_inner_models is not None
            else replace(query, finite_queries=(bindings[0], bindings[1]))
        )
    return query


@dataclass(frozen=True, kw_only=True)
class NBEGMQuery:
    """Select native NB rows and actual matched corners before discrete aggregation."""

    model: NBEGMQueryModel
    """Independent canonical target authority and interval preimages."""
    lower: Callable[..., FloatND] | None
    """The canonical actual fixed-cell lower bound."""
    resources: Callable[..., FloatND]
    """Actual target resources, with the current discrete action bound."""
    predicates: tuple[Callable[..., BoolND], ...]
    """Actual current-state/discrete predicates, applied after tuple recovery."""
    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    """Bind actual target utility or the existing positive EZ flow."""
    read_discount_factor: Callable[[EconFunctionKwargs], ScalarFloat]
    """Bind actual target discounting."""
    adaptive_queries: tuple[NBEGMQuery, NBEGMQuery] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Actual target-period keeper and adjuster readers with independent derived
    authority.
    """
    finite_queries: tuple[NBEGMQuery, NBEGMQuery] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Actual finite keeper and adjuster readers with their own interval recipes."""

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Offer value and native liquid marginal through this installed reader."""
        return ContinuationCapabilities(
            value=True, marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE})
        )

    @property
    def input_names(self) -> frozenset[ReferenceName]:
        """Include every interval/floor/predicate dependency in branch grouping."""
        bound = self.model.authority.bound
        internal = {bound.action, bound.post_decision, "utility", "CE"}
        adaptive = self.model.authority.adaptive_outer
        if adaptive is not None:
            if self.adaptive_queries is None:
                raise TypeError(
                    "Adaptive query inputs require their actual role bindings."
                )
            names = (
                set()
                if self.lower is None
                else set(inspect.signature(self.lower).parameters)
            )
            for reader in self.adaptive_queries:
                names.update(reader.input_names)
            for target in (adaptive.keeper_target, adaptive.fixed_cost_target):
                if target is not None:
                    function = concatenate_regime_function(
                        functions=self.model.functions, target=target
                    )
                    names.update(inspect.signature(function).parameters)
            names.discard(adaptive.outer_post_decision)
            return frozenset(names - internal)
        finite = self.model.authority.finite_outer
        if finite is not None:
            if self.finite_queries is None:
                raise TypeError(
                    "Finite query inputs require their actual role bindings."
                )
            names = (
                set()
                if self.lower is None
                else set(inspect.signature(self.lower).parameters)
            )
            for reader in self.finite_queries:
                names.update(reader.input_names)
            if finite.keeper_target is None:
                names.add(finite.outer_state)
            else:
                keeper = concatenate_regime_function(
                    functions=self.model.functions, target=finite.keeper_target
                )
                names.update(inspect.signature(keeper).parameters)
            return frozenset(
                names - internal - {finite.outer_action, finite.outer_post_decision}
            )
        return (
            frozenset(
                name
                for _, names in self.model.authority.evaluator_inputs
                for name in names
                if name not in internal
            )
            | self.model.intervals.input_names
        )

    def evaluate_floor(
        self, *, pool: EconFunctionKwargs
    ) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
        """Evaluate the actual cell's bound exactly once before current eligibility."""
        if self.lower is None:
            literal = self.model.authority.bound.literal_value
            nodes = self.model.literal_nodes
            if literal is None or nodes is None:
                raise TypeError(
                    "A literal NB query requires its original physical grid."
                )
            floor = jnp.asarray(literal, dtype=nodes.dtype)
            status = BoundInputStatus(
                nonfinite_floor=~jnp.isfinite(floor),
                invalid_shifted_grid=(
                    ~jnp.all(jnp.isfinite(nodes)) | jnp.any(nodes[1:] <= nodes[:-1])
                ),
            )
            return floor, nodes, status
        floor = _evaluate(function=self.lower, pool=pool)
        nodes, status = shift_savings_grid(floor=floor, offsets=self.model.offsets)
        invalid = status.nonfinite_floor | status.invalid_shifted_grid
        return (
            jnp.where(invalid, 0.0, floor),
            jnp.where(invalid, self.model.offsets, nodes),
            status,
        )

    @overload
    def read_choices(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        action_names: tuple[ActionName, ...],
        action_values: tuple[FloatND | IntND, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        paired_marginal_read: bool = False,
        collect_continuation_read_counts: Literal[False] = False,
    ) -> _NBChoiceResult: ...

    @overload
    def read_choices(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        action_names: tuple[ActionName, ...],
        action_values: tuple[FloatND | IntND, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        paired_marginal_read: bool = False,
        collect_continuation_read_counts: Literal[True],
    ) -> tuple[_NBChoiceResult, ContinuationReadCounts]: ...

    @overload
    def read_choices(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        action_names: tuple[ActionName, ...],
        action_values: tuple[FloatND | IntND, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        paired_marginal_read: bool = False,
        collect_continuation_read_counts: bool,
    ) -> _NBChoiceResult | tuple[_NBChoiceResult, ContinuationReadCounts]: ...

    def read_choices(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        action_names: tuple[ActionName, ...],
        action_values: tuple[FloatND | IntND, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        paired_marginal_read: bool = False,
        collect_continuation_read_counts: bool = False,
    ) -> _NBChoiceResult | tuple[_NBChoiceResult, ContinuationReadCounts]:
        """Read conditional NB rows with one floor and optional original-row counts."""
        if self.model.authority.adaptive_outer is not None:
            from _lcm.egm.adaptive_nbegm_query import (  # noqa: PLC0415
                read_adaptive_choices,
            )

            return read_adaptive_choices(
                query=self,
                grid=grid,
                value=value,
                marginal=marginal,
                data=data,
                row_queries=row_queries,
                row_gradients=row_gradients,
                passive_values=passive_values,
                passive_grids=passive_grids,
                action_names=action_names,
                action_values=action_values,
                pool=pool,
                euler_gradient=euler_gradient,
                paired_marginal_read=paired_marginal_read,
                collect_continuation_read_counts=collect_continuation_read_counts,
            )
        if self.model.authority.finite_outer is not None:
            from _lcm.egm.finite_nbegm_query import (  # noqa: PLC0415
                read_finite_nbegm_choices,
            )

            return read_finite_nbegm_choices(
                query=self,
                grid=grid,
                value=value,
                marginal=marginal,
                data=data,
                row_queries=row_queries,
                row_gradients=row_gradients,
                passive_values=passive_values,
                passive_grids=passive_grids,
                action_names=action_names,
                action_values=action_values,
                pool=pool,
                euler_gradient=euler_gradient,
                paired_marginal_read=paired_marginal_read,
                collect_continuation_read_counts=collect_continuation_read_counts,
            )
        if action_names != self.model.authority.action_names:
            raise TypeError("NB conditional actions differ from canonical authority.")
        floor_state = self.evaluate_floor(pool=pool)
        action_shape = tuple(values.size for values in action_values)

        def read_one(
            index: ScalarInt,
        ) -> tuple[NBEGMQueryResult, ContinuationReadCounts | None]:
            action_index = (
                jnp.unravel_index(index, action_shape) if action_shape else ()
            )
            positions = (slice(None),) * len(passive_grids) + action_index
            actual = {
                **pool,
                **{
                    name: values[position]
                    for name, values, position in zip(
                        action_names, action_values, action_index, strict=True
                    )
                },
            }
            return self._read_at_floor_with_summary(
                grid=read_at_codes(array=grid, positions=positions),
                value=read_at_codes(array=value, positions=positions),
                marginal=read_at_codes(array=marginal, positions=positions),
                data=select_query_data(
                    data=data, positions=positions, floor_positions=()
                ),
                row_queries=read_at_codes(array=row_queries, positions=positions),
                row_gradients=read_at_codes(array=row_gradients, positions=positions),
                passive_values=passive_values,
                passive_grids=passive_grids,
                pool=actual,
                euler_gradient=euler_gradient,
                floor_state=floor_state,
                paired_marginal_read=paired_marginal_read,
                read_rows=self._read_rows_with_counts
                if collect_continuation_read_counts
                else self._read_rows_without_counts,
            )

        result, counts = jax.vmap(read_one)(
            jnp.arange(math.prod(action_shape), dtype=jnp.int32)
        )
        if collect_continuation_read_counts:
            if counts is None:
                raise TypeError("Checked choices require their original row summary.")
            return result, ContinuationReadCounts(
                *(sum_counts(counts=count) for count in counts)
            )
        return result

    def read_prepared(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus],
        paired_marginal_read: bool = False,
    ) -> NBEGMQueryResult:
        """Blend whole native liquid tuples and recover an actual legal corner."""
        result, _ = self._read_at_floor_with_summary(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=row_queries,
            row_gradients=row_gradients,
            passive_values=passive_values,
            passive_grids=passive_grids,
            pool=pool,
            euler_gradient=euler_gradient,
            floor_state=floor_state,
            paired_marginal_read=paired_marginal_read,
            read_rows=self._read_rows_without_counts,
        )
        return result

    def read_prepared_with_counts(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus],
        paired_marginal_read: bool = False,
    ) -> tuple[NBEGMQueryResult, ContinuationReadCounts]:
        """Count original native NB rows before corner recovery or current masking."""
        return self._read_at_floor_with_summary(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=row_queries,
            row_gradients=row_gradients,
            passive_values=passive_values,
            passive_grids=passive_grids,
            pool=pool,
            euler_gradient=euler_gradient,
            floor_state=floor_state,
            paired_marginal_read=paired_marginal_read,
            read_rows=self._read_rows_with_counts,
        )

    def _read_at_floor_with_summary[Summary: ContinuationReadCounts | None](
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus],
        paired_marginal_read: bool,
        read_rows: Callable[..., tuple[_BlendedRow, Summary]],
    ) -> tuple[NBEGMQueryResult, Summary]:
        """Share the actual NB recovery with one statically selected owned-row read."""
        topology = self.model.authority.row_topology
        if topology is None or topology.kind != "nb_budget_interval_run":
            raise TypeError("An NB read requires its independent budget-run topology.")
        floor, nodes, bound_status = floor_state
        bound = self.model.authority.bound
        resources = _evaluate(function=self.resources, pool=pool)
        eligible = jnp.asarray(a=True)
        for predicate in self.predicates:
            eligible &= _evaluate(function=predicate, pool=pool)
        domain = resources > floor
        weights, active, negative = _passive_weights(
            values=passive_values, grids=passive_grids
        )
        weights = tuple(weight.reshape(-1) for weight in weights)
        active, negative = active.reshape(-1), negative.reshape(-1)
        row, summary = read_rows(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=row_queries,
            row_gradients=row_gradients,
            weights=weights,
            active=active,
            negative=negative,
            paired_marginal_read=paired_marginal_read,
        )
        row_legal = (
            row.supported
            & ~row.invalid_numeric
            & (row.savings >= floor)
            & (row.savings <= nodes[-1])
            & (resources > row.savings)
        )
        corner = _read_nb_compact_corner(
            recipe=self.model.intervals,
            data=data,
            pool=pool,
            passive_names=self.model.passive_state_names,
            passive_grids=passive_grids,
            weights=weights,
            active=active,
            negative=negative,
        )
        preferences = self.build_preferences(pool)
        beta = self.read_discount_factor(pool)
        consumption = jnp.where(domain & eligible, resources - floor, 1.0)

        def resources_at(liquid: ScalarFloat) -> ScalarFloat:
            return _evaluate(
                function=self.resources, pool={**pool, bound.liquid_state: liquid}
            )

        resource_slope = jax.grad(resources_at)(pool[bound.liquid_state])
        if self.model.inverse_eis_name is None:
            corner_value = preferences.utility(consumption) + zero_safe_weighted_term(
                weight=beta, value=corner.value, subnormal_is_accounted_for=True
            )
            corner_native_marginal = (
                preferences.marginal_utility(consumption) * resource_slope
            )
        else:
            elasticity = pool[self.model.inverse_eis_name]
            if not isinstance(elasticity, jax.Array | float):
                raise TypeError("NB inverse elasticity must be a numeric leaf.")
            inverse_eis = 1.0 / elasticity
            flow = preferences.utility(consumption)
            corner_value = ez_period_value(
                flow=flow,
                nu=corner.value,
                discount_factor=beta,
                inverse_eis=inverse_eis,
            )
            corner_native_marginal = (
                ez_marginal_of_resource(
                    log_flow_marginal=jnp.log(preferences.marginal_utility(consumption))
                    - inverse_eis * jnp.log(flow),
                    value=corner_value,
                    discount_factor=beta,
                    inverse_eis=inverse_eis,
                )
                * resource_slope
            )
        economic = jnp.isneginf(corner_value) & (self.model.inverse_eis_name is None)
        corner_marginal = jnp.where(
            economic, 0.0, corner_native_marginal * euler_gradient
        )
        invalid_corner = corner.invalid_numeric | (
            eligible
            & domain
            & corner.supported
            & (
                (~jnp.isfinite(corner_value) & ~economic)
                | ~jnp.isfinite(corner_marginal)
            )
        )
        corner_legal = (
            corner.supported & ~invalid_corner & (jnp.isfinite(corner_value) | economic)
        )
        take_row = row_legal & (~corner_legal | (row.value >= corner_value))
        represented = row_legal | corner_legal
        return NBEGMQueryResult(
            value=jnp.where(
                eligible & domain & represented,
                jnp.where(take_row, row.value, corner_value),
                -jnp.inf,
            ),
            marginal=jnp.where(
                eligible & domain & represented,
                jnp.where(take_row, row.marginal, corner_marginal),
                0.0,
            ),
            savings=jnp.where(take_row, row.savings, floor),
            coverage_missing=row.invalid_numeric
            | invalid_corner
            | (eligible & domain & ~represented),
            bound_status=bound_status,
        ), summary

    def _read_rows_without_counts(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        weights: tuple[Float1D, ...],
        active: BoolND,
        negative: BoolND,
        paired_marginal_read: bool,
    ) -> tuple[_BlendedRow, None]:
        """Keep ordinary row reads free of optional count work and output leaves."""
        return self._read_rows(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=row_queries,
            row_gradients=row_gradients,
            weights=weights,
            active=active,
            negative=negative,
            paired_marginal_read=paired_marginal_read,
        ), None

    def _read_rows(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        weights: tuple[Float1D, ...],
        active: BoolND,
        negative: BoolND,
        paired_marginal_read: bool,
    ) -> _BlendedRow:
        """Keep the owned tuple and every active stencil status associated."""
        topology = self.model.authority.row_topology
        if topology is None or topology.kind != "nb_budget_interval_run":
            raise TypeError("An NB read requires its independent budget-run topology.")
        capacity = grid.shape[-1]
        n_rows = row_queries.size
        rows = jax.tree.map(
            lambda leaf: leaf.reshape(n_rows, *leaf.shape[row_queries.ndim :]), data
        )
        read = _read_owned_derivative if paired_marginal_read else read_owned_row
        if not paired_marginal_read and not self.model.authority.state_names:
            read = _read_scalar_nbegm_owned_row
        result = jax.vmap(
            functools.partial(
                read,
                owner_capacity=topology.owner_capacity,
                side_is_row_side=True,
            )
        )(
            query=row_queries.reshape(n_rows),
            grid=grid.reshape(n_rows, capacity),
            value=value.reshape(n_rows, capacity),
            marginal=marginal.reshape(n_rows, capacity),
            data=rows,
        )
        return _blend_owned_rows(
            result=result,
            row_gradients=row_gradients,
            weights=weights,
            active=active,
            negative=negative,
        )

    def _read_rows_with_counts(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        weights: tuple[Float1D, ...],
        active: BoolND,
        negative: BoolND,
        paired_marginal_read: bool,
    ) -> tuple[_BlendedRow, ContinuationReadCounts]:
        """Reduce one event per active original row using its validated node mask."""
        topology = self.model.authority.row_topology
        if topology is None or topology.kind != "nb_budget_interval_run":
            raise TypeError("An NB read requires its independent budget-run topology.")
        capacity = grid.shape[-1]
        n_rows = row_queries.size
        rows = jax.tree.map(
            lambda leaf: leaf.reshape(n_rows, *leaf.shape[row_queries.ndim :]), data
        )
        read = (
            _read_owned_derivative_with_support
            if paired_marginal_read
            else _read_owned_row_with_support
        )
        if not paired_marginal_read and not self.model.authority.state_names:
            read = _read_scalar_nbegm_owned_row_with_support
        result, supported_node = jax.vmap(
            functools.partial(
                read,
                owner_capacity=topology.owner_capacity,
                side_is_row_side=True,
            )
        )(
            query=row_queries.reshape(n_rows),
            grid=grid.reshape(n_rows, capacity),
            value=value.reshape(n_rows, capacity),
            marginal=marginal.reshape(n_rows, capacity),
            data=rows,
        )
        has_first = jnp.any(supported_node, axis=-1)
        first_index = jnp.argmax(supported_node, axis=-1)
        first = jnp.take_along_axis(
            grid.reshape(n_rows, capacity), first_index[:, None], axis=-1
        )[:, 0]
        eligible = active & has_first
        counts = ContinuationReadCounts(
            below_first_live=sum_counts(
                counts=eligible & (row_queries.reshape(n_rows) < first)
            ),
            eligible=sum_counts(counts=eligible),
        )
        return _blend_owned_rows(
            result=result,
            row_gradients=row_gradients,
            weights=weights,
            active=active,
            negative=negative,
        ), counts


def _read_scalar_nbegm_owned_row(
    *,
    query: ScalarFloat,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    owner_capacity: int,
    side_is_row_side: bool = True,
) -> OwnedRowRead:
    """Preserve scalar NB linear values and their authenticated ordinary tuple."""
    result = read_owned_row(
        query=query,
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        owner_capacity=owner_capacity,
        side_is_row_side=side_is_row_side,
    )
    return _with_scalar_nbegm_linear_value(
        result=result,
        query=query,
        grid=grid,
        value=value,
    )


def _read_scalar_nbegm_owned_row_with_support(
    *,
    query: ScalarFloat,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    owner_capacity: int,
    side_is_row_side: bool = True,
) -> tuple[OwnedRowRead, BoolND]:
    """Keep the original counted-row mask beside the scalar linear tuple."""
    result, supported_node = _read_owned_row_with_support(
        query=query,
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        owner_capacity=owner_capacity,
        side_is_row_side=side_is_row_side,
    )
    return _with_scalar_nbegm_linear_value(
        result=result,
        query=query,
        grid=grid,
        value=value,
    ), supported_node


def _with_scalar_nbegm_linear_value(
    *,
    result: OwnedRowRead,
    query: ScalarFloat,
    grid: Float1D,
    value: Float1D,
) -> OwnedRowRead:
    """Use the scalar route's linear ordinate without changing owned support.

    Both readers locate the same right-continuous bracket. At exact nodes the
    selected ordinate is independent of the adjacent piece's derivative side;
    the padded reader preserves exact finite records beside absent neighbors.
    """
    linear_value = interp_on_padded_grid(x_query=query, xp=grid, fp=value)
    return result._replace(
        value=jnp.where(result.supported, linear_value, result.value)
    )


def _read_nb_compact_corner_rows(
    *,
    recipe: NBEGMIntervalRecipe,
    data: EGMQueryData,
    pool: EconFunctionKwargs,
    passive_names: tuple[ReferenceName, ...],
    passive_grids: tuple[Float1D, ...],
) -> tuple[FloatND, BoolND, BoolND, BoolND, BoolND]:
    """Match economic source-side vectors, never sorted-slot ordinals across cells."""
    if (
        data.corner_continuation is None
        or data.corner_kind is None
        or data.corner_interval_id is None
        or data.corner_side_id is None
    ):
        raise TypeError("An NB query requires its declared compact interval bank.")
    if len(passive_names) != len(passive_grids):
        raise TypeError(
            "NB passive names and stencil grids differ from canonical authority."
        )
    actual_preimages, _actual_increasing = recipe.evaluate_preimages(pool)
    actual_above = recipe.evaluate_sides(pool)
    n_intervals = len(recipe.source_identity) + 1
    shape = tuple(grid.size for grid in passive_grids)
    mesh = jnp.meshgrid(*passive_grids, indexing="ij") if passive_grids else ()
    coordinates = tuple(axis.reshape(-1) for axis in mesh)
    n_rows = math.prod(shape)
    values = data.corner_continuation.reshape(n_rows, n_intervals)
    kinds = data.corner_kind.reshape(n_rows, n_intervals)
    ids = data.corner_interval_id.reshape(n_rows, n_intervals)
    sides = data.corner_side_id.reshape(n_rows, n_intervals)

    def read_one(index: ScalarInt) -> tuple[ScalarFloat, BoolND, BoolND, BoolND]:
        row_pool = {
            **pool,
            **{
                name: values[index]
                for name, values in zip(passive_names, coordinates, strict=True)
            },
        }
        preimages, increasing = recipe.evaluate_preimages(row_pool)
        order = jnp.argsort(preimages, stable=True)
        ranks = jnp.argsort(order, stable=True)
        assignments = (jnp.arange(n_intervals)[:, None] > ranks[None, :]) == increasing[
            None, :
        ]
        sorted_preimages = preimages[order]
        lower = jnp.concatenate(
            (jnp.asarray([-jnp.inf], dtype=preimages.dtype), sorted_preimages)
        )
        upper = jnp.concatenate(
            (sorted_preimages, jnp.asarray([jnp.inf], dtype=preimages.dtype))
        )
        matches = jnp.all(assignments == actual_above[None, :], axis=-1) & (
            lower < upper
        )
        match_count = jnp.sum(matches)
        position = jnp.argmax(matches)
        kind = kinds[index, position]
        selected = values[index, position]
        labelled = (ids[index, position] == position) & (sides[index, position] == 0)
        finite = kind == QueryNodeKind.FINITE
        economic = kind == QueryNodeKind.ECONOMIC_MINUS_INF
        known = finite | economic
        supported = (match_count == 1) & labelled & known
        invalid = (
            ~jnp.all(jnp.isfinite(preimages))
            | (match_count > 1)
            | (
                (match_count == 1)
                & (kind != QueryNodeKind.PAD)
                & (
                    ~known
                    | ~labelled
                    | (finite & ~jnp.isfinite(selected))
                    | (economic & ~jnp.isneginf(selected))
                )
            )
        )
        return (
            jnp.where(finite & supported, selected, 0.0),
            supported,
            economic & supported,
            invalid,
        )

    selected, supported, economic, invalid = jax.vmap(read_one)(
        jnp.arange(n_rows, dtype=jnp.int32)
    )
    return (
        selected,
        supported,
        economic,
        invalid,
        ~jnp.all(jnp.isfinite(actual_preimages)),
    )


def _read_nb_compact_corner(
    *,
    recipe: NBEGMIntervalRecipe,
    data: EGMQueryData,
    pool: EconFunctionKwargs,
    passive_names: tuple[ReferenceName, ...],
    passive_grids: tuple[Float1D, ...],
    weights: tuple[Float1D, ...],
    active: BoolND,
    negative: BoolND,
) -> _NBCornerRead:
    """Match held interval rows before the canonical signed passive reduction."""
    selected, supported, economic, invalid, invalid_preimages = (
        _read_nb_compact_corner_rows(
            recipe=recipe,
            data=data,
            pool=pool,
            passive_names=passive_names,
            passive_grids=passive_grids,
        )
    )
    negative_economic = jnp.any(active & negative & economic)
    positive_economic = jnp.any(active & ~negative & economic)
    has_support = jnp.all(~active | supported) & ~negative_economic
    blended = _weighted_sum(weights=weights, values=selected)
    return _NBCornerRead(
        value=jnp.where(positive_economic, -jnp.inf, blended),
        supported=has_support,
        invalid_numeric=invalid_preimages
        | jnp.any(active & invalid)
        | (has_support & ~positive_economic & ~jnp.isfinite(blended)),
    )
