"""Genuine finite NNBEGM choices over complete original NB query-row banks."""

import functools
import inspect
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm import continuation as _nn_continuation
from _lcm.egm import nbegm_breakpoints as _nn_breakpoints
from _lcm.egm import nbegm_step as _nn_step
from _lcm.egm import preferences as _nn_preferences
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_balanced import _ordered_points
from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.egm.finite_nbegm_query import _read_finite_nb_owned_corners
from _lcm.egm.finite_negm_query import (
    _collapse_finite_selection,
    _FiniteOuterProfile,
    _read_finite_native_row,
    _select_finite_outer_profiles,
    _single_row_finite_corner_read,
    _validate_finite_outer_rows,
)
from _lcm.egm.nbegm_query_model import NBEGMIntervalRecipe
from _lcm.egm.nbegm_step import _OwnedNBEGMQueryStep
from _lcm.egm.outer_inversion import DeclaredOuterInverse, invert_declared_outer_target
from _lcm.egm.preferences import Preferences, preferences_from_utility
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution import nbegm as _nn_inner_module
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.feasibility import (
    BoundedResult,
    QueryInputStatus,
    merge_bound_input_status,
    merge_continuation_read_counts,
    merge_query_input_status,
)
from _lcm.solution.nbegm import (
    _BoundScalarFunction,
    _NBEGMBranchSolver,
    _NBEGMCellBinder,
    _NBEGMCellContinuation,
    _NBEGMCellSolver,
    _NBEGMQueryIntervalEvaluators,
    _NBEGMTiledCore,
    _RideAlongNBEGMPeriodKernel,
)
from _lcm.typing import EconFunctionKwargs
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import (
    ActionName,
    Float1D,
    FloatND,
    IntND,
    RegimeName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    StateName,
)

if TYPE_CHECKING:
    from _lcm.egm.continuation import ContinuationPlan
    from _lcm.solution.nnbegm import _FiniteNNBEGMPeriodKernel
else:
    # Original numerical modules are initialized before these annotation types.
    type ContinuationPlan = object
    # The original NN owner imports the numerical engine before its class exists.
    type _FiniteNNBEGMPeriodKernel = object

_ORIGINAL_PREIMAGES = _NBEGMQueryIntervalEvaluators.preimages
_ORIGINAL_SIDES = _NBEGMQueryIntervalEvaluators.sides
_ORIGINAL_PREIMAGES_CODE = _ORIGINAL_PREIMAGES.__code__
_ORIGINAL_SIDES_CODE = _ORIGINAL_SIDES.__code__
_ORIGINAL_RAW_PREIMAGES = inspect.unwrap(_ORIGINAL_PREIMAGES)
_ORIGINAL_RAW_SIDES = inspect.unwrap(_ORIGINAL_SIDES)
_ORIGINAL_RAW_PREIMAGES_CODE = _ORIGINAL_RAW_PREIMAGES.__code__
_ORIGINAL_RAW_SIDES_CODE = _ORIGINAL_RAW_SIDES.__code__


class NNBEGMOwnedRow(NamedTuple):
    """Original complete sampled row, with native winners distinct from runs."""

    grid: Float1D
    row: _OwnedNBEGMQueryStep
    requested_index: ScalarInt
    support_valid: ScalarBool


@dataclass(frozen=True, kw_only=True, eq=False)
class NNBEGMInnerSource:
    """Actual original inner owner, numerical core and declaration recipe."""

    kernel: _RideAlongNBEGMPeriodKernel
    core: _NBEGMTiledCore
    recipe: NBEGMIntervalRecipe
    borrowing_limit: ScalarFloat

    @property
    def savings_nodes(self) -> Float1D:
        return self.core.envelope_build.bind_cell.savings_grid

    @property
    def continuation_plan(self) -> ContinuationPlan:
        return self.core.continuation_plan

    def has_original_linear_plan(self, *, context: SolverBuildContext) -> bool:
        """Match the original NB plan's unset linear CE to the declared source."""
        declared = context.certainty_equivalent
        plan = self.continuation_plan
        return (
            (declared is None or type(declared) is LinearExpectation)
            and plan.certainty_equivalent is None
            and plan.risk_aversion_param_name is None
        )

    def __post_init__(self) -> None:
        if (
            type(self.kernel) is not _RideAlongNBEGMPeriodKernel
            or self.kernel.core_programs()["main"].function is not self.core
            or type(self.core) is not _NBEGMTiledCore
        ):
            raise TypeError("NNBEGM inner source lost its original numerical owner.")
        binder = self.core.envelope_build.bind_cell
        preimages = self.recipe.evaluate_preimages
        sides = self.recipe.evaluate_sides
        if (
            not inspect.ismethod(preimages)
            or not inspect.ismethod(sides)
            or type(preimages.__self__) is not _NBEGMQueryIntervalEvaluators
            or sides.__self__ is not preimages.__self__
            or preimages.__self__.sources is not self.core.schedule_spec.sources
            or preimages.__self__.liquid_name != self.core.statics.liquid_name
            or preimages.__func__ is not _ORIGINAL_PREIMAGES
            or sides.__func__ is not _ORIGINAL_SIDES
            or _ORIGINAL_PREIMAGES.__code__ is not _ORIGINAL_PREIMAGES_CODE
            or _ORIGINAL_SIDES.__code__ is not _ORIGINAL_SIDES_CODE
            or inspect.unwrap(preimages.__func__) is not _ORIGINAL_RAW_PREIMAGES
            or inspect.unwrap(sides.__func__) is not _ORIGINAL_RAW_SIDES
            or _ORIGINAL_RAW_PREIMAGES.__code__ is not _ORIGINAL_RAW_PREIMAGES_CODE
            or _ORIGINAL_RAW_SIDES.__code__ is not _ORIGINAL_RAW_SIDES_CODE
            or self.recipe.source_identity is not self.core.schedule_spec.sources
            or self.recipe.input_names
            != frozenset(
                name
                for source in self.core.schedule_spec.sources
                for name in (
                    source.threshold_param_name,
                    *source.derived_param_names,
                    *source.derived_state_names,
                    *(
                        ()
                        if source.threshold_index_state is None
                        else (source.threshold_index_state,)
                    ),
                )
            )
            or self.recipe.equal_above
            != tuple(
                source.equality_owner == "above"
                for source in self.core.schedule_spec.sources
            )
        ):
            raise TypeError("NNBEGM query recipe lost its original declared sources.")
        if (
            self.core.statics.has_jump
            or self.core.statics.continuation_reads_liquid
            or self.core.statics.co_map_state_names
            or self.core.cliff_candidates
            or self.core.continuation_plan.query_targets
            or self.core.continuation_plan.scalar_targets
            or binder.is_epstein_zin
            or binder.current_floor is not None
            or binder.current_predicates is not None
            or binder.feasibility_constraints
            or binder.n_published_boundaries
            or binder.schedule_spec.discrete_actions
        ):
            raise UnsupportedOperationError(
                "Finite NNBEGM U requires original continuous additive literal-floor "
                "ordinary-child NB cells without inner discrete or jump branches."
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class NNBEGMCaseUPoint:
    """Original finite nested owner and its genuine complete conditional NB banks."""

    context: SolverBuildContext
    plan: CurrentIIDPlan
    period: int
    owner: _FiniteNNBEGMPeriodKernel
    keeper: NNBEGMInnerSource
    adjuster: NNBEGMInnerSource
    source_grid: Float1D
    savings_nodes: Float1D
    borrowing_limit: ScalarFloat
    outer_nodes: Float1D
    coefficient: Fraction
    outer_state: StateName
    outer_action: ActionName
    keeper_target: Callable[..., FloatND]
    outer_function: Callable[..., Mapping[str, FloatND]]
    outer_inverse: DeclaredOuterInverse

    @property
    def euler_state_name(self) -> StateName:
        return self.owner.liquid_state_name

    @property
    def action_name(self) -> ActionName:
        return self.owner.inner_action

    @property
    def passive_names(self) -> tuple[StateName, ...]:
        return (self.outer_state,)

    @property
    def target(self) -> RegimeName:
        (target,) = self.keeper.continuation_plan.stateful_targets
        return target

    @property
    def transition_target_names(self) -> tuple[RegimeName, ...]:
        return tuple(self.context.transitions)

    def __post_init__(self) -> None:
        from _lcm.solution.nnbegm import _FiniteNNBEGMPeriodKernel  # noqa: PLC0415

        if (
            type(self.owner) is not _FiniteNNBEGMPeriodKernel
            or self.owner.regime_name != self.context.regime_name
            or self.context.current_iid_plans.get(self.period) is not self.plan
            or self.owner.keeper_kernel is not self.keeper.kernel
            or self.owner.adjuster_kernel is not self.adjuster.kernel
            or self.owner.outer_action != self.outer_action
            or self.outer_nodes.shape != self.owner.outer_grid_values.shape
        ):
            raise TypeError("NNBEGM point lost its genuine original finite owner.")
        for inner in (self.keeper, self.adjuster):
            if (
                tuple(inner.core.statics.ride_names) != (self.outer_state,)
                or inner.core.regime_name != self.context.regime_name
                or not inner.has_original_linear_plan(context=self.context)
                or tuple(inner.continuation_plan.stateful_targets) != (self.target,)
                or inner.core.statics.liquid_name != self.euler_state_name
            ):
                raise UnsupportedOperationError(
                    "Finite NNBEGM U requires its actual liquid/durable ordinary cells."
                )

    def with_operands(
        self,
        *,
        source_grid: Float1D,
        savings_nodes: Float1D,
        borrowing_limit: ScalarFloat,
        outer_nodes: Float1D,
    ) -> NNBEGMCaseUPoint:
        return replace(
            self,
            source_grid=source_grid,
            savings_nodes=savings_nodes,
            borrowing_limit=borrowing_limit,
            outer_nodes=outer_nodes,
        )

    def _construct_inner(
        self,
        *,
        inner: NNBEGMInnerSource,
        actual: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
        liquid: ScalarFloat,
    ) -> BoundedResult[NNBEGMOwnedRow]:
        # Fixed capacity preserves every original support coordinate and reserves
        # exactly one request slot. Duplicates are compacted, never truncated.
        raw = jnp.concatenate((self.source_grid, liquid[None]))
        queries, length, fits = _ordered_points(
            raw=raw,
            active=jnp.ones(raw.shape, dtype=jnp.bool_),
            capacity=self.source_grid.shape[0] + 1,
        )
        index = jnp.argmax(queries == liquid).astype(jnp.int32)
        valid = (
            fits
            & jnp.isfinite(liquid)
            & jnp.any(queries == liquid)
            & jnp.all(jnp.isfinite(self.source_grid))
            & jnp.all(self.source_grid[1:] > self.source_grid[:-1])
            & (self.borrowing_limit == self.savings_nodes[0])
            & (length >= self.source_grid.shape[0])
        )
        core = inner.core
        kwargs = {**actual, self.euler_state_name: self.source_grid}
        coordinates: list[jax.Array] = []
        for name in core.statics.ride_names:
            coordinate = kwargs[name]
            if not isinstance(coordinate, jax.Array) or coordinate.ndim != 0:
                raise TypeError("NNBEGM U requires its actual scalar durable cell.")
            coordinates.append(coordinate)
        ride_values = tuple(coordinates)
        solver = replace(
            core.envelope_build.bind_cell, savings_grid=self.savings_nodes
        )(kwargs)
        continuation = _NBEGMCellContinuation(
            kwargs=kwargs,
            carry=MappingProxyType(dict(children)),
            comap_bindings=MappingProxyType({}),
            savings_grid=self.savings_nodes,
            continuation_plan=core.continuation_plan,
            statics=core.statics,
            regime_name=core.regime_name,
            cliff_candidates=False,
            schedule_spec=core.schedule_spec,
        )(ride_values)
        bounds = ()
        statuses = ()
        counts = None
        if isinstance(continuation, BoundedResult):
            bounds = continuation.bound_status
            statuses = continuation.query_status or ()
            counts = continuation.continuation_read_counts
            continuation = continuation.output
        if not isinstance(continuation, tuple) or len(continuation) != len(
            ("value", "marginal")
        ):
            raise TypeError("NNBEGM inner source lost its original continuation pair.")
        value, marginal = continuation
        if not isinstance(value, jax.Array) or not isinstance(marginal, jax.Array):
            raise TypeError("NNBEGM continuation requires original numerical rows.")
        row = solver.construct_owned_query_row(
            ride_values=ride_values,
            query=queries,
            cont_value=value,
            cont_marginal=marginal,
        )
        no = jnp.asarray(a=False)
        return BoundedResult(
            output=NNBEGMOwnedRow(queries, row, index, valid),
            bound_status=bounds,
            query_status=(*statuses, QueryInputStatus(no, no, ~valid)),
            continuation_read_counts=counts,
        )

    # keyword-only-exempt: library-callback=_read_finite_nb_owned_corners
    def _preferences(
        self, pool: EconFunctionKwargs, *, inner: NNBEGMInnerSource
    ) -> Preferences:
        binder = inner.core.envelope_build.bind_cell
        function = binder.schedule_spec.utility_dag
        utility = _BoundScalarFunction(
            func=function,
            argument_name=self.action_name,
            bound={
                name: pool[name]
                for name in inspect.signature(function).parameters
                if name != self.action_name
            },
        )
        return preferences_from_utility(
            utility_of_action=utility,
            action_lower=binder.action_lower,
            action_upper=binder.action_upper,
        )

    # keyword-only-exempt: library-callback=_read_finite_nb_owned_corners
    def _discount(
        self, pool: EconFunctionKwargs, *, inner: NNBEGMInnerSource
    ) -> ScalarFloat:
        function = inner.core.schedule_spec.discount_factor_dag
        if function is None:
            value = pool["koopmans_aggregator__discount_factor"]
            if not isinstance(value, jax.Array) or value.ndim != 0:
                raise TypeError("NNBEGM source discount factor must be scalar.")
            return value
        return function(
            **{name: pool[name] for name in inspect.signature(function).parameters}
        )

    def _profile(
        self,
        *,
        inner: NNBEGMInnerSource,
        bank: NNBEGMOwnedRow,
        actual: EconFunctionKwargs,
        liquid: ScalarFloat,
    ) -> tuple[_FiniteOuterProfile, ScalarBool]:
        row = bank.row
        data = row.query_data
        function = inner.core.schedule_spec.coh_of_liquid_dag
        resources = function(
            **{name: actual[name] for name in inspect.signature(function).parameters}
        )
        eligible = resources > self.borrowing_limit
        raw = _read_finite_native_row(
            coordinate=liquid,
            # Original owned readers count a NaN-padded valid prefix. Keep
            # the constructed bank's +inf padding outside that read boundary.
            grid=jnp.where(jnp.isfinite(bank.grid), bank.grid, jnp.nan),
            value=row.value,
            marginal=row.marginal,
            data=data,
            owner_capacity=len(inner.recipe.source_identity) + 1,
            side_is_row_side=True,
        )
        corners = _read_finite_nb_owned_corners(
            recipe=inner.recipe,
            passive_names=(),
            resources_function=function,
            build_preferences=functools.partial(self._preferences, inner=inner),
            read_discount_factor=functools.partial(self._discount, inner=inner),
            liquid_state=self.euler_state_name,
            inverse_eis_name=None,
            data=data,
            pool=actual,
            passive_grids=(),
            resources=resources,
            floor=self.borrowing_limit,
            eligible=eligible,
            euler_gradient=jnp.asarray(1.0, dtype=bank.grid.dtype),
        )
        # The original compact corner reader retains its passive-cell row axis.
        # This original branch has already bound the sole durable coordinate.
        if any(channel.shape != (1,) for channel in corners):
            raise TypeError("NNBEGM requires its original single bound corner row.")
        corners = _single_row_finite_corner_read(
            value=corners.value[0],
            marginal=corners.marginal[0],
            savings=corners.savings[0],
            supported=corners.supported[0],
            finite=corners.finite[0],
            invalid=corners.invalid[0],
        )
        profile = _validate_finite_outer_rows(
            raw=raw,
            queries=liquid,
            gradients=jnp.asarray(1.0, dtype=bank.grid.dtype),
            floor=self.borrowing_limit,
            upper_saving=self.savings_nodes[-1],
            resources=resources,
            eligible=eligible,
            corner_value=corners.value,
            corner_marginal=corners.marginal,
            supported_corner=corners.supported,
            finite_corner=corners.finite,
            invalid_corner=corners.invalid,
            # This exact requested bank node retains its original economic M;
            # the value interpolant's directional germ remains separate.
            paired_marginal_read=False,
        )
        live = jnp.isfinite(bank.grid)
        node_resources = jax.vmap(
            lambda coordinate: function(
                **{
                    name: {**actual, self.euler_state_name: coordinate}[name]
                    for name in inspect.signature(function).parameters
                }
            )
        )(jnp.where(live, bank.grid, 0.0))
        original_coordinate = jnp.any(
            bank.grid[:, None] == self.source_grid[None, :], axis=1
        )
        empty = data.node_kind == QueryNodeKind.PAD
        allowed_empty = empty & (
            (node_resources <= self.borrowing_limit) | ~original_coordinate
        )
        n_liquid = self.source_grid.shape[0]
        n_savings = self.savings_nodes.shape[0]
        # The original continuous bank stacks Euler nodes and dense point pairs;
        # native identities name consecutive links followed by self-brackets.
        native_candidates = n_savings + 2 * n_liquid * n_savings
        native_capacity = 2 * native_candidates - 1
        finite = data.node_kind == QueryNodeKind.FINITE
        known = (data.node_kind == QueryNodeKind.FINITE) | (
            data.node_kind == QueryNodeKind.ECONOMIC_MINUS_INF
        )
        # Native winner and budget-run owner are distinct. Every represented
        # finite query node keeps its actual original selected native winner.
        integrity = (
            bank.support_valid
            & jnp.all(~live | jnp.isfinite(node_resources))
            & jnp.all(~live | known | allowed_empty)
            & jnp.all(
                ~live
                | jnp.isfinite(row.value)
                | jnp.isneginf(row.value)
                | allowed_empty
            )
            & jnp.all(~live | jnp.isfinite(row.marginal) | allowed_empty)
            & jnp.all(~empty | (row.owner == -1))
            & jnp.all(~finite | ((row.owner >= 0) & (row.owner < native_capacity)))
            & jnp.all(~finite | jnp.isfinite(row.policy))
            & jnp.all(
                ~finite
                | (
                    jnp.isfinite(data.savings)
                    & (data.savings >= self.borrowing_limit)
                    & (data.savings <= self.savings_nodes[-1])
                )
            )
            & jnp.all(data.floor_used == self.borrowing_limit)
            & ~jnp.any(profile.coverage_missing)
            & ~jnp.any(corners.invalid)
        )
        return profile, integrity

    def select(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> BoundedResult[tuple[OwnedResourcesRead, ScalarFloat, IntND]]:
        return self.read_constructed(
            liquid=liquid,
            physical=physical,
            pool=pool,
            banks=self.construct(
                liquid=liquid, physical=physical, pool=pool, children=children
            ),
        )

    def construct(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> tuple[BoundedResult[NNBEGMOwnedRow], ...]:
        if tuple(children) != (self.target,):
            raise TypeError("NNBEGM U lost its original future roster.")
        actual = {**pool, **physical, self.euler_state_name: liquid}
        targets = (self.keeper_target(**actual), *tuple(self.outer_nodes))
        return tuple(
            self._construct_inner(
                inner=self.keeper if position == 0 else self.adjuster,
                actual={**actual, self.owner.outer_post_decision: target},
                children=children,
                liquid=liquid,
            )
            for position, target in enumerate(targets)
        )

    def read_constructed(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        banks: tuple[BoundedResult[NNBEGMOwnedRow], ...],
    ) -> BoundedResult[tuple[OwnedResourcesRead, ScalarFloat, IntND]]:
        actual = {**pool, **physical, self.euler_state_name: liquid}
        targets = (self.keeper_target(**actual), *tuple(self.outer_nodes))
        if len(banks) != len(targets):
            raise TypeError("NNBEGM U requires every original finite alternative.")
        rows = []
        profiles = []
        valid = jnp.asarray(a=True)
        bounds = ()
        statuses = None
        counts = None
        for position, (target, constructed) in enumerate(
            zip(targets, banks, strict=True)
        ):
            inner = self.keeper if position == 0 else self.adjuster
            cell = {**actual, self.owner.outer_post_decision: target}
            profile, integrity = self._profile(
                inner=inner, bank=constructed.output, actual=cell, liquid=liquid
            )
            rows.append(constructed.output)
            profiles.append(profile)
            valid &= integrity
            if position == 0:
                bounds = constructed.bound_status
            elif len(bounds) != len(constructed.bound_status):
                raise TypeError("NNBEGM branches changed original status slots.")
            else:
                bounds = tuple(
                    merge_bound_input_status(left=left, right=right)
                    for left, right in zip(
                        bounds, constructed.bound_status, strict=True
                    )
                )
            statuses = (
                constructed.query_status
                if position == 0
                else merge_query_input_status(
                    left=statuses, right=constructed.query_status
                )
            )
            counts = merge_continuation_read_counts(
                left=counts, right=constructed.continuation_read_counts
            )
        candidates = jax.tree.map(lambda *values: jnp.stack(values, axis=-1), *profiles)
        selection = _select_finite_outer_profiles(profiles=candidates)
        value, marginal, _left, _direction = _collapse_finite_selection(
            selection=selection
        )
        index = selection.outer_index[0]
        side = selection.right
        use_corner = side.use_corner[0]
        ordinary = side.ordinary
        corner = side.corner
        # The right-extension germ orders ties; its dead mask must not erase
        # the selected original bank's economic marginal at a closed node.
        marginal = jnp.where(
            jnp.isnan(value),
            jnp.nan,
            jnp.where(use_corner, corner.marginal[0], ordinary.marginal[0]),
        )
        saving = jnp.where(use_corner, corner.savings[0], ordinary.savings[0])
        selected_owner = jnp.stack(
            [bank.row.owner[bank.requested_index] for bank in rows]
        )[index]
        selected_policy = jnp.stack(
            [bank.row.policy[bank.requested_index] for bank in rows]
        )[index]
        selected_saving = jnp.stack(
            [bank.row.query_data.savings[bank.requested_index] for bank in rows]
        )[index]
        selected_value = jnp.stack(
            [bank.row.value[bank.requested_index] for bank in rows]
        )[index]
        selected_marginal = jnp.stack(
            [bank.row.marginal[bank.requested_index] for bank in rows]
        )[index]
        supported = jnp.where(use_corner, corner.supported[0], ordinary.supported[0])
        resources = jnp.stack(
            [
                inner.core.schedule_spec.coh_of_liquid_dag(
                    **{
                        name: {**actual, self.owner.outer_post_decision: target}[name]
                        for name in inspect.signature(
                            inner.core.schedule_spec.coh_of_liquid_dag
                        ).parameters
                    }
                )
                for inner, target in zip(
                    (self.keeper, *((self.adjuster,) * self.outer_nodes.shape[0])),
                    targets,
                    strict=True,
                )
            ]
        )[index]
        valid &= (
            ~supported
            | use_corner
            | (
                (selected_owner >= 0)
                & (saving == selected_saving)
                & (ordinary.value[0] == selected_value)
                & (ordinary.marginal[0] == selected_marginal)
            )
        )

        def forward(action: FloatND) -> FloatND:
            return self.outer_function(**{**actual, self.outer_action: action})[
                self.owner.outer_post_decision
            ]

        target = jnp.stack(targets)[index]
        inversion = invert_declared_outer_target(
            inverse=self.outer_inverse,
            target=target,
            at_zero=forward(jnp.zeros_like(target)),
            forward=forward,
        )
        valid &= inversion.admissible
        no = jnp.asarray(a=False)
        return BoundedResult(
            output=(
                OwnedResourcesRead(
                    value=value,
                    marginal=jnp.where(supported, marginal, 0.0),
                    policy=jnp.where(use_corner, resources - saving, selected_policy),
                    savings=saving,
                    native_owner=jnp.where(use_corner, -1, selected_owner).astype(
                        jnp.int32
                    ),
                    side_id=jnp.where(use_corner, 0, ordinary.side_id[0]).astype(
                        jnp.int32
                    ),
                    corner=use_corner,
                    feasible=supported,
                    valid=valid,
                ),
                inversion.action,
                index,
            ),
            bound_status=bounds,
            query_status=(*(statuses or ()), QueryInputStatus(no, no, ~valid)),
            continuation_read_counts=counts,
        )


_ORIGINAL_NNBEGM_CLASSES = (
    NNBEGMCaseUPoint,
    NNBEGMInnerSource,
    _NBEGMCellBinder,
    _NBEGMCellContinuation,
    _NBEGMCellSolver,
    _NBEGMBranchSolver,
    _BoundScalarFunction,
    _NBEGMQueryIntervalEvaluators,
)
_ORIGINAL_NNBEGM_METHODS = tuple(
    (owner, name, method, method.__code__, raw, raw.__code__)
    for owner in _ORIGINAL_NNBEGM_CLASSES
    for name, method in vars(owner).items()
    if inspect.isfunction(method)
    and (
        name == "__init__"
        or name not in {"__repr__", "__eq__", "__hash__", "__setattr__", "__delattr__"}
    )
    for raw in (inspect.unwrap(method),)
) + tuple(
    (owner, name, method, method.__code__, raw, raw.__code__)
    for owner, name in ((_RideAlongNBEGMPeriodKernel, "core_programs"),)
    for method in (vars(owner)[name],)
    for raw in (inspect.unwrap(method),)
)
_ORIGINAL_NNBEGM_PROPERTIES = tuple(
    (owner, name, descriptor, getter, getter.__code__, raw, raw.__code__)
    for owner in _ORIGINAL_NNBEGM_CLASSES
    for name, descriptor in vars(owner).items()
    if isinstance(descriptor, property)
    for getter in (descriptor.fget,)
    if getter is not None
    for raw in (inspect.unwrap(getter),)
)


_ORIGINAL_NNBEGM_NEW = tuple(
    (owner, inspect.getattr_static(owner, "__new__"))
    for owner in _ORIGINAL_NNBEGM_CLASSES
)
_ORIGINAL_NNBEGM_IMPORTS = tuple(
    (
        module,
        sys.modules[module.__name__],
        name,
        function,
        function.__code__,
        raw,
        raw.__code__,
    )
    for module, names in (
        (_nn_continuation, ("bind_continuation",)),
        (
            _nn_breakpoints,
            (
                "interval_midpoints",
                "interval_segment_coefficients",
                "clamp_breakpoints_to_grid",
                "linear_asset_selection_preimage",
            ),
        ),
        (
            _nn_step,
            (
                "nbegm_per_interval_continuation_step_savings",
                "nbegm_multi_interval_step_savings",
                "nbegm_unified_step_savings",
            ),
        ),
        (_nn_preferences, ("preferences_from_utility",)),
    )
    for name in names
    for function in (vars(module)[name],)
    for raw in (inspect.unwrap(function),)
)


_ORIGINAL_NNBEGM_HELPERS = tuple(
    (_nn_inner_module, name, function, function.__code__, raw, raw.__code__)
    for name in (
        "_solve_ride_along_cell_step",
        "_nbegm_cell_breakpoints",
        "_cell_breakpoint",
    )
    for function in (vars(_nn_inner_module)[name],)
    for raw in (inspect.unwrap(function),)
)


_ORIGINAL_NNBEGM_CLASS_BINDINGS = tuple(
    (module, name, owner)
    for module in (sys.modules[__name__], _nn_inner_module)
    for name, owner in vars(module).items()
    if any(owner is original[0] for original in _ORIGINAL_NNBEGM_METHODS)
)


def require_original_nnbegm_case_u_methods() -> None:  # noqa: C901 - keep the bounded original origin checks together
    """Authenticate original engine constructors before any owned preparation."""
    from _lcm.solution import nnbegm as nn_owner_module  # noqa: PLC0415
    from _lcm.solution.fingerprint import (  # noqa: PLC0415
        _unwrap_exact_beartype_wrapper,
    )
    from _lcm.solution.nnbegm import _FiniteNNBEGMPeriodKernel  # noqa: PLC0415

    if (
        sys.modules.get(nn_owner_module.__name__) is not nn_owner_module
        or nn_owner_module._ORIGINAL_FINITE_NNBEGM_PERIOD_KERNEL  # noqa: SLF001
        is not _FiniteNNBEGMPeriodKernel
    ):
        raise TypeError("Original NNBEGM finite owner authority changed.")
    for module, name, original_class in _ORIGINAL_NNBEGM_CLASS_BINDINGS:
        if (
            sys.modules.get(module.__name__) is not module
            or vars(module).get(name) is not original_class
        ):
            raise TypeError("Original NNBEGM constructor class binding changed.")
    for owner, original_new in _ORIGINAL_NNBEGM_NEW:
        if (
            inspect.getattr_static(owner, "__new__") is not original_new
            or original_new is not object.__new__
        ):
            raise TypeError("Original NNBEGM constructor authority changed.")
    for (
        module,
        original_module,
        name,
        original,
        code,
        raw,
        raw_code,
    ) in _ORIGINAL_NNBEGM_IMPORTS:
        if (
            sys.modules.get(module.__name__) is not original_module
            or module is not original_module
            or vars(module).get(name) is not original
            or original.__code__ is not code
            or (_unwrap_exact_beartype_wrapper(original) or original) is not raw
            or raw.__code__ is not raw_code
        ):
            raise TypeError("Original NNBEGM imported numerical binding changed.")
    for owner, name, method, code, raw, raw_code in (
        *_ORIGINAL_NNBEGM_METHODS,
        *_ORIGINAL_NNBEGM_HELPERS,
    ):
        if (
            vars(owner).get(name) is not method
            or method.__code__ is not code
            or (_unwrap_exact_beartype_wrapper(method) or method) is not raw
            or raw.__code__ is not raw_code
        ):
            raise TypeError("Original NNBEGM shape method authority changed.")
    for (
        owner,
        name,
        descriptor,
        getter,
        code,
        raw,
        raw_code,
    ) in _ORIGINAL_NNBEGM_PROPERTIES:
        if (
            vars(owner).get(name) is not descriptor
            or descriptor.fget is not getter
            or getter.__code__ is not code
            or (_unwrap_exact_beartype_wrapper(getter) or getter) is not raw
            or raw.__code__ is not raw_code
        ):
            raise TypeError("Original NNBEGM shape property authority changed.")
