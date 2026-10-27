"""Original finite NEGM candidate construction under an actual current frame."""

import functools
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import aggregates_nonlinearly
from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.egm.finite_negm_query import (
    _collapse_finite_selection,
    _FiniteOuterProfile,
    _read_finite_native_row,
    _select_finite_outer_profiles,
    _validate_finite_outer_rows,
)
from _lcm.egm.outer_inversion import DeclaredOuterInverse, invert_declared_outer_target
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.step import _EGMStep
from _lcm.egm.step_core import OwnedResourcesRow, _EgmKernelPieces, _SolveOneCombo
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import (
    BoundedResult,
    QueryInputStatus,
    merge_bound_input_status,
    merge_continuation_read_counts,
    merge_query_input_status,
)
from _lcm.solution.negm import _NEGMPeriodKernel, _NodeSolver
from _lcm.typing import EconFunctionArg, EconFunctionKwargs
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import (
    ActionName,
    Float1D,
    FloatND,
    RegimeName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    StateName,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class NEGMCaseUPoint:
    """The actual keeper and every declared outer node, with their native rows.

    Construction consumes the complete realised frame before either nested
    branch runs. The original owned row constructor keeps each inner candidate's
    value, marginal and saving associated; the existing finite outer selector
    ranks those complete tuples only at the common pre-shock state.
    """

    context: SolverBuildContext
    """Full original outer source context, retaining its dated law authority."""
    plan: CurrentIIDPlan
    """Exact original current law owned by the outer dated source."""
    period: int
    """Original source date, shared by both actual inner branches."""
    owner: _NEGMPeriodKernel
    """Original solver-owned keeper, adjuster, outer nodes and cost binding."""
    keeper: _EgmKernelPieces
    """Exact original keeper pieces, including the actual continuation plan."""
    adjuster: _EgmKernelPieces
    """Exact original adjuster pieces from the same nested build."""
    refine: Callable[..., OwnedEnvelope]
    """The configured original owned inner envelope backend."""
    source_grid: Float1D
    """Original full liquid support used for each transient row's domain."""
    savings_nodes: Float1D
    """Actual original saving operands, explicit in source and replay programs."""
    borrowing_limit: ScalarFloat
    """Literal original lower saving operand, shared by both branches."""
    outer_nodes: Float1D
    """Actual original finite outer operands, in the owner's declared order."""
    coefficient: Fraction
    """Structurally certified common positive affine liquid-resource Jacobian."""
    outer_state: StateName
    """Actual fixed durable coordinate at this conditional source cell."""
    outer_action: ActionName
    """Original outer action recovered through the declared inverse."""
    keeper_target: Callable[..., FloatND]
    """Original injected no-adjustment post-decision function."""
    outer_function: Callable[..., Mapping[str, FloatND]]
    """Original declared outer post-decision DAG, not a substitute inverse."""
    outer_inverse: DeclaredOuterInverse
    """Canonical structural inverse used to recover the actual outer action."""
    owner_capacity: int
    """Original inner segment-owner capacity, independent of outer identity."""

    @property
    def euler_state_name(self) -> StateName:
        return self.keeper.euler_state_name

    @property
    def action_name(self) -> ActionName:
        return self.keeper.action_name

    @property
    def target(self) -> RegimeName:
        (target,) = self.keeper.continuation_plan.stateful_targets
        return target

    @property
    def passive_names(self) -> tuple[StateName, ...]:
        return (self.outer_state,)

    @property
    def transition_target_names(self) -> tuple[RegimeName, ...]:
        return self.owner.transition_target_names

    def with_operands(
        self,
        *,
        source_grid: Float1D,
        savings_nodes: Float1D,
        borrowing_limit: ScalarFloat,
        outer_nodes: Float1D,
    ) -> NEGMCaseUPoint:
        """Borrow the actual dated operands without replacing original statics."""
        return replace(
            self,
            source_grid=source_grid,
            savings_nodes=savings_nodes,
            borrowing_limit=borrowing_limit,
            outer_nodes=outer_nodes,
        )

    def __post_init__(self) -> None:
        if (
            type(self.owner) is not _NEGMPeriodKernel
            or self.owner.regime_name != self.context.regime_name
            or self.context.current_iid_plans.get(self.period) is not self.plan
        ):
            raise TypeError("NEGM current points require their original nested owner.")
        for kernel, pieces in (
            (self.owner.keeper_kernel, self.keeper),
            (self.owner.adjuster_kernel, self.adjuster),
        ):
            if (
                not isinstance(kernel, _DCEGMPeriodKernel)
                or not isinstance(kernel.demand_step, _EGMStep)
                or kernel.demand_step.pieces is not pieces
                or pieces.continuation_plan.certainty_equivalent
                is not self.context.certainty_equivalent
            ):
                raise TypeError("NEGM current points lost their genuine inner pieces.")
            if (
                pieces.query_spec is not None
                or pieces.lower_func is not None
                or pieces.continuation_plan.query_targets
                or aggregates_nonlinearly(pieces.continuation_plan.certainty_equivalent)
                or pieces.combo_names != (self.outer_state,)
                or kernel.demand_step.own_runtime_process_names
                or len(pieces.continuation_plan.stateful_targets) != 1
                or pieces.continuation_plan.scalar_targets
            ):
                raise UnsupportedOperationError(
                    "Finite NEGM U requires its original literal-floor single "
                    "durable cell and linear ordinary future reads."
                )
        if (
            self.keeper.euler_state_name != self.adjuster.euler_state_name
            or self.keeper.action_name != self.adjuster.action_name
            or self.keeper.n_pad != self.adjuster.n_pad
            or self.coefficient <= 0
            or self.owner_capacity < 1
        ):
            raise TypeError("NEGM inner branches disagree on their original schema.")

    def construct(
        self,
        *,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> tuple[OwnedResourcesRow, OwnedResourcesRow]:
        """Keep the original keeper and candidate-stacked adjuster rows transient."""
        if tuple(children) != (self.target,):
            raise TypeError("Finite NEGM U requires its actual original child roster.")
        actual = {**pool, **physical}
        keeper = self._construct_inner(
            pieces=self.keeper, actual=actual, children=children
        )
        adjusters = jax.lax.map(
            _NodeSolver(
                inner_core=functools.partial(
                    self._construct_adjuster, children=children
                ),
                outer_post_decision=self.owner.outer_post_decision,
                adjuster_arguments=MappingProxyType(actual),
            ),
            self.outer_nodes,
        )
        return keeper, adjusters

    def _construct_adjuster(
        self, *, children: Mapping[RegimeName, EGMCarry], **actual: EconFunctionArg
    ) -> OwnedResourcesRow:
        return self._construct_inner(
            pieces=self.adjuster, actual=actual, children=children
        )

    def _construct_inner(
        self,
        *,
        pieces: _EgmKernelPieces,
        actual: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> OwnedResourcesRow:
        fixed: dict[StateName, ScalarFloat] = {}
        for name in pieces.combo_names:
            value = actual[name]
            if not isinstance(value, jax.Array) or value.ndim != 0:
                raise TypeError("NEGM current construction requires its actual cell.")
            fixed[name] = value
        pieces = replace(
            pieces,
            savings_nodes=self.savings_nodes,
            borrowing_limit=self.borrowing_limit,
        )
        resources = jax.vmap(
            lambda liquid: pieces.own_resources_func(
                **{**actual, pieces.euler_state_name: liquid}
            )
        )(self.source_grid)
        return _SolveOneCombo(
            pieces=pieces,
            pool={name: value for name, value in actual.items() if name not in fixed},
            state_grid=self.source_grid,
            next_regime_to_continuation=MappingProxyType(dict(children)),
            savings_point_width=None,
            stochastic_node_width=None,
            envelope_cell_width=1,
            resolved_process_grids=MappingProxyType({}),
        ).owned_resources_row(
            combo_values=tuple(fixed[name] for name in pieces.combo_names),
            resources_domain=resources,
            refine=self.refine,
        )

    def select(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> BoundedResult[tuple[OwnedResourcesRead, ScalarFloat, ScalarInt]]:
        """Select the original complete finite outer tuple before any expectation."""
        keeper, adjusters = self.construct(
            physical=physical, pool=pool, children=children
        )
        return self.read_constructed(
            liquid=liquid,
            physical=physical,
            pool=pool,
            keeper=keeper,
            adjusters=adjusters,
        )

    def read_constructed(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        keeper: OwnedResourcesRow,
        adjusters: OwnedResourcesRow,
    ) -> BoundedResult[tuple[OwnedResourcesRead, ScalarFloat, ScalarInt]]:
        """Read complete original rows at one common pre-shock coordinate."""
        actual = {**pool, **physical, self.euler_state_name: liquid}
        targets = (self.keeper_target(**actual), *tuple(self.outer_nodes))
        rows = (
            keeper,
            *(
                jax.tree.map(operator.itemgetter(index), adjusters)
                for index in range(self.outer_nodes.shape[0])
            ),
        )
        profiles = []
        valid = jnp.asarray(a=True)
        bound = ()
        statuses = None
        counts = None
        for index, (target, row) in enumerate(zip(targets, rows, strict=True)):
            pieces = self.keeper if index == 0 else self.adjuster
            cell = (
                actual
                if index == 0
                else {**actual, self.owner.outer_post_decision: target}
            )
            profile, integrity = self._profile(row=row, pieces=pieces, actual=cell)
            profiles.append(profile)
            valid &= integrity
            row_bound = () if row.floor_status is None else (row.floor_status,)
            if index == 0:
                bound = row_bound
            else:
                if len(bound) != len(row_bound):
                    raise TypeError(
                        "NEGM branches lost the original shared floor slots."
                    )
                bound = tuple(
                    merge_bound_input_status(left=a, right=b)
                    for a, b in zip(bound, row_bound, strict=True)
                )
            statuses = merge_query_input_status(left=statuses, right=row.child_status)
            counts = merge_continuation_read_counts(
                left=counts, right=row.continuation_read_counts
            )
        selection = _select_finite_outer_profiles(
            profiles=jax.tree.map(lambda *leaves: jnp.stack(leaves), *profiles)
        )
        value, marginal, _left_marginal, _right_direction = _collapse_finite_selection(
            selection=selection
        )
        index = selection.outer_index[0]
        target = jnp.stack(targets)[index]
        side = selection.right
        ordinary, corner = side.ordinary, side.corner
        use_corner = side.use_corner[0]
        supported = jnp.where(use_corner, corner.supported[0], ordinary.supported[0])
        saving = jnp.where(use_corner, corner.savings[0], ordinary.savings[0])
        cash = jnp.stack(
            (
                self.keeper.own_resources_func(**actual),
                *(
                    self.adjuster.own_resources_func(
                        **{**actual, self.owner.outer_post_decision: node}
                    )
                    for node in self.outer_nodes
                ),
            )
        )[index]

        def forward(action: FloatND) -> FloatND:
            return self.outer_function(**{**actual, self.outer_action: action})[
                self.owner.outer_post_decision
            ]

        inversion = invert_declared_outer_target(
            inverse=self.outer_inverse,
            target=target,
            at_zero=forward(jnp.zeros_like(target)),
            forward=forward,
        )
        valid &= inversion.admissible
        valid &= self.borrowing_limit == self.savings_nodes[0]
        valid &= jnp.isfinite(value) | jnp.isneginf(value)
        no = jnp.zeros((), dtype=jnp.bool_)
        selected = OwnedResourcesRead(
            value=value,
            marginal=jnp.where(
                supported & ~jnp.isneginf(value),
                marginal,
                0.0,
            ),
            policy=jnp.where(supported, cash - saving, 0.0),
            savings=saving,
            native_owner=jnp.where(use_corner, -1, ordinary.owner_id[0]).astype(
                jnp.int32
            ),
            side_id=jnp.where(use_corner, 0, ordinary.side_id[0]).astype(jnp.int32),
            corner=use_corner,
            feasible=supported,
            valid=valid,
        )
        return BoundedResult(
            output=(selected, inversion.action, index),
            bound_status=bound,
            query_status=(*(statuses or ()), QueryInputStatus(no, no, ~valid)),
            continuation_read_counts=counts,
        )

    def _profile(
        self,
        *,
        row: OwnedResourcesRow,
        pieces: _EgmKernelPieces,
        actual: EconFunctionKwargs,
    ) -> tuple[_FiniteOuterProfile, ScalarBool]:
        """Use the existing native side/corner records and finite outer selector."""
        resources = pieces.own_resources_func(**actual)
        known_floor = jnp.isfinite(row.expected_floor_value) | jnp.isneginf(
            row.expected_floor_value
        )
        corner_kind = jnp.where(
            known_floor,
            jnp.where(
                jnp.isneginf(row.expected_floor_value),
                QueryNodeKind.ECONOMIC_MINUS_INF,
                QueryNodeKind.FINITE,
            ),
            QueryNodeKind.PAD,
        ).astype(jnp.int32)
        data = EGMQueryData(
            savings=row.envelope.savings,
            node_kind=row.node_kind,
            link_valid=row.envelope.link_valid,
            owner_id=row.envelope.owner_id,
            side_id=row.envelope.side_id,
            floor_used=row.borrowing_limit,
            corner_continuation=row.expected_floor_value[None],
            corner_kind=corner_kind[None],
            corner_interval_id=jnp.zeros((1,), dtype=jnp.int32),
            corner_side_id=jnp.zeros((1,), dtype=jnp.int32),
        )
        native = _read_finite_native_row(
            coordinate=resources,
            grid=row.envelope.grid,
            value=row.envelope.value,
            marginal=row.marginal,
            data=data,
            owner_capacity=self.owner_capacity,
            side_is_row_side=True,
        )
        coefficient = jnp.asarray(float(self.coefficient), dtype=resources.dtype)
        # Every candidate has the same structurally proved positive affine
        # resource slope. Multiplying the kth germ by that common positive
        # slope**k leaves each original finite selector comparison unchanged.
        # Published economic marginals still include the resource Jacobian.
        utility = BoundUtilityOfAction(
            utility_func=pieces.utility_func,
            action_name=pieces.action_name,
            bound=actual,
        )
        domain = resources > row.borrowing_limit
        eligible = domain
        if pieces.feasibility_func is not None:
            eligible &= pieces.feasibility_func(**actual)
        evaluated = jnp.where(eligible, resources - row.borrowing_limit, 1.0)
        value = utility(evaluated) + zero_safe_weighted_term(
            weight=row.discount_factor,
            value=row.expected_floor_value,
            subnormal_is_accounted_for=True,
        )
        marginal = jax.grad(utility)(evaluated) * coefficient
        invalid_corner = (
            eligible
            & known_floor
            & ((~jnp.isfinite(value) & ~jnp.isneginf(value)) | ~jnp.isfinite(marginal))
        )
        profile = _validate_finite_outer_rows(
            raw=native,
            queries=resources,
            gradients=coefficient,
            floor=row.borrowing_limit,
            upper_saving=row.upper_savings,
            resources=resources,
            eligible=eligible,
            corner_value=value,
            corner_marginal=marginal,
            supported_corner=known_floor,
            finite_corner=known_floor & jnp.isfinite(value) & jnp.isfinite(marginal),
            invalid_corner=invalid_corner,
            paired_marginal_read=False,
        )
        valid = ~row.construction_invalid & ~profile.coverage_missing
        if row.floor_status is not None:
            valid &= ~(
                row.floor_status.nonfinite_floor | row.floor_status.invalid_shifted_grid
            )
        for status in row.child_status or ():
            for leaf in jax.tree.leaves(status):
                valid &= ~jnp.any(leaf)
        return profile, valid
