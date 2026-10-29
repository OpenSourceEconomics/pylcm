"""Guarded BALANCED U projection for fixed conditional original candidate banks.

Only the saturated-span class has fixed Hermite operands. Other U sources keep
exact-coordinate publication. Conditional rows are temporary; the durable guard
retains original producer operands so `validate` can check ownership ex post.
"""

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.continuation import ContinuationPayload
from _lcm.current_iid import AdmittedGroupParams, NodeOperands
from _lcm.egm.asset_row import (
    _ConstructedAssetNode,
    _node_value_slope,
    _publish_node_selection,
)
from _lcm.egm.current_iid_balanced import _ordered_points
from _lcm.egm.current_iid_case_u import (
    CaseUCurrentNodes,
    CaseUPointSolver,
    realise_case_u_nodes,
)
from _lcm.egm.current_iid_log_utility import LogUtilityProgram
from _lcm.egm.current_iid_projection import refine_projected_row
from _lcm.egm.current_iid_pullback import pullback_resource_boundary
from _lcm.egm.current_iid_resources_proof import prove_resources_only
from _lcm.egm.interp import (
    _hermite_query_derivative,
    materialize_hermite_shadow,
    prepare_padded_grid,
)
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.projected_continuation import ProjectedEGMCarry, ProjectedRowRead
from _lcm.egm.query_data import QueryNodeKind
from _lcm.egm.step import _EGMStep
from _lcm.egm.upper_envelope import _bracket_from_owned_row, get_upper_envelope
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.egm.validation import _dag_ancestors, _savings_stage_candidates
from _lcm.probability import is_represented_zero
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.current_iid_case_r_build import _initial_resources_coefficient
from _lcm.solution.current_iid_case_u_build import (
    CaseUProgramFunction,
    _bind_numeric_pieces,
    _carry_subset,
)
from _lcm.solution.dcegm import EGMStepBuild, ExactEnvelope, _BoundDCEGM
from _lcm.solution.feasibility import BoundInputStatus
from _lcm.typing import (
    EconFunctionArg,
    EconFunctionKwargs,
    FootprintTree,
    PytreeChild,
    PytreeValue,
    ShapeDtypePytree,
)
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import LawValue, ShockGroupName, ShockName
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


@dataclass(frozen=True, kw_only=True)
class CaseUBalancedOperands:
    """Closed numerical source inputs with immutable original name rosters."""

    pool_values: tuple[EconFunctionArg, ...]
    pool_names: tuple[ReferenceName, ...]
    child_values: tuple[ContinuationPayload, ...]
    child_names: tuple[RegimeName, ...]
    group_values: tuple[tuple[LawValue | BoolND, ...], ...]
    group_names: tuple[ShockGroupName, ...]
    parameter_names: tuple[tuple[ReferenceName, ...], ...]
    node_values: tuple[tuple[FloatND | tuple[IntND, ...], ...], ...]
    node_group_names: tuple[ShockGroupName, ...]
    node_names: tuple[tuple[str, ...], ...]
    code_names: tuple[tuple[tuple[ShockName, ...] | None, ...], ...]
    indices: IntND
    savings_nodes: FloatND
    borrowing_limit: ScalarFloat
    floor_values: tuple[ScalarBool, ...]
    has_floor: bool
    child_nodes: tuple[tuple[FloatND | IntND, ...], ...]
    lower: ScalarFloat
    upper: ScalarFloat

    @classmethod
    def create(
        cls,
        *,
        pool: EconFunctionKwargs,
        children: MappingProxyType[RegimeName, ContinuationPayload],
        group_params: AdmittedGroupParams,
        node_operands: NodeOperands,
        indices: IntND,
        savings_nodes: FloatND,
        borrowing_limit: ScalarFloat,
        floor_status: BoundInputStatus | None,
        child_nodes: tuple[tuple[FloatND | IntND, ...], ...],
        lower: ScalarFloat,
        upper: ScalarFloat,
    ) -> "CaseUBalancedOperands":  # noqa: UP037 - PEP 649 classmethod evaluated before its class is bound
        """Preserve the original JAX mapping leaf order without storing mappings."""
        pool_names = _guard_mapping_names(pool)
        child_names = _guard_mapping_names(children)
        group_names = _guard_mapping_names(group_params)
        parameter_names = tuple(
            _guard_mapping_names(group_params[name]) for name in group_names
        )
        node_groups = _guard_mapping_names(node_operands)
        node_names = tuple(
            _guard_mapping_names(node_operands[name]) for name in node_groups
        )
        values = []
        codes = []
        for group, names in zip(node_groups, node_names, strict=True):
            group_values = []
            group_codes = []
            for name in names:
                value = node_operands[group][name]
                if isinstance(value, jax.Array):
                    group_values.append(value)
                    group_codes.append(None)
                else:
                    keys = _guard_mapping_names(value)
                    group_values.append(tuple(value[key] for key in keys))
                    group_codes.append(keys)
            values.append(tuple(group_values))
            codes.append(tuple(group_codes))
        return cls(
            pool_values=tuple(pool[name] for name in pool_names),
            pool_names=pool_names,
            child_values=tuple(children[name] for name in child_names),
            child_names=child_names,
            group_values=tuple(
                tuple(group_params[group][name] for name in names)
                for group, names in zip(group_names, parameter_names, strict=True)
            ),
            group_names=group_names,
            parameter_names=parameter_names,
            node_values=tuple(values),
            node_group_names=node_groups,
            node_names=node_names,
            code_names=tuple(codes),
            indices=indices,
            savings_nodes=savings_nodes,
            borrowing_limit=borrowing_limit,
            floor_values=() if floor_status is None else tuple(floor_status),
            has_floor=floor_status is not None,
            child_nodes=child_nodes,
            lower=lower,
            upper=upper,
        )

    @property
    def floor_status(self) -> BoundInputStatus | None:
        """Restore the original optional status over the stored numerical flags."""
        expected = len(BoundInputStatus._fields) if self.has_floor else 0
        if len(self.floor_values) != expected:
            raise TypeError("U guard lost its original floor status layout.")
        return (
            tuple.__new__(BoundInputStatus, self.floor_values)
            if self.has_floor
            else None
        )

    @property
    def pool(self) -> EconFunctionKwargs:
        """Borrow the original numerical pool under its original names."""
        return MappingProxyType(
            dict(zip(self.pool_names, self.pool_values, strict=True))
        )

    @property
    def children(self) -> MappingProxyType[RegimeName, ContinuationPayload]:
        """Borrow the original child carries under their original names."""
        return MappingProxyType(
            dict(zip(self.child_names, self.child_values, strict=True))
        )

    @property
    def group_params(self) -> AdmittedGroupParams:
        """Borrow actual grouped law parameters without changing their order."""
        return MappingProxyType(
            {
                group: MappingProxyType(dict(zip(names, values, strict=True)))
                for group, names, values in zip(
                    self.group_names,
                    self.parameter_names,
                    self.group_values,
                    strict=True,
                )
            }
        )

    @property
    def node_operands(self) -> NodeOperands:
        """Borrow actual node banks and code columns from closed stored tuples."""
        result = {}
        for group, names, values, codes in zip(
            self.node_group_names,
            self.node_names,
            self.node_values,
            self.code_names,
            strict=True,
        ):
            operands = {}
            for name, value, keys in zip(names, values, codes, strict=True):
                if keys is not None:
                    if not isinstance(value, tuple):
                        raise TypeError("U guard lost its original node code columns.")
                    operands[name] = MappingProxyType(
                        dict(zip(keys, value, strict=True))
                    )
                else:
                    if isinstance(value, tuple):
                        raise TypeError(
                            "U guard lost its original numerical node bank."
                        )
                    operands[name] = value
            result[group] = MappingProxyType(operands)
        return MappingProxyType(result)


def _guard_mapping_names[T](values: Mapping[str, T]) -> tuple[str, ...]:
    """Follow the original registered mapping tree's exact numerical order."""
    if type(values) is MappingProxyType:
        return tuple(values)
    if type(values) is dict:
        return tuple(sorted(values))
    raise TypeError("U guard requires its original engine mapping container.")


@dataclass(frozen=True, kw_only=True)
class BalancedCaseUCarry(ProjectedEGMCarry):
    """Stored economics and original source inputs for ex-post ownership checks."""

    value_slope: FloatND
    """Derivative of the reference value at each finite record; zero elsewhere."""
    guard: CaseUBalancedOperands

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.value_slope.shape != self.grid.shape:
            raise ValueError("Case-U value slopes have a different row schema.")
        if self.value_slope.dtype != self.grid.dtype:
            raise TypeError("Case-U value slopes must share the grid dtype.")

    @property
    def guard_required(self) -> bool:
        return True

    @property
    def value_slope_channel(self) -> FloatND:
        return self.value_slope

    def leaves(self) -> Mapping[tuple[str, ...], FloatND | IntND | BoolND]:
        return MappingProxyType(
            {path: leaf for path, (_, leaf) in self._paths().items()}
        )

    def input_path(self, *, leaf_path: tuple[str, ...]) -> tuple[str | int, ...]:
        return self._paths()[leaf_path][0]

    def _paths(  # noqa: C901 - validate each actual native key kind
        self,
    ) -> Mapping[tuple[str, ...], tuple[tuple[str | int, ...], jax.Array]]:
        """Resolve the actual registered tree, including immutable mappings."""
        result = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(self)[0]:
            node: FootprintTree = self
            identity: list[str] = []
            consumer: list[str | int] = []
            for original_key in path:
                key = original_key
                if isinstance(key, jax.tree_util.FlattenedIndexKey):
                    if not isinstance(node, MappingProxyType):
                        raise TypeError(
                            "U guard flattened key has no immutable mapping."
                        )
                    key = jax.tree_util.DictKey(tuple(node)[key.key])
                if isinstance(key, jax.tree_util.DictKey):
                    if not isinstance(node, Mapping) or not isinstance(key.key, str):
                        raise TypeError(
                            "U guard requires original named mapping leaves."
                        )
                    identity.append(key.key)
                    consumer.append(key.key)
                    node = node[key.key]
                elif isinstance(key, jax.tree_util.SequenceKey):
                    if not isinstance(node, tuple):
                        raise TypeError("U guard sequence lost its original tuple.")
                    identity.append(str(key.idx))
                    consumer.append(key.idx)
                    node = node[key.idx]
                elif isinstance(key, jax.tree_util.GetAttrKey):
                    identity.append(key.name)
                    names = getattr(node, "_fields", ())
                    consumer.append(
                        names.index(key.name) if isinstance(node, tuple) else key.name
                    )
                    node = getattr(node, key.name)
                else:
                    raise TypeError("U guard has an unsupported native leaf key.")
            if node is not leaf or tuple(identity) in result:
                raise TypeError("U guard native leaf identity is ambiguous.")
            result[tuple(identity)] = (tuple(consumer), leaf)
        return MappingProxyType(result)


class _ConditionalRow(NamedTuple):
    """One node's original conditional construction at the row's lower end."""

    envelope: OwnedEnvelope
    saturated: ScalarBool
    borrowing_limit: ScalarFloat
    savings_ceiling: ScalarFloat
    discounted_floor_value: ScalarFloat
    construction_valid: ScalarBool
    """No mandatory child summary of the original construction flagged."""

    @classmethod
    def of(
        cls,
        *,
        node: _ConstructedAssetNode,
        envelope: OwnedEnvelope,
        saturated: ScalarBool,
    ) -> "_ConditionalRow":  # noqa: UP037 - PEP 649 classmethod evaluated before its class is bound
        """Keep only the arrays that topology and reads use from the original node."""
        construction_valid = jnp.ones((), dtype=jnp.bool_)
        for leaf in jax.tree.leaves(node.child_status):
            construction_valid &= ~jnp.any(leaf)
        return cls(
            envelope=envelope,
            saturated=saturated,
            borrowing_limit=node.pieces.borrowing_limit,
            savings_ceiling=node.pieces.savings_nodes[-1],
            discounted_floor_value=node.discount_factor * node.expected_values[0],
            construction_valid=construction_valid,
        )


def _flatten_conditional_row(
    record: _ConditionalRow,
) -> tuple[tuple[PytreeChild | OwnedEnvelope, ...], None]:
    return tuple(record), None


def _flatten_conditional_row_with_keys(
    record: _ConditionalRow,
) -> tuple[
    tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild | OwnedEnvelope], ...], None
]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(_ConditionalRow._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_conditional_row(
    metadata: None, children: Iterable[PytreeChild | OwnedEnvelope]
) -> _ConditionalRow:
    """Restore stacked, abstract or per-node leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(_ConditionalRow._fields):
        raise TypeError("U conditional row reconstruction requires its exact fields.")
    return cast("_ConditionalRow", tuple.__new__(_ConditionalRow, leaves))


jax.tree_util.register_pytree_with_keys(
    _ConditionalRow,
    _flatten_conditional_row_with_keys,
    _unflatten_conditional_row,
    _flatten_conditional_row,
)


class _NodeReads(NamedTuple):
    """Selected reads off conditional rows at a query; scalar or node-stacked."""

    value: FloatND
    slope: FloatND
    """Derivative of the selected value piece along the liquid direction."""
    marginal: FloatND
    """Current utility's derivative at the frozen selected saving."""
    feasible: BoolND
    valid: BoolND


@dataclass(frozen=True, kw_only=True, eq=False)
class SaturatedCaseUProgram:
    """Model-owned original constructor and fixed-conditional topology proof."""

    point: CaseUProgramFunction
    utility: LogUtilityProgram
    refine: Callable[..., OwnedEnvelope]

    def solver(self, *, guard: CaseUBalancedOperands) -> CaseUPointSolver:
        pieces = _bind_numeric_pieces(
            original=self.point.pieces,
            savings_nodes=guard.savings_nodes,
            borrowing_limit=guard.borrowing_limit,
            floor_status=guard.floor_status,
            child_node_values=guard.child_nodes,
        )
        return CaseUPointSolver(
            pieces=pieces,
            pool=guard.pool,
            children=_carry_subset(
                next_regime_to_continuation=guard.children,
                stateful_targets=frozenset(
                    (
                        *pieces.continuation_plan.stateful_targets,
                        *pieces.continuation_plan.scalar_targets,
                    )
                ),
            ),
            savings_point_width=self.point.savings_point_width,
            stochastic_node_width=self.point.stochastic_node_width,
            envelope_cell_width=self.point.envelope_cell_width,
        )

    def nodes(self, *, guard: CaseUBalancedOperands) -> CaseUCurrentNodes:
        return realise_case_u_nodes(
            plan=self.point.plan,
            liquid=guard.lower,
            liquid_state=self.point.pieces.euler_state_name,
            arguments=guard.pool,
            group_params=guard.group_params,
            node_operands=guard.node_operands,
            indices=guard.indices,
        )

    def reference(
        self, *, guard: CaseUBalancedOperands, query: ScalarFloat
    ) -> ProjectedRowRead:
        """Original integrated level/economic derivative, rebuilt at each query.

        Publication reads ``hoisted_reference``; ex-post validation and tests
        keep this rebuilding oracle.
        """
        result, law_valid = self.point.evaluate_with_law(
            next_regime_to_continuation=guard.children,
            _lcm_current_iid=guard.group_params,
            _lcm_current_iid_nodes=guard.node_operands,
            _lcm_current_iid_rows=guard.indices,
            _lcm_case_u_savings_nodes=guard.savings_nodes,
            _lcm_case_u_borrowing_limit=guard.borrowing_limit,
            _lcm_case_u_floor_status=guard.floor_status,
            _lcm_case_u_child_node_values=guard.child_nodes,
            **{**guard.pool, self.point.pieces.euler_state_name: query[None]},
        )
        valid = result.output.valid[0] & jnp.all(law_valid)
        for status in (*result.bound_status, *(result.query_status or ())):
            for leaf in jax.tree.leaves(status):
                valid &= ~jnp.any(leaf)
        return ProjectedRowRead(
            result.output.value[0],
            result.output.economic_marginal[0],
            jnp.int32(0),
            jnp.int32(0),
            valid,
        )

    def value_slope(
        self,
        *,
        guard: CaseUBalancedOperands,
        conditionals: _ConditionalRow,
        query: ScalarFloat,
    ) -> ScalarFloat:
        """Derivative of the reference value, read off the fixed conditional rows.

        Each node's level derivative is the selected piece's own: the borrowing
        corner's ``u'(R - L)`` or the Hermite piece of a non-collapsed envelope
        bracket, never an automatic derivative of the exact-node bracket
        selection. Weights and their tangents combine as in the integration.
        """
        nodes, reads = self._node_reads(
            guard=guard, conditionals=conditionals, query=query
        )
        factors = nodes.factors
        terms = factored_weighted_term(factors=factors, values=reads.slope)
        for column in range(factors.shape[0]):
            terms += factored_weighted_term(
                factors=_with_tangent(nodes=nodes, column=column), values=reads.value
            )
        return jnp.sum(terms)

    def hoisted_reference(
        self,
        *,
        guard: CaseUBalancedOperands,
        conditionals: _ConditionalRow,
        query: ScalarFloat,
    ) -> ProjectedRowRead:
        """``reference`` read off the fixed conditional rows, without rebuilding.

        ``conditionals`` is ``conditionals()`` of this guard, built once per row.
        Selection, saving and corner are the reference's own; admission makes
        discount, future reads and child operands query-invariant, so the
        economic marginal is the current utility's at the frozen saving. The
        integration and validity follow the reference's.
        """
        nodes, reads = self._node_reads(
            guard=guard, conditionals=conditionals, query=query
        )
        factors = nodes.factors
        live = jnp.all(~is_represented_zero(factors), axis=0)
        derivative_needed = _derivative_needed(nodes=nodes)
        needed = live | derivative_needed
        value_terms = factored_weighted_term(factors=factors, values=reads.value)
        marginal_terms = factored_weighted_term(factors=factors, values=reads.marginal)
        for column in range(factors.shape[0]):
            marginal_terms += factored_weighted_term(
                factors=_with_tangent(nodes=nodes, column=column), values=reads.value
            )
        value = jnp.sum(value_terms)
        raw_marginal = jnp.sum(marginal_terms)
        economic = jnp.any(live & jnp.isneginf(reads.value))
        empty = jnp.any(live & ~reads.feasible)
        physical_valid = jnp.ones((), dtype=jnp.bool_)
        for tangent in nodes.physical_tangent.values():
            physical_valid &= jnp.all(jnp.isfinite(tangent))
        valid = (
            nodes.law_valid
            & jnp.all(~needed | reads.valid)
            & jnp.all(jnp.isfinite(nodes.factor_tangent))
            & physical_valid
            & jnp.all(
                ~live
                | (jnp.isfinite(reads.value) & jnp.isfinite(value_terms))
                | (jnp.isneginf(reads.value) & jnp.isneginf(value_terms))
            )
            & (jnp.isfinite(value) | (economic & jnp.isneginf(value)))
            & jnp.all(jnp.isfinite(marginal_terms))
            & jnp.isfinite(raw_marginal)
            & jnp.all(~derivative_needed | jnp.isfinite(reads.value))
            & (guard.pool["period"] == self.point.period)
        )
        for leaf in jax.tree.leaves(guard.floor_status):
            valid &= ~jnp.any(leaf)
        return ProjectedRowRead(
            value,
            jnp.where(empty | economic, 0.0, raw_marginal),
            jnp.int32(0),
            jnp.int32(0),
            valid,
        )

    def _node_reads(
        self,
        *,
        guard: CaseUBalancedOperands,
        conditionals: _ConditionalRow,
        query: ScalarFloat,
    ) -> tuple[CaseUCurrentNodes, _NodeReads]:
        """Each node's selected value, slope and economic marginal at ``query``."""
        pieces = self.point.pieces
        state = pieces.euler_state_name
        nodes = realise_case_u_nodes(
            plan=self.point.plan,
            liquid=query,
            liquid_state=state,
            arguments=guard.pool,
            group_params=guard.group_params,
            node_operands=guard.node_operands,
            indices=guard.indices,
        )

        def one(
            inputs: tuple[
                Mapping[ShockName, ScalarFloat | ScalarInt],
                Mapping[ShockName, ScalarFloat],
                ScalarBool,
                _ConditionalRow,
            ],
        ) -> _NodeReads:
            physical, tangent, active, conditional = inputs
            floating = {name: physical[name] for name in tangent}

            def read() -> _NodeReads:
                # keyword-only-exempt: library-callback=jax.jvp
                def resources(
                    point: ScalarFloat, draw: Mapping[ShockName, ScalarFloat]
                ) -> ScalarFloat:
                    return self.resources(
                        guard=guard, physical={**physical, **draw}, query=point
                    )

                at, direction = jax.jvp(
                    resources,
                    (query, floating),
                    (jnp.ones_like(query), tangent),
                )
                utility = BoundUtilityOfAction(
                    utility_func=pieces.utility_func,
                    action_name=pieces.action_name,
                    bound={**guard.pool, **physical, state: query},
                )
                row = conditional.envelope
                owned = _bracket_from_owned_row(refined=row, x_query=at)
                value, policy, corner = _publish_node_selection(
                    bracket=owned.bracket,
                    n_pad=pieces.n_pad,
                    resources_at_node=at,
                    borrowing_limit=conditional.borrowing_limit,
                    utility_of_action=utility,
                    discounted_expected_value_at_limit=(
                        conditional.discounted_floor_value
                    ),
                )
                search, length = prepare_padded_grid(row.grid)
                envelope = _hermite_query_derivative(
                    x_query=at,
                    search_grid=search,
                    valid_length=length,
                    xp=row.grid,
                    fp=row.value,
                    fp_slopes=jax.vmap(
                        lambda policy, value: _node_value_slope(
                            policy=policy, value=value, utility_of_action=utility
                        )
                    )(row.policy, row.value),
                )
                floor = jax.grad(utility)(
                    jnp.maximum(
                        at - conditional.borrowing_limit, jnp.finfo(at.dtype).tiny
                    )
                )
                # The selected saving is held fixed in the economic marginal.
                saving = jnp.where(corner, conditional.borrowing_limit, owned.savings)

                # keyword-only-exempt: library-callback=jax.jvp
                def current_utility(
                    point: ScalarFloat, draw: Mapping[ShockName, ScalarFloat]
                ) -> ScalarFloat:
                    return BoundUtilityOfAction(
                        utility_func=pieces.utility_func,
                        action_name=pieces.action_name,
                        bound={**guard.pool, **physical, **draw, state: point},
                    )(resources(point, draw) - saving)

                _, marginal = jax.jvp(
                    current_utility,
                    (query, floating),
                    (jnp.ones_like(query), tangent),
                )
                finite_or_economic = jnp.isfinite(value) | jnp.isneginf(value)
                feasible = finite_or_economic & (policy > 0.0)
                if pieces.feasibility_func is not None:
                    feasible &= pieces.feasibility_func(
                        **{**guard.pool, **physical, state: query}
                    )
                smooth = feasible & jnp.isfinite(value)
                valid = (
                    conditional.construction_valid
                    & ~owned.construction_invalid
                    & jnp.isfinite(saving)
                    & jnp.isfinite(policy)
                    & jnp.isfinite(marginal)
                    & (saving >= conditional.borrowing_limit)
                    & (saving <= conditional.savings_ceiling)
                    & (corner | owned.read_supported)
                    & finite_or_economic
                )
                return _NodeReads(
                    value=jnp.where(feasible, value, -jnp.inf),
                    slope=jnp.where(
                        smooth, jnp.where(corner, floor, envelope) * direction, 0.0
                    ),
                    marginal=jnp.where(smooth, marginal, 0.0),
                    feasible=feasible,
                    valid=valid,
                )

            zero = jnp.zeros_like(query)
            no = jnp.zeros((), dtype=jnp.bool_)
            return jax.lax.cond(
                active,
                read,
                lambda: _NodeReads(
                    value=zero, slope=zero, marginal=zero, feasible=no, valid=~no
                ),
            )

        return nodes, jax.lax.map(
            one,
            (
                nodes.physical,
                nodes.physical_tangent,
                _needed(nodes=nodes),
                conditionals,
            ),
        )

    def conditional(
        self,
        *,
        guard: CaseUBalancedOperands,
        physical: Mapping[ShockName, ScalarFloat | ScalarInt],
        tangent: Mapping[ShockName, ScalarFloat],
    ) -> tuple[_ConstructedAssetNode, ScalarBool]:
        prepared = self.solver(guard=guard).prepare_node(
            liquid=guard.lower,
            physical=physical,
            physical_tangent=tangent,
            owned_refine=self.refine,
        )
        node = prepared[0]
        # Structural admission excludes pre-state candidate/law ancestry. This
        # actual rounded min therefore selects the same operand for the domain.
        saturated = (
            (node.resources_at_node >= node.first_endogenous_point)
            & jnp.isfinite(node.first_endogenous_point)
            & (node.first_endogenous_point > node.pieces.borrowing_limit)
        )
        return node, saturated

    def conditionals(self, *, guard: CaseUBalancedOperands) -> _ConditionalRow:
        """Build each needed node's conditional row once; it never reads a query."""
        nodes = self.nodes(guard=guard)

        def one(
            inputs: tuple[
                Mapping[ShockName, ScalarFloat | ScalarInt],
                Mapping[ShockName, ScalarFloat],
                ScalarBool,
            ],
        ) -> _ConditionalRow:
            physical, tangent, active = inputs

            def produce() -> _ConditionalRow:
                node, saturated = self.conditional(
                    guard=guard, physical=physical, tangent=tangent
                )
                if node.owned_envelope is None:
                    raise TypeError(
                        "U topology requires its original full owned envelope."
                    )
                return _ConditionalRow.of(
                    node=node, envelope=node.owned_envelope, saturated=saturated
                )

            # Inactive nodes are never read; zeros keep the stacked schema.
            return jax.lax.cond(
                active, produce, lambda: jax.tree.map(jnp.zeros_like, produce())
            )

        return jax.lax.map(
            one, (nodes.physical, nodes.physical_tangent, _needed(nodes=nodes))
        )

    def resources(
        self,
        *,
        guard: CaseUBalancedOperands,
        physical: Mapping[ShockName, ScalarFloat | ScalarInt],
        query: ScalarFloat,
    ) -> ScalarFloat:
        function = self.point.pieces.own_resources_func
        values = {**guard.pool, **physical, self.point.pieces.euler_state_name: query}
        return function(
            **{name: values[name] for name in inspect.signature(function).parameters}
        )

    def cell(
        self,
        *,
        guard: CaseUBalancedOperands,
        left: ScalarFloat,
        right: ScalarFloat,
        query: ScalarFloat | None = None,
        conditionals: _ConditionalRow | None = None,
    ) -> ScalarBool:
        """Link a cell whose ends keep one selection; with a query, check it ex post.

        ``conditionals`` is ``conditionals()`` of this guard, built once per row.
        """
        nodes = self.nodes(guard=guard)
        needed = _needed(nodes=nodes)
        if conditionals is None:
            conditionals = self.conditionals(guard=guard)

        def one(
            inputs: tuple[
                Mapping[ShockName, ScalarFloat | ScalarInt],
                Mapping[ShockName, ScalarFloat],
                ScalarBool,
                _ConditionalRow,
            ],
        ) -> ScalarBool:
            physical, tangent, active, conditional = inputs

            def check() -> ScalarBool:
                utility_of_action = BoundUtilityOfAction(
                    utility_func=self.point.pieces.utility_func,
                    action_name=self.point.pieces.action_name,
                    bound={
                        **guard.pool,
                        **physical,
                        self.point.pieces.euler_state_name: guard.lower,
                    },
                )
                low = self.resources(guard=guard, physical=physical, query=left)
                high = self.resources(guard=guard, physical=physical, query=right)
                _, _, utility_valid = self.utility.coefficients(
                    pool={**guard.pool, **physical}, dtype=left.dtype
                )
                supported, owner, corner = _classify(
                    conditional=conditional,
                    utility_of_action=utility_of_action,
                    lower=low,
                    upper=high,
                )
                valid = conditional.saturated & utility_valid & supported
                if query is not None:
                    actual = self.solver(guard=guard).node(
                        liquid=query, physical=physical, physical_tangent=tangent
                    )
                    selected = actual.output
                    valid &= (
                        selected.valid
                        & (selected.native_owner == owner)
                        & (selected.side_id == 0)
                        & (selected.corner == corner)
                    )
                return valid

            return jax.lax.cond(active, check, lambda: jnp.ones((), dtype=jnp.bool_))

        matches = jax.lax.map(
            one, (nodes.physical, nodes.physical_tangent, needed, conditionals)
        )
        return nodes.law_valid & (left < right) & jnp.all(matches)

    def validate(self, *, carry: ProjectedEGMCarry, query: ScalarFloat) -> ScalarBool:
        if not isinstance(carry, BalancedCaseUCarry):
            raise TypeError("U source guard requires its original numeric operands.")
        guard = carry.guard
        active = jnp.arange(carry.grid.size) < carry.valid_length
        exact = jnp.any(active & (carry.grid == query))
        search = jnp.where(active, carry.grid, jnp.inf)
        index = jnp.clip(
            jnp.searchsorted(search, query, side="right") - 1, 0, carry.grid.size - 2
        )
        supported = jax.lax.cond(
            exact,
            lambda: jnp.ones((), dtype=jnp.bool_),
            lambda: (
                carry.link_valid[index]
                & self.cell(
                    guard=guard,
                    left=carry.grid[index],
                    right=carry.grid[index + 1],
                    query=query,
                )
            ),
        )
        # Reference levels never replace the acquired PWA channels.
        return (
            supported
            & self.reference(guard=guard, query=query).valid
            & (guard.pool["period"] == self.point.period)
        )

    def publish(
        self, *, initial: ProjectedEGMCarry, guard: CaseUBalancedOperands
    ) -> BalancedCaseUCarry:
        """Union actual conditional native cuts, classify, then refine economics."""
        capacity = initial.grid.size
        nodes = self.nodes(guard=guard)
        needed = _needed(nodes=nodes)
        # Query-independent per node; every cut and cell check below reuses it.
        conditionals = self.conditionals(guard=guard)

        def cuts(
            inputs: tuple[
                Mapping[ShockName, ScalarFloat | ScalarInt],
                ScalarBool,
                _ConditionalRow,
            ],
        ) -> tuple[Float1D, ScalarBool]:
            physical, active, conditional = inputs

            def produce() -> tuple[Float1D, ScalarBool]:
                row, saturated = conditional.envelope, conditional.saturated

                def pull(cut: ScalarFloat) -> tuple[Float1D, ScalarBool]:
                    def resource(query: ScalarFloat) -> ScalarFloat:
                        return self.resources(
                            guard=guard, physical=physical, query=query
                        )

                    ge = pullback_resource_boundary(
                        resources=resource,
                        boundary=cut,
                        lower=guard.lower,
                        upper=guard.upper,
                    )
                    gt = pullback_resource_boundary(
                        resources=resource,
                        boundary=cut,
                        lower=guard.lower,
                        upper=guard.upper,
                        strict=True,
                    )
                    points = jnp.stack(
                        (
                            jnp.nextafter(ge.point, guard.lower),
                            ge.point,
                            jnp.nextafter(gt.point, guard.lower),
                            gt.point,
                        )
                    )
                    active_cut = jnp.stack(
                        (ge.interior, ge.interior, gt.interior, gt.interior)
                    )
                    return jnp.where(active_cut, points, jnp.inf), ge.valid & gt.valid

                points, valid = jax.lax.map(
                    pull,
                    jnp.where(
                        jnp.arange(row.grid.size) < row.n_kept, row.grid, row.grid[0]
                    ),
                )
                keep = jnp.arange(row.grid.size) < row.n_kept
                return jnp.where(keep[:, None] & saturated, points, jnp.inf).reshape(
                    -1
                ), jnp.all(~keep | valid)

            return jax.lax.cond(
                active,
                produce,
                lambda: (
                    jnp.full(
                        (4 * self.point.pieces.n_pad,),
                        jnp.inf,
                        dtype=initial.grid.dtype,
                    ),
                    jnp.ones((), dtype=jnp.bool_),
                ),
            )

        extra, valid = jax.lax.map(cuts, (nodes.physical, needed, conditionals))
        raw = jnp.concatenate((initial.grid, extra.reshape(-1)))
        active = jnp.concatenate(
            (
                jnp.arange(capacity) < initial.valid_length,
                jnp.isfinite(extra.reshape(-1)),
            )
        )
        points, length, retained = _ordered_points(
            raw=raw, active=active, capacity=capacity
        )
        grid = jnp.where(jnp.arange(capacity) < length, points, initial.domain_lower)
        values = jax.vmap(
            lambda query: self.hoisted_reference(
                guard=guard, conditionals=conditionals, query=query
            )
        )(grid)
        kinds = jnp.where(
            jnp.arange(capacity) < length,
            jnp.where(
                jnp.isneginf(values.value),
                QueryNodeKind.ECONOMIC_MINUS_INF,
                QueryNodeKind.FINITE,
            ),
            QueryNodeKind.PAD,
        ).astype(jnp.int32)
        row = replace(
            initial,
            grid=grid,
            value=jnp.where(jnp.arange(capacity) < length, values.value, 0),
            economic_marginal=jnp.where(
                jnp.arange(capacity) < length, values.economic_marginal, 0
            ),
            owner_id=jnp.zeros_like(initial.owner_id),
            side_id=jnp.zeros_like(initial.side_id),
            node_kind=kinds,
            valid_length=length,
            valid=initial.valid
            & retained
            & jnp.all(valid)
            & jnp.all((jnp.arange(capacity) >= length) | values.valid),
        )
        links = jax.vmap(
            lambda left, right: self.cell(
                guard=guard, left=left, right=right, conditionals=conditionals
            )
        )(grid[:-1], grid[1:])
        links = (jnp.arange(capacity - 1) < length - 1) & links
        components = jnp.cumsum(
            jnp.concatenate(
                (jnp.zeros((1,), dtype=jnp.int32), (~links).astype(jnp.int32))
            ),
            dtype=jnp.int32,
        )
        # IDs describe connected PWA regions, not a collapsed native roster.
        # Original native ownership is checked by validate() ex post only.
        row = replace(
            row,
            link_valid=links,
            owner_id=jnp.where(jnp.arange(capacity) < length, components, -1),
        )

        def owned_reference(query: ScalarFloat) -> ProjectedRowRead:
            # Refinement only bisects cells of this row, so the owning region
            # is the one the query falls in here.
            read = self.hoisted_reference(
                guard=guard, conditionals=conditionals, query=query
            )
            search = jnp.where(
                jnp.arange(capacity) < row.valid_length, row.grid, jnp.inf
            )
            index = jnp.clip(
                jnp.searchsorted(search, query, side="right") - 1, 0, capacity - 2
            )
            return read._replace(owner_id=row.owner_id[index])

        refined, _, slopes = refine_projected_row(
            initial=row,
            reference=owned_reference,
            accuracy="balanced",
            value_slope=lambda query: self.value_slope(
                guard=guard, conditionals=conditionals, query=query
            ),
        )
        if slopes is None:
            raise TypeError("Case-U refinement lost its value slopes.")
        return BalancedCaseUCarry(
            **{
                field.name: getattr(refined, field.name)
                for field in fields(ProjectedEGMCarry)
            },
            value_slope=slopes,
            guard=guard,
        )


def _needed(*, nodes: CaseUCurrentNodes) -> BoolND:
    live = jnp.all(~is_represented_zero(nodes.factors), axis=0)
    return live | _derivative_needed(nodes=nodes)


def _derivative_needed(*, nodes: CaseUCurrentNodes) -> BoolND:
    """Nodes whose value enters only through a differentiated probability factor."""
    needed = jnp.zeros(nodes.factors.shape[1:], dtype=jnp.bool_)
    for column in range(nodes.factors.shape[0]):
        others = jnp.concatenate(
            (nodes.factors[:column], nodes.factors[column + 1 :]), axis=0
        )
        needed |= ~is_represented_zero(nodes.factor_tangent[column]) & jnp.all(
            ~is_represented_zero(others), axis=0
        )
    return needed


def _with_tangent(*, nodes: CaseUCurrentNodes, column: int) -> FloatND:
    """The factor bank with one factor replaced by its liquid derivative."""
    return jnp.concatenate(
        (
            nodes.factors[:column],
            nodes.factor_tangent[column : column + 1],
            nodes.factors[column + 1 :],
        ),
        axis=0,
    )


def _select(
    *, node: _ConstructedAssetNode, row: OwnedEnvelope, resources: ScalarFloat
) -> tuple[ScalarInt, ScalarBool, ScalarBool]:
    """Select the original owner of one constructed node at ``resources``."""
    return _select_conditional(
        conditional=_ConditionalRow.of(
            node=node, envelope=row, saturated=jnp.ones((), dtype=jnp.bool_)
        ),
        utility_of_action=node.utility_of_action,
        resources=resources,
    )


def _select_conditional(
    *,
    conditional: _ConditionalRow,
    utility_of_action: BoundUtilityOfAction,
    resources: ScalarFloat,
) -> tuple[ScalarInt, ScalarBool, ScalarBool]:
    row = conditional.envelope
    bracket = _bracket_from_owned_row(refined=row, x_query=resources)
    value, policy, corner = _publish_node_selection(
        bracket=bracket.bracket,
        n_pad=row.grid.size,
        resources_at_node=resources,
        borrowing_limit=conditional.borrowing_limit,
        utility_of_action=utility_of_action,
        discounted_expected_value_at_limit=conditional.discounted_floor_value,
    )
    saving = jnp.where(corner, conditional.borrowing_limit, bracket.savings)
    valid = (
        ~bracket.construction_invalid
        & jnp.isfinite(value)
        & jnp.isfinite(policy)
        & (policy > 0)
        & (saving >= conditional.borrowing_limit)
        & (saving <= conditional.savings_ceiling)
        & (corner | bracket.read_supported)
    )
    return jnp.where(corner, -1, bracket.owner_id), corner, valid


def _classify(
    *,
    conditional: _ConditionalRow,
    utility_of_action: BoundUtilityOfAction,
    lower: ScalarFloat,
    upper: ScalarFloat,
) -> tuple[ScalarBool, ScalarInt, ScalarBool]:
    row = conditional.envelope
    search = jnp.where(jnp.arange(row.grid.size) < row.n_kept, row.grid, jnp.inf)
    index = jnp.clip(
        jnp.searchsorted(search, lower, side="right") - 1, 0, row.grid.size - 2
    )
    x0, x1 = row.grid[index], row.grid[index + 1]
    c0, c1 = row.policy[index], row.policy[index + 1]
    h, _, _, applicable = materialize_hermite_shadow(
        xp_lower=x0,
        xp_upper=x1,
        fp_lower=row.value[index],
        fp_upper=row.value[index + 1],
        slope_lower=_node_value_slope(
            policy=c0, value=row.value[index], utility_of_action=utility_of_action
        ),
        slope_upper=_node_value_slope(
            policy=c1,
            value=row.value[index + 1],
            utility_of_action=utility_of_action,
        ),
    )
    # Ordinary rounded legality of the piece's node savings; the cell links
    # when both ends then keep one owner/corner selection. No interior proof.
    floor, ceiling = conditional.borrowing_limit, conditional.savings_ceiling
    s0, s1 = x0 - c0, (x0 + h) - c1
    legal = (
        (c0 > 0)
        & (c1 > 0)
        & (jnp.minimum(s0, s1) >= floor)
        & (jnp.maximum(s0, s1) <= ceiling)
    )
    left_owner, left_corner, left_valid = _select_conditional(
        conditional=conditional, utility_of_action=utility_of_action, resources=lower
    )
    right_owner, right_corner, right_valid = _select_conditional(
        conditional=conditional, utility_of_action=utility_of_action, resources=upper
    )
    geometry = (
        (lower < upper)
        & (lower >= x0)
        & (upper <= x1)
        & (index + 1 < row.n_kept)
        & applicable
        & row.link_valid[index]
        & (row.owner_id[index] == row.owner_id[index + 1])
    )
    same = (left_owner == right_owner) & (left_corner == right_corner)
    return (
        geometry & legal & same & left_valid & right_valid,
        left_owner,
        left_corner,
    )


type _UGuardChild = PytreeChild | CaseUBalancedOperands | ContinuationPayload


type _UGuardMetadata = tuple[
    tuple[ReferenceName, ...],
    tuple[RegimeName, ...],
    tuple[ShockGroupName, ...],
    tuple[tuple[ReferenceName, ...], ...],
    tuple[ShockGroupName, ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[ShockName, ...] | None, ...], ...],
    bool,
]
_UGUARD_METADATA_FIELDS = (
    "pool_names",
    "child_names",
    "group_names",
    "parameter_names",
    "node_group_names",
    "node_names",
    "code_names",
    "has_floor",
)


def _flatten_guard(
    value: CaseUBalancedOperands,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _UGuardChild], ...], _UGuardMetadata]:
    return tuple(
        (jax.tree_util.GetAttrKey(field.name), getattr(value, field.name))
        for field in fields(value)
        if field.name not in _UGUARD_METADATA_FIELDS
    ), (
        value.pool_names,
        value.child_names,
        value.group_names,
        value.parameter_names,
        value.node_group_names,
        value.node_names,
        value.code_names,
        value.has_floor,
    )


def _flatten_carry(
    value: BalancedCaseUCarry,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _UGuardChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(field.name), getattr(value, field.name))
        for field in fields(value)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_guard(
    metadata: _UGuardMetadata, children: Iterable[_UGuardChild]
) -> CaseUBalancedOperands:
    result = object.__new__(CaseUBalancedOperands)
    for name, value in zip(_UGUARD_METADATA_FIELDS, metadata, strict=True):
        object.__setattr__(result, name, value)
    dynamic = (
        field
        for field in fields(CaseUBalancedOperands)
        if field.name not in _UGUARD_METADATA_FIELDS
    )
    for field, child in zip(dynamic, children, strict=True):
        object.__setattr__(result, field.name, child)
    return result


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_carry(
    metadata: None, children: Iterable[_UGuardChild]
) -> BalancedCaseUCarry:
    del metadata
    result = object.__new__(BalancedCaseUCarry)
    for field, child in zip(fields(BalancedCaseUCarry), children, strict=True):
        object.__setattr__(result, field.name, child)
    return result


jax.tree_util.register_pytree_with_keys(
    CaseUBalancedOperands, _flatten_guard, _unflatten_guard
)
jax.tree_util.register_pytree_with_keys(
    BalancedCaseUCarry, _flatten_carry, _unflatten_carry
)


_CURRENT_IID_ROW_RANK = 2


def prepare_saturated_case_u(  # noqa: C901, PLR0911, PLR0912 - exact-route capability boundaries
    *,
    context: SolverBuildContext,
    solver: _BoundDCEGM,
    build: EGMStepBuild,
    period: int,
    point: CaseUProgramFunction,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
) -> SaturatedCaseUProgram | None:
    """Keep unsupported U economics on their original exact-coordinate route.

    This is a declaration/schema proof, never an inference from sampled owner
    agreement. The query-dependent span is checked on real producing operands.
    """
    numerical = build.steps[period]
    if not isinstance(numerical, _EGMStep) or not isinstance(
        solver.envelope, ExactEnvelope
    ):
        return None
    declared = arguments.get(solver.continuous_state)
    indices = arguments.get("_lcm_current_iid_rows")
    demands = arguments.get("_lcm_parent_demand_savings", ())
    if (
        not isinstance(declared, (jax.Array, jax.ShapeDtypeStruct))
        or declared.ndim != 1
        or not isinstance(indices, (jax.Array, jax.ShapeDtypeStruct))
        or indices.ndim != _CURRENT_IID_ROW_RANK
        or not isinstance(demands, tuple)
    ):
        return None
    demand_width = 0
    for demand in demands:
        if (
            not isinstance(demand, (jax.Array, jax.ShapeDtypeStruct))
            or demand.ndim != 1
        ):
            return None
        demand_width += demand.shape[0]
    # Each original conditional native knot contributes >=/> pullbacks and
    # their predecessors. This bound uses actual input widths, never capacity.
    mandatory_bound = (
        declared.shape[0] + demand_width + 4 * point.pieces.n_pad * indices.shape[0]
    )
    if solver.current_iid_projection_capacity < mandatory_bound:
        return None
    proof = prove_resources_only(context=context, solver=solver, plan=point.plan)
    dependencies = dict(proof.root_dependencies)
    # At the actual future stage savings is an independent candidate input.
    # Utility/inverse/current roots retain their complete original ancestry.
    stage_functions = {
        name: function
        for name, function in context.solve_functions.items()
        if name not in (solver.resources, solver.post_decision_function)
    }
    for _, label, function in _savings_stage_candidates(
        user_regime=context.user_regimes[context.regime_name],
        law=context.laws[context.regime_name],
        solver=solver,
    ):
        dependencies[label] = _dag_ancestors(
            functions=stage_functions, target_func=function
        )
    if (
        proof.unsupported
        or solver.continuous_state in proof.context_inputs
        or any(
            solver.resources in names or solver.continuous_state in names
            for names in dependencies.values()
        )
        or numerical.pieces.feasibility_func is not None
    ):
        return None
    pool = {
        name: value
        for name, value in arguments.items()
        if name != solver.continuous_state
    }
    try:
        _, _, physical_dtypes = _initial_resources_coefficient(
            numerical=numerical,
            plan=point.plan,
            actual_pool=pool,
        )
        abstract = {}
        for name in inspect.signature(point.pieces.utility_func).parameters:
            if name == point.pieces.action_name:
                value = jax.ShapeDtypeStruct((), point.pieces.savings_nodes.dtype)
            elif name in physical_dtypes:
                value = jax.ShapeDtypeStruct((), physical_dtypes[name])
            elif name in pool:
                actual = pool[name]
                if not isinstance(actual, (jax.Array, jax.ShapeDtypeStruct)):
                    return None
                value = actual
            else:
                return None
            if value.shape != ():
                return None
            abstract[name] = value
        utility = LogUtilityProgram.authenticate(
            function=point.pieces.utility_func,
            action_name=point.pieces.action_name,
            abstract_pool=abstract,
        )
    except UnsupportedOperationError:
        # Unsupported structure keeps exact U; no numerical producer ran here.
        return None
    return SaturatedCaseUProgram(
        point=point,
        utility=utility,
        refine=get_upper_envelope(
            solver=solver, n_refined=point.pieces.n_pad, publish_provenance=True
        ),
    )
