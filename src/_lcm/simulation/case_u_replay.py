"""Native realised-frame Case-U reconstruction using the original node solver."""

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.current_iid import AdmittedGroupParams, CurrentIIDDrawPlan
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u import CaseUPointSolver
from _lcm.egm.current_iid_case_u_replay import (
    CaseUReplayChildren,
    CaseUReplayInput,
    CaseUReplayPeriod,
)
from _lcm.engine import Regime
from _lcm.shock_overrides import CurrentIIDOverride
from _lcm.simulation.case_r_replay import case_r_status
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.current_iid_case_u_build import _bind_numeric_pieces
from _lcm.solution.feasibility import BoundedResult, BoundInputStatus
from _lcm.typing import (
    ConstraintFunction,
    EconFunctionKwargs,
    FlatParams,
    PRNGKeyND,
    PytreeChild,
    PytreeValue,
    QAndFArg,
    ShapeDtypePytree,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.shocks import BaseInnovations, PhysicalValues, ShockName
from lcm.typing import (
    ActionName,
    BoolND,
    FloatND,
    IntND,
    ReferenceName,
    ScalarFloat,
)


class CaseUReplayOutput(NamedTuple):
    """Original action, same realised frame and mandatory construction summaries."""

    actions: Mapping[ActionName, FloatND | IntND]
    value: FloatND
    physical: Mapping[ShockName, FloatND | IntND]
    law_invalid: BoolND
    invalid: BoolND
    feasible: BoolND
    native_owner: IntND
    side_id: IntND
    corner: BoolND


# NamedTuple is already a JAX node; semantic keys are needed by role/materializer
# paths, so register its actual field layout with trusted abstract reconstruction.
type _CaseUReplayChild = PytreeChild | Mapping[ReferenceName, PytreeChild]


def _flatten_case_u_output(
    output: CaseUReplayOutput,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _CaseUReplayChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), child)
        for name, child in zip(CaseUReplayOutput._fields, output, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_case_u_output(
    metadata: None, children: Iterable[_CaseUReplayChild]
) -> CaseUReplayOutput:
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(CaseUReplayOutput._fields):
        raise TypeError("Case-U reconstruction lost its actual output field roster.")
    return cast("CaseUReplayOutput", tuple.__new__(CaseUReplayOutput, leaves))


jax.tree_util.register_pytree_with_keys(
    CaseUReplayOutput, _flatten_case_u_output, _unflatten_case_u_output
)


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUActionReplay:
    """One genuine original-node constructor over a single addressed subject."""

    declaration: CaseUReplayPeriod
    draw_plan: CurrentIIDDrawPlan
    feasibility: ConstraintFunction

    def __call__(  # noqa: C901, PLR0912, PLR0915 - genuine source branches share one actual frame
        self,
        *,
        future: tuple[EGMCarry, ...],
        states: Mapping[ReferenceName, FloatND | IntND],
        keys: Mapping[ShockName, PRNGKeyND],
        overrides: Mapping[ShockName, CurrentIIDOverride],
        params: EconFunctionKwargs,
        group_params: AdmittedGroupParams,
        period: IntND,
        age: FloatND | IntND,
        savings_nodes: FloatND,
        borrowing_limit: ScalarFloat,
        floor_status: BoundInputStatus | None,
        child_node_values: tuple[tuple[FloatND | IntND, ...], ...],
        child_age_values: tuple[FloatND | IntND, ...],
        nb_source_grid: FloatND | None,
        negm_source_grid: FloatND | None = None,
        negm_outer_nodes: FloatND | None = None,
    ) -> CaseUReplayOutput:
        declaration = self.declaration
        payload = CaseUReplayChildren(
            source=declaration.future.source,
            period=declaration.future.period,
            child_names=tuple(declaration.future.specs),
            child_values=future,
        )
        actual_overrides = {}
        active = {}
        for name, supplied in overrides.items():
            value, mask = supplied.values, supplied.active
            if isinstance(value, jax.ShapeDtypeStruct) or isinstance(
                mask, jax.ShapeDtypeStruct
            ):
                raise ExecutionPlanningError(
                    "Case-U override descriptors must be traced before execution."
                )
            actual_overrides[name] = (
                BaseInnovations(values=value)
                if supplied.kind == "base"
                else PhysicalValues(values=value)
            )
            active[name] = jnp.asarray(mask)
            if active[name].dtype != jnp.bool_:
                raise ExecutionPlanningError(
                    "Case-U override activation is not boolean."
                )
        pool = {**params, "period": period, "age": age}
        frame = self.draw_plan.realise(
            keys=keys,
            arguments=cast("Mapping[ReferenceName, QAndFArg]", {**pool, **states}),
            group_params=group_params,
            overrides=actual_overrides,
            override_active=active,
        )
        if declaration.negm_point is not None:
            if (
                negm_source_grid is None
                or negm_outer_nodes is None
                or floor_status is not None
            ):
                raise TypeError(
                    "Finite NEGM replay lost its original numerical operands."
                )
            point = declaration.negm_point.with_operands(
                source_grid=negm_source_grid,
                savings_nodes=savings_nodes,
                borrowing_limit=borrowing_limit,
                outer_nodes=negm_outer_nodes,
            )
            nested = point.select(
                liquid=states[point.euler_state_name],
                physical=frame.physical,
                pool={**pool, **{name: states[name] for name in point.passive_names}},
                children=payload.children,
            )
            original, outer_action, _outer_index = nested.output
            invalid = ~original.valid | (period != declaration.future.period)
            for status in (*nested.bound_status, *(nested.query_status or ())):
                invalid |= jnp.any(jnp.stack(tuple(status)))
            actions = {
                point.action_name: original.policy,
                point.outer_action: outer_action,
            }
            actual = {**pool, **states, **frame.physical, **actions}
            feasible = self.feasibility(
                **{
                    name: actual[name]
                    for name in inspect.signature(self.feasibility).parameters
                }
            )
            return CaseUReplayOutput(
                actions=actions,
                value=original.value,
                physical=dict(frame.physical),
                law_invalid=~frame.valid,
                invalid=invalid,
                feasible=original.feasible & feasible,
                native_owner=original.native_owner,
                side_id=original.side_id,
                corner=original.corner,
            )
        if declaration.nbegm_point is not None:
            from _lcm.egm.current_iid_nbegm import NBCaseURidePoint  # noqa: PLC0415

            if nb_source_grid is None:
                raise TypeError("NB U replay lost its original interval support.")
            point = declaration.nbegm_point.with_savings(nodes=savings_nodes)
            point = replace(point, source_grid=nb_source_grid)
            selected_actions = {}
            if isinstance(point, NBCaseURidePoint):
                ride = point.select(
                    liquid=states[point.euler_state_name],
                    physical=frame.physical,
                    pool={
                        **pool,
                        **{name: states[name] for name in point.passive_names},
                    },
                    children=payload.children,
                )
                selected = BoundedResult(
                    output=ride.output.original,
                    bound_status=ride.bound_status,
                    query_status=ride.query_status,
                    continuation_read_counts=ride.continuation_read_counts,
                )
                selected_actions = {
                    name: ride.output.action_codes[index]
                    for index, name in enumerate(point.discrete_action_names)
                }
            else:
                selected = point.node(
                    liquid=states[point.euler_state_name],
                    physical=frame.physical,
                    pool=pool,
                    children=payload.children,
                )
            invalid = ~selected.output.valid | (period != declaration.future.period)
            for status in (*selected.bound_status, *(selected.query_status or ())):
                invalid |= jnp.any(jnp.stack(tuple(status)))
            actual = {
                **pool,
                **states,
                **frame.physical,
                point.action_name: selected.output.policy,
                **selected_actions,
            }
            feasible = self.feasibility(
                **{
                    name: actual[name]
                    for name in inspect.signature(self.feasibility).parameters
                }
            )
            return CaseUReplayOutput(
                actions={point.action_name: selected.output.policy, **selected_actions},
                value=selected.output.value,
                physical=dict(frame.physical),
                law_invalid=~frame.valid,
                invalid=invalid,
                feasible=selected.output.feasible & feasible,
                native_owner=selected.output.native_owner,
                side_id=selected.output.side_id,
                corner=selected.output.corner,
            )
        if declaration.pieces is None:
            raise TypeError("Case-U replay lost its original solver constructor.")
        pieces = _bind_numeric_pieces(
            original=declaration.pieces,
            savings_nodes=savings_nodes,
            borrowing_limit=borrowing_limit,
            floor_status=floor_status,
            child_node_values=child_node_values,
        )
        reads = pieces.continuation_plan.child_reads
        if len(reads) != len(child_age_values):
            raise TypeError("Case-U reconstruction lost its original child clocks.")
        pieces = replace(
            pieces,
            continuation_plan=replace(
                pieces.continuation_plan,
                child_reads=MappingProxyType(
                    {
                        target: replace(read, age_values=ages)
                        for (target, read), ages in zip(
                            reads.items(), child_age_values, strict=True
                        )
                    }
                ),
            ),
        )
        # Conditional realised-frame reconstruction holds the physical values
        # fixed. The unconditional solve still uses the original law tangent.
        physical = frame.physical
        tangent = {
            name: jnp.zeros_like(value)
            for name, value in physical.items()
            if jnp.issubdtype(value.dtype, jnp.floating)
        }
        selected = CaseUPointSolver(
            pieces=pieces,
            pool={
                **pool,
                **{
                    name: value
                    for name, value in states.items()
                    if name != pieces.euler_state_name
                },
            },
            children=MappingProxyType(dict(payload.children)),
            savings_point_width=declaration.savings_point_width,
            stochastic_node_width=declaration.stochastic_node_width,
            envelope_cell_width=declaration.envelope_cell_width,
        ).node(
            liquid=states[pieces.euler_state_name],
            physical=physical,
            physical_tangent=tangent,
        )
        invalid = ~selected.output.valid | (period != declaration.future.period)
        for status in (*selected.bound_status, *(selected.query_status or ())):
            invalid |= jnp.any(jnp.stack(tuple(status)))
        actual = {
            **pool,
            **states,
            **frame.physical,
            pieces.action_name: selected.output.policy,
        }
        feasible = self.feasibility(
            **{
                name: actual[name]
                for name in inspect.signature(self.feasibility).parameters
            }
        )
        return CaseUReplayOutput(
            actions={pieces.action_name: selected.output.policy},
            value=selected.output.value,
            physical=dict(frame.physical),
            law_invalid=~frame.valid,
            invalid=invalid,
            feasible=selected.output.feasible & feasible,
            native_owner=selected.output.native_owner,
            side_id=selected.output.side_id,
            corner=selected.output.corner,
        )


def case_u_arguments(
    *,
    regime: Regime,
    replay: CaseUReplayInput,
    future: PytreeValue | ShapeDtypePytree,
    states: PytreeValue | ShapeDtypePytree,
    keys: PytreeValue | ShapeDtypePytree,
    overrides: PytreeValue | ShapeDtypePytree,
    flat_params: FlatParams,
    period: int,
    period_value: PytreeValue | ShapeDtypePytree,
    age: PytreeValue | ShapeDtypePytree,
) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Dispatch original fixed/runtime/target/law/clock and future operands."""
    spec = regime.simulation.case_u_replay_spec
    plan = regime.simulation.current_iid_draw_plan
    if spec is None or plan is None:
        raise ExecutionPlanningError("Case-U reconstruction lacks its actual route.")
    replay.require(spec=spec, period=period)
    if not isinstance(future, CaseUReplayChildren):
        raise TypeError("Case-U reconstruction lost its acquired future payload.")
    declaration = spec.periods[period]
    pieces = declaration.pieces
    nb = declaration.nbegm_point
    nested = declaration.negm_point
    if nested is not None:
        savings = nested.savings_nodes
        floor, status = nested.borrowing_limit, None
        continuation = nested.keeper.continuation_plan
    elif nb is not None:
        savings = nb.savings_nodes
        floor, status = nb.borrowing_limit, None
        continuation = nb.continuation_plan
    elif pieces is not None:
        savings = pieces.savings_nodes
        floor, status = pieces.borrowing_limit, pieces.floor_status
        continuation = pieces.continuation_plan
    else:
        raise TypeError("Case-U replay lacks its original solver constructor.")
    fixed = declaration.fixed_flat_params
    return MappingProxyType(
        {
            "future": future.child_values,
            "states": states,
            "keys": keys,
            "overrides": overrides,
            "params": {
                **union_fixed_params(
                    fixed_flat_params=fixed,
                    regime_name=regime.name,
                    transition_target_names=declaration.transition_target_names,
                ),
                **union_free_params(
                    flat_params=flat_params,
                    regime_name=regime.name,
                    transition_target_names=declaration.transition_target_names,
                ),
            },
            "group_params": {
                name: {**fixed.get(name, {}), **flat_params.get(name, {})}
                for name in plan.bindings
            },
            "period": period_value,
            "age": age,
            "savings_nodes": savings,
            "borrowing_limit": floor,
            "floor_status": status,
            "child_node_values": tuple(
                read.stochastic_node_values
                for read in continuation.child_reads.values()
            ),
            "child_age_values": tuple(
                read.age_values for read in continuation.child_reads.values()
            ),
            "nb_source_grid": None if nb is None else nb.source_grid,
            "negm_source_grid": None if nested is None else nested.source_grid,
            "negm_outer_nodes": None if nested is None else nested.outer_nodes,
        }
    )


class CaseUReplayProfile(NamedTuple):
    """Actual compiler result descriptors, separate from numerical return values."""

    actions: Mapping[str, jax.ShapeDtypeStruct]
    value: jax.ShapeDtypeStruct
    physical: Mapping[str, jax.ShapeDtypeStruct]
    law_invalid: jax.ShapeDtypeStruct
    invalid: jax.ShapeDtypeStruct
    feasible: jax.ShapeDtypeStruct
    native_owner: jax.ShapeDtypeStruct
    side_id: jax.ShapeDtypeStruct
    corner: jax.ShapeDtypeStruct


def case_u_profile_output(*, output: ShapeDtypePytree) -> CaseUReplayProfile:
    """Consume this original constructor's actual compiled output schema."""
    from _lcm.simulation.case_r_replay import (  # noqa: PLC0415
        _profile_leaf,
        _profile_mapping,
    )

    if not isinstance(output, tuple):
        raise ExecutionPlanningError("Case-U profile omitted its actual output record.")
    children = tuple(output)
    if type(output) is not CaseUReplayOutput or len(children) != len(
        CaseUReplayOutput._fields
    ):
        raise ExecutionPlanningError("Case-U profile has a foreign output schema.")
    actions, value, physical, law, invalid, feasible, owner, side, corner = children
    return CaseUReplayProfile(
        actions=_profile_mapping(values=actions),
        value=_profile_leaf(value=value),
        physical=_profile_mapping(values=physical),
        law_invalid=_profile_leaf(value=law),
        invalid=_profile_leaf(value=invalid),
        feasible=_profile_leaf(value=feasible),
        native_owner=_profile_leaf(value=owner),
        side_id=_profile_leaf(value=side),
        corner=_profile_leaf(value=corner),
    )


# The host summary operates on generic law/integrity/feasibility flags only.
case_u_status = case_r_status
