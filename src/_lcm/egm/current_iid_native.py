"""Private single-row native Case-R producer and original-candidate replay.

Public admission remains closed until the solver's graph, statuses, producer
schemas and replay routes install these callables together.
"""

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import NamedTuple, NotRequired, TypedDict, cast

import jax
import jax.numpy as jnp

from _lcm.current_iid import AdmittedGroupParams, CurrentIIDPlan, NodeOperands
from _lcm.egm.carry import EGMCarry
from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.current_iid_balanced import BalancedResourcesProgram
from _lcm.egm.current_iid_demands import (
    ParentResourcesDemand,
    prepare_parent_resources_demands,
)
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluator
from _lcm.egm.current_iid_publication import (
    ProjectedResourcesReference,
    assemble_exact_resources_samples,
    publish_balanced_resources,
)
from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.egm.current_iid_topology import (
    pullback_native_partition,
    require_resources_topology,
)
from _lcm.egm.nbegm_resources import NBEGMResourcesBank
from _lcm.egm.nbegm_resources_read import read_nbegm_resources_bank
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.step_core import OwnedResourcesRow, _EgmKernelPieces, _SolveOneCombo
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.probability import is_represented_zero
from _lcm.typing import EconFunctionKwargs, PytreeChild, QAndFArg
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import ShockName
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


class NativeCaseRInputs(TypedDict):
    """Actual keyword tree used by native Case-R materialization and tracing."""

    liquid_grid: Float1D
    pool: EconFunctionKwargs
    next_regime_to_continuation: MappingProxyType[
        RegimeName, EGMCarry | ProjectedEGMCarry
    ]
    law_arguments: Mapping[ReferenceName, QAndFArg]
    group_params: AdmittedGroupParams
    node_operands: NodeOperands
    indices: IntND
    savings_point_width: int | None
    stochastic_node_width: int | None
    envelope_cell_width: int
    demand_pools: NotRequired[tuple[EconFunctionKwargs, ...]]
    demand_savings: NotRequired[tuple[Float1D, ...]]


type _NativeCaseRChild = (
    PytreeChild
    | ProjectedEGMCarry
    | OwnedResourcesRow
    | tuple[OwnedResourcesRead, BoolND]
)


class NativeCaseROutput(NamedTuple):
    """Separate public value, projected child channels and original replay row."""

    value: Float1D
    projected: ProjectedEGMCarry
    resources_row: OwnedResourcesRow
    valid: ScalarBool
    current_iid_invalid: ScalarBool
    empty_count: ScalarInt
    other_inf_count: ScalarInt
    selection: tuple[OwnedResourcesRead, BoolND]
    projection_records_needed: ScalarInt
    """Least record count the projected row needed; above capacity only if exhausted."""

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[_NativeCaseRChild]], NativeCaseROutput],
        tuple[None, tuple[_NativeCaseRChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_native_case_r_output, (None, tuple(self))


@dataclass(frozen=True, kw_only=True, eq=False)
class BoundResourcesMap:
    """Actual resources callback with fixed context and explicit physical input."""

    function: Callable[..., FloatND]
    """Same canonical concatenated function certified by the model builder."""
    liquid_state: ReferenceName
    """Original physical pre-shock liquid state name."""
    pool: EconFunctionKwargs
    """Actual call operands; no target buffers are captured in model statics."""
    names: tuple[ReferenceName, ...]
    """Canonical signature, fixed once by the native builder."""

    # keyword-only-exempt: library-callback=typing.Callable
    def __call__(
        self, liquid: ScalarFloat, physical: Mapping[ShockName, ScalarFloat | ScalarInt]
    ) -> ScalarFloat:
        values = {**self.pool, **physical, self.liquid_state: liquid}
        return self.function(**{name: values[name] for name in self.names})


@dataclass(frozen=True, kw_only=True, eq=False)
class NativeCaseRCore:
    """A complete invocation of the canonical single-row economic producer.

    The first supported shape has one liquid state and no other state/action
    combo axes. Current law banks and child artifacts are dynamic arguments;
    only canonical callbacks, numerical capacities and the certified rational
    resources coefficient belong to model statics.
    """

    pieces: _EgmKernelPieces
    """Canonical producer pieces, including the complete future expectation."""
    plan: CurrentIIDPlan
    """Actual dated current-node law and joint-row plan."""
    coefficient: Fraction
    """Positive resources coefficient from structural and rounded-map proofs."""
    refine: Callable[..., OwnedEnvelope]
    """Existing owned ExactEnvelope backend, without query-authority fabrication."""
    owner_capacity: int
    """Real monotone-run capacity of that owned backend."""
    projection_capacity: int
    """Declared bounded output/refinement allocation, never grown after admission."""

    parent_demands: tuple[ParentResourcesDemand, ...] | None = None
    """Closed model consumer population; None retains the full-topology refusal."""

    balanced: BalancedResourcesProgram | None = None
    """Authenticated private balanced class; the public solver guard stays closed."""

    def __post_init__(self) -> None:
        if self.pieces.combo_names:
            raise UnsupportedOperationError(
                "Case-R passive-state and discrete-action row axes are not installed."
            )
        if self.coefficient <= 0:
            raise UnsupportedOperationError(
                "Case-R resources require a positive slope."
            )
        if self.projection_capacity < _MIN_PROJECTION_CAPACITY:
            raise ValueError(
                "Case-R projection capacity cannot represent its endpoints."
            )

    def __call__(
        self,
        *,
        liquid_grid: Float1D,
        pool: EconFunctionKwargs,
        next_regime_to_continuation: MappingProxyType[
            RegimeName, EGMCarry | ProjectedEGMCarry
        ],
        law_arguments: Mapping[ReferenceName, QAndFArg],
        group_params: AdmittedGroupParams,
        node_operands: NodeOperands,
        indices: IntND,
        savings_point_width: int | None,
        stochastic_node_width: int | None,
        envelope_cell_width: int,
        demand_pools: tuple[EconFunctionKwargs, ...] = (),
        demand_savings: tuple[Float1D, ...] = (),
    ) -> NativeCaseROutput:
        """Construct once, then select current-node policies before integration."""
        if self.parent_demands is None and self.balanced is None:
            require_resources_topology()
        physical, factors, law_valid = self.plan.realise_nodes(
            arguments=law_arguments,
            group_params=group_params,
            node_operands=node_operands,
            indices=indices,
        )
        resources = BoundResourcesMap(
            function=self.pieces.own_resources_func,
            liquid_state=self.pieces.euler_state_name,
            pool=pool,
            names=tuple(inspect.signature(self.pieces.own_resources_func).parameters),
        )
        # This common domain is derived from the declared pre-state grid and
        # every original joint quadrature row, not a row borrowed from node 0.
        live = jnp.all(~is_represented_zero(factors), axis=0)

        def resource_bounds(
            inputs: tuple[Mapping[ShockName, ScalarFloat | ScalarInt], ScalarBool],
        ) -> FloatND:
            draw, active = inputs
            return jax.lax.cond(
                active,
                lambda: jnp.stack(
                    (resources(liquid_grid[0], draw), resources(liquid_grid[-1], draw))
                ),
                lambda: jnp.asarray((jnp.inf, -jnp.inf), dtype=liquid_grid.dtype),
            )

        bounds = jax.lax.map(resource_bounds, (physical, live))
        common_domain = jnp.stack((jnp.min(bounds[:, 0]), jnp.max(bounds[:, 1])))
        construct = _SolveOneCombo(
            pieces=self.pieces,
            pool=pool,
            state_grid=liquid_grid,
            next_regime_to_continuation=next_regime_to_continuation,
            savings_point_width=savings_point_width,
            stochastic_node_width=stochastic_node_width,
            envelope_cell_width=envelope_cell_width,
            resolved_process_grids=MappingProxyType({}),
        )
        row = construct.owned_resources_row(
            combo_values=(), resources_domain=common_domain, refine=self.refine
        )
        utility = BoundUtilityOfAction(
            utility_func=self.pieces.utility_func,
            action_name=self.pieces.action_name,
            bound=pool,
        )
        evaluator = CurrentResourcesEvaluator(
            row=row,
            resources=resources,
            resources_slope=jnp.asarray(
                float(self.coefficient), dtype=liquid_grid.dtype
            ),
            utility_of_action=utility,
            owner_capacity=self.owner_capacity,
        )
        demanded_queries = jnp.empty((0,), dtype=liquid_grid.dtype)
        demands_valid = jnp.ones((), dtype=jnp.bool_)
        if self.parent_demands is not None:
            demanded_queries, demands_valid = prepare_parent_resources_demands(
                declarations=self.parent_demands,
                pools=demand_pools,
                savings=demand_savings,
                dtype=liquid_grid.dtype,
            )
        partition = pullback_native_partition(
            row=row,
            resources=resources,
            physical=physical,
            live=live,
            liquid_grid=liquid_grid,
            demanded_queries=demanded_queries,
            capacity=self.projection_capacity,
        )
        reference = ProjectedResourcesReference(
            evaluator=evaluator,
            physical=physical,
            factors=factors,
            law_valid=law_valid & demands_valid,
            partition=partition,
        )
        projected, selected = assemble_exact_resources_samples(
            reference=reference, capacity=self.projection_capacity
        )
        records_needed = partition.records_needed
        if self.balanced is not None:
            projected, records_needed = publish_balanced_resources(
                reference=reference,
                program=self.balanced,
                pool=pool,
                capacity=self.projection_capacity,
            )
        # Exact public solve nodes use their original selected candidates. The
        # affine projected carry serves future queries; it does not rebaseline V.
        node_values = jax.vmap(
            lambda liquid: evaluator.integrate(
                liquid=liquid,
                physical=physical,
                factors=factors,
                law_valid=law_valid,
            )
        )(liquid_grid)
        valid = projected.valid & jnp.all(node_values.valid)
        return NativeCaseROutput(
            value=node_values.value,
            projected=projected,
            resources_row=row,
            valid=valid,
            current_iid_invalid=~law_valid,
            empty_count=jnp.sum(node_values.empty, dtype=jnp.int32),
            other_inf_count=jnp.sum(
                ~node_values.empty & ~jnp.isfinite(node_values.value), dtype=jnp.int32
            ),
            selection=(selected, live),
            projection_records_needed=records_needed,
        )


def replay_case_r_candidate(
    *,
    row: OwnedResourcesRow | NBEGMResourcesBank,
    liquid: ScalarFloat,
    physical: Mapping[ShockName, ScalarFloat | ScalarInt],
    frame_valid: ScalarBool,
    resources: BoundResourcesMap,
    utility: BoundUtilityOfAction,
    coefficient: Fraction,
    owner_capacity: int,
    arithmetic: ComparisonArithmetic,
) -> OwnedResourcesRead:
    """Read original candidates with the caller's same addressed physical frame.

    This pure row reader is intended for the model-owned row/chunk replay
    program. It does not reconstruct a whole regime or sample another draw.
    The enclosing forward graph must retain this same frame for flow/transition.
    """
    if type(row) is NBEGMResourcesBank:
        selected = read_nbegm_resources_bank(
            bank=row,
            resources=resources(liquid, physical),
            arithmetic=arithmetic,
        )
        return replace(selected, valid=selected.valid & frame_valid)
    if type(row) is not OwnedResourcesRow:
        raise TypeError("Case-R replay requires its original solver's resource row.")
    evaluator = CurrentResourcesEvaluator(
        row=row,
        resources=resources,
        resources_slope=jnp.asarray(float(coefficient), dtype=liquid.dtype),
        utility_of_action=utility,
        owner_capacity=owner_capacity,
    )
    selected = evaluator.node(liquid=liquid, physical=physical)
    return replace(selected, valid=selected.valid & frame_valid)


_MIN_PROJECTION_CAPACITY = 2


def _flatten_native_case_r_output(
    record: NativeCaseROutput,
) -> tuple[tuple[_NativeCaseRChild, ...], None]:
    return tuple(record), None


def _flatten_native_case_r_output_with_keys(
    record: NativeCaseROutput,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _NativeCaseRChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(NativeCaseROutput._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_native_case_r_output(
    metadata: None,
    children: Iterable[_NativeCaseRChild],
) -> NativeCaseROutput:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(NativeCaseROutput._fields):
        raise TypeError(
            "Native NativeCaseROutput reconstruction requires its exact fields."
        )
    return cast("NativeCaseROutput", tuple.__new__(NativeCaseROutput, leaves))


jax.tree_util.register_pytree_with_keys(
    NativeCaseROutput,
    _flatten_native_case_r_output_with_keys,
    _unflatten_native_case_r_output,
    _flatten_native_case_r_output,
)
