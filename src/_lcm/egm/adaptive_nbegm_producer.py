"""Bind the genuine nested NB floor and preserve all executed native summaries."""

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from _lcm.egm.nbegm_query_model import NBEGMQueryModel
from _lcm.egm.nested_floor import (
    NESTED_FLOOR_INPUT,
    PreparedNestedFloor,
    prepare_nested_floor,
)
from _lcm.egm.outer_candidates import OuterCandidateResult
from _lcm.egm.preferences import concatenate_regime_function
from _lcm.execution.core_program import CoreArgumentBuilder, CoreBuildContext
from _lcm.execution.output_layout import OutputRoleTree, StateAxesLeading
from _lcm.solution.contract import PeriodKernel
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    ContinuationReadCounts,
    QueryInputStatus,
    merge_continuation_read_counts,
)
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm.solver_api import KernelOutput
from lcm.typing import BoolND, Float1D, Int1D, StateName

if TYPE_CHECKING:
    from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
else:
    # The numerical solver imports this module while its class body is loading.
    _RideAlongNBEGMPeriodKernel = PeriodKernel

_NATIVE_CONDITIONAL_ARITY = 3


def _tuple_roles(*, roles: OutputRoleTree) -> BoundedResult[tuple[OutputRoleTree, ...]]:
    """Narrow an actual bounded role record after checking its tuple structure."""
    if type(roles) is not BoundedResult or type(roles.output) is not tuple:
        raise TypeError("A nested NB program must declare bounded tuple roles.")
    return cast("BoundedResult[tuple[OutputRoleTree, ...]]", roles)


def _floor_template(*, shape: tuple[int, ...], offsets: Float1D) -> PreparedNestedFloor:
    """Declare only the private operand's numeric shapes for native lowering."""
    scalar = jnp.asarray(a=False)
    status = jax.tree.map(
        lambda _flag: jnp.zeros(shape, dtype=jnp.bool_),
        BoundInputStatus(scalar, scalar),
    )
    return BoundedResult(
        output=jnp.zeros(shape, dtype=offsets.dtype),
        bound_status=(status,),
    )


@dataclass(frozen=True, kw_only=True)
class PreparedFloorArguments:
    """Add the canonical-shaped private operand used by actual adjuster calls."""

    inner: CoreArgumentBuilder
    template: PreparedNestedFloor

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        """Use one signature for lowering and the invocation-local runtime bank."""
        return MappingProxyType(
            {**self.inner(context), NESTED_FLOOR_INPUT: self.template}
        )


def with_keeper_floor(
    *,
    inner: Callable[..., PytreeValue],
    model: NBEGMQueryModel,
    state_names: tuple[StateName, ...],
    **arguments: PytreeValue,
) -> BoundedResult[tuple[PytreeValue, ...]]:
    """Prepare b(C) before any keeper cost and append the genuine shared bank."""
    target = model.authority.bound.lower_target
    if target is None:
        raise TypeError("An adaptive callable floor requires its canonical DAG target.")
    prepared = prepare_nested_floor(
        lower=concatenate_regime_function(functions=model.functions, target=target),
        offsets=model.offsets,
        state_names=state_names,
        **arguments,
    )
    result = inner(**arguments, **{NESTED_FLOOR_INPUT: prepared})
    if (
        type(result) is not BoundedResult
        or len(result.output) != _NATIVE_CONDITIONAL_ARITY
    ):
        raise TypeError("A nested NB keeper must retain genuine conditional channels.")
    native = cast("BoundedResult[tuple[PytreeValue, ...]]", result)
    output: tuple[PytreeValue, ...] = (*native.output, prepared)
    return BoundedResult(
        output=output,
        bound_status=native.bound_status,
        query_status=native.query_status,
        continuation_read_counts=native.continuation_read_counts,
    )


def prepare_nested_inner_kernels(
    *,
    keeper: _RideAlongNBEGMPeriodKernel,
    adjuster: _RideAlongNBEGMPeriodKernel,
    model: NBEGMQueryModel,
    state_grids: Mapping[StateName, Float1D | Int1D],
) -> tuple[_RideAlongNBEGMPeriodKernel, _RideAlongNBEGMPeriodKernel]:
    """Wrap native role programs without an extra floor-only program or solve."""
    # Import after solver construction; the public solver facade stays acyclic.
    from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel  # noqa: PLC0415

    if not isinstance(keeper, _RideAlongNBEGMPeriodKernel) or not isinstance(
        adjuster, _RideAlongNBEGMPeriodKernel
    ):
        raise TypeError(
            "Adaptive floor sharing requires the actual native NB ride kernels."
        )
    names = model.authority.state_names
    shape = tuple(state_grids[name].size for name in names)
    template = _floor_template(shape=shape, offsets=model.offsets)
    bank_roles = jax.tree.map(
        lambda _leaf: StateAxesLeading(state_names=names), template
    )
    keeper_programs, adjuster_programs = {}, {}
    for name, program in keeper.core_programs().items():
        if type(program.output_roles) is not BoundedResult:
            raise TypeError("A nested NB keeper requires mandatory integrity roles.")
        roles = _tuple_roles(roles=program.output_roles)
        checked = program.checked_variant
        if checked is not None and type(checked.output_roles) is not BoundedResult:
            raise TypeError(
                "A checked nested NB keeper requires the same bounded roles."
            )
        checked_roles = (
            None if checked is None else _tuple_roles(roles=checked.output_roles)
        )
        keeper_programs[name] = replace(
            program,
            function=functools.partial(
                with_keeper_floor,
                inner=program.function,
                model=model,
                state_names=names,
            ),
            output_roles=replace(roles, output=(*roles.output, bank_roles)),
            checked_variant=None
            if checked is None or checked_roles is None
            else replace(
                checked,
                function=functools.partial(
                    with_keeper_floor,
                    inner=checked.function,
                    model=model,
                    state_names=names,
                ),
                output_roles=replace(
                    checked_roles, output=(*checked_roles.output, bank_roles)
                ),
            ),
        )
    for name, program in adjuster.core_programs().items():
        adjuster_programs[name] = replace(
            program,
            argument_builder=PreparedFloorArguments(
                inner=program.argument_builder, template=template
            ),
        )
    return (
        replace(keeper, _core_programs=MappingProxyType(keeper_programs)),
        replace(adjuster, _core_programs=MappingProxyType(adjuster_programs)),
    )


def fold_executed_summaries(
    *,
    keeper: KernelOutput,
    candidates: Sequence[OuterCandidateResult],
    floor_mismatch: BoolND,
) -> tuple[
    tuple[BoundInputStatus, ...],
    tuple[QueryInputStatus, ...] | None,
    ContinuationReadCounts | None,
]:
    """OR every executed status and sum actual read counts, including rejected
    probes.
    """
    from _lcm.solution.nbegm import _NBEGMNestedKernelOutput  # noqa: PLC0415

    if (
        not isinstance(keeper, _NBEGMNestedKernelOutput)
        or len(keeper.bound_status) != 1
    ):
        raise TypeError("An adaptive fold requires its genuine keeper's source owner.")
    bounds, queries, counts = (
        keeper.bound_status,
        keeper.query_status,
        keeper.continuation_read_counts,
    )
    for candidate in candidates:
        if (
            len(candidate.bound_status) != len(bounds)
            or (candidate.query_status is None) != (queries is None)
            or len(candidate.query_status or ()) != len(queries or ())
        ):
            raise TypeError("Nested NB role integrity owners disagree.")
        bounds = tuple(
            jax.tree.map(jnp.logical_or, left, right)
            for left, right in zip(bounds, candidate.bound_status, strict=True)
        )
        if queries is not None:
            if candidate.query_status is None:
                raise TypeError("A nested NB candidate lost its declared query slots.")
            queries = tuple(
                jax.tree.map(jnp.logical_or, left, right)
                for left, right in zip(queries, candidate.query_status, strict=True)
            )
        counts = merge_continuation_read_counts(
            left=counts, right=candidate.continuation_read_counts
        )
    own = bounds[0]
    bounds = (
        BoundInputStatus(
            own.nonfinite_floor, own.invalid_shifted_grid | floor_mismatch
        ),
    )
    return bounds, queries, counts
