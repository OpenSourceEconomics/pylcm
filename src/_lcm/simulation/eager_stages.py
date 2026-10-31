"""Shared exact-layout compiler admission for separately dispatched eager stages.

Process support and declaration-validation stages use the same placement, actual
compiler report and existing workspace planner. This module holds no array cache
or independent memory policy.
"""

from collections.abc import Callable, Hashable, Mapping
from types import MappingProxyType
from typing import cast

import jax

from _lcm.execution.workspace_planning import plan_workspace
from _lcm.simulation.host_operations import (
    ProfiledSimulationOperations,
    StaticArgument,
    _OperandLeaf,
    _operation_memory,
    _OperationCompiler,
)
from _lcm.simulation.operand_placement import (
    _OperandValue,
    place_simulation_arguments,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
    union_buffer_footprints,
)
from _lcm.solution.backward_induction import _lowering_key
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm.typing import ReferenceName

type _StageOperand = _OperandLeaf | complex


def _abstract_stage_operand(
    value: _StageOperand,
) -> jax.ShapeDtypeStruct | _StageOperand:
    """Retain the original placed shape, weak type and layout."""
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=value.sharding, weak_type=value.weak_type
        )
    return value


def run_eager_stage[T: PytreeValue](
    *,
    function: Callable[..., T],
    arguments: Mapping[ReferenceName, _OperandValue],
    static_arguments: MappingProxyType[ReferenceName, StaticArgument],
    operations: ProfiledSimulationOperations,
    devices: tuple[jax.Device, ...],
    required: jax.sharding.Sharding,
    budget_bytes: int,
    live_footprint: Callable[[], DeviceBufferFootprint],
    program_identity: Hashable,
    layout_identity: Hashable,
) -> T:
    """Reserve the actual stage with all original and placed operands still live."""
    execution_devices = tuple(
        device for device in devices if device in required.device_set
    )
    current = live_footprint()
    budget_devices = tuple(dict.fromkeys((*devices, *current.spans)))
    placed = place_simulation_arguments(
        arguments=arguments,
        subject_arg_names=(),
        value_reads=(),
        devices=execution_devices,
        budget_bytes=budget_bytes,
        live_footprint=current,
        budget_devices=budget_devices,
    )
    argument_buffers = measure_buffer_footprint(tree=placed)
    live = union_buffer_footprints(footprints=(live_footprint(), argument_buffers))
    abstract: Mapping[ReferenceName, ShapeDtypePytree] = jax.tree.map(
        _abstract_stage_operand, dict(placed)
    )
    compiler_key = _lowering_key(
        program_identity=program_identity,
        arguments=abstract,
        specialization_key=tuple(static_arguments.items()),
        layout_key=layout_identity,
    )
    external = resident_bytes_by_device(
        live=live, arguments=argument_buffers, devices=execution_devices
    )
    plan = plan_workspace(
        axes=(),
        compile_candidate=_OperationCompiler(
            owner=operations,
            key=compiler_key,
            function=function,
            arguments=abstract,
            static_arguments=static_arguments,
            output_sharding=required,
        ),
        budget_bytes=budget_bytes,
        resident_bytes=max(external.values()),
        memory_for=_operation_memory,
    )
    result = plan.compiled.executable(**placed)
    jax.block_until_ready(result)
    return cast("T", result)
