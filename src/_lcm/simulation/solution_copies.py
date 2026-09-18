"""Profile foreign eager value copies within one execution backend.

Copies retain the source layout, including devices outside the simulation subset.
Mixed CPU/accelerator copies need a separate host budget and are refused. Unrelated
CPU host storage does not consume an accelerator allocator's device ceiling.
"""

from collections.abc import Callable
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.execution.workspace_planning import plan_workspace
from _lcm.simulation.host_operations import (
    ProfiledSimulationOperations,
    _operation_memory,
    _OperationCompiler,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    require_transfer_headroom,
    resident_bytes_by_device,
    union_buffer_footprints,
)
from _lcm.solution.backward_induction import _lowering_key
from lcm.exceptions import ExecutionPlanningError


def copy_solution_leaf(
    *,
    leaf: jax.Array,
    operations: ProfiledSimulationOperations,
    live_footprint: Callable[[], DeviceBufferFootprint],
    budget_devices: tuple[jax.Device, ...],
    budget_bytes: int,
) -> jax.Array:
    """Admit one exact-layout private copy against every currently owned buffer.

    The compiler includes input and output payloads. Only its actual source argument
    is excluded from external residency; all earlier copies remain charged. Copy
    executables and abstract metadata are cached, never concrete arrays or owners.
    """
    platforms = {device.platform for device in budget_devices}
    if len(platforms) != 1 or any(
        device.platform not in platforms for device in leaf.sharding.device_set
    ):
        raise ExecutionPlanningError(
            "Budgeted foreign value copies require source and execution devices "
            "on the same backend; mixed CPU/accelerator copy admission is not profiled."
        )
    argument_buffers = measure_buffer_footprint(tree=leaf)
    live = union_buffer_footprints(footprints=(live_footprint(), argument_buffers))
    devices = tuple(argument_buffers.spans)
    # Empty arrays still have a physical layout and need an executing device.
    if not devices:
        devices = tuple(
            dict.fromkeys(shard.device for shard in leaf.addressable_shards)
        )
    all_devices = tuple(
        device
        for device in dict.fromkeys((*budget_devices, *live.spans, *devices))
        if device.platform in platforms
    )
    require_transfer_headroom(
        live=live,
        destination_bytes={},
        scratch_bytes={},
        budget_bytes=budget_bytes,
        devices=all_devices,
    )
    external = resident_bytes_by_device(
        live=live, arguments=argument_buffers, devices=devices
    )
    arguments = MappingProxyType(
        {
            "value": jax.ShapeDtypeStruct(
                leaf.shape, leaf.dtype, sharding=leaf.sharding, weak_type=leaf.weak_type
            )
        }
    )
    key = _lowering_key(
        program_identity=_copy_value_leaf,
        arguments=arguments,
        layout_key=("foreign_solution_copy", leaf.sharding),
    )
    plan = plan_workspace(
        axes=(),
        compile_candidate=_OperationCompiler(
            owner=operations,
            key=key,
            function=_copy_value_leaf,
            arguments=arguments,
            static_arguments=MappingProxyType({}),
            output_sharding=leaf.sharding,
        ),
        budget_bytes=budget_bytes,
        resident_bytes=max(external.values()),
        memory_for=_operation_memory,
    )
    result = plan.compiled.executable(value=leaf).block_until_ready()
    copied = measure_buffer_footprint(tree=result)
    exclusive = resident_bytes_by_device(
        live=copied, arguments=argument_buffers, devices=devices
    )
    if any(
        exclusive[device] != sum(stop - start for start, stop in spans)
        for device, spans in copied.spans.items()
    ):
        raise ExecutionPlanningError("Foreign solution copy aliases its source buffer.")
    return result


def _copy_value_leaf(*, value: jax.Array) -> jax.Array:
    """Keep the canonical private-copy expression inside the measured executable."""
    return jnp.array(value, copy=True)
