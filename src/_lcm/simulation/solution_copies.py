"""Profile foreign solution copies within one execution backend.

Copies retain the source layout, including devices outside the simulation subset.
Mixed CPU/accelerator copies need a separate host budget and are refused. Unrelated
CPU host storage does not consume an accelerator allocator's device ceiling. Before
the first copy, the payload bytes a supplied result's values and simulation policies
must occupy are checked against the remaining headroom, so a result that cannot fit
is refused before anything is decoded or copied.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

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
from lcm._solver_api.entries import _CanonicalArtifactEntry, _CanonicalValueEntry
from lcm._solver_api.result import SolutionResult
from lcm._solver_api.stores import ArtifactStore, ValueStore
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import SIMULATION_POLICY


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


def require_foreign_solution_headroom(
    *,
    solution: object,
    live: DeviceBufferFootprint,
    budget_devices: tuple[jax.Device, ...],
    budget_bytes: int,
    budget_note: str,
) -> None:
    """Refuse a supplied result whose consumed payloads cannot fit, before any copy.

    Counts one private copy of every value and simulation policy, sized from the
    arrays an engine-owned entry holds or from an archive manifest's declared leaf
    shapes and dtypes. Archive uploads, second copies, flags, replay payloads and
    compiler workspace are not counted, so the total is a lower bound: a refusal
    here is certain, and every copy is still admitted individually afterwards.
    """
    if type(solution) is not SolutionResult:
        return
    platforms = {device.platform for device in budget_devices}
    devices = tuple(
        device
        for device in dict.fromkeys((*budget_devices, *live.spans))
        if device.platform in platforms
    )
    resident = resident_bytes_by_device(
        live=live, arguments=DeviceBufferFootprint(spans={}), devices=devices
    )
    available = sum(max(0, budget_bytes - resident[device]) for device in devices)
    needed = _consumed_payload_bytes(solution=solution)
    if needed > available:
        raise ExecutionPlanningError(
            f"Private copies of the supplied solution need at least {needed} bytes "
            f"of device memory, but {available} bytes remain under the budget "
            f"after the {sum(resident.values())} bytes already resident."
            f"{budget_note}"
        )


def _consumed_payload_bytes(*, solution: SolutionResult) -> int:
    """Sum the declared bytes of every value and simulation-policy entry."""
    entries: list[object] = []
    if type(solution.values) is ValueStore:
        entries.extend(solution.values._entries.values())  # noqa: SLF001
    if type(solution.replay_artifacts) is ArtifactStore:
        entries.extend(
            entry
            for ref, entry in solution.replay_artifacts._entries.items()  # noqa: SLF001
            if ref.key == SIMULATION_POLICY
        )
    return sum(_entry_bytes(entry) for entry in entries)


def _entry_bytes(entry: object) -> int:
    """Return an entry's payload bytes without reading or decoding it."""
    # The archive imports solution ownership; resolve its entry type at call time.
    from _lcm.persistence.solution import _LazyHdf5Entry  # noqa: PLC0415

    if type(entry) is _CanonicalValueEntry:
        return _array_bytes(entry.value)
    if type(entry) is _CanonicalArtifactEntry:
        return sum(_array_bytes(leaf) for leaf in entry.leaves)
    if type(entry) is _LazyHdf5Entry:
        return sum(
            np.dtype(str(leaf["dtype"])).itemsize
            * int(np.prod(cast("tuple[int, ...]", leaf["shape"]), dtype=np.int64))
            for leaf in entry.leaves
        )
    return 0


def _array_bytes(value: object) -> int:
    return int(value.nbytes) if isinstance(value, jax.Array) else 0


def _copy_value_leaf(*, value: jax.Array) -> jax.Array:
    """Keep the canonical private-copy expression inside the measured executable."""
    return jnp.array(value, copy=True)
