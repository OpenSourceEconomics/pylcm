"""Admit completed-chunk copies while retaining every compute source until ready.

The transfer convention reserves one destination payload and one source-shard
payload of scratch per copied leaf occurrence. Shared input aliases are measured
physically by SimulationMemory; uncertain destination aliases are not assumed.
GPU-to-CPU offload explicitly excludes host RAM from the selected GPU ceiling.
CPU execution has no such exclusion and assembles on a selected CPU device.
"""

import jax

from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import require_transfer_headroom
from lcm.exceptions import ExecutionPlanningError


def chunk_host_device(*, subject_devices: tuple[jax.Device, ...]) -> jax.Device:
    """Keep CPU assembly on a selected CPU; GPU offload uses host CPU RAM."""
    if all(device.platform == "cpu" for device in subject_devices):
        return subject_devices[0]
    return jax.devices("cpu")[0]


def offload_chunk[T](
    *, tree: T, host_device: jax.Device, memory: SimulationMemory | None
) -> T:
    """Check source/destination/scratch overlap before the actual device copy."""
    if memory is None:
        return jax.block_until_ready(jax.device_put(tree, host_device))
    if host_device.platform != "cpu":
        raise ExecutionPlanningError(
            "Completed-chunk offload requires an actual CPU destination."
        )
    host_excluded = host_device not in memory.devices
    if host_excluded and not all(device.platform == "gpu" for device in memory.devices):
        raise ExecutionPlanningError(
            "CPU chunk offload requires a budgeted CPU destination."
        )
    memory.hold(tree=tree)
    destination, scratch, copies_needed = _copy_reservation(
        tree=tree,
        host_device=host_device,
        host_excluded=host_excluded,
        devices=memory.devices,
    )
    require_transfer_headroom(
        live=memory.snapshot(),
        destination_bytes=destination,
        scratch_bytes=scratch,
        budget_bytes=memory.budget_bytes,
        devices=memory.devices,
    )
    if not copies_needed:
        return tree
    result = jax.block_until_ready(jax.device_put(tree, host_device))
    memory.hold(tree=result)
    return result


def _copy_reservation(
    *,
    tree: object,
    host_device: jax.Device,
    host_excluded: bool,
    devices: tuple[jax.Device, ...],
) -> tuple[dict[jax.Device, int], dict[jax.Device, int], bool]:
    """Reserve destinations and source-shard scratch for each actual copied leaf."""
    leaves = jax.tree.leaves(tree)
    destination: dict[jax.Device, int] = {}
    scratch: dict[jax.Device, int] = {}
    copies_needed = False
    for leaf in leaves:
        if not isinstance(leaf, jax.Array):
            raise ExecutionPlanningError(
                "A completed chunk must expose its actual JAX leaves."
            )
        if not leaf.devices() <= set(devices):
            raise ExecutionPlanningError(
                "A chunk copy has an unbudgeted compute source."
            )
        if leaf.devices() == {host_device}:
            continue
        copies_needed = True
        if not host_excluded:
            destination[host_device] = destination.get(host_device, 0) + leaf.nbytes
        for shard in leaf.addressable_shards:
            scratch[shard.device] = scratch.get(shard.device, 0) + shard.data.nbytes
    return destination, scratch, copies_needed
