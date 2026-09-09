"""Profile final result assembly on its actual compute or offload device.

CPU assembly after GPU offload uses host RAM outside the selected GPU ceiling.
That narrow boundary still compiles the exact body and retains its real profile;
it does not extend to ordinary operand uploads or value transfers. CPU execution
charges assembly against its selected CPU device as every other operation does.
"""

from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.simulation.host_operations import StaticArgument, _abstract_operand
from _lcm.simulation.memory import SimulationMemory


def concatenate_arrays(
    *, arrays: tuple[jax.Array, ...], memory: SimulationMemory | None = None
) -> jax.Array:
    """Concatenate one field, retaining earlier fields until assembly finishes."""
    if memory is None:
        return _concatenate_arrays(arrays=arrays)
    return _run_assembly(
        memory=memory, arguments={"arrays": arrays}, static_arguments={}
    )


def slice_array(
    *,
    array: jax.Array,
    start: int,
    stop: int,
    memory: SimulationMemory | None = None,
) -> jax.Array:
    """Slice a published field without admitting a same-extent identity as a copy."""
    if start == 0 and stop == array.shape[0]:
        return array
    if memory is None:
        return _slice_array(array=array, start=start, stop=stop)
    return _run_assembly(
        memory=memory,
        arguments={"array": array},
        static_arguments={"start": start, "stop": stop},
    )


def _run_assembly(
    *,
    memory: SimulationMemory,
    arguments: Mapping[str, object],
    static_arguments: Mapping[str, StaticArgument],
) -> jax.Array:
    """Apply the CPU-host exclusion only to the two final assembly bodies here."""
    function = _slice_array if "array" in arguments else _concatenate_arrays
    leaves = jax.tree.leaves(arguments)
    devices = set().union(*(leaf.devices() for leaf in leaves))
    host_assembly = len(devices) == 1 and next(iter(devices)).platform == "cpu"
    executing = tuple(devices) if host_assembly else memory.subject_devices
    if (
        host_assembly
        and not devices.intersection(memory.devices)
        and all(device.platform == "gpu" for device in memory.devices)
    ):
        # These buffers have already completed the admitted GPU->CPU transfer.
        # Profile CPU assembly without pretending a GPU limit bounds host RAM.
        profile = memory.operations.prepare_abstract(
            function=function,
            arguments=jax.tree.map(_abstract_operand, arguments),
            subject_arg_names=tuple(arguments),
            static_arguments=static_arguments,
            devices=executing,
        )
        result = profile.executable(**arguments)
        jax.block_until_ready(result)
    else:
        result = memory.operations.dispatch(
            function=function,
            arguments=arguments,
            subject_arg_names=tuple(arguments),
            static_arguments=static_arguments,
            devices=executing,
            budget_devices=memory.devices,
            budget_bytes=memory.budget_bytes,
            live_footprint=memory.snapshot,
        )
    memory.hold(tree=result)
    return cast("jax.Array", result)


def _concatenate_arrays(*, arrays: tuple[jax.Array, ...]) -> jax.Array:
    """The original concatenate along the global subject axis."""
    return jnp.concatenate(arrays)


def _slice_array(*, array: jax.Array, start: int, stop: int) -> jax.Array:
    """The original contiguous subject slice with immutable bounds."""
    return array[start:stop]
