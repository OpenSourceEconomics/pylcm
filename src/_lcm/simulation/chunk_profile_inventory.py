"""Metadata-only inventory of actual compiled stages and future payload owners.

Future logical slots cannot establish physical aliases. Each declared occurrence
is therefore charged independently; concrete entry residency remains a separate
interval union. Argument-placement reservations deliberately overlap compiler
input bytes and retained originals. This is a conservative admission bound.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import jax

from _lcm.simulation.chunk_planning import SimulationStageProfile
from _lcm.simulation.host_operations import StaticArgument
from _lcm.simulation.operand_placement import subject_operand_sharding
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.value_placement import simulation_value_sharding
from lcm.exceptions import ExecutionPlanningError


def abstract_tree(*, tree: object, sharding: jax.sharding.Sharding) -> object:
    """Copy only shape/dtype metadata onto the required ordered layout."""

    def abstract(leaf: object) -> jax.ShapeDtypeStruct:
        if not isinstance(leaf, jax.Array | jax.ShapeDtypeStruct):
            raise ExecutionPlanningError(
                "Chunk profiles need canonical array metadata."
            )
        return jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, weak_type=leaf.weak_type, sharding=sharding
        )

    return jax.tree.map(abstract, tree)


def payload_bytes(*, tree: object) -> dict[jax.Device, int]:
    """Size declared shard payloads, including extended PRNG-key dtypes."""
    result: dict[jax.Device, int] = {}
    for leaf in jax.tree.leaves(tree):
        if not isinstance(leaf, jax.ShapeDtypeStruct) or leaf.sharding is None:
            raise ExecutionPlanningError(
                "A future payload requires placed abstract metadata."
            )
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            raw = jax.eval_shape(
                jax.random.key_data, jax.ShapeDtypeStruct((), leaf.dtype)
            )
            item_bytes = math.prod(raw.shape) * raw.dtype.itemsize
        else:
            item_bytes = leaf.dtype.itemsize
        per_device = math.prod(leaf.sharding.shard_shape(leaf.shape)) * item_bytes
        for device in leaf.sharding.device_set:
            result[device] = result.get(device, 0) + per_device
    return result


def add_bytes(
    *, target: dict[jax.Device, int], source: Mapping[jax.Device, int]
) -> None:
    """Sum distinct declared storage slots per actual backend device."""
    for device, count in source.items():
        target[device] = target.get(device, 0) + count


def maximum_bytes(
    *, target: dict[jax.Device, int], source: Mapping[jax.Device, int]
) -> None:
    """Combine phases that cannot execute concurrently."""
    for device, count in source.items():
        target[device] = max(target.get(device, 0), count)


@dataclass(kw_only=True)
class ChunkProfileInventory:
    """Compile actual pure operations while accumulating only future metadata."""

    runtime: SimulationRuntime
    stages: list[SimulationStageProfile] = field(default_factory=list)
    unit: dict[jax.Device, int] = field(default_factory=dict)
    maximum_unit: dict[jax.Device, int] = field(default_factory=dict)

    def operation(
        self,
        *,
        function: Callable[..., object],
        arguments: Mapping[str, object],
        subject_arg_names: tuple[str, ...] = (),
        static_arguments: Mapping[str, StaticArgument] = MappingProxyType({}),
        subject_outputs: bool = False,
        devices: tuple[jax.Device, ...] | None = None,
    ) -> object:
        """Use the same argument placement, compiler and numerical body as dispatch."""
        devices = self.runtime.subject_devices if devices is None else devices
        subject = subject_operand_sharding(devices=devices)
        shared = simulation_value_sharding(stored_sharding=subject, devices=devices)
        placed = {
            name: abstract_tree(
                tree=value, sharding=subject if name in subject_arg_names else shared
            )
            for name, value in arguments.items()
        }
        compiled = self.runtime.operations.prepare_abstract(
            function=function,
            arguments=placed,
            subject_arg_names=subject_arg_names,
            static_arguments=static_arguments,
            subject_outputs=subject_outputs,
            devices=devices,
        )
        return self.compiled(
            name=getattr(function, "__name__", type(function).__qualname__),
            executable=compiled.executable,
            arguments=placed,
            devices=devices,
        )

    def compiled(
        self,
        *,
        name: str,
        executable: jax.stages.Compiled,
        arguments: Mapping[str, object],
        devices: tuple[jax.Device, ...] | None = None,
    ) -> object:
        """Keep raw compiler peaks separate from conservative future owner slots."""
        self.stages.append(
            SimulationStageProfile(
                name=name,
                executable=executable,
                devices=self.runtime.subject_devices if devices is None else devices,
            )
        )
        # An operation may copy every argument and keep every returned leaf alive
        # through the unit. DCE and abstract pointer equality cannot erase owners.
        add_bytes(target=self.unit, source=payload_bytes(tree=arguments))
        add_bytes(target=self.unit, source=payload_bytes(tree=executable.out_info))
        return executable.out_info

    def close_unit(self) -> None:
        """Bound sequential units by their largest temporary inventory."""
        maximum_bytes(target=self.maximum_unit, source=self.unit)
        self.unit.clear()
