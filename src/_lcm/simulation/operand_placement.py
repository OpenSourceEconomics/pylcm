"""Place forward operands without acquiring or retaining solved-value copies."""

import dataclasses
import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import jax
import numpy as np

from _lcm.execution.core_program import ValueRead
from _lcm.execution.footprint import layout_footprint
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    require_transfer_headroom,
    union_buffer_footprints,
)
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.typing import ArgumentTree, HostArray
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ReferenceName, ValueND

# One process-grid parameter: a runtime or fixed numerical value, which may be a
# Python complex.
type _ProcessParameter = ValueND | HostArray | np.generic | int | float | complex

# A placement operand as callers hand it over: a core-program argument tree, or a
# process-grid parameter, alone or in its parameter mapping.
type _OperandValue = ArgumentTree | _ProcessParameter | Mapping[str, _ProcessParameter]

if TYPE_CHECKING:
    # An operand tree as the placement walk descends and rebuilds it.
    type _Operand = (
        _OperandValue | tuple[_Operand, ...] | list[_Operand] | Mapping[str, _Operand]
    )
else:
    # beartype cannot build a check for a recursive alias over other recursive
    # pytree aliases; callers' operands are checked against `_OperandValue`.
    type _Operand = object


@runtime_checkable
class SubjectArgumentNames(Protocol):
    """Argument-builder metadata naming the population's dynamic operands."""

    @property
    def subject_arg_names(self) -> tuple[str, ...]:
        """Exact subject operands, independently of their array sizes."""
        ...


def place_simulation_arguments[T: _OperandValue](
    *,
    arguments: Mapping[str, T],
    subject_arg_names: tuple[ReferenceName, ...],
    value_reads: tuple[ValueRead, ...],
    devices: tuple[jax.Device, ...],
    budget_bytes: int | None = None,
    live_footprint: DeviceBufferFootprint | None = None,
    argument_footprint: DeviceBufferFootprint | None = None,
    budget_devices: tuple[jax.Device, ...] = (),
) -> Mapping[str, T]:
    """Share scalars/grids/params and shard subjects on the declared device order.

    Addressed value leaves are already supplied by the period's value owner and
    pass through unchanged. Placement is call-local; this module caches no arrays.
    An already-correct array passes through by identity, including subject states.

    Each argument comes back with its own container structure; only its numeric
    leaves are replaced by device arrays, so a tree of device arrays keeps its
    exact type and a host leaf comes back as the device array it was placed as.

    A budgeted caller supplies the live inventory in `live_footprint`; the
    operands are charged on top of it here. A caller that already measured
    `arguments` for its own accounting may pass that measurement as
    `argument_footprint` instead of having it taken again: the charge is the
    same, and one `unsafe_buffer_pointer` barrier over the operand tree is not
    repeated in the same dispatch.
    """
    subject_sharding = subject_operand_sharding(devices=devices)
    shared_sharding = simulation_value_sharding(
        stored_sharding=subject_sharding, devices=devices
    )
    protected = frozenset(
        (read.source.argument or read.source.channel.value, *read.source.path)
        for read in value_reads
    )
    if budget_bytes is not None:
        if live_footprint is None or not budget_devices:
            raise ExecutionPlanningError(
                "Operand placement requires a live budget inventory."
            )
        _require_operand_headroom(
            arguments=arguments,
            subject_arg_names=subject_arg_names,
            subject_sharding=subject_sharding,
            shared_sharding=shared_sharding,
            protected=protected,
            budget_bytes=budget_bytes,
            # Only the budgeted devices are read below, and projection precedes
            # normalization, so retained history elsewhere costs a key lookup.
            live=union_buffer_footprints(
                footprints=(
                    live_footprint,
                    argument_footprint
                    if argument_footprint is not None
                    else measure_buffer_footprint(tree=arguments),
                ),
                devices=budget_devices,
            ),
            budget_devices=budget_devices,
        )
    placed = MappingProxyType(
        {
            name: _place_operand_tree(
                tree=value,
                sharding=(
                    subject_sharding if name in subject_arg_names else shared_sharding
                ),
                protected=_paths_below(paths=protected, segment=name),
            )
            for name, value in arguments.items()
        }
    )
    if budget_bytes is not None:
        # The complete projected scratch reservation remains charged until all
        # placements finish. Subsequent compiler planning therefore has no
        # outstanding scratch from these operand copies.
        jax.block_until_ready(placed)
    # Rebuilding a container keeps its structure and places its leaves.
    return cast("Mapping[str, T]", placed)


def _require_operand_headroom(
    *,
    arguments: Mapping[str, _Operand],
    subject_arg_names: tuple[ReferenceName, ...],
    subject_sharding: jax.sharding.Sharding,
    shared_sharding: jax.sharding.Sharding,
    protected: frozenset[tuple[str | int, ...]],
    budget_bytes: int,
    live: DeviceBufferFootprint,
    budget_devices: tuple[jax.Device, ...],
) -> None:
    """Reserve the whole operand tree's new destinations and overlapping scratch.

    Each nonaligned leaf contributes its required layout payload, plus that
    amount of transfer scratch on participating source/destination devices, the
    existing value-transfer cost convention. Repeated leaves are conservatively
    projected as separate copies; actual post-placement aliases are deduplicated
    before dispatch. No fabricated value address is needed for shared operands.
    """
    destination_bytes: dict[jax.Device, int] = {}
    scratch_bytes: dict[jax.Device, int] = {}
    for name, tree in arguments.items():
        sharding = subject_sharding if name in subject_arg_names else shared_sharding
        for leaf in _operand_leaves(
            tree=tree, protected=_paths_below(paths=protected, segment=name)
        ):
            if isinstance(leaf, jax.Array) and leaf.sharding == sharding:
                continue
            if not isinstance(
                leaf, jax.Array | np.ndarray | np.generic | bool | int | float | complex
            ):
                continue
            byte_count = _required_operand_bytes(leaf=leaf, sharding=sharding)
            source_devices = (
                leaf.sharding.device_set if isinstance(leaf, jax.Array) else set()
            )
            for device in sharding.device_set:
                destination_bytes[device] = (
                    destination_bytes.get(device, 0) + byte_count
                )
            for device in source_devices | sharding.device_set:
                scratch_bytes[device] = scratch_bytes.get(device, 0) + byte_count
    require_transfer_headroom(
        live=live,
        destination_bytes=destination_bytes,
        scratch_bytes=scratch_bytes,
        budget_bytes=budget_bytes,
        devices=budget_devices,
    )


def subject_operand_sharding(
    *, devices: tuple[jax.Device, ...]
) -> jax.sharding.Sharding:
    """Use one ordered leading-axis placement for subject inputs and outputs."""
    if not devices:
        raise ValueError("Simulation operands require explicit subject devices.")
    if len(devices) == 1:
        return jax.sharding.SingleDeviceSharding(devices[0])
    mesh = jax.make_mesh(
        (len(devices),), ("X",), (jax.sharding.AxisType.Auto,), devices=devices
    )
    return jax.NamedSharding(mesh, jax.P("X"))


def _required_operand_bytes(
    *,
    leaf: jax.Array | HostArray | np.generic | complex,
    sharding: jax.sharding.Sharding,
) -> int:
    """Size canonical numeric payloads without materializing a device array."""
    if isinstance(leaf, jax.Array):
        shape = tuple(leaf.shape)
        count = math.prod(shape)
        # nbytes also sizes extended PRNG-key dtypes, which are not NumPy dtypes.
        item_bytes = leaf.nbytes // count if count else 0
    else:
        # JAX applies the current dtype rules without tracing a conversion
        # or retaining the concrete operand between calls.
        abstract = jax.typeof(leaf)
        shape = tuple(abstract.shape)
        item_bytes = abstract.dtype.itemsize
    return layout_footprint(
        sharding=sharding, shape=shape, item_bytes=item_bytes
    ).bytes_per_device


def _operand_leaves(
    *, tree: _Operand, protected: frozenset[tuple[str | int, ...]]
) -> Iterator[_Operand]:
    """Walk the same operand containers while leaving addressed values untouched."""
    if () in protected:
        return
    if isinstance(tree, Mapping):
        children = tree.items()
    elif isinstance(tree, tuple | list):
        children = enumerate(tree)
    elif dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        children = (
            (field.name, getattr(tree, field.name))
            for field in dataclasses.fields(tree)
            if field.init
        )
    else:
        yield tree
        return
    for name, value in children:
        yield from _operand_leaves(
            tree=value, protected=_paths_below(paths=protected, segment=name)
        )


def _paths_below(
    *, paths: frozenset[tuple[str | int, ...]], segment: str | int
) -> frozenset[tuple[str | int, ...]]:
    """Keep the protected value leaves below this exact argument-tree step."""
    return frozenset(path[1:] for path in paths if path and path[0] == segment)


def _place_operand_tree(
    *,
    tree: _Operand,
    sharding: jax.sharding.Sharding,
    protected: frozenset[tuple[str | int, ...]],
) -> _Operand:
    """Rebuild only operand containers around the explicitly protected leaves."""
    if () in protected:
        return tree
    if isinstance(tree, Mapping):
        return MappingProxyType(
            {
                key: _place_operand_tree(
                    tree=value,
                    sharding=sharding,
                    protected=_paths_below(paths=protected, segment=key),
                )
                for key, value in tree.items()
            }
        )
    if isinstance(tree, tuple | list):
        values = [
            _place_operand_tree(
                tree=value,
                sharding=sharding,
                protected=_paths_below(paths=protected, segment=index),
            )
            for index, value in enumerate(tree)
        ]
        return tuple(values) if isinstance(tree, tuple) else values
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return dataclasses.replace(
            tree,
            **{
                field.name: _place_operand_tree(
                    tree=getattr(tree, field.name),
                    sharding=sharding,
                    protected=_paths_below(paths=protected, segment=field.name),
                )
                for field in dataclasses.fields(tree)
                if field.init
            },
        )
    return _place_operand_leaf(leaf=tree, sharding=sharding)


def _place_operand_leaf(*, leaf: _Operand, sharding: jax.sharding.Sharding) -> _Operand:
    """Move a numeric leaf while preserving already-correct buffers and metadata."""
    if isinstance(leaf, jax.Array) and leaf.sharding == sharding:
        return leaf
    if isinstance(
        leaf, jax.Array | np.ndarray | np.generic | bool | int | float | complex
    ):
        return jax.device_put(leaf, sharding)
    return leaf
