"""Known simulation payload bytes, deduplicated on each actual device.

Snapshots hold device descriptors and integer address intervals, never arrays. Their
caller owns the buffers and must discard a snapshot when those owners are released:
an allocator can reuse a departed owner's pointer for an unrelated allocation.

The concrete measurement uses JAX's low-level `unsafe_buffer_pointer` on individual
addressable shards and assumes the shard's logical payload occupies a contiguous
interval starting there. It measures known payload bytes, not allocator capacity,
padding, reserved memory, executable storage or another process's allocations. A
view can only establish the extent it exposes; retain the original's footprint too
when its larger allocation remains live. Required non-addressable shards are refused.

Executable arguments are subtracted only from these external resident bytes, under
the solve planner's convention that compiler peaks already include their payload.
This convention must be verified for the execution backend before budget use.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax

from lcm.exceptions import ExecutionPlanningError

type _Spans = tuple[tuple[int, int], ...]


@dataclass(frozen=True, kw_only=True)
class DeviceBufferFootprint:
    """A union of half-open payload address intervals on each actual device."""

    spans: Mapping[jax.Device, _Spans]
    """Merged address intervals; CPU0 and GPU0 remain different device keys."""

    def __post_init__(self) -> None:
        """Own normalized immutable interval metadata without retaining arrays."""
        object.__setattr__(
            self,
            "spans",
            MappingProxyType(
                {
                    device: _merge_spans(spans=tuple(spans))
                    for device, spans in self.spans.items()
                }
            ),
        )


def measure_buffer_footprint(*, tree: object) -> DeviceBufferFootprint:
    """Measure every live addressable JAX payload in an explicitly supplied tree.

    Plain Python and NumPy leaves occupy no JAX device storage until placed. Opaque
    containers are not introspected: callers supply their actual array-bearing trees.
    A deleted or partly non-addressable array cannot establish a complete inventory.
    """
    spans_by_device: dict[jax.Device, list[tuple[int, int]]] = {}
    for leaf in jax.tree.leaves(tree):
        if not isinstance(leaf, jax.Array):
            continue
        if leaf.is_deleted():
            raise ExecutionPlanningError(
                "A deleted array has no live payload footprint."
            )
        if not leaf.is_fully_addressable:
            raise ExecutionPlanningError(
                "Simulation residency requires every stored shard to be addressable."
            )
        for shard in leaf.addressable_shards:
            size = int(shard.data.nbytes)
            if size:
                start = shard.data.unsafe_buffer_pointer()
                spans_by_device.setdefault(shard.device, []).append(
                    (start, start + size)
                )
    return DeviceBufferFootprint(
        spans={device: tuple(spans) for device, spans in spans_by_device.items()}
    )


def union_buffer_footprints(
    *, footprints: tuple[DeviceBufferFootprint, ...]
) -> DeviceBufferFootprint:
    """Combine input, publication and transient metadata without duplicate aliases."""
    spans_by_device: dict[jax.Device, list[tuple[int, int]]] = {}
    for footprint in footprints:
        for device, spans in footprint.spans.items():
            spans_by_device.setdefault(device, []).extend(spans)
    return DeviceBufferFootprint(
        spans={device: tuple(spans) for device, spans in spans_by_device.items()}
    )


def resident_bytes_by_device(
    *,
    live: DeviceBufferFootprint,
    arguments: DeviceBufferFootprint,
    devices: tuple[jax.Device, ...],
) -> Mapping[jax.Device, int]:
    """Count live bytes outside the actual arguments already in compiler peaks.

    Subtract only each argument's covered address range on the same actual device.
    A source array whose copied destination is an argument remains charged, and a
    larger source sharing a smaller argument's start retains its uncovered bytes.
    """
    return MappingProxyType(
        {
            device: _uncovered_bytes(
                live=live.spans.get(device, ()),
                arguments=arguments.spans.get(device, ()),
            )
            for device in devices
        }
    )


def require_transfer_headroom(
    *,
    live: DeviceBufferFootprint,
    destination_bytes: Mapping[jax.Device, int],
    scratch_bytes: Mapping[jax.Device, int],
    budget_bytes: int,
    devices: tuple[jax.Device, ...],
) -> None:
    """Refuse a fresh transfer before its source, destination and scratch can overlap.

    Callers project fresh destination payloads onto their required devices and include
    every still-pending temporary reservation in `scratch_bytes`. Live source and
    existing-copy storage is already in `live`; this stage excludes no arguments.
    Uncertain destination aliases may be conservatively projected as fresh storage.
    The callback authorizes no allocation beyond the explicitly supplied cost model.
    """
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ExecutionPlanningError("A transfer budget must be a positive integer.")
    if not devices or len(set(devices)) != len(devices):
        raise ExecutionPlanningError("A transfer budget needs distinct actual devices.")
    for costs in (destination_bytes, scratch_bytes):
        if any(type(value) is not int or value < 0 for value in costs.values()):
            raise ExecutionPlanningError(
                "Transfer destination and scratch bytes must be nonnegative integers."
            )
    resident = resident_bytes_by_device(
        live=live, arguments=DeviceBufferFootprint(spans={}), devices=devices
    )
    for device in devices:
        required = (
            resident[device]
            + destination_bytes.get(device, 0)
            + scratch_bytes.get(device, 0)
        )
        if required > budget_bytes:
            raise ExecutionPlanningError(
                f"Simulation transfer requires {required} bytes on {device!s} "
                f"against a {budget_bytes}-byte budget before allocation."
            )


def _merge_spans(*, spans: _Spans) -> _Spans:
    """Merge overlapping or adjacent half-open byte intervals."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(spans):
        if type(start) is not int or type(stop) is not int or start < 0 or stop < start:
            raise ExecutionPlanningError(
                "Payload intervals require nonnegative integer bounds, "
                f"got {(start, stop)!r}."
            )
        if start == stop:
            continue
        if merged and start <= merged[-1][1]:
            previous_start, previous_stop = merged[-1]
            merged[-1] = (previous_start, max(previous_stop, stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _uncovered_bytes(*, live: _Spans, arguments: _Spans) -> int:
    """Subtract argument intersections by walking two disjoint sorted interval lists."""
    remaining = sum(stop - start for start, stop in live)
    live_index = argument_index = 0
    while live_index < len(live) and argument_index < len(arguments):
        live_start, live_stop = live[live_index]
        argument_start, argument_stop = arguments[argument_index]
        remaining -= max(
            0, min(live_stop, argument_stop) - max(live_start, argument_start)
        )
        if live_stop <= argument_stop:
            live_index += 1
        else:
            argument_index += 1
    return remaining
