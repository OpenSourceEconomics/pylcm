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

Compiler-retained executable arguments are subtracted only from these external
resident bytes, under the convention that compiler peaks already include their
payload. Eliminated arguments stay resident while their callers own them. The
compiler argument-byte convention must be verified for the execution backend.

A call-local `OwnerLedger` records each live binding once, at its placement, and
reuses the merged union only while its ownership epoch is unchanged. It is metadata,
not an owner: the existing owner managers keep the arrays alive for their declared
lifetime, and no pointer survives the call that measured it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
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
    *,
    footprints: tuple[DeviceBufferFootprint, ...],
    devices: tuple[jax.Device, ...] | None = None,
) -> DeviceBufferFootprint:
    """Combine live metadata, optionally only on an explicit consumer device set.

    Projection precedes interval normalization: unrelated retained history costs
    only a device-key lookup. The original footprints remain complete for other
    consumers. Callers must select every device their admission check requires.
    """
    spans_by_device: dict[jax.Device, list[tuple[int, int]]] = {}
    for footprint in footprints:
        selected = footprint.spans if devices is None else devices
        for device in selected:
            spans = footprint.spans.get(device, ())
            spans_by_device.setdefault(device, []).extend(spans)
    return DeviceBufferFootprint(
        spans={device: tuple(spans) for device, spans in spans_by_device.items()}
    )


def resolve_budget_devices(
    *, execution_devices: tuple[jax.Device, ...], live: DeviceBufferFootprint
) -> tuple[jax.Device, ...]:
    """Include actual retained sources under the execution backend's device ceiling.

    Preserve execution order, then append same-backend sources in footprint order.
    Host storage on another backend does not acquire an accelerator memory ceiling.
    This inventory changes neither execution placement nor any array's sharding.
    """
    platforms = {device.platform for device in execution_devices}
    return tuple(
        dict.fromkeys(
            (
                *execution_devices,
                *(
                    device
                    for device, spans in live.spans.items()
                    if spans and device.platform in platforms
                ),
            )
        )
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


@dataclass(kw_only=True, eq=False)
class OwnerLedger:
    """Call-local metadata for owners measured once at a known ownership epoch.

    A ledger records the address spans of bindings an external owner manager keeps
    alive; it never retains an array and never outlives its call. Every bind,
    rebind and release advances `epoch` and discards the cached unions, so a union
    can be reused only while the recorded ownership is provably unchanged.

    A stale binding can only over-charge: spans are unioned, so an address recycled
    by a live owner is counted once, which is that owner's own charge. Admission can
    therefore never become more permissive than a full re-measure would be.
    """

    epoch: int = 0
    """Advances on every ownership mutation; a cached union is valid for one value."""

    _bindings: dict[str, DeviceBufferFootprint] = field(default_factory=dict)
    """Measured spans per owner name, in binding order."""

    _unions: dict[tuple[jax.Device, ...] | None, DeviceBufferFootprint] = field(
        default_factory=dict
    )
    """Merged projections of the current epoch, one per requested device set."""

    def bind(self, *, owner: str, footprint: DeviceBufferFootprint) -> None:
        """Record already-measured spans for one owner, replacing any earlier one."""
        self._bindings[owner] = footprint
        self._invalidate()

    def measure(self, *, owner: str, tree: object) -> None:
        """Await and measure one owner tree exactly once, at its placement.

        The readiness barrier moves to the binding instead of repeating on every
        later snapshot. It is never removed and never deferred, so the completion
        boundary each admission check relied on is preserved.
        """
        jax.block_until_ready(tree)
        self.bind(owner=owner, footprint=measure_buffer_footprint(tree=tree))

    def release(self, *, owner: str) -> None:
        """Drop one owner binding, whether or not it was ever recorded."""
        self._bindings.pop(owner, None)
        self._invalidate()

    def release_prefix(self, *, prefix: str) -> None:
        """Drop every binding of one owner family, such as a unit's temporaries."""
        for owner in [name for name in self._bindings if name.startswith(prefix)]:
            del self._bindings[owner]
        self._invalidate()

    def clear(self) -> None:
        """Drop every binding at the end of the owning scope."""
        self._bindings.clear()
        self._invalidate()

    def bump(self) -> None:
        """Invalidate for an ownership change this ledger does not itself record."""
        self._invalidate()

    def union(
        self, *, devices: tuple[jax.Device, ...] | None = None
    ) -> DeviceBufferFootprint:
        """Merge the current epoch's spans once per requested device projection."""
        cached = self._unions.get(devices)
        if cached is None:
            cached = union_buffer_footprints(
                footprints=tuple(self._bindings.values()), devices=devices
            )
            self._unions[devices] = cached
        return cached

    def _invalidate(self) -> None:
        """Make every cached union unusable before any owner can disappear."""
        self.epoch += 1
        self._unions.clear()
