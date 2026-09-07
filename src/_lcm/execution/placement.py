"""Submesh placement: the devices each regime's nodes run on.

The planner receives the number of visible devices and assigns each regime a
device set before anything is lowered. A regime with a distributed grid runs on
the mesh its extent defines; a regime without one runs on a single device,
filling devices the sharded meshes leave idle in the periods it is active and
otherwise taking the device with the smallest planned footprint. The assignment
is a fact of the plan and of every compilation key, and it never changes a
value.

Every decision here is a pure function of the requests and the device count:
the caller resolves the visible devices and passes their number in, so the same
requests always yield the same placement.
"""

import dataclasses
import math
from collections.abc import Hashable, Mapping, Sequence
from types import MappingProxyType

from _lcm.typing import RegimeName
from lcm.exceptions import ExecutionPlanningError


@dataclasses.dataclass(frozen=True, kw_only=True)
class PlacementRequest:
    """What the planner needs to know about one regime."""

    regime_name: RegimeName
    """The regime's name; declaration order is the order of requests."""

    distributed_extents: tuple[int, ...]
    """Extents of the regime's distributed grids in declaration order; empty for
    a regime that runs on one device."""

    active_periods: tuple[int, ...]
    """Periods the regime has a node in."""

    template_bytes: int
    """Bytes of the regime's value template; the footprint weight of a node."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubmeshPlacement:
    """The device ids each regime's nodes run on."""

    device_ids_by_regime: Mapping[RegimeName, tuple[int, ...]]
    """Device ids per regime, ascending; one id for a single-device regime."""

    n_devices: int
    """Number of visible devices the plan was made for."""

    def __post_init__(self) -> None:
        """Freeze the mapping."""
        object.__setattr__(
            self,
            "device_ids_by_regime",
            MappingProxyType(dict(self.device_ids_by_regime)),
        )

    def devices_for(self, *, regime_name: RegimeName) -> tuple[int, ...]:
        """Return the device ids of one regime's nodes."""
        if regime_name not in self.device_ids_by_regime:
            msg = (
                f"Regime {regime_name!r} has no placement; the planner was given "
                f"{sorted(self.device_ids_by_regime)}."
            )
            raise ExecutionPlanningError(msg)
        return self.device_ids_by_regime[regime_name]

    @property
    def key(self) -> Hashable:
        """Return the placement as a hashable, order-independent record."""
        return tuple(sorted(self.device_ids_by_regime.items()))

    @property
    def is_canonical(self) -> bool:
        """Whether every regime sits where a one-device or full-mesh solve puts it."""
        every_device = tuple(range(self.n_devices))
        return all(
            ids in {(0,), every_device} for ids in self.device_ids_by_regime.values()
        )


def mesh_size_for_extents(*, extents: Sequence[int], n_devices: int) -> int:
    """Return the number of devices a regime's distributed grids define.

    One distributed grid runs on the largest divisor of its extent that does not
    exceed the visible devices, so the extent is divisible by the mesh size.
    Several distributed grids scatter one point per device and need exactly their
    product of extents.
    """
    if n_devices < 1:
        msg = f"A placement needs at least one device; got {n_devices}."
        raise ExecutionPlanningError(msg)
    if len(extents) == 1:
        extent = extents[0]
        return max(size for size in range(1, n_devices + 1) if extent % size == 0)
    product = math.prod(extents)
    if product > n_devices:
        msg = (
            "When distributing over multiple grids, the product of the number of "
            "points in the grids must equal the number of available devices. "
            f"Gridpoints product: {product} Available devices: {n_devices}"
        )
        raise ExecutionPlanningError(msg)
    return product


def plan_submesh_placement(
    *, requests: Sequence[PlacementRequest], n_devices: int
) -> SubmeshPlacement:
    """Assign every regime its device ids.

    Sharded regimes take consecutive blocks in declaration order, wrapping to the
    first block when no full block is left. A single-device regime takes the
    lowest device — among the devices every sharded mesh leaves idle when there
    are any, else among all — that no regime co-active with it occupies; when
    every candidate is occupied it takes the device with the smallest planned
    footprint, ties to the lowest id.
    """
    if n_devices < 1:
        msg = f"A placement needs at least one device; got {n_devices}."
        raise ExecutionPlanningError(msg)
    device_ids: dict[RegimeName, tuple[int, ...]] = {}
    footprint = dict.fromkeys(range(n_devices), 0)
    offset = 0
    for request in requests:
        if not request.distributed_extents:
            continue
        size = mesh_size_for_extents(
            extents=request.distributed_extents, n_devices=n_devices
        )
        start = offset if offset + size <= n_devices else 0
        block = tuple(range(start, start + size))
        offset = start + size
        device_ids[request.regime_name] = block
        for device in block:
            footprint[device] += request.template_bytes // size
    sharded_devices = {device for ids in device_ids.values() for device in ids}
    idle = tuple(device for device in range(n_devices) if device not in sharded_devices)
    candidates = idle or tuple(range(n_devices))
    for request in requests:
        if request.distributed_extents:
            continue
        busy = {
            device
            for other in requests
            if other.regime_name in device_ids
            and set(other.active_periods) & set(request.active_periods)
            for device in device_ids[other.regime_name]
        }
        free = tuple(device for device in candidates if device not in busy)
        if free:
            device = free[0]
        else:
            device = min(range(n_devices), key=_FootprintRank(footprint=footprint))
        device_ids[request.regime_name] = (device,)
        footprint[device] += request.template_bytes
    return SubmeshPlacement(device_ids_by_regime=device_ids, n_devices=n_devices)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FootprintRank:
    """Rank devices by planned footprint, then by id; a key for `min`."""

    footprint: Mapping[int, int]
    """Bytes already planned for each device id."""

    def __call__(self, device: int) -> tuple[int, int]:
        """Return the sort key of one device id."""
        return (self.footprint[device], device)
