"""The per-device bytes the plan predicts for solve-lifetime artifacts.

A static model, read off the solve-lifetime templates before anything runs:
every artifact the execution plan knows has a footprint — the bytes of its
shard and the devices it holds a shard on — and every dispatch unit of the
schedule knows which devices it runs on and which artifacts its outputs are
registered under. A replicated array costs its full size on each of its
devices; a sharded one costs a single shard.
"""

import dataclasses
import math
from collections.abc import Hashable

import jax

from _lcm.typing import RegimeName


@dataclasses.dataclass(frozen=True, kw_only=True)
class ArtifactFootprint:
    """One artifact's resident bytes per device and the devices it occupies."""

    bytes_per_device: int
    """Bytes of the artifact's shard on each of its devices."""

    device_ids: tuple[int, ...]
    """Ascending ids of the devices holding one shard each."""

    def __post_init__(self) -> None:
        """Reject a byte count or device set no allocator could produce."""
        _fail_if_negative(value=self.bytes_per_device, label="Artifact footprint")
        _fail_if_not_a_device_set(
            device_ids=self.device_ids, label="An artifact footprint"
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScheduledUnit:
    """One dispatch unit of the static schedule."""

    period: int
    """The period the unit solves."""

    regime: RegimeName
    """The regime whose kernel the unit dispatches."""

    device_ids: tuple[int, ...]
    """Devices the unit runs on."""

    produces: tuple[Hashable, ...]
    """Artifact keys the unit's outputs are registered under."""

    output_bytes_per_device: int
    """Bytes the unit's outputs occupy on each of its devices."""

    def __post_init__(self) -> None:
        """Reject a unit that runs nowhere or claims a negative output size."""
        _fail_if_negative(value=self.output_bytes_per_device, label="Unit output")
        _fail_if_not_a_device_set(device_ids=self.device_ids, label="A scheduled unit")


def per_device_footprint(*, array: jax.Array) -> ArtifactFootprint:
    """Read one array's footprint off its sharding."""
    sharding = array.sharding
    shard_shape = sharding.shard_shape(tuple(int(size) for size in array.shape))
    return ArtifactFootprint(
        bytes_per_device=math.prod(shard_shape) * array.dtype.itemsize,
        device_ids=tuple(sorted(device.id for device in sharding.device_set)),
    )


def _fail_if_negative(*, value: int, label: str) -> None:
    """Reject a byte count below zero, naming what carried it."""
    if value < 0:
        msg = f"{label} bytes cannot be negative, got {value}."
        raise ValueError(msg)


def _fail_if_not_a_device_set(*, device_ids: tuple[int, ...], label: str) -> None:
    """Reject an empty device tuple or one that names a device twice."""
    if not device_ids:
        msg = f"{label} must name at least one device."
        raise ValueError(msg)
    if len(set(device_ids)) != len(device_ids):
        msg = f"{label} names a repeated device: {device_ids!r}."
        raise ValueError(msg)
