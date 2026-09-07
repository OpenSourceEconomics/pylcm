"""The per-device bytes the plan predicts for solve-lifetime artifacts.

A static model, read off the solve-lifetime templates before anything runs:
every artifact the execution plan knows has a footprint — the bytes of its
shard and the devices it holds a shard on — and every dispatch unit of the
schedule knows which devices it runs on and which artifacts its outputs are
registered under. A replicated array costs its full size on each of its
devices; a sharded one costs a single shard.

`plan_resident_bytes` walks that schedule the way the loop will run it and
reports, per unit, what it already finds on its busiest device.
"""

import dataclasses
import math
from collections.abc import Hashable, Mapping
from types import MappingProxyType

import jax

from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.typing import RegimeName
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ValueND


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


def plan_resident_bytes(
    *,
    waves_by_period: Mapping[int, tuple[tuple[ScheduledUnit, ...], ...]],
    fold_dispatches: Mapping[tuple[int, RegimeName, RegimeName], Hashable],
    ledger: PlannedInputLiveness,
    footprints: Mapping[Hashable, ArtifactFootprint],
) -> MappingProxyType[tuple[int, RegimeName], int]:
    """Return, per unit, the busiest per-device resident bytes at its position.

    The schedule is walked the way the loop will run it: periods descending,
    waves in order, and a period's gated-edge folds after its last wave. A
    unit's number is what is already resident on its busiest device plus the
    outputs of the units dispatched concurrently with it on that device; its
    own outputs are not part of it, since the width it is being planned for
    decides them.

    `fold_dispatches` maps each fold dispatch id `(period, source, target)` to
    the artifact key it produces. Every unit's `(period, regime)` and every
    fold id must be a planned dispatch of the ledger, with or without
    accesses. An artifact the ledger counts but no template sizes occupies
    nothing; one the ledger does not know at all is refused, because its
    lifetime has no answer.

    An artifact leaves the footprint exactly where the ledger would release it:
    every key of its alias group at zero remaining consumers, none pinned and
    none retained. A donated buffer needs no rule of its own, because a
    dispatch donates only an artifact it is the sole remaining consumer of,
    with no pin, no retention and no alias peer — the same position at which
    the walk already releases it.
    """
    _fail_if_footprint_is_unplanned(ledger=ledger, footprints=footprints)
    counts = dict(ledger.remaining_counts)
    live: dict[Hashable, ArtifactFootprint] = {}
    resident: dict[tuple[int, RegimeName], int] = {}
    for period in sorted(waves_by_period, reverse=True):
        for wave in waves_by_period[period]:
            _walk_wave(
                period=period,
                wave=wave,
                ledger=ledger,
                footprints=footprints,
                counts=counts,
                live=live,
                resident=resident,
            )
        _walk_period_folds(
            period=period,
            fold_dispatches=fold_dispatches,
            ledger=ledger,
            footprints=footprints,
            counts=counts,
            live=live,
        )
    return MappingProxyType(resident)


def per_device_footprint(*, array: ValueND) -> ArtifactFootprint:
    """Read one array's footprint off its own sharding."""
    return layout_footprint(
        sharding=array.sharding,
        shape=tuple(int(size) for size in array.shape),
        item_bytes=array.dtype.itemsize,
    )


def layout_footprint(
    *,
    sharding: jax.sharding.Sharding,
    shape: tuple[int, ...],
    item_bytes: int,
) -> ArtifactFootprint:
    """Return the bytes one device holds of `shape` under `sharding`.

    The shape need not be realized: a planned layout is sized here before any
    array is placed on it, and a shape no mesh axis divides evenly is refused
    rather than reported as a fractional shard.
    """
    try:
        shard_shape = sharding.shard_shape(shape)
    except ValueError as error:
        msg = (
            f"An array of shape {shape} does not divide evenly over the planned "
            f"sharding {sharding}: {error}"
        )
        raise ExecutionPlanningError(msg) from error
    return ArtifactFootprint(
        bytes_per_device=item_bytes * math.prod(shard_shape),
        device_ids=sharding_device_ids(sharding=sharding),
    )


def sharding_device_ids(*, sharding: jax.sharding.Sharding) -> tuple[int, ...]:
    """Return the ascending ids of the devices one sharding places on."""
    return tuple(sorted(device.id for device in sharding.device_set))


def _walk_wave(
    *,
    period: int,
    wave: tuple[ScheduledUnit, ...],
    ledger: PlannedInputLiveness,
    footprints: Mapping[Hashable, ArtifactFootprint],
    counts: dict[Hashable, int],
    live: dict[Hashable, ArtifactFootprint],
    resident: dict[tuple[int, RegimeName], int],
) -> None:
    """Record what one wave's units find resident, then commit the whole wave.

    The three passes are ordered as the runtime dispatches them: every unit is
    measured against the state it starts from, all outputs land together, and
    only then do the accesses of the wave commit.
    """
    for unit in wave:
        _fail_if_period_disagrees(unit=unit, period=period)
        resident[(period, unit.regime)] = _busiest_device_bytes(
            unit=unit, wave=wave, live=live
        )
    for unit in wave:
        _register_outputs(produces=unit.produces, footprints=footprints, live=live)
    for unit in wave:
        _release_after_dispatch(
            dispatch=(period, unit.regime), ledger=ledger, counts=counts, live=live
        )


def _walk_period_folds(
    *,
    period: int,
    fold_dispatches: Mapping[tuple[int, RegimeName, RegimeName], Hashable],
    ledger: PlannedInputLiveness,
    footprints: Mapping[Hashable, ArtifactFootprint],
    counts: dict[Hashable, int],
    live: dict[Hashable, ArtifactFootprint],
) -> None:
    """Register and commit the gated-edge folds that run after a period's waves."""
    for dispatch, produced in fold_dispatches.items():
        if dispatch[0] != period:
            continue
        _register_outputs(produces=(produced,), footprints=footprints, live=live)
        _release_after_dispatch(
            dispatch=dispatch, ledger=ledger, counts=counts, live=live
        )


def _register_outputs(
    *,
    produces: tuple[Hashable, ...],
    footprints: Mapping[Hashable, ArtifactFootprint],
    live: dict[Hashable, ArtifactFootprint],
) -> None:
    """Make the sized outputs of one dispatch resident."""
    for artifact in produces:
        if artifact in footprints:
            live[artifact] = footprints[artifact]


def _busiest_device_bytes(
    *,
    unit: ScheduledUnit,
    wave: tuple[ScheduledUnit, ...],
    live: Mapping[Hashable, ArtifactFootprint],
) -> int:
    """Return the resident bytes on the unit's most occupied device."""
    return max(
        _device_bytes(live=live, device=device)
        + sum(
            peer.output_bytes_per_device
            for peer in wave
            if peer is not unit and device in peer.device_ids
        )
        for device in unit.device_ids
    )


def _device_bytes(*, live: Mapping[Hashable, ArtifactFootprint], device: int) -> int:
    """Sum the resident bytes of every live artifact holding a shard on `device`."""
    return sum(
        footprint.bytes_per_device
        for footprint in live.values()
        if device in footprint.device_ids
    )


def _release_after_dispatch(
    *,
    dispatch: Hashable,
    ledger: PlannedInputLiveness,
    counts: dict[Hashable, int],
    live: dict[Hashable, ArtifactFootprint],
) -> None:
    """Decrement one dispatch's accesses and drop what the ledger would release."""
    for artifact in ledger.accesses_of(dispatch=dispatch):
        counts[artifact] -= 1
    for artifact in tuple(live):
        if all(
            counts[member] == 0
            and not ledger.is_pinned(artifact=member)
            and not ledger.is_retained(artifact=member)
            for member in ledger.alias_group(artifact=artifact)
        ):
            del live[artifact]


def _fail_if_footprint_is_unplanned(
    *,
    ledger: PlannedInputLiveness,
    footprints: Mapping[Hashable, ArtifactFootprint],
) -> None:
    """Reject a sized artifact the ledger cannot report a lifetime for."""
    for artifact in footprints:
        if not ledger.is_known(artifact=artifact):
            msg = (
                f"The footprint of {artifact!r} names an artifact outside the "
                "immutable plan, whose lifetime the ledger cannot report."
            )
            raise ExecutionPlanningError(msg)


def _fail_if_period_disagrees(*, unit: ScheduledUnit, period: int) -> None:
    """Reject a wave listed under a period its units do not solve."""
    if unit.period != period:
        msg = (
            f"Unit {unit.regime!r} carries period {unit.period}, but the "
            f"schedule lists it under period {period}."
        )
        raise ValueError(msg)


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
