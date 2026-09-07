"""The per-device bytes the plan predicts for solve-lifetime artifacts.

A static model, read off the solve-lifetime templates before anything runs:
every artifact the execution plan knows has a footprint — the bytes of its
shard and the devices it holds a shard on — and every dispatch unit of the
schedule knows which devices it runs on and which artifacts its outputs are
registered under. A replicated array costs its full size on each of its
devices; a sharded one costs a single shard.

`plan_resident_bytes` walks that schedule the way the loop will run it and
reports, per unit, what it already finds on its busiest device. Residency is
per buffer rather than per key: the keys of one alias group name a single
allocation and are charged once. What a unit is handed as a device argument is
left out, under the convention `plan_resident_bytes` states about the compiler
report this number is meant to be added to.
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

    consumes: tuple[Hashable, ...]
    """Artifact keys the unit's executables are handed as device arguments.

    A buffer named here is left out of the unit's resident bytes under the
    argument convention `plan_resident_bytes` states. A value the plan copies
    or reshards before the read does not belong here: the stored buffer and the
    copy are both live, and only the copy reaches the executable.
    """

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
    decides them, and neither are the buffers it is handed as arguments — see
    the argument convention below.

    **The argument convention.** This number is meant to be added to a
    compiler-reported peak, and it assumes that report already counts the
    buffers the executable receives as arguments. That is what the XLA CPU
    backend does: `peak_memory_in_bytes` there equals `argument + output +
    temp` and tracks the argument size one for one. So a unit's own arguments
    are left out here, and each buffer is charged exactly once across the sum.
    On a backend whose report excludes arguments the sum under-counts by
    exactly those bytes, which is the direction that overruns a device; such a
    backend needs the exclusion dropped rather than the budget widened, and the
    convention re-measured before either.

    Two approximations sit inside that convention. A program declaring no reads
    names no arguments, so the inputs the ledger pins on its behalf are charged
    both here and inside its peak — an over-count, the safe direction. And the
    exclusion is per unit: a buffer another unit holds stays charged to that
    unit, since it is not in that unit's peak.

    `fold_dispatches` maps each fold dispatch id `(period, source, target)` to
    the artifact key it produces. Every unit's `(period, regime)` and every
    fold id must be a planned dispatch of the ledger, with or without
    accesses. An artifact the ledger counts but no template sizes occupies
    nothing; one the ledger does not know at all is refused, because its
    lifetime has no answer.

    An alias group is one buffer under several names, so it is resident once,
    charged on each device at the largest claim any of its keys makes there;
    adding the keys up would bill one allocation several times. It leaves the
    footprint exactly where the ledger would release it: every key at zero
    remaining consumers, none pinned and none retained.

    A donated buffer needs no rule of its own. Donation aliases one of the
    unit's own arguments into its output, and an argument is what the
    convention above assigns to the peak, so the buffer is charged once like
    any other — never twice, and never not at all.
    """
    _fail_if_footprint_is_unplanned(ledger=ledger, footprints=footprints)
    counts = dict(ledger.remaining_counts)
    live: dict[frozenset[Hashable], dict[Hashable, ArtifactFootprint]] = {}
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
    live: dict[frozenset[Hashable], dict[Hashable, ArtifactFootprint]],
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
        _register_outputs(
            produces=unit.produces, footprints=footprints, ledger=ledger, live=live
        )
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
    live: dict[frozenset[Hashable], dict[Hashable, ArtifactFootprint]],
) -> None:
    """Register and commit the gated-edge folds that run after a period's waves."""
    for dispatch, produced in fold_dispatches.items():
        if dispatch[0] != period:
            continue
        _register_outputs(
            produces=(produced,), footprints=footprints, ledger=ledger, live=live
        )
        _release_after_dispatch(
            dispatch=dispatch, ledger=ledger, counts=counts, live=live
        )


def _register_outputs(
    *,
    produces: tuple[Hashable, ...],
    footprints: Mapping[Hashable, ArtifactFootprint],
    ledger: PlannedInputLiveness,
    live: dict[frozenset[Hashable], dict[Hashable, ArtifactFootprint]],
) -> None:
    """Make the sized outputs of one dispatch resident, one entry per buffer.

    Every key of an alias group names the same buffer, so they share one entry
    and each key contributes only its own claim on the devices it names.
    """
    for artifact in produces:
        if artifact in footprints:
            group = ledger.alias_group(artifact=artifact)
            live.setdefault(group, {})[artifact] = footprints[artifact]


def _busiest_device_bytes(
    *,
    unit: ScheduledUnit,
    wave: tuple[ScheduledUnit, ...],
    live: Mapping[frozenset[Hashable], Mapping[Hashable, ArtifactFootprint]],
) -> int:
    """Return the resident bytes on the unit's most occupied device."""
    consumed = frozenset(unit.consumes)
    return max(
        _device_bytes(live=live, device=device, consumed=consumed)
        + sum(
            peer.output_bytes_per_device
            for peer in wave
            if peer is not unit and device in peer.device_ids
        )
        for device in unit.device_ids
    )


def _device_bytes(
    *,
    live: Mapping[frozenset[Hashable], Mapping[Hashable, ArtifactFootprint]],
    device: int,
    consumed: frozenset[Hashable],
) -> int:
    """Sum what every live buffer holding a shard on `device` occupies there.

    A buffer is charged once, at the largest claim any of its keys makes on
    that device: the keys of an alias group are names for one allocation, so
    adding them up would bill the same bytes several times. A buffer the
    measured unit is handed as an argument on this device is not charged at
    all, under the argument convention `plan_resident_bytes` states.
    """
    return sum(
        _group_bytes(members=members, device=device)
        for members in live.values()
        if _group_is_present(members=members, device=device)
        and not _group_is_consumed(members=members, device=device, consumed=consumed)
    )


def _group_bytes(*, members: Mapping[Hashable, ArtifactFootprint], device: int) -> int:
    """Return one buffer's largest claim on `device` among the keys naming it."""
    return max(
        footprint.bytes_per_device
        for footprint in members.values()
        if device in footprint.device_ids
    )


def _group_is_present(
    *, members: Mapping[Hashable, ArtifactFootprint], device: int
) -> bool:
    """Whether any key of one buffer holds a shard on `device`."""
    return any(device in footprint.device_ids for footprint in members.values())


def _group_is_consumed(
    *,
    members: Mapping[Hashable, ArtifactFootprint],
    device: int,
    consumed: frozenset[Hashable],
) -> bool:
    """Whether the measured unit is handed this buffer as an argument there."""
    return any(
        key in consumed and device in footprint.device_ids
        for key, footprint in members.items()
    )


def _release_after_dispatch(
    *,
    dispatch: Hashable,
    ledger: PlannedInputLiveness,
    counts: dict[Hashable, int],
    live: dict[frozenset[Hashable], dict[Hashable, ArtifactFootprint]],
) -> None:
    """Decrement one dispatch's accesses and drop what the ledger would release."""
    for artifact in ledger.accesses_of(dispatch=dispatch):
        counts[artifact] -= 1
    for group in tuple(live):
        if all(
            counts[member] == 0
            and not ledger.is_pinned(artifact=member)
            and not ledger.is_retained(artifact=member)
            for member in group
        ):
            del live[group]


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
    """Reject a wave listed under a period its units do not solve.

    A `ValueError`, deliberately, where the rest of this module raises
    `ExecutionPlanningError`: a schedule mapping that contradicts its own keys
    is a malformed argument, not a plan the engine could have produced and has
    to refuse.
    """
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
