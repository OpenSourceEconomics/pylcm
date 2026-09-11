"""Period-local ownership of the solved values read by simulation dispatches.

NumPy replay originals are uploaded directly to the ordered subject layout and
kept unchanged. Their independent device copies share the normal output-alias
guards and last-unit release barrier. Host uploads are explicitly unsupported
when the owner has only a device-transfer budget callback.
"""

import logging
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax
import numpy as np
from jaxtyping import PyTree

from _lcm.execution.core_program import ValueRead
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.scheduler import (
    BufferRegistry,
    PeriodTransferCache,
    release_closed_artifacts,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueTransferKind,
    _validate_edge_identity,
    apply_value_transfer,
    classify_value_transfer,
    resolve_value_transfer,
)
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.typing import RegimeName
from lcm.exceptions import ExecutionPlanningError

type _CopyKey = tuple[ValueArtifactAddress, jax.sharding.Sharding]

_logger = logging.getLogger(__name__)


@runtime_checkable
class BeforeValueTransfer(Protocol):
    """Check copy residency before a nonaligned transfer can allocate."""

    def __call__(
        self,
        *,
        transfer: ResolvedValueTransfer,
        live_values: tuple[jax.Array, ...],
    ) -> None:
        """Inspect exact transfer metadata and transient owner-held arrays."""
        ...


class PeriodSimulationReads:
    """Acquire value arguments and close one entire active regime at a time."""

    def __init__(
        self,
        *,
        period: int,
        devices: tuple[jax.Device, ...],
        reads_by_unit: Mapping[RegimeName, tuple[ValueRead, ...]],
        release_enabled: bool,
        before_transfer: BeforeValueTransfer | None = None,
    ) -> None:
        """Keep exact reader occurrences and count distinct pending regime units."""
        self._devices = devices
        self._reads_by_unit = MappingProxyType(
            {unit: tuple(reads) for unit, reads in reads_by_unit.items()}
        )
        for unit, reads in self._reads_by_unit.items():
            for read in reads:
                if (
                    read.source.source_period != period
                    or read.source.source_regime != unit
                ):
                    msg = (
                        f"Declared reader {read.source!r} does not belong to "
                        f"period {period} and unit {unit!r}."
                    )
                    raise ExecutionPlanningError(msg)
        self._finished = False
        self._pending_units = set(reads_by_unit)
        self._release_enabled = release_enabled
        self._before_transfer = before_transfer
        self._registry = BufferRegistry()
        self._caches: dict[_CopyKey, PeriodTransferCache] = {}
        self._units_by_key: dict[_CopyKey, frozenset[RegimeName]] = {}
        self._source_values: dict[ValueArtifactAddress, jax.Array | np.ndarray] = {}
        self._copied_values: dict[_CopyKey, jax.Array] = {}
        self._host_ledgers: dict[
            _CopyKey, PlannedInputLiveness[RegimeName, _CopyKey]
        ] = {}
        self._inputs_by_unit: dict[RegimeName, list[jax.Array]] = {}
        self._pending_outputs: list[jax.Array] = []

    def read(
        self, *, unit: RegimeName, read: ValueRead, value: jax.Array | np.ndarray
    ) -> jax.Array:
        """Place a declared JAX value or NumPy replay leaf on the subject devices.

        A NumPy original must remain unchanged while its period is open. Its
        identity is retained without first materializing a default-device array.
        """
        self._check_open_unit(unit=unit)
        if read not in self._reads_by_unit[unit]:
            msg = f"Unit {unit!r} requested an undeclared value read: {read!r}."
            raise ExecutionPlanningError(msg)
        previous = self._source_values.get(read.target)
        if previous is not None and previous is not value:
            msg = f"The source array changed for value artifact {read.target!r}."
            raise ExecutionPlanningError(msg)
        if isinstance(value, np.ndarray):
            return self._read_host(unit=unit, read=read, value=value)
        required = simulation_value_sharding(
            stored_sharding=value.sharding, devices=self._devices
        )
        transfer = resolve_value_transfer(
            target=read.target,
            source=read.source,
            kind=classify_value_transfer(
                stored_sharding=value.sharding, required_sharding=required
            ),
            stored_template=value,
            source_sharding=required,
        )
        key = (read.target, required)
        cache = self._caches.get(key)
        if cache is None:
            units = frozenset(
                candidate
                for candidate in self._pending_units
                if any(
                    occurrence.target == read.target
                    for occurrence in self._reads_by_unit[candidate]
                )
            )
            self._units_by_key[key] = units
            cache = PeriodTransferCache(
                registry=self._registry,
                consumer_counts={key: len(units)},
                pending_outputs=self._pending_outputs,
                release_enabled=self._release_enabled,
            )
            self._caches[key] = cache
        transfer = replace(
            transfer,
            reused_by_several_consumers=len(self._units_by_key[key]) > 1,
        )
        self._source_values[read.target] = value
        self._registry.declare_not_produced(tree=value)
        copied = cache.get(transfer=transfer)
        if copied is None:
            if (
                transfer.kind is not ValueTransferKind.ALIGNED_LOCAL
                and self._before_transfer is not None
            ):
                self._before_transfer(transfer=transfer, live_values=self.live_values)
            copied = apply_value_transfer(value=value, transfer=transfer)
            cache.put(transfer=transfer, array=copied, stored=value)
            self._copied_values[key] = copied
        self._inputs_by_unit.setdefault(unit, []).append(copied)
        return copied

    def commit(self, *, unit: RegimeName, outputs: PyTree) -> None:
        """Retain the whole result/carry tree before closing this regime's reads."""
        self._check_open_unit(unit=unit)
        self._registry.declare_passed_through(
            inputs=self._inputs_by_unit.get(unit, ()), outputs=outputs
        )
        self._registry.declare_not_produced(tree=outputs)
        self._pending_outputs.extend(
            leaf for leaf in jax.tree.leaves(outputs) if isinstance(leaf, jax.Array)
        )
        jax.block_until_ready(outputs)
        for key, cache in self._caches.items():
            if unit in self._units_by_key[key]:
                cache.commit_consumer(key=key)
        for key, ledger in self._host_ledgers.items():
            if unit in ledger.pending_dispatches:
                closed = ledger.commit_successful_dispatch(dispatch=unit)
                if self._release_enabled:
                    release_closed_artifacts(
                        ledger=ledger,
                        registry=self._registry,
                        artifacts=closed,
                        arrays_by_artifact={key: self._copied_values[key]},
                        pending_outputs=self._pending_outputs,
                        closing_dispatch=unit,
                        logger=_logger,
                    )
        self._pending_units.remove(unit)
        self._inputs_by_unit.pop(unit, None)

    @property
    def live_values(self) -> tuple[jax.Array, ...]:
        """Return a transient view of undeleted sources, copies and pending outputs.

        Identity deduplication removes repeated wrappers, leaving physical shard
        overlap to the residency accountant. No snapshot is cached here; callers
        must not retain it beyond their immediate accounting or readiness check.
        """
        values = (
            *self._source_values.values(),
            *self._copied_values.values(),
            *self._pending_outputs,
        )
        return tuple(
            {
                id(value): value
                for value in values
                if isinstance(value, jax.Array) and not value.is_deleted()
            }.values()
        )

    def finish(self) -> None:
        """Verify every commit and discard concrete references, including on error.

        An incomplete scope reports its missing units after dropping its local
        wrappers. Explicit deletion remains solely the successful unit's job;
        cleanup never manually deletes an uncommitted input or eager alias.
        """
        self._finished = True
        try:
            jax.block_until_ready(tuple(self._pending_outputs))
            if self._pending_units:
                msg = (
                    "Simulation value reads have uncommitted units: "
                    f"{tuple(sorted(self._pending_units))!r}."
                )
                raise ExecutionPlanningError(msg)
        finally:
            self._caches.clear()
            self._source_values.clear()
            self._copied_values.clear()
            self._host_ledgers.clear()
            self._inputs_by_unit.clear()
            self._pending_outputs.clear()
            self._before_transfer = None

    def _read_host(
        self, *, unit: RegimeName, read: ValueRead, value: np.ndarray
    ) -> jax.Array:
        """Upload one replay leaf without inventing a stored device transfer."""
        if read.target.kind is not ValueArtifactKind.REPLAY_ARTIFACT_LEAF:
            raise ExecutionPlanningError("Host value reads require a replay artifact.")
        _validate_edge_identity(target=read.target, source=read.source)
        if self._before_transfer is not None:
            raise ExecutionPlanningError(
                "Host replay placement is not supported by the device-transfer "
                "budget callback; a host-allocation budget is required."
            )
        required = _host_replay_sharding(devices=self._devices)
        key = (read.target, required)
        copied = self._copied_values.get(key)
        if copied is None:
            units = tuple(
                candidate
                for candidate in self._pending_units
                if any(
                    occurrence.target == read.target
                    for occurrence in self._reads_by_unit[candidate]
                )
            )
            ledger: PlannedInputLiveness[RegimeName, _CopyKey] = PlannedInputLiveness(
                dispatch_accesses=dict.fromkeys(units, (key,))
            )
            uploaded = jax.device_put(value, required, may_alias=False)
            # Aligned NumPy storage can alias a CPU upload despite may_alias=False.
            # Give each artifact its own ready device buffer before registration.
            copied = uploaded.copy()
            copied.block_until_ready()
            self._registry.register(array=copied, artifact=key)
            self._host_ledgers[key] = ledger
            self._copied_values[key] = copied
        self._source_values[read.target] = value
        self._inputs_by_unit.setdefault(unit, []).append(copied)
        return copied

    def _check_open_unit(self, *, unit: RegimeName) -> None:
        """Require a declared unit whose single commit has not closed its reads."""
        if self._finished:
            msg = "The simulation value-read period is already finished."
            raise ExecutionPlanningError(msg)
        if unit not in self._reads_by_unit:
            msg = f"Unknown simulation value-read unit {unit!r}."
            raise ExecutionPlanningError(msg)
        if unit not in self._pending_units:
            msg = f"Simulation value-read unit {unit!r} is already committed."
            raise ExecutionPlanningError(msg)


def _host_replay_sharding(*, devices: tuple[jax.Device, ...]) -> jax.sharding.Sharding:
    """Replicate a host artifact directly on the ordered subject destination."""
    if not devices:
        raise ValueError("Simulation value placement requires at least one device.")
    if len(devices) == 1:
        return jax.sharding.SingleDeviceSharding(devices[0])
    mesh = jax.make_mesh(
        (len(devices),), ("X",), (jax.sharding.AxisType.Auto,), devices=devices
    )
    return jax.NamedSharding(mesh, jax.P())
