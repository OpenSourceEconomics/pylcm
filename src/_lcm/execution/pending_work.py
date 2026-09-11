"""Call-local completion ownership for budgeted solve execution."""

import sys
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import jax

from _lcm.execution.compiler_inputs import compiler_input_paths
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    TransferCache,
    apply_value_transfer_plan,
)
from lcm.typing import ValueND


@runtime_checkable
class BeforeArrayDelete(Protocol):
    """Completion hook called only after the existing owner authorizes deletion."""

    def __call__(self, *, arrays: tuple[ValueND, ...]) -> None:
        """Discharge witnesses before any selected array wrapper is invalidated."""
        ...


@dataclass(frozen=True, kw_only=True)
class _PendingRecord:
    """Returned arrays and the actual devices their outstanding work can occupy."""

    arrays: tuple[ValueND, ...]
    """Every returned array view, without guessing equivalence between aliases."""
    devices: frozenset[jax.Device]
    """Union of execution, transfer endpoints, and actual returned placements."""


class PendingSolveWork:
    """Own completion witnesses for one budgeted solve, without owning liveness."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        """Start with no dispatched work or retained numerical owners."""
        self._records: list[_PendingRecord] = []

    def before(self, *, devices: AbstractSet[jax.Device]) -> None:
        """Complete only previous work whose physical device footprint intersects."""
        selected = tuple(record for record in self._records if record.devices & devices)
        self._records[:] = [
            record for record in self._records if not record.devices & devices
        ]
        _drain(records=selected)

    def record(self, *, outputs: object, devices: AbstractSet[jax.Device]) -> None:
        """Own every returned array before a later output contract can raise."""
        arrays = tuple(
            {
                id(leaf): leaf
                for leaf in jax.tree.leaves(outputs)
                if isinstance(leaf, jax.Array)
            }.values()
        )
        if not arrays:
            return
        record = _PendingRecord(arrays=arrays, devices=frozenset(devices))
        # Take ownership first: even an invalid/deleted output must not prevent
        # cleanup of the other returned arrays if inspecting its placement fails.
        self._records.append(record)
        actual_devices = frozenset(
            device for array in arrays for device in array.devices()
        )
        self._records[-1] = replace(record, devices=record.devices | actual_devices)

    def before_delete(self, *, arrays: tuple[ValueND, ...]) -> None:
        """Discharge tracked wrappers before the existing owner invalidates them."""
        self.before(
            devices=frozenset(device for array in arrays for device in array.devices())
        )

    def close(self) -> None:
        """Drain remaining outputs and clear owners on normal or exceptional exit."""
        active_error = sys.exception()
        records = tuple(self._records)
        self._records.clear()
        try:
            _drain(records=records)
        except Exception as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(
                "Solve completion cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        finally:
            records = ()


@dataclass(frozen=True, kw_only=True)
class _MaterializedCopies:
    """Transient observation of returned transfers, including partial-plan failure."""

    arrays: list[ValueND] = field(default_factory=list)
    """Fresh copy wrappers returned in this call and not yet known complete."""

    def __call__(self, *, transfer: ResolvedValueTransfer, array: ValueND) -> None:
        """Hold the result before the transfer adapter validates its metadata."""
        # Complete declared endpoints were collected before allocation.
        del transfer
        self.arrays.append(array)

    def close(
        self, *, owner: PendingSolveWork, devices: AbstractSet[jax.Device]
    ) -> None:
        """Hand copies to the solve owner without obscuring a dispatch error."""
        active_error = sys.exception()
        try:
            owner.record(outputs=self.arrays, devices=devices)
        except Exception as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(
                "Retaining solve transfer completion witnesses also failed: "
                f"{cleanup_error}"
            )
        finally:
            self.arrays.clear()


def execute_with_pending_work(
    *,
    owner: PendingSolveWork,
    compiled: jax.stages.Compiled,
    arguments: Mapping[str, object],
    transfers: tuple[ResolvedValueTransfer, ...],
    cache: TransferCache | None,
    donates: bool,
) -> object:
    """Admit one asynchronous executable into a call-local completion lifetime.

    Before transfers, all concrete argument devices are conservative potential
    source endpoints. This can serialize a compiler-dead ordinary argument on a
    shared source device. After transfers, the exact selected executable proves
    which copy wrappers have a runtime dependency; no read-locator grammar or
    equivalence between different alias views is inferred here.
    """
    devices = (
        frozenset(
            device
            for leaf in jax.tree.leaves(
                (compiled.input_shardings, compiled.output_shardings)
            )
            if isinstance(leaf, jax.sharding.Sharding)
            for device in leaf.device_set
        )
        | frozenset(
            device
            for leaf in jax.tree.leaves(arguments)
            if isinstance(leaf, jax.Array)
            for device in leaf.devices()
        )
        | frozenset(
            device
            for transfer in transfers
            for sharding in (transfer.stored_sharding, transfer.source_sharding)
            for device in sharding.device_set
        )
    )
    owner.before(devices=devices)
    copies = _MaterializedCopies()
    try:
        planned_arguments = (
            apply_value_transfer_plan(
                arguments=arguments,
                plan=transfers,
                cache=cache,
                on_materialized=copies,
            )
            if transfers
            else arguments
        )
        kept_paths = compiler_input_paths(
            compiled=compiled, arguments=planned_arguments
        )
        with_paths, _ = jax.tree_util.tree_flatten_with_path(dict(planned_arguments))
        kept_arrays = {
            id(leaf)
            for path, leaf in with_paths
            if path in kept_paths and isinstance(leaf, jax.Array)
        }
        # A DCE'd input cannot make an output witness prove transfer completion.
        # Donation may invalidate the input wrapper at dispatch, so complete and
        # drop every fresh copy witness beforehand on that route as well.
        must_wait = tuple(
            array for array in copies.arrays if donates or id(array) not in kept_arrays
        )
        if must_wait:
            jax.block_until_ready(must_wait)
            waited = {id(array) for array in must_wait}
            copies.arrays[:] = [
                array for array in copies.arrays if id(array) not in waited
            ]
        output = compiled(**planned_arguments)
        # Include auxiliary and pass-through outputs before any layout check can
        # fail. Kept copy witnesses travel with the complete output tree.
        owner.record(outputs=(output, tuple(copies.arrays)), devices=devices)
        copies.arrays.clear()
        return output
    finally:
        copies.close(owner=owner, devices=devices)


def _drain(*, records: tuple[_PendingRecord, ...]) -> None:
    """Attempt every completion even when another returned witness is invalid."""
    failures: list[Exception] = []
    for record in records:
        for array in record.arrays:
            try:
                _complete_array(array=array)
            except Exception as error:  # noqa: BLE001 -- drain all, then re-raise
                failures.append(error)
    if failures:
        first, *additional = failures
        for error in additional:
            first.add_note(
                f"Additional completion failure: {type(error).__name__}: {error}"
            )
        raise first


def _complete_array(*, array: ValueND) -> None:
    """Never treat an invalidated completion witness as already ready."""
    if array.is_deleted():
        msg = "A solve completion witness was deleted before being discharged."
        raise RuntimeError(msg)
    array.block_until_ready()
