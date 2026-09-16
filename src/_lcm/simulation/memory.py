"""Call-scoped known-buffer residency for forward execution.

The simulation loop owns the arrays. This scope retains only the current unit's
explicit transient roots; permanent inputs and published outputs are represented
by address metadata while their actual owners remain alive in the loop.

Each owner is measured once, when it is bound, and its spans are kept in a
call-local `OwnerLedger`. Every field assignment and every owner method advances the
ledger's epoch, so a merged union is reused only while the recorded ownership is
unchanged and admission always runs on the latest snapshot. Owners this scope does
not control — a period owner's materialized reads, a caller's transient tree — are
re-measured conservatively on every snapshot.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import jax

from _lcm.execution.value_transfer import ResolvedValueTransfer
from _lcm.simulation.host_operations import (
    ProfiledSimulationOperations,
    StaticArgument,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    OwnerLedger,
    measure_buffer_footprint,
    require_transfer_headroom,
    union_buffer_footprints,
)
from _lcm.simulation.value_reads import PeriodSimulationReads

# Fields already holding measured spans; the ledger binds them without a walk.
_LEDGER_FOOTPRINT_FIELDS = ("inputs", "outputs", "chunk_inputs")

# Fields holding live owner trees; assigning one measures it exactly once.
_LEDGER_TREE_FIELDS = ("unit_inputs", "derived")

# Owner-name family of the current unit's transient owners, released together.
_HELD_PREFIX = "held:"

type _Footprint = DeviceBufferFootprint

# Shared immutable metadata for an owner set holding no device payload.
_EMPTY_FOOTPRINT = DeviceBufferFootprint(spans={})


def _is_empty(*, tree: object) -> bool:
    """Recognize the empty transient tree without touching a numeric leaf."""
    return type(tree) is tuple and len(tree) == 0


@dataclass(kw_only=True)
class SimulationMemory:
    """Account for retained originals, growing outputs and the current period."""

    budget_bytes: int
    devices: tuple[jax.Device, ...]
    subject_devices: tuple[jax.Device, ...]
    operations: ProfiledSimulationOperations
    inputs: DeviceBufferFootprint
    axis_widths: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    outputs: DeviceBufferFootprint = field(
        default_factory=lambda: DeviceBufferFootprint(spans={})
    )
    chunk_inputs: DeviceBufferFootprint = field(
        default_factory=lambda: DeviceBufferFootprint(spans={})
    )
    unit_inputs: object = ()
    derived: object = ()
    period_owner: PeriodSimulationReads | None = None
    ledger: OwnerLedger = field(default_factory=OwnerLedger, repr=False)
    """Call-local span metadata for every owner this scope binds."""

    def __post_init__(self) -> None:
        """Own the call's selected common specialization independently of its caller."""
        self._held: list[object] = []
        self._period_generation: object = None
        self._period_footprint = _EMPTY_FOOTPRINT
        self._period_stable = True
        self._snapshots: dict[
            tuple[jax.Device, ...] | None, tuple[tuple[int, object], _Footprint]
        ] = {}
        self.axis_widths = MappingProxyType(dict(self.axis_widths))
        # The dataclass assigned the owner fields before `ledger` existed, so bind
        # their initial values now; every later assignment goes through the hook.
        for name in (*_LEDGER_FOOTPRINT_FIELDS, *_LEDGER_TREE_FIELDS):
            setattr(self, name, getattr(self, name))

    # keyword-only-exempt: library-callback=object.__setattr__
    def __setattr__(self, name: str, value: object) -> None:
        """Advance the ownership epoch on every mutation of a tracked owner field.

        Direct field assignment is the established way several call sites rebind
        unit roots and entry inputs. Routing every one of them through the ledger
        keeps a future call site from silently reusing a stale union.
        """
        object.__setattr__(self, name, value)
        if "ledger" not in self.__dict__:
            return
        if name in _LEDGER_FOOTPRINT_FIELDS:
            self.ledger.bind(owner=name, footprint=cast("DeviceBufferFootprint", value))
        elif name in _LEDGER_TREE_FIELDS:
            if name == "unit_inputs":
                # Rebinding the unit roots has always discarded the temporaries
                # held since the previous rebinding; release their owners too.
                self._held = []
                self.ledger.release_prefix(prefix=_HELD_PREFIX)
            if _is_empty(tree=value):
                # Releasing an owner set needs no measurement, only an epoch.
                self.ledger.bind(owner=name, footprint=_EMPTY_FOOTPRINT)
            else:
                self.ledger.measure(owner=name, tree=value)
        elif name == "period_owner":
            self._period_generation = None
            self._period_footprint = _EMPTY_FOOTPRINT
            self._period_stable = True
            self.ledger.bump()

    @property
    def residency_epoch(self) -> tuple[int, object]:
        """Identify the current ownership epoch, including the period owner's."""
        return (self.ledger.epoch, self._period_generation)

    def snapshot(
        self,
        *,
        additional: object = (),
        devices: tuple[jax.Device, ...] | None = None,
    ) -> DeviceBufferFootprint:
        """Drain known transfers and inventory the current explicit live roots."""
        period = self._period_snapshot()
        key = (self.ledger.epoch, self._period_generation)
        cached = self._snapshots.get(devices)
        if self._period_stable and cached is not None and cached[0] == key:
            combined = cached[1]
        else:
            base = self.ledger.union(devices=devices)
            combined = (
                base
                if not period.spans
                else union_buffer_footprints(footprints=(base, period), devices=devices)
            )
            if self._period_stable:
                self._snapshots[devices] = (key, combined)
        if _is_empty(tree=additional):
            return combined
        jax.block_until_ready(additional)
        return union_buffer_footprints(
            footprints=(combined, measure_buffer_footprint(tree=additional)),
            devices=devices,
        )

    def budget_snapshot(self, *, additional: object = ()) -> DeviceBufferFootprint:
        """Read current live roots while projecting retained admission metadata."""
        return self.snapshot(additional=additional, devices=self.devices)

    def set_chunk_inputs(self, *, tree: object) -> None:
        """Replace a chunk's grids/params while its actual owners remain alive."""
        self.chunk_inputs = measure_buffer_footprint(tree=tree)

    def publish(self, *, tree: object) -> None:
        """Record results whose actual owners survive the period."""
        self.outputs = union_buffer_footprints(
            footprints=(self.outputs, measure_buffer_footprint(tree=tree))
        )

    def replace_outputs(self, *, tree: object) -> None:
        """Reset publication metadata after offload and release of the old owners."""
        self.outputs = measure_buffer_footprint(tree=tree)

    def set_derived(self, tree: object) -> None:
        """Replace the current host adapter's live derived-input snapshot."""
        self.derived = tree

    def hold(self, tree: object) -> None:
        """Keep host intermediates alive and counted through the unit's commit.

        The owner joins the unit's live list and its spans are measured once here,
        so an admission check later in the same unit never re-walks it.
        """
        self._held.append(tree)
        self.ledger.measure(owner=f"{_HELD_PREFIX}{len(self._held)}", tree=tree)

    def before_transfer(
        self, *, transfer: ResolvedValueTransfer, live_values: tuple[jax.Array, ...]
    ) -> None:
        """Check new copy and scratch storage before the physical transfer starts."""
        cost = transfer.cost
        required_devices = transfer.source_sharding.device_set
        participating_devices = required_devices | transfer.stored_sharding.device_set
        require_transfer_headroom(
            live=self.budget_snapshot(additional=live_values),
            destination_bytes=dict.fromkeys(required_devices, cost.per_device_bytes),
            scratch_bytes=dict.fromkeys(participating_devices, cost.temporary_bytes),
            budget_bytes=self.budget_bytes,
            devices=self.devices,
        )

    def check_resident(self) -> None:
        """Refuse an already-infeasible known-buffer inventory."""
        require_transfer_headroom(
            live=self.budget_snapshot(),
            destination_bytes={},
            scratch_bytes={},
            budget_bytes=self.budget_bytes,
            devices=self.devices,
        )

    def run[T](
        self,
        *,
        function: Callable[..., T],
        arguments: Mapping[str, object],
        subject_arg_names: tuple[str, ...] = (),
        static_arguments: Mapping[str, StaticArgument] = MappingProxyType({}),
        subject_outputs: bool = False,
    ) -> T:
        """Profile one pure host operation and own its ready result for this unit."""
        result = cast(
            "T",
            self.operations.dispatch(
                function=function,
                arguments=arguments,
                subject_arg_names=subject_arg_names,
                static_arguments=static_arguments,
                subject_outputs=subject_outputs,
                devices=self.subject_devices,
                live_footprint=self.budget_snapshot,
                budget_devices=self.devices,
                budget_bytes=self.budget_bytes,
            ),
        )
        self.hold(tree=result)
        return result

    def close_unit(self) -> None:
        """Drop transient unit roots after whole-output readiness and publication."""
        self.unit_inputs = ()
        self.derived = ()

    def _period_snapshot(self) -> DeviceBufferFootprint:
        """Charge the period owner's live values, reusing only a proved generation.

        A period owner materializes reads, completes transfers and releases donated
        artifacts behind this scope. It is reused only while it publishes a
        `generation` that has not advanced; any other owner is re-measured.
        """
        owner = self.period_owner
        if owner is None:
            self._period_stable = True
            return _EMPTY_FOOTPRINT
        generation = getattr(owner, "generation", None)
        self._period_stable = generation is not None
        if self._period_stable and generation == self._period_generation:
            return self._period_footprint
        live_values = owner.live_values
        jax.block_until_ready(live_values)
        footprint = measure_buffer_footprint(tree=live_values)
        self._period_generation = generation
        self._period_footprint = footprint
        return footprint


def run_simulation_operation[T](
    *,
    memory: SimulationMemory | None,
    function: Callable[..., T],
    arguments: Mapping[str, object],
    subject_arg_names: tuple[str, ...] = (),
    static_arguments: Mapping[str, StaticArgument] = MappingProxyType({}),
    subject_outputs: bool = False,
) -> T:
    """Execute a pure operation under the current optional workspace budget."""
    if memory is None:
        return function(**arguments, **static_arguments)
    return memory.run(
        function=function,
        arguments=arguments,
        subject_arg_names=subject_arg_names,
        static_arguments=static_arguments,
        subject_outputs=subject_outputs,
    )
