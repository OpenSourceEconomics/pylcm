"""Call-scoped known-buffer residency for forward execution.

The simulation loop owns the arrays. This scope retains only the current unit's
explicit transient roots; permanent inputs and published outputs are represented
by address metadata while their actual owners remain alive in the loop.
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
    measure_buffer_footprint,
    require_transfer_headroom,
    union_buffer_footprints,
)
from _lcm.simulation.value_reads import PeriodSimulationReads


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

    def __post_init__(self) -> None:
        """Own the call's selected common specialization independently of its caller."""
        self.axis_widths = MappingProxyType(dict(self.axis_widths))

    def snapshot(self, *, additional: object = ()) -> DeviceBufferFootprint:
        """Drain known transfers and inventory the current explicit live roots."""
        roots = (
            self.unit_inputs,
            self.derived,
            () if self.period_owner is None else self.period_owner.live_values,
            additional,
        )
        jax.block_until_ready(roots)
        return union_buffer_footprints(
            footprints=(
                self.inputs,
                self.chunk_inputs,
                self.outputs,
                measure_buffer_footprint(tree=roots),
            )
        )

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
        """Keep host intermediates alive and counted through the unit's commit."""
        self.unit_inputs = (self.unit_inputs, tree)

    def before_transfer(
        self, *, transfer: ResolvedValueTransfer, live_values: tuple[jax.Array, ...]
    ) -> None:
        """Check new copy and scratch storage before the physical transfer starts."""
        cost = transfer.cost
        required_devices = transfer.source_sharding.device_set
        participating_devices = required_devices | transfer.stored_sharding.device_set
        require_transfer_headroom(
            live=self.snapshot(additional=live_values),
            destination_bytes=dict.fromkeys(required_devices, cost.per_device_bytes),
            scratch_bytes=dict.fromkeys(participating_devices, cost.temporary_bytes),
            budget_bytes=self.budget_bytes,
            devices=self.devices,
        )

    def check_resident(self) -> None:
        """Refuse an already-infeasible known-buffer inventory."""
        require_transfer_headroom(
            live=self.snapshot(),
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
                live_footprint=self.snapshot,
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
