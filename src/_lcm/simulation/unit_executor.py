"""Keep one regime unit's budget context outside the persistent runtime cache."""

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType

import jax

from _lcm.execution.core_program import CoreProgram
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    union_buffer_footprints,
)
from _lcm.simulation.runtime import SimulationDispatchContext, SimulationRuntime
from lcm.exceptions import ExecutionPlanningError


@dataclasses.dataclass(kw_only=True, eq=False)
class SimulationUnitExecutor:
    """Own raw program outputs while one regime unit consumes and publishes them.

    A host merge may replace a raw program output before the next program runs.
    Keep its actual owner beside its footprint until the whole-unit commit, then
    explicitly close this wrapper. The caller separately retains published and
    merged results. It never stores the wrapper in a persisted regime bundle.
    The callback drains represented transfers before measuring live buffers.
    No arrays, metadata or callback enter the shared executable cache.
    """

    runtime: SimulationRuntime
    live_footprint: Callable[[], DeviceBufferFootprint]
    budget_devices: tuple[jax.Device, ...]
    axis_widths: Mapping[str, int] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    on_output: Callable[[object], None] | None = None
    _outputs: list[DeviceBufferFootprint] = dataclasses.field(
        default_factory=list, init=False, repr=False
    )
    _owned_outputs: list[object] = dataclasses.field(
        default_factory=list, init=False, repr=False
    )
    _closed: bool = dataclasses.field(default=False, init=False, repr=False)

    def _live(self) -> DeviceBufferFootprint:
        if self._closed:
            raise ExecutionPlanningError("This simulation unit is closed.")
        return union_buffer_footprints(
            footprints=(self.live_footprint(), *self._outputs)
        )

    def dispatch(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[str, object],
        period: int,
        n_subjects: int,
    ) -> object:
        """Budget each dispatch afresh and finish its workspace before the next one."""
        if self._closed:
            raise ExecutionPlanningError("This simulation unit is closed.")
        budgeted = self.runtime.execution.device_memory_bytes is not None
        result = self.runtime.dispatch(
            program=program,
            arguments=arguments,
            period=period,
            n_subjects=n_subjects,
            residency=(
                SimulationDispatchContext(
                    live_footprint=self._live,
                    budget_devices=self.budget_devices,
                    axis_widths=self.axis_widths,
                )
                if budgeted
                else None
            ),
        )
        if budgeted:
            jax.block_until_ready(result)
            self._owned_outputs.append(result)
            self._outputs.append(measure_buffer_footprint(tree=result))
            if self.on_output is not None:
                self.on_output(result)
        return result

    def close(self) -> None:
        """End intermediate ownership after the caller's whole-unit commit barrier."""
        self._outputs.clear()
        self._owned_outputs.clear()
        self._closed = True
