"""One forward-device selection shared by compilation, profiles and carriers."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import jax

from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.typing import RegimeName
from lcm.exceptions import ExecutionPlanningError


@runtime_checkable
class _SubjectDeviceExecutor(Protocol):
    """A bound forward runtime supplies the authoritative ordered devices."""

    @property
    def subject_devices(self) -> tuple[jax.Device, ...]:
        """Return the execution devices used by its compiler and live admission."""
        ...


def simulation_subject_devices(
    *,
    regimes: Mapping[RegimeName, Regime],
    device_ids: tuple[int, ...],
    execution: ResolvedExecution | None = None,
) -> tuple[jax.Device, ...]:
    """Keep solve placement distinct from a requested population partition.

    Runtime binding supplies ``execution``. Later population/parameter setup
    reads that bound runtime instead of independently inferring devices from
    solve axes. Unbound internal callers retain the legacy fallback.
    """
    devices = placed_devices_for_ids(
        submesh_device_ids=(), visible_device_ids=device_ids
    )
    if execution is not None:
        if execution.simulation_sharding == "subjects":
            return devices
    else:
        executors = tuple(
            regime.simulation.programs.executor for regime in regimes.values()
        )
        bound = tuple(
            item for item in executors if isinstance(item, _SubjectDeviceExecutor)
        )
        if bound:
            executor = bound[0]
            if len(bound) != len(executors) or any(
                item is not executor for item in bound
            ):
                raise ExecutionPlanningError(
                    "Forward regimes must share one subject runtime."
                )
            selected = executor.subject_devices
            if (
                not selected
                or len(set(selected)) != len(selected)
                or not set(selected).issubset(devices)
            ):
                raise ExecutionPlanningError(
                    "Subject runtime devices escape model selection."
                )
            return selected
    return (
        devices
        if any(regime.solution.sharded_state_names for regime in regimes.values())
        else devices[:1]
    )
