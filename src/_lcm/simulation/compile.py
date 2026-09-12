"""Bind call-local runtime executors to the declared simulation programs."""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

import jax

from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.typing import RegimeName


def bind_simulation_runtime(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    execution: ResolvedExecution,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, Regime]:
    """Attach one shared executor to a call-local copy of the program bundles."""
    executor = SimulationRuntime(
        execution=execution,
        enable_jit=enable_jit,
        subject_devices=_subject_devices(
            regimes=regimes, device_ids=execution.device_ids
        ),
    )
    return MappingProxyType(
        {
            name: dataclasses.replace(
                regime,
                simulation=dataclasses.replace(
                    regime.simulation,
                    programs=dataclasses.replace(
                        regime.simulation.programs, executor=executor
                    ),
                ),
            )
            for name, regime in regimes.items()
        }
    )


def _subject_devices(
    *, regimes: Mapping[RegimeName, Regime], device_ids: tuple[int, ...]
) -> tuple[jax.Device, ...]:
    """Resolve the actual population devices from the canonical regime axes."""
    devices = placed_devices_for_ids(
        submesh_device_ids=(), visible_device_ids=device_ids
    )
    return (
        devices
        if any(regime.solution.sharded_state_names for regime in regimes.values())
        else devices[:1]
    )
