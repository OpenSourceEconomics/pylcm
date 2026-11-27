"""Bind call-local runtime executors to the declared simulation programs."""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

import jax

from _lcm.engine import Regime
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation.agent_devices import simulation_agent_devices
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
        agent_devices=_agent_devices(
            regimes=regimes, device_ids=execution.device_ids, execution=execution
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


def _agent_devices(
    *,
    regimes: Mapping[RegimeName, Regime],
    device_ids: tuple[int, ...],
    execution: ResolvedExecution | None = None,
) -> tuple[jax.Device, ...]:
    """Resolve the same forward devices used by profiles and population carriers."""
    return simulation_agent_devices(
        regimes=regimes, device_ids=device_ids, execution=execution
    )
