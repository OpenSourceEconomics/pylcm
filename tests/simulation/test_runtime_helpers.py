"""Bind explicit execution owners for tests invoking the engine without a Model."""

from types import MappingProxyType

import jax

from _lcm.engine import Regime
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation.compile import bind_simulation_runtime
from _lcm.typing import RegimeName


def bind_eager_simulation(
    *, regimes: MappingProxyType[RegimeName, Regime]
) -> MappingProxyType[RegimeName, Regime]:
    """Attach a default executor to raw canonical regimes in kernel-level tests."""
    return bind_simulation_runtime(
        regimes=regimes,
        execution=ResolvedExecution(
            device_ids=tuple(device.id for device in jax.devices()),
            sharded_states=frozenset(
                name
                for regime in regimes.values()
                for name in regime.solution.sharded_state_names
            ),
            axis_widths=MappingProxyType({}),
            device_memory_bytes=None,
        ),
        enable_jit=False,
    )
