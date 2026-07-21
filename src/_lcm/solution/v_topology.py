"""Per-regime value-function topology: shapes and device shardings.

Shared leaf: the backward-induction hot path sizes its continuation-input
templates from it, the failure-path diagnostics rebuild the rolling V mapping
from it, and the simulate-side AOT compile reuses the same templates.
"""

from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.engine import Regime, _build_regime_sharding
from _lcm.typing import FlatParams, RegimeName, StateName
from lcm.typing import FloatND


@dataclass(frozen=True)
class _RegimeVTopology:
    """Shape and (optional) sharding of a single regime's V-array."""

    shape: tuple[int, ...]
    """V-array shape, with one entry per state."""

    sharding: jax.NamedSharding | None
    """Device sharding for the V-array, or `None` when no state is distributed."""


def _get_regime_V_shapes_and_shardings(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> dict[RegimeName, _RegimeVTopology]:
    """Compute V-array shapes and shardings for every regime.

    The V-array has one dimension per state variable, sized by that state's
    grid. When at least one state grid in a regime is distributed, the
    V-array is sharded across devices along those axes; otherwise the
    sharding is `None`.

    Args:
        regimes: Immutable mapping of regime names to internal regimes.
        flat_params: Regime parameters (needed for runtime grid shapes).

    Returns:
        Dict of regime names to `_RegimeVTopology` (shape and sharding).

    """
    n_devices = len(jax.devices())
    topology: dict[RegimeName, _RegimeVTopology] = {}
    for regime_name, regime in regimes.items():
        state_action_space = regime.solution.state_action_space(
            regime_params=flat_params[regime_name],
        )
        state_order: tuple[StateName, ...] = tuple(state_action_space.states.keys())
        shape = tuple(len(v) for v in state_action_space.states.values())
        sharding_plan = _build_regime_sharding(
            grids=regime.solution.grids, n_devices=n_devices
        )
        sharding = (
            sharding_plan.V_arr_sharding(state_order)
            if sharding_plan is not None
            else None
        )
        topology[regime_name] = _RegimeVTopology(shape=shape, sharding=sharding)
    return topology


def _build_zero_V_arr(*, topology: _RegimeVTopology) -> FloatND:
    """Build the zero V-array template for a regime, sharded where requested."""
    zeros = jnp.zeros(topology.shape)
    if topology.sharding is None:
        return zeros
    return jax.device_put(zeros, topology.sharding)
