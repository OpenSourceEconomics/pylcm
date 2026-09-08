"""Per-regime value-function topology: shapes and device shardings.

Shared leaf: the backward-induction hot path sizes its continuation-input
templates from it, the failure-path diagnostics rebuild the rolling V mapping
from it, and the simulate-side AOT compile reuses the same templates.

Backward induction stores values on each regime's assigned devices. Simulation
reads period-owned replicas on the actual subject devices. This module computes
only their shapes and destination layouts; it never relocates a stored solution.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp

from _lcm.engine import (
    Regime,
    _build_regime_sharding,
    _RegimeSharding,
    placed_devices_for_ids,
)
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.typing import FlatParams, RegimeName, StateName
from lcm.typing import FloatND


@dataclass(frozen=True)
class _RegimeVTopology:
    """Shape and (optional) sharding of a single regime's V-array."""

    shape: tuple[int, ...]
    """V-array shape, with one entry per state."""

    sharding: jax.sharding.Sharding
    """Device sharding the V-array is committed to."""


def expected_V_rank(*, regime: Regime) -> int:
    """Return the number of axes this regime's stored value function has.

    Single source of truth for the rank: the topology below builds V to it, and
    the boundary that accepts a caller-supplied solution checks against it.

    - One axis per solve state that is kept as a grid axis. A folded state is
      integrated out by quadrature at solve time and contributes none.
    - One trailing stakeholder axis when the regime is collective.

    Args:
        regime: Canonical regime whose stored value function is being sized.

    Returns:
        The rank of this regime's value-function array.

    """
    n_state_axes = sum(
        1 for name in regime.solution.state_names if name not in regime.fold_state_names
    )
    return n_state_axes + (1 if regime.stakeholders is not None else 0)


def placed_V_sharding(
    *,
    sharding_plan: _RegimeSharding | None,
    state_order: tuple[StateName, ...],
    devices: tuple[jax.Device, ...],
) -> jax.sharding.Sharding:
    """Return the sharding a regime's value template is committed to.

    A regime with a distributed state takes its mesh's spec; one without takes
    the first of the devices it is placed on. Every value is committed, so a
    model that names a subset of the devices JAX reports never publishes on
    one it excluded — the process default device is a device like any other,
    and a model that does not own it must not land there.

    Args:
        sharding_plan: The regime's mesh plan, or `None` when no state grid of
            it is distributed.
        state_order: The V-array's state axes, in order.
        devices: Tuple of the devices the regime's nodes run on.

    Returns:
        The sharding to commit the template to.

    """
    if sharding_plan is not None:
        return sharding_plan.V_arr_sharding(state_order)
    return jax.sharding.SingleDeviceSharding(devices[0])


def _get_regime_V_shapes_and_shardings(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    phase: Literal["solve", "simulate"] = "solve",
    device_ids: tuple[int, ...] = (),
) -> dict[RegimeName, _RegimeVTopology]:
    """Compute V-array shapes and shardings for every regime.

    The V-array has one dimension per state variable, sized by that state's
    grid. Solve templates use each regime's assigned placement. Simulation
    templates replicate the value across the devices evaluating subjects;
    deriving their shape never requires a state grid to divide the larger
    simulation mesh.

    Args:
        regimes: Immutable mapping of regime names to internal regimes.
        flat_params: Regime parameters (needed for runtime grid shapes).
        phase: Which placement to build against — `"solve"` reads each
            regime's assigned devices; `"simulate"` uses a replicated read
            layout over the model's subject devices. With no sharded state,
            subjects run on the model's first device.
        device_ids: The model's device ids, ascending. Empty names every
            device JAX reports.

    Returns:
        Dict of regime names to `_RegimeVTopology` (shape and sharding).

    """
    topology: dict[RegimeName, _RegimeVTopology] = {}
    for regime_name, regime in regimes.items():
        state_action_space = regime.solution.state_action_space(
            regime_params=flat_params[regime_name],
        )
        # Folded IID-process states are integrated out of the stored value by
        # quadrature at solve time (`get_max_Q_over_a`'s fold reduction), so
        # they are NOT an axis of this regime's V-array — exclude them from
        # the shape/sharding topology the same way a co-mapped state's axis
        # is still present (co-map only relocates an axis for sharding; fold
        # removes it).
        state_order: tuple[StateName, ...] = tuple(
            name
            for name in state_action_space.states
            if name not in regime.fold_state_names
        )
        shape = tuple(
            len(v)
            for name, v in state_action_space.states.items()
            if name not in regime.fold_state_names
        )
        # A collective regime's V carries a trailing
        # stakeholder axis, so the zero template and the roll must too. The
        # sharding plan spans the state axes only; the trailing stakeholder
        # axis is replicated.
        if regime.stakeholders is not None:
            shape = (*shape, len(regime.stakeholders))
        assert len(shape) == expected_V_rank(regime=regime), (  # noqa: S101
            f"regime {regime_name!r}: V topology built rank {len(shape)}, "
            f"while the rank rule states {expected_V_rank(regime=regime)}"
        )
        devices = placed_devices_for_ids(
            submesh_device_ids=regime.solution.submesh_device_ids,
            visible_device_ids=device_ids,
        )
        sharding = placed_V_sharding(
            sharding_plan=_build_regime_sharding(
                grids=regime.solution.grids,
                sharded_state_names=regime.solution.sharded_state_names,
                devices=devices,
            ),
            state_order=state_order,
            devices=devices,
        )
        if phase == "simulate":
            simulation_devices = placed_devices_for_ids(
                submesh_device_ids=(), visible_device_ids=device_ids
            )
            if not any(item.solution.sharded_state_names for item in regimes.values()):
                simulation_devices = simulation_devices[:1]
            sharding = simulation_value_sharding(
                stored_sharding=sharding, devices=simulation_devices
            )
        topology[regime_name] = _RegimeVTopology(
            shape=shape,
            sharding=sharding,
        )
    return topology


def _build_zero_V_arr(*, topology: _RegimeVTopology) -> FloatND:
    """Build the zero V-array template for a regime, on the devices it is placed on."""
    return jax.device_put(jnp.zeros(topology.shape), topology.sharding)
