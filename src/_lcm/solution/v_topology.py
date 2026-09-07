"""Per-regime value-function topology: shapes and device shardings.

Shared leaf: the backward-induction hot path sizes its continuation-input
templates from it, the failure-path diagnostics rebuild the rolling V mapping
from it, and the simulate-side AOT compile reuses the same templates.

The two phases read different placements. Backward induction places a regime's
value on the devices the planner assigned it, which can be a submesh of the
visible devices. Simulation spreads subjects over every device, so it reads the
canonical layout instead — the one a solve without a per-regime placement
produces — and `canonical_solution_values` brings a solved value onto it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp

from _lcm.engine import Regime, _build_regime_sharding, _RegimeSharding
from _lcm.typing import FlatParams, RegimeName, StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND, ValueND


@dataclass(frozen=True)
class _RegimeVTopology:
    """Shape and (optional) sharding of a single regime's V-array."""

    shape: tuple[int, ...]
    """V-array shape, with one entry per state."""

    sharding: jax.sharding.Sharding | None
    """Device sharding for the V-array, or `None` for the default placement."""


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


def canonical_solution_values(
    *,
    values: Mapping[int, Mapping[RegimeName, ValueND]],
    regimes: MappingProxyType[RegimeName, Regime],
) -> MappingProxyType[int, MappingProxyType[RegimeName, ValueND]]:
    """Return the values on the layout the simulate programs are lowered against.

    A single-device value placed off the default device is copied onto it.
    Serves every array a solve publishes per period and regime — the value
    functions and the collective dissolution flags beside them — since all of
    them leave the solve on their regime's own placement.

    Args:
        values: Mapping of period to a mapping of regime name to the array the
            solve published there.
        regimes: Immutable mapping of regime names to canonical regimes.

    Returns:
        Immutable mapping of period to an immutable mapping of regime name to
        the array on the canonical layout.

    """
    fail_if_a_value_is_on_a_proper_submesh(regimes=regimes)
    default = jax.devices()[0]
    result: dict[int, MappingProxyType[RegimeName, ValueND]] = {}
    for period, by_regime in values.items():
        placed: dict[RegimeName, ValueND] = {}
        for regime_name, value in by_regime.items():
            device_ids = regimes[regime_name].solution.submesh_device_ids
            placed[regime_name] = (
                jax.device_put(value, default)
                if len(device_ids) == 1 and value.sharding.device_set != {default}
                else value
            )
        result[period] = MappingProxyType(placed)
    return MappingProxyType(result)


def fail_if_a_value_is_on_a_proper_submesh(
    *, regimes: MappingProxyType[RegimeName, Regime]
) -> None:
    """Refuse a solve whose values simulation cannot read.

    Simulation spreads its subjects over every visible device, so it reads a
    value that is either on one device or on all of them. A value on a proper
    submesh — a distributed extent the visible device count does not divide —
    meets no such population and is refused rather than silently gathered.

    Args:
        regimes: Immutable mapping of regime names to canonical regimes.

    """
    n_devices = len(jax.devices())
    for regime_name, regime in regimes.items():
        device_ids = regime.solution.submesh_device_ids
        if 1 < len(device_ids) < n_devices:
            extents = tuple(
                grid.to_jax().shape[0]
                for grid in regime.solution.grids.values()
                if grid.distributed
            )
            msg = (
                f"Regime {regime_name!r} was solved on a submesh of "
                f"{len(device_ids)} of {n_devices} devices (distributed extents "
                f"{extents!r}); simulation spreads subjects over every device "
                "and cannot read a value from a proper submesh."
            )
            raise ExecutionPlanningError(msg)


def placed_V_sharding(
    *,
    sharding_plan: _RegimeSharding | None,
    state_order: tuple[StateName, ...],
    devices: tuple[jax.Device, ...],
) -> jax.sharding.Sharding | None:
    """Return the sharding a regime's value template is committed to.

    A regime with a distributed state takes its mesh's spec. A regime without
    one keeps an uncommitted template — the default placement — wherever a
    single-device solve would have put it anyway, and is committed to its own
    device only when the placement moved it elsewhere.

    Args:
        sharding_plan: The regime's mesh plan, or `None` when no state grid of
            it is distributed.
        state_order: The V-array's state axes, in order.
        devices: Tuple of the devices the regime's nodes run on.

    Returns:
        The sharding to commit the template to, or `None` for the default
        placement.

    """
    if sharding_plan is not None:
        return sharding_plan.V_arr_sharding(state_order)
    visible = tuple(jax.devices())
    if devices == (visible[0],) or len(devices) == len(visible):
        return None
    return jax.sharding.SingleDeviceSharding(devices[0])


def _get_regime_V_shapes_and_shardings(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    phase: Literal["solve", "simulate"] = "solve",
) -> dict[RegimeName, _RegimeVTopology]:
    """Compute V-array shapes and shardings for every regime.

    The V-array has one dimension per state variable, sized by that state's
    grid. When at least one state grid in a regime is distributed, the
    V-array is sharded across the devices the phase places the regime on;
    otherwise it is committed only where the placement moved the regime off
    the default device.

    Args:
        regimes: Immutable mapping of regime names to internal regimes.
        flat_params: Regime parameters (needed for runtime grid shapes).
        phase: Which placement to build against — `"solve"` reads each
            regime's assigned devices, `"simulate"` every visible device,
            which is the canonical layout simulation's programs are lowered
            against.

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
        devices = (
            regime.solution.placed_devices()
            if phase == "solve"
            else tuple(jax.devices())
        )
        topology[regime_name] = _RegimeVTopology(
            shape=shape,
            sharding=placed_V_sharding(
                sharding_plan=_build_regime_sharding(
                    grids=regime.solution.grids, devices=devices
                ),
                state_order=state_order,
                devices=devices,
            ),
        )
    return topology


def _build_zero_V_arr(*, topology: _RegimeVTopology) -> FloatND:
    """Build the zero V-array template for a regime, sharded where requested."""
    zeros = jnp.zeros(topology.shape)
    if topology.sharding is None:
        return zeros
    return jax.device_put(zeros, topology.sharding)
