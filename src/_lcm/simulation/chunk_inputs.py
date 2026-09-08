"""Prepare call-invariant operands for one forward subject chunk."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax

from _lcm.engine import Regime, StateActionSpace, placed_devices_for_ids
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.typing import FlatParams, RegimeName, StateOrActionName
from lcm.typing import Float1D, Int1D, IntND


@dataclass(frozen=True, kw_only=True)
class SimulationChunkInputs:
    """Keep one chunk's shared grids and initial subject operands."""

    devices: tuple[jax.Device, ...]
    """Actual ordered devices that evaluate subjects."""
    flat_params: FlatParams
    """Parameters replicated on the subject devices."""
    initial_regime_ids: Int1D
    """Seeded regime ids in subject layout."""
    initial_own_stakeholder: Int1D
    """Seeded stakeholder roles in subject layout."""
    starting_periods: Int1D
    """Entry periods in subject layout."""
    base_state_action_spaces: Mapping[RegimeName, StateActionSpace]
    """Parameters-completed spaces shared across the chunk's periods."""


def prepare_simulation_chunk_inputs(
    *,
    initial_states: Mapping[StateOrActionName, Float1D | IntND],
    initial_regime_ids: Int1D,
    initial_own_stakeholder: Int1D,
    starting_periods: Int1D,
    flat_params: FlatParams,
    regimes: MappingProxyType[RegimeName, Regime],
    device_ids: tuple[int, ...],
    memory: SimulationMemory | None,
) -> SimulationChunkInputs:
    """Place initial operands and complete grids once for this subject chunk."""
    devices = placed_devices_for_ids(
        submesh_device_ids=(), visible_device_ids=device_ids
    )
    if not any(regime.solution.sharded_state_names for regime in regimes.values()):
        devices = devices[:1]
    if memory is not None:
        memory.unit_inputs = (
            initial_states,
            initial_regime_ids,
            initial_own_stakeholder,
            starting_periods,
        )
    shared = place_simulation_arguments(
        arguments={"params": flat_params},
        subject_arg_names=(),
        value_reads=(),
        devices=devices,
        budget_bytes=None if memory is None else memory.budget_bytes,
        live_footprint=None if memory is None else memory.snapshot(),
        budget_devices=() if memory is None else memory.devices,
    )
    flat_params = cast("FlatParams", shared["params"])
    if memory is not None:
        memory.hold(tree=flat_params)
    subject_inputs = place_simulation_arguments(
        arguments={
            "regime_ids": initial_regime_ids,
            "roles": initial_own_stakeholder,
            "starting_periods": starting_periods,
        },
        subject_arg_names=("regime_ids", "roles", "starting_periods"),
        value_reads=(),
        devices=devices,
        budget_bytes=None if memory is None else memory.budget_bytes,
        live_footprint=None if memory is None else memory.snapshot(),
        budget_devices=() if memory is None else memory.devices,
    )
    initial_regime_ids = cast("Int1D", subject_inputs["regime_ids"])
    initial_own_stakeholder = cast("Int1D", subject_inputs["roles"])
    starting_periods = cast("Int1D", subject_inputs["starting_periods"])
    base_state_action_spaces = MappingProxyType(
        {
            name: regime.solution.state_action_space(regime_params=flat_params[name])
            for name, regime in regimes.items()
        }
    )
    if memory is not None:
        memory.set_chunk_inputs(
            tree=(
                flat_params,
                tuple(
                    (space.states, space.actions)
                    for space in base_state_action_spaces.values()
                ),
                initial_states,
                initial_regime_ids,
                initial_own_stakeholder,
                starting_periods,
            )
        )
        memory.check_resident()
    return SimulationChunkInputs(
        devices=devices,
        flat_params=flat_params,
        initial_regime_ids=initial_regime_ids,
        initial_own_stakeholder=initial_own_stakeholder,
        starting_periods=starting_periods,
        base_state_action_spaces=base_state_action_spaces,
    )
