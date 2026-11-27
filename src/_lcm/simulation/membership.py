"""Admit agent-membership setup before allocating its device arrays."""

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.collective import NO_ROLE
from _lcm.simulation.initial_conditions import MISSING_CAT_CODE
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from lcm.typing import Int1D, ScalarInt


def initialize_agent_membership(
    *,
    initial_regime_ids: Int1D,
    initial_own_member: Int1D,
    memory: SimulationMemory | None = None,
) -> tuple[Int1D, Int1D]:
    """Create empty regime and member carriers for an agent chunk."""
    return run_simulation_operation(
        memory=memory,
        function=_empty_agent_membership,
        arguments={
            "initial_regime_ids": initial_regime_ids,
            "initial_own_member": initial_own_member,
        },
        agent_arg_names=("initial_regime_ids", "initial_own_member"),
        agent_outputs=True,
    )


def activate_agent_membership(
    *,
    period: int,
    starting_periods: Int1D,
    initial_regime_ids: Int1D,
    initial_own_member: Int1D,
    regime_ids: Int1D,
    own_member: Int1D,
    memory: SimulationMemory | None = None,
) -> tuple[Int1D, Int1D]:
    """Seed both membership carriers exactly when each agent enters."""
    return run_simulation_operation(
        memory=memory,
        function=_activate_agent_membership,
        arguments={
            "period": period if memory is None else np.int32(period),
            "starting_periods": starting_periods,
            "initial_regime_ids": initial_regime_ids,
            "initial_own_member": initial_own_member,
            "regime_ids": regime_ids,
            "own_member": own_member,
        },
        agent_arg_names=(
            "starting_periods",
            "initial_regime_ids",
            "initial_own_member",
            "regime_ids",
            "own_member",
        ),
        agent_outputs=True,
    )


def _empty_agent_membership(
    *, initial_regime_ids: Int1D, initial_own_member: Int1D
) -> tuple[Int1D, Int1D]:
    """Represent agents that have not yet entered by the two missing codes."""
    return (
        jnp.full_like(initial_regime_ids, MISSING_CAT_CODE, dtype=jnp.int32),
        jnp.full_like(initial_own_member, NO_ROLE, dtype=jnp.int32),
    )


def _activate_agent_membership(
    *,
    period: int | ScalarInt,
    starting_periods: Int1D,
    initial_regime_ids: Int1D,
    initial_own_member: Int1D,
    regime_ids: Int1D,
    own_member: Int1D,
) -> tuple[Int1D, Int1D]:
    """Preserve existing memberships except for agents entering this period."""
    entering = starting_periods == period
    return (
        jnp.where(entering, initial_regime_ids, regime_ids),
        jnp.where(entering, initial_own_member, own_member),
    )
