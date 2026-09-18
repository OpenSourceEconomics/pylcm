"""Admit subject-membership setup before allocating its device arrays."""

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.collective import NO_ROLE
from _lcm.simulation.initial_conditions import MISSING_CAT_CODE
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from lcm.typing import Int1D, ScalarInt


def initialize_subject_membership(
    *,
    initial_regime_ids: Int1D,
    initial_own_stakeholder: Int1D,
    memory: SimulationMemory | None = None,
) -> tuple[Int1D, Int1D]:
    """Create empty regime and stakeholder carriers for a subject chunk."""
    return run_simulation_operation(
        memory=memory,
        function=_empty_subject_membership,
        arguments={
            "initial_regime_ids": initial_regime_ids,
            "initial_own_stakeholder": initial_own_stakeholder,
        },
        subject_arg_names=("initial_regime_ids", "initial_own_stakeholder"),
        subject_outputs=True,
    )


def activate_subject_membership(
    *,
    period: int,
    starting_periods: Int1D,
    initial_regime_ids: Int1D,
    initial_own_stakeholder: Int1D,
    regime_ids: Int1D,
    own_stakeholder: Int1D,
    memory: SimulationMemory | None = None,
) -> tuple[Int1D, Int1D]:
    """Seed both membership carriers exactly when each subject enters."""
    return run_simulation_operation(
        memory=memory,
        function=_activate_subject_membership,
        arguments={
            "period": period if memory is None else np.int32(period),
            "starting_periods": starting_periods,
            "initial_regime_ids": initial_regime_ids,
            "initial_own_stakeholder": initial_own_stakeholder,
            "regime_ids": regime_ids,
            "own_stakeholder": own_stakeholder,
        },
        subject_arg_names=(
            "starting_periods",
            "initial_regime_ids",
            "initial_own_stakeholder",
            "regime_ids",
            "own_stakeholder",
        ),
        subject_outputs=True,
    )


def _empty_subject_membership(
    *, initial_regime_ids: Int1D, initial_own_stakeholder: Int1D
) -> tuple[Int1D, Int1D]:
    """Represent subjects that have not yet entered by the two missing codes."""
    return (
        jnp.full_like(initial_regime_ids, MISSING_CAT_CODE, dtype=jnp.int32),
        jnp.full_like(initial_own_stakeholder, NO_ROLE, dtype=jnp.int32),
    )


def _activate_subject_membership(
    *,
    period: int | ScalarInt,
    starting_periods: Int1D,
    initial_regime_ids: Int1D,
    initial_own_stakeholder: Int1D,
    regime_ids: Int1D,
    own_stakeholder: Int1D,
) -> tuple[Int1D, Int1D]:
    """Preserve existing memberships except for subjects entering this period."""
    entering = starting_periods == period
    return (
        jnp.where(entering, initial_regime_ids, regime_ids),
        jnp.where(entering, initial_own_stakeholder, own_stakeholder),
    )
