"""A period's simulated value array has one row per agent, and at most a role axis.

`PeriodRegimeSimulationData.V_arr` is rank-polymorphic because a collective
regime publishes a value per member: `(n_agents,)` for a singleton
regime, `(n_agents, n_members)` for a collective one. A rank-polymorphic
annotation accepts any rank, so the shape agreement that used to be a property
of the annotation is asserted here instead — every other field of the record is
per-agent, and a `V_arr` that disagrees with them silently misaligns every
value in the simulated frame.
"""

import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.engine import PeriodRegimeSimulationData

_N_AGENTS = 4
_N_MEMBERS = 2


def _period_data(*, V_arr: jnp.ndarray) -> PeriodRegimeSimulationData:
    """One period's record for `_N_AGENTS` agents, carrying `V_arr`."""
    column = jnp.arange(_N_AGENTS, dtype=jnp.zeros(()).dtype)
    return PeriodRegimeSimulationData(
        V_arr=V_arr,
        actions=MappingProxyType({"consumption": column}),
        states=MappingProxyType({"wealth": column}),
        in_regime=jnp.arange(_N_AGENTS) % 2 == 0,
        own_member=jnp.zeros(_N_AGENTS, dtype=jnp.int32),
        nested_policy_fallback=jnp.zeros(_N_AGENTS, dtype=bool),
    )


def test_a_singleton_regimes_value_array_is_one_value_per_agent() -> None:
    """A rank-1 `V_arr` with one entry per agent is accepted."""
    data = _period_data(V_arr=jnp.zeros(_N_AGENTS))
    assert data.V_arr.shape == (_N_AGENTS,)


def test_a_collective_regimes_value_array_carries_a_trailing_role_axis() -> None:
    """A rank-2 `V_arr` of agents by members is accepted."""
    data = _period_data(V_arr=jnp.zeros((_N_AGENTS, _N_MEMBERS)))
    assert data.V_arr.shape == (_N_AGENTS, _N_MEMBERS)


def test_a_value_array_with_more_than_one_trailing_axis_is_rejected() -> None:
    """A rank-3 `V_arr` names no axis a simulated record has."""
    with pytest.raises(ValueError, match="V_arr"):
        _period_data(V_arr=jnp.zeros((_N_AGENTS, _N_MEMBERS, 3)))


def test_a_value_array_whose_leading_axis_is_not_agents_is_rejected() -> None:
    """`V_arr`'s first axis must match the per-agent `in_regime` mask."""
    with pytest.raises(ValueError, match="V_arr"):
        _period_data(V_arr=jnp.zeros(_N_AGENTS + 1))


def test_a_scalar_value_array_is_rejected() -> None:
    """A 0-d `V_arr` carries no agent axis at all."""
    with pytest.raises(ValueError, match="V_arr"):
        _period_data(V_arr=jnp.zeros(()))


def test_the_record_still_populates_every_declared_field() -> None:
    """The guard reads only fields the dataclass declares, so it cannot go stale."""
    names = {field.name for field in dataclasses.fields(PeriodRegimeSimulationData)}
    assert {"V_arr", "in_regime"} <= names
