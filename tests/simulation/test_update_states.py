from types import MappingProxyType

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from _lcm.simulation.transitions import _advance_states_for_agents


def test_advance_states_writes_new_values_for_selected_agents() -> None:
    """Selected agents receive the next-period array; others stay put."""
    states_per_regime = MappingProxyType(
        {
            "working": MappingProxyType({"wealth": jnp.array([10.0, 20.0, 30.0])}),
        }
    )
    next_states_per_regime = MappingProxyType(
        {
            "working": MappingProxyType({"wealth": jnp.array([15.0, 25.0, 35.0])}),
        }
    )
    agent_indices = jnp.array([True, False, True])

    next_states = _advance_states_for_agents(
        states_per_regime=states_per_regime,
        next_states_per_regime=next_states_per_regime,
        agent_indices=agent_indices,
    )

    assert_array_equal(next_states["working"]["wealth"], jnp.array([15.0, 20.0, 35.0]))


def test_advance_states_handles_multiple_regimes_and_states() -> None:
    """Regimes named in `next_states_per_regime` get updated; others stay intact."""
    states_per_regime = MappingProxyType(
        {
            "working": MappingProxyType(
                {
                    "wealth": jnp.array([10.0, 20.0]),
                    "health": jnp.array([1.0, 2.0]),
                }
            ),
            "retired": MappingProxyType({"wealth": jnp.array([100.0, 200.0])}),
        }
    )
    next_states_per_regime = MappingProxyType(
        {
            "working": MappingProxyType(
                {
                    "wealth": jnp.array([15.0, 25.0]),
                    "health": jnp.array([1.5, 2.5]),
                }
            ),
        }
    )
    agent_indices = jnp.array([True, True])

    next_states = _advance_states_for_agents(
        states_per_regime=states_per_regime,
        next_states_per_regime=next_states_per_regime,
        agent_indices=agent_indices,
    )

    assert_array_equal(next_states["working"]["wealth"], jnp.array([15.0, 25.0]))
    assert_array_equal(next_states["working"]["health"], jnp.array([1.5, 2.5]))
    assert_array_equal(next_states["retired"]["wealth"], jnp.array([100.0, 200.0]))


def test_advance_states_no_agents_selected_leaves_carrier_unchanged() -> None:
    """When no agent is selected, the next-period values are ignored entirely."""
    states_per_regime = MappingProxyType(
        {
            "r": MappingProxyType({"wealth": jnp.array([10.0, 20.0])}),
        }
    )
    next_states_per_regime = MappingProxyType(
        {
            "r": MappingProxyType({"wealth": jnp.array([99.0, 99.0])}),
        }
    )
    agent_indices = jnp.array([False, False])

    next_states = _advance_states_for_agents(
        states_per_regime=states_per_regime,
        next_states_per_regime=next_states_per_regime,
        agent_indices=agent_indices,
    )

    assert_array_equal(next_states["r"]["wealth"], jnp.array([10.0, 20.0]))
