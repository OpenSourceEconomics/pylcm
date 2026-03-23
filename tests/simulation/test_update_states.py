from types import MappingProxyType

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from lcm.simulation.utils import _update_states_for_subjects


def test_update_states_strips_next_prefix():
    all_states = MappingProxyType(
        {
            "working__wealth": jnp.array([10.0, 20.0, 30.0]),
        }
    )
    computed_next_states = MappingProxyType(
        {
            "working__next_wealth": jnp.array([15.0, 25.0, 35.0]),
        }
    )
    subject_indices = jnp.array([True, False, True])

    result = _update_states_for_subjects(
        all_states=all_states,
        computed_next_states=computed_next_states,
        subject_indices=subject_indices,
    )

    assert_array_equal(result["working__wealth"], jnp.array([15.0, 20.0, 35.0]))


def test_update_states_multiple_regimes_and_states():
    all_states = MappingProxyType(
        {
            "working__wealth": jnp.array([10.0, 20.0]),
            "working__health": jnp.array([1.0, 2.0]),
            "retired__wealth": jnp.array([100.0, 200.0]),
        }
    )
    computed_next_states = MappingProxyType(
        {
            "working__next_wealth": jnp.array([15.0, 25.0]),
            "working__next_health": jnp.array([1.5, 2.5]),
        }
    )
    subject_indices = jnp.array([True, True])

    result = _update_states_for_subjects(
        all_states=all_states,
        computed_next_states=computed_next_states,
        subject_indices=subject_indices,
    )

    assert_array_equal(result["working__wealth"], jnp.array([15.0, 25.0]))
    assert_array_equal(result["working__health"], jnp.array([1.5, 2.5]))
    # Untouched state remains unchanged
    assert_array_equal(result["retired__wealth"], jnp.array([100.0, 200.0]))


def test_update_states_no_subjects_selected():
    all_states = MappingProxyType(
        {
            "r__wealth": jnp.array([10.0, 20.0]),
        }
    )
    computed_next_states = MappingProxyType(
        {
            "r__next_wealth": jnp.array([99.0, 99.0]),
        }
    )
    subject_indices = jnp.array([False, False])

    result = _update_states_for_subjects(
        all_states=all_states,
        computed_next_states=computed_next_states,
        subject_indices=subject_indices,
    )

    assert_array_equal(result["r__wealth"], jnp.array([10.0, 20.0]))
