import jax
import jax.numpy as jnp

from lcm.simulation.random import generate_simulation_keys


def test_generate_simulation_keys():
    """Each per-name key stream split from a seed key is mutually distinct."""
    key = jax.random.key(0)
    stochastic_next_functions = ["a", "b"]
    got = generate_simulation_keys(
        key=key, names=stochastic_next_functions, n_initial_states=1
    )
    # Compare the raw key data: distinct keys give a rank-2 matrix of key rows.
    matrix = jnp.array(
        [
            jax.random.key_data(key),
            jax.random.key_data(got[0]),
            jax.random.key_data(got[1]["key_a"][0]),
            jax.random.key_data(got[1]["key_b"][0]),
        ]
    )
    assert jnp.linalg.matrix_rank(matrix) == 2
