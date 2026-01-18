import jax.numpy as jnp

from lcm.random import generate_simulation_keys


def test_generate_simulation_keys():
    key = jnp.arange(2, dtype="uint32")  # PRNG dtype
    stochastic_next_functions = ["a", "b"]
    got = generate_simulation_keys(key, stochastic_next_functions, 1)
    # assert that all generated keys are different from each other
    matrix = jnp.array([key, got[0], got[1]["key_a"][0], got[1]["key_b"][0]])
    assert jnp.linalg.matrix_rank(matrix) == 2
