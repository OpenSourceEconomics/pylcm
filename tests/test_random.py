from __future__ import annotations

import jax
import jax.numpy as jnp

from lcm.random import generate_simulation_keys, random_choice


def test_random_choice():
    key = jax.random.key(0)
    probs = jnp.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    labels = jnp.array([1, 2, 3])
    got = random_choice(labels=labels, probs=probs, key=key)
    assert jnp.array_equal(got, jnp.array([3, 1, 2]))


def test_generate_simulation_keys():
    key = jnp.arange(2, dtype="uint32")  # PRNG dtype
    stochastic_next_functions = ["a", "b"]
    got = generate_simulation_keys(key, stochastic_next_functions)
    # assert that all generated keys are different from each other
    matrix = jnp.array([key, got[0], got[1]["a"], got[1]["b"]])
    assert jnp.linalg.matrix_rank(matrix) == 2
