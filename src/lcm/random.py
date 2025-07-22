from __future__ import annotations

import os
from functools import partial

import jax
from jax import Array


def random_choice(
    labels: Array,
    probs: Array,
    key: Array,
) -> Array:
    """Draw multiple random choices.

    Args:
        labels: 1d array of labels.
        probs: 2d array of probabilities. Second dimension must be the same length as
            the dimension of labels.
        key: Random key.

    Returns:
        Selected labels. 1d array of length len(probs).

    """
    keys = jax.random.split(key, probs.shape[0])
    return _vmapped_choice(keys, probs, labels)


@partial(jax.vmap, in_axes=(0, 0, None))
def _vmapped_choice(key: Array, probs: Array, labels: Array) -> Array:
    return jax.random.choice(key, a=labels, p=probs)


def generate_simulation_keys(
    key: Array, ids: list[str]
) -> tuple[Array, dict[str, Array]]:
    """Generate pseudo-random number generator keys (PRNG keys) for simulation.

    PRNG keys in JAX are immutable objects used to control random number generation.
    A key can be used to generate a stream of random numbers, e.g., given a key, one can
    call jax.random.normal(key) to generate a stream of normal random numbers. In order
    to ensure that each simulation is based on a different stream of random numbers, we
    split the key into one key per stochastic variable, and one key that will be passed
    to the next iteration in order to generate new keys.

    See the JAX documentation for more details:
    https://docs.jax.dev/en/latest/random-numbers.html#random-numbers-in-jax

    Args:
        key: Random key.
        ids: List of names for which a key is to be generated.

    Returns:
        - Updated random key.
        - Dict with random keys for each id in ids.

    """
    keys = jax.random.split(key, num=len(ids) + 1)

    next_key = keys[0]
    simulation_keys = dict(zip(ids, keys[1:], strict=True))

    return next_key, simulation_keys


def draw_random_seed() -> int:
    """Generate a random seed using the operating system's secure entropy pool.

    Returns:
        Random seed.

    """
    return int.from_bytes(os.urandom(4), "little")
