import os

import jax
from jax import Array


def generate_simulation_keys(
    *, key: Array, names: list[str], n_initial_states: int
) -> tuple[Array, dict[str, Array]]:
    """Generate pseudo-random number generator keys (PRNG keys) for simulation.

    PRNG keys in JAX are immutable objects used to control random number generation.
    A key can be used to generate a stream of random numbers, e.g., given a key, one can
    call jax.random.normal(key) to generate a stream of normal random numbers. In order
    to ensure that each simulation is based on a different stream of random numbers, we
    split the key into one key per initial state for each stochastic variable,
    and one key that will be passed to the next iteration in order to generate new keys.

    See the JAX documentation for more details:
    https://docs.jax.dev/en/latest/random-numbers.html#random-numbers-in-jax

    Args:
        key: Random key.
        names: List of names for which a key is to be generated.
        n_initial_states: Number of initial states.

    Returns:
        - Updated random key.
        - Mapping with n=n_initial_states new random keys for each name in names.

    """
    simulation_keys = {}
    next_key = key
    for name in names:
        keys = jax.random.split(next_key, num=n_initial_states + 1)
        next_key = keys[0]
        simulation_keys[f"key_{name}"] = keys[1:]
    return next_key, simulation_keys


def draw_random_seed() -> int:
    """Generate a random seed using the operating system's secure entropy pool.

    Returns:
        Random seed.

    """
    return int.from_bytes(os.urandom(4), "little")
