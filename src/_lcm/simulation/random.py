import os

import jax
import jax.numpy as jnp

from _lcm.typing import PRNGKeyND


def generate_simulation_keys(
    *,
    key: PRNGKeyND,
    names: list[str],
    n_initial_states: int,
    original_n_subjects: int | None = None,
) -> tuple[PRNGKeyND, dict[str, PRNGKeyND]]:
    """Generate pseudo-random number generator keys (PRNG keys) for simulation.

    PRNG keys in JAX are immutable objects used to control random number generation.
    A key can be used to generate a stream of random numbers, e.g., given a key, one can
    call jax.random.normal(key) to generate a stream of normal random numbers. In order
    to ensure that each simulation is based on a different stream of random numbers, we
    split the key into one key per initial state for each stochastic variable,
    and one key that will be passed to the next iteration in order to generate new keys.

    When `original_n_subjects` is smaller than `n_initial_states`, the leading
    `original_n_subjects` per-subject keys come from a split sized for the
    user's real subject count — so RNG draws for the real subjects are
    independent of the per-device padding applied by
    `pad_initial_conditions_for_devices`. The remaining (pad) slots replicate
    the last real subject's key, so pad rows produce deterministic outputs
    that match a duplicate-last-subject simulation; pad rows are trimmed
    away in `Model.simulate`.

    See the JAX documentation for more details:
    https://docs.jax.dev/en/latest/random-numbers.html#random-numbers-in-jax

    Args:
        key: Random key.
        names: List of names for which a key is to be generated.
        n_initial_states: Number of initial states the simulate dispatch sees
            (possibly padded for sharding).
        original_n_subjects: Number of subjects before per-device padding.
            Defaults to `n_initial_states` (no padding). When smaller, the
            per-subject split is sized to `original_n_subjects` so real
            subjects' RNG draws are device-count-invariant.

    Returns:
        - Updated random key (deterministic given the same `original_n_subjects`).
        - Mapping with `n_initial_states` keys per name (the last
          `n_initial_states - original_n_subjects` are duplicates of the
          last real-subject key).

    """
    if original_n_subjects is None:
        original_n_subjects = n_initial_states
    pad = n_initial_states - original_n_subjects
    simulation_keys: dict[str, PRNGKeyND] = {}
    next_key = key
    for name in names:
        keys = jax.random.split(next_key, num=original_n_subjects + 1)
        next_key = keys[0]
        per_subject = keys[1:]
        if pad > 0:
            per_subject = jnp.concatenate(
                [per_subject, jnp.repeat(per_subject[-1:], pad, axis=0)]
            )
        simulation_keys[f"key_{name}"] = per_subject
    return next_key, simulation_keys


def draw_random_seed() -> int:
    """Generate a random seed using the operating system's secure entropy pool.

    Returns:
        Random seed.

    """
    return int.from_bytes(os.urandom(4), "little")
