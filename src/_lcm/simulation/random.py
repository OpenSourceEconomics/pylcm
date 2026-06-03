import os

import jax
import jax.numpy as jnp

from _lcm.typing import PRNGKeyND


def generate_simulation_keys(
    *,
    key: PRNGKeyND,
    names: list[str],
    n_initial_states: int,
    subject_slice: slice | None = None,
    original_n_subjects: int | None = None,
) -> tuple[PRNGKeyND, dict[str, PRNGKeyND]]:
    """Generate pseudo-random number generator keys (PRNG keys) for simulation.

    PRNG keys in JAX are immutable objects used to control random number generation.
    A key can be used to generate a stream of random numbers, e.g., given a key, one can
    call jax.random.normal(key) to generate a stream of normal random numbers. In order
    to ensure that each simulation is based on a different stream of random numbers, we
    split the key into one key per initial state for each stochastic variable,
    and one key that will be passed to the next iteration in order to generate new keys.

    The per-subject split is sized to `original_n_subjects` (the user's real
    population, defaulting to `n_initial_states`), so each real subject's key — and
    the carry key (`keys[0]`) — is independent of both how subjects are chunked and
    of any per-device padding. The stream is then assembled to length
    `n_initial_states`:

    - **Device padding** (`original_n_subjects < n_initial_states`): the trailing
      `n_initial_states - original_n_subjects` slots duplicate the last real
      subject's key, so pad rows reproduce a duplicate-last-subject simulation.
    - **Chunking** (`subject_slice` given): the assembled full-population stream is
      sliced to the chunk's global-index window, so a chunk reproduces exactly the
      keys it would get in a single pass.

    See the JAX documentation for more details:
    https://docs.jax.dev/en/latest/random-numbers.html#random-numbers-in-jax

    Args:
        key: Random key.
        names: List of names for which a key is to be generated.
        n_initial_states: Number of initial states the simulate dispatch sees
            (the full population, possibly padded for sharding).
        subject_slice: When given, the contiguous global-index slice of the subjects
            being simulated in this chunk. `None` returns keys for all subjects.
        original_n_subjects: Number of subjects before per-device padding. Defaults
            to `n_initial_states` (no padding). When smaller, the per-subject split
            is sized to it so real subjects' draws are device-count-invariant.

    Returns:
        - Updated random key.
        - Mapping with one per-subject key array for each name in names (padded to
          `n_initial_states`, then sliced to `subject_slice` when given).

    """
    if original_n_subjects is None:
        original_n_subjects = n_initial_states
    pad = n_initial_states - original_n_subjects
    simulation_keys = {}
    next_key = key
    for name in names:
        keys = jax.random.split(next_key, num=original_n_subjects + 1)
        next_key = keys[0]
        per_subject_keys = keys[1:]
        if pad > 0:
            per_subject_keys = jnp.concatenate(
                [per_subject_keys, jnp.repeat(per_subject_keys[-1:], pad, axis=0)]
            )
        if subject_slice is not None:
            per_subject_keys = per_subject_keys[subject_slice]
        simulation_keys[f"key_{name}"] = per_subject_keys
    return next_key, simulation_keys


def draw_random_seed() -> int:
    """Generate a random seed using the operating system's secure entropy pool.

    Returns:
        Random seed.

    """
    return int.from_bytes(os.urandom(4), "little")
