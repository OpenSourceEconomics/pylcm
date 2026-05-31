import os

import jax

from _lcm.typing import PRNGKeyND


def generate_simulation_keys(
    *,
    key: PRNGKeyND,
    names: list[str],
    n_initial_states: int,
    subject_slice: slice | None = None,
) -> tuple[PRNGKeyND, dict[str, PRNGKeyND]]:
    """Generate pseudo-random number generator keys (PRNG keys) for simulation.

    PRNG keys in JAX are immutable objects used to control random number generation.
    A key can be used to generate a stream of random numbers, e.g., given a key, one can
    call jax.random.normal(key) to generate a stream of normal random numbers. In order
    to ensure that each simulation is based on a different stream of random numbers, we
    split the key into one key per initial state for each stochastic variable,
    and one key that will be passed to the next iteration in order to generate new keys.

    `n_initial_states` is always the **total** subject count, so the split — and
    therefore each subject's key — is independent of how subjects are chunked. When
    simulating a chunk, pass `subject_slice` to return only that chunk's per-subject
    keys (`keys[1:][subject_slice]`). The carry key (`keys[0]`) comes from the full-
    population split and is unaffected by the slice, so the per-period key stream
    reproduces identically across chunks.

    See the JAX documentation for more details:
    https://docs.jax.dev/en/latest/random-numbers.html#random-numbers-in-jax

    Args:
        key: Random key.
        names: List of names for which a key is to be generated.
        n_initial_states: Total number of subjects (the full population).
        subject_slice: When given, the contiguous global-index slice of the subjects
            being simulated in this chunk. `None` returns keys for all subjects.

    Returns:
        - Updated random key.
        - Mapping with one per-subject key array for each name in names (sliced to
          `subject_slice` when given).

    """
    simulation_keys = {}
    next_key = key
    for name in names:
        keys = jax.random.split(next_key, num=n_initial_states + 1)
        next_key = keys[0]
        per_subject_keys = keys[1:]
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
