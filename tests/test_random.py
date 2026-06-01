import jax
import jax.numpy as jnp

from _lcm.simulation.random import generate_simulation_keys


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


def test_generate_simulation_keys_pads_with_last_real_key():
    """Padding the dispatch size leaves real keys unchanged; pad slots reuse the last.

    When `original_n_subjects < n_initial_states`, the per-subject split is sized to
    the real subject count, so the leading real keys (and the carry key) are
    identical to an unpadded call; the trailing pad slots duplicate the last real
    subject's key.
    """
    key = jax.random.key(0)
    real = generate_simulation_keys(key=key, names=["a"], n_initial_states=3)
    padded = generate_simulation_keys(
        key=key, names=["a"], n_initial_states=5, original_n_subjects=3
    )

    real_data = jax.random.key_data(real[1]["key_a"])
    padded_data = jax.random.key_data(padded[1]["key_a"])

    assert jnp.array_equal(padded_data[:3], real_data)
    assert jnp.array_equal(padded_data[3], real_data[2])
    assert jnp.array_equal(padded_data[4], real_data[2])
    assert jnp.array_equal(jax.random.key_data(padded[0]), jax.random.key_data(real[0]))


def test_generate_simulation_keys_slice_selects_chunk_from_padded_stream():
    """`subject_slice` extracts a chunk's keys from the padded full population."""
    key = jax.random.key(0)
    full = generate_simulation_keys(
        key=key, names=["a"], n_initial_states=5, original_n_subjects=3
    )
    chunk = generate_simulation_keys(
        key=key,
        names=["a"],
        n_initial_states=5,
        original_n_subjects=3,
        subject_slice=slice(2, 4),
    )

    assert jnp.array_equal(
        jax.random.key_data(chunk[1]["key_a"]),
        jax.random.key_data(full[1]["key_a"][2:4]),
    )
