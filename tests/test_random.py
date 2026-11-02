import hashlib
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import random as simulation_random
from _lcm.simulation.random import generate_simulation_keys
from _lcm.simulation.random_address import RandomAddress, inverse_cdf_index
from _lcm.simulation.subject_groups import SubjectRows


def test_address_encoding_matches_independent_compact_utf8_json() -> None:
    """Version, stream domain, Unicode name and stable component all enter identity."""
    address = RandomAddress(kind="shock", name="wage_ε", component=3)
    encoded = json.dumps(
        ["pylcm.random.v1", "shock", "wage_ε", 3],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert address.words() == struct.unpack(">8I", hashlib.sha256(encoded).digest())
    assert (
        address.words()
        != RandomAddress(kind="state", name="wage_ε", component=3).words()
    )
    assert (
        address.words()
        != RandomAddress(kind="shock", name="wage_ε", component=4).words()
    )


@pytest.mark.parametrize("uniform", [0.0, 0.25, 0.75])
def test_categorical_cutpoints_are_half_open_and_skip_zero_mass(uniform: float) -> None:
    """A draw on a CDF boundary belongs to the following positive-mass cell."""
    probabilities = jnp.asarray([0.0, 0.25, 0.0, 0.5, 0.25, 0.0])
    expected = {0.0: 1, 0.25: 3, 0.75: 4}[uniform]
    for operation in (inverse_cdf_index, jax.jit(inverse_cdf_index)):
        assert (
            int(operation(probabilities=probabilities, uniform=jnp.asarray(uniform)))
            == expected
        )


def test_domains_and_periods_are_distinct_but_binding_aliases_are_not_identity() -> (
    None
):
    """DAG bindings may change while the model-level scalar innovation stays fixed."""
    root = jax.random.key(41)
    address = RandomAddress(kind="shock", name="wage")
    outputs = []
    for binding, kind, period in (
        ("first__next_wage", "shock", 2),
        ("second__next_wage", "shock", 2),
        ("first__next_wage", "state", 2),
        ("first__next_wage", "shock", 3),
    ):
        _, keys = generate_simulation_keys(
            key=root,
            names=[binding],
            n_initial_states=3,
            period=period,
            addresses={binding: RandomAddress(kind=kind, name=address.name)},
        )
        outputs.append(np.asarray(jax.random.key_data(keys[f"key_{binding}"])))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert np.any(outputs[0] != outputs[2])
    assert np.any(outputs[0] != outputs[3])


def test_grouped_keys_follow_original_rows_on_both_sides_of_uint32() -> None:
    """Grouping preserves original uint64 rows without constructing a full bank."""
    root = jax.random.key(41)
    start = 2**32 - 2
    _, contiguous = generate_simulation_keys(
        key=root,
        names=["wage"],
        n_initial_states=start + 4,
        subject_slice=slice(start, start + 4),
        period=2,
    )
    _, grouped = generate_simulation_keys(
        key=root,
        names=["wage"],
        n_initial_states=start + 4,
        subject_slice=SubjectRows(
            rows=np.asarray([start + 3, start, start + 2], dtype=np.uint64)
        ),
        period=2,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(grouped["key_wage"]),
        np.asarray(jax.random.key_data(contiguous["key_wage"]))[[3, 0, 2]],
    )


def test_trailing_zero_category_never_acquires_rounding_residual_mass() -> None:
    """The final positive category owns the upper tail even after CDF rounding."""
    one = jnp.asarray(1.0)
    high_uniform = jnp.nextafter(one, jnp.asarray(0.0))
    probabilities = jnp.asarray(
        [0.25, jnp.nextafter(jnp.asarray(0.75), jnp.asarray(0.0)), 0.0]
    )
    assert bool(jnp.all(jnp.isfinite(probabilities)))
    assert float(probabilities[-1]) == 0.0
    assert bool(jnp.sum(probabilities) < one)
    assert (
        int(inverse_cdf_index(probabilities=probabilities, uniform=high_uniform)) == 1
    )


def test_named_keys_survive_inserting_and_permuting_other_names() -> None:
    """A scalar innovation keeps its keys when unrelated declarations change."""
    assert (
        Path(simulation_random.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[1] / "src")
    )
    key = jax.random.key(41)
    _, original = generate_simulation_keys(
        key=key, names=["health", "wage"], n_initial_states=3
    )
    _, altered = generate_simulation_keys(
        key=key, names=["new", "wage", "health"], n_initial_states=3
    )
    for name in ("health", "wage"):
        np.testing.assert_array_equal(
            jax.random.key_data(original[f"key_{name}"]),
            jax.random.key_data(altered[f"key_{name}"]),
        )


def test_named_draws_preserve_the_root_and_existing_rows_when_population_grows() -> (
    None
):
    """Appending subjects cannot change another subject's base innovations."""
    key = jax.random.key(41)
    small_root, small = generate_simulation_keys(
        key=key, names=["health"], n_initial_states=3
    )
    large_root, large = generate_simulation_keys(
        key=key, names=["health"], n_initial_states=5
    )
    for returned in (small_root, large_root):
        np.testing.assert_array_equal(
            jax.random.key_data(returned), jax.random.key_data(key)
        )
    np.testing.assert_array_equal(
        jax.random.key_data(small["key_health"]),
        jax.random.key_data(large["key_health"][:3]),
    )


@pytest.mark.parametrize("period", [0, 7])
def test_addressed_chunks_use_uint64_original_rows(period: int) -> None:
    """A chunk can address high original rows without building its population."""
    key = jax.random.key(41)
    start = 2**32 - 1
    _, combined = generate_simulation_keys(
        key=key,
        names=["health"],
        n_initial_states=start + 4,
        subject_slice=slice(start, start + 4),
        period=period,
    )
    _, first = generate_simulation_keys(
        key=key,
        names=["health"],
        n_initial_states=start + 4,
        subject_slice=slice(start, start + 2),
        period=period,
    )
    _, second = generate_simulation_keys(
        key=key,
        names=["health"],
        n_initial_states=start + 4,
        subject_slice=slice(start + 2, start + 4),
        period=period,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(combined["key_health"]),
        np.concatenate(
            (
                jax.random.key_data(first["key_health"]),
                jax.random.key_data(second["key_health"]),
            )
        ),
    )
    assert np.unique(jax.random.key_data(combined["key_health"]), axis=0).shape[0] == 4


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
