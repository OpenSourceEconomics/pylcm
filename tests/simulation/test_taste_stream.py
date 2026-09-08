"""Common taste shocks follow semantic time/domain and global row addresses."""

from fractions import Fraction
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import taste_stream
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.random import generate_simulation_keys
from _lcm.simulation.residency import measure_buffer_footprint
from lcm import AgeGrid, DiscreteGrid, categorical
from lcm.typing import ScalarInt


@categorical(ordered=False)
class _Choice:
    leave: ScalarInt
    stay: ScalarInt


@categorical(ordered=False)
class _ReorderedChoice:
    stay: ScalarInt
    leave: ScalarInt


@categorical(ordered=False)
class _LargerChoice:
    leave: ScalarInt
    stay: ScalarInt
    wait: ScalarInt


def _domain() -> MappingProxyType[str, DiscreteGrid]:
    return MappingProxyType({"enrollment": DiscreteGrid(category_class=_Choice)})


def _words(value: int) -> np.ndarray:
    return np.asarray([value >> 32, value & (2**32 - 1)], dtype=np.uint32)


def _address() -> np.ndarray:
    return np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32)


def _oracle_keys(
    *, key: jax.Array, address: np.ndarray, start: int, count: int, real: int
) -> jax.Array:
    base = key
    for word in address:
        base = jax.random.fold_in(base, word)
    keys = []
    for row in range(start, start + count):
        high, low = _words(min(row, real - 1))
        keys.append(jax.random.fold_in(jax.random.fold_in(base, high), low))
    return jnp.stack(keys)


def test_addresses_share_exact_ages_and_domains_across_regimes_and_horizons() -> None:
    short = taste_stream.build_taste_stream_addresses(
        ages=AgeGrid(exact_values=(1, Fraction(3, 2), 2)),
        discrete_actions_by_regime={"student": _domain()},
    )
    long = taste_stream.build_taste_stream_addresses(
        ages=AgeGrid(exact_values=(0, 1, Fraction(3, 2), 2, 3)),
        discrete_actions_by_regime={"other": _domain(), "counterfactual": _domain()},
    )
    assert short[(1, "student")] == long[(2, "counterfactual")]
    assert long[(2, "counterfactual")] == long[(2, "other")]
    assert short[(0, "student")] != short[(1, "student")]
    assert isinstance(short, MappingProxyType)
    assert len(short[(0, "student")]) == 8


def test_numeric_age_equality_ignores_integer_fraction_representation() -> None:
    integer = AgeGrid(exact_values=(1, Fraction(3, 2)))
    rational = AgeGrid(exact_values=(Fraction(1, 1), Fraction(3, 2)))
    assert type(integer.exact_values[0]) is int
    assert type(rational.exact_values[0]) is Fraction
    first = taste_stream.build_taste_stream_addresses(
        ages=integer, discrete_actions_by_regime={"first": _domain()}
    )
    second = taste_stream.build_taste_stream_addresses(
        ages=rational, discrete_actions_by_regime={"second": _domain()}
    )
    assert first[(0, "first")] == second[(0, "second")]


def test_distinct_exact_ages_do_not_merge_when_float_storage_rounds_them() -> None:
    first_age = Fraction(1, 3)
    second_age = first_age + Fraction(1, 2**80)
    first = AgeGrid(exact_values=(first_age,))
    second = AgeGrid(exact_values=(second_age,))
    assert first_age != second_age
    np.testing.assert_array_equal(first.values, second.values)
    first_addresses = taste_stream.build_taste_stream_addresses(
        ages=first, discrete_actions_by_regime={"same": _domain()}
    )
    second_addresses = taste_stream.build_taste_stream_addresses(
        ages=second, discrete_actions_by_regime={"same": _domain()}
    )
    assert first_addresses[(0, "same")] != second_addresses[(0, "same")]


@pytest.mark.parametrize("change", ["name", "order", "size", "axis_order"])
def test_incompatible_discrete_domains_get_distinct_addresses(change: str) -> None:
    original = {
        "a": DiscreteGrid(category_class=_Choice),
        "b": DiscreteGrid(category_class=_Choice),
    }
    altered = dict(original)
    if change == "name":
        altered = {"renamed": original["a"], "b": original["b"]}
    elif change == "order":
        altered["a"] = DiscreteGrid(category_class=_ReorderedChoice)
    elif change == "size":
        altered["a"] = DiscreteGrid(category_class=_LargerChoice)
    else:
        altered = {"b": original["b"], "a": original["a"]}
    addresses = taste_stream.build_taste_stream_addresses(
        ages=AgeGrid(exact_values=(1,)),
        discrete_actions_by_regime={"first": original, "second": altered},
    )
    assert addresses[(0, "first")] != addresses[(0, "second")]


@pytest.mark.parametrize("start", [0, 5, 2**32 - 2, 2**32 + 7])
def test_keys_use_both_subject_words_and_match_scalar_oracle(start: int) -> None:
    key = jax.random.key(17, impl="threefry2x32")
    real = start + 3
    actual = taste_stream.draw_taste_shock_keys(
        key=key,
        address_words=_address(),
        subject_start=_words(start),
        last_real_subject=_words(real - 1),
        subject_count=5,
    )
    expected = _oracle_keys(
        key=key, address=_address(), start=start, count=5, real=real
    )
    np.testing.assert_array_equal(
        jax.random.key_data(actual), jax.random.key_data(expected)
    )


@pytest.mark.parametrize(
    ("start", "last"), [(2**64 - 2, 2**64 - 1), (2**64 - 1, 2**64 - 3)]
)
def test_uint64_overflow_and_padded_start_repeat_the_last_real_key(
    *, start: int, last: int
) -> None:
    key = jax.random.key(17, impl="threefry2x32")
    actual = taste_stream.draw_taste_shock_keys(
        key=key,
        address_words=_address(),
        subject_start=_words(start),
        last_real_subject=_words(last),
        subject_count=5,
    )
    expected = _oracle_keys(
        key=key, address=_address(), start=start, count=5, real=last + 1
    )
    np.testing.assert_array_equal(
        jax.random.key_data(actual), jax.random.key_data(expected)
    )


def test_empty_chunk_preserves_the_typed_key_shape_without_a_subject() -> None:
    key = jax.random.key(17, impl="threefry2x32")
    actual = taste_stream.draw_taste_shock_keys(
        key=key,
        address_words=_address(),
        subject_start=_words(2**64 - 1),
        last_real_subject=_words(2**64 - 1),
        subject_count=0,
    )
    assert actual.shape == (0,)
    assert actual.dtype == key.dtype
    assert jax.random.key_data(actual).shape == (0, 2)


def test_cached_ordinary_keys_follow_current_threefry_partition_mode() -> None:
    """One retained operation cache cannot reuse a different split implementation."""
    key = jax.random.key(41, impl="threefry2x32")
    original_bits = np.asarray(jax.random.key_data(key)).copy()
    devices = (jax.devices()[0],)
    operations = ProfiledSimulationOperations()
    memory = SimulationMemory(
        budget_bytes=2**24,
        devices=devices,
        subject_devices=devices,
        operations=operations,
        inputs=measure_buffer_footprint(tree=key),
    )
    references = []
    for partitionable in (True, False, True):
        with jax.threefry_partitionable(partitionable):
            expected_carry, expected = generate_simulation_keys(
                key=key,
                names=["taste_shock"],
                n_initial_states=8,
                subject_slice=slice(4, 8),
                original_n_subjects=7,
            )
            actual_carry, actual = generate_simulation_keys(
                key=key,
                names=["taste_shock"],
                n_initial_states=8,
                subject_slice=slice(4, 8),
                original_n_subjects=7,
                memory=memory,
            )
            references.append(np.asarray(jax.random.key_data(expected_carry)).copy())
            np.testing.assert_array_equal(
                jax.random.key_data(actual_carry),
                jax.random.key_data(expected_carry),
                err_msg=f"Cached ordinary carry ignored partitionable={partitionable}",
            )
            np.testing.assert_array_equal(
                jax.random.key_data(actual["key_taste_shock"]),
                jax.random.key_data(expected["key_taste_shock"]),
            )
        memory.close_unit()
    assert np.any(references[0] != references[1])
    np.testing.assert_array_equal(references[0], references[2])
    np.testing.assert_array_equal(jax.random.key_data(key), original_bits)


def test_chunk_keys_equal_their_full_population_window() -> None:
    key = jax.random.key(29, impl="threefry2x32")
    full = taste_stream.draw_taste_shock_keys(
        key=key,
        address_words=_address(),
        subject_start=_words(0),
        last_real_subject=_words(6),
        subject_count=10,
    )
    chunk = taste_stream.draw_taste_shock_keys(
        key=key,
        address_words=_address(),
        subject_start=_words(4),
        last_real_subject=_words(6),
        subject_count=6,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(chunk), jax.random.key_data(full[4:])
    )


def test_one_compilation_accepts_dynamic_age_chunk_and_population_addresses() -> None:
    function = taste_stream.draw_taste_shock_keys
    key = jax.random.key(31, impl="threefry2x32")
    compiled = (
        jax.jit(function, static_argnames=("subject_count",))
        .lower(
            key=key,
            address_words=_address(),
            subject_start=_words(0),
            last_real_subject=_words(6),
            subject_count=4,
        )
        .compile()
    )
    for address, start, real in (
        (_address(), 3, 7),
        (_address() + 100, 2**32 - 1, 2**32 + 2),
    ):
        actual = compiled(
            key=key,
            address_words=address,
            subject_start=_words(start),
            last_real_subject=_words(real - 1),
        )
        expected = _oracle_keys(
            key=key, address=address, start=start, count=4, real=real
        )
        np.testing.assert_array_equal(
            jax.random.key_data(actual), jax.random.key_data(expected)
        )
