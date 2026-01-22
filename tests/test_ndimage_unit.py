import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from lcm.ndimage import (
    _compute_indices_and_weights,
    _multiply_all,
    _round_half_away_from_zero,
    _sum_all,
    map_coordinates,
)


def test_map_coordinates_wrong_input_dimensions():
    values = jnp.arange(2)  # ndim = 1
    coordinates = [jnp.array([0]), jnp.array([1])]  # len = 2
    with pytest.raises(ValueError, match="coordinates must be a sequence of length"):
        map_coordinates(values, coordinates)


def test_map_coordinates_extrapolation():
    x = jnp.arange(3.0)
    c = [jnp.array([-2.0, -1.0, 5.0, 10.0])]

    got = map_coordinates(x, c)
    expected = c[0]

    assert_array_equal(got, expected)


def test_nonempty_sum():
    a = jnp.arange(3)

    expected = a + a + a
    got = _sum_all([a, a, a])

    assert_array_equal(got, expected)


def test_nonempty_prod():
    a = jnp.arange(3)

    expected = a * a * a
    got = _multiply_all([a, a, a])

    assert_array_equal(got, expected)


def test_round_half_away_from_zero_integer():
    a = jnp.array([1, 2], dtype=jnp.int32)
    assert_array_equal(_round_half_away_from_zero(a), a)


def test_round_half_away_from_zero_float():
    a = jnp.array([0.5, 1.5], dtype=jnp.float32)

    expected = jnp.array([1, 2], dtype=jnp.int32)
    got = _round_half_away_from_zero(a)

    assert_array_equal(got, expected)


def test_linear_indices_and_weights_inside_domain():
    """Test that the indices and weights are correct for a points inside the domain."""
    coordinates = jnp.array([0, 0.5, 1])

    (idx_low, weight_low), (idx_high, weight_high) = _compute_indices_and_weights(
        coordinates, input_size=2
    )

    assert_array_equal(idx_low, jnp.array([0, 0, 0], dtype=jnp.int32))
    assert_array_equal(weight_low, jnp.array([1, 0.5, 0], dtype=jnp.float32))
    assert_array_equal(idx_high, jnp.array([1, 1, 1], dtype=jnp.int32))
    assert_array_equal(weight_high, jnp.array([0, 0.5, 1], dtype=jnp.float32))


def test_linear_indices_and_weights_outside_domain():
    coordinates = jnp.array([-1, 2])

    (idx_low, weight_low), (idx_high, weight_high) = _compute_indices_and_weights(
        coordinates, input_size=2
    )

    assert_array_equal(idx_low, jnp.array([0, 0], dtype=jnp.int32))
    assert_array_equal(weight_low, jnp.array([2, -1], dtype=jnp.float32))
    assert_array_equal(idx_high, jnp.array([1, 1], dtype=jnp.int32))
    assert_array_equal(weight_high, jnp.array([-1, 2], dtype=jnp.float32))
