import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from _lcm.regime_building.ndimage import (
    _compute_indices_and_weights,
    _multiply_all,
    _round_half_away_from_zero,
    _sum_all,
    map_coordinates,
)


def test_map_coordinates_wrong_input_dimensions():
    values = jnp.arange(2, dtype=jnp.int32)  # ndim = 1
    coordinates = [
        jnp.array([0], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    ]  # len = 2
    with pytest.raises(ValueError, match="coordinates must be a sequence of length"):
        map_coordinates(input=values, coordinates=coordinates)


def test_map_coordinates_extrapolation():
    x = jnp.arange(3.0)
    c = [jnp.array([-2.0, -1.0, 5.0, 10.0])]

    got = map_coordinates(input=x, coordinates=c)
    expected = c[0]

    assert_array_equal(got, expected)


def test_nonempty_sum():
    a = jnp.arange(3, dtype=jnp.int32)

    expected = a + a + a
    got = _sum_all([a, a, a])

    assert_array_equal(got, expected)


def test_nonempty_prod():
    a = jnp.arange(3, dtype=jnp.int32)

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
        coordinate=coordinates, input_size=2
    )

    assert_array_equal(idx_low, jnp.array([0, 0, 0], dtype=jnp.int32))
    assert_array_equal(weight_low, jnp.array([1, 0.5, 0], dtype=jnp.float32))
    assert_array_equal(idx_high, jnp.array([1, 1, 1], dtype=jnp.int32))
    assert_array_equal(weight_high, jnp.array([0, 0.5, 1], dtype=jnp.float32))


def test_linear_indices_and_weights_outside_domain():
    coordinates = jnp.array([-1.0, 2.0])

    (idx_low, weight_low), (idx_high, weight_high) = _compute_indices_and_weights(
        coordinate=coordinates, input_size=2
    )

    assert_array_equal(idx_low, jnp.array([0, 0], dtype=jnp.int32))
    assert_array_equal(weight_low, jnp.array([2, -1], dtype=jnp.float32))
    assert_array_equal(idx_high, jnp.array([1, 1], dtype=jnp.int32))
    assert_array_equal(weight_high, jnp.array([-1, 2], dtype=jnp.float32))


@pytest.mark.parametrize(
    ("index", "expected"), [(-1, 201.0), (-4, 1.0), (3, 201.0), (10, 201.0)]
)
def test_indexed_axis_preserves_native_negative_and_out_of_bounds_reads(
    *,
    index: int,
    expected: float,
) -> None:
    """Fixed discrete indices retain JAX wrapping and clipping while interpolating."""
    values = 100 * jnp.arange(3.0)[:, None] + 2 * jnp.arange(4.0)[None, :]
    actual = map_coordinates(
        input=values,
        coordinates=(jnp.asarray(0.5),),
        indexed_axes=(0,),
        indices=(index,),
    )
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("indices", "expected"), [((1, 2), 6.0), ((-1, -1), 11.0), ((9, 9), 11.0)]
)
def test_all_indexed_axes_need_no_interpolation_coordinates(
    *,
    indices: tuple[int, int],
    expected: float,
) -> None:
    """An entirely indexed array returns its selected cell without interpolation."""
    actual = map_coordinates(
        input=jnp.arange(12.0).reshape(3, 4),
        coordinates=(),
        indexed_axes=(0, 1),
        indices=indices,
    )
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("axes", "indices", "pinned"),
    [
        ((0, 0), (0, 0), ()),
        ((-1,), (0,), ()),
        ((2,), (0,), ()),
        ((0,), (), ()),
        ((0,), (0,), (0,)),
    ],
)
def test_indexed_axes_reject_inconsistent_metadata(
    *,
    axes: tuple[int, ...],
    indices: tuple[int, ...],
    pinned: tuple[int, ...],
) -> None:
    """Indexed axes are distinct valid dimensions with exactly one integer index."""
    with pytest.raises(ValueError, match="indexed axes must be unique"):
        map_coordinates(
            input=jnp.arange(12.0).reshape(3, 4),
            coordinates=(jnp.asarray(0.5),),
            indexed_axes=axes,
            indices=indices,
            pinned_axes=pinned,
        )


def test_pinned_axis_positions_refer_to_original_array() -> None:
    """Removing an indexed dimension does not renumber the remaining pinned axes."""
    values = (
        100 * jnp.arange(3.0)[:, None, None]
        + 10 * jnp.arange(2.0)[None, :, None]
        + jnp.arange(4.0)[None, None, :]
    )
    actual = map_coordinates(
        input=values,
        coordinates=(jnp.asarray(1.0), jnp.asarray(1.5)),
        indexed_axes=(1,),
        indices=(1,),
        pinned_axes=(0,),
    )
    assert_array_equal(actual, 111.5)
