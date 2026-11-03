from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from _lcm.regime_building.ndimage import (
    _compute_indices_and_weights,
    _multiply_all,
    _round_half_away_from_zero,
    _sum_all,
    map_coordinates,
)
from lcm.typing import Float1D, FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


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


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("pinned", [False, True])
def test_extrapolation_preserves_a_constant_at_small_and_large_scales(
    *, pinned: bool
) -> None:
    """Equal finite stencil values survive large positive and negative extensions."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    limits = jnp.finfo(dtype)
    coordinates = jnp.asarray([-36349786.25959628, -2.0, 2.0, 36349786.25959628])
    for constant in (
        0.48520302639196167,
        -0.48520302639196167,
        float(limits.tiny) * 4,
        float(limits.max) / 16,
    ):
        grid = jnp.full((2, 2), constant, dtype=dtype)
        assert np.isfinite(np.asarray(grid)).all()
        assert np.isfinite(np.asarray(coordinates)).all()
        got = map_coordinates(
            input=grid,
            coordinates=[coordinates, jnp.asarray(0.0)],
            pinned_axes=(1,) if pinned else (),
        )
        assert_array_equal(got, np.full(coordinates.shape, constant, dtype=dtype))


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("far_axis", [0, 1])
def test_extrapolation_does_not_amplify_an_irrelevant_axis(*, far_axis: int) -> None:
    """A field constant along an axis has no sensitivity to extending that axis."""
    row = jnp.asarray([0.48520302639196167, 0.6931471805599453])
    grid = jnp.stack((row, row))
    grid = grid if far_axis == 0 else grid.T
    far = jnp.asarray([-36349786.25959628, 36349786.25959628])
    within = jnp.asarray(0.125)
    assert_array_equal(grid[0] if far_axis == 0 else grid[:, 0], row)
    exact = Fraction(7, 8) * Fraction(float(row[0])) + Fraction(1, 8) * Fraction(
        float(row[1])
    )
    coordinates = [far, within] if far_axis == 0 else [within, far]
    got = map_coordinates(input=grid, coordinates=coordinates)
    np.testing.assert_allclose(got, float(exact), rtol=0, atol=10**-DECIMAL_PRECISION)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_extrapolation_keeps_coordinate_and_value_sensitivities() -> None:
    """Extending a constant axis keeps zero slope and unit level sensitivity."""
    row = jnp.asarray([0.48520302639196167, 0.6931471805599453])
    grid = jnp.stack((row, row))
    far = jnp.asarray(36349786.25959628)
    within = jnp.asarray(0.125)

    def read(*, values: FloatND, far: ScalarFloat, within: ScalarFloat) -> ScalarFloat:
        return map_coordinates(input=values, coordinates=[far, within])

    far_gradient = jax.grad(lambda point: read(values=grid, far=point, within=within))(
        far
    )
    within_gradient = jax.grad(lambda point: read(values=grid, far=far, within=point))(
        within
    )
    _, constant_direction = jax.jvp(
        lambda values: read(values=values, far=far, within=within),
        (grid,),
        (jnp.ones_like(grid),),
    )
    assert_array_equal(far_gradient, 0)
    np.testing.assert_allclose(
        within_gradient, float(row[1] - row[0]), rtol=0, atol=10**-DECIMAL_PRECISION
    )
    assert_array_equal(constant_direction, 1)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_extrapolation_preserves_bilinear_derivatives_and_linear_pullbacks() -> None:
    """A bilinear extension keeps its analytical gradient and mixed derivative."""
    grid = jnp.asarray([[1.0, 4.0], [3.0, 11.0]])
    point = jnp.asarray([2.25, -1.5])

    def read(*, values: FloatND, point: Float1D) -> ScalarFloat:
        return map_coordinates(input=values, coordinates=[point[0], point[1]])

    def at(point: Float1D) -> ScalarFloat:
        return read(values=grid, point=point)

    # The represented stencil is exactly 1 + 2*x + 3*y + 5*x*y.
    assert_array_equal(jax.jit(at)(point), -15.875)
    gradient = jnp.asarray([-5.5, 14.25])
    assert_array_equal(jax.jit(jax.grad(at))(point), gradient)
    assert_array_equal(jax.jit(jax.hessian(at))(point), [[0.0, 5.0], [5.0, 0.0]])
    _, pullback = jax.vjp(at, point)
    assert_array_equal(pullback(jnp.asarray(0.0))[0], jnp.zeros_like(point))
    assert_array_equal(pullback(jnp.asarray(0.75))[0], 0.75 * gradient)
    assert_array_equal(
        pullback(jnp.asarray(0.75))[0],
        pullback(jnp.asarray(0.25))[0] + pullback(jnp.asarray(0.5))[0],
    )
    assert_array_equal(
        jax.grad(lambda values: read(values=values, point=point))(grid),
        [[-3.125, 1.875], [5.625, -3.375]],
    )
    _, zero_tangent = jax.jvp(at, (point,), (jnp.zeros_like(point),))
    assert_array_equal(zero_tangent, 0.0)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("direction", [-1, 1], ids=["left", "right"])
def test_extrapolation_keeps_a_tiny_endpoint_difference(*, direction: int) -> None:
    """A subnormal endpoint difference must survive a normal-sized extension."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    limits = np.finfo(dtype)
    unsigned_dtype = np.uint32 if dtype.itemsize == 4 else np.uint64
    smallest_normal_bits = 1 << limits.nmant
    coordinate = jnp.asarray(direction * 2 ** (limits.nmant + 8), dtype=dtype)
    tiny = Fraction(float(limits.tiny))
    for steps in (1, 2, 17):
        for sign in (-1, 1):
            endpoints = np.asarray(
                [smallest_normal_bits, smallest_normal_bits + steps],
                dtype=unsigned_dtype,
            ).view(dtype)
            endpoints = sign * endpoints
            grid = jnp.asarray(endpoints)
            assert_array_equal(
                np.asarray(grid).view(unsigned_dtype), endpoints.view(unsigned_dtype)
            )
            left, right = (Fraction(float(value)) for value in endpoints)
            assert 0 < abs(right - left) < tiny
            assert np.isfinite(endpoints).all()
            assert (np.abs(endpoints) >= limits.tiny).all()
            point = Fraction(float(coordinate))
            exact = left + point * (right - left)
            got = map_coordinates(input=grid, coordinates=[coordinate])
            error = abs(Fraction(float(got)) - exact)
            # The guarded corner sum can lose the unit in 1-coordinate at
            # this scale. Its resulting error is one smallest normal value.
            assert error <= tiny, (direction, steps, sign, error, tiny)


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


def test_linear_indices_and_weights_read_the_coordinate_once():
    """Index and weights derive from one materialized copy of the coordinate.

    Left free, the compiler may recompute the coordinate separately for the
    gather index and for the weight. When the two copies round differently, a
    coordinate within an ulp of a node pairs the index of one cell with the
    weight of the neighbouring cell, so the read jumps by the neighbour's value.
    """
    closed = jax.make_jaxpr(
        lambda c: _compute_indices_and_weights(coordinate=c, input_size=5)
    )(1.0)
    coordinate = closed.jaxpr.invars[0]

    readers = [
        eqn.primitive.name for eqn in closed.jaxpr.eqns if coordinate in eqn.invars
    ]

    assert readers == ["optimization_barrier"]
