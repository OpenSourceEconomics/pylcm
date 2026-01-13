# Copyright 2019 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made by Tim Mensinger, 2024.
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import jax.scipy.ndimage
import numpy as np
import pytest
import scipy.ndimage
from numpy.testing import assert_allclose, assert_array_almost_equal

import lcm.ndimage
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED

# Use 64-bit dtypes when x64 is enabled, 32-bit otherwise
TEST_DTYPES = [np.int64, np.float64] if X64_ENABLED else [np.int32, np.float32]

jax_map_coordinates = partial(jax.scipy.ndimage.map_coordinates, order=1, cval=0)
scipy_map_coordinates = partial(scipy.ndimage.map_coordinates, order=1, cval=0)
lcm_map_coordinates = lcm.ndimage.map_coordinates

JAX_BASED_IMPLEMENTATIONS = [jax_map_coordinates, lcm_map_coordinates]


TEST_SHAPES = [
    (5,),
    (3, 4),
    (3, 4, 5),
]

TEST_COORDINATES_SHAPES = [
    (7,),
    (2, 3, 4),
]


def _make_test_data(shape, coordinates_shape, dtype):
    rng = np.random.default_rng()
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    c = [(size - 1) * rng.random(coordinates_shape).astype(dtype) for size in shape]
    return x, c


@pytest.mark.parametrize("map_coordinates", JAX_BASED_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("coordinates_shape", TEST_COORDINATES_SHAPES)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_map_coordinates_against_scipy(
    map_coordinates, shape, coordinates_shape, dtype
):
    """Test that JAX and LCM implementations behave as scipy."""
    x, c = _make_test_data(shape, coordinates_shape, dtype=dtype)

    x_jax = jnp.asarray(x)
    c_jax = [jnp.asarray(c_i) for c_i in c]

    expected = scipy_map_coordinates(x, c)
    got = map_coordinates(x_jax, c_jax)

    assert_array_almost_equal(got, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("map_coordinates", JAX_BASED_IMPLEMENTATIONS)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_map_coordinates_round_half_against_scipy(map_coordinates, dtype):
    """Test that JAX and LCM implementations round as scipy."""
    x = np.arange(-5, 5, dtype=dtype)
    c = np.array([[0.5, 1.5, 2.5, 6.5, 8.5]])

    x_jax = jnp.asarray(x)
    c_jax = [jnp.asarray(c_i) for c_i in c]

    expected = scipy_map_coordinates(x, c)
    got = map_coordinates(x_jax, c_jax)

    assert_array_almost_equal(got, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("map_coordinates", JAX_BASED_IMPLEMENTATIONS)
def test_gradients(map_coordinates):
    """Test that JAX and LCM implementations exhibit same gradient behavior."""
    x = jnp.arange(9.0)
    border = 3  # square root of 9, as we are considering a parabola on x.

    def f(step):
        coordinates = x + step
        shifted = map_coordinates(x, [coordinates])
        return ((x - shifted) ** 2)[border:-border].mean()

    # Gradient of f(step) is 2 * step
    assert_allclose(jax.grad(f)(0.5), 1.0)
    assert_allclose(jax.grad(f)(1.0), 2.0)
