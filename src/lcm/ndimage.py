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
# Modifications made by Tim Mensinger, 2024
from __future__ import annotations

import functools
import itertools
import operator
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, jit, lax

if TYPE_CHECKING:
    from collections.abc import Sequence


@jit
def map_coordinates(
    input: Array,
    coordinates: Sequence[Array],
) -> Array:
    """Map the input array to new coordinates using linear interpolation.

    Modified from JAX implementation of :func:`scipy.ndimage.map_coordinates`.

    Given an input array and a set of coordinates, this function returns the
    interpolated values of the input array at those coordinates. For coordinates outside
    the input array, linear extrapolation is used.

    Args:
      input: N-dimensional input array from which values are interpolated.
      coordinates: length-N sequence of arrays specifying the coordinates
        at which to evaluate the interpolated values

    Returns:
      The interpolated (extrapolated) values at the specified coordinates.

    """
    if len(coordinates) != input.ndim:
        raise ValueError(
            "coordinates must be a sequence of length input.ndim, but "
            f"{len(coordinates)} != {input.ndim}"
        )

    interpolation_data = [
        _compute_indices_and_weights(coordinate, size)
        for coordinate, size in zip(coordinates, input.shape, strict=True)
    ]

    interpolation_values = []
    for indices_and_weights in itertools.product(*interpolation_data):
        indices, weights = zip(*indices_and_weights, strict=True)
        contribution = input[indices]
        weighted_value = _multiply_all(weights) * contribution
        interpolation_values.append(weighted_value)

    result = _sum_all(interpolation_values)

    if jnp.issubdtype(input.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)

    return result.astype(input.dtype)


def _compute_indices_and_weights(
    coordinate: Array, input_size: int
) -> list[tuple[Array, Array]]:
    """Compute indices and weights for linear interpolation."""
    lower_index = jnp.clip(jnp.floor(coordinate), 0, input_size - 2).astype(jnp.int32)
    upper_weight = coordinate - lower_index
    lower_weight = 1 - upper_weight
    return [(lower_index, lower_weight), (lower_index + 1, upper_weight)]


def _multiply_all(arrs: Sequence[Array]) -> Array:
    """Multiply all arrays in the sequence."""
    return functools.reduce(operator.mul, arrs)


def _sum_all(arrs: Sequence[Array]) -> Array:
    """Sum all arrays in the sequence."""
    return functools.reduce(operator.add, arrs)


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)
