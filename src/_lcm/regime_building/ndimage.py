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
"""Multilinear interpolation at arbitrary coordinates, adapted from JAX.

`map_coordinates` reads an array at fractional coordinates by weighting its
neighbouring cells; the continuation's value interpolator is its only caller.

- Outside the array's extent the read extrapolates linearly from the two
  outermost nodes of each axis, so a corner weight there is negative.
- `-inf` marks a state at which no action is feasible. A read whose stencil
  gives such a corner a nonzero weight, of either sign, is itself `-inf`; a
  corner of exactly zero weight contributes nothing, whatever stands at it.
"""

import functools
import itertools
import operator
from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit, lax

from _lcm.probability import is_represented_zero
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import FloatND, IntND


@functools.partial(jit, static_argnames=("pinned_axes",))
def map_coordinates(
    *,
    input: FloatND | IntND,  # noqa: A002
    coordinates: Sequence[FloatND | IntND] | FloatND | IntND,
    pinned_axes: tuple[int, ...] = (),
) -> FloatND | IntND:
    """Map the input array to new coordinates using linear interpolation.

    Modified from JAX implementation of `scipy.ndimage.map_coordinates`.

    Given an input array and a set of coordinates, this function returns the
    interpolated values of the input array at those coordinates. For coordinates outside
    the input array, linear extrapolation is used.

    An axis named in `pinned_axes` is read at one node instead of two. Nothing
    lies between two nodes of such an axis — its coordinate already names one
    exactly — so the second corner would carry weight zero and contribute
    nothing. Dropping it halves the corners of the interpolation box per pinned
    axis, and the box is a product over ALL axes, so pinning three of them
    leaves an eighth of the reads.

    A corner whose weight is exactly zero contributes nothing, whatever value
    stands at it. That matters because backward induction writes `-inf` for a
    state at which no action is feasible, and an unguarded `0 * -inf` is NaN that
    then travels through the sum into every neighbouring read — one infeasible
    state would take out the states around it.

    The neutralization tests the weight for a *represented zero* rather than for
    positivity. Extrapolation is what makes the distinction load-bearing: outside
    the grid the corner weights are legitimately negative, so discarding
    non-positive weights would truncate an extrapolated read rather than drop a
    null event.

    The same negative weight is why feasibility is decided per corner rather
    than left to the arithmetic: a negative weight on `-inf` gives `+inf`, which
    would report an infeasible read as the best one available. Any corner
    holding `-inf` under a nonzero weight therefore makes the read `-inf`.

    Args:
      input: N-dimensional input array from which values are interpolated.
      coordinates: length-N sequence of arrays specifying the coordinates
        at which to evaluate the interpolated values
      pinned_axes: Tuple of axis positions whose coordinate names a node
        exactly, so the axis is read at that one node rather than
        interpolated between two.

    Returns:
      The interpolated (extrapolated) values at the specified coordinates.

    """
    if len(coordinates) != input.ndim:
        raise ValueError(
            "coordinates must be a sequence of length input.ndim, but "
            f"{len(coordinates)} != {input.ndim}"
        )

    interpolation_data = [
        _compute_pinned_index_and_weight(coordinate=coordinate, input_size=size)
        if axis in pinned_axes
        else _compute_indices_and_weights(coordinate=coordinate, input_size=size)
        for axis, (coordinate, size) in enumerate(
            zip(coordinates, input.shape, strict=True)
        )
    ]

    interpolation_values = []
    touches_infeasible = jnp.zeros((), dtype=bool)
    for indices_and_weights in itertools.product(*interpolation_data):
        indices, weights = zip(*indices_and_weights, strict=True)
        contribution = input[indices]
        corner_weight = _multiply_all(weights)
        # Only a floating grid can hold the `+-inf` that makes a zero-weight
        # corner undefined or marks an infeasible state, and only a floating
        # weight has a sign bit and an exponent field to read. An integer read
        # has neither hazard, so it multiplies as it always did.
        if jnp.issubdtype(corner_weight.dtype, jnp.floating) and jnp.issubdtype(
            contribution.dtype, jnp.floating
        ):
            weighted_value = zero_safe_weighted_term(
                weight=corner_weight,
                value=contribution,
                subnormal_is_accounted_for=True,
            )
            touches_infeasible = touches_infeasible | (
                jnp.isneginf(contribution) & ~is_represented_zero(corner_weight)
            )
        else:
            weighted_value = corner_weight * contribution
        interpolation_values.append(weighted_value)

    result = _sum_all(interpolation_values)
    if jnp.issubdtype(result.dtype, jnp.floating):
        result = jnp.where(
            touches_infeasible, jnp.asarray(-jnp.inf, dtype=result.dtype), result
        )

    if jnp.issubdtype(input.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)

    return result.astype(input.dtype)


def _compute_indices_and_weights(
    *, coordinate: FloatND | IntND, input_size: int
) -> list[tuple[IntND, FloatND | IntND]]:
    """Compute indices and weights for linear interpolation.

    The index and the weights derive from one materialized copy of the
    coordinate. Left free, the compiler may recompute the coordinate separately
    for the gather index and for the weight, and the copies can round
    differently; a coordinate within an ulp of a node then pairs the index of
    one cell with the weight of its neighbour.
    """
    coordinate = lax.optimization_barrier(coordinate)
    lower_index = jnp.clip(jnp.floor(coordinate), 0, input_size - 2).astype(jnp.int32)
    upper_weight = coordinate - lower_index
    lower_weight = 1 - upper_weight
    return [(lower_index, lower_weight), (lower_index + 1, upper_weight)]


def _compute_pinned_index_and_weight(
    *, coordinate: FloatND | IntND, input_size: int
) -> list[tuple[IntND, FloatND]]:
    """Return the single full-weight corner of an axis read at one node.

    A coordinate that names no node of such an axis arrives as NaN, which no
    index can represent, so it is replaced by a node the read can land on. The
    caller is the one that knows the axis, and is where such a read is
    discarded.
    """
    representable = jnp.where(jnp.isfinite(coordinate), coordinate, 0)
    index = jnp.clip(jnp.round(representable), 0, input_size - 1).astype(jnp.int32)
    # Shape and dtype as the interpolated weights of any other axis, so a
    # corner's weight product is formed in the dtype it would be either way.
    return [(index, jnp.ones_like(coordinate))]


def _multiply_all(arrs: Sequence[FloatND | IntND]) -> FloatND | IntND:
    """Multiply all arrays in the sequence."""
    return functools.reduce(operator.mul, arrs)


def _sum_all(arrs: Sequence[FloatND | IntND]) -> FloatND | IntND:
    """Sum all arrays in the sequence."""
    return functools.reduce(operator.add, arrs)


def _round_half_away_from_zero(a: FloatND | IntND) -> FloatND | IntND:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)
