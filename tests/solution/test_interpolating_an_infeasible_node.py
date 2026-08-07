"""A state at which no action is feasible does not destroy the states beside it.

Backward induction writes `-inf` for a state where every action is infeasible —
`max_Q_over_a` reduces with `initial=-jnp.inf` — and the next step reads that
array back by interpolation. Multiplying an interpolation weight of exactly zero
by that `-inf` gives NaN, and the NaN then travels through the sum into every
neighbouring read, so one infeasible state takes out the states around it.

Neutralizing the corner on the *value*, an operand of the multiplication, is what
the rest of the engine already does with a zero-weight node. The alternative
spelling — testing the weight for positivity after the multiply — is wrong here
for a reason specific to this routine: `map_coordinates` extrapolates rather than
clamping, so outside the grid its corner weights are legitimately negative, and
discarding them discards real signal rather than a null event.
"""

import functools
import itertools
import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.ndimage import _compute_indices_and_weights, map_coordinates
from lcm.typing import FloatND


def _infeasible() -> float:
    """The value of a state at which no action is feasible."""
    return -np.inf


def _read_every_node(grid: FloatND) -> FloatND:
    """Interpolate `grid` at each of its own nodes."""
    axes = [range(size) for size in grid.shape]
    reads = [
        map_coordinates(grid, [jnp.asarray(float(i)) for i in index])
        for index in itertools.product(*axes)
    ]
    return jnp.asarray(reads).reshape(grid.shape)


def _bare_multiply_reference(grid: FloatND, coordinates: list[FloatND]) -> FloatND:
    """Interpolate with an unguarded product, correct wherever the grid is finite."""
    data = [
        _compute_indices_and_weights(coordinate, size)
        for coordinate, size in zip(coordinates, grid.shape, strict=True)
    ]
    terms = []
    for indices_and_weights in itertools.product(*data):
        indices, weights = zip(*indices_and_weights, strict=True)
        weight = functools.reduce(operator.mul, weights)
        terms.append(weight * grid[indices])
    return functools.reduce(operator.add, terms)


def test_an_infeasible_node_does_not_poison_its_neighbours() -> None:
    """Reading a one-dimensional grid at its own nodes returns that grid."""
    grid = jnp.asarray([0.0, 1.0, 2.0, _infeasible(), 4.0])

    np.testing.assert_array_equal(np.asarray(_read_every_node(grid)), np.asarray(grid))


def test_an_infeasible_node_does_not_poison_its_neighbourhood() -> None:
    """A single infeasible state does not take out the block of states around it."""
    grid = jnp.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, _infeasible(), 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    np.testing.assert_array_equal(np.asarray(_read_every_node(grid)), np.asarray(grid))


def test_a_read_between_a_feasible_and_an_infeasible_node_is_infeasible() -> None:
    """A genuinely positive weight on `-inf` yields `-inf`, not a finite number."""
    grid = jnp.asarray([1.0, _infeasible()])

    read = map_coordinates(grid, [jnp.asarray(0.5)])

    assert bool(jnp.isneginf(read))


@pytest.mark.parametrize(
    "coordinate",
    [-1.5, -0.5, 3.5, 4.5],
    ids=["far-below", "below", "above", "far-above"],
)
def test_extrapolation_outside_the_grid_is_unchanged(coordinate: float) -> None:
    """Outside the grid, corner weights are negative and must keep contributing.

    Discarding non-positive weights would silently replace an extrapolated read
    with a truncated one; the guard tests for a represented zero instead.
    """
    grid = jnp.asarray([0.0, 1.0, 4.0, 9.0, 16.0])
    coord = [jnp.asarray(coordinate)]

    np.testing.assert_allclose(
        float(map_coordinates(grid, coord)),
        float(_bare_multiply_reference(grid, coord)),
        rtol=1e-14,
    )


def test_the_read_stays_differentiable_at_a_node() -> None:
    """The derivative at a node keeps the corner whose weight vanishes there.

    Exactly at a node one corner weight is zero, and the value beside it is what
    the derivative is made of: `d(w * v)/dw = v`. Neutralizing that corner by
    replacing its value would flatten the slope to zero at every node.
    """
    grid = jnp.asarray([0.0, 2.0, 4.0, 6.0])

    def read(coordinate: FloatND) -> FloatND:
        return map_coordinates(grid, [coordinate])

    np.testing.assert_allclose(float(jax.grad(read)(jnp.asarray(1.0))), 2.0)


def test_finite_reads_never_reverse_a_discrete_choice() -> None:
    """On finite grids the guard never changes which alternative is best.

    The guarded read sits in the value function that feeds every subsequent
    comparison, so agreeing to a few units in the last place is not enough on its
    own: what matters is that no `argmax` moves.
    """
    rng = np.random.default_rng(seed=20260807)
    grid = jnp.asarray(rng.normal(size=(9, 9)) * 10.0)

    reversals = 0
    for _ in range(200):
        coordinates = rng.uniform(-1.0, 9.0, size=(5, 2))
        guarded = [
            float(map_coordinates(grid, [jnp.asarray(c[0]), jnp.asarray(c[1])]))
            for c in coordinates
        ]
        bare = [
            float(
                _bare_multiply_reference(grid, [jnp.asarray(c[0]), jnp.asarray(c[1])])
            )
            for c in coordinates
        ]
        reversals += int(np.argmax(guarded) != np.argmax(bare))

    assert reversals == 0
