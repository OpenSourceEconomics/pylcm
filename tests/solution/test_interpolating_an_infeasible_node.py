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
        map_coordinates(input=grid, coordinates=[jnp.asarray(float(i)) for i in index])
        for index in itertools.product(*axes)
    ]
    return jnp.asarray(reads).reshape(grid.shape)


def _bare_multiply_reference(*, grid: FloatND, coordinates: list[FloatND]) -> FloatND:
    """Interpolate with an unguarded product, correct wherever the grid is finite."""
    data = [
        _compute_indices_and_weights(coordinate=coordinate, input_size=size)
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

    read = map_coordinates(input=grid, coordinates=[jnp.asarray(0.5)])

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
        float(map_coordinates(input=grid, coordinates=coord)),
        float(_bare_multiply_reference(grid=grid, coordinates=coord)),
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
        return map_coordinates(input=grid, coordinates=[coordinate])

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
            float(
                map_coordinates(
                    input=grid, coordinates=[jnp.asarray(c[0]), jnp.asarray(c[1])]
                )
            )
            for c in coordinates
        ]
        bare = [
            float(
                _bare_multiply_reference(
                    grid=grid, coordinates=[jnp.asarray(c[0]), jnp.asarray(c[1])]
                )
            )
            for c in coordinates
        ]
        reversals += int(np.argmax(guarded) != np.argmax(bare))

    assert reversals == 0


_DTYPES = [jnp.float32, jnp.float64]


def _grid_with_infeasible_node(*, infeasible_index: int, dtype: type) -> FloatND:
    """Twelve finite nodes, the one at `infeasible_index` replaced by `-inf`.

    The last node holds the value of the feasible neighbour from a production
    solve, so the top-edge reads below reproduce it.
    """
    values = np.linspace(-1.0, -0.1, 12)
    values[11] = -0.07254466811182257
    values[infeasible_index] = -np.inf
    grid = jnp.asarray(values, dtype=dtype)
    assert grid.dtype == dtype
    return grid


def _read(*, grid: FloatND, coordinate: float) -> FloatND:
    coord = jnp.asarray(coordinate, dtype=grid.dtype)
    assert coord.dtype == grid.dtype
    return map_coordinates(input=grid, coordinates=[coord])


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
def test_extrapolation_above_the_grid_onto_an_infeasible_node_is_infeasible(
    dtype: type,
) -> None:
    """A read beyond the top node whose stencil holds `-inf` is `-inf`.

    The corner below the top node carries a negative weight out there, and a
    negative weight on `-inf` must not turn the read into `+inf`.
    """
    grid = _grid_with_infeasible_node(infeasible_index=10, dtype=dtype)

    read = _read(grid=grid, coordinate=11.62304782083577)

    assert float(read) == -np.inf


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
def test_extrapolation_below_the_grid_onto_an_infeasible_node_is_infeasible(
    dtype: type,
) -> None:
    """A read below the bottom node whose stencil holds `-inf` is `-inf`."""
    grid = _grid_with_infeasible_node(infeasible_index=1, dtype=dtype)

    read = _read(grid=grid, coordinate=-0.62304782083577)

    assert float(read) == -np.inf


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
def test_interpolation_inside_the_grid_onto_an_infeasible_node_is_infeasible(
    dtype: type,
) -> None:
    """A read strictly between a feasible and an infeasible node is `-inf`."""
    grid = _grid_with_infeasible_node(infeasible_index=10, dtype=dtype)

    read = _read(grid=grid, coordinate=10.25)

    assert float(read) == -np.inf


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
@pytest.mark.parametrize("coordinate", [11.0, 0.0], ids=["top-node", "bottom-node"])
def test_a_zero_weight_infeasible_corner_leaves_the_read_feasible(
    *, dtype: type, coordinate: float
) -> None:
    """A read exactly at a feasible node returns that node beside an `-inf` one."""
    infeasible_index = 10 if coordinate > 0 else 1
    grid = _grid_with_infeasible_node(infeasible_index=infeasible_index, dtype=dtype)

    read = _read(grid=grid, coordinate=coordinate)

    assert float(read) == float(grid[int(coordinate)])


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
@pytest.mark.parametrize(
    "coordinate",
    [-1.62304782083577, -0.5, 3.25, 11.62304782083577, 13.5],
    ids=["far-below", "below", "inside", "above", "far-above"],
)
def test_a_finite_read_is_bitwise_the_plain_weighted_sum(
    *, dtype: type, coordinate: float
) -> None:
    """On a finite grid every read, extrapolated or not, is the bare corner sum."""
    grid = jnp.asarray(np.linspace(-1.0, 3.0, 12) ** 3, dtype=dtype)
    coord = [jnp.asarray(coordinate, dtype=dtype)]
    assert grid.dtype == dtype

    np.testing.assert_array_equal(
        np.asarray(map_coordinates(input=grid, coordinates=coord)),
        np.asarray(_bare_multiply_reference(grid=grid, coordinates=coord)),
    )
