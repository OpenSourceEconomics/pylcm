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
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.ndimage import _compute_indices_and_weights, map_coordinates
from lcm.typing import FloatND, IntND


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


def _rational_stencil_reference(
    *, grid: np.ndarray, coordinates: tuple[float, ...], pinned_axes: tuple[int, ...]
) -> float:
    """Evaluate one linear stencil exactly, with infeasibility decided per corner.

    Independent of the production kernel: weights and finite values are exact
    rationals, so the reference applies only to small grids whose coordinates
    are integers or dyadic fractions and whose finite values are small integers.
    A corner with an exactly zero weight contributes nothing; any other corner
    holding `-inf` makes the read `-inf`; otherwise any other NaN corner makes
    it NaN. A pinned coordinate names an existing node exactly.
    """
    axis_terms = []
    for axis, coordinate in enumerate(coordinates):
        query = Fraction(coordinate)
        if axis in pinned_axes:
            assert query.denominator == 1
            assert 0 <= query < grid.shape[axis]
            axis_terms.append([(int(query), Fraction(1))])
        else:
            lower = min(
                max(query.numerator // query.denominator, 0), grid.shape[axis] - 2
            )
            upper_weight = query - lower
            axis_terms.append(
                [(lower, Fraction(1) - upper_weight), (lower + 1, upper_weight)]
            )
    active = []
    for corner in itertools.product(*axis_terms):
        index = tuple(node for node, _ in corner)
        weight = functools.reduce(operator.mul, (factor for _, factor in corner))
        if weight:
            active.append((weight, float(grid[index])))
    if any(value == -np.inf for _, value in active):
        return -np.inf
    if any(np.isnan(value) for _, value in active):
        return np.nan
    return float(sum(weight * Fraction(value) for weight, value in active))


def _read_at_integer_coordinate(case: tuple[FloatND, IntND]) -> FloatND:
    """Read one floating grid at its own int32 coordinate."""
    grid, coordinate = case
    return map_coordinates(input=grid, coordinates=[coordinate])


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
@pytest.mark.parametrize(
    ("coordinate", "expected"),
    [(2, -np.inf), (1, 4.0)],
    ids=["negative-weight-on-infeasible", "zero-weight-on-infeasible"],
)
def test_an_integer_coordinate_on_a_floating_grid_obeys_the_feasibility_rule(
    *, dtype: type, coordinate: int, expected: float
) -> None:
    """An int32 read returns `-inf` under a live infeasible corner, else its node.

    At coordinate 2 the corner holding `-inf` carries weight -1, so the read is
    infeasible rather than `+inf`; at coordinate 1 that corner carries weight 0,
    so the read is the finite neighbour 4 rather than NaN.
    """
    grid = jnp.asarray([-jnp.inf, 4.0], dtype=dtype)

    read = map_coordinates(
        input=grid, coordinates=[jnp.asarray(coordinate, dtype=jnp.int32)]
    )

    assert read.dtype == dtype
    assert float(read) == expected


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager-vmap", "jit-vmap"])
def test_every_two_node_stencil_at_integer_coordinates_matches_the_exact_reference(
    *, dtype: type, compiled: bool
) -> None:
    """Each two-node stencil read at an int32 coordinate equals the exact reference.

    Node values range over every pair from {-inf, NaN, -2, 0, 4} and coordinates
    over -2..3, so weights are zero, positive and negative.
    """
    values = [-np.inf, np.nan, -2.0, 0.0, 4.0]
    cases = list(itertools.product(itertools.product(values, repeat=2), range(-2, 4)))
    grids = jnp.asarray([grid for grid, _ in cases], dtype=dtype)
    coordinates = jnp.asarray([coordinate for _, coordinate in cases], dtype=jnp.int32)
    expected = np.asarray(
        [
            _rational_stencil_reference(
                grid=np.asarray(grid), coordinates=(float(coordinate),), pinned_axes=()
            )
            for grid, coordinate in cases
        ],
        dtype=grids.dtype,
    )
    read = jax.vmap(_read_at_integer_coordinate)
    if compiled:
        actual = jax.jit(read)((grids, coordinates))
    else:
        with jax.disable_jit():
            actual = read((grids, coordinates))

    assert actual.dtype == dtype
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
@pytest.mark.parametrize("pinned_axes", [(), (0,), (1,), (0, 1)])
@pytest.mark.parametrize("mixed_coordinates", [False, True], ids=["integer", "mixed"])
def test_batched_two_dimensional_reads_keep_each_reads_own_feasibility(
    *, dtype: type, pinned_axes: tuple[int, ...], mixed_coordinates: bool
) -> None:
    """Each read of a batch of 2x2 grids equals the exact reference.

    Every grid holds `-inf`, NaN or both at some corner; reads use int32 or mixed
    int32 and floating coordinates, with any subset of axes pinned.
    """
    variants = []
    for corner in itertools.product(range(2), repeat=2):
        for value in (-np.inf, np.nan):
            grid = np.asarray([[1.0, 2.0], [3.0, 4.0]])
            grid[corner] = value
            variants.append(grid)
        grid = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        grid[corner] = -np.inf
        grid[tuple(1 - index for index in corner)] = np.nan
        variants.append(grid)
    axis_points = [
        (0.0, 1.0)
        if axis in pinned_axes
        else (-0.5, 0.0, 0.5, 1.0, 1.5)
        if mixed_coordinates and axis == 1
        else (-1.0, 0.0, 1.0, 2.0)
        for axis in range(2)
    ]
    points = list(itertools.product(*axis_points))
    coordinates = [
        jnp.asarray([point[0] for point in points], dtype=jnp.int32),
        jnp.asarray(
            [point[1] for point in points],
            dtype=dtype if mixed_coordinates else jnp.int32,
        ),
    ]
    grids = jnp.asarray(np.stack(variants), dtype=dtype)
    expected = np.asarray(
        [
            [
                _rational_stencil_reference(
                    grid=grid, coordinates=point, pinned_axes=pinned_axes
                )
                for point in points
            ]
            for grid in variants
        ],
        dtype=grids.dtype,
    )

    def read_grid(grid: FloatND) -> FloatND:
        return map_coordinates(
            input=grid, coordinates=coordinates, pinned_axes=pinned_axes
        )

    actual = jax.jit(jax.vmap(read_grid))(grids)

    assert actual.dtype == dtype
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
def test_a_finite_integer_coordinate_read_is_bitwise_the_plain_weighted_sum(
    dtype: type,
) -> None:
    """On finite grids an int32 read has the bits of the bare corner sum.

    Signed zeros are included among the node values.
    """
    values = [-0.0, 0.0, -2.0, 4.0]
    cases = list(itertools.product(itertools.product(values, repeat=2), range(-2, 4)))
    grids = jnp.asarray([grid for grid, _ in cases], dtype=dtype)
    coordinates = jnp.asarray([coordinate for _, coordinate in cases], dtype=jnp.int32)

    def bare_read(case: tuple[FloatND, IntND]) -> FloatND:
        grid, coordinate = case
        return (1 - coordinate) * grid[0] + coordinate * grid[1]

    actual = jax.jit(jax.vmap(_read_at_integer_coordinate))((grids, coordinates))
    expected = jax.jit(jax.vmap(bare_read))((grids, coordinates))
    bits_dtype = np.uint32 if np.dtype(dtype).itemsize == 4 else np.uint64

    assert actual.dtype == expected.dtype == dtype
    np.testing.assert_array_equal(
        np.asarray(actual).view(bits_dtype), np.asarray(expected).view(bits_dtype)
    )


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp64"])
def test_the_compiled_read_stays_differentiable_at_a_node(dtype: type) -> None:
    """Under `jit`, the slope at a node of a linear grid is its analytical value 2."""
    grid = jnp.asarray([0.0, 2.0, 4.0, 6.0], dtype=dtype)

    def read(coordinate: FloatND) -> FloatND:
        return map_coordinates(input=grid, coordinates=[coordinate])

    derivative = jax.jit(jax.grad(read))(jnp.asarray(1.0, dtype=dtype))

    assert float(derivative) == 2.0


def test_an_integer_grid_keeps_integer_interpolation_and_extrapolation() -> None:
    """An int32 plane read at int32 coordinates in and around it is the plane."""
    grid = jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32)
    coordinates = jnp.asarray(
        list(itertools.product(range(-1, 3), repeat=2)), dtype=jnp.int32
    )

    read = map_coordinates(
        input=grid, coordinates=[coordinates[:, 0], coordinates[:, 1]]
    )

    assert read.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.asarray(read), np.asarray(1 + 2 * coordinates[:, 0] + coordinates[:, 1])
    )
