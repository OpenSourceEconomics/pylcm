import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.grids.coordinates import (
    get_irreg_coordinate,
    get_linspace_coordinate,
    get_logspace_coordinate,
    linspace,
    logspace,
)
from _lcm.regime_building.ndimage import map_coordinates
from tests.conftest import DECIMAL_PRECISION


def test_linspace():
    calculated = linspace(start=jnp.asarray(1.0), stop=jnp.asarray(2.0), n_points=6)
    expected = np.array([1, 1.2, 1.4, 1.6, 1.8, 2])
    aaae(calculated, expected, decimal=DECIMAL_PRECISION)


def test_linspace_mapped_value():
    """For reference of the grid values, see expected grid in `test_linspace`."""
    start = jnp.asarray(1.0)
    stop = jnp.asarray(2.0)
    # Get position corresponding to a value in the grid
    calculated = get_linspace_coordinate(
        value=jnp.asarray(1.2),
        start=start,
        stop=stop,
        n_points=jnp.int32(6),
    )
    assert np.allclose(calculated, 1.0)

    # Get position corresponding to a value that is between two grid points.
    # Here, the value is 1.3, that is in the middle of 1.2 and 1.4, which have the
    # positions 1 and 2, respectively. Therefore, we want the position to be 1.5.
    calculated = get_linspace_coordinate(
        value=jnp.asarray(1.3),
        start=start,
        stop=stop,
        n_points=jnp.int32(6),
    )
    assert np.allclose(calculated, 1.5)

    # Get position corresponding to a value that is outside the grid
    calculated = get_linspace_coordinate(
        value=jnp.asarray(0.6),
        start=start,
        stop=stop,
        n_points=jnp.int32(6),
    )
    assert np.allclose(calculated, -2.0)


def test_logspace():
    calculated = logspace(start=jnp.asarray(1.0), stop=jnp.asarray(100.0), n_points=7)
    expected = np.array(
        [
            1.0,
            2.154434690031884,
            4.641588833612779,
            10.000000000000002,
            21.54434690031884,
            46.41588833612779,
            100.00000000000001,
        ],
    )
    aaae(calculated, expected, decimal=DECIMAL_PRECISION)


def test_logspace_mapped_value():
    """For reference of the grid values, see expected grid in `test_logspace`."""
    calculated = get_logspace_coordinate(
        value=jnp.asarray((2.15443469 + 4.64158883) / 2),
        start=jnp.asarray(1.0),
        stop=jnp.asarray(100.0),
        n_points=jnp.int32(7),
    )
    assert np.allclose(calculated, 1.5)


@pytest.mark.illustrative
def test_map_coordinates_linear():
    """Illustrative test on how the output of get_linspace_coordinate can be used."""
    grid_info = {
        "start": jnp.asarray(0.0),
        "stop": jnp.asarray(1.0),
        "n_points": jnp.int32(3),
    }

    grid = linspace(
        start=grid_info["start"],
        stop=grid_info["stop"],
        n_points=int(grid_info["n_points"]),
    )  # [0, 0.5, 1]

    values = 2 * grid  # [0, 1.0, 2.0]

    # We choose a coordinate that is exactly in the middle between the first and second
    # entry of the grid.
    coordinate = get_linspace_coordinate(
        value=jnp.asarray(0.25),
        **grid_info,
    )

    # Perform the linear interpolation
    interpolated_value = map_coordinates(input=values, coordinates=[coordinate])
    assert np.allclose(interpolated_value, 0.5)


@pytest.mark.illustrative
def test_map_coordinates_logarithmic():
    """Illustrative test on how the output of get_logspace_coordinate can be used."""
    grid_info = {
        "start": jnp.asarray(1.0),
        "stop": jnp.asarray(2.0),
        "n_points": jnp.int32(3),
    }

    grid = logspace(
        start=grid_info["start"],
        stop=grid_info["stop"],
        n_points=int(grid_info["n_points"]),
    )  # [1.0, 1.414213562373095, 2.0]

    values = 2 * grid  # [2.0, 2.82842712474619, 4.0]

    # We choose a coordinate that is exactly in the middle between the first and second
    # entry of the grid.
    coordinate = get_logspace_coordinate(
        value=jnp.asarray((1.0 + 1.414213562373095) / 2),
        **grid_info,
    )

    # Perform the linear interpolation
    interpolated_value = map_coordinates(input=values, coordinates=[coordinate])
    assert np.allclose(interpolated_value, (2.0 + 2.82842712474619) / 2)


@pytest.mark.illustrative
def test_map_coordinates_linear_outside_grid():
    """Illustrative test on what happens to values outside the grid."""
    grid_info = {
        "start": jnp.asarray(1.0),
        "stop": jnp.asarray(2.0),
        "n_points": jnp.int32(2),
    }

    grid = linspace(
        start=grid_info["start"],
        stop=grid_info["stop"],
        n_points=int(grid_info["n_points"]),
    )  # [1, 2]

    values = 2 * grid  # [2, 4]

    # Get coordinates corresponding to values outside the grid [1, 2]
    coordinates = jnp.array(
        [
            get_linspace_coordinate(value=jnp.asarray(grid_val), **grid_info)
            for grid_val in [-1.0, 0.0, 3.0]
        ]
    )

    interpolated_value = map_coordinates(input=values, coordinates=[coordinates])

    aaae(interpolated_value, [-2, 0, 6], decimal=DECIMAL_PRECISION)


def test_get_linspace_coordinate_with_array():
    values = jnp.array([1.0, 1.2, 1.5])
    coords = get_linspace_coordinate(
        value=values,
        start=jnp.asarray(1.0),
        stop=jnp.asarray(2.0),
        n_points=jnp.int32(6),
    )
    expected = jnp.array([0.0, 1.0, 2.5])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


def test_get_logspace_coordinate_with_array():
    grid = logspace(start=jnp.asarray(1.0), stop=jnp.asarray(100.0), n_points=7)
    mid = (float(grid[1]) + float(grid[2])) / 2
    values = jnp.array([mid])
    coords = get_logspace_coordinate(
        value=values,
        start=jnp.asarray(1.0),
        stop=jnp.asarray(100.0),
        n_points=jnp.int32(7),
    )
    aaae(coords, jnp.array([1.5]), decimal=DECIMAL_PRECISION)


def test_get_irreg_coordinate_with_array():
    points = jnp.array([0.0, 1.0, 3.0, 6.0])
    values = jnp.array([0.5, 3.0, 4.5])
    coords = get_irreg_coordinate(value=values, points=points)
    expected = jnp.array([0.5, 2.0, 2.5])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


def test_concrete_linspace_coordinate_compiles_one_division_by_the_value() -> None:
    """A concrete grid's spacing enters a compiled program as one constant.

    The spacing is the correctly rounded quotient computed before tracing, so the
    only division left in the program is the value's, and no compiler folding of
    `(stop - start) / (n_points - 1)` can make it differ between two programs.
    """
    start, stop, n_points = jnp.asarray(1.0), jnp.asarray(10.0), jnp.int32(8)
    lowered = jax.jit(
        lambda value: get_linspace_coordinate(
            value=value, start=start, stop=stop, n_points=n_points
        )
    ).lower(jnp.asarray(2.0))
    assert lowered.as_text().count("stablehlo.divide") == 1


def test_irreg_coordinate_compiles_without_a_loop() -> None:
    """The interval search of an irregular grid is straight-line code.

    A loop inside the coordinate lookup splits every compiled program that reads
    a value function on such a grid at that loop, once per lookup.
    """
    points = jnp.array([0.0, 1.0, 3.0, 6.0, 10.0])
    lowered = jax.jit(
        jax.vmap(lambda value: get_irreg_coordinate(value=value, points=points))
    ).lower(jnp.linspace(-1.0, 11.0, 7))
    assert lowered.as_text().count("stablehlo.while") == 0


def _irreg_coordinate_by_counting(*, value: float, points: np.ndarray) -> float:
    """Coordinate from the count of points at or below the value."""
    upper = min(max(int(np.sum(points <= value)), 1), len(points) - 1)
    lower_point, upper_point = points[upper - 1], points[upper]
    return upper - 1 + (value - lower_point) / (upper_point - lower_point)


@pytest.mark.parametrize("n_points", range(2, 10))
def test_irreg_coordinate_matches_a_count_of_points_below(n_points: int) -> None:
    """Nodes, their adjacent floats, interior and exterior values all agree."""
    points = np.cumsum(np.random.default_rng(seed=n_points).uniform(0.1, 2.0, n_points))
    values = np.concatenate(
        [
            points,
            np.nextafter(points, -np.inf),
            np.nextafter(points, np.inf),
            (points[:-1] + points[1:]) / 2,
            [points[0] - 1.0, points[-1] + 1.0],
        ]
    )
    expected = [_irreg_coordinate_by_counting(value=v, points=points) for v in values]
    got = jax.jit(get_irreg_coordinate)(
        value=jnp.asarray(values), points=jnp.asarray(points)
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
