from __future__ import annotations

from dataclasses import make_dataclass

import jax.numpy as jnp
import numpy as np
import portion
import pytest

from lcm.exceptions import GridInitializationError
from lcm.grids import (
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Piece,
    PiecewiseLinSpacedGrid,
    _get_field_names_and_values,
    _validate_continuous_grid,
    _validate_discrete_grid,
    validate_category_class,
)


def test_validate_discrete_grid_empty():
    category_class = make_dataclass("Category", [])
    error_msg = "category_class must have at least one field"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_scalar_input():
    category_class = make_dataclass("Category", [("a", int, 1), ("b", str, "s")])
    error_msg = (
        "Field values of the category_class can only be int "
        r"values. The values to the following fields are not: \['b'\]"
    )
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_none_input():
    category_class = make_dataclass("Category", [("a", int), ("b", int, 1)])
    error_msg = (
        "Field values of the category_class can only be int "
        r"values. The values to the following fields are not: \['a'\]"
    )
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_unique():
    category_class = make_dataclass("Category", [("a", int, 1), ("b", int, 1)])
    error_msg = "Field values of the category_class must be unique."
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_consecutive_unordered():
    category_class = make_dataclass("Category", [("a", int, 1), ("b", int, 0)])
    error_msg = "Field values of the category_class must be consecutive integers"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_consecutive_jumps():
    category_class = make_dataclass("Category", [("a", int, 0), ("b", int, 2)])
    error_msg = "Field values of the category_class must be consecutive integers"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_get_fields_with_defaults():
    category_class = make_dataclass("Category", [("a", int, 1), ("b", int, 2)])
    assert _get_field_names_and_values(category_class) == {"a": 1, "b": 2}


def test_get_fields_no_defaults():
    category_class = make_dataclass("Category", [("a", int), ("b", int)])
    assert _get_field_names_and_values(category_class) == {"a": None, "b": None}


def test_get_fields_instance():
    category_class = make_dataclass("Category", [("a", int), ("b", int)])
    assert _get_field_names_and_values(category_class(a=1, b=2)) == {"a": 1, "b": 2}


def test_get_fields_empty():
    category_class = make_dataclass("Category", [])
    assert _get_field_names_and_values(category_class) == {}


def test_validate_continuous_grid_invalid_start():
    error_msg = "start must be a scalar int or float value"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_continuous_grid("a", 1, 10)  # ty: ignore[invalid-argument-type]


def test_validate_continuous_grid_invalid_stop():
    error_msg = "stop must be a scalar int or float value"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_continuous_grid(1, "a", 10)  # ty: ignore[invalid-argument-type]


def test_validate_continuous_grid_invalid_n_points():
    error_msg = "n_points must be an int greater than 0 but is a"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_continuous_grid(1, 2, "a")  # ty: ignore[invalid-argument-type]


def test_validate_continuous_grid_negative_n_points():
    error_msg = "n_points must be an int greater than 0 but is -1"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_continuous_grid(1, 2, -1)


def test_validate_continuous_grid_start_greater_than_stop():
    error_msg = "start must be less than stop"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_continuous_grid(2, 1, 10)


def test_linspace_grid_creation():
    grid = LinSpacedGrid(start=1, stop=5, n_points=5)
    assert np.allclose(grid.to_jax(), np.linspace(1, 5, 5))


def test_logspace_grid_creation():
    grid = LogSpacedGrid(start=1, stop=10, n_points=3)
    assert np.allclose(grid.to_jax(), np.logspace(np.log10(1), np.log10(10), 3))


def test_discrete_grid_creation():
    category_class = make_dataclass(
        "Category", [("a", int, 0), ("b", int, 1), ("c", int, 2)]
    )
    grid = DiscreteGrid(category_class)
    assert np.allclose(grid.to_jax(), np.arange(3))


def test_linspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LinSpacedGrid(start=1, stop=0, n_points=10)


def test_logspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LogSpacedGrid(start=1, stop=0, n_points=10)


def test_discrete_grid_invalid_category_class():
    category_class = make_dataclass(
        "Category", [("a", int, 0), ("b", str, "wrong_type")]
    )
    with pytest.raises(
        GridInitializationError,
        match="Field values of the category_class can only be int",
    ):
        DiscreteGrid(category_class)


def test_replace_mixin():
    grid = LinSpacedGrid(start=1, stop=5, n_points=5)
    new_grid = grid.replace(start=0)
    assert new_grid.start == 0
    assert new_grid.stop == 5
    assert new_grid.n_points == 5


# ======================================================================================
# Tests for validate_category_class (reusable validation)
# ======================================================================================


def test_validate_category_class_valid():
    """Valid category class should return empty error list."""
    category_class = make_dataclass("Category", [("a", int, 0), ("b", int, 1)])
    errors = validate_category_class(category_class)
    assert errors == []


def test_validate_category_class_not_dataclass():
    """Non-dataclass should return error."""

    class NotDataclass:
        a = 0
        b = 1

    errors = validate_category_class(NotDataclass)
    assert len(errors) == 1
    assert "must be a dataclass" in errors[0]


def test_validate_category_class_non_consecutive():
    """Non-consecutive values should return error."""
    category_class = make_dataclass("Category", [("a", int, 0), ("b", int, 2)])
    errors = validate_category_class(category_class)
    assert len(errors) == 1
    assert "consecutive integers" in errors[0]


def test_validate_category_class_not_starting_at_zero():
    """Values not starting at 0 should return error."""
    category_class = make_dataclass("Category", [("a", int, 1), ("b", int, 2)])
    errors = validate_category_class(category_class)
    assert len(errors) == 1
    assert "consecutive integers starting from 0" in errors[0]


# ======================================================================================
# Tests for IrregSpacedGrid
# ======================================================================================


def test_irreg_spaced_grid_creation():
    grid = IrregSpacedGrid(points=(1.0, 2.0, 5.0, 10.0))
    assert np.allclose(grid.to_jax(), np.array([1.0, 2.0, 5.0, 10.0]))
    assert grid.n_points == 4


def test_irreg_spaced_grid_invalid_too_few_points():
    with pytest.raises(GridInitializationError, match="at least 2 elements"):
        IrregSpacedGrid(points=(1.0,))


def test_irreg_spaced_grid_invalid_non_numeric():
    with pytest.raises(GridInitializationError, match="must be int or float"):
        IrregSpacedGrid(points=(1.0, "a", 3.0))  # type: ignore[arg-type]


def test_irreg_spaced_grid_invalid_not_ascending():
    with pytest.raises(GridInitializationError, match="strictly ascending order"):
        IrregSpacedGrid(points=(1.0, 3.0, 2.0))


# ======================================================================================
# Tests for coordinate equivalence between LinSpacedGrid and IrregSpacedGrid
# ======================================================================================


@pytest.fixture
def equivalent_grids():
    """Create a LinSpacedGrid and IrregSpacedGrid with identical points."""
    lin_grid = LinSpacedGrid(start=0.0, stop=10.0, n_points=11)
    irreg_grid = IrregSpacedGrid(points=tuple(float(x) for x in lin_grid.to_jax()))
    return lin_grid, irreg_grid


@pytest.mark.parametrize(
    "value", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
)
def test_linspaced_and_irregspaced_exact_grid_points(equivalent_grids, value: float):
    """Coordinates match at exact grid points."""
    lin_grid, irreg_grid = equivalent_grids
    assert np.isclose(lin_grid.get_coordinate(value), irreg_grid.get_coordinate(value))


@pytest.mark.parametrize("value", [0.5, 1.25, 3.7, 7.9, 9.99])
def test_linspaced_and_irregspaced_interpolation(equivalent_grids, value: float):
    """Coordinates match for interpolation (values between grid points)."""
    lin_grid, irreg_grid = equivalent_grids
    assert np.isclose(lin_grid.get_coordinate(value), irreg_grid.get_coordinate(value))


@pytest.mark.parametrize("value", [-1.0, -0.5, 10.5, 12.0])
def test_linspaced_and_irregspaced_extrapolation(equivalent_grids, value: float):
    """Coordinates match for extrapolation (values outside grid range)."""
    lin_grid, irreg_grid = equivalent_grids
    assert np.isclose(lin_grid.get_coordinate(value), irreg_grid.get_coordinate(value))


def _create_equivalent_grid(
    grid_type: str, lin_grid: LinSpacedGrid
) -> IrregSpacedGrid | PiecewiseLinSpacedGrid:
    """Create a grid equivalent to the given LinSpacedGrid."""
    if grid_type == "IrregSpacedGrid":
        return IrregSpacedGrid(points=tuple(float(x) for x in lin_grid.to_jax()))
    if grid_type == "PiecewiseLinSpacedGrid":
        return PiecewiseLinSpacedGrid(
            pieces=(
                Piece(
                    interval=f"[{lin_grid.start}, {lin_grid.stop}]",
                    n_points=lin_grid.n_points,
                ),
            )
        )
    msg = f"Unknown grid type: {grid_type}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    ("start", "stop", "n_points"),
    [
        (0.0, 1.0, 5),
        (1.0, 100.0, 10),
        (-10.0, 10.0, 21),
        (0.5, 2.5, 3),
    ],
)
@pytest.mark.parametrize("grid_type", ["IrregSpacedGrid", "PiecewiseLinSpacedGrid"])
def test_linspaced_coordinates_match_other_grid_types(
    start: float, stop: float, n_points: int, grid_type: str, x64_enabled: bool
):
    """LinSpacedGrid coordinates should match equivalent grids of other types."""
    lin_grid = LinSpacedGrid(start=start, stop=stop, n_points=n_points)
    other_grid = _create_equivalent_grid(grid_type, lin_grid)

    # Generate test values: grid points, interpolation, and extrapolation
    grid_points = [float(x) for x in lin_grid.to_jax()]
    step = (stop - start) / (n_points - 1)

    # Interpolation: midpoints between consecutive grid points
    interpolation_values = [
        (grid_points[i] + grid_points[i + 1]) / 2 for i in range(n_points - 1)
    ]

    # Extrapolation: outside the grid range
    extrapolation_values = [start - step, start - 0.1, stop + 0.1, stop + step]

    all_test_values = grid_points + interpolation_values + extrapolation_values

    # Tolerance depends on precision and grid value magnitude
    base_rtol = 1e-6 if x64_enabled else 1e-4
    max_magnitude = max(abs(start), abs(stop), 1.0)
    rtol = base_rtol * max_magnitude

    for value in all_test_values:
        lin_coord = float(lin_grid.get_coordinate(value))
        other_coord = float(other_grid.get_coordinate(value))
        assert np.isclose(lin_coord, other_coord, rtol=rtol), (
            f"Mismatch at value {value} for {grid_type} vs LinSpacedGrid "
            f"({start}, {stop}, {n_points}): "
            f"LinSpacedGrid={lin_coord}, {grid_type}={other_coord}"
        )


# ======================================================================================
# Tests for PiecewiseLinSpacedGrid
# ======================================================================================


def test_piecewise_lin_spaced_grid_creation_with_strings():
    """PiecewiseLinSpacedGrid can be created with string intervals."""
    grid = PiecewiseLinSpacedGrid(
        pieces=(
            Piece(interval="[1, 4)", n_points=3),
            Piece(interval="[4, 10]", n_points=7),
        )
    )
    assert grid.n_points == 10


def test_piecewise_lin_spaced_grid_creation_with_portion_objects():
    """PiecewiseLinSpacedGrid can be created with portion.Interval objects."""
    grid = PiecewiseLinSpacedGrid(
        pieces=(
            Piece(interval=portion.closedopen(0, 5), n_points=5),
            Piece(interval=portion.closed(5, 10), n_points=6),
        )
    )
    assert grid.n_points == 11


def test_piecewise_lin_spaced_grid_closed_boundary_is_exact():
    """Closed boundaries should produce exact endpoint values."""
    grid = PiecewiseLinSpacedGrid(pieces=(Piece(interval="[0, 5]", n_points=6),))
    points = grid.to_jax()
    assert float(points[0]) == 0.0
    assert float(points[-1]) == 5.0


def test_piecewise_lin_spaced_grid_open_boundary_excludes_endpoint():
    """Open boundaries should not include the exact endpoint value."""
    grid = PiecewiseLinSpacedGrid(
        pieces=(
            Piece(interval="[0, 5)", n_points=5),
            Piece(interval="[5, 10]", n_points=6),
        )
    )
    points = grid.to_jax()
    # First piece should end just before 5
    assert float(points[4]) < 5.0
    # Second piece should start exactly at 5
    assert float(points[5]) == 5.0


def test_piecewise_lin_spaced_grid_no_representable_value_between_pieces():
    """No representable float between adjacent open/closed boundaries."""
    grid = PiecewiseLinSpacedGrid(
        pieces=(
            Piece(interval="[0, 5)", n_points=5),
            Piece(interval="[5, 10]", n_points=6),
        )
    )
    points = grid.to_jax()

    # The last point of the first piece (open boundary)
    last_of_first = points[4]
    # The first point of the second piece (closed boundary)
    first_of_second = points[5]

    # nextafter should give us exactly the first point of the second piece
    assert jnp.nextafter(last_of_first, jnp.inf) == first_of_second


def test_piecewise_lin_spaced_grid_adjacent_closedopen_closedclosed():
    """[a, x) followed by [x, b] should be valid (adjacent)."""
    grid = PiecewiseLinSpacedGrid(
        pieces=(
            Piece(interval="[1, 4)", n_points=3),
            Piece(interval="[4, 7]", n_points=4),
        )
    )
    assert grid.n_points == 7


def test_piecewise_lin_spaced_grid_adjacent_closed_openclosed():
    """[a, x] followed by (x, b] should be valid (adjacent)."""
    grid = PiecewiseLinSpacedGrid(
        pieces=(
            Piece(interval="[1, 4]", n_points=4),
            Piece(interval="(4, 7]", n_points=3),
        )
    )
    assert grid.n_points == 7


def test_piecewise_lin_spaced_grid_invalid_gap():
    """[a, x) followed by (x, b] should be invalid (gap at x)."""
    with pytest.raises(GridInitializationError, match="not adjacent"):
        PiecewiseLinSpacedGrid(
            pieces=(
                Piece(interval="[1, 4)", n_points=3),
                Piece(interval="(4, 7]", n_points=3),
            )
        )


def test_piecewise_lin_spaced_grid_invalid_overlap():
    """[a, x] followed by [x, b] should be invalid (overlap at x)."""
    with pytest.raises(GridInitializationError, match="not adjacent"):
        PiecewiseLinSpacedGrid(
            pieces=(
                Piece(interval="[1, 4]", n_points=3),
                Piece(interval="[4, 7]", n_points=3),
            )
        )


def test_piecewise_lin_spaced_grid_invalid_numeric_gap():
    """Pieces with numeric gap between them should be invalid."""
    with pytest.raises(GridInitializationError, match="gap"):
        PiecewiseLinSpacedGrid(
            pieces=(
                Piece(interval="[1, 3]", n_points=3),
                Piece(interval="[5, 7]", n_points=3),
            )
        )


def test_piecewise_lin_spaced_grid_invalid_not_tuple():
    """Pieces must be a tuple."""
    with pytest.raises(GridInitializationError, match="must be a tuple"):
        PiecewiseLinSpacedGrid(pieces=[Piece(interval="[1, 4]", n_points=3)])  # type: ignore[arg-type]


def test_piecewise_lin_spaced_grid_invalid_not_piece():
    """Each element in pieces must be a Piece object."""
    with pytest.raises(GridInitializationError, match="must be a Piece"):
        PiecewiseLinSpacedGrid(
            pieces=({"interval": "[1, 4]", "n_points": 3},)  # type: ignore[arg-type]
        )


def test_piecewise_lin_spaced_grid_invalid_n_points():
    """n_points must be >= 2."""
    with pytest.raises(GridInitializationError, match="n_points must be an int >= 2"):
        PiecewiseLinSpacedGrid(pieces=(Piece(interval="[1, 4]", n_points=1),))


def test_piecewise_lin_spaced_grid_invalid_interval_string():
    """Invalid interval string should raise error."""
    with pytest.raises(GridInitializationError, match="invalid"):
        PiecewiseLinSpacedGrid(pieces=(Piece(interval="invalid", n_points=3),))


def test_piecewise_lin_spaced_grid_three_pieces():
    """Grid with three adjacent pieces should work."""
    grid = PiecewiseLinSpacedGrid(
        pieces=(
            Piece(interval="[0, 3)", n_points=3),
            Piece(interval="[3, 7)", n_points=4),
            Piece(interval="[7, 10]", n_points=4),
        )
    )
    assert grid.n_points == 11
    points = grid.to_jax()
    assert float(points[0]) == 0.0
    assert float(points[-1]) == 10.0
