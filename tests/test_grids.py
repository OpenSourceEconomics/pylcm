from dataclasses import make_dataclass

import numpy as np
import pytest

from lcm.exceptions import GridInitializationError
from lcm.grids import (
    DiscreteGrid,
    LinspaceGrid,
    LogspaceGrid,
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
    grid = LinspaceGrid(start=1, stop=5, n_points=5)
    assert np.allclose(grid.to_jax(), np.linspace(1, 5, 5))


def test_logspace_grid_creation():
    grid = LogspaceGrid(start=1, stop=10, n_points=3)
    assert np.allclose(grid.to_jax(), np.logspace(np.log10(1), np.log10(10), 3))


def test_discrete_grid_creation():
    category_class = make_dataclass(
        "Category", [("a", int, 0), ("b", int, 1), ("c", int, 2)]
    )
    grid = DiscreteGrid(category_class)
    assert np.allclose(grid.to_jax(), np.arange(3))


def test_linspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LinspaceGrid(start=1, stop=0, n_points=10)


def test_logspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LogspaceGrid(start=1, stop=0, n_points=10)


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
    grid = LinspaceGrid(start=1, stop=5, n_points=5)
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
