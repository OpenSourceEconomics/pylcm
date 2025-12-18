from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp

from lcm import grid_helpers
from lcm.exceptions import GridInitializationError, format_messages
from lcm.utils import find_duplicates

if TYPE_CHECKING:
    from lcm.typing import Float1D, Int1D, ParamsDict, ScalarFloat


class Grid(ABC):
    """LCM Grid base class."""

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""


class DiscreteGrid(Grid):
    """A class representing a discrete grid.

    Args:
        category_class (type): The category class representing the grid categories. Must
            be a dataclass with fields that have unique scalar int or float values.

    Attributes:
        categories: The list of category names.
        codes: The list of category codes.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with scalar
            int or float fields.

    """

    def __init__(self, category_class: type) -> None:
        """Initialize the DiscreteGrid.

        Args:
            category_class (type): The category class representing the grid categories.
                Must be a dataclass with fields that have unique scalar int or float
                values.

        """
        _validate_discrete_grid(category_class)

        names_and_values = _get_field_names_and_values(category_class)

        self.__categories = tuple(names_and_values.keys())
        self.__codes = tuple(names_and_values.values())

    @property
    def categories(self) -> tuple[str, ...]:
        """Get the list of category names."""
        return self.__categories

    @property
    def codes(self) -> tuple[int | float, ...]:
        """Get the list of category codes."""
        return self.__codes

    def to_jax(self) -> Int1D:
        """Convert the grid to a Jax array."""
        return jnp.array(self.codes)


@dataclass(frozen=True, kw_only=True)
class ContinuousGrid(Grid, ABC):
    """LCM Continuous Grid base class."""

    start: int | float
    stop: int | float
    n_points: int

    def __post_init__(self) -> None:
        _validate_continuous_grid(
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )

    @abstractmethod
    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""

    @abstractmethod
    def get_coordinate(self, value: ScalarFloat, params: ParamsDict) -> ScalarFloat:
        """Get the generalized coordinate of a value in the grid."""

    def replace(self, **kwargs: float) -> ContinuousGrid:
        """Replace the attributes of the grid.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the grid.

        Returns:
            A new grid with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)  # type: ignore[arg-type]
        except TypeError as e:
            raise GridInitializationError(
                f"Failed to replace attributes of the grid. The error was: {e}"
            ) from e


class LinspaceGrid(ContinuousGrid):
    """A linear grid of continuous values.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 50.5, 100]`.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_helpers.linspace(self.start, self.stop, self.n_points)

    def get_coordinate(self, value: ScalarFloat, params: ParamsDict) -> ScalarFloat:  # noqa: ARG002
        """Get the generalized coordinate of a value in the grid."""
        return grid_helpers.get_linspace_coordinate(
            value, self.start, self.stop, self.n_points
        )


class LogspaceGrid(ContinuousGrid):
    """A logarithmic grid of continuous values.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 10, 100]`.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_helpers.logspace(self.start, self.stop, self.n_points)

    def get_coordinate(self, value: ScalarFloat, params: ParamsDict) -> ScalarFloat:  # noqa: ARG002
        """Get the generalized coordinate of a value in the grid."""
        return grid_helpers.get_logspace_coordinate(
            value, self.start, self.stop, self.n_points
        )


@dataclass(frozen=True, kw_only=True)
class ShockGrid(ContinuousGrid):
    """An empty grid for discretized continuous shocks.

    The actual values will be calculated once the prameters for the shock are
    available during the solution or simulation.


    Attributes:
        start: This argument is not used.
        stop: This argument is not used.
        n_points: The number of points in the grid. Must be an int greater than 0.
        type: The shock type.

    """

    start: int | float = 0
    stop: int | float = 1
    type: Literal["uniform", "normal", "tauchen", "rouwenhorst"]

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return jnp.zeros(shape=self.n_points)

    def get_coordinate(self, value: ScalarFloat, params: ParamsDict) -> ScalarFloat:
        """Get the generalized coordinate of a value in the grid."""
        return grid_helpers.get_shock_coordinate(
            value, n_points=self.n_points, params=params, distribution_type=self.type
        )


# ======================================================================================
# Validate user input
# ======================================================================================


def _validate_discrete_grid(category_class: type) -> None:
    """Validate the field names and values of the category_class passed to DiscreteGrid.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique scalar int or float values.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with scalar
            int or float fields.

    """
    error_messages = validate_category_class(category_class)
    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)


def validate_category_class(category_class: type) -> list[str]:
    """Validate a category class has proper structure for discrete grids.

    This validates that:
    - The class is a dataclass
    - It has at least one field
    - All field values are scalar int or float
    - All field values are unique
    - Field values are consecutive integers starting from 0

    Args:
        category_class: The category class to validate. Must be a dataclass with fields
            that have unique scalar int or float values.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages: list[str] = []

    if not is_dataclass(category_class):
        error_messages.append(
            "category_class must be a dataclass with scalar int or float fields, "
            f"but is {category_class}."
        )
        return error_messages

    names_and_values = _get_field_names_and_values(category_class)

    if not names_and_values:
        error_messages.append("category_class must have at least one field.")

    names_with_non_numerical_values = [
        name
        for name, value in names_and_values.items()
        if not isinstance(value, int | float)
    ]
    if names_with_non_numerical_values:
        error_messages.append(
            "Field values of the category_class can only be scalar int or float "
            f"values. The values to the following fields are not: "
            f"{names_with_non_numerical_values}"
        )

    values = list(names_and_values.values())

    duplicated_values = find_duplicates(values)
    if duplicated_values:
        error_messages.append(
            "Field values of the category_class must be unique. "
            f"The following values are duplicated: {duplicated_values}"
        )

    if values != list(range(len(values))):
        error_messages.append(
            "Field values of the category_class must be consecutive integers "
            "starting from 0 (e.g., 0, 1, 2, ...)."
        )

    return error_messages


def _get_field_names_and_values(dc: type) -> dict[str, Any]:
    """Get the fields of a dataclass.

    Args:
        dc: The dataclass to get the fields of.

    Returns:
        A dictionary with the field names as keys and the field values as values. If
        no value is provided for a field, the value is set to None.

    """
    return {field.name: getattr(dc, field.name, None) for field in fields(dc)}


def _validate_continuous_grid(
    start: float,
    stop: float,
    n_points: int,
) -> None:
    """Validate the continuous grid parameters.

    Args:
        start: The start value of the grid.
        stop: The stop value of the grid.
        n_points: The number of points in the grid.

    Raises:
        GridInitializationError: If the grid parameters are invalid.

    """
    error_messages = []

    valid_start_type = isinstance(start, int | float)
    if not valid_start_type:
        error_messages.append("start must be a scalar int or float value")

    valid_stop_type = isinstance(stop, int | float)
    if not valid_stop_type:
        error_messages.append("stop must be a scalar int or float value")

    if not isinstance(n_points, int) or n_points < 1:
        error_messages.append(
            f"n_points must be an int greater than 0 but is {n_points}",
        )

    if valid_start_type and valid_stop_type and start >= stop:
        error_messages.append("start must be less than stop")

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)
