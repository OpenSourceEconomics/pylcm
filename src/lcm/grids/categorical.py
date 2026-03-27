import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass

import pandas as pd

from lcm.exceptions import GridInitializationError, format_messages
from lcm.utils.containers import find_duplicates, get_field_names_and_values


def categorical[T](*, ordered: bool) -> Callable[[type[T]], type[T]]:
    """Create a categorical class with auto-assigned integer values.

    Transforms a class with int annotations into a frozen dataclass where each
    field is assigned a consecutive integer value starting from 0.

    Example:
        @categorical(ordered=False)
        class LaborSupply:
            work: int
            retire: int

        # Equivalent to:
        @dataclass(frozen=True)
        class LaborSupply:
            work: int = 0
            retire: int = 1

        # Usage:
        LaborSupply.work   # 0
        LaborSupply.retire # 1

    Args:
        ordered: Whether the categories have a meaningful ordering. Must be
            explicitly specified.

    Returns:
        A decorator that creates a frozen dataclass with auto-assigned integer values.

    """

    def decorator(cls: type[T]) -> type[T]:
        annotations = getattr(cls, "__annotations__", {})

        # Assign sequential integers as defaults
        for i, name in enumerate(annotations):
            setattr(cls, name, i)

        cls._ordered = ordered  # ty: ignore[unresolved-attribute]

        @classmethod
        def _to_categorical_dtype(cls: type) -> pd.CategoricalDtype:
            """Return a `pd.CategoricalDtype` with the category names of this class."""
            import pandas as pd  # noqa: PLC0415

            names = [f.name for f in dataclasses.fields(cls)]
            return pd.CategoricalDtype(categories=names, ordered=cls._ordered)  # ty: ignore[unresolved-attribute]

        cls.to_categorical_dtype = _to_categorical_dtype  # ty: ignore[unresolved-attribute]

        # Apply dataclass decorator
        return dataclass(frozen=True)(cls)

    return decorator


def validate_category_class(category_class: type) -> list[str]:
    """Validate a category class has proper structure for discrete grids.

    This validates that:
    - The class is a dataclass
    - It has at least one field
    - All field values are int
    - All field values are unique
    - Field values are consecutive integers starting from 0

    Args:
        category_class: The category class to validate. Must be a dataclass with fields
            that have unique int values.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages: list[str] = []

    if not is_dataclass(category_class):
        error_messages.append(
            "category_class must be a dataclass with int fields, "
            f"but is {category_class}."
        )
        return error_messages

    names_and_values = get_field_names_and_values(category_class)

    if not names_and_values:
        error_messages.append("category_class must have at least one field.")

    names_with_non_int_values = [
        name for name, value in names_and_values.items() if not isinstance(value, int)
    ]
    if names_with_non_int_values:
        error_messages.append(
            "Field values of the category_class can only be int values. "
            f"The values to the following fields are not: "
            f"{names_with_non_int_values}"
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


def _validate_discrete_grid(category_class: type) -> None:
    """Validate the field names and values of the category_class passed to DiscreteGrid.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields.

    """
    error_messages = validate_category_class(category_class)
    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)
