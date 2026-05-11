import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass

import jax
import jax.numpy as jnp
import pandas as pd

from lcm.exceptions import (
    CategoricalDefinitionError,
    GridInitializationError,
    format_messages,
)
from lcm.typing import ScalarInt
from lcm.utils.containers import find_duplicates, get_field_names_and_values


def categorical[T](*, ordered: bool) -> Callable[[type[T]], type[T]]:
    """Create a categorical class with auto-assigned `ScalarInt` values.

    Transforms a class with `ScalarInt`-annotated fields into a frozen
    dataclass where each field is assigned a consecutive 0-d `jnp.int32`
    scalar starting from 0. Decoration fails with
    `CategoricalDefinitionError` if any field is annotated differently.

    Example:
        from lcm.typing import ScalarInt

        @categorical(ordered=False)
        class LaborSupply:
            work: ScalarInt
            retire: ScalarInt

        LaborSupply.work       # Array(0, dtype=int32)
        LaborSupply.retire     # Array(1, dtype=int32)

    Args:
        ordered: Whether the categories have a meaningful ordering. Must be
            explicitly specified.

    Returns:
        A decorator that creates a frozen dataclass with auto-assigned
        `ScalarInt` values.

    """

    def decorator(cls: type[T]) -> type[T]:
        annotations = getattr(cls, "__annotations__", {})

        bad_fields = {
            name: annot for name, annot in annotations.items() if annot is not ScalarInt
        }
        if bad_fields:
            details = ", ".join(
                f"`{name}: {getattr(annot, '__name__', annot)}`"
                for name, annot in bad_fields.items()
            )
            raise CategoricalDefinitionError(
                f"@categorical-decorated class {cls.__qualname__!r} must annotate "
                f"every field as `ScalarInt` (the 0-d int32 scalar pylcm produces "
                f"for category codes at runtime). The following fields are "
                f"annotated otherwise: {details}. Import via "
                f"`from lcm.typing import ScalarInt`."
            )

        # Mark fields as `init=False` with sequential int defaults so the
        # generated `__init__` doesn't write per-instance values. The class
        # attributes are then swapped to `jnp.int32` scalars post-decoration
        # (frozen dataclasses reject `jax.Array` defaults directly under
        # Python 3.14's mutable-default check, but class-level overrides via
        # `type.__setattr__` are allowed).
        for i, name in enumerate(annotations):
            setattr(cls, name, field(default=i, init=False))

        cls._ordered = ordered  # ty: ignore[unresolved-attribute]

        @classmethod
        def _to_categorical_dtype(cls: type) -> pd.CategoricalDtype:
            """Return a `pd.CategoricalDtype` with the category names of this class."""
            import pandas as pd  # noqa: PLC0415

            names = [f.name for f in dataclasses.fields(cls)]
            return pd.CategoricalDtype(categories=names, ordered=cls._ordered)  # ty: ignore[unresolved-attribute]

        cls.to_categorical_dtype = _to_categorical_dtype  # ty: ignore[unresolved-attribute]

        new_cls = dataclass(frozen=True)(cls)

        # Class-level access (`X.foo`) and instance-level fall-through
        # (`X().foo`, with `init=False` leaving instance __dict__ empty)
        # both resolve to these `ScalarInt` scalars.
        for i, name in enumerate(annotations):
            type.__setattr__(new_cls, name, jnp.int32(i))

        return new_cls

    return decorator


def validate_category_class(category_class: type) -> list[str]:
    """Validate a category class has proper structure for discrete grids.

    This validates that:
    - The class is a dataclass
    - It has at least one field
    - All field values are `ScalarInt`s
    - All field values are unique
    - Field values are consecutive integers starting from 0

    Args:
        category_class: The category class to validate. Must be a dataclass with fields
            whose values are unique `ScalarInt`s.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages: list[str] = []

    if not is_dataclass(category_class):
        error_messages.append(
            "category_class must be a dataclass with `ScalarInt` fields, "
            f"but is {category_class}."
        )
        return error_messages

    names_and_values = get_field_names_and_values(category_class)

    if not names_and_values:
        error_messages.append("category_class must have at least one field.")

    names_with_bad_values = [
        name for name, value in names_and_values.items() if not _is_scalar_int(value)
    ]
    if names_with_bad_values:
        error_messages.append(
            "Field values of the category_class must be `ScalarInt` "
            "(0-d int32 jax scalars). The values to the following fields are "
            f"not: {names_with_bad_values}"
        )
        # The remaining checks coerce via `int(...)`; bail out if any value
        # cannot be coerced cleanly.
        return error_messages

    values_as_py = [int(v) for v in names_and_values.values()]

    duplicated_values = find_duplicates(values_as_py)
    if duplicated_values:
        error_messages.append(
            "Field values of the category_class must be unique. "
            f"The following values are duplicated: {duplicated_values}"
        )

    if values_as_py != list(range(len(values_as_py))):
        error_messages.append(
            "Field values of the category_class must be consecutive integers "
            "starting from 0 (e.g., 0, 1, 2, ...)."
        )

    return error_messages


def _is_scalar_int(value: object) -> bool:
    """Return True iff `value` is a 0-d integer jax array (`ScalarInt`)."""
    return (
        isinstance(value, jax.Array)
        and value.shape == ()
        and jnp.issubdtype(value.dtype, jnp.integer)
    )


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
