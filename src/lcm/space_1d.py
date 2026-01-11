"""Decorator for creating 1D space classes."""

from __future__ import annotations

from dataclasses import dataclass


def space_1d[T](cls: type[T]) -> type[T]:
    """Decorator to create a 1D space class with auto-assigned integer values.

    Transforms a class with int annotations into a frozen dataclass where each
    field is assigned a consecutive integer value starting from 0.

    Example:
        @space_1d
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
        cls: The class to decorate.

    Returns:
        A frozen dataclass with auto-assigned integer values.

    """
    annotations = getattr(cls, "__annotations__", {})

    # Assign sequential integers as defaults
    for i, name in enumerate(annotations):
        setattr(cls, name, i)

    # Apply dataclass decorator
    return dataclass(frozen=True)(cls)
