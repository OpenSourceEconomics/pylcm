from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING, TypeVar, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T")


def find_duplicates(*containers: Iterable[T]) -> set[T]:
    combined = chain.from_iterable(containers)
    counts = Counter(combined)
    return {v for v, count in counts.items() if count > 1}


def first_non_none(*args: T | None) -> T:
    """Return the first non-None argument.

    Args:
        *args: Arguments to check.

    Returns:
        The first non-None argument.

    Raises:
        ValueError: If all arguments are None.

    """
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError("All arguments are None")


def set_frozen_attr(obj: Any, name: str, value: Any) -> None:  # noqa: ANN401
    """Robust attribute setting for frozen dataclasses.

    Args:
        obj: The frozen dataclass instance
        name: Name of the attribute to set
        value: Value to set

    """
    object.__setattr__(obj, name, value)