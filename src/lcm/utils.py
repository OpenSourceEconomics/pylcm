from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING, Any, TypeVar

from dags.tree import QNAME_DELIMITER, flatten_to_qnames, unflatten_from_qnames

# Re-export for use in other modules. This is the separator used by dags to
# concatenate nested dictionary keys into qualified names (e.g., "work__next_wealth").
# User-defined regime names and function names must NOT contain this separator.
REGIME_SEPARATOR = QNAME_DELIMITER

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lcm.typing import RegimeName

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


def flatten_regime_namespace(d: dict[RegimeName, Any]) -> dict[str, Any]:
    return flatten_to_qnames(d)


def unflatten_regime_namespace(d: dict[str, Any]) -> dict[RegimeName, Any]:
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]
