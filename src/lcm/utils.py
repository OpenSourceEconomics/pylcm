from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import fields
from itertools import chain
from types import MappingProxyType
from typing import Any, TypeVar

import jax.numpy as jnp
from dags.tree import QNAME_DELIMITER, flatten_to_qnames, unflatten_from_qnames
from jax import Array

from lcm.typing import RegimeName

# Re-export for use in other modules. This is the separator used by dags to
# concatenate nested dictionary keys into qualified names (e.g., "work__next_wealth").
# User-defined regime names and function names must NOT contain this separator.
REGIME_SEPARATOR = QNAME_DELIMITER

T = TypeVar("T")


def find_duplicates(*containers: Iterable[T]) -> set[T]:
    combined = chain.from_iterable(containers)
    counts = Counter(combined)
    return {v for v, count in counts.items() if count > 1}


def get_field_names_and_values(dc: type) -> MappingProxyType[str, Any]:
    """Return the fields of a dataclass.

    Args:
        dc: The dataclass to get the fields of.

    Returns:
        An immutable dictionary with the field names as keys and the field values as
        values. If no value is provided for a field, the value is set to None.

    """
    return MappingProxyType(
        {field.name: getattr(dc, field.name, None) for field in fields(dc)}
    )


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


def flatten_regime_namespace(d: Mapping[RegimeName, Any]) -> MappingProxyType[str, Any]:
    return MappingProxyType(flatten_to_qnames(d))


def unflatten_regime_namespace(d: dict[str, Any]) -> dict[RegimeName, Any]:
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]


def normalize_regime_transition_probs(
    probs: MappingProxyType[str, Array],
    active_regimes_next_period: tuple[str, ...],
) -> MappingProxyType[str, Array]:
    """Normalize regime transition probabilities over active regimes only."""
    if not active_regimes_next_period:
        return MappingProxyType({})
    active_probs = jnp.stack([probs[r] for r in active_regimes_next_period])
    total = jnp.sum(active_probs, axis=0)
    return MappingProxyType({r: probs[r] / total for r in active_regimes_next_period})
