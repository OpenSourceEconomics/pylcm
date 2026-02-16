from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import fields
from itertools import chain
from types import MappingProxyType
from typing import Any, TypeVar, cast

import jax.numpy as jnp
from dags.tree import flatten_to_qnames, unflatten_from_qnames
from jax import Array

from lcm.params import MappingLeaf
from lcm.typing import RegimeName

T = TypeVar("T")


def _make_immutable(value: Any) -> Any:  # noqa: ANN401
    """Recursively convert a value to its immutable equivalent."""
    if isinstance(value, MappingLeaf):
        return MappingLeaf(
            MappingProxyType({k: _make_immutable(v) for k, v in value.data.items()})
        )
    if isinstance(value, (MappingProxyType, tuple, frozenset)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({k: _make_immutable(v) for k, v in value.items()})
    if isinstance(value, set):
        return frozenset(_make_immutable(v) for v in value)
    if isinstance(value, list):
        return tuple(_make_immutable(v) for v in value)
    return value


def ensure_containers_are_immutable[K, V](
    value: Mapping[K, V],
) -> MappingProxyType[K, V]:
    """Recursively convert mutable containers to immutable equivalents.

    Conversions:
        - dict/Mapping -> MappingProxyType
        - list -> tuple
        - set -> frozenset

    This utility ensures deep immutability of nested data structures. Values that
    are already immutable (MappingProxyType, tuple, frozenset) are returned as-is.

    Args:
        value: Any Mapping to convert.

    Returns:
        A MappingProxyType containing the mapping's items, with all nested containers
        converted to their immutable equivalents.

    """
    return cast("MappingProxyType[K, V]", _make_immutable(value))


def _make_mutable(value: Any) -> Any:  # noqa: ANN401
    """Recursively convert a value to its mutable equivalent."""
    if isinstance(value, MappingLeaf):
        return MappingLeaf({k: _make_mutable(v) for k, v in value.data.items()})
    if isinstance(value, (set, list)):
        return value
    if isinstance(value, (MappingProxyType, Mapping)):
        return {k: _make_mutable(v) for k, v in value.items()}
    if isinstance(value, frozenset):
        return {_make_mutable(v) for v in value}
    if isinstance(value, tuple):
        return [_make_mutable(v) for v in value]
    return value


def ensure_containers_are_mutable[K, V](value: Mapping[K, V]) -> dict[K, V]:
    """Recursively convert immutable containers to mutable equivalents.

    Conversions:
        - MappingProxyType/Mapping -> dict
        - tuple -> list
        - frozenset -> set

    This utility ensures deep mutability of nested data structures. Values that
    are already mutable (dict, list, set) are returned as-is.

    Args:
        value: Any Mapping to convert.

    Returns:
        A dict containing the mapping's items, with all nested containers
        converted to their mutable equivalents.

    """
    return cast("dict[K, V]", _make_mutable(value))


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
