"""Container helpers the engine uses to keep its own data immutable.

`ensure_containers_are_immutable` and its inverse convert nested mappings and
sequences between the frozen and mutable forms. The remaining helpers —
`find_duplicates`, `invert_regime_ids`, `first_non_none` — are the small
lookups several build stages share.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import TypeVar, cast, overload

from _lcm.params.mapping_leaf import LeafEntry
from lcm.params import UserMappingLeaf, UserSequenceLeaf
from lcm.typing import ScalarInt

T = TypeVar("T")

# A value inside a nested container: a leaf the caller stores, such as a grid, a
# function, an array or a params leaf, or a container of such values.
type _ContainerValue = object  # noqa: PAN001 - callers nest any leaf, and the conversion changes the container type at every level


class Unset:
    """Sentinel for parameters that haven't been explicitly set."""

    def __repr__(self) -> str:
        return "Unset()"


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
    """Return elements that appear more than once across all containers."""
    combined = chain.from_iterable(containers)
    counts = Counter(combined)
    return {v for v, count in counts.items() if count > 1}


def invert_regime_ids[K](
    mapping: Mapping[K, ScalarInt | int],
) -> MappingProxyType[int, K]:
    """Return the inverse of a regime-name → id mapping, with Python-`int` keys.

    `@categorical` assigns `jnp.int32` scalars to class attributes, so
    `regime_names_to_ids` values are 0-d jax arrays — which aren't
    hashable and therefore can't serve as `dict` keys. This helper
    coerces each value to a Python `int` so the inverted lookup
    (`id → name`) is usable wherever JAX promotion isn't already
    happening.
    """
    return MappingProxyType({int(v): k for k, v in mapping.items()})


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


@overload
def _make_immutable(value: LeafEntry) -> LeafEntry: ...
@overload
def _make_immutable(value: _ContainerValue) -> _ContainerValue: ...
def _make_immutable(value: _ContainerValue) -> _ContainerValue:
    """Recursively convert a value to its immutable equivalent.

    A frozen params leaf entry is again a leaf entry: mappings become
    `MappingProxyType`, lists become tuples, and leaves are returned as they are.
    """
    if isinstance(value, (UserMappingLeaf, UserSequenceLeaf)):
        return value  # already immutable by construction
    if isinstance(value, (MappingProxyType, tuple, frozenset)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({k: _make_immutable(v) for k, v in value.items()})
    if isinstance(value, set):
        return frozenset(_make_immutable(v) for v in value)
    if isinstance(value, list):
        return tuple(_make_immutable(v) for v in value)
    return value


def _make_mutable(value: _ContainerValue) -> _ContainerValue:  # noqa: PLR0911
    """Recursively convert a value to its mutable equivalent."""
    if isinstance(value, UserMappingLeaf):
        return {k: _make_mutable(v) for k, v in value.data.items()}
    if isinstance(value, UserSequenceLeaf):
        return [_make_mutable(v) for v in value.data]
    if isinstance(value, (set, list)):
        return value
    if isinstance(value, (MappingProxyType, Mapping)):
        return {k: _make_mutable(v) for k, v in value.items()}
    if isinstance(value, frozenset):
        return {_make_mutable(v) for v in value}
    if isinstance(value, tuple):
        return [_make_mutable(v) for v in value]
    return value
