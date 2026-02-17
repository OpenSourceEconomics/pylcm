"""A Mapping wrapper that is a JAX pytree but not itself a Mapping."""

from collections.abc import Mapping
from typing import Any

import jax


class MappingLeaf:
    """A Mapping wrapper that is a JAX pytree but not itself a Mapping.

    Prevents flatten_regime_namespace from recursing into contents while
    allowing JAX to trace through array values.

    Data is frozen to immutable containers on construction.
    """

    __slots__ = ("data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        from lcm.utils import ensure_containers_are_immutable  # noqa: PLC0415

        self.data = ensure_containers_are_immutable(data)

    def __repr__(self) -> str:
        return f"MappingLeaf({dict(self.data)!r})"

    __hash__ = None  # MappingProxyType is not hashable

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MappingLeaf):
            return NotImplemented
        return self.data == other.data


def _flatten(nmp: MappingLeaf) -> tuple[list[Any], tuple[str, ...]]:
    keys = tuple(sorted(nmp.data.keys()))
    values = [nmp.data[k] for k in keys]
    return values, keys


def _unflatten(keys: tuple[str, ...], values: list[Any]) -> MappingLeaf:
    return MappingLeaf(dict(zip(keys, values, strict=True)))


jax.tree_util.register_pytree_node(MappingLeaf, _flatten, _unflatten)
