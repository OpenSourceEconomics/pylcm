"""A Sequence wrapper that is a JAX pytree but not itself a Sequence."""

from collections.abc import Sequence
from typing import Any

import jax


class SequenceLeaf:
    """A Sequence wrapper that is a JAX pytree but not itself a Sequence.

    Prevents flatten_regime_namespace from recursing into contents while
    allowing JAX to trace through array values.

    Data is frozen to immutable containers on construction.
    """

    __slots__ = ("data",)

    def __init__(self, data: Sequence[Any]) -> None:
        from lcm.utils import _make_immutable  # noqa: PLC0415

        self.data = tuple(_make_immutable(v) for v in data)

    def __repr__(self) -> str:
        return f"SequenceLeaf({list(self.data)!r})"

    def __hash__(self) -> int:
        return hash(self.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceLeaf):
            return NotImplemented
        return self.data == other.data


def _flatten(sl: SequenceLeaf) -> tuple[list[Any], None]:
    return list(sl.data), None


def _unflatten(_aux: None, values: list[Any]) -> SequenceLeaf:
    return SequenceLeaf(values)


jax.tree_util.register_pytree_node(SequenceLeaf, _flatten, _unflatten)
