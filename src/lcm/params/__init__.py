from collections.abc import Mapping, Sequence
from typing import Any, overload

from lcm.params.mapping_leaf import MappingLeaf
from lcm.params.sequence_leaf import SequenceLeaf


@overload
def as_leaf(data: Mapping[str, Any]) -> MappingLeaf: ...


@overload
def as_leaf(data: Sequence[Any]) -> SequenceLeaf: ...


def as_leaf(data: Mapping[str, Any] | Sequence[Any]) -> MappingLeaf | SequenceLeaf:
    """Wrap a Mapping or Sequence as a JAX-pytree leaf."""
    if isinstance(data, Mapping):
        return MappingLeaf(dict(data))
    if isinstance(data, Sequence):
        return SequenceLeaf(data)
    msg = f"as_leaf() expects a Mapping or Sequence, got {type(data).__name__}"
    raise TypeError(msg)


__all__ = ["MappingLeaf", "SequenceLeaf", "as_leaf"]
