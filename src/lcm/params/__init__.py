from collections.abc import Mapping, Sequence
from typing import Any, overload

from lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from lcm.params.sequence_leaf import SequenceLeaf, UserSequenceLeaf


@overload
def as_leaf(data: Mapping[str, Any]) -> UserMappingLeaf: ...


@overload
def as_leaf(data: Sequence[Any]) -> UserSequenceLeaf: ...


def as_leaf(
    data: Mapping[str, Any] | Sequence[Any],
) -> UserMappingLeaf | UserSequenceLeaf:
    """Wrap a Mapping or Sequence as a JAX-pytree leaf.

    Returns the boundary (`User...Leaf`) variant — accepts Python scalars,
    numpy arrays, `pd.Series`, JAX arrays, and nested leaves. The
    canonical narrowed variants (`MappingLeaf` / `SequenceLeaf`) are the
    output of `cast_params_to_canonical_dtypes`.

    """
    if isinstance(data, Mapping):
        return UserMappingLeaf(dict(data))
    if isinstance(data, Sequence):
        return UserSequenceLeaf(data)
    msg = f"as_leaf() expects a Mapping or Sequence, got {type(data).__name__}"
    raise TypeError(msg)


def __getattr__(name: str) -> object:
    if name == "process_params":
        from lcm.params.processing import process_params  # noqa: PLC0415

        return process_params
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "MappingLeaf",
    "SequenceLeaf",
    "UserMappingLeaf",
    "UserSequenceLeaf",
    "as_leaf",
    "process_params",
]
