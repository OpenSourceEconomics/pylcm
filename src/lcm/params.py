"""Public params helpers — the JAX-pytree leaf wrappers.

`as_leaf` wraps a `Mapping` or `Sequence` of parameters as a single JAX-pytree
leaf. The leaf classes themselves are re-exported here; their definitions and
the engine-side params machinery live in `_lcm.params`.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, overload

from beartype import beartype

from _lcm.beartype_conf import PARAMS_CONF
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf, UserSequenceLeaf
from lcm.temporal import (
    TimeVarying,
    UnlabelledTimeParameterWarning,
    time_varying_params,
)

if TYPE_CHECKING:
    from lcm.typing import UserParamsLeaf
else:
    # `lcm.typing` imports this module before it defines `UserParamsLeaf`, and
    # `as_leaf` is decorated at import; the canonical dtype cast refuses an
    # unsupported entry.
    type UserParamsLeaf = object


@overload
def as_leaf(data: Mapping[str, UserParamsLeaf]) -> UserMappingLeaf: ...


@overload
def as_leaf(data: Sequence[UserParamsLeaf]) -> UserSequenceLeaf: ...


@beartype(conf=PARAMS_CONF)
def as_leaf(
    data: Mapping[str, UserParamsLeaf] | Sequence[UserParamsLeaf],
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


__all__ = [
    "MappingLeaf",
    "SequenceLeaf",
    "TimeVarying",
    "UnlabelledTimeParameterWarning",
    "UserMappingLeaf",
    "UserSequenceLeaf",
    "as_leaf",
    "time_varying_params",
]
