"""A Mapping wrapper that is a JAX pytree but not itself a Mapping."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax

if TYPE_CHECKING:
    from _lcm.typing import _ParamsLeaf
    from lcm.typing import _UserParamsLeaf


class UserMappingLeaf:
    """A Mapping wrapper that is a JAX pytree but not itself a Mapping.

    Holds the boundary leaf type — accepts the same wide value union as
    `Model.solve` / `Model.simulate` parameters (Python scalars, numpy
    arrays, `pd.Series`, JAX arrays, nested `UserMappingLeaf` /
    `UserSequenceLeaf`).

    Prevents `flatten_regime_namespace` from recursing into contents while
    allowing JAX to trace through array values. The `data` attribute is
    deep-converted to immutable containers (`MappingProxyType`, `tuple`,
    `frozenset`) on construction; instances themselves are not hashable
    (`__hash__ = None`), since `MappingProxyType` isn't.

    The constructor accepts `Mapping[str, Any]` at runtime so beartype's
    O(n) per-leaf check doesn't fire on user-supplied scalars or arrays;
    the precise leaf-type contract is enforced statically through the
    `data` class-attribute annotation.

    """

    __slots__ = ("data",)

    if TYPE_CHECKING:
        data: Mapping[str, _UserParamsLeaf]

    def __init__(self, data: Mapping[str, Any]) -> None:
        from _lcm.utils.containers import (  # noqa: PLC0415
            ensure_containers_are_immutable,
        )

        self.data = ensure_containers_are_immutable(data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self.data)!r})"

    __hash__ = None  # MappingProxyType is not hashable

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UserMappingLeaf):
            return NotImplemented
        return self.data == other.data


class MappingLeaf(UserMappingLeaf):
    """Mapping leaf carrying only canonical-dtype values.

    Output of `cast_params_to_canonical_dtypes`. Every value is either a
    canonical JAX array (`FloatND` / `IntND` / `BoolND`) or another
    canonical leaf (`MappingLeaf` / `SequenceLeaf`).

    Subclasses `UserMappingLeaf`, so any code that accepts the wide user
    form via `isinstance(_, UserMappingLeaf)` also accepts the canonical
    form. Code requiring canonical values uses the narrower `MappingLeaf`
    type explicitly.

    """

    __slots__ = ()

    if TYPE_CHECKING:
        data: Mapping[str, _ParamsLeaf]


def _user_flatten(
    leaf: UserMappingLeaf,
) -> tuple[list[Any], tuple[str, ...]]:
    keys = tuple(sorted(leaf.data.keys()))
    values = [leaf.data[k] for k in keys]
    return values, keys


def _user_unflatten(keys: tuple[str, ...], values: Sequence[Any]) -> UserMappingLeaf:
    return UserMappingLeaf(dict(zip(keys, values, strict=True)))


def _canonical_unflatten(keys: tuple[str, ...], values: Sequence[Any]) -> MappingLeaf:
    return MappingLeaf(dict(zip(keys, values, strict=True)))


jax.tree_util.register_pytree_node(UserMappingLeaf, _user_flatten, _user_unflatten)
jax.tree_util.register_pytree_node(MappingLeaf, _user_flatten, _canonical_unflatten)
