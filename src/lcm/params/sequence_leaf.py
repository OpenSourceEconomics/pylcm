"""A Sequence wrapper that is a JAX pytree but not itself a Sequence."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax

if TYPE_CHECKING:
    from lcm.api.typing import _UserParamsLeaf
    from lcm.typing import _ParamsLeaf


class UserSequenceLeaf:
    """A Sequence wrapper that is a JAX pytree but not itself a Sequence.

    Holds the boundary leaf type — accepts the same wide value union as
    `Model.solve` / `Model.simulate` parameters (Python scalars, numpy
    arrays, `pd.Series`, JAX arrays, nested `UserMappingLeaf` /
    `UserSequenceLeaf`).

    Prevents `flatten_regime_namespace` from recursing into contents while
    allowing JAX to trace through array values. Data is frozen to
    immutable containers on construction.

    The constructor accepts `Sequence[Any]` at runtime so beartype's
    O(n) per-leaf check doesn't fire on user-supplied scalars or arrays;
    the precise leaf-type contract is enforced statically through the
    `data` class-attribute annotation.

    """

    __slots__ = ("data",)

    if TYPE_CHECKING:
        data: tuple[_UserParamsLeaf, ...]

    def __init__(self, data: Sequence[Any]) -> None:
        from lcm.utils.containers import _make_immutable  # noqa: PLC0415

        self.data = tuple(_make_immutable(v) for v in data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.data)!r})"

    def __hash__(self) -> int:
        return hash(self.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UserSequenceLeaf):
            return NotImplemented
        return self.data == other.data


class SequenceLeaf(UserSequenceLeaf):
    """Sequence leaf carrying only canonical-dtype values.

    Output of `cast_params_to_canonical_dtypes`. Every value is either a
    canonical JAX array (`FloatND` / `IntND` / `BoolND`) or another
    canonical leaf (`MappingLeaf` / `SequenceLeaf`).

    Subclasses `UserSequenceLeaf`, so any code that accepts the wide user
    form via `isinstance(_, UserSequenceLeaf)` also accepts the canonical
    form. Code requiring canonical values uses the narrower `SequenceLeaf`
    type explicitly.

    """

    __slots__ = ()

    if TYPE_CHECKING:
        data: tuple[_ParamsLeaf, ...]


def _user_flatten(leaf: UserSequenceLeaf) -> tuple[list[Any], None]:
    return list(leaf.data), None


def _user_unflatten(_aux: None, values: Sequence[Any]) -> UserSequenceLeaf:
    return UserSequenceLeaf(values)


def _canonical_unflatten(_aux: None, values: Sequence[Any]) -> SequenceLeaf:
    return SequenceLeaf(values)


jax.tree_util.register_pytree_node(UserSequenceLeaf, _user_flatten, _user_unflatten)
jax.tree_util.register_pytree_node(SequenceLeaf, _user_flatten, _canonical_unflatten)
