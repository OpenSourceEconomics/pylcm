"""Make jaxtyping's anonymous-variadic-dim sentinel survive pickling.

jaxtyping marks a `"..."` axis with a module-level `object()` sentinel
(`_anonymous_variadic_dim`). A plain `object()` does not keep its identity
across a pickle round-trip, so cloudpickling a value whose type annotations
reference a `Foo[Array, "..."]` type — which the beartype claw makes
pervasive — yields a type whose variadic-dim marker no longer matches the
live module global. jaxtyping's shape check then trips
`assert type(variadic_dim) is _NamedVariadicDim`.

Replacing the sentinel with a `__reduce__`-backed singleton makes it
round-trip to the same object, so unpickled annotation types stay valid.
This module must be imported before any `jaxtyping`-subscripted type is
created — `lcm/__init__.py` imports it before every other `lcm` submodule.
"""

from typing import Self

from jaxtyping import _array_types


class _AnonymousVariadicDim:
    """Picklable singleton for jaxtyping's `"..."` axis marker."""

    _instance: Self | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self) -> tuple[type[_AnonymousVariadicDim], tuple[()]]:
        return (_AnonymousVariadicDim, ())


_array_types._anonymous_variadic_dim = _AnonymousVariadicDim()  # noqa: SLF001
