"""Thin wrappers around `dags.signature` that default to `forwarder=True`.

Every direct caller of `dags.with_signature` / `dags.rename_arguments`
inside pylcm produces a generic `*args, **kwargs` forwarder whose
annotations describe the inner function's contract, not the wrapper's
own runtime call protocol. Setting `forwarder=True` on each call tells
beartype's import claw to treat the wrapper as permissive — matching
the wrapper's actual behaviour — instead of enforcing user annotations
against JAX tracers and other forwarded objects.

Importing these wrappers (rather than the dags ones) keeps that intent
visible at each call site and concentrates the configuration in one
place if pylcm ever needs to flip the default.

"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

from dags import rename_arguments as _rename_arguments
from dags import with_signature as _with_signature
from dags.typing import P, R


@overload
def with_signature(
    func: Callable[P, R],
    *,
    args: Mapping[str, str] | Sequence[str] | None = None,
    kwargs: Mapping[str, str] | Sequence[str] | None = None,
    enforce: bool = True,
    return_annotation: Any = inspect.Parameter.empty,  # noqa: ANN401
) -> Callable[P, R]: ...


@overload
def with_signature(
    *,
    args: Mapping[str, str] | Sequence[str] | None = None,
    kwargs: Mapping[str, str] | Sequence[str] | None = None,
    enforce: bool = True,
    return_annotation: Any = inspect.Parameter.empty,  # noqa: ANN401
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def with_signature(
    func: Callable[P, R] | None = None,
    *,
    args: Mapping[str, str] | Sequence[str] | None = None,
    kwargs: Mapping[str, str] | Sequence[str] | None = None,
    enforce: bool = True,
    return_annotation: Any = inspect.Parameter.empty,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """`dags.with_signature` with `forwarder=True` baked in.

    See `dags.signature.with_signature` for argument semantics. The
    `forwarder=True` setting advertises the resulting wrapper as a
    permissive `*args, **kwargs` forwarder on `__annotations__`,
    keeping beartype's claw from enforcing per-parameter annotations
    against the forwarder's actual call arguments.
    """
    return _with_signature(
        func,  # ty: ignore[invalid-argument-type]
        args=args,
        kwargs=kwargs,
        enforce=enforce,
        return_annotation=return_annotation,
        forwarder=True,
    )


@overload
def rename_arguments(
    func: Callable[P, R], *, mapper: Mapping[str, str]
) -> Callable[..., R]: ...


@overload
def rename_arguments(
    *, mapper: Mapping[str, str]
) -> Callable[[Callable[P, R]], Callable[..., R]]: ...


def rename_arguments(
    func: Callable[P, R] | None = None,
    *,
    mapper: Mapping[str, str] | None = None,
) -> Callable[..., R] | Callable[[Callable[P, R]], Callable[..., R]]:
    """`dags.rename_arguments` with `forwarder=True` baked in.

    See `dags.signature.rename_arguments` for argument semantics.
    """
    return _rename_arguments(
        func,  # ty: ignore[invalid-argument-type]
        mapper=mapper,  # ty: ignore[invalid-argument-type]
        forwarder=True,
    )
