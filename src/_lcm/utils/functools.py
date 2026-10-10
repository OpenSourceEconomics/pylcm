"""Call-convention adapters that wrap a function without defining one per call.

Each adapter is a frozen dataclass whose `__call__` is defined once at import. A
wrapper defined inside the adapting function would be decorated by the package's
beartype claw on every call and memoized for the life of the process, pinning the
wrapped function and everything it closes over. The adapters carry the wrapped
function's name, docstring, and attributes so that debugging, `inspect.signature`,
and `dags` see the function they stand in for.
"""

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeIs, TypeVar, cast

import jax.numpy as jnp
from jax import Array

from _lcm.typing import ArrayTree, PytreeValue
from lcm.typing import ReferenceName, UserFunction, UserFunctionResult, ValueND

ReturnType = TypeVar("ReturnType")

# Names `functools.wraps` copies, minus the deferred annotations (PEP 649):
# an adapter's own `(*args: PytreeValue, **kwargs: PytreeValue)` annotations
# must stay in force so that the beartype claw never enforces user-model types
# on a forwarder.
_WRAPPER_ASSIGNMENTS: tuple[str, ...] = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__type_params__",
)


def allow_only_kwargs(
    *, func: Callable[..., ReturnType], enforce: bool = True
) -> Callable[..., ReturnType]:
    """Restrict a function to be called with only keyword arguments.

    Args:
        func: The function to be wrapped.
        enforce: Whether to enforce the signature.

    Returns:
        A Callable with the same arguments as func (but with the additional restriction
            that it is only callable with keyword arguments).

    """
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Create new signature without positional-only arguments
    new_parameters = [
        p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in parameters.values()
    ]
    new_signature = signature.replace(parameters=new_parameters)

    # We cast to F here to signal ty that the return type is the same as the input
    # type. This ignores the change of parameters from positional to keyword-only
    # arguments.
    # TODO(@timmens): Remove this cast once we find an explicit way to specify the
    # change from positional to keyword-only parameter in the function signature
    # https://github.com/opensourceeconomics/pylcm/issues/80.
    return cast(
        "Callable[..., ReturnType]",
        _KeywordOnlyAdapter(
            func=func,
            signature_parameter_names=tuple(parameters),
            keyword_only_names=tuple(
                p.name
                for p in parameters.values()
                if p.kind == inspect.Parameter.KEYWORD_ONLY
            ),
            enforce=enforce,
            signature=new_signature,
        ),
    )


def _split_bound_arguments[T](
    *,
    bound: inspect.BoundArguments,
    n_positional: int,
    original_signature: inspect.Signature,
    adapted_signature: inspect.Signature,
) -> tuple[list[T], dict[ReferenceName, T]]:
    """Preserve how positional-or-keyword values reached the adapter."""
    positional_origins: set[str] = set()
    remaining_positional = n_positional
    for parameter in adapted_signature.parameters.values():
        if remaining_positional == 0:
            break
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            positional_origins.add(parameter.name)
            remaining_positional -= 1
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            positional_origins.add(parameter.name)
            break

    forwarded_args: list[T] = []
    forwarded_kwargs: dict[ReferenceName, T] = {}
    for name, value in bound.arguments.items():
        kind = original_signature.parameters[name].kind
        if kind == inspect.Parameter.VAR_POSITIONAL:
            forwarded_args.extend(value)
        elif kind == inspect.Parameter.VAR_KEYWORD:
            forwarded_kwargs.update(value)
        elif kind == inspect.Parameter.POSITIONAL_ONLY or (
            kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and name in positional_origins
        ):
            forwarded_args.append(value)
        else:
            forwarded_kwargs[name] = value

    return forwarded_args, forwarded_kwargs


def allow_args(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
    """Allow a function to be called with positional arguments.

    In comparison to `allow_only_kwargs`, the `allow_args` decorator does not enforce
    that the function is called only with positional arguments.

    Args:
        func: The function to be wrapped.

    Returns:
        A Callable with the same arguments as func (but with the additional possibility
            to call it with positional arguments).

    """
    try:
        signature = inspect.signature(func)
    except TypeError, ValueError:
        return func
    parameters = signature.parameters

    # Create new signature without keyword-only arguments
    new_parameters = [
        (
            p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            else p
        )
        for p in parameters.values()
    ]
    new_signature = signature.replace(parameters=new_parameters)

    # We cast to F here to signal ty that the return type is the same as the input
    # type. This ignores the change of parameters from positional to keyword-only
    # arguments.
    # TODO(@timmens): Remove this cast once we find an explicit way to specify the
    # change from positional to keyword-only parameter in the function signature
    # https://github.com/opensourceeconomics/pylcm/issues/80.
    return cast(
        "Callable[..., ReturnType]",
        _PositionalAdapter(
            func=func,
            original_signature=signature,
            signature=new_signature,
            accepts_variadic=any(
                parameter.kind
                in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
                for parameter in parameters.values()
            ),
        ),
    )


def publish_signature(
    *, target: Callable[..., ArrayTree], signature: inspect.Signature
) -> None:
    """Set the signature `inspect.signature` reports for `target`.

    A plain function takes the attribute directly; a frozen adapter takes it through
    `object.__setattr__`, which is the one assignment its immutability permits.
    """
    object.__setattr__(target, "__signature__", signature)


def is_user_function[T](value: T | UserFunction) -> TypeIs[UserFunction]:
    """Whether a declared value is a user function rather than a constant or spec.

    Every callable in a model declaration is a user function: its parameters are
    resolved by name and its result is used as the model value it declares.

    Args:
        value: A declared value, such as a weight, a law or a kernel slot.

    Returns:
        Whether `value` is callable.

    """
    return callable(value)


def array_result(value: UserFunctionResult) -> ValueND:
    """Return a user function's result where its role requires an array.

    A utility, a resources map, a constraint or a probability cell is evaluated on
    arrays and returns one. A Python scalar, such as a constant function's value,
    becomes a 0-d array; a mapping or tuple there is a model error.

    Args:
        value: What the user function returned.

    Returns:
        `value` as an array.

    Raises:
        TypeError: If `value` is a mapping or a tuple.

    """
    if isinstance(value, Array):
        return value
    if isinstance(value, float | int):
        return jnp.asarray(value)
    msg = (
        f"A model function returned a {type(value).__name__} where an array is "
        "required."
    )
    raise TypeError(msg)


def get_union_of_args[R](
    list_of_functions: Sequence[Callable[..., R]],
) -> frozenset[str]:
    """Return the union of arguments of a list of functions.

    Args:
        list_of_functions: A list of functions.

    Returns:
        The union of arguments of all functions in list_of_functions.

    """
    arguments = [inspect.signature(f).parameters for f in list_of_functions]
    return frozenset(set().union(*arguments))


def all_as_kwargs[T](
    *,
    args: tuple[T, ...],
    kwargs: dict[ReferenceName, T],
    arg_names: list[ReferenceName],
) -> MappingProxyType[ReferenceName, T]:
    """Return kwargs dictionary containing all arguments.

    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.
        arg_names: Names of arguments.

    Returns:
        A dictionary of all arguments.

    """
    return MappingProxyType(
        dict(zip(arg_names[: len(args)], args, strict=True)) | kwargs
    )


def all_as_args[T](
    *,
    args: tuple[T, ...],
    kwargs: dict[ReferenceName, T],
    arg_names: list[ReferenceName],
) -> tuple[T, ...]:
    """Return args tuple containing all arguments.

    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.
        arg_names: Names of arguments.

    Returns:
        A tuple of all arguments.

    """
    return args + tuple(convert_kwargs_to_args(kwargs=kwargs, arg_names=arg_names))


def convert_kwargs_to_args[T](
    *, kwargs: dict[ReferenceName, T], arg_names: list[ReferenceName]
) -> tuple[T, ...]:
    """Convert kwargs to args in the order of arg_names.

    Args:
        kwargs: Keyword arguments.
        arg_names: List of argument names in the order they should be.

    Returns:
        List of arguments in the order of arg_names.

    """
    unknown = set(kwargs).difference(arg_names)
    if unknown:
        raise ValueError(f"Arguments {sorted(unknown)} are not among {arg_names}.")
    return tuple(kwargs[name] for name in arg_names if name in kwargs)


@dataclass(frozen=True, eq=False)
class _WrappedCallable[R]:
    """Base of the adapters: a callable standing in for `func` under `signature`."""

    func: Callable[..., R]
    """The wrapped function."""
    signature: inspect.Signature
    """The signature `inspect.signature` reports for the adapter."""

    def __post_init__(self) -> None:
        # The attributes `functools.wraps` would copy, set through
        # `object.__setattr__` because the adapter is frozen. The wrapped
        # function's own attributes come first so that a marker another library
        # set on it survives, then the adapter's identity overrides them.
        for name, value in getattr(self.func, "__dict__", {}).items():
            if name not in _PROTECTED_ATTRIBUTES:
                object.__setattr__(self, name, value)
        for name in _WRAPPER_ASSIGNMENTS:
            if hasattr(self.func, name):
                object.__setattr__(self, name, getattr(self.func, name))
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(self, "__signature__", self.signature)
        object.__setattr__(
            self, "__annotations__", dict(type(self).__call__.__annotations__)
        )

    def __call__(self, *args: PytreeValue, **kwargs: PytreeValue) -> R:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class _KeywordOnlyAdapter[R](_WrappedCallable[R]):
    """Forward keyword arguments to `func`, refusing positional ones."""

    signature_parameter_names: tuple[str, ...]
    """Every parameter of `func`, in declaration order."""
    keyword_only_names: tuple[str, ...]
    """The parameters `func` itself takes keyword-only."""
    enforce: bool
    """Whether an argument `func` does not take is an error."""

    def __call__(self, *args: PytreeValue, **kwargs: PytreeValue) -> R:
        if args:
            raise ValueError(
                (
                    "This function has been decorated so that it allows only kwargs, "
                    "but was called with positional arguments."
                ),
            )

        if self.enforce:
            extra = set(kwargs).difference(self.signature_parameter_names)
            if extra:
                raise ValueError(
                    f"Expected arguments: {list(self.signature_parameter_names)}, "
                    f"got extra: {extra}",
                )

        missing = set(self.signature_parameter_names).difference(kwargs)
        if missing:
            raise ValueError(
                f"Expected arguments: {list(self.signature_parameter_names)}, "
                f"missing: {missing}",
            )

        # Retrieve keyword-only arguments
        kw_only_kwargs = {k: kwargs[k] for k in self.keyword_only_names}

        # Get kwargs that must be converted to positional arguments
        positional_kwargs = {
            k: v
            for k, v in kwargs.items()
            if (k not in self.keyword_only_names)
            and (k in self.signature_parameter_names)
        }

        # Collect all positional arguments in correct order
        positional = convert_kwargs_to_args(
            kwargs=positional_kwargs, arg_names=list(self.signature_parameter_names)
        )

        return self.func(*positional, **kw_only_kwargs)


@dataclass(frozen=True, eq=False)
class _PositionalAdapter[R](_WrappedCallable[R]):
    """Accept positional arguments for `func` and forward them as `func` takes them."""

    original_signature: inspect.Signature
    """The signature of `func` itself, which fixes how each value is forwarded."""
    accepts_variadic: bool
    """Whether `func` takes `*args` or `**kwargs`, so no argument count is too many."""

    def __call__(self, *args: PytreeValue, **kwargs: PytreeValue) -> R:
        parameters = self.original_signature.parameters
        if len(args) + len(kwargs) > len(parameters) and not self.accepts_variadic:
            raise ValueError("Too many arguments provided.")

        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError as error:
            if any(
                phrase in str(error)
                for phrase in (
                    "too many positional arguments",
                    "multiple values for argument",
                    "unexpected keyword argument",
                )
            ):
                raise ValueError("Too many arguments provided.") from error
            raise ValueError("Not all arguments provided.") from error

        forwarded_args, forwarded_kwargs = _split_bound_arguments(
            bound=bound,
            n_positional=len(args),
            original_signature=self.original_signature,
            adapted_signature=self.signature,
        )

        return self.func(*forwarded_args, **forwarded_kwargs)


# Attributes of a wrapped function an adapter never inherits: its own fields and
# the introspection attributes the adapter sets itself.
_PROTECTED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "func",
        "signature",
        "signature_parameter_names",
        "keyword_only_names",
        "enforce",
        "original_signature",
        "accepts_variadic",
        "__signature__",
        "__wrapped__",
        "__annotations__",
        "__annotate__",
        "__dict__",
    }
)
