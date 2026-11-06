"""Validated construction of user-facing configuration records.

Configuration records whose fields are plain data validate every field, and
every entry of every container field, when they are constructed. The records
are Pydantic dataclasses in strict mode, with three scalar admission rules
shared across them:

- `CONFIG_FLOAT` admits a builtin `int` or `float` and stores a Python
  `float`. Booleans, strings, NumPy scalars, `Decimal`, `Fraction` and other
  objects implementing `__float__` are refused.
- `EXACT_INT` admits a builtin `int` only. Booleans, floats and `int`
  subclasses are refused.
- `NON_EMPTY_NAME` admits a non-empty `str`, unchanged.

Process-law values do not use `CONFIG_FLOAT`: their numeric kind is part of
their identity.

`translate_validation_errors` maps a record's `pydantic.ValidationError` to a
project exception that names the class, each failing field path and the
violated requirement. A call with positional, missing or unexpected arguments
raises `TypeError` before any field is validated. Package-wide beartype
checking of ordinary functions and methods is unaffected.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Annotated

from pydantic import (
    AllowInfNan,
    BeforeValidator,
    ConfigDict,
    StringConstraints,
    ValidationError,
)
from pydantic_core import PydanticCustomError

# Shared by every validated configuration record: no lax coercion, defaults are
# validated like supplied values, and unknown keywords are refused.
STRICT_CONFIG = ConfigDict(strict=True, validate_default=True, extra="forbid")


def _admit_builtin_real(value: object) -> object:  # noqa: PAN001 - admits arbitrary caller input
    if type(value) is int or type(value) is float:
        return float(value)
    raise PydanticCustomError(
        "builtin_real_type", "Input should be a Python int or float"
    )


def _admit_exact_int(value: object) -> object:  # noqa: PAN001 - admits arbitrary caller input
    if type(value) is int:
        return value
    raise PydanticCustomError("exact_int_type", "Input should be an exact Python int")


# Admit a builtin `int` or `float` and convert it to a Python `float`.
CONFIG_FLOAT = BeforeValidator(_admit_builtin_real)

# Admit a builtin `int`, refusing `bool` and other `int` subclasses.
EXACT_INT = BeforeValidator(_admit_exact_int)

# Refuse NaN and both infinities.
FINITE = AllowInfNan(allow_inf_nan=False)

# A non-empty string name, stored unchanged.
type NonEmptyName = Annotated[str, StringConstraints(min_length=1)]


def translate_validation_errors[T](
    error_type: type[Exception],
) -> Callable[[type[T]], type[T]]:
    """Map a validated record's construction errors to a project exception.

    Apply the returned decorator outermost, on top of the Pydantic dataclass
    decorator. The class object is returned unchanged except for its
    initializer.

    Args:
        error_type: Exception raised for invalid field data. It is constructed
            from one message and chained to the `ValidationError`.

    Returns:
        A class decorator.

    """

    def decorate(cls: type[T]) -> type[T]:
        validating_init = cls.__init__
        signature = inspect.signature(cls)

        @functools.wraps(validating_init, assigned=_ASSIGNED_WITHOUT_ANNOTATIONS)
        def __init__(self: object, *args: object, **kwargs: object) -> None:  # noqa: N807, PAN001 - forwards the record's own call
            signature.bind(*args, **kwargs)
            try:
                validating_init(self, *args, **kwargs)
            except ValidationError as error:
                raise error_type(
                    format_validation_error(class_name=cls.__name__, error=error)
                ) from error

        cls.__init__ = __init__  # ty: ignore[invalid-assignment]
        return cls

    return decorate


def validate_field[T](
    *,
    validate: Callable[[object], T],  # noqa: PAN001 - validates arbitrary caller input
    value: object,  # noqa: PAN001 - arbitrary caller input
    owner: str,
    field_name: str,
    error_type: type[Exception],
) -> T:
    """Validate one field of a record that keeps its own constructor.

    Args:
        validate: `validate_python` of a module-level `TypeAdapter` holding the
            field's schema.
        value: The value the caller supplied.
        owner: Name of the record, for the error message.
        field_name: Name of the field, for the error message.
        error_type: Exception raised for an invalid value, chained to the
            `ValidationError`.

    Returns:
        The validated and normalized value.

    """
    try:
        return validate(value)
    except ValidationError as error:
        raise error_type(
            format_validation_error(class_name=owner, error=error, prefix=(field_name,))
        ) from error


def format_validation_error(
    *, class_name: str, error: ValidationError, prefix: tuple[str, ...] = ()
) -> str:
    """Render each failing field path of `error` with its violated requirement.

    Args:
        class_name: Name of the record that refused its input.
        error: The Pydantic validation error.
        prefix: Path of the validated value inside the record, when the error
            comes from validating one field on its own.

    Returns:
        One line per failure, `<class>.<path>: <requirement> (got <input>)`.

    """
    lines = [f"Invalid {class_name}:"]
    for detail in error.errors(include_url=False):
        if detail["type"] == "value_error":
            # A relation check names its own fields in its message.
            lines.append(f"  {detail['ctx']['error']}")
            continue
        path = "".join(
            f"[{part!r}]" if isinstance(part, int) or index > 0 else f".{part}"
            for index, part in enumerate((*prefix, *detail["loc"]))
        )
        lines.append(f"  {class_name}{path}: {detail['msg']} (got {detail['input']!r})")
    return "\n".join(lines)


_ASSIGNED_WITHOUT_ANNOTATIONS = tuple(
    name
    for name in functools.WRAPPER_ASSIGNMENTS
    if name not in ("__annotations__", "__annotate__")
)
