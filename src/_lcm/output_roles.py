"""The role of a user function's output, and the one wrapper that converts it.

Every user function whose output the engine consumes is wrapped once, where it is
registered, in the role its consumer needs: a float, a Boolean or a discrete code.
The wrapper converts the output with the matching adapter from `_lcm.dtypes`,
which reads only the dtype, never a traced value, so engine code receives
functions that already return their role's kind.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import cast, no_type_check

from dags import get_annotations, get_free_arguments

from _lcm.dtypes import (
    OutputSource,
    UserOutput,
    as_bool,
    as_canonical_float,
    as_code,
    name_of,
)
from _lcm.grids import Grid
from _lcm.typing import PytreeValue
from _lcm.utils.functools import _WrappedCallable
from lcm.grids import DiscreteGrid
from lcm.typing import ValueND

# What a registered user function returns before it is put in role: a user
# output, or any pytree value a composed DAG node passes through.
type RawOutput = UserOutput | PytreeValue


class OutputRole(Enum):
    """The kind of value the engine consumes from a user function."""

    FLOAT = "float"
    BOOL = "bool"
    CODE = "code"


_ADAPTERS = {
    OutputRole.FLOAT: as_canonical_float,
    OutputRole.BOOL: as_bool,
    OutputRole.CODE: as_code,
}


def target_role(grid: Grid) -> OutputRole:
    """Return the role of a value that lands on a state with this grid.

    A discrete grid's state is stored as a code, whatever values the grid names;
    every other state is a float.
    """
    return OutputRole.CODE if isinstance(grid, DiscreteGrid) else OutputRole.FLOAT


@dataclass(frozen=True, eq=False, kw_only=True)
class InRole(_WrappedCallable[RawOutput]):
    """A user function whose output is converted to its role's kind.

    A frozen callable instance rather than a nested function: the beartype claw
    memoizes every function it decorates, so a function defined per model build
    would pin that build's user function after the model is dropped.
    """

    role: OutputRole
    """The kind the output is converted to."""
    source: OutputSource
    """How an output-kind error names the function, e.g. "Law 'next_wealth'"."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Consumers read the user's annotations; `__call__` is not type-checked,
        # so the beartype claw never enforces user-model types on the wrapper.
        object.__setattr__(self, "__annotations__", get_annotations(self.func))

    @no_type_check
    def __call__(self, *args: PytreeValue, **kwargs: PytreeValue) -> ValueND:
        value = cast("UserOutput", self.func(*args, **kwargs))
        return _ADAPTERS[self.role](value=value, source=self.source)


def in_role[F: Callable[..., RawOutput]](*, func: F, role: OutputRole, kind: str) -> F:
    """Return `func` with its output converted to `role`'s kind.

    Args:
        func: The user function, as registered.
        role: The kind the engine consumes from it.
        kind: What the function is, for error messages, e.g. "Law" or "Gate".

    Returns:
        A callable with `func`'s signature and name. A function already in this
        role is returned unchanged.

    Raises:
        TypeError: If `func` is already wrapped in another role.

    """
    if isinstance(func, InRole):
        if func.role is not role:
            msg = f"{func.source} is already in role {func.role}, not {role}."
            raise TypeError(msg)
        return func
    # A `partial`'s bound keywords stay in its signature with defaults; the DAG
    # reads only its free arguments, so the wrapper publishes only those.
    signature = inspect.signature(func)
    free = set(get_free_arguments(func))
    return InRole(  # ty: ignore[invalid-return-type]
        func=func,
        signature=signature.replace(
            parameters=[p for p in signature.parameters.values() if p.name in free]
        ),
        role=role,
        source=f"{kind} '{name_of(func)}'",
    )
