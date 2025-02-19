from enum import Enum
from typing import Any, Protocol

from jax import Array

# Many JAX functions are designed to work with scalar numerical values. This also
# includes zero dimensional jax arrays.
Scalar = int | float | Array


ParamsDict = dict[str, Any]


class UserFunction(Protocol):
    """A function provided by the user.

    Only used for type checking.

    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D102


class InternalUserFunction(Protocol):
    """The internal representation of a function provided by the user.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, *args: Scalar, params: ParamsDict, **kwargs: Scalar
    ) -> Scalar: ...


class DiscreteProblemSolverFunction(Protocol):
    """The function that solves the discrete problem.

    Only used for type checking.

    """

    def __call__(self, values: Array, params: ParamsDict) -> Array: ...  # noqa: D102


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None


class Target(Enum):
    """Target of the function."""

    SOLVE = "solve"
    SIMULATE = "simulate"
