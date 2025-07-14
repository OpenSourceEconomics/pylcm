from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

from jax import Array
from jaxtyping import Bool, Float, Int, Scalar

T = TypeVar("T")


class Param(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value


type ContinuousState = Float[Array, "..."]
type ContinuousAction = Float[Array, "..."]
type DiscreteState = Int[Array, "..."]
type DiscreteAction = Int[Array, "..."]
type DerivedFloat = Float[Array, "..."]
type DerivedInt = Int[Array, "..."]
type DerivedBool = Bool[Array, "..."]

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


class MaxQOverCFunction(Protocol):
    """The function that maximizes Q over the continuous actions.

    Q is the state-action value function. The MaxQOverCFunction returns the maximum of Q
    over the continuous actions.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, next_V_arr: Array, params: ParamsDict, **kwargs: Scalar
    ) -> Array: ...


class ArgmaxQOverCFunction(Protocol):
    """The function that finds the argmax of Q over the continuous actions.

    Q is the state-action value function. The ArgmaxQOverCFunction returns the argmax
    and the maximum of Q over the continuous actions.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, next_V_arr: Array, params: ParamsDict, **kwargs: Scalar
    ) -> tuple[Array, Array]: ...


class MaxQcOverDFunction(Protocol):
    """The function that maximizes Qc over the discrete actions.

    Qc is the maximum of the state-action value function (Q) over the continuous
    actions, conditional on the discrete action. It depends on a state and the discrete
    actions. The MaxQcFunction returns the maximum of Qc over the discrete actions.

    Only used for type checking.

    """

    def __call__(self, Qc_arr: Array, params: ParamsDict) -> Array: ...  # noqa: D102


class ArgmaxQcOverDFunction(Protocol):
    """The function that finds the argmax of Qc over the discrete actions.

    Qc is the maximum of the state-action value function (Q) over the continuous
    actions, conditional on the discrete action. It depends on a state and the discrete
    actions. The ArgmaxQcFunction returns the argmax of Qc over the discrete actions.

    Only used for type checking.

    """

    def __call__(self, Qc_arr: Array, params: ParamsDict) -> tuple[Array, Array]: ...  # noqa: D102


class StochasticNextFunction(Protocol):
    """The function that simulates the next state of a stochastic variable.

    Only used for type checking.

    """

    def __call__(self, keys: dict[str, Array], **kwargs: Array) -> Array: ...  # noqa: D102


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None


class Target(Enum):
    """Target of the function."""

    SOLVE = "solve"
    SIMULATE = "simulate"
