from typing import Any, Protocol

from jax import Array
from jaxtyping import Bool, Float, Int, Scalar

type ContinuousState = Float[Array, "..."]
type ContinuousAction = Float[Array, "..."]
type DiscreteState = Int[Array, "..."]
type DiscreteAction = Int[Array, "..."]

type FloatND = Float[Array, "..."]
type IntND = Int[Array, "..."]
type BoolND = Bool[Array, "..."]

type Float1D = Float[Array, "_"]  # noqa: F821
type Int1D = Int[Array, "_"]  # noqa: F821
type Bool1D = Bool[Array, "_"]  # noqa: F821

# Many JAX functions are designed to work with scalar numerical values. This also
# includes zero dimensional jax arrays.
type ScalarInt = int | Int[Scalar, ""]  # noqa: F722
type ScalarFloat = float | Float[Scalar, ""]  # noqa: F722


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
        self, *args: Array | int, params: ParamsDict, **kwargs: Array | int
    ) -> Array: ...


class QAndFFunction(Protocol):
    """The function that computes Q and F.

    Q is the state-action value function. F is a boolean array that indicates whether
    the state-action pair is feasible.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, next_V_arr: Array, params: ParamsDict, **kwargs: Array
    ) -> tuple[FloatND, BoolND]: ...


class MaxQOverCFunction(Protocol):
    """The function that maximizes Q over the continuous actions.

    Q is the state-action value function. The MaxQOverCFunction returns the maximum of Q
    over the continuous actions.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, next_V_arr: Array, params: ParamsDict, **kwargs: Array
    ) -> Array: ...


class ArgmaxQOverCFunction(Protocol):
    """The function that finds the argmax of Q over the continuous actions.

    Q is the state-action value function. The ArgmaxQOverCFunction returns the argmax
    and the maximum of Q over the continuous actions.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, next_V_arr: Array, params: ParamsDict, **kwargs: Array
    ) -> tuple[Array, Array]: ...


class MaxQOverAFunction(Protocol):
    """The function that maximizes Q over all actions.

    Q is the state-action value function. The MaxQOverCFunction returns the maximum of Q
    over all actions.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, next_V_arr: Array, params: ParamsDict, **kwargs: Array
    ) -> Array: ...


class ArgmaxQOverAFunction(Protocol):
    """The function that finds the argmax of Q over all actions.

    Q is the state-action value function. The ArgmaxQOverCFunction returns the argmax
    and the maximum of Q over all actions.

    Only used for type checking.

    """

    def __call__(  # noqa: D102
        self, next_V_arr: Array, params: ParamsDict, **kwargs: Array
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

    def __call__(self, **kwargs: Array) -> Array: ...  # noqa: D102
