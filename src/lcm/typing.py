from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol

from jax import Array
from jaxtyping import Bool, Float, Int, Scalar

from lcm.nested_mapping_params import NestedMappingParams

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

type Period = int | Int1D
type Age = float
type RegimeName = str
type RegimeNamesToIds = MappingProxyType[RegimeName, int]

type GridsDict = MappingProxyType[RegimeName, MappingProxyType[str, Array]]

type TransitionFunctionsMapping = MappingProxyType[
    RegimeName, MappingProxyType[str, InternalUserFunction]
]


type UserParams = Mapping[
    str,
    bool
    | float
    | Array
    | NestedMappingParams
    | Mapping[
        str,
        bool
        | float
        | Array
        | NestedMappingParams
        | Mapping[str, bool | float | Array | NestedMappingParams],
    ],
]

# Internal regime parameters: A flat mapping with function-qualified names.
# Keys are always function-qualified (e.g., "utility__risk_aversion",
# "H__discount_factor"). Values are scalars or arrays.
type FlatRegimeParams = MappingProxyType[str, bool | float | Array]
type InternalParams = MappingProxyType[RegimeName, FlatRegimeParams]

# Immutable templates, used internally
type RegimeParamsTemplate = MappingProxyType[str, MappingProxyType[str, type]]
type ParamsTemplate = MappingProxyType[RegimeName, RegimeParamsTemplate]

# Dictionary-templates; returned to users.
type MutableRegimeParamsTemplate = dict[str, dict[str, type]]
type MutableParamsTemplate = dict[RegimeName, MutableRegimeParamsTemplate]

# Type aliases for value function arrays
type VArrMapping = MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]


class UserFunction(Protocol):
    """A function provided by the user.

    Only used for type checking.

    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401


class InternalUserFunction(Protocol):
    """The internal representation of a function provided by the user.

    Only used for type checking.

    """

    def __call__(
        self,
        *args: Array | float,
        **kwargs: Array | float,
    ) -> Array: ...


class RegimeTransitionFunction(Protocol):
    """The regime transition function provided by the user.

    Returns an array of transition probabilities indexed by regime ID.

    Only used for type checking.

    """

    def __call__(
        self,
        *args: Array | float,
        **kwargs: Array | float,
    ) -> Float1D: ...


class VmappedRegimeTransitionFunction(Protocol):
    """The vmapped regime transition function.

    Returns a 2D array of transition probabilities with shape [n_regimes, n_subjects].

    Only used for type checking.

    """

    def __call__(
        self,
        *args: Array | float,
        **kwargs: Array | float,
    ) -> FloatND: ...


class QAndFFunction(Protocol):
    """The function that computes Q and F.

    Q is the state-action value function. F is a boolean array that indicates whether
    the state-action pair is feasible.

    Only used for type checking.

    """

    def __call__(
        self,
        next_V_arr: FloatND,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, BoolND]: ...


class MaxQOverCFunction(Protocol):
    """The function that maximizes Q over the continuous actions.

    Q is the state-action value function. The MaxQOverCFunction returns the maximum of Q
    over the continuous actions.

    Only used for type checking.

    """

    def __call__(
        self,
        next_V_arr: MappingProxyType[RegimeName, Array],
        period: Period,
        **kwargs: Array,
    ) -> Array: ...


class ArgmaxQOverCFunction(Protocol):
    """The function that finds the argmax of Q over the continuous actions.

    Q is the state-action value function. The ArgmaxQOverCFunction returns the argmax
    and the maximum of Q over the continuous actions.

    Only used for type checking.

    """

    def __call__(
        self,
        next_V_arr: MappingProxyType[RegimeName, Array],
        period: Period,
        **kwargs: Array,
    ) -> tuple[Array, Array]: ...


class MaxQOverAFunction(Protocol):
    """The function that maximizes Q over all actions.

    Q is the state-action value function. The MaxQOverCFunction returns the maximum of Q
    over all actions.

    Only used for type checking.

    """

    def __call__(
        self,
        next_V_arr: MappingProxyType[RegimeName, Array],
        **kwargs: Any,  # noqa: ANN401
    ) -> Array: ...


class ArgmaxQOverAFunction(Protocol):
    """The function that finds the argmax of Q over all actions.

    Q is the state-action value function. The ArgmaxQOverCFunction returns the argmax
    and the maximum of Q over all actions.

    Only used for type checking.

    """

    def __call__(
        self,
        next_V_arr: MappingProxyType[RegimeName, Array],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[Array, Array]: ...


class MaxQcOverDFunction(Protocol):
    """The function that maximizes Qc over the discrete actions.

    Qc is the maximum of the state-action value function (Q) over the continuous
    actions, conditional on the discrete action. It depends on a state and the discrete
    actions. The MaxQcFunction returns the maximum of Qc over the discrete actions.

    Only used for type checking.

    """

    def __call__(self, Qc_arr: Array, **kwargs: Array) -> Array: ...


class ArgmaxQcOverDFunction(Protocol):
    """The function that finds the argmax of Qc over the discrete actions.

    Qc is the maximum of the state-action value function (Q) over the continuous
    actions, conditional on the discrete action. It depends on a state and the discrete
    actions. The ArgmaxQcFunction returns the argmax of Qc over the discrete actions.

    Only used for type checking.

    """

    def __call__(self, Qc_arr: Array, **kwargs: Array) -> tuple[Array, Array]: ...


class StochasticNextFunction(Protocol):
    """The function that simulates the next state of a stochastic variable.

    Only used for type checking.

    """

    def __call__(self, **kwargs: Array) -> Array: ...


class NextStateSimulationFunction(Protocol):
    """The function that computes the next states during the simulation.

    Only used for type checking.

    """

    def __call__(
        self,
        **kwargs: Array | Period | Age,
    ) -> MappingProxyType[str, DiscreteState | ContinuousState]: ...


class ActiveFunction(Protocol):
    """Function that determines if a regime is active at a given age.

    Only used for type checking.

    """

    def __call__(self, age: float, /) -> bool: ...
