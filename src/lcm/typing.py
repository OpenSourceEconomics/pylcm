from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from jax import Array
from jaxtyping import Bool, Float, Int32, Scalar

from lcm.params import MappingLeaf
from lcm.params.sequence_leaf import SequenceLeaf

type ContinuousState = Float[Array, "..."]
type ContinuousAction = Float[Array, "..."]
type DiscreteState = Int32[Array, "..."]
type DiscreteAction = Int32[Array, "..."]

type FloatND = Float[Array, "..."]
type IntND = Int32[Array, "..."]
type BoolND = Bool[Array, "..."]

type Float1D = Float[Array, "_"]  # noqa: F821
type Int1D = Int32[Array, "_"]  # noqa: F821
type Bool1D = Bool[Array, "_"]  # noqa: F821

# Zero-dimensional JAX scalars — pylcm's canonical scalar form post boundary cast.
type ScalarInt = Int32[Scalar, ""]
type ScalarFloat = Float[Scalar, ""]
type ScalarBool = Bool[Scalar, ""]

type Period = ScalarInt
type Age = ScalarInt | ScalarFloat
type RegimeName = str
type StateName = str
type ActionName = str
type StateOrActionName = str
type ShockName = str
type FunctionName = str
type TransitionFunctionName = str
type RegimeNamesToIds = MappingProxyType[RegimeName, ScalarInt]
type RegimeIdsToNames = MappingProxyType[int, RegimeName]

type FunctionsMapping = MappingProxyType[FunctionName, InternalUserFunction]

type TransitionFunctionsMapping = MappingProxyType[
    RegimeName, MappingProxyType[TransitionFunctionName, InternalUserFunction]
]

type RegimeStates = MappingProxyType[StateName, Array]
type StatesPerRegime = MappingProxyType[RegimeName, RegimeStates]


type _ParamsLeaf = bool | float | Array | pd.Series | MappingLeaf | SequenceLeaf
type UserParams = Mapping[
    str,
    _ParamsLeaf | Mapping[str, _ParamsLeaf | Mapping[str, _ParamsLeaf]],
]

# Internal regime parameters: A flat mapping with function-qualified names.
# Keys are always function-qualified (e.g., "utility__risk_aversion",
# "H__discount_factor"). Values are scalars or arrays.
type FlatRegimeParams = MappingProxyType[str, Array]
type InternalParams = MappingProxyType[RegimeName, FlatRegimeParams]

# Immutable templates, used internally
type RegimeParamsTemplate = MappingProxyType[FunctionName, MappingProxyType[str, str]]
type ParamsTemplate = MappingProxyType[RegimeName, RegimeParamsTemplate]

# User-facing template; types rendered as strings.
type UserFacingParamsTemplate = dict[RegimeName, dict[FunctionName, dict[str, str]]]

# Type aliases for value function arrays
type PeriodToRegimeToVArr = MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]


@runtime_checkable
class UserFunction(Protocol):
    """A function provided by the user.

    Used for both type checking and beartype runtime checks on perimeter
    constructors. Any callable satisfies this protocol structurally.

    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401


@runtime_checkable
class InternalUserFunction(Protocol):
    """The internal representation of a function provided by the user.

    Only used for type checking.

    """

    def __call__(
        self,
        *args: Array | float,
        **kwargs: Array | float,
    ) -> Array: ...


@runtime_checkable
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


@runtime_checkable
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


@runtime_checkable
class QAndFFunction(Protocol):
    """The function that computes Q and F.

    Q is the state-action value function. F is a boolean array that indicates whether
    the state-action pair is feasible.

    Only used for type checking.

    """

    def __call__(
        self,
        next_regime_to_V_arr: FloatND,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, BoolND]: ...


@runtime_checkable
class MaxQOverAFunction(Protocol):
    """The function that maximizes Q over all actions.

    Q is the state-action value function. The MaxQOverCFunction returns the maximum of Q
    over all actions.

    Only used for type checking.

    """

    def __call__(
        self,
        next_regime_to_V_arr: MappingProxyType[RegimeName, Array],
        **kwargs: Any,  # noqa: ANN401
    ) -> Array: ...


@runtime_checkable
class ArgmaxQOverAFunction(Protocol):
    """The function that finds the argmax of Q over all actions.

    Q is the state-action value function. The ArgmaxQOverCFunction returns the argmax
    and the maximum of Q over all actions.

    Only used for type checking.

    """

    def __call__(
        self,
        next_regime_to_V_arr: MappingProxyType[RegimeName, Array],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[Array, Array]: ...


@runtime_checkable
class StochasticNextFunction(Protocol):
    """The function that simulates the next state of a stochastic variable.

    Only used for type checking.

    """

    def __call__(self, **kwargs: Array) -> Array: ...


@runtime_checkable
class NextStateSimulationFunction(Protocol):
    """The function that computes the next states during the simulation.

    Returns a nested mapping `{target_regime: {next_<state>: array}}`. Only
    used for type checking.

    """

    def __call__(
        self,
        **kwargs: Array | Period | Age,
    ) -> MappingProxyType[
        RegimeName, MappingProxyType[str, DiscreteState | ContinuousState]
    ]: ...


@runtime_checkable
class ActiveFunction(Protocol):
    """Function that determines if a regime is active at a given age.

    Only used for type checking.

    """

    def __call__(self, age: float, /) -> bool: ...
