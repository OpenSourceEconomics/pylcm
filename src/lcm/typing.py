from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from jax import Array
from jaxtyping import Bool, Float, Int32, Key, Scalar

from lcm.params import MappingLeaf, UserMappingLeaf
from lcm.params.sequence_leaf import SequenceLeaf, UserSequenceLeaf

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

# JAX PRNG keys (`jax.random`) carry the dedicated `key<fry>` dtype, which
# jaxtyping matches via `Key` — distinct from `FloatND`/`IntND`. Covers both a
# single 0-d key and a batched 1-d array of keys.
type PRNGKeyND = Key[Array, "..."]

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

type RegimeStates = MappingProxyType[StateName, FloatND | IntND]
type StatesPerRegime = MappingProxyType[RegimeName, RegimeStates]

# Boundary form of initial conditions — accepted by `Model.simulate` and
# canonicalized by `canonicalize_initial_conditions`.
type UserInitialConditions = Mapping[
    StateName | Literal["regime_id"], Array | np.ndarray
]

# Post-canonicalization form — emitted by `canonicalize_initial_conditions`
# and consumed by `validate_initial_conditions`, `simulate`, and persistence.
# Read-protocol typing so callers don't have to wrap a dict in
# `MappingProxyType` before passing it in; pylcm producers still wrap on
# the way out to preserve immutability at runtime.
type InitialConditions = Mapping[StateName | Literal["regime_id"], FloatND | IntND]


# Boundary leaf type — accepted by `Model.__init__` / `Model.solve` /
# `Model.simulate` and canonicalized by `cast_params_to_canonical_dtypes`.
type _UserParamsLeaf = (
    bool
    | int
    | float
    | FloatND
    | IntND
    | BoolND
    | np.ndarray
    | pd.Series
    | UserMappingLeaf
    | UserSequenceLeaf
)
type UserParams = Mapping[
    str,
    _UserParamsLeaf | Mapping[str, _UserParamsLeaf | Mapping[str, _UserParamsLeaf]],
]

# Post-canonicalization leaf type — output of
# `cast_params_to_canonical_dtypes`. Only canonical-dtype JAX arrays and
# canonical-narrow `MappingLeaf` / `SequenceLeaf` instances survive.
type _ParamsLeaf = FloatND | IntND | BoolND | MappingLeaf | SequenceLeaf
type Params = Mapping[
    str,
    _ParamsLeaf | Mapping[str, _ParamsLeaf | Mapping[str, _ParamsLeaf]],
]

# Internal regime parameters: A flat mapping with function-qualified names.
# Keys are always function-qualified (e.g., "utility__risk_aversion",
# "H__discount_factor"). Values are canonical-dtype JAX arrays or
# canonical-narrow container leaves.
type FlatRegimeParams = MappingProxyType[
    str, FloatND | IntND | BoolND | MappingLeaf | SequenceLeaf
]
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
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> FloatND | IntND | BoolND: ...


@runtime_checkable
class RegimeTransitionFunction(Protocol):
    """The processed regime transition function for the solve phase.

    Wraps the user's `next_regime` function so its output is a mapping of
    target regime name to a transition-probability array, rather than a
    raw array indexed by regime id.

    Only used for type checking.

    """

    def __call__(
        self,
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> MappingProxyType[RegimeName, FloatND]: ...


@runtime_checkable
class VmappedRegimeTransitionFunction(Protocol):
    """The processed regime transition function for the simulate phase.

    The `vmap`-over-subjects counterpart of `RegimeTransitionFunction`:
    same mapping output, with each probability array carrying a leading
    per-subject axis.

    Only used for type checking.

    """

    def __call__(
        self,
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> MappingProxyType[RegimeName, FloatND]: ...


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
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **kwargs: Any,  # noqa: ANN401
    ) -> FloatND: ...


@runtime_checkable
class ArgmaxQOverAFunction(Protocol):
    """The function that finds the argmax of Q over all actions.

    Q is the state-action value function. The ArgmaxQOverCFunction returns the argmax
    and the maximum of Q over all actions.

    Only used for type checking.

    """

    def __call__(
        self,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[IntND, FloatND]: ...


@runtime_checkable
class StochasticNextFunction(Protocol):
    """The function that simulates the next state of a stochastic variable.

    Only used for type checking.

    """

    def __call__(self, **kwargs: FloatND | IntND) -> FloatND | IntND: ...


@runtime_checkable
class NextStateSimulationFunction(Protocol):
    """The function that computes the next states during the simulation.

    Returns a nested mapping `{target_regime: {next_<state>: array}}`. Only
    used for type checking.

    """

    def __call__(
        self,
        **kwargs: FloatND | IntND | Period | Age | MappingLeaf | SequenceLeaf,
    ) -> MappingProxyType[
        RegimeName, MappingProxyType[str, DiscreteState | ContinuousState]
    ]: ...


@runtime_checkable
class ActiveFunction(Protocol):
    """Function that determines if a regime is active at a given age.

    Only used for type checking.

    """

    def __call__(self, age: float, /) -> bool: ...
