from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from jax import Array
from jaxtyping import Bool, Float, Int32, Key, Scalar

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
type ProcessName = str
type FunctionName = str
type TransitionFunctionName = str
type RegimeNamesToIds = MappingProxyType[RegimeName, ScalarInt]
type RegimeIdsToNames = MappingProxyType[int, RegimeName]

type EconFunctionsMapping = MappingProxyType[FunctionName, EconFunction]
type ConstraintFunctionsMapping = MappingProxyType[FunctionName, ConstraintFunction]

type TransitionFunctionsMapping = MappingProxyType[
    RegimeName, MappingProxyType[TransitionFunctionName, TransitionFunction]
]

type RegimeStates = MappingProxyType[StateName, Float1D | Int1D]
type StatesPerRegime = MappingProxyType[RegimeName, RegimeStates]

# Post-canonicalization form — emitted by `canonicalize_initial_conditions`
# and consumed by `validate_initial_conditions`, `simulate`, and persistence.
# Read-protocol typing so callers don't have to wrap a dict in
# `MappingProxyType` before passing it in; pylcm producers still wrap on
# the way out to preserve immutability at runtime. Values are 1-D arrays
# of length `n_subjects`; the validator checks the rank-1 invariant.
type InitialConditions = Mapping[StateName | Literal["regime_id"], Float1D | Int1D]


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
type FlatParams = MappingProxyType[RegimeName, FlatRegimeParams]

# Immutable templates, used internally
type RegimeParamsTemplate = MappingProxyType[FunctionName, MappingProxyType[str, str]]
type ParamsTemplate = MappingProxyType[RegimeName, RegimeParamsTemplate]

# Type aliases for value function arrays
type PeriodToRegimeToVArr = MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]


@runtime_checkable
class EconFunction(Protocol):
    """A numeric model function after processing into the engine signature.

    Covers the *value-side* user-supplied content of a regime: the period
    utility, the Bellman aggregator `H`, and any helper / DAG functions
    whose output is consumed by them. Returns a numeric array
    (`FloatND` or `IntND`). Feasibility predicates live in
    `ConstraintFunction`; state / regime / process transitions live in
    `TransitionFunction`.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> FloatND | IntND: ...


@runtime_checkable
class ConstraintFunction(Protocol):
    """A feasibility predicate over (state, action, params).

    Returns a boolean array indicating whether each grid point is
    feasible. Stored on `Regime.constraints` and combined into the
    `F` array of `Q_and_F`.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> BoolND: ...


@runtime_checkable
class TransitionFunction(Protocol):
    """A state / regime / process transition function.

    Stored on `Regime.transition` (regime transition), in
    `Regime.state_transitions` (per-state, plus per-target dicts),
    and as the auto-generated stubs for process-derived transitions.
    Returns the deterministic next-period value (`IntND` / `FloatND`)
    or, for stochastic / weight functions, the corresponding numeric
    array (probability mass, weight, etc.).

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> FloatND | IntND: ...


@runtime_checkable
class RegimeTransitionFunction(Protocol):
    """The processed regime transition function for the solve phase.

    Wraps the user's `next_regime` function so its output is a mapping of
    target regime name to a transition-probability array, rather than a
    raw array indexed by regime id.

    Used for both type checking and beartype runtime checks.

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

    Used for both type checking and beartype runtime checks.

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

    Used for both type checking and beartype runtime checks.

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

    Used for both type checking and beartype runtime checks.

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

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[IntND, FloatND]: ...


@runtime_checkable
class StochasticNextFunction(Protocol):
    """The function that simulates the next state of a stochastic variable.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(self, **kwargs: FloatND | IntND) -> FloatND | IntND: ...


@runtime_checkable
class NextStateSimulationFunction(Protocol):
    """The function that computes the next states during the simulation.

    Returns a nested mapping `{target_regime: {next_<state>: array}}`. Used for
    both type checking and beartype runtime checks.

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

    Used for both type checking and beartype runtime checks.

    """

    def __call__(self, age: Any, /) -> bool: ...  # noqa: ANN401


# Backwards-compatibility shim: User* aliases live in `lcm.api.typing`.
# Re-exported here so existing `from lcm.typing import UserParams` etc.
# keep working. Placed at the bottom of the module so the core type
# names imported by `lcm.api.typing` are already defined.
from lcm.api.typing import (  # noqa: E402, F401
    UserAge,
    UserFacingParamsTemplate,
    UserFunction,
    UserInitialConditions,
    UserParams,
)
