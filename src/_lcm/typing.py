"""Engine-internal type aliases and protocols.

Compound mapping aliases, canonical post-processing forms, and the structural
`Protocol` classes used for type checking and beartype runtime checks. The
string-label aliases and the other user-facing aliases live in `lcm.typing`;
they are re-exported here so engine-internal code can import everything from
`_lcm.typing`.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from jax import Array
from jaxtyping import Key

from _lcm.egm.carry import EgmCarry
from _lcm.params.mapping_leaf import MappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf

# String-label and array aliases are defined in `lcm.typing`. The `noqa`-marked
# labels are re-exported here — not referenced in this module's body — so
# `from _lcm.typing import ActionName` etc. keeps working engine-wide.
from lcm.typing import (
    ActionName,  # noqa: F401
    Age,
    BoolND,
    ContinuousState,
    DiscreteState,
    Float1D,
    FloatND,
    FunctionName,
    Int1D,
    IntND,
    Period,
    ProcessName,  # noqa: F401
    RegimeName,
    ScalarInt,
    StateName,
    StateOrActionName,  # noqa: F401
    TransitionFunctionName,
)

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

# JAX PRNG keys (`jax.random`) carry the dedicated `key<fry>` dtype, which
# jaxtyping matches via `Key` — distinct from `FloatND`/`IntND`. Covers both a
# single 0-d key and a batched 1-d array of keys.
type PRNGKeyND = Key[Array, "..."]


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
class EgmStepFunction(Protocol):
    """The per-period DC-EGM kernel for one regime.

    Consumes the regime's exogenous state grids, the rolling value-function
    and EGM-carry mappings, and the regime's flat params; returns the
    regime's value-function array on the exogenous state grid together with
    the carry its parents interpolate.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EgmCarry],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, EgmCarry]: ...


@runtime_checkable
class EgmCarryProducer(Protocol):
    """Closed-form carry producer for a terminal regime.

    Maps the regime's solved value-function array (plus its state grids and
    flat params) to the EGM carry a DC-EGM parent interpolates.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *,
        V_arr: FloatND,
        **kwargs: Any,  # noqa: ANN401
    ) -> EgmCarry: ...


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
