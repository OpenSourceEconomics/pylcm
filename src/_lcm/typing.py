"""Engine-internal type aliases and protocols.

Compound mapping aliases, canonical post-processing forms, and the structural
`Protocol` classes used for type checking and beartype runtime checks. The
string-label aliases and the other user-facing aliases live in `lcm.typing`;
they are re-exported here so engine-internal code can import everything from
`_lcm.typing`.
"""

import types
from collections.abc import Mapping, Sequence
from dataclasses import Field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    TypeAliasType,
    runtime_checkable,
)

import jax
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Int, Key

from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.published_policy import (
    EGMSimPolicy,
    NBEGMGridPolicy,
    NNBEGMSimPolicy,
)
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
    ReferenceName,
    RegimeName,
    ScalarInt,
    StateName,
    StateOrActionName,  # noqa: F401
    TransitionFunctionName,
    ValueND,
)

# A `__`-joined path through the params or function namespace, such as
# `"utility__risk_aversion"`. Flat params are keyed by these names, while a
# `ParameterName` names one parameter on its own.
type QualifiedName = str

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
type InitialConditions = Mapping[
    StateName | Literal["regime_id", "own_stakeholder"], Float1D | Int1D
]

# JAX PRNG keys (`jax.random`) carry the dedicated `key<fry>` dtype, which
# jaxtyping matches via `Key` — distinct from `FloatND`/`IntND`. Covers both a
# single 0-d key and a batched 1-d array of keys.
type PRNGKeyND = Key[Array, "..."]


# Post-canonicalization leaf type — output of
# `cast_params_to_canonical_dtypes`. Only canonical-dtype JAX arrays and
# canonical-narrow `MappingLeaf` / `SequenceLeaf` instances survive.
type ParamsLeaf = FloatND | IntND | BoolND | MappingLeaf | SequenceLeaf

# One argument of a user economic function, exactly as `EconFunction.__call__`
# accepts it. Named so a call site binding such arguments can say so. Integer
# arrays may have any width: with x64 enabled, states, actions and indices are
# int64.
type EconFunctionArg = (
    FloatND | Int[Array, "..."] | BoolND | float | MappingLeaf | SequenceLeaf
)

# The arguments of a user economic function, keyed by the names they reference.
type EconFunctionKwargs = Mapping[ReferenceName, EconFunctionArg]

# One argument of a generated `QAndFFunction` or `MaxQOverAFunction`: an argument
# of a user function, or a per-regime mapping of value arrays or of flat params.
type QAndFArg = (
    EconFunctionArg
    | Mapping[RegimeName, FloatND]
    | Mapping[RegimeName, Mapping[QualifiedName, ParamsLeaf]]
)
type QAndFKwargs = Mapping[ReferenceName, QAndFArg]

# A value tree as the engine hands it to JAX: arrays and Python scalars at the
# leaves, nested in tuples, lists and string-keyed mappings. The beartype claw
# checks the outer levels of a recursive alias; ty checks every level.
type ArrayTree = (
    ValueND
    | bool
    | int
    | float
    | tuple[ArrayTree, ...]
    | list[ArrayTree]
    | Mapping[str, ArrayTree]
    | None
)

# The abstract counterpart of an `ArrayTree`, for lowering and memory profiling.
type ShapeDtypeTree = (
    jax.ShapeDtypeStruct
    | tuple[ShapeDtypeTree, ...]
    | list[ShapeDtypeTree]
    | Mapping[str, ShapeDtypeTree]
    | None
)

# A value tree as JAX flattens it, wider than `ArrayTree` at both ends:
# - a node may also be a registered pytree class: a params leaf, or a dataclass such
#   as `EGMCarry` or a published policy, which the claw checks as a whole;
# - a leaf may also be a host NumPy array or scalar.
# Every `ArrayTree` is a `PytreeValue`; naming it as a member keeps that true for
# ty, which would otherwise hold `list[ArrayTree]` apart from `list[PytreeValue]`.
type PytreeValue = (
    ArrayTree
    | ValueND
    | HostArray
    | np.generic
    | bool
    | int
    | float
    | MappingLeaf
    | SequenceLeaf
    | DataclassInstance
    | _PluginPytree
    | tuple[PytreeValue, ...]
    | list[PytreeValue]
    | Mapping[str, PytreeValue]
    | None
)

# A value tree keyed by period at its top levels, as the solve and the simulation
# hold their per-period inputs, outputs and intermediates. Period levels sit only at
# the top or under other period levels; a site whose period levels sit below a name
# level or inside a tuple spells that outer level, as in
# `Mapping[RegimeName, PytreeByPeriod]`. One self-reference keeps the alias
# checkable by the beartype claw.
type PytreeByPeriod = PytreeValue | Mapping[int, PytreeByPeriod]

# The abstract counterpart of a `PytreeValue`, for lowering and memory profiling.
type ShapeDtypePytree = (
    ShapeDtypeTree
    | jax.ShapeDtypeStruct
    | MappingLeaf
    | SequenceLeaf
    | DataclassInstance
    | tuple[ShapeDtypePytree, ...]
    | list[ShapeDtypePytree]
    | Mapping[str, ShapeDtypePytree]
    | None
)

if TYPE_CHECKING:
    from _lcm.solution.solver_diagnostics import SolverDiagnostics
    from lcm._solver_api.replay import ContinuationArtifact

    # A payload a solver publishes under an artifact key: a simulation policy, a
    # continuation artifact, solver diagnostics, or a value tree.
    type ArtifactPayload = (
        SimulationPolicy | ContinuationArtifact | SolverDiagnostics | PytreeValue
    )
    # A solver plugin's continuation artifact, which a value tree may carry.
    type _PluginPytree = ContinuationArtifact
else:
    # Solver plugins publish payload classes of their own; the claw checks nothing
    # here rather than run `isinstance` against a protocol on plugin objects.
    type ArtifactPayload = object
    # A plugin artifact inside a value tree is checked as the registered
    # dataclass it is, never against the protocol.
    type _PluginPytree = DataclassInstance

if TYPE_CHECKING:
    from lcm._solver_api.authority import _ArtifactLeafToken

    # A child handed to a registered pytree's `unflatten`:
    # - a concrete or traced array, or a host array or scalar, during calls;
    # - a `jax.ShapeDtypeStruct` or a `jax.stages.ArgInfo` during AOT lowering;
    # - a `jax.sharding.Sharding` when JAX lays shardings out like the tree;
    # - the artifact authority's opaque token when it compiles a payload template;
    # - `None` for an absent leaf.
    type PytreeChild = (
        ValueND
        | HostArray
        | np.generic
        | jax.ShapeDtypeStruct
        | jax.stages.ArgInfo
        | jax.sharding.Sharding
        | _ArtifactLeafToken
        | bool
        | int
        | float
        | None
    )
else:
    # JAX also unflattens with placeholder leaves of its own (`PytreeLeaf` proxies,
    # `object()` sentinels), so the claw accepts any child.
    type PytreeChild = object

# A runtime annotation object: a class, a `type` alias, a subscripted generic, an
# `X | Y` union, or a string forward reference.
type AnnotationForm = type | TypeAliasType | types.GenericAlias | types.UnionType | str  # noqa: PAN006 - an annotation may name any class

# Shardings laid out like the tree they place.
type ShardingTree = (
    jax.sharding.Sharding
    | tuple[ShardingTree, ...]
    | list[ShardingTree]
    | Mapping[str, ShardingTree]
    | None
)

# A copied lowering descriptor: strings, integers, Booleans, bytes and `None` at
# the leaves, nested in tuples, frozensets and read-only mappings. It retains no
# live payload.
type LoweringDescriptor = (
    str
    | int
    | bool
    | bytes
    | tuple[LoweringDescriptor, ...]
    | frozenset[LoweringDescriptor]
    | MappingProxyType[LoweringDescriptor, LoweringDescriptor]
    | None
)

# A value that `json.dumps` writes and `json.loads` reads back.
type JSONValue = (
    bool | int | float | str | Sequence[JSONValue] | Mapping[str, JSONValue] | None
)

# A host NumPy operand, as opposed to a device `jax.Array`.
type HostArray = npt.NDArray[np.generic]


@runtime_checkable
class DataclassInstance(Protocol):
    """An instance of any dataclass."""

    __dataclass_fields__: ClassVar[dict[str, Field[object]]]  # noqa: PAN001 - a dataclass field may hold any value


type Params = Mapping[
    str,
    ParamsLeaf | Mapping[str, ParamsLeaf | Mapping[str, ParamsLeaf]],
]

# Internal regime parameters: A flat mapping with function-qualified names.
# Keys are always function-qualified (e.g., "utility__risk_aversion",
# "koopmans_aggregator__discount_factor"). Values are canonical-dtype JAX arrays or
# canonical-narrow container leaves.
type FlatRegimeParams = MappingProxyType[
    QualifiedName, FloatND | IntND | BoolND | MappingLeaf | SequenceLeaf
]
# The `edges` level of the internal params: per source regime, the flat params of
# the callables its edges declare, keyed by their declaration path below
# `params["edges"][source]` joined by the qname delimiter.
type FlatEdgeParams = MappingProxyType[RegimeName, FlatRegimeParams]
# Every regime's own flat params under its name, and the edge level under
# `"edges"`.
type FlatParams = MappingProxyType[RegimeName, FlatRegimeParams | FlatEdgeParams]

# Immutable templates, used internally. Within a regime, a key is either:
# - a function name ⇒ that function's params (`{param: type-string}`)
# - a target regime's name ⇒ target-local transition params. Ordinary output
#   laws nest one level deeper (`{next_state: {param: type-string}}`); a joint
#   kernel additionally owns role branches for its support and probabilities
#   (`{kernel: {support: {...}, probabilities: {...}}}`).
type RegimeParamsTemplateNode = str | MappingProxyType[str, RegimeParamsTemplateNode]
type RegimeParamsTemplate = MappingProxyType[
    FunctionName | RegimeName, MappingProxyType[str, RegimeParamsTemplateNode]
]
# One source's branch of the `edges` template: nested by declaration path below
# `params["edges"][source]`, so its first level mixes parameters (a law over all
# targets) and target names.
type EdgeParamsTemplate = MappingProxyType[str, RegimeParamsTemplateNode]
# One branch per regime, plus an `"edges"` branch mapping each source regime to
# its `EdgeParamsTemplate` when any source's edges declare a parameter.
type ParamsTemplate = MappingProxyType[RegimeName, RegimeParamsTemplate]

# Type aliases for value function arrays
type PeriodToRegimeToVArr = MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]
# The payload classes the shipped replay routes declare. A regime's own
# `SimulationPhase.replay_route` names which one it publishes and under which
# `ReplayMode`; this alias only spells the union of those declarations.
type SimulationPolicy = (
    EGMSimPolicy | NBEGMGridPolicy | NNBEGMSimPolicy | NestedEGMSimPolicy
)
# Sparse over regimes: the inner mapping carries an entry only for regimes
# whose kernels publish a simulation policy. Regimes that publish none are
# absent — callers must not assume the full regime keyset.
type PeriodToRegimeToSimulationPolicy = MappingProxyType[
    int, MappingProxyType[RegimeName, SimulationPolicy]
]
# Sparse over regimes: the inner mapping carries an entry only for COLLECTIVE
# regimes. `True` on the state cells whose action mask is empty
# (distinct from a numeric `-inf` value); empty inner mappings for models
# without collective regimes. Returned as `backward_induction.solve`'s third
# element and consumed by `simulate` to route a dissolution-gated edge whose gate
# reads `D_target`.
type PeriodToRegimeToDissolutionFlags = MappingProxyType[
    int, MappingProxyType[RegimeName, BoolND]
]


@runtime_checkable
class EconFunction(Protocol):
    """A numeric model function after processing into the engine signature.

    Covers the *value-side* user-supplied content of a regime: the period
    utility and any helper / DAG functions
    whose output is consumed by them. Returns a numeric array
    (`FloatND` or `IntND`). Feasibility predicates live in
    `ConstraintFunction`; state / regime / process transitions live in
    `TransitionFunction`.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *args: EconFunctionArg,
        **kwargs: EconFunctionArg,
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
        *args: EconFunctionArg,
        **kwargs: EconFunctionArg,
    ) -> BoolND: ...


@runtime_checkable
class TransitionFunction(Protocol):
    """A state / regime / process transition function.

    Bound from `Model(edges=...)` as a source's regime law, in
    `Regime.state_transitions` (per-state, plus per-target dicts),
    and as the auto-generated stubs for process-derived transitions.
    Returns the deterministic next-period value (`IntND` / `FloatND`)
    or, for stochastic / weight functions, the corresponding numeric
    array (probability mass, weight, etc.).

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *args: EconFunctionArg,
        **kwargs: EconFunctionArg,
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
        *args: EconFunctionArg,
        **kwargs: EconFunctionArg,
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
        *args: EconFunctionArg,
        **kwargs: EconFunctionArg,
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
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **kwargs: QAndFArg,
    ) -> tuple[FloatND, BoolND]: ...


@runtime_checkable
class MaxQOverAFunction(Protocol):
    """The function that maximizes Q over all actions.

    Q is the state-action value function. The MaxQOverCFunction returns the maximum of Q
    over all actions. For a collective regime the core returns the pair
    `(V, D)` — the stakeholder-axis value array plus the boolean dissolution
    flag — instead of the plain V array.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **kwargs: QAndFArg,
    ) -> FloatND | tuple[FloatND, BoolND]: ...


@runtime_checkable
class EGMStepFunction(Protocol):
    """The per-period DC-EGM kernel for one regime.

    Consumes the regime's exogenous state grids, the rolling EGM-carry
    mapping, and the regime's flat params; returns the regime's value-function
    array on the exogenous state grid, the carry its parents interpolate, and
    the published consumption policy simulation interpolates off-grid. The
    `_lcm_*` keywords are the static block widths the execution plan tiles the
    kernel's loops with.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        _lcm_stochastic_node_width: int | None = None,
        _lcm_cell_width: int | None = None,
        _lcm_savings_point_width: int | None = None,
        _lcm_euler_point_width: int | None = None,
        _lcm_envelope_cell_width: int = 1,
        **kwargs: EconFunctionArg,
    ) -> tuple[FloatND, EGMCarry, EGMSimPolicy]: ...


@runtime_checkable
class EGMCarryProducer(Protocol):
    """Closed-form carry producer for a terminal regime.

    Maps the regime's solved value-function array (plus its state grids and
    flat params) to the EGM carry a DC-EGM parent interpolates.

    Used for both type checking and beartype runtime checks.

    """

    def __call__(
        self,
        *,
        V_arr: FloatND,
        **kwargs: EconFunctionArg,
    ) -> EGMCarry: ...


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
        **kwargs: QAndFArg,
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
