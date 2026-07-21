"""The solver contract: what every regime solver provides to the engine.

A regime's `solver` field selects its backward-induction algorithm. The engine
dispatches polymorphically on the solver instance — `solver.validate(context)`
then `solver.build_period_kernels(context)` — with no switch on solver type.
Add a solver by subclassing `Solver` and implementing `build_period_kernels`;
override `validate` for a build-time model-contract check (the default is a
no-op). `SolverBuildContext` carries everything a solver may read to build one
regime's kernels; `SolutionKernels` is what it hands back.

Each entry of `SolutionKernels.period_kernels` is a `PeriodKernel`: a single
non-jitted period adapter that wraps the solver's shared jitted core, calls it
with the solver's own argument layout, and assembles a `KernelResult` outside
JIT. The solve loop invokes the same adapter for every solver, branching only on
which optional outputs (`continuation`, `simulation_policy`) are present, never
on solver type.

This module is an engine leaf. Reaching `lcm.regime` would close an import
cycle — it imports the `lcm.solvers` façade, which re-exports `Solver` from
here. `UserRegime` and `VInterpolationInfo` (whose module imports `lcm.regime`)
are therefore referenced through two-form aliases: precise element types for ty
under `TYPE_CHECKING`, a bare container for the beartype claw at runtime. The
remaining engine types (`StateActionSpace`, `EGMCarry`, `EGMSimPolicy`) live in
sibling leaves with no path back to `lcm.solvers`, so they import normally and
beartype checks them precisely. The widened runtime aliases are required because
the claw beartypes each dataclass `__init__`, and under PEP 649 that forces the
field annotations to resolve to real objects when an instance is constructed.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.typing import (
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    FlatParams,
    PeriodToRegimeToSimulationPolicy,
    PeriodToRegimeToVArr,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.ages import AgeGrid
from lcm.typing import BoolND, FloatND

# The cross-period continuation channel a continuation-based parent
# interpolates. Named solver-agnostically on the seam so the engine threads it
# without knowing it is an EGM carry; today the only continuation payload is
# the EGM carry itself.
type ContinuationPayload = EGMCarry

# The published off-grid simulation-policy artifact, under the same rule: the
# engine stores and returns it opaquely — a solver-supplied reader is the only
# consumer that looks inside (DC-EGM's flat `EGMSimPolicy`, the
# continuous-outer NNBEGM's `NestedEGMSimPolicy`).
type SimulationPolicy = EGMSimPolicy | NestedEGMSimPolicy

if TYPE_CHECKING:
    from _lcm.regime_building.V import VInterpolationInfo
    from lcm.regime import Regime as UserRegime

    UserRegimesMapping: TypeAlias = Mapping[RegimeName, UserRegime]  # noqa: UP040
    RegimeToVInterpolationInfo: TypeAlias = MappingProxyType[  # noqa: UP040
        RegimeName, VInterpolationInfo
    ]
else:
    # `lcm.regime` — reached directly, and transitively through
    # `_lcm.regime_building.V` — closes a cycle via the `lcm.solvers` façade,
    # which re-exports `Solver` from this module. ty reads the precise element
    # types above; the beartype claw checks only the outer container at runtime.
    UserRegimesMapping = Mapping
    RegimeToVInterpolationInfo = MappingProxyType


@dataclass(frozen=True, kw_only=True)
class SolverBuildContext:
    """Everything a solver may read to build one regime's kernels.

    Bundled so the solver method signature stays stable as solvers with
    different needs are added; each solver reads only the fields it uses.
    """

    regime_name: RegimeName
    """Name of the regime the kernels are built for."""

    user_regimes: UserRegimesMapping
    """Mapping of regime names to user-provided `Regime` instances."""

    state_action_space: StateActionSpace
    """The regime's state-action space."""

    Q_and_F_functions: MappingProxyType[int, QAndFFunction]
    """Immutable mapping of period to Q-and-F closures."""

    grids: MappingProxyType[StateOrActionName, Grid]
    """Immutable mapping of the regime's variable names to grid objects."""

    functions: EconFunctionsMapping
    """The regime's processed functions (params renamed to qualified names)."""

    constraints: ConstraintFunctionsMapping
    """Immutable mapping of the regime's constraint names to functions."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of target regime names to transition functions."""

    stochastic_transition_names: frozenset[TransitionFunctionName]
    """Frozenset of stochastic transition function names."""

    compute_regime_transition_probs: RegimeTransitionFunction | None
    """Regime transition probability function, or `None` for terminal regimes."""

    regime_to_v_interpolation_info: RegimeToVInterpolationInfo
    """Immutable mapping of regime names to V-interpolation info."""

    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]]
    """Immutable mapping of regime names to their active period tuples."""

    flat_param_names: frozenset[str]
    """Frozenset of flat parameter names for the regime."""

    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]]
    """Immutable mapping of every regime name to its flat parameter names.

    A DC-EGM source carrying into a different target regime reads the target's
    params in its per-asset-node solve, so the kernel build admits and binds
    the union of the source and its reachable carry targets' params.
    """

    enable_jit: bool
    """Whether to JIT-compile the kernels."""

    has_taste_shocks: bool
    """Whether the regime declares EV1 taste shocks on its discrete actions."""

    certainty_equivalent: CertaintyEquivalent | None = None
    """Nonlinear certainty equivalent declared by the regime, if any.

    `GridSearch` consumes it via the compiled Q-and-F closures; solvers
    that exploit the linear-expectation structure of the continuation
    (e.g. Euler-inversion EGM) must reject regimes that declare one.
    """

    co_map_state_names: tuple[StateName, ...] = ()
    """Fixed, distributed state names co-mapped with the continuation V.

    Their axes are the leading axes of the value-function array; a solver that reads
    the continuation V slices each off and reads only the device-local slice, so no
    all-gather is inserted. Empty when no state qualifies.
    """

    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = ()
    """Per-co-map-state `in_axes` for the continuation-V mapping, aligned with
    `co_map_state_names`.

    Each entry maps a regime name to `0` (its value-function leaf carries that state —
    slice it) or `None` (the state is pruned from that regime — pass the leaf through).
    """

    stakeholders: tuple[str, ...] | None = None
    """Ordered stakeholder names for a collective regime, or `None` (singleton).

    COLLECTIVE-REGIMES (E1). When set, the grid-search kernel reads off each
    stakeholder's own value at the shared household argmax of the Pareto-weighted
    scalarization, and the regime's value-function array gains a trailing
    stakeholder axis.
    """

    weights: Mapping[str, float] | None = None
    """Household Pareto weights per stakeholder; set together with `stakeholders`."""

    edge_target_regimes: tuple[RegimeName, ...] = ()
    """Target regimes this regime reaches through a gated edge, or empty (E3').

    COLLECTIVE-REGIMES (E3'). Non-empty only for a source regime declaring
    `gated_edges`. The grid-search kernel then substitutes each such target's
    gated continuation object ``Wbar`` (supplied by the solve loop under
    ``edge_regime_to_V_arr``) for the raw target V in the ``next_regime_to_V_arr``
    mapping it reads and lowers against. Empty for every other regime.
    """

    fold_state_names: tuple[StateName, ...] = ()
    """IID-process states declared `fold=True`, or empty (the default).

    Only `GridSearch` consumes this: the grid-search kernel weighted-averages
    each named state's axis out of the stored value immediately after the
    max-over-actions / collective readout, using the process's own
    quadrature weights. Empty keeps the default path byte-identical.
    """

    same_period_ref_regimes: tuple[RegimeName, ...] = ()
    """Reference regimes whose SAME-period V this regime's kernels read (E2).

    COLLECTIVE-REGIMES (E2). Non-empty only for a collective regime declaring
    `same_period_refs`. The grid-search kernel then accepts the extra call
    argument `same_period_regime_to_V_arr` (the mapping of these regimes to
    their current-period V arrays, supplied by the solve loop after solving
    them earlier in the same period) and includes matching zero templates in
    its lowering arguments. Empty for every other regime, whose kernel
    signatures are unchanged.
    """


@dataclass(frozen=True, kw_only=True)
class KernelResult:
    """One regime-period solve output, assembled outside JIT.

    The solve loop reads `V_arr` from every kernel and branches only on whether
    the optional generic outputs are present — never on solver type:

    - `continuation` is the cross-period payload a continuation-based parent
      interpolates; `None` for a regime that publishes no continuation.
    - `simulation_policy` is the off-grid policy forward simulation can
      interpolate; `None` for a regime that publishes none.
    - `diagnostics` is the solver's numerical self-report; `None` for a solver
      that measures nothing (every finite-grid solver today).
    """

    V_arr: FloatND
    """The regime's value-function array on its exogenous state grid."""

    continuation: ContinuationPayload | None = None
    """Continuation payload for a continuation-based parent, or `None`."""

    simulation_policy: SimulationPolicy | None = None
    """Published off-grid simulation policy, or `None`."""

    dissolution: BoolND | None = None
    """The dissolution / empty-feasible-set flag `D` on the state axes, or `None`.

    COLLECTIVE-REGIMES (E2). Published by every collective regime's kernel:
    `True` exactly where NO action satisfies the combined (ordinary AND value)
    constraints, so the household argmax was taken over an empty set. Distinct
    from a numeric `-inf` value, which occurs on-path; gates must consume this
    flag, never test `V == -inf`. `None` for singleton regimes (the default
    path is unchanged).
    """


@runtime_checkable
class PeriodKernel(Protocol):
    """One regime's per-period solve adapter — the loop's uniform call target.

    A single non-jitted closure per regime-period that wraps the solver's shared
    jitted core(s) (deduped across periods by core identity), calls them with the
    solver's own argument layout, and assembles a `KernelResult` outside JIT.
    Plain closures satisfy this structurally; the loop never inspects the solver
    type. `cores()` exposes the shared jitted function(s) keyed by a stable
    per-kernel name so AOT compilation can deduplicate and lower each;
    `build_lower_args` builds a named core's lowering kwargs.

    Most kernels carry exactly one core (`{"main": ...}`); a multi-core kernel
    carries several under its own keys, one per distinct traced program it must
    lower (for example a passive keeper alongside an adjuster sweep). The AOT
    contract lowers, compiles, and dispatches each core by its key, so a
    multi-core kernel never collapses into one program.
    """

    def cores(self) -> Mapping[str, Callable]:
        """Return the shared jitted core(s), keyed by stable per-kernel name.

        Each value is a distinct traced program AOT compilation lowers and
        deduplicates independently; `build_lower_args(core_key=...)` builds the
        matching lowering kwargs and `__call__` reads the compiled cores back by
        the same key.
        """
        ...

    @property
    def core(self) -> Callable:
        """The kernel's `"main"` core, for any single-core reader.

        Defaults to `cores()["main"]`; multi-core kernels override or omit it.
        """
        ...

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> PeriodKernel:
        """Return a copy with the regime's fixed params bound into the core(s).

        The adapter owns its solver's binding rule — which fixed params reach
        the core (and any inline closure it wraps) — so the engine binds fixed
        params without a solver-type switch.
        """
        ...

    def build_lower_args(
        self,
        *,
        core_key: str,
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the named core's lowering arguments for this period.

        Single-core kernels ignore `core_key`; a multi-core kernel dispatches
        its per-core lowering off it.
        """
        ...

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Invoke the compiled core(s) and assemble the period's `KernelResult`.

        Single-core kernels read `compiled_cores["main"]`; a multi-core kernel
        reads each of its own core keys.
        """
        ...


@runtime_checkable
class SimulationPolicyReader(Protocol):
    """Solver-supplied off-grid policy read for forward simulation.

    Matches the engine's one policy-read seam: given the grid-argmax actions
    and the solver's published payload, return the (possibly) updated action
    mapping for this regime-period. The reader owns every payload-specific
    decision — row indexing, interpolation, branch re-decision, acceptance,
    fallbacks — so the simulation loop never switches on payload type. A
    reader must fall back to the grid-argmax value for any subject it cannot
    read confidently, never fabricate one.
    """

    def __call__(
        self,
        *,
        payload: SimulationPolicy,
        optimal_actions: MappingProxyType[StateOrActionName, FloatND],
        states: Mapping[StateOrActionName, FloatND],
        flat_params: FlatParams,
        period: int,
        age: FloatND,
    ) -> MappingProxyType[StateOrActionName, FloatND]:
        """Return the action mapping with off-grid reads applied."""
        ...


@dataclass(frozen=True, kw_only=True)
class SolutionKernels:
    """Per-period solve adapters produced by a solver."""

    period_kernels: Mapping[int, PeriodKernel]
    """Immutable mapping of period to the regime's uniform period adapter."""

    continuation_template: ContinuationPayload | None = None
    """All-finite template continuation with the regime's static shapes.

    `None` for a regime that publishes no continuation. Initializes the rolling
    `next_regime_to_continuation` mapping and serves as the lowering argument when
    AOT-compiling a parent's kernel.
    """


class Solver(ABC):
    """Base class for regime solvers — the polymorphic dispatch target.

    The engine calls `validate` then `build_period_kernels` on the instance,
    matching the engine's own polymorphism (`Grid(ABC)`, the stochastic
    processes). Subclasses are frozen dataclasses carrying the solver's
    configuration.
    """

    @abstractmethod
    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build the regime's per-period solve adapters."""

    @property
    def carry_retains_discrete_action_rows(self) -> bool:
        """Whether this regime's continuation carry keeps per-discrete-action rows.

        A reading parent aggregates the child's discrete choices (the DC-EGM
        logsum) only when the carry retains a row per discrete-action combo. A
        value-only solver that publishes an already-action-maxed value array
        (brute `GridSearch`, the case-piece `NBEGM`) sets this `False`, so the
        parent reads the maxed value directly without spurious action rows.
        """
        return True

    @property
    def carry_rows_share_state_grid(self) -> bool:
        """Whether every published carry row shares the state grid as abscissae.

        Grid-aligned rows (a value array published on the regime's own state
        grid) all interpolate with the same bracket structure, so reads that
        are linear in the rows commute with expectations over the carry's
        node axes. Endogenous-grid solvers publish per-row abscissae and set
        this `False`.
        """
        return False

    @property
    def n_stacked_carry_candidates(self) -> int:
        """Length of the published carry's stacked outer-candidate axis.

        A solver that publishes one carry row per outer durable candidate
        (keeper plus one per outer-grid node), stacked on an axis before the
        grid axis, declares that axis length here; a reading parent broadcasts
        its queries over exactly that many candidates and collapses them by
        the hard max. A solver whose carry has no candidate axis — no outer
        margin, or an outer margin already folded inside the solve — declares
        `0`, and the parent queries each carry row once.
        """
        return 0

    def validate(self, *, context: SolverBuildContext) -> None:  # noqa: B027
        """Check the regime is in scope for this solver. Default: no-op."""

    def build_simulation_policy_reader(
        self,
        *,
        context: SolverBuildContext,
    ) -> SimulationPolicyReader | None:
        """Build the reader that consumes this solver's `sim_policy` payload.

        `None` (the default) keeps the engine's built-in behavior: the DC-EGM
        `EGMSimPolicy` read where the regime qualifies, grid argmax otherwise.
        A solver that publishes a payload the engine cannot read (the nested
        continuous-outer policy) supplies its own reader here, so the
        simulation loop dispatches through the solver — never through an
        `isinstance` switch on the payload.
        """
        _ = context
        return None

    @property
    def requires_continuation(self) -> bool:
        """Whether this solver reads a continuation payload from its targets.

        An endogenous-grid solver inverts the Euler equation against its
        target regimes' value *and marginal* on a continuation grid, so each
        target — including a terminal one — must publish a continuation the
        engine rolls alongside `next_regime_to_V_arr`. Grid search reads only
        the value array, so it needs none. The engine reads this off every
        regime's solver to decide whether terminal regimes produce their
        closed-form continuations, without forking on the solver type.
        """
        return False
