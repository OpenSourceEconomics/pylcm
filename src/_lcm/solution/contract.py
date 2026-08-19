"""The solver contract: what every regime solver provides to the engine.

A regime's `solver` field selects its backward-induction algorithm. The engine
dispatches polymorphically on the solver instance. Model finalization calls
`solver.validate_model(context)`; the processed build calls
`solver.validate_build(context)` and then
`solver.build_period_kernels(context)`, with no switch on solver type. Add a
solver by subclassing `Solver` and implementing `build_period_kernels`; override
either validation hook for the stage whose information it needs (both default
to no-ops).

Each entry of `SolutionKernels.period_kernels` is a `PeriodKernel`: a single
non-jitted period adapter that wraps the solver's shared jitted core, calls it
with the solver's own argument layout, and assembles a `KernelResult` outside
JIT. The solve loop invokes the same adapter for every solver, branching only on
which optional outputs (`continuation`, `simulation_policy`) are present, never
on solver type.

This module is an engine leaf. Resolving finalized user-regime or
`VInterpolationInfo` types at runtime would close an import cycle through the
`lcm.solvers` façade, which re-exports `Solver` from here. They are therefore
referenced through two-form aliases: precise element types for ty under
`TYPE_CHECKING`, a bare container for the beartype claw at runtime. The
remaining engine types (`StateActionSpace`, `EGMCarry`, `EGMSimPolicy`) live in
sibling leaves with no path back to `lcm.solvers`, so they import normally and
beartype checks them precisely. The widened runtime aliases are required because
the claw beartypes each dataclass `__init__`, and under PEP 649 that forces the
field annotations to resolve to real objects when an instance is constructed.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.continuation import (
    ContinuationPayload,
    EGMContinuationLayout,
    EGMContinuationSpec,
)
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.reachability import PhaseReachability
from _lcm.transition_laws import TransitionLaws
from _lcm.typing import (
    ActionName,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    FlatParams,
    FunctionName,
    PeriodToRegimeToSimulationPolicy,
    PeriodToRegimeToVArr,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionsMapping,
)
from lcm.ages import AgeGrid
from lcm.typing import Float1D, FloatND

# The continuation channel is defined once in `_lcm.continuation`. Backward
# induction treats `ContinuationPayload` opaquely; EGM solvers additionally
# publish one `EGMContinuationSpec` bundling the template and layout metadata.

# The published off-grid simulation-policy artifact. The rule here is a **fixed
# read set** rather than opacity, because simulation genuinely interpolates the
# rows: it reads the row arrays (`endog_grid` to locate a query, `value` or
# `policy` as the field asked for, `marginal_utility` for the node slopes the
# value convention needs) and the row-axis name tuples that say which regime
# variable each leading axis stands for. Those two groups are the contract, and
# a second implementation has to publish both. Any *further* field the engine
# reaches for is the signal to introduce a protocol rather than to grow this
# one — the seam is a stated read set, not permission to read whatever is there.
type SimulationPolicy = EGMSimPolicy

if TYPE_CHECKING:
    from _lcm.regime_building.finalize import FinalizedUserRegime
    from _lcm.regime_building.V import VInterpolationInfo

    UserRegimesMapping: TypeAlias = Mapping[  # noqa: UP040
        RegimeName, FinalizedUserRegime
    ]
    RegimeToVInterpolationInfo: TypeAlias = MappingProxyType[  # noqa: UP040
        RegimeName, VInterpolationInfo
    ]
else:
    # Resolving the element types closes a cycle via the `lcm.solvers` façade,
    # which re-exports `Solver` from this module. ty reads the precise types
    # above; the beartype claw checks only the outer container at runtime.
    UserRegimesMapping = Mapping
    RegimeToVInterpolationInfo = MappingProxyType


@dataclass(frozen=True, kw_only=True)
class SolverModelContext:
    """Finalized user-level information available to solver validation."""

    regime_name: RegimeName
    """Name of the regime whose solver is being validated."""

    user_regimes: UserRegimesMapping
    """Mapping of every finalized user regime in the model."""

    solution_reachability: PhaseReachability | None = None
    """Static solution graph when available; `None` during early validation."""


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

    solution_reachability: PhaseReachability
    """Static solution graph used to derive solver layouts."""

    Q_and_F_functions: MappingProxyType[int, QAndFFunction]
    """Immutable mapping of period to Q-and-F closures."""

    grids: MappingProxyType[StateOrActionName, Grid]
    """Immutable mapping of the regime's variable names to grid objects.

    Age-invariant: for an `AgeSpecializedGrid` state this holds the representative
    age's grid. Read it for a grid's *shape traits* — kind, `n_points`, dtype,
    `batch_size` — which are invariant across ages by contract. For a grid's *node
    values* in a particular period, read `period_to_state_nodes`.
    """

    period_to_state_nodes: (
        MappingProxyType[int, MappingProxyType[StateName, Float1D]] | None
    ) = None
    """Immutable mapping of period to that period's age-specialized state nodes.

    `None` when the regime has no age-specialized state, in which case `grids` is
    already the whole story. A solver that lifts a state's nodes into a numerical
    computation must consult this per period: capturing one array outside its
    per-period loop silently pins every period to the representative age.
    """

    functions: EconFunctionsMapping
    """The regime's processed functions (params renamed to qualified names)."""

    koopmans_aggregator: EconFunction | None
    """The regime's processed Koopmans aggregator, or `None` in a terminal regime.

    Processed like every other function, so its parameters carry their qualified
    names — a solver that reads them off the runtime pool must use this object,
    not the user-facing `LinearAggregator`.
    """

    constraints: ConstraintFunctionsMapping
    """Immutable mapping of the regime's constraint names to functions."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of target regime names to transition functions."""

    transition_laws: TransitionLaws
    """Immutable mapping of target regime names to their transition laws."""

    compute_regime_transition_probs: RegimeTransitionFunction | None
    """Regime transition probability function, or `None` for terminal regimes."""

    regime_to_v_interpolation_info: RegimeToVInterpolationInfo
    """Immutable mapping of regime names to V-interpolation info."""

    period_to_regime_v_interp: (
        MappingProxyType[int, RegimeToVInterpolationInfo] | None
    ) = None
    """Immutable mapping of period to that period's V-interpolation info per regime.

    A period-`t` kernel reads its continuation on the *target's* period-`t+1` grid,
    so a solver that interpolates `V_{t+1}` must look the target's info up under
    `period + 1` rather than reuse `regime_to_v_interpolation_info`, which carries
    the representative age. `None` when no regime has an age-specialized state.
    """

    period_to_regime_grid_signature: (
        MappingProxyType[int, MappingProxyType[RegimeName, Hashable]] | None
    ) = None
    """Immutable mapping of period to each regime's age-specialized grid signature.

    The user's own `AgeSpecializedGrid.signature(age)` values, so a solver that
    groups periods into shared compiled programs can fold its targets' signatures
    at `period + 1` into the group key. Periods whose continuation grids differ
    then never share a trace. `None` when no regime has an age-specialized state.
    """

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

    certainty_equivalent: CertaintyEquivalent | None
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


@dataclass(frozen=True, kw_only=True)
class KernelResult:
    """One regime-period solve output, assembled outside JIT.

    The solve loop reads `V_arr` from every kernel and branches only on whether
    the optional generic outputs are present — never on solver type:

    - `continuation` is the cross-period payload a continuation-based parent
      interpolates; `None` for a regime that publishes no continuation.
    - `simulation_policy` is the off-grid policy forward simulation can
      interpolate; `None` for a regime that publishes none.
    """

    V_arr: FloatND
    """The regime's value-function array on its exogenous state grid."""

    continuation: ContinuationPayload | None = None
    """Continuation payload for a continuation-based parent, or `None`."""

    simulation_policy: SimulationPolicy | None = None
    """Published off-grid simulation policy, or `None`."""


@dataclass(frozen=True, kw_only=True)
class BackwardInductionResult:
    """The generic outputs of one backward-induction run.

    Internal to the engine: the public `Model.solve` unpacks it into its
    documented mapping-or-tuple return shape.
    """

    value_functions: PeriodToRegimeToVArr
    """Immutable mapping of period to each regime's value-function array."""

    simulation_policies: PeriodToRegimeToSimulationPolicy
    """Immutable mapping of period to each regime's published simulation policy.

    Sparse over regimes: only kernels that publish a policy contribute entries.
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


@dataclass(frozen=True)
class _BoundLiquidMargin:
    """Resolved DAG role names carried privately by an EGM-family solver."""

    state: StateName
    action: ActionName
    resources: FunctionName
    post_decision_state: FunctionName
    before_cost: FunctionName | None = None
    cost: FunctionName | None = None


@dataclass(frozen=True)
class _BoundOuterContinuousMargin:
    """Resolved outer-margin DAG role names carried privately by a solver.

    `no_adjustment` is `None` where the regime declares the identity map with
    `lcm.outer_unchanged`, which is the form every consumer branches on.
    """

    state: StateName
    action: ActionName
    post_decision_state: FunctionName
    no_adjustment: FunctionName | None


@dataclass(frozen=True, kw_only=True)
class SolutionKernels:
    """Per-period solve adapters produced by a solver."""

    period_kernels: Mapping[int, PeriodKernel]
    """Immutable mapping of period to the regime's uniform period adapter."""

    continuation_spec: EGMContinuationSpec | None = None
    """Concrete EGM continuation template bundled with its static layout."""

    @property
    def continuation_template(self) -> ContinuationPayload | None:
        """Return the template payload for generic rolling and lowering code."""
        return (
            None if self.continuation_spec is None else self.continuation_spec.template
        )


class Solver(ABC):
    """Base class for regime solvers — the polymorphic dispatch target.

    The engine calls `validate_model`, then later `validate_build` and
    `build_period_kernels` on the instance, matching the engine's own
    polymorphism (`Grid(ABC)`, the stochastic processes). Subclasses are frozen
    dataclasses carrying the solver's configuration.
    """

    @abstractmethod
    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build the regime's per-period solve adapters."""

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """Declare how a reading parent interprets this solver's EGM carry.

        The default is an endogenous-grid row per discrete-action combination,
        with row-specific abscissae and no stacked outer-candidate axis. Solvers
        whose representation differs override this one bundled declaration.
        """
        return EGMContinuationLayout()

    def validate_model(self, *, context: SolverModelContext) -> None:  # noqa: B027
        """Validate finalized user declarations. Default: no-op."""

    def validate_build(self, *, context: SolverBuildContext) -> None:  # noqa: B027
        """Validate processed period/build context. Default: no-op."""

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


class OneMarginSolver(Solver):
    """Marker base for solvers consuming one explicit liquid margin.

    Future NB-EGM implementations can join this family without changing the
    regime interface; the only shared operation is immutable role binding.
    """

    @abstractmethod
    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> OneMarginSolver:
        """Return an immutable solver copy bound to one liquid margin."""


class TwoMarginSolver(Solver):
    """Marker base for solvers consuming liquid and outer continuous margins."""

    @abstractmethod
    def _with_margins(
        self,
        *,
        liquid: _BoundLiquidMargin,
        outer: _BoundOuterContinuousMargin,
    ) -> TwoMarginSolver:
        """Return an immutable solver copy bound to both margins."""


_BOUND_TYPES: dict[tuple[type, type], type] = {}


def bind_roles(*, solver: Solver, role_type: type, **roles: object) -> Solver:
    """Return a copy of `solver` carrying `roles`, keeping its own type.

    The regime resolves a solver's DAG role names and hands them back to the
    solver, which has to answer with an object that is still the one the user
    constructed: a subclass keeps the fields it added and the methods it
    overrides, so a custom solver reaches the engine as itself.

    `role_type` declares the role fields for the stock solver and is returned
    directly for it. For any other concrete type, the bound class is derived
    from that type and `role_type` together, so the subclass sits ahead of the
    stock solver in the method resolution order.

    Args:
        solver: The solver whose roles are being bound.
        role_type: The bound dataclass declaring the role fields, itself a
            subclass of the stock solver.
        **roles: The resolved role names, one per field `role_type` declares.
            An entry naming a field the solver already carries replaces it, which
            is how a nest hands on its *bound* inner solver rather than the
            public one it was declared with.

    Returns:
        A frozen solver of `solver`'s own type carrying the resolved roles.

    """
    solver_type = type(solver)
    stock_type = role_type.__bases__[0]
    values = {
        field.name: getattr(solver, field.name)
        for field in fields(solver_type)  # ty: ignore[invalid-argument-type]
    }
    if solver_type is stock_type:
        bound_type = role_type
    elif issubclass(solver_type, role_type):
        # Already bound: a regime that replaces one of its own fields rebinds
        # the solver it is already carrying, and its type is the answer.
        bound_type = solver_type
    else:
        key = (solver_type, role_type)
        if key not in _BOUND_TYPES:
            derived = type(
                f"_Bound{solver_type.__name__}", (solver_type, role_type), {}
            )
            _BOUND_TYPES[key] = dataclass(frozen=True, kw_only=True)(derived)
        bound_type = _BOUND_TYPES[key]
    return bound_type(**(values | roles))
