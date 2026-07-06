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
which optional outputs (`carry`, `sim_policy`) are present, never on solver type.

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
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.typing import (
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    FlatParams,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.ages import AgeGrid
from lcm.typing import FloatND

# The cross-period continuation channel a DC-EGM parent interpolates. Named
# solver-agnostically on the seam so the engine threads it without knowing it is
# an EGM carry; today the only continuation payload is the EGM carry itself.
type ContinuationPayload = EGMCarry

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


@dataclass(frozen=True, kw_only=True)
class KernelResult:
    """One regime-period solve output, assembled outside JIT.

    The solve loop reads `V_arr` from every kernel and branches only on whether
    the optional generic outputs are present — never on solver type:

    - `carry` is the cross-period continuation a DC-EGM parent interpolates;
      `None` for a regime that publishes no continuation.
    - `sim_policy` is the off-grid consumption policy DC-EGM forward simulation
      can interpolate; `None` for a regime that publishes none.
    """

    V_arr: FloatND
    """The regime's value-function array on its exogenous state grid."""

    carry: ContinuationPayload | None = None
    """Continuation payload for a DC-EGM parent, or `None`."""

    sim_policy: EGMSimPolicy | None = None
    """Published off-grid simulation policy, or `None`."""


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

    Most kernels carry exactly one core (`{"main": ...}`); the NEGM kernel
    carries two (`{"keeper": ..., "adjuster": ...}`), a per-durable-state passive
    DC-EGM keeper alongside the adjuster sweep. The AOT contract lowers, compiles,
    and dispatches each core by its key, so a multi-core kernel never collapses
    into one program.
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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the named core's lowering arguments for this period.

        Single-core kernels ignore `core_key`; the NEGM kernel dispatches the
        keeper-vs-adjuster lowering off it.
        """
        ...

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Invoke the compiled core(s) and assemble the period's `KernelResult`.

        Single-core kernels read `compiled_cores["main"]`; the NEGM kernel reads
        `["keeper"]` and `["adjuster"]`.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class SolutionKernels:
    """Per-period solve adapters produced by a solver."""

    period_kernels: Mapping[int, PeriodKernel]
    """Immutable mapping of period to the regime's uniform period adapter."""

    continuation_template: ContinuationPayload | None = None
    """All-finite template continuation with the regime's static shapes.

    `None` for a regime that publishes no continuation. Initializes the rolling
    `next_regime_to_egm_carry` mapping and serves as the lowering argument when
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

    def validate(self, *, context: SolverBuildContext) -> None:  # noqa: B027
        """Check the regime is in scope for this solver. Default: no-op."""

    @property
    def requires_continuation_carries(self) -> bool:
        """Whether this solver reads a continuation carry from its targets.

        An endogenous-grid solver inverts the Euler equation against its
        target regimes' value *and marginal* on a continuation grid, so each
        target — including a terminal one — must publish a carry the engine
        rolls alongside `next_regime_to_V_arr`. Grid search reads only the
        value array, so it needs no carry. The engine reads this off every
        regime's solver to decide whether terminal regimes produce their
        closed-form carries, without forking on the solver type.
        """
        return False
