"""The solver contract: what every regime solver provides to the engine.

A regime's `solver` field selects its backward-induction algorithm. The engine
dispatches polymorphically on the solver instance — `solver.validate(context)`
then `solver.build_period_kernels(context)` — with no switch on solver type.
Add a solver by subclassing `Solver` and implementing `build_period_kernels`;
override `validate` for a build-time model-contract check (the default is a
no-op). `SolverBuildContext` carries everything a solver may read to build one
regime's kernels; `SolverKernels` is what it hands back.

This module is an engine leaf. Reaching `lcm.regime` would close an import
cycle — it imports the `lcm.solvers` façade, which re-exports `Solver` from
here. `UserRegime` and `VInterpolationInfo` (whose module imports `lcm.regime`)
are therefore referenced through two-form aliases: precise element types for ty
under `TYPE_CHECKING`, a bare container for the beartype claw at runtime. The
remaining engine types (`StateActionSpace`, `EGMCarry`) live in sibling leaves
with no path back to `lcm.solvers`, so they import normally and beartype checks
them precisely. The widened runtime aliases are required because the claw
beartypes each dataclass `__init__`, and under PEP 649 that forces the field
annotations to resolve to real objects when a context is constructed.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

from _lcm.egm.carry import EGMCarry
from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.typing import (
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    EGMStepFunction,
    MaxQOverAFunction,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)

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


@dataclass(frozen=True, kw_only=True)
class SolverKernels:
    """Per-period solve kernels produced by a solver."""

    max_Q_over_a: MappingProxyType[int, MaxQOverAFunction]
    """Immutable mapping of period to max-Q-over-actions kernels.

    Empty for solvers that replace the grid search with their own kernels.
    """

    egm_step: MappingProxyType[int, EGMStepFunction] | None = None
    """Immutable mapping of period to DC-EGM kernels, or `None`."""

    egm_carry_template: EGMCarry | None = None
    """All-finite template carry with the regime's static shapes, or `None`."""

    egm_reachable_targets: frozenset[RegimeName] = frozenset()
    """The regime's reachable-target names — the only carry keys its kernels
    read. The solve loop filters the rolling carry mapping to these before
    handing it to each kernel, so the device need not hold every regime's
    carry at once."""


class Solver(ABC):
    """Base class for regime solvers — the polymorphic dispatch target.

    The engine calls `validate` then `build_period_kernels` on the instance,
    matching the engine's own polymorphism (`Grid(ABC)`, the stochastic
    processes). Subclasses are frozen dataclasses carrying the solver's
    configuration.
    """

    @abstractmethod
    def build_period_kernels(self, *, context: SolverBuildContext) -> SolverKernels:
        """Build the regime's per-period solve kernels."""

    def validate(self, *, context: SolverBuildContext) -> None:  # noqa: B027
        """Check the regime is in scope for this solver. Default: no-op."""
