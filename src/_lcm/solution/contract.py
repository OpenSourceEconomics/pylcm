"""The solver contract: what every regime solver provides to the engine.

A regime's `solver` field selects its backward-induction algorithm. The engine
dispatches polymorphically on the solver instance — `solver.validate(context)`
then `solver.build_period_kernels(context)` — with no switch on solver type.
Add a solver by subclassing `Solver` and implementing `build_period_kernels`;
override `validate` for a build-time model-contract check (the default is a
no-op). `SolverBuildContext` carries everything a solver may read to build one
regime's kernels; `SolverKernels` is what it hands back.

This module is an engine leaf: it imports only `_lcm.engine` / `_lcm.grids` /
`_lcm.typing` (none of which reach `lcm.solvers`), so the public solver façade
can re-export it without forming an import cycle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.typing import (
    MaxQOverAFunction,
    QAndFFunction,
    RegimeName,
    StateName,
    StateOrActionName,
)


@dataclass(frozen=True, kw_only=True)
class SolverBuildContext:
    """Everything a solver may read to build one regime's kernels.

    Bundled so the solver method signature stays stable as solvers with
    different needs are added; each solver reads only the fields it uses.
    """

    state_action_space: StateActionSpace
    """The regime's state-action space."""

    Q_and_F_functions: MappingProxyType[int, QAndFFunction]
    """Immutable mapping of period to Q-and-F closures."""

    grids: MappingProxyType[StateOrActionName, Grid]
    """Immutable mapping of the regime's variable names to grid objects."""

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
class SolverKernels:
    """Per-period solve kernels produced by a solver."""

    max_Q_over_a: MappingProxyType[int, MaxQOverAFunction]
    """Immutable mapping of period to max-Q-over-actions kernels.

    Empty for solvers that replace the grid search with their own kernels.
    """


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
