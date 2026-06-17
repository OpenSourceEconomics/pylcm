"""Registry mapping solver configuration types to kernel builders.

`process_regimes` dispatches on `type(regime.solver)` through
`SOLVER_KERNEL_BUILDERS` to obtain the per-period solve kernels for each
regime. Adding a solver means adding one public configuration class in
`lcm.solvers` and registering one private builder here — dispatch sites
never grow `isinstance` chains.

"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import jax

from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a
from _lcm.typing import (
    MaxQOverAFunction,
    QAndFFunction,
    StateOrActionName,
)
from lcm.solvers import DCEGM, BruteForce


@dataclass(frozen=True, kw_only=True)
class SolverBuildContext:
    """Everything a solver kernel builder may need for one regime.

    Bundled so the registry's builder signature stays stable as solvers with
    different needs are added; each builder reads the fields it uses.
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


@dataclass(frozen=True, kw_only=True)
class SolverKernels:
    """Per-period solve kernels produced by a solver kernel builder."""

    max_Q_over_a: MappingProxyType[int, MaxQOverAFunction]
    """Immutable mapping of period to max-Q-over-actions kernels.

    Empty for solvers that replace the grid search with their own kernels.
    """


class SolverKernelBuilder(Protocol):
    """Build the per-period solve kernels for one regime."""

    def __call__(
        self,
        *,
        solver: BruteForce | DCEGM,
        context: SolverBuildContext,
    ) -> SolverKernels:
        """Return the per-period solve kernels keyed by period index."""
        ...


def _build_brute_force_kernels(
    *,
    solver: BruteForce | DCEGM,  # noqa: ARG001
    context: SolverBuildContext,
) -> SolverKernels:
    """Build max-Q-over-a closures for each period.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    """
    built: dict[int, MaxQOverAFunction] = {}
    result: dict[int, MaxQOverAFunction] = {}
    for period, Q_and_F in context.Q_and_F_functions.items():
        q_id = id(Q_and_F)
        if q_id not in built:
            func = get_max_Q_over_a(
                Q_and_F=Q_and_F,
                batch_sizes={
                    name: grid.batch_size
                    for name, grid in context.grids.items()
                    if name in context.state_action_space.state_names
                },
                action_names=context.state_action_space.action_names,
                state_names=context.state_action_space.state_names,
                n_discrete_action_axes=len(context.state_action_space.discrete_actions),
                has_taste_shocks=context.has_taste_shocks,
            )
            built[q_id] = jax.jit(func) if context.enable_jit else func
        result[period] = built[q_id]
    return SolverKernels(max_Q_over_a=MappingProxyType(result))


SOLVER_KERNEL_BUILDERS: MappingProxyType[type, SolverKernelBuilder] = MappingProxyType(
    {
        BruteForce: _build_brute_force_kernels,
    }
)
