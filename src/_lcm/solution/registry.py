"""Registry mapping solver configuration types to kernel builders.

`process_regimes` dispatches on `type(regime.solver)` through
`SOLVER_KERNEL_BUILDERS` to obtain the per-period solve kernels for each
regime. Adding a solver means adding one public configuration class in
`lcm.solvers` and registering one private builder here — dispatch sites
never grow `isinstance` chains.

"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import jax

from _lcm.egm.carry import EgmCarry
from _lcm.egm.step import build_egm_step_functions
from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    EgmStepFunction,
    MaxQOverAFunction,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, BruteForce


@dataclass(frozen=True, kw_only=True)
class SolverBuildContext:
    """Everything a solver kernel builder may need for one regime.

    Bundled so the registry's builder signature stays stable as solvers with
    different needs are added; each builder reads the fields it uses.
    """

    regime_name: RegimeName
    """Name of the regime the kernels are built for."""

    user_regimes: Mapping[RegimeName, UserRegime]
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

    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo]
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
    """Per-period solve kernels produced by a solver kernel builder."""

    max_Q_over_a: MappingProxyType[int, MaxQOverAFunction]
    """Immutable mapping of period to max-Q-over-actions kernels.

    Empty for solvers that replace the grid search with their own kernels.
    """

    egm_step: MappingProxyType[int, EgmStepFunction] | None = None
    """Immutable mapping of period to DC-EGM kernels, or `None`."""

    egm_carry_template: EgmCarry | None = None
    """All-finite template carry with the regime's static shapes, or `None`."""


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


def _build_dcegm_kernels(
    *,
    solver: BruteForce | DCEGM,
    context: SolverBuildContext,
) -> SolverKernels:
    """Build per-period DC-EGM kernels and the regime's carry template.

    Validation guarantees the regime is non-terminal, so the regime
    transition probability function exists.
    """
    assert isinstance(solver, DCEGM)  # noqa: S101
    assert context.compute_regime_transition_probs is not None  # noqa: S101
    egm_step, egm_carry_template = build_egm_step_functions(
        solver=solver,
        regime_name=context.regime_name,
        user_regimes=context.user_regimes,
        functions=context.functions,
        constraints=context.constraints,
        transitions=context.transitions,
        stochastic_transition_names=context.stochastic_transition_names,
        compute_regime_transition_probs=context.compute_regime_transition_probs,
        regime_to_v_interpolation_info=context.regime_to_v_interpolation_info,
        regimes_to_active_periods=context.regimes_to_active_periods,
        flat_param_names=context.flat_param_names,
        regime_to_flat_param_names=context.regime_to_flat_param_names,
        state_action_space=context.state_action_space,
        has_taste_shocks=context.has_taste_shocks,
    )
    if context.enable_jit:
        jitted_by_id: dict[int, EgmStepFunction] = {}
        for func in egm_step.values():
            if id(func) not in jitted_by_id:
                jitted_by_id[id(func)] = jax.jit(func)
        egm_step = MappingProxyType(
            {period: jitted_by_id[id(func)] for period, func in egm_step.items()}
        )
    return SolverKernels(
        max_Q_over_a=MappingProxyType({}),
        egm_step=egm_step,
        egm_carry_template=egm_carry_template,
    )


SOLVER_KERNEL_BUILDERS: MappingProxyType[type, SolverKernelBuilder] = MappingProxyType(
    {
        BruteForce: _build_brute_force_kernels,
        DCEGM: _build_dcegm_kernels,
    }
)
