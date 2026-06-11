"""Registry mapping solver configuration types to kernel builders.

`process_regimes` dispatches on `type(regime.solver)` through
`SOLVER_KERNEL_BUILDERS` to obtain the per-period solve kernels for each
regime. Adding a solver means adding one public configuration class in
`lcm.solvers` and registering one private builder here — dispatch sites
never grow `isinstance` chains.

"""

from types import MappingProxyType
from typing import Any, Protocol

import jax

from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a
from _lcm.typing import (
    MaxQOverAFunction,
    QAndFFunction,
    RegimeName,
    StateOrActionName,
)
from lcm.solvers import DCEGM, BruteForce
from lcm.typing import FloatND

DCEGM_NOT_IMPLEMENTED_MSG = (
    "The DCEGM solver is not implemented yet. A regime configured with "
    "`solver=DCEGM(...)` validates the DC-EGM model contract at `Model` "
    "construction time, but solving it is not supported; use "
    "`solver=BruteForce()` (the default) to solve the model."
)


class SolverKernelBuilder(Protocol):
    """Build the per-period solve kernels for one regime."""

    def __call__(
        self,
        *,
        solver: BruteForce | DCEGM,
        state_action_space: StateActionSpace,
        Q_and_F_functions: MappingProxyType[int, QAndFFunction],
        grids: MappingProxyType[StateOrActionName, Grid],
        enable_jit: bool,
        has_taste_shocks: bool,
    ) -> MappingProxyType[int, MaxQOverAFunction]:
        """Return the per-period solve kernels keyed by period index."""
        ...


def _build_brute_force_kernels(
    *,
    solver: BruteForce | DCEGM,  # noqa: ARG001
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    grids: MappingProxyType[StateOrActionName, Grid],
    enable_jit: bool,
    has_taste_shocks: bool,
) -> MappingProxyType[int, MaxQOverAFunction]:
    """Build max-Q-over-a closures for each period.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    """
    built: dict[int, MaxQOverAFunction] = {}
    result: dict[int, MaxQOverAFunction] = {}
    for period, Q_and_F in Q_and_F_functions.items():
        q_id = id(Q_and_F)
        if q_id not in built:
            func = get_max_Q_over_a(
                Q_and_F=Q_and_F,
                batch_sizes={
                    name: grid.batch_size
                    for name, grid in grids.items()
                    if name in state_action_space.state_names
                },
                action_names=state_action_space.action_names,
                state_names=state_action_space.state_names,
                n_discrete_action_axes=len(state_action_space.discrete_actions),
                has_taste_shocks=has_taste_shocks,
            )
            built[q_id] = jax.jit(func) if enable_jit else func
        result[period] = built[q_id]
    return MappingProxyType(result)


def _build_dcegm_kernels(
    *,
    solver: BruteForce | DCEGM,  # noqa: ARG001
    state_action_space: StateActionSpace,  # noqa: ARG001
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    grids: MappingProxyType[StateOrActionName, Grid],  # noqa: ARG001
    enable_jit: bool,  # noqa: ARG001
    has_taste_shocks: bool,  # noqa: ARG001
) -> MappingProxyType[int, MaxQOverAFunction]:
    """Build per-period kernel stubs that raise at solve time.

    `Model` construction with a valid DC-EGM regime succeeds (the contract is
    validated separately); invoking the solver raises `NotImplementedError`.
    """

    def _raise_dcegm_not_implemented(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **kwargs: Any,  # noqa: ANN401
    ) -> FloatND:
        raise NotImplementedError(DCEGM_NOT_IMPLEMENTED_MSG)

    return MappingProxyType(
        dict.fromkeys(Q_and_F_functions, _raise_dcegm_not_implemented)
    )


SOLVER_KERNEL_BUILDERS: MappingProxyType[type, SolverKernelBuilder] = MappingProxyType(
    {
        BruteForce: _build_brute_force_kernels,
        DCEGM: _build_dcegm_kernels,
    }
)
