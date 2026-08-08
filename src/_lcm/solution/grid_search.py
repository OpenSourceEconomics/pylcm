"""The default grid-search solver.

`GridSearch` runs the max-Q-over-a grid search. Its `build_period_kernels`
returns one `PeriodKernel` per period — a non-jitted adapter that wraps the
shared jitted core (identity-deduped by `id(Q_and_F)`, so periods sharing a
core reuse one compiled program), calls it with the grid-search argument
layout, and assembles a `KernelResult` outside JIT.

The kernel-building imports (`jax`, `get_max_Q_over_a`) are function-local so
the public `lcm.solvers` façade stays a thin re-export that pulls in no
numerical engine modules.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.engine import StateActionSpace
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SolutionKernels,
    Solver,
    SolverBuildContext,
)
from _lcm.typing import (
    FlatParams,
    MaxQOverAFunction,
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.typing import (
    FloatND,
)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GridSearch(Solver):
    """Grid-search solver over the full state-action product (the default)."""

    @property
    def carry_retains_discrete_action_rows(self) -> bool:
        """A brute living child publishes an already-action-maxed value array."""
        return False

    @property
    def carry_rows_share_state_grid(self) -> bool:
        """Grid-search carries sit on the regime's own state grid."""
        return True

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one max-Q-over-a period adapter per period.

        Periods sharing the same Q_and_F object reuse a single jitted core,
        and therefore a single compiled program.
        """
        from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a  # noqa: PLC0415

        built: dict[int, MaxQOverAFunction] = {}
        result: dict[int, PeriodKernel] = {}
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
                    n_discrete_action_axes=len(
                        context.state_action_space.discrete_actions
                    ),
                    has_taste_shocks=context.has_taste_shocks,
                    co_map_state_names=context.co_map_state_names,
                    co_map_v_arr_in_axes=context.co_map_v_arr_in_axes,
                )
                built[q_id] = jax.jit(func) if context.enable_jit else func
            result[period] = _GridSearchPeriodKernel(
                core=built[q_id], regime_name=context.regime_name
            )
        return SolutionKernels(period_kernels=MappingProxyType(result))


@dataclass(frozen=True, kw_only=True)
class _GridSearchPeriodKernel:
    """The grid-search period adapter — wraps one max-Q-over-a core.

    Closes over the regime name (to project its flat params) and the shared
    jitted core. Calling it evaluates Q on the full state-action product and
    maximizes over the actions, returning a `KernelResult` whose only output is
    the value-function array — no continuation, no simulation policy.
    """

    core: Callable
    """The shared jitted max-Q-over-a core (`id`-deduped across periods)."""

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    def cores(self) -> Mapping[str, Callable]:
        """Return the single max-Q-over-a core under the `"main"` key."""
        return MappingProxyType({"main": self.core})

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _GridSearchPeriodKernel:
        """Bind the regime's fixed params into the core.

        The core threads its `**kwargs` into the per-combo pool, so binding the
        regime's own fixed params restores the values removed from the live
        `flat_params`; the captured functions read only the keys they need.
        """
        regime_fixed = dict(
            fixed_flat_params.get(self.regime_name, MappingProxyType({}))
        )
        if not regime_fixed:
            return self
        return replace(self, core=functools.partial(self.core, **regime_fixed))

    def build_lower_args(
        self,
        *,
        core_key: str = "main",  # noqa: ARG002
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],  # noqa: ARG002
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: the full state-action product."""
        return {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(flat_params[self.regime_name]),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],  # noqa: ARG002
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Evaluate the grid search and assemble the `KernelResult`."""
        V_arr = compiled_cores["main"](
            **state_action_space.states,
            **state_action_space.actions,
            next_regime_to_V_arr=next_regime_to_V_arr,
            **flat_params[self.regime_name],
            period=jnp.int32(period),
            age=ages.values[period],
        )
        return KernelResult(V_arr=V_arr)
