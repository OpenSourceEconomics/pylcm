"""The one-asset EGM solver.

`OneAssetEGM` runs the single-asset endogenous grid method for a regime with
one continuous (Euler) state and no discrete kinks — the specialization whose
step needs no upper envelope. The kernel-building imports are function-local
so the public `lcm.solvers` façade stays a thin re-export that pulls in no
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
from _lcm.egm.carry import EGMCarry
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid
from _lcm.solution.continuation_target import (
    _period_to_continuation_target,
    _union_fixed_params,
    _union_free_params,
)
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
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.typing import (
    Float1D,
    FloatND,
)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class OneAssetEGM(Solver):
    """Endogenous-grid solver for a 1-D consumption--saving regime.

    A regime with exactly one continuous state (the liquid wealth), one
    continuous consumption action, and no discrete choice is a plain
    consumption--saving problem. The single continuous state needs no upper
    envelope: inverting the consumption Euler equation on the post-decision
    savings grid and mapping the resulting endogenous wealth back onto the
    regular grid solves the period exactly. The step carries the marginal
    value of liquid backward (the envelope theorem makes it exact, unlike a
    finite difference of a coarse value array), so each period both reads its
    continuation's marginal and publishes its own.
    """

    savings_grid: ContinuousGrid
    """Exogenous post-decision savings grid `s = liquid - consumption` (>= 0)."""

    @property
    def requires_continuation(self) -> bool:
        """The 1-D EGM step reads its continuation's marginal value of liquid."""
        return True

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one 1-D EGM period adapter per active period.

        Each period's adapter knows the single deterministic continuation
        target (the transition target whose regime is active next period), so
        it reads that target's value array and marginal-utility carry.
        """

        savings_grid = self.savings_grid.to_jax()
        liquid_grid = context.grids[context.state_action_space.state_names[0]].to_jax()

        period_to_target = _period_to_continuation_target(context=context)
        cores: dict[RegimeName, Callable] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        for period, target in period_to_target.items():
            if target not in cores:
                core = _build_one_asset_core(savings_grid=savings_grid, target=target)
                cores[target] = jax.jit(core) if context.enable_jit else core
            period_kernels[period] = _OneAssetEGMPeriodKernel(
                core=cores[target],
                regime_name=context.regime_name,
                continuation_target=target,
                transition_target_names=tuple(context.transitions),
            )
        return SolutionKernels(
            period_kernels=MappingProxyType(period_kernels),
            continuation_template=_build_one_asset_carry_template(
                liquid_grid=liquid_grid
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _OneAssetEGMPeriodKernel:
    """The 1-D EGM period adapter — wraps the shared `egm_one_asset_step` core.

    Closes over the regime name, the period's single deterministic
    continuation target (whose value array and marginal carry feed the Euler
    inversion), and the transition target names (to union their params).
    Returns a `KernelResult` carrying the value array and the marginal-value
    carry a parent EGM regime interpolates.
    """

    core: Callable
    """The shared jitted 1-D EGM-step core."""

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    continuation_target: RegimeName
    """The regime active next period; its value and marginal continue this one."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    def cores(self) -> Mapping[str, Callable]:
        """Return the single EGM-step core under the `"main"` key."""
        return MappingProxyType({"main": self.core})

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _OneAssetEGMPeriodKernel:
        """Bind the regime's and its targets' fixed params into the core."""
        bound = _union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not bound:
            return self
        return replace(self, core=functools.partial(self.core, **bound))

    def build_lower_args(
        self,
        *,
        core_key: str = "main",  # noqa: ARG002
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: state, continuation, params."""
        return {
            **dict(state_action_space.states),
            "next_value": next_regime_to_V_arr[self.continuation_target],
            "next_marginal": next_regime_to_continuation[
                self.continuation_target
            ].marginal_utility,
            **_union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        }

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> KernelResult:
        """Run the 1-D EGM step and assemble the `KernelResult`."""
        V_arr, carry = compiled_cores["main"](
            **state_action_space.states,
            next_value=next_regime_to_V_arr[self.continuation_target],
            next_marginal=next_regime_to_continuation[
                self.continuation_target
            ].marginal_utility,
            **_union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        )
        return KernelResult(V_arr=V_arr, continuation=carry)


def _build_one_asset_core(*, savings_grid: Float1D, target: RegimeName) -> Callable:
    """Build the jitted-able 1-D EGM core closing over the savings grid.

    The core reads the state grid (`liquid`), the continuation value and
    marginal, and the regime's scalar params, runs `egm_one_asset_step`, and
    returns the value array and the marginal-value carry on the liquid grid.
    The liquid law params are qualified by the continuation target's transition
    (`{target}__next_liquid__...`).
    """
    from _lcm.egm.one_asset_egm_step import egm_one_asset_step  # noqa: PLC0415

    def core(
        *,
        liquid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        step = egm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            savings_grid=savings_grid,
            discount_factor=params["H__discount_factor"],
            crra=params["utility__crra"],
            return_liquid=params[f"{target}__next_liquid__return_liquid"],
            income=params[f"{target}__next_liquid__retirement_income"],
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=step.value,
            marginal_utility=step.marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=step.value.dtype),
        )
        return step.value, carry

    return core


def _build_one_asset_carry_template(*, liquid_grid: Float1D) -> EGMCarry:
    """Build the all-finite 1-D EGM carry template on the liquid grid."""
    return EGMCarry(
        endog_grid=liquid_grid,
        value=jnp.zeros_like(liquid_grid),
        marginal_utility=jnp.zeros_like(liquid_grid),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
    )
