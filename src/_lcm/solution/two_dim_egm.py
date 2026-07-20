"""The two-dimensional EGM solver (G2EGM / RFC).

`TwoDimEGM` runs the two-continuous-state endogenous grid method with the
selected candidate-refinement step (`"g2egm"` or `"rfc"`). The kernel-building
imports are function-local so the public `lcm.solvers` façade stays a thin
re-export that pulls in no numerical engine modules.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

import jax
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
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
class TwoDimEGM(Solver):
    """Two-asset G2EGM solver for a regime with two continuous Euler states.

    The working phase of the DS pension model couples a liquid state `m` and a
    pension state `n` through the budget, with two continuous actions
    (consumption and a one-directional pension deposit). The G2EGM step builds
    the four KKT constraint segments on the post-decision `(a, b)` and
    `(consumption, b)` grids, triangulates each into the current `(m, n)`
    plane, and selects the best feasible policy by the recomputed Bellman
    objective.

    A working->working period reads the regime's own next-period value on the
    `(m, n)` grid; the single working->retired boundary period reads the 1-D
    retired continuation (value and marginal) through the lump-sum pension
    payout. The continuation target per period is resolved at build time from
    the active-period structure, so the right step is selected without a
    runtime fork.
    """

    a_grid: ContinuousGrid
    """Liquid post-decision grid for the `ucon`/`dcon` segments (include 0)."""

    b_grid: ContinuousGrid
    """Pension post-decision grid shared by all segments."""

    consumption_grid: ContinuousGrid
    """Consumption sweep for the `acon`/`con` segments at `a = 0`."""

    threshold: float = 0.25
    """Barycentric extrapolation tolerance for triangle admissibility."""

    upper_envelope: Literal["g2egm", "rfc"] = "g2egm"
    """Multidimensional upper-envelope backend.

    `"g2egm"` triangulates each KKT segment and takes within- then across-segment
    maxima; `"rfc"` merges the segment clouds and selects by the Dobrescu-Shanker
    rooftop-cut delete plus a single local-simplex publish. The retirement-boundary
    period always uses the G2EGM step (the RFC step has no retiring variant yet).
    """

    @property
    def requires_continuation(self) -> bool:
        """The boundary step reads the retired regime's marginal value of liquid."""
        return True

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one G2EGM period adapter per active period.

        Periods whose next period stays in this regime use the working->working
        step; the single period whose next period leaves it (the retirement
        boundary) uses the retiring step reading the 1-D retired continuation.
        All periods share one jitted core (the boundary branch is selected by a
        static Python flag), so they reuse a single compiled program.
        """

        a_grid = self.a_grid.to_jax()
        b_grid = self.b_grid.to_jax()
        consumption_grid = self.consumption_grid.to_jax()

        period_to_target = _period_to_continuation_target(context=context)
        own_name = context.regime_name
        boundary_targets = {
            target for target in period_to_target.values() if target != own_name
        }
        boundary_prefix = next(iter(boundary_targets), own_name)
        cores: dict[bool, Callable] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        for period, target in period_to_target.items():
            is_boundary = target != own_name
            if is_boundary not in cores:
                core = _build_two_dim_core(
                    a_grid=a_grid,
                    b_grid=b_grid,
                    consumption_grid=consumption_grid,
                    threshold=self.threshold,
                    is_boundary=is_boundary,
                    interior_prefix=own_name,
                    boundary_prefix=boundary_prefix,
                    upper_envelope=self.upper_envelope,
                )
                cores[is_boundary] = jax.jit(core) if context.enable_jit else core
            period_kernels[period] = _TwoDimEGMPeriodKernel(
                core=cores[is_boundary],
                regime_name=own_name,
                continuation_target=target,
                is_boundary=is_boundary,
                transition_target_names=tuple(context.transitions),
            )
        return SolutionKernels(period_kernels=MappingProxyType(period_kernels))


@dataclass(frozen=True, kw_only=True)
class _TwoDimEGMPeriodKernel:
    """The two-asset G2EGM period adapter — wraps one G2EGM-step core.

    Closes over the regime name, the period's continuation target, and the
    transition target names. The working->working core reads the regime's own
    next-period value on `(m, n)`; the boundary core additionally reads the
    retired continuation's value and marginal carry. Returns a `KernelResult`
    whose only output is the value array — a working parent reads it directly
    as its 2-D continuation, so no carry is published.
    """

    core: Callable
    """The shared jitted G2EGM-step core (one per boundary/interior branch)."""

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    continuation_target: RegimeName
    """The regime active next period; equals this regime except at the boundary."""

    is_boundary: bool
    """Whether next period leaves this regime (the retirement boundary step)."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    def cores(self) -> Mapping[str, Callable]:
        """Return the single EGM-step core under the `"main"` key."""
        return MappingProxyType({"main": self.core})

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _TwoDimEGMPeriodKernel:
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
        """Build the core's lowering arguments: states, continuation, params."""
        return self._core_args(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
        )

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
        """Run the G2EGM step and assemble the `KernelResult`."""
        V_arr = compiled_cores["main"](
            **self._core_args(
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
            )
        )
        return KernelResult(V_arr=V_arr)

    def _core_args(
        self,
        *,
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
    ) -> dict[str, object]:
        """Assemble the core's keyword arguments for one period.

        The state grids come from the state-action space. The interior step
        reads the regime's own next-period value on `(m, n)`; the boundary step
        reads the retired continuation's value and marginal-utility carry. Each
        branch's core takes only the continuation it consumes, so the two
        signatures differ and a working continuation (which carries no marginal)
        is never demanded at the boundary.
        """
        states = dict(state_action_space.states)
        continuation: dict[str, object]
        if self.is_boundary:
            continuation = {
                "next_value_retired": next_regime_to_V_arr[self.continuation_target],
                "next_marginal_retired": next_regime_to_continuation[
                    self.continuation_target
                ].marginal_utility,
            }
        else:
            continuation = {
                "next_value_working": next_regime_to_V_arr[self.continuation_target],
            }
        return {
            "liquid": states["liquid"],
            "pension": states["pension"],
            **continuation,
            **_union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        }


def _build_two_dim_core(
    *,
    a_grid: Float1D,
    b_grid: Float1D,
    consumption_grid: Float1D,
    threshold: float,
    is_boundary: bool,
    interior_prefix: RegimeName,
    boundary_prefix: RegimeName,
    upper_envelope: Literal["g2egm", "rfc"] = "g2egm",
) -> Callable:
    """Build the jitted-able two-asset core for one branch (interior or boundary).

    The interior branch reads the regime's own next-period working value on the
    `(m, n)` grid; the boundary branch reads the 1-D retired value and marginal
    through the lump-sum payout. Both subtract the additive work disutility the
    generic envelope objective omits, so the returned value matches the engine's
    working value (whose utility carries the disutility). Transition params are
    qualified by the regime's own name (interior) or the retirement target
    (boundary), since the boundary reads the retired liquid law.

    `upper_envelope` selects the interior step's envelope — the G2EGM mesh or the
    combined-cloud RFC. The boundary (retiring) step is always G2EGM.
    """
    from _lcm.egm.rfc_two_asset_step import rfc_two_asset_step  # noqa: PLC0415
    from _lcm.egm.two_asset_g2egm_step import (  # noqa: PLC0415
        g2egm_retiring_step,
        g2egm_step,
    )

    def boundary_core(
        *,
        liquid: Float1D,
        pension: Float1D,
        next_value_retired: Float1D,
        next_marginal_retired: Float1D,
        **params: FloatND,
    ) -> FloatND:
        result = g2egm_retiring_step(
            next_value_retired=next_value_retired,
            next_marginal_retired=next_marginal_retired,
            liquid_grid=liquid,
            m_grid=liquid,
            n_grid=pension,
            a_grid=a_grid,
            b_grid=b_grid,
            consumption_grid=consumption_grid,
            discount_factor=params["H__discount_factor"],
            crra=params["utility__crra"],
            match_rate=params[f"{boundary_prefix}__next_liquid__match_rate"],
            return_liquid=params[f"{boundary_prefix}__next_liquid__return_liquid"],
            pension_payout_return=params[
                f"{boundary_prefix}__next_liquid__pension_payout_return"
            ],
            retirement_income=params[
                f"{boundary_prefix}__next_liquid__retirement_income"
            ],
            threshold=threshold,
        )
        return result.value - params["utility__work_disutility"]

    def interior_core(
        *,
        liquid: Float1D,
        pension: Float1D,
        next_value_working: FloatND,
        **params: FloatND,
    ) -> FloatND:
        discount_factor = params["H__discount_factor"]
        crra = params["utility__crra"]
        match_rate = params[f"{interior_prefix}__next_pension__match_rate"]
        return_liquid = params[f"{interior_prefix}__next_liquid__return_liquid"]
        return_pension = params[f"{interior_prefix}__next_pension__return_pension"]
        wage = params[f"{interior_prefix}__next_liquid__wage"]
        if upper_envelope == "rfc":
            result = rfc_two_asset_step(
                next_value=next_value_working,
                m_grid=liquid,
                n_grid=pension,
                a_grid=a_grid,
                b_grid=b_grid,
                consumption_grid=consumption_grid,
                discount_factor=discount_factor,
                crra=crra,
                match_rate=match_rate,
                return_liquid=return_liquid,
                return_pension=return_pension,
                wage=wage,
            )
        else:
            result = g2egm_step(
                next_value=next_value_working,
                m_grid=liquid,
                n_grid=pension,
                a_grid=a_grid,
                b_grid=b_grid,
                consumption_grid=consumption_grid,
                discount_factor=discount_factor,
                crra=crra,
                match_rate=match_rate,
                return_liquid=return_liquid,
                return_pension=return_pension,
                wage=wage,
                threshold=threshold,
            )
        return result.value - params["utility__work_disutility"]

    return boundary_core if is_boundary else interior_core
