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
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid
from _lcm.solution.continuation_target import (
    _period_to_continuation_target,
    _union_fixed_params,
    _union_free_params,
    target_period_grid,
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
from lcm.exceptions import RegimeInitializationError
from lcm.typing import (
    Float1D,
    FloatND,
    StateName,
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

    liquid_state: StateName = "liquid"
    """Name of the regime's liquid (cash-on-hand) continuous state.

    Two continuous states cannot be told apart positionally, so the modeller
    names which one fills each economic role. The kernel's own `(m, n)`
    vocabulary stays private to it.
    """

    pension_state: StateName = "pension"
    """Name of the regime's illiquid (pension) continuous state."""

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

    def validate(self, *, context: SolverBuildContext) -> None:
        """Check the regime and its targets fit the two-asset G2EGM step.

        Every message names the regime's own states and the role field that
        assigns them, so a mismatch is fixed without knowing the kernel's
        internal `(m, n)` vocabulary.
        """
        own_states = context.regime_to_v_interpolation_info[
            context.regime_name
        ].continuous_states
        if len(own_states) != 2:  # noqa: PLR2004
            msg = (
                f"TwoDimEGM regime '{context.regime_name}' must have exactly two "
                f"continuous states, but has {len(own_states)}: "
                f"{sorted(own_states)}."
            )
            raise RegimeInitializationError(msg)
        missing = {
            field: name
            for field, name in (
                ("liquid_state", self.liquid_state),
                ("pension_state", self.pension_state),
            )
            if name not in own_states
        }
        if missing:
            msg = (
                f"TwoDimEGM regime '{context.regime_name}' has continuous states "
                f"{sorted(own_states)}, which do not include "
                + ", ".join(f"{field}='{name}'" for field, name in missing.items())
                + ". Set the role fields to the regime's own state names."
            )
            raise RegimeInitializationError(msg)
        if self.liquid_state == self.pension_state:
            msg = (
                f"TwoDimEGM regime '{context.regime_name}' assigns the same state "
                f"'{self.liquid_state}' to both the liquid and the pension role."
            )
            raise RegimeInitializationError(msg)
        # The step's value array has one axis per role and nothing else, so a
        # third state has no axis to land on. Rejecting here beats publishing a
        # value array whose axes silently mean something other than they claim.
        declared = tuple(context.state_action_space.state_names)
        if set(declared) != {self.liquid_state, self.pension_state}:
            msg = (
                f"TwoDimEGM regime '{context.regime_name}' declares states "
                f"{list(declared)}; the two-asset step supports exactly the two "
                f"role states '{self.liquid_state}' and '{self.pension_state}' "
                f"and no others."
            )
            raise RegimeInitializationError(msg)

        period_to_target = _period_to_continuation_target(context=context)
        boundary_targets = {
            target
            for target in period_to_target.values()
            if target != context.regime_name
        }
        # The boundary step reads exactly one target regime's continuation (the
        # retirement boundary). More than one distinct boundary target has no
        # single well-defined prefix, so fail loud rather than pick one by set
        # iteration order.
        if len(boundary_targets) > 1:
            msg = (
                f"TwoDimEGM regime '{context.regime_name}' leaves to more than one "
                f"target regime ({sorted(boundary_targets)}); the boundary step "
                f"supports a single continuation target."
            )
            raise RegimeInitializationError(msg)
        for target in boundary_targets:
            target_states = context.regime_to_v_interpolation_info[
                target
            ].continuous_states
            if len(target_states) != 1:
                msg = (
                    f"TwoDimEGM regime '{context.regime_name}' leaves to target "
                    f"regime '{target}', whose continuous states are "
                    f"{sorted(target_states)}. The boundary step pays the pension "
                    f"out as a lump sum into a single continuous state, so the "
                    f"target must declare exactly one."
                )
                raise RegimeInitializationError(msg)

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
        # `validate` has already established there is at most one of these.
        boundary_targets = {
            target for target in period_to_target.values() if target != own_name
        }
        boundary_prefix = next(iter(boundary_targets), own_name)
        # The boundary target's own name for the state the pension is paid into.
        # It is resolved from that regime, not assumed to match this regime's
        # liquid role: the two are different regimes and may name it differently.
        boundary_liquid_state = (
            next(
                iter(
                    context.regime_to_v_interpolation_info[
                        boundary_prefix
                    ].continuous_states
                )
            )
            if boundary_targets
            else self.liquid_state
        )
        # The step always works in its own `(liquid, pension)` axis order, while
        # the regime's value array follows the order the states were declared in.
        # `validate` has established the two are a permutation of each other.
        own_liquid_grid = context.grids[self.liquid_state].to_jax()
        own_pension_grid = context.grids[self.pension_state].to_jax()
        publishes_pension_first = tuple(context.state_action_space.state_names) == (
            self.pension_state,
            self.liquid_state,
        )
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
                    liquid_state=self.liquid_state,
                    pension_state=self.pension_state,
                    boundary_liquid_state=boundary_liquid_state,
                )
                cores[is_boundary] = jax.jit(core) if context.enable_jit else core
            period_kernels[period] = _TwoDimEGMPeriodKernel(
                core=cores[is_boundary],
                regime_name=own_name,
                continuation_target=target,
                is_boundary=is_boundary,
                transition_target_names=tuple(context.transitions),
                liquid_state=self.liquid_state,
                pension_state=self.pension_state,
                publishes_pension_first=publishes_pension_first,
                next_liquid_grid=target_period_grid(
                    context=context,
                    period=period,
                    target=target,
                    target_state_name=self.liquid_state,
                )
                if not is_boundary
                else own_liquid_grid,
                next_pension_grid=target_period_grid(
                    context=context,
                    period=period,
                    target=target,
                    target_state_name=self.pension_state,
                )
                if not is_boundary
                else own_pension_grid,
                next_boundary_liquid_grid=target_period_grid(
                    context=context,
                    period=period,
                    target=target,
                    target_state_name=boundary_liquid_state,
                )
                if is_boundary
                else own_liquid_grid,
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

    liquid_state: StateName
    """The regime's own name for the state filling the kernel's liquid role."""

    pension_state: StateName
    """The regime's own name for the state filling the kernel's pension role."""

    next_liquid_grid: Float1D
    """The interior target's liquid nodes at `period + 1`.

    Unused on the boundary branch, which reads a 1-D continuation instead.
    """

    next_pension_grid: Float1D
    """The interior target's pension nodes at `period + 1`."""

    next_boundary_liquid_grid: Float1D
    """The boundary target's sole continuous nodes at `period + 1`.

    The abscissae of the retired value and marginal the boundary step reads
    through the lump-sum payout. Unused on the interior branch.
    """

    publishes_pension_first: bool
    """Whether the regime declares its pension state before its liquid state.

    The step returns its value on the `(liquid, pension)` grid; the regime's
    value array follows the declaration order. When the two disagree the result
    is transposed on the way out, so a regime's axis order is its own business.
    """

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
        if self.publishes_pension_first:
            V_arr = jnp.swapaxes(V_arr, 0, 1)
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
                "next_boundary_liquid": self.next_boundary_liquid_grid,
                "next_value_retired": next_regime_to_V_arr[self.continuation_target],
                "next_marginal_retired": next_regime_to_continuation[
                    self.continuation_target
                ].marginal_utility,
            }
        else:
            continuation = {
                "next_liquid": self.next_liquid_grid,
                "next_pension": self.next_pension_grid,
                "next_value_working": next_regime_to_V_arr[self.continuation_target],
            }
        return {
            "liquid": states[self.liquid_state],
            "pension": states[self.pension_state],
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
    liquid_state: StateName,
    pension_state: StateName,
    boundary_liquid_state: StateName,
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

    Each law's parameters are qualified by the state it moves, under that
    regime's own name for it: this regime's `liquid_state` / `pension_state`
    for its own laws, and the boundary target's `boundary_liquid_state` for
    the payout law it reads.
    """
    boundary_liquid_law = f"{boundary_prefix}__next_{boundary_liquid_state}"
    interior_liquid_law = f"{interior_prefix}__next_{liquid_state}"
    interior_pension_law = f"{interior_prefix}__next_{pension_state}"
    from _lcm.egm.rfc_two_asset_step import rfc_two_asset_step  # noqa: PLC0415
    from _lcm.egm.two_asset_g2egm_step import (  # noqa: PLC0415
        g2egm_retiring_step,
        g2egm_step,
    )

    def boundary_core(
        *,
        liquid: Float1D,
        pension: Float1D,
        next_boundary_liquid: Float1D,
        next_value_retired: Float1D,
        next_marginal_retired: Float1D,
        **params: FloatND,
    ) -> FloatND:
        result = g2egm_retiring_step(
            next_value_retired=next_value_retired,
            next_marginal_retired=next_marginal_retired,
            liquid_grid=next_boundary_liquid,
            m_grid=liquid,
            n_grid=pension,
            a_grid=a_grid,
            b_grid=b_grid,
            consumption_grid=consumption_grid,
            discount_factor=params["H__discount_factor"],
            crra=params["utility__crra"],
            match_rate=params[f"{boundary_liquid_law}__match_rate"],
            return_liquid=params[f"{boundary_liquid_law}__return_liquid"],
            pension_payout_return=params[
                f"{boundary_liquid_law}__pension_payout_return"
            ],
            retirement_income=params[f"{boundary_liquid_law}__retirement_income"],
            threshold=threshold,
        )
        return result.value - params["utility__work_disutility"]

    def interior_core(
        *,
        liquid: Float1D,
        pension: Float1D,
        next_liquid: Float1D,
        next_pension: Float1D,
        next_value_working: FloatND,
        **params: FloatND,
    ) -> FloatND:
        discount_factor = params["H__discount_factor"]
        crra = params["utility__crra"]
        match_rate = params[f"{interior_pension_law}__match_rate"]
        return_liquid = params[f"{interior_liquid_law}__return_liquid"]
        return_pension = params[f"{interior_pension_law}__return_pension"]
        wage = params[f"{interior_liquid_law}__wage"]
        if upper_envelope == "rfc":
            result = rfc_two_asset_step(
                next_value=next_value_working,
                m_grid=liquid,
                n_grid=pension,
                next_m_grid=next_liquid,
                next_n_grid=next_pension,
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
                next_m_grid=next_liquid,
                next_n_grid=next_pension,
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
