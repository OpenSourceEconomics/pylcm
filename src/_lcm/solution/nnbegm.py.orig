"""The N-NB-EGM solver: nested outer search around an NB-EGM inner solve.

`NNBEGM` runs the NEGM-style outer keeper/adjuster search over a durable
margin with an inner `NBEGM` consumption-saving solve, so declared liquid
kinks, jumps, and hard constraints keep their exact NB-EGM treatment inside
every outer candidate. `NNBEGMInnerSpec` normalizes the 1-D inner solver
config (DC-EGM or NB-EGM vocabulary) so outer kernel code never dispatches
on the inner type.

The kernel-building imports are function-local so the public `lcm.solvers`
façade stays a thin re-export that pulls in no numerical engine modules.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.egm.carry import EGMCarry
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SolutionKernels,
    Solver,
    SolverBuildContext,
)
from _lcm.solution.dcegm import DCEGM
from _lcm.solution.nbegm import NBEGM
from _lcm.solution.negm import (
    _fail_if_outer_batch_size_negative,
    _fail_if_outer_grid_is_stochastic,
    _no_adjustment_outer_transition,
    _strip_outer_transition,
    _with_no_adjustment_outer_function,
    _with_outer_post_decision,
)
from _lcm.typing import FlatParams, RegimeName
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ActionName, FloatND, FunctionName, StateName


@dataclass(frozen=True, kw_only=True)
class NNBEGMInnerSpec:
    """Normalized view of a 1-D inner EGM solver config for nested outer search.

    A nested outer-search solver wraps a one-dimensional inner EGM solver and
    needs its Euler-state slots under one vocabulary. The two inner families
    name the budget node differently and NBEGM leaves two slots inferable
    standalone; the spec makes every slot explicit so outer kernel code never
    dispatches on the inner type.
    """

    solver: Solver
    """The inner solver config itself."""
    continuous_state: StateName
    """The inner Euler (liquid) state."""
    post_decision_function: FunctionName
    """The inner post-decision (savings) node."""
    budget_target: FunctionName
    """The inner budget node (DCEGM `resources`, NBEGM `budget_target`)."""
    savings_grid: ContinuousGrid
    """The inner exogenous post-decision grid."""


def get_nnbegm_inner_spec(*, inner: Solver) -> NNBEGMInnerSpec:
    """Normalize an inner EGM solver config into a `NNBEGMInnerSpec`.

    Inside a nested solver the regime carries two continuous states, so the
    single-continuous-state inference NBEGM applies standalone is ambiguous —
    both `continuous_state` and `post_decision_function` must be explicit.

    Args:
        inner: The inner 1-D EGM solver config; must be `DCEGM` or `NBEGM`.

    Returns:
        The normalized spec.

    Raises:
        RegimeInitializationError: If an NBEGM inner leaves `continuous_state`
            or `post_decision_function` to inference.
        TypeError: If `inner` is not a 1-D EGM solver config.

    """
    match inner:
        case DCEGM():
            return NNBEGMInnerSpec(
                solver=inner,
                continuous_state=inner.continuous_state,
                post_decision_function=inner.post_decision_function,
                budget_target=inner.resources,
                savings_grid=inner.savings_grid,
            )
        case NBEGM():
            if inner.continuous_state is None:
                msg = (
                    "An NNBEGM inner requires an explicit "
                    "`continuous_state`: the regime carries two continuous "
                    "states, so the single-state inference is ambiguous."
                )
                raise RegimeInitializationError(msg)
            if inner.post_decision_function is None:
                msg = (
                    "An NNBEGM inner requires an explicit "
                    "`post_decision_function` naming the inner savings node."
                )
                raise RegimeInitializationError(msg)
            return NNBEGMInnerSpec(
                solver=inner,
                continuous_state=inner.continuous_state,
                post_decision_function=inner.post_decision_function,
                budget_target=inner.budget_target,
                savings_grid=inner.savings_grid,
            )
        case _:
            msg = (
                "A nested inner solver must be DCEGM or NBEGM, got "
                f"{type(inner).__name__}."
            )
            raise TypeError(msg)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NNBEGM(Solver):
    """N-NB-EGM — an outer durable grid search over inner 1-D NB-EGM solves.

    The regime carries two continuous margins. The outer post-decision margin
    (a durable/illiquid stock) is selected by a finite search: a *keeper* holds
    it unchanged for free, and an *adjuster* sweep binds it to each exogenous
    outer-grid node. Conditional on the outer node, the remaining problem is a
    one-dimensional consumption-saving solve on the liquid state, handled by
    the inner `NBEGM` config — so declared liquid kinks, jumps, and hard
    constraints keep their exact NB-EGM treatment inside every outer candidate.

    The outer axis is collapsed by `V = max(V_keeper, max_j W_j)`; the solution
    is exact relative to the finite outer candidate set (grid plus keeper). The
    published continuation is the pointwise upper envelope of the candidates'
    carry rows on the shared liquid state grid — a finite-grid (bridged) outer
    envelope, so the inner config must not publish jump-topology rows.

    No outer Euler condition is assumed: adjustment frictions and caps make a
    second Euler inversion unreliable, which is the reason to nest rather than
    to solve two coupled first-order conditions (that case belongs to
    `TwoDimEGM`).
    """

    inner: NBEGM
    """The inner 1-D NB-EGM config solved per outer candidate.

    `continuous_state` and `post_decision_function` must be explicit — the
    regime carries two continuous states, so the standalone single-state
    inference is ambiguous.
    """

    outer_action: ActionName
    """The regime's outer continuous action (e.g. the durable investment)."""

    outer_post_decision: FunctionName
    """The outer post-decision function fixed at each outer node.

    The auto-generated transition name of the durable state
    (`next_<durable>`); the inner budget reads it as a bound constant.
    """

    outer_grid: ContinuousGrid
    """Exogenous candidate grid for the outer post-decision margin."""

    outer_no_adjustment_candidate: FunctionName | None = None
    """State-dependent no-adjustment map `s' = keep(s)` the keeper holds.

    `None` keeps the durable stock unchanged (identity)."""

    outer_batch_size: int = 0
    """Outer-grid nodes solved per chunk before folding into the running
    maximum; `0` solves every node at once. A memory knob only —
    value-invariant."""

    def __post_init__(self) -> None:
        spec = get_nnbegm_inner_spec(inner=self.inner)
        _fail_if_outer_grid_is_stochastic(self.outer_grid)
        _fail_if_nnbegm_outer_post_decision_is_inner(
            outer_post_decision=self.outer_post_decision,
            inner_post_decision=spec.post_decision_function,
        )
        _fail_if_outer_batch_size_negative(self.outer_batch_size)

    @property
    def requires_continuation(self) -> bool:
        """NNBEGM runs an inner NB-EGM solve that inverts the Euler equation."""
        return True

    @property
    def carry_retains_discrete_action_rows(self) -> bool:
        """The inner NB-EGM merges discrete branches inside its envelope."""
        return self.inner.carry_retains_discrete_action_rows

    @property
    def carry_rows_share_state_grid(self) -> bool:
        """The inner NB-EGM publishes carry rows on the shared state grid."""
        return self.inner.carry_rows_share_state_grid

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one nested period adapter per period, wrapping inner kernels.

        Mirrors the NEGM keeper/adjuster split with an NB-EGM inner:

        - the *adjuster* strips the outer post-decision transition and admits
          the outer value as a flat param bound per outer-grid node;
        - the *keeper* injects `next_<durable> = keep(<durable>)` into the
          transitions and the econ functions, so the durable becomes a genuine
          passive ride-along state.
        """
        adjuster_context = replace(
            context,
            transitions=_strip_outer_transition(
                transitions=context.transitions,
                outer_post_decision=self.outer_post_decision,
            ),
            flat_param_names=context.flat_param_names | {self.outer_post_decision},
        )
        adjuster_kernels = self.inner.build_period_kernels(context=adjuster_context)
        no_adjustment_func = (
            context.functions[self.outer_no_adjustment_candidate]
            if self.outer_no_adjustment_candidate is not None
            else None
        )
        keeper_context = replace(
            context,
            transitions=_no_adjustment_outer_transition(
                transitions=context.transitions,
                outer_post_decision=self.outer_post_decision,
                no_adjustment_func=no_adjustment_func,
            ),
            functions=_with_no_adjustment_outer_function(
                functions=context.functions,
                outer_post_decision=self.outer_post_decision,
                no_adjustment_func=no_adjustment_func,
            ),
        )
        keeper_kernels = self.inner.build_period_kernels(context=keeper_context)
        template = keeper_kernels.continuation_template
        _fail_if_nnbegm_carry_publishes_topology_rows(template=template)
        outer_grid_values = self.outer_grid.to_jax()
        period_kernels = MappingProxyType(
            {
                period: _NNBEGMPeriodKernel(
                    keeper_kernel=keeper_kernels.period_kernels[period],
                    adjuster_kernel=adjuster_kernel,
                    regime_name=context.regime_name,
                    outer_grid_values=outer_grid_values,
                    outer_post_decision=self.outer_post_decision,
                    outer_batch_size=self.outer_batch_size,
                )
                for period, adjuster_kernel in (adjuster_kernels.period_kernels.items())
            }
        )
        # The bridged outer envelope folds candidates pointwise on the shared
        # liquid state grid, so the published rows keep the keeper's shape —
        # no carry widening.
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_template=template,
        )


@dataclass(frozen=True, kw_only=True)
class _NNBEGMPeriodKernel:
    """The NNBEGM period adapter — a keeper plus an adjuster outer sweep.

    Holds two inner NB-EGM period adapters and the exogenous outer grid. Each
    inner adapter can expose several independently-traced cores (the ride-along
    NB-EGM kernel splits into a continuation and an envelope core), so the
    nested adapter republishes every inner core under a `keeper:`/`adjuster:`
    prefix and strips the prefix when delegating.

    Calling it runs the keeper once and the adjuster once per outer-grid node,
    collapses the outer axis by `V = max(V_keeper, max_j W_j)`, and publishes
    the pointwise (bridged) upper envelope of the candidates' carry rows on the
    shared liquid state grid. The adapter is non-jitted: it dispatches the
    shared jitted inner cores, matching `_NEGMPeriodKernel`.
    """

    keeper_kernel: PeriodKernel
    """The keeper inner adapter — a passive per-durable-state NB-EGM."""

    adjuster_kernel: PeriodKernel
    """The adjuster inner adapter whose shared jitted cores are swept."""

    regime_name: RegimeName
    """Name of the regime whose flat params the outer node binds into."""

    outer_grid_values: FloatND
    """Exogenous grid over the outer post-decision margin `s'`."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision function bound per outer-grid node."""

    outer_batch_size: int
    """Outer-grid nodes solved per chunk before folding into the running
    maximum; `0` solves every node at once."""

    @property
    def core(self) -> Callable:
        """The adjuster's primary core, exposed for any single-core reader."""
        return self.adjuster_kernel.core

    def cores(self) -> Mapping[str, Callable]:
        """Return every inner core under a `keeper:`/`adjuster:` prefix.

        The keeper and adjuster are distinct traced programs built from
        different contexts, and each inner adapter may expose several cores of
        its own; prefixing keeps every (role, inner-core) pair under its own
        AOT compilation key.
        """
        return MappingProxyType(
            {
                **{
                    f"keeper:{name}": core
                    for name, core in self.keeper_kernel.cores().items()
                },
                **{
                    f"adjuster:{name}": core
                    for name, core in self.adjuster_kernel.cores().items()
                },
            }
        )

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _NNBEGMPeriodKernel:
        """Bind the regime's fixed params into both inner kernels."""
        return replace(
            self,
            keeper_kernel=self.keeper_kernel.with_fixed_params(
                fixed_flat_params=fixed_flat_params
            ),
            adjuster_kernel=self.adjuster_kernel.with_fixed_params(
                fixed_flat_params=fixed_flat_params
            ),
        )

    def build_lower_args(
        self,
        *,
        core_key: str,
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Delegate the named inner core's lowering arguments.

        The prefix selects the role; the remainder is the inner adapter's own
        core key. The adjuster binds `outer_post_decision` at the first
        outer-grid node so its lowered program matches the shape every per-node
        call traces; the keeper lowers with no outer binding.
        """
        role, inner_key = core_key.split(":", maxsplit=1)
        if role == "keeper":
            return self.keeper_kernel.build_lower_args(
                core_key=inner_key,
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        return self.adjuster_kernel.build_lower_args(
            core_key=inner_key,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=_with_outer_post_decision(
                flat_params=flat_params,
                regime_name=self.regime_name,
                outer_post_decision=self.outer_post_decision,
                value=self.outer_grid_values[0],
            ),
            period=period,
            ages=ages,
        )

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Run keeper and adjuster sweep, collapse by `max`, fold the carry.

        The keeper's carry rows and every adjuster's carry rows live on the
        shared liquid state grid, so the outer envelope is a pointwise maximum
        per row entry — value and marginal follow the winning candidate. `max`
        is associative, so the chunked fold is value-identical to a single
        stacked maximum regardless of `outer_batch_size`.
        """
        keeper_result = self.keeper_kernel(
            compiled_cores=_subcores(compiled_cores=compiled_cores, role="keeper"),
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        V_arr = keeper_result.V_arr
        carry = cast("EGMCarry", keeper_result.continuation)
        adjuster_cores = _subcores(compiled_cores=compiled_cores, role="adjuster")
        nodes = list(self.outer_grid_values)
        chunk_size = self.outer_batch_size or len(nodes)
        for chunk_start in range(0, len(nodes), chunk_size):
            chunk_results = [
                self.adjuster_kernel(
                    compiled_cores=adjuster_cores,
                    state_action_space=state_action_space,
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_continuation=next_regime_to_continuation,
                    flat_params=_with_outer_post_decision(
                        flat_params=flat_params,
                        regime_name=self.regime_name,
                        outer_post_decision=self.outer_post_decision,
                        value=node,
                    ),
                    period=period,
                    ages=ages,
                )
                for node in nodes[chunk_start : chunk_start + chunk_size]
            ]
            for adjuster_result in chunk_results:
                # `fmax`, not `maximum`: the inner NB-EGM NaN-dead masks cells
                # an outer node makes infeasible, and one infeasible candidate
                # must not poison a cell another candidate solves. A cell stays
                # NaN only when every candidate is infeasible there.
                V_arr = jnp.fmax(V_arr, adjuster_result.V_arr)
                carry = _fold_bridged_outer_carry(
                    running=carry,
                    candidate=cast("EGMCarry", adjuster_result.continuation),
                )
            # Force the running maximum to device before the next chunk so the
            # lazy fold's peak stays bounded to one chunk of candidates and the
            # chunk's independent solves can overlap.
            V_arr, carry = jax.block_until_ready((V_arr, carry))
        # The simulate phase re-optimizes the outer durable action by grid
        # argmax over the next-period value array, so the keeper's published
        # simulation policy rides through unchanged.
        return KernelResult(
            V_arr=V_arr,
            continuation=carry,
            simulation_policy=keeper_result.simulation_policy,
        )


def _subcores(
    *, compiled_cores: Mapping[str, Callable], role: str
) -> Mapping[str, Callable]:
    """Select one role's inner cores, stripping the `role:` prefix."""
    token = f"{role}:"
    return MappingProxyType(
        {
            key.removeprefix(token): core
            for key, core in compiled_cores.items()
            if key.startswith(token)
        }
    )


def _fold_bridged_outer_carry(*, running: EGMCarry, candidate: EGMCarry) -> EGMCarry:
    """Fold one adjuster candidate into the running bridged outer envelope.

    Every candidate's carry rows live on the shared liquid state grid, so the
    outer envelope is a pointwise maximum: where the candidate's value row
    beats the running one, its value and marginal replace them. The row
    abscissae and the taste-shock scale are shared, so they ride through.
    NaN-dead cells never win, and a candidate that solves a cell the running
    envelope holds as NaN-dead takes it over.
    """
    take = (candidate.value > running.value) | (
        jnp.isnan(running.value) & ~jnp.isnan(candidate.value)
    )
    return replace(
        running,
        value=jnp.where(take, candidate.value, running.value),
        marginal_utility=jnp.where(
            take, candidate.marginal_utility, running.marginal_utility
        ),
    )


def _fail_if_nnbegm_outer_post_decision_is_inner(
    *, outer_post_decision: FunctionName, inner_post_decision: FunctionName
) -> None:
    if outer_post_decision == inner_post_decision:
        msg = (
            f"NNBEGM.outer_post_decision '{outer_post_decision}' coincides "
            f"with the inner NB-EGM post-decision function "
            f"'{inner_post_decision}'. The outer post-decision (the next-period "
            "durable stock) and the inner post-decision (the liquid savings) "
            "must be distinct functions."
        )
        raise RegimeInitializationError(msg)


def _fail_if_nnbegm_carry_publishes_topology_rows(
    *, template: ContinuationPayload | None
) -> None:
    if isinstance(template, EGMCarry) and template.breakpoints is not None:
        msg = (
            "NNBEGM publishes a bridged (pointwise, finite-grid) outer "
            "envelope, which cannot represent the inner config's jump-topology "
            "rows. Use `jump_read='bridged'` on the inner NBEGM or remove the "
            "declared jump breakpoints."
        )
        raise RegimeInitializationError(msg)
