"""The N-NB-EGM solver: nested outer search around an NB-EGM inner solve.

`NNBEGM` runs the NEGM-style outer keeper/adjuster search over a durable
margin with an inner `NBEGM` consumption-saving solve, so declared liquid
kinks, jumps, and hard constraints keep their exact NB-EGM treatment inside
every outer candidate. The regime owns both margins' DAG role names; the
public solver contains numerical configuration only, and a private bound
companion carries the resolved names into the kernels.

The kernel-building imports are function-local so the public `lcm.solvers`
façade stays a thin re-export that pulls in no numerical engine modules.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.bounds import without_proved_lower_bounds
from _lcm.constraints.capabilities import ConstraintCapabilities
from _lcm.constraints.processed import normalize_constraints
from _lcm.continuation import EGMContinuationLayout, EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid, Grid
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    SolverModelContext,
    TwoMarginSolver,
    _BoundLiquidMargin,
    _BoundOuterContinuousMargin,
)
from _lcm.solution.nbegm import NBEGM, _BoundNBEGM, proved_post_decision_of
from _lcm.solution.negm import (
    _fail_if_outer_batch_size_negative,
    _fail_if_outer_grid_is_stochastic,
    _with_no_adjustment_outer_function,
    _with_outer_post_decision,
    _without_outer_post_decision,
)
from _lcm.typing import FlatParams, RegimeName
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.typing import ActionName, FloatND, FunctionName, StateName, UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NNBEGM(TwoMarginSolver):
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
    the two-continuous-state solver published with its own paper).
    """

    inner: NBEGM
    """Numerical configuration of the inner 1-D NB-EGM solve."""

    outer_grid: ContinuousGrid
    """Exogenous candidate grid for the outer post-decision margin."""

    outer_batch_size: int = 0
    """Outer-grid nodes solved per chunk before folding into the running
    maximum; `0` solves every node at once. A memory knob only —
    value-invariant."""

    def __post_init__(self) -> None:
        _fail_if_inner_is_not_nbegm(self.inner)
        _fail_if_outer_grid_is_stochastic(self.outer_grid)
        _fail_if_outer_batch_size_negative(self.outer_batch_size, solver_name="NNBEGM")

    def _with_margins(
        self,
        *,
        liquid: _BoundLiquidMargin,
        outer: _BoundOuterContinuousMargin,
    ) -> _BoundNNBEGM:
        """Bind both regime-owned margins into a private runtime config."""
        kwargs = {
            field.name: getattr(self, field.name)
            for field in fields(NNBEGM)
            if field.name != "inner"
        }
        inner = self.inner._with_liquid_margin(liquid)  # noqa: SLF001
        return _BoundNNBEGM(
            **kwargs,
            inner=inner,
            outer_action=outer.action,
            outer_state=outer.state,
            outer_post_decision=outer.post_decision_state,
            outer_no_adjustment_candidate=outer.no_adjustment,
        )

    @property
    def requires_continuation(self) -> bool:
        """NNBEGM runs an inner NB-EGM solve that inverts the Euler equation."""
        return True

    @property
    def supports_nonlinear_certainty_equivalent(self) -> bool:
        """The inner NB-EGM solve inverts the recursive Euler equation."""
        return self.inner.supports_nonlinear_certainty_equivalent

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """The bridged outer envelope republishes the inner solver's rows."""
        return self.inner.egm_continuation_layout

    @property
    def constraint_capabilities(self) -> ConstraintCapabilities:
        """What this kernel can do with a declared constraint.

        The inner case-piece solve is where a liquid constraint would have to be
        evaluated, and it evaluates none, so the nested solver inherits the
        inner declaration rather than restating it.
        """
        return self.inner.constraint_capabilities

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Validate the user-level nested NB-EGM contract for this regime.

        A declared lower bound on the inner post-decision state is admitted:
        it states the number the inner savings grid already enforces, so it is
        proved against that grid and leaves no predicate for either margin to
        evaluate. Which declarations qualify is asked of the same function that
        drops them from the engine's constraint set, so the exemption here and
        the drop there cannot come to disagree.

        Keep it that way: a local test for the bound's shape here would be a
        second spelling of a question that already has an answer, and the two
        would drift without a symptom. A bound exempted here but not dropped
        there reaches the engine's constraint set, which is built per discrete
        combo — no place a continuous post-decision state can be read.
        """
        from _lcm.egm.nnbegm_validation import (  # noqa: PLC0415
            validate_nnbegm_regime,
        )
        from _lcm.egm.validation import (  # noqa: PLC0415
            fail_if_declared_lower_bound_disagrees_with_the_grid,
            fail_if_kernel_grids_withhold_their_points,
        )

        user_regime = context.user_regimes[context.regime_name]
        validate_nnbegm_regime(
            regime_name=context.regime_name,
            user_regime=user_regime,
        )
        bound = cast("_BoundNNBEGM", self)
        outer_state = bound.outer_state
        liquid = bound.inner.continuous_state
        fail_if_kernel_grids_withhold_their_points(
            grids={
                "outer grid": bound.outer_grid,
                "inner savings grid": bound.inner.savings_grid,
                f"grid of the outer state '{outer_state}'": cast(
                    "Grid", user_regime.states[outer_state]
                ),
                f"grid of the liquid state '{liquid}'": cast(
                    "Grid", user_regime.states[liquid]
                ),
            },
            regime_name=context.regime_name,
            solver_name="NNBEGM",
        )
        fail_if_declared_lower_bound_disagrees_with_the_grid(
            regime_name=context.regime_name,
            user_regime=user_regime,
            solver=bound.inner,
            solver_name="NNBEGM",
        )
        unenforceable = without_proved_lower_bounds(
            # `Phased` is rejected in the constraints slot by the phase
            # grammar, so every value here is a bare declaration.
            constraints=normalize_constraints(
                constraints=cast(
                    "Mapping[FunctionName, UserFunction]", user_regime.constraints
                )
            ),
            proved_post_decision=proved_post_decision_of(solver=self.inner),
        )
        if unenforceable:
            constraint_names = sorted(unenforceable)
            msg = (
                f"NNBEGM regime '{context.regime_name}' declares constraints "
                f"{constraint_names}. The inner NB-EGM solve inverts the Euler "
                "equation and the outer sweep scores an exogenous grid, so no "
                "user constraint is evaluated in either margin; encode the "
                "borrowing limit in the first node of the inner `savings_grid` "
                "and the budget identity in the post-decision function, or use "
                "GridSearch."
            )
            raise ModelInitializationError(msg)

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Apply the inner solver's build-time gates to the liquid margin.

        The inner NB-EGM kernels run unchanged inside every outer candidate, so
        a piece that hides branching breaks the inner Euler inversion here
        exactly as it would under a bare `NBEGM`, and declared taste shocks are
        ignored by the inner envelopes here exactly as they would be there. The
        smoothness gate is pointed at the inner spec's Euler state rather than
        the regime's first state, because the regime also carries the outer
        margin the pieces never see.
        """
        from _lcm.solution.nbegm import (  # noqa: PLC0415
            fail_if_taste_shocks_declared,
            validate_case_piece_smoothness,
        )

        bound = cast("_BoundNNBEGM", self)
        fail_if_taste_shocks_declared(context=context)
        validate_case_piece_smoothness(
            context=context,
            liquid_state_name=bound.inner.continuous_state,
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        r"""Build one nested period adapter per period, wrapping inner kernels.

        Mirrors the NEGM keeper/adjuster split with an NB-EGM inner:

        - the *adjuster* strips the outer post-decision transition and admits
          the outer value as a flat param bound per outer-grid node;
        - the *keeper* injects $s_t^\textit{post-dec} = keep(\textit{durable}_t)$
          into the econ functions, so the durable becomes a genuine passive
          ride-along state.
        """
        bound = cast("_BoundNNBEGM", self)
        # The adjuster's outer post-decision arrives per outer-grid node as a
        # bound param, so the function declaring the chosen stock leaves the
        # inner DAG — leaving it in would let the inner scope check walk through
        # it to the outer action, which is exactly what binding the node
        # removes.
        #
        # The durable's own law of motion stays exactly as the regime declares
        # it. It reads the post-decision, which is that bound leaf here, so it
        # is decision-independent without being replaced — and a declared
        # `next_<durable>` $= (1 - \delta)\, s_t^\textit{post-dec}$ is therefore
        # the stock the continuation is read at, not the raw node the outer
        # search picked.
        adjuster_context = replace(
            context,
            functions=_without_outer_post_decision(
                functions=context.functions,
                outer_post_decision=bound.outer_post_decision,
            ),
            flat_param_names=context.flat_param_names | {bound.outer_post_decision},
        )
        adjuster_kernels = bound.inner.build_period_kernels(context=adjuster_context)
        no_adjustment_func = (
            context.functions[bound.outer_no_adjustment_candidate]
            if bound.outer_no_adjustment_candidate is not None
            else None
        )
        # The keeper computes the post-decision from the durable leaf instead of
        # taking it as a bound param, so the declared law again stands and what
        # the keeper carries is `next_<durable>(keep(<durable>))`.
        keeper_context = replace(
            context,
            functions=_with_no_adjustment_outer_function(
                functions=context.functions,
                durable_state=bound.outer_state,
                outer_post_decision=bound.outer_post_decision,
                no_adjustment_func=no_adjustment_func,
            ),
        )
        keeper_kernels = bound.inner.build_period_kernels(context=keeper_context)
        template = keeper_kernels.continuation_template
        _fail_if_inner_carry_rows_not_grid_aligned(inner=bound.inner)
        _fail_if_nnbegm_carry_publishes_topology_rows(template=template)
        outer_grid_values = self.outer_grid.to_jax()
        period_kernels = MappingProxyType(
            {
                period: _NNBEGMPeriodKernel(
                    keeper_kernel=keeper_kernels.period_kernels[period],
                    adjuster_kernel=adjuster_kernel,
                    regime_name=context.regime_name,
                    outer_grid_values=outer_grid_values,
                    outer_post_decision=bound.outer_post_decision,
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
            continuation_spec=(
                None
                if template is None
                else EGMContinuationSpec(
                    template=template,
                    layout=self.egm_continuation_layout,
                )
            ),
            # Both inner margins are solved by the inner solver, so both sets of
            # parameter-dependent preconditions still apply to this regime.
            param_checks=(
                *adjuster_kernels.param_checks,
                *keeper_kernels.param_checks,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _BoundNNBEGM(NNBEGM):
    """Internal N-NB-EGM config with both regime margins resolved."""

    inner: _BoundNBEGM
    outer_action: ActionName
    outer_state: StateName
    outer_post_decision: FunctionName
    outer_no_adjustment_candidate: FunctionName | None


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
    r"""Exogenous grid over the outer post-decision margin $s_t^\textit{post-dec}$."""

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


def _fail_if_inner_is_not_nbegm(inner: object) -> None:
    """Enforce the public NNBEGM composition despite inert type stubs.

    The planned DCEGM-or-NBEGM inner unification belongs to the follow-on NEGM
    fold.  The current public NNBEGM solver is the NBEGM-specific composition,
    so accepting another object here would defer a structural error until
    private margin binding or kernel construction.
    """
    if not isinstance(inner, NBEGM):
        cls = type(inner)
        raise RegimeInitializationError(
            "NNBEGM.inner must be an NBEGM numerical configuration, got "
            f"{cls.__module__}.{cls.__qualname__}."
        )


def _fail_if_inner_carry_rows_not_grid_aligned(*, inner: Solver) -> None:
    """Refuse an inner solver whose carry rows do not sit on the state grid.

    The bridged outer envelope replaces `value` and `marginal_utility` per
    candidate and rides the keeper's `endog_grid` through unchanged, which is
    only correct when every candidate publishes rows at the same abscissae.
    """
    if not inner.egm_continuation_layout.rows_share_state_grid:
        msg = (
            f"NNBEGM's inner solver {type(inner).__name__} publishes carry rows "
            "off the shared state grid, but the bridged outer envelope folds "
            "candidates pointwise and reuses the keeper's `endog_grid`, so the "
            "folded rows would pair one candidate's values with another's "
            "abscissae. Use an inner solver whose "
            "`egm_continuation_layout.rows_share_state_grid` is True."
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
