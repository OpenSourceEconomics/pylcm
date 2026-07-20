"""The NEGM solver (nested endogenous grid method, Druedahl 2021).

`NEGM` nests a `DCEGM` inner solve: a per-durable-node passive keeper alongside
one adjuster sweep per exogenous outer post-decision node, whose conditional
carries are lifted into common cash-on-hand (via the declared outer cost) and
stacked; the parent read collapses the candidate axis by the exact query-side
maximum. `_NEGMPeriodKernel` carries two cores (`"keeper"` / `"adjuster"`)
wrapped from the inner DC-EGM kernels; the AOT contract lowers and dispatches
each by key.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype
from dags import concatenate_functions, get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.beartype_conf import REGIME_CONF
from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_envelope import build_stacked_outer_carry
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SolutionKernels,
    Solver,
    SolverBuildContext,
)
from _lcm.solution.dcegm import DCEGM
from _lcm.typing import (
    EconFunction,
    EconFunctionsMapping,
    FlatParams,
    RegimeName,
    TransitionFunction,
    TransitionFunctionsMapping,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError, RegimeInitializationError
from lcm.typing import (
    ActionName,
    ContinuousState,
    FloatND,
    FunctionName,
    StateName,
    StateOrActionName,
)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NEGM(Solver):
    """Nested-EGM solver: an outer grid search over a durable/illiquid margin.

    NEGM solves a model with one continuous margin the Euler equation cleanly
    inverts on (liquid consumption-savings) plus a second continuous margin
    that does not admit a clean inverse-Euler (a durable/illiquid stock with
    adjustment frictions) by nesting:

    - an *inner* standard 1-D DC-EGM solve of the consumption-savings problem,
      conditional on the outer margin being fixed (this is exactly the existing
      `DCEGM` kernel, with the outer margin entering inner resources and utility
      as a constant and indexing the child durable state);
    - an *outer* deterministic `max` over a grid of the outer post-decision
      margin plus mandatory kink candidates (the no-adjustment point `s' = s`,
      the floor corner).

    The outer step is a search, not a second inverse-Euler: the outer value is
    generically non-concave (adjustment-cost kink, floor corners), so a second
    EGM inversion would be invalid there.

    The `NEGM(inner=DCEGM(...), …)` composition makes "NEGM nests the 1-D
    DC-EGM" literal: it reuses every inner field and its upper-envelope backend,
    reuses `DCEGM.__post_init__` validation wholesale, and keeps the
    outer-margin contract in one place. The model-contract check
    `validate_negm_regimes` rejects, at `Model` construction, any model NEGM
    does not fit (no outer margin, a coupled-2-Euler pension shape, a
    taste-shock-ordering violation), naming the offending feature and the
    correct alternative solver.
    """

    inner: DCEGM
    """The inner 1-D DC-EGM config.

    Carries the liquid Euler state, the consumption action, the resources and
    post-decision functions, the savings grid, and the upper-envelope backend.
    Its `__post_init__` guards run on construction, so an invalid inner config
    is rejected before NEGM's own guards.
    """

    outer_action: ActionName
    """The outer continuous action — the durable/illiquid choice.

    Forbidden as the inner DC-EGM continuous action (the two margins must be
    distinct).
    """

    outer_post_decision: FunctionName
    """The outer post-decision function `s'` in `Regime.functions`.

    The inner `resources` and the child-state index both read its value as a
    constant. Forbidden as the inner DC-EGM post-decision function.
    """

    outer_grid: ContinuousGrid
    """Exogenous grid over the outer post-decision margin `s'`."""

    outer_no_adjustment_candidate: FunctionName | None = None
    """State-specific kink candidate (the no-adjustment point `s' = s`).

    Inserted per node because a fixed exogenous outer grid misses
    state-specific kinks. `None` only when the model provably has no adjustment
    kink.
    """

    outer_cost: FunctionName | None = None
    """The credited outer-cost function in `Regime.functions`, or `None`.

    The declared contract behind the stacked-carry lift. With a cost declared,
    the regime must NOT define the inner resources function itself: it defines
    the cost-free base `<inner.resources>_before_outer_cost`, and pylcm
    composes `resources = base - cost` at model build — so the resources'
    affine use of the cost (coefficient exactly `-1`) holds by construction,
    not by inference. Model build additionally enforces that the cost reads
    only the durable state, the outer post-decision, and params, and that the
    base does not read the outer post-decision. The per-cell cash-on-hand
    shift then derives directly from the declaration
    (`cost(z, z') - cost(z, keep(z))`). `None` declares the regime
    outer-cost-free: the regime defines the resources function directly, it
    must be independent of the outer post-decision (enforced), and every
    candidate already shares the keeper's cash-on-hand axis.
    """

    outer_batch_size: int = 0
    """Number of outer-grid nodes solved per chunk of the outer sweep.

    Bounds the *solve-side* chunk transients only: each chunk's per-node
    intermediate buffers are materialised and released together, so they never
    grow with the whole outer grid. It does not bound the period's peak, whose
    remaining candidate-scaled contributions chunking cannot remove:

    - the candidate *carries* are all retained — the published stacked
      continuation holds every outer candidate (`(A+1) * n_pad` grid slots per
      leading cell), inherent to the exact query-side outer maximum,
    - while the stack is built, the unstacked candidate carries and the
      stacked output coexist transiently,
    - the parent's continuation read prepares a search key of the full stacked
      shape and evaluates every candidate per query.

    A positive value processes that many nodes at once (their independent
    solves overlap); `0` (the default) solves every node at once — fastest,
    but its solve-side peak grows with the outer-grid size. It is a
    memory-vs-parallelism knob only: the solved value function is identical
    across batch sizes.
    """

    def __post_init__(self) -> None:
        _fail_if_outer_grid_is_stochastic(self.outer_grid)
        _fail_if_outer_action_is_inner_action(
            outer_action=self.outer_action, inner=self.inner
        )
        _fail_if_outer_post_decision_is_inner_post_decision(
            outer_post_decision=self.outer_post_decision, inner=self.inner
        )
        _fail_if_outer_batch_size_negative(self.outer_batch_size)

    @property
    def requires_continuation(self) -> bool:
        """NEGM nests a DC-EGM solve that inverts the Euler equation."""
        return True

    @property
    def n_stacked_carry_candidates(self) -> int:
        """The published carry stacks the keeper plus one row per outer node."""
        return int(self.outer_grid.to_jax().shape[0]) + 1

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one NEGM period adapter per period, wrapping the inner kernels.

        The standalone `validate_negm_regimes` model-contract check (run during
        regime processing) guarantees the outer margin is present and distinct
        from the inner margin, that the outer margin is not Euler-coupled to the
        inner state, and that any taste-shocked discrete choice is the outermost
        aggregation. The inner DC-EGM period kernels are built once (with the
        outer margin bound so it enters the inner resources and utility as a
        constant and indexes the child durable state); each is wrapped in an
        outer adapter that sweeps the outer grid plus the mandatory per-node
        candidates and collapses the outer axis by `max`.
        """
        # The adjuster is the inner DC-EGM with the outer post-decision
        # transition stripped: its value is supplied per outer-grid node
        # (`_with_outer_post_decision`), not recomputed from the outer action, so
        # the child-carry next-state function must not demand the outer action;
        # `read_child` sources the bound value from the combo pool instead.
        # `_with_outer_post_decision` binds `outer_post_decision` into the
        # regime's flat params at runtime, so the inner kernel reads it as a
        # bound param — admit it as a flat param at build time too, so the inner
        # scope check accepts the inner resources / utility reading it (the
        # service-flow `utility(serviced(next_<durable>))` pattern).
        adjuster_context = replace(
            context,
            transitions=_strip_outer_transition(
                transitions=context.transitions,
                outer_post_decision=self.outer_post_decision,
            ),
            flat_param_names=context.flat_param_names | {self.outer_post_decision},
        )
        adjuster_kernels = self.inner.build_period_kernels(context=adjuster_context)
        # The keeper is a normal passive DC-EGM: the outer post-decision is held
        # at its no-adjustment level (`next_<durable> = keep(<durable>)`), so the
        # durable becomes a genuine decision-independent passive state and
        # `credited(<durable>, keep(<durable>)) = 0` makes keeping free. The keeper
        # map is injected in two places that both read the outer post-decision: the
        # transitions (so the child read indexes the kept durable) and the econ
        # functions (so the inner resources DAG computes it from the durable leaf
        # rather than demanding it as a bound param, as the adjuster does). With no
        # `outer_no_adjustment_candidate`, `keep` is the identity (hold the stock);
        # a depreciating `keep(d) = d (1 - delta)` lands the kept stock off the
        # durable grid and the inner passive read blends it over the grid.
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
        outer_grid_values = self.outer_grid.to_jax()
        durable_state = self.outer_post_decision.removeprefix("next_")
        coh_shift_func = _build_coh_shift_function(
            functions=context.functions,
            durable_state_name=durable_state,
            outer_post_decision=self.outer_post_decision,
            no_adjustment_func=no_adjustment_func,
            outer_cost_name=self.outer_cost,
        )
        durable_grid_values = context.grids[durable_state].to_jax()
        period_kernels = MappingProxyType(
            {
                period: _NEGMPeriodKernel(
                    keeper_kernel=keeper_kernels.period_kernels[period],
                    adjuster_kernel=adjuster_kernel,
                    regime_name=context.regime_name,
                    outer_grid_values=outer_grid_values,
                    outer_post_decision=self.outer_post_decision,
                    coh_shift_func=coh_shift_func,
                    durable_grid_values=durable_grid_values,
                    outer_batch_size=self.outer_batch_size,
                )
                for period, adjuster_kernel in adjuster_kernels.period_kernels.items()
            }
        )
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_template=_stack_carry_template(
                template=keeper_kernels.continuation_template,
                n_candidates=outer_grid_values.shape[0] + 1,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _NEGMPeriodKernel:
    """The NEGM period adapter — a keeper plus an adjuster outer search.

    Holds two inner DC-EGM period adapters and the exogenous outer grid. The
    outer durable choice splits into two distinct traced programs:

    - the *keeper* — a per-durable-state passive DC-EGM (`next_illiquid =
      illiquid`, identity) that keeps the durable stock unchanged for free
      (`credited(s, s) = 0`), run once over the full durable grid; and
    - the *adjuster* — the inner DC-EGM with the outer transition stripped, run
      once per exogenous outer-grid node with `outer_post_decision` bound to that
      node as a constant.

    Calling it runs the keeper once and the adjuster once per outer-grid node,
    then collapses the stacked outer axis by `V = max(V_keeper, V_adjuster_sweep)`
    — the same shape brute would search, but with the inner consumption margin
    off-grid (the accuracy win). The adapter is non-jitted: it calls the shared
    jitted inner cores, matching `_DCEGMPeriodKernel`.
    """

    keeper_kernel: PeriodKernel
    """The keeper inner adapter — a passive per-durable-state DC-EGM."""

    adjuster_kernel: PeriodKernel
    """The adjuster inner adapter whose shared jitted core is swept."""

    regime_name: RegimeName
    """Name of the regime whose flat params the outer node binds into."""

    outer_grid_values: FloatND
    """Exogenous grid over the outer post-decision margin `s'`."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision function bound per outer-grid node."""

    coh_shift_func: Callable[..., FloatND]
    """Per-(durable, outer-node) cash-on-hand shift of each adjuster candidate.

    Maps the durable grid, the outer grid, and the regime's flat params to the
    shift matrix `credited(z, z'_j)` that lifts each adjuster's endogenous grid
    into the keeper's cash-on-hand axis.
    """

    durable_grid_values: FloatND
    """The durable state's grid — the carry's last leading (passive) axis.

    Any discrete/process states precede the passive durable margin in the carry's
    leading axes, so the durable is carry axis `-2`.
    """

    outer_batch_size: int
    """Outer-grid nodes solved per chunk of the outer sweep.

    `0` solves every node at once; a positive value bounds the *solve-side*
    chunk transients only. It does not bound the period's peak: every candidate
    carry stays resident for the stacked continuation, the unstacked candidate
    list and the stacked output coexist while the stack is built, and the
    parent read evaluates the full candidate axis per query. A
    memory-vs-parallelism knob only — value-invariant.
    """

    @property
    def core(self) -> Callable:
        """The shared jitted adjuster core, exposed for any single-core reader."""
        return self.adjuster_kernel.core

    def cores(self) -> Mapping[str, Callable]:
        """Return the keeper and adjuster inner cores, keyed independently.

        Each is a distinct traced DC-EGM program — the keeper is a passive
        per-durable-state solve (`next_illiquid = illiquid`, identity), the
        adjuster strips that transition and binds the outer node as a constant —
        so AOT compilation lowers and compiles each under its own key rather than
        collapsing both into one program.
        """
        return MappingProxyType(
            {
                "keeper": self.keeper_kernel.core,
                "adjuster": self.adjuster_kernel.core,
            }
        )

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _NEGMPeriodKernel:
        """Bind the regime's fixed params into both inner kernels and the shift.

        The cash-on-hand shift evaluates the regime's inner resources, which may
        read a fixed param, so the same fixed params removed from the live
        `flat_params` are bound into `coh_shift_func` as well.
        """
        regime_fixed = dict(
            fixed_flat_params.get(self.regime_name, MappingProxyType({}))
        )
        coh_shift_func = self.coh_shift_func
        if regime_fixed:
            coh_shift_func = functools.partial(coh_shift_func, **regime_fixed)
        return replace(
            self,
            keeper_kernel=self.keeper_kernel.with_fixed_params(
                fixed_flat_params=fixed_flat_params
            ),
            adjuster_kernel=self.adjuster_kernel.with_fixed_params(
                fixed_flat_params=fixed_flat_params
            ),
            coh_shift_func=coh_shift_func,
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

        The keeper is a normal passive DC-EGM, lowered straight off the inner
        kernel with no outer-node binding (its `next_illiquid = illiquid`
        identity sources the durable as a genuine passive state). The adjuster
        binds `outer_post_decision` into the regime's flat params at the first
        outer-grid node, so its lowered inner program matches the shape every
        per-node call traces; the outer axis is added outside the jitted core by
        the `__call__` sweep.
        """
        if core_key == "keeper":
            return self.keeper_kernel.build_lower_args(
                core_key="main",
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        return self.adjuster_kernel.build_lower_args(
            core_key="main",
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
        """Run keeper and adjuster sweep, collapse by `max`, assemble the result.

        The keeper runs the passive DC-EGM once, yielding the value of leaving the
        durable stock unchanged at every durable state. For each exogenous
        outer-grid node `s'_j`, the adjuster runs with `outer_post_decision` bound
        to `s'_j`, yielding `W_j`, the inner value over the liquid endogenous grid
        at durable `s'_j`. The outer axis is collapsed by
        `V = max(V_keeper, max_j W_j)`; the argmax over that stacked axis is the
        outer simulation policy.
        """
        keeper_result = self.keeper_kernel(
            compiled_cores={"main": compiled_cores["keeper"]},
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        # The published continuation retains every outer candidate: the keeper
        # and each adjuster carry, lifted into a common cash-on-hand axis and
        # stacked on a candidate axis. The parent period's continuation read
        # takes the exact `max_j V_j(q)` over that axis at its own query, so an
        # adjuster that wins strictly between keeper nodes is read at its true
        # value there — never bridged upward by interpolating a node-sampled
        # maximum.
        coh_shifts = self.coh_shift_func(
            durable_values=self.durable_grid_values,
            outer_values=self.outer_grid_values,
            **flat_params[self.regime_name],
        )
        # Sweep the adjuster outer-grid nodes in chunks of `outer_batch_size`:
        # each chunk's independent solves overlap, and blocking on the chunk
        # bounds the *solve*-side peak (the per-node intermediate buffers) to
        # one chunk. The candidate carries themselves are all retained — the
        # `(A+1) * n_pad` resident width is inherent to the exact query-side
        # maximum, not a fold artifact. The keeper and every adjuster are
        # DC-EGM kernels, so each always publishes a continuation carry.
        keeper_carry = cast("EGMCarry", keeper_result.continuation)
        V_arr = keeper_result.V_arr
        nodes = self._outer_nodes()
        adjuster_carries: list[EGMCarry] = []
        chunk_size = self.outer_batch_size or len(nodes)
        for chunk_start in range(0, len(nodes), chunk_size):
            chunk_results = [
                self.adjuster_kernel(
                    compiled_cores={"main": compiled_cores["adjuster"]},
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
                V_arr = jnp.maximum(V_arr, adjuster_result.V_arr)
                adjuster_carries.append(cast("EGMCarry", adjuster_result.continuation))
            # Force the chunk's results to device before the next chunk. Without
            # this the lazy sweep accumulates a dependency on every chunk's
            # solves at once — the solve-side peak would grow with the whole
            # outer grid rather than one chunk — and the chunk's independent
            # solves could not overlap.
            V_arr, _ = jax.block_until_ready((V_arr, adjuster_carries[chunk_start:]))
        carry = build_stacked_outer_carry(
            keeper_carry=keeper_carry,
            adjuster_carries=tuple(adjuster_carries),
            coh_shifts=coh_shifts,
        )
        # The simulate phase re-optimizes the outer durable action by grid argmax
        # over the next-period value array, so the published `sim_policy` (the
        # keeper's off-grid inner consumption function) is not the channel that
        # drives simulated durable choice; it rides through unchanged.
        return KernelResult(
            V_arr=V_arr,
            continuation=carry,
            simulation_policy=keeper_result.simulation_policy,
        )

    def _outer_nodes(self) -> list[FloatND]:
        """The exogenous outer post-decision nodes, each a scalar `s'_j`.

        The state-specific no-adjustment kink `s' = s` a fixed exogenous grid
        would miss is covered by the keeper, not appended here.
        """
        return [
            self.outer_grid_values[index]
            for index in range(self.outer_grid_values.shape[0])
        ]


def _stack_carry_template(
    *, template: EGMCarry | None, n_candidates: int
) -> EGMCarry | None:
    """Stack a keeper carry template into the published candidate-axis shape.

    The NEGM continuation carry retains every outer candidate (the keeper plus
    one per outer-grid node) on a candidate axis inserted just before the grid
    axis. The parent period's kernel is AOT-compiled against this template, so
    the template must carry that axis: the keeper template is broadcast across
    the `n_candidates` slots, keeping every row finite and ascending so a parent
    evaluated against the template stays finite.
    """
    if template is None:
        return None

    def stack(arr: FloatND) -> FloatND:
        return jnp.broadcast_to(
            arr[..., None, :], (*arr.shape[:-1], n_candidates, arr.shape[-1])
        )

    return EGMCarry(
        endog_grid=stack(template.endog_grid),
        value=stack(template.value),
        marginal_utility=stack(template.marginal_utility),
        taste_shock_scale=template.taste_shock_scale,
    )


def _build_coh_shift_function(
    *,
    functions: EconFunctionsMapping,
    durable_state_name: StateName,
    outer_post_decision: FunctionName,
    no_adjustment_func: EconFunction | None,
    outer_cost_name: FunctionName | None,
) -> Callable[..., FloatND]:
    """Build the per-(durable, outer-node) cash-on-hand shift of each adjuster.

    Adjuster `j`'s inner endogenous grid lives in resources space `R_j = coh -
    cost(z, z'_j)`; mapping it into the keeper's cash-on-hand axis adds back
    the credited cost relative to the free keep:

    `shift(z, z'_j) = cost(z, z'_j) - cost(z, keep(z))`,

    evaluated directly on the regime's declared outer-cost DAG
    (`NEGM.outer_cost`), whose inputs are only the durable state, the outer
    post-decision, and params. Nothing about the shift is inferred from the
    wider resources function — with a declared cost the resources are composed
    at model build as `<resources>_before_outer_cost - <outer_cost>`, so their
    affine use of the cost (coefficient exactly `-1`) holds by construction
    and the credited difference is exactly the resources translation. `keep`
    is the keeper's no-adjustment map (`no_adjustment_func`; the identity when
    the regime declares none) — the level whose credited cost is zero, e.g.
    the depreciated stock `z (1 - delta)`. The axis change has derivative 1,
    so each candidate's value and resource-marginal transfer into coh space
    unchanged. With `outer_cost_name=None` — validated at model build to mean
    the resources never read the outer post-decision — the shift is
    identically zero.

    The returned callable takes the durable grid (`durable_values`), the outer
    grid (`outer_values`), and the regime's flat params, and returns the shift
    matrix of shape `(n_durable, n_outer)`.
    """
    if outer_cost_name is None:

        def zero_shifts(
            *, durable_values: FloatND, outer_values: FloatND, **params: object
        ) -> FloatND:
            del params
            return jnp.zeros(
                (durable_values.shape[0], outer_values.shape[0]),
                dtype=durable_values.dtype,
            )

        return zero_shifts

    non_h_functions = {name: func for name, func in functions.items() if name != "H"}
    cost_func = concatenate_functions(
        functions=non_h_functions,
        targets=outer_cost_name,
        enforce_signature=False,
        set_annotations=True,
    )
    cost_arg_names = set(get_annotations(cost_func)) - {"return"}

    def keeper_level(durable: FloatND) -> FloatND:
        # The keeper core realises the outer post-decision at its own
        # no-adjustment level `keep(durable)` — the level whose credited cost
        # is zero. With an identity keeper this is `durable` itself.
        return durable if no_adjustment_func is None else no_adjustment_func(durable)

    def coh_shifts(
        *, durable_values: FloatND, outer_values: FloatND, **params: object
    ) -> FloatND:
        # Defense in depth behind the model-build ancestor check: a cost DAG
        # demanding any binding other than the durable, the outer
        # post-decision, and params cannot be evaluated per (durable, outer)
        # cell.
        cost_extra_arg_names = (
            cost_arg_names - {durable_state_name, outer_post_decision} - set(params)
        )
        if cost_extra_arg_names:
            msg = (
                f"The declared NEGM outer cost '{outer_cost_name}' reads "
                f"{sorted(cost_extra_arg_names)}. It may read only the durable "
                f"state '{durable_state_name}', the outer post-decision "
                f"'{outer_post_decision}', and params — the credited-cost lift "
                "is a constant per (durable, outer-node) cell, so no other "
                "state or action can vary inside it."
            )
            raise InvalidParamsError(msg)

        def cost_at(durable: FloatND, outer: FloatND) -> FloatND:
            bindings = {durable_state_name: durable, outer_post_decision: outer}
            return cost_func(
                **{
                    name: value
                    for name, value in bindings.items()
                    if name in cost_arg_names
                },
                **params,
            )

        return jax.vmap(
            lambda durable: jax.vmap(
                lambda outer: (
                    cost_at(durable, outer) - cost_at(durable, keeper_level(durable))
                )
            )(outer_values)
        )(durable_values)

    return coh_shifts


def _strip_outer_transition(
    *,
    transitions: TransitionFunctionsMapping,
    outer_post_decision: FunctionName,
) -> TransitionFunctionsMapping:
    """Drop the outer post-decision transition from every target's transitions.

    The adjuster supplies the outer post-decision value per outer-grid node, so
    its child-carry next-state function must not demand the outer action;
    `read_child` sources the bound value from the combo pool instead.
    """
    return MappingProxyType(
        {
            target: MappingProxyType(
                {
                    name: func
                    for name, func in target_transitions.items()
                    if name != outer_post_decision
                }
            )
            for target, target_transitions in transitions.items()
        }
    )


def _no_adjustment_outer_transition(
    *,
    transitions: TransitionFunctionsMapping,
    outer_post_decision: FunctionName,
    no_adjustment_func: EconFunction | None,
) -> TransitionFunctionsMapping:
    """Replace the outer post-decision transition with the keeper's durable map.

    The keeper holds the durable at its no-adjustment level
    (`next_<durable> = keep(<durable>)`), so each target's outer transition
    becomes that law on the durable state. The durable state name is the
    transition name with the `next_` prefix removed, mirroring the engine's
    `next_<state>` auto-naming. The durable then reads as a genuine
    decision-independent passive state in the inner DC-EGM.

    With no `no_adjustment_func` the keeper holds the stock unchanged via the
    auto-identity transition (`next_<durable> = <durable>`), and `read_child`
    indexes the unchanged durable on its grid. A depreciating
    `keep(d) = d (1 - delta)` lands the kept stock off the durable grid; the
    inner passive read blends the child value over the grid's neighbouring nodes.
    """
    durable_state = outer_post_decision.removeprefix("next_")
    if no_adjustment_func is None:
        keeper_transition = cast(
            "TransitionFunction",
            _IdentityTransition(durable_state, annotation=ContinuousState),
        )
    else:
        keeper_transition = _durable_keeper_transition(
            no_adjustment_func=no_adjustment_func,
            durable_state=durable_state,
            outer_post_decision=outer_post_decision,
        )
    return MappingProxyType(
        {
            target: MappingProxyType(
                {
                    name: (keeper_transition if name == outer_post_decision else func)
                    for name, func in target_transitions.items()
                }
            )
            for target, target_transitions in transitions.items()
        }
    )


def _with_no_adjustment_outer_function(
    *,
    functions: EconFunctionsMapping,
    outer_post_decision: FunctionName,
    no_adjustment_func: EconFunction | None,
) -> EconFunctionsMapping:
    """Add the keeper's outer post-decision to the econ-function DAG.

    The inner resources function reads the outer post-decision (`next_<durable>`)
    by name. The adjuster binds it as a per-node param; the keeper instead holds
    it at its no-adjustment level, so the resources DAG computes
    `next_<durable> = keep(<durable>)` from the durable leaf state. The injected
    function declares the durable state as its single parameter, so DAG
    concatenation wires the durable combo value into resources. With no
    `no_adjustment_func`, `keep` is the identity (hold the stock).
    """
    durable_state = outer_post_decision.removeprefix("next_")
    # Copy the durable's annotation (and the outer post-decision's consumer
    # annotation) off the existing functions so the DAG's annotation-consistency
    # check, which requires every consumer of a leaf to agree, stays satisfied.
    durable_annotation = _annotation_of_arg(functions=functions, arg_name=durable_state)
    outer_annotation = _annotation_of_arg(
        functions=functions, arg_name=outer_post_decision
    )

    @with_signature(
        args={durable_state: durable_annotation},
        return_annotation=outer_annotation,
    )
    def keep_outer_post_decision(**kwargs: FloatND) -> FloatND:
        if no_adjustment_func is None:
            return kwargs[durable_state]
        return no_adjustment_func(**{durable_state: kwargs[durable_state]})

    keep_outer_post_decision.__name__ = outer_post_decision
    return MappingProxyType(
        {
            **dict(functions),
            outer_post_decision: cast("EconFunction", keep_outer_post_decision),
        }
    )


def _durable_keeper_transition(
    *,
    no_adjustment_func: EconFunction,
    durable_state: StateName,
    outer_post_decision: FunctionName,
) -> TransitionFunction:
    """Wrap the no-adjustment map as the keeper's durable transition.

    The map reads the durable state alone and returns the kept next stock, so it
    is a decision-independent passive law `next_<durable> = keep(<durable>)`. The
    wrapper carries the `next_<durable>` name and a `ContinuousState` signature so
    the engine's transition collector classifies it like any passive durable law.
    """

    @with_signature(
        args={durable_state: "ContinuousState"},
        return_annotation="ContinuousState",
    )
    def keeper_transition(**kwargs: ContinuousState) -> ContinuousState:
        return no_adjustment_func(**{durable_state: kwargs[durable_state]})

    keeper_transition.__name__ = outer_post_decision
    return cast("TransitionFunction", keeper_transition)


def _annotation_of_arg(
    *, functions: EconFunctionsMapping, arg_name: StateOrActionName
) -> str:
    """Return the annotation the regime's functions use for one argument.

    The DAG's annotation-consistency check requires every consumer of a leaf to
    agree on its annotation, so the injected keeper function copies it from the
    first regime function that declares the argument. Falls back to `"FloatND"`
    when no function annotates it.
    """
    for func in functions.values():
        annotations = ensure_annotations_are_strings(get_annotations(func))
        annotation = annotations.get(arg_name, "no_annotation_found")
        if annotation != "no_annotation_found":
            return annotation
    return "FloatND"


def _with_outer_post_decision(
    *,
    flat_params: FlatParams,
    regime_name: RegimeName,
    outer_post_decision: FunctionName,
    value: FloatND,
) -> FlatParams:
    """Bind the outer post-decision value into the regime's flat params.

    The inner DC-EGM core threads its per-combo pool from `flat_params`, so
    binding `outer_post_decision` there makes the inner resources and the
    child-state index read the fixed outer node as a constant.
    """
    regime_params = {**dict(flat_params[regime_name]), outer_post_decision: value}
    return MappingProxyType(
        {
            name: (
                MappingProxyType(regime_params) if name == regime_name else regime_pool
            )
            for name, regime_pool in flat_params.items()
        }
    )


def _fail_if_outer_batch_size_negative(outer_batch_size: int) -> None:
    if outer_batch_size < 0:
        msg = (
            f"NEGM.outer_batch_size must be non-negative, got {outer_batch_size}. "
            "Use 0 to solve every outer-grid node at once, or a positive value to "
            "fold the outer search in chunks of that many nodes."
        )
        raise RegimeInitializationError(msg)


def _fail_if_outer_grid_is_stochastic(outer_grid: ContinuousGrid) -> None:
    if isinstance(outer_grid, _ContinuousStochasticProcess):
        msg = (
            "NEGM.outer_grid must be a deterministic continuous grid, not a "
            f"stochastic process ({type(outer_grid).__name__}). The outer grid "
            "is the exogenous search grid over the durable post-decision margin; "
            "it carries no transition. A stochastic durable margin belongs in a "
            "process state, not the NEGM outer search."
        )
        raise RegimeInitializationError(msg)


def _fail_if_outer_action_is_inner_action(
    *, outer_action: ActionName, inner: DCEGM
) -> None:
    if outer_action == inner.continuous_action:
        msg = (
            f"NEGM.outer_action '{outer_action}' coincides with the inner "
            f"DC-EGM continuous action '{inner.continuous_action}'. The outer "
            "durable/illiquid margin and the inner consumption margin must be "
            "distinct actions."
        )
        raise RegimeInitializationError(msg)


def _fail_if_outer_post_decision_is_inner_post_decision(
    *, outer_post_decision: FunctionName, inner: DCEGM
) -> None:
    if outer_post_decision == inner.post_decision_function:
        msg = (
            f"NEGM.outer_post_decision '{outer_post_decision}' coincides with "
            f"the inner DC-EGM post-decision function "
            f"'{inner.post_decision_function}'. The outer post-decision (the "
            "next-period durable stock) and the inner post-decision (the liquid "
            "savings) must be distinct functions."
        )
        raise RegimeInitializationError(msg)
