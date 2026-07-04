"""The built-in regime solvers.

`GridSearch` (the default) runs the existing max-Q-over-a grid search; `DCEGM`
runs the discrete-continuous endogenous grid method. Both are `Solver`
subclasses whose `build_period_kernels` returns one `PeriodKernel` per period —
a non-jitted adapter that wraps the solver's shared jitted core, calls it with
the solver's own argument layout, and assembles a `KernelResult` outside JIT.
The solve loop invokes every adapter the same way, so no solver-type fork
survives in the loop.

Compilation reuse is preserved: only the shared core is `jax.jit`'d and
identity-deduped (`id(Q_and_F)` for grid search, function identity for the EGM
step), so periods sharing a core reuse one compiled program. The adapters that
wrap a shared core are themselves never jitted.

The kernel-building imports (`jax`, `get_max_Q_over_a`, `build_egm_step_functions`)
are function-local so the public `lcm.solvers` façade stays a thin re-export that
pulls in no numerical engine modules.
"""

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from beartype import beartype
from dags import concatenate_functions, get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.beartype_conf import REGIME_CONF
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.bqsegm import BQSEGMRegistry
from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_envelope import (
    finalize_outer_envelope,
    fold_outer_envelope,
    init_outer_envelope,
)
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
from _lcm.typing import (
    EconFunction,
    EconFunctionsMapping,
    EGMStepFunction,
    FlatParams,
    MaxQOverAFunction,
    RegimeName,
    TransitionFunction,
    TransitionFunctionsMapping,
)
from lcm.ages import AgeGrid
from lcm.case_piece import EqualityOwner
from lcm.exceptions import RegimeInitializationError
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousState,
    Float1D,
    FloatND,
    FunctionName,
    IntND,
    StateName,
    StateOrActionName,
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
        import jax  # noqa: PLC0415

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


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class DCEGM(Solver):
    """Configuration of the DC-EGM solver for one regime.

    DC-EGM inverts the Euler equation on an exogenous end-of-period
    (post-decision) grid instead of searching a dense grid for the continuous
    action. It requires a specific model structure — exactly one continuous
    (*Euler*) state and one continuous action, a declared resources function
    `R` with consumption recovery `c = R - A`, a post-decision function `A`,
    and an `inverse_marginal_utility` regime function — which is validated at
    `Model` construction time.

    The configuration is published so a model can name the solver and its
    parameters, but the solver engine is not yet wired in: `validate` rejects a
    regime requesting it. `GridSearch` is the only available solver.

    Forward simulation works but is *grid-restricted*: `simulate` recomputes
    the argmax over the regime's gridded continuous action against the
    stored value function, rather than interpolating the exact EGM policy.
    Simulated continuous actions therefore live on the action grid, and with
    taste shocks the simulated choice frequencies follow the grid-restricted
    choice-specific values, not exactly the solve's choice probabilities.
    The budget constraint the solve enforces intrinsically
    (`continuous_action <= resources - savings_grid lower bound`) is applied
    as a feasibility mask during simulation.

    """

    continuous_state: StateName
    """Name of the Euler continuous state (e.g. `"wealth"`).

    Its transition must consume the post-decision function and reach the
    state and the continuous action only through it.
    """

    continuous_action: ActionName
    """Name of the continuous action (e.g. `"consumption"`)."""

    resources: FunctionName
    """Name of the resources function `R` in `Regime.functions`.

    Resources are what consumption is paid out of; the endogenous grid lives
    in R-space. Required even in the classic case, where it is the identity
    (e.g. `"resources": lambda wealth: wealth`). Must not depend on the
    continuous action and must be non-decreasing in the continuous state.
    """

    post_decision_function: FunctionName
    """Name of the post-decision function in `Regime.functions`.

    The end-of-period state (e.g. savings), satisfying
    `post_decision = resources - continuous_action`.
    """

    savings_grid: ContinuousGrid
    """Exogenous end-of-period grid; its lower bound is the borrowing limit.

    The endogenous grid inherits this grid's spacing, and the published value
    function is interpolated linearly between endogenous points — so this grid
    controls where the solution is accurate. With sharply curved utility (e.g.
    CRRA), cluster the nodes toward the borrowing limit (`LogSpacedGrid`, or
    an `IrregSpacedGrid` clustered at the low end): a uniform grid
    under-resolves the value function near the limit, and that interpolation
    error compounds across periods.
    """

    upper_envelope: Literal["fues", "rfc", "ltm", "mss"] = "fues"
    """Upper-envelope refinement backend removing dominated Euler candidates.

    - `"fues"`: the Fast Upper-Envelope Scan — a sequential scan that inserts
      exact segment-crossing points.
    - `"rfc"`: the Rooftop-Cut algorithm — a parallel dominance test that only
      deletes points (a kink lands between retained points, recovered by the
      Hermite carry read) and generalizes to multidimensional grids.
    - `"ltm"`: the local-upper-bound brute method — an `O(K^2)` dense segment
      scan that evaluates the envelope at every candidate abscissa (the
      quadratic baseline of Dobrescu & Shanker 2026; a kink lands between
      output nodes, recovered by the downstream read).
    - `"mss"`: HARK's EGM upper envelope — a left-to-right sweep that keeps the
      max-value branch at every abscissa *and* inserts the exact
      segment-crossing point, so it tracks the FUES envelope tightly (the `MSS`
      method of Dobrescu & Shanker 2026).
    """

    fues_jump_thresh: float = 2.0
    """Segment-switch threshold on `|ΔA / ΔR|` in the FUES scan."""

    fues_n_points_to_scan: int = 10
    """Number of points the FUES forward scan inspects after a candidate."""

    fues_scan_unroll: int = 1
    """Loop-unroll factor for the FUES candidate `lax.scan`.

    Passed to `jax.lax.scan(..., unroll=fues_scan_unroll)` in both the
    full-envelope and streaming-bracket scans. The scan is sequential and
    latency-bound on accelerators; unrolling `k` iterations into one loop body
    trades compile time and code size for fewer loop-carry round trips, which can
    cut the per-row exec wall on GPU. `1` (no unroll) is the default; the refined
    envelope is numerically identical across values, so this is a pure
    performance knob.
    """

    rfc_jump_thresh: float = 2.0
    """Segment-switch threshold on `|Δc / ΔR|` in the rooftop cut."""

    rfc_search_radius: int = 10
    """Number of neighbors on each side the rooftop-cut dominance test inspects."""

    refined_grid_factor: float = 1.2
    """Headroom factor sizing the refined (NaN-padded) envelope arrays."""

    n_constrained_points: int = 20
    """Number of closed-form points on the credit-constrained segment."""

    stochastic_node_batch_size: int = 0
    """Block size for splaying the child stochastic-node expectation.

    The continuation expectation runs over the product of the child regime's
    stochastic process nodes — a single mesh, not a per-grid axis, so it gets
    its own solve-level knob rather than a per-grid `batch_size`. A positive
    value below the mesh length processes that expectation in `lax.map` blocks
    instead of one fused vmap, shedding the dominant `egm_step` working buffer
    (which carries this node axis); `0` keeps the fused vmap. Like the savings
    grid's `batch_size`, this is a memory knob only — the solved value function
    is identical to the unsplayed solve.
    """

    def __post_init__(self) -> None:
        _fail_if_savings_grid_is_stochastic(self.savings_grid)
        _fail_if_refined_grid_factor_too_small(self.refined_grid_factor)
        _fail_if_fues_jump_thresh_non_positive(self.fues_jump_thresh)
        _fail_if_n_constrained_points_too_few(self.n_constrained_points)
        _fail_if_fues_n_points_to_scan_too_few(self.fues_n_points_to_scan)
        _fail_if_fues_scan_unroll_too_few(self.fues_scan_unroll)
        _fail_if_rfc_jump_thresh_non_positive(self.rfc_jump_thresh)
        _fail_if_rfc_search_radius_too_few(self.rfc_search_radius)
        _fail_if_stochastic_node_batch_size_negative(self.stochastic_node_batch_size)

    @property
    def requires_continuation_carries(self) -> bool:
        """DC-EGM inverts the Euler equation against its targets' marginals."""
        return True

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one DC-EGM period adapter per period and the carry template.

        The standalone `validate_dcegm_regimes` model-contract check (run during
        regime processing) guarantees the regime is non-terminal, so the regime
        transition probability function exists. Periods sharing one EGM-step core
        reuse a single jitted core, and therefore a single compiled program.
        Numerical-builder imports are function-local so the public `lcm.solvers`
        façade stays a thin re-export that pulls in no engine modules.
        """
        import jax  # noqa: PLC0415

        from _lcm.egm.step import build_egm_step_functions  # noqa: PLC0415

        assert context.compute_regime_transition_probs is not None  # noqa: S101
        egm_step, egm_carry_template, egm_reachable_targets = build_egm_step_functions(
            solver=self,
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
            jitted_by_id: dict[int, EGMStepFunction] = {}
            for func in egm_step.values():
                if id(func) not in jitted_by_id:
                    jitted_by_id[id(func)] = jax.jit(func)
            egm_step = MappingProxyType(
                {period: jitted_by_id[id(func)] for period, func in egm_step.items()}
            )
        period_kernels = MappingProxyType(
            {
                period: _DCEGMPeriodKernel(
                    core=core,
                    regime_name=context.regime_name,
                    reachable_targets=egm_reachable_targets,
                    transition_target_names=tuple(context.transitions),
                )
                for period, core in egm_step.items()
            }
        )
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_template=egm_carry_template,
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

    outer_batch_size: int = 0
    """Number of outer-grid nodes solved per chunk before folding into the
    running outer maximum.

    The outer search folds each node's solve into a running `(V_arr, envelope)`,
    so the peak device memory holds one chunk of candidates regardless of the
    outer-grid size. A positive value processes that many nodes at once (their
    independent solves overlap) before reducing them; `0` (the default) solves
    every node at once — fastest, but its peak grows with the outer-grid size.
    It is a memory-vs-parallelism knob only: `max` is associative, so the solved
    value function is identical across batch sizes.
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
    def requires_continuation_carries(self) -> bool:
        """NEGM nests a DC-EGM solve that inverts the Euler equation."""
        return True

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
            resources_name=self.inner.resources,
            euler_state_name=self.inner.continuous_state,
            durable_state_name=durable_state,
            outer_post_decision=self.outer_post_decision,
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
            continuation_template=_widen_carry_template(
                template=keeper_kernels.continuation_template,
                n_extra=outer_grid_values.shape[0],
            ),
        )


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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],  # noqa: ARG002
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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],  # noqa: ARG002
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


@dataclass(frozen=True, kw_only=True)
class _DCEGMPeriodKernel:
    """The DC-EGM period adapter — wraps one EGM-step core.

    Closes over the regime name, its reachable carry targets, and the names of
    its transition targets (to union their params). Calling it inverts the Euler
    equation on the savings grid and returns a `KernelResult` carrying the value
    function, the continuation a parent interpolates, and the published off-grid
    simulation policy.
    """

    core: Callable
    """The shared jitted EGM-step core (`id`-deduped across periods)."""

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    reachable_targets: frozenset[RegimeName]
    """The carry keys the EGM core reads; the rolling carry is filtered to these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    def cores(self) -> Mapping[str, Callable]:
        """Return the single EGM-step core under the `"main"` key."""
        return MappingProxyType({"main": self.core})

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _DCEGMPeriodKernel:
        """Bind the regime's and its carry targets' fixed params into the core.

        A DC-EGM source carrying into a *different* target regime evaluates that
        target's resources / transition functions in its per-asset-node solve,
        reading the target's fixed params. The core threads its `**kwargs`
        straight into the per-combo pool those captured functions read, so
        binding the union of the regime's and its carry targets' fixed params
        restores the values removed from the live `flat_params` for all of them
        at once.
        """
        egm_fixed = dict(fixed_flat_params.get(self.regime_name, MappingProxyType({})))
        for target_name in self.transition_target_names:
            for key, value in fixed_flat_params.get(
                target_name, MappingProxyType({})
            ).items():
                egm_fixed.setdefault(key, value)
        if not egm_fixed:
            return self
        return replace(self, core=functools.partial(self.core, **egm_fixed))

    def build_lower_args(
        self,
        *,
        core_key: str = "main",  # noqa: ARG002
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: states, carries, EGM params."""
        return {
            **dict(state_action_space.states),
            "next_regime_to_egm_carry": _reachable_carry_subset(
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                reachable_targets=self.reachable_targets,
            ),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **self._egm_kernel_params(flat_params=flat_params),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Run the DC-EGM step and assemble the `KernelResult`."""
        V_arr, egm_carry, sim_policy = compiled_cores["main"](
            **state_action_space.states,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_egm_carry=_reachable_carry_subset(
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                reachable_targets=self.reachable_targets,
            ),
            **self._egm_kernel_params(flat_params=flat_params),
            period=jnp.int32(period),
            age=ages.values[period],
        )
        return KernelResult(V_arr=V_arr, carry=egm_carry, sim_policy=sim_policy)

    def _egm_kernel_params(self, *, flat_params: FlatParams) -> dict[str, object]:
        """Flat params fed into the DC-EGM core: the source's plus its targets'.

        A DC-EGM source carrying into a *different* target regime evaluates that
        target's resources / transition functions in its per-asset-node solve,
        reading the target's params (e.g. a pension factor the source never
        reads). These are model-level shared values, so the target's
        `flat_params` entry carries the right value; union them in. The core
        threads its `**kwargs` into the per-combo pool, and its captured
        functions read only the keys they need, so a target's extra params are
        harmless to the source functions that do not. Mirrors the fixed-param
        binding done at model build (`_partial_fixed_params_into_regimes`) for
        the free-param path.
        """
        params: dict[str, object] = dict(flat_params[self.regime_name])
        for target_name in self.transition_target_names:
            for key, value in flat_params.get(
                target_name, MappingProxyType({})
            ).items():
                params.setdefault(key, value)
        return params


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
    """Outer-grid nodes solved per chunk before folding into the running maximum.

    `0` solves every node at once; a positive value bounds the peak memory to one
    chunk of candidates. A memory-vs-parallelism knob only — value-invariant.
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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
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
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        return self.adjuster_kernel.build_lower_args(
            core_key="main",
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
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
            next_regime_to_egm_carry=next_regime_to_egm_carry,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        # The published continuation is the genuine cash-on-hand upper envelope of
        # the keeper and every adjuster candidate, so the parent period's
        # keeper-identity continuation read sees the best durable choice at every
        # coh — not only the keeper's, and an adjuster that wins strictly between
        # keeper nodes survives.
        coh_shifts = self.coh_shift_func(
            durable_values=self.durable_grid_values,
            outer_values=self.outer_grid_values,
            **flat_params[self.regime_name],
        )
        # Fold the adjuster outer-grid nodes into a running outer maximum — `V =
        # max_j W_j` on the state grid and the coh-space envelope carry — in chunks
        # of `outer_batch_size`, rather than materialising every node's solve at
        # once. Each chunk's independent solves overlap; reducing them into the
        # running `(V_arr, envelope)` bounds the peak to one chunk of candidates.
        # `max` is associative and the envelope's shared coh grid is fixed at the
        # keeper's, so the chunked fold is value-identical to a single stacked
        # maximum regardless of the chunk size. The keeper and every adjuster are
        # DC-EGM kernels, so each always publishes a continuation carry.
        keeper_carry = cast("EGMCarry", keeper_result.carry)
        V_arr = keeper_result.V_arr
        nodes = self._outer_nodes()
        envelope = init_outer_envelope(keeper_carry, len(nodes))
        chunk_size = self.outer_batch_size or len(nodes)
        for chunk_start in range(0, len(nodes), chunk_size):
            chunk_results = [
                self.adjuster_kernel(
                    compiled_cores={"main": compiled_cores["adjuster"]},
                    state_action_space=state_action_space,
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_egm_carry=next_regime_to_egm_carry,
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
            for offset, adjuster_result in enumerate(chunk_results):
                V_arr = jnp.maximum(V_arr, adjuster_result.V_arr)
                envelope = fold_outer_envelope(
                    envelope,
                    cast("EGMCarry", adjuster_result.carry),
                    coh_shifts[:, chunk_start + offset],
                    chunk_start + offset,
                )
            # Force the running maximum to device before the next chunk. Without
            # this the lazy fold accumulates a dependency on every chunk's solves
            # at once — the peak would grow with the whole outer grid rather than
            # one chunk — and the chunk's independent solves could not overlap.
            V_arr, envelope = jax.block_until_ready((V_arr, envelope))
        carry = finalize_outer_envelope(envelope)
        # The simulate phase re-optimizes the outer durable action by grid argmax
        # over the next-period value array, so the published `sim_policy` (the
        # keeper's off-grid inner consumption function) is not the channel that
        # drives simulated durable choice; it rides through unchanged.
        return KernelResult(
            V_arr=V_arr,
            carry=carry,
            sim_policy=keeper_result.sim_policy,
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


def _widen_carry_template(
    *, template: EGMCarry | None, n_extra: int
) -> EGMCarry | None:
    """Widen a keeper carry template by `n_extra` trailing island-peak slots.

    The NEGM outer envelope publishes the keeper's shared-grid envelope plus one
    spliced island-peak slot per adjuster, so the published carry is `n_extra`
    nodes wider than the keeper's. The parent period's kernel is AOT-compiled
    against this template, so the template's last axis must match the published
    width. The padding is `NaN` (trailing dead nodes), consistent with the
    interpolator's first-finite-node search.
    """
    if template is None:
        return None
    pad = [(0, 0)] * (template.endog_grid.ndim - 1) + [(0, n_extra)]
    widen = lambda arr: jnp.pad(arr, pad, constant_values=jnp.nan)  # noqa: E731
    return EGMCarry(
        endog_grid=widen(template.endog_grid),
        value=widen(template.value),
        marginal_utility=widen(template.marginal_utility),
        taste_shock_scale=template.taste_shock_scale,
    )


def _build_coh_shift_function(
    *,
    functions: EconFunctionsMapping,
    resources_name: FunctionName,
    euler_state_name: StateName,
    durable_state_name: StateName,
    outer_post_decision: FunctionName,
) -> Callable[..., FloatND]:
    """Build the per-(durable, outer-node) cash-on-hand shift of each adjuster.

    Adjuster `j`'s inner endogenous grid lives in resources space `R_j = coh -
    credited(z, z'_j)`; mapping it into the keeper's cash-on-hand axis adds back
    `credited(z, z'_j)`. That credited cost is the keeper-minus-adjuster
    difference of the regime's own inner resources at any fixed liquid wealth
    (`resources` is affine in wealth, so wealth cancels):

    `shift(z, z'_j) = resources(w0, z, next=z) - resources(w0, z, next=z'_j)`.

    The returned callable takes the durable grid (`durable_values`), the outer
    grid (`outer_values`), and the regime's flat params, and returns the shift
    matrix of shape `(n_durable, n_outer)`.
    """
    resources_func = concatenate_functions(
        functions={name: func for name, func in functions.items() if name != "H"},
        targets=resources_name,
        enforce_signature=False,
        set_annotations=True,
    )
    resources_arg_names = set(get_annotations(resources_func)) - {"return"}
    bound_arg_names = {euler_state_name, durable_state_name, outer_post_decision}

    def coh_shifts(
        *, durable_values: FloatND, outer_values: FloatND, **params: object
    ) -> FloatND:
        zero_reference = jnp.zeros((), dtype=durable_values.dtype)
        # Every resources leaf other than the Euler state, the durable state, the
        # outer post-decision, and the regime params is constant in the outer
        # choice (the credited cost reads only the durable margin), so it appears
        # identically in the keeper and adjuster legs and cancels in their
        # difference. Hold each at a fixed reference — exactly as the Euler state
        # is held at zero — so the shift is the pure credited-cost difference.
        separable_arg_names = resources_arg_names - bound_arg_names - set(params)
        separable_references = dict.fromkeys(separable_arg_names, zero_reference)

        def shift_one(durable: FloatND, outer: FloatND) -> FloatND:
            # The keeper reference holds the outer post-decision at the durable
            # itself (`next = durable`), not at the keeper core's no-adjustment
            # level `keep(durable)`. For an identity keeper (`keep(d) = d`) the two
            # coincide and this is the exact credited-cost lift. Under a
            # depreciating keeper (`keep(d) = d(1 - delta)`) the design-exact
            # reference would be `keep(durable)`, but using it in isolation lifts
            # each adjuster to a coh that the fixed keeper-grid envelope reads by
            # extrapolation and the one-island-per-adjuster splice then over-counts
            # — empirically regressing the DS-2024 delta=0.10 oracle agreement from
            # a converging ~0.08 to a plateaued ~0.18. The identity reference and
            # the island splice are tuned together; correcting both belongs with
            # the segment-topology outer-envelope redesign, not here.
            keeper_resources = resources_func(
                **{
                    euler_state_name: zero_reference,
                    durable_state_name: durable,
                    outer_post_decision: durable,
                },
                **separable_references,
                **params,
            )
            adjuster_resources = resources_func(
                **{
                    euler_state_name: zero_reference,
                    durable_state_name: durable,
                    outer_post_decision: outer,
                },
                **separable_references,
                **params,
            )
            return keeper_resources - adjuster_resources

        return jax.vmap(
            lambda durable: jax.vmap(lambda outer: shift_one(durable, outer))(
                outer_values
            )
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


def _reachable_carry_subset(
    *,
    next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
    reachable_targets: frozenset[RegimeName],
) -> MappingProxyType[RegimeName, EGMCarry]:
    """The carries a regime's EGM core actually reads.

    Each core only ever indexes `next_regime_to_egm_carry[target]` for its
    reachable targets, so the full all-regimes mapping is needlessly large.
    Filtering to the reachable subset keeps the core's carry pytree input
    minimal — only this subset is passed per call rather than every regime's
    carry at once.

    Iterates the source mapping's key order (stable across rolls) so the
    filtered pytree structure matches between lowering and call. Membership is
    tested defensively; reachable targets are always carry-producing.
    """
    return MappingProxyType(
        {
            name: next_regime_to_egm_carry[name]
            for name in next_regime_to_egm_carry
            if name in reachable_targets
        }
    )


def _fail_if_savings_grid_is_stochastic(savings_grid: ContinuousGrid) -> None:
    if isinstance(savings_grid, _ContinuousStochasticProcess):
        msg = (
            "DCEGM.savings_grid must be a deterministic continuous grid, not a "
            f"stochastic process ({type(savings_grid).__name__}). The savings "
            "grid is the exogenous end-of-period grid; it carries no transition."
        )
        raise RegimeInitializationError(msg)


def _fail_if_refined_grid_factor_too_small(refined_grid_factor: float) -> None:
    # `not (x > 1.0)` rejects NaN too — `nan <= 1.0` is False, so a bare
    # `<= 1.0` guard would admit a non-finite factor that later sizes the
    # refined envelope arrays and corrupts the scatter.
    if not (math.isfinite(refined_grid_factor) and refined_grid_factor > 1.0):
        msg = (
            f"DCEGM.refined_grid_factor must be a finite value greater than 1.0, "
            f"got {refined_grid_factor}. It is the headroom factor sizing the "
            "refined envelope arrays; a value at or below 1.0 leaves no room "
            "for the constrained points and overflows the scatter."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_jump_thresh_non_positive(fues_jump_thresh: float) -> None:
    # `not (x > 0.0)` rejects NaN too: `nan <= 0.0` is False, so the segment-
    # switch comparison would silently misbehave on a non-finite threshold.
    if not (math.isfinite(fues_jump_thresh) and fues_jump_thresh > 0.0):
        msg = (
            f"DCEGM.fues_jump_thresh must be a finite positive value, got "
            f"{fues_jump_thresh}. It is the segment-switch threshold on "
            "`|ΔA / ΔR|` in the FUES scan."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_jump_thresh_non_positive(rfc_jump_thresh: float) -> None:
    # `not (x > 0.0)` rejects NaN too: `nan <= 0.0` is False, so the segment-
    # switch comparison would silently misbehave on a non-finite threshold.
    if not (math.isfinite(rfc_jump_thresh) and rfc_jump_thresh > 0.0):
        msg = (
            f"DCEGM.rfc_jump_thresh must be a finite positive value, got "
            f"{rfc_jump_thresh}. It is the segment-switch threshold on "
            "`|Δc / ΔR|` in the rooftop cut."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_search_radius_too_few(rfc_search_radius: int) -> None:
    if rfc_search_radius < 1:
        msg = (
            f"DCEGM.rfc_search_radius must be at least 1, got "
            f"{rfc_search_radius}. The rooftop-cut dominance test must inspect "
            "at least one neighbor on each side of a candidate."
        )
        raise RegimeInitializationError(msg)


def _fail_if_n_constrained_points_too_few(n_constrained_points: int) -> None:
    if n_constrained_points < 2:  # noqa: PLR2004
        msg = (
            f"DCEGM.n_constrained_points must be at least 2, got "
            f"{n_constrained_points}. The credit-constrained segment needs at "
            "least two closed-form points to interpolate between."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_n_points_to_scan_too_few(fues_n_points_to_scan: int) -> None:
    if fues_n_points_to_scan < 1:
        msg = (
            f"DCEGM.fues_n_points_to_scan must be at least 1, got "
            f"{fues_n_points_to_scan}. The FUES forward scan must inspect at "
            "least one point after each candidate."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_scan_unroll_too_few(fues_scan_unroll: int) -> None:
    if fues_scan_unroll < 1:
        msg = (
            f"DCEGM.fues_scan_unroll must be at least 1, got "
            f"{fues_scan_unroll}. It is the `lax.scan` unroll factor for the "
            "FUES candidate scan; 1 means no unrolling."
        )
        raise RegimeInitializationError(msg)


def _fail_if_stochastic_node_batch_size_negative(
    stochastic_node_batch_size: int,
) -> None:
    if stochastic_node_batch_size < 0:
        msg = (
            f"DCEGM.stochastic_node_batch_size must be non-negative, got "
            f"{stochastic_node_batch_size}. It is the block size for splaying the "
            "child stochastic-node expectation into `lax.map` blocks; 0 keeps the "
            "fused vmap."
        )
        raise RegimeInitializationError(msg)


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
    def requires_continuation_carries(self) -> bool:
        """The 1-D EGM step reads its continuation's marginal value of liquid."""
        return True

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one 1-D EGM period adapter per active period.

        Each period's adapter knows the single deterministic continuation
        target (the transition target whose regime is active next period), so
        it reads that target's value array and marginal-utility carry.
        """
        import jax  # noqa: PLC0415

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


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class BQSEGM(Solver):
    """Case-piece endogenous-grid solver for a 1-D consumption--saving regime.

    A regime whose budget is split by case boundaries on the liquid state (e.g. a
    Medicaid asset test) is smooth within each case. BQSEGM solves each case by
    ordinary 1-D EGM, masks each case's candidates to the region where its
    predicate is consistent with the recovered state, and merges the cases on the
    liquid grid with the branch-aware upper envelope. The strict/non-strict
    consistency split gives the boundary point to the side that owns equality.
    The step carries the marginal value of liquid backward, like the plain 1-D
    EGM, so this regime both reads and publishes a continuation carry.

    The regime's declarations select the kernel:

    - Case-piece split (`lcm.case_boundary` / `lcm.piece`): the binary jump step
      on the two masked subsidy cases. The v1 scope is one binary predicate
      splitting additive cash-on-hand contributions; multi-predicate and
      non-additive pieces are deferred.
    - Piecewise-affine schedule (`lcm.piecewise_affine`): the breakpoint
      kinds pick the step — kinks/floors only, jumps only, or mixed —
      solved by `coh` inversion per continuous run and masked across the jumps.
    - Schedule with ride-along co-states: two independently-jitted cores per
      period (transition-aware continuation read, then the per-cell envelope
      solve in savings space), batched over the ride-along cells.
    - Discrete action over a smooth budget: one continuous subproblem per
      discrete-action value, merged by the discrete upper envelope.
    """

    savings_grid: ContinuousGrid
    """Exogenous post-decision savings grid `s = coh - consumption` (>= 0)."""
    budget_target: str = "coh"
    """DAG node carrying the consumption budget (cash-on-hand).

    Names the model node the continuous EGM inverts against, mirroring `DCEGM`'s
    `resources=`. The default `"coh"` matches the solver's convention; a model
    that names its budget node differently (`resources`, `cash_on_hand`) selects
    it here.
    """
    post_decision_function: FunctionName | None = None
    """Name of the post-decision savings function (the `savings = coh - c` slot).

    Required when the regime carries a ride-along co-state: the continuation is
    then read through the transition-aware continuation reader (which consumes
    the savings slot), so the 1-D case-piece solve batches over the ride-along
    axes. `None` for a single-liquid-axis regime, whose continuation is read
    directly off the next period's liquid grid.
    """
    continuous_state: StateName | None = None
    """Name of the liquid (Euler) continuous state, like `DCEGM.continuous_state`.

    Its post-decision law of motion reads `post_decision_function`; every other
    state — discrete, a continuous co-state (e.g. AIME), or a stochastic process —
    rides along, integrated by the continuation reader. `None` lets the solver
    infer it as the regime's single continuous state (the single-liquid case), and
    is rejected when the regime carries more than one continuous state, where the
    Euler axis must be named to separate it from the ride-along axes.
    """
    jump_read: Literal["one_sided", "bridged"] = "one_sided"
    """How the parent's continuation read treats the child value's cliffs.

    The within-period case solve is jump-aware in both modes (masked cases,
    boundary-owner equality); the mode selects what the carry publishes for the
    parents that read it:

    - `"one_sided"` — each carry row holds every jump preimage as a duplicated
      abscissa carrying the exact one-sided value and marginal limits, so reads
      near a cliff are one-sided by construction. Publishing breakpoints gates
      the stochastic-dim fold off on jump-bearing reads, so this mode trades
      runtime for cliff fidelity.
    - `"bridged"` — plain liquid-grid rows with no breakpoints; the parent's
      interpolation may average across a cliff, like any finite-grid solver
      reading the same rows. The fold stays available, so this is the fast
      mode for solves whose consumer tolerates finite-grid cliff error (e.g.
      inner estimation loops, polished afterwards under `"one_sided"`).
    """
    stochastic_node_batch_size: int = 0
    """Block size for splaying the child stochastic-node expectation.

    The continuation read integrates the child's stochastic next-states (health,
    health-cost shocks, the wage residual) over their joint node mesh. `0` reads the
    whole mesh in one vectorized pass; a positive block size loops the mesh in chunks
    of that many nodes, trading compile/runtime for a smaller peak intermediate. Like
    `DCEGM.stochastic_node_batch_size`; raise it when the joint node mesh dominates
    the per-cell memory budget.
    """
    envelope_segment_block_size: int = 0
    """Block size for streaming the merged upper envelope over candidate segments.

    The per-interval envelope brackets every candidate segment against every liquid
    query point; `0` materialises that matrix in one pass, a positive block size
    streams it in blocks of that many segments (identical result, smaller peak
    intermediate). Raise it when the query grid is large enough that the per-cell
    bracket matrix dominates the per-cell memory budget.
    """
    interval_batch_size: int = 0
    """Batch size for the per-interval continuation read.

    When a carry target's next-state law reads the current liquid state, the
    continuation core evaluates the continuation DAG once per declared liquid
    interval. `0` evaluates all intervals in one vectorized pass; a positive
    batch size runs them in sequential chunks of that many intervals
    (identical result, peak intermediates bounded by one chunk). Raise it when
    the per-interval continuation buffers dominate the per-cell memory budget.
    """
    cell_block_size: int = 0
    """Block size for streaming the ride-along solve over ride cells.

    Both ride-along cores fan out per cell — the continuation core's transition/
    child-interpolation read and the envelope core's candidate solve; `0` vmaps
    the whole flattened ride mesh at once in each, so every cell's buffers are in
    flight together — the dominant peak-memory term at production mesh sizes. A
    positive block size scans the mesh in blocks of that many cells in both cores
    (identical result, peak bounded by one block's buffers).
    """

    @property
    def requires_continuation_carries(self) -> bool:
        """The case-piece EGM step reads its continuation's marginal value."""
        return True

    @property
    def carry_retains_discrete_action_rows(self) -> bool:
        """The case-piece carry publishes a value maxed over the continuous action."""
        return False

    @property
    def carry_rows_share_state_grid(self) -> bool:
        """The case-piece ride-along carry sits on the shared liquid grid."""
        return True

    def validate(self, *, context: SolverBuildContext) -> None:
        """Check case coverage and reject hidden branching in user pieces.

        Collecting the metadata enforces strict coverage (each split output has a
        `when` and an `otherwise` piece, every boundary declares a surface). Two
        complementary gates then run on the user pieces:

        - AST: rejects Python branching / hidden comparisons in a smooth piece and
          any non-comparison branching in the boundary predicate.
        - JAXPR: traces each smooth piece and rejects piecewise primitives
          (`select_n`, `lt`, …) hidden inside a called helper that the AST cannot
          see. A piece attested with `lcm.smooth_helper` is exempt.

        The boundary predicate is meant to compare, so only the AST gate runs on
        it; the JAXPR gate runs on the smooth pieces alone.
        """
        import inspect  # noqa: PLC0415

        import jax.numpy as jnp  # noqa: PLC0415

        from _lcm.egm.bqsegm import collect_bqsegm_metadata  # noqa: PLC0415
        from _lcm.egm.bqsegm_validation import (  # noqa: PLC0415
            find_ast_violations,
            find_jaxpr_violations,
            is_smooth_helper,
        )

        functions = cast(
            "Mapping[FunctionName, Callable[..., object]]",
            context.user_regimes[context.regime_name].functions,
        )
        registry = collect_bqsegm_metadata(functions=functions)
        space = context.state_action_space
        _validate_bqsegm_boundary_scope(
            registry=registry,
            functions=functions,
            liquid_state_name=space.state_names[0],
            reserved_names=frozenset(space.state_names) | frozenset(space.action_names),
        )
        violations: list[str] = []
        for predicate_name in registry.boundaries:
            violations += find_ast_violations(
                functions[predicate_name], mode="boundary"
            )
        for piece_set in registry.piece_sets:
            for piece_name in (piece_set.when_func, piece_set.otherwise_func):
                piece = functions[piece_name]
                if is_smooth_helper(piece):
                    continue
                violations += find_ast_violations(piece, mode="smooth_user")
                n_params = len(inspect.signature(piece).parameters)
                abstract_args = tuple(jnp.asarray(1.0) for _ in range(n_params))
                violations += find_jaxpr_violations(
                    piece, abstract_args=abstract_args, mode="smooth_user"
                )
        if violations:
            from lcm.exceptions import BQSEGMCaseError  # noqa: PLC0415

            msg = "BQSEGM smoothness gate failed:\n" + "\n".join(violations)
            raise BQSEGMCaseError(msg)

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one case-piece EGM period adapter per active period."""
        from _lcm.egm.bqsegm import collect_bqsegm_metadata  # noqa: PLC0415

        savings_grid = self.savings_grid.to_jax()

        functions = cast(
            "Mapping[FunctionName, Callable[..., object]]",
            context.user_regimes[context.regime_name].functions,
        )
        registry = collect_bqsegm_metadata(functions=functions)
        has_schedule = bool(registry.piecewise_affine_schedules) and not (
            registry.piece_sets
        )
        has_discrete = bool(context.state_action_space.discrete_actions)
        # A discrete action over a cliffed single-liquid budget composes the
        # discrete upper envelope with the schedule's per-branch intervals. A
        # discrete action alongside a ride-along schedule stays rejected below.
        is_schedule_discrete = (
            has_schedule
            and has_discrete
            and not self._schedule_has_ride_along(context=context)
        )
        is_schedule = has_schedule and not is_schedule_discrete
        is_discrete = not has_schedule and not registry.piece_sets and has_discrete
        schedule_discrete_spec = (
            _collect_bqsegm_schedule_discrete_spec(
                context=context,
                budget_target=self.budget_target,
                continuous_state=self.continuous_state,
            )
            if is_schedule_discrete
            else None
        )
        schedule_spec = (
            _collect_bqsegm_schedule_spec(
                context=context,
                budget_target=self.budget_target,
                continuous_state=self.continuous_state,
            )
            if is_schedule
            else None
        )
        if schedule_spec is not None and schedule_spec.ride_along_state_names:
            if context.state_action_space.discrete_actions:
                self._fail_if_unsupported_ride_discrete(
                    context=context, schedule_spec=schedule_spec
                )
            return self._build_ride_along_kernels(
                context=context,
                savings_grid=savings_grid,
                schedule_spec=schedule_spec,
            )

        liquid_state_name = (
            schedule_spec.liquid_state_name
            if schedule_spec is not None
            else context.state_action_space.state_names[0]
        )
        liquid_grid = context.grids[liquid_state_name].to_jax()
        discrete_spec = (
            _collect_bqsegm_discrete_spec(
                context=context, budget_target=self.budget_target
            )
            if is_discrete
            else None
        )
        case_spec = (
            _collect_bqsegm_case_spec(context=context)
            if not is_schedule and not is_discrete and schedule_discrete_spec is None
            else None
        )

        period_to_target = _period_to_continuation_target(context=context)
        cores: dict[RegimeName, Callable] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        for period, target in period_to_target.items():
            if target not in cores:
                if schedule_discrete_spec is not None:
                    core = _build_bqsegm_schedule_discrete_core(
                        savings_grid=savings_grid,
                        target=target,
                        spec=schedule_discrete_spec,
                        taste_shock_scale=0.0,
                    )
                elif schedule_spec is not None:
                    core = _build_bqsegm_continuous_core(
                        savings_grid=savings_grid,
                        target=target,
                        schedule_spec=schedule_spec,
                    )
                elif discrete_spec is not None:
                    core = _build_bqsegm_discrete_core(
                        savings_grid=savings_grid,
                        target=target,
                        discrete_spec=discrete_spec,
                        taste_shock_scale=0.0,
                    )
                else:
                    assert case_spec is not None  # noqa: S101
                    core = _build_bqsegm_core(
                        savings_grid=savings_grid, target=target, case_spec=case_spec
                    )
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

    def _fail_if_unsupported_ride_discrete(
        self, *, context: SolverBuildContext, schedule_spec: _BQSEGMScheduleSpec
    ) -> None:
        """Reject a ride-along discrete action the envelope path cannot handle.

        The ride-along discrete envelope solves the continuous subproblem per
        discrete branch and takes the upper envelope, but only when the action
        stays inside the budget schedule and the schedule carries no jump:

        - more than one discrete action is unsupported (the envelope is over one
          action's grid),
        - a jump breakpoint would need the published one-sided limits to be the
          max over branches (topology through the envelope), not yet wired,
        - an action the period utility reads directly is not bound per branch.
        """
        import inspect  # noqa: PLC0415

        actions = context.state_action_space.discrete_actions
        if len(actions) != 1:
            msg = (
                "BQSEGM's schedule+ride-along discrete envelope supports exactly "
                f"one discrete action; the regime {context.regime_name!r} declares "
                f"{tuple(actions)}."
            )
            raise RegimeInitializationError(msg)
        action_name = next(iter(actions))
        if any(source.kind == "jump" for source in schedule_spec.sources):
            msg = (
                "BQSEGM's schedule+ride-along discrete envelope handles kink "
                "schedules only; a jump breakpoint with a discrete action needs "
                "the published one-sided limits taken over branches. Regime "
                f"{context.regime_name!r} declares a jump and a discrete action."
            )
            raise RegimeInitializationError(msg)
        utility_args = tuple(inspect.signature(schedule_spec.utility_dag).parameters)
        if action_name in utility_args:
            msg = (
                f"BQSEGM's schedule+ride-along discrete envelope binds the action "
                f"{action_name!r} into cash-on-hand only; the regime "
                f"{context.regime_name!r} reads it in the period utility, which the "
                "envelope core does not yet bind per branch."
            )
            raise RegimeInitializationError(msg)
        _fail_if_discrete_action_feeds_continuation(
            context=context,
            action_name=action_name,
            liquid_state_name=schedule_spec.liquid_state_name,
        )

    def _schedule_has_ride_along(self, *, context: SolverBuildContext) -> bool:
        """Whether the schedule regime carries a ride-along co-state.

        A ride-along axis is any state other than the liquid (Euler) axis. The
        Euler axis is `continuous_state` when named, else the regime's single
        continuous state; discrete actions are not states and never ride along.
        """
        space = context.state_action_space
        continuous_states = tuple(
            name
            for name in space.state_names
            if isinstance(context.grids[name], ContinuousGrid)
        )
        if self.continuous_state is not None:
            liquid_state_name = self.continuous_state
        elif len(continuous_states) == 1:
            liquid_state_name = continuous_states[0]
        else:
            # Ambiguous Euler axis: treat as ride-along so the schedule path (and
            # its explicit multi-continuous-state error) handles it.
            return True
        return any(name != liquid_state_name for name in space.state_names)

    def _build_ride_along_kernels(
        self,
        *,
        context: SolverBuildContext,
        savings_grid: Float1D,
        schedule_spec: _BQSEGMScheduleSpec,
    ) -> SolutionKernels:
        """Build the case-piece kernels for a regime carrying a ride-along co-state.

        The continuation is read through the transition-aware reader, so each
        period's plan depends on its reachable carry/scalar target split; cores
        are deduplicated by that split. The 1-D liquid solve runs once per
        ride-along cell, batched.
        """
        from _lcm.egm.validation import _reachable_target_names  # noqa: PLC0415

        if self.post_decision_function is None:
            msg = (
                "BQSEGM with a ride-along co-state requires `post_decision_function` "
                "(the savings slot the continuation reader consumes); the regime "
                f"{context.regime_name!r} leaves it unset."
            )
            raise RegimeInitializationError(msg)

        liquid_grid = context.grids[schedule_spec.liquid_state_name].to_jax()
        ride_shape = tuple(
            int(context.grids[name].to_jax().shape[0])
            for name in schedule_spec.ride_along_state_names
        )
        reachable_targets = frozenset(
            _reachable_target_names(
                user_regime=context.user_regimes[context.regime_name],
                user_regimes=context.user_regimes,
            )
        )
        transition_target_names = tuple(context.transitions)

        # The ride-along kernel takes the continuation as a probability-weighted
        # blend over the full reachable target set (`bind_continuation` sums the
        # per-target carries by `compute_regime_transition_probs`), so it admits a
        # stochastic multi-target lifecycle transition. Enumerate the regime's
        # active periods directly rather than resolving a single target per period.
        active_periods = sorted(context.regimes_to_active_periods[context.regime_name])
        continuation_cores: dict[tuple[RegimeName, ...], Callable] = {}
        envelope_cores: dict[tuple[RegimeName, ...], Callable] = {}
        statics_by_key: dict[tuple[RegimeName, ...], _BQSEGMRideAlongStatics] = {}
        cliff_candidates_by_key: dict[tuple[RegimeName, ...], bool] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        for period in active_periods:
            plan = _build_bqsegm_continuation_plan(
                context=context,
                period=period,
                reachable_targets=reachable_targets,
                post_decision_name=self.post_decision_function,
                stochastic_node_batch_size=self.stochastic_node_batch_size,
            )
            key = (*plan.carry_targets, "|", *plan.scalar_targets)
            if key not in continuation_cores:
                statics = _bqsegm_ride_along_statics(
                    savings_grid=savings_grid,
                    schedule_spec=schedule_spec,
                    continuation_plan=plan,
                    envelope_segment_block_size=self.envelope_segment_block_size,
                    cell_block_size=self.cell_block_size,
                    interval_batch_size=self.interval_batch_size,
                    publish_jump_topology=self.jump_read == "one_sided",
                )
                # Save-to-cliff candidates need the regime's own carry read
                # (the cliffs are the self-schedule's); a period whose targets
                # exclude the regime itself solves without them.
                cliff_candidates = (
                    statics.n_published_jumps > 0
                    and context.regime_name in plan.child_reads
                )
                continuation_core = _build_bqsegm_continuation_core(
                    savings_grid=savings_grid,
                    continuation_plan=plan,
                    statics=statics,
                    regime_name=context.regime_name,
                    cliff_candidates=cliff_candidates,
                )
                envelope_core = _build_bqsegm_envelope_core(
                    savings_grid=savings_grid,
                    schedule_spec=schedule_spec,
                    statics=statics,
                )
                continuation_cores[key] = (
                    jax.jit(continuation_core)
                    if context.enable_jit
                    else continuation_core
                )
                envelope_cores[key] = (
                    jax.jit(envelope_core) if context.enable_jit else envelope_core
                )
                statics_by_key[key] = statics
                cliff_candidates_by_key[key] = cliff_candidates
            period_kernels[period] = _RideAlongBQSEGMPeriodKernel(
                continuation_core=continuation_cores[key],
                envelope_core=envelope_cores[key],
                statics=statics_by_key[key],
                cliff_candidates=cliff_candidates_by_key[key],
                regime_name=context.regime_name,
                reachable_targets=reachable_targets,
                transition_target_names=transition_target_names,
            )
        return SolutionKernels(
            period_kernels=MappingProxyType(period_kernels),
            continuation_template=_build_ride_along_carry_template(
                liquid_grid=liquid_grid,
                ride_shape=ride_shape,
                n_breakpoints=(
                    next(iter(statics_by_key.values())).n_published_jumps
                    if statics_by_key
                    else 0
                ),
            ),
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
    def requires_continuation_carries(self) -> bool:
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
        import jax  # noqa: PLC0415

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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: state, continuation, params."""
        return {
            **dict(state_action_space.states),
            "next_value": next_regime_to_V_arr[self.continuation_target],
            "next_marginal": next_regime_to_egm_carry[
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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> KernelResult:
        """Run the 1-D EGM step and assemble the `KernelResult`."""
        V_arr, carry = compiled_cores["main"](
            **state_action_space.states,
            next_value=next_regime_to_V_arr[self.continuation_target],
            next_marginal=next_regime_to_egm_carry[
                self.continuation_target
            ].marginal_utility,
            **_union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        )
        return KernelResult(V_arr=V_arr, carry=carry)


@dataclass(frozen=True, kw_only=True)
class _RideAlongBQSEGMPeriodKernel:
    """The case-piece EGM adapter for a regime carrying a ride-along co-state.

    The solve splits into two independently-jitted cores so neither XLA program
    carries the other's instruction graph:

    - `continuation`: reads `next_regime_to_egm_carry` and binds one continuation per
      ride-along cell through the transition-aware reader, returning the
      probability-weighted expected value and marginal over the savings grid.
    - `envelope`: re-derives each cell's budget and utility and solves the 1-D liquid
      step against the continuation core's stacks, returning the value array and the
      ride-along-axis-leading continuation carry a parent interpolates.

    Calling the adapter runs `continuation` then `envelope` unjitted and assembles the
    `KernelResult`; no JIT spans the two calls.
    """

    continuation_core: Callable
    """The jitted continuation half (`id`-deduped across periods)."""

    envelope_core: Callable
    """The jitted EGM/envelope half (`id`-deduped across periods)."""

    statics: _BQSEGMRideAlongStatics
    """Build-time config — supplies the envelope core's placeholder stack shapes."""

    cliff_candidates: bool
    """Whether this period's cores exchange save-to-cliff candidate columns.

    True only when the carry publishes jump topology and the period's targets
    include the regime itself (the cliffs are the self-schedule's).
    """

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    reachable_targets: frozenset[RegimeName]
    """The carry keys the core reads; the rolling carry is filtered to these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    @property
    def core(self) -> Callable:
        """The continuation core, exposed for any single-core reader."""
        return self.continuation_core

    def cores(self) -> Mapping[str, Callable]:
        """Return the continuation and envelope cores under their own keys."""
        return MappingProxyType(
            {
                "continuation": self.continuation_core,
                "envelope": self.envelope_core,
            }
        )

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _RideAlongBQSEGMPeriodKernel:
        """Bind the regime's and its carry targets' fixed params into both cores."""
        bound = _union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not bound:
            return self
        return replace(
            self,
            continuation_core=functools.partial(self.continuation_core, **bound),
            envelope_core=functools.partial(self.envelope_core, **bound),
        )

    def build_lower_args(
        self,
        *,
        core_key: str = "continuation",
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the named core's lowering arguments.

        The continuation core takes the state grids, the filtered carries, and the
        regime's flat params. The envelope core takes the same state and param args
        minus the carries, plus correctly-shaped zero placeholders for the two
        continuation stacks (statically derivable from the ride-along grid sizes, the
        savings grid, and the interval count).
        """
        states = dict(state_action_space.states)
        params = self._kernel_params(flat_params=flat_params)
        if core_key == "envelope":
            return {
                **states,
                **self._stack_placeholders(states=states),
                **params,
                "period": jnp.int32(period),
                "age": ages.values[period],
            }
        return {
            **states,
            "next_regime_to_egm_carry": _reachable_carry_subset(
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                reachable_targets=self.reachable_targets,
            ),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **params,
            "period": jnp.int32(period),
            "age": ages.values[period],
        }

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Run the continuation then envelope core and assemble the `KernelResult`."""
        states = dict(state_action_space.states)
        params = self._kernel_params(flat_params=flat_params)
        continuation_stacks = compiled_cores["continuation"](
            **states,
            next_regime_to_egm_carry=_reachable_carry_subset(
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                reachable_targets=self.reachable_targets,
            ),
            next_regime_to_V_arr=next_regime_to_V_arr,
            **params,
            period=jnp.int32(period),
            age=ages.values[period],
        )
        if self.cliff_candidates:
            cont_value_stack, cont_marginal_stack, cliff_stack = continuation_stacks
            cliff_kwargs = {"cliff_savings_stack": cliff_stack}
        else:
            cont_value_stack, cont_marginal_stack = continuation_stacks
            cliff_kwargs = {}
        V_arr, carry = compiled_cores["envelope"](
            **states,
            cont_value_stack=cont_value_stack,
            cont_marginal_stack=cont_marginal_stack,
            **cliff_kwargs,
            **params,
            period=jnp.int32(period),
            age=ages.values[period],
        )
        return KernelResult(V_arr=V_arr, carry=carry)

    def _stack_placeholders(self, *, states: Mapping[str, object]) -> dict[str, object]:
        """Zero placeholders for the envelope core's continuation stacks.

        The interval regime reads one continuation row per declared interval, so the
        stacks carry an interval axis between the ride-cell and savings axes; the
        non-interval regime reads a single row over the savings grid.
        """
        n_ride_cells = self.statics.n_ride_cells(states=states)
        n_extra = 2 * self.statics.n_published_jumps if self.cliff_candidates else 0
        n_savings = self.statics.n_savings + n_extra
        shape: tuple[int, ...] = (
            (n_ride_cells, self.statics.n_intervals, n_savings)
            if self.statics.next_state_reads_liquid
            else (n_ride_cells, n_savings)
        )
        zeros = jnp.zeros(shape, dtype=canonical_float_dtype())
        placeholders: dict[str, object] = {
            "cont_value_stack": zeros,
            "cont_marginal_stack": zeros,
        }
        if self.cliff_candidates:
            cliff_shape: tuple[int, ...] = (
                (n_ride_cells, self.statics.n_intervals, n_extra)
                if self.statics.next_state_reads_liquid
                else (n_ride_cells, n_extra)
            )
            placeholders["cliff_savings_stack"] = jnp.zeros(
                cliff_shape, dtype=canonical_float_dtype()
            )
        return placeholders

    def _kernel_params(self, *, flat_params: FlatParams) -> dict[str, object]:
        """Flat params fed into the cores: the regime's plus its targets'."""
        return _union_free_params(
            flat_params=flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )


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
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: states, continuation, params."""
        return self._core_args(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
            flat_params=flat_params,
        )

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> KernelResult:
        """Run the G2EGM step and assemble the `KernelResult`."""
        V_arr = compiled_cores["main"](
            **self._core_args(
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                flat_params=flat_params,
            )
        )
        return KernelResult(V_arr=V_arr)

    def _core_args(
        self,
        *,
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_egm_carry: Mapping[RegimeName, ContinuationPayload],
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
                "next_marginal_retired": next_regime_to_egm_carry[
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


@dataclass(frozen=True, kw_only=True)
class _BQSEGMCaseSpec:
    """Build-time statics describing one binary case split (v1 scope)."""

    when_callable: Callable
    """The `when` piece — its contribution applies where the predicate holds."""
    otherwise_callable: Callable
    """The `otherwise` piece — its contribution applies where the predicate fails."""
    when_func: FunctionName
    """Qualified-name prefix of the `when` piece's params."""
    otherwise_func: FunctionName
    """Qualified-name prefix of the `otherwise` piece's params."""
    when_param_names: tuple[str, ...]
    """Parameter names of the `when` piece."""
    otherwise_param_names: tuple[str, ...]
    """Parameter names of the `otherwise` piece."""
    predicate_name: FunctionName
    """Qualified-name prefix of the boundary predicate's params."""
    threshold_name: str
    """Name of the predicate's threshold parameter."""
    equality_owner: EqualityOwner
    """Predicate side owning the exact-boundary point (`when` or `otherwise`)."""


# The only split output v1 knows how to route — an additive cash-on-hand shift.
_BQSEGM_V1_OUTPUT = "subsidy"


def _validate_bqsegm_boundary_scope(
    *,
    registry: BQSEGMRegistry,
    functions: Mapping[FunctionName, Callable[..., object]],
    liquid_state_name: str,
    reserved_names: frozenset[str],
) -> None:
    """Reject case-piece declarations outside the v1 BQSEGM scope.

    v1 implements exactly one binary split of an additive cash-on-hand `subsidy`
    across one jump boundary on the liquid state, owned by the `otherwise` side,
    with pieces that read only the flat params (not states or actions). Anything
    else (a `when`-owned boundary, a continuous kink or hard constraint, a
    boundary on another variable, a non-`subsidy` output, a state-dependent piece)
    is rejected here rather than silently solved under the wrong convention.
    """
    import inspect  # noqa: PLC0415

    from lcm.exceptions import BQSEGMCaseError  # noqa: PLC0415

    for piece_set in registry.piece_sets:
        if piece_set.output != _BQSEGM_V1_OUTPUT:
            msg = (
                f"BQSEGM v1 only splits an additive cash-on-hand "
                f"{_BQSEGM_V1_OUTPUT!r} output; the regime splits "
                f"{piece_set.output!r}. Richer split outputs are deferred."
            )
            raise BQSEGMCaseError(msg)
        for piece_name in (piece_set.when_func, piece_set.otherwise_func):
            params = inspect.signature(functions[piece_name]).parameters
            state_action_deps = sorted(set(params) & reserved_names)
            if state_action_deps:
                msg = (
                    f"BQSEGM v1 pieces read only the flat params; piece "
                    f"{piece_name!r} depends on the state/action "
                    f"{state_action_deps!r}. State-dependent pieces are deferred."
                )
                raise BQSEGMCaseError(msg)
    for predicate_name, meta in registry.boundaries.items():
        for surface in meta.boundaries:
            if surface.equality_owner != "otherwise":
                msg = (
                    f"BQSEGM v1 only supports `equality='otherwise'` boundaries; "
                    f"{predicate_name!r} owns equality on the "
                    f"{surface.equality_owner!r} side."
                )
                raise BQSEGMCaseError(msg)
            if surface.kind != "jump":
                msg = (
                    f"BQSEGM v1 only supports `kind='jump'` boundaries; "
                    f"{predicate_name!r} declares {surface.kind!r}."
                )
                raise BQSEGMCaseError(msg)
            if surface.variable != liquid_state_name:
                msg = (
                    f"BQSEGM v1 only supports a boundary on the liquid state "
                    f"{liquid_state_name!r}; {predicate_name!r} compares "
                    f"{surface.variable!r}."
                )
                raise BQSEGMCaseError(msg)


def _collect_bqsegm_case_spec(*, context: SolverBuildContext) -> _BQSEGMCaseSpec:
    """Collect the single binary case split from the regime's user functions."""
    import inspect  # noqa: PLC0415

    from _lcm.egm.bqsegm import collect_bqsegm_metadata  # noqa: PLC0415

    functions = cast(
        "Mapping[FunctionName, Callable[..., object]]",
        context.user_regimes[context.regime_name].functions,
    )
    registry = collect_bqsegm_metadata(functions=functions)
    if len(registry.piece_sets) != 1:
        msg = (
            "BQSEGM v1 supports exactly one split output; the regime declares "
            f"{len(registry.piece_sets)}."
        )
        raise RegimeInitializationError(msg)
    piece_set = registry.piece_sets[0]
    surfaces = registry.boundaries[piece_set.predicate_name].boundaries
    if len(surfaces) != 1:
        msg = (
            "BQSEGM v1 supports exactly one boundary surface; the predicate "
            f"{piece_set.predicate_name!r} declares {len(surfaces)}."
        )
        raise RegimeInitializationError(msg)
    space = context.state_action_space
    _validate_bqsegm_boundary_scope(
        registry=registry,
        functions=functions,
        liquid_state_name=space.state_names[0],
        reserved_names=frozenset(space.state_names) | frozenset(space.action_names),
    )
    when_callable = functions[piece_set.when_func]
    otherwise_callable = functions[piece_set.otherwise_func]
    return _BQSEGMCaseSpec(
        when_callable=when_callable,
        otherwise_callable=otherwise_callable,
        when_func=piece_set.when_func,
        otherwise_func=piece_set.otherwise_func,
        when_param_names=tuple(inspect.signature(when_callable).parameters),
        otherwise_param_names=tuple(inspect.signature(otherwise_callable).parameters),
        predicate_name=piece_set.predicate_name,
        threshold_name=surfaces[0].threshold,
        equality_owner=surfaces[0].equality_owner,
    )


def _build_bqsegm_core(
    *, savings_grid: Float1D, target: RegimeName, case_spec: _BQSEGMCaseSpec
) -> Callable:
    """Build the jittable case-piece EGM core closing over the case split.

    The core evaluates each piece's additive contribution and the boundary
    threshold from the regime's flat params, runs the two-case EGM merge, and
    returns the value array and the marginal-value carry on the liquid grid.
    """
    from _lcm.egm.bqsegm_step import bqsegm_one_asset_step  # noqa: PLC0415

    def core(
        *,
        liquid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        subsidy_when = case_spec.when_callable(
            **{
                p: params[f"{case_spec.when_func}__{p}"]
                for p in case_spec.when_param_names
            }
        )
        subsidy_otherwise = case_spec.otherwise_callable(
            **{
                p: params[f"{case_spec.otherwise_func}__{p}"]
                for p in case_spec.otherwise_param_names
            }
        )
        asset_limit = params[f"{case_spec.predicate_name}__{case_spec.threshold_name}"]
        value, marginal, _policy = bqsegm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            savings_grid=savings_grid,
            discount_factor=params["H__discount_factor"],
            crra=params["utility__crra"],
            return_liquid=params[f"{target}__next_liquid__return_liquid"],
            income=params[f"{target}__next_liquid__income"],
            subsidy_when=subsidy_when,
            subsidy_otherwise=subsidy_otherwise,
            asset_limit=asset_limit,
            equality_owner=case_spec.equality_owner,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        )
        return value, carry

    return core


@dataclass(frozen=True)
class _BQSEGMSource:
    """One breakpoint of one schedule, in solver-facing form.

    A regime may declare several piecewise-affine schedules, each bracketing on
    its own monotone income variable; every threshold of every schedule becomes
    one source. The solver maps each source to its per-ride-along-cell asset
    preimage in its own variable and merges all sources into one sorted partition.
    """

    variable: str
    """Name of the monotone schedule variable this breakpoint brackets on."""
    threshold_param_name: str
    """Qualified parameter name of this breakpoint's threshold."""
    kind: str
    """Discontinuity kind: `continuous_kink`, `jump`, or `hard_constraint`."""
    derived_of_liquid_dag: Callable | None
    """Composed schedule variable as a function of the liquid state, or `None`
    when the schedule varies in the liquid state directly (no preimage needed)."""
    derived_param_names: tuple[str, ...]
    """Unqualified parameter names the schedule variable reads (non-state args)."""
    derived_state_names: tuple[str, ...] = ()
    """Ride-along state names the schedule variable reads, so the per-cell call
    passes only the cell entries the derived DAG accepts."""
    threshold_index_state: str | None = None
    """Ride-along state indexing this breakpoint's threshold table, or `None` for a
    scalar threshold. When set, the threshold is read per cell as
    `threshold[cell_state, static_index]`."""
    threshold_static_index: int | None = None
    """Static column index into the threshold table, applied after the ride-along
    row index. `None` leaves the row-indexed value as-is."""
    threshold_subkey: str | None = None
    """Entry to select inside a `MappingLeaf` threshold param (`leaf.data[subkey]`),
    resolved before the ride-along row index and static column index. `None` when
    the threshold param is a bare array."""


@dataclass(frozen=True)
class _BQSEGMScheduleSpec:
    """Build-time statics for a continuous piecewise-affine schedule regime."""

    coh_of_liquid_dag: Callable
    """Composed `coh` as a function of the liquid state and qualified params."""
    coh_param_names: tuple[str, ...]
    """Qualified parameter names `coh` reads (everything but the state axes)."""
    utility_dag: Callable
    """Composed period utility as a function of the consumption action, the
    ride-along states it reads, and qualified utility params. The ride-along core
    binds it per cell to invert the Euler equation and evaluate the period value."""
    consumption_action_name: ActionName
    """Name of the continuous consumption action the period utility reads."""
    liquid_state_name: str
    """Name of the liquid state the schedule and budget vary in."""
    ride_along_state_names: tuple[str, ...]
    """State axes other than the liquid axis (the budget varies per ride-along cell)."""
    liquid_axis_pos: int
    """Index of the liquid axis in the canonical productmap state order. The
    ride-along core solves in working layout (ride axes leading the liquid axis)
    and moves the liquid axis to this position so the published value array follows
    the productmap order — a no-op when every ride-along axis is a discrete state
    sorting ahead of the liquid axis, a genuine transpose for a continuous co-state
    declared after it."""
    threshold_param_names: tuple[str, ...]
    """Qualified parameter names of the schedule's thresholds."""
    breakpoint_kinds: tuple[str, ...]
    """Discontinuity kind per threshold, in the schedule's declared order."""
    sources: tuple[_BQSEGMSource, ...] = ()
    """Every breakpoint across all declared schedules, merged on the liquid axis.
    The ride-along core maps each source to its own per-cell asset preimage."""
    discount_factor_dag: Callable | None = None
    """Composed `discount_factor` as a function of its ride-along state arguments and
    qualified params, or `None` when the regime uses pylcm's flat `H__discount_factor`
    parameter. When set, the ride-along core resolves the discount factor per cell."""
    discrete_action_name: str | None = None
    """Name of a single discrete action the budget shifts, enveloped over per ride
    cell, or `None` when the regime carries no discrete action. Excluded from
    `coh_param_names` — the envelope core binds it per branch."""
    discrete_action_codes: tuple[int, ...] = ()
    """Integer codes of the discrete action's grid values, in envelope order."""


def _fail_if_discrete_action_feeds_continuation(
    *,
    context: SolverBuildContext,
    action_name: str,
    liquid_state_name: str,
) -> None:
    """Reject a discrete action that shifts the continuation, not just the budget.

    The discrete envelope solves every branch against one shared next-period
    continuation, valid only when the action enters the current budget and
    utility alone. If the action feeds the regime transition or a non-liquid
    state's law of motion, each branch's continuation differs and the shared read
    is wrong. Feeding cash-on-hand (hence the liquid post-decision state) is the
    intended budget channel and is allowed.
    """
    import inspect  # noqa: PLC0415

    from lcm.transition import MarkovTransition  # noqa: PLC0415

    def _reject(where: str) -> None:
        msg = (
            f"BQSEGM's discrete envelope shares one continuation across the "
            f"branches of {action_name!r}, so the action may shift only the "
            f"current budget and utility; regime {context.regime_name!r} reads it "
            f"in {where}. Fix the action there, or use a solver that carries a "
            "branch-specific continuation."
        )
        raise RegimeInitializationError(msg)

    transition_probs = context.compute_regime_transition_probs
    if (
        transition_probs is not None
        and action_name in inspect.signature(transition_probs).parameters
    ):
        _reject("the regime transition")

    regime = context.user_regimes[context.regime_name]
    funcs: dict[str, Callable[..., object]] = {
        name: func for name, func in regime.functions.items() if callable(func)
    }

    def _law_reads_action(law: Callable[..., object]) -> bool:
        try:
            combined = concatenate_functions(
                {**funcs, "__continuation_target__": law},
                targets="__continuation_target__",
            )
        except Exception:  # noqa: BLE001  # unanalysable law: leave to other gates
            return False
        return action_name in inspect.signature(combined).parameters

    for state_name, law in regime.state_transitions.items():
        if state_name == liquid_state_name:
            continue
        candidates = law.values() if isinstance(law, Mapping) else [law]
        for candidate in candidates:
            func = (
                candidate.func if isinstance(candidate, MarkovTransition) else candidate
            )
            if callable(func) and _law_reads_action(func):
                _reject(f"the law of motion for {state_name!r}")


def _ride_discrete_action(
    *, context: SolverBuildContext
) -> tuple[str | None, tuple[int, ...]]:
    """Identify a single budget-shifting discrete action and its grid codes.

    Returns `(None, ())` when the regime carries no discrete action.
    """
    discrete_actions = context.state_action_space.discrete_actions
    if not discrete_actions:
        return None, ()
    name = next(iter(discrete_actions))
    return name, tuple(int(code) for code in discrete_actions[name])


def _collect_bqsegm_schedule_spec(
    *,
    context: SolverBuildContext,
    budget_target: str = "coh",
    continuous_state: StateName | None = None,
) -> _BQSEGMScheduleSpec:
    """Collect a regime's piecewise-affine schedules into one breakpoint partition.

    A regime may declare several schedules, each bracketing on its own monotone
    income variable (taxable income, MAGI, …); every threshold becomes a
    breakpoint source. Each source maps to its per-ride-along-cell asset preimage
    in its own variable, and the sources merge into one sorted liquid partition.
    The budget node (`budget_target`) is composed once as a function of the liquid
    state, read per interval to recover the active affine segment.
    """
    import inspect  # noqa: PLC0415

    from _lcm.egm.bqsegm import collect_bqsegm_metadata  # noqa: PLC0415

    user_functions = cast(
        "Mapping[FunctionName, Callable[..., object]]",
        context.user_regimes[context.regime_name].functions,
    )
    registry = collect_bqsegm_metadata(functions=user_functions)
    schedules = registry.piecewise_affine_schedules
    if not schedules:
        msg = "BQSEGM schedule path needs at least one piecewise-affine schedule."
        raise RegimeInitializationError(msg)
    state_names = context.state_action_space.state_names
    # The Euler axis is one continuous state, not the first state axis: the
    # canonical order leads with discrete states, so a ride-along co-state sorts
    # ahead of the liquid axis. The remaining continuous states — a co-state (AIME)
    # or stochastic processes — ride along, integrated by the continuation reader.
    # When the regime carries more than one continuous state the Euler axis is named
    # via the solver's `continuous_state`; a single continuous state is the liquid
    # axis unambiguously. A schedule on the liquid state varies in it directly; a
    # schedule on a derived monotone quantity (gross income, MAGI) maps each
    # threshold to a per-ride-along-cell asset preimage.
    continuous_states = tuple(
        name for name in state_names if isinstance(context.grids[name], ContinuousGrid)
    )
    if continuous_state is not None:
        if continuous_state not in continuous_states:
            msg = (
                f"BQSEGM `continuous_state={continuous_state!r}` is not a continuous "
                f"state of the regime; its continuous states are {continuous_states}."
            )
            raise RegimeInitializationError(msg)
        liquid_state_name = continuous_state
    elif len(continuous_states) != 1:
        msg = (
            "BQSEGM schedule path needs exactly one continuous (liquid) state, or "
            "`continuous_state` naming the Euler axis when the regime carries a "
            f"continuous co-state; the regime has {continuous_states}."
        )
        raise RegimeInitializationError(msg)
    else:
        liquid_state_name = continuous_states[0]
    ride_along_state_names = tuple(
        name for name in state_names if name != liquid_state_name
    )
    has_derived = any(schedule.variable != liquid_state_name for schedule in schedules)
    if has_derived and not ride_along_state_names:
        derived_vars = tuple(
            schedule.variable
            for schedule in schedules
            if schedule.variable != liquid_state_name
        )
        msg = (
            f"BQSEGM schedule varies in the derived quantity/quantities "
            f"{derived_vars} but the regime has no ride-along co-state; a derived "
            "schedule maps thresholds to per-cell asset preimages and is only "
            "wired on the ride-along path."
        )
        raise RegimeInitializationError(msg)

    # Cache the composed derived-variable DAG per variable across its breakpoints.
    derived_dags: dict[str, tuple[Callable, tuple[str, ...], tuple[str, ...]]] = {}

    def _derived_dag(
        variable: str,
    ) -> tuple[Callable, tuple[str, ...], tuple[str, ...]]:
        if variable not in derived_dags:
            dag = concatenate_functions(dict(context.functions), targets=variable)
            dag_params = tuple(inspect.signature(dag).parameters)
            params = tuple(name for name in dag_params if name not in state_names)
            states_read = tuple(
                name for name in dag_params if name in ride_along_state_names
            )
            derived_dags[variable] = (dag, params, states_read)
        return derived_dags[variable]

    sources: list[_BQSEGMSource] = []
    for schedule in schedules:
        is_liquid_direct = schedule.variable == liquid_state_name
        dag, params, states_read = (
            (None, (), ()) if is_liquid_direct else _derived_dag(schedule.variable)
        )
        sources.extend(
            _BQSEGMSource(
                variable=schedule.variable,
                threshold_param_name=f"{schedule.output}__{bracket.threshold}",
                kind=bracket.kind,
                derived_of_liquid_dag=dag,
                derived_param_names=params,
                derived_state_names=states_read,
                threshold_index_state=bracket.indexed_by,
                threshold_static_index=bracket.static_index,
                threshold_subkey=bracket.threshold_subkey,
            )
            for bracket in schedule.breakpoints
        )

    # A single discrete action shifting the budget is enveloped over per ride
    # cell; it is neither a state nor a coh param, so exclude it from
    # `coh_param_names` (the envelope core binds it per branch).
    discrete_action_name, discrete_action_codes = _ride_discrete_action(context=context)
    coh_dag = concatenate_functions(dict(context.functions), targets=budget_target)
    coh_args = tuple(inspect.signature(coh_dag).parameters)
    coh_param_names = tuple(
        name
        for name in coh_args
        if name not in state_names and name != discrete_action_name
    )
    utility_dag = concatenate_functions(dict(context.functions), targets="utility")
    # A regime whose discount factor is a DAG function (e.g. a per-preference-type
    # beta indexed by a ride-along state) exposes it as a target; absent that, the
    # default flat `H__discount_factor` param drives discounting.
    discount_factor_dag = (
        concatenate_functions(dict(context.functions), targets="discount_factor")
        if "discount_factor" in context.functions
        else None
    )
    consumption_action_name = next(iter(context.state_action_space.continuous_actions))
    # `threshold_param_names` / `breakpoint_kinds` mirror the first schedule and
    # drive the non-ride-along continuous core, which is reached only for a
    # regime with no ride-along axis (a single liquid-direct schedule).
    first = schedules[0]
    threshold_param_names = tuple(
        f"{first.output}__{bp.threshold}" for bp in first.breakpoints
    )
    breakpoint_kinds = tuple(bp.kind for bp in first.breakpoints)
    return _BQSEGMScheduleSpec(
        coh_of_liquid_dag=coh_dag,
        coh_param_names=coh_param_names,
        utility_dag=utility_dag,
        consumption_action_name=consumption_action_name,
        liquid_state_name=liquid_state_name,
        ride_along_state_names=ride_along_state_names,
        liquid_axis_pos=state_names.index(liquid_state_name),
        threshold_param_names=threshold_param_names,
        breakpoint_kinds=breakpoint_kinds,
        sources=tuple(sources),
        discount_factor_dag=discount_factor_dag,
        discrete_action_name=discrete_action_name,
        discrete_action_codes=discrete_action_codes,
    )


def _schedule_kind_flags(
    kinds: tuple[str, ...],
) -> tuple[bool, bool, bool, tuple[bool, ...], tuple[bool, ...] | None]:
    """Classify a schedule's breakpoint kinds into the step-dispatch flags.

    Returns `(is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask)`:

    - `is_single_jump` — one jump, the binary recurring case.
    - `is_multi_jump` — every breakpoint a jump, the N-cliff recurring case.
    - `is_mixed` — jumps and kinks together, solved by the unified step.
    - `jump_mask` — per breakpoint, whether it is a jump (for the unified step).
    - `flat_mask` — per interval (N+1), whether a hard-constraint floors it, or
      `None` when no breakpoint is a hard constraint.
    """
    is_single_jump = kinds == ("jump",)
    is_multi_jump = len(kinds) > 1 and all(kind == "jump" for kind in kinds)
    is_mixed = "jump" in kinds and not all(kind == "jump" for kind in kinds)
    jump_mask = tuple(kind == "jump" for kind in kinds)
    has_floor = "hard_constraint" in kinds
    flat_mask = (
        tuple(
            j < len(kinds) and kinds[j] == "hard_constraint"
            for j in range(len(kinds) + 1)
        )
        if has_floor
        else None
    )
    return is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask


def _solve_cliffed_budget(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid: Float1D,
    savings_grid: Float1D,
    discount_factor: FloatND,
    crra: FloatND,
    return_liquid: FloatND,
    income: FloatND,
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    is_single_jump: bool,
    is_multi_jump: bool,
    is_mixed: bool,
    jump_mask: tuple[bool, ...],
    flat_mask: tuple[bool, ...] | None,
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of a cliffed single-liquid budget, dispatching on kind.

    Reads the continuation jump-aware at every jump (no bridging), so the solve
    is exact through recurring jumps, not only at a terminal-adjacent period.
    The kind flags come from `_schedule_kind_flags`. Returns this period's value,
    marginal value of liquid, and consumption policy on `liquid`.
    """
    from _lcm.egm.bqsegm_step import (  # noqa: PLC0415
        bqsegm_multi_interval_step,
        bqsegm_one_asset_step,
        bqsegm_recurring_jump_step,
        bqsegm_unified_step,
    )

    gross_return = 1.0 + return_liquid
    if is_single_jump:
        # A single jump in cash-on-hand is the binary case the v1 step solves
        # exactly, including its recurring jumped continuation: each interval's
        # affine segment has slope 1, so its intercept is the additive cash-on-hand
        # level on that side of the cliff.
        return bqsegm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            crra=crra,
            return_liquid=return_liquid,
            income=income,
            subsidy_when=coh_intercepts[0],
            subsidy_otherwise=coh_intercepts[1],
            asset_limit=breakpoints[0],
            equality_owner="otherwise",
        )
    if is_multi_jump:
        # N cliffs: each affine segment has slope 1, so its intercept is the additive
        # cash-on-hand level on that side, and the recurring step resolves every jump
        # (boundary-targeting + jump-aware continuation).
        return bqsegm_recurring_jump_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            crra=crra,
            gross_return=gross_return,
            income=income,
            subsidy_levels=coh_intercepts,
            jump_breakpoints=breakpoints,
            equality_owner="otherwise",
        )
    if is_mixed:
        # Jumps and kinks together: the unified step solves each continuous case by
        # coh inversion and masks across the jumps. The jump_mask is aligned with the
        # sorted breakpoints (the schedule declares its thresholds ascending).
        return bqsegm_unified_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            crra=crra,
            gross_return=gross_return,
            income=income,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            jump_mask=jump_mask,
        )
    return bqsegm_multi_interval_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
        flat_interval_mask=flat_mask,
    )


def _build_bqsegm_continuous_core(
    *, savings_grid: Float1D, target: RegimeName, schedule_spec: _BQSEGMScheduleSpec
) -> Callable:
    """Build the jittable continuous-schedule EGM core for one continuation target.

    The core reads the schedule's thresholds as liquid breakpoints, recovers the
    active affine cash-on-hand segment per interval by differentiating the composed
    `coh` at each interval's representative, and runs the kind-appropriate EGM step.
    """
    from _lcm.egm.bqsegm_breakpoints import (  # noqa: PLC0415
        interval_midpoints,
        interval_segment_coefficients,
    )

    kinds = schedule_spec.breakpoint_kinds
    is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask = (
        _schedule_kind_flags(kinds)
    )

    def core(
        *,
        liquid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        coh_params = {name: params[name] for name in schedule_spec.coh_param_names}

        def coh_of_liquid(scalar_liquid: FloatND) -> FloatND:
            return schedule_spec.coh_of_liquid_dag(
                **{schedule_spec.liquid_state_name: scalar_liquid}, **coh_params
            )

        breakpoints = jnp.sort(
            jnp.stack([params[name] for name in schedule_spec.threshold_param_names])
        )
        midpoints = interval_midpoints(liquid_grid=liquid, breakpoints=breakpoints)
        coh_slopes, coh_intercepts = interval_segment_coefficients(
            schedule=coh_of_liquid, interval_midpoints=midpoints
        )
        value, marginal, _policy = _solve_cliffed_budget(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid=liquid,
            savings_grid=savings_grid,
            discount_factor=params["H__discount_factor"],
            crra=params["utility__crra"],
            return_liquid=params[f"{target}__next_liquid__return_liquid"],
            income=params[f"{target}__next_liquid__income"],
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            is_single_jump=is_single_jump,
            is_multi_jump=is_multi_jump,
            is_mixed=is_mixed,
            jump_mask=jump_mask,
            flat_mask=flat_mask,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        )
        return value, carry

    return core


def _build_bqsegm_continuation_plan(
    *,
    context: SolverBuildContext,
    period: int,
    reachable_targets: frozenset[RegimeName],
    post_decision_name: FunctionName,
    stochastic_node_batch_size: int = 0,
) -> Any:  # noqa: ANN401  # `ContinuationPlan`; not annotated precisely (importing
    # module scope closes an import cycle (`continuation` → … → `lcm.solvers`).
    """Assemble the period's continuation plan for the ride-along case-piece core."""
    from _lcm.egm.continuation import (  # noqa: PLC0415
        build_continuation_plan,
        get_egm_continuation_targets,
    )

    # A regime running the case-piece solver is non-terminal, so it always has a
    # regime transition; narrow the optional for the continuation reader.
    compute_regime_transition_probs = context.compute_regime_transition_probs
    if compute_regime_transition_probs is None:
        msg = (
            f"BQSEGM regime {context.regime_name!r} has no regime transition; the "
            "case-piece solver is for non-terminal regimes only."
        )
        raise RegimeInitializationError(msg)
    carry_targets, scalar_targets = get_egm_continuation_targets(
        period=period,
        transitions=context.transitions,
        reachable_targets=reachable_targets,
        regimes_to_active_periods=context.regimes_to_active_periods,
        regime_to_v_interpolation_info=context.regime_to_v_interpolation_info,
    )
    return build_continuation_plan(
        user_regimes=context.user_regimes,
        functions=context.functions,
        transitions=context.transitions,
        stochastic_transition_names=context.stochastic_transition_names,
        carry_targets=carry_targets,
        scalar_targets=scalar_targets,
        compute_regime_transition_probs=compute_regime_transition_probs,
        post_decision_name=post_decision_name,
        stochastic_node_batch_size=stochastic_node_batch_size,
        regime_to_v_interpolation_info=context.regime_to_v_interpolation_info,
    )


def _solve_ride_along_cell_step(
    *,
    has_jump: bool,
    jump_positions: tuple[Any, ...],
    cont_value: Float1D,
    cont_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: FloatND,
    utility_of_action: Callable[[FloatND], FloatND],
    inverse_marginal_utility: Callable[[FloatND], FloatND],
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    extra_savings: Float1D | None = None,
    extra_cont_value: Float1D | None = None,
) -> tuple[Float1D, Float1D, Float1D]:
    """Run one ride-along cell's 1-D case-piece step against savings continuation.

    A pure-kink schedule uses the continuous multi-interval step; a schedule with a
    jump breakpoint uses the unified jump-and-kink step, both reading the expected
    value and marginal already evaluated on the savings grid. The Euler inversion,
    the period value, and the marginal value of liquid all read the regime's own
    utility through `utility_of_action` and `inverse_marginal_utility` (bound to this
    cell). The jump positions locate the jump breakpoints in the sorted partition —
    static for a single variable, a per-cell traced tuple when several variables
    reorder per cell.
    """
    from _lcm.egm.bqsegm_step import (  # noqa: PLC0415
        bqsegm_multi_interval_step_savings,
        bqsegm_unified_step_savings,
    )

    if has_jump:
        return bqsegm_unified_step_savings(
            cont_value=cont_value,
            cont_marginal=cont_marginal,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            inverse_marginal_utility=inverse_marginal_utility,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            jump_positions=jump_positions,
            extra_savings=extra_savings,
            extra_cont_value=extra_cont_value,
        )
    return bqsegm_multi_interval_step_savings(
        cont_value=cont_value,
        cont_marginal=cont_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        utility_of_action=utility_of_action,
        inverse_marginal_utility=inverse_marginal_utility,
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
    )


def _ride_along_jump_config(
    kinds: tuple[str, ...], *, n_variables: int
) -> tuple[BoolND, int, bool, tuple[int, ...], bool]:
    """Derive the merged partition's jump statics from the declared breakpoint kinds.

    Returns the per-breakpoint jump flags, the static jump count, whether any jump
    is present, the declared-order jump positions, and whether the jump positions
    must be recovered per cell — true only when jump and kink breakpoints declared
    on several variables interleave differently in each ride-along cell.
    """
    jump_flags = tuple(kind == "jump" for kind in kinds)
    n_jumps = sum(jump_flags)
    static_jump_positions = tuple(
        index for index, is_jump in enumerate(jump_flags) if is_jump
    )
    dynamic_jumps = n_variables > 1 and 0 < n_jumps < len(kinds)
    return (
        jnp.asarray(jump_flags),
        n_jumps,
        n_jumps > 0,
        static_jump_positions,
        dynamic_jumps,
    )


def _partition_jumps(
    preimages: Float1D,
    *,
    dynamic_jumps: bool,
    jump_flags: BoolND,
    n_jumps: int,
    static_jump_positions: tuple[int, ...],
) -> tuple[Float1D, tuple[Any, ...]]:
    """Sort a cell's breakpoint preimages and locate the jumps in the sorted order.

    With fixed jump positions the declared-order positions carry over; when the
    jumps reorder per cell the sorted-order jump indices are recovered from the
    permutation that sorts the preimages.
    """
    if dynamic_jumps:
        order = jnp.argsort(preimages)
        sorted_jumps = jnp.nonzero(jump_flags[order], size=n_jumps)[0]
        return preimages[order], tuple(sorted_jumps[k] for k in range(n_jumps))
    return jnp.sort(preimages), static_jump_positions


def _indexed_threshold_value(
    *,
    table: Any,  # noqa: ANN401  # scalar param, threshold table, or mapping leaf
    subkey: str | None,
    index_state: str | None,
    static_index: int | None,
    cell: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Read a breakpoint threshold from its param for one ride-along cell.

    The param is resolved to a value in this order:
    - `subkey` selects an entry inside a `MappingLeaf` (`leaf.data[subkey]`).
    - `index_state` reads the row at that ride-along state's code in this cell.
    - `static_index` selects a column (e.g. a bracket edge).

    A scalar threshold leaves every step disabled and passes through unchanged.
    """
    value = table
    if subkey is not None:
        value = value.data[subkey]
    if index_state is not None:
        value = value[cell[index_state]]
    if static_index is not None:
        value = value[static_index]
    return value


@dataclass(frozen=True)
class _BQSEGMRideAlongStatics:
    """Build-time config the ride-along continuation and envelope cores share.

    Both cores rebuild each ride-along cell's breakpoint partition, budget schedule,
    discount factor, and utility identically off this config; the continuation core
    additionally reads the regime transition through `bind_continuation`. Every field
    is a Python-level static derived once from the schedule spec and continuation plan.
    """

    sources: tuple[_BQSEGMSource, ...]
    """Every declared breakpoint, merged on the liquid axis."""
    jump_flags_arr: BoolND
    """Per-source jump indicator in declared order."""
    n_jumps: int
    """Number of jump breakpoints across all sources."""
    publish_jump_topology: bool
    """Whether the carry publishes jump preimages as duplicated row abscissae.

    `False` (`BQSEGM.jump_read == "bridged"`) keeps the within-period case solve
    jump-aware but carries plain liquid-grid rows with no breakpoints, so
    parents interpolate across the cliffs and the stochastic-dim fold stays
    available.
    """
    has_jump: bool
    """Whether any declared breakpoint is a jump (vs. a continuous kink)."""
    static_jump_positions: tuple[int, ...]
    """Jump indices in the sorted partition when a single variable fixes the order."""
    dynamic_jumps: bool
    """Whether the sorted-order jump indices must be recovered per cell."""
    liquid_name: str
    """Name of the liquid (Euler) state."""
    ride_names: tuple[str, ...]
    """Ride-along state axes (the budget varies per cell over these)."""
    state_names: tuple[str, ...]
    """Liquid plus ride-along state names — the kwargs that are state grids."""
    next_state_reads_liquid: bool
    """Whether a carry target's next-state law reads the current liquid state, so the
    continuation is piecewise-constant across declared intervals and the per-interval
    path applies."""
    interval_batch_size: int
    """Batch size for the per-interval continuation read: `0` evaluates all
    intervals in one vectorized pass, a positive size runs sequential chunks of
    that many intervals."""
    consumption_action_name: ActionName
    """Name of the continuous consumption action the period utility reads."""
    utility_param_names: tuple[str, ...]
    """Qualified utility params (excluding the consumption action and states)."""
    utility_state_names: tuple[str, ...]
    """Ride-along states the period utility reads, bound per cell."""
    coh_state_names: tuple[str, ...]
    """Ride-along states the cash-on-hand schedule reads, bound per cell."""
    discount_param_names: tuple[str, ...]
    """Qualified params the discount-factor DAG reads, or empty for flat discount."""
    discount_state_names: tuple[str, ...]
    """Ride-along states the discount-factor DAG reads, or empty for flat discount."""
    n_intervals: int
    """Number of liquid intervals the breakpoints split each cell into (N + 1)."""
    n_savings: int
    """Length of the post-decision savings grid."""
    envelope_segment_block_size: int
    """Block size for streaming the merged upper envelope over candidate segments;
    `0` keeps the one-shot dense envelope (see `BQSEGM.envelope_segment_block_size`)."""
    cell_block_size: int
    """Block size for streaming both ride-along cores over ride cells; `0` vmaps
    the whole flattened mesh at once (see `BQSEGM.cell_block_size`)."""

    @property
    def n_published_jumps(self) -> int:
        """Number of jump preimages the carry publishes per row."""
        return self.n_jumps if self.publish_jump_topology else 0

    def n_ride_cells(self, *, states: Mapping[str, object]) -> int:
        """Number of flattened ride-along cells for the given state grids."""
        count = 1
        for name in self.ride_names:
            count *= int(jnp.asarray(states[name]).shape[0])
        return count


def _bqsegm_ride_along_statics(
    *,
    savings_grid: Float1D,
    schedule_spec: _BQSEGMScheduleSpec,
    continuation_plan: Any,  # noqa: ANN401  # `ContinuationPlan`; import-cycle-safe
    envelope_segment_block_size: int = 0,
    cell_block_size: int = 0,
    interval_batch_size: int = 0,
    publish_jump_topology: bool = True,
) -> _BQSEGMRideAlongStatics:
    """Derive the static config the ride-along continuation and envelope cores share.

    Partitions the schedule's breakpoints, classifies the jump structure, and reads
    each component DAG's argument names (utility, cash-on-hand, discount factor) into
    the per-cell parameter and state splits both cores apply identically.
    """
    import inspect  # noqa: PLC0415

    sources = schedule_spec.sources
    kinds = tuple(source.kind for source in sources)
    if "hard_constraint" in kinds:
        msg = (
            "BQSEGM ride-along path supports continuous-kink and jump schedules; "
            f"got breakpoint kinds {kinds}. A hard-constraint (floor) breakpoint "
            "with a ride-along co-state is a later slice."
        )
        raise RegimeInitializationError(msg)
    n_variables = len({source.variable for source in sources})
    jump_flags_arr, n_jumps, has_jump, static_jump_positions, dynamic_jumps = (
        _ride_along_jump_config(kinds, n_variables=n_variables)
    )

    liquid_name = schedule_spec.liquid_state_name
    ride_names = schedule_spec.ride_along_state_names
    state_names = (liquid_name, *ride_names)

    # When a carry target's next-state law reads the liquid (Euler) state — a
    # current-asset boundary in `next_<liquid>` (e.g. a Medicaid transfer or pension
    # adjustment that switches at a declared cliff) — the continuation is constant
    # only within each declared interval. Detect it once: the per-interval path then
    # binds the liquid state to each interval's node and solves interval by interval.
    def _next_state_reads_liquid(target: str) -> bool:
        next_state_func = continuation_plan.child_reads[target].next_state_func
        return liquid_name in inspect.signature(next_state_func).parameters

    next_state_reads_liquid = any(
        _next_state_reads_liquid(target) for target in continuation_plan.carry_targets
    )

    # The period utility reads the consumption action, the ride-along states it
    # depends on (bound per cell), and qualified utility params (bound from kwargs).
    consumption_action_name = schedule_spec.consumption_action_name
    utility_arg_names = tuple(inspect.signature(schedule_spec.utility_dag).parameters)
    utility_param_names = tuple(
        name
        for name in utility_arg_names
        if name not in state_names and name != consumption_action_name
    )
    utility_state_names = tuple(
        name for name in ride_names if name in utility_arg_names
    )
    # The cash-on-hand schedule reads the liquid state plus whichever ride-along states
    # and params enter its DAG; bind exactly those per cell so unread ride-along states
    # (e.g. a preference type the budget ignores) are not forwarded to the DAG.
    coh_arg_names = tuple(inspect.signature(schedule_spec.coh_of_liquid_dag).parameters)
    coh_state_names = tuple(name for name in ride_names if name in coh_arg_names)
    # The discount factor is either pylcm's flat `H__discount_factor` param or, when
    # the regime supplies a `discount_factor` DAG function (e.g. a per-preference-type
    # beta read off a ride-along state), resolved per cell from that function's
    # qualified params and ride-along state arguments.
    discount_factor_dag = schedule_spec.discount_factor_dag
    if discount_factor_dag is None:
        discount_param_names: tuple[str, ...] = ()
        discount_state_names: tuple[str, ...] = ()
    else:
        discount_arg_names = tuple(inspect.signature(discount_factor_dag).parameters)
        discount_param_names = tuple(
            name for name in discount_arg_names if name not in state_names
        )
        discount_state_names = tuple(
            name for name in ride_names if name in discount_arg_names
        )

    return _BQSEGMRideAlongStatics(
        sources=sources,
        jump_flags_arr=jump_flags_arr,
        n_jumps=n_jumps,
        publish_jump_topology=publish_jump_topology,
        has_jump=has_jump,
        static_jump_positions=static_jump_positions,
        dynamic_jumps=dynamic_jumps,
        liquid_name=liquid_name,
        ride_names=ride_names,
        state_names=state_names,
        next_state_reads_liquid=next_state_reads_liquid,
        consumption_action_name=consumption_action_name,
        utility_param_names=utility_param_names,
        utility_state_names=utility_state_names,
        coh_state_names=coh_state_names,
        discount_param_names=discount_param_names,
        discount_state_names=discount_state_names,
        n_intervals=len(sources) + 1,
        n_savings=int(savings_grid.shape[0]),
        envelope_segment_block_size=envelope_segment_block_size,
        cell_block_size=cell_block_size,
        interval_batch_size=interval_batch_size,
    )


def _bqsegm_cell_breakpoints(
    *,
    statics: _BQSEGMRideAlongStatics,
    kwargs: Mapping[str, Any],
    cell: dict[str, Any],
    liquid_grid: Float1D,
    dtype: Any,  # noqa: ANN401  # canonical float dtype
) -> tuple[Float1D, tuple[Any, ...]]:
    """Build one ride-along cell's sorted liquid breakpoints and jump positions.

    Each declared schedule's threshold maps to its asset value in its own variable
    (directly for a liquid-state schedule, via the per-cell affine preimage for a
    derived-variable schedule), and the sources merge into one sorted partition. A
    degenerate boundary — a derived variable with (near-)zero asset slope in this cell,
    so the threshold is never crossed — has a non-finite preimage; clamping to a margin
    just outside the grid collapses it to an empty edge interval instead of poisoning a
    live interval's affine segment.
    """
    from _lcm.egm.bqsegm_breakpoints import (  # noqa: PLC0415
        clamp_breakpoints_to_grid,
        linear_asset_preimage,
    )

    liquid_name = statics.liquid_name

    def cell_breakpoint(source: _BQSEGMSource) -> FloatND:
        threshold_value = _indexed_threshold_value(
            table=kwargs[source.threshold_param_name],
            subkey=source.threshold_subkey,
            index_state=source.threshold_index_state,
            static_index=source.threshold_static_index,
            cell=cell,
        )
        threshold = jnp.asarray(threshold_value, dtype=dtype)
        if source.derived_of_liquid_dag is None:
            return threshold
        dag = source.derived_of_liquid_dag
        derived_params = {name: kwargs[name] for name in source.derived_param_names}
        cell_for_dag = {name: cell[name] for name in source.derived_state_names}

        def derived_of_liquid(scalar_liquid: FloatND) -> FloatND:
            return dag(**{liquid_name: scalar_liquid}, **cell_for_dag, **derived_params)

        return linear_asset_preimage(derived_of_liquid, threshold=threshold)

    preimages = clamp_breakpoints_to_grid(
        breakpoints=jnp.stack([cell_breakpoint(source) for source in statics.sources]),
        liquid_grid=liquid_grid,
    )
    return _partition_jumps(
        preimages,
        dynamic_jumps=statics.dynamic_jumps,
        jump_flags=statics.jump_flags_arr,
        n_jumps=statics.n_jumps,
        static_jump_positions=statics.static_jump_positions,
    )


def _cliff_savings_targets(
    *,
    continuation_plan: Any,  # noqa: ANN401  # `ContinuationPlan`; import-cycle-safe
    regime_name: RegimeName,
    statics: _BQSEGMRideAlongStatics,
    kwargs: dict[str, Any],
    cell: dict[str, Any],
    combo_pool: dict[str, Any],
    liquid_grid: Float1D,
    savings_grid: Float1D,
    dtype: Any,  # noqa: ANN401
    midpoints: Float1D | None = None,
) -> FloatND:
    """Map the self-read child's value cliffs to one-sided savings targets.

    A child value jump creates a legitimate one-sided optimum — save to just
    inside the cliff's owning side — that generically falls strictly between
    savings nodes. Per ride cell this recovers the cell's jump preimages in
    the child's liquid space, inverts the affine savings-form liquid law, and
    returns one target one float margin inside each side of every jump
    (`2 * n_jumps` entries). Targets outside the savings grid's span, or under
    a non-increasing liquid law, are NaN — the envelope's point-candidate
    family treats NaN entries as dead.
    """
    read = continuation_plan.child_reads[regime_name]
    post_decision_name = continuation_plan.post_decision_name
    breakpoints, jump_positions = _bqsegm_cell_breakpoints(
        statics=statics, kwargs=kwargs, cell=cell, liquid_grid=liquid_grid, dtype=dtype
    )
    jumps = jnp.stack([breakpoints[position] for position in jump_positions])

    def targets_for_pool(pool: dict[str, Any]) -> FloatND:
        def next_euler_state(savings_value: FloatND) -> FloatND:
            next_states = read.next_state_func(
                **pool, **{post_decision_name: savings_value}
            )
            return jnp.asarray(next_states[read.next_state_key], dtype=dtype)

        intercept = next_euler_state(jnp.asarray(0.0, dtype=dtype))
        slope = next_euler_state(jnp.asarray(1.0, dtype=dtype)) - intercept
        s_star = (jumps - intercept) / slope
        margin = jnp.maximum(jnp.abs(s_star), 1.0) * jnp.finfo(dtype).eps * 1e4
        candidates = jnp.stack([s_star - margin, s_star + margin], axis=-1).reshape(-1)
        valid = (
            (candidates >= savings_grid[0])
            & (candidates <= savings_grid[-1])
            & (slope > 0.0)
        )
        return jnp.where(valid, candidates, jnp.nan)

    if midpoints is None:
        return targets_for_pool(combo_pool)
    # An interval-bound liquid law: the savings-to-liquid map (and so each
    # cliff's savings preimage) is specific to the interval whose node the
    # liquid state is bound to — one target row per interval.
    liquid_name = statics.liquid_name
    return jax.vmap(
        lambda midpoint: targets_for_pool({**combo_pool, liquid_name: midpoint})
    )(midpoints)


def _build_bqsegm_continuation_core(
    *,
    savings_grid: Float1D,
    continuation_plan: Any,  # noqa: ANN401  # `ContinuationPlan`; import-cycle-safe
    statics: _BQSEGMRideAlongStatics,
    regime_name: RegimeName,
    cliff_candidates: bool,
) -> Callable:
    """Build the continuation half of the ride-along solve, jitted in isolation.

    Per ride-along cell the continuation is read through `bind_continuation` —
    integrating the next-period regime transition, stochastic shocks, the ride-along
    co-state transition, and the child value interpolation — and evaluated over the
    savings grid. The interval regime binds the liquid state to each interval's node
    and returns one continuation row per interval; the non-interval regime returns one
    row over the savings grid. The cells stack into `(n_ride_cells, [n_intervals,]
    n_savings)` expected-value and expected-marginal arrays the envelope core consumes.

    The heavy fan-out lives only here: this core builds no utility, cash-on-hand, or
    discount closure, so its compiled program never carries the EGM/envelope math.
    """
    from _lcm.egm.bqsegm_breakpoints import interval_midpoints  # noqa: PLC0415
    from _lcm.egm.continuation import bind_continuation  # noqa: PLC0415

    liquid_name = statics.liquid_name
    ride_names = statics.ride_names
    state_names = statics.state_names

    def continuation_core(
        *,
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        **kwargs: Any,  # noqa: ANN401  # state grids + flat params (mixed dtypes)
    ) -> tuple[FloatND, ...]:
        dtype = canonical_float_dtype()
        liquid = jnp.asarray(kwargs[liquid_name], dtype=dtype)
        param_pool = {key: v for key, v in kwargs.items() if key not in state_names}

        def cell_continuation(
            ride_values: tuple[Any, ...],
        ) -> tuple[FloatND, ...]:
            cell = dict(zip(ride_names, ride_values, strict=True))
            combo_pool = {**param_pool, **cell}

            def cliff_targets_for(midpoints: Float1D | None) -> FloatND:
                # Under the one-sided read, the cell also evaluates the blended
                # continuation at each self-read cliff's one-sided savings
                # targets; the extra columns ride at the end of the savings
                # axis and the envelope core adds them as point candidates.
                return _cliff_savings_targets(
                    continuation_plan=continuation_plan,
                    regime_name=regime_name,
                    statics=statics,
                    kwargs=kwargs,
                    cell=cell,
                    combo_pool=combo_pool,
                    liquid_grid=liquid,
                    savings_grid=savings_grid,
                    dtype=dtype,
                    midpoints=midpoints,
                )

            def query_with(targets: FloatND) -> Float1D:
                return jnp.concatenate(
                    [
                        savings_grid,
                        jnp.where(jnp.isnan(targets), savings_grid[0], targets),
                    ]
                )

            if statics.next_state_reads_liquid:
                # The next-period state law carries a current-asset boundary, so the
                # continuation is constant only within each declared interval. Bind
                # the liquid (Euler) state to each interval's representative node,
                # building one continuation row per interval. `lax.map` compiles the
                # continuation DAG once and XLA iterates, rather than a Python unroll
                # that bakes one copy of the per-cell DAG into the graph per interval.
                breakpoints, _ = _bqsegm_cell_breakpoints(
                    statics=statics,
                    kwargs=kwargs,
                    cell=cell,
                    liquid_grid=liquid,
                    dtype=dtype,
                )
                midpoints = interval_midpoints(
                    liquid_grid=liquid, breakpoints=breakpoints
                )
                cliff_targets = (
                    cliff_targets_for(midpoints) if cliff_candidates else None
                )

                def interval_rows(
                    interval_inputs: tuple[FloatND, ...],
                    combo_pool: dict[str, Any] = combo_pool,
                ) -> tuple[Float1D, Float1D]:
                    midpoint, *interval_targets = interval_inputs
                    interval_pool = {**combo_pool, liquid_name: midpoint}
                    interval_continuation = bind_continuation(
                        plan=continuation_plan,
                        combo_pool=interval_pool,
                        next_regime_to_egm_carry=next_regime_to_egm_carry,
                        dtype=dtype,
                    )
                    query = (
                        savings_grid
                        if not interval_targets
                        else query_with(interval_targets[0])
                    )
                    return jax.vmap(interval_continuation)(query)

                interval_inputs = (
                    (midpoints,)
                    if cliff_targets is None
                    else (midpoints, cliff_targets)
                )
                if statics.interval_batch_size:
                    rows = jax.lax.map(
                        interval_rows,
                        interval_inputs,
                        batch_size=statics.interval_batch_size,
                    )
                else:
                    rows = jax.vmap(interval_rows)(interval_inputs)
                if cliff_targets is None:
                    return rows
                return (*rows, cliff_targets)

            continuation = bind_continuation(
                plan=continuation_plan,
                combo_pool=combo_pool,
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                dtype=dtype,
            )
            cliff_targets = cliff_targets_for(None) if cliff_candidates else None
            rows = jax.vmap(continuation)(
                savings_grid if cliff_targets is None else query_with(cliff_targets)
            )
            if cliff_targets is None:
                return rows
            return (*rows, cliff_targets)

        ride_grids = tuple(jnp.asarray(kwargs[name]) for name in ride_names)
        mesh = jnp.meshgrid(*ride_grids, indexing="ij")
        flat_cells = tuple(grid.ravel() for grid in mesh)
        solve_cells = jax.vmap(lambda *vals: cell_continuation(vals))
        return _stream_cell_solves(
            solve_cells=solve_cells,
            inputs=flat_cells,
            cell_block=statics.cell_block_size,
        )

    return continuation_core


def _split_cliff_columns(
    *,
    cont_value: FloatND,
    cont_marginal: FloatND,
    n_nodes: int,
    has_cliff_columns: bool,
) -> tuple[FloatND, FloatND, FloatND | None]:
    """Split a cell's continuation rows into node columns and cliff columns.

    The continuation core rides the save-to-cliff targets' values at the end
    of the savings axis; the leading `n_nodes` columns are the savings-node
    rows the EGM step consumes, the rest feed the point-candidate family.
    """
    if not has_cliff_columns:
        return cont_value, cont_marginal, None
    return (
        cont_value[..., :n_nodes],
        cont_marginal[..., :n_nodes],
        cont_value[..., n_nodes:],
    )


def _vmapped_cell_solver(
    *,
    solve_one_cell: Callable,
    flat_cells: tuple[FloatND | IntND, ...],
    cont_value_stack: FloatND,
    cont_marginal_stack: FloatND,
    cliff_savings_stack: FloatND | None,
) -> tuple[Callable, tuple[FloatND | IntND, ...]]:
    """Vmap the per-cell solve over the ride mesh and its continuation stacks.

    The trailing per-cell inputs are the continuation core's stacks — value and
    marginal rows, plus the save-to-cliff savings targets when the one-sided
    read publishes jump topology.
    """
    if cliff_savings_stack is None:
        return jax.vmap(lambda *args: solve_one_cell(args[:-2], args[-2], args[-1])), (
            *flat_cells,
            cont_value_stack,
            cont_marginal_stack,
        )
    return jax.vmap(
        lambda *args: solve_one_cell(args[:-3], args[-3], args[-2], args[-1])
    ), (*flat_cells, cont_value_stack, cont_marginal_stack, cliff_savings_stack)


def _build_bqsegm_envelope_core(  # noqa: C901
    *,
    savings_grid: Float1D,
    schedule_spec: _BQSEGMScheduleSpec,
    statics: _BQSEGMRideAlongStatics,
) -> Callable:
    """Build the EGM/envelope half of the ride-along solve, jitted in isolation.

    Per ride-along cell this re-derives the budget schedule, discount factor, and
    utility from the same (states, params), then solves the 1-D continuous-budget step
    against the cell's continuation row supplied by the continuation core. The interval
    regime runs the per-interval continuation step; the non-interval regime runs the
    multi-interval or unified jump step. The cells stack into the value array and carry
    with the ride-along axes leading the liquid axis, matching the canonical layout.

    Re-deriving the breakpoints, cash-on-hand coefficients, and discount factor here is
    cheap closed-form work; this core calls no continuation reader, so the heavy
    transition fan-out never enters its compiled program.
    """
    from _lcm.egm.bqsegm_breakpoints import (  # noqa: PLC0415
        interval_midpoints,
        interval_segment_coefficients,
    )
    from _lcm.egm.bqsegm_step import (  # noqa: PLC0415
        bqsegm_per_interval_continuation_step_savings,
    )
    from _lcm.egm.numeric_inverse import (  # noqa: PLC0415
        numeric_inverse_marginal_utility,
    )

    liquid_name = statics.liquid_name
    ride_names = statics.ride_names
    discount_factor_dag = schedule_spec.discount_factor_dag
    # The continuous action solving the Euler equation is bracketed numerically when
    # the regime supplies no analytic inverse: a small floor up to a generous
    # multiple of the savings grid's top node (the resources scale). The clamped
    # near-zero-marginal corner whose root exceeds the bracket lands far to the right
    # and is discarded by the upper envelope.
    action_upper = savings_grid[-1] * 1000.0 + 1000.0
    action_lower = jnp.asarray(1e-8, dtype=action_upper.dtype)

    def envelope_core(
        *,
        cont_value_stack: FloatND,
        cont_marginal_stack: FloatND,
        cliff_savings_stack: FloatND | None = None,
        **kwargs: Any,  # noqa: ANN401  # state grids + flat params (mixed dtypes)
    ) -> tuple[FloatND, EGMCarry]:
        dtype = canonical_float_dtype()
        liquid = jnp.asarray(kwargs[liquid_name], dtype=dtype)
        coh_params = {name: kwargs[name] for name in schedule_spec.coh_param_names}
        utility_params = {name: kwargs[name] for name in statics.utility_param_names}
        discount_params = {name: kwargs[name] for name in statics.discount_param_names}

        def solve_one_cell(
            ride_values: tuple[Any, ...],
            cont_value: FloatND,
            cont_marginal: FloatND,
            cliff_savings: FloatND | None = None,
        ) -> tuple[Float1D, ...]:
            cont_value, cont_marginal, extra_cont_value = _split_cliff_columns(
                cont_value=cont_value,
                cont_marginal=cont_marginal,
                n_nodes=savings_grid.shape[0],
                has_cliff_columns=cliff_savings is not None,
            )
            cell = dict(zip(ride_names, ride_values, strict=True))
            cell_discount_factor = (
                kwargs["H__discount_factor"]
                if discount_factor_dag is None
                else discount_factor_dag(
                    **{name: cell[name] for name in statics.discount_state_names},
                    **discount_params,
                )
            )

            breakpoints, jump_positions = _bqsegm_cell_breakpoints(
                statics=statics,
                kwargs=kwargs,
                cell=cell,
                liquid_grid=liquid,
                dtype=dtype,
            )
            midpoints = interval_midpoints(liquid_grid=liquid, breakpoints=breakpoints)

            def utility_of_consumption(consumption_value: FloatND) -> FloatND:
                return schedule_spec.utility_dag(
                    **{statics.consumption_action_name: consumption_value},
                    **{name: cell[name] for name in statics.utility_state_names},
                    **utility_params,
                )

            marginal_utility = jax.grad(utility_of_consumption)

            def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
                return numeric_inverse_marginal_utility(
                    marginal_continuation=marginal_continuation,
                    marginal_utility=marginal_utility,
                    c_lower=action_lower,
                    c_upper=action_upper,
                )

            # With published jump breakpoints, the cell also publishes each
            # jump's preimage and its exact one-sided value limits: the liquid
            # query grid is augmented with a point just inside each side of
            # every jump, solved in the same call, and split back out
            # positionally. The bridged read skips the augmentation and
            # carries plain liquid-grid rows.
            if statics.n_published_jumps:
                jumps = jnp.stack([breakpoints[p] for p in jump_positions])
                query_grid, endog_row, unsort = _augment_liquid_with_jump_sides(
                    liquid_grid=liquid, jumps=jumps
                )
            else:
                query_grid = liquid

            def solve_branch(
                action_binding: Mapping[str, IntND],
            ) -> tuple[Float1D, Float1D, Float1D]:
                """Solve the cell's continuous subproblem for one discrete branch.

                `action_binding` binds the discrete action into cash-on-hand (empty
                when the regime carries no discrete action). The breakpoint partition,
                utility, and jump augmentation are action-independent and computed once
                in the enclosing scope.
                """

                def coh_of_liquid(scalar_liquid: FloatND) -> FloatND:
                    return schedule_spec.coh_of_liquid_dag(
                        **{liquid_name: scalar_liquid},
                        **{name: cell[name] for name in statics.coh_state_names},
                        **coh_params,
                        **action_binding,
                    )

                coh_slopes, coh_intercepts = interval_segment_coefficients(
                    schedule=coh_of_liquid, interval_midpoints=midpoints
                )
                if statics.next_state_reads_liquid:
                    # True cash-on-hand per liquid grid point keeps the step's corners
                    # feasible where a partly-binding kink makes an interval's recovered
                    # affine budget extrapolate below zero.
                    coh_grid = jax.vmap(coh_of_liquid)(query_grid)
                    return bqsegm_per_interval_continuation_step_savings(
                        cont_value=cont_value,
                        cont_marginal=cont_marginal,
                        liquid_grid=query_grid,
                        savings_grid=savings_grid,
                        discount_factor=cell_discount_factor,
                        utility_of_action=utility_of_consumption,
                        inverse_marginal_utility=inverse_marginal_utility,
                        coh_slopes=coh_slopes,
                        coh_intercepts=coh_intercepts,
                        breakpoints=breakpoints,
                        coh_grid=coh_grid,
                        envelope_segment_block_size=statics.envelope_segment_block_size,
                        extra_savings=cliff_savings,
                        extra_cont_value=extra_cont_value,
                    )
                return _solve_ride_along_cell_step(
                    has_jump=statics.has_jump,
                    jump_positions=jump_positions,
                    extra_savings=cliff_savings,
                    extra_cont_value=extra_cont_value,
                    cont_value=cont_value,
                    cont_marginal=cont_marginal,
                    liquid_grid=query_grid,
                    savings_grid=savings_grid,
                    discount_factor=cell_discount_factor,
                    utility_of_action=utility_of_consumption,
                    inverse_marginal_utility=inverse_marginal_utility,
                    coh_slopes=coh_slopes,
                    coh_intercepts=coh_intercepts,
                    breakpoints=breakpoints,
                )

            # A discrete action shifting the budget is enveloped over per cell: each
            # branch solves the continuous subproblem with the action bound into
            # cash-on-hand, then the discrete choice is taken by the upper envelope.
            # The ride-along discrete envelope is gated to jump-free schedules, so the
            # branch rows are plain liquid-grid rows.
            action_name = schedule_spec.discrete_action_name
            if action_name is not None:
                branch_steps = [
                    solve_branch({action_name: jnp.asarray(code, dtype=jnp.int32)})
                    for code in schedule_spec.discrete_action_codes
                ]
                value_row, marginal_row = _discrete_envelope_over_branches(
                    value_stack=jnp.stack([step[0] for step in branch_steps]),
                    marginal_stack=jnp.stack([step[1] for step in branch_steps]),
                    taste_shock_scale=0.0,
                )
                return (value_row, marginal_row)

            value_row, marginal_row, _policy_row = solve_branch({})
            if statics.n_published_jumps == 0:
                return (value_row, marginal_row)
            # The carry keeps the whole augmented row — the jump rides inside
            # the endogenous grid as a duplicated abscissa carrying its exact
            # one-sided value and marginal limits. Only the published value
            # array needs the original liquid nodes, sliced back out through
            # the sort permutation.
            value_at_liquid = value_row[unsort][: liquid.shape[0]]
            return (value_at_liquid, endog_row, value_row, marginal_row, jumps)

        ride_grids = tuple(jnp.asarray(kwargs[name]) for name in ride_names)
        ride_shape = tuple(int(grid.shape[0]) for grid in ride_grids)
        mesh = jnp.meshgrid(*ride_grids, indexing="ij")
        flat_cells = tuple(grid.ravel() for grid in mesh)
        solve_cells, stream_inputs = _vmapped_cell_solver(
            solve_one_cell=solve_one_cell,
            flat_cells=flat_cells,
            cont_value_stack=cont_value_stack,
            cont_marginal_stack=cont_marginal_stack,
            cliff_savings_stack=cliff_savings_stack,
        )
        stacks = _stream_cell_solves(
            solve_cells=solve_cells,
            inputs=stream_inputs,
            cell_block=statics.cell_block_size,
        )
        value_arr, carry = _assemble_ride_carry(
            stacks=stacks,
            n_jumps=statics.n_published_jumps,
            liquid=liquid,
            ride_shape=ride_shape,
            liquid_axis_pos=schedule_spec.liquid_axis_pos,
            dtype=dtype,
        )
        return value_arr, carry

    return envelope_core


@dataclass(frozen=True)
class _BQSEGMDiscreteSpec:
    """Build-time statics for a discrete-action regime with a smooth budget.

    The discrete action shifts cash-on-hand; the continuous consumption/savings
    subproblem is solved per discrete-action value by BQSEGM and the discrete choice
    is taken by the upper envelope over the branch values.
    """

    coh_of_liquid_dag: Callable
    """Composed `coh` as a function of the liquid state, the discrete action, and
    qualified params."""
    coh_param_names: tuple[str, ...]
    """Qualified parameter names `coh` reads (excluding the liquid state and the
    discrete action)."""
    liquid_state_name: str
    """Name of the liquid state the budget varies in."""
    discrete_action_name: str
    """Name of the discrete action enveloped over."""
    discrete_action_codes: tuple[int, ...]
    """Integer codes of the discrete action's grid values."""


def _collect_bqsegm_discrete_spec(
    *, context: SolverBuildContext, budget_target: str = "coh"
) -> _BQSEGMDiscreteSpec:
    """Collect the single binary/multi-valued discrete action of a smooth regime."""
    import inspect  # noqa: PLC0415

    space = context.state_action_space
    if len(space.discrete_actions) != 1:
        msg = (
            "BQSEGM discrete-envelope path supports exactly one discrete action; "
            f"the regime declares {len(space.discrete_actions)}."
        )
        raise RegimeInitializationError(msg)
    discrete_action_name = next(iter(space.discrete_actions))
    codes = tuple(int(code) for code in space.discrete_actions[discrete_action_name])
    liquid_state_name = space.state_names[0]
    _fail_if_discrete_action_feeds_continuation(
        context=context,
        action_name=discrete_action_name,
        liquid_state_name=liquid_state_name,
    )
    coh_dag = concatenate_functions(dict(context.functions), targets=budget_target)
    coh_args = tuple(inspect.signature(coh_dag).parameters)
    coh_param_names = tuple(
        name
        for name in coh_args
        if name not in (liquid_state_name, discrete_action_name)
    )
    return _BQSEGMDiscreteSpec(
        coh_of_liquid_dag=coh_dag,
        coh_param_names=coh_param_names,
        liquid_state_name=liquid_state_name,
        discrete_action_name=discrete_action_name,
        discrete_action_codes=codes,
    )


@dataclass(frozen=True)
class _BQSEGMScheduleDiscreteSpec:
    """Build-time statics for a discrete action over a cliffed single-liquid budget.

    Each discrete-action value shifts cash-on-hand and the budget also carries a
    declared schedule (kinks/jumps) on the liquid state. Per action value the
    continuous subproblem is solved by the multi-interval EGM step honouring the
    schedule, and the discrete choice is taken by the upper envelope over the
    branch values.
    """

    coh_of_liquid_action_dag: Callable
    """Composed budget node as a function of the liquid state, the discrete action,
    and qualified params."""
    coh_param_names: tuple[str, ...]
    """Qualified parameter names the budget reads (excluding the liquid state and the
    discrete action)."""
    liquid_state_name: str
    """Name of the liquid (Euler) state the budget varies in."""
    discrete_action_name: str
    """Name of the discrete action enveloped over."""
    discrete_action_codes: tuple[int, ...]
    """Integer codes of the discrete action's grid values."""
    threshold_param_names: tuple[str, ...]
    """Qualified parameter names of the schedule's thresholds (liquid breakpoints)."""
    breakpoint_kinds: tuple[str, ...]
    """Discontinuity kind per threshold, in the schedule's declared order."""


def _collect_bqsegm_schedule_discrete_spec(
    *,
    context: SolverBuildContext,
    budget_target: str = "coh",
    continuous_state: StateName | None = None,
) -> _BQSEGMScheduleDiscreteSpec:
    """Collect a single discrete action layered over a single-liquid cliff schedule."""
    import inspect  # noqa: PLC0415

    from _lcm.egm.bqsegm import collect_bqsegm_metadata  # noqa: PLC0415

    space = context.state_action_space
    if len(space.discrete_actions) != 1:
        msg = (
            "BQSEGM schedule+discrete path supports exactly one discrete action; "
            f"the regime declares {len(space.discrete_actions)}."
        )
        raise RegimeInitializationError(msg)
    discrete_action_name = next(iter(space.discrete_actions))
    codes = tuple(int(code) for code in space.discrete_actions[discrete_action_name])

    continuous_states = tuple(
        name
        for name in space.state_names
        if isinstance(context.grids[name], ContinuousGrid)
    )
    if continuous_state is not None:
        liquid_state_name = continuous_state
    elif len(continuous_states) == 1:
        liquid_state_name = continuous_states[0]
    else:
        msg = (
            "BQSEGM schedule+discrete path needs exactly one continuous (liquid) "
            f"state; the regime has {continuous_states}."
        )
        raise RegimeInitializationError(msg)

    _fail_if_discrete_action_feeds_continuation(
        context=context,
        action_name=discrete_action_name,
        liquid_state_name=liquid_state_name,
    )
    user_functions = {
        name: func for name, func in context.functions.items() if callable(func)
    }
    registry = collect_bqsegm_metadata(functions=user_functions)
    schedules = registry.piecewise_affine_schedules
    if any(schedule.variable != liquid_state_name for schedule in schedules):
        msg = (
            "BQSEGM schedule+discrete path handles schedules on the liquid state "
            "only; a derived-variable schedule needs the ride-along path."
        )
        raise RegimeInitializationError(msg)

    coh_dag = concatenate_functions(dict(context.functions), targets=budget_target)
    coh_args = tuple(inspect.signature(coh_dag).parameters)
    coh_param_names = tuple(
        name
        for name in coh_args
        if name not in (liquid_state_name, discrete_action_name)
    )
    first = schedules[0]
    threshold_param_names = tuple(
        f"{first.output}__{bp.threshold}" for bp in first.breakpoints
    )
    breakpoint_kinds = tuple(bp.kind for bp in first.breakpoints)
    return _BQSEGMScheduleDiscreteSpec(
        coh_of_liquid_action_dag=coh_dag,
        coh_param_names=coh_param_names,
        liquid_state_name=liquid_state_name,
        discrete_action_name=discrete_action_name,
        discrete_action_codes=codes,
        threshold_param_names=threshold_param_names,
        breakpoint_kinds=breakpoint_kinds,
    )


def _discrete_envelope_over_branches(
    *,
    value_stack: FloatND,
    marginal_stack: FloatND,
    taste_shock_scale: float,
) -> tuple[Float1D, Float1D]:
    """Take the discrete choice by the upper envelope over branch solves.

    `value_stack` and `marginal_stack` are `(n_branches, n_liquid)` — one solved
    branch per discrete-action value. Returns the enveloped value and marginal on
    the liquid grid:

    - Hard maximum (`taste_shock_scale == 0`): `max` over branches, with the
      winning branch's marginal by Danskin's theorem. At a value tie the envelope
      has a kink and the derivative is a subgradient set; the `argmax` convention
      selects the lowest-index tied branch's marginal — a well-defined subgradient,
      not the true (set-valued) derivative.
    - EV1 taste shocks (`taste_shock_scale > 0`): the scaled logsum value and the
      choice-probability-weighted branch marginal.
    """
    if taste_shock_scale == 0.0:
        modal = jnp.argmax(value_stack, axis=0)
        index = jnp.arange(value_stack.shape[1])
        return value_stack[modal, index], marginal_stack[modal, index]
    scaled = value_stack / taste_shock_scale
    probabilities = jax.nn.softmax(scaled, axis=0)
    value = taste_shock_scale * jax.scipy.special.logsumexp(scaled, axis=0)
    marginal = jnp.sum(probabilities * marginal_stack, axis=0)
    return value, marginal


def _build_bqsegm_schedule_discrete_core(
    *,
    savings_grid: Float1D,
    target: RegimeName,
    spec: _BQSEGMScheduleDiscreteSpec,
    taste_shock_scale: float,
) -> Callable:
    """Build the discrete-envelope core over a cliffed single-liquid budget.

    Per discrete-action value the core recovers the schedule's per-interval affine
    cash-on-hand and the liquid breakpoints and solves that branch with the
    kind-appropriate step (reading the continuation jump-aware, so the solve is
    exact through recurring jumps). The discrete choice is then taken by the upper
    envelope over the branch values — the hard maximum, or the EV1 logsum under a
    taste-shock scale.
    """
    from _lcm.egm.bqsegm_breakpoints import (  # noqa: PLC0415
        interval_midpoints,
        interval_segment_coefficients,
    )

    is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask = (
        _schedule_kind_flags(spec.breakpoint_kinds)
    )

    def core(
        *,
        liquid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        coh_params = {name: params[name] for name in spec.coh_param_names}
        breakpoints = jnp.sort(
            jnp.stack([params[name] for name in spec.threshold_param_names])
        )
        midpoints = interval_midpoints(liquid_grid=liquid, breakpoints=breakpoints)
        values: list[Float1D] = []
        marginals: list[Float1D] = []
        for code in spec.discrete_action_codes:

            def coh_of_liquid(scalar_liquid: FloatND, code: int = code) -> FloatND:
                return spec.coh_of_liquid_action_dag(
                    **{
                        spec.liquid_state_name: scalar_liquid,
                        spec.discrete_action_name: jnp.asarray(code),
                    },
                    **coh_params,
                )

            coh_slopes, coh_intercepts = interval_segment_coefficients(
                schedule=coh_of_liquid, interval_midpoints=midpoints
            )
            branch_value, branch_marginal, _policy = _solve_cliffed_budget(
                next_value=next_value,
                next_marginal=next_marginal,
                liquid=liquid,
                savings_grid=savings_grid,
                discount_factor=params["H__discount_factor"],
                crra=params["utility__crra"],
                return_liquid=params[f"{target}__next_liquid__return_liquid"],
                income=params[f"{target}__next_liquid__income"],
                coh_slopes=coh_slopes,
                coh_intercepts=coh_intercepts,
                breakpoints=breakpoints,
                is_single_jump=is_single_jump,
                is_multi_jump=is_multi_jump,
                is_mixed=is_mixed,
                jump_mask=jump_mask,
                flat_mask=flat_mask,
            )
            values.append(branch_value)
            marginals.append(branch_marginal)

        value, marginal = _discrete_envelope_over_branches(
            value_stack=jnp.stack(values),
            marginal_stack=jnp.stack(marginals),
            taste_shock_scale=taste_shock_scale,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        )
        return value, carry

    return core


def _build_bqsegm_discrete_core(
    *,
    savings_grid: Float1D,
    target: RegimeName,
    discrete_spec: _BQSEGMDiscreteSpec,
    taste_shock_scale: float,
) -> Callable:
    """Build the jittable discrete-envelope core for one continuation target.

    Per discrete-action value the core recovers the smooth budget's affine cash-on-
    hand and solves the continuous subproblem with the multi-interval step, then
    takes the discrete choice by the upper envelope (`bqsegm_discrete_envelope_step`).
    """
    from _lcm.egm.bqsegm_breakpoints import affine_coefficients  # noqa: PLC0415
    from _lcm.egm.bqsegm_step import (  # noqa: PLC0415
        bqsegm_discrete_envelope_step,
    )

    def core(
        *,
        liquid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        coh_params = {name: params[name] for name in discrete_spec.coh_param_names}
        empty_breakpoints = jnp.zeros((0,), dtype=liquid.dtype)
        choices: list[dict[str, Float1D]] = []
        for code in discrete_spec.discrete_action_codes:

            def coh_of_liquid(scalar_liquid: FloatND, code: int = code) -> FloatND:
                return discrete_spec.coh_of_liquid_dag(
                    **{
                        discrete_spec.liquid_state_name: scalar_liquid,
                        discrete_spec.discrete_action_name: jnp.asarray(code),
                    },
                    **coh_params,
                )

            slope, intercept = affine_coefficients(coh_of_liquid)
            choices.append(
                {
                    "coh_slopes": jnp.reshape(slope, (1,)),
                    "coh_intercepts": jnp.reshape(intercept, (1,)),
                    "breakpoints": empty_breakpoints,
                }
            )
        value, marginal, _policy, _choice = bqsegm_discrete_envelope_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            savings_grid=savings_grid,
            discount_factor=params["H__discount_factor"],
            crra=params["utility__crra"],
            gross_return=1.0 + params[f"{target}__next_liquid__return_liquid"],
            income=params[f"{target}__next_liquid__income"],
            choices=tuple(choices),
            taste_shock_scale=taste_shock_scale,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        )
        return value, carry

    return core


def _build_one_asset_carry_template(*, liquid_grid: Float1D) -> EGMCarry:
    """Build the all-finite 1-D EGM carry template on the liquid grid."""
    return EGMCarry(
        endog_grid=liquid_grid,
        value=jnp.zeros_like(liquid_grid),
        marginal_utility=jnp.zeros_like(liquid_grid),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
    )


def _stream_cell_solves(
    *,
    solve_cells: Callable,
    inputs: tuple[FloatND | IntND, ...],
    cell_block: int,
) -> tuple[FloatND, ...]:
    """Run the vmapped per-cell solve over the flattened ride mesh.

    A non-positive (or mesh-covering) `cell_block` solves every cell in one
    vmap; otherwise the mesh is scanned in cell blocks so only one block's
    candidate buffers are in flight — padding repeats the last cell and its
    results are dropped after the scan.
    """
    n_cells = int(inputs[0].shape[0])
    if cell_block <= 0 or cell_block >= n_cells:
        return solve_cells(*inputs)
    pad = (-n_cells) % cell_block

    def to_blocks(arr: FloatND | IntND) -> FloatND | IntND:
        padded = (
            jnp.concatenate([arr, jnp.repeat(arr[-1:], pad, axis=0)]) if pad else arr
        )
        return padded.reshape(-1, cell_block, *arr.shape[1:])

    blocked = jax.lax.map(
        lambda args: solve_cells(*args), tuple(to_blocks(arr) for arr in inputs)
    )

    def from_blocks(arr: FloatND | IntND) -> FloatND | IntND:
        return arr.reshape(-1, *arr.shape[2:])[:n_cells]

    return tuple(from_blocks(arr) for arr in blocked)


def _assemble_ride_carry(
    *,
    stacks: tuple[FloatND, ...],
    n_jumps: int,
    liquid: Float1D,
    ride_shape: tuple[int, ...],
    liquid_axis_pos: int,
    dtype: Any,  # noqa: ANN401  # jnp dtype object
) -> tuple[FloatND, EGMCarry]:
    """Reshape the per-cell solve stacks into the value array and the carry.

    - With jump breakpoints, the cell solve returns the value at the liquid
      nodes plus the augmented carry rows (duplicated jump abscissae with
      one-sided limits) and the jump locations.
    - Without jumps, it returns plain liquid-grid rows and the carry sits on
      the shared broadcast grid.

    The published value array follows the productmap state order, so the
    liquid axis moves from the working layout's trailing position to its
    canonical index. The carry keeps the working layout (ride axes leading
    the row axis): it is read back only by `bind_continuation`, which
    produced it, so the round-trip stays self-consistent.
    """
    n_liquid = liquid.shape[0]
    if n_jumps:
        (
            value_stack,
            endog_stack,
            row_value_stack,
            row_marginal_stack,
            breakpoint_stack,
        ) = stacks
        n_row = n_liquid + 2 * n_jumps
        carry_rows = (
            endog_stack.reshape(*ride_shape, n_row).astype(dtype),
            row_value_stack.reshape(*ride_shape, n_row).astype(dtype),
            row_marginal_stack.reshape(*ride_shape, n_row).astype(dtype),
        )
        breakpoint_rows = breakpoint_stack.reshape(*ride_shape, n_jumps).astype(dtype)
    else:
        value_stack, marginal_stack = stacks
        carry_rows = (
            jnp.broadcast_to(liquid, (*ride_shape, n_liquid)).astype(dtype),
            value_stack.reshape(*ride_shape, n_liquid).astype(dtype),
            marginal_stack.reshape(*ride_shape, n_liquid).astype(dtype),
        )
        breakpoint_rows = None
    value_arr = jnp.moveaxis(
        value_stack.reshape(*ride_shape, n_liquid), -1, liquid_axis_pos
    )
    carry = EGMCarry(
        endog_grid=carry_rows[0],
        value=carry_rows[1],
        marginal_utility=carry_rows[2],
        taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        breakpoints=breakpoint_rows,
    )
    return value_arr, carry


def _augment_liquid_with_jump_sides(
    *, liquid_grid: Float1D, jumps: Float1D
) -> tuple[Float1D, Float1D, IntND]:
    """Insert a query point one float step inside each side of every jump.

    Returns the sorted augmented query grid, the matching published
    abscissae — the same order with each side point relabeled to its exact
    jump location, so the row carries the jump as a duplicated abscissa —
    and the permutation mapping sorted positions back to concatenation
    order (liquid nodes first, then left-side points, then right-side
    points).
    """
    evaluation_points = jnp.concatenate(
        [
            liquid_grid,
            jnp.nextafter(jumps, -jnp.inf),
            jnp.nextafter(jumps, jnp.inf),
        ]
    )
    published_abscissae = jnp.concatenate([liquid_grid, jumps, jumps])
    sort_order = jnp.argsort(evaluation_points)
    return (
        evaluation_points[sort_order],
        published_abscissae[sort_order],
        # int32 permutation: the augmented grid has at most a few hundred
        # entries.
        jnp.argsort(sort_order).astype(jnp.int32),
    )


def _build_ride_along_carry_template(
    *, liquid_grid: Float1D, ride_shape: tuple[int, ...], n_breakpoints: int
) -> EGMCarry:
    """Build the all-finite case-piece carry template with ride-along axes leading.

    Each ride-along cell publishes one liquid-grid carry row; the template carries
    the ride-along (discrete/passive) axes ahead of the liquid axis, matching the
    canonical value-function layout the continuation reader interpolates.
    """
    # Same pytree as the runtime carry: a regime with jump breakpoints holds
    # each jump inside its rows as a duplicated abscissa (two extra row slots
    # per jump) and publishes the jump locations (kink breakpoints leave the
    # value continuous and add no row slots), so the lowering template shares
    # both fixed shapes. Repeating the top node keeps the template rows
    # weakly ascending and all-finite.
    row = jnp.concatenate(
        [liquid_grid, jnp.repeat(liquid_grid[-1:], 2 * n_breakpoints)]
    )
    block = jnp.broadcast_to(row, (*ride_shape, row.shape[0]))
    return EGMCarry(
        endog_grid=block,
        value=jnp.zeros_like(block),
        marginal_utility=jnp.zeros_like(block),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
        breakpoints=(
            jnp.zeros((*ride_shape, n_breakpoints), dtype=liquid_grid.dtype)
            if n_breakpoints
            else None
        ),
    )


def _period_to_continuation_target(
    *, context: SolverBuildContext
) -> dict[int, RegimeName]:
    """Resolve each active period's single deterministic continuation target.

    The model's deterministic lifecycle transition reaches exactly one target
    next period: the regime among this regime's transition targets that is
    active at `period + 1`. The last active period continues into the target
    active at the period beyond the horizon's interior (the terminal regime).
    """
    own_active = set(context.regimes_to_active_periods[context.regime_name])
    target_active = {
        target: set(context.regimes_to_active_periods[target])
        for target in context.transitions
    }
    result: dict[int, RegimeName] = {}
    for period in sorted(own_active):
        reached = [
            target for target, active in target_active.items() if (period + 1) in active
        ]
        if len(reached) != 1:
            msg = (
                f"Regime '{context.regime_name}' does not reach exactly one "
                f"active target at period {period + 1}: candidates {reached}. "
                "The endogenous-grid solvers require a deterministic "
                "lifecycle transition (one active target per period)."
            )
            raise RegimeInitializationError(msg)
        result[period] = reached[0]
    return result


def _union_free_params(
    *,
    flat_params: FlatParams,
    regime_name: RegimeName,
    transition_target_names: tuple[RegimeName, ...],
) -> dict[str, object]:
    """Union the regime's free params with its transition targets' free params.

    The boundary step evaluates the target regime's transition params (e.g. the
    pension payout factor the source never reads), so the core needs the union;
    captured functions read only the keys they need.
    """
    params: dict[str, object] = dict(flat_params[regime_name])
    for target_name in transition_target_names:
        for key, value in flat_params.get(target_name, MappingProxyType({})).items():
            params.setdefault(key, value)
    return params


def _union_fixed_params(
    *,
    fixed_flat_params: FlatParams,
    regime_name: RegimeName,
    transition_target_names: tuple[RegimeName, ...],
) -> dict[str, object]:
    """Union the regime's and its targets' fixed params for core binding."""
    bound = dict(fixed_flat_params.get(regime_name, MappingProxyType({})))
    for target_name in transition_target_names:
        for key, value in fixed_flat_params.get(
            target_name, MappingProxyType({})
        ).items():
            bound.setdefault(key, value)
    return bound


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
