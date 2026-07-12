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
from typing import Literal, cast

import jax
import jax.numpy as jnp
from beartype import beartype
from dags import concatenate_functions, get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.beartype_conf import REGIME_CONF
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
from lcm.exceptions import RegimeInitializationError
from lcm.typing import (
    ActionName,
    ContinuousState,
    Float1D,
    FloatND,
    FunctionName,
    StateName,
    StateOrActionName,
)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GridSearch(Solver):
    """Grid-search solver over the full state-action product (the default)."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one max-Q-over-a period adapter per period.

        Periods sharing the same Q_and_F object reuse a single jitted core,
        and therefore a single compiled program.
        """
        import jax  # noqa: PLC0415

        from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a  # noqa: PLC0415

        built: dict[int, MaxQOverAFunction] = {}
        result: dict[int, PeriodKernel] = {}
        # Fold weights are the folded process's own marginal distribution
        # (`compute_transition_probs` returns an (n_points, n_points) matrix
        # whose every row is that marginal — the "IID" part — so row 0 is it).
        # `_validate_fold_declarations` rejects a runtime-parameterized
        # process, so this is a plain constant computed once here, at
        # kernel-build time — never inside the traced core.
        fold_weights = {
            name: cast(
                "_ContinuousStochasticProcess", context.grids[name]
            ).get_transition_probs()[0]
            for name in context.fold_state_names
        }
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
                    stakeholders=context.stakeholders,
                    weights=context.weights,
                    fold_state_names=context.fold_state_names,
                    fold_weights=fold_weights,
                )
                built[q_id] = jax.jit(func) if context.enable_jit else func
            result[period] = _GridSearchPeriodKernel(
                core=built[q_id],
                regime_name=context.regime_name,
                collective=context.stakeholders is not None,
                same_period_ref_regimes=context.same_period_ref_regimes,
                edge_target_regimes=context.edge_target_regimes,
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

    collective: bool = False
    """Whether the core is a collective (stakeholder-valued) reduction.

    COLLECTIVE-REGIMES (E1/E2). A collective core returns the pair `(V, D)` —
    the stakeholder-axis value array plus the boolean dissolution flag — instead of
    the plain V array; the adapter unpacks it into the `KernelResult`. `False`
    keeps the singleton default byte-identical.
    """

    same_period_ref_regimes: tuple[RegimeName, ...] = ()
    """Reference regimes whose same-period V the core reads (E2), or empty.

    When non-empty, `__call__` forwards the solve loop's
    `same_period_regime_to_V_arr` mapping into the core, and
    `build_lower_args` supplies matching zero templates (reusing the
    period-invariant `next_regime_to_V_arr` templates — a regime's V shape does
    not change across periods, so the next-period template is also the correct
    same-period lowering shape).
    """

    edge_target_regimes: tuple[RegimeName, ...] = ()
    """Target regimes reached through a gated edge (E3'), or empty.

    COLLECTIVE-REGIMES (E3'). When non-empty, `build_lower_args` and `__call__`
    replace each such target's entry in `next_regime_to_V_arr` with the gated
    continuation object ``Wbar`` supplied under `edge_regime_to_V_arr` (a
    per-source template at lowering, the freshly folded array at run time), so
    the source's continuation reads ``Wbar`` in place of the raw target V with no
    change to the compiled core. Empty keeps every other kernel byte-identical.
    """

    def _with_edge_substitution(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None,
    ) -> Mapping[RegimeName, FloatND]:
        """Replace edge targets' raw V with their gated ``Wbar`` (E3')."""
        if not self.edge_target_regimes:
            return next_regime_to_V_arr
        if edge_regime_to_V_arr is None:
            msg = (
                f"Regime '{self.regime_name}' declares gated edges into "
                f"{self.edge_target_regimes} but the solve loop passed no edge "
                "continuation arrays."
            )
            raise RuntimeError(msg)
        return MappingProxyType(
            {
                name: (
                    edge_regime_to_V_arr[name]
                    if name in self.edge_target_regimes
                    else arr
                )
                for name, arr in next_regime_to_V_arr.items()
            }
        )

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
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: the full state-action product.

        For an E2 regime (`same_period_ref_regimes` non-empty) the same-period
        reference V arrays are lowered with the zero templates already built for
        `next_regime_to_V_arr` — the same-period array of a reference regime has
        exactly its (period-invariant) V shape and sharding. For an E3' source
        (`edge_target_regimes` non-empty) each edge target's continuation is
        lowered with its ``Wbar`` template (target grid + source stakeholder
        axis), substituted for the raw target V.
        """
        next_regime_to_V_arr = self._with_edge_substitution(
            next_regime_to_V_arr=next_regime_to_V_arr,
            edge_regime_to_V_arr=edge_regime_to_V_arr,
        )
        lower_args: dict[str, object] = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(flat_params[self.regime_name]),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }
        if self.same_period_ref_regimes:
            lower_args["same_period_regime_to_V_arr"] = MappingProxyType(
                {
                    regime_name: next_regime_to_V_arr[regime_name]
                    for regime_name in self.same_period_ref_regimes
                }
            )
        return lower_args

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
        same_period_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelResult:
        """Evaluate the grid search and assemble the `KernelResult`.

        `same_period_regime_to_V_arr` is passed by the solve loop only for a
        regime declaring `same_period_refs` (E2); `edge_regime_to_V_arr` only for
        a regime declaring `gated_edges` (E3', substituted into
        `next_regime_to_V_arr` before the core call). Every other kernel keeps
        the uniform `PeriodKernel` call signature.
        """
        next_regime_to_V_arr = self._with_edge_substitution(
            next_regime_to_V_arr=next_regime_to_V_arr,
            edge_regime_to_V_arr=edge_regime_to_V_arr,
        )
        extra_kwargs: dict[str, object] = {}
        if self.same_period_ref_regimes:
            if same_period_regime_to_V_arr is None:
                msg = (
                    f"Regime '{self.regime_name}' declares same_period_refs on "
                    f"{self.same_period_ref_regimes} but the solve loop passed "
                    "no same-period V arrays."
                )
                raise RuntimeError(msg)
            extra_kwargs["same_period_regime_to_V_arr"] = same_period_regime_to_V_arr
        out = compiled_cores["main"](
            **state_action_space.states,
            **state_action_space.actions,
            next_regime_to_V_arr=next_regime_to_V_arr,
            **flat_params[self.regime_name],
            period=jnp.int32(period),
            age=ages.values[period],
            **extra_kwargs,
        )
        if self.collective:
            # COLLECTIVE-REGIMES (E1/E2): the collective core returns the pair
            # (stakeholder-axis V, dissolution flag D).
            V_arr, dissolution = out
            return KernelResult(V_arr=V_arr, dissolution=dissolution)
        return KernelResult(V_arr=out)


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


def _build_one_asset_carry_template(*, liquid_grid: Float1D) -> EGMCarry:
    """Build the all-finite 1-D EGM carry template on the liquid grid."""
    return EGMCarry(
        endog_grid=liquid_grid,
        value=jnp.zeros_like(liquid_grid),
        marginal_utility=jnp.zeros_like(liquid_grid),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
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
