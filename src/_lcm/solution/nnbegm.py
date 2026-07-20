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
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.egm.branch_aggregation import (
    DeterministicOuterMaximum,
    OuterBranchAggregator,
    UniformObservedFixedCost,
)
from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import (
    NestedEGMSimPolicy,
    OuterPolicyBank,
    derive_inner_sim_policy,
)
from _lcm.egm.numeric_inverse import numeric_inverse_marginal_utility
from _lcm.egm.outer_candidates import (
    OuterCandidateBank,
    OuterCandidateResult,
    build_outer_candidate_bank,
)
from _lcm.egm.outer_carry import collapse_continuous_candidate_bank
from _lcm.egm.outer_refinement import refine_outer_mesh
from _lcm.egm.outer_search import AdaptiveOuterMesh, FiniteOuterGrid, OuterSearch
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid, DiscreteGrid
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SimulationPolicy,
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
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.typing import FlatParams, RegimeName
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ActionName, Float1D, FloatND, FunctionName, StateName


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

    outer_search: OuterSearch | None = None
    """How the outer margin's candidates are generated and refined.

    `FiniteOuterGrid` reproduces the historical finite-candidate behavior;
    `AdaptiveOuterMesh` is the canonical continuous-outer approximation.
    Exactly one of `outer_search` and the legacy `outer_grid` must be set."""

    outer_grid: ContinuousGrid | None = None
    """Legacy exogenous candidate grid for the outer post-decision margin.

    Shorthand for `outer_search=FiniteOuterGrid(grid=...,
    batch_size=outer_batch_size)`; scheduled for deprecation once downstream
    models migrate to `outer_search`."""

    outer_no_adjustment_candidate: FunctionName | None = None
    """State-dependent no-adjustment map `s' = keep(s)` the keeper holds.

    `None` keeps the durable stock unchanged (identity)."""

    outer_batch_size: int = 0
    """Legacy companion to `outer_grid`: nodes solved per chunk before
    folding into the running maximum; `0` solves every node at once. A memory
    knob only — value-invariant. With `outer_search`, set the strategy's own
    `batch_size` instead."""

    branch_aggregator: OuterBranchAggregator = field(
        default_factory=DeterministicOuterMaximum
    )
    """How the keeper and adjuster branch values combine per state cell.

    `DeterministicOuterMaximum()` (default) is the historical hard maximum.
    `UniformObservedFixedCost(...)` integrates a uniform observed fixed
    adjustment cost analytically into the fold — continuous-outer
    (`AdaptiveOuterMesh`) only, and the shock must not appear as a solve
    state (its integration is exact, no grid needed)."""

    def __post_init__(self) -> None:
        spec = get_nnbegm_inner_spec(inner=self.inner)
        _fail_if_outer_batch_size_negative(self.outer_batch_size)
        search = self.resolved_outer_search
        match search:
            case FiniteOuterGrid():
                _fail_if_outer_grid_is_stochastic(search.grid)
            case AdaptiveOuterMesh():
                _fail_if_outer_grid_is_stochastic(search.initial_grid)
            case _:
                pass
        _fail_if_nnbegm_outer_post_decision_is_inner(
            outer_post_decision=self.outer_post_decision,
            inner_post_decision=spec.post_decision_function,
        )
        if isinstance(
            self.branch_aggregator, UniformObservedFixedCost
        ) and not isinstance(search, AdaptiveOuterMesh):
            msg = (
                "UniformObservedFixedCost aggregates the keeper/adjuster "
                "branches through the continuous collapse; it requires "
                "`outer_search=AdaptiveOuterMesh(...)`."
            )
            raise RegimeInitializationError(msg)

    @property
    def resolved_outer_search(self) -> OuterSearch:
        """The normalized outer-search strategy, legacy fields folded in.

        Raises:
            RegimeInitializationError: If neither or both of `outer_search`
                and `outer_grid` are set, or a nonzero `outer_batch_size`
                accompanies `outer_search`.

        """
        if self.outer_search is None:
            if self.outer_grid is None:
                msg = (
                    "NNBEGM requires an outer search: pass `outer_search=` "
                    "(canonical) or the legacy `outer_grid=`."
                )
                raise RegimeInitializationError(msg)
            return FiniteOuterGrid(
                grid=self.outer_grid, batch_size=self.outer_batch_size
            )
        if self.outer_grid is not None:
            msg = "Pass either `outer_search` or the legacy `outer_grid`, not both."
            raise RegimeInitializationError(msg)
        if self.outer_batch_size != 0:
            msg = (
                "`outer_batch_size` belongs to the legacy `outer_grid` "
                "interface; set the strategy's own `batch_size` on "
                "`outer_search` instead."
            )
            raise RegimeInitializationError(msg)
        return self.outer_search

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
        search = self.resolved_outer_search
        match search:
            case FiniteOuterGrid():
                outer_grid_values = search.grid.to_jax()
                outer_batch_size = search.batch_size
            case AdaptiveOuterMesh():
                outer_grid_values = search.initial_grid.to_jax()
                outer_batch_size = search.batch_size
            case _:
                msg = (
                    f"NNBEGM outer search strategy {type(search).__name__} "
                    "is not wired into the period kernels; use "
                    "FiniteOuterGrid or AdaptiveOuterMesh."
                )
                raise RegimeInitializationError(msg)
        spec = get_nnbegm_inner_spec(inner=self.inner)
        inner_action = _nnbegm_inner_action(
            context=context, outer_action=self.outer_action
        )
        # Carry-row axis names, in the carry contract's order: discrete states
        # first (V state order), then passive continuous states (every
        # continuous state except the inner Euler axis). Used to derive the
        # published inner policies for the nested simulation reader.
        row_discrete_state_names = tuple(
            name
            for name in context.state_action_space.state_names
            if isinstance(context.grids[name], DiscreteGrid)
        )
        row_passive_state_names = tuple(
            name
            for name in context.state_action_space.state_names
            if isinstance(context.grids[name], ContinuousGrid)
            and name != spec.continuous_state
        )
        inverse_marginal = _nested_inverse_marginal(
            context=context,
            rows_on_state_grid=self.carry_rows_share_state_grid,
            inner_action=inner_action,
            savings_top=float(spec.savings_grid.to_jax()[-1]),
        )
        branch_fixed_cost, branch_scale_function = _resolve_branch_fixed_cost(
            aggregator=self.branch_aggregator, context=context
        )
        period_kernels = MappingProxyType(
            {
                period: _NNBEGMPeriodKernel(
                    keeper_kernel=keeper_kernels.period_kernels[period],
                    adjuster_kernel=adjuster_kernel,
                    regime_name=context.regime_name,
                    outer_grid_values=outer_grid_values,
                    outer_post_decision=self.outer_post_decision,
                    outer_batch_size=outer_batch_size,
                    outer_search=search,
                    outer_action=self.outer_action,
                    inner_action=inner_action,
                    resources_target=spec.budget_target,
                    savings_lower_bound=float(spec.savings_grid.to_jax()[0]),
                    liquid_grid_values=context.grids[spec.continuous_state].to_jax(),
                    liquid_state_name=spec.continuous_state,
                    outer_no_adjustment_name=self.outer_no_adjustment_candidate,
                    inverse_marginal=inverse_marginal,
                    row_discrete_state_names=row_discrete_state_names,
                    row_passive_state_names=row_passive_state_names,
                    branch_fixed_cost=branch_fixed_cost,
                    branch_scale_function=branch_scale_function,
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

    outer_search: OuterSearch
    """The resolved outer-search strategy: `FiniteOuterGrid` collapses the
    exact finite candidate set, `AdaptiveOuterMesh` adaptively refines the
    shared mesh and collapses continuously."""

    outer_action: ActionName
    """The regime's outer continuous action (published for the nested
    simulation reader)."""

    inner_action: ActionName
    """The regime's inner continuous action (the consumption the published
    inner policies map resources to)."""

    resources_target: FunctionName
    """The inner budget node the published policy rows are read at."""

    savings_lower_bound: float
    """Lower bound of the inner savings grid (the intrinsic budget check of
    the simulation policy read)."""

    liquid_grid_values: Float1D
    """The inner Euler (liquid) state grid — the shared abscissae the inner
    NB-EGM's published carry rows are re-read on
    (`carry_rows_share_state_grid`)."""

    liquid_state_name: StateName
    """Name of the inner Euler (liquid) state (published for the nested
    simulation reader's row query)."""

    outer_no_adjustment_name: FunctionName | None
    """The keeper's no-adjustment candidate function name, or `None` when
    keeping holds the current durable unchanged (published for the nested
    simulation reader's keeper-action recovery)."""

    inverse_marginal: Callable[..., FloatND] | None
    """The regime's closed-form inverse marginal utility with
    `marginal_continuation` as its only free parameter, or `None` when
    unavailable — then no nested simulation payload is derived and simulate
    keeps the grid-argmax path."""

    row_discrete_state_names: tuple[StateName, ...]
    """Names of the carry rows' leading discrete-state axes, in axis order."""

    row_passive_state_names: tuple[StateName, ...]
    """Names of the carry rows' passive continuous-state axes (every
    continuous state except the inner Euler state), after the discrete
    states."""

    branch_fixed_cost: UniformObservedFixedCost | None
    """The uniform observed fixed-cost aggregator, or `None` for the
    deterministic keeper/adjuster maximum."""

    branch_scale_function: Callable[..., FloatND] | None
    """The fixed cost's scale function, arguments restricted to
    `period`/`age`/flat params (resolved per period at call time)."""

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
        """Solve keeper and adjuster bank, then collapse the finite candidates.

        The adjuster sweep first materializes every node's exact conditional
        solve into an `OuterCandidateBank` (the structure the continuous-outer
        interpolant and adaptive mesh consume), then the finite collapse folds
        the bank into the keeper exactly as the pre-bank incremental sweep did.
        `outer_batch_size` bounds how many node solves are dispatched before
        forcing them to device; the bank itself retains all candidates by
        design, so peak retention is one full bank regardless of batching.
        """
        keeper_result = self._solve_keeper(
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        if isinstance(self.outer_search, AdaptiveOuterMesh):
            return self._solve_continuous(
                keeper_result=keeper_result,
                config=self.outer_search,
                compiled_cores=compiled_cores,
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        bank = self._build_candidate_bank(
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        return _collapse_finite_candidate_bank(keeper=keeper_result, bank=bank)

    def _solve_continuous(
        self,
        *,
        keeper_result: KernelResult,
        config: AdaptiveOuterMesh,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Adaptively refine the shared outer mesh, then collapse continuously.

        The mesh driver's exact-solve callback runs the adjuster's inner
        solve per requested node (chunked by the strategy's `batch_size`)
        and caches every `OuterCandidateResult` by node value, so the final
        bank reuses the refinement solves instead of re-solving. The keeper
        stays a separate exact branch throughout; its `sim_policy` rides
        through unchanged until the continuous simulation reader lands.
        """
        adjuster_cores = _subcores(compiled_cores=compiled_cores, role="adjuster")
        cache: dict[float, OuterCandidateResult] = {}

        def solve_nodes(nodes_arr: Float1D) -> FloatND:
            requested = [float(node) for node in np.asarray(nodes_arr)]
            pending = [node for node in requested if node not in cache]
            chunk_size = config.batch_size or max(len(pending), 1)
            for chunk_start in range(0, len(pending), chunk_size):
                chunk = pending[chunk_start : chunk_start + chunk_size]
                chunk_results = [
                    self._solve_adjuster_node(
                        node=jnp.asarray(node),
                        adjuster_cores=adjuster_cores,
                        state_action_space=state_action_space,
                        next_regime_to_V_arr=next_regime_to_V_arr,
                        next_regime_to_continuation=next_regime_to_continuation,
                        flat_params=flat_params,
                        period=period,
                        ages=ages,
                    )
                    for node in chunk
                ]
                jax.block_until_ready(
                    [(result.V_arr, result.carry) for result in chunk_results]
                )
                cache.update(zip(chunk, chunk_results, strict=True))
            return jnp.stack([cache[node].V_arr for node in requested])

        mesh = refine_outer_mesh(
            initial_nodes=self.outer_grid_values,
            solve_at=solve_nodes,
            config=config,
        )
        bank = build_outer_candidate_bank(
            outer_nodes=mesh.nodes,
            results=[cache[float(node)] for node in np.asarray(mesh.nodes)],
        )
        if self.branch_fixed_cost is None:
            fixed_cost_scale = None
            fixed_cost_support = None
        else:
            fixed_cost_scale = _resolve_branch_scale(
                scale_function=self.branch_scale_function,
                regime_params=flat_params[self.regime_name],
                period=period,
                ages=ages,
            )
            fixed_cost_support = (
                self.branch_fixed_cost.lower,
                self.branch_fixed_cost.upper,
            )
        collapse = collapse_continuous_candidate_bank(
            keeper_v_arr=keeper_result.V_arr,
            keeper_carry=cast("EGMCarry", keeper_result.continuation),
            bank=bank,
            config=config,
            fixed_cost_scale=fixed_cost_scale,
            fixed_cost_support=fixed_cost_support,
        )
        # Derive both branches' inner simulation policies. An NB-EGM inner
        # publishes no `EGMSimPolicy` of its own; on the smooth v1 scope its
        # unrefined carry rows determine the policy exactly (`consumption =
        # resources - savings` node by node), so derive both sides from the
        # carries and fail closed (no nested payload, grid simulation
        # unchanged) whenever the rows are not derivation-safe.
        keeper_policy = (
            keeper_result.simulation_policy
            if isinstance(keeper_result.simulation_policy, EGMSimPolicy)
            else derive_inner_sim_policy(
                carry=cast("EGMCarry", keeper_result.continuation),
                state_grid_values=self.liquid_grid_values,
                inverse_marginal=self.inverse_marginal,
                row_discrete_state_names=self.row_discrete_state_names,
                row_passive_state_names=self.row_passive_state_names,
            )
        )
        adjuster_policies = (
            bank.sim_policy
            if bank.sim_policy is not None
            else derive_inner_sim_policy(
                carry=bank.carry,
                state_grid_values=self.liquid_grid_values,
                inverse_marginal=self.inverse_marginal,
                row_discrete_state_names=self.row_discrete_state_names,
                row_passive_state_names=self.row_passive_state_names,
                extra_leading_axes=1,
            )
        )
        # Publish the nested payload only when both inner policies are
        # derivation-safe AND the branch is a deterministic hard maximum: the
        # continuous reader replays keeper vs adjuster off-grid from exactly
        # these conditional ingredients. Under a fixed-cost aggregation the
        # realized branch depends on the drawn cost, so the reader cannot
        # replay it and simulation falls back to the grid argmax — which is
        # precisely what `policy_fallback_mask` reports, so the mask is set from
        # this same condition rather than hard-coded.
        nested_published = (
            keeper_policy is not None
            and adjuster_policies is not None
            and self.branch_fixed_cost is None
        )
        diagnostics = SolverDiagnostics(
            max_outer_interpolation_error=jnp.asarray(mesh.max_validation_error),
            max_outer_bracket_width=jnp.max(collapse.value_search.bracket_width),
            outer_nodes_used=jnp.asarray(bank.n_candidates, dtype=jnp.int32),
            outer_at_lower_bound=collapse.value_search.at_lower_bound,
            outer_at_upper_bound=collapse.value_search.at_upper_bound,
            keeper_adjuster_margin=collapse.keeper_adjuster_margin,
            best_second_best_margin=collapse.best_second_best_margin,
            policy_fallback_mask=jnp.asarray(not nested_published),
            unresolved_mask=jnp.asarray(mesh.unresolved),
            adjustment_probability=collapse.adjustment_probability,
        )
        sim_policy: SimulationPolicy | None = keeper_result.simulation_policy
        if nested_published:
            sim_policy = NestedEGMSimPolicy(
                keeper=keeper_policy,
                adjuster=OuterPolicyBank(
                    outer_nodes=mesh.nodes,
                    policies=adjuster_policies,
                ),
                outer_action_name=self.outer_action,
                outer_post_decision_name=self.outer_post_decision,
                inner_action_name=self.inner_action,
                liquid_state_name=self.liquid_state_name,
                outer_no_adjustment_name=self.outer_no_adjustment_name,
                resources_target_name=self.resources_target,
                savings_lower_bound=self.savings_lower_bound,
                golden_iterations=config.golden_iterations,
            )
        return KernelResult(
            V_arr=collapse.V_arr,
            continuation=collapse.carry,
            simulation_policy=sim_policy,
            diagnostics=diagnostics,
        )

    def _solve_keeper(
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
        """Run the keeper inner solve — the state-dependent no-adjustment branch."""
        return self.keeper_kernel(
            compiled_cores=_subcores(compiled_cores=compiled_cores, role="keeper"),
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )

    def _solve_adjuster_node(
        self,
        *,
        node: FloatND,
        adjuster_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> OuterCandidateResult:
        """Run one adjuster node's exact conditional inner solve."""
        result = self.adjuster_kernel(
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
        return OuterCandidateResult(
            outer_node=node,
            V_arr=result.V_arr,
            carry=cast("EGMCarry", result.continuation),
            # An inner 1-D kernel publishes a flat policy or nothing; the
            # isinstance narrows the widened payload union for the bank.
            sim_policy=(
                result.simulation_policy
                if isinstance(result.simulation_policy, EGMSimPolicy)
                else None
            ),
        )

    def _build_candidate_bank(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> OuterCandidateBank:
        """Solve every outer node and stack the results into a candidate bank.

        Nodes are dispatched in `outer_batch_size` chunks, each chunk forced to
        device before the next is dispatched, so a chunk's independent solves
        can overlap while the number of in-flight solves stays bounded. Every
        adjuster's `sim_policy` is collected into the bank (the continuous
        simulation reader will consume them); the finite collapse does not.
        """
        adjuster_cores = _subcores(compiled_cores=compiled_cores, role="adjuster")
        nodes = list(self.outer_grid_values)
        chunk_size = self.outer_batch_size or len(nodes)
        results: list[OuterCandidateResult] = []
        for chunk_start in range(0, len(nodes), chunk_size):
            chunk_results = [
                self._solve_adjuster_node(
                    node=node,
                    adjuster_cores=adjuster_cores,
                    state_action_space=state_action_space,
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_continuation=next_regime_to_continuation,
                    flat_params=flat_params,
                    period=period,
                    ages=ages,
                )
                for node in nodes[chunk_start : chunk_start + chunk_size]
            ]
            jax.block_until_ready(
                [(result.V_arr, result.carry) for result in chunk_results]
            )
            results.extend(chunk_results)
        return build_outer_candidate_bank(
            outer_nodes=self.outer_grid_values,
            results=results,
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


def _collapse_finite_candidate_bank(
    *, keeper: KernelResult, bank: OuterCandidateBank
) -> KernelResult:
    """Collapse a finite candidate bank into the keeper — the exact grid search.

    Reproduces the pre-bank incremental sweep exactly, including tie-breaking:
    the keeper initializes the running envelope and every fold compares with a
    strict `>`, so the keeper wins exact ties and an earlier node beats a
    later one. `V = max(V_keeper, max_j W_j)` uses `fmax` — the inner NB-EGM
    NaN-dead masks cells an outer node makes infeasible, and one infeasible
    candidate must not poison a cell another candidate solves; a cell stays
    NaN only when every candidate is infeasible there. The carry rows all live
    on the shared liquid state grid, so the outer envelope is a pointwise
    maximum per row entry — value and marginal follow the winning candidate.

    The simulate phase re-optimizes the outer durable action by grid argmax
    over the next-period value array, so the keeper's published `sim_policy`
    rides through unchanged; the bank's collected adjuster policies are not
    consumed here (the continuous simulation reader will consume them).
    """
    V_arr = keeper.V_arr
    carry = cast("EGMCarry", keeper.continuation)
    for index in range(bank.n_candidates):
        V_arr = jnp.fmax(V_arr, bank.candidate_v_arr(index))
        carry = _fold_bridged_outer_carry(
            running=carry,
            candidate=bank.candidate_carry(index),
        )
    return KernelResult(
        V_arr=V_arr,
        continuation=carry,
        simulation_policy=keeper.simulation_policy,
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


def _nnbegm_inner_action(
    *, context: SolverBuildContext, outer_action: ActionName
) -> ActionName:
    """The regime's single inner continuous action (not the outer one).

    The v1 nested scope carries exactly one inner continuous action; its
    name identifies which recorded action the published inner policy
    replaces in simulation.
    """
    names = [
        name
        for name in context.state_action_space.continuous_actions
        if name != outer_action
    ]
    if len(names) != 1:
        msg = (
            "NNBEGM supports exactly one inner continuous action besides "
            f"the outer action '{outer_action}', found {sorted(names)}."
        )
        raise RegimeInitializationError(msg)
    return names[0]


def _nested_inverse_marginal(
    *,
    context: SolverBuildContext,
    rows_on_state_grid: bool,
    inner_action: ActionName,
    savings_top: float,
) -> Callable[..., FloatND] | None:
    """The regime's inverse marginal utility, if payload-derivation-safe.

    The nested simulation payload derives the inner consumption rows from the
    carry's marginal via the envelope theorem, which requires (a) the inner
    carry rows to live on the shared liquid state grid and (b) an inverse of
    `u'` free of state/param bindings (a state-dependent utility would need
    per-row bindings the kernel-level derivation does not perform). Mirrors
    the inner solve's own choice: the model's closed-form
    `inverse_marginal_utility` when its only parameter is
    `marginal_continuation`, else the iEGM numeric inversion of the utility's
    action-derivative under the same bracket convention as the solve
    (`step_core`), provided utility is a function of the inner action alone.
    Anything else returns `None`: the solve is unaffected and simulation
    keeps the grid-argmax path.
    """
    import inspect  # noqa: PLC0415

    if not rows_on_state_grid:
        return None
    closed_form = context.functions.get("inverse_marginal_utility")
    if closed_form is not None and tuple(inspect.signature(closed_form).parameters) == (
        "marginal_continuation",
    ):
        return closed_form
    utility = context.functions.get("utility")
    if utility is None or tuple(inspect.signature(utility).parameters) != (
        inner_action,
    ):
        return None
    marginal_utility = jax.grad(lambda c: utility(**{inner_action: c}))
    action_upper = jnp.asarray(savings_top * 1000.0 + 1000.0)
    action_lower = jnp.asarray(1e-8, dtype=action_upper.dtype)

    def inverse_marginal(*, marginal_continuation: FloatND) -> FloatND:
        flat = jnp.ravel(jnp.asarray(marginal_continuation))
        roots = jax.vmap(
            lambda m: numeric_inverse_marginal_utility(
                marginal_continuation=m,
                marginal_utility=marginal_utility,
                c_lower=action_lower,
                c_upper=action_upper,
            )
        )(flat)
        return roots.reshape(jnp.shape(marginal_continuation))

    return inverse_marginal


def _resolve_branch_fixed_cost(
    *,
    aggregator: OuterBranchAggregator,
    context: SolverBuildContext,
) -> tuple[UniformObservedFixedCost | None, Callable[..., FloatND] | None]:
    """Validate and resolve a fixed-cost branch aggregator at build time.

    Returns `(None, None)` for the deterministic maximum. For
    `UniformObservedFixedCost`, checks the analytic-integration contract:

    - the shock must *not* be a solve state (the closed form replaces its
      grid; a leftover state would integrate the cost twice);
    - the scale function must exist and read only `period`, `age`, and flat
      params — the collapse applies one scalar scale per period, so a state-
      dependent scale is out of the supported scope.
    """
    import inspect  # noqa: PLC0415

    if not isinstance(aggregator, UniformObservedFixedCost):
        return None, None
    if aggregator.shock_name in context.state_action_space.states:
        msg = (
            f"UniformObservedFixedCost integrates the shock "
            f"'{aggregator.shock_name}' analytically; remove its solve-state "
            f"grid from regime '{context.regime_name}' (keeping it would "
            "integrate the cost twice)."
        )
        raise RegimeInitializationError(msg)
    scale_function = context.functions.get(aggregator.scale_function)
    if scale_function is None:
        msg = (
            f"UniformObservedFixedCost.scale_function "
            f"'{aggregator.scale_function}' is not a function of regime "
            f"'{context.regime_name}'."
        )
        raise RegimeInitializationError(msg)
    unresolvable = [
        name
        for name in inspect.signature(scale_function).parameters
        if name not in ("period", "age") and name not in context.flat_param_names
    ]
    if unresolvable:
        msg = (
            f"UniformObservedFixedCost.scale_function "
            f"'{aggregator.scale_function}' reads {sorted(unresolvable)}; the "
            "per-period scalar scale may only read `period`, `age`, and flat "
            "params (a state-dependent scale is outside the supported scope)."
        )
        raise RegimeInitializationError(msg)
    return aggregator, scale_function


def _resolve_branch_scale(
    *,
    scale_function: Callable[..., FloatND] | None,
    regime_params: Mapping[str, object],
    period: int,
    ages: AgeGrid,
) -> FloatND:
    """Evaluate the fixed cost's per-period scalar scale at kernel-call time."""
    import inspect  # noqa: PLC0415

    if scale_function is None:  # pragma: no cover - guarded at build time
        msg = "branch_fixed_cost set without a resolved scale function"
        raise RegimeInitializationError(msg)
    kwargs: dict[str, object] = {}
    for name in inspect.signature(scale_function).parameters:
        if name == "period":
            kwargs[name] = jnp.asarray(period)
        elif name == "age":
            kwargs[name] = jnp.asarray(ages.values[period])
        else:
            kwargs[name] = regime_params[name]
    return jnp.asarray(scale_function(**kwargs))


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
