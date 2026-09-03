"""The DC-EGM solver (discrete-continuous endogenous grid method).

`DCEGM` configures one regime's Euler-inversion solve. Its
`build_period_kernels` returns one `PeriodKernel` per period — a non-jitted
kernel declaring the shared jitted EGM step (deduped by function identity, so
periods sharing a core reuse one compiled program) as its one native dense
program; it builds the step's arguments through the declared builder and
publishes a `KernelOutput` (value array, continuation carry, published
simulation policy) outside JIT.

The kernel-building imports (`jax`, `build_egm_step_functions`) are
function-local so the public `lcm.solvers` façade stays a thin re-export that
pulls in no numerical engine modules.
"""

import functools
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.routes import (
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import EGMCarry, egm_carry_role_tree
from _lcm.egm.published_policy import egm_sim_policy_role_tree
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.grids import ContinuousGrid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.continuation_target import _union_fixed_params, _union_free_params
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    OneMarginSolver,
    PeriodKernel,
    SolutionKernels,
    SolverBuildContext,
    SolverModelContext,
    _BoundLiquidMargin,
    bind_roles,
    simulation_route,
)
from _lcm.typing import (
    EGMStepFunction,
    FlatParams,
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.exceptions import (
    ExactAffineKernelUnavailableError,
    RegimeInitializationError,
)
from lcm.solver_api import EGM_CONTINUATION, SIMULATION_POLICY, KernelOutput
from lcm.typing import (
    ActionName,
    FloatND,
    FunctionName,
    StateName,
)


def _fail_if_exact_affine_kernel_unavailable(
    *, regime_name: RegimeName, selection: str
) -> None:
    """Refuse a certified envelope selection without its native payload."""
    from _lcm.egm.upper_envelope._exact_affine import ffi  # noqa: PLC0415

    if ffi.kernel_available_for_current_backend():
        return
    msg = (
        f"Regime {regime_name!r} selects {selection}, but the native "
        "exact-affine library is unavailable or unloadable. Install a pylcm "
        "build carrying the library or reinstall it with `pixi reinstall pylcm`; "
        "select ordinary or approximate envelope arithmetic only when its "
        "documented contract is acceptable."
    )
    raise ExactAffineKernelUnavailableError(msg)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ExactEnvelope:
    """Certified segment-envelope backend configuration.

    The native exact-affine library must be available when a regime selects this
    backend. The backend is exact for the finite candidate set supplied to it;
    discretization of a continuous constrained branch remains controlled by
    `DCEGM.n_constrained_points`.
    """

    max_runs: int = 24
    """Maximum resource-increasing runs folded into one node cell."""

    cell_batch_size: int | None = None
    """Number of independent node cells resolved in parallel; `None` is serial."""

    def __post_init__(self) -> None:
        _fail_if_envelope_max_runs_too_few(self.max_runs)
        _fail_if_envelope_cell_batch_size_non_positive(self.cell_batch_size)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class FUESEnvelope:
    """Fast Upper-Envelope Scan configuration."""

    jump_thresh: float = 2.0
    """Segment-switch threshold on `|ΔA / ΔR|`."""

    n_points_to_scan: int | None = None
    """Forward-scan width; `None` performs the exhaustive scan."""

    scan_unroll: int = 1
    """Loop-unroll factor for the sequential candidate scan."""

    def __post_init__(self) -> None:
        _fail_if_fues_jump_thresh_non_positive(self.jump_thresh)
        _fail_if_fues_n_points_to_scan_too_few(self.n_points_to_scan)
        _fail_if_fues_scan_unroll_too_few(self.scan_unroll)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class RFCEnvelope:
    """Rooftop-Cut upper-envelope configuration."""

    jump_thresh: float = 2.0
    """Segment-switch threshold on `|Δc / ΔR|`."""

    search_radius: int = 10
    """Neighbors inspected on each side of a candidate."""

    def __post_init__(self) -> None:
        _fail_if_rfc_jump_thresh_non_positive(self.jump_thresh)
        _fail_if_rfc_search_radius_too_few(self.search_radius)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class LTMEnvelope:
    """Local-upper-bound brute-force envelope configuration."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class MSSEnvelope:
    """HARK-style left-to-right segment envelope configuration."""


type EnvelopeConfig = (
    ExactEnvelope | FUESEnvelope | RFCEnvelope | LTMEnvelope | MSSEnvelope
)


# A non-concave candidate chain folds into at least two resource-increasing runs,
# so a smaller fold capacity could never publish one.
_MIN_ENVELOPE_MAX_RUNS: int = 2


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class DCEGM(OneMarginSolver):
    """Configuration of the DC-EGM solver for one regime.

    DC-EGM inverts the Euler equation on an exogenous end-of-period
    (post-decision) grid instead of searching a dense grid for the continuous
    action. It requires a specific model structure — exactly one continuous
    (*Euler*) state and one continuous action, a declared resources function
    `R` with consumption recovery `c = R - A`, and a post-decision function `A`.
    An analytical `inverse_marginal_utility` function is optional: when it is
    absent, pylcm differentiates `utility` and numerically inverts the marginal
    utility with an expanding bracket. Structural and per-period applicability
    checks run during `Model(...)` through the solver's staged validation hooks.

    The period kernel publishes the off-grid EGM policy on the replay channel
    of every active period. A solve retains it in its labelled replay artifacts
    only when the model declares a policy-read consumer; without that route the
    loop drops it at the period boundary and the result records the omission as
    not applicable.
    No shipped envelope currently satisfies the conservative off-grid read gate,
    so ordinary simulation recomputes the action on its grid.

    Otherwise, `simulate` recomputes the argmax over the regime's
    gridded continuous action against the stored value function. Simulated
    continuous actions then live on the action grid, and with taste shocks the
    simulated choice frequencies follow the grid-restricted choice-specific
    values rather than exactly the solve's choice probabilities. On both paths
    the budget constraint the solve enforces intrinsically
    (`continuous_action <= resources - savings_grid lower bound`) is applied
    as a feasibility mask during simulation.

    """

    @property
    def publishes_simulation_policy(self) -> bool:
        """The kernel publishes an off-grid EGM policy on every active period.

        Whether a solve retains it follows the declared policy-read route; this
        declaration lets the result ledger record a dropped policy as not
        applicable rather than treating the cell as one that never publishes.
        """
        return True

    savings_grid: ContinuousGrid
    """Exogenous end-of-period grid; its lower bound is the borrowing limit.

    The endogenous grid inherits this grid's spacing, so it controls where the
    solution is accurate. Value reads use a slope-limited cubic Hermite
    interpolant when marginal-utility slopes are available; policy reads remain
    piecewise linear. Reads extrapolate along the nearest segment below support
    and use the endpoint above support. With sharply curved utility (e.g. CRRA),
    cluster nodes toward the borrowing limit: interpolation error there compounds
    across periods.
    """

    envelope: EnvelopeConfig = field(default_factory=ExactEnvelope)
    """Typed upper-envelope backend configuration.

    Backend-specific controls live on the selected frozen configuration object,
    so controls for an inactive algorithm cannot be supplied accidentally:

    - `ExactEnvelope`: certified ownership of the represented candidates. It
      requires pylcm's native exact-affine library and fails during `Model(...)`
      when that library is unavailable.
    - `FUESEnvelope`: Fast Upper-Envelope Scan.
    - `RFCEnvelope`: Rooftop-Cut algorithm.
    - `LTMEnvelope`: quadratic local-upper-bound baseline.
    - `MSSEnvelope`: HARK-style left-to-right segment sweep.

    `ExactEnvelope` certifies the envelope of the candidates actually supplied;
    it does not turn the sampled credit-constrained branch into a continuous
    exact representation.
    """

    refined_grid_factor: float = 2.0
    """Headroom factor sizing the refined (NaN-padded) envelope arrays.

    The refined row holds one slot per envelope point, and every kink costs two
    (the outgoing owner's reading, then the incoming one). How often a row kinks
    is a property of the candidate cloud rather than of the grid, so this is
    headroom, not a bound: a row needing more slots than it has reports overflow
    and is NaN-poisoned rather than silently truncated. The default is sized for
    the `"exact"` upper envelope, which emits every ownership change — including
    the interior-only and node-aligned ones the fast scans miss — and therefore
    kinks more often than they do.
    """

    n_constrained_points: int = 20
    """Resolution of the sampled credit-constrained branch.

    The current period evaluates the analytical constrained value as a floor,
    but the continuation carry handed to a parent contains only these sampled
    constrained candidates. Increasing the count can therefore improve values
    and policies in earlier periods at the cost of larger envelope rows, longer
    compilation, and more execution work. `ExactEnvelope` is exact only for this
    supplied finite candidate set, not for the unsampled continuous branch.
    """

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
        _fail_if_n_constrained_points_too_few(self.n_constrained_points)
        _fail_if_stochastic_node_batch_size_negative(self.stochastic_node_batch_size)

    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> _BoundDCEGM:
        """Bind regime-owned DAG names without exposing them on public `DCEGM`."""
        return cast(
            "_BoundDCEGM",
            bind_roles(
                solver=self,
                role_type=_BoundDCEGM,
                continuous_state=margin.state,
                continuous_action=margin.action,
                resources=margin.resources,
                post_decision_function=margin.post_decision_state,
            ),
        )

    @property
    def requires_continuation(self) -> bool:
        """DC-EGM inverts the Euler equation against its targets' marginals."""
        return True

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Validate the user-level DC-EGM contract for this regime."""
        from _lcm.egm.validation import validate_dcegm_regime  # noqa: PLC0415

        validate_dcegm_regime(
            regime_name=context.regime_name,
            user_regime=context.user_regimes[context.regime_name],
            user_regimes=context.user_regimes,
            solution_reachability=context.solution_reachability,
        )

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Validate capabilities required by the selected envelope backend.

        Model semantics are checked earlier by `validate_model`, without
        consulting machine-local capabilities. Reaching this build-stage hook
        therefore means the regime satisfies the DC-EGM contract; only then is
        it meaningful to ask whether this installation can execute its selected
        backend.
        """
        if isinstance(self.envelope, ExactEnvelope):
            _fail_if_exact_affine_kernel_unavailable(
                regime_name=context.regime_name,
                selection="ExactEnvelope",
            )

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the one route DC-EGM walks in each phase.

        The solve pipeline builds its feasibility predicate once per discrete
        combination, before the continuous inner stage runs. What is bound
        there is every discrete state, every passive continuous state, every
        discrete action, and the regime's params — but not the Euler state,
        whose nodes the inversion produces, and not the continuous action,
        which the inversion solves for. A constraint reading either of those is
        refused rather than evaluated against whichever value happens to be in
        scope.

        One site, not two. The kernel evaluates constraints per discrete
        combination and nowhere else, so declaring a savings-stage site as well
        would promise a place of evaluation that does not exist; what the
        savings stage offers is a proof, and a proof belongs to a site rather
        than needing one of its own.

        One route, not one per period: DC-EGM reuses a single core across every
        period whose pool resolves alike, and does not rebuild per age group.
        """
        bound = cast("_BoundDCEGM", self)
        proofs = (
            proves_the_savings_grids_lower_bound(
                post_decision=bound.post_decision_function
            ),
        )
        if context.phase == "simulate":
            return (simulation_route(context=context, solver_path=("dcegm",)),)
        return (
            ConstraintRoute(
                key=ConstraintRouteKey(
                    phase="solve", period_group=None, solver_path=("dcegm",)
                ),
                sites=(
                    ConstraintSite(
                        stage="discrete_combo",
                        function_pool=context.functions,
                        available_names=_combination_inputs(
                            context=context, euler_state=bound.continuous_state
                        ),
                        structural_proofs=proofs,
                    ),
                ),
            ),
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one DC-EGM period adapter per period and the carry template.

        The solver's model-stage and build-stage validation hooks guarantee the
        regime is non-terminal, so the regime-transition probability function
        exists. Periods sharing one EGM-step core reuse a single jitted core, and
        therefore a single compiled program. Numerical-builder imports are
        function-local so the public `lcm.solvers` façade stays a thin re-export
        that pulls in no engine modules.
        """

        from _lcm.egm.step import build_egm_step_functions  # noqa: PLC0415

        assert context.compute_regime_transition_probs is not None  # noqa: S101
        assert context.koopmans_aggregator is not None  # noqa: S101
        build = build_egm_step_functions(
            solver=cast("_BoundDCEGM", self),
            regime_name=context.regime_name,
            user_regimes=context.user_regimes,
            functions=context.functions,
            koopmans_aggregator=context.koopmans_aggregator,
            constraints=context.constraints,
            processed_constraints=context.processed_constraints,
            transitions=context.transitions,
            transition_plans=context.transition_plans,
            compute_regime_transition_probs=context.compute_regime_transition_probs,
            regime_to_v_interpolation_info=context.regime_to_v_interpolation_info,
            period_to_regime_v_interp=context.period_to_regime_v_interp,
            period_to_regime_grid_signature=context.period_to_regime_grid_signature,
            solution_reachability=context.solution_reachability,
            flat_param_names=context.flat_param_names,
            regime_to_flat_param_names=context.regime_to_flat_param_names,
            state_action_space=context.state_action_space,
            has_taste_shocks=context.has_taste_shocks,
        )
        steps: Mapping[int, Callable] = build.steps
        if context.enable_jit:
            jitted_by_id: dict[int, Callable] = {}
            for func in steps.values():
                if id(func) not in jitted_by_id:
                    jitted_by_id[id(func)] = jax.jit(func)
            steps = MappingProxyType(
                {period: jitted_by_id[id(func)] for period, func in steps.items()}
            )
        argument_builder = _DCEGMArgumentBuilder(
            regime_name=context.regime_name,
            stateful_targets=build.stateful_targets,
            transition_target_names=tuple(context.transitions),
        )
        output_roles = _dcegm_output_roles(build=build)
        # Periods sharing one core share one program, and so one compiled
        # executable.
        programs_by_core: dict[int, MappingProxyType[str, CoreProgram]] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        for period, core in steps.items():
            if id(core) not in programs_by_core:
                programs_by_core[id(core)] = MappingProxyType(
                    {
                        "main": CoreProgram(
                            name="main",
                            function=core,
                            argument_builder=argument_builder,
                            requirements=CoreExecutionRequirements(),
                            output_roles=output_roles,
                            disposition=CoreExecutionDisposition.DENSE,
                            disposition_reason=_DCEGM_DENSE_REASON,
                            donation_candidates=(),
                        )
                    }
                )
            period_kernels[period] = _DCEGMPeriodKernel(
                _core_programs=programs_by_core[id(core)],
                regime_name=context.regime_name,
                stateful_targets=build.stateful_targets,
                transition_target_names=tuple(context.transitions),
            )
        return SolutionKernels(
            period_kernels=MappingProxyType(period_kernels),
            continuation_spec=EGMContinuationSpec(
                template=build.carry_template,
                layout=self.egm_continuation_layout,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _BoundDCEGM(DCEGM):
    """Internal DC-EGM configuration with regime-resolved DAG role names."""

    continuous_state: StateName
    """Name of the liquid state DC-EGM inverts the Euler equation over."""

    continuous_action: ActionName
    """Name of the liquid action the endogenous grid is expressed in."""

    resources: FunctionName
    """Name of the function giving resources available for that action."""

    post_decision_function: FunctionName
    """Name of the function giving the savings the exogenous grid spans."""


# Why the DC-EGM program executes dense: the kernel owns its stochastic-node
# and refined-grid batching, and no product axis of it is planner-streamable.
_DCEGM_DENSE_REASON = "deliberately_dense:dcegm_solver_owned_node_and_grid_batching"


@dataclass(frozen=True, kw_only=True)
class EGMStepBuild:
    """What one regime's DC-EGM step build hands its period kernels."""

    steps: MappingProxyType[int, EGMStepFunction]
    """Per-period step function; periods sharing a configuration share one."""

    carry_template: EGMCarry
    """The regime's all-finite carry template (discrete states, then passive
    states, then discrete actions leading; the refined row axis last)."""

    stateful_targets: frozenset[RegimeName]
    """The carry keys the steps read; the rolling carry is filtered to these."""

    row_discrete_state_names: tuple[StateName, ...]
    """Discrete states leading every carry and policy row, in row order."""

    row_passive_state_names: tuple[StateName, ...]
    """Passive continuous states following the discrete states on every row."""

    row_discrete_action_names: tuple[ActionName, ...]
    """Discrete actions following the passive states on every row."""


def _dcegm_output_roles(*, build: EGMStepBuild) -> tuple[object, ...]:
    """Describe the step's outputs by the state axes that lead them.

    - The value array is the regime's value on the productmap state order.
    - Every carry row and every policy row leads with the discrete and passive
      state axes; the discrete-action axes and the refined row axis follow and
      are replicated. The taste-shock scale is a replicated scalar; the carry
      publishes neither breakpoints nor an exact policy row.
    """
    row = StateAxesLeading(
        state_names=build.row_discrete_state_names + build.row_passive_state_names
    )
    return (
        VALUE,
        egm_carry_role_tree(
            row=row,
            scalar=StateAxesLeading(state_names=(), shape=()),
            breakpoints=None,
            policy=None,
        ),
        egm_sim_policy_role_tree(
            row=row,
            row_discrete_state_names=build.row_discrete_state_names,
            row_passive_state_names=build.row_passive_state_names,
            row_discrete_action_names=build.row_discrete_action_names,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class _DCEGMArgumentBuilder:
    """Build the DC-EGM program's arguments for lowering and execution.

    The program takes the state grids, the carries of the regime's carry
    targets, the free params of the regime and of its transition targets, and
    the period and age. No target value array is read: the continuation is
    interpolated from the carries alone.

    A DC-EGM source carrying into a *different* target regime evaluates that
    target's resources and transition functions in its per-asset-node solve,
    reading the target's params (a pension factor the source never reads, for
    example). These are model-level shared values, so the target's flat params
    carry the right value; the union is passed and the captured functions read
    only the keys they need.
    """

    regime_name: RegimeName
    """Name of the regime whose flat params the program projects."""

    stateful_targets: frozenset[RegimeName]
    """The carry keys the step reads; the rolling carry is filtered to these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the exact kwargs shared by lowering and the runtime call."""
        state_action_space = cast("StateActionSpace", context.state_action_space)
        flat_params = cast("FlatParams", context.flat_params)
        ages = cast("AgeGrid", context.ages)
        return MappingProxyType(
            {
                **dict(state_action_space.states),
                "next_regime_to_continuation": _carry_subset(
                    next_regime_to_continuation=cast(
                        "Mapping[RegimeName, ContinuationPayload]",
                        context.next_regime_to_continuation,
                    ),
                    stateful_targets=self.stateful_targets,
                ),
                **_union_free_params(
                    flat_params=flat_params,
                    regime_name=self.regime_name,
                    transition_target_names=self.transition_target_names,
                ),
                "period": jnp.int32(context.period),
                "age": ages.values[context.period],
            }
        )


@dataclass(frozen=True, kw_only=True)
class _DCEGMPeriodKernel:
    """The DC-EGM period kernel: one native dense program around the step core.

    `main` inverts the Euler equation on the savings grid and publishes the
    value function, the continuation a parent interpolates, and the off-grid
    simulation policy. The kernel owns its stochastic-node and refined-grid
    batching, so the program is deliberately dense; it reads its continuation
    from the carries alone and declares no target value access. Calling the
    kernel builds the program's arguments through its declared builder and
    returns a public `KernelOutput` with the carry on the continuation channel
    and the policy on the replay channel.
    """

    _core_programs: Mapping[str, CoreProgram]
    """The immutable one-node program graph, `main`."""

    regime_name: RegimeName
    """Name of the regime whose flat params this kernel projects."""

    stateful_targets: frozenset[RegimeName]
    """The carry keys the step reads; the rolling carry is filtered to these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    def __post_init__(self) -> None:
        """Snapshot the graph and require exactly the `main` program."""
        programs = MappingProxyType(dict(self._core_programs))
        if tuple(programs) != ("main",):
            msg = (
                "A DC-EGM kernel publishes exactly the program 'main'; got "
                f"{tuple(programs)}."
            )
            raise ValueError(msg)
        object.__setattr__(self, "_core_programs", programs)

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the native graph used by eager, JIT, AOT, and replay paths."""
        return self._core_programs

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _DCEGMPeriodKernel:
        """Bind the regime's and its carry targets' fixed params into the program.

        The step threads its `**kwargs` straight into the per-combo pool its
        captured functions read, so binding the union of the regime's and its
        carry targets' fixed params restores the values removed from the live
        `flat_params` for all of them at once.
        """
        egm_fixed = _union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not egm_fixed:
            return self
        program = self._core_programs["main"]
        return replace(
            self,
            _core_programs=MappingProxyType(
                {
                    "main": replace(
                        program,
                        function=functools.partial(program.function, **egm_fixed),
                    )
                }
            ),
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
        logger: logging.Logger,  # noqa: ARG002
    ) -> KernelOutput:
        """Run the compiled `main` program and publish its typed artifacts."""
        program = self._core_programs["main"]
        arguments = program.argument_builder(
            CoreBuildContext(
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        )
        V_arr, egm_carry, sim_policy = compiled_cores["main"](**arguments)
        return KernelOutput(
            value=V_arr,
            continuations={EGM_CONTINUATION: egm_carry},
            replay={SIMULATION_POLICY: sim_policy},
        )


def _carry_subset(
    *,
    next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
    stateful_targets: frozenset[RegimeName],
) -> MappingProxyType[RegimeName, EGMCarry]:
    """Return the carries a regime's EGM core actually reads.

    Each core only ever indexes `next_regime_to_continuation[target]` for its
    carry targets, so the full all-regimes mapping is needlessly large.
    Filtering to that subset keeps the core's carry pytree input
    minimal — only this subset is passed per call rather than every regime's
    carry at once.

    Iterates the source mapping's key order (stable across rolls) so the
    filtered pytree structure matches between lowering and call. Membership is
    tested defensively because a target need not publish a carry in every model.
    """
    return MappingProxyType(
        {
            name: next_regime_to_continuation[name]
            for name in next_regime_to_continuation
            if name in stateful_targets
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
            f"FUESEnvelope.jump_thresh must be a finite positive value, got "
            f"{fues_jump_thresh}. It is the segment-switch threshold on "
            "`|ΔA / ΔR|` in the FUES scan."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_jump_thresh_non_positive(rfc_jump_thresh: float) -> None:
    # `not (x > 0.0)` rejects NaN too: `nan <= 0.0` is False, so the segment-
    # switch comparison would silently misbehave on a non-finite threshold.
    if not (math.isfinite(rfc_jump_thresh) and rfc_jump_thresh > 0.0):
        msg = (
            f"RFCEnvelope.jump_thresh must be a finite positive value, got "
            f"{rfc_jump_thresh}. It is the segment-switch threshold on "
            "`|Δc / ΔR|` in the rooftop cut."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_search_radius_too_few(rfc_search_radius: int) -> None:
    if rfc_search_radius < 1:
        msg = (
            f"RFCEnvelope.search_radius must be at least 1, got "
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


def _fail_if_fues_n_points_to_scan_too_few(fues_n_points_to_scan: int | None) -> None:
    # `None` requests the exhaustive scan; only an explicit finite width is bounded.
    if fues_n_points_to_scan is not None and fues_n_points_to_scan < 1:
        msg = (
            f"FUESEnvelope.n_points_to_scan must be at least 1, got "
            f"{fues_n_points_to_scan}. The FUES forward scan must inspect at "
            "least one point after each candidate."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_scan_unroll_too_few(fues_scan_unroll: int) -> None:
    if fues_scan_unroll < 1:
        msg = (
            f"FUESEnvelope.scan_unroll must be at least 1, got "
            f"{fues_scan_unroll}. It is the `lax.scan` unroll factor for the "
            "FUES candidate scan; 1 means no unrolling."
        )
        raise RegimeInitializationError(msg)


def _fail_if_envelope_max_runs_too_few(envelope_max_runs: int) -> None:
    if envelope_max_runs < _MIN_ENVELOPE_MAX_RUNS:
        msg = (
            f"ExactEnvelope.max_runs must be at least {_MIN_ENVELOPE_MAX_RUNS}, "
            f"got {envelope_max_runs}. It is the fold capacity of the exact "
            "upper envelope; a non-concave candidate chain folds into at "
            "least two resource-increasing runs."
        )
        raise RegimeInitializationError(msg)


def _fail_if_envelope_cell_batch_size_non_positive(
    envelope_cell_batch_size: int | None,
) -> None:
    if envelope_cell_batch_size is not None and envelope_cell_batch_size < 1:
        msg = (
            f"ExactEnvelope.cell_batch_size must be at least 1, got "
            f"{envelope_cell_batch_size}. It is how many node cells the exact "
            "upper envelope resolves in parallel; use None to resolve them one "
            "at a time."
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


def _combination_inputs(
    *, context: ConstraintRouteContext, euler_state: StateName
) -> frozenset[str]:
    """Names bound per discrete combination, which a constraint may read there.

    Every discrete state, every continuous state other than the Euler state
    (those are passive, so one node each per combination), every discrete
    action, the regime's params, and the lifecycle clock. The Euler state and
    the continuous action are absent because the inversion produces them.
    """
    return (
        frozenset(context.variables.discrete_state_names)
        | (frozenset(context.variables.continuous_state_names) - {euler_state})
        | frozenset(context.variables.discrete_action_names)
        | context.flat_param_names
        | {"age", "period"}
    )
