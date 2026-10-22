"""The DC-EGM solver (discrete-continuous endogenous grid method).

`DCEGM` configures one regime's Euler-inversion solve. Its
`build_period_kernels` returns one `PeriodKernel` per period — a non-jitted
kernel declaring output-specialized values and replay variants of the shared
jitted EGM step (deduped by function identity). It builds the step's arguments
through the declared builder and publishes a `KernelOutput` outside JIT. The
values variant returns only value and continuation carry; the replay variant
also returns the simulation policy.

The kernel-building imports (`jax`, `build_egm_step_functions`) are
function-local so the public `lcm.solvers` façade stays a thin re-export that
pulls in no numerical engine modules.
"""

import functools
import logging
import math
from collections.abc import Callable, Hashable, Mapping
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
from _lcm.egm.carry import EGMCarry, egm_carry_role_tree, egm_query_role_tree
from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.published_policy import EGMSimPolicy, egm_sim_policy_role_tree
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ProgramScope,
    ReducedAxis,
    TiledOutputAxis,
    _CheckedCoreVariant,
    bind_core_program_fixed_params,
)
from _lcm.execution.output_layout import VALUE, OutputRoleTree, StateAxesLeading
from _lcm.execution.reductions import WEIGHTED_EXPECTATION_REDUCTION
from _lcm.grids import ContinuousGrid
from _lcm.post_decision_bound import PostDecisionBoundSpec, grid_origin_bound_spec
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.continuation_reads import (
    continuation_leaf_reads,
    published_continuation_templates,
    rekeyed_value_reads,
    with_continuation_leaf_reads,
)
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
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
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    ContinuationCountedKernelOutput,
    FeasibilityCounts,
    FeasibilityKernelOutput,
    QueryInputStatus,
    _unflatten_continuation_read_counts,
)
from _lcm.time import TimeAxis
from _lcm.transition_plans import SupportOrigin
from _lcm.typing import (
    EGMStepFunction,
    FlatParams,
    NextValue,
    PytreeValue,
    RegimeName,
    ShapeDtypePytree,
)
from lcm._solver_api.capabilities import SolverExecutionCapabilities
from lcm.exceptions import (
    ExactAffineKernelUnavailableError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.solver_api import (
    EGM_CONTINUATION,
    EGM_ENDOGENOUS_COORDINATE,
    SIMULATION_POLICY,
    ArtifactKey,
    ContinuationCapabilities,
    KernelOutput,
)
from lcm.typing import (
    ActionName,
    BoolND,
    FloatND,
    FunctionName,
    ReferenceName,
    StateName,
)

type _DCEGMStepOutput = (
    tuple[FloatND, EGMCarry, EGMSimPolicy]
    | BoundedResult[tuple[FloatND, EGMCarry, EGMSimPolicy, FeasibilityCounts | None]]
    | BoundedResult[_NestedDCEGMStepChannels]
)
type _NestedDCEGMStepChannels = tuple[
    FloatND, EGMCarry, EGMSimPolicy, FeasibilityCounts | None, BoolND
]
type _DCEGMQuerySlots = tuple[tuple[RegimeName, int, PostDecisionBoundSpec], ...]
_DCEGM_VALUES_FIELD_COUNT = 3
_DCEGM_REPLAY_FIELD_COUNT = 4
_NESTED_DCEGM_STEP_FIELD_COUNT = 5


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

    def __post_init__(self) -> None:
        _fail_if_envelope_max_runs_too_few(self.max_runs)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class FUESEnvelope:
    """Fast Upper-Envelope Scan configuration."""

    jump_thresh: float = 2.0
    """Segment-switch threshold on `|ΔA / ΔR|`."""

    n_points_to_scan: int | None = None
    """Forward-scan width; `None` performs the exhaustive scan."""

    def __post_init__(self) -> None:
        _fail_if_fues_jump_thresh_non_positive(self.jump_thresh)
        _fail_if_fues_n_points_to_scan_too_few(self.n_points_to_scan)


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

    arithmetic: ComparisonArithmetic = "certified"
    """Which arithmetic settles a comparison between two candidate chords.

    The geometry is the same either way: which stored piece covers an interval,
    which node owns a query, and where two branches hand over. Only the
    comparison changes.

    - `"certified"` decides on the stored operands, so an ordering the working
      format cannot separate is still settled, and a comparison the arithmetic
      cannot decide publishes `NaN` rather than a guess. Requires the installed
      exact-affine payload for the active backend.
    - `"ordinary"` compares two readings formed in the working format. A
      reading is a slope and then an affine step, so it carries no bound in
      representable steps: cancellation or a large common level can reverse an
      ordering the correctly rounded values would separate, and candidates
      whose readings coincide are separated by the declared tie order. It
      reaches no native kernel, so it is the route available when that payload
      is absent; its values, owners and crossings are the caller's to validate
      for the intended model at every precision, backend and transformation in
      use.
    """


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

    The period kernel can publish the off-grid EGM policy on the replay channel
    of every active period. A solve selects that output-specialized program only
    when an exact artifact retention requires the policy; without such a route
    it runs the values program and records the omission as not applicable.
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
    def capabilities(self) -> SolverExecutionCapabilities:
        """Describe this configured solver without building numerical kernels."""
        return SolverExecutionCapabilities(
            required_declaration="ConsumptionSavingsRegime with one LiquidMargin",
            problem_shape="One liquid Euler margin and optional discrete choice",
            prerequisites=(
                "Valid resources and post-decision roles; lower bound; supported "
                "passive states and continuation layout; EV1 taste shocks"
            ),
            main_tradeoff=(
                "Constrained candidates and upper envelope; simulation may "
                "re-optimize on the action grid"
            ),
            reduced_axes=("stochastic_node",),
            tiled_axes=("cell", "savings_point", "euler_point")
            + (("envelope_cell",) if isinstance(self.envelope, ExactEnvelope) else ()),
            host_axes=(),
            host_driven_programs=(),
            supports_ev1_taste_shocks=True,
            supports_nonlinear_certainty_equivalent=False,
        )

    @property
    def transition_local_lottery_origins(self) -> frozenset[SupportOrigin]:
        """The child read enumerates an edge's local draws as node axes."""
        return frozenset({SupportOrigin.SOURCE_PROCESS})

    @property
    def publishes_simulation_policy(self) -> bool:
        """The kernel can publish an off-grid EGM policy on every active period.

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

    def __post_init__(self) -> None:
        _fail_if_savings_grid_is_stochastic(self.savings_grid)
        _fail_if_refined_grid_factor_too_small(self.refined_grid_factor)
        _fail_if_n_constrained_points_too_few(self.n_constrained_points)

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
                post_decision_bound=grid_origin_bound_spec(
                    post_decision=margin.post_decision_state,
                    liquid_state=margin.state,
                    action=margin.action,
                    resources_target=margin.resources,
                    grid=self.savings_grid,
                ),
            ),
        )

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """DC-EGM inverts the Euler equation against its targets' marginals."""
        return frozenset({EGM_CONTINUATION})

    @property
    def required_continuation_capabilities(self) -> ContinuationCapabilities:
        """The EGM step reads the target's value and its marginal in resources."""
        return ContinuationCapabilities(
            value=True, marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE})
        )

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Validate the user-level DC-EGM contract for this regime."""
        from _lcm.egm.validation import validate_dcegm_regime  # noqa: PLC0415

        validate_dcegm_regime(
            regime_name=context.regime_name,
            user_regime=context.user_regimes[context.regime_name],
            user_regimes=context.user_regimes,
            laws=context.laws,
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
        if context.sharded_state_names:
            raise ModelInitializationError(
                f"DCEGM regime {context.regime_name!r} cannot shard discrete "
                f"state axes {sorted(context.sharded_state_names)!r}: its child "
                "carry indexing requires whole discrete axes. Remove these "
                "states from ExecutionConfig.sharded_states."
            )
        if isinstance(self.envelope, ExactEnvelope):
            _fail_if_exact_affine_kernel_unavailable(
                regime_name=context.regime_name,
                selection="ExactEnvelope",
            )
        if (
            isinstance(self.envelope, MSSEnvelope)
            and self.envelope.arithmetic == "certified"
        ):
            _fail_if_exact_affine_kernel_unavailable(
                regime_name=context.regime_name,
                selection="the certified MSSEnvelope arithmetic",
            )

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the actual current-feasibility site of the selected envelope.

        Owned conditional rows keep their continuous subproblem unmasked and
        evaluate predicates at published current-state nodes. The Euler state
        is bound there, alongside discrete choices, passive states and time.
        Continuous actions remain unavailable to predicates. Other envelopes
        evaluate feasibility per combination.
        """
        bound = cast("_BoundDCEGM", self)
        proofs = (
            proves_the_savings_grids_lower_bound(
                post_decision=bound.post_decision_function,
                bound_spec=bound.post_decision_bound,
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
                        stage=(
                            "state_node"
                            if isinstance(self.envelope, ExactEnvelope)
                            else "discrete_combo"
                        ),
                        function_pool=context.functions,
                        available_names=(
                            _combination_inputs(
                                context=context, euler_state=bound.continuous_state
                            )
                            | (
                                {bound.continuous_state}
                                if isinstance(self.envelope, ExactEnvelope)
                                else set()
                            )
                        ),
                        structural_proofs=proofs,
                    ),
                ),
            ),
        )

    def declare_continuation_reads(
        self, *, kernels: SolutionKernels, context: SolverBuildContext
    ) -> SolutionKernels:
        """Declare every published row of every carry target each period reads."""
        return declare_dcegm_carry_reads(kernels=kernels, context=context)

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one DC-EGM period adapter per period and the carry template.

        The solver's model-stage and build-stage validation hooks guarantee the
        regime is non-terminal, so the regime-transition probability function
        exists. Periods sharing one EGM-step core reuse a single jitted core, and
        therefore a single compiled program. Numerical-builder imports are
        function-local so the public `lcm.solvers` façade stays a thin re-export
        that pulls in no engine modules.

        An enclosing outer solver may request per-action current eligibility
        when the actual continuation plan installs child query readers. This
        transport request grants no own floor or query authority.
        """

        from _lcm.egm.step import _EGMStep, build_egm_step_functions  # noqa: PLC0415

        assert context.compute_regime_transition_probs is not None  # noqa: S101
        assert context.koopmans_aggregator is not None  # noqa: S101
        query_model = context.dcegm_source_query_model or (
            context.dcegm_query_models.get(context.regime_name)
            if context.dcegm_query_models is not None
            else None
        )
        build = build_egm_step_functions(
            solver=cast("_BoundDCEGM", self),
            regime_name=context.regime_name,
            user_regimes=context.user_regimes,
            laws=context.laws,
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
            period_to_user_regimes=context.period_to_user_regimes,
            solution_reachability=context.solution_reachability,
            flat_param_names=context.flat_param_names,
            regime_to_flat_param_names=context.regime_to_flat_param_names,
            state_action_space=context.state_action_space,
            has_taste_shocks=context.has_taste_shocks,
            age_values=context.ages.values,
            query_model=query_model,
            dcegm_query_models=context.dcegm_query_models,
            nbegm_query_models=context.nbegm_query_models,
        )
        steps: Mapping[int, EGMStepFunction[_DCEGMStepOutput]] = build.steps
        argument_builder = _DCEGMArgumentBuilder(
            regime_name=context.regime_name,
            stateful_targets=build.stateful_targets,
            transition_target_names=tuple(context.transitions),
        )
        # Periods sharing one numerical core share both output-specialized
        # programs. The values variant drops the optional policy inside the
        # traced body so XLA can eliminate policy assembly when no selected
        # retention authority needs it.
        programs_by_core: dict[int, MappingProxyType[str, CoreProgram]] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        period_group_keys: dict[int, Hashable] = {}
        own_bound_specs = () if query_model is None else (query_model.authority.bound,)
        query_slots_by_period: dict[int, _DCEGMQuerySlots] = {}
        reader_capabilities: dict[
            int, Mapping[RegimeName, ContinuationCapabilities]
        ] = {}
        for period, core in steps.items():
            plan = cast("_EGMStep", core).pieces.continuation_plan
            query_slots, capabilities = _dcegm_query_slots_and_capabilities(
                core=core,
                regime_name=context.regime_name,
                period=period,
                own_bound_specs=own_bound_specs,
            )
            if capabilities is not None:
                reader_capabilities[period] = capabilities
            if query_slots is not None:
                query_slots_by_period[period] = query_slots
            numerical = cast("_EGMStep", core)
            retain_child_mask = (
                cast("_BoundDCEGM", self)._retain_child_feasibility  # noqa: SLF001
                and bool(plan.query_targets)
                and not numerical.pieces._retain_nested_feasibility  # noqa: SLF001
            )
            if id(core) not in programs_by_core:
                if retain_child_mask:
                    numerical = replace(
                        numerical,
                        pieces=replace(
                            numerical.pieces, _retain_nested_feasibility=True
                        ),
                    )
                checked_step = replace(
                    numerical,
                    pieces=replace(
                        numerical.pieces, collect_continuation_read_counts=True
                    ),
                )
                values_core = functools.partial(_dcegm_values_core, core=numerical)
                replay_core = numerical
                checked_values = functools.partial(
                    _dcegm_values_core, core=checked_step
                )
                checked_replay = checked_step
                if numerical.pieces._retain_nested_feasibility:  # noqa: SLF001
                    values_core = functools.partial(
                        values_core, retain_nested_feasibility=True
                    )
                    checked_values = functools.partial(
                        checked_values, retain_nested_feasibility=True
                    )
                if context.enable_jit:
                    # The plan's width is a compile-time choice, so the solver's
                    # own compilation must hold it static too; the engine's outer
                    # jit then hands this one a concrete width.
                    values_core = jax.jit(values_core, static_argnames=_WIDTH_KEYWORDS)
                    replay_core = jax.jit(replay_core, static_argnames=_WIDTH_KEYWORDS)
                    checked_values = jax.jit(
                        checked_values, static_argnames=_WIDTH_KEYWORDS
                    )
                    checked_replay = jax.jit(
                        checked_replay, static_argnames=_WIDTH_KEYWORDS
                    )
                requirements = CoreExecutionRequirements(
                    reduced_axes=_stochastic_node_axis(
                        node_axes=build.stochastic_node_axes_by_period[period],
                        state_action_space=context.state_action_space,
                    ),
                    tiled_axes=_tiled_axes(build=build),
                )
                main_roles = _dcegm_output_roles(
                    build=build,
                    publish_replay=False,
                    own_bound_specs=own_bound_specs,
                    query_slots=query_slots,
                )
                replay_roles = _dcegm_output_roles(
                    build=build,
                    publish_replay=True,
                    own_bound_specs=own_bound_specs,
                    query_slots=query_slots,
                )
                if numerical.pieces._retain_nested_feasibility:  # noqa: SLF001
                    current_role = StateAxesLeading(
                        state_names=context.regime_to_v_interpolation_info[
                            context.regime_name
                        ].state_names,
                        dtype=jnp.bool_,
                    )
                    main_roles = _with_nested_feasibility_role(
                        roles=main_roles, current_role=current_role
                    )
                    replay_roles = _with_nested_feasibility_role(
                        roles=replay_roles, current_role=current_role
                    )
                programs_by_core[id(core)] = MappingProxyType(
                    {
                        "main": CoreProgram(
                            name="main",
                            function=values_core,
                            argument_builder=argument_builder,
                            requirements=requirements,
                            output_roles=main_roles,
                            disposition=CoreExecutionDisposition.PLANNED,
                            donation_candidates=(),
                            scope=ProgramScope.VALUES_ONLY,
                            checked_variant=_CheckedCoreVariant(
                                function=checked_values,
                                output_roles=_counted_dcegm_output_roles(
                                    roles=main_roles,
                                ),
                            ),
                            compiler_options=(
                                (("scan_unroll", 1),)
                                if isinstance(self.envelope, FUESEnvelope)
                                else ()
                            ),
                        ),
                        "replay": CoreProgram(
                            name="replay",
                            function=replay_core,
                            argument_builder=argument_builder,
                            requirements=requirements,
                            output_roles=replay_roles,
                            disposition=CoreExecutionDisposition.PLANNED,
                            donation_candidates=(),
                            scope=ProgramScope.REPLAY,
                            checked_variant=_CheckedCoreVariant(
                                function=checked_replay,
                                output_roles=_counted_dcegm_output_roles(
                                    roles=replay_roles,
                                ),
                            ),
                            compiler_options=(
                                (("scan_unroll", 1),)
                                if isinstance(self.envelope, FUESEnvelope)
                                else ()
                            ),
                            retained_artifact_keys=(SIMULATION_POLICY,),
                            retained_artifact_payload_types={
                                SIMULATION_POLICY: EGMSimPolicy
                            },
                            replaces_program="main",
                        ),
                    }
                )
            period_group_keys[period] = build.step_group_keys[period]
            period_kernels[period] = _DCEGMPeriodKernel(
                _core_programs=programs_by_core[id(core)],
                regime_name=context.regime_name,
                stateful_targets=build.stateful_targets,
                transition_target_names=tuple(context.transitions),
                own_bound_specs=own_bound_specs,
                query_input_authorities=query_slots,
                retain_nested_feasibility=cast(  # noqa: SLF001
                    "_EGMStep", core
                ).pieces._retain_nested_feasibility,
                retain_child_feasibility=retain_child_mask,
            )
        return SolutionKernels(
            continuation_reader_capabilities=reader_capabilities or None,
            query_input_authorities=query_slots_by_period or None,
            period_kernels=MappingProxyType(period_kernels),
            period_group_keys=MappingProxyType(period_group_keys),
            continuation_spec=EGMContinuationSpec(
                template=context.place_on_regime_devices(template=build.carry_template),
                layout=(
                    self.egm_continuation_layout
                    if query_model is None
                    else replace(
                        self.egm_continuation_layout,
                        query_spec=query_model.authority,
                    )
                ),
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
    post_decision_bound: PostDecisionBoundSpec | None = None
    """Literal grid floor installed as a construction capability."""

    _negm_current_publication: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Use native floor ownership for eligible nested current-value publication."""

    _retain_nested_feasibility: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Keep current eligibility per ordinary action for an enclosing outer choice."""

    _retain_child_feasibility: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Request transient eligibility only when actual child query readers exist."""


# Planner name of the child stochastic-node mesh the DC-EGM continuation folds.
STOCHASTIC_NODE_AXIS = "stochastic_node"

# Planner name of the exact envelope's resource-node cells.
ENVELOPE_CELL_AXIS = "envelope_cell"

# Planner name of the output state cells the per-combo solve is tiled over.
CELL_AXIS = "cell"

# Planner name of the exogenous savings nodes the continuation is tiled over.
SAVINGS_POINT_AXIS = "savings_point"

# Planner name of the exogenous Euler nodes the asset-row solve is tiled over.
EULER_POINT_AXIS = "euler_point"

_STOCHASTIC_NODE_WIDTH_KEYWORD = "_lcm_stochastic_node_width"

_TILED_AXIS_WIDTH_KEYWORDS = MappingProxyType(
    {
        CELL_AXIS: "_lcm_cell_width",
        SAVINGS_POINT_AXIS: "_lcm_savings_point_width",
        EULER_POINT_AXIS: "_lcm_euler_point_width",
        ENVELOPE_CELL_AXIS: "_lcm_envelope_cell_width",
    }
)


def _dcegm_query_slots_and_capabilities(
    *,
    core: EGMStepFunction[_DCEGMStepOutput],
    regime_name: RegimeName,
    period: int,
    own_bound_specs: tuple[PostDecisionBoundSpec, ...],
) -> tuple[
    _DCEGMQuerySlots | None, Mapping[RegimeName, ContinuationCapabilities] | None
]:
    """Describe actual installed child readers and the current query bound."""
    from _lcm.egm.step import _EGMStep  # noqa: PLC0415

    plan = cast("_EGMStep", core).pieces.continuation_plan
    child_queries = {
        target: query
        for target in plan.query_targets
        if (query := plan.child_reads[target].query) is not None
    }
    if len(child_queries) != len(plan.query_targets):
        raise TypeError("DC child slots require actual installed query readers.")
    child_slots = tuple(
        (
            target,
            period + 1,
            child_queries[target].model.authority.bound,
        )
        for target in plan.query_targets
    )
    own_slot = tuple((regime_name, period, bound) for bound in own_bound_specs)
    query_slots = (*child_slots, *own_slot) or None
    capabilities = (
        MappingProxyType(
            {
                target: child_queries[target].capabilities
                for target in plan.query_targets
            }
        )
        if plan.query_targets
        else None
    )
    return query_slots, capabilities


def _tiled_axes(*, build: EGMStepBuild) -> tuple[TiledOutputAxis, ...]:
    """Declare the loops the DC-EGM kernel runs in planner-sized tiles.

    Each is a loop whose per-tile results are concatenated rather than folded,
    so the axis is a tiled output axis and every width names the same result:

    - `cell` over the regime's output state cells (its discrete and passive
      states); discrete actions stay outside it, because the action
      aggregation needs every action's value at once;
    - `savings_point` over the exogenous savings nodes of the continuation;
    - `euler_point` over the exogenous Euler nodes, which only the asset-row
      kernel loops over;
    - `envelope_cell` over adjacent candidate abscissae, which only the exact
      envelope visits with a tiled cell scan.

    A loop of one cell has nothing to tile and carries no declaration, so a
    regime whose kernel does not run it — the single-post-state kernel's node
    loop, a single-cell state product — declares no axis for it and
    `ExecutionConfig(axis_widths=...)` refuses that name.
    """
    extents = {
        CELL_AXIS: build.cell_extent,
        SAVINGS_POINT_AXIS: build.savings_point_extent,
        EULER_POINT_AXIS: build.euler_point_extent,
        ENVELOPE_CELL_AXIS: build.envelope_cell_extent,
    }
    state_names = {
        CELL_AXIS: build.row_discrete_state_names + build.row_passive_state_names,
        SAVINGS_POINT_AXIS: (),
        EULER_POINT_AXIS: (),
        ENVELOPE_CELL_AXIS: (),
    }
    return tuple(
        TiledOutputAxis(
            name=name,
            state_names=state_names[name],
            extent=extents[name],
            width_keyword=keyword,
        )
        for name, keyword in _TILED_AXIS_WIDTH_KEYWORDS.items()
        if extents[name] > 1
    )


# Every planner width the DC-EGM core takes, in one tuple: the width is a
# compile-time choice, so the solver's own jit has to hold all of them static.
_WIDTH_KEYWORDS = (
    _STOCHASTIC_NODE_WIDTH_KEYWORD,
    *_TILED_AXIS_WIDTH_KEYWORDS.values(),
)


def _stochastic_node_axis(
    *,
    node_axes: tuple[tuple[StateName, int], ...],
    state_action_space: StateActionSpace,
) -> tuple[ReducedAxis, ...]:
    """Declare the child stochastic-node mesh the continuation folds, where it is one.

    An axis's coordinates are grids the program receives, so the mesh is
    declarable exactly when the regime carries every child stochastic state on
    a grid of the child's own node count. Three meshes carry no such
    declaration and fold in one block instead:

    - a regime whose reachable targets carry no stochastic state at all;
    - a one-node mesh, which has nothing to partition;
    - a cross-grid mesh, whose child integrates a state on a shorter grid than
      the parent's own — the node axis then belongs to the child and no grid
      the program receives has its length.
    """
    grids = state_action_space.states
    declarable = all(
        name in grids and int(jnp.asarray(grids[name]).shape[0]) == count
        for name, count in node_axes
    )
    extent = math.prod(count for _, count in node_axes)
    if not node_axes or not declarable or extent <= 1:
        return ()
    return (
        ReducedAxis(
            name=STOCHASTIC_NODE_AXIS,
            coordinate_names=tuple(name for name, _ in node_axes),
            coordinate_extents=tuple(count for _, count in node_axes),
            canonical_order="c",
            reduction=WEIGHTED_EXPECTATION_REDUCTION,
            width_keyword=_STOCHASTIC_NODE_WIDTH_KEYWORD,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class EGMStepBuild:
    """What one regime's DC-EGM step build hands its period kernels."""

    steps: MappingProxyType[int, EGMStepFunction[_DCEGMStepOutput]]
    """Per-period step function; periods sharing a configuration share one."""

    step_group_keys: MappingProxyType[int, Hashable]
    """Per-period key of the configuration group whose step the period got.

    Built from continuation targets and declared signatures alone, so two
    builds of one model assign a period the same key. Two periods share a step
    exactly when they share this key.
    """

    carry_template: EGMCarry
    """The regime's all-finite carry template (discrete states, then passive
    states, then discrete actions leading; the refined row axis last)."""

    stateful_targets: frozenset[RegimeName]
    """The carry keys the steps read; the rolling carry is filtered to these."""

    row_discrete_state_names: tuple[StateName, ...]
    """Discrete states leading every carry and policy row, in row order."""

    stochastic_node_axes_by_period: MappingProxyType[
        int, tuple[tuple[StateName, int], ...]
    ]
    """Each period core's actual child stochastic mesh, as state names and counts."""

    row_passive_state_names: tuple[StateName, ...]
    """Passive continuous states following the discrete states on every row."""

    row_discrete_action_names: tuple[ActionName, ...]
    """Discrete actions following the passive states on every row."""

    cell_extent: int
    """Number of output state cells: the discrete-by-passive state product."""

    savings_point_extent: int
    """Number of nodes on the solver's exogenous savings grid."""

    euler_point_extent: int
    """Number of asset-row nodes, or `0` where the kernel runs no node loop."""

    envelope_cell_extent: int
    """Number of exact-envelope node cells, or `0` for another backend."""


def _counted_dcegm_output_roles(*, roles: OutputRoleTree) -> OutputRoleTree:
    """Describe optional counts using role placeholders, without numeric allocation."""
    counts = _unflatten_continuation_read_counts(
        None,
        tuple(
            StateAxesLeading(state_names=(), shape=(), dtype=jnp.int32)
            for _ in range(2)
        ),
    )
    if isinstance(roles, BoundedResult):
        return replace(roles, continuation_read_counts=counts)
    if not isinstance(roles, tuple):
        raise TypeError("DC output roles require the declared ordinary tuple.")
    return BoundedResult(
        output=(*roles, None), bound_status=(), continuation_read_counts=counts
    )


def _dcegm_output_roles(
    *,
    build: EGMStepBuild,
    publish_replay: bool,
    own_bound_specs: tuple[PostDecisionBoundSpec, ...] = (),
    query_slots: _DCEGMQuerySlots | None = None,
) -> OutputRoleTree:
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
    common = (
        VALUE,
        egm_carry_role_tree(
            row=row,
            scalar=StateAxesLeading(state_names=(), shape=()),
            breakpoints=None,
            policy=None,
            query_data=(
                None
                if build.carry_template.query_data is None
                else egm_query_role_tree(
                    query_data=build.carry_template.query_data,
                    state_names=row.state_names,
                )
            ),
        ),
    )
    output: tuple[OutputRoleTree, ...] = common
    if publish_replay:
        output = (
            *common,
            egm_sim_policy_role_tree(
                row=row,
                row_discrete_state_names=build.row_discrete_state_names,
                row_passive_state_names=build.row_passive_state_names,
                row_discrete_action_names=build.row_discrete_action_names,
            ),
        )
    if query_slots is None:
        return output
    count = StateAxesLeading(state_names=(), shape=(), dtype=jnp.int32)
    template = _dcegm_role_template(
        build=build,
        publish_replay=publish_replay,
        own_bound_specs=own_bound_specs,
        query_slots=query_slots,
    )
    return jax.tree.unflatten(
        jax.tree.structure(template),
        jax.tree.leaves(
            (
                (*output, (count, count)),
                jax.tree.map(_dcegm_status_role, template.bound_status),
                jax.tree.map(_dcegm_status_role, template.query_status),
            )
        ),
    )


def _dcegm_role_template(
    *,
    build: EGMStepBuild,
    publish_replay: bool,
    own_bound_specs: tuple[PostDecisionBoundSpec, ...],
    query_slots: _DCEGMQuerySlots,
) -> BoundedResult[tuple[PytreeValue, ...]]:
    """Construct the numeric structure whose registered unflatten restores roles."""
    carry = build.carry_template
    output: tuple[PytreeValue, ...] = (jnp.asarray(0.0), carry)
    if publish_replay:
        output = (
            *output,
            EGMSimPolicy(
                endog_grid=carry.endog_grid,
                policy=carry.endog_grid,
                value=carry.value,
                marginal_utility=carry.marginal_utility,
                row_discrete_state_names=build.row_discrete_state_names,
                row_passive_state_names=build.row_passive_state_names,
                row_discrete_action_names=build.row_discrete_action_names,
            ),
        )
    inactive = jnp.asarray(a=False)
    return BoundedResult(
        output=(*output, (jnp.int32(0), jnp.int32(0))),
        bound_status=tuple(
            BoundInputStatus(inactive, inactive) for _ in own_bound_specs
        ),
        query_status=tuple(
            QueryInputStatus(inactive, inactive, inactive) for _ in query_slots
        ),
    )


def _dcegm_status_role(flag: BoolND) -> StateAxesLeading:
    """Describe bounded Boolean diagnostics independently of every numeric row axis."""
    return StateAxesLeading(state_names=(), shape=flag.shape, dtype=flag.dtype)


def _dcegm_values_core(
    *,
    core: Callable[..., _DCEGMStepOutput],
    retain_nested_feasibility: bool = False,
    **kwargs: PytreeValue,
) -> (
    tuple[FloatND, EGMCarry]
    | BoundedResult[tuple[FloatND, EGMCarry, FeasibilityCounts | None]]
    | BoundedResult[tuple[FloatND, EGMCarry, FeasibilityCounts | None, BoolND]]
):
    """Run DC-EGM without retaining its optional simulation-policy output."""
    evaluated = core(**kwargs)
    if retain_nested_feasibility:
        if (
            type(evaluated) is not BoundedResult
            or type(evaluated.output) is not tuple
            or len(evaluated.output) != _NESTED_DCEGM_STEP_FIELD_COUNT
        ):
            raise TypeError(
                "A nested DC producer must retain current action eligibility."
            )
        value, carry, _policy, summary, current_by_action = cast(
            "_NestedDCEGMStepChannels", evaluated.output
        )
        return BoundedResult[
            tuple[FloatND, EGMCarry, FeasibilityCounts | None, BoolND]
        ](
            output=(value, carry, summary, current_by_action),
            bound_status=evaluated.bound_status,
            query_status=evaluated.query_status,
            continuation_read_counts=evaluated.continuation_read_counts,
        )
    if isinstance(evaluated, BoundedResult):
        if len(evaluated.output) != _DCEGM_REPLAY_FIELD_COUNT:
            raise TypeError(
                "An ordinary DC producer must declare four bounded channels."
            )
        value, carry, _policy, summary = cast(
            "tuple[FloatND, EGMCarry, EGMSimPolicy, FeasibilityCounts | None]",
            evaluated.output,
        )
        return BoundedResult(
            output=(value, carry, summary),
            bound_status=evaluated.bound_status,
            query_status=evaluated.query_status,
            continuation_read_counts=evaluated.continuation_read_counts,
        )
    value, carry, _policy = evaluated
    return value, carry


def _with_nested_feasibility_role(
    *, roles: OutputRoleTree, current_role: StateAxesLeading
) -> BoundedResult[tuple[OutputRoleTree, ...]]:
    """Append actual per-action eligibility with canonical leading state axes."""
    if isinstance(roles, BoundedResult):
        if type(roles.output) is not tuple:
            raise TypeError("A bounded DC program must declare tuple output roles.")
        bounded_roles = cast("BoundedResult[tuple[OutputRoleTree, ...]]", roles)
        return replace(bounded_roles, output=(*bounded_roles.output, current_role))
    if type(roles) is not tuple:
        raise TypeError("A DC program must declare its numeric output tuple.")
    tuple_roles = cast("tuple[OutputRoleTree, ...]", roles)
    return BoundedResult(output=(*tuple_roles, None, current_role), bound_status=())


def declare_dcegm_carry_reads(
    *, kernels: SolutionKernels, context: SolverBuildContext
) -> SolutionKernels:
    """Attach each DC-EGM period's declared carry reads to its shared programs.

    Periods sharing one numerical core share the programs built around it; the
    reads are a fact about the period, so each period gets its own program
    mapping around the same core.
    """
    return replace(
        kernels,
        period_kernels=MappingProxyType(
            {
                period: dcegm_kernel_with_declared_reads(
                    kernel=kernel, context=context, period=period
                )
                for period, kernel in kernels.period_kernels.items()
            }
        ),
    )


def dcegm_kernel_with_declared_reads(
    *, kernel: PeriodKernel, context: SolverBuildContext, period: int
) -> PeriodKernel:
    """Return one DC-EGM period's adapter with its reachable targets' rows declared.

    The builder hands the step the whole rolling payload, keyed by every carry
    target the core indexes across all periods for pytree stability. Of those,
    only the targets this period actually reaches publish a continuation at
    `period + 1` — a target inactive there still occupies a carry key, but the
    rolling mapping holds the build template for it, not a produced artifact.
    Reads are declared for the reachable subset, one per published leaf, each
    under its own leaf path inside the rolling mapping. The replay variant
    runs the same reads under its own core name. Shared with the composite
    solvers that embed this adapter.
    """
    if not isinstance(kernel, _DCEGMPeriodKernel):
        return kernel
    reachable_targets = (
        ()
        if period == context.solution_reachability.n_periods - 1
        else context.solution_reachability.targets(
            period=period,
            source=context.regime_name,
        )
    )
    reads = tuple(
        read
        for target, template in published_continuation_templates(
            continuation_specs=context.continuation_specs,
            targets=kernel.stateful_targets & frozenset(reachable_targets),
        ).items()
        for read in continuation_leaf_reads(
            template=template,
            artifact_key=EGM_CONTINUATION,
            target=target,
            source_regime=kernel.regime_name,
            source_period=period,
            core_key="main",
        )
    )
    return replace(
        kernel,
        _core_programs=with_continuation_leaf_reads(
            programs=kernel.core_programs(),
            reads_by_core_key={
                "main": reads,
                "replay": rekeyed_value_reads(reads=reads, core_key="replay"),
            },
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

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Return the exact kwargs shared by lowering and the runtime call."""
        state_action_space = cast("StateActionSpace", context.state_action_space)
        flat_params = cast("FlatParams", context.flat_params)
        ages = cast("TimeAxis", context.ages)
        return MappingProxyType(
            {
                **dict(state_action_space.states),
                "next_regime_to_continuation": _carry_subset(
                    next_regime_to_continuation=context.next_regime_to_continuation,
                    stateful_targets=self.stateful_targets,
                ),
                **union_free_params(
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
    """The DC-EGM period kernel: two output-specialized dense programs.

    `main` inverts the Euler equation on the savings grid and publishes the
    value function plus the continuation a parent interpolates. `replay` is the
    output-specialized variant that also publishes the off-grid simulation
    policy. Both declare the child stochastic-node mesh as a reduced axis, so
    the execution plan owns the width their continuation expectation folds at.
    Calling the kernel builds the selected program's arguments through its
    declared builder.
    """

    _core_programs: Mapping[str, CoreProgram]
    """The immutable values/replay program variants."""

    regime_name: RegimeName
    """Name of the regime whose flat params this kernel projects."""

    stateful_targets: frozenset[RegimeName]
    """The carry keys the step reads; the rolling carry is filtered to these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    own_bound_specs: tuple[PostDecisionBoundSpec, ...] = ()
    """Canonical source-floor arithmetic slots installed by this kernel's builder."""

    query_input_authorities: _DCEGMQuerySlots | None = None
    """Installed query/construction owners, absent on the legacy output layout."""

    retain_nested_feasibility: bool = False
    """Consume a private action-eligibility channel at the standalone host boundary."""

    retain_child_feasibility: bool = False
    """Keep per-action eligibility for an enclosing solver with child query reads."""

    @property
    def retains_current_feasibility(self) -> bool:
        """Whether either nested composition owns the transient action mask."""
        return self.retain_nested_feasibility or self.retain_child_feasibility

    def __post_init__(self) -> None:
        """Snapshot the graph and require the two retention-scoped variants."""
        programs = MappingProxyType(dict(self._core_programs))
        if tuple(programs) != ("main", "replay"):
            msg = (
                "A DC-EGM kernel publishes exactly 'main' and 'replay'; got "
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
        egm_fixed = union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not egm_fixed:
            return self
        return replace(
            self,
            _core_programs=MappingProxyType(
                {
                    name: bind_core_program_fixed_params(
                        program=program, params=egm_fixed
                    )
                    for name, program in self._core_programs.items()
                }
            ),
        )

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, NextValue],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: TimeAxis,
        logger: logging.Logger,  # noqa: ARG002
    ) -> KernelOutput:
        """Run the selected output variant and publish its typed artifacts."""
        name = "replay" if "replay" in compiled_cores else "main"
        program = self._core_programs[name]
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
        outputs = compiled_cores[name](**arguments)
        if self.retains_current_feasibility:
            outputs = _strip_nested_feasibility(evaluated=outputs, variant=name)
        if self.query_input_authorities is not None:
            if type(outputs) is not BoundedResult:
                raise TypeError("DC query outputs require declared integrity records.")
            return _consume_bounded_dcegm_output(
                evaluated=outputs,
                variant=name,
                own_bound_specs=self.own_bound_specs,
                query_slots=self.query_input_authorities,
            )
        if isinstance(outputs, BoundedResult):
            if (
                outputs.bound_status
                or outputs.query_status is not None
                or outputs.continuation_read_counts is None
            ):
                raise TypeError(
                    "DC integrity records require model-owned output slots."
                )
            if name == "main":
                V_arr, egm_carry, feasibility = outputs.output
                replay = {}
            else:
                V_arr, egm_carry, sim_policy, feasibility = outputs.output
                replay = {SIMULATION_POLICY: sim_policy}
            if feasibility is not None:
                raise TypeError("Count-only DC output cannot measure feasibility.")
            return ContinuationCountedKernelOutput(
                value=V_arr,
                continuations={EGM_CONTINUATION: egm_carry},
                replay=replay,
                continuation_read_counts=outputs.continuation_read_counts,
            )
        if name == "main":
            V_arr, egm_carry = cast("tuple[FloatND, EGMCarry]", outputs)
            return KernelOutput(
                value=V_arr,
                continuations={EGM_CONTINUATION: egm_carry},
            )
        V_arr, egm_carry, sim_policy = cast(
            "tuple[FloatND, EGMCarry, EGMSimPolicy]", outputs
        )
        return KernelOutput(
            value=V_arr,
            continuations={EGM_CONTINUATION: egm_carry},
            replay={SIMULATION_POLICY: sim_policy},
        )


def _strip_nested_feasibility(*, evaluated: PytreeValue, variant: str) -> PytreeValue:
    """Remove only a validated transient action mask at the host publication edge."""
    expected = 4 if variant == "main" else 5
    if (
        type(evaluated) is not BoundedResult
        or type(evaluated.output) is not tuple
        or len(evaluated.output) != expected
    ):
        raise TypeError("A nested DC producer lacks its action-eligibility channel.")
    value, *_, current_by_action = evaluated.output
    if (
        current_by_action.dtype != jnp.bool_
        or current_by_action.shape[:-1] != value.shape
        or current_by_action.shape[-1] < 1
    ):
        raise TypeError("Nested DC eligibility must preserve every state and action.")
    numeric = evaluated.output[:-1]
    if (
        not evaluated.bound_status
        and evaluated.query_status is None
        and evaluated.continuation_read_counts is None
    ):
        if numeric[-1] is not None:
            raise TypeError("An unbounded nested DC output cannot measure feasibility.")
        return numeric[:-1]
    return replace(evaluated, output=numeric)


def _consume_bounded_dcegm_output[Output: PytreeValue](
    *,
    evaluated: BoundedResult[Output],
    variant: str,
    own_bound_specs: tuple[PostDecisionBoundSpec, ...],
    query_slots: _DCEGMQuerySlots,
) -> FeasibilityKernelOutput:
    """Forward bounded diagnostics for acceptance after every invariant block."""
    if (
        len(evaluated.bound_status) != len(own_bound_specs)
        or evaluated.query_status is None
        or len(evaluated.query_status) != len(query_slots)
    ):
        raise TypeError("DC integrity records differ from model-owned output slots.")
    fields = (
        _DCEGM_VALUES_FIELD_COUNT if variant == "main" else _DCEGM_REPLAY_FIELD_COUNT
    )
    if type(evaluated.output) is not tuple or len(evaluated.output) != fields:
        raise TypeError("DC bounded output lacks its declared numeric channels.")
    if evaluated.output[-1] is None:
        raise TypeError("Mandatory DC output requires measured feasibility.")
    if variant == "main":
        value, carry, counts = cast(
            "tuple[FloatND, EGMCarry, FeasibilityCounts]", evaluated.output
        )
        replay = {}
    else:
        value, carry, policy, counts = cast(
            "tuple[FloatND, EGMCarry, EGMSimPolicy, FeasibilityCounts]",
            evaluated.output,
        )
        replay = {SIMULATION_POLICY: policy}
    return FeasibilityKernelOutput(
        value=value,
        continuations={EGM_CONTINUATION: carry},
        replay=replay,
        feasibility_counts=counts,
        bound_status=evaluated.bound_status,
        query_status=evaluated.query_status,
        continuation_read_counts=evaluated.continuation_read_counts,
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
            name: cast("EGMCarry", next_regime_to_continuation[name])
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


def _fail_if_envelope_max_runs_too_few(envelope_max_runs: int) -> None:
    if envelope_max_runs < _MIN_ENVELOPE_MAX_RUNS:
        msg = (
            f"ExactEnvelope.max_runs must be at least {_MIN_ENVELOPE_MAX_RUNS}, "
            f"got {envelope_max_runs}. It is the fold capacity of the exact "
            "upper envelope; a non-concave candidate chain folds into at "
            "least two resource-increasing runs."
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
