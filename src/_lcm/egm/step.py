"""The per-(period, regime) DC-EGM solve kernel.

`build_egm_step_functions` turns a regime's processed functions, transitions,
and `DCEGM` solver configuration into per-period kernels that replace the
brute-force `max_Q_over_a` during backward induction. Each kernel runs the
DC-EGM step per combination of the regime's discrete states, passive
continuous states (deterministic non-process states whose transitions are
independent of the decision, e.g. an AIME-like skill level — one node per
combo), and discrete actions, with the exogenous savings grid as the inner
axis:

1. compute child states at every savings node (the post-decision function is
   removed from the DAG so the savings node enters as an external input),
2. map the child state into the child's resources space, select the carry
   rows matching the child's discrete-state values, and interpolate the
   child's value and marginal utility there per child discrete-action combo.
   The child's resources space is *per combo*: when the child's resources
   function reads its discrete states, passive states, or discrete actions,
   each carry row is queried at its own $R'$ and carries its own composed
   gradient $\\partial R'/\\partial A$ of the map
   $A \\mapsto R'(\\mathcal{T}(A), z', d', p')$ — the non-Euler inputs are
   savings-independent (validated) and ride as constants through the
   gradient. When the resources function reads only the child's Euler state,
   a single query and gradient is computed and broadcast across the rows.
   The child's passive values land off-grid, so the read is *mixed*: linear
   weights on the two neighboring nodes of the child's passive grid
   (edge-clamped), each neighbor's row interpolated 1-D in its own node's
   resources space, then blended per discrete-action row — before the choice
   aggregation, so the logsum sees blended choice-specific values,
3. aggregate over the child's discrete-action rows with the child's EV1
   taste-shock scale: the smoothed value is the logsum and the smoothed
   marginal is the choice-probability-weighted marginal
   $\\sum_{d'} P_{d'} \\mu_{d'} (\\partial R'/\\partial A)_{d'}$ — the
   composed factor sits inside the aggregation because each choice's
   envelope lives in its own resources space (exact for EV1 by Danskin's
   theorem; scale zero degrades to the hard max / one-hot argmax),
4. take the stochastic-node and regime-transition expectations: a child
   stochastic state's node — a continuous AR(1) process state or a
   Markov-discrete state — is distributed per its intrinsic transition weights
   $w(\\text{node}' \\mid \\text{node}, \\text{params})$, so steps 2-3 run
   per child node combo and the results are weight-summed — the stochastic
   expectation sits *outside* the action aggregation (the shock realizes
   before the next period's choice), matching the brute-force solver's
   weighted average of the already action-aggregated next-period V — and the
   per-target results are regime-transition-probability weighted,
5. invert the Euler equation per savings node (with the degenerate-inversion
   guard) to obtain the optimal action and the endogenous resources grid,
6. add the closed-form credit-constrained segment as additional candidates,
7. refine the candidate correspondence through the configured upper-envelope
   backend (one envelope per discrete combo),
8. publish the value function on the regime's exogenous state grid —
   discrete-state and passive-state combos remain axes of the value-function
   array (the Euler axis is moved to its canonical position among the
   continuous states), the regime's own discrete-action combos are
   aggregated with the regime's own taste-shock scale — and assemble the
   per-combo carry rows for the regime's parents. The publish step never
   evaluates anything at off-grid passive values: each combo's rows are
   interpolated in resources only, on the regime's own grid axes.

When any savings-stage function reads the current Euler state — the Euler
state's own law (an additive residual like a means-tested capital-income
supplement), the regime-transition probabilities, a stochastic transition
weight, or a passive state's law — the kernel solves *per exogenous asset
node* instead: conditional on one node of the Euler grid every such read is
a per-combo constant, so the single-post-state pipeline above is exact
within the node's row, and the row publishes only its own node. This is
brute-force-equivalent by construction — the brute solver evaluates the
same decision-time functions at the same exogenous nodes. The per-combo
carry then holds the per-node published points (one row per combo, no asset
axis): abscissa the node resources, policy the published optimal action,
value the published V, and marginal
$dV/dR = u'(c^*) + \\beta\\, (\\partial W/\\partial a)\\big|_{A^*} / R'(a)$
— the envelope term $u'(c^*)$ plus the savings-stage functions' direct
Euler-state channels through the continuation $W$, divided by the resources
slope. The Euler-state channel is differentiated through the *whole*
continuation closure: regime-transition probabilities and transition
weights are evaluated inside it, so first-order terms like
$\\sum_{targets} \\partial P/\\partial a \\cdot EV_{target}$ survive — the
probabilities are not the softmax of the values they weight, so Danskin
does not cancel them, and hoisting them out of the differentiated closure
would silently drop them. The mode requires every savings-stage Euler-state
read to be *continuous* in the Euler state at the resolution of the Euler
grid (kinks are fine): a residual that jumps makes the child's value
function discontinuous, the true policy bunches next-period wealth exactly
at the discontinuity — an Euler-equation-free corner outside EGM's
candidate families (interior Euler inversions plus the closed-form
credit-constrained segment) — and a jump in the probabilities or weights
breaks the smoothness-at-node-resolution assumption the per-node solve
relies on, so such functions are rejected at build time rather than solved
approximately. The switch is a build-time Python branch; the default path
is untouched and costs nothing when the mode is unused.

The canonical combo-axis order — single-sourced through
`_EgmKernelPieces.combo_names` and the carry template's leading shape — is
discrete states (V state order, process states included as node-valued
discrete dimensions), then passive states (V continuous-state order), then
discrete actions (action-grid order), then the carry's grid axis. Discrete
states lead so the child carry can be selected by integer indexing; passive
axes follow so the mixed read blends them away next, leaving exactly the
discrete-action rows the choice aggregation consumes.

Discrete-only constraints mask infeasible discrete combos: their value rows
are $-\\infty$ and their marginal-utility rows are exactly zero, so they
carry zero choice probability and stay finite inside the parent's
probability-weighted expectation.

A terminal carry target may carry discrete states, but only those shared
with the parent's own discrete combo axes and reached by the identity
(fixed) transition — at a parent combo with the state at value $k$ the
terminal carry row is its utility evaluated at $k$, selected by the parent's
own integer combo index, exactly as the non-terminal child read selects its
combo. A fixed `pref_type` whose terminal bequest differs by type is the
motivating case.

A child Markov-discrete state — one whose `next_<name>` is a stochastic
transition into the target — is integrated exactly like a process state: its
node axis is the *child's* discrete grid (codes $0, 1, \\dots$, which also
index the carry's leading axis), distributed per the intrinsic weights
$w(\\text{node}' \\mid \\text{node}, \\text{params})$, summed outside the
action aggregation. The child grid need not match the source's: a 3-state
health remapped onto a 2-state target carries a length-2 weight vector, and
the integration ranges over the child axis.

Out of scope:
terminal carry targets with actions, with a discrete state the parent does
not carry, or with a non-identity transition into a shared discrete state,
child resources functions reading anything beyond the child's states and
discrete actions (e.g. free child params), and child process states whose
grid points are supplied at runtime while feeding the child's resources
function. Such configurations build kernels that raise `NotImplementedError`
at solve time, so `Model` construction always succeeds for a validated
DC-EGM regime.
"""

import functools
import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions, with_signature

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.constraints.processed import ProcessedConstraintsMapping
from _lcm.continuation import ProjectedContinuationSpec, ProjectedProducerDate
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.asset_row import _get_solve_one_combo_asset_rows
from _lcm.egm.carry import EGMCarry, build_template_egm_carry
from _lcm.egm.continuation import (
    _is_runtime_process,
    build_continuation_plan,
    get_egm_continuation_targets,
    stochastic_node_axes,
)
from _lcm.egm.continuation_grids import (
    continuation_grid_signature,
    continuation_v_interpolation_info,
)
from _lcm.egm.current_iid_envelope import DatedCaseUStepAuthority
from _lcm.egm.current_iid_euler_grid import (
    ResolvedDatedEulerGrid,
    materialize_step_euler_grid,
)
from _lcm.egm.dcegm_query_model import DCEGMQueryModel
from _lcm.egm.kernel_scope import _find_unsupported_feature
from _lcm.egm.nbegm_query_model import NBEGMQueryModel
from _lcm.egm.nested_floor import PreparedNestedFloor
from _lcm.egm.preferences import concatenate_regime_function
from _lcm.egm.preparation import prepare_egm_carry, prepare_egm_query_data
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.egm.query_data import EGMQueryData, build_template_egm_query_data
from _lcm.egm.regime_introspection import (
    _get_discrete_state_names,
    _get_passive_state_names,
    _get_process_state_names,
)
from _lcm.egm.step_core import (
    CONSTRAINED_OFFSET_FRACTION,
    _ComboRows,
    _ConditionalComboRows,
    _EgmKernelPieces,
    _get_solve_one_combo,
    tile_block_size,
)
from _lcm.egm.upper_envelope import get_bracket_finder, get_upper_envelope
from _lcm.egm.validation import savings_stage_reads_euler_state
from _lcm.egm.validation_materialization import _validation_grid_first
from _lcm.engine import StateActionSpace
from _lcm.logsum import logsum_and_softmax
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.reachability import PhaseReachability
from _lcm.regime_building.age_normalization import (
    periodized_tree_signature,
    resolve_periodized_nodes,
)
from _lcm.regime_building.max_Q_over_a import TASTE_SHOCK_SCALE_PARAM
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.regime_building.w_dag import _get_build_W_kwargs
from _lcm.regime_law import RegimeLaws
from _lcm.solution.dcegm import (
    EGMStepBuild,
    ExactEnvelope,
    _BoundDCEGM,
    _DCEGMStepOutput,
    _NestedDCEGMStepChannels,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    FeasibilityCounts,
    count_current_feasibility,
)
from _lcm.transition_plans import TargetTransitionPlans
from _lcm.typing import (
    ActionName,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionArg,
    EconFunctionsMapping,
    EGMStepFunction,
    QualifiedName,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.dispatchers import map_over_leading_axis, productmap
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    BoolND,
    DiscreteAction,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)

type _MappedConditionalRows = tuple[
    FloatND,
    FloatND,
    FloatND,
    FloatND,
    FloatND,
    BoolND,
    EGMQueryData | None,
    BoolND,
]


def build_egm_step_functions(
    *,
    solver: _BoundDCEGM,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None = None,
    constraints: ConstraintFunctionsMapping,
    processed_constraints: ProcessedConstraintsMapping,
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    solution_reachability: PhaseReachability,
    flat_param_names: frozenset[QualifiedName],
    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]],
    state_action_space: StateActionSpace,
    has_taste_shocks: bool,
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
    period_to_regime_grid_signature: (
        MappingProxyType[int, MappingProxyType[RegimeName, Hashable]] | None
    ) = None,
    period_to_user_regimes: (
        MappingProxyType[int, Mapping[RegimeName, UserRegime]] | None
    ) = None,
    age_values: FloatND | IntND,
    query_model: DCEGMQueryModel | None = None,
    dcegm_query_models: Mapping[RegimeName, DCEGMQueryModel] | None = None,
    nbegm_query_models: Mapping[RegimeName, NBEGMQueryModel] | None = None,
    projected_specs: Mapping[
        RegimeName, ProjectedContinuationSpec | ProjectedProducerDate
    ] = MappingProxyType({}),
    periods: tuple[int, ...] | None = None,
    case_u_source: DatedCaseUStepAuthority | None = None,
    resolved_euler_grid: ResolvedDatedEulerGrid | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> EGMStepBuild:
    """Build per-period DC-EGM kernels and the regime's carry template.

    Periods sharing the same continuation-target configuration *and* the same
    continuation-grid signature share one kernel (and hence one compiled
    program), mirroring the per-period grouping of the Q-and-F builders. The
    period-`t` kernel interpolates its targets' `V_{t+1}` on the targets'
    period-`t+1` grids, so an age-specialized target splits the grouping.

    Args:
        solver: The regime's DC-EGM solver configuration.
        regime_name: Name of the regime the kernels solve.
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances (the carry targets' resources functions are read from
            here).
        laws: Each regime's law, bound from `Model(edges=...)`, by regime name.
        functions: The regime's processed functions (params renamed to
            qualified names).
        certainty_equivalent: Actual source declaration, retained by identity
            for its private economic continuation reader.
        constraints: Immutable mapping of the regime's constraint names to
            constraint functions (discrete-only after DC-EGM validation).
        transitions: Immutable mapping of target regime names to their state
            transition functions.
        transition_plans: Immutable mapping of target regime names to their
            transition
            function names.
        compute_regime_transition_probs: Regime transition probability
            function for solve.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.
        solution_reachability: Static solution graph shared by every solver.
        flat_param_names: Frozenset of flat parameter names for the regime.
        regime_to_flat_param_names: Immutable mapping of every regime name to
            its flat parameter names. A carry target's resources / transition
            functions read the target regime's params, so the validator admits
            and the kernel binds the union of the source and its reachable
            carry targets' params.
        state_action_space: The regime's state-action space (source of the
            discrete-action grids and their canonical order).
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.
        query_model: Canonical source evaluator and authority pool, when installed.
        period_to_regime_v_interp: Immutable mapping of period to that period's
            V-interpolation info per regime, or `None` for an age-invariant
            model. A period-`t` kernel reads its targets' entries at `t + 1`.
        period_to_regime_grid_signature: Immutable mapping of period to each
            regime's user-declared age-specialization signature, or `None` for
            an age-invariant model. Folded into the kernel-sharing group key.
        period_to_user_regimes: Immutable mapping of period to every regime's
            declaration with its age markers resolved at that period, or `None`
            for an age-invariant model. A period-`t` kernel reads its carry
            targets' resources maps from their declarations at `t + 1`.

    Returns:
        The per-period kernel mapping, the regime's all-finite carry template
        (leading axes: discrete states, then passive states, then discrete
        actions), the regime's carry-target names — the only carry keys any of
        its kernels read, used to filter the rolling carry mapping the solve
        loop hands each kernel — and the names of the axes leading every carry
        and policy row.

    """
    n_pad = compute_egm_carry_length(solver=solver, materializer=materializer)
    n_savings_nodes = int(
        (
            solver.savings_grid.to_jax()
            if materializer is None
            else materializer.materialize_grid(spec=solver.savings_grid)
        ).shape[0]
    )
    own_v_info = regime_to_v_interpolation_info[regime_name]
    # Any savings-stage Euler-state read (the Euler law's residual, regime
    # transition probabilities, stochastic transition weights, non-Euler
    # laws) switches the kernel to the per-exogenous-asset-node solve; the
    # single-post-state kernel has no defined Euler value at the savings
    # stage, so dispatching it would be silently wrong.
    asset_row_mode = savings_stage_reads_euler_state(
        user_regime=user_regimes[regime_name], law=laws[regime_name], solver=solver
    )
    # The persisted carry length is split from the envelope-workspace length
    # `n_pad`: in asset-row mode the stored row holds one published point per
    # exogenous Euler node and the dense workspace is transient (FUES/RFC runs
    # per node and publishes a single point), so the carry needs only
    # `n_euler_nodes` rows. The padding to `n_pad` it would otherwise carry is
    # pure dead storage (`interp_on_padded_grid` masks the NaN tail). In the
    # single-post-state kernel the carry *is* the refined envelope, so its
    # length stays `n_pad`.
    n_carry_rows = n_pad
    n_euler_nodes = int(
        materialize_step_euler_grid(
            spec=own_v_info.continuous_states[solver.continuous_state],
            resolved=resolved_euler_grid,
            materializer=materializer,
        ).shape[0]
    )
    if asset_row_mode:
        n_pad = max(n_pad, n_euler_nodes)
        n_carry_rows = n_euler_nodes
    own_discrete_state_names = _get_discrete_state_names(
        v_interpolation_info=own_v_info
    )
    own_passive_state_names = _get_passive_state_names(
        v_interpolation_info=own_v_info,
        euler_state_name=solver.continuous_state,
    )
    own_discrete_action_values = MappingProxyType(
        dict(state_action_space.discrete_actions)
    )
    # Process states whose distribution params arrive at runtime have their
    # grids resolved per solve; thread those resolved grids into the
    # continuation so a process-reading resources function integrates over the
    # solve-time nodes rather than the build-time NaN placeholder.
    own_runtime_process_names = tuple(
        name
        for name in _get_process_state_names(v_interpolation_info=own_v_info)
        if _is_runtime_process(own_v_info.discrete_states[name])
    )
    # Canonical position of the Euler axis in the published V array: after
    # the discrete-state axes, at its slot within the continuous-state order.
    euler_axis_in_V = len(own_discrete_state_names) + tuple(
        own_v_info.continuous_states
    ).index(solver.continuous_state)
    leading_shape = (
        tuple(
            int(
                (
                    own_v_info.discrete_states[name].to_jax()
                    if materializer is None
                    else materializer.materialize_grid(
                        spec=own_v_info.discrete_states[name]
                    )
                ).shape[0]
            )
            for name in own_discrete_state_names
        )
        + tuple(
            int(
                (
                    own_v_info.continuous_states[name].to_jax()
                    if materializer is None
                    else materializer.materialize_grid(
                        spec=own_v_info.continuous_states[name]
                    )
                ).shape[0]
            )
            for name in own_passive_state_names
        )
        + tuple(int(v.shape[0]) for v in own_discrete_action_values.values())
    )
    n_cell_axes = len(own_discrete_state_names) + len(own_passive_state_names)
    if materializer is None:
        carry_template = build_template_egm_carry(
            n_rows=n_carry_rows,
            leading_shape=leading_shape,
            query_data=(
                None
                if query_model is None
                else build_template_egm_query_data(
                    state_shape=leading_shape[:n_cell_axes],
                    action_shape=leading_shape[n_cell_axes:],
                    row_capacity=n_carry_rows,
                    interval_capacity=len(query_model.authority.interval_ids),
                    liquid_corner=asset_row_mode,
                )
            ),
        )
    else:
        query_data = (
            None
            if query_model is None
            else prepare_egm_query_data(
                state_shape=leading_shape[:n_cell_axes],
                action_shape=leading_shape[n_cell_axes:],
                row_capacity=n_carry_rows,
                interval_capacity=len(query_model.authority.interval_ids),
                liquid_corner=asset_row_mode,
                materializer=materializer,
            )
        )
        carry_template = prepare_egm_carry(
            n_rows=n_carry_rows,
            leading_shape=leading_shape,
            query_data=query_data,
            materializer=materializer,
        )

    configs: dict[_EGMGroupKey, list[int]] = {}
    stateful_targets_union = frozenset(
        solution_reachability.union_targets(source=regime_name)
    )
    active_periods = tuple(
        period
        for period, active_regimes in enumerate(
            solution_reachability.active_regimes_by_period
        )
        if regime_name in active_regimes
    )
    if periods is not None:
        if len(set(periods)) != len(periods) or not set(periods) <= set(active_periods):
            raise TypeError("Prepared EGM dates must be unique actual active dates.")
        active_periods = tuple(period for period in active_periods if period in periods)
    for period in active_periods:
        targets = (
            ()
            if period == solution_reachability.n_periods - 1
            else solution_reachability.targets(period=period, source=regime_name)
        )
        carry, scalar = get_egm_continuation_targets(
            targets=targets,
            regime_to_v_interpolation_info=continuation_v_interpolation_info(
                period=period,
                regime_to_v_interpolation_info=regime_to_v_interpolation_info,
                period_to_regime_v_interp=period_to_regime_v_interp,
            ),
        )
        configs.setdefault(
            (
                carry,
                scalar,
                (
                    continuation_grid_signature(
                        period=period,
                        targets=carry + scalar,
                        period_to_regime_grid_signature=period_to_regime_grid_signature,
                    ),
                    # An `AgeSpecializedFunction` is a different function at each
                    # age, so periods may share a program only where the user's
                    # declared signature says the closures agree.
                    periodized_tree_signature(tree=functions, period=period),
                    periodized_tree_signature(tree=constraints, period=period),
                    tuple(
                        (
                            target,
                            periodized_tree_signature(
                                tree=model.functions, period=period + 1
                            ),
                            periodized_tree_signature(
                                tree={"koopmans_aggregator": model.koopmans_aggregator},
                                period=period + 1,
                            ),
                        )
                        for target in carry
                        if (
                            model := (
                                (dcegm_query_models or {}).get(target)
                                or (nbegm_query_models or {}).get(target)
                            )
                        )
                        is not None
                    )
                    + tuple(
                        ("projected_pre_shock", target, period + 1)
                        for target in carry
                        if target in projected_specs
                    ),
                ),
            ),
            [],
        ).append(period)

    built: dict[_EGMGroupKey, EGMStepFunction[_DCEGMStepOutput]] = {}
    node_axes_by_group: dict[_EGMGroupKey, tuple[tuple[StateName, int], ...]] = {}
    for group_key, group_periods in configs.items():
        stateful_targets, scalar_targets, _ = group_key
        # Every period in the group shares a continuation-grid signature and the
        # signatures of its periodized functions and constraints, so the first
        # one's resolved economics is the whole group's.
        representative_period = group_periods[0]
        group_v_interp = continuation_v_interpolation_info(
            period=representative_period,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            period_to_regime_v_interp=period_to_regime_v_interp,
        )
        group_functions = cast(
            "EconFunctionsMapping",
            resolve_periodized_nodes(mapping=functions, period=representative_period),
        )
        group_constraints = cast(
            "ConstraintFunctionsMapping",
            resolve_periodized_nodes(mapping=constraints, period=representative_period),
        )
        unsupported = _find_unsupported_feature(
            case_u_source=case_u_source,
            source_periods=tuple(group_periods),
            projected_specs=projected_specs,
            solver=solver,
            regime_name=regime_name,
            user_regimes=user_regimes,
            laws=laws,
            functions=group_functions,
            constraints=group_constraints,
            processed_constraints=processed_constraints,
            stateful_targets=stateful_targets,
            transitions=transitions,
            transition_plans=transition_plans,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=group_v_interp,
            flat_param_names=flat_param_names,
            regime_to_flat_param_names=regime_to_flat_param_names,
            own_discrete_state_names=own_discrete_state_names,
            own_passive_state_names=own_passive_state_names,
            own_discrete_action_names=tuple(own_discrete_action_values),
            asset_row_mode=asset_row_mode,
            post_inversion_predicate_names=(
                frozenset(
                    target
                    for purpose, target in query_model.authority.evaluator_targets
                    if purpose == f"predicate:{target}"
                    and target in group_constraints
                    and target in query_model.functions
                    and solver.continuous_state
                    in dict(query_model.authority.evaluator_inputs)[purpose]
                )
                if isinstance(query_model, DCEGMQueryModel)
                and isinstance(solver.envelope, ExactEnvelope)
                else frozenset()
            ),
        )
        if unsupported is not None:
            raise ModelInitializationError(unsupported)
        # A target's resources map is the child's own at `t + 1`: its age
        # markers resolve there, and the group key holds the targets'
        # signatures at `t + 1`, so the representative period stands for all.
        child_user_regimes = (
            user_regimes
            if period_to_user_regimes is None
            else period_to_user_regimes.get(representative_period + 1, user_regimes)
        )
        kernel = _get_egm_step(
            materializer=materializer,
            solver=solver,
            user_regimes=child_user_regimes,
            laws=laws,
            functions=group_functions,
            koopmans_aggregator=koopmans_aggregator,
            certainty_equivalent=certainty_equivalent,
            constraints=group_constraints,
            transitions=transitions,
            transition_plans=transition_plans,
            compute_regime_transition_probs=compute_regime_transition_probs,
            stateful_targets=stateful_targets,
            scalar_targets=scalar_targets,
            n_pad=n_pad,
            n_carry_rows=n_carry_rows,
            own_discrete_state_names=own_discrete_state_names,
            own_passive_state_names=own_passive_state_names,
            own_discrete_action_values=own_discrete_action_values,
            own_runtime_process_names=own_runtime_process_names,
            euler_axis_in_V=euler_axis_in_V,
            has_taste_shocks=has_taste_shocks,
            regime_to_v_interpolation_info=group_v_interp,
            asset_row_mode=asset_row_mode,
            age_values=age_values,
            query_model=query_model,
            dcegm_query_models=dcegm_query_models,
            nbegm_query_models=nbegm_query_models,
            projected_specs=projected_specs,
            source_period=representative_period,
        )
        built[group_key] = _canonical_case_u_step(
            kernel=kernel, solver=solver, authority=case_u_source
        )
        # Each group owns a distinct numerical core and its child mesh. A
        # target active at another age may carry the same state name on a
        # different domain; that does not change this group's expectation.
        node_axes_by_group[group_key] = stochastic_node_axes(
            plan=cast("_EGMStep", kernel).pieces.continuation_plan
        )

    result: dict[int, EGMStepFunction[_DCEGMStepOutput]] = {}
    group_keys: dict[int, _EGMGroupKey] = {}
    for key, grouped_periods in configs.items():
        for period in grouped_periods:
            result[period] = built[key]
            group_keys[period] = key

    return EGMStepBuild(
        native_case_u_sources=MappingProxyType(
            {} if case_u_source is None else dict.fromkeys(result, case_u_source)
        ),
        steps=MappingProxyType(dict(sorted(result.items()))),
        step_group_keys=MappingProxyType(dict(sorted(group_keys.items()))),
        carry_template=carry_template,
        stateful_targets=stateful_targets_union,
        row_discrete_state_names=own_discrete_state_names,
        row_passive_state_names=own_passive_state_names,
        row_discrete_action_names=tuple(own_discrete_action_values),
        stochastic_node_axes_by_period=MappingProxyType(
            {period: node_axes_by_group[key] for period, key in group_keys.items()}
        ),
        cell_extent=math.prod(leading_shape[:n_cell_axes]),
        savings_point_extent=n_savings_nodes,
        euler_point_extent=n_euler_nodes if asset_row_mode else 0,
        envelope_cell_extent=(
            solver.n_constrained_points + n_savings_nodes - 1
            if isinstance(solver.envelope, ExactEnvelope)
            else 0
        ),
    )


def _canonical_case_u_step(
    *,
    kernel: EGMStepFunction[_DCEGMStepOutput],
    solver: _BoundDCEGM,
    authority: DatedCaseUStepAuthority | None,
) -> EGMStepFunction[_DCEGMStepOutput]:
    """Mint U's publication pieces before exposing its original dated step."""
    if authority is None:
        return kernel
    if not isinstance(kernel, _EGMStep):
        raise TypeError("Native U source lost its original Euler step.")
    return replace(
        kernel,
        pieces=replace(
            kernel.pieces,
            refine_to_bracket=get_bracket_finder(
                solver=solver,
                n_refined=kernel.pieces.n_pad,
                publish_provenance=True,
            ),
            check_query_construction=True,
            collect_continuation_read_counts=True,
        ),
    )


type _EGMGroupKey = tuple[tuple[RegimeName, ...], tuple[RegimeName, ...], Hashable]


def compute_egm_carry_length(
    *, solver: DCEGM, materializer: DatedGenerationMaterializer | None = None
) -> int:
    """Static carry-row length for a DC-EGM regime.

    Covers the Euler candidates (one per savings node), the closed-form
    constrained candidates, and the headroom factor for envelope-kink
    insertions (each costs two slots).

    Args:
        solver: The regime's DC-EGM solver configuration.

    Returns:
        Number of slots in the regime's carry rows.

    """
    n_savings = int(
        (
            solver.savings_grid.to_jax()
            if materializer is None
            else materializer.materialize_grid(spec=solver.savings_grid)
        ).shape[0]
    )
    n_candidates = n_savings + solver.n_constrained_points
    return math.ceil(solver.refined_grid_factor * n_candidates)


def _get_egm_step(
    *,
    solver: _BoundDCEGM,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None = None,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    compute_regime_transition_probs: RegimeTransitionFunction,
    stateful_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    n_pad: int,
    n_carry_rows: int,
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_values: MappingProxyType[ActionName, DiscreteAction],
    own_runtime_process_names: tuple[StateName, ...],
    euler_axis_in_V: int,
    has_taste_shocks: bool,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    asset_row_mode: bool,
    age_values: FloatND | IntND,
    query_model: DCEGMQueryModel | None = None,
    dcegm_query_models: Mapping[RegimeName, DCEGMQueryModel] | None = None,
    nbegm_query_models: Mapping[RegimeName, NBEGMQueryModel] | None = None,
    projected_specs: Mapping[
        RegimeName, ProjectedContinuationSpec | ProjectedProducerDate
    ] = MappingProxyType({}),
    source_period: int | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> EGMStepFunction[_DCEGMStepOutput]:
    """Build the EGM kernel for one continuation-target configuration.

    `asset_row_mode` selects the per-combo computation at build time: the
    per-exogenous-asset-node solve when any savings-stage function reads the
    current Euler state, the single-post-state default otherwise. Only the
    asset-row kernel runs a per-node loop, so only it reads the `euler_point`
    tile width.
    """
    get_solve_one_combo = (
        _get_solve_one_combo_asset_rows if asset_row_mode else _get_solve_one_combo
    )
    pieces = _build_kernel_pieces(
        materializer=materializer,
        solver=solver,
        user_regimes=user_regimes,
        laws=laws,
        functions=functions,
        koopmans_aggregator=koopmans_aggregator,
        certainty_equivalent=certainty_equivalent,
        constraints=constraints,
        transitions=transitions,
        transition_plans=transition_plans,
        compute_regime_transition_probs=compute_regime_transition_probs,
        stateful_targets=stateful_targets,
        scalar_targets=scalar_targets,
        n_pad=n_pad,
        n_carry_rows=n_carry_rows,
        own_discrete_state_names=own_discrete_state_names,
        own_passive_state_names=own_passive_state_names,
        own_discrete_action_values=own_discrete_action_values,
        euler_axis_in_V=euler_axis_in_V,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        age_values=age_values,
        query_model=query_model,
        dcegm_query_models=dcegm_query_models,
        nbegm_query_models=nbegm_query_models,
        projected_specs=projected_specs,
        source_period=source_period,
        asset_row_mode=asset_row_mode,
    )

    return _EGMStep(
        pieces=pieces,
        get_solve_one_combo=get_solve_one_combo,
        has_taste_shocks=has_taste_shocks,
        own_discrete_state_names=own_discrete_state_names,
        own_passive_state_names=own_passive_state_names,
        own_discrete_action_values=own_discrete_action_values,
        own_runtime_process_names=own_runtime_process_names,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _EGMStep:
    """The DC-EGM kernel of one regime for one continuation-target configuration.

    Runs the per-combo solve over the regime's combo product and publishes V
    on the exogenous grid, the carry its parents interpolate, and the
    simulation policy. The continuation is read from the carries alone; no
    next-period value array enters the step.
    """

    pieces: _EgmKernelPieces
    """Build-time statics shared by every per-combo computation."""

    get_solve_one_combo: Callable[
        ...,
        Callable[
            [tuple[ScalarInt | ScalarFloat, ...]],
            _ComboRows | BoundedResult[_ConditionalComboRows],
        ],
    ]
    """Builder of the per-combo solve (single-post-state or asset-row)."""

    has_taste_shocks: bool
    """Whether the regime smooths its discrete actions with EV1 taste shocks."""

    own_discrete_state_names: tuple[StateName, ...]
    """The regime's discrete states in carry-axis order."""

    own_passive_state_names: tuple[StateName, ...]
    """The regime's passive continuous states in carry-axis order."""

    own_discrete_action_values: MappingProxyType[ActionName, DiscreteAction]
    """The regime's discrete actions and their grid values."""

    own_runtime_process_names: tuple[StateName, ...]
    """Process states whose grids are resolved at runtime."""

    def __call__(
        self,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        _lcm_stochastic_node_width: int | None = None,
        _lcm_cell_width: int | None = None,
        _lcm_savings_point_width: int | None = None,
        _lcm_euler_point_width: int | None = None,
        _lcm_envelope_cell_width: int = 1,
        _lcm_prepared_nested_floor: PreparedNestedFloor | None = None,
        **kwargs: EconFunctionArg,
    ) -> _DCEGMStepOutput:
        """Run the DC-EGM step and publish V on the exogenous grid.

        Args:
            next_regime_to_continuation: The next period's EGM carries.
            _lcm_stochastic_node_width: Block width the execution plan folds
                the child stochastic-node expectation at; `None` folds the
                whole node mesh in one block.
            _lcm_cell_width: Tile width the plan runs the output state-cell
                loop at; `None` runs the whole product in one tile.
            _lcm_savings_point_width: Tile width the plan runs the
                per-savings-node loop at; `None` runs it in one tile.
            _lcm_euler_point_width: Tile width the plan runs the asset-row
                per-node loop at; `None` runs it in one tile.
            _lcm_envelope_cell_width: Number of exact-envelope node cells the
                plan resolves together; defaults to a serial scan.
            **kwargs: The regime's state grids, flat params, `period`, and
                `age`.

        Returns:
            Tuple of the value-function array on the exogenous state grid
            (discrete-state axes leading, continuous states in canonical
            order), the regime's carry (one row per discrete-state x
            passive-node x discrete-action combo), and the simulation policy.

        """
        pieces = self.pieces
        if _lcm_prepared_nested_floor is not None:
            pieces = replace(
                pieces,
                _prepared_nested_floor=_lcm_prepared_nested_floor,
                _prepared_nested_axes=tuple(
                    (name, jnp.asarray(kwargs[name]))
                    for name in self.own_discrete_state_names
                    + self.own_passive_state_names
                ),
            )
        dtype = canonical_float_dtype()
        own_state_names = {
            pieces.euler_state_name,
            *self.own_discrete_state_names,
            *self.own_passive_state_names,
        }
        pool = {k: v for k, v in kwargs.items() if k not in own_state_names}
        state_grid = jnp.asarray(kwargs[pieces.euler_state_name], dtype=dtype)
        own_taste_shock_scale = (
            jnp.asarray(kwargs[TASTE_SHOCK_SCALE_PARAM], dtype=dtype)
            if self.has_taste_shocks
            else jnp.asarray(0.0, dtype=dtype)
        )
        resolved_process_grids = MappingProxyType(
            {
                name: jnp.asarray(kwargs[name], dtype=dtype)
                for name in self.own_runtime_process_names
            }
        )
        solve_one_combo = self.get_solve_one_combo(
            pieces=pieces,
            pool=pool,
            state_grid=state_grid,
            next_regime_to_continuation=next_regime_to_continuation,
            euler_point_width=_lcm_euler_point_width,
            savings_point_width=_lcm_savings_point_width,
            envelope_cell_width=_lcm_envelope_cell_width,
            stochastic_node_width=_lcm_stochastic_node_width,
            resolved_process_grids=resolved_process_grids,
        )

        if pieces.combo_names:
            combo_var_names = (
                self.own_discrete_state_names
                + self.own_passive_state_names
                + tuple(self.own_discrete_action_values)
            )
            solve_one_combo_over_axes = with_signature(args=list(combo_var_names))(
                _SolveOneComboOverAxes(
                    solve_one_combo=solve_one_combo, combo_var_names=combo_var_names
                )
            )

            # Map the per-combo solve over the Cartesian product of the combo
            # axes. The state part of that product is the tiled `cell` axis;
            # actions stay fused, because the discrete-action aggregation needs
            # every action's value at once. Outputs come back with the combo
            # axes in `combo_var_names` order, preserving whole discrete axes
            # for the carry.
            combo_axis_values = {
                **{
                    name: jnp.asarray(kwargs[name])
                    for name in self.own_discrete_state_names
                    + self.own_passive_state_names
                },
                **{
                    name: jnp.asarray(values)
                    for name, values in self.own_discrete_action_values.items()
                },
            }
            mapped = _map_combo_product(
                func=solve_one_combo_over_axes,
                combo_var_names=combo_var_names,
                combo_axis_values=combo_axis_values,
                state_names=self.own_discrete_state_names
                + self.own_passive_state_names,
                cell_width=_lcm_cell_width,
                bound_summary=(
                    pieces.query_spec is not None
                    or pieces.check_query_construction
                    or bool(pieces.continuation_plan.query_targets)
                    or pieces.collect_continuation_read_counts
                    or pieces._retain_nested_feasibility  # noqa: SLF001
                ),
            )
            if isinstance(mapped, BoundedResult):
                return _publish_conditional_product(
                    rows=mapped,
                    pieces=pieces,
                    taste_shock_scale=own_taste_shock_scale,
                    has_taste_shocks=self.has_taste_shocks,
                    discrete_state_names=self.own_discrete_state_names,
                    passive_state_names=self.own_passive_state_names,
                    discrete_action_names=tuple(self.own_discrete_action_values),
                )
            (
                V_stack,
                grid_stack,
                policy_stack,
                value_stack,
                marginal_stack,
                _,
            ) = mapped
            n_state_axes = len(self.own_discrete_state_names) + len(
                self.own_passive_state_names
            )
            action_axes = tuple(range(n_state_axes, len(combo_var_names)))
            if action_axes and self.has_taste_shocks:
                V_arr, _ = logsum_and_softmax(
                    values=V_stack, scale=own_taste_shock_scale, axes=action_axes
                )
            elif action_axes:
                # Without taste shocks the smoothed maximum is the hard maximum
                # over the discrete-action axes (the logsum is for `scale > 0`).
                V_arr = jnp.max(V_stack, axis=action_axes)
            else:
                V_arr = V_stack
            # The combo layout puts the Euler axis last; the canonical V
            # layout interleaves it with the passive axes in V state order.
            V_arr = jnp.moveaxis(V_arr, -1, pieces.euler_axis_in_V)
            carry = EGMCarry(
                endog_grid=grid_stack,
                value=value_stack,
                marginal_utility=marginal_stack,
                taste_shock_scale=own_taste_shock_scale,
            )
            sim_policy = EGMSimPolicy(
                endog_grid=grid_stack,
                policy=policy_stack,
                value=value_stack,
                marginal_utility=marginal_stack,
                row_discrete_state_names=self.own_discrete_state_names,
                row_passive_state_names=self.own_passive_state_names,
                row_discrete_action_names=tuple(self.own_discrete_action_values),
            )
        else:
            combo = solve_one_combo(())
            if isinstance(combo, BoundedResult):
                return _publish_conditional_combo(
                    combo=combo,
                    taste_shock_scale=own_taste_shock_scale,
                    retain_nested_feasibility=pieces._retain_nested_feasibility,  # noqa: SLF001
                )
            V_arr, grid_row, policy_row, value_row, marginal_row, _ = combo
            carry = EGMCarry(
                endog_grid=grid_row,
                value=value_row,
                marginal_utility=marginal_row,
                taste_shock_scale=own_taste_shock_scale,
            )
            sim_policy = EGMSimPolicy(
                endog_grid=grid_row,
                policy=policy_row,
                value=value_row,
                marginal_utility=marginal_row,
            )
        return V_arr, carry, sim_policy


def _publish_conditional_product(
    *,
    rows: BoundedResult[_MappedConditionalRows],
    pieces: _EgmKernelPieces,
    taste_shock_scale: ScalarFloat,
    has_taste_shocks: bool,
    discrete_state_names: tuple[StateName, ...],
    passive_state_names: tuple[StateName, ...],
    discrete_action_names: tuple[ActionName, ...],
) -> (
    BoundedResult[tuple[FloatND, EGMCarry, EGMSimPolicy, FeasibilityCounts | None]]
    | BoundedResult[_NestedDCEGMStepChannels]
):
    """Publish conditional choice rows and reduce feasibility on state nodes."""
    V, grid, policy, value, marginal, _, data, feasible = rows.output
    n_state_axes = len(discrete_state_names) + len(passive_state_names)
    action_axes = tuple(range(n_state_axes, len(pieces.combo_names)))
    if action_axes and has_taste_shocks:
        V_state, _ = logsum_and_softmax(
            values=V, scale=taste_shock_scale, axes=action_axes
        )
    elif action_axes:
        V_state = jnp.max(V, axis=action_axes)
    else:
        V_state = V
    if data is not None and action_axes:
        # The installed floor excludes actions; retain its one fixed-cell leaf.
        data = replace(
            data,
            floor_used=data.floor_used[
                (slice(None),) * n_state_axes + (0,) * len(action_axes)
            ],
        )
    carry = EGMCarry(
        endog_grid=grid,
        value=value,
        marginal_utility=marginal,
        taste_shock_scale=taste_shock_scale,
        query_data=data,
    )
    sim_policy = EGMSimPolicy(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        row_discrete_state_names=discrete_state_names,
        row_passive_state_names=passive_state_names,
        row_discrete_action_names=discrete_action_names,
    )
    if pieces._retain_nested_feasibility:  # noqa: SLF001
        node_feasible = jnp.moveaxis(feasible, -1, n_state_axes).reshape(
            *V_state.shape, -1
        )
        counts = (
            count_current_feasibility(
                budget_feasible=node_feasible,
                predicate_feasible=jnp.ones((*node_feasible.shape, 0), dtype=jnp.bool_),
                value=V_state,
                active_states=jnp.ones(V_state.shape, dtype=jnp.bool_),
            )
            if rows.bound_status or rows.query_status is not None
            else None
        )
        return BoundedResult(
            output=(
                jnp.moveaxis(V_state, -1, pieces.euler_axis_in_V),
                carry,
                sim_policy,
                counts,
                jnp.moveaxis(node_feasible, -2, pieces.euler_axis_in_V),
            ),
            bound_status=rows.bound_status,
            query_status=rows.query_status,
            continuation_read_counts=rows.continuation_read_counts,
        )
    if not rows.bound_status and rows.query_status is None:
        return BoundedResult(
            output=(
                jnp.moveaxis(V_state, -1, pieces.euler_axis_in_V),
                carry,
                sim_policy,
                None,
            ),
            bound_status=(),
            continuation_read_counts=rows.continuation_read_counts,
        )
    node_feasible = jnp.moveaxis(feasible, -1, n_state_axes).reshape(*V_state.shape, -1)
    counts = count_current_feasibility(
        budget_feasible=node_feasible,
        predicate_feasible=jnp.ones((*node_feasible.shape, 0), dtype=jnp.bool_),
        value=V_state,
        active_states=jnp.ones(V_state.shape, dtype=jnp.bool_),
    )
    return BoundedResult(
        output=(
            jnp.moveaxis(V_state, -1, pieces.euler_axis_in_V),
            carry,
            sim_policy,
            counts,
        ),
        bound_status=rows.bound_status,
        query_status=rows.query_status,
        continuation_read_counts=rows.continuation_read_counts,
    )


def _publish_conditional_combo(
    *,
    combo: BoundedResult[_ConditionalComboRows],
    taste_shock_scale: ScalarFloat,
    retain_nested_feasibility: bool = False,
) -> (
    BoundedResult[tuple[FloatND, EGMCarry, EGMSimPolicy, FeasibilityCounts | None]]
    | BoundedResult[_NestedDCEGMStepChannels]
):
    """Publish one conditional row with independent counts and bounded integrity."""
    value, grid, policy, row_value, marginal, _, data, feasible = combo.output
    carry = EGMCarry(
        endog_grid=grid,
        value=row_value,
        marginal_utility=marginal,
        taste_shock_scale=taste_shock_scale,
        query_data=data,
    )
    sim_policy = EGMSimPolicy(
        endog_grid=grid,
        policy=policy,
        value=row_value,
        marginal_utility=marginal,
    )
    if retain_nested_feasibility:
        counts = (
            count_current_feasibility(
                budget_feasible=feasible[:, None],
                predicate_feasible=jnp.ones((*feasible.shape, 1, 0), dtype=jnp.bool_),
                value=value,
                active_states=jnp.ones(feasible.shape, dtype=jnp.bool_),
            )
            if combo.bound_status or combo.query_status is not None
            else None
        )
        return BoundedResult(
            output=(value, carry, sim_policy, counts, feasible[:, None]),
            bound_status=combo.bound_status,
            query_status=combo.query_status,
            continuation_read_counts=combo.continuation_read_counts,
        )
    if not combo.bound_status and combo.query_status is None:
        return BoundedResult(
            output=(value, carry, sim_policy, None),
            bound_status=(),
            continuation_read_counts=combo.continuation_read_counts,
        )
    counts = count_current_feasibility(
        budget_feasible=feasible[:, None],
        predicate_feasible=jnp.ones((*feasible.shape, 1, 0), dtype=jnp.bool_),
        value=value,
        active_states=jnp.ones(feasible.shape, dtype=jnp.bool_),
    )
    return BoundedResult(
        output=(value, carry, sim_policy, counts),
        bound_status=combo.bound_status,
        query_status=combo.query_status,
        continuation_read_counts=combo.continuation_read_counts,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _SolveOneComboOverAxes:
    """The per-combo solve, keyed by combo variable name for `productmap`."""

    solve_one_combo: Callable[
        [tuple[ScalarInt | ScalarFloat, ...]],
        _ComboRows | BoundedResult[_ConditionalComboRows],
    ]
    """The positional per-combo solve."""

    combo_var_names: tuple[StateOrActionName, ...]
    """The combo variables in the order the solve takes their values."""

    def __call__(
        self, **combo_values: ScalarFloat | ScalarInt
    ) -> _ComboRows | BoundedResult[_ConditionalComboRows]:
        return self.solve_one_combo(
            tuple(combo_values[name] for name in self.combo_var_names)
        )


def _map_combo_product(
    *,
    func: Callable[..., _ComboRows | BoundedResult[_ConditionalComboRows]],
    combo_var_names: tuple[StateOrActionName, ...],
    combo_axis_values: dict[StateOrActionName, FloatND | IntND],
    state_names: tuple[StateName, ...],
    cell_width: int | None,
    bound_summary: bool = False,
) -> tuple[FloatND | BoolND, ...] | BoundedResult[_MappedConditionalRows]:
    """Map the per-combo solve over the Cartesian product of the combo axes.

    `func` has a `combo_var_names` keyword signature and returns one tuple of
    per-combo outputs (1-D float arrays plus the scalar read-support flag).
    `state_names` are the combo axes forming the tiled `cell` axis — the output
    state cells — and everything else is vmapped. Returns the stacked outputs
    with the combo axes as leading dims in `combo_var_names` order (the
    canonical carry layout).

    A `cell_width` covering the whole state product is plain `productmap` with
    every axis vmapped. A narrower one runs a *single* `lax.map` (one scan
    carry) over the flattened state product, vmapping the action axes within
    each step, then transposes the result back into `combo_var_names` order.
    The two schedules name the same result: tiles are concatenated, never
    folded.
    """
    extent = math.prod(int(combo_axis_values[name].shape[0]) for name in state_names)
    block = tile_block_size(width=cell_width, extent=extent)
    if not block:
        return productmap(
            func=func,
            variables=combo_var_names,
            batch_sizes=dict.fromkeys(combo_var_names, 0),
            bound_summary=bound_summary,
        )(**combo_axis_values)

    splayed = tuple(name for name in combo_var_names if name in state_names)
    vmapped = tuple(name for name in combo_var_names if name not in state_names)
    inner = productmap(
        func=func,
        variables=vmapped,
        batch_sizes=dict.fromkeys(vmapped, 0),
        bound_summary=bound_summary,
    )
    vmapped_values = {name: combo_axis_values[name] for name in vmapped}
    splayed_dims = tuple(int(combo_axis_values[name].shape[0]) for name in splayed)
    mesh = jnp.meshgrid(*(combo_axis_values[name] for name in splayed), indexing="ij")
    flat_splayed = tuple(grid.ravel() for grid in mesh)

    at_one = functools.partial(
        _at_one_splayed_combo,
        inner=inner,
        splayed=splayed,
        vmapped_values=vmapped_values,
    )
    if bound_summary:
        stacked = map_over_leading_axis(
            func=at_one,
            xs=flat_splayed,
            batch_size=block,
            bound_summary=True,
        )
    else:
        stacked = jax.lax.map(at_one, flat_splayed, batch_size=block)

    # Restore `combo_var_names` order: `stacked` axes are
    # (flattened splayed, *vmapped, *per-element); reshape the flat axis back
    # to the splayed dims, then transpose the combo axes into canonical order.
    current_order = splayed + vmapped
    perm = tuple(current_order.index(name) for name in combo_var_names)
    n_combo = len(combo_var_names)
    if isinstance(stacked, BoundedResult):
        return BoundedResult(
            output=jax.tree.map(
                functools.partial(
                    _reorder_combo_axes,
                    splayed_dims=splayed_dims,
                    perm=perm,
                    n_combo=n_combo,
                ),
                stacked.output,
            ),
            bound_status=stacked.bound_status,
            query_status=stacked.query_status,
            continuation_read_counts=stacked.continuation_read_counts,
        )
    return tuple(
        _reorder_combo_axes(
            arr=arr, splayed_dims=splayed_dims, perm=perm, n_combo=n_combo
        )
        for arr in stacked
    )


# keyword-only-exempt: library-callback=jax.lax.map
def _at_one_splayed_combo(
    splayed_values: tuple[ScalarFloat | ScalarInt, ...],
    *,
    inner: Callable[
        ..., tuple[FloatND | BoolND, ...] | BoundedResult[_MappedConditionalRows]
    ],
    splayed: tuple[StateOrActionName, ...],
    vmapped_values: dict[StateOrActionName, FloatND | IntND],
) -> tuple[FloatND | BoolND, ...] | BoundedResult[_MappedConditionalRows]:
    """Run the vmapped inner product at one flattened splayed combo."""
    return inner(**dict(zip(splayed, splayed_values, strict=True)), **vmapped_values)


# keyword-only-exempt: library-callback=jax.tree.map
def _reorder_combo_axes(
    arr: FloatND | BoolND | IntND,
    *,
    splayed_dims: tuple[int, ...],
    perm: tuple[int, ...],
    n_combo: int,
) -> FloatND | BoolND | IntND:
    """Unflatten the splayed axis and transpose the combo axes into canonical order."""
    arr = arr.reshape(*splayed_dims, *arr.shape[1:])
    trailing = tuple(range(n_combo, arr.ndim))
    return jnp.transpose(arr, (*perm, *trailing))


def _build_kernel_pieces(
    *,
    solver: _BoundDCEGM,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None = None,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    compute_regime_transition_probs: RegimeTransitionFunction,
    stateful_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    n_pad: int,
    n_carry_rows: int,
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_values: MappingProxyType[ActionName, DiscreteAction],
    euler_axis_in_V: int,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    age_values: FloatND | IntND,
    query_model: DCEGMQueryModel | None = None,
    dcegm_query_models: Mapping[RegimeName, DCEGMQueryModel] | None = None,
    nbegm_query_models: Mapping[RegimeName, NBEGMQueryModel] | None = None,
    projected_specs: Mapping[
        RegimeName, ProjectedContinuationSpec | ProjectedProducerDate
    ] = MappingProxyType({}),
    source_period: int | None = None,
    asset_row_mode: bool = False,
    materializer: DatedGenerationMaterializer | None = None,
) -> _EgmKernelPieces:
    """Assemble the build-time statics of the EGM kernel."""
    savings_nodes = (
        jnp.asarray(solver.savings_grid.to_jax(), dtype=canonical_float_dtype())
        if materializer is None
        else materializer(
            value=materializer.materialize_grid(spec=solver.savings_grid),
            dtype=np.dtype(canonical_float_dtype()),
            name="egm_savings_nodes",
        )
    )
    n_constrained = solver.n_constrained_points
    continuation_plan = build_continuation_plan(
        materializer=materializer,
        user_regimes=user_regimes,
        laws=laws,
        functions=functions,
        transitions=transitions,
        transition_plans=transition_plans,
        stateful_targets=stateful_targets,
        scalar_targets=scalar_targets,
        compute_regime_transition_probs=compute_regime_transition_probs,
        post_decision_name=solver.post_decision_function,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        age_values=age_values,
        certainty_equivalent=certainty_equivalent,
        dcegm_query_models=dcegm_query_models,
        nbegm_query_models=nbegm_query_models,
        projected_specs=projected_specs,
        source_period=source_period,
    )
    return _EgmKernelPieces(
        euler_state_name=solver.continuous_state,
        action_name=solver.continuous_action,
        savings_nodes=savings_nodes,
        borrowing_limit=(
            savings_nodes[0]
            if materializer is None
            else materializer.evaluate_validation(
                function=_validation_grid_first, arguments={"points": savings_nodes}
            )
        ),
        n_constrained=n_constrained,
        # Static geometric ratio: the constrained actions run from
        # `span * CONSTRAINED_OFFSET_FRACTION` up to `span`, so the ratio
        # depends only on the offset fraction and the point count.
        constrained_ratio=(1.0 / CONSTRAINED_OFFSET_FRACTION)
        ** (1.0 / max(n_constrained - 1, 1)),
        n_pad=n_pad,
        n_carry_rows=n_carry_rows,
        combo_names=own_discrete_state_names
        + own_passive_state_names
        + tuple(own_discrete_action_values),
        euler_axis_in_V=euler_axis_in_V,
        utility_func=concatenate_regime_function(functions=functions, target="utility"),
        inverse_marginal_utility_func=(
            concatenate_regime_function(
                functions=functions, target="inverse_marginal_utility"
            )
            if "inverse_marginal_utility" in functions
            else None
        ),
        own_resources_func=concatenate_regime_function(
            functions=functions, target=solver.resources
        ),
        feasibility_func=_build_feasibility_function(
            functions=functions, constraints=constraints
        ),
        build_W_kwargs=_get_build_W_kwargs(
            functions=functions, koopmans_aggregator=koopmans_aggregator
        ),
        refine=(
            get_upper_envelope(solver=solver, n_refined=n_pad, publish_provenance=True)
            if query_model is not None
            else get_upper_envelope(solver=solver, n_refined=n_pad)
        ),
        refine_to_bracket=(
            get_bracket_finder(solver=solver, n_refined=n_pad, publish_provenance=True)
            if query_model is not None
            else get_bracket_finder(solver=solver, n_refined=n_pad)
        ),
        continuation_plan=continuation_plan,
        query_spec=None if query_model is None else query_model.authority,
        _retain_nested_feasibility=solver._retain_nested_feasibility,  # noqa: SLF001
        _negm_current_publication=(
            solver._negm_current_publication  # noqa: SLF001
            and isinstance(solver.envelope, ExactEnvelope)
            and not asset_row_mode
        ),
        lower_func=(
            concatenate_regime_function(
                functions=functions, target=solver.post_decision_bound.lower_target
            )
            if solver.post_decision_bound is not None
            and solver.post_decision_bound.kind == "dag"
            and solver.post_decision_bound.lower_target is not None
            else None
        ),
    )


def _build_feasibility_function(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
) -> Callable[..., ScalarBool] | None:
    """Build current feasibility, or `None` when no predicate remains.

    The route binds either a discrete combo or a published current-state node.
    Continuous actions are unavailable at both predicate sites.

    Returns:
        Callable mapping the bound current-state pool to a scalar feasibility
        indicator, or `None` without constraints.

    """
    if not constraints:
        return None
    return _ComboFeasibility(
        constraints_func=concatenate_functions(
            functions={
                **dict(functions),
                **dict(constraints),
            },
            targets=list(constraints),
            return_type="dict",
            enforce_signature=False,
            set_annotations=True,
        )
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _ComboFeasibility:
    """Evaluate and conjoin current constraints at the route's bound state."""

    constraints_func: Callable[..., Mapping[str, BoolND]]
    """The concatenated constraint DAG, returning every constraint by name."""

    def __call__(self, **combo_pool: EconFunctionArg) -> ScalarBool:
        outputs = self.constraints_func(**combo_pool)
        return jnp.all(jnp.stack([jnp.asarray(out) for out in outputs.values()]))
