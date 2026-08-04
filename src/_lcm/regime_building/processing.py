import functools
import inspect
import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
from dags import concatenate_functions, get_annotations, with_signature
from dags.signature import rename_arguments
from dags.tree import QNAME_DELIMITER, qname_from_tree_path, tree_path_from_qname
from jax import numpy as jnp

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.coarse_transition import _CoarseTransitionCell
from _lcm.egm.budget import (
    DCEGM_BUDGET_CONSTRAINT_NAME,
    get_intrinsic_budget_constraint,
)
from _lcm.egm.carry import EGMCarry, build_template_egm_carry, shard_carry_template
from _lcm.egm.negm_validation import validate_negm_regimes
from _lcm.egm.nnbegm_validation import validate_nnbegm_regimes
from _lcm.egm.terminal import (
    N_STATELESS_CARRY_ROWS,
    get_brute_child_carry_producer,
    get_stateless_terminal_carry_producer,
    get_terminal_wealth_carry_producer,
)
from _lcm.egm.validation import (
    savings_stage_reads_euler_state,
    validate_dcegm_regimes,
)
from _lcm.engine import (
    EGMPolicyRead,
    Regime,
    SimulationPhase,
    SolutionPhase,
    StateActionSpace,
    Variables,
)
from _lcm.grids import (
    DiscreteGrid,
    Grid,
)
from _lcm.grids.coordinates import get_irreg_coordinate
from _lcm.identity_transition import _IdentityTransition
from _lcm.params.processing import get_flat_param_names
from _lcm.params.regime_template import create_regime_params_template
from _lcm.processes import _ContinuousStochasticProcess, _IIDProcess
from _lcm.processes.ar1 import TauchenAR1Process
from _lcm.processes.iid import NormalIIDProcess
from _lcm.processes.state_conditioned import (
    Family,
    StateConditioned,
    conditioned_row,
    gather_sigma,
    sigma_array_by_code,
)
from _lcm.reachability import (
    ModelReachability,
    PhaseName,
    PhaseReachability,
    build_model_reachability,
    candidate_targets_from_transition,
)
from _lcm.regime_building.age_normalization import (
    AgeGridSchedule,
    PeriodizedEconFunction,
    PeriodizedUserFunction,
    assert_continuation_grids_agree,
    continuation_group_key,
    continuation_info_lookup,
    expand_groups_to_periods,
    group_periods_by_key,
    normalize_age_specialization,
    periodized_tree_signature,
    resolve_periodized_nodes,
)
from _lcm.regime_building.canonicalize import canonicalize_phased_regimes
from _lcm.regime_building.diagnostics import _build_compute_intermediates_per_period
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.gated_edges import (
    ResolvedEdgeLeg,
    ResolvedGatedEdge,
    build_fallback_state_projector,
    get_edge_fold,
    get_edge_simulate_gate_evaluator,
)
from _lcm.regime_building.max_Q_over_a import (
    get_argmax_and_max_Q_over_a,
)
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.next_state import get_next_state_function_for_simulation
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
    normalize_all_regime_phases,
)
from _lcm.regime_building.Q_and_F import (
    LAW_SOURCE_ATTR,
    ResolvedSamePeriodRef,
    _get_deterministic_transitions,
    get_Q_and_F,
    get_Q_and_F_collective,
    get_Q_and_F_terminal,
    get_Q_and_F_terminal_collective,
    partition_continuation_targets,
)
from _lcm.regime_building.stochastic_state_transitions import (
    collect_stochastic_state_transitions,
)
from _lcm.regime_building.V import (
    VInterpolationInfo,
    create_v_interpolation_info,
)
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SolverBuildContext,
)
from _lcm.state_action_space import create_state_action_space
from _lcm.typing import (
    ArgmaxQOverAFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    EGMCarryProducer,
    FlatParams,
    FunctionName,
    MappingLeaf,
    NextStateSimulationFunction,
    ProcessName,
    QAndFFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
    SequenceLeaf,
    StateName,
    StateOrActionName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
    VmappedRegimeTransitionFunction,
)
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.dispatchers import simulation_spacemap, vmap_1d
from _lcm.utils.error_messages import format_messages
from _lcm.utils.functools import get_union_of_args
from _lcm.utils.namespace import flatten_regime_namespace, unflatten_regime_namespace
from _lcm.variables import (
    from_regime,
    get_grids,
    simulate_variables_from_regime,
)
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, NEGM, Solver
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, UserFunction

type _TransitionBundles = dict[
    RegimeName, dict[TransitionFunctionName, UserFunction | _CoarseTransitionCell]
]


def compute_active_periods_by_regime(
    *,
    ages: AgeGrid,
    user_regimes: Mapping[RegimeName, object],
) -> MappingProxyType[RegimeName, tuple[int, ...]]:
    """Evaluate every regime's `active` predicate exactly once.

    The single canonical activity schedule for the model: every other
    subsystem that needs to know which periods a regime is active in
    (reachability, age specialization, broadcast pruning, model-input
    validation) consumes this mapping instead of re-evaluating
    `Regime.active` or calling `AgeGrid.get_periods_where` itself.
    """
    return MappingProxyType(
        {
            regime_name: tuple(ages.get_periods_where(regime.active))  # ty: ignore[unresolved-attribute]
            for regime_name, regime in user_regimes.items()
        }
    )


@dataclass(frozen=True, kw_only=True)
class PreparedModelStructure:
    """Construction-time declarations shared by validation and compilation."""

    representative_user_regimes: MappingProxyType[RegimeName, FinalizedUserRegime]
    """Age-normalized user regimes used for phase compilation."""

    phased_specs: MappingProxyType[RegimeName, PhasedRegimeSpec]
    """Age-normalized phase declarations."""

    grid_schedule: AgeGridSchedule | None
    """Concrete period grids for age-specialized states."""

    reachability: ModelReachability
    """Static solution and simulation regime graphs."""

    active_periods_by_regime: MappingProxyType[RegimeName, tuple[int, ...]]
    """Periods in which each regime is locally active."""


def prepare_model_structure(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    ages: AgeGrid,
    active_periods_by_regime: MappingProxyType[RegimeName, tuple[int, ...]],
) -> PreparedModelStructure:
    """Prepare normalized declarations and static phase graphs once.

    `active_periods_by_regime` must be the single canonical activity
    mapping from `compute_active_periods_by_regime`, computed once by the
    caller — this function does not evaluate `Regime.active` itself.
    """
    raw_phase_specs = normalize_all_regime_phases(user_regimes=user_regimes)
    age_normalization = normalize_age_specialization(
        user_regimes=user_regimes,
        phased_specs=raw_phase_specs,
        ages=ages,
        active_periods_by_regime=active_periods_by_regime,
    )
    phased_specs = age_normalization.phased_specs
    transitions_by_phase: Mapping[PhaseName, Mapping[RegimeName, object]] = {
        "solution": {
            regime_name: spec.solution.regime_transition
            for regime_name, spec in phased_specs.items()
        },
        "simulation": {
            regime_name: spec.simulation.regime_transition
            for regime_name, spec in phased_specs.items()
        },
    }
    try:
        reachability = build_model_reachability(
            n_periods=ages.n_periods,
            active_periods_by_regime=active_periods_by_regime,
            transitions_by_phase=transitions_by_phase,
            terminal_regimes={
                regime_name
                for regime_name, regime in user_regimes.items()
                if regime.terminal
            },
        )
    except ValueError as error:
        raise ModelInitializationError(str(error)) from error
    return PreparedModelStructure(
        representative_user_regimes=age_normalization.representative_user_regimes,
        phased_specs=phased_specs,
        grid_schedule=age_normalization.grid_schedule,
        reachability=reachability,
        active_periods_by_regime=active_periods_by_regime,
    )


def process_regimes(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    prepared_structure: PreparedModelStructure,
) -> MappingProxyType[RegimeName, Regime]:
    """Process finalized regimes into canonical regimes.

    Normalizes phases, then age specialization (the single model-level boundary
    that resolves every `AgeSpecializedFunction` / `AgeSpecializedGrid` into
    concrete build-time objects), then canonicalizes every regime's laws into
    target-granular form, then compiles the per-phase function sets. Stochastic
    process transitions are generated from the grid's intrinsic transition logic.

    Args:
        user_regimes: Mapping of regime names to finalized regimes.
        ages: The AgeGrid for the model.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        enable_jit: Whether to jit the functions of the canonical regime.
        prepared_structure: Normalized declarations and static phase graphs,
            from `prepare_model_structure`. The caller builds this once and
            shares it with every other consumer that needs the same
            reachability graph, instead of each recomputing its own copy.

    Returns:
        The processed canonical regimes.

    """
    representative_user_regimes = prepared_structure.representative_user_regimes
    phased_specs = prepared_structure.phased_specs
    grid_schedule = prepared_structure.grid_schedule
    reachability = prepared_structure.reachability
    regimes_to_active_periods = prepared_structure.active_periods_by_regime
    all_regime_names = frozenset(user_regimes)
    state_handoff_errors = _state_handoff_errors(
        phase_name="solution",
        phase_reachability=reachability.solution,
        specs=phased_specs,
        ages=ages,
    )
    state_handoff_errors += _state_handoff_errors(
        phase_name="simulation",
        phase_reachability=reachability.simulation,
        specs=phased_specs,
        ages=ages,
    )
    if state_handoff_errors:
        raise ModelInitializationError(format_messages(state_handoff_errors))

    # DC-EGM regimes must satisfy the EGM model contract before any kernel is built.
    # `Model.__init__` validates earlier (so contract violations beat the generic
    # unused-variable check); this call covers direct `process_regimes` callers.
    # All three read the representative regimes: the contract is about a state's kind
    # and shape, both invariant across ages, and a raw `AgeSpecializedGrid` marker is
    # not a `Grid`, so a type-filtered collection of continuous states would drop it.
    validate_dcegm_regimes(
        user_regimes=representative_user_regimes,
        solution_reachability=reachability.solution,
    )
    validate_negm_regimes(user_regimes=representative_user_regimes)
    validate_nnbegm_regimes(user_regimes=representative_user_regimes)

    # Per-period continuation interpolation info, built from the schedule's cached
    # concrete grids (never an age factory). `None` for an age-invariant model.
    period_to_regime_v_interp = _build_period_v_interpolation_info(
        representative_user_regimes=representative_user_regimes,
        grid_schedule=grid_schedule,
    )

    # The canonical specs hold every law in target-granular form, resolved per
    # phase: the simulate slice additionally holds every carried-only state
    # and its law of motion, so the canonical mapping carries the law toward
    # each retained target that carries the state — including targets reached
    # through nothing but the carried state.
    specs = canonicalize_phased_regimes(
        raw_specs=phased_specs,
        all_regime_names=all_regime_names,
        solution_reachability=reachability.solution,
        simulation_reachability=reachability.simulation,
    )
    solve_nested_transitions = {
        regime_name: _extract_phase_transitions(phase_slice=spec.solution)
        for regime_name, spec in specs.items()
    }
    simulate_nested_transitions = {
        regime_name: _extract_phase_transitions(phase_slice=spec.simulation)
        for regime_name, spec in specs.items()
    }
    _fail_if_collective_regime_targets_unsupported(
        user_regimes=user_regimes,
        nested_transitions_by_regime=solve_nested_transitions,
    )
    _validate_categoricals(representative_user_regimes)

    regime_to_variables = MappingProxyType(
        {
            regime_name: from_regime(user_regime)
            for regime_name, user_regime in representative_user_regimes.items()
        }
    )
    all_grids = MappingProxyType(
        {
            regime_name: get_grids(user_regime)
            for regime_name, user_regime in representative_user_regimes.items()
        }
    )

    _fail_if_action_has_batch_size(user_regimes)

    regime_to_v_interpolation_info = MappingProxyType(
        {
            regime_name: create_v_interpolation_info(user_regime)
            for regime_name, user_regime in representative_user_regimes.items()
        }
    )
    state_action_spaces = MappingProxyType(
        {
            regime_name: create_state_action_space(
                variables=regime_to_variables[regime_name],
                grids=all_grids[regime_name],
            )
            for regime_name in user_regimes
        }
    )

    # fold-review F2 (folded slice): regimes whose EVERY state is a folded IID
    # process. Their stored V has all state axes integrated out
    # (`_get_regime_V_shapes_and_shardings`), so V is a scalar (per any trailing
    # stakeholder axis). Such a regime reached ONLY via a regime transition has
    # no ordinary state law and no source-carried process edge, so its
    # transition bundle would be empty and it would be silently dropped from
    # the continuation graph / E[V]. The solve phase keeps it enumerable with an
    # explicit empty bundle and reads its scalar continuation (see
    # `_process_regime_core` and `_build_solution_phase`).
    fold_only_regimes = frozenset(
        regime_name
        for regime_name in user_regimes
        if (state_names := state_action_spaces[regime_name].state_names)
        and _fold_state_names(state_names=state_names, grids=all_grids[regime_name])
        == state_names
    )

    # COLLECTIVE-REGIMES (E2): same-period reference declarations are a
    # cross-regime contract — validate it (existence, stakeholder layout,
    # projection coverage, co-activity) and reject reference cycles before any
    # kernel is built.
    _fail_if_same_period_refs_invalid(
        user_regimes=user_regimes,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        regimes_to_active_periods=regimes_to_active_periods,
    )
    _fail_if_same_period_ref_cycle(user_regimes=user_regimes)

    # COLLECTIVE-REGIMES (E3'): gated-edge declarations are a cross-regime
    # contract — validate endpoints, stakeholder layout, and projection coverage
    # before any kernel is built.
    _fail_if_gated_edges_invalid(
        user_regimes=user_regimes,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        regimes_to_active_periods=regimes_to_active_periods,
    )

    # COLLECTIVE-REGIMES (fold / E2 / E3' interaction, completed guard):
    # `fold=True` on a regime that another regime's gate reads nodewise (as a
    # gated-edge target, a leg fallback, or a same-period reference) is
    # rejected here, regardless of stakeholder cardinality — see the function
    # docstring.
    _fail_if_folded_regime_is_same_period_endpoint(user_regimes=user_regimes)

    model_has_egm_regime = any(
        user_regime.solver.requires_continuation
        for user_regime in user_regimes.values()
    )

    # Each regime's flat param names in the engine's binding vocabulary, keyed
    # by regime. A DC-EGM source regime that carries into a *different* target
    # regime evaluates that target's resources / transition functions in its
    # per-asset-node solve, so it must know the target's param leaves (e.g. a
    # pension factor the source itself never reads); the kernel binds them from
    # the union of the source and its reachable carry targets' fixed params.
    regime_to_params_template = MappingProxyType(
        {
            # The representative regime already carries first-active concrete
            # functions and representative-age grids, so the template is built
            # from it and needs no age argument of its own.
            regime_name: create_regime_params_template(user_regime)
            for regime_name, user_regime in representative_user_regimes.items()
        }
    )
    regime_to_granular_param_expansions = MappingProxyType(
        {
            regime_name: _granular_param_expansions(
                nested_transitions_by_phase=(
                    solve_nested_transitions[regime_name],
                    simulate_nested_transitions[regime_name],
                ),
                regime_params_template=regime_to_params_template[regime_name],
                declaration_param_expansions=_declaration_param_expansions(
                    source_regime_name=regime_name,
                    specs=phased_specs,
                    all_regime_names=all_regime_names,
                    regime_params_template=regime_to_params_template[regime_name],
                ),
            )
            for regime_name in user_regimes
        }
    )
    regime_to_flat_param_names = MappingProxyType(
        {
            regime_name: _engine_flat_param_names(
                regime_params_template=regime_to_params_template[regime_name],
                granular_param_expansions=regime_to_granular_param_expansions[
                    regime_name
                ],
            )
            for regime_name in user_regimes
        }
    )

    canonical_regimes: dict[RegimeName, Regime] = {}
    # Iterate the representative-resolved regimes: identical to the user regimes
    # except that any `AgeSpecializedGrid` state is a concrete representative-age
    # grid, so every grid-derived call below is age-invariant.
    for regime_name, user_regime in representative_user_regimes.items():
        spec = specs[regime_name]
        regime_params_template = regime_to_params_template[regime_name]
        granular_param_expansions = regime_to_granular_param_expansions[regime_name]

        # COLLECTIVE-REGIMES (E1): resolve the household Pareto weights once (equal
        # weights when unspecified) and thread the stakeholder names / weights into
        # both phase builds. `None` for the singleton default, so the existing path
        # is byte-identical.
        stakeholders = user_regime.stakeholders
        weights = _resolve_stakeholder_weights(user_regime)
        # COLLECTIVE-REGIMES (E2): the (deduplicated, order-preserving) regimes
        # whose same-period V this regime reads; drives the within-period
        # topological solve order and the kernel's same-period V threading.
        same_period_ref_regimes = tuple(
            dict.fromkeys(ref.regime for ref in user_regime.same_period_refs.values())
        )
        # Folded IID-process states — collected purely from this regime's own
        # states and grids, so (unlike `co_map_state_names`) the caller can
        # compute it before `_build_solution_phase` builds `core.transitions`.
        fold_state_names = _fold_state_names(
            state_names=state_action_spaces[regime_name].state_names,
            grids=all_grids[regime_name],
        )

        solution = _build_solution_phase(
            spec=spec,
            regime_name=regime_name,
            # Representative, not raw: these reach `SolverBuildContext`, and a solver
            # reading a state grid off them wants a concrete `Grid`. Every such read
            # is of a shape trait (`batch_size`, the `ContinuousGrid` kind), which is
            # invariant across ages, so the representative grid answers it exactly.
            # Node *values*, which do vary by age, come from the period's own axes.
            user_regimes=representative_user_regimes,
            declared_regime_transition=phased_specs[
                regime_name
            ].solution.regime_transition,
            phase_reachability=reachability.solution,
            nested_transitions=solve_nested_transitions[regime_name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            granular_param_expansions=granular_param_expansions,
            regime_to_flat_param_names=regime_to_flat_param_names,
            regime_names_to_ids=regime_names_to_ids,
            variables=regime_to_variables[regime_name],
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            period_to_regime_v_interp=period_to_regime_v_interp,
            grid_schedule=grid_schedule,
            state_action_space=state_action_spaces[regime_name],
            ages=ages,
            enable_jit=enable_jit,
            certainty_equivalent=user_regime.certainty_equivalent,
            solver=user_regime.solver,
            model_has_egm_regime=model_has_egm_regime,
            has_taste_shocks=user_regime.taste_shocks is not None,
            stakeholders=stakeholders,
            weights=weights,
            same_period_ref_regimes=same_period_ref_regimes,
            edge_target_regimes=tuple(user_regime.gated_edges),
            fold_state_names=fold_state_names,
            fold_only_regimes=fold_only_regimes,
        )

        simulation = _build_simulation_phase(
            spec=spec,
            user_regime=user_regime,
            regime_name=regime_name,
            user_regimes=user_regimes,
            solution_reachability=reachability.solution,
            simulation_reachability=reachability.simulation,
            nested_transitions=simulate_nested_transitions[regime_name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            granular_param_expansions=granular_param_expansions,
            regime_names_to_ids=regime_names_to_ids,
            variables=regime_to_variables[regime_name],
            simulation_variables=simulate_variables_from_regime(user_regime),
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            period_to_regime_v_interp=period_to_regime_v_interp,
            grid_schedule=grid_schedule,
            state_action_space=state_action_spaces[regime_name],
            ages=ages,
            enable_jit=enable_jit,
            # Unresolved (marker-bearing) solve pool so the simulation continuation
            # resolves AgeSpecializedFunction per period age, not at the representative
            # age frozen into `solution.functions` (round-11 F1). Falls back to the
            # published pool when no specialization is present.
            solve_functions=solution.continuation_functions,
            solve_transitions=solution.transitions,
            solve_stochastic_transition_names=solution.stochastic_transition_names,
            solve_compute_regime_transition_probs=solution.compute_regime_transition_probs,
            has_taste_shocks=user_regime.taste_shocks is not None,
            solver=user_regime.solver,
            certainty_equivalent=user_regime.certainty_equivalent,
            stakeholders=stakeholders,
            weights=weights,
            fold_only_regimes=fold_only_regimes,
        )

        stochastic_state_transitions = collect_stochastic_state_transitions(
            user_regime=user_regime,
            user_regimes=representative_user_regimes,
        )

        canonical_regimes[regime_name] = Regime(
            name=regime_name,
            terminal=spec.terminal,
            active_periods=tuple(regimes_to_active_periods[regime_name]),
            regime_params_template=regime_params_template,
            solution=solution,
            simulation=simulation,
            stochastic_state_transitions=stochastic_state_transitions,
            granular_param_expansions=granular_param_expansions,
            has_taste_shocks=user_regime.taste_shocks is not None,
            certainty_equivalent=user_regime.certainty_equivalent,
            stakeholders=stakeholders,
            same_period_ref_regimes=same_period_ref_regimes,
            fold_state_names=fold_state_names,
        )

    _fail_if_folded_state_persists(canonical_regimes=canonical_regimes)

    # COLLECTIVE-REGIMES (E3'): build the gated-edge folds in a second pass, now
    # that every regime's grid and processed functions are known. Each edge's
    # fold lands on its TARGET regime's grid and reads the target's functions, so
    # it can only be built after all regimes exist.
    canonical_regimes = _attach_gated_edge_folds(
        canonical_regimes=canonical_regimes,
        user_regimes=user_regimes,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        enable_jit=enable_jit,
    )

    return ensure_containers_are_immutable(canonical_regimes)


def _attach_gated_edge_folds(
    *,
    canonical_regimes: dict[RegimeName, Regime],
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    enable_jit: bool,
) -> dict[RegimeName, Regime]:
    """Resolve and compile each source regime's gated-edge folds (E3').

    For every source regime declaring `gated_edges`, resolve each user
    `GatedEdge` to its engine form and build the ``(Wbar, gate)`` producer on the
    target regime's grid (reading the target's processed functions), plus one
    per-leg FALLBACK state projector and a gate interpolator (E4 simulate
    routing — see `build_fallback_state_projector`). The resolved edges,
    folds, and projectors are stored back on the source's canonical regime.
    """
    for source_name, user_regime in user_regimes.items():
        if not user_regime.gated_edges:
            continue
        resolved: dict[RegimeName, ResolvedGatedEdge] = {}
        folds: dict[RegimeName, Callable] = {}
        leg_projectors: dict[RegimeName, tuple[Callable, ...]] = {}
        simulate_gate_evaluators: dict[RegimeName, Callable] = {}
        for target_name, edge in user_regime.gated_edges.items():
            resolved_edge = _resolve_gated_edge(
                source_name=source_name,
                target_name=target_name,
                edge=edge,
                user_regimes=user_regimes,
            )
            resolved[target_name] = resolved_edge
            target_solution = canonical_regimes[target_name].solution
            target_deterministic_transitions, _ = _get_deterministic_transitions(
                transitions=target_solution.transitions,
                stochastic_transition_names=(
                    target_solution.stochastic_transition_names
                ),
            )
            fold = get_edge_fold(
                edge=resolved_edge,
                target_v_info=regime_to_v_interpolation_info[target_name],
                target_functions=target_solution.functions,
                target_deterministic_transitions=target_deterministic_transitions,
                reference_v_info=regime_to_v_interpolation_info,
                target_stakeholders=user_regimes[target_name].stakeholders,
            )
            folds[target_name] = jax.jit(fold) if enable_jit else fold
            # COLLECTIVE-REGIMES (E4, simulate F1 fix). The simulate-side
            # router needs its own gate re-evaluated at a REALIZED (candidate
            # target-state) point — no longer by interpolating the fold's
            # baked boolean `gate` array and thresholding it (which does not
            # commute with a nonlinear predicate), but by recomputing the
            # predicate from interpolated VALUE operands
            # (`get_edge_simulate_gate_evaluator`). The candidate target
            # state fed in at simulate is a genuine (possibly off-grid) VALUE
            # for every target state — including a non-folded process state
            # (`_ContinuousStochasticProcess`, classified `discrete_states`
            # for the Markov-chain solve path but never fed an exact on-grid
            # index here). Mirror `_build_same_period_ref_reader`'s
            # (`Q_and_F.py`) auto-select of `get_V_interpolator`'s
            # process-aware mode so that axis is linearly interpolated
            # instead of integer-looked-up; a target without a process state
            # is unaffected (byte-identical ordinary path).
            target_has_process_axis = any(
                isinstance(grid, _ContinuousStochasticProcess)
                for grid in regime_to_v_interpolation_info[
                    target_name
                ].discrete_states.values()
            )
            # Never wrapped in `jax.jit` here — mirrors the previous gate
            # interpolator (a plain `get_V_interpolator` product, likewise
            # unjitted), consumed only inside `route_gated_edges`'s own
            # `jax.vmap` over subjects.
            simulate_gate_evaluators[target_name] = get_edge_simulate_gate_evaluator(
                edge=resolved_edge,
                target_v_info=regime_to_v_interpolation_info[target_name],
                target_functions=target_solution.functions,
                target_deterministic_transitions=target_deterministic_transitions,
                reference_v_info=regime_to_v_interpolation_info,
                target_stakeholders=user_regimes[target_name].stakeholders,
                target_has_process_axis=target_has_process_axis,
            )
            leg_projectors[target_name] = tuple(
                build_fallback_state_projector(
                    ref=leg.fallback,
                    fallback_v_info=regime_to_v_interpolation_info[leg.fallback.regime],
                    target_regime_name=target_name,
                    target_state_names=regime_to_v_interpolation_info[
                        target_name
                    ].state_names,
                    target_functions=target_solution.functions,
                    target_deterministic_transitions=target_deterministic_transitions,
                )
                for leg in resolved_edge.legs
            )
        canonical_regimes[source_name] = dataclass_replace(
            canonical_regimes[source_name],
            gated_edges=MappingProxyType(resolved),
            gated_edge_folds=MappingProxyType(folds),
            gated_edge_leg_projectors=MappingProxyType(leg_projectors),
            gated_edge_simulate_gate_evaluators=MappingProxyType(
                simulate_gate_evaluators
            ),
        )
    return canonical_regimes


def _resolve_gated_edge(
    *,
    source_name: RegimeName,
    target_name: RegimeName,
    edge: object,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> ResolvedGatedEdge:
    """Resolve one user `GatedEdge` to its engine form (E3').

    Validated already by `_fail_if_gated_edges_invalid`, so every named regime
    and stakeholder exists. Legs are ordered by the SOURCE's stakeholder tuple
    (a singleton source has one leg with `source_stakeholder=None`); each leg's
    OPEN-branch target component and its fallback stakeholder become trailing-
    axis indices.
    """
    edge = cast("Any", edge)
    target_stakeholders = user_regimes[target_name].stakeholders
    source_stakeholders = user_regimes[source_name].stakeholders

    def _stakeholder_index(
        regime_name: RegimeName, stakeholder: str | None
    ) -> int | None:
        regime_stakeholders = user_regimes[regime_name].stakeholders
        if regime_stakeholders is None:
            return None
        return regime_stakeholders.index(cast("str", stakeholder))

    def _resolve_ref(ref: object) -> ResolvedSamePeriodRef:
        ref = cast("Any", ref)
        return ResolvedSamePeriodRef(
            regime=ref.regime,
            projection=ref.projection,
            stakeholder_index=_stakeholder_index(ref.regime, ref.stakeholder),
        )

    # A collective source's legs are iterated in its stakeholder order so
    # ``Wbar``'s trailing axis matches the source's stakeholder axis; a singleton
    # source has exactly one leg (keyed arbitrarily), stored with source
    # stakeholder `None`.
    if source_stakeholders is None:
        leg_order: list[tuple[str, str | None]] = [(next(iter(edge.legs)), None)]
    else:
        leg_order = [(s, s) for s in source_stakeholders]

    legs: list[ResolvedEdgeLeg] = []
    for leg_key, source_stakeholder in leg_order:
        leg = edge.legs[leg_key]
        legs.append(
            ResolvedEdgeLeg(
                source_stakeholder=source_stakeholder,
                target_component_index=(
                    None
                    if target_stakeholders is None
                    else target_stakeholders.index(cast("str", leg.target_stakeholder))
                ),
                fallback=_resolve_ref(leg.fallback),
            )
        )
    gate_refs = {name: _resolve_ref(ref) for name, ref in edge.gate_refs.items()}
    reference_regimes = tuple(
        dict.fromkeys(
            [leg.fallback.regime for leg in legs]
            + [ref.regime for ref in gate_refs.values()]
        )
    )
    return ResolvedGatedEdge(
        target=target_name,
        gate=edge.gate,
        gate_refs=MappingProxyType(gate_refs),
        legs=tuple(legs),
        reference_regimes=reference_regimes,
    )


def _resolve_stakeholder_weights(
    user_regime: UserRegime,
) -> MappingProxyType[str, float] | None:
    """Resolve a collective regime's household Pareto weights.

    COLLECTIVE-REGIMES (E1). Returns `None` for a singleton regime (the default,
    keeping the existing path untouched). For a collective regime, uses the
    user's explicit `weights` when given, else equal weights `1/len(stakeholders)`
    (validated to match the stakeholder names at regime construction).
    """
    stakeholders = user_regime.stakeholders
    if stakeholders is None:
        return None
    if user_regime.weights is not None:
        user_weights = user_regime.weights
        return MappingProxyType({s: float(user_weights[s]) for s in stakeholders})
    equal = 1.0 / len(stakeholders)
    return MappingProxyType(dict.fromkeys(stakeholders, equal))


def _fail_if_collective_regime_targets_unsupported(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    nested_transitions_by_regime: Mapping[RegimeName, _TransitionBundles],
) -> None:
    """Reject regime transitions mixing collective and mismatched stakeholders.

    COLLECTIVE-REGIMES (E1, slice 2). The E1 continuation reads each target's
    `next_V_arr` with the SOURCE regime's stakeholder layout: a collective
    source expects every target leaf to carry the identical trailing
    stakeholder axis, and a singleton source expects none. Per-stakeholder
    routing across stakeholder layouts (dissolution into `single_f`/`single_m`,
    marriage formation) is the gated-edge machinery (E3', slice 4), so any
    non-terminal regime whose reachable target declares a different
    `stakeholders` tuple is rejected here. Both-`None` (the singleton default)
    never enters the comparison, so today's path is untouched.

    Args:
        user_regimes: Mapping of regime names to finalized user regimes.
        nested_transitions_by_regime: Per-regime solve-phase transition
            bundles; their keys are the regime's reachable targets.

    Raises:
        NotImplementedError: If a non-terminal regime's reachable target
            declares a different `stakeholders` tuple.

    """
    for regime_name, user_regime in user_regimes.items():
        if user_regime.terminal:
            continue
        for target_regime_name in nested_transitions_by_regime.get(regime_name, {}):
            target_regime = user_regimes.get(target_regime_name)
            if target_regime is None:
                continue
            if user_regime.stakeholders is None and target_regime.stakeholders is None:
                continue
            # COLLECTIVE-REGIMES (E3'): a target reached through a DECLARED gated
            # edge is exempt — the edge folds a gated continuation object matching
            # the SOURCE's stakeholder layout, so the mixed-topology read is safe.
            if target_regime_name in user_regime.gated_edges:
                continue
            if user_regime.stakeholders != target_regime.stakeholders:
                msg = (
                    f"Regime '{regime_name}' (stakeholders="
                    f"{user_regime.stakeholders}) can reach regime "
                    f"'{target_regime_name}' (stakeholders="
                    f"{target_regime.stakeholders}), but every transition "
                    "target of a collective regime must be a collective regime "
                    "with the identical `stakeholders` tuple (and a singleton "
                    "regime cannot target a collective one). Per-stakeholder "
                    "routing to different regimes (e.g. dissolution into "
                    "single-person regimes) is the gated-edge machinery (E3', "
                    "slice 4) and is not yet implemented. See the design doc "
                    "`pylcm-extension-collective-regimes.md` (v2.1)."
                )
                raise NotImplementedError(msg)


def _resolve_same_period_refs(
    *,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> MappingProxyType[str, ResolvedSamePeriodRef]:
    """Resolve a regime's user `SamePeriodRef` declarations to the engine form.

    COLLECTIVE-REGIMES (E2). The declarations were validated by
    `_fail_if_same_period_refs_invalid`, so the reference regime exists and its
    stakeholder naming is consistent; here the named stakeholder becomes the
    index on the reference V's trailing stakeholder axis (`None` for a
    singleton reference).
    """
    resolved: dict[str, ResolvedSamePeriodRef] = {}
    for ref_name, ref in user_regime.same_period_refs.items():
        ref_stakeholders = user_regimes[ref.regime].stakeholders
        stakeholder_index = (
            None
            if ref_stakeholders is None
            else ref_stakeholders.index(cast("str", ref.stakeholder))
        )
        resolved[ref_name] = ResolvedSamePeriodRef(
            regime=ref.regime,
            projection=ref.projection,
            stakeholder_index=stakeholder_index,
        )
    return MappingProxyType(resolved)


def _fail_if_same_period_refs_invalid(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
) -> None:
    """Validate every `same_period_refs` declaration against the other regimes.

    COLLECTIVE-REGIMES (E2). Checks, per reference: the reference regime
    exists; a collective reference names one of its stakeholders while a
    singleton reference names none; the projection covers exactly the reference
    regime's solve states (the coordinates its V is interpolated over); and the
    reference regime is active in every period the declaring regime is active
    (its same-period V must exist whenever the reader solves).

    Raises:
        ModelInitializationError: On the first violated declaration, naming the
            regime, the reference, and the violated property.

    """
    for regime_name, user_regime in user_regimes.items():
        for ref_name, ref in user_regime.same_period_refs.items():
            prefix = (
                f"Regime '{regime_name}', same_period_refs['{ref_name}'] "
                f"(reference regime '{ref.regime}'): "
            )
            target_regime = user_regimes.get(ref.regime)
            if target_regime is None:
                msg = (
                    f"{prefix}the reference regime is not part of the model. "
                    f"Known regimes: {sorted(user_regimes)}."
                )
                raise ModelInitializationError(msg)
            if target_regime.stakeholders is None:
                if ref.stakeholder is not None:
                    msg = (
                        f"{prefix}names stakeholder '{ref.stakeholder}', but "
                        "the reference regime is a singleton — its V carries "
                        "no stakeholder axis. Drop the `stakeholder`."
                    )
                    raise ModelInitializationError(msg)
            elif ref.stakeholder is None:
                msg = (
                    f"{prefix}the reference regime is collective "
                    f"(stakeholders={target_regime.stakeholders}), so the "
                    "reference must name WHOSE value to read via "
                    "`stakeholder=...`."
                )
                raise ModelInitializationError(msg)
            elif ref.stakeholder not in target_regime.stakeholders:
                msg = (
                    f"{prefix}names stakeholder '{ref.stakeholder}', which is "
                    "not one of the reference regime's stakeholders "
                    f"{target_regime.stakeholders}."
                )
                raise ModelInitializationError(msg)
            expected_states = set(
                regime_to_v_interpolation_info[ref.regime].state_names
            )
            if set(ref.projection) != expected_states:
                msg = (
                    f"{prefix}the projection must supply exactly one coordinate "
                    "function per state of the reference regime "
                    f"({sorted(expected_states)}); got "
                    f"{sorted(ref.projection)}."
                )
                raise ModelInitializationError(msg)
            missing_periods = sorted(
                set(regimes_to_active_periods[regime_name])
                - set(regimes_to_active_periods[ref.regime])
            )
            if missing_periods:
                msg = (
                    f"{prefix}the reference regime must be active (and hence "
                    "solved) in every period the declaring regime is active, "
                    "but it is not active in period(s) "
                    f"{missing_periods}. A same-period reference V that was "
                    "never solved cannot be read."
                )
                raise ModelInitializationError(msg)


def _fail_if_gated_edges_invalid(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
) -> None:
    """Validate every `gated_edges` declaration against the other regimes (E3').

    COLLECTIVE-REGIMES (E3'). Checks, per edge: the target regime exists; each
    leg's OPEN-branch target component names a target stakeholder (or is `None`
    for a singleton target); each fallback and gate reference names an existing
    regime, with a stakeholder iff that regime is collective, and a projection
    covering exactly that regime's states; and every regime the edge reads
    (fallbacks + gate references) is active in every period its target is active
    (F4 co-activity: the edge's Wbar reads those same-period Vs whenever the
    target is solved, so a reference that was never solved cannot be read).

    Raises:
        ModelInitializationError: On the first violated declaration.
    """
    for regime_name, user_regime in user_regimes.items():
        for target_name, edge in user_regime.gated_edges.items():
            prefix = f"Regime '{regime_name}', gated_edges['{target_name}']: "
            target = user_regimes.get(target_name)
            if target is None:
                msg = (
                    f"{prefix}the target regime is not part of the model. "
                    f"Known regimes: {sorted(user_regimes)}."
                )
                raise ModelInitializationError(msg)
            for leg_key, leg in edge.legs.items():
                leg_prefix = f"{prefix}leg '{leg_key}': "
                _fail_if_target_stakeholder_invalid(
                    leg_prefix=leg_prefix,
                    target=target,
                    target_name=target_name,
                    target_stakeholder=leg.target_stakeholder,
                )
                _fail_if_ref_invalid(
                    prefix=f"{leg_prefix}fallback ",
                    ref=leg.fallback,
                    user_regimes=user_regimes,
                    regime_to_v_interpolation_info=regime_to_v_interpolation_info,
                )
            _fail_if_duplicate_fallback_regimes(prefix=prefix, edge=edge)
            for ref_name, ref in edge.gate_refs.items():
                _fail_if_ref_invalid(
                    prefix=f"{prefix}gate_refs['{ref_name}'] ",
                    ref=ref,
                    user_regimes=user_regimes,
                    regime_to_v_interpolation_info=regime_to_v_interpolation_info,
                )
            _fail_if_gated_edge_references_inactive(
                prefix=prefix,
                source_name=regime_name,
                target_name=target_name,
                edge=edge,
                regimes_to_active_periods=regimes_to_active_periods,
            )


def _fail_if_gated_edge_references_inactive(
    *,
    prefix: str,
    source_name: RegimeName,
    target_name: RegimeName,
    edge: object,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
) -> None:
    """Reject a gated edge whose fallback / gate reference is not co-active with
    its target on every period whose ``Wbar`` is actually consumed (E3', F4).

    COLLECTIVE-REGIMES (E3', F4). `_roll_gated_edges` folds the edge's ``Wbar``
    in each period ``t`` the TARGET regime is solved, reading each fallback's and
    gate reference's SAME-PERIOD V, and that ``Wbar`` is consumed by the SOURCE
    one period earlier (at ``t-1``). If a referenced regime is inactive in a
    period whose ``Wbar`` is consumed, its V does not exist and silently rolling
    the previous ``Wbar`` would feed the source a STALE later-period value.

    The consumed periods are exactly ``{t : t in active(target) and
    (t-1) in active(source)}`` — NOT every period the target is active. A period
    ``t`` where the target is active but the source is not active at ``t-1``
    (e.g. a self-loop edge at the target's earliest active period) produces a
    ``Wbar`` no source ever reads, so a reference may legitimately be inactive
    there; requiring co-activity across ALL target-active periods would wrongly
    reject such a model (see
    ``test_repeating_self_loop_gated_edge_simulates_past_activity_boundary``).
    Require every referenced regime to be active on every consumed period so the
    roll is well defined wherever its result is used.
    """
    edge = cast("Any", edge)
    reference_regime_names = {leg.fallback.regime for leg in edge.legs.values()} | {
        ref.regime for ref in edge.gate_refs.values()
    }
    source_periods = set(regimes_to_active_periods[source_name])
    consumed_periods = {
        t for t in regimes_to_active_periods[target_name] if (t - 1) in source_periods
    }
    for ref_regime_name in sorted(reference_regime_names):
        missing_periods = sorted(
            consumed_periods - set(regimes_to_active_periods[ref_regime_name])
        )
        if missing_periods:
            msg = (
                f"{prefix}reference regime '{ref_regime_name}' must be active "
                "(and hence solved) in every period the edge target "
                f"'{target_name}' is active AND the source '{source_name}' is "
                "active one period earlier (so the edge's Wbar is actually "
                f"consumed); but it is not active in such period(s) "
                f"{missing_periods}. A gated-edge reference V that was never "
                "solved cannot be read."
            )
            raise ModelInitializationError(msg)


def _fail_if_duplicate_fallback_regimes(*, prefix: str, edge: object) -> None:
    """Reject an edge whose legs share a fallback regime (F4).

    COLLECTIVE-REGIMES (E4, F4 guard). `route_gated_edges`
    (`_lcm.simulation.gated_routing`) writes EVERY leg's own projected
    fallback state into `leg.fallback.regime`'s per-subject state slot, one
    leg at a time, masked by `subjects_in_regime` — never keyed by leg. If
    two legs of the SAME edge name the same fallback regime (even with
    different projections, e.g. two stakeholders both falling back to a
    shared regime name), the second leg's write silently overwrites the
    first's for every subject in the source regime, regardless of which leg
    `_select_own_leg` would actually select for a given row — a genuine data
    corruption, not merely an unused branch. A singleton source (exactly one
    leg) can never trigger this; only a multi-leg (collective) source can.
    """
    edge = cast("Any", edge)
    fallback_regimes = [leg.fallback.regime for leg in edge.legs.values()]
    seen: set[RegimeName] = set()
    duplicates: list[RegimeName] = []
    for regime_name in fallback_regimes:
        if regime_name in seen and regime_name not in duplicates:
            duplicates.append(regime_name)
        seen.add(regime_name)
    if duplicates:
        msg = (
            f"{prefix}two or more legs share the same fallback regime "
            f"{sorted(duplicates)}. Forward simulation writes each leg's "
            "own projected fallback state into its fallback regime's "
            "per-subject slot; legs sharing a fallback regime would have "
            "one leg's write silently overwrite the other's for every "
            "subject, regardless of which leg is actually selected. Give "
            "each leg its own fallback regime."
        )
        raise ModelInitializationError(msg)


def _fail_if_target_stakeholder_invalid(
    *,
    leg_prefix: str,
    target: UserRegime,
    target_name: RegimeName,
    target_stakeholder: str | None,
) -> None:
    """Reject an edge leg whose OPEN-branch target component is inconsistent."""
    if target.stakeholders is None:
        if target_stakeholder is not None:
            msg = (
                f"{leg_prefix}names target_stakeholder "
                f"'{target_stakeholder}', but the target regime "
                f"'{target_name}' is a singleton — drop it."
            )
            raise ModelInitializationError(msg)
    elif target_stakeholder is None:
        msg = (
            f"{leg_prefix}the target regime '{target_name}' is collective "
            f"(stakeholders={target.stakeholders}); the leg must name which "
            "component the gate-open branch takes via `target_stakeholder=...`."
        )
        raise ModelInitializationError(msg)
    elif target_stakeholder not in target.stakeholders:
        msg = (
            f"{leg_prefix}names target_stakeholder '{target_stakeholder}', "
            f"which is not one of the target's stakeholders {target.stakeholders}."
        )
        raise ModelInitializationError(msg)


def _fail_if_ref_invalid(
    *,
    prefix: str,
    ref: object,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> None:
    """Reject an edge fallback / gate reference with an invalid endpoint (E3')."""
    ref = cast("Any", ref)
    reference = user_regimes.get(ref.regime)
    if reference is None:
        msg = (
            f"{prefix}reference regime '{ref.regime}' is not part of the model. "
            f"Known regimes: {sorted(user_regimes)}."
        )
        raise ModelInitializationError(msg)
    if reference.stakeholders is None:
        if ref.stakeholder is not None:
            msg = (
                f"{prefix}names stakeholder '{ref.stakeholder}', but the "
                f"reference regime '{ref.regime}' is a singleton. Drop it."
            )
            raise ModelInitializationError(msg)
    elif ref.stakeholder is None:
        msg = (
            f"{prefix}the reference regime '{ref.regime}' is collective "
            f"(stakeholders={reference.stakeholders}); name whose value to read "
            "via `stakeholder=...`."
        )
        raise ModelInitializationError(msg)
    elif ref.stakeholder not in reference.stakeholders:
        msg = (
            f"{prefix}names stakeholder '{ref.stakeholder}', not one of the "
            f"reference regime's stakeholders {reference.stakeholders}."
        )
        raise ModelInitializationError(msg)
    expected_states = set(regime_to_v_interpolation_info[ref.regime].state_names)
    if set(ref.projection) != expected_states:
        msg = (
            f"{prefix}the projection must supply exactly one coordinate function "
            f"per state of the reference regime ({sorted(expected_states)}); got "
            f"{sorted(ref.projection)}."
        )
        raise ModelInitializationError(msg)


def _fail_if_same_period_ref_cycle(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Reject cyclic `same_period_refs` declarations at model build.

    COLLECTIVE-REGIMES (E2). Within one period, a regime's value constraints
    read reference regimes solved EARLIER in that period, so the reference
    graph must be acyclic (a self-reference is a one-node cycle). Depth-first
    three-color search; the error names one offending cycle.
    """
    graph = {
        regime_name: tuple(
            dict.fromkeys(ref.regime for ref in user_regime.same_period_refs.values())
        )
        for regime_name, user_regime in user_regimes.items()
    }
    visiting: set[RegimeName] = set()
    done: set[RegimeName] = set()
    stack: list[RegimeName] = []

    def visit(node: RegimeName) -> None:
        if node in done or node not in graph:
            return
        if node in visiting:
            cycle = [*stack[stack.index(node) :], node]
            msg = (
                "`same_period_refs` declarations form a cycle: "
                f"{' -> '.join(cycle)}. Within a period, a reference regime "
                "must be solved before the regime that reads its value, so "
                "the reference graph must be acyclic."
            )
            raise ModelInitializationError(msg)
        visiting.add(node)
        stack.append(node)
        for successor in graph[node]:
            visit(successor)
        stack.pop()
        visiting.discard(node)
        done.add(node)

    for regime_name in graph:
        visit(regime_name)


def _fail_if_folded_regime_is_same_period_endpoint(
    *, user_regimes: Mapping[RegimeName, UserRegime]
) -> None:
    """Reject `fold=True` on a regime read nodewise by another gate/reference.

    COLLECTIVE-REGIMES (fold / E2 / E3' interaction — F1-F4 audit, completed
    guard). `fold=True` integrates a shock's node axis out of a regime's
    stored V — and, for a COLLECTIVE regime, additionally collapses its
    dissolution flag `D` by `jnp.any` — immediately after that regime's OWN
    period solve (`_wrap_with_fold_reduction` in
    `regime_building/max_Q_over_a.py`). E3' gated-edge routing and E2
    same-period reads both require gate-THEN-integrate: each realized shock
    node must be routed through its own gate / consent decision before any
    node is averaged away (`jnp.where(gate, V_target, V_fallback)` per node,
    then integrate — never integrate first and gate the average). If the
    folding regime is itself read nodewise by another regime's gate or
    same-period reference, the two orderings conflict: the reader only ever
    sees the already-averaged V (and, if collective, the already-`jnp.any`-
    reduced D), not the per-node values gate-then-integrate needs.
    Counterexample: target node values ``[-inf, 1]``, nodewise dissolution
    ``[True, False]``, fallback ``0``, equal weights — the correct
    route-then-average is ``0.5``, but fold-first (`D_any=True`) routes the
    whole cell to the fallback, ``0.0``.

    A regime is unsafe to fold when it is any of:

    - an inbound gated-edge TARGET — named as a key of another regime's
      `gated_edges`, so its V/D would be read nodewise by that edge's gate
      and the leg routing;
    - a gated-edge leg FALLBACK — named by the `regime` of a
      `gated_edges[...].legs[...].fallback` entry: `get_edge_fold` reads the
      fallback nodewise, `jnp.where(gate, V_target, V_fallback)`, at every
      target grid node, BEFORE any integration — the same ordering hazard as
      the edge target itself; or
    - a same-period REFERENCE — named by the `regime` of another regime's
      `same_period_refs` entry, or of a `gated_edges[...].gate_refs` entry —
      so its per-node V would be read through a projection.

    This applies regardless of stakeholder cardinality: gate-then-integrate
    is violated whether the folding regime is collective or a SINGLETON — a
    singleton has a node-valued V (just no D) that a gate/reference reads
    exactly the same way.

    This is the bounded INTERIM fix for the finding: reject the unsafe
    combination at construction rather than implement the full transient
    V_node/D_node split that would let a gate read the pre-fold, per-node
    values (a separate, larger slice). A folded regime that is none of the
    above — neither a gated-edge target, a leg fallback, nor a same-period
    reference — is unaffected and stays allowed — mirroring the cross-regime
    graph walks in `_fail_if_same_period_refs_invalid` /
    `_fail_if_gated_edges_invalid` above and `_fail_if_folded_state_persists`
    below, this runs once every regime's declarations are known, not in the
    regime-local `_validate_fold_declarations`.

    Raises:
        ModelInitializationError: Naming every offending regime, its folded
            state(s), and which role(s) (gated-edge target / leg fallback /
            same-period reference) make the fold unsafe.
    """
    gated_edge_targets: set[RegimeName] = {
        target_name
        for regime in user_regimes.values()
        for target_name in regime.gated_edges
    }
    # Build the reference set from each resolved edge's COMPLETE reference
    # set (fallbacks + gate_refs), rather than re-enumerating `edge.legs` /
    # `edge.gate_refs` by hand here — `_resolve_gated_edge` is the single
    # source of truth for "which regimes does this edge read nodewise", and
    # every named regime is already known to exist
    # (`_fail_if_gated_edges_invalid` runs before this).
    same_period_reference_regimes: set[RegimeName] = {
        ref.regime
        for regime in user_regimes.values()
        for ref in regime.same_period_refs.values()
    } | {
        reference_name
        for source_name, regime in user_regimes.items()
        for target_name, edge in regime.gated_edges.items()
        for reference_name in _resolve_gated_edge(
            source_name=source_name,
            target_name=target_name,
            edge=edge,
            user_regimes=user_regimes,
        ).reference_regimes
    }

    error_messages: list[str] = []
    for regime_name, regime in user_regimes.items():
        fold_names = sorted(
            name
            for name, grid in regime.states.items()
            if isinstance(grid, _IIDProcess) and grid.fold
        )
        if not fold_names:
            continue

        roles: list[str] = []
        if regime_name in gated_edge_targets:
            roles.append("the TARGET of another regime's `gated_edges`")
        if regime_name in same_period_reference_regimes:
            roles.append(
                "a same-period REFERENCE (named by another regime's "
                "`same_period_refs`, `gated_edges[...].gate_refs`, or "
                "`gated_edges[...].legs[...].fallback`)"
            )
        if not roles:
            continue

        d_clause = (
            " (and collapses its dissolution flag D by `jnp.any`)"
            if regime.stakeholders is not None
            else ""
        )
        cardinality = (
            f"a collective (stakeholders={regime.stakeholders}) regime"
            if regime.stakeholders is not None
            else "a singleton regime"
        )
        error_messages.append(
            f"Regime '{regime_name}' declares fold=True on state(s) "
            f"{fold_names} while being {cardinality} that is "
            f"{' and '.join(roles)}. Folding integrates the shock's node "
            f"axis out of this regime's stored V{d_clause} before that gate "
            "/ reference can route or read per node: E2/E3' require "
            "gate-then-integrate, not integrate-then-gate. Drop "
            "`fold=True` on this regime, or stop targeting / referencing it "
            "from a gated edge or same-period reference."
        )

    if error_messages:
        raise ModelInitializationError(format_messages(error_messages))


def _period_to_state_nodes(
    *, regime_name: RegimeName, grid_schedule: AgeGridSchedule | None
) -> MappingProxyType[int, MappingProxyType[StateName, Float1D]] | None:
    """This regime's age-specialized state nodes, per active period.

    `None` when the model has no age-specialized grid, or when this regime declares
    none, so an age-invariant regime hands its solver exactly what it handed before.
    """
    if grid_schedule is None:
        return None
    specialized = grid_schedule.specialized_states_by_regime.get(
        regime_name, frozenset()
    )
    if not specialized:
        return None
    return MappingProxyType(
        {
            period: MappingProxyType(
                {
                    state_name: by_regime[regime_name][state_name].nodes
                    for state_name in sorted(specialized)
                }
            )
            for period, by_regime in grid_schedule.by_period.items()
            if regime_name in by_regime
        }
    )


def _period_to_regime_grid_signature(
    *, grid_schedule: AgeGridSchedule | None
) -> MappingProxyType[int, MappingProxyType[RegimeName, Hashable]] | None:
    """Every regime's user-declared grid signature, per period.

    `None` for an age-invariant model, so its solvers group periods exactly as
    they did before.
    """
    if grid_schedule is None:
        return None
    return MappingProxyType(
        {
            period: MappingProxyType(
                {
                    regime_name: grid_schedule.grid_signature(
                        period=period, regime_name=regime_name
                    )
                    for regime_name in by_regime
                }
            )
            for period, by_regime in grid_schedule.by_period.items()
        }
    )


def _build_period_v_interpolation_info(
    *,
    representative_user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    grid_schedule: AgeGridSchedule | None,
) -> MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None:
    """Per-period continuation interpolation info, from cached concrete grids.

    For every active period holding an age-specialized grid, overlay that period's
    concrete grids on the representative regime and build its `VInterpolationInfo`,
    so period `t`'s continuation `V_{t+1}` is tabulated on the target regime's
    grid **at period `t+1`**. Never calls an age factory — it only transforms
    already-built concrete grids. `None` when no state is age-specialized.
    """
    if grid_schedule is None:
        return None
    result: dict[int, MappingProxyType[RegimeName, VInterpolationInfo]] = {}
    for period, regimes_at_period in grid_schedule.by_period.items():
        result[period] = MappingProxyType(
            {
                regime_name: create_v_interpolation_info(
                    representative_user_regimes[regime_name].replace(
                        states={
                            **representative_user_regimes[regime_name].states,
                            **{
                                state_name: resolved.grid
                                for state_name, resolved in states.items()
                            },
                        }
                    )
                )
                for regime_name, states in regimes_at_period.items()
            }
        )
    return MappingProxyType(result)


def _state_handoff_errors(
    *,
    phase_name: PhaseName,
    phase_reachability: PhaseReachability,
    specs: Mapping[RegimeName, PhasedRegimeSpec],
    ages: AgeGrid,
) -> list[str]:
    """Return errors for target states without a valid retained-edge handoff."""
    phase_slices: dict[RegimeName, RegimePhaseSpec] = {
        regime_name: getattr(spec, phase_name) for regime_name, spec in specs.items()
    }
    error_messages: list[str] = []
    for period in range(phase_reachability.n_periods - 1):
        for source in sorted(phase_reachability.active_regimes_by_period[period]):
            source_slice = phase_slices[source]
            for target in phase_reachability.targets(period=period, source=source):
                target_slice = phase_slices[target]
                for state_name, target_grid in target_slice.grid_states.items():
                    # An IID draw does not depend on its previous value, so the
                    # target's entry distribution is the process's own
                    # unconditional law and there is nothing for the source to
                    # hand over. Every other state -- including an AR(1)
                    # process, whose next draw does depend on a previous value
                    # the source would have to supply -- needs a handoff.
                    if isinstance(target_grid, _IIDProcess):
                        continue
                    if _has_valid_state_handoff(
                        source_slice=source_slice,
                        target=target,
                        state_name=state_name,
                    ):
                        continue
                    status = phase_reachability.edge_status(
                        period=period,
                        source=source,
                        target=target,
                    )
                    state_kind = (
                        "stochastic process"
                        if isinstance(target_grid, _ContinuousStochasticProcess)
                        else "state"
                    )
                    error_messages.append(
                        f"{phase_name} phase, period {period} "
                        f"(age {_display_age(ages, period)}), source '{source}' -> "
                        f"period {period + 1} "
                        f"(age {_display_age(ages, period + 1)}), "
                        f"target '{target}' retains a {status.name} edge. The "
                        f"target declares {state_kind} '{state_name}', but the "
                        f"source does not carry '{state_name}' and defines no "
                        f"entry law, so no next-period value exists. Declare "
                        f"'{state_name}' on '{source}', define a target-specific "
                        f"entry law, or narrow the transition's static target "
                        f"support."
                    )
    return error_messages


def _display_age(ages: AgeGrid, period: int) -> float:
    """Return period's age as a decimal float, never a raw `Fraction`."""
    return float(ages.exact_values[period])


def _has_valid_state_handoff(
    *,
    source_slice: RegimePhaseSpec,
    target: RegimeName,
    state_name: StateName,
) -> bool:
    """Return whether one target state obtains a valid next-period value.

    Every target state on a retained edge must be supplied by one of: a
    carried/shared state, a deterministic law, a stochastic law, or an explicit
    entry/reset law. A target-only state has no implicit initialization value.
    """
    if state_name in source_slice.grid_states:
        return True

    law = source_slice.state_transitions.get(state_name)
    return law is not None and (not isinstance(law, Mapping) or target in law)


def _build_solution_phase(
    *,
    spec: PhasedRegimeSpec,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    declared_regime_transition: object,
    phase_reachability: PhaseReachability,
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]],
    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]],
    regime_names_to_ids: RegimeNamesToIds,
    variables: Variables,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
    grid_schedule: AgeGridSchedule | None = None,
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
    certainty_equivalent: CertaintyEquivalent | None,
    solver: Solver,
    model_has_egm_regime: bool,
    has_taste_shocks: bool,
    stakeholders: tuple[str, ...] | None = None,
    weights: Mapping[str, float] | None = None,
    same_period_ref_regimes: tuple[RegimeName, ...] = (),
    edge_target_regimes: tuple[RegimeName, ...] = (),
    fold_state_names: tuple[StateName, ...] = (),
    fold_only_regimes: frozenset[RegimeName] = frozenset(),
) -> SolutionPhase:
    """Build all compiled functions for the backward-induction (solve) phase.

    Args:
        spec: The regime's per-phase specification.
        regime_name: The name of the regime.
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        declared_regime_transition: Solve transition before temporal filtering.
        phase_reachability: Static graph for the solution phase.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        granular_param_expansions: Immutable mapping of coarse-template law
            keys to granular qname prefixes.
        regime_to_flat_param_names: Immutable mapping of every regime name to
            its flat param names in the engine's binding vocabulary. A DC-EGM
            source carrying into a different target regime reads the target's
            params in its per-asset-node solve, so the kernel build needs the
            whole mapping, not only the source regime's own params.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        variables: States and actions of the regime with kind/topology/process tags.
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.
        solver: The regime's solver; the engine calls `validate` then
            `build_period_kernels` on it to obtain the per-period kernels.
        model_has_egm_regime: Whether any regime of the model uses a solver
            that reads continuation carries (an endogenous-grid solver);
            terminal regimes then produce their closed-form carries.
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.

    Returns:
        Complete solve functions container.

    """
    core = _process_regime_core(
        source_regime_name=regime_name,
        active_periods_by_regime=regimes_to_active_periods,
        functions=spec.solution.functions,
        constraints=spec.solution.constraints,
        state_transitions=spec.solution.state_transitions,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
        fold_only_regimes=fold_only_regimes,
        coarse_state_law_names=_phase_coarse_state_law_names(
            user_regime=user_regimes[regime_name], phase="solve"
        ),
        phase_reachability=phase_reachability,
    )

    # fold-review F2 (folded slice): a folded-only target enters `core.transitions`
    # with an EMPTY bundle (see `_process_regime_core`). Its stored V has the
    # folded axes integrated out, but its `VInterpolationInfo` still lists the
    # folded states, so the continuation interpolator would otherwise demand a
    # `next_<shock>` coordinate the source cannot supply and index an axis the
    # scalar V no longer has. Strip the folded axes from the interpolation info
    # of exactly those empty-bundle folded targets so the continuation is read
    # as the scalar V (no coordinate). Only the Q-and-F continuation read is
    # overridden; every other consumer keeps the unstripped info.
    regime_to_v_interpolation_info_for_Q = _strip_folded_axes_for_scalar_targets(
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        transitions=core.transitions,
        fold_only_regimes=fold_only_regimes,
        all_grids=all_grids,
    )

    flat_param_names = _engine_flat_param_names(
        regime_params_template=regime_params_template,
        granular_param_expansions=granular_param_expansions,
    )

    # Fixed, distributed states are co-mapped with the continuation V so the solve
    # kernel reads only its device-local slice (no all-gather). Terminal regimes have
    # no continuation, so the set is empty there.
    co_map_state_names: tuple[StateName, ...] = ()
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = ()

    next_state_names = _declared_next_state_names(user_regimes[regime_name])

    if spec.terminal:
        compute_regime_transition_probs = None
        validation_regime_transition_probs = None
        if stakeholders is not None:
            # COLLECTIVE-REGIMES (E1): the collective terminal kernel builds one
            # `U^s`-and-`F` per stakeholder and stacks the utilities on a trailing
            # stakeholder axis. Separate builder so the singleton path is untouched.
            terminal_func = get_Q_and_F_terminal_collective(
                flat_param_names=flat_param_names,
                functions=core.functions,
                constraints=core.constraints,
                stakeholders=stakeholders,
                next_state_names=next_state_names,
            )
        else:
            terminal_func = get_Q_and_F_terminal(
                flat_param_names=flat_param_names,
                functions=core.functions,
                constraints=core.constraints,
                next_state_names=next_state_names,
            )
        Q_and_F_functions = MappingProxyType(
            dict.fromkeys(range(ages.n_periods), terminal_func)
        )
        compute_intermediates: MappingProxyType[int, Callable] = MappingProxyType({})
    else:
        compute_regime_transition_probs = build_regime_transition_probs_functions(
            functions=core.functions,
            compute_regime_transition_probs=core.next_regime_func,
            grids=all_grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            flat_param_names=flat_param_names,
            is_stochastic=spec.solution.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="solve",
            next_regime_cells=(
                core.next_regime_cells
                if core.next_regime_func is not None
                or core.next_regime_cells is not None
                else MappingProxyType({})
            ),
        )
        validation_regime_transition_probs = _build_validation_regime_transition_probs(
            declared_regime_transition=declared_regime_transition,
            compute_regime_transition_probs=compute_regime_transition_probs,
            functions=core.functions,
            grids=all_grids[regime_name],
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            flat_param_names=flat_param_names,
            enable_jit=enable_jit,
        )
        co_map_state_names = _co_map_state_names(
            state_names=state_action_space.state_names,
            grids=all_grids[regime_name],
            transitions=core.transitions,
        )
        # A co-mapped state's axis is sliced only off the leaves that carry it; a
        # target regime where the state is pruned keeps its full leaf (`None`).
        co_map_v_arr_in_axes = tuple(
            MappingProxyType(
                {
                    target: 0
                    if state in regime_to_v_interpolation_info[target].state_names
                    else None
                    for target in regime_to_v_interpolation_info
                }
            )
            for state in co_map_state_names
        )
        # COLLECTIVE-REGIMES (E2): value-constraint predicates carry user params
        # exactly like ordinary constraints — rename them to their qnames; the
        # user's same-period reference declarations are resolved to the
        # engine-side form (the stakeholder name becomes the index on the
        # reference V's trailing stakeholder axis).
        user_regime = user_regimes[regime_name]
        value_constraints = MappingProxyType(
            {
                name: _rename_params_to_qnames(
                    func=func,
                    regime_params_template=regime_params_template,
                    param_key=name,
                )
                for name, func in user_regime.value_constraints.items()
            }
        )
        same_period_refs = _resolve_same_period_refs(
            user_regime=user_regime, user_regimes=user_regimes
        )
        Q_and_F_functions = _build_Q_and_F_per_period(
            active_periods=regimes_to_active_periods[regime_name],
            phase_reachability=phase_reachability,
            source_regime_name=regime_name,
            functions=core.functions,
            constraints=core.constraints,
            transitions=core.transitions,
            stochastic_transition_names=core.stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info_for_Q,
            flat_param_names=flat_param_names,
            period_to_regime_v_interp=period_to_regime_v_interp,
            co_map_state_names=co_map_state_names,
            certainty_equivalent=certainty_equivalent,
            stakeholders=stakeholders,
            value_constraints=value_constraints,
            same_period_refs=same_period_refs,
            next_state_names=next_state_names,
            grid_schedule=grid_schedule,
        )
        if stakeholders is not None:
            # COLLECTIVE-REGIMES (E1): the NaN-diagnostics intermediates mirror
            # the singleton Q evaluation (one `utility` target), which a
            # collective regime does not carry. The failure path handles a
            # missing closure gracefully (no U/F/E/Q breakdown).
            compute_intermediates = MappingProxyType({})
        else:
            compute_intermediates = _build_compute_intermediates_per_period(
                active_periods=regimes_to_active_periods[regime_name],
                flat_param_names=flat_param_names,
                phase_reachability=phase_reachability,
                source_regime_name=regime_name,
                functions=core.functions,
                constraints=core.constraints,
                transitions=core.transitions,
                stochastic_transition_names=core.stochastic_transition_names,
                compute_regime_transition_probs=compute_regime_transition_probs,
                regime_to_v_interpolation_info=regime_to_v_interpolation_info,
                state_action_space=state_action_space,
                grids=all_grids[regime_name],
                enable_jit=enable_jit,
                certainty_equivalent=certainty_equivalent,
                next_state_names=next_state_names,
                # F4: diagnostics recompute on the SAME period-specific target
                # grid as the primary solve (not the representative grid).
                grid_schedule=grid_schedule,
                period_to_regime_v_interp=period_to_regime_v_interp,
            )

    # Dispatch the per-period kernel build polymorphically on the regime's
    # solver: `validate` rejects out-of-scope configurations at build time,
    # then `build_period_kernels` returns one uniform period adapter per period
    # plus the regime's continuation template. `GridSearch` wraps the
    # max-Q-over-a grid search; `DCEGM` wraps the EGM step.
    context = SolverBuildContext(
        regime_name=regime_name,
        user_regimes=user_regimes,
        state_action_space=state_action_space,
        solution_reachability=phase_reachability,
        Q_and_F_functions=Q_and_F_functions,
        grids=all_grids[regime_name],
        period_to_state_nodes=_period_to_state_nodes(
            regime_name=regime_name, grid_schedule=grid_schedule
        ),
        functions=core.functions,
        constraints=core.constraints,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        period_to_regime_v_interp=period_to_regime_v_interp,
        period_to_regime_grid_signature=_period_to_regime_grid_signature(
            grid_schedule=grid_schedule
        ),
        regimes_to_active_periods=regimes_to_active_periods,
        flat_param_names=flat_param_names,
        regime_to_flat_param_names=regime_to_flat_param_names,
        enable_jit=enable_jit,
        has_taste_shocks=has_taste_shocks,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
        co_map_v_arr_in_axes=co_map_v_arr_in_axes,
        stakeholders=stakeholders,
        weights=weights,
        same_period_ref_regimes=same_period_ref_regimes,
        edge_target_regimes=edge_target_regimes,
        fold_state_names=fold_state_names,
    )
    solver.validate(context=context)
    solver_kernels = solver.build_period_kernels(context=context)

    # The terminal continuation publisher is a cross-solver concern, not the
    # grid search's: a terminal regime in a model with a DC-EGM regime must
    # publish a closed-form carry so a DC-EGM parent can interpolate its value
    # and marginal utility. Build the producer engine-side and compose it as an
    # output decorator around each period adapter, so the solver stays unaware
    # of the continuation it is being asked to emit.
    egm_carry_producer, egm_carry_template = _build_egm_child_carry_producer(
        user_regime=user_regimes[regime_name],
        functions=core.functions,
        variables=variables,
        grids=all_grids[regime_name],
        model_has_egm_regime=model_has_egm_regime,
        solver_produces_carry=solver_kernels.continuation_template is not None,
        enable_jit=enable_jit,
    )
    period_kernels = solver_kernels.period_kernels
    continuation_template = solver_kernels.continuation_template
    if egm_carry_producer is not None:
        period_kernels = MappingProxyType(
            {
                period: _TerminalCarryPeriodKernel(
                    base=kernel,
                    carry_producer=egm_carry_producer,
                    regime_name=regime_name,
                )
                for period, kernel in period_kernels.items()
            }
        )
        continuation_template = egm_carry_template

    # The published function set is consumed unresolved by feasibility checks and
    # additional-target computation, so resolve any `PeriodizedEconFunction` to its
    # representative-period concrete function here (the per-period Q_and_F build
    # keeps resolving `core.functions` per period).
    solution_active_periods = regimes_to_active_periods[regime_name]
    published_solution_functions = (
        cast(
            "EconFunctionsMapping",
            resolve_periodized_nodes(core.functions, solution_active_periods[0]),
        )
        if solution_active_periods
        else core.functions
    )

    period_state_axes = _build_period_state_axes(
        regime_name=regime_name,
        grid_schedule=grid_schedule,
        active_periods=regimes_to_active_periods[regime_name],
    )

    return SolutionPhase(
        _variables=variables,
        grids=all_grids[regime_name],
        functions=published_solution_functions,
        # Marker-bearing (unresolved) pool for the simulation continuation, resolved
        # per period's age there rather than frozen at the representative age (F1).
        _continuation_functions=core.functions,
        constraints=core.constraints,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        reachability=phase_reachability,
        compute_regime_transition_probs=compute_regime_transition_probs,
        period_kernels=period_kernels,
        validation_regime_transition_probs=validation_regime_transition_probs,
        compute_intermediates=compute_intermediates,
        continuation_template=continuation_template,
        _base_state_action_space=state_action_space,
        period_state_axes=period_state_axes,
    )


def _filter_kwargs_for_func(
    *, func: Callable, kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    """Filter kwargs to only those accepted by func's signature."""
    try:
        sig = inspect.signature(func)
    except ValueError, TypeError:
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


@dataclass(frozen=True, kw_only=True)
class _TerminalCarryPeriodKernel:
    """Engine-owned output decorator publishing a terminal regime's carry.

    Wraps a grid-search period adapter so that, after the base kernel computes
    the value-function array, the regime's closed-form carry producer turns that
    array into the continuation a DC-EGM parent interpolates. Publishing a carry
    is a cross-solver concern, so the wrapped solver stays unaware of it.

    `core` and `build_lower_args` delegate to the base adapter: the carry
    producer is a separately built (and jitted) closure invoked inline, not part
    of the AOT-compiled core, so AOT compilation deduplicates and lowers exactly
    the base grid-search core.
    """

    base: PeriodKernel
    """The wrapped grid-search period adapter."""

    carry_producer: EGMCarryProducer
    """Closed-form producer mapping the value array to the regime's carry."""

    regime_name: RegimeName
    """Name of the terminal regime whose flat params the producer reads."""

    @property
    def core(self) -> Callable:
        """The base adapter's shared jitted core, for any single-core reader."""
        return self.base.core

    def cores(self) -> Mapping[str, Callable]:
        """Delegate to the base adapter's cores (a terminal regime is single-core)."""
        return self.base.cores()

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _TerminalCarryPeriodKernel:
        """Bind fixed params into both the base core and the carry producer.

        A terminal regime's carry producer evaluates the regime's own bequest
        utility on the wealth grid; that utility may reach a model-level fixed
        param (e.g. a consumption-equivalence scale) through a helper. The solve
        loop invokes the producer with only the live (free) params, so bind the
        regime's fixed params here — matching the base adapter's core binding.
        """
        regime_fixed = dict(
            fixed_flat_params.get(self.regime_name, MappingProxyType({}))
        )
        base = self.base.with_fixed_params(fixed_flat_params=fixed_flat_params)
        carry_producer = self.carry_producer
        if regime_fixed:
            carry_producer = functools.partial(
                carry_producer,
                **_filter_kwargs_for_func(func=carry_producer, kwargs=regime_fixed),
            )
        return dataclass_replace(self, base=base, carry_producer=carry_producer)

    def build_lower_args(
        self,
        *,
        core_key: str = "main",
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the base core's lowering arguments (the carry producer is jitted
        separately at build time, so it is not part of the AOT-compiled core)."""
        return self.base.build_lower_args(
            core_key=core_key,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
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
        """Run the base kernel, then publish the regime's continuation carry."""
        result = self.base(
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        carry = self.carry_producer(
            V_arr=result.V_arr,
            **state_action_space.states,
            **flat_params[self.regime_name],
            period=jnp.int32(period),
            age=ages.values[period],
        )
        # `dissolution` (E2) rides through unchanged — unreachable today (collective
        # terminals carry actions, so no closed-form carry producer wraps them),
        # but the decorator must not silently drop a base kernel's output.
        return KernelResult(
            V_arr=result.V_arr,
            continuation=carry,
            simulation_policy=result.simulation_policy,
            dissolution=result.dissolution,
        )


def _build_egm_child_carry_producer(
    *,
    user_regime: UserRegime,
    functions: EconFunctionsMapping,
    variables: Variables,
    grids: MappingProxyType[StateOrActionName, Grid],
    model_has_egm_regime: bool,
    solver_produces_carry: bool,
    enable_jit: bool,
) -> tuple[EGMCarryProducer | None, EGMCarry | None]:
    """Build the carry producer and template for an EGM regime's carry target.

    A regime an endogenous-grid regime transitions into must publish its value
    and marginal value of resources so the parent can interpolate them. The
    regime's own solver already builds its carry when it is endogenous-grid
    (`solver_produces_carry`); this engine-side producer covers the brute
    (`GridSearch`) targets that do not. Cases:

    - no states (terminal) ⇒ constant-value, zero-marginal-utility broadcast rows
    - terminal with ≥1 continuous state, no actions, and discrete states only of
      the fixed (non-process) kind ⇒ terminal utility and its wealth gradient on
      the regime's own state grid, the discrete states leading
    - living brute regime with a continuous Euler state ⇒ its solved value array
      and the array's Euler-state gradient, discrete states (process states
      included) and passive continuous states leading
    - anything else ⇒ no producer (an EGM regime targeting an unsupported shape
      is rejected by the EGM kernel builder)

    Returns:
        Tuple of the producer and the regime's carry template, both `None` for
        models without an endogenous-grid regime, for regimes whose own solver
        already produces a carry, and for unsupported shapes.

    """
    if not model_has_egm_regime or solver_produces_carry:
        return None, None
    producer: EGMCarryProducer
    discrete_state_names = tuple(
        name
        for name in variables.state_names
        if name in set(variables.discrete_state_names)
    )
    continuous_state_names = tuple(variables.continuous_state_names)
    euler_state_name = next(iter(user_regime.states), None)
    if user_regime.terminal:
        has_only_fixed_discrete_states = all(
            not isinstance(grids[name], _ContinuousStochasticProcess)
            for name in discrete_state_names
        )
        if not variables.state_names:
            producer = get_stateless_terminal_carry_producer()
            template = build_template_egm_carry(n_rows=N_STATELESS_CARRY_ROWS)
        elif (
            len(continuous_state_names) >= 1
            and has_only_fixed_discrete_states
            and not user_regime.actions
            and euler_state_name in continuous_state_names
        ):
            # The parent's child read picks the terminal's Euler state as its
            # first declared state (`_get_child_state_name`); the remaining
            # continuous states are the passive (durable / outer) margins it
            # interpolates as leading carry axes — the NEGM housing-bequest shape.
            passive_state_names = tuple(
                name for name in continuous_state_names if name != euler_state_name
            )
            producer = get_terminal_wealth_carry_producer(
                functions=functions,
                state_name=euler_state_name,
                discrete_state_names=discrete_state_names,
                passive_state_names=passive_state_names,
                continuous_state_order=continuous_state_names,
            )
            leading_shape = tuple(
                int(grids[name].to_jax().shape[0])
                for name in discrete_state_names + passive_state_names
            )
            template = shard_carry_template(
                template=build_template_egm_carry(
                    n_rows=int(grids[euler_state_name].to_jax().shape[0]),
                    leading_shape=leading_shape,
                ),
                grids=grids,
                leading_axis_names=discrete_state_names + passive_state_names,
            )
        else:
            return None, None
    elif (
        len(continuous_state_names) >= 1 and euler_state_name in continuous_state_names
    ):
        # A living brute target: its solved value array is the carry value and the
        # array's Euler-state gradient is the marginal value of resources. The
        # parent reads it in M-space ($R \\equiv M$) via the same child read it
        # uses for an endogenous-grid target.
        passive_state_names = tuple(
            name for name in continuous_state_names if name != euler_state_name
        )
        producer = get_brute_child_carry_producer(
            state_name=euler_state_name,
            discrete_state_names=discrete_state_names,
            passive_state_names=passive_state_names,
            continuous_state_order=continuous_state_names,
        )
        leading_shape = tuple(
            int(grids[name].to_jax().shape[0])
            for name in discrete_state_names + passive_state_names
        )
        template = shard_carry_template(
            template=build_template_egm_carry(
                n_rows=int(grids[euler_state_name].to_jax().shape[0]),
                leading_shape=leading_shape,
            ),
            grids=grids,
            leading_axis_names=discrete_state_names + passive_state_names,
        )
    else:
        return None, None
    if enable_jit:
        producer = jax.jit(producer)
    return producer, template


def _build_simulation_phase(  # noqa: PLR0915
    *,
    spec: PhasedRegimeSpec,
    user_regime: UserRegime,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    solution_reachability: PhaseReachability,
    simulation_reachability: PhaseReachability,
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]],
    regime_names_to_ids: RegimeNamesToIds,
    variables: Variables,
    simulation_variables: Variables,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
    grid_schedule: AgeGridSchedule | None = None,
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
    solve_functions: EconFunctionsMapping,
    solve_transitions: TransitionFunctionsMapping,
    solve_stochastic_transition_names: frozenset[TransitionFunctionName],
    solve_compute_regime_transition_probs: RegimeTransitionFunction | None,
    has_taste_shocks: bool,
    solver: Solver,
    certainty_equivalent: CertaintyEquivalent | None,
    stakeholders: tuple[str, ...] | None = None,
    weights: Mapping[str, float] | None = None,
    fold_only_regimes: frozenset[RegimeName] = frozenset(),
) -> SimulationPhase:
    """Build all compiled functions for the forward-simulation phase.

    The decision functions (Q_and_F, argmax, regime-transition probs) are
    built from the simulation slice's functions plus each carried state's
    solve-phase imputation — the agent decides on the value the solved policy
    was computed for. The published function pool strips the imputations so
    every other simulate consumer reads the carried value.

    Q_and_F always uses the solve (non-vmapped) regime transition probs because
    it evaluates on the Cartesian grid, not per-subject.

    For a DC-EGM or NEGM regime, the budget constraint the EGM solve enforces
    intrinsically is synthesized and injected into the constraint set: the
    simulate-phase grid argmax needs it as a feasibility mask exactly like a
    user-declared borrowing constraint of a brute-force regime. NEGM nests the
    same inner 1-D solve, so the mask comes from its inner DC-EGM config. The
    solve phase is unaffected — the EGM kernels never see it.

    Args:
        spec: The regime's per-phase specification.
        user_regime: The finalized user regime, scanned for `Phased`
            declarations by the policy-replay gate.
        regime_name: The name of the regime.
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances. COLLECTIVE-REGIMES (E4): only consulted for a
            collective regime, to resolve its `value_constraints` (E2) and
            `same_period_refs` (E2) exactly as the solve phase does — the
            simulate-phase Q_and_F must apply the identical value-aware
            feasibility mask so the simulated argmax never picks an action
            the solved value function excluded.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        granular_param_expansions: Immutable mapping of coarse-template law
            keys to granular qname prefixes.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        variables: States and actions of the regime with kind/topology/process tags.
        simulation_variables: Simulate-phase variables (solve variables plus
            carried-only states, appended).
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.
        solve_functions: The solve phase's function pool. Q prices the continuation
            under the agent's perceived law, so the continuation sub-DAG (state laws,
            stochastic weights, and every helper they read) is resolved against this
            pool rather than the simulate one.
        solve_transitions: Transitions from the solve phase (reused).
        solve_stochastic_transition_names: Stochastic transition names from solve
            (reused).
        solve_compute_regime_transition_probs: Solve-phase regime transition prob
            function, used for Q_and_F in both phases.
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.
        solver: The regime's solver configuration; a DC-EGM or NEGM regime
            gets the synthesized intrinsic budget constraint.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.
        stakeholders: Ordered stakeholder names for a collective regime, or
            `None` (the singleton default).
        weights: Household Pareto weights per stakeholder; required (and only
            used) when `stakeholders` is set — feeds the simulate-side
            argmax's household scalarization (E4).

    Returns:
        Complete simulate functions container.

    """
    carried_only = spec.carried_only_state_names
    decision_functions = dict(spec.simulation.functions) | {
        name: spec.solution.functions[name] for name in carried_only
    }
    core = _process_regime_core(
        source_regime_name=regime_name,
        active_periods_by_regime=regimes_to_active_periods,
        functions=decision_functions,
        constraints=spec.simulation.constraints,
        state_transitions=spec.simulation.state_transitions,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
        fold_only_regimes=fold_only_regimes,
        coarse_state_law_names=_phase_coarse_state_law_names(
            user_regime=user_regime, phase="simulate"
        ),
        phase_reachability=simulation_reachability,
    )
    functions = core.functions
    constraints = core.constraints
    if isinstance(solver, (DCEGM, NEGM)):
        if (
            DCEGM_BUDGET_CONSTRAINT_NAME in core.functions
            or DCEGM_BUDGET_CONSTRAINT_NAME in core.constraints
        ):
            msg = (
                f"Regime '{regime_name}' declares a function or constraint "
                f"named '{DCEGM_BUDGET_CONSTRAINT_NAME}'. That name is "
                "reserved for the budget constraint the simulate phase "
                "synthesizes for DC-EGM and NEGM regimes; rename it."
            )
            raise ModelInitializationError(msg)
        constraints = MappingProxyType(
            {
                **core.constraints,
                DCEGM_BUDGET_CONSTRAINT_NAME: get_intrinsic_budget_constraint(
                    solver=solver, functions=core.functions
                ),
            }
        )

    # Every published simulate-phase consumer (next_state, the realized
    # regime draw, the feasibility check, additional targets) reads each
    # carried state as its carried true value, not the solve-phase
    # imputation. Dropping the imputation turns the name into a leaf supplied
    # by the simulator, and `core.transitions` (built from the simulation
    # slice) carries every simulate-phase law — including each carried
    # state's `next_<name>` and any `Phased` law's simulate variant. Only the
    # decision functions (Q_and_F / argmax) keep the imputation — the agent
    # decides on the value the solved policy was computed for. To make the
    # realized regime draw decide on the imputation instead, declare the
    # imputation under a second name in `functions` and read that.
    if carried_only:
        simulate_functions: EconFunctionsMapping = MappingProxyType(
            {k: v for k, v in core.functions.items() if k not in carried_only}
        )
    else:
        simulate_functions = core.functions
    # Carried states are `Phased(simulate=Grid)` by the phase grammar, so the
    # isinstance check is a no-op at runtime; it narrows the type (an
    # `AgeSpecializedGrid` can never be carried-only).
    carried_grids = {
        name: grid
        for name, grid in spec.simulation.grid_states.items()
        if name in carried_only and isinstance(grid, Grid)
    }
    simulate_grids = MappingProxyType({**all_grids[regime_name], **carried_grids})

    flat_param_names = _engine_flat_param_names(
        regime_params_template=regime_params_template,
        granular_param_expansions=granular_param_expansions,
    )

    # COLLECTIVE-REGIMES (E4): forward simulation of a collective regime
    # reuses the SAME Q_and_F builders the solve phase uses (E1/E2's
    # `get_Q_and_F_terminal_collective` / `get_Q_and_F_collective`,
    # value-masked exactly like solve), so the simulated argmax never picks
    # an action the solved value function excluded. Only the ARGMAX step
    # differs from the singleton path — `get_argmax_and_max_Q_over_a`'s
    # collective branch (below) recomputes the household argmax and gathers
    # each stakeholder's own value at it.
    collective = stakeholders is not None
    next_state_names = _declared_next_state_names(user_regime)
    if spec.terminal:
        compute_regime_transition_probs = None
        if collective:
            terminal_func = get_Q_and_F_terminal_collective(
                flat_param_names=flat_param_names,
                functions=functions,
                constraints=constraints,
                stakeholders=stakeholders,
                next_state_names=next_state_names,
            )
        else:
            terminal_func = get_Q_and_F_terminal(
                flat_param_names=flat_param_names,
                functions=functions,
                constraints=constraints,
                next_state_names=next_state_names,
            )
        Q_and_F_functions = MappingProxyType(
            dict.fromkeys(range(ages.n_periods), terminal_func)
        )
    else:
        compute_regime_transition_probs = build_regime_transition_probs_functions(
            functions=simulate_functions,
            compute_regime_transition_probs=core.next_regime_func,
            grids=simulate_grids,
            regime_names_to_ids=regime_names_to_ids,
            flat_param_names=flat_param_names,
            is_stochastic=spec.simulation.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="simulate",
            next_regime_cells=(
                core.next_regime_cells
                if core.next_regime_func is not None
                or core.next_regime_cells is not None
                else MappingProxyType({})
            ),
        )
        # Q_and_F uses the solve (non-vmapped) regime transition probs since
        # it evaluates on the Cartesian grid, not per-subject. The solve
        # phase built that function unconditionally for non-terminal regimes.
        assert solve_compute_regime_transition_probs is not None  # noqa: S101
        if collective:
            # COLLECTIVE-REGIMES (E2): resolve the same value_constraints /
            # same_period_refs the solve phase reads, so simulate applies the
            # identical value-aware feasibility mask (E2's `Q_<s>`-conditioned
            # IR predicates, and same-period reference reads).
            assert user_regime is not None  # noqa: S101
            value_constraints = MappingProxyType(
                {
                    name: _rename_params_to_qnames(
                        func=func,
                        regime_params_template=regime_params_template,
                        param_key=name,
                    )
                    for name, func in user_regime.value_constraints.items()
                }
            )
            same_period_refs = _resolve_same_period_refs(
                user_regime=user_regime, user_regimes=user_regimes
            )
        else:
            value_constraints = MappingProxyType({})
            same_period_refs = MappingProxyType({})
        # The simulated agent acts on its BELIEFS about the FUTURE and lives in the
        # TRUTH NOW. So Q is built from two phase-closed halves:
        #   flow         = simulate transitions + simulate pool (`functions`)
        #   continuation = solve transitions    + solve pool (`solve_functions`)
        # Each half needs BOTH its transitions and its function pool: `dags` resolves a
        # transition's argument names against the pool it is handed, transitively, so
        # passing `transitions=solve_transitions` alone would still read `Phased`
        # helpers (and, for `MarkovTransition`, the whole `weight_*` node) from the
        # simulate phase — and passing the solve transitions into the flow would leave
        # the flow's `next_<state>` a solve law wearing simulate helpers, a sub-DAG that
        # is neither phase. The same `next_<state>` name therefore legitimately resolves
        # to different callables in the two halves. The realized next state stays on the
        # simulate laws — see `next_state`/`compute_regime_transition_probs` below.
        #
        # ONE EXCEPTION to "flow = simulate truth": `functions` here is the
        # imputation-augmented decision pool built above — for a carried-only state
        # it carries the solve IMPUTATION, not the realized carried value. So flow
        # utility/feasibility that reads a carried state decides on the imputation,
        # by design (the continuation was solved there; policy-consistency). The
        # realized carried value is used for the forward transition via
        # `simulate_functions`. See the carried-state comment at the top of this
        # function and the `Phased` semantics contract in `lcm/phased.py`.
        #
        # fold-round6/round7 simulate parity: a folded-only target's stored V is a
        # scalar with its folded axes integrated out, while its `VInterpolationInfo`
        # still lists them — so the unstripped info makes the continuation read demand
        # a `next_<shock>` coordinate the source never realises. Strip those folded
        # axes for the simulate Q read exactly as the solve phase does (reusing the
        # same helper, keyed on the transitions the Q read actually enumerates); every
        # other simulate consumer keeps the unstripped info.
        regime_to_v_interpolation_info_for_Q = _strip_folded_axes_for_scalar_targets(
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            transitions=solve_transitions,
            fold_only_regimes=fold_only_regimes,
            all_grids=all_grids,
        )
        Q_and_F_functions = _build_Q_and_F_per_period(
            active_periods=regimes_to_active_periods[regime_name],
            phase_reachability=solution_reachability,
            source_regime_name=regime_name,
            functions=functions,
            constraints=constraints,
            transitions=solve_transitions,
            stochastic_transition_names=solve_stochastic_transition_names,
            compute_regime_transition_probs=solve_compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info_for_Q,
            flat_param_names=flat_param_names,
            period_to_regime_v_interp=period_to_regime_v_interp,
            certainty_equivalent=certainty_equivalent,
            stakeholders=stakeholders,
            value_constraints=value_constraints,
            same_period_refs=same_period_refs,
            continuation_functions=solve_functions,
            flow_transitions=core.transitions,
            flow_stochastic_transition_names=core.stochastic_transition_names,
            next_state_names=next_state_names,
            grid_schedule=grid_schedule,
        )

    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        enable_jit=enable_jit,
        has_taste_shocks=has_taste_shocks,
        stakeholders=stakeholders,
        weights=weights,
    )

    pointwise_Q_and_F = _build_pointwise_Q_and_F_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        enable_jit=enable_jit,
    )

    next_state = _build_next_state_vmapped(
        active_periods=regimes_to_active_periods[regime_name],
        phase_reachability=simulation_reachability,
        source_regime_name=regime_name,
        functions=simulate_functions,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        all_grids=all_grids,
        variables=variables,
        flat_param_names=flat_param_names,
        enable_jit=enable_jit,
    )

    # Replaying the solve-phase EGM policy in simulation is valid only when
    # the simulate-phase decision problem is the one the solve optimized and
    # the published rows carry both the coordinates and the branch topology
    # the read interpolates over at every state dimension:
    # - a standalone `DCEGM` solver — a NEGM regime maximizes its value over
    #   keeper and adjuster candidates but publishes only the keeper's inner
    #   consumption function, so replaying it would pair an adjuster-won
    #   value with the keeper's policy;
    # - a crossing-certifying upper-envelope backend (MSS only) — RFC and LTM
    #   leave the envelope switch between two retained nodes, and FUES decides
    #   segment identity by a slope-threshold heuristic with no labels from
    #   the kernel, so its row can bridge a missed switch; only MSS inserts
    #   the exact crossing by construction (see
    #   `_envelope_publishes_crossings`);
    # - the single-post-state kernel (not asset-row mode) — when a
    #   savings-stage function reads the Euler state, DC-EGM solves per
    #   exogenous asset node and publishes one optimal point per node, not a
    #   crossing-complete resources-space row, so interpolating across nodes
    #   would mix branches wherever the winner changes between adjacent nodes;
    # - no continuous stochastic-process state — a process is stored as a
    #   node-valued row axis, but its simulation transition draws a continuous
    #   value that need not land on a node, so nearest-node row selection
    #   reads the wrong conditional policy;
    # - no passive continuous state — each row is the upper-envelope policy
    #   conditional on one passive node, and blending two rows across a
    #   passive-dimension branch switch yields an action from neither branch;
    # - no `Phased` declaration anywhere on the regime (a phase-variant
    #   utility, budget, transition, or state domain changes the
    #   simulate-phase FOC or the policy-row coordinates even under an
    #   unchanged `H`) — `Phased` is the grammar's only source of phase
    #   variance, so its absence is the exact test;
    # - no carried-only states (their simulate-phase domain has no solve-side
    #   row coordinates to read the policy on);
    # - no taste shocks (the realized discrete draw perturbs the decision).
    # The process, passive, and asset-row exclusions each lift once the read
    # publishes conditional values and re-decides the branch at the simulated
    # state; today the gate keeps those regimes on the grid-argmax path.
    phase_invariant = (
        not regime_declares_phased(user_regime) and not spec.carried_only_state_names
    )
    own_v_info = regime_to_v_interpolation_info[regime_name]
    egm_policy_read = None
    if (
        isinstance(solver, DCEGM)
        and _envelope_publishes_crossings(solver)
        and phase_invariant
        and not has_taste_shocks
        and not _regime_has_process_state(own_v_info)
        and not _regime_has_passive_state(
            v_interpolation_info=own_v_info, euler_state_name=solver.continuous_state
        )
        and not savings_stage_reads_euler_state(user_regime=user_regime, solver=solver)
    ):
        egm_policy_read = EGMPolicyRead(
            action_name=solver.continuous_action,
            resources_target=solver.resources,
            savings_lower_bound=float(solver.savings_grid.to_jax()[0]),
        )

    # Inventory the periodized nodes the additional-target guard must reject —
    # built from the (pre-publication) `functions` AND `constraints`.
    # `_process_regime_core` excludes constraint names from `functions`, but the
    # additional-target pool re-merges constraints (`_build_functions_pool`) and
    # advertises them as targets; without the constraint namespace here a
    # periodized constraint would escape the guard. Both mappings are
    # core-processed and still carry `PeriodizedEconFunction` markers.
    age_specialized_function_names = frozenset(
        name
        for name, func in (*simulate_functions.items(), *constraints.items())
        if isinstance(func, PeriodizedEconFunction)
    )

    # Publish representative-period-resolved functions AND constraints: the
    # feasibility check (`_get_feasibility` at initial-conditions validation) and
    # additional-target computation consume both as plain callables, so an
    # unresolved `PeriodizedEconFunction` leaking into either raises at build/eval
    # time. `next_state` above keeps resolving `simulate_functions` per period;
    # per-period *target* reads of a periodized node are still rejected by the guard
    # via `age_specialized_function_names` (a rep-period closure would be wrong).
    simulation_active_periods = regimes_to_active_periods[regime_name]
    published_simulate_functions = (
        cast(
            "EconFunctionsMapping",
            resolve_periodized_nodes(simulate_functions, simulation_active_periods[0]),
        )
        if simulation_active_periods
        else simulate_functions
    )
    published_simulate_constraints = (
        cast(
            "ConstraintFunctionsMapping",
            resolve_periodized_nodes(constraints, simulation_active_periods[0]),
        )
        if simulation_active_periods
        else constraints
    )

    return SimulationPhase(
        _variables=simulation_variables,
        grids=simulate_grids,
        carried_only_state_names=frozenset(carried_grids),
        functions=published_simulate_functions,
        constraints=published_simulate_constraints,
        age_specialized_function_names=age_specialized_function_names,
        transitions=core.transitions,
        reachability=simulation_reachability,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        argmax_and_max_Q_over_a=argmax_and_max_Q_over_a,
        Q_and_F=pointwise_Q_and_F,
        next_state=next_state,
        egm_policy_read=egm_policy_read,
    )


# Upper envelopes whose published row carries every switch a read crosses.
# Empty: no shipped backend provides the guarantee, so the branch-faithful policy
# read is off and simulation keeps its grid-argmax. An envelope earns a place here
# by passing the crossing- and support-completeness suites, not by intending to.
_CROSSING_COMPLETE_ENVELOPES: frozenset[str] = frozenset()


def _envelope_publishes_crossings(solver: DCEGM) -> bool:
    """Whether the solver's upper envelope certifies every segment crossing.

    No shipped backend qualifies, so the read is off everywhere and simulation
    keeps its grid-argmax. A branch-faithful policy read interpolates a row whose
    envelope switches sit at duplicated abscissae carrying both branch records:
    - `"mss"` ⇒ no: the refinement inserts crossings only between adjacent query
      winners, so a branch that owns ground strictly inside one candidate
      interval leaves no record in the row, and a switch landing exactly on a
      candidate abscissa is suppressed by the strict-interior test. Opening the
      read on such a row does not merely blur a switch — it publishes a
      different action, and one the canonical-Q safeguard cannot recover,
      because that safeguard rescores the emitted candidate against the finite
      action grid and an omitted owner is in neither set.
    - `"exact"` ⇒ no, and this is the default, so the branch-faithful read is
      off unless a regime asks for `"mss"`. The exclusion is narrower than the
      ones below: ownership is resolved per node cell and certified, a boundary
      whose sides differ in value *or* policy already emits both one-sided
      records, and a chain with more owned sub-cells than the row has slots
      overflows loudly through `n_kept`. What is missing is the other half of
      MSS's guarantee — the exact backend publishes no read-support verdict, so
      a row whose segment chain splits (NaN-dead candidates, or a value
      decrease between consecutive candidates) carries no signal that the
      reader would be bridging a gap rather than interpolating one branch.
    - `"fues"` ⇒ no: segment identity is decided by thresholding the
      implied-savings slope (`fues_jump_thresh`) — a heuristic — and the
      DC-EGM kernel supplies no segment labels. Two value branches whose
      cross-segment slope stays below the threshold merge into one row with
      no crossing inserted; the row then bridges the switch, and neither an
      exhaustive scan nor a wider window repairs that, because the scan can
      only search within the segment identity it was given. A FUES row is
      therefore not certified crossing-complete for the read.
    - `"rfc"` / `"ltm"` ⇒ no: the switch lands between retained nodes.
    """
    return solver.envelope in _CROSSING_COMPLETE_ENVELOPES


def _regime_has_process_state(v_interpolation_info: VInterpolationInfo) -> bool:
    """Whether the regime carries a continuous stochastic-process state.

    A process is stored among the node-valued discrete states, so its policy
    row is a discrete axis. Its simulation transition draws a continuous value
    off the process grid, which a nearest-node row read cannot resolve.
    """
    return any(
        isinstance(grid, _ContinuousStochasticProcess)
        for grid in v_interpolation_info.discrete_states.values()
    )


def _regime_has_passive_state(
    *, v_interpolation_info: VInterpolationInfo, euler_state_name: StateName
) -> bool:
    """Whether the regime carries a passive continuous state.

    Every continuous state other than the Euler state is passive; the DC-EGM
    policy row is conditional on each passive node, so blending rows across the
    passive axis is only branch-faithful where no envelope switch lies between.
    """
    return any(
        name != euler_state_name for name in v_interpolation_info.continuous_states
    )


def regime_declares_phased(user_regime: UserRegime) -> bool:
    """Whether any regime slot carries a `Phased` (phase-variant) declaration.

    Scans `functions`, `state_transitions` (including per-target dicts), the
    regime `transition`, and `states`. `Phased` is outermost-only in every
    slot except a per-target transition dict, whose cells it may wrap.
    """

    def is_phased(value: object) -> bool:
        if isinstance(value, Phased):
            return True
        if isinstance(value, Mapping):
            return any(isinstance(cell, Phased) for cell in value.values())
        return False

    return (
        any(is_phased(value) for value in user_regime.functions.values())
        or any(is_phased(value) for value in user_regime.state_transitions.values())
        or is_phased(user_regime.transition)
        or any(is_phased(value) for value in user_regime.states.values())
    )


@dataclass(frozen=True)
class _CoreResult:
    """Result of core regime function processing for one phase."""

    functions: EconFunctionsMapping
    """User functions (utility, helpers) with params renamed to qnames."""

    constraints: ConstraintFunctionsMapping
    """Constraint functions with params renamed to qnames."""

    transitions: TransitionFunctionsMapping
    """Nested mapping of transition names to transition functions."""

    stochastic_transition_names: frozenset[TransitionFunctionName]
    """Frozenset of stochastic transition function names."""

    next_regime_func: TransitionFunction | None
    """The coarse regime transition function; `None` for terminal regimes and
    for per-target regime transitions."""

    next_regime_cells: MappingProxyType[RegimeName, EconFunction] | None
    """Per-target regime transition probability functions (params renamed),
    or `None` when the regime transition is coarse or absent."""


def _process_regime_core(
    *,
    source_regime_name: RegimeName,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
    functions: Mapping[FunctionName, UserFunction],
    constraints: Mapping[FunctionName, UserFunction],
    state_transitions: Mapping[StateName, object],
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    variables: Variables,
    coarse_state_law_names: frozenset[StateName],
    fold_only_regimes: frozenset[RegimeName] = frozenset(),
    phase_reachability: PhaseReachability,
) -> _CoreResult:
    """Process one phase's regime functions and transitions.

    The caller supplies phase-resolved inputs (a slice of the regime's
    `PhasedRegimeSpec`, possibly augmented): rename params to qualified names,
    classify and process transitions.

    Args:
        source_regime_name: The name of the regime being processed (the
            transition SOURCE). Used, with `active_periods_by_regime`, to
            decide coarse-transition candidate reachability and to reject an
            ambiguous folded-coarse topology (see the reachability
            construction below).
        active_periods_by_regime: Active-period tuples for every regime. A
            coarse candidate `T` is "reachable next" from the source when `T`
            is active in some period immediately after a period the source is
            active — the test that gates the folded-coarse scope error.
        functions: Phase-resolved regime functions for this build.
        constraints: Phase-resolved constraint functions.
        state_transitions: This phase's `state_transitions` slice, used to
            detect per-target dicts and stochastic transitions.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        variables: States and actions of the regime with kind/topology/process tags.
        coarse_state_law_names: State names whose deterministic law is COARSE
            (a bare law, not a per-target dict) in THIS phase. A coarse law's
            `next_<state>` cells are stamped at the shared bare location so a
            within-period read MERGES; a per-target law's cells keep their
            target-qualified location so the read CONFLICTS. See
            `_phase_coarse_state_law_names`.
        fold_only_regimes: Regime names whose every state is a folded IID process
            (scalar stored V). A reachable target in this set that would otherwise
            get an empty bundle is kept enumerable with an explicit empty bundle
            (fold-review F2), so its scalar continuation enters E[V].
        phase_reachability: This phase's static regime graph, the sole source
            of which targets need a continuation built.
        source_regime_name: This regime's name, used to read its retained
            targets from `phase_reachability`.

    Returns:
        Core processing result with functions, constraints, transitions, stochastic
        transition names, and the next_regime function.

    """
    flat_grids = flatten_regime_namespace(all_grids)

    # The canonical regime transition rides in the bundles as each target's
    # `"next_regime"` cell. Cells are not state laws: split them off before
    # the flat-namespace processing, dropping bundles that held nothing else
    # so the per-target `transitions` mapping keys exactly the targets with
    # at least one state law (the period-target enumeration reads those
    # keys).
    next_regime_cells_by_target: dict[
        RegimeName, UserFunction | _CoarseTransitionCell
    ] = {}
    state_law_bundles: dict[RegimeName, dict[TransitionFunctionName, UserFunction]] = {}
    for target_regime_name, bundle in nested_transitions.items():
        if "next_regime" in bundle:
            next_regime_cells_by_target[target_regime_name] = bundle["next_regime"]
        laws = {
            law_name: cast("UserFunction", law)
            for law_name, law in bundle.items()
            if law_name != "next_regime"
        }
        if laws:
            state_law_bundles[target_regime_name] = laws

    flat_nested_transitions = flatten_regime_namespace(state_law_bundles)

    all_functions: dict[str, UserFunction] = {
        **functions,
        **constraints,
        **flat_nested_transitions,
    }

    stochastic_transition_names = _get_stochastic_transition_names(
        state_transitions=state_transitions, variables=variables
    )

    stochastic_transition_functions = {
        func_name: func
        for func_name, func in flat_nested_transitions.items()
        if tree_path_from_qname(func_name)[-1] in stochastic_transition_names
    }

    deterministic_transition_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name in flat_nested_transitions
        and func_name not in stochastic_transition_functions
    }

    deterministic_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name not in stochastic_transition_functions
        and func_name not in deterministic_transition_functions
    }

    processed_functions: dict[str, EconFunction] = {}

    for func_name, func in deterministic_functions.items():
        # `_process_one_function` (not a bare `_rename_params_to_qnames`) so an
        # `AgeSpecializedFunction` is wrapped into the `_SpecializedEconFunction`
        # marker the per-period builders (and `resolve_periodized_nodes`)
        # resolve; a plain function is renamed identically. A cascade merge onto
        # the collective-port function loop had dropped this call site, leaving
        # age-specialized deterministic functions unwrapped (they then reached
        # `jax.jit(...).lower()` as a bare `(*args, **kwargs)` marker).
        processed_functions[func_name] = _process_one_function(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
        )

    for func_name, func in deterministic_transition_functions.items():
        # `func_name` is `<target>__next_<state>`; its last path part is the bare
        # law name `next_<state>` and the state name drops the `next_` prefix.
        bare_law_name = tree_path_from_qname(func_name)[-1]
        state_name = bare_law_name.removeprefix("next_")
        # Provenance location keys a within-period read's merge/conflict (see
        # `_law_sources_differ`). For a law COARSE in THIS phase, stamp at the bare
        # `next_<state>` shared by all its target cells so the read MERGES; a
        # per-target law keeps its target-qualified `names_key` so the read CONFLICTS.
        # Decoupled from `names_key` (param binding), so the coarse side of a
        # map-vs-bare `Phased` merges even though the phase-union template binds its
        # params per target.
        names_key = _extract_template_names_key(func_name, regime_params_template)
        law_source_location = (
            bare_law_name if state_name in coarse_state_law_names else names_key
        )
        processed_functions[func_name] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
            names_key=names_key,
            law_source_location=law_source_location,
        )

    for func_name, func in stochastic_transition_functions.items():
        processed_functions[f"weight_{func_name}"] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
            names_key=_extract_template_names_key(func_name, regime_params_template),
        )
        processed_functions[func_name] = _get_discrete_markov_next_function(
            func=func,
            grid=flat_grids[func_name.replace("next_", "")].to_jax(),
        )

    # Transitions of continuous stochastic processes bypass the stub pipeline
    # entirely. Build weight and next functions for reachable target regimes
    # from each target's grid. Scope to genuinely reachable targets to avoid
    # spurious entries for unreachable regimes.
    #
    # A target is reachable if it is DECLARED by the regime transition
    # (`nested_transitions` keys — #407 `ec96c12`: declared reachability is the
    # single source of truth, not state laws), OR if a PER-TARGET regime
    # transition names it explicitly, OR if it is a coarse candidate. Deriving
    # from `flat_nested_transitions` alone — i.e. from ordinary (non-process)
    # state laws — was the historical defect on both counts: a target whose sole
    # content is a process state has an empty state-law bundle, so it was left
    # with no process transitions and silently dropped from the continuation,
    # and hence its continuation from E[V].
    #
    # The components are UNIONed rather than one replacing the others. The
    # asymmetry is spelled out below and runs one way: a spurious candidate
    # carries zero regime-transition probability and contributes nothing to E[V],
    # whereas omitting a genuinely-routed one silently drops a whole
    # continuation. Note this keeps `_fail_if_coarse_candidate_folds_ambiguously`
    # reachable, which a straight `reachable_targets = frozenset(nested_
    # transitions)` would have bypassed along with the fold-round4 F2 guarantee.
    #
    # A coarse `transition=func` emits a `next_regime` cell for EVERY regime
    # (the routing is decided at runtime from the returned id), so its cell keys
    # are the candidate UNIVERSE, not the transition's actual support -- which is
    # unknowable at build time. Each candidate CAN be routed to (INCLUDING the
    # source itself: a coarse transition may legitimately return its own regime
    # when that regime is still active next period), so its continuation must be
    # built: a candidate the function never returns simply carries zero
    # regime-transition probability and contributes nothing to E[V], whereas
    # OMITTING a genuinely-routed candidate silently drops its whole continuation
    # (fold-round3 F1). So the source is NOT excluded (fold-round4 F1: excluding
    # it by name erased real coarse self-transitions).
    #
    # The one thing `target != source` cannot stand in for is transition SUPPORT,
    # and support is load-bearing for a FOLDED process: `_fail_if_folded_state_
    # persists` is STRUCTURAL (it inspects built `next_<process>` edges, not
    # transition probability), so if we built a `next_<process>` continuation for
    # a folded candidate we would either accept a real persisting self-fold that
    # deletes the fold axis (F1) or reject a folded candidate the function never
    # returns (fold-round4 F2). We therefore (a) do NOT build a continuation for a
    # candidate's FOLDED process -- a folded process is consumed in its own period
    # and its stored `V` has no such axis, so no continuation is needed; and
    # (b) REJECT the ambiguous case at build time: a coarse candidate that folds a
    # process AND is active in a period immediately after the source (so the
    # coarse edge could carry that fold into a period where it persists) must be
    # declared with explicit PER-TARGET transition cells, whose support IS known,
    # so the persistence guard can validate it exactly.
    process_names = variables.process_names
    per_target_regime_targets = {
        target
        for target, cell in next_regime_cells_by_target.items()
        if not isinstance(cell, _CoarseTransitionCell)
    }
    coarse_candidate_targets = {
        target
        for target, cell in next_regime_cells_by_target.items()
        if isinstance(cell, _CoarseTransitionCell)
    }
    _fail_if_coarse_candidate_folds_ambiguously(
        source_regime_name=source_regime_name,
        source_process_names=frozenset(process_names),
        coarse_candidate_targets=coarse_candidate_targets,
        all_grids=all_grids,
        active_periods_by_regime=active_periods_by_regime,
    )
    # The per-target process grids below are scoped to the processes the *source*
    # carries: an intrinsic law is a transition from this period's value of the
    # process, so a target-only process has no from-value here to condition on.
    reachable_targets = (
        frozenset(nested_transitions)
        | {
            tree_path_from_qname(k)[0]
            for k in flat_nested_transitions
            if QNAME_DELIMITER in k
        }
        | per_target_regime_targets
        | coarse_candidate_targets
    )

    # entirely. Build weight and next functions for every graph-retained
    # continuation target's grid. Scope to the phase reachability graph's
    # retained targets for this source — not to whichever targets happen to
    # have a non-process law bundle — so a target reached solely by carrying
    # a shared process state (no other law) still gets its intrinsic
    # process transition synthesized. Read the process names off the target's
    # own grids rather than the source's variables: an IID process the source
    # does not carry is entered at its unconditional law, so it needs its
    # intrinsic transition built here too.
    continuation_targets = phase_reachability.union_targets(source=source_regime_name)
    target_process_grids: dict[
        tuple[RegimeName, ProcessName], _ContinuousStochasticProcess
    ] = {
        (user_regime, process): grid
        for user_regime, grids in all_grids.items()
        if user_regime in continuation_targets
        for process, grid in grids.items()
        if isinstance(grid, _ContinuousStochasticProcess)
        # A folded process gets no continuation transition when the SOURCE does
        # not carry it, or on a coarse candidate. Its stored V has no such axis,
        # and building the edge trips `_fail_if_folded_state_persists`, which is
        # STRUCTURAL -- it inspects built `next_<process>` edges, not transition
        # probability. This has to be decided here, not only at entry: now that
        # the process names come off the TARGET's own grids, a folded
        # target-only process is visible to synthesis for the first time, and
        # the source's variables used to hide it.
        #
        # The `process not in process_names` half is what keeps the guard armed.
        # A folded process the source DOES carry can genuinely persist, and the
        # guard needs its edge as evidence; a target-only one cannot persist,
        # because there is no prior-period axis to carry. Excluding folded
        # processes outright disarms the guard -- measured: it silences
        # test_fold_on_persisting_shock_is_rejected_at_model_processing and
        # test_fold_on_persisting_shock_reached_only_via_regime_transition_is_rejected.
        # The coarse half is fold-round4 F2, unchanged.
        if not (
            getattr(grid, "fold", False)
            and (
                process not in process_names or user_regime in coarse_candidate_targets
            )
        )
    }
    _validate_all_conditioned_processes(all_grids=all_grids)
    # A process the source carries is transitioned from its current value. One
    # it does not carry is entered at its own law -- unless the source declared
    # an explicit entry law for it, which is the more specific statement and
    # wins.
    carried_processes = set(variables.process_names)
    target_process_grids = {
        (user_regime, process): grid
        for (user_regime, process), grid in target_process_grids.items()
        if process in carried_processes
        or f"{user_regime}__next_{process}" not in flat_nested_transitions
    }
    # Only an IID process can be entered without a handoff, which
    # `_state_handoff_errors` has already enforced.
    #
    # The condition to extend here is storage: entering a process builds it a
    # next-state and weight function, and therefore an axis. A process that is
    # integrated out of the value function rather than stored on one must be
    # excluded, or entry would reintroduce exactly the axis its treatment
    # removes -- it needs no entry law for the same reason it needs no handoff.
    entered_process_grids = {
        key: grid
        for key, grid in target_process_grids.items()
        if key[1] not in carried_processes and isinstance(grid, _IIDProcess)
    }
    carried_process_grids = {
        key: grid
        for key, grid in target_process_grids.items()
        if key[1] in carried_processes
    }
    processed_functions |= (
        {
            f"weight_{user_regime}__next_{process}": _get_weights_func_for_process(
                # A state-conditioned process resolves `state_conditioned.on`
                # against the regime's own grids; without them the lookup finds
                # nothing and the build fails with a misleading "must name a
                # DiscreteGrid" error.
                name=process,
                grid=grid,
                grids=all_grids[user_regime],
            )
            for (user_regime, process), grid in carried_process_grids.items()
        }
        | {
            f"{user_regime}__next_{process}": _get_stochastic_next_function_for_process(
                name=process, grid=grid.to_jax()
            )
            for (user_regime, process), grid in carried_process_grids.items()
        }
        | {
            f"weight_{user_regime}__next_{process}": _get_entry_weights_for_process(
                name=process, grid=grid
            )
            for (user_regime, process), grid in entered_process_grids.items()
        }
        | {
            f"{user_regime}__next_{process}": _get_entry_next_for_process(
                grid=grid.to_jax()
            )
            for (user_regime, process), grid in entered_process_grids.items()
        }
    )

    process_transition_keys = {
        f"{user_regime}__next_{process}"
        for user_regime, process in target_process_grids
    }
    internal_transition = {
        func_name: processed_functions[func_name]
        for func_name in flat_nested_transitions
    } | {key: processed_functions[key] for key in process_transition_keys}

    processed_constraints: ConstraintFunctionsMapping = MappingProxyType(
        {func_name: processed_functions[func_name] for func_name in constraints}
    )
    excluded_from_functions = (
        set(flat_nested_transitions) | set(constraints) | process_transition_keys
    )
    phase_functions = MappingProxyType(
        {
            func_name: processed_functions[func_name]
            for func_name in processed_functions
            if func_name not in excluded_from_functions
        }
    )

    nested_transitions_by_target = unflatten_regime_namespace(internal_transition)
    # fold-review F2 (folded slice): a reachable target whose ONLY state is a
    # target-local folded IID process has no ordinary state law and no
    # source-carried process edge, so it is absent from
    # `nested_transitions_by_target` and would be silently dropped from
    # the continuation graph / E[V] — valuing the source action that routes to it
    # as worthless (a reversed policy). Its stored V is a SCALAR (the folded
    # axis is integrated out), so it needs no continuation coordinate: keep an
    # explicit EMPTY bundle so it stays enumerable and its scalar V enters E[V]
    # (the solve phase strips the folded axis from its interpolation info; see
    # `_build_solution_phase`). Scoped strictly to folded-only targets; the
    # general non-folded empty-bundle hole (an ordinary state reached only via
    # the regime transition) stays deferred.
    for target in reachable_targets:
        if target in fold_only_regimes and target not in nested_transitions_by_target:
            nested_transitions_by_target[target] = {}
    transitions = _wrap_transitions(nested_transitions_by_target)

    next_regime_func, next_regime_cells = _process_next_regime_cells(
        next_regime_cells_by_target=next_regime_cells_by_target,
        regime_params_template=regime_params_template,
    )

    return _CoreResult(
        functions=phase_functions,
        constraints=processed_constraints,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        next_regime_func=next_regime_func,
        next_regime_cells=next_regime_cells,
    )


def _process_next_regime_cells(
    *,
    next_regime_cells_by_target: Mapping[
        RegimeName, UserFunction | _CoarseTransitionCell
    ],
    regime_params_template: RegimeParamsTemplate,
) -> tuple[
    TransitionFunction | None, MappingProxyType[RegimeName, EconFunction] | None
]:
    """Process the canonical regime-transition cells of one phase.

    Dispatch on the cell type of the canonical per-target mapping:

    - empty mapping (terminal regime) ⇒ `(None, None)`
    - `_CoarseTransitionCell` cells ⇒ the shared underlying transition is
      processed once under the `next_regime` template key, so the engine
      evaluates it once and indexes per target
    - `MarkovTransition` cells (user per-target dict) ⇒ each cell is
      processed under its nested `template[target]["next_regime"]` branch

    Args:
        next_regime_cells_by_target: The canonical regime-transition cells,
            keyed by target regime name.
        regime_params_template: The regime's parameter template.

    Returns:
        Tuple of the processed coarse transition (`None` unless coarse) and
        the processed per-target cells (`None` unless per-target).

    """
    if not next_regime_cells_by_target:
        return None, None
    cells = tuple(next_regime_cells_by_target.values())
    first_cell = cells[0]
    if isinstance(first_cell, _CoarseTransitionCell):
        assert all(  # noqa: S101
            isinstance(cell, _CoarseTransitionCell)
            and cell.underlying is first_cell.underlying
            for cell in cells
        ), "Coarse regime-transition cells must share one underlying object."
        next_regime_func = _rename_params_to_qnames(
            func=cast("UserFunction", first_cell.underlying),
            regime_params_template=regime_params_template,
            param_key="next_regime",
        )
        return next_regime_func, None
    next_regime_cells = MappingProxyType(
        {
            target_regime_name: _rename_params_to_qnames(
                func=cast("UserFunction", cell),
                regime_params_template=regime_params_template,
                param_key=qname_from_tree_path((target_regime_name, "next_regime")),
            )
            for target_regime_name, cell in next_regime_cells_by_target.items()
        }
    )
    return None, next_regime_cells


def _build_validation_regime_transition_probs(
    *,
    declared_regime_transition: object,
    compute_regime_transition_probs: RegimeTransitionFunction,
    functions: EconFunctionsMapping,
    grids: MappingProxyType[StateOrActionName, Grid],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    flat_param_names: frozenset[str],
    enable_jit: bool,
) -> RegimeTransitionFunction:
    """Build a validation function that retains every declared target cell."""
    if not isinstance(declared_regime_transition, Mapping):
        return compute_regime_transition_probs

    _, declared_cells = _process_next_regime_cells(
        next_regime_cells_by_target=cast(
            "Mapping[RegimeName, UserFunction | _CoarseTransitionCell]",
            declared_regime_transition,
        ),
        regime_params_template=regime_params_template,
    )
    assert declared_cells is not None  # noqa: S101
    return build_regime_transition_probs_functions(
        functions=functions,
        compute_regime_transition_probs=None,
        grids=grids,
        regime_names_to_ids=regime_names_to_ids,
        flat_param_names=flat_param_names,
        is_stochastic=True,
        enable_jit=enable_jit,
        phase="solve",
        next_regime_cells=declared_cells,
    )


def _extract_phase_transitions(*, phase_slice: RegimePhaseSpec) -> _TransitionBundles:
    """Transpose one canonical phase slice into per-target transition bundles.

    The slice's `state_transitions` values and its regime transition are
    canonical per-target mappings (`canonicalize_regimes` resolved
    reachability and desugared identities), so the extraction is a pure
    transpose: bundle each target regime's `next_<state>` laws plus its
    regime-transition cell under `"next_regime"`. A target reachable through
    the regime transition alone contributes a bundle holding only its cell.
    Stochastic process transitions are handled separately during internal
    function processing.

    Args:
        phase_slice: One canonical phase slice of the regime specification.

    Returns:
        Per-target transition bundles for internal processing.

    """
    if phase_slice.regime_transition is None:
        return {}

    per_target: _TransitionBundles = {}
    for state_name, canonical in phase_slice.state_transitions.items():
        for target_regime_name, law in cast(
            "Mapping[RegimeName, UserFunction]", canonical
        ).items():
            per_target.setdefault(target_regime_name, {})[f"next_{state_name}"] = law
    for target_regime_name, cell in cast(
        "Mapping[RegimeName, UserFunction | _CoarseTransitionCell]",
        phase_slice.regime_transition,
    ).items():
        per_target.setdefault(target_regime_name, {})["next_regime"] = cell

    return per_target


def _wrap_transitions(
    transitions: dict[RegimeName, dict[TransitionFunctionName, TransitionFunction]],
) -> TransitionFunctionsMapping:
    """Wrap nested transitions dict in MappingProxyType."""
    return MappingProxyType(
        {name: MappingProxyType(inner) for name, inner in transitions.items()}
    )


def _get_stochastic_transition_names(
    *,
    state_transitions: Mapping[StateName, object],
    variables: Variables,
) -> frozenset[TransitionFunctionName]:
    """Compute stochastic transition names from one phase's state transitions.

    Args:
        state_transitions: One phase's `state_transitions` slice.
        variables: States and actions of the regime with kind/topology/process tags.

    Returns:
        Frozenset of stochastic transition function names (e.g., "next_health").

    """
    markov_state_names: set[StateName] = set()
    for name, raw in state_transitions.items():
        if isinstance(raw, MarkovTransition) or (
            isinstance(raw, Mapping)
            and any(isinstance(v, MarkovTransition) for v in raw.values())
        ):
            markov_state_names.add(name)
    return frozenset(
        f"next_{name}" for name in markov_state_names | set(variables.process_names)
    )


def _declared_next_state_names(user_regime: UserRegime) -> frozenset[str]:
    """Engine `next_<state>` node names for a regime (own + target-only states).

    A `next_<state>` DAG node exists for every state the regime grids AND every
    target-only state it hands off (a `state_transitions` key not in `states`). This
    is phase-invariant and read from the USER declaration, so it still lists a
    target-only state whose carrier does not grid it in some phase — exactly the
    no-producer case the read guard must catch. A user may LEGALLY name a current
    state/action `next_stock`; its own node is `next_next_stock`, so `next_stock`
    itself is absent here and a read of it is not mistaken for a next-state node.
    """
    return frozenset(
        f"next_{name}"
        for name in set(user_regime.states) | set(user_regime.state_transitions)
    )


def _phase_coarse_state_law_names(
    *, user_regime: UserRegime, phase: Literal["solve", "simulate"]
) -> frozenset[StateName]:
    """State names whose deterministic law is COARSE (bare) in the given phase.

    A coarse law binds ONE shared law across all its target cells, so a within-period
    read of its `next_<state>` is unambiguous and must MERGE across targets; a
    per-target dict binds a distinct cell per target, so the read is target-dependent
    and CONFLICTS. `Phased` resolves per phase — the side facing `phase` decides.

    Read from the USER declaration shape (bare callable vs `Mapping`), because after
    canonicalization a broadcast coarse law and a per-target dict that reuses one
    callable are indistinguishable by object identity — the very ambiguity the
    provenance stamp exists to resolve. `Phased(solve=coarse, simulate={...})` (and its
    mirror) is therefore coarse in exactly one phase, which is why the location must be
    keyed per phase rather than off the phase-union template.
    """
    coarse: set[StateName] = set()
    for state_name, raw in user_regime.state_transitions.items():
        side: object = raw
        if isinstance(raw, Phased):
            side = raw.solve if phase == "solve" else raw.simulate
        if side is not None and not isinstance(side, Mapping):
            coarse.add(state_name)
    return frozenset(coarse)


def _process_one_function(
    *,
    func: UserFunction | PeriodizedUserFunction,
    regime_params_template: RegimeParamsTemplate,
    param_key: str,
    names_key: str | None = None,
) -> EconFunction:
    """Rename a function's params to qnames, periodizing an age-specialized node.

    A plain function is renamed once. A `PeriodizedUserFunction` (the normalized
    form of an `AgeSpecializedFunction`, already resolved to concrete per-period
    callables) becomes a `PeriodizedEconFunction`: each distinct signature's
    concrete function is renamed once under the **same** `param_key` / `names_key`,
    so every period carries identical qnames — sound because the call signature is
    age-invariant by contract. No user factory is retained or called here.
    """
    if isinstance(func, PeriodizedUserFunction):
        processed_by_signature: dict[Hashable, EconFunction] = {}
        for period, concrete in func.concrete_by_period.items():
            signature = func.signature_by_period[period]
            if signature not in processed_by_signature:
                processed_by_signature[signature] = _rename_params_to_qnames(
                    func=concrete,
                    regime_params_template=regime_params_template,
                    param_key=param_key,
                    names_key=names_key,
                )
        representative_signature = func.signature_by_period[
            min(func.signature_by_period)
        ]
        return PeriodizedEconFunction(
            representative=processed_by_signature[representative_signature],
            function_by_signature=MappingProxyType(processed_by_signature),
            signature_by_period=func.signature_by_period,
        )
    return _rename_params_to_qnames(
        func=func,
        regime_params_template=regime_params_template,
        param_key=param_key,
        names_key=names_key,
    )


def _rename_params_to_qnames(
    *,
    func: UserFunction,
    regime_params_template: RegimeParamsTemplate,
    param_key: str,
    names_key: str | None = None,
    law_source_location: str | None = None,
) -> EconFunction:
    """Rename function params to qualified names using dags.signature.rename_arguments.

    E.g., risk_aversion -> utility__risk_aversion.

    Args:
        func: The user function.
        regime_params_template: The parameter template for the regime.
        param_key: The qname prefix the renamed params carry (e.g., "utility",
            "retired__next_wealth").
        names_key: The template key under which the param names live, when it
            differs from `param_key` — a coarse law's names sit at the bare
            law name while its params bind per target. Defaults to
            `param_key`.
        law_source_location: The location recorded in the provenance stamp used
            by `_law_sources_differ` for the within-period conflict guard, when
            it must differ from `names_key`. This decouples a `next_<state>`
            cell's merge/conflict identity from where its params BIND: the caller
            passes the bare `next_<state>` for a law that is COARSE in THIS phase
            (so its cells share a location and MERGE) and the target-qualified
            `names_key` for a per-target law (so its cells CONFLICT), even in a
            `Phased` map-vs-bare law whose phase-union template would otherwise
            give the coarse side per-target leaves. Defaults to `names_key`.

    Returns:
        The function with renamed parameters.

    """
    # Per-target keys are qnames (`<target>__<func>`) addressing a nested
    # template branch; walk the tree path instead of subscripting directly.
    branch: Mapping[str, object] = regime_params_template
    for part in tree_path_from_qname(names_key if names_key is not None else param_key):
        branch = cast("Mapping[str, object]", branch[part])
    # Scope the rename to THIS cell's OWN parameters. A `Phased` state-transition
    # cell's template branch is the UNION of both phases' params, so a cell that is
    # parameter-free (or coarse) in its own phase must not be renamed/stamped merely
    # because the OTHER phase contributed a param to the union — that is the false
    # conflict of the map-vs-bare and asymmetric both-per-target cases.
    own_args = get_union_of_args([func])
    param_names = [p for p in branch if p in own_args]
    if not param_names:
        # No engine wrapper is added, so no provenance stamp: the cell's own object
        # identity distinguishes one coarse law (the same object broadcast to every
        # target) from distinct per-target laws. A parameter-free law reused across
        # per-target cells is genuinely identical (no parameter can differ), so
        # shared identity is the right verdict here. See `_law_sources_differ`.
        return cast("EconFunction", func)
    mapper = {p: qname_from_tree_path((param_key, p)) for p in param_names}
    renamed = rename_arguments(func, mapper=mapper)
    # Stamp the engine-created wrapper with (user law, provenance location). The
    # conflict guard compares this token instead of unwrapping (which cannot tell the
    # engine's rename layer from a user's own `rename_arguments` wrapper). The location
    # is decisive: a COARSE law (this phase) is stamped at the BARE `next_<state>`,
    # shared by every target cell (same location -> merge), whereas a PER-TARGET dict's
    # cells carry a TARGET-QUALIFIED location, distinct per cell -- so cells that reuse
    # the SAME callable still get different tokens and remain a conflict. The caller
    # supplies `law_source_location` from this phase's declaration shape; it defaults to
    # `names_key`. Keep the base user law, not a nested token, as origin.
    location = law_source_location
    if location is None:
        location = names_key if names_key is not None else param_key
    prior = getattr(func, LAW_SOURCE_ATTR, None)
    base = prior[0] if isinstance(prior, tuple) else (func if prior is None else prior)
    setattr(renamed, LAW_SOURCE_ATTR, (base, location))
    return cast("EconFunction", renamed)


def _engine_flat_param_names(
    *,
    regime_params_template: RegimeParamsTemplate,
    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]],
) -> frozenset[str]:
    """Return the regime's flat param names in the engine's binding vocabulary.

    Template names whose function key has a granular expansion are replaced
    by their per-target spellings (`<target>__<law>__<param>`); everything
    else passes through unchanged.
    """
    names: set[str] = set()
    for name in get_flat_param_names(regime_params_template):
        path = tree_path_from_qname(name)
        prefixes = granular_param_expansions.get(path[0]) if len(path) > 1 else None
        if prefixes:
            names.update(
                qname_from_tree_path((prefix, path[-1])) for prefix in prefixes
            )
        else:
            names.add(name)
    return frozenset(names)


def _granular_param_expansions(
    *,
    nested_transitions_by_phase: tuple[_TransitionBundles, ...],
    regime_params_template: RegimeParamsTemplate,
    declaration_param_expansions: Mapping[FunctionName, tuple[str, ...]],
) -> MappingProxyType[FunctionName, tuple[str, ...]]:
    """Map each coarse-template law key to its granular qname prefixes.

    A state law whose params the template keys coarsely binds them per target
    in the engine; this collects, across the given phase bundles, every
    `<target>__<law>` prefix for laws whose names live at the bare law name
    (mirroring `_extract_template_names_key`) and that carry params at all.
    Canonical flat params materialize one shared leaf per prefix.
    """
    expansions = {
        law_name: set(prefixes)
        for law_name, prefixes in declaration_param_expansions.items()
    }
    for bundles in nested_transitions_by_phase:
        for target_regime_name, bundle in bundles.items():
            for law_name in bundle:
                if law_name == "next_regime":
                    continue
                qname = qname_from_tree_path((target_regime_name, law_name))
                names_key = _extract_template_names_key(qname, regime_params_template)
                if names_key != qname and regime_params_template.get(names_key):
                    expansions.setdefault(names_key, set()).add(qname)
    return MappingProxyType(
        {law_name: tuple(sorted(v)) for law_name, v in expansions.items()}
    )


def _declaration_param_expansions(
    *,
    source_regime_name: RegimeName,
    specs: Mapping[RegimeName, PhasedRegimeSpec],
    all_regime_names: frozenset[RegimeName],
    regime_params_template: RegimeParamsTemplate,
) -> MappingProxyType[FunctionName, tuple[str, ...]]:
    """Retain granular parameter names as declaration provenance."""
    expansions: dict[FunctionName, set[str]] = {}
    for phase_name in ("solution", "simulation"):
        source_slice: RegimePhaseSpec = getattr(specs[source_regime_name], phase_name)
        candidate_targets = candidate_targets_from_transition(
            transition=source_slice.regime_transition,
            all_regime_names=all_regime_names,
        )
        for state_name, law in source_slice.state_transitions.items():
            if law is None or isinstance(law, Mapping):
                continue
            law_name = f"next_{state_name}"
            if not regime_params_template.get(law_name):
                continue
            for target in candidate_targets:
                target_slice: RegimePhaseSpec = getattr(specs[target], phase_name)
                if state_name in target_slice.grid_states:
                    expansions.setdefault(law_name, set()).add(
                        qname_from_tree_path((target, law_name))
                    )
    return MappingProxyType(
        {law_name: tuple(sorted(prefixes)) for law_name, prefixes in expansions.items()}
    )


def _extract_template_names_key(
    func_name: str,
    regime_params_template: RegimeParamsTemplate,
) -> str:
    """Extract the template key under which a function's param names live.

    The template mirrors the user's coarseness — a per-target dict yields
    params nested under the target (`template[target_regime][func]`), a broadcast
    law a single coarse `next_<state>` key — while the engine-side function
    names are always target-prefixed (canonical form). The template therefore
    decides where the names live:

    - "work__next_health" with `template["work"]["next_health"]` present
      (user wrote a per-target dict) ⇒ "work__next_health"
    - "work__next_wealth" without such a branch (broadcast law) ⇒
      "next_wealth"
    - unprefixed names ⇒ unchanged

    Either way the params *bind* under the engine function's qname; for
    broadcast laws the canonical flat params materialize one shared leaf
    per target (`Regime.granular_param_expansions`).
    """
    path = tree_path_from_qname(func_name)
    if len(path) > 1:
        suffix = qname_from_tree_path(path[1:])
        target_branch = regime_params_template.get(path[0])
        if isinstance(target_branch, Mapping) and isinstance(
            target_branch.get(suffix), Mapping
        ):
            return func_name
        return suffix
    return func_name


def _get_discrete_markov_next_function(
    *, func: UserFunction, grid: Int1D
) -> UserFunction:
    @with_signature(args=None, return_annotation="Int1D")
    @functools.wraps(func)
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ANN401, ARG001
        return grid

    return next_func


def _get_stochastic_next_function_for_process(
    *, name: str, grid: Float1D
) -> UserFunction:
    """Get function returning the indices in the vf arr of the next process states."""

    @with_signature(args={f"{name}": "ContinuousState"}, return_annotation="Int1D")
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return jnp.arange(grid.shape[0], dtype=jnp.int32)

    return next_func


def _process_family(grid: _ContinuousStochasticProcess) -> Family:
    """Map a supported process to its state-conditioned family (audit F2).

    Only families whose transition CDF carries ``sigma`` can express a fixed-node,
    state-conditioned ``sigma``: CDF-binned ``NormalIIDProcess`` and
    ``TauchenAR1Process``. Gauss-Hermite IID (nodes scale with ``sigma``) and
    Rouwenhorst (``rho``-only transition) are rejected at construction.
    """
    if isinstance(grid, NormalIIDProcess | TauchenAR1Process) and grid.gauss_hermite:
        # Blanket rejection, per the stated v1 scope (code-review F6). GH-placed nodes
        # are derived from the *grid* sigma, so a state-conditioned sigma would bin a
        # different law on a quadrature rule chosen for another one; untested here.
        msg = (
            "state-conditioned sigma is not supported for Gauss-Hermite node placement "
            "(the nodes are built from the grid sigma); use gauss_hermite=False "
            "(CDF binning)."
        )
        raise ModelInitializationError(msg)
    if isinstance(grid, NormalIIDProcess):
        return "iid_normal"
    if isinstance(grid, TauchenAR1Process):
        return "tauchen"
    msg = (
        "state-conditioned sigma is only supported for CDF-binned NormalIIDProcess "
        f"and TauchenAR1Process (audit F2); got {type(grid).__name__}."
    )
    raise ModelInitializationError(msg)


def _validate_conditioning_codes_agree_across_regimes(
    *,
    on: str,
    all_grids: Mapping[RegimeName, Mapping[StateOrActionName, Grid]],
) -> None:
    """Every regime carrying the conditioning state must map categories to codes alike.

    The sigma array is ordered by the *target* regime's grid, but the code gathering
    it is the *current* (source) state's. Those agree only if the category-to-code maps
    do; a regime that relabels `{low: 0, high: 1}` to `{high: 0, low: 1}` would silently
    swap the two volatilities (code-review F4). Rather than thread the source grid
    through every seam, v1 requires one shared map and rejects otherwise.

    Round-2 review F3 noted this global scan is not edge-precise (it fails a
    state-less reachable source late rather than at the guard, and over-rejects
    disconnected components reusing the name), but established **no silent policy
    error**. Kept deliberately for v1: the conservative rule is safe, and the
    per-edge alternative would need source-reachability plumbing that could itself
    introduce the sigma-swap this prevents. Accepted limitation, not an oversight.
    """
    maps = {
        regime: dict(zip(grid.categories, grid.codes, strict=True))
        for regime, grids in all_grids.items()
        if isinstance(grid := grids.get(on), DiscreteGrid)
    }
    distinct = {tuple(sorted(m.items())) for m in maps.values()}
    if len(distinct) > 1:
        msg = (
            f"state_conditioned.on='{on}' must map categories to the same integer "
            f"codes in every regime that carries it, because the per-category sigma is "
            f"indexed by that code; got {maps}."
        )
        raise ModelInitializationError(msg)


def _validate_conditioned_sigmas(by: Mapping[str, float]) -> None:
    """Every per-category sigma must be a finite, strictly positive number.

    ``None``, ``NaN`` and ``inf`` all sail through a bare ``v <= 0`` test and then
    poison every transition row silently (code-review F3).
    """
    bad = {
        k: v
        for k, v in by.items()
        if v is None or not math.isfinite(float(v)) or float(v) <= 0.0
    }
    if bad:
        msg = f"state_conditioned.by values must be finite positive sigmas; got {bad}"
        raise ModelInitializationError(msg)


#: A CDF row bins on node midpoints, so it needs at least one interior edge.
_MIN_CONDITIONED_NODES = 2


def _validate_conditioned_grid_is_fixed(
    *, name: str, grid: _ContinuousStochasticProcess
) -> None:
    """A state-conditioned process must have every grid parameter fixed at build time.

    The conditioned branch is chosen *before* the runtime-parameter mechanism, so a
    parameter left for runtime is never bound: ``get_gridpoints()`` returns all-NaN and
    the closure captures those nodes permanently (code-review F3). Reject instead.
    """
    if not grid.is_fully_specified:
        missing = ", ".join(sorted(grid.params_to_pass_at_runtime))
        msg = (
            f"state-conditioned process '{name}' requires every grid parameter fixed "
            f"at construction (v1); {missing} would be passed at runtime. Pass them to "
            f"{type(grid).__name__}(...), or drop state_conditioned."
        )
        raise ModelInitializationError(msg)
    nodes = grid.get_gridpoints()
    if not bool(jnp.all(jnp.isfinite(nodes))):
        msg = (
            f"state-conditioned process '{name}' resolved to non-finite nodes: {nodes}"
        )
        raise ModelInitializationError(msg)
    # Finiteness is not enough. The direct-CDF row bins on the MIDPOINTS of these nodes,
    # so a strictly increasing axis of >= 2 points is what makes the CDF differences
    # probabilities at all (code-review round 2, F2). All of this is reachable from the
    # public API: `sigma=-0.3` or `n_std=-3.0` yields a DESCENDING axis (pylcm does not
    # require a positive sigma), `sigma=0.0` collapses every node, and a large `mu` in
    # float32 rounds them together. The descending case is the dangerous one — the row
    # still sums to 1.0, so a row-sum check passes while masses go negative and a
    # continuation decision flips sign.
    if nodes.ndim != 1 or nodes.shape[0] < _MIN_CONDITIONED_NODES:
        msg = (
            f"state-conditioned process '{name}' needs a 1-D axis of at least "
            f"{_MIN_CONDITIONED_NODES} nodes; got shape {nodes.shape}. A single node "
            f"leaves the CDF row no bin edges."
        )
        raise ModelInitializationError(msg)
    if not bool(jnp.all(jnp.diff(nodes) > 0)):
        msg = (
            f"state-conditioned process '{name}' resolved to a node axis that is not "
            f"strictly increasing: {nodes}. Midpoint-CDF binning would return negative "
            f"transition masses (which still sum to 1.0, so this fails silently). "
            f"Check for a negative or zero sigma / n_std, or nodes collapsed by "
            f"float32 precision at a large mu."
        )
        raise ModelInitializationError(msg)


def _validate_all_conditioned_processes(
    *, all_grids: Mapping[RegimeName, Mapping[StateOrActionName, Grid]]
) -> None:
    """Check every state-conditioned declaration in the model.

    The domain is EVERY regime, deliberately not the continuation targets the
    transitions are built over. That construction scope is the phase-reachability
    graph's retained target set -- correct for deciding which transitions to
    synthesize, wrong as a validation domain: a regime whose only edge the temporal
    activity intersection drops would never be checked, and an invalid declaration
    would then solve silently. Every check is a property of the declaration alone, so
    reachability cannot turn an invalid one valid.
    """
    for regime_grids in all_grids.values():
        for process_name, process_grid in regime_grids.items():
            if not isinstance(process_grid, _ContinuousStochasticProcess):
                continue
            state_conditioned = process_grid.state_conditioned
            if state_conditioned is None:
                continue
            _validate_conditioned_process(
                name=process_name,
                grid=process_grid,
                sc=state_conditioned,
                grids=regime_grids,
            )
            _validate_conditioning_codes_agree_across_regimes(
                on=state_conditioned.on, all_grids=all_grids
            )


def _validate_conditioned_process(
    *,
    name: str,
    grid: _ContinuousStochasticProcess,
    sc: StateConditioned,
    grids: Mapping[StateOrActionName, Grid],
) -> tuple[DiscreteGrid, Family]:
    """Structural checks on one state-conditioned declaration.

    Separated from `_get_conditioned_weights_func` so that validity is not a side
    effect of *building* the weights function. Every check here is a property of the
    declaration alone -- which grid `on` names, whether the sigmas are usable, whether
    the process family admits conditioning, whether the node grid is fixed -- so none
    of them depends on the process ever being reached.

    Returns the conditioning grid and the process family, both of which the builder
    needs, so the checks are not repeated.
    """
    conditioning_grid = grids.get(sc.on)
    if not isinstance(conditioning_grid, DiscreteGrid):
        msg = (
            f"state_conditioned.on='{sc.on}' must name a DiscreteGrid state in the "
            f"same regime as the process."
        )
        raise ModelInitializationError(msg)
    _validate_conditioned_sigmas(sc.by)
    family = _process_family(grid)
    _validate_conditioned_grid_is_fixed(name=name, grid=grid)
    return conditioning_grid, family


def _get_conditioned_weights_func(
    *,
    name: str,
    grid: _ContinuousStochasticProcess,
    sc: StateConditioned,
    grids: Mapping[StateOrActionName, Grid],
) -> UserFunction:
    """Weights function for a state-conditioned process (direct-CDF, audit F1/F5/F6).

    The transition row is computed DIRECTLY at the from-value on the FIXED common nodes
    (from the scalar ``sigma``), with the per-regime ``sigma`` gathered by the
    conditioning state's integer code. No precomputed-row interpolation (F1).
    """
    conditioning_grid, family = _validate_conditioned_process(
        name=name, grid=grid, sc=sc, grids=grids
    )
    nodes = grid.get_gridpoints()
    sigma_by_code = sigma_array_by_code(conditioning_grid, sc.by)
    fixed = dict(grid.params)
    mu_fixed, rho_fixed = fixed["mu"], fixed.get("rho")

    args = {name: "ContinuousState", sc.on: "DiscreteState"}

    @with_signature(args=args, return_annotation="FloatND", enforce=False)
    def weights_func_conditioned(*a: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
        sigma = gather_sigma(sigma_by_code, kwargs[sc.on])
        return conditioned_row(
            family=family,
            nodes=nodes,
            sigma=sigma,
            from_value=kwargs[name],
            mu=mu_fixed,
            rho=rho_fixed,
        )

    return weights_func_conditioned


def _get_weights_func_for_process(
    *,
    name: str,
    grid: _ContinuousStochasticProcess,
    grids: Mapping[StateOrActionName, Grid] = MappingProxyType({}),
) -> UserFunction:
    """Get function that uses linear interpolation to calculate the process weights.

    For processes whose params are supplied at runtime, the grid points and
    transition probabilities are computed inside JIT from those runtime params. For a
    state-conditioned process the row is instead computed directly at the from-value
    (``_get_conditioned_weights_func``), and `grids` is that regime's grid mapping, from
    which the conditioning state is resolved.

    """
    sc = grid.state_conditioned
    if sc is not None:
        # `sc.on` may be absent from `grids`; the callee raises a clear message naming
        # the offending state.
        return _get_conditioned_weights_func(name=name, grid=grid, sc=sc, grids=grids)

    if grid.params_to_pass_at_runtime:
        n_points = grid.n_points
        fixed_params = dict(grid.params)
        runtime_param_names = {
            qname_from_tree_path((name, p)): p for p in grid.params_to_pass_at_runtime
        }
        args = {
            name: "ContinuousState",
            **dict.fromkeys(runtime_param_names, "FloatND"),
        }

        @with_signature(args=args, return_annotation="FloatND", enforce=False)
        def weights_func_runtime(*a: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
            # `grid.params` is canonical (0-d JAX scalars) from its own
            # boundary cast; `kwargs` arrive as JAX tracers from JIT.
            process_kw: dict[str, FloatND | IntND] = {
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
            gridpoints = grid.compute_gridpoints(**process_kw)
            transition_probs = grid.compute_transition_probs(**process_kw)
            coord = get_irreg_coordinate(value=kwargs[name], points=gridpoints)
            return map_coordinates(
                input=transition_probs,
                coordinates=[
                    jnp.full(n_points, fill_value=coord),
                    jnp.arange(n_points, dtype=jnp.int32),
                ],
            )

        return weights_func_runtime

    gridpoints = grid.get_gridpoints()
    transition_probs = grid.get_transition_probs()

    @with_signature(
        args={f"{name}": "ContinuousState"},
        return_annotation="FloatND",
        enforce=False,
    )
    def weights_func(*args: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
        coordinate = get_irreg_coordinate(value=kwargs[f"{name}"], points=gridpoints)
        return map_coordinates(
            input=transition_probs,
            coordinates=[
                jnp.full(grid.n_points, fill_value=coordinate),
                jnp.arange(grid.n_points, dtype=jnp.int32),
            ],
        )

    return weights_func


def _get_entry_next_for_process(*, grid: Float1D) -> UserFunction:
    """Get the next-index function for a process the source does not carry.

    The target's nodes are the process's own, and no current value selects
    among them, so the signature is empty.
    """

    @with_signature(args={}, return_annotation="Int1D")
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return jnp.arange(grid.shape[0], dtype=jnp.int32)

    return next_func


def _get_entry_weights_for_process(*, name: str, grid: _IIDProcess) -> UserFunction:
    """Get the entry weights of an IID process the source does not carry.

    Every row of an IID transition matrix is the same unconditional
    distribution, so the entry weights are row zero and no current value is
    needed to choose the row.
    """
    if grid.params_to_pass_at_runtime:
        fixed_params = dict(grid.params)
        runtime_param_names = {
            qname_from_tree_path((name, p)): p for p in grid.params_to_pass_at_runtime
        }

        @with_signature(
            args=dict.fromkeys(runtime_param_names, "FloatND"),
            return_annotation="FloatND",
            enforce=False,
        )
        def entry_weights_runtime(*a: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
            process_kw: dict[str, FloatND | IntND] = {
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
            return grid.compute_transition_probs(**process_kw)[0]

        return entry_weights_runtime

    transition_probs = grid.get_transition_probs()

    @with_signature(args={}, return_annotation="FloatND", enforce=False)
    def entry_weights(*args: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
        return transition_probs[0]

    return entry_weights


def _validate_categoricals(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Validate that simple transitions don't span mismatched discrete grids.

    When a non-per-target-dict transition is used for a `DiscreteGrid` state, the same
    function is applied to all target regimes. If a target regime has a different number
    of categories for that state, JAX silently clips indices producing wrong results.

    Also validates that the `ordered` flag is consistent across regimes for the same
    discrete state variable. Mixed ordered flags (one True, one False) are not allowed.

    When both regimes are ordered with different categories, the per-regime orderings
    are merged via topological sort. If the merge is ambiguous or contradictory, an
    error is raised.

    Raises:
        ModelInitializationError: If a category count mismatch or ordered flag
            inconsistency is found.

    """
    error_messages: list[str] = []

    for source_name, source_regime in user_regimes.items():
        if source_regime.terminal:
            continue

        for state_name, raw in source_regime.state_transitions.items():
            source_grid = _get_simple_transition_discrete_grid(
                source_regime, state_name, raw
            )
            if source_grid is None:
                continue

            for target_regime_name, target_regime in user_regimes.items():
                target_grid = target_regime.states.get(state_name)
                if not isinstance(target_grid, DiscreteGrid):
                    continue

                if source_grid.categories != target_grid.categories:
                    error_messages.append(
                        f"Discrete state '{state_name}' in regime '{source_name}' "
                        f"has categories {source_grid.categories}, but regime "
                        f"'{target_regime_name}' has categories "
                        f"{target_grid.categories}. A single transition function "
                        f"cannot map between different category sets — use a "
                        f"per-target dict in state_transitions to specify the "
                        f"mapping for each target regime.",
                    )

    # Validate ordered flag consistency across regimes
    _validate_ordered_flags(user_regimes, error_messages)

    if error_messages:
        raise ModelInitializationError(format_messages(error_messages))


def compute_merged_discrete_categories(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> tuple[dict[str, tuple[str, ...]], dict[str, bool]]:
    """Compute merged categories and ordered flags for all discrete variables.

    Returns:
        Tuple of (categories dict, ordered_flags dict).

    """
    var_grids: dict[str, list[tuple[str, DiscreteGrid]]] = {}
    for regime_name, user_regime in user_regimes.items():
        for var_name, grid in {**user_regime.states, **user_regime.actions}.items():
            if isinstance(grid, DiscreteGrid):
                var_grids.setdefault(var_name, []).append((regime_name, grid))

    categories: dict[str, tuple[str, ...]] = {}
    ordered_flags: dict[str, bool] = {}
    for var_name, entries in var_grids.items():
        first_grid = entries[0][1]
        ordered_flags[var_name] = first_grid.ordered

        if len(entries) == 1 or not first_grid.ordered:
            categories[var_name] = first_grid.categories
            continue

        all_cats = [grid.categories for _, grid in entries]
        if len(set(all_cats)) <= 1:
            categories[var_name] = first_grid.categories
            continue

        merged = _merge_ordered_categories(
            [(rn, grid.categories) for rn, grid in entries]
        )
        # Validation already passed, so merge must succeed
        assert merged is not None  # noqa: S101
        categories[var_name] = merged

    return categories, ordered_flags


def _validate_ordered_flags(
    user_regimes: Mapping[RegimeName, UserRegime],
    error_messages: list[str],
) -> None:
    """Validate that the ordered flag is consistent for each discrete variable.

    For each discrete state/action variable that appears in multiple regimes:
    - Mixed ordered flags (True in one, False in another) -> error.
    - Both ordered with different categories -> merge via topological sort; ambiguous
      or contradictory merges -> error.
    """
    # Collect per-variable: list of (regime_name, grid)
    var_grids: dict[str, list[tuple[str, DiscreteGrid]]] = {}
    for regime_name, user_regime in user_regimes.items():
        for var_name, grid in {**user_regime.states, **user_regime.actions}.items():
            if isinstance(grid, DiscreteGrid):
                var_grids.setdefault(var_name, []).append((regime_name, grid))

    for var_name, entries in var_grids.items():
        if len(entries) < 2:  # noqa: PLR2004
            continue

        ordered_flags = {grid.ordered for _, grid in entries}
        if len(ordered_flags) > 1:
            regime_details = ", ".join(
                f"'{rn}' (ordered={g.ordered})" for rn, g in entries
            )
            error_messages.append(
                f"Discrete variable '{var_name}' has inconsistent ordered flags "
                f"across regimes: {regime_details}. All regimes must agree on "
                f"whether the variable is ordered or unordered.",
            )
            continue

        is_ordered = next(iter(ordered_flags))
        if not is_ordered:
            continue

        # Both ordered — check if categories differ and need merging
        all_categories = [grid.categories for _, grid in entries]
        if len(set(all_categories)) <= 1:
            continue

        # Attempt topological sort merge
        merged = _merge_ordered_categories(
            [(rn, grid.categories) for rn, grid in entries]
        )
        if merged is None:
            regime_details = ", ".join(
                f"'{rn}': {list(g.categories)}" for rn, g in entries
            )
            error_messages.append(
                f"Discrete variable '{var_name}' is ordered in multiple regimes "
                f"with different categories that cannot be merged into a unique "
                f"total order. Regime orderings: {regime_details}.",
            )


def _merge_ordered_categories(
    regime_categories: list[tuple[str, tuple[str, ...]]],
) -> tuple[str, ...] | None:
    """Merge per-regime category orderings into a total order via topological sort.

    Each regime contributes a chain of ordering constraints from its field declaration
    order. Returns the unique total order if one exists, or None if ambiguous or
    contradictory.
    """
    edges, all_nodes, in_degree = _build_ordering_graph(regime_categories)
    return _unique_topological_sort(edges, all_nodes, in_degree)


def _build_ordering_graph(
    regime_categories: list[tuple[str, tuple[str, ...]]],
) -> tuple[dict[str, set[str]], set[str], dict[str, int]]:
    """Build a directed graph of ordering constraints from regime categories."""
    edges: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    in_degree: dict[str, int] = defaultdict(int)

    for _regime_name, categories in regime_categories:
        for cat in categories:
            all_nodes.add(cat)
            if cat not in in_degree:
                in_degree[cat] = 0
        for i in range(len(categories) - 1):
            a, b = categories[i], categories[i + 1]
            if b not in edges[a]:
                edges[a].add(b)
                in_degree[b] += 1

    return edges, all_nodes, in_degree


def _unique_topological_sort(
    edges: dict[str, set[str]],
    all_nodes: set[str],
    in_degree: dict[str, int],
) -> tuple[str, ...] | None:
    """Return the unique topological order, or None if ambiguous or cyclic."""
    queue = [n for n in all_nodes if in_degree[n] == 0]
    result: list[str] = []

    while queue:
        if len(queue) > 1:
            return None
        node = queue[0]
        queue = []
        result.append(node)
        for neighbor in sorted(edges.get(node, set())):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(all_nodes):
        return None

    return tuple(result)


def _get_simple_transition_discrete_grid(
    user_regime: UserRegime,
    state_name: StateName,
    raw: object,
) -> DiscreteGrid | None:
    """Return the source DiscreteGrid for a simple transition.

    Returns None if the transition is a per-target dict, an identity law
    (fixed state), not a DiscreteGrid, or the state is not present in the
    source regime.

    A `Phased` entry is unwrapped: the source grid is the same for both variants, so
    the only question is whether either variant is a *simple* (broadcast) law. The grid
    is returned when at least one variant is — that variant is the one that could
    silently clip — and None only when every variant handles targets explicitly. Left
    wrapped, a `Phased` of two per-target dicts would be mistaken for one broadcast law
    and rejected for the very category difference the dicts exist to express.

    """
    if isinstance(raw, Phased):
        variants = [
            variant
            for variant in (raw.solve, raw.simulate)
            if _get_simple_transition_discrete_grid(user_regime, state_name, variant)
            is not None
        ]
        if not variants:
            return None
        raw = variants[0]
    # Per-target dicts handle category differences explicitly
    if isinstance(raw, Mapping) and not isinstance(raw, MarkovTransition):
        return None
    # An identity law (fixed state) only maps within its own regime
    if isinstance(raw, _IdentityTransition):
        return None
    # Target-only state — no source grid to compare
    if state_name not in user_regime.states:
        return None
    source_grid = user_regime.states[state_name]
    return source_grid if isinstance(source_grid, DiscreteGrid) else None


def build_regime_transition_probs_functions(
    *,
    functions: EconFunctionsMapping,
    compute_regime_transition_probs: TransitionFunction | None,
    grids: MappingProxyType[StateOrActionName, Grid],
    regime_names_to_ids: RegimeNamesToIds,
    flat_param_names: frozenset[str],
    is_stochastic: bool,
    enable_jit: bool,
    phase: Literal["solve", "simulate"],
    next_regime_cells: MappingProxyType[RegimeName, EconFunction] | None = None,
) -> RegimeTransitionFunction | VmappedRegimeTransitionFunction:
    """Build a regime transition probability function for the given phase.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        compute_regime_transition_probs: The user's coarse next_regime
            function; `None` for per-target regime transitions.
        grids: Immutable mapping of state and action variable names to grid objects.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        flat_param_names: Frozenset of flat parameter names in the engine's
            binding vocabulary.
        is_stochastic: Whether the regime transition is stochastic.
        enable_jit: Whether to JIT-compile the functions.
        phase: Which phase to build for.
        next_regime_cells: Per-target regime transition probability functions;
            `None` for coarse regime transitions.

    """
    if next_regime_cells is not None:
        wrapped_regime_transition_probs = _assemble_granular_regime_transition_probs(
            next_regime_cells=next_regime_cells
        )
    else:
        if compute_regime_transition_probs is None:
            msg = "Either a coarse regime transition or per-target cells is required."
            raise ModelInitializationError(msg)
        # Wrap deterministic next_regime to return one-hot probability array
        if is_stochastic:
            probs_func = compute_regime_transition_probs
        else:
            probs_func = _wrap_deterministic_regime_transition(
                func=compute_regime_transition_probs,
                regime_names_to_ids=regime_names_to_ids,
            )

        # Wrap to convert array output to dict format
        wrapped_regime_transition_probs = _wrap_regime_transition_probs(
            func=probs_func, regime_names_to_ids=regime_names_to_ids
        )

    functions_pool = dict(functions) | {
        "regime_transition_probs": wrapped_regime_transition_probs
    }

    next_regime = concatenate_functions(
        functions=functions_pool,
        targets="regime_transition_probs",
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )
    if phase == "solve":
        return jax.jit(next_regime) if enable_jit else next_regime

    sig_args = list(inspect.signature(next_regime).parameters)

    # We do this because a transition function without any parameters will throw
    # an error with vmap
    next_regime_accepting_all = with_signature(
        next_regime,
        args=sig_args + [state for state in grids if state not in sig_args],
    )

    next_regime_vmapped = vmap_1d(
        func=next_regime_accepting_all,
        variables=_get_vmap_params(
            all_args=tuple(inspect.signature(next_regime_accepting_all).parameters),
            flat_param_names=flat_param_names,
        ),
    )

    return jax.jit(next_regime_vmapped) if enable_jit else next_regime_vmapped


def _assemble_granular_regime_transition_probs(
    *,
    next_regime_cells: MappingProxyType[RegimeName, EconFunction],
) -> Callable[..., MappingProxyType[RegimeName, FloatND]]:
    """Assemble per-target probability cells into the probs-dict contract.

    Produces the same regime-name → probability mapping that
    `_wrap_regime_transition_probs` builds from a coarse probability vector,
    restricted to the declared targets: omitted regimes are structurally
    unreachable.

    Args:
        next_regime_cells: Per-target probability functions with qname params.

    Returns:
        A function over the union of the cells' arguments returning an
        immutable mapping of declared regime names to probability scalars.

    """
    cell_arg_names = {
        target_regime_name: tuple(
            name for name in get_annotations(cell) if name != "return"
        )
        for target_regime_name, cell in next_regime_cells.items()
    }
    merged_annotations: dict[str, str] = {}
    for cell in next_regime_cells.values():
        annotations = get_annotations(cell)
        annotations.pop("return", None)
        merged_annotations |= annotations
    return_annotation = MappingProxyType[RegimeName, FloatND]

    @with_signature(args=merged_annotations, return_annotation=return_annotation)
    def regime_transition_probs(
        **kwargs: FloatND | IntND | int,
    ) -> MappingProxyType[RegimeName, FloatND]:
        return MappingProxyType(
            {
                target_regime_name: jnp.asarray(
                    cell(
                        **{
                            name: kwargs[name]
                            for name in cell_arg_names[target_regime_name]
                        }
                    )
                )
                for target_regime_name, cell in next_regime_cells.items()
            }
        )

    # Pin `__annotations__` on the final wrapper: `concatenate_functions`
    # reads `__annotations__` (not `__signature__`) to reconcile the DAG.
    regime_transition_probs.__annotations__ = {
        **merged_annotations,
        "return": return_annotation,
    }
    return regime_transition_probs


def _wrap_regime_transition_probs(
    *,
    func: TransitionFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> Callable[..., MappingProxyType[RegimeName, FloatND]]:
    """Wrap next_regime function to convert array output to dict format.

    The next_regime function returns a JAX array of probabilities indexed by
    the regime's id. This wrapper converts the array to dict format for internal
    processing.

    Args:
        func: The user's next_regime function (with qname parameters).
        regime_names_to_ids: Immutable mapping of regime names to integer indices.

    Returns:
        A wrapped function that returns an immutable mapping of regime
        names to probability scalars.

    """
    # Get regime names in index order from regime_names_to_ids. Coerce
    # `ScalarInt` ids to Python `int` so `sorted` has a comparable key.
    regime_names_by_id: list[tuple[int, str]] = sorted(
        [(int(idx), name) for name, idx in regime_names_to_ids.items()],
        key=lambda x: x[0],
    )
    regime_names = [name for _, name in regime_names_by_id]

    # `wrapped` converts `func`'s probability array into a regime-name → prob
    # mapping. The return annotation describes that mapping; `func`'s own
    # return annotation (a bare probability array) does not survive the
    # conversion and must not be carried through.
    annotations = get_annotations(func)
    annotations.pop("return", None)
    return_annotation = MappingProxyType[RegimeName, FloatND]

    @with_signature(
        args=annotations,
        return_annotation=return_annotation,
    )
    @functools.wraps(func)
    def wrapped(
        *args: FloatND | IntND | int,
        **kwargs: FloatND | IntND | int,
    ) -> MappingProxyType[RegimeName, FloatND]:
        result = func(*args, **kwargs)
        # Convert array to dict using ordering by regime id
        return MappingProxyType(
            {name: result[idx] for idx, name in enumerate(regime_names)}
        )

    # Pin `__annotations__` on the final wrapper: `concatenate_functions`
    # reads `__annotations__` (not `__signature__`) to reconcile the DAG, and
    # the decorator stack can drop them when `func` carries deferred (PEP 649)
    # annotations through `functools.wraps`.
    wrapped.__annotations__ = {**annotations, "return": return_annotation}
    return wrapped


def _wrap_deterministic_regime_transition(
    *,
    func: TransitionFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> TransitionFunction:
    """Wrap deterministic next_regime to return one-hot probability array.

    Converts a deterministic regime transition function that returns an integer
    regime ID to a function that returns a one-hot probability array, matching
    the interface of stochastic regime transitions.

    Args:
        func: The user's deterministic next_regime function (returns int).
        regime_names_to_ids: Immutable mapping of regime names to integer indices.

    Returns:
        A wrapped function that returns a one-hot probability array.

    """
    n_regimes = len(regime_names_to_ids)

    # Preserve original annotations but update return type
    annotations = {k: v for k, v in get_annotations(func).items() if k != "return"}

    @with_signature(args=annotations, return_annotation="FloatND")
    @functools.wraps(func)
    def wrapped(
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> FloatND:
        regime_idx = func(*args, **kwargs)
        return jax.nn.one_hot(regime_idx, n_regimes)

    # Pin `__annotations__` on the final wrapper: `concatenate_functions`
    # reads `__annotations__` (not `__signature__`) to reconcile the DAG, and
    # the decorator stack can drop them when `func` carries deferred (PEP 649)
    # annotations through `functools.wraps`.
    wrapped.__annotations__ = {**annotations, "return": "FloatND"}
    return wrapped


def _get_vmap_params(
    *,
    all_args: tuple[str, ...],
    flat_param_names: frozenset[str],
) -> tuple[str, ...]:
    """Get parameter names that should be vmapped (states and actions)."""
    non_vmap = {"period", "age"} | flat_param_names
    return tuple(arg for arg in all_args if arg not in non_vmap)


def _co_map_state_names(
    *,
    state_names: tuple[StateName, ...],
    grids: MappingProxyType[StateOrActionName, Grid],
    transitions: TransitionFunctionsMapping,
) -> tuple[StateName, ...]:
    """Return the distributed, never-transitioning states, in state-axis order.

    A state qualifies when its grid is distributed and its law of motion is the
    identity in every target bundle that carries it — so its next value equals its
    current value, and the continuation V can be read from the device-local slice
    rather than all-gathered. Distributed states sort first in `state_names`, so the
    result is a leading prefix of it (what the co-map requires).
    """
    co_map: list[StateName] = []
    for name in state_names:
        grid = grids.get(name)
        if grid is None or not grid.distributed:
            continue
        next_key = f"next_{name}"
        carrying = [
            bundle[next_key] for bundle in transitions.values() if next_key in bundle
        ]
        if carrying and all(
            getattr(law, "_is_auto_identity", False) for law in carrying
        ):
            co_map.append(name)
    return tuple(co_map)


def _fold_state_names(
    *,
    state_names: tuple[StateName, ...],
    grids: MappingProxyType[StateOrActionName, Grid],
) -> tuple[StateName, ...]:
    """Return the IID-process states declared `fold=True`, in state-axis order.

    A folded state is integrated out of the stored value by quadrature
    immediately after the period's max-over-actions / collective readout — a
    state-topology property collected the same way `_co_map_state_names`
    collects distributed states, not a per-node computation.
    """
    return tuple(
        name
        for name in state_names
        if isinstance(grid := grids.get(name), _IIDProcess) and grid.fold
    )


def _strip_folded_axes_for_scalar_targets(
    *,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    transitions: TransitionFunctionsMapping,
    fold_only_regimes: frozenset[RegimeName],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
) -> MappingProxyType[RegimeName, VInterpolationInfo]:
    """Override the continuation interpolation info of empty-bundle folded targets.

    fold-review F2 (folded slice): a folded-only target carries an EMPTY
    transition bundle so the continuation graph keeps it enumerable
    (`_process_regime_core`). Its stored V has every folded axis integrated out
    (`_get_regime_V_shapes_and_shardings`), but `create_v_interpolation_info`
    still lists the folded states, so the ordinary continuation interpolator
    would demand a `next_<shock>` coordinate — which the source never realises —
    and index an axis the scalar V no longer has. For exactly those targets,
    return a `VInterpolationInfo` with the folded states stripped, so the
    interpolator becomes a plain scalar read. Every other regime's info is
    passed through unchanged.

    Args:
        regime_to_v_interpolation_info: The model's per-regime interpolation info.
        transitions: This source regime's processed transition bundles.
        fold_only_regimes: Regimes whose every state is a folded IID process.
        all_grids: Immutable mapping of regime names to Grid spec objects.

    Returns:
        The interpolation-info mapping with folded-only empty-bundle targets
        stripped of their folded axes (a copy only when an override is needed).

    """
    scalar_targets = [
        target
        for target, bundle in transitions.items()
        if not bundle and target in fold_only_regimes
    ]
    if not scalar_targets:
        return regime_to_v_interpolation_info
    overridden = dict(regime_to_v_interpolation_info)
    for target in scalar_targets:
        info = regime_to_v_interpolation_info[target]
        grids = all_grids[target]
        folded = {
            name
            for name in info.state_names
            if isinstance(grid := grids.get(name), _IIDProcess) and grid.fold
        }
        overridden[target] = VInterpolationInfo(
            state_names=tuple(n for n in info.state_names if n not in folded),
            discrete_states=MappingProxyType(
                {k: v for k, v in info.discrete_states.items() if k not in folded}
            ),
            continuous_states=MappingProxyType(
                {k: v for k, v in info.continuous_states.items() if k not in folded}
            ),
        )
    return MappingProxyType(overridden)


def _fail_if_coarse_candidate_folds_ambiguously(
    *,
    source_regime_name: RegimeName,
    source_process_names: frozenset[ProcessName],
    coarse_candidate_targets: set[RegimeName],
    all_grids: Mapping[RegimeName, Mapping[StateOrActionName, Grid]],
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
) -> None:
    """Reject an ambiguous folded-coarse topology; require per-target cells.

    A coarse `transition=func` decides its target at runtime, so its candidate
    universe is every regime but its actual SUPPORT is unknown at build time.
    That is fine for a non-folded continuation (a never-returned candidate just
    carries zero probability), but not for a FOLDED process: whether a fold
    genuinely persists across the coarse edge -- which `_fail_if_folded_state_
    persists` must decide STRUCTURALLY, without seeing probabilities -- depends
    on the unknown support. Admitting such a candidate would either accept a real
    persisting self-fold (fold-round4 F1) or reject a folded candidate the
    function never returns (fold-round4 F2). So a coarse candidate that folds a
    process AND is active in a period immediately after the source (so the coarse
    edge could carry that fold into a period where it persists) is rejected here:
    the modeller must declare it with explicit per-target transition cells, whose
    support IS known, so the persistence guard can validate it exactly.

    fold-round5 F1: persistence across the coarse edge is only POSSIBLE for a
    folded process the SOURCE itself carries -- the continuation builder auto-wires
    an intrinsic `next_<process>` edge only for the source's own `process_names`
    (see the `target_process_grids` comprehension above and the `next_<name>`
    keys in `_stochastic_transition_names`). A candidate that folds a
    TARGET-LOCAL process whose name the source does NOT carry can never receive a
    `next_<process>` edge from this source, so the fold cannot persist across the
    edge and there is nothing to validate; rejecting it is a false positive. Only
    the intersection with the source's process names is genuinely ambiguous.
    """
    source_active = active_periods_by_regime.get(source_regime_name, ())
    next_periods = {p + 1 for p in source_active}
    error_messages: list[str] = []
    for target in sorted(coarse_candidate_targets):
        if not next_periods.intersection(active_periods_by_regime.get(target, ())):
            continue
        target_grids = all_grids.get(target, {})
        folded = sorted(
            name
            for name, grid in target_grids.items()
            if isinstance(grid, _ContinuousStochasticProcess)
            and getattr(grid, "fold", False)
            and name in source_process_names
        )
        error_messages.extend(
            f"regime '{source_regime_name}' uses a COARSE `transition=func`, "
            f"and candidate target '{target}' folds process '{process}' and is "
            f"active in the period immediately after '{source_regime_name}'. A "
            f"coarse transition's actual support is unknown at build time, so a "
            f"folded process that could persist across it cannot be validated: "
            f"declare the transition into '{target}' with an explicit PER-TARGET "
            f"cell (`transition={{'{target}': ...}}`) so its reachability is "
            f"known and the fold-persistence check can be applied exactly."
            for process in folded
        )
    if error_messages:
        raise ModelInitializationError(format_messages(error_messages))


def _fail_if_folded_state_persists(
    *, canonical_regimes: Mapping[RegimeName, Regime]
) -> None:
    """Reject a folded state that structurally persists past its own period.

    A fold weighted-averages a state's axis out of the stored value: the
    stored `V` of the regime that declares `fold=True` on it has no such
    axis (`_get_regime_V_shapes_and_shardings`). If ANY regime's transitions
    — including the declaring regime's own, via a self-transition — still
    carry an intrinsic `next_<name>` continuation for that state (i.e. the
    state is ALSO declared, hence auto-wired with its own weight/index
    functions, in some regime reachable from another), the continuation
    machinery would try to interpolate a `next_<name>` axis that the target's
    stored `V` no longer has — a shape mismatch, or worse, a silent wrong
    read. This is a cross-regime property (every regime's `solution.transitions`
    must be known), so it is checked once here, after every regime is built —
    not in the regime-local `_validate_fold_declarations`.

    Folding is therefore restricted, for now, to states that do not persist:
    declared in exactly the one regime that folds them, and not redeclared
    (directly or via a self-transition) in any regime any transition reaches.
    A genuinely persistent IID shock (redrawn every period a regime is
    active) needs the fold to also be recognized by the *continuation* side
    (`regime_to_v_interpolation_info` / `stochastic_transition_names`) of
    every regime that reads into it — out of scope for this slice.
    """
    error_messages: list[str] = []
    for regime_name, regime in canonical_regimes.items():
        if not regime.fold_state_names:
            continue
        for fold_name in regime.fold_state_names:
            next_key = f"next_{fold_name}"
            offending = [
                (source_name, target_name)
                for source_name, source_regime in canonical_regimes.items()
                for target_name, bundle in source_regime.solution.transitions.items()
                if target_name == regime_name and next_key in bundle
            ]
            if offending:
                sources = sorted({source for source, _ in offending})
                error_messages.append(
                    f"fold=True on regime '{regime_name}' state '{fold_name}' "
                    f"is not supported: '{fold_name}' is also declared as a "
                    f"state reachable via a next-period continuation from "
                    f"{sources} into '{regime_name}' — i.e. it structurally "
                    f"persists. A folded state's stored value has no "
                    f"'{fold_name}' axis, so that continuation could no "
                    f"longer interpolate over it. Folding is only supported "
                    f"for a state that does not persist past the period that "
                    f"folds it (not redeclared, directly or via a "
                    f"self-transition, in any reachable regime)."
                )
    if error_messages:
        raise ModelInitializationError(format_messages(error_messages))


def _build_period_state_axes(
    *,
    regime_name: RegimeName,
    grid_schedule: AgeGridSchedule | None,
    active_periods: tuple[int, ...],
) -> MappingProxyType[int, MappingProxyType[StateOrActionName, Float1D]] | None:
    """Per-period node arrays for this regime's age-specialized continuous states.

    Read straight from the schedule: every state declared via `AgeSpecializedGrid`
    gets an explicit per-period axis table (the period's concrete grid nodes, so V
    is tabulated on the current grid), regardless of whether some resolved values
    happen to be equal. Returns `{period: {state: nodes}}`, or `None` when the
    regime has no age-specialized state (age-invariant regimes are unchanged).
    """
    if grid_schedule is None or not active_periods:
        return None
    specialized = grid_schedule.specialized_states_by_regime.get(
        regime_name, frozenset()
    )
    if not specialized:
        return None
    return MappingProxyType(
        {
            period: MappingProxyType(
                {
                    state_name: grid_schedule.by_period[period][regime_name][
                        state_name
                    ].nodes
                    for state_name in sorted(specialized)
                }
            )
            for period in active_periods
        }
    )


def _build_Q_and_F_per_period(
    *,
    active_periods: tuple[int, ...],
    phase_reachability: PhaseReachability,
    source_regime_name: RegimeName,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
    co_map_state_names: tuple[StateName, ...] = (),
    certainty_equivalent: CertaintyEquivalent | None = None,
    stakeholders: tuple[str, ...] | None = None,
    value_constraints: ConstraintFunctionsMapping = MappingProxyType({}),
    same_period_refs: MappingProxyType[str, ResolvedSamePeriodRef] = (
        MappingProxyType({})
    ),
    continuation_functions: EconFunctionsMapping | None = None,
    flow_transitions: TransitionFunctionsMapping | None = None,
    flow_stochastic_transition_names: frozenset[TransitionFunctionName] | None = None,
    next_state_names: frozenset[TransitionFunctionName] = frozenset(),
    grid_schedule: AgeGridSchedule | None = None,
) -> MappingProxyType[int, QAndFFunction]:
    """Build Q-and-F closures for each active period of a non-terminal regime.

    Periods sharing the same target-regime configuration and static signature
    reuse a single closure, reducing distinct JIT compilations. The caller is
    responsible for handling terminal regimes. Only the source regime's active
    periods are built — the solve/simulate loops only ever index those.

    When the model has `AgeSpecializedGrid` states, period `t`'s continuation
    `V_{t+1}` is interpolated on the target regimes' grids **at period `t+1`**
    (`period_to_regime_v_interp`), and `continuation_group_key` folds in those
    grids' explicit user signatures (`grid_schedule`) so periods with different
    continuation grids do not false-share a compiled `Q_and_F`. With no
    age-specialized objects the grouping collapses to the target configuration
    exactly as an age-invariant model.

    Args:
        active_periods: The source regime's active periods (the periods built).
        phase_reachability: Static graph for this phase.
        source_regime_name: Regime whose continuation targets are requested.
        functions: Immutable mapping of internal (possibly periodized) functions.
        constraints: Immutable mapping of constraint functions.
        transitions: Immutable mapping of regime-to-regime transition functions.
        stochastic_transition_names: Frozenset of stochastic transition names.
        compute_regime_transition_probs: Regime transition probability function.
        regime_to_v_interpolation_info: Mapping of regime names to representative
            V-interpolation info (the age-invariant fallback).
        flat_param_names: Frozenset of flat parameter names for the regime.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.
        stakeholders: Ordered stakeholder names for a collective regime, or
            `None` (the singleton default). When set, the per-period closures
            come from `get_Q_and_F_collective` (per-stakeholder continuation,
            trailing stakeholder axis on Q); only the solve site passes this.
        value_constraints: Value-aware feasibility predicates (E2), params
            already renamed; only used for collective regimes.
        same_period_refs: Resolved same-period reference declarations (E2);
            only used for collective regimes.
        continuation_functions: Solve-phase pool the continuation sub-DAG resolves
            against; `None` in the solve phase, where it coincides with `functions`.
        flow_transitions: Simulate-phase transitions the flow `next_<state>` nodes are
            taken from; `None` in the solve phase, where they coincide with
            `transitions`. See `get_Q_and_F` for the phase-closure contract.
        flow_stochastic_transition_names: Stochastic names of `flow_transitions`.
        next_state_names: Declared `next_<state>` node names for this regime,
            used by the no-producer guard to distinguish a genuine missing
            next-state producer from a current variable merely named `next_*`.
        grid_schedule: Concrete age-specialized grid schedule, or `None`.
        period_to_regime_v_interp: Per-period continuation interpolation info
            built from the schedule, or `None`.

    Returns:
        Immutable mapping of period index to the per-period Q-and-F closure.

    """

    # `continuation_info`: all-regime interpolation info for period `t`'s
    # continuation V_{t+1}. The last period's continuation is the zero template.
    # `group_key` folds in the continuation targets' explicit user grid signatures
    # at t+1 (`grid_schedule`), so periods with different continuation grids do
    # not false-share a compiled kernel, and their stateless-target partition, so
    # periods differing only in which stateless regimes they can reach never
    # share a closure either.
    continuation_info = continuation_info_lookup(
        period_to_regime_v_interp=period_to_regime_v_interp,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )
    # `continuation_functions` rides in the key too (round-11 F1): the perceived
    # (solve) continuation pool is resolved per period at the group's
    # representative period below, so periods whose pool resolves to different
    # closures must NOT share a kernel. `None` contributes a constant, which
    # collapses the grouping exactly as before.
    group_key = continuation_group_key(
        phase_reachability=phase_reachability,
        source_regime_name=source_regime_name,
        functions=functions,
        constraints=constraints,
        grid_schedule=grid_schedule,
        continuation_info=continuation_info,
        continuation_functions=continuation_functions,
    )

    configs = group_periods_by_key(active_periods, group_key)

    # Build one Q_and_F per distinct group, resolving periodized functions and
    # constraints at the group's representative period. Equal signature ⇒ identical
    # closures, so any period in the group is a valid representative.
    built: dict[tuple[tuple[RegimeName, ...], Hashable], QAndFFunction] = {}
    for key, periods in configs.items():
        targets = key[0]
        representative_period = periods[0]
        # Upstream added this check; it is a general invariant about the group's
        # continuation grids, so it belongs BEFORE the collective/singleton split
        # rather than only on the singleton twin.
        stateful_targets, scalar_targets = partition_continuation_targets(
            targets=targets,
            regime_to_v_interpolation_info=continuation_info(representative_period),
        )
        assert_continuation_grids_agree(
            grid_schedule=grid_schedule,
            target_regimes=targets,
            periods=tuple(periods),
        )
        if stakeholders is not None:
            # COLLECTIVE-REGIMES (E1): separate builder so the singleton path
            # is byte-identical. No certainty equivalent — rejected at regime
            # construction for collective regimes. Functions/constraints are
            # resolved at the group's age exactly as the singleton branch — a
            # passthrough for a collective regime with no age-specialized nodes
            # (EKL), so the tested collective behaviour is unchanged.
            built[key] = get_Q_and_F_collective(
                flat_param_names=flat_param_names,
                functions=cast(
                    "EconFunctionsMapping",
                    resolve_periodized_nodes(functions, representative_period),
                ),
                constraints=cast(
                    "ConstraintFunctionsMapping",
                    resolve_periodized_nodes(constraints, representative_period),
                ),
                # The stateful targets: the ones whose continuation is actually
                # interpolated this period, which is what this branch's
                # `period_targets` has always meant. Upstream's
                # `partition_continuation_targets` splits the graph targets into
                # exactly that set and the stateless one; the collective builder
                # takes no stateless targets, as before.
                period_targets=stateful_targets,
                transitions=transitions,
                stochastic_transition_names=stochastic_transition_names,
                compute_regime_transition_probs=compute_regime_transition_probs,
                regime_to_v_interpolation_info=continuation_info(representative_period),
                stakeholders=stakeholders,
                co_map_state_names=co_map_state_names,
                value_constraints=value_constraints,
                same_period_refs=same_period_refs,
                # Thread the flow/continuation phase split, exactly as the singleton
                # branch below: dropping these made the collective simulator build
                # current flow from the SOLVE transitions and the continuation from
                # the SIMULATE/decision pool — a hybrid sub-DAG that can reverse the
                # household argmax (collective-regimes-branch-2 audit F1). `None` /
                # solve-phase values are a passthrough, so the solve build is
                # unchanged.
                continuation_functions=(
                    cast(
                        "EconFunctionsMapping",
                        resolve_periodized_nodes(
                            continuation_functions, representative_period
                        ),
                    )
                    if continuation_functions is not None
                    else None
                ),
                flow_transitions=flow_transitions,
                flow_stochastic_transition_names=flow_stochastic_transition_names,
                next_state_names=next_state_names,
            )
            continue
        built[key] = get_Q_and_F(
            flat_param_names=flat_param_names,
            functions=cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(functions, representative_period),
            ),
            constraints=cast(
                "ConstraintFunctionsMapping",
                resolve_periodized_nodes(constraints, representative_period),
            ),
            period_targets=stateful_targets,
            scalar_targets=scalar_targets,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=continuation_info(representative_period),
            co_map_state_names=co_map_state_names,
            certainty_equivalent=certainty_equivalent,
            # Resolve the perceived continuation pool at THIS group's representative
            # period (F1) — not a period frozen into the published solve functions.
            # Valid because the pool's per-period signature is folded into the group
            # key above, so every period in the group resolves it identically.
            continuation_functions=(
                cast(
                    "EconFunctionsMapping",
                    resolve_periodized_nodes(
                        continuation_functions, representative_period
                    ),
                )
                if continuation_functions is not None
                else None
            ),
            flow_transitions=flow_transitions,
            flow_stochastic_transition_names=flow_stochastic_transition_names,
            next_state_names=next_state_names,
        )

    return expand_groups_to_periods(configs, built)


def _build_argmax_and_max_Q_over_a_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    enable_jit: bool,
    has_taste_shocks: bool = False,
    stakeholders: tuple[str, ...] | None = None,
    weights: Mapping[str, float] | None = None,
) -> MappingProxyType[int, ArgmaxQOverAFunction]:
    """Build argmax-and-max-Q-over-a closures for each period.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    With taste shocks, the per-subject Gumbel key is vmapped alongside the
    simulated states.

    COLLECTIVE-REGIMES (E4): `stakeholders`/`weights`, when set, thread into
    `get_argmax_and_max_Q_over_a`'s collective branch — the returned V carries
    a trailing stakeholder axis, which `simulation_spacemap` below preserves
    (it only maps over `state_names`, never the trailing axis).
    """
    spacemapped_names = tuple(state_action_space.states)
    if has_taste_shocks:
        spacemapped_names = (*spacemapped_names, "taste_shock_key")

    built: dict[int, ArgmaxQOverAFunction] = {}
    result: dict[int, ArgmaxQOverAFunction] = {}
    for period, Q_and_F in Q_and_F_functions.items():
        q_id = id(Q_and_F)
        if q_id not in built:
            func = get_argmax_and_max_Q_over_a(
                Q_and_F=Q_and_F,
                action_names=state_action_space.action_names,
                state_names=state_action_space.state_names,
                n_discrete_action_axes=len(state_action_space.discrete_actions),
                has_taste_shocks=has_taste_shocks,
                stakeholders=stakeholders,
                weights=weights,
            )
            if enable_jit:
                func = jax.jit(func)
            built[q_id] = simulation_spacemap(
                func=func,
                action_names=(),
                state_names=spacemapped_names,
            )
        result[period] = built[q_id]
    return MappingProxyType(result)


def _build_pointwise_Q_and_F_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    enable_jit: bool,
) -> MappingProxyType[int, QAndFFunction]:
    """Build per-subject pointwise Q-and-feasibility closures for each period.

    Unlike `argmax_and_max_Q_over_a`, which maximizes over the Cartesian product
    of the action grids, this maps every action alongside the states over the
    subject axis: each subject supplies its own action value and receives that
    one state-action pair's value and feasibility. Off-grid candidate actions are
    scored through here, so their values share the grid maximization's objective.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    """
    subject_mapped_names = (
        *state_action_space.state_names,
        *state_action_space.action_names,
    )

    built: dict[int, QAndFFunction] = {}
    result: dict[int, QAndFFunction] = {}
    for period, Q_and_F in Q_and_F_functions.items():
        q_id = id(Q_and_F)
        if q_id not in built:
            # A terminal period's Q_and_F takes no actions, so only the names it
            # actually declares can be mapped over the subject axis.
            declared = frozenset(inspect.signature(Q_and_F).parameters)
            func = jax.jit(Q_and_F) if enable_jit else Q_and_F
            built[q_id] = simulation_spacemap(
                func=func,
                action_names=(),
                state_names=tuple(
                    name for name in subject_mapped_names if name in declared
                ),
            )
        result[period] = built[q_id]
    return MappingProxyType(result)


def _build_next_state_vmapped(
    *,
    active_periods: tuple[int, ...],
    phase_reachability: PhaseReachability,
    source_regime_name: RegimeName,
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    variables: Variables,
    flat_param_names: frozenset[str],
    enable_jit: bool,
) -> MappingProxyType[int, NextStateSimulationFunction]:
    """Build a per-period vmapped next-state function for simulation.

    A law of motion can read a periodized function (e.g. `next_wealth` reading
    `net_income`), so next-state is resolved per period just like `Q_and_F`. Only
    the regime's active periods are built. Periods whose functions resolve to the
    same closures share one compiled function; with no age-specialized node every
    period shares a single function, exactly as an age-invariant model.
    """
    configs = group_periods_by_key(
        active_periods,
        lambda period: (
            (
                ()
                if period == phase_reachability.n_periods - 1
                else phase_reachability.targets(
                    period=period, source=source_regime_name
                )
            ),
            periodized_tree_signature(functions, period),
        ),
    )

    built: dict[
        tuple[tuple[RegimeName, ...], Hashable], NextStateSimulationFunction
    ] = {}
    for key, periods in configs.items():
        period_targets, _ = key
        representative_period = periods[0]
        period_transitions = MappingProxyType(
            {
                target: transitions[target]
                for target in period_targets
                if target in transitions
            }
        )
        next_state = get_next_state_function_for_simulation(
            functions=cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(functions, representative_period),
            ),
            transitions=period_transitions,
            stochastic_transition_names=stochastic_transition_names,
            all_grids=all_grids,
            variables=variables,
        )
        sig_args = tuple(inspect.signature(next_state).parameters)
        non_vmap = {"period", "age"} | flat_param_names
        vmap_variables = tuple(arg for arg in sig_args if arg not in non_vmap)
        next_state_vmapped = vmap_1d(func=next_state, variables=vmap_variables)
        next_state_vmapped = with_signature(
            next_state_vmapped, kwargs=sig_args, enforce=False
        )
        built[key] = jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped

    return expand_groups_to_periods(configs, built)


def _fail_if_action_has_batch_size(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Raise if any action grid has a non-zero batch_size.

    Batching applies only to the outer state loop during solving, not to the
    inner action optimization. A non-zero batch_size on an action grid would be
    silently ignored, so we reject it early.

    """
    for regime_name, user_regime in user_regimes.items():
        for action_name, grid in user_regime.actions.items():
            if grid is not None and grid.batch_size != 0:
                msg = (
                    f"batch_size > 0 is not supported on action grids. Only state "
                    f"grids can be batched. Found batch_size={grid.batch_size} on "
                    f"action '{action_name}' in regime '{regime_name}'."
                )
                raise ValueError(msg)
