import functools
import inspect
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
from dags import concatenate_functions, get_annotations, with_signature
from dags.signature import rename_arguments
from dags.tree import qname_from_tree_path, tree_path_from_qname
from jax import numpy as jnp

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.coarse_transition import _CoarseTransitionCell
from _lcm.engine import (
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
from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.next_state import get_next_state_function_for_simulation
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
    normalize_all_regime_phases,
)
from _lcm.regime_building.Q_and_F import (
    get_Q_and_F,
    get_Q_and_F_terminal,
)
from _lcm.regime_building.stochastic_state_transitions import (
    collect_stochastic_state_transitions,
)
from _lcm.regime_building.V import VInterpolationInfo, create_v_interpolation_info
from _lcm.solution.contract import SolverBuildContext
from _lcm.state_action_space import create_state_action_space
from _lcm.transition_laws import (
    TransitionLawInfo,
    TransitionLaws,
)
from _lcm.typing import (
    ArgmaxQOverAFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    FunctionName,
    NextStateSimulationFunction,
    ProcessName,
    QAndFFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
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
from _lcm.utils.namespace import flatten_regime_namespace, unflatten_regime_namespace
from _lcm.variables import (
    from_regime,
    get_grids,
    simulate_variables_from_regime,
)
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.solvers import Solver
from lcm.transition import MarkovTransition
from lcm.typing import Float1D, FloatND, Int1D, IntND, UserFunction

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

    canonical_regimes: dict[RegimeName, Regime] = {}
    # Iterate the representative-resolved regimes: identical to the user regimes
    # except that any `AgeSpecializedGrid` state is a concrete representative-age
    # grid, so every grid-derived call below is age-invariant.
    for regime_name, user_regime in representative_user_regimes.items():
        spec = specs[regime_name]
        # The representative regime already carries first-active concrete functions,
        # so the parameter template no longer needs to know about age specialization.
        regime_params_template = create_regime_params_template(user_regime)
        granular_param_expansions = _granular_param_expansions(
            nested_transitions_by_phase=(
                solve_nested_transitions[regime_name],
                simulate_nested_transitions[regime_name],
            ),
            regime_params_template=regime_params_template,
            declaration_param_expansions=_declaration_param_expansions(
                source_regime_name=regime_name,
                specs=phased_specs,
                all_regime_names=all_regime_names,
                regime_params_template=regime_params_template,
            ),
        )

        solution = _build_solution_phase(
            spec=spec,
            regime_name=regime_name,
            declared_regime_transition=phased_specs[
                regime_name
            ].solution.regime_transition,
            phase_reachability=reachability.solution,
            nested_transitions=solve_nested_transitions[regime_name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            granular_param_expansions=granular_param_expansions,
            regime_names_to_ids=regime_names_to_ids,
            variables=regime_to_variables[regime_name],
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            period_to_regime_v_interp=period_to_regime_v_interp,
            grid_schedule=grid_schedule,
            state_action_space=state_action_spaces[regime_name],
            ages=ages,
            enable_jit=enable_jit,
            has_taste_shocks=user_regime.taste_shocks is not None,
            certainty_equivalent=user_regime.certainty_equivalent,
            solver=user_regime.solver,
        )

        simulation = _build_simulation_phase(
            spec=spec,
            regime_name=regime_name,
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
            solve_transitions=solution.transitions,
            solve_transition_laws=solution.transition_laws,
            solve_compute_regime_transition_probs=solution.compute_regime_transition_probs,
            has_taste_shocks=user_regime.taste_shocks is not None,
            certainty_equivalent=user_regime.certainty_equivalent,
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
        )

    return ensure_containers_are_immutable(canonical_regimes)


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
                        entry_error = _runtime_param_entry_error(
                            phase_name=phase_name,
                            phase_reachability=phase_reachability,
                            ages=ages,
                            period=period,
                            source=source,
                            target=target,
                            state_name=state_name,
                            target_grid=target_grid,
                        )
                        if entry_error is not None:
                            error_messages.append(entry_error)
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


def _runtime_param_entry_error(
    *,
    phase_name: PhaseName,
    phase_reachability: PhaseReachability,
    ages: AgeGrid,
    period: int,
    source: RegimeName,
    target: RegimeName,
    state_name: StateName,
    target_grid: _IIDProcess,
) -> str | None:
    """Return an error if an entered IID process needs runtime parameters.

    The entry weights are the process's unconditional row, and they are
    evaluated inside the *source's* Bellman equation, which reads only the
    source's own parameters. A law supplied at runtime therefore has no value
    the source can read: restating it under the source would create a second
    knob for one law, free to disagree with the nodes the target's value
    function is built on. Entry is available for a law fixed at construction.

    Args:
        phase_name: Phase whose slices are being checked.
        phase_reachability: Static graph for this phase.
        ages: The AgeGrid for the model.
        period: Period of the source regime.
        source: Regime being left.
        target: Regime being entered.
        state_name: Name of the target's process state.
        target_grid: The target's grid for `state_name`.

    Returns:
        The error message, or `None` when the process's law is fully fixed.

    """
    runtime_params = target_grid.params_to_pass_at_runtime
    if not runtime_params:
        return None
    status = phase_reachability.edge_status(period=period, source=source, target=target)
    named = ", ".join(f"'{param}'" for param in runtime_params)
    return (
        f"{phase_name} phase, period {period} "
        f"(age {_display_age(ages, period)}), source '{source}' -> "
        f"period {period + 1} (age {_display_age(ages, period + 1)}), "
        f"target '{target}' retains a {status.name} edge. The target declares "
        f"stochastic process '{state_name}', which the source does not carry, "
        f"so it is entered at the process's own law. That law is priced in "
        f"'{source}', which reads only its own parameters, but '{state_name}' "
        f"passes {named} at runtime. Fix '{state_name}' at construction, "
        f"declare it on '{source}', or narrow the transition's static target "
        f"support."
    )


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
    declared_regime_transition: object,
    phase_reachability: PhaseReachability,
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]],
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
    has_taste_shocks: bool,
    certainty_equivalent: CertaintyEquivalent | None,
    solver: Solver,
) -> SolutionPhase:
    """Build all compiled functions for the backward-induction (solve) phase.

    Args:
        spec: The regime's per-phase specification.
        regime_name: The name of the regime.
        declared_regime_transition: Solve transition before temporal filtering.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        granular_param_expansions: Immutable mapping of coarse-template law
            keys to granular qname prefixes.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        variables: States and actions of the regime with kind/topology/process tags.
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.
        solver: The regime's solver; the engine calls `validate` then
            `build_period_kernels` on it to obtain the per-period kernels.

    Returns:
        Complete solve functions container.

    """
    core = _process_regime_core(
        functions=spec.solution.functions,
        constraints=spec.solution.constraints,
        state_transitions=spec.solution.state_transitions,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
        phase_reachability=phase_reachability,
        source_regime_name=regime_name,
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

    if spec.terminal:
        compute_regime_transition_probs = None
        validation_regime_transition_probs = None
        terminal_func = get_Q_and_F_terminal(
            flat_param_names=flat_param_names,
            functions=core.functions,
            constraints=core.constraints,
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
        Q_and_F_functions = _build_Q_and_F_per_period(
            active_periods=regimes_to_active_periods[regime_name],
            phase_reachability=phase_reachability,
            source_regime_name=regime_name,
            functions=core.functions,
            constraints=core.constraints,
            transitions=core.transitions,
            transition_laws=core.transition_laws,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            flat_param_names=flat_param_names,
            co_map_state_names=co_map_state_names,
            certainty_equivalent=certainty_equivalent,
            grid_schedule=grid_schedule,
            period_to_regime_v_interp=period_to_regime_v_interp,
        )
        compute_intermediates = _build_compute_intermediates_per_period(
            active_periods=regimes_to_active_periods[regime_name],
            flat_param_names=flat_param_names,
            phase_reachability=phase_reachability,
            source_regime_name=regime_name,
            functions=core.functions,
            constraints=core.constraints,
            transitions=core.transitions,
            transition_laws=core.transition_laws,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            state_action_space=state_action_space,
            grids=all_grids[regime_name],
            enable_jit=enable_jit,
            certainty_equivalent=certainty_equivalent,
            # F4: diagnostics recompute on the SAME period-specific target grid as the
            # primary solve (not the representative grid).
            grid_schedule=grid_schedule,
            period_to_regime_v_interp=period_to_regime_v_interp,
        )

    # Dispatch the per-period kernel build polymorphically on the regime's
    # solver: `validate` rejects out-of-scope configurations at build time,
    # then `build_period_kernels` returns the per-period kernels. `GridSearch`
    # builds the max-Q-over-a grid-search kernels.
    context = SolverBuildContext(
        state_action_space=state_action_space,
        solution_reachability=phase_reachability,
        Q_and_F_functions=Q_and_F_functions,
        grids=all_grids[regime_name],
        enable_jit=enable_jit,
        has_taste_shocks=has_taste_shocks,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
        co_map_v_arr_in_axes=co_map_v_arr_in_axes,
    )
    solver.validate(context=context)
    solver_kernels = solver.build_period_kernels(context=context)
    max_Q_over_a = solver_kernels.max_Q_over_a

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
        constraints=core.constraints,
        transitions=core.transitions,
        transition_laws=core.transition_laws,
        reachability=phase_reachability,
        compute_regime_transition_probs=compute_regime_transition_probs,
        validation_regime_transition_probs=validation_regime_transition_probs,
        max_Q_over_a=max_Q_over_a,
        compute_intermediates=compute_intermediates,
        _base_state_action_space=state_action_space,
        period_state_axes=period_state_axes,
    )


def _build_simulation_phase(
    *,
    spec: PhasedRegimeSpec,
    regime_name: RegimeName,
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
    solve_transitions: TransitionFunctionsMapping,
    solve_transition_laws: TransitionLaws,
    solve_compute_regime_transition_probs: RegimeTransitionFunction | None,
    has_taste_shocks: bool,
    certainty_equivalent: CertaintyEquivalent | None,
) -> SimulationPhase:
    """Build all compiled functions for the forward-simulation phase.

    The decision functions (Q_and_F, argmax, regime-transition probs) are
    built from the simulation slice's functions plus each carried state's
    solve-phase imputation — the agent decides on the value the solved policy
    was computed for. The published function pool strips the imputations so
    every other simulate consumer reads the carried value.

    Q_and_F always uses the solve (non-vmapped) regime transition probs because
    it evaluates on the Cartesian grid, not per-subject.

    Args:
        spec: The regime's per-phase specification.
        regime_name: The name of the regime.
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
        solve_transitions: Transitions from the solve phase (reused).
        solve_transition_laws: Immutable mapping of target regime names to their
            transition laws, built in the solve phase and reused here.
        solve_compute_regime_transition_probs: Solve-phase regime transition prob
            function, used for Q_and_F in both phases.
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.

    Returns:
        Complete simulate functions container.

    """
    carried_only = spec.carried_only_state_names
    decision_functions = dict(spec.simulation.functions) | {
        name: spec.solution.functions[name] for name in carried_only
    }
    core = _process_regime_core(
        functions=decision_functions,
        constraints=spec.simulation.constraints,
        state_transitions=spec.simulation.state_transitions,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
        phase_reachability=simulation_reachability,
        source_regime_name=regime_name,
    )
    functions = core.functions
    constraints = core.constraints

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

    if spec.terminal:
        compute_regime_transition_probs = None
        terminal_func = get_Q_and_F_terminal(
            flat_param_names=flat_param_names,
            functions=functions,
            constraints=constraints,
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
        Q_and_F_functions = _build_Q_and_F_per_period(
            active_periods=regimes_to_active_periods[regime_name],
            phase_reachability=solution_reachability,
            source_regime_name=regime_name,
            functions=functions,
            constraints=constraints,
            transitions=solve_transitions,
            transition_laws=solve_transition_laws,
            compute_regime_transition_probs=solve_compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            flat_param_names=flat_param_names,
            certainty_equivalent=certainty_equivalent,
            grid_schedule=grid_schedule,
            period_to_regime_v_interp=period_to_regime_v_interp,
        )

    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        enable_jit=enable_jit,
        has_taste_shocks=has_taste_shocks,
    )

    next_state = _build_next_state_vmapped(
        active_periods=regimes_to_active_periods[regime_name],
        phase_reachability=simulation_reachability,
        source_regime_name=regime_name,
        functions=simulate_functions,
        transitions=core.transitions,
        transition_laws=core.transition_laws,
        all_grids=all_grids,
        flat_param_names=flat_param_names,
        enable_jit=enable_jit,
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
        transition_laws=core.transition_laws,
        compute_regime_transition_probs=compute_regime_transition_probs,
        argmax_and_max_Q_over_a=argmax_and_max_Q_over_a,
        next_state=next_state,
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

    transition_laws: TransitionLaws
    """Immutable mapping of target regime names to their transition laws."""

    next_regime_func: TransitionFunction | None
    """The coarse regime transition function; `None` for terminal regimes and
    for per-target regime transitions."""

    next_regime_cells: MappingProxyType[RegimeName, EconFunction] | None
    """Per-target regime transition probability functions (params renamed),
    or `None` when the regime transition is coarse or absent."""


def _process_regime_core(
    *,
    functions: Mapping[FunctionName, UserFunction],
    constraints: Mapping[FunctionName, UserFunction],
    state_transitions: Mapping[StateName, object],
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    variables: Variables,
    phase_reachability: PhaseReachability,
    source_regime_name: RegimeName,
) -> _CoreResult:
    """Process one phase's regime functions and transitions.

    The caller supplies phase-resolved inputs (a slice of the regime's
    `PhasedRegimeSpec`, possibly augmented): rename params to qualified names,
    classify and process transitions.

    Args:
        functions: Phase-resolved regime functions for this build.
        constraints: Phase-resolved constraint functions.
        state_transitions: This phase's `state_transitions` slice, used to
            detect per-target dicts and stochastic transitions.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        variables: States and actions of the regime with kind/topology/process tags.
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
        processed_functions[func_name] = _process_one_function(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
        )

    for func_name, func in deterministic_transition_functions.items():
        processed_functions[func_name] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
            names_key=_extract_template_names_key(func_name, regime_params_template),
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
    }
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
    # Two conditions bound this and both must be added here as they become
    # expressible, because entry builds a next-state and a weight function and
    # so commits to both an axis and a distribution:
    #
    # - **storage.** A process integrated out of the value function rather than
    #   stored on one must be excluded, or entry reintroduces exactly the axis
    #   its treatment removes -- it needs no entry law for the same reason it
    #   needs no handoff.
    # - **conditioning.** The entry weights below are the unconditional row,
    #   which is the whole distribution only for a process whose law depends on
    #   nothing. A process conditioned on another state must not be entered at
    #   that row: the conditioner is a live state of the target, and entry has
    #   no reason to ignore it.
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
                name=process, grid=grid
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

    transitions = _wrap_transitions(unflatten_regime_namespace(internal_transition))

    transition_laws = _build_transition_laws(
        transitions=transitions,
        processed_functions=processed_functions,
        all_grids=all_grids,
        entered_processes=frozenset(entered_process_grids),
    )

    next_regime_func, next_regime_cells = _process_next_regime_cells(
        next_regime_cells_by_target=next_regime_cells_by_target,
        regime_params_template=regime_params_template,
    )

    return _CoreResult(
        functions=phase_functions,
        constraints=processed_constraints,
        transitions=transitions,
        transition_laws=transition_laws,
        next_regime_func=next_regime_func,
        next_regime_cells=next_regime_cells,
    )


def _build_transition_laws(
    *,
    transitions: TransitionFunctionsMapping,
    processed_functions: Mapping[str, UserFunction],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    entered_processes: frozenset[tuple[RegimeName, ProcessName]],
) -> TransitionLaws:
    """Describe every target's transition laws, after synthesis has built them.

    Stochasticity is read off the synthesized functions rather than re-derived
    from the user's declarations: a law is stochastic exactly when a
    target-qualified weight function exists for it. Building the description here
    -- once both explicit and intrinsic laws are in `processed_functions` -- is
    what lets a process the source does not carry be described at all.

    Args:
        transitions: Immutable mapping of target regime names to their bundles of
            unqualified `next_<state>` transition functions.
        processed_functions: Mapping of qualified function names to functions,
            carrying the synthesized `weight_<target>__next_<state>` laws.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        entered_processes: Frozenset of `(target, process)` pairs entered at the
            process's own unconditional law.

    Returns:
        Immutable mapping of target regime names to their transition laws.

    """
    laws: dict[RegimeName, MappingProxyType[TransitionFunctionName, TransitionLawInfo]]
    laws = {}
    for target, bundle in transitions.items():
        target_laws: dict[TransitionFunctionName, TransitionLawInfo] = {}
        for next_state_name in bundle:
            state_name = next_state_name.removeprefix("next_")
            qualified_name = qname_from_tree_path((target, next_state_name))
            weight_name = f"weight_{qualified_name}"
            target_laws[next_state_name] = TransitionLawInfo(
                target=target,
                next_state_name=next_state_name,
                qualified_name=qualified_name,
                stochastic=weight_name in processed_functions,
                continuous_process=isinstance(
                    all_grids.get(target, MappingProxyType({})).get(state_name),
                    _ContinuousStochasticProcess,
                ),
                intrinsic_entry=(target, state_name) in entered_processes,
                weight_name=(
                    weight_name if weight_name in processed_functions else None
                ),
            )
        laws[target] = MappingProxyType(target_laws)
    return MappingProxyType(laws)


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

    Returns:
        The function with renamed parameters.

    """
    # Per-target keys are qnames (`<target>__<func>`) addressing a nested
    # template branch; walk the tree path instead of subscripting directly.
    branch: Mapping[str, object] = regime_params_template
    for part in tree_path_from_qname(names_key if names_key is not None else param_key):
        branch = cast("Mapping[str, object]", branch[part])
    param_names = list(branch)
    if not param_names:
        return cast("EconFunction", func)
    mapper = {p: qname_from_tree_path((param_key, p)) for p in param_names}

    return cast("EconFunction", rename_arguments(func, mapper=mapper))


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


def _get_weights_func_for_process(
    *, name: str, grid: _ContinuousStochasticProcess
) -> UserFunction:
    """Get function that uses linear interpolation to calculate the process weights.

    For processes whose params are supplied at runtime, the grid points and
    transition probabilities are computed inside JIT from those runtime params.

    """
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

    """
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
        *args: FloatND | IntND | int,
        **kwargs: FloatND | IntND | int,
    ) -> FloatND:
        regime_idx = func(*args, **kwargs)
        return jax.nn.one_hot(regime_idx, n_regimes)

    # Pin `__annotations__` on the final wrapper: `concatenate_functions`
    # reads `__annotations__` (not `__signature__`) to reconcile the DAG, and
    # the decorator stack can drop them when `func` carries deferred (PEP 649)
    # annotations through `functools.wraps`.
    wrapped.__annotations__ = {**annotations, "return": "FloatND"}
    return wrapped  # ty: ignore[invalid-return-type]


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
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
    co_map_state_names: tuple[StateName, ...] = (),
    certainty_equivalent: CertaintyEquivalent | None = None,
    grid_schedule: AgeGridSchedule | None = None,
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
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
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Regime transition probability function.
        regime_to_v_interpolation_info: Mapping of regime names to representative
            V-interpolation info (the age-invariant fallback).
        flat_param_names: Frozenset of flat parameter names for the regime.
        certainty_equivalent: Nonlinear certainty equivalent, or `None`.
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
    # not false-share a compiled kernel.
    continuation_info = continuation_info_lookup(
        period_to_regime_v_interp=period_to_regime_v_interp,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )
    group_key = continuation_group_key(
        phase_reachability=phase_reachability,
        source_regime_name=source_regime_name,
        functions=functions,
        constraints=constraints,
        grid_schedule=grid_schedule,
    )

    configs = group_periods_by_key(active_periods, group_key)

    # Build one Q_and_F per distinct group, resolving periodized functions and
    # constraints at the group's representative period. Equal signature ⇒ identical
    # closures, so any period in the group is a valid representative.
    built: dict[tuple[tuple[RegimeName, ...], Hashable], QAndFFunction] = {}
    for key, periods in configs.items():
        period_targets = key[0]
        representative_period = periods[0]
        assert_continuation_grids_agree(
            grid_schedule=grid_schedule,
            target_regimes=period_targets,
            periods=tuple(periods),
        )
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
            period_targets=period_targets,
            transitions=transitions,
            transition_laws=transition_laws,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=continuation_info(representative_period),
            co_map_state_names=co_map_state_names,
            certainty_equivalent=certainty_equivalent,
        )

    return expand_groups_to_periods(configs, built)


def _build_argmax_and_max_Q_over_a_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    enable_jit: bool,
    has_taste_shocks: bool = False,
) -> MappingProxyType[int, ArgmaxQOverAFunction]:
    """Build argmax-and-max-Q-over-a closures for each period.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    With taste shocks, the per-subject Gumbel key is vmapped alongside the
    simulated states.
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


def _build_next_state_vmapped(
    *,
    active_periods: tuple[int, ...],
    phase_reachability: PhaseReachability,
    source_regime_name: RegimeName,
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
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
            transition_laws=transition_laws,
            all_grids=all_grids,
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
