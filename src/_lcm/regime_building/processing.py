import functools
import inspect
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
from dags import concatenate_functions, get_annotations, with_signature
from dags.signature import rename_arguments
from dags.tree import QNAME_DELIMITER, qname_from_tree_path, tree_path_from_qname
from jax import numpy as jnp

from _lcm.engine import (
    Regime,
    SimulationPhase,
    SolutionPhase,
    StateActionSpace,
    Variables,
)
from _lcm.grids import DiscreteGrid, Grid
from _lcm.grids.coordinates import get_irreg_coordinate
from _lcm.params.processing import get_flat_param_names
from _lcm.params.regime_template import create_regime_params_template
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.diagnostics import _build_compute_intermediates_per_period
from _lcm.regime_building.max_Q_over_a import (
    get_argmax_and_max_Q_over_a,
    get_max_Q_over_a,
)
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.next_state import get_next_state_function_for_simulation
from _lcm.regime_building.Q_and_F import (
    get_complete_targets,
    get_Q_and_F,
    get_Q_and_F_terminal,
)
from _lcm.regime_building.stochastic_state_transitions import (
    collect_stochastic_state_transitions,
)
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.regime_building.V import VInterpolationInfo, create_v_interpolation_info
from _lcm.state_action_space import create_state_action_space
from _lcm.typing import (
    ArgmaxQOverAFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    MaxQOverAFunction,
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
    state_pair_grids,
)
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import (
    MarkovTransition,
    SolveSimulateFunctionPair,
    SolveSimulateStatePair,
)
from lcm.typing import Float1D, FloatND, Int1D, IntND, UserFunction


def process_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, Regime]:
    """Process user regimes into canonical regimes.

    Extract state transitions from `user_regime.state_transitions` and
    regime transitions from `user_regime.transition`. For fixed states
    (value `None` in `state_transitions`), an identity transition is
    auto-generated. Stochastic process transitions are generated from the
    grid's intrinsic transition logic.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        ages: The AgeGrid for the model.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        enable_jit: Whether to jit the functions of the canonical regime.

    Returns:
        The processed canonical regimes.

    """
    states_per_regime: dict[RegimeName, set[StateName]] = {
        regime_name: set(user_regime.states.keys())
        for regime_name, user_regime in user_regimes.items()
    }

    nested_transitions: dict[
        RegimeName,
        dict[
            RegimeName | TransitionFunctionName,
            dict[TransitionFunctionName, UserFunction] | UserFunction,
        ],
    ] = {}
    for regime_name, user_regime in user_regimes.items():
        nested_transitions[regime_name] = _extract_transitions_from_regime(
            user_regime=user_regime,
            states_per_regime=states_per_regime,
        )
    # The simulate phase additionally carries each `SolveSimulateStatePair` as a
    # true state and evolves it via `pair.transition`; the solve phase does not.
    simulate_nested_transitions = {
        regime_name: _augment_nested_transitions_with_state_pairs(
            nested_transitions=nested_transitions[regime_name],
            user_regime=user_regime,
            states_per_regime=states_per_regime,
        )
        for regime_name, user_regime in user_regimes.items()
    }
    _validate_categoricals(user_regimes)

    regime_to_variables = MappingProxyType(
        {
            regime_name: from_regime(user_regime)
            for regime_name, user_regime in user_regimes.items()
        }
    )
    all_grids = MappingProxyType(
        {
            regime_name: get_grids(user_regime)
            for regime_name, user_regime in user_regimes.items()
        }
    )

    _fail_if_action_has_batch_size(user_regimes)

    regime_to_v_interpolation_info = MappingProxyType(
        {
            regime_name: create_v_interpolation_info(user_regime)
            for regime_name, user_regime in user_regimes.items()
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
    regimes_to_active_periods = MappingProxyType(
        {
            regime_name: ages.get_periods_where(user_regime.active)
            for regime_name, user_regime in user_regimes.items()
        }
    )

    canonical_regimes: dict[RegimeName, Regime] = {}
    for regime_name, user_regime in user_regimes.items():
        regime_params_template = create_regime_params_template(user_regime)

        solution = _build_solution_phase(
            user_regime=user_regime,
            regime_name=regime_name,
            nested_transitions=nested_transitions[regime_name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            variables=regime_to_variables[regime_name],
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            state_action_space=state_action_spaces[regime_name],
            ages=ages,
            enable_jit=enable_jit,
        )

        simulation = _build_simulation_phase(
            user_regime=user_regime,
            regime_name=regime_name,
            nested_transitions=simulate_nested_transitions[regime_name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            variables=regime_to_variables[regime_name],
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            state_action_space=state_action_spaces[regime_name],
            ages=ages,
            enable_jit=enable_jit,
            solve_transitions=solution.transitions,
            solve_stochastic_transition_names=solution.stochastic_transition_names,
            solve_compute_regime_transition_probs=solution.compute_regime_transition_probs,
        )

        stochastic_state_transitions = collect_stochastic_state_transitions(
            user_regime=user_regime,
            user_regimes=user_regimes,
        )

        canonical_regimes[regime_name] = Regime(
            name=regime_name,
            terminal=user_regime.terminal,
            active_periods=tuple(regimes_to_active_periods[regime_name]),
            regime_params_template=regime_params_template,
            solution=solution,
            simulation=simulation,
            stochastic_state_transitions=stochastic_state_transitions,
        )

    return ensure_containers_are_immutable(canonical_regimes)


def _build_solution_phase(
    *,
    user_regime: UserRegime,
    regime_name: RegimeName,
    nested_transitions: dict[
        RegimeName | TransitionFunctionName,
        dict[TransitionFunctionName, UserFunction] | UserFunction,
    ],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    variables: Variables,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
) -> SolutionPhase:
    """Build all compiled functions for the backward-induction (solve) phase.

    Args:
        regime: The user regime.
        regime_name: The name of the regime.
        nested_transitions: Nested transitions dict for internal processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        variables: States and actions of the regime with kind/topology/process tags.
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.

    Returns:
        Complete solve functions container.

    """
    core = _process_regime_core(
        user_regime=user_regime,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
        phase="solve",
    )

    flat_param_names = frozenset(get_flat_param_names(regime_params_template))

    if user_regime.terminal:
        compute_regime_transition_probs = None
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
            compute_regime_transition_probs=core.next_regime_func,  # ty: ignore[invalid-argument-type]
            grids=all_grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            regime_params_template=regime_params_template,
            is_stochastic=user_regime.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="solve",
        )
        Q_and_F_functions = _build_Q_and_F_per_period(
            regimes_to_active_periods=regimes_to_active_periods,
            functions=core.functions,
            constraints=core.constraints,
            transitions=core.transitions,
            stochastic_transition_names=core.stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            ages=ages,
            flat_param_names=flat_param_names,
        )
        compute_intermediates = _build_compute_intermediates_per_period(
            flat_param_names=flat_param_names,
            regimes_to_active_periods=regimes_to_active_periods,
            functions=core.functions,
            constraints=core.constraints,
            transitions=core.transitions,
            stochastic_transition_names=core.stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            state_action_space=state_action_space,
            grids=all_grids[regime_name],
            ages=ages,
            enable_jit=enable_jit,
        )

    max_Q_over_a = _build_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        grids=all_grids[regime_name],
        enable_jit=enable_jit,
    )

    return SolutionPhase(
        variables=variables,
        grids=all_grids[regime_name],
        functions=core.functions,
        constraints=core.constraints,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        max_Q_over_a=max_Q_over_a,
        compute_intermediates=compute_intermediates,
        _base_state_action_space=state_action_space,
    )


def _build_simulation_phase(
    *,
    user_regime: UserRegime,
    regime_name: RegimeName,
    nested_transitions: dict[
        RegimeName | TransitionFunctionName,
        dict[TransitionFunctionName, UserFunction] | UserFunction,
    ],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    variables: Variables,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
    solve_transitions: TransitionFunctionsMapping,
    solve_stochastic_transition_names: frozenset[TransitionFunctionName],
    solve_compute_regime_transition_probs: RegimeTransitionFunction | None,
) -> SimulationPhase:
    """Build all compiled functions for the forward-simulation phase.

    When the regime has `SolveSimulateFunctionPair` entries, simulate-specific
    function variants are resolved. Otherwise, functions and constraints are
    identical to the solve phase.

    Q_and_F always uses the solve (non-vmapped) regime transition probs because
    it evaluates on the Cartesian grid, not per-subject.

    Args:
        regime: The user regime.
        regime_name: The name of the regime.
        nested_transitions: Nested transitions dict for internal processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        variables: States and actions of the regime with kind/topology/process tags.
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.
        solve_transitions: Transitions from the solve phase (reused).
        solve_stochastic_transition_names: Stochastic transition names from solve
            (reused).
        solve_compute_regime_transition_probs: Solve-phase regime transition prob
            function, used for Q_and_F in both phases.

    Returns:
        Complete simulate functions container.

    """
    core = _process_regime_core(
        user_regime=user_regime,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
        phase="simulate",
    )
    # Only functions/constraints vary by phase; core.transitions and
    # core.stochastic_transition_names are phase-independent and reused from solve.
    functions = core.functions
    constraints = core.constraints

    flat_param_names = frozenset(get_flat_param_names(regime_params_template))

    if user_regime.terminal:
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
            functions=functions,
            compute_regime_transition_probs=core.next_regime_func,  # ty: ignore[invalid-argument-type]
            grids=all_grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            regime_params_template=regime_params_template,
            is_stochastic=user_regime.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="simulate",
        )
        # Q_and_F uses the solve (non-vmapped) regime transition probs since
        # it evaluates on the Cartesian grid, not per-subject. The solve
        # phase built that function unconditionally for non-terminal regimes.
        assert solve_compute_regime_transition_probs is not None  # noqa: S101
        Q_and_F_functions = _build_Q_and_F_per_period(
            regimes_to_active_periods=regimes_to_active_periods,
            functions=functions,
            constraints=constraints,
            transitions=solve_transitions,
            stochastic_transition_names=solve_stochastic_transition_names,
            compute_regime_transition_probs=solve_compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            ages=ages,
            flat_param_names=flat_param_names,
        )

    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        enable_jit=enable_jit,
    )

    # Every published simulate-phase consumer (next_state, the feasibility
    # check, additional targets) reads each `SolveSimulateStatePair` as its
    # carried true value, not the solve-phase imputation. Dropping the pair's
    # solve function turns the name into a leaf supplied by the simulator, and
    # `core.transitions` (built from the pair-augmented nested transitions)
    # carries each pair's `next_<name>` evolution. Only the decision functions
    # built above (Q_and_F / argmax / regime-transition probs) keep the
    # imputation — the agent decides on the value the solved policy was
    # computed for. Without pairs the solve sets are reused unchanged.
    pair_state_names = frozenset(
        name
        for name, spec in user_regime.states.items()
        if isinstance(spec, SolveSimulateStatePair)
    )
    if pair_state_names:
        simulate_functions: EconFunctionsMapping = MappingProxyType(
            {k: v for k, v in core.functions.items() if k not in pair_state_names}
        )
        next_state_transitions = core.transitions
        next_state_stochastic_names = core.stochastic_transition_names
    else:
        simulate_functions = core.functions
        next_state_transitions = solve_transitions
        next_state_stochastic_names = solve_stochastic_transition_names

    next_state = _build_next_state_vmapped(
        functions=simulate_functions,
        transitions=next_state_transitions,
        stochastic_transition_names=next_state_stochastic_names,
        all_grids=all_grids,
        variables=variables,
        regime_params_template=regime_params_template,
        enable_jit=enable_jit,
    )

    pair_grids = state_pair_grids(user_regime)
    return SimulationPhase(
        variables=simulate_variables_from_regime(user_regime),
        grids=MappingProxyType({**all_grids[regime_name], **pair_grids}),
        pair_state_names=frozenset(pair_grids),
        functions=simulate_functions,
        constraints=constraints,
        transitions=next_state_transitions,
        stochastic_transition_names=next_state_stochastic_names,
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

    stochastic_transition_names: frozenset[TransitionFunctionName]
    """Frozenset of stochastic transition function names."""

    next_regime_func: TransitionFunction | None
    """The regime transition function, or `None` for terminal regimes."""


def _process_regime_core(
    *,
    user_regime: UserRegime,
    nested_transitions: dict[
        RegimeName | TransitionFunctionName,
        dict[TransitionFunctionName, UserFunction] | UserFunction,
    ],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    variables: Variables,
    phase: Literal["solve", "simulate"],
) -> _CoreResult:
    """Process regime functions and transitions for a single phase.

    Resolve `SolveSimulateFunctionPair` entries by picking the variant matching
    `phase`, rename params to qualified names, classify and process transitions.

    Args:
        regime: The user regime.
        nested_transitions: Nested transitions dict for internal processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        variables: States and actions of the regime with kind/topology/process tags.
        phase: Which phase variant to use for function pairs.

    Returns:
        Core processing result with functions, constraints, transitions, stochastic
        transition names, and the next_regime function.

    """
    flat_grids = flatten_regime_namespace(all_grids)
    flat_nested_transitions = flatten_regime_namespace(nested_transitions)

    # Resolve phase-variant function entries for this phase.
    resolved_functions: dict[str, UserFunction] = {}
    for name, func in user_regime.functions.items():
        if isinstance(func, SolveSimulateFunctionPair | Phased):
            resolved_functions[name] = cast(
                "UserFunction",
                func.solve if phase == "solve" else func.simulate,
            )
        else:
            resolved_functions[name] = cast("UserFunction", func)

    # A SolveSimulateStatePair contributes its `solve` variant as a derived
    # function under the state's name, so the function DAG computes the imputed
    # value (e.g. pension wealth from AIME).
    for name, spec in user_regime.states.items():
        if isinstance(spec, SolveSimulateStatePair):
            resolved_functions[name] = cast("UserFunction", spec.solve)

    all_functions: dict[str, UserFunction] = {
        **resolved_functions,
        **cast("Mapping[str, UserFunction]", user_regime.constraints),
        **flat_nested_transitions,
    }

    per_target_next_names = frozenset(
        f"next_{name}"
        for name, raw in user_regime.state_transitions.items()
        if isinstance(raw, Mapping) and not isinstance(raw, MarkovTransition)
    )

    stochastic_transition_names = _get_stochastic_transition_names(
        user_regime=user_regime, variables=variables
    )

    stochastic_transition_functions = {
        func_name: func
        for func_name, func in flat_nested_transitions.items()
        if tree_path_from_qname(func_name)[-1] in stochastic_transition_names
        and func_name != "next_regime"
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

    functions: dict[str, EconFunction] = {}

    for func_name, func in deterministic_functions.items():
        functions[func_name] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
        )

    for func_name, func in deterministic_transition_functions.items():
        param_key = _extract_param_key(func_name, per_target_next_names)
        functions[func_name] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=param_key,
        )

    for func_name, func in stochastic_transition_functions.items():
        param_key = _extract_param_key(func_name, per_target_next_names)
        functions[f"weight_{func_name}"] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=param_key,
        )
        functions[func_name] = _get_discrete_markov_next_function(
            func=func,
            grid=flat_grids[func_name.replace("next_", "")].to_jax(),
        )

    # Transitions of continuous stochastic processes bypass the stub pipeline
    # entirely. Build weight and next functions for reachable target regimes
    # from each target's grid. Scope to targets already present in non-process
    # transitions to avoid spurious entries for unreachable regimes.
    process_names = variables.process_names
    reachable_targets = {
        tree_path_from_qname(k)[0]
        for k in flat_nested_transitions
        if QNAME_DELIMITER in k
    }
    target_process_grids: dict[
        tuple[RegimeName, ProcessName], _ContinuousStochasticProcess
    ] = {
        (user_regime, process): grid
        for user_regime, grids in all_grids.items()
        if user_regime in reachable_targets
        for process in process_names
        if isinstance(grid := grids.get(process), _ContinuousStochasticProcess)
    }
    functions |= {
        f"weight_{user_regime}__next_{process}": _get_weights_func_for_process(
            name=process, grid=grid
        )
        for (user_regime, process), grid in target_process_grids.items()
    } | {
        f"{user_regime}__next_{process}": _get_stochastic_next_function_for_process(
            name=process, grid=grid.to_jax()
        )
        for (user_regime, process), grid in target_process_grids.items()
    }

    process_transition_keys = {
        f"{user_regime}__next_{process}"
        for user_regime, process in target_process_grids
    }
    internal_transition = {
        func_name: functions[func_name]
        for func_name in flat_nested_transitions
        if func_name != "next_regime"
    } | {key: functions[key] for key in process_transition_keys}

    constraints: ConstraintFunctionsMapping = MappingProxyType(
        {func_name: functions[func_name] for func_name in user_regime.constraints}
    )
    excluded_from_functions = (
        set(flat_nested_transitions)
        | set(user_regime.constraints)
        | process_transition_keys
    )
    phase_functions = MappingProxyType(
        {
            func_name: functions[func_name]
            for func_name in functions
            if func_name not in excluded_from_functions
        }
    )

    transitions = _wrap_transitions(unflatten_regime_namespace(internal_transition))

    next_regime_func: TransitionFunction | None = functions.get("next_regime")

    return _CoreResult(
        functions=phase_functions,
        constraints=constraints,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        next_regime_func=next_regime_func,
    )


def _extract_transitions_from_regime(
    *,
    user_regime: UserRegime,
    states_per_regime: Mapping[RegimeName, set[StateName]],
) -> dict[
    RegimeName | TransitionFunctionName,
    dict[TransitionFunctionName, UserFunction] | UserFunction,
]:
    """Extract transitions from `regime.state_transitions` and regime transition.

    For non-terminal regimes, reads state transitions from `regime.state_transitions`
    and auto-generates identity transitions for fixed states (`None` values).
    Stochastic process transitions are handled separately during internal
    function processing.

    For per-target dicts, selects the transition function matching each target regime.

    Args:
        regime: The user regime.
        states_per_regime: Mapping of regime names to their state names.

    Returns:
        Nested transitions dict for internal processing.

    """
    if user_regime.terminal:
        return {}

    state_transitions = collect_state_transitions(
        user_regime.states, user_regime.state_transitions
    )
    simple_transitions, per_target_transitions = _classify_transitions(
        state_transitions
    )

    nested: dict[
        RegimeName | TransitionFunctionName,
        dict[TransitionFunctionName, UserFunction] | UserFunction,
    ] = {"next_regime": cast("UserFunction", user_regime.transition)}

    reachable_targets = _get_reachable_targets(
        per_target_transitions=per_target_transitions,
        simple_transitions=simple_transitions,
        states_per_regime=states_per_regime,
        pair_state_names=_pair_state_names(user_regime),
    )

    for target_regime_name in reachable_targets:
        target_regime_state_names = states_per_regime[target_regime_name]
        target_dict: dict[TransitionFunctionName, UserFunction] = {}
        for state_name in target_regime_state_names:
            next_key = f"next_{state_name}"
            if next_key in simple_transitions:
                target_dict[next_key] = simple_transitions[next_key]
            elif next_key in per_target_transitions:
                variants = per_target_transitions[next_key]
                if target_regime_name in variants:
                    target_dict[next_key] = variants[target_regime_name]
        if target_dict:
            nested[target_regime_name] = target_dict

    return nested


def _augment_nested_transitions_with_state_pairs(
    *,
    nested_transitions: dict[
        RegimeName | TransitionFunctionName,
        dict[TransitionFunctionName, UserFunction] | UserFunction,
    ],
    user_regime: UserRegime,
    states_per_regime: Mapping[RegimeName, set[StateName]],
) -> dict[
    RegimeName | TransitionFunctionName,
    dict[TransitionFunctionName, UserFunction] | UserFunction,
]:
    """Add each state pair's simulate transition to every target that carries it.

    The solve phase omits `SolveSimulateStatePair` transitions (the name is a
    derived function there). The simulate phase carries the pair as a true state
    and evolves it via `pair.transition`, registered as `next_<name>` for every
    reachable target regime that also declares the pair as a state. Returns the
    input unchanged when the regime has no state pairs.

    A reachable target absent from the input (no ordinary state transitions
    land there — e.g. retirement keeps only the carried pension wealth) gets a
    fresh entry holding just the pair's `next_<name>`, so the carried value is
    evolved rather than silently frozen on the crossing. Such targets stay
    absent from the solve-phase transitions, which is correct: in solve the
    pair is a derived function in the target regime, not a handed-over state.
    """
    if user_regime.terminal:
        return nested_transitions
    pair_transitions = {
        name: cast("UserFunction", spec.transition)
        for name, spec in user_regime.states.items()
        if isinstance(spec, SolveSimulateStatePair)
    }
    if not pair_transitions:
        return nested_transitions

    augmented: dict[
        RegimeName | TransitionFunctionName,
        dict[TransitionFunctionName, UserFunction] | UserFunction,
    ] = {}
    for key, value in nested_transitions.items():
        # `next_regime`'s value is the regime-transition callable, not a
        # per-target dict; pass it through untouched.
        if not isinstance(value, dict):
            augmented[key] = value
            continue
        target_states = states_per_regime.get(key, set())
        merged: dict[TransitionFunctionName, UserFunction] = dict(
            cast("dict[TransitionFunctionName, UserFunction]", value)
        )
        for name, func in pair_transitions.items():
            if name in target_states:
                merged[f"next_{name}"] = func
        augmented[key] = merged

    simple_transitions, per_target_transitions = _classify_transitions(
        collect_state_transitions(user_regime.states, user_regime.state_transitions)
    )
    reachable_targets = _get_reachable_targets(
        per_target_transitions=per_target_transitions,
        simple_transitions=simple_transitions,
        states_per_regime=states_per_regime,
        pair_state_names=set(pair_transitions),
    )
    for target_regime_name in reachable_targets:
        if target_regime_name in augmented:
            continue
        carried = {
            f"next_{name}": func
            for name, func in pair_transitions.items()
            if name in states_per_regime[target_regime_name]
        }
        if carried:
            augmented[target_regime_name] = carried
    return augmented


def _get_reachable_targets(
    *,
    per_target_transitions: dict[
        TransitionFunctionName, dict[RegimeName, UserFunction]
    ],
    simple_transitions: dict[TransitionFunctionName, UserFunction],
    states_per_regime: Mapping[RegimeName, set[StateName]],
    pair_state_names: set[StateName],
) -> set[RegimeName]:
    """Determine which target regimes need transition entries.

    When per-target transitions exist, start from the explicitly named targets
    and add any target whose state needs are fully covered by simple
    (non-per-target) transitions. A `SolveSimulateStatePair` state supplies its
    own transition, so it never counts toward a target's needs. Without
    per-target transitions, all regimes are reachable.

    """
    if not per_target_transitions:
        return set(states_per_regime.keys())

    targets: set[RegimeName] = set()
    for variants in per_target_transitions.values():
        targets |= variants.keys()
    for target_name, target_states in states_per_regime.items():
        if target_name not in targets:
            needed = {f"next_{s}" for s in target_states - pair_state_names}
            if needed and needed.issubset(simple_transitions):
                targets.add(target_name)
    return targets


def _pair_state_names(user_regime: UserRegime) -> set[StateName]:
    """Return the names of the regime's `SolveSimulateStatePair` states."""
    return {
        name
        for name, spec in user_regime.states.items()
        if isinstance(spec, SolveSimulateStatePair)
    }


def _classify_transitions(
    state_transitions: dict[TransitionFunctionName, UserFunction],
) -> tuple[
    dict[TransitionFunctionName, UserFunction],
    dict[TransitionFunctionName, dict[RegimeName, UserFunction]],
]:
    """Split collected transitions into simple and per-target groups.

    Qualified names like "next_health__working" (produced by
    `collect_state_transitions` for per-target dicts) are decomposed via
    `tree_path_from_qname`.

    Returns:
        Tuple of (simple_transitions, per_target_transitions).

    """
    simple: dict[TransitionFunctionName, UserFunction] = {}
    per_target: dict[TransitionFunctionName, dict[RegimeName, UserFunction]] = {}
    for key, func in state_transitions.items():
        path = tree_path_from_qname(key)
        if len(path) == 1:
            simple[key] = func
        else:
            state_key = path[0]
            target_name = qname_from_tree_path(path[1:])
            per_target.setdefault(state_key, {})[target_name] = func
    return simple, per_target


def _wrap_transitions(
    transitions: dict[RegimeName, dict[TransitionFunctionName, TransitionFunction]],
) -> TransitionFunctionsMapping:
    """Wrap nested transitions dict in MappingProxyType."""
    return MappingProxyType(
        {name: MappingProxyType(inner) for name, inner in transitions.items()}
    )


def _get_stochastic_transition_names(
    *,
    user_regime: UserRegime,
    variables: Variables,
) -> frozenset[TransitionFunctionName]:
    """Compute stochastic transition names from regime state transitions.

    Args:
        regime: The user regime.
        variables: States and actions of the regime with kind/topology/process tags.

    Returns:
        Frozenset of stochastic transition function names (e.g., "next_health").

    """
    markov_state_names: set[StateName] = set()
    for name in user_regime.state_transitions:
        raw = user_regime.state_transitions[name]
        if isinstance(raw, MarkovTransition) or (
            isinstance(raw, Mapping)
            and any(isinstance(v, MarkovTransition) for v in raw.values())
        ):
            markov_state_names.add(name)
    return frozenset(
        f"next_{name}" for name in markov_state_names | set(variables.process_names)
    )


def _rename_params_to_qnames(
    *,
    func: UserFunction,
    regime_params_template: RegimeParamsTemplate,
    param_key: str,
) -> EconFunction:
    """Rename function params to qualified names using dags.signature.rename_arguments.

    E.g., risk_aversion -> utility__risk_aversion.

    Args:
        func: The user function.
        regime_params_template: The parameter template for the regime.
        param_key: The key to look up in regime_params_template (e.g., "utility").

    Returns:
        The function with renamed parameters.

    """
    param_names = list(regime_params_template[param_key])
    if not param_names:
        return cast("EconFunction", func)
    mapper = {p: qname_from_tree_path((param_key, p)) for p in param_names}

    return cast("EconFunction", rename_arguments(func, mapper=mapper))


def _extract_param_key(
    func_name: str,
    per_target_next_names: frozenset[str] = frozenset(),
) -> str:
    """Extract the param template key from a possibly prefixed function name.

    For prefixed names like "work__next_wealth", returns "next_wealth".
    For per-target transitions like "work__next_health" where "next_health" is in
    `per_target_next_names`, returns "to_work_next_health" to match the template key.
    For unprefixed names like "next_regime", returns the name unchanged.

    """
    path = tree_path_from_qname(func_name)
    if len(path) > 1:
        suffix = qname_from_tree_path(path[1:])
        if suffix in per_target_next_names:
            return f"to_{path[0]}_{suffix}"
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

            for target_name, target_regime in user_regimes.items():
                target_grid = target_regime.states.get(state_name)
                if not isinstance(target_grid, DiscreteGrid):
                    continue

                if source_grid.categories != target_grid.categories:
                    error_messages.append(
                        f"Discrete state '{state_name}' in regime '{source_name}' "
                        f"has categories {source_grid.categories}, but regime "
                        f"'{target_name}' has categories "
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

    Returns None if the transition is a per-target dict, None (identity), not a
    DiscreteGrid, or the state is not present in the source regime.

    """
    # Per-target dicts handle category differences explicitly
    if isinstance(raw, Mapping) and not isinstance(raw, MarkovTransition):
        return None
    # None means identity (fixed state) — only maps within its own regime
    if raw is None:
        return None
    # Target-only state — no source grid to compare
    if state_name not in user_regime.states:
        return None
    source_grid = user_regime.states[state_name]
    return source_grid if isinstance(source_grid, DiscreteGrid) else None


def build_regime_transition_probs_functions(
    *,
    functions: EconFunctionsMapping,
    compute_regime_transition_probs: TransitionFunction,
    grids: MappingProxyType[StateOrActionName, Grid],
    regime_names_to_ids: RegimeNamesToIds,
    regime_params_template: RegimeParamsTemplate,
    is_stochastic: bool,
    enable_jit: bool,
    phase: Literal["solve", "simulate"],
) -> RegimeTransitionFunction | VmappedRegimeTransitionFunction:
    """Build a regime transition probability function for the given phase.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        compute_regime_transition_probs: The user's next_regime function.
        grids: Immutable mapping of state and action variable names to grid objects.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        regime_params_template: The regime's parameter template.
        is_stochastic: Whether the regime transition is stochastic.
        enable_jit: Whether to JIT-compile the functions.
        phase: Which phase to build for.

    """
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
            regime_params_template=regime_params_template,
        ),
    )

    return jax.jit(next_regime_vmapped) if enable_jit else next_regime_vmapped


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
    return wrapped


def _get_vmap_params(
    *,
    all_args: tuple[str, ...],
    regime_params_template: RegimeParamsTemplate,
) -> tuple[str, ...]:
    """Get parameter names that should be vmapped (states and actions)."""
    non_vmap = {"period", "age"} | get_flat_param_names(regime_params_template)
    return tuple(arg for arg in all_args if arg not in non_vmap)


def _build_Q_and_F_per_period(
    *,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    ages: AgeGrid,
    flat_param_names: frozenset[str],
) -> MappingProxyType[int, QAndFFunction]:
    """Build Q-and-F closures for each period of a non-terminal regime.

    Periods sharing the same target-regime configuration reuse a single
    closure, reducing the number of distinct JIT compilations. The caller
    is responsible for handling terminal regimes.

    Args:
        regimes_to_active_periods: Immutable mapping of regime names to
            their active period tuples.
        functions: Immutable mapping of internal user functions.
        constraints: Immutable mapping of constraint functions.
        transitions: Immutable mapping of regime-to-regime transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition
            function names.
        compute_regime_transition_probs: Regime transition probability
            function for the current regime.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.
        ages: Age grid for the model.
        flat_param_names: Frozenset of flat parameter names for the regime.

    Returns:
        Immutable mapping of period index to the per-period Q-and-F closure.

    """
    # Group periods by target configuration
    configs: dict[tuple[RegimeName, ...], list[int]] = {}
    for period in range(ages.n_periods):
        complete = get_complete_targets(
            period=period,
            transitions=transitions,
            regimes_to_active_periods=regimes_to_active_periods,
            stochastic_transition_names=stochastic_transition_names,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )
        configs.setdefault(complete, []).append(period)

    # Build one Q_and_F per distinct configuration
    built: dict[tuple[RegimeName, ...], QAndFFunction] = {}
    for complete_targets in configs:
        built[complete_targets] = get_Q_and_F(
            flat_param_names=flat_param_names,
            functions=functions,
            constraints=constraints,
            complete_targets=complete_targets,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )

    # Map each period to its group's function
    result: dict[int, QAndFFunction] = {}
    for key, periods in configs.items():
        for period in periods:
            result[period] = built[key]

    return MappingProxyType(result)


def _build_max_Q_over_a_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    grids: MappingProxyType[StateOrActionName, Grid],
    enable_jit: bool,
) -> MappingProxyType[int, MaxQOverAFunction]:
    """Build max-Q-over-a closures for each period.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    """
    built: dict[int, MaxQOverAFunction] = {}
    result: dict[int, MaxQOverAFunction] = {}
    for period, Q_and_F in Q_and_F_functions.items():
        q_id = id(Q_and_F)
        if q_id not in built:
            func = get_max_Q_over_a(
                Q_and_F=Q_and_F,
                batch_sizes={
                    name: grid.batch_size
                    for name, grid in grids.items()
                    if name in state_action_space.state_names
                },
                action_names=state_action_space.action_names,
                state_names=state_action_space.state_names,
            )
            built[q_id] = jax.jit(func) if enable_jit else func
        result[period] = built[q_id]
    return MappingProxyType(result)


def _build_argmax_and_max_Q_over_a_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    enable_jit: bool,
) -> MappingProxyType[int, ArgmaxQOverAFunction]:
    """Build argmax-and-max-Q-over-a closures for each period.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    """
    built: dict[int, ArgmaxQOverAFunction] = {}
    result: dict[int, ArgmaxQOverAFunction] = {}
    for period, Q_and_F in Q_and_F_functions.items():
        q_id = id(Q_and_F)
        if q_id not in built:
            func = get_argmax_and_max_Q_over_a(
                Q_and_F=Q_and_F,
                action_names=state_action_space.action_names,
                state_names=state_action_space.state_names,
            )
            if enable_jit:
                func = jax.jit(func)
            built[q_id] = simulation_spacemap(
                func=func,
                action_names=(),
                state_names=tuple(state_action_space.states),
            )
        result[period] = built[q_id]
    return MappingProxyType(result)


def _build_next_state_vmapped(
    *,
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    variables: Variables,
    regime_params_template: RegimeParamsTemplate,
    enable_jit: bool,
) -> NextStateSimulationFunction:
    """Build a vmapped next-state function for simulation."""
    next_state = get_next_state_function_for_simulation(
        functions=functions,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        all_grids=all_grids,
        variables=variables,
    )
    sig_args = tuple(inspect.signature(next_state).parameters)

    non_vmap = {"period", "age"} | get_flat_param_names(regime_params_template)
    vmap_variables = tuple(arg for arg in sig_args if arg not in non_vmap)

    next_state_vmapped = vmap_1d(func=next_state, variables=vmap_variables)
    next_state_vmapped = with_signature(
        next_state_vmapped, kwargs=sig_args, enforce=False
    )

    return jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped


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
            if grid.batch_size != 0:
                msg = (
                    f"batch_size > 0 is not supported on action grids. Only state "
                    f"grids can be batched. Found batch_size={grid.batch_size} on "
                    f"action '{action_name}' in regime '{regime_name}'."
                )
                raise ValueError(msg)
