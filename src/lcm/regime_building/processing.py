import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
import pandas as pd
from dags import concatenate_functions, get_annotations, with_signature
from dags.signature import rename_arguments
from dags.tree import QNAME_DELIMITER, qname_from_tree_path, tree_path_from_qname
from jax import Array
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, format_messages
from lcm.grids import DiscreteGrid, Grid
from lcm.grids.coordinates import get_irreg_coordinate
from lcm.interfaces import (
    InternalRegime,
    SimulateFunctions,
    SolveFunctions,
    SolveSimulateFunctionPair,
    StateActionSpace,
)
from lcm.params.processing import get_flat_param_names
from lcm.params.regime_template import create_regime_params_template
from lcm.regime import MarkovTransition, Regime
from lcm.regime_building.max_Q_over_a import (
    get_argmax_and_max_Q_over_a,
    get_max_Q_over_a,
)
from lcm.regime_building.ndimage import map_coordinates
from lcm.regime_building.next_state import get_next_state_function_for_simulation
from lcm.regime_building.Q_and_F import (
    get_compute_intermediates,
    get_Q_and_F,
    get_Q_and_F_terminal,
)
from lcm.regime_building.V import VInterpolationInfo, create_v_interpolation_info
from lcm.regime_building.validation import collect_state_transitions
from lcm.regime_building.variable_info import get_grids, get_variable_info
from lcm.shocks import _ShockGrid
from lcm.state_action_space import create_state_action_space
from lcm.typing import (
    ArgmaxQOverAFunction,
    Float1D,
    FunctionsMapping,
    Int1D,
    InternalUserFunction,
    MaxQOverAFunction,
    NextStateSimulationFunction,
    QAndFFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
    TransitionFunctionsMapping,
    UserFunction,
    VmappedRegimeTransitionFunction,
)
from lcm.utils.containers import ensure_containers_are_immutable
from lcm.utils.dispatchers import productmap, simulation_spacemap, vmap_1d
from lcm.utils.namespace import flatten_regime_namespace, unflatten_regime_namespace


def process_regimes(
    *,
    regimes: Mapping[str, Regime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, InternalRegime]:
    """Process user regimes into internal regimes.

    Extract state transitions from `regime.state_transitions` and
    regime transitions from `regime.transition`. For fixed states (value `None`
    in `state_transitions`), an identity transition is auto-generated. ShockGrid
    transitions are generated from the grid's intrinsic transition logic.

    Args:
        regimes: Mapping of regime names to Regime instances.
        ages: The AgeGrid for the model.
        regime_names_to_ids: Immutable mapping from regime names to integer indices.
        enable_jit: Whether to jit the functions of the internal regime.

    Returns:
        The processed regimes.

    """
    states_per_regime: dict[RegimeName, set[str]] = {
        name: set(regime.states.keys()) for name, regime in regimes.items()
    }

    nested_transitions = {}
    for name, regime in regimes.items():
        nested_transitions[name] = _extract_transitions_from_regime(
            regime=regime,
            states_per_regime=states_per_regime,
        )
    _validate_categoricals(regimes)

    variable_info = MappingProxyType(
        {n: get_variable_info(r) for n, r in regimes.items()}
    )
    all_grids = MappingProxyType({n: get_grids(r) for n, r in regimes.items()})

    _fail_if_action_has_batch_size(regimes)

    regime_to_v_interpolation_info = MappingProxyType(
        {n: create_v_interpolation_info(r) for n, r in regimes.items()}
    )
    state_action_spaces = MappingProxyType(
        {
            n: create_state_action_space(
                variable_info=variable_info[n], grids=all_grids[n]
            )
            for n in regimes
        }
    )
    regimes_to_active_periods = MappingProxyType(
        {n: ages.get_periods_where(r.active) for n, r in regimes.items()}
    )

    internal_regimes = {}
    for name, regime in regimes.items():
        regime_params_template = create_regime_params_template(regime)

        solve_functions = _build_solve_functions(
            regime=regime,
            regime_name=name,
            nested_transitions=nested_transitions[name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            variable_info=variable_info[name],
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            state_action_space=state_action_spaces[name],
            ages=ages,
            enable_jit=enable_jit,
        )

        simulate_functions = _build_simulate_functions(
            regime=regime,
            regime_name=name,
            nested_transitions=nested_transitions[name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            variable_info=variable_info[name],
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            state_action_space=state_action_spaces[name],
            ages=ages,
            enable_jit=enable_jit,
            solve_transitions=solve_functions.transitions,
            solve_stochastic_transition_names=solve_functions.stochastic_transition_names,
            solve_compute_regime_transition_probs=solve_functions.compute_regime_transition_probs,
        )

        internal_regimes[name] = InternalRegime(
            name=name,
            terminal=regime.terminal,
            grids=all_grids[name],
            variable_info=variable_info[name],
            active_periods=tuple(regimes_to_active_periods[name]),
            regime_params_template=regime_params_template,
            solve_functions=solve_functions,
            simulate_functions=simulate_functions,
            _base_state_action_space=state_action_spaces[name],
        )

    return ensure_containers_are_immutable(internal_regimes)


def _build_solve_functions(
    *,
    regime: Regime,
    regime_name: RegimeName,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[str, Grid]],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    variable_info: pd.DataFrame,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
) -> SolveFunctions:
    """Build all compiled functions for the backward-induction (solve) phase.

    Args:
        regime: The user regime.
        regime_name: The name of the regime.
        nested_transitions: Nested transitions dict for internal processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        regime_names_to_ids: Mapping from regime names to integer indices.
        variable_info: Variable info of the regime.
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.

    Returns:
        Complete solve functions container.

    """
    core = _process_regime_core(
        regime=regime,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variable_info=variable_info,
        phase="solve",
    )

    if regime.terminal:
        compute_regime_transition_probs = None
    else:
        compute_regime_transition_probs = build_regime_transition_probs_functions(
            functions=core.functions,
            compute_regime_transition_probs=core.next_regime_func,  # ty: ignore[invalid-argument-type]
            grids=all_grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            regime_params_template=regime_params_template,
            is_stochastic=regime.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="solve",
        )

    Q_and_F_functions = _build_Q_and_F_per_period(
        regime=regime,
        regimes_to_active_periods=regimes_to_active_periods,
        functions=core.functions,
        constraints=core.constraints,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        ages=ages,
        regime_params_template=regime_params_template,
    )

    max_Q_over_a = _build_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        grids=all_grids[regime_name],
        enable_jit=enable_jit,
    )

    compute_intermediates = _build_compute_intermediates_per_period(
        regime=regime,
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

    return SolveFunctions(
        functions=core.functions,
        constraints=core.constraints,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        max_Q_over_a=max_Q_over_a,
        compute_intermediates=compute_intermediates,
    )


def _build_simulate_functions(
    *,
    regime: Regime,
    regime_name: RegimeName,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[str, Grid]],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    variable_info: pd.DataFrame,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
    solve_transitions: TransitionFunctionsMapping,
    solve_stochastic_transition_names: frozenset[str],
    solve_compute_regime_transition_probs: RegimeTransitionFunction | None,
) -> SimulateFunctions:
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
        regime_names_to_ids: Mapping from regime names to integer indices.
        variable_info: Variable info of the regime.
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
        regime=regime,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variable_info=variable_info,
        phase="simulate",
    )
    # Only functions/constraints vary by phase; core.transitions and
    # core.stochastic_transition_names are phase-independent and reused from solve.
    functions = core.functions
    constraints = core.constraints

    if regime.terminal:
        compute_regime_transition_probs = None
    else:
        compute_regime_transition_probs = build_regime_transition_probs_functions(
            functions=functions,
            compute_regime_transition_probs=core.next_regime_func,  # ty: ignore[invalid-argument-type]
            grids=all_grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            regime_params_template=regime_params_template,
            is_stochastic=regime.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="simulate",
        )

    # Q_and_F uses the solve (non-vmapped) regime transition probs since it
    # evaluates on the Cartesian grid, not per-subject.
    Q_and_F_functions = _build_Q_and_F_per_period(
        regime=regime,
        regimes_to_active_periods=regimes_to_active_periods,
        functions=functions,
        constraints=constraints,
        transitions=solve_transitions,
        stochastic_transition_names=solve_stochastic_transition_names,
        compute_regime_transition_probs=solve_compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        ages=ages,
        regime_params_template=regime_params_template,
    )

    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        enable_jit=enable_jit,
    )

    # State transitions are phase-independent (only utility/H have phase variants),
    # so core.functions from either phase produces the same transitions.
    next_state = _build_next_state_vmapped(
        functions=core.functions,
        transitions=solve_transitions,
        stochastic_transition_names=solve_stochastic_transition_names,
        all_grids=all_grids,
        variable_info=variable_info,
        regime_params_template=regime_params_template,
        enable_jit=enable_jit,
    )

    return SimulateFunctions(
        functions=functions,
        constraints=constraints,
        transitions=solve_transitions,
        stochastic_transition_names=solve_stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        argmax_and_max_Q_over_a=argmax_and_max_Q_over_a,
        next_state=next_state,
    )


@dataclass(frozen=True)
class _CoreResult:
    """Result of core regime function processing for one phase."""

    functions: FunctionsMapping
    """User functions (utility, helpers) with params renamed to qnames."""

    constraints: FunctionsMapping
    """Constraint functions with params renamed to qnames."""

    transitions: TransitionFunctionsMapping
    """Nested mapping of transition names to transition functions."""

    stochastic_transition_names: frozenset[str]
    """Frozenset of stochastic transition function names."""

    next_regime_func: InternalUserFunction | None
    """The regime transition function, or `None` for terminal regimes."""


def _process_regime_core(
    *,
    regime: Regime,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[str, Grid]],
    regime_params_template: RegimeParamsTemplate,
    variable_info: pd.DataFrame,
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
        variable_info: Variable info of the regime.
        phase: Which phase variant to use for function pairs.

    Returns:
        Core processing result with functions, constraints, transitions, stochastic
        transition names, and the next_regime function.

    """
    flat_grids = flatten_regime_namespace(all_grids)
    flat_nested_transitions = flatten_regime_namespace(nested_transitions)

    # Resolve function pairs for this phase.
    resolved_functions: dict[str, UserFunction] = {}
    for name, func in regime.functions.items():
        if isinstance(func, SolveSimulateFunctionPair):
            resolved_functions[name] = cast(
                "UserFunction",
                func.solve if phase == "solve" else func.simulate,
            )
        else:
            resolved_functions[name] = func

    all_functions: dict[str, UserFunction] = {
        **resolved_functions,
        **regime.constraints,
        **flat_nested_transitions,
    }

    per_target_next_names = frozenset(
        f"next_{name}"
        for name, raw in regime.state_transitions.items()
        if isinstance(raw, Mapping) and not isinstance(raw, MarkovTransition)
    )

    stochastic_transition_names = _get_stochastic_transition_names(
        regime=regime, variable_info=variable_info
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

    functions: dict[str, InternalUserFunction] = {}

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

    # Shock transitions bypass the stub pipeline entirely. Build weight and
    # next functions for reachable target regimes from each target's grid.
    # Scope to targets already present in non-shock transitions to avoid
    # spurious entries for unreachable regimes.
    shock_names = variable_info.query("is_shock").index.tolist()
    reachable_targets = {
        tree_path_from_qname(k)[0]
        for k in flat_nested_transitions
        if QNAME_DELIMITER in k
    }
    target_shock_grids: dict[tuple[RegimeName, str], _ShockGrid] = {
        (regime, shock): grid
        for regime, grids in all_grids.items()
        if regime in reachable_targets
        for shock in shock_names
        if isinstance(grid := grids.get(shock), _ShockGrid)
    }
    functions |= {
        f"weight_{regime}__next_{shock}": _get_weights_func_for_shock(
            name=shock, grid=grid
        )
        for (regime, shock), grid in target_shock_grids.items()
    } | {
        f"{regime}__next_{shock}": _get_stochastic_next_function_for_shock(
            name=shock, grid=grid.to_jax()
        )
        for (regime, shock), grid in target_shock_grids.items()
    }

    shock_transition_keys = {
        f"{regime}__next_{shock}" for regime, shock in target_shock_grids
    }
    internal_transition = {
        func_name: functions[func_name]
        for func_name in flat_nested_transitions
        if func_name != "next_regime"
    } | {key: functions[key] for key in shock_transition_keys}

    constraints = MappingProxyType(
        {func_name: functions[func_name] for func_name in regime.constraints}
    )
    excluded_from_functions = (
        set(flat_nested_transitions) | set(regime.constraints) | shock_transition_keys
    )
    phase_functions = MappingProxyType(
        {
            func_name: functions[func_name]
            for func_name in functions
            if func_name not in excluded_from_functions
        }
    )

    transitions = _wrap_transitions(unflatten_regime_namespace(internal_transition))

    next_regime_func = functions.get("next_regime")

    return _CoreResult(
        functions=phase_functions,
        constraints=constraints,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        next_regime_func=next_regime_func,
    )


def _extract_transitions_from_regime(
    *,
    regime: Regime,
    states_per_regime: Mapping[RegimeName, set[str]],
) -> dict[str, dict[str, UserFunction] | UserFunction]:
    """Extract transitions from `regime.state_transitions` and regime transition.

    For non-terminal regimes, reads state transitions from `regime.state_transitions`
    and auto-generates identity transitions for fixed states (`None` values).
    ShockGrid transitions are handled separately during internal function processing.

    For per-target dicts, selects the transition function matching each target regime.

    Args:
        regime: The user regime.
        states_per_regime: Mapping of regime names to their state names.

    Returns:
        Nested transitions dict for internal processing.

    """
    if regime.terminal:
        return {}

    state_transitions = collect_state_transitions(
        regime.states, regime.state_transitions
    )
    simple_transitions, per_target_transitions = _classify_transitions(
        state_transitions
    )

    nested = cast(
        "dict[str, dict[str, UserFunction] | UserFunction]",
        {"next_regime": regime.transition},
    )

    reachable_targets = _get_reachable_targets(
        per_target_transitions=per_target_transitions,
        simple_transitions=simple_transitions,
        states_per_regime=states_per_regime,
    )

    for target_regime_name in reachable_targets:
        target_regime_state_names = states_per_regime[target_regime_name]
        target_dict: dict[str, UserFunction] = {}
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


def _get_reachable_targets(
    *,
    per_target_transitions: dict[str, dict[str, UserFunction]],
    simple_transitions: dict[str, UserFunction],
    states_per_regime: Mapping[RegimeName, set[str]],
) -> set[RegimeName]:
    """Determine which target regimes need transition entries.

    When per-target transitions exist, start from the explicitly named targets
    and add any target whose state needs are fully covered by simple
    (non-per-target) transitions. Without per-target transitions, all regimes
    are reachable.

    """
    if not per_target_transitions:
        return set(states_per_regime.keys())

    targets: set[RegimeName] = set()
    for variants in per_target_transitions.values():
        targets |= variants.keys()
    for target_name, target_states in states_per_regime.items():
        if target_name not in targets:
            needed = {f"next_{s}" for s in target_states}
            if needed and needed.issubset(simple_transitions):
                targets.add(target_name)
    return targets


def _classify_transitions(
    state_transitions: dict[str, UserFunction],
) -> tuple[dict[str, UserFunction], dict[str, dict[str, UserFunction]]]:
    """Split collected transitions into simple and per-target groups.

    Qualified names like "next_health__working" (produced by
    `collect_state_transitions` for per-target dicts) are decomposed via
    `tree_path_from_qname`.

    Returns:
        Tuple of (simple_transitions, per_target_transitions).

    """
    simple: dict[str, UserFunction] = {}
    per_target: dict[str, dict[str, UserFunction]] = {}
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
    transitions: dict[RegimeName, dict[str, InternalUserFunction]],
) -> TransitionFunctionsMapping:
    """Wrap nested transitions dict in MappingProxyType."""
    return MappingProxyType(
        {name: MappingProxyType(inner) for name, inner in transitions.items()}
    )


def _get_stochastic_transition_names(
    *,
    regime: Regime,
    variable_info: pd.DataFrame,
) -> frozenset[str]:
    """Compute stochastic transition names from regime state transitions and shocks.

    Args:
        regime: The user regime.
        variable_info: Variable info of the regime.

    Returns:
        Frozenset of stochastic transition function names (e.g., "next_health").

    """
    markov_state_names: set[str] = set()
    for name in regime.state_transitions:
        raw = regime.state_transitions[name]
        if isinstance(raw, MarkovTransition) or (
            isinstance(raw, Mapping)
            and any(isinstance(v, MarkovTransition) for v in raw.values())
        ):
            markov_state_names.add(name)
    shock_state_names = set(variable_info.query("is_shock").index.tolist())
    return frozenset(f"next_{name}" for name in markov_state_names | shock_state_names)


def _rename_params_to_qnames(
    *,
    func: UserFunction,
    regime_params_template: RegimeParamsTemplate,
    param_key: str,
) -> InternalUserFunction:
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
        return cast("InternalUserFunction", func)
    mapper = {p: qname_from_tree_path((param_key, p)) for p in param_names}

    return cast("InternalUserFunction", rename_arguments(func, mapper=mapper))


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


def _get_stochastic_next_function_for_shock(
    *, name: str, grid: Float1D
) -> UserFunction:
    """Get function that returns the indices in the vf arr of the next shock states."""

    @with_signature(args={f"{name}": "ContinuousState"}, return_annotation="Int1D")
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return jnp.arange(grid.shape[0])

    return next_func


def _get_weights_func_for_shock(*, name: str, grid: _ShockGrid) -> UserFunction:
    """Get function that uses linear interpolation to calculate the shock weights.

    For shocks whose params are supplied at runtime, the grid points and transition
    probabilities are computed inside JIT from those runtime params.

    """
    if grid.params_to_pass_at_runtime:
        n_points = grid.n_points
        fixed_params = dict(grid.params)
        runtime_param_names = {
            qname_from_tree_path((name, p)): p for p in grid.params_to_pass_at_runtime
        }
        args = {name: "ContinuousState", **dict.fromkeys(runtime_param_names, "float")}

        @with_signature(args=args, return_annotation="FloatND", enforce=False)
        def weights_func_runtime(*a: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
            shock_kw: dict[str, float] = {  # ty: ignore[invalid-assignment]
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
            gridpoints = grid.compute_gridpoints(**shock_kw)
            transition_probs = grid.compute_transition_probs(**shock_kw)
            coord = get_irreg_coordinate(value=kwargs[name], points=gridpoints)
            return map_coordinates(
                input=transition_probs,
                coordinates=[
                    jnp.full(n_points, fill_value=coord),
                    jnp.arange(n_points),
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
    def weights_func(*args: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
        coordinate = get_irreg_coordinate(value=kwargs[f"{name}"], points=gridpoints)
        return map_coordinates(
            input=transition_probs,
            coordinates=[
                jnp.full(grid.n_points, fill_value=coordinate),
                jnp.arange(grid.n_points),
            ],
        )

    return weights_func


def _validate_categoricals(
    regimes: Mapping[str, Regime],
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

    for source_name, source_regime in regimes.items():
        if source_regime.terminal:
            continue

        for state_name, raw in source_regime.state_transitions.items():
            source_grid = _get_simple_transition_discrete_grid(
                source_regime, state_name, raw
            )
            if source_grid is None:
                continue

            for target_name, target_regime in regimes.items():
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
    _validate_ordered_flags(regimes, error_messages)

    if error_messages:
        raise ModelInitializationError(format_messages(error_messages))


def compute_merged_discrete_categories(
    regimes: Mapping[str, Regime],
) -> tuple[dict[str, tuple[str, ...]], dict[str, bool]]:
    """Compute merged categories and ordered flags for all discrete variables.

    Returns:
        Tuple of (categories dict, ordered_flags dict).

    """
    var_grids: dict[str, list[tuple[str, DiscreteGrid]]] = {}
    for regime_name, regime in regimes.items():
        for var_name, grid in {**regime.states, **regime.actions}.items():
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
    regimes: Mapping[str, Regime],
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
    for regime_name, regime in regimes.items():
        for var_name, grid in {**regime.states, **regime.actions}.items():
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
    from collections import defaultdict  # noqa: PLC0415

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
    regime: Regime,
    state_name: str,
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
    if state_name not in regime.states:
        return None
    source_grid = regime.states[state_name]
    return source_grid if isinstance(source_grid, DiscreteGrid) else None


def build_regime_transition_probs_functions(
    *,
    functions: FunctionsMapping,
    compute_regime_transition_probs: InternalUserFunction,
    grids: MappingProxyType[str, Grid],
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
        grids: Immutable mapping of grid names to grid objects.
        regime_names_to_ids: Mapping from regime names to integer indices.
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
    func: InternalUserFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> InternalUserFunction:
    """Wrap next_regime function to convert array output to dict format.

    The next_regime function returns a JAX array of probabilities indexed by
    the regime's id. This wrapper converts the array to dict format for internal
    processing.

    Args:
        func: The user's next_regime function (with qname parameters).
        regime_names_to_ids: Mapping from regime names to integer indices.

    Returns:
        A wrapped function that returns MappingProxyType[str, float|Array].

    """
    # Get regime names in index order from regime_names_to_ids
    regime_names_by_id: list[tuple[int, str]] = sorted(
        [(idx, name) for name, idx in regime_names_to_ids.items()],
        key=lambda x: x[0],
    )
    regime_names = [name for _, name in regime_names_by_id]

    annotations = get_annotations(func)
    return_annotation = annotations.pop("return", "dict[str, Any]")

    @with_signature(
        args=annotations,
        return_annotation=return_annotation,
    )
    @functools.wraps(func)
    def wrapped(
        *args: Array | int,
        **kwargs: Array | int,
    ) -> MappingProxyType[str, Any]:
        result = func(*args, **kwargs)
        # Convert array to dict using ordering by regime id
        return MappingProxyType(
            {name: result[idx] for idx, name in enumerate(regime_names)}
        )

    return wrapped


def _wrap_deterministic_regime_transition(
    *,
    func: InternalUserFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> InternalUserFunction:
    """Wrap deterministic next_regime to return one-hot probability array.

    Converts a deterministic regime transition function that returns an integer
    regime ID to a function that returns a one-hot probability array, matching
    the interface of stochastic regime transitions.

    Args:
        func: The user's deterministic next_regime function (returns int).
        regime_names_to_ids: Mapping from regime names to integer indices.

    Returns:
        A wrapped function that returns a one-hot probability array.

    """
    n_regimes = len(regime_names_to_ids)

    # Preserve original annotations but update return type
    annotations = {k: v for k, v in get_annotations(func).items() if k != "return"}

    @with_signature(args=annotations, return_annotation="Array")
    @functools.wraps(func)
    def wrapped(
        *args: Array | int,
        **kwargs: Array | int,
    ) -> Array:
        regime_idx = func(*args, **kwargs)
        return jax.nn.one_hot(regime_idx, n_regimes)

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
    regime: Regime,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[str],
    compute_regime_transition_probs: RegimeTransitionFunction | None,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    ages: AgeGrid,
    regime_params_template: RegimeParamsTemplate,
) -> MappingProxyType[int, QAndFFunction]:
    """Build Q-and-F closures for each period."""
    flat_param_names = frozenset(get_flat_param_names(regime_params_template))

    Q_and_F_functions = {}
    for period, age in enumerate(ages.values):
        if regime.terminal:
            Q_and_F_functions[period] = get_Q_and_F_terminal(
                flat_param_names=flat_param_names,
                age=age,
                period=period,
                functions=functions,
                constraints=constraints,
            )
        else:
            assert compute_regime_transition_probs is not None  # noqa: S101
            Q_and_F_functions[period] = get_Q_and_F(
                flat_param_names=flat_param_names,
                age=age,
                period=period,
                functions=functions,
                constraints=constraints,
                transitions=transitions,
                stochastic_transition_names=stochastic_transition_names,
                regimes_to_active_periods=regimes_to_active_periods,
                compute_regime_transition_probs=compute_regime_transition_probs,
                regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            )

    return MappingProxyType(Q_and_F_functions)


def _build_compute_intermediates_per_period(
    *,
    regime: Regime,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[str],
    compute_regime_transition_probs: RegimeTransitionFunction | None,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    state_action_space: StateActionSpace,
    grids: MappingProxyType[str, Grid],
    ages: AgeGrid,
    enable_jit: bool,
) -> MappingProxyType[int, Callable]:
    """Build diagnostic intermediate closures for each period.

    The closures return all Q_and_F intermediates over the full state-action
    space. Used in the error path when `validate_V` detects NaN. They follow
    the same productmap + JIT structure as `max_Q_over_a`.

    """
    if regime.terminal:
        return MappingProxyType({})

    assert compute_regime_transition_probs is not None  # noqa: S101

    state_batch_sizes = {
        name: grid.batch_size
        for name, grid in grids.items()
        if name in state_action_space.state_names
    }

    variable_names = (
        *state_action_space.state_names,
        *state_action_space.action_names,
    )

    intermediates: dict[int, Callable] = {}
    for period, age in enumerate(ages.values):
        scalar = get_compute_intermediates(
            age=age,
            period=period,
            functions=functions,
            constraints=constraints,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            regimes_to_active_periods=regimes_to_active_periods,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )
        mapped = _productmap_over_state_action_space(
            func=scalar,
            action_names=state_action_space.action_names,
            state_names=state_action_space.state_names,
            state_batch_sizes=state_batch_sizes,
        )
        fused = _wrap_with_reduction(
            func=mapped,
            variable_names=variable_names,
        )
        intermediates[period] = jax.jit(fused) if enable_jit else fused

    return MappingProxyType(intermediates)


def _wrap_with_reduction(
    *,
    func: Callable,
    variable_names: tuple[str, ...],
) -> Callable:
    """Fuse a productmap'd intermediates function with on-device reductions.

    The wrapped function returns a flat pytree of scalars and per-dimension
    vectors instead of full state-action-shaped arrays. When JIT-compiled,
    XLA can fuse the compute and reduce steps so the full-shape
    intermediates never materialise.

    Returns:
        Callable taking the same kwargs as `func` and returning a dict with
        `{Y}_overall` scalars and `{Y}_by_{name}` vectors for `Y` in
        {`U_nan`, `E_nan`, `Q_nan`, `F_feasible`}, plus `regime_probs` as
        a dict of per-target scalar means.

    """

    def reduced(**kwargs: Array) -> dict[str, Any]:
        U_arr, F_arr, E_next_V, Q_arr, regime_probs = func(**kwargs)
        arrays: dict[str, Array] = {
            "U_nan": jnp.isnan(U_arr).astype(float),
            "E_nan": jnp.isnan(E_next_V).astype(float),
            "Q_nan": jnp.isnan(Q_arr).astype(float),
            "F_feasible": F_arr.astype(float),
        }
        out: dict[str, Any] = {}
        for key, arr in arrays.items():
            out[f"{key}_overall"] = jnp.mean(arr)
            for i, name in enumerate(variable_names):
                if i < arr.ndim:
                    axes = tuple(j for j in range(arr.ndim) if j != i)
                    out[f"{key}_by_{name}"] = jnp.mean(arr, axis=axes)
        out["regime_probs"] = {k: jnp.mean(v) for k, v in regime_probs.items()}
        return out

    return reduced


def _productmap_over_state_action_space(
    *,
    func: Callable,
    action_names: tuple[str, ...],
    state_names: tuple[str, ...],
    state_batch_sizes: dict[str, int],
) -> Callable:
    """Wrap a scalar state-action function with productmap over actions then states.

    Matches the pattern used by `get_max_Q_over_a`: actions form the inner
    Cartesian product (unbatched), states form the outer loop (with batching).
    """
    inner = productmap(
        func=func,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )
    return productmap(
        func=inner,
        variables=state_names,
        batch_sizes=state_batch_sizes,
    )


def _build_max_Q_over_a_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    grids: MappingProxyType[str, Grid],
    enable_jit: bool,
) -> MappingProxyType[int, MaxQOverAFunction]:
    """Build max-Q-over-a closures for each period."""
    result = {}
    for period, Q_and_F in Q_and_F_functions.items():
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
        result[period] = jax.jit(func) if enable_jit else func
    return MappingProxyType(result)


def _build_argmax_and_max_Q_over_a_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    enable_jit: bool,
) -> MappingProxyType[int, ArgmaxQOverAFunction]:
    """Build argmax-and-max-Q-over-a closures for each period."""
    result = {}
    for period, Q_and_F in Q_and_F_functions.items():
        func = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F,
            action_names=state_action_space.action_names,
            state_names=state_action_space.state_names,
        )
        if enable_jit:
            func = jax.jit(func)
        result[period] = simulation_spacemap(
            func=func,
            action_names=(),
            state_names=tuple(state_action_space.states),
        )
    return MappingProxyType(result)


def _build_next_state_vmapped(
    *,
    functions: FunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[str],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[str, Grid]],
    variable_info: pd.DataFrame,
    regime_params_template: RegimeParamsTemplate,
    enable_jit: bool,
) -> NextStateSimulationFunction:
    """Build a vmapped next-state function for simulation."""
    next_state = get_next_state_function_for_simulation(
        functions=functions,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        all_grids=all_grids,
        variable_info=variable_info,
    )
    sig_args = tuple(inspect.signature(next_state).parameters)

    non_vmap = {"period", "age"} | get_flat_param_names(regime_params_template)
    vmap_variables = tuple(arg for arg in sig_args if arg not in non_vmap)

    next_state_vmapped = vmap_1d(func=next_state, variables=vmap_variables)
    next_state_vmapped = with_signature(
        next_state_vmapped, kwargs=sig_args, enforce=False
    )

    return jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped


def _fail_if_action_has_batch_size(regimes: Mapping[str, Regime]) -> None:
    """Raise if any action grid has a non-zero batch_size.

    Batching applies only to the outer state loop during solving, not to the
    inner action optimization. A non-zero batch_size on an action grid would be
    silently ignored, so we reject it early.

    """
    for regime_name, regime in regimes.items():
        for action_name, grid in regime.actions.items():
            if grid.batch_size != 0:
                msg = (
                    f"batch_size > 0 is not supported on action grids. Only state "
                    f"grids can be batched. Found batch_size={grid.batch_size} on "
                    f"action '{action_name}' in regime '{regime_name}'."
                )
                raise ValueError(msg)
