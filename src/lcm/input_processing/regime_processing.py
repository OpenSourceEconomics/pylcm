import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
from dags.signature import rename_arguments, with_signature
from dags.tree import qname_from_tree_path, tree_path_from_qname
from jax import Array
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, format_messages
from lcm.grid_helpers import get_irreg_coordinate
from lcm.grids import ContinuousGrid, DiscreteGrid, Grid
from lcm.input_processing.create_regime_params_template import (
    create_regime_params_template,
)
from lcm.input_processing.regime_components import (
    build_argmax_and_max_Q_over_a_functions,
    build_max_Q_over_a_functions,
    build_next_state_simulation_functions,
    build_Q_and_F_functions,
    build_regime_transition_probs_functions,
)
from lcm.input_processing.util import (
    get_grids,
    get_variable_info,
)
from lcm.interfaces import (
    InternalFunctions,
    InternalRegime,
    PhaseVariant,
    StateSpaceInfo,
)
from lcm.ndimage import map_coordinates
from lcm.regime import MarkovTransition, Regime, _collect_state_transitions
from lcm.shocks import _ShockGrid
from lcm.state_action_space import create_state_action_space
from lcm.typing import (
    Float1D,
    Int1D,
    InternalUserFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplate,
    TransitionFunctionsMapping,
    UserFunction,
)
from lcm.utils import (
    ensure_containers_are_immutable,
    flatten_regime_namespace,
    unflatten_regime_namespace,
)


def _wrap_transitions(
    transitions: dict[RegimeName, dict[str, InternalUserFunction]],
) -> TransitionFunctionsMapping:
    """Wrap nested transitions dict in MappingProxyType."""
    return MappingProxyType(
        {name: MappingProxyType(inner) for name, inner in transitions.items()}
    )


def process_regimes(
    *,
    regimes: Mapping[str, Regime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, InternalRegime]:
    """Process user regimes into internal regimes.

    Extracts state transitions from `regime.state_transitions` and
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

    # ----------------------------------------------------------------------------------
    # Extract transitions from grid attributes and regime transition
    # ----------------------------------------------------------------------------------
    states_per_regime: dict[str, set[str]] = {
        name: set(regime.states.keys()) for name, regime in regimes.items()
    }

    nested_transitions = {}
    for name, regime in regimes.items():
        nested_transitions[name] = _extract_transitions_from_regime(
            regime=regime,
            states_per_regime=states_per_regime,
        )
    _validate_categoricals(regimes)

    # ----------------------------------------------------------------------------------
    # Stage 1: Initialize regime components that do not depend on other regimes
    # ----------------------------------------------------------------------------------
    variable_info = MappingProxyType(
        {n: get_variable_info(r) for n, r in regimes.items()}
    )
    grids = MappingProxyType({n: get_grids(r) for n, r in regimes.items()})

    state_space_infos = MappingProxyType(
        {n: _create_state_space_info(r) for n, r in regimes.items()}
    )
    state_action_spaces = MappingProxyType(
        {
            n: create_state_action_space(variable_info=variable_info[n], grids=grids[n])
            for n in regimes
        }
    )
    regimes_to_active_periods = MappingProxyType(
        {n: ages.get_periods_where(r.active) for n, r in regimes.items()}
    )

    # ----------------------------------------------------------------------------------
    # Stage 2: Initialize regime components that depend on other regimes
    # ----------------------------------------------------------------------------------
    internal_regimes = {}
    for name, regime in regimes.items():
        regime_params_template = create_regime_params_template(regime)

        internal_functions = _get_internal_functions(
            regime=regime,
            regime_name=name,
            nested_transitions=nested_transitions[name],
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            grids=grids,
            variable_info=variable_info[name],
            enable_jit=enable_jit,
        )

        # Build Q_and_F for solve (uses the default functions, i.e. solve variants).
        Q_and_F_solve = build_Q_and_F_functions(
            regime=regime,
            regimes_to_active_periods=regimes_to_active_periods,
            internal_functions=internal_functions,
            state_space_infos=state_space_infos,
            ages=ages,
            regime_params_template=regime_params_template,
        )

        # Build Q_and_F for simulate (uses simulate overrides if any).
        if _has_phase_variants(regime):
            Q_and_F_simulate = build_Q_and_F_functions(
                regime=regime,
                regimes_to_active_periods=regimes_to_active_periods,
                internal_functions=internal_functions.with_simulate_overrides(),
                state_space_infos=state_space_infos,
                ages=ages,
                regime_params_template=regime_params_template,
            )
        else:
            Q_and_F_simulate = Q_and_F_solve

        max_Q_over_a_functions = build_max_Q_over_a_functions(
            state_action_space=state_action_spaces[name],
            Q_and_F_functions=Q_and_F_solve,
            enable_jit=enable_jit,
        )
        argmax_and_max_Q_over_a_functions = build_argmax_and_max_Q_over_a_functions(
            state_action_space=state_action_spaces[name],
            Q_and_F_functions=Q_and_F_simulate,
            enable_jit=enable_jit,
        )
        next_state_simulation_function = build_next_state_simulation_functions(
            internal_functions=internal_functions,
            grids=grids,
            variable_info=variable_info[name],
            regime_params_template=regime_params_template,
            enable_jit=enable_jit,
        )

        # ------------------------------------------------------------------------------
        # Collect all components into the internal regime
        # ------------------------------------------------------------------------------
        internal_regimes[name] = InternalRegime(
            name=name,
            terminal=regime.terminal,
            grids=grids[name],
            variable_info=variable_info[name],
            functions=MappingProxyType(internal_functions.functions),
            constraints=MappingProxyType(internal_functions.constraints),
            active_periods=tuple(regimes_to_active_periods[name]),
            regime_transition_probs=internal_functions.regime_transition_probs,
            internal_functions=internal_functions,
            transitions=internal_functions.transitions,
            regime_params_template=regime_params_template,
            max_Q_over_a_functions=MappingProxyType(max_Q_over_a_functions),
            argmax_and_max_Q_over_a_functions=MappingProxyType(
                argmax_and_max_Q_over_a_functions
            ),
            next_state_simulation_function=next_state_simulation_function,
            _base_state_action_space=state_action_spaces[name],
        )

    return ensure_containers_are_immutable(internal_regimes)


def _get_internal_functions(
    *,
    regime: Regime,
    regime_name: str,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    grids: MappingProxyType[RegimeName, MappingProxyType[str, Grid]],
    variable_info: pd.DataFrame,
    enable_jit: bool,
) -> InternalFunctions:
    """Process the user provided regime functions.

    Args:
        regime: The regime as provided by the user.
        regime_name: The name of the regime.
        nested_transitions: Nested transitions dict for internal processing.
            Format: {"regime_name": {"next_state": func, ...}, "next_regime": func}
        regime_params_template: The regime's parameter template.
        regime_names_to_ids: Mapping from regime names to integer indices.
        grids: Immutable mapping of regime names to Grid spec objects.
        variable_info: Variable info of the regime.
        enable_jit: Whether to jit the internal functions.

    Returns:
        The processed regime functions.

    """
    flat_grids = flatten_regime_namespace(grids)
    # Flatten nested transitions to get prefixed names like "regime__next_wealth"
    flat_nested_transitions = flatten_regime_namespace(nested_transitions)

    # ==================================================================================
    # Rename function parameters to qualified names
    # ==================================================================================
    # We rename user function parameters to qualified names (e.g.,
    # risk_aversion -> utility__risk_aversion) so they can be matched with flat params.

    # Collect PhaseVariant entries from regime.functions for separate handling.
    phase_variant_entries: dict[str, PhaseVariant] = {
        name: func
        for name, func in regime.functions.items()
        if isinstance(func, PhaseVariant)
    }
    # For the purpose of building all_functions, use the solve variant as
    # the representative callable (get_all_functions already does this).
    resolved_functions: dict[str, UserFunction] = {}
    for name, func in regime.functions.items():
        resolved_functions[name] = cast(
            "UserFunction", func.solve if isinstance(func, PhaseVariant) else func
        )

    # Build all_functions using nested_transitions (to get prefixed names)
    all_functions: dict[str, UserFunction] = {
        **resolved_functions,
        **regime.constraints,
        **flat_nested_transitions,
    }

    # Compute per-target next names for param key extraction
    per_target_next_names = frozenset(
        f"next_{name}"
        for name, raw in regime.state_transitions.items()
        if isinstance(raw, Mapping) and not isinstance(raw, MarkovTransition)
    )

    # Compute stochastic state names from regime.state_transitions
    markov_state_names: set[str] = set()
    for name in regime.state_transitions:
        raw = regime.state_transitions[name]
        if isinstance(raw, MarkovTransition) or (
            isinstance(raw, Mapping)
            and any(isinstance(v, MarkovTransition) for v in raw.values())
        ):
            markov_state_names.add(name)
    shock_state_names = set(variable_info.query("is_shock").index.tolist())
    stochastic_transition_names = frozenset(
        f"next_{name}" for name in markov_state_names | shock_state_names
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
    simulate_overrides: dict[str, InternalUserFunction] = {}

    for func_name, func in deterministic_functions.items():
        if func_name in phase_variant_entries:
            pv = phase_variant_entries[func_name]
            functions[func_name] = _rename_params_to_qnames(
                func=pv.solve,
                regime_params_template=regime_params_template,
                param_key=func_name,
            )
            simulate_overrides[func_name] = _rename_params_to_qnames(
                func=pv.simulate,
                regime_params_template=regime_params_template,
                param_key=func_name,
            )
        else:
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
        # The user-specified next function is the weighting function for the
        # stochastic transition. For the solution, we must also define a next function
        # that returns the whole grid of possible values.
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
    for shock_name in variable_info.query("is_shock").index.tolist():
        relative_name = f"{regime_name}__next_{shock_name}"
        functions[f"weight_{relative_name}"] = _get_weights_func_for_shock(
            name=shock_name,
            gridspec=cast("_ShockGrid", grids[regime_name][shock_name]),
        )
        functions[relative_name] = _get_stochastic_next_function_for_shock(
            name=shock_name,
            grid=flat_grids[relative_name.replace("next_", "")].to_jax(),
        )
    internal_transition = {
        func_name: functions[func_name]
        for func_name in flat_nested_transitions
        if func_name != "next_regime"
    }
    internal_constraints = MappingProxyType(
        {func_name: functions[func_name] for func_name in regime.constraints}
    )
    excluded_from_functions = set(flat_nested_transitions) | set(regime.constraints)
    internal_functions = MappingProxyType(
        {
            func_name: functions[func_name]
            for func_name in functions
            if func_name not in excluded_from_functions
        }
    )
    is_stochastic_regime_transition = regime.stochastic_regime_transition

    if regime.terminal:
        internal_regime_transition_probs = None
    else:
        internal_regime_transition_probs = build_regime_transition_probs_functions(
            internal_functions=internal_functions,
            regime_transition_probs=functions["next_regime"],
            grids=grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            regime_params_template=regime_params_template,
            is_stochastic=is_stochastic_regime_transition,
            enable_jit=enable_jit,
        )
    return InternalFunctions(
        functions=internal_functions,
        constraints=internal_constraints,
        transitions=_wrap_transitions(unflatten_regime_namespace(internal_transition)),
        regime_transition_probs=internal_regime_transition_probs,
        stochastic_transition_names=stochastic_transition_names,
        simulate_overrides=MappingProxyType(simulate_overrides),
    )


def _classify_transitions(
    state_transitions: dict[str, UserFunction],
) -> tuple[dict[str, UserFunction], dict[str, dict[str, UserFunction]]]:
    """Split collected transitions into simple and per-target groups.

    Qualified names like "next_health__working" (produced by
    `_collect_state_transitions` for per-target dicts) are decomposed via
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


def _has_phase_variants(regime: Regime) -> bool:
    """Check if any function in the regime is a PhaseVariant."""
    return any(isinstance(f, PhaseVariant) for f in regime.functions.values())


def _extract_transitions_from_regime(
    *,
    regime: Regime,
    states_per_regime: Mapping[str, set[str]],
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
        Nested transitions dict in the format expected by _get_internal_functions.

    """
    if regime.terminal:
        return {}

    state_transitions = _collect_state_transitions(
        regime.states, regime.state_transitions
    )
    simple_transitions, per_target_transitions = _classify_transitions(
        state_transitions
    )

    nested = cast(
        "dict[str, dict[str, UserFunction] | UserFunction]",
        {"next_regime": regime.transition},
    )

    for target_regime_name, target_regime_state_names in states_per_regime.items():
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


def _get_weights_func_for_shock(*, name: str, gridspec: _ShockGrid) -> UserFunction:
    """Get function that uses linear interpolation to calculate the shock weights.

    For shocks whose params are supplied at runtime, the grid points and transition
    probabilities are computed inside JIT from those runtime params.

    """
    if gridspec.params_to_pass_at_runtime:
        n_points = gridspec.n_points
        fixed_params = dict(gridspec.params)
        runtime_param_names = {
            qname_from_tree_path((name, p)): p
            for p in gridspec.params_to_pass_at_runtime
        }
        args = {name: "ContinuousState", **dict.fromkeys(runtime_param_names, "float")}

        @with_signature(args=args, return_annotation="FloatND", enforce=False)
        def weights_func_runtime(*a: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
            shock_kw: dict[str, float] = {  # ty: ignore[invalid-assignment]
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
            gridpoints = gridspec.compute_gridpoints(**shock_kw)
            transition_probs = gridspec.compute_transition_probs(**shock_kw)
            coord = get_irreg_coordinate(value=kwargs[name], points=gridpoints)
            return map_coordinates(
                input=transition_probs,
                coordinates=[
                    jnp.full(n_points, fill_value=coord),
                    jnp.arange(n_points),
                ],
            )

        return weights_func_runtime

    gridpoints = gridspec.get_gridpoints()
    transition_probs = gridspec.get_transition_probs()

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
                jnp.full(gridspec.n_points, fill_value=coordinate),
                jnp.arange(gridspec.n_points),
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


def get_simulation_output_dtypes(
    regimes: Mapping[str, Regime],
    regime_names_to_ids: Mapping[str, int],
) -> MappingProxyType[str, pd.CategoricalDtype]:
    """Compute pandas CategoricalDtype for all discrete output columns.

    Merge ordered categories across regimes via topological sort. This must be
    called after model validation (which guarantees merges succeed).

    Args:
        regimes: Mapping of regime names to Regime instances.
        regime_names_to_ids: Mapping of regime names to integer IDs.

    Returns:
        Immutable mapping of variable name to `pd.CategoricalDtype`. Includes
        all discrete state/action variables plus the ``"regime"`` column.

    """
    merged_categories, ordered_flags = _compute_merged_discrete_categories(regimes)

    dtypes: dict[str, pd.CategoricalDtype] = {}
    for var_name, categories in merged_categories.items():
        dtypes[var_name] = pd.CategoricalDtype(
            categories=list(categories),
            ordered=ordered_flags[var_name],
        )

    dtypes["regime"] = pd.CategoricalDtype(
        categories=list(regime_names_to_ids.keys()),
        ordered=False,
    )

    return MappingProxyType(dtypes)


def _compute_merged_discrete_categories(
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
    - Mixed ordered flags (True in one, False in another) → error.
    - Both ordered with different categories → merge via topological sort; ambiguous
      or contradictory merges → error.
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


def _create_state_space_info(regime: Regime) -> StateSpaceInfo:
    """Create state space info for V-function interpolation.

    For terminal regimes, only states entering concurrent valuation are included.

    Args:
        regime: Regime instance.

    Returns:
        State space information for the regime.

    """
    vi = get_variable_info(regime)
    grids = get_grids(regime)

    if regime.terminal:
        vi = vi.query("enters_concurrent_valuation")

    state_names = vi.query("is_state").index.tolist()

    discrete_states = {
        name: grid_spec
        for name, grid_spec in grids.items()
        if (name in state_names and isinstance(grid_spec, DiscreteGrid))
        or isinstance(grid_spec, _ShockGrid)
    }

    continuous_states = {
        name: grid_spec
        for name, grid_spec in grids.items()
        if name in state_names
        and isinstance(grid_spec, ContinuousGrid)
        and not isinstance(grid_spec, _ShockGrid)
    }

    return StateSpaceInfo(
        state_names=tuple(state_names),
        discrete_states=MappingProxyType(discrete_states),
        continuous_states=MappingProxyType(continuous_states),
    )
