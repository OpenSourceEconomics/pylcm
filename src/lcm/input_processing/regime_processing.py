import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
from dags.signature import rename_arguments, with_signature
from dags.tree import QNAME_DELIMITER
from jax import Array
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.grid_helpers import get_irreg_coordinate
from lcm.grids import Grid, ShockGrid
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
    get_gridspecs,
    get_variable_info,
    is_stochastic_transition,
)
from lcm.interfaces import InternalFunctions, InternalRegime, ShockType
from lcm.mark import stochastic
from lcm.ndimage import map_coordinates
from lcm.regime import Regime
from lcm.state_action_space import create_state_action_space, create_state_space_info
from lcm.typing import (
    FlatRegimeParams,
    Float1D,
    Int1D,
    InternalUserFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplate,
    TransitionFunctionsMapping,
    UserFunction,
    UserParams,
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
    regimes: Mapping[str, Regime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    *,
    fixed_params: UserParams,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, InternalRegime]:
    """Process the user regime.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the regime specification is valid.

    Args:
        regimes: Mapping of regime names to Regime instances.
        ages: The AgeGrid for the model.
        regime_names_to_ids: Immutable mapping from regime names to integer indices.
        fixed_params: Parameters that can be fixed at model initialization.
        enable_jit: Whether to jit the functions of the internal regime.

    Returns:
        The processed regime.

    """

    # ----------------------------------------------------------------------------------
    # Convert flat transitions to nested format
    # ----------------------------------------------------------------------------------
    # User provides flat format, internal processing uses nested format.
    # First, collect state names for each regime to know which transitions map where.
    states_per_regime: dict[str, set[str]] = {
        name: set(regime.states.keys()) for name, regime in regimes.items()
    }

    # Convert each regime's flat transitions to nested format
    nested_transitions = {}
    for name, regime in regimes.items():
        nested_transitions[name] = _convert_flat_to_nested_transitions(
            flat_transitions=regime.transitions,
            states_per_regime=states_per_regime,
            terminal=regime.terminal,
        )
    # ----------------------------------------------------------------------------------
    # Stage 1: Initialize regime components that do not depend on other regimes
    # ----------------------------------------------------------------------------------
    variable_info = MappingProxyType(
        {n: get_variable_info(r) for n, r in regimes.items()}
    )
    flat_regime_fixed_params = {
        n: _extract_regime_fixed_params(
            regime=r, regime_name=n, fixed_params=fixed_params
        )
        for n, r in regimes.items()
    }
    gridspecs = MappingProxyType(
        {
            n: _init_shock_gridspecs(
                gridspecs=get_gridspecs(r),
                flat_regime_fixed_params=flat_regime_fixed_params[n],
            )
            for n, r in regimes.items()
        }
    )
    grids = MappingProxyType(
        {
            n: MappingProxyType(
                {name: spec.to_jax() for name, spec in gridspecs[n].items()}
            )
            for n in regimes
        }
    )

    state_space_infos = MappingProxyType(
        {n: create_state_space_info(r) for n, r in regimes.items()}
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
            regime,
            regime_name=name,
            nested_transitions=nested_transitions[name],
            grids=grids,
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            gridspecs=gridspecs[name],
            variable_info=variable_info[name],
            enable_jit=enable_jit,
        )

        Q_and_F_functions = build_Q_and_F_functions(
            regime_name=name,
            regime=regime,
            regimes_to_active_periods=regimes_to_active_periods,
            internal_functions=internal_functions,
            state_space_infos=state_space_infos,
            ages=ages,
            regime_params_template=regime_params_template,
        )
        max_Q_over_a_functions = build_max_Q_over_a_functions(
            state_action_space=state_action_spaces[name],
            Q_and_F_functions=Q_and_F_functions,
            enable_jit=enable_jit,
        )
        argmax_and_max_Q_over_a_functions = build_argmax_and_max_Q_over_a_functions(
            state_action_space=state_action_spaces[name],
            Q_and_F_functions=Q_and_F_functions,
            enable_jit=enable_jit,
        )
        next_state_simulation_function = build_next_state_simulation_functions(
            internal_functions=internal_functions,
            grids=grids,
            gridspecs=gridspecs[name],
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
            gridspecs=gridspecs[name],
            variable_info=variable_info[name],
            functions=MappingProxyType(internal_functions.functions),
            constraints=MappingProxyType(internal_functions.constraints),
            active_periods=tuple(regimes_to_active_periods[name]),
            regime_transition_probs=internal_functions.regime_transition_probs,
            internal_functions=internal_functions,
            flat_regime_fixed_params=flat_regime_fixed_params[name],
            transitions=internal_functions.transitions,
            regime_params_template=regime_params_template,
            state_action_space=state_action_spaces[name],
            state_space_info=state_space_infos[name],
            max_Q_over_a_functions=MappingProxyType(max_Q_over_a_functions),
            argmax_and_max_Q_over_a_functions=MappingProxyType(
                argmax_and_max_Q_over_a_functions
            ),
            next_state_simulation_function=next_state_simulation_function,
            # currently no additive utility shocks are supported
            random_utility_shocks=ShockType.NONE,
        )

    return ensure_containers_are_immutable(internal_regimes)


def _get_internal_functions(
    regime: Regime,
    regime_name: str,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    grids: MappingProxyType[RegimeName, MappingProxyType[str, Array]],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    gridspecs: MappingProxyType[str, Grid],
    variable_info: pd.DataFrame,
    *,
    enable_jit: bool,
) -> InternalFunctions:
    """Process the user provided regime functions.

    Args:
        regime: The regime as provided by the user.
        regime_name: The name of the regime.
        nested_transitions: Nested transitions dict for internal processing.
            Format: {"regime_name": {"next_state": fn, ...}, "next_regime": fn}
        grids: Dict containing the state grids for each regime.
        regime_params_template: The regime's parameter template.
        regime_names_to_ids: Mapping from regime names to integer indices.
        gridspecs: The specifications of the current regimes grids.
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

    # Build all_functions using nested_transitions (to get prefixed names)
    all_functions = {
        **regime.functions,
        **regime.constraints,
        **flat_nested_transitions,
    }

    stochastic_transition_functions = {
        fn_name: fn
        for fn_name, fn in all_functions.items()
        if is_stochastic_transition(fn) and fn_name != "next_regime"
    }

    deterministic_transition_functions = {
        fn_name: fn
        for fn_name, fn in all_functions.items()
        if fn_name in flat_nested_transitions
        and fn_name not in stochastic_transition_functions
    }

    deterministic_functions = {
        fn_name: fn
        for fn_name, fn in all_functions.items()
        if fn_name not in stochastic_transition_functions
        and fn_name not in deterministic_transition_functions
    }

    functions: dict[str, InternalUserFunction] = {}

    for fn_name, fn in deterministic_functions.items():
        functions[fn_name] = _rename_params_to_qnames(
            fn=fn,
            fn_key=fn_name,
            regime_params_template=regime_params_template,
        )

    for fn_name, fn in deterministic_transition_functions.items():
        fn_key = _extract_param_key(fn_name)
        functions[fn_name] = _rename_params_to_qnames(
            fn=fn,
            fn_key=fn_key,
            regime_params_template=regime_params_template,
        )

    for fn_name, fn in stochastic_transition_functions.items():
        # The user-specified next function is the weighting function for the
        # stochastic transition. For the solution, we must also define a next function
        # that returns the whole grid of possible values.
        fn_key = _extract_param_key(fn_name)
        functions[f"weight_{fn_name}"] = _rename_params_to_qnames(
            fn=fn,
            fn_key=fn_key,
            regime_params_template=regime_params_template,
        )
        functions[fn_name] = _get_stochastic_next_function(
            fn=fn,
            grid=flat_grids[fn_name.replace("next_", "")],
        )
    for shock_name in variable_info.query("is_shock").index.tolist():
        relative_name = f"{regime_name}__next_{shock_name}"
        functions[f"weight_{relative_name}"] = _get_weights_fn_for_shock(
            name=shock_name,
            flat_grid=flat_grids[relative_name.replace("next_", "")],
            gridspec=gridspecs[shock_name],
        )
        functions[relative_name] = _get_stochastic_next_function_for_shock(
            name=shock_name,
            grid=flat_grids[relative_name.replace("next_", "")],
        )
    internal_transition = {
        fn_name: functions[fn_name]
        for fn_name in flat_nested_transitions
        if fn_name != "next_regime"
    }
    internal_constraints = MappingProxyType(
        {fn_name: functions[fn_name] for fn_name in regime.constraints}
    )
    excluded_from_functions = set(flat_nested_transitions) | set(regime.constraints)
    internal_functions = MappingProxyType(
        {
            fn_name: functions[fn_name]
            for fn_name in functions
            if fn_name not in excluded_from_functions
        }
    )
    # Determine if next_regime is stochastic (decorated with @lcm.mark.stochastic)
    # next_regime is at top level in both flat and nested formats
    next_regime_fn = nested_transitions.get("next_regime")
    is_stochastic_regime_transition = (
        next_regime_fn is not None
        and is_stochastic_transition(
            next_regime_fn  # ty: ignore[invalid-argument-type]
        )
    )

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
    )


def _extract_param_key(fn_name: str) -> str:
    """Extract the param template key from a possibly prefixed function name.

    For prefixed names like "work__next_wealth", returns "next_wealth".
    For unprefixed names like "next_regime", returns the name unchanged.

    """
    if QNAME_DELIMITER in fn_name:
        return fn_name.split(QNAME_DELIMITER, 1)[1]
    return fn_name


def _rename_params_to_qnames(
    fn: UserFunction,
    fn_key: str,
    regime_params_template: RegimeParamsTemplate,
) -> InternalUserFunction:
    """Rename function params to qualified names using dags.signature.rename_arguments.

    E.g., risk_aversion -> utility__risk_aversion.

    Args:
        fn: The user function.
        fn_key: The key to look up in regime_params_template (e.g., "utility").
        regime_params_template: The parameter template for the regime.

    Returns:
        The function with renamed parameters.

    """
    param_names = list(regime_params_template[fn_key])
    if not param_names:
        return cast("InternalUserFunction", fn)
    mapper = {p: f"{fn_key}{QNAME_DELIMITER}{p}" for p in param_names}
    return cast("InternalUserFunction", rename_arguments(fn, mapper=mapper))


def _get_stochastic_next_function(fn: UserFunction, grid: Int1D) -> UserFunction:
    @with_signature(args=None, return_annotation="Int1D")
    @functools.wraps(fn)
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ANN401, ARG001
        return grid

    return next_func


def _get_stochastic_next_function_for_shock(name: str, grid: Float1D) -> UserFunction:
    """Get function that returns the indices in the vf arr of the next shock states."""

    @with_signature(args={f"{name}": "ContinuousState"}, return_annotation="Int1D")
    @stochastic
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return jnp.arange(grid.shape[0])

    return next_func


def _get_weights_fn_for_shock(
    name: str, flat_grid: Float1D, gridspec: Grid
) -> UserFunction:
    """Get function that uses linear interpolation to calculate the shock weights."""
    transition_probs = gridspec.shock.get_transition_probs()  # ty: ignore[unresolved-attribute]

    @with_signature(
        args={f"{name}": "ContinuousState"},
        return_annotation="FloatND",
        enforce=False,
    )
    def weights_func(*args: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
        coordinate = get_irreg_coordinate(kwargs[f"{name}"], flat_grid)
        return map_coordinates(
            input=transition_probs,
            coordinates=[
                jnp.full(gridspec.n_points, fill_value=coordinate),  # ty: ignore[unresolved-attribute]
                jnp.arange(gridspec.n_points),  # ty: ignore[unresolved-attribute]
            ],
        )

    return weights_func


def _extract_regime_fixed_params(
    regime: Regime, regime_name: str, fixed_params: UserParams
) -> FlatRegimeParams:
    """Extract and process fixed params relevant to a regime's shocks.

    Fixed params can be provided at two levels:
    - Model level: {"state_name": {...}} - applies to all regimes
    - Regime level: {"regime_name": {"state_name": {...}}} - applies to specific regime

    Regime-level params take precedence over model-level params.

    """
    result: dict[str, Any] = {}

    # Get regime-specific fixed_params if provided
    regime_fixed_params = fixed_params.get(regime_name, {})
    if not isinstance(regime_fixed_params, Mapping):
        regime_fixed_params = {}

    for state_name, state in regime.states.items():
        if isinstance(state, ShockGrid):
            # Check regime-level first, then model-level
            if state_name in regime_fixed_params:
                result[state_name] = regime_fixed_params[state_name]
            elif state_name in fixed_params:
                result[state_name] = fixed_params[state_name]

    return ensure_containers_are_immutable(result)


def _init_shock_gridspecs(
    gridspecs: MappingProxyType[str, Grid],
    flat_regime_fixed_params: FlatRegimeParams,
) -> MappingProxyType[str, Grid]:
    """Initialize ShockGrid instances with their fixed parameters."""
    result: dict[str, Grid] = {}
    for name, spec in gridspecs.items():
        if isinstance(spec, ShockGrid) and name in flat_regime_fixed_params:
            result[name] = spec.init_params(
                cast("MappingProxyType[str, float]", flat_regime_fixed_params[name])
            )
        else:
            result[name] = spec
    return MappingProxyType(result)


def _convert_flat_to_nested_transitions(
    flat_transitions: Mapping[str, UserFunction],
    states_per_regime: Mapping[str, set[str]],
    *,
    terminal: bool = False,
) -> dict[str, dict[str, UserFunction] | UserFunction]:
    """Convert flat transitions dictionary to nested format.

    Takes a user-provided flat transitions dictionary and converts it to the nested
    format expected by internal processing. Each transition function is mapped to
    all target regimes that have the corresponding state.

    Args:
        flat_transitions: Dictionary mapping transition names to functions.
        states_per_regime: Dictionary mapping regime names to their state names.
        terminal: Whether the regime is terminal (no transitions).

    Returns:
        Nested dictionary with state transitions mapped to their target regimes.

    """
    if terminal:
        return {}

    next_regime_fn = flat_transitions["next_regime"]
    state_transitions = {
        name: fn for name, fn in flat_transitions.items() if name != "next_regime"
    }

    transitioned_state_names = {
        name.removeprefix("next_") for name in state_transitions
    }

    nested: dict[str, dict[str, UserFunction] | UserFunction] = {}
    nested["next_regime"] = next_regime_fn
    for regime_name, regime_state_names in states_per_regime.items():
        if regime_state_names <= transitioned_state_names:
            nested[regime_name] = {
                f"next_{state}": state_transitions[f"next_{state}"]
                for state in regime_state_names & transitioned_state_names
            }
    return nested
