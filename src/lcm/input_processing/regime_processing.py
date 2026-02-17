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
from lcm.grids import DiscreteGrid, Grid
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
from lcm.regime import Regime, _make_identity_func
from lcm.shocks import _ShockGrid
from lcm.state_action_space import create_state_action_space, create_state_space_info
from lcm.typing import (
    ContinuousState,
    DiscreteState,
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

    Extracts state transitions from grid `transition` attributes and
    regime transitions from `regime.transition`. For fixed states (grids
    without a transition), an identity transition is auto-generated. ShockGrid
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
    # ----------------------------------------------------------------------------------
    # Stage 1: Initialize regime components that do not depend on other regimes
    # ----------------------------------------------------------------------------------
    variable_info = MappingProxyType(
        {n: get_variable_info(r) for n, r in regimes.items()}
    )
    gridspecs = MappingProxyType({n: get_gridspecs(r) for n, r in regimes.items()})
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
            regime=regime,
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
            transitions=internal_functions.transitions,
            regime_params_template=regime_params_template,
            state_space_info=state_space_infos[name],
            max_Q_over_a_functions=MappingProxyType(max_Q_over_a_functions),
            argmax_and_max_Q_over_a_functions=MappingProxyType(
                argmax_and_max_Q_over_a_functions
            ),
            next_state_simulation_function=next_state_simulation_function,
            # currently no additive utility shocks are supported
            random_utility_shocks=ShockType.NONE,
            _base_state_action_space=state_action_spaces[name],
        )

    return ensure_containers_are_immutable(internal_regimes)


def _get_internal_functions(
    *,
    regime: Regime,
    regime_name: str,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    grids: MappingProxyType[RegimeName, MappingProxyType[str, Array]],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    gridspecs: MappingProxyType[str, Grid],
    variable_info: pd.DataFrame,
    enable_jit: bool,
) -> InternalFunctions:
    """Process the user provided regime functions.

    Args:
        regime: The regime as provided by the user.
        regime_name: The name of the regime.
        nested_transitions: Nested transitions dict for internal processing.
            Format: {"regime_name": {"next_state": func, ...}, "next_regime": func}
        grids: Immutable mapping of regime names to grid arrays.
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
        func_name: func
        for func_name, func in all_functions.items()
        if is_stochastic_transition(func) and func_name != "next_regime"
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
        param_key = _extract_param_key(func_name)
        functions[func_name] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=param_key,
        )

    for func_name, func in stochastic_transition_functions.items():
        # The user-specified next function is the weighting function for the
        # stochastic transition. For the solution, we must also define a next function
        # that returns the whole grid of possible values.
        param_key = _extract_param_key(func_name)
        functions[f"weight_{func_name}"] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=param_key,
        )
        functions[func_name] = _get_stochastic_next_function(
            func=func,
            grid=flat_grids[func_name.replace("next_", "")],
        )
    for shock_name in variable_info.query("is_shock").index.tolist():
        relative_name = f"{regime_name}__next_{shock_name}"
        functions[f"weight_{relative_name}"] = _get_weights_func_for_shock(
            name=shock_name,
            gridspec=cast("_ShockGrid", gridspecs[shock_name]),
        )
        functions[relative_name] = _get_stochastic_next_function_for_shock(
            name=shock_name,
            grid=flat_grids[relative_name.replace("next_", "")],
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
    # Determine if next_regime is stochastic (decorated with @lcm.mark.stochastic)
    # next_regime is at top level in both flat and nested formats
    next_regime_func = nested_transitions.get("next_regime")
    is_stochastic_regime_transition = (
        next_regime_func is not None
        and is_stochastic_transition(
            next_regime_func  # ty: ignore[invalid-argument-type]
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


def _extract_transitions_from_regime(
    *,
    regime: Regime,
    states_per_regime: Mapping[str, set[str]],
) -> dict[str, dict[str, UserFunction] | UserFunction]:
    """Extract transitions from grid attributes and auto-generate identity transitions.

    For non-terminal regimes, collects state transitions from grid `transition`
    attributes and auto-generates identity transitions for fixed states (grids
    without a transition). ShockGrid transitions are handled separately during
    internal function processing.

    Args:
        regime: The user regime.
        states_per_regime: Mapping of regime names to their state names.

    Returns:
        Nested transitions dict in the format expected by _get_internal_functions.

    """
    if regime.terminal:
        return {}

    # Collect state transitions from grids
    state_transitions: dict[str, UserFunction] = {}
    for state_name, grid in regime.states.items():
        if isinstance(grid, _ShockGrid):
            # ShockGrids need an entry so they appear in internal transitions.
            # The actual transition function is generated during internal processing.
            state_transitions[f"next_{state_name}"] = stochastic(lambda: None)
        elif (grid_transition := getattr(grid, "transition", None)) is not None:
            state_transitions[f"next_{state_name}"] = grid_transition
        else:
            # Fixed state: auto-generate identity transition
            ann = DiscreteState if isinstance(grid, DiscreteGrid) else ContinuousState
            state_transitions[f"next_{state_name}"] = _make_identity_func(
                state_name, annotation=ann
            )

    # Build nested format
    transitioned_state_names = {
        name.removeprefix("next_") for name in state_transitions
    }

    nested: dict[str, dict[str, UserFunction] | UserFunction] = {}
    # Guaranteed non-None: terminal regimes return early in the caller.
    nested["next_regime"] = regime.transition  # ty: ignore[invalid-assignment]
    for target_regime_name, target_regime_state_names in states_per_regime.items():
        if target_regime_state_names <= transitioned_state_names:
            nested[target_regime_name] = {
                f"next_{state}": state_transitions[f"next_{state}"]
                for state in target_regime_state_names & transitioned_state_names
            }
    return nested


def _extract_param_key(func_name: str) -> str:
    """Extract the param template key from a possibly prefixed function name.

    For prefixed names like "work__next_wealth", returns "next_wealth".
    For unprefixed names like "next_regime", returns the name unchanged.

    """
    if QNAME_DELIMITER in func_name:
        return func_name.split(QNAME_DELIMITER, 1)[1]
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
    mapper = {p: f"{param_key}{QNAME_DELIMITER}{p}" for p in param_names}
    return cast("InternalUserFunction", rename_arguments(func, mapper=mapper))


def _get_stochastic_next_function(*, func: UserFunction, grid: Int1D) -> UserFunction:
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
    @stochastic
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
            f"{name}{QNAME_DELIMITER}{p}": p for p in gridspec.params_to_pass_at_runtime
        }
        args = {name: "ContinuousState", **dict.fromkeys(runtime_param_names, "float")}

        _compute_gridpoints = gridspec.compute_gridpoints
        _compute_transition_probs = gridspec.compute_transition_probs

        @with_signature(args=args, return_annotation="FloatND", enforce=False)
        def weights_func_runtime(*a: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
            shock_kw = {
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
            grid_points = _compute_gridpoints(n_points, **shock_kw)  # ty: ignore[invalid-argument-type]
            transition_probs = _compute_transition_probs(n_points, **shock_kw)  # ty: ignore[invalid-argument-type]
            coord = get_irreg_coordinate(value=kwargs[name], points=grid_points)
            return map_coordinates(
                input=transition_probs,
                coordinates=[
                    jnp.full(n_points, fill_value=coord),
                    jnp.arange(n_points),
                ],
            )

        return weights_func_runtime

    grid_points = gridspec.get_gridpoints()
    transition_probs = gridspec.get_transition_probs()

    @with_signature(
        args={f"{name}": "ContinuousState"},
        return_annotation="FloatND",
        enforce=False,
    )
    def weights_func(*args: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
        coordinate = get_irreg_coordinate(value=kwargs[f"{name}"], points=grid_points)
        return map_coordinates(
            input=transition_probs,
            coordinates=[
                jnp.full(gridspec.n_points, fill_value=coordinate),
                jnp.arange(gridspec.n_points),
            ],
        )

    return weights_func
