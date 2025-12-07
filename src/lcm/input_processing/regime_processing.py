from __future__ import annotations

import functools
import warnings
from copy import deepcopy
from dataclasses import make_dataclass
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
from dags import get_annotations
from dags.signature import with_signature

from lcm import mark
from lcm.input_processing.create_params_template import create_params_template
from lcm.input_processing.regime_components import (
    build_argmax_and_max_Q_over_a_functions,
    build_max_Q_over_a_functions,
    build_next_state_simulation_functions,
    build_Q_and_F_functions,
    build_regime_transition_probs_functions,
    build_state_action_spaces,
    build_state_space_infos,
)
from lcm.input_processing.util import (
    get_grids,
    get_gridspecs,
    get_variable_info,
    is_stochastic_transition,
)
from lcm.interfaces import InternalFunctions, InternalRegime, ShockType
from lcm.utils import flatten_regime_namespace, unflatten_regime_namespace

if TYPE_CHECKING:
    from jax import Array

    from lcm.regime import Regime
    from lcm.typing import (
        Int1D,
        InternalUserFunction,
        ParamsDict,
        RegimeName,
        UserFunction,
    )


def process_regimes(
    regimes: list[Regime],
    n_periods: int,
    regime_id_cls: type,
    *,
    enable_jit: bool,
) -> dict[str, InternalRegime]:
    """Process the user regime.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the regime specification is valid.

    Args:
        regimes: The regimes as provided by the user.
        n_periods: Number of periods of the model.
        regime_id_cls: A dataclass mapping regime names to integer indices.
        enable_jit: Whether to jit the functions of the internal regime.

    Returns:
        The processed regime.

    """
    # ----------------------------------------------------------------------------------
    # Override next_regime for single-regime models to ensure probability is always 1.0
    # ----------------------------------------------------------------------------------
    if len(regimes) == 1:
        regime = regimes[0]

        if "next_regime" in regime.transitions:
            warnings.warn(
                f"Single-regime model '{regime.name}' has a user-defined 'next_regime' "
                "function, but this will be ignored. For single-regime models, the "
                "regime transition probability is always 1.0 for the same regime.",
                UserWarning,
                stacklevel=3,
            )

        @mark.stochastic
        def _default_next_regime() -> Array:
            return jnp.array([1.0])

        regimes = [
            regime.replace(
                transitions={
                    k: v for k, v in regime.transitions.items() if k != "next_regime"
                }
                | {"next_regime": _default_next_regime}
            )
        ]

    # ----------------------------------------------------------------------------------
    # Convert flat transitions to nested format
    # ----------------------------------------------------------------------------------
    # User provides flat format, internal processing uses nested format.
    # First, collect state names for each regime to know which transitions map where.
    states_per_regime: dict[str, set[str]] = {
        regime.name: set(regime.states.keys()) for regime in regimes
    }

    # Convert each regime's flat transitions to nested format
    nested_transitions = {
        regime.name: convert_flat_to_nested_transitions(
            flat_transitions=regime.transitions,
            states_per_regime=states_per_regime,
        )
        for regime in regimes
    }

    # ----------------------------------------------------------------------------------
    # Stage 1: Initialize regime components that do not depend on other regimes
    # ----------------------------------------------------------------------------------
    grids = {}
    gridspecs = {}
    variable_info = {}
    state_space_infos = {}
    state_action_spaces = {}

    for regime in regimes:
        grids[regime.name] = get_grids(regime)
        gridspecs[regime.name] = get_gridspecs(regime)
        variable_info[regime.name] = get_variable_info(regime)
        state_space_infos[regime.name] = build_state_space_infos(regime)
        state_action_spaces[regime.name] = build_state_action_spaces(regime)

    # ----------------------------------------------------------------------------------
    # Stage 2: Initialize regime components that depend on other regimes
    # ----------------------------------------------------------------------------------
    internal_regimes = {}
    for regime in regimes:
        params_template = create_params_template(
            regime,
            nested_transitions=nested_transitions[regime.name],
            grids=grids,
            n_periods=n_periods,
        )

        internal_functions = _get_internal_functions(
            regime,
            nested_transitions=nested_transitions[regime.name],
            grids=grids,
            params=params_template,
            regime_id_cls=regime_id_cls,
            enable_jit=enable_jit,
        )

        Q_and_F_functions = build_Q_and_F_functions(
            regime=regime,
            internal_functions=internal_functions,
            state_space_infos=state_space_infos,
            grids=grids,
        )
        max_Q_over_a_functions = build_max_Q_over_a_functions(
            regime=regime, Q_and_F_functions=Q_and_F_functions, enable_jit=enable_jit
        )
        argmax_and_max_Q_over_a_functions = build_argmax_and_max_Q_over_a_functions(
            regime=regime, Q_and_F_functions=Q_and_F_functions, enable_jit=enable_jit
        )
        next_state_simulation_function = build_next_state_simulation_functions(
            internal_functions=internal_functions,
            grids=grids,
            enable_jit=enable_jit,
        )

        # ------------------------------------------------------------------------------
        # Collect all components into the internal regime
        # ------------------------------------------------------------------------------
        internal_regimes[regime.name] = InternalRegime(
            name=regime.name,
            grids=grids[regime.name],
            gridspecs=gridspecs[regime.name],
            variable_info=variable_info[regime.name],
            functions=internal_functions.functions,
            utility=internal_functions.utility,
            constraints=internal_functions.constraints,
            regime_transition_probs=internal_functions.regime_transition_probs,
            internal_functions=internal_functions,
            transitions=internal_functions.transitions,
            params_template=params_template,
            state_action_spaces=state_action_spaces[regime.name],
            state_space_infos=state_space_infos[regime.name],
            max_Q_over_a_functions=max_Q_over_a_functions,
            argmax_and_max_Q_over_a_functions=argmax_and_max_Q_over_a_functions,
            next_state_simulation_function=next_state_simulation_function,
            # currently no additive utility shocks are supported
            random_utility_shocks=ShockType.NONE,
        )

    return internal_regimes


def _get_internal_functions(
    regime: Regime,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    grids: dict[RegimeName, dict[str, Array]],
    params: ParamsDict,
    regime_id_cls: type,
    *,
    enable_jit: bool,
) -> InternalFunctions:
    """Process the user provided regime functions.

    Args:
        regime: The regime as provided by the user.
        nested_transitions: Nested transitions dict for internal processing.
            Format: {"regime_name": {"next_state": fn, ...}, "next_regime": fn}
        grids: Dict containing the state grids for each regime.
        params: The parameters of the regime.
        regime_id_cls: Dataclass mapping regime names to integer indices.
        enable_jit: Whether to jit the internal functions.

    Returns:
        The processed regime functions.

    """
    flat_grids = flatten_regime_namespace(grids)

    # Flatten nested transitions to get prefixed names like "regime__next_wealth"
    flat_nested_transitions = flatten_regime_namespace(nested_transitions)

    # ==================================================================================
    # Add 'params' argument to functions
    # ==================================================================================
    # We wrap the user functions such that they can be called with the 'params' argument
    # instead of the individual parameters.

    # Build all_functions using nested_transitions (to get prefixed names)
    all_functions = deepcopy(
        flatten_regime_namespace(
            regime.functions
            | {"utility": regime.utility}
            | regime.constraints
            | nested_transitions
        )
    )

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
        functions[fn_name] = _ensure_fn_only_depends_on_params(
            fn=fn,
            fn_name=fn_name,
            params=params,
        )

    for fn_name, fn in deterministic_transition_functions.items():
        # For transition functions with prefixed names like "work__next_wealth",
        # extract the flat param key "next_wealth" to look up in params
        if fn_name == "next_regime":
            param_key = fn_name
        elif "__" in fn_name:
            param_key = fn_name.split("__", 1)[
                1
            ]  # "work__next_wealth" -> "next_wealth"
        else:
            param_key = fn_name
        functions[fn_name] = _ensure_fn_only_depends_on_params(
            fn=fn,
            fn_name=fn_name,
            param_key=param_key,
            params=params,
        )

    for fn_name, fn in stochastic_transition_functions.items():
        # The user-specified next function is the weighting function for the
        # stochastic transition. For the solution, we must also define a next function
        # that returns the whole grid of possible values.
        # For prefixed names, extract the flat param key
        param_key = fn_name.split("__", 1)[1] if "__" in fn_name else fn_name
        functions[f"weight_{fn_name}"] = _ensure_fn_only_depends_on_params(
            fn=fn,
            fn_name=fn_name,
            param_key=param_key,
            params=params,
        )
        functions[fn_name] = _get_stochastic_next_function(
            fn=fn,
            grid=flat_grids[fn_name.replace("next_", "")],
        )

    internal_transition = {
        fn_name: functions[fn_name]
        for fn_name in flat_nested_transitions
        if fn_name != "next_regime"
    }
    internal_utility = functions["utility"]
    internal_constraints = {
        fn_name: functions[fn_name] for fn_name in regime.constraints
    }
    internal_functions = {
        fn_name: functions[fn_name]
        for fn_name in functions
        if fn_name not in flat_nested_transitions
        and fn_name not in regime.constraints
        and fn_name not in {"utility", "next_regime"}
    }
    # Determine if next_regime is stochastic (decorated with @lcm.mark.stochastic)
    # next_regime is at top level in both flat and nested formats
    next_regime_fn = nested_transitions.get("next_regime")
    is_stochastic_regime_transition = (
        next_regime_fn is not None
        and is_stochastic_transition(
            next_regime_fn  # type: ignore[arg-type]
        )
    )

    internal_regime_transition_probs = build_regime_transition_probs_functions(
        internal_functions=internal_functions,
        regime_transition_probs=functions["next_regime"],
        grids=grids[regime.name],
        regime_id_cls=regime_id_cls,
        is_stochastic=is_stochastic_regime_transition,
        enable_jit=enable_jit,
    )

    return InternalFunctions(
        functions=internal_functions,
        utility=internal_utility,
        constraints=internal_constraints,
        transitions=unflatten_regime_namespace(internal_transition),
        regime_transition_probs=internal_regime_transition_probs,
    )


def _replace_func_parameters_by_params(
    fn: UserFunction,
    params: ParamsDict,
    param_key: str,
) -> InternalUserFunction:
    """Wrap a function to get its parameters from the params dict.

    Args:
        fn: The user function to wrap.
        params: The params dict template.
        param_key: The key to look up in params (e.g., "next_wealth").

    Returns:
        A wrapped function that accepts a params dict and extracts its parameters.
    """
    annotations = {
        k: v for k, v in get_annotations(fn).items() if k not in params[param_key]
    }
    annotations_with_params = annotations | {"params": "ParamsDict"}
    return_annotation = annotations_with_params.pop("return")

    @with_signature(args=annotations_with_params, return_annotation=return_annotation)
    @functools.wraps(fn)
    def processed_func(*args: Array, params: ParamsDict, **kwargs: Array) -> Array:
        return fn(*args, **kwargs, **params[param_key])

    return cast("InternalUserFunction", processed_func)


def _add_dummy_params_argument(fn: UserFunction) -> InternalUserFunction:
    annotations = get_annotations(fn) | {"params": "ParamsDict"}
    return_annotation = annotations.pop("return")

    @with_signature(args=annotations, return_annotation=return_annotation)
    @functools.wraps(fn)
    def processed_func(*args: Array, params: ParamsDict, **kwargs: Array) -> Array:  # noqa: ARG001
        return fn(*args, **kwargs)

    return cast("InternalUserFunction", processed_func)


def _get_stochastic_next_function(fn: UserFunction, grid: Int1D) -> UserFunction:
    @with_signature(args=None, return_annotation="Int1D")
    @functools.wraps(fn)
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return grid

    return next_func


def _ensure_fn_only_depends_on_params(
    fn: UserFunction,
    fn_name: str,
    params: ParamsDict,
    param_key: str | None = None,
) -> InternalUserFunction:
    # param_key is the key to look up in params (may differ from fn_name).
    key = param_key if param_key is not None else fn_name
    # params[key] contains the dictionary of parameters used by the function, which
    # is empty if the function does not depend on any regime parameters.
    if params[key]:
        return _replace_func_parameters_by_params(
            fn=fn,
            params=params,
            param_key=key,
        )
    return _add_dummy_params_argument(fn)


def create_default_regime_id_cls(regime_name: str) -> type:
    """Create a default RegimeID class for single-regime models.

    Args:
        regime_name: The name of the single regime.

    Returns:
        A dataclass with a single field mapping the regime name to index 0.

    """
    return make_dataclass("RegimeID", [(regime_name, int, 0)])


def convert_flat_to_nested_transitions(
    flat_transitions: dict[str, UserFunction],
    states_per_regime: dict[str, set[str]],
) -> dict[str, dict[str, UserFunction] | UserFunction]:
    """Convert flat transitions dictionary to nested format.

    Takes a user-provided flat transitions dictionary and converts it to the nested
    format expected by internal processing. Each transition function is mapped to
    all target regimes that have the corresponding state.

    Only regimes with COMPLETE transitions are included. A regime has complete
    transitions if there is a transition function for every state in that regime.
    This ensures we don't create entries for regimes that cannot be transitioned to.

    Args:
        flat_transitions: Flat dictionary mapping transition names to functions.
            Example: {"next_wealth": fn, "next_health": fn, "next_regime": fn}
        states_per_regime: Dictionary mapping regime names to their state names.
            Example: {"work": {"wealth", "health"}, "retirement": {"wealth", "health"}}

    Returns:
        Nested dictionary with state transitions mapped to their target regimes.
        Only includes regimes where ALL states have transition functions.
        Example: {
            "work": {"next_wealth": fn, "next_health": fn},
            "retirement": {"next_wealth": fn, "next_health": fn},
            "next_regime": fn
        }

    """
    # Separate next_regime from state transitions
    next_regime_fn = flat_transitions.get("next_regime")
    state_transitions = {
        name: fn for name, fn in flat_transitions.items() if name != "next_regime"
    }

    # Get the set of states that have transition functions
    states_with_transitions = {name.removeprefix("next_") for name in state_transitions}

    # Build nested structure, only including regimes with complete transitions
    nested: dict[str, dict[str, UserFunction] | UserFunction] = {}

    for regime_name, state_names in states_per_regime.items():
        # Check if ALL states in this regime have transition functions
        if state_names <= states_with_transitions:
            # All states covered - include this regime
            nested[regime_name] = {
                f"next_{state}": state_transitions[f"next_{state}"]
                for state in state_names
            }

    # Add next_regime at top level if it exists
    if next_regime_fn is not None:
        nested["next_regime"] = next_regime_fn

    return nested
