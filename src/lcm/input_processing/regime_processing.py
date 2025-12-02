from __future__ import annotations

import functools
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from dags import get_annotations
from dags.signature import with_signature

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
    regimes: list[Regime], n_periods: int, *, enable_jit: bool
) -> dict[str, InternalRegime]:
    """Process the user regime.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the regime specification is valid.

    Args:
        regimes: The regimes as provided by the user.
        n_periods: Number of periods of the model.
        enable_jit: Whether to jit the functions of the internal regime.

    Returns:
        The processed regime.

    """
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
            regime, grids=grids, n_periods=n_periods
        )

        internal_functions = _get_internal_functions(
            regime, grids=grids, params=params_template, enable_jit=enable_jit
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
    grids: dict[RegimeName, dict[str, Array]],
    params: ParamsDict,
    *,
    enable_jit: bool,
) -> InternalFunctions:
    """Process the user provided regime functions.

    Args:
        regime: The regime as provided by the user.
        grids: Dict containing the state grids for each regime.
        params: The parameters of the regime.
        enable_jit: Whether to jit the internal functions.

    Returns:
        The processed regime functions.

    """
    flat_grids = flatten_regime_namespace(grids)

    # ==================================================================================
    # Add 'params' argument to functions
    # ==================================================================================
    # We wrap the user functions such that they can be called with the 'params' argument
    # instead of the individual parameters.

    all_functions = deepcopy(regime.get_all_functions())

    stochastic_transition_functions = {
        fn_name: fn
        for fn_name, fn in all_functions.items()
        if is_stochastic_transition(fn)
    }

    deterministic_transition_functions = {
        fn_name: fn
        for fn_name, fn in all_functions.items()
        if fn_name in flatten_regime_namespace(regime.transitions)
        and fn_name not in stochastic_transition_functions
    }

    deterministic_functions = {
        fn_name: fn
        for fn_name, fn in all_functions.items()
        if fn_name
        not in (stochastic_transition_functions, deterministic_transition_functions)
    }

    functions: dict[str, InternalUserFunction] = {}

    for fn_name, fn in deterministic_functions.items():
        functions[fn_name] = _ensure_fn_only_depends_on_params(
            fn=fn,
            fn_name=fn_name,
            params=params,
        )

    for fn_name, fn in deterministic_transition_functions.items():
        regime_name = None if fn_name == "next_regime" else fn_name.split("__", 1)[0]
        functions[fn_name] = _ensure_fn_only_depends_on_params(
            fn=fn,
            fn_name=fn_name,
            params=params,
            regime_name=regime_name,
        )

    for fn_name, fn in stochastic_transition_functions.items():
        # The user-specified next function is the weighting function for the
        # stochastic transition. For the solution, we must also define a next function
        # that returns the whole grid of possible values.
        functions[f"weight_{fn_name}"] = _ensure_fn_only_depends_on_params(
            fn=fn,
            fn_name=fn_name,
            params=params,
        )
        functions[fn_name] = _get_stochastic_next_function(
            fn=fn,
            grid=flat_grids[fn_name.replace("next_", "")],
        )

    internal_transition = {
        fn_name: functions[fn_name]
        for fn_name in flatten_regime_namespace(regime.transitions)
        if fn_name != "next_regime"
    }
    internal_utility = functions["utility"]
    internal_constraints = {
        fn_name: functions[fn_name] for fn_name in regime.constraints
    }
    internal_functions = {
        fn_name: functions[fn_name]
        for fn_name in functions
        if fn_name not in flatten_regime_namespace(regime.transitions)
        and fn_name not in regime.constraints
        and fn_name not in {"utility", "next_regime"}
    }
    internal_regime_transition_probs = build_regime_transition_probs_functions(
        internal_functions=internal_functions,
        regime_transition_probs=functions["next_regime"],
        grids=grids[regime.name],
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
    fn: UserFunction, params: ParamsDict, name: str, regime_name: str | None
) -> InternalUserFunction:
    annotations = {
        k: v for k, v in get_annotations(fn).items() if k not in params[name]
    }
    annotations_with_params = annotations | {"params": "ParamsDict"}
    return_annotation = annotations_with_params.pop("return")

    @with_signature(args=annotations_with_params, return_annotation=return_annotation)
    @functools.wraps(fn)
    def processed_func(*args: Array, params: ParamsDict, **kwargs: Array) -> Array:
        return fn(*args, **kwargs, **params[name])

    @with_signature(args=annotations_with_params, return_annotation=return_annotation)
    @functools.wraps(fn)
    def processed_func_regime(
        *args: Array, params: ParamsDict, **kwargs: Array
    ) -> Array:
        return fn(*args, **kwargs, **params[name])

    if regime_name is None:
        return cast("InternalUserFunction", processed_func)
    return cast("InternalUserFunction", processed_func_regime)


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
    fn: UserFunction, fn_name: str, params: ParamsDict, regime_name: str | None = None
) -> InternalUserFunction:
    # params[fn_name] contains the dictionary of parameters used by the function, which
    # is empty if the function does not depend on any regime parameters.
    if params[fn_name]:
        return _replace_func_parameters_by_params(
            fn=fn,
            params=params,
            name=fn_name,
            regime_name=regime_name,
        )
    return _add_dummy_params_argument(fn)
