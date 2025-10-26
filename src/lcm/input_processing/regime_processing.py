from __future__ import annotations

import functools
import inspect
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from dags import get_annotations
from dags.signature import with_signature

from lcm.functools import convert_kwargs_to_args
from lcm.input_processing.create_params_template import create_params_template
from lcm.input_processing.regime_components import (
    build_argmax_and_max_Q_over_a_functions,
    build_max_Q_over_a_functions,
    build_next_state_simulation_functions,
    build_Q_and_F_functions,
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

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from lcm.regime import Regime
    from lcm.typing import (
        DiscreteAction,
        DiscreteState,
        FloatND,
        Int1D,
        InternalUserFunction,
        ParamsDict,
        UserFunction,
    )


def process_regime(regime: Regime, *, enable_jit: bool) -> InternalRegime:
    """Process the user regime.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the regime specification is valid.

    Args:
        regime: The regime as provided by the user.
        enable_jit: Whether to jit the functions of the internal regime.

    Returns:
        The processed regime.

    """
    params_template = create_params_template(regime)

    internal_functions = _get_internal_functions(regime, params=params_template)
    grids = get_grids(regime)
    gridspecs = get_gridspecs(regime)
    variable_info = get_variable_info(regime)

    Q_and_F_functions = build_Q_and_F_functions(
        regime=regime, internal_functions=internal_functions
    )

    state_space_info = build_state_space_infos(
        regime=regime,
    )
    state_action_space = build_state_action_spaces(
        regime=regime,
    )
    max_Q_over_a_functions = build_max_Q_over_a_functions(
        regime=regime, Q_and_F_functions=Q_and_F_functions, enable_jit=enable_jit
    )
    argmax_and_max_Q_over_a_functions = build_argmax_and_max_Q_over_a_functions(
        regime=regime, Q_and_F_functions=Q_and_F_functions, enable_jit=enable_jit
    )
    next_state_simulation_functions = build_next_state_simulation_functions(
        regime=regime, internal_functions=internal_functions, grids=grids
    )

    return InternalRegime(
        grids=grids,
        gridspecs=gridspecs,
        variable_info=variable_info,
        functions=internal_functions.functions,
        utility=internal_functions.utility,
        constraints=internal_functions.constraints,
        internal_functions=internal_functions,
        transitions=internal_functions.transitions,
        params_template=params_template,
        state_action_spaces=state_action_space,
        state_space_infos=state_space_info,
        max_Q_over_a_functions=max_Q_over_a_functions,
        argmax_and_max_Q_over_a_functions=argmax_and_max_Q_over_a_functions,
        next_state_simulation_functions=next_state_simulation_functions,
        # currently no additive utility shocks are supported
        random_utility_shocks=ShockType.NONE,
        active=regime.active,
    )


def _get_internal_functions(
    regime: Regime,
    params: ParamsDict,
) -> InternalFunctions:
    """Process the user provided regime functions.

    Args:
        regime: The regime as provided by the user.
        params: The parameters of the regime.

    Returns:
        The processed regime functions.

    """
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)

    raw_functions = deepcopy(regime.get_all_functions())

    # ==================================================================================
    # Create functions for stochastic transitions
    # ==================================================================================
    for next_fn_name, next_fn in regime.transitions.items():
        if is_stochastic_transition(next_fn):
            state = next_fn_name.removeprefix("next_")

            raw_functions[next_fn_name] = _get_stochastic_next_function(
                raw_func=next_fn,
                grid=grids[state],
            )

            raw_functions[f"weight_{next_fn_name}"] = _get_stochastic_weight_function(
                raw_func=next_fn,
                name=state,
                variable_info=variable_info,
            )

    # ==================================================================================
    # Add 'params' argument to functions
    # ==================================================================================
    # We wrap the user functions such that they can be called with the 'params' argument
    # instead of the individual parameters. This is done for all functions except for
    # the dynamically generated weighting functions for stochastic next functions, since
    # they are constructed to accept the 'params' argument by default.

    functions: dict[str, InternalUserFunction] = {}

    for func_name, func in raw_functions.items():
        is_weight_next_function = func_name.startswith("weight_next_")

        if is_weight_next_function:
            processed_func = cast("InternalUserFunction", func)

        # params[name] contains the dictionary of parameters for the function, which
        # is empty if the function does not depend on any regime parameters.
        elif params[func_name]:
            processed_func = _replace_func_parameters_by_params(
                func=func,
                params=params,
                name=func_name,
            )

        else:
            processed_func = _add_dummy_params_argument(func)

        functions[func_name] = processed_func

    internal_transition = {
        fn_name: functions[fn_name] for fn_name in regime.transitions
    }
    internal_utility = functions["utility"]
    internal_constraints = {
        fn_name: functions[fn_name] for fn_name in regime.constraints
    }
    internal_functions = {
        fn_name: functions[fn_name]
        for fn_name in functions
        if fn_name not in regime.transitions
        and fn_name not in regime.constraints
        and fn_name != "utility"
    }

    return InternalFunctions(
        functions=internal_functions,
        utility=internal_utility,
        constraints=internal_constraints,
        transitions=internal_transition,
    )


def _replace_func_parameters_by_params(
    func: UserFunction, params: ParamsDict, name: str
) -> InternalUserFunction:
    annotations = {
        k: v for k, v in get_annotations(func).items() if k not in params[name]
    }
    annotations_with_params = annotations | {"params": "ParamsDict"}
    return_annotation = annotations_with_params.pop("return")

    @with_signature(args=annotations_with_params, return_annotation=return_annotation)
    @functools.wraps(func)
    def processed_func(*args: Array, params: ParamsDict, **kwargs: Array) -> Array:
        return func(*args, **kwargs, **params[name])

    return cast("InternalUserFunction", processed_func)


def _add_dummy_params_argument(func: UserFunction) -> InternalUserFunction:
    annotations = get_annotations(func) | {"params": "ParamsDict"}
    return_annotation = annotations.pop("return")

    @with_signature(args=annotations, return_annotation=return_annotation)
    @functools.wraps(func)
    def processed_func(*args: Array, params: ParamsDict, **kwargs: Array) -> Array:  # noqa: ARG001
        return func(*args, **kwargs)

    return cast("InternalUserFunction", processed_func)


def _get_stochastic_next_function(raw_func: UserFunction, grid: Int1D) -> UserFunction:
    annotations = get_annotations(raw_func)
    annotations.pop("return")

    @with_signature(args=annotations, return_annotation="Int1D")
    @functools.wraps(raw_func)
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return grid

    return next_func


def _get_stochastic_weight_function(
    raw_func: UserFunction, name: str, variable_info: pd.DataFrame
) -> InternalUserFunction:
    """Get a function that returns the transition weights of a stochastic variable.

    Example:
    Consider a stochastic variable 'health' that takes two values {0, 1}. The transition
    matrix is thus 2x2. We create the weighting function and then select the weights
    that correspond to the case where 'health' is 0.

    >>> from lcm.mark import StochasticInfo
    >>> def next_health(health):
    >>>     pass
    >>> next_health._stochastic_info = StochasticInfo()
    >>> params = {"shocks": {"health": np.arange(4).reshape(2, 2)}}
    >>> weight_func = _get_stochastic_weight_function(
    >>>     raw_func=next_health,
    >>>     name="health"
    >>>     variable_info=variable_info,
    >>>     grids=grids,
    >>> )
    >>> weight_func(health=0, params=params)
    >>> array([0, 1])


    Args:
        raw_func: The raw next function of the stochastic variable.
        name: The name of the stochastic variable.
        variable_info: A table with information about regime variables.

    Returns:
        A function that returns the transition weights of the stochastic variable.

    """
    function_parameters = list(inspect.signature(raw_func).parameters)

    # Assert that stochastic next function only depends on discrete variables or period
    invalid = {
        arg
        for arg in function_parameters
        if arg != "_period" and not variable_info.loc[arg, "is_discrete"]
    }

    if invalid:
        raise ValueError(
            "Stochastic variables can only depend on discrete variables and '_period', "
            f"but {name} depends on {invalid}.",
        )

    annotations = get_annotations(raw_func) | {"params": "ParamsDict"}
    annotations.pop("return")

    @with_signature(args=annotations, return_annotation="FloatND")
    def weight_func(
        params: ParamsDict, **kwargs: DiscreteState | DiscreteAction | int
    ) -> FloatND:
        args = convert_kwargs_to_args(kwargs, parameters=function_parameters)
        return params["shocks"][name][*args]

    return cast("InternalUserFunction", weight_func)
