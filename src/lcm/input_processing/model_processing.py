from __future__ import annotations

import functools
import inspect
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from dags import get_annotations
from dags.signature import with_signature

from lcm.functools import convert_kwargs_to_args
from lcm.input_processing.create_params_template import create_params_template
from lcm.input_processing.util import (
    get_all_user_functions,
    get_grids,
    get_gridspecs,
    get_variable_info,
    is_stochastic_transition,
)
from lcm.interfaces import InternalModel, ShockType

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from lcm.typing import (
        DiscreteAction,
        DiscreteState,
        FloatND,
        Int1D,
        InternalUserFunction,
        ParamsDict,
        UserFunction,
    )
    from lcm.user_model import Model


def process_model(model: Model) -> InternalModel:
    """Process the user model.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the model specification is valid.

    Args:
        model: The model as provided by the user.

    Returns:
        The processed model.

    """
    params = create_params_template(model)

    internal_functions = _get_internal_functions(model, params=params)

    return InternalModel(
        grids=get_grids(model),
        gridspecs=get_gridspecs(model),
        variable_info=get_variable_info(model),
        functions=internal_functions["functions"],  # type: ignore[arg-type]
        utility=internal_functions["utility"],  # type: ignore[arg-type]
        constraints=internal_functions["constraints"],  # type: ignore[arg-type]
        transitions=internal_functions["transitions"],  # type: ignore[arg-type]
        params=params,
        # currently no additive utility shocks are supported
        random_utility_shocks=ShockType.NONE,
        n_periods=model.n_periods,
    )


def _get_internal_functions(
    model: Model,
    params: ParamsDict,
) -> dict[str, InternalUserFunction | dict[str, InternalUserFunction]]:
    """Process the user provided model functions.

    Args:
        model: The model as provided by the user.
        params: The parameters of the model.

    Returns:
        Dictionary containing all functions of the model. The keys are the names of the
        functions. The values are the processed functions. The main difference between
        processed and unprocessed functions is that processed functions take `params` as
        argument.

    """
    variable_info = get_variable_info(model)
    grids = get_grids(model)

    raw_functions = deepcopy(get_all_user_functions(model))

    # ==================================================================================
    # Create functions for stochastic transitions
    # ==================================================================================
    for next_fn_name, next_fn in model.transitions.items():
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
        # is empty if the function does not depend on any model parameters.
        elif params[func_name]:
            processed_func = _replace_func_parameters_by_params(
                func=func,
                params=params,
                name=func_name,
            )

        else:
            processed_func = _add_dummy_params_argument(func)

        functions[func_name] = processed_func

    internal_transition = {fn_name: functions[fn_name] for fn_name in model.transitions}
    internal_utility = functions["utility"]
    internal_constraints = {
        fn_name: functions[fn_name] for fn_name in model.constraints
    }
    internal_functions = {
        fn_name: functions[fn_name]
        for fn_name in functions
        if fn_name not in model.transitions
        and fn_name not in model.constraints
        and fn_name != "utility"
    }

    return {
        "functions": internal_functions,
        "utility": internal_utility,
        "constraints": internal_constraints,
        "transitions": internal_transition,
    }


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
        variable_info: A table with information about model variables.

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
