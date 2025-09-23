from __future__ import annotations

import dataclasses
import functools
import inspect
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from dags import get_annotations
from dags.signature import with_signature

from lcm.functools import convert_kwargs_to_args
from lcm.input_processing.create_params_template import create_params_template
from lcm.input_processing.util import (
    get_function_info,
    get_grids,
    get_gridspecs,
    get_variable_info,
)
from lcm.interfaces import ShockType

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
    from collections.abc import Sequence
    from lcm.user_model import Model
    from lcm.regime import Regime
    from lcm.grids import Grid
    from lcm.typing import Float1D
    

@dataclasses.dataclass(frozen=True)
class InternalRegime:

    name: str
    description: str | None
    active: list[int]
    grids: dict[str, Float1D | Int1D]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    functions: dict[str, InternalUserFunction]
    function_info: pd.DataFrame
    params: ParamsDict
    # Not properly processed yet
    random_utility_shocks: ShockType


def process_regimes(regimes: Regime | Sequence[Regime]) -> list[InternalRegime]:
    """Process the user regimes.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the regime specification is valid.

    Args:
        regimes: The regimes as provided by the user. Can be a single Regime or a
            sequence of Regimes.

    Returns:
        The list of processed regimes.

    """
    if isinstance(regimes, Regime):
        regimes = [regimes]

    return [_process_regime(regime) for regime in regimes]


def _process_regime(regime: Regime) -> InternalRegime:
    params = create_params_template(regime)

    return InternalRegime(
        name=regime.name,
        description=regime.description,
        active=list(regime.active),
        grids=get_grids(regime),
        gridspecs=get_gridspecs(regime),
        variable_info=get_variable_info(regime),
        functions=_get_internal_functions(regime, params=params),
        function_info=get_function_info(regime),
        params=params,
        # currently no additive utility shocks are supported
        random_utility_shocks=ShockType.NONE,
    )


def _get_internal_functions(
    model: Model,
    params: ParamsDict,
) -> dict[str, InternalUserFunction]:
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

    raw_functions = deepcopy(model.functions)

    for var in model.states:
        if variable_info.loc[var, "is_stochastic"]:
            raw_functions[f"next_{var}"] = _get_stochastic_next_function(
                raw_func=raw_functions[f"next_{var}"],
                grid=grids[var],
            )

            raw_functions[f"weight_next_{var}"] = _get_stochastic_weight_function(
                raw_func=raw_functions[f"next_{var}"],
                name=var,
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

    return functions


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
