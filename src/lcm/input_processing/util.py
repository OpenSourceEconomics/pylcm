from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from dags import get_ancestors

from lcm.grids import ContinuousGrid, Grid

if TYPE_CHECKING:
    from jax import Array

    from lcm.typing import UserFunction
    from lcm.user_model import Model


def get_function_info(model: Model) -> pd.DataFrame:
    """Derive information about functions in the model.

    Args:
        model: The model as provided by the user.

    Returns:
        A table with information about all functions in the model. The index contains
        the name of a model function. The columns are booleans that are True if the
        function has the corresponding property. The columns are: is_next,
        is_stochastic_next, is_constraint.

    """
    info = pd.DataFrame(index=list(model.functions))
    # Convert both filter and constraint to constraints, until we forbid filters.
    info["is_constraint"] = info.index.str.endswith(("_constraint", "_filter"))
    info["is_next"] = info.index.str.startswith("next_") & ~info["is_constraint"]
    info["is_stochastic_next"] = [
        hasattr(func, "_stochastic_info") and info.loc[func_name]["is_next"]
        for func_name, func in model.functions.items()
    ]
    return info


def get_variable_info(model: Model) -> pd.DataFrame:
    """Derive information about all variables in the model.

    Args:
        model: The model as provided by the user.

    Returns:
        A table with information about all variables in the model. The index contains
        the name of a model variable. The columns are booleans that are True if the
        variable has the corresponding property. The columns are: is_state, is_action,
        is_continuous, is_discrete.

    """
    function_info = get_function_info(model)

    variables = model.states | model.actions

    info = pd.DataFrame(index=list(variables))

    info["is_state"] = info.index.isin(model.states)
    info["is_action"] = ~info["is_state"]

    info["is_continuous"] = [
        isinstance(spec, ContinuousGrid) for spec in variables.values()
    ]
    info["is_discrete"] = ~info["is_continuous"]

    info["is_stochastic"] = [
        (var in model.states and function_info.loc[f"next_{var}", "is_stochastic_next"])
        for var in variables
    ]

    enter_concurrent_valuation = _get_variables_that_enter_concurrent_valuation(
        states_and_actions_names=list(variables),
        function_info=function_info,
        user_functions=model.functions,
    )
    info["enters_concurrent_valuation"] = [
        var in enter_concurrent_valuation for var in variables
    ]

    enter_transition = _get_variables_that_enter_transition(
        states_and_actions_names=list(variables),
        function_info=function_info,
        user_functions=model.functions,
    )
    info["enters_transition"] = [var in enter_transition for var in variables]

    order = info.query("is_discrete & is_state").index.tolist()
    order += info.query("is_discrete & is_action").index.tolist()
    order += info.query("is_continuous & is_state").index.tolist()
    order += info.query("is_continuous & is_action").index.tolist()

    if set(order) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[order]


def _get_variables_that_enter_concurrent_valuation(
    states_and_actions_names: list[str],
    function_info: pd.DataFrame,
    user_functions: dict[str, UserFunction],
) -> list[str]:
    """Get variables that enter the concurrent valuation.

    The concurrent valuation is the evaluation of the Q_and_F function. Hence, all
    variables that influence the "utility" (Q), as well as the constraints (F), count
    as relevant for the concurrent valuation.

    Special variables such as the "_period" or parameters will be ignored.

    """
    enters_Q_and_F_fn_names = [
        "utility",
        *function_info.query("is_constraint").index.tolist(),
    ]
    ancestors = get_ancestors(
        user_functions,
        targets=enters_Q_and_F_fn_names,
        include_targets=False,
    )
    return list(set(states_and_actions_names).intersection(set(ancestors)))


def _get_variables_that_enter_transition(
    states_and_actions_names: list[str],
    function_info: pd.DataFrame,
    user_functions: dict[str, UserFunction],
) -> list[str]:
    """Get state and action variables that enter the transition functions.

    Transition functions correspond to the "next_" functions in the model. This function
    returns all state and action variables that occur as inputs to these functions.

    Special variables such as the "_period" or parameters will be ignored.

    """
    next_fn_names = function_info.query("is_next").index.tolist()
    ancestors = get_ancestors(
        user_functions,
        targets=next_fn_names,
        include_targets=False,
    )
    return list(set(states_and_actions_names).intersection(set(ancestors)))


def get_gridspecs(
    model: Model,
) -> dict[str, Grid]:
    """Create a dictionary of grid specifications for each variable in the model.

    Args:
        model (dict): The model as provided by the user.

    Returns:
        Dictionary containing all variables of the model. The keys are the names of the
        variables. The values describe which values the variable can take. For discrete
        variables these are the codes. For continuous variables this is information
        about how to build the grids.

    """
    variable_info = get_variable_info(model)

    raw_variables = model.states | model.actions
    order = variable_info.index.tolist()
    return {k: raw_variables[k] for k in order}


def get_grids(
    model: Model,
) -> dict[str, Array]:
    """Create a dictionary of array grids for each variable in the model.

    Args:
        model: The model as provided by the user.

    Returns:
        Dictionary containing all variables of the model. The keys are the names of the
        variables. The values are the grids.

    """
    variable_info = get_variable_info(model)
    gridspecs = get_gridspecs(model)

    grids = {name: spec.to_jax() for name, spec in gridspecs.items()}
    order = variable_info.index.tolist()
    return {k: grids[k] for k in order}
