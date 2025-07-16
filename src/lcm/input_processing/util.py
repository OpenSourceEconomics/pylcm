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

    auxiliary_variables = _get_auxiliary_variables(
        state_variables=info.query("is_state").index.tolist(),
        function_info=function_info,
        user_functions=model.functions,
    )
    info["is_auxiliary"] = [var in auxiliary_variables for var in variables]

    order = info.query("is_discrete & is_state").index.tolist()
    order += info.query("is_discrete & is_action").index.tolist()
    order += info.query("is_continuous & is_state").index.tolist()
    order += info.query("is_continuous & is_action").index.tolist()

    if set(order) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[order]


def _get_auxiliary_variables(
    state_variables: list[str],
    function_info: pd.DataFrame,
    user_functions: dict[str, UserFunction],
) -> list[str]:
    """Get state variables that only occur in next functions.

    Args:
        state_variables: List of state variable names.
        function_info: A table with information about all
            functions in the model. The index contains the name of a function. The
            columns are booleans that are True if the function has the corresponding
            property. The columns are: is_filter, is_constraint, is_next.
        user_functions: Dictionary that maps names of functions to functions.

    Returns:
        List of state variable names that are only used in next functions.

    """
    non_next_functions = function_info.query("~is_next").index.tolist()
    user_functions = {name: user_functions[name] for name in non_next_functions}
    ancestors = get_ancestors(
        user_functions,
        targets=list(user_functions),
        include_targets=True,
    )
    return list(set(state_variables).difference(set(ancestors)))


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
