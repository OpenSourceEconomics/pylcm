from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from dags import get_ancestors

from lcm.grids import ContinuousGrid, Grid

if TYPE_CHECKING:
    from jax import Array

    from lcm.typing import UserFunction
    from lcm.user_model import Model


def get_transition_info(model: Model) -> pd.DataFrame:
    info = pd.DataFrame(index=list(model.transitions))
    info["is_stochastic_next"] = [
        hasattr(func, "_stochastic_info") for func in model.transitions.values()
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
    transition_info = get_transition_info(model)

    variables = model.states | model.actions

    info = pd.DataFrame(index=list(variables))

    info["is_state"] = info.index.isin(model.states)
    info["is_action"] = ~info["is_state"]

    info["is_continuous"] = [
        isinstance(spec, ContinuousGrid) for spec in variables.values()
    ]
    info["is_discrete"] = ~info["is_continuous"]

    info["is_stochastic"] = [
        (
            var in model.states
            and transition_info.loc[f"next_{var}", "is_stochastic_next"]
        )
        for var in variables
    ]

    info["enters_concurrent_valuation"] = _indicator_enters_concurrent_valuation(
        states_and_actions_names=list(variables),
        model=model,
    )

    info["enters_transition"] = _indicator_enters_transition(
        states_and_actions_names=list(variables),
        model=model,
    )

    order = info.query("is_discrete & is_state").index.tolist()
    order += info.query("is_discrete & is_action").index.tolist()
    order += info.query("is_continuous & is_state").index.tolist()
    order += info.query("is_continuous & is_action").index.tolist()

    if set(order) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[order]


def _indicator_enters_concurrent_valuation(
    states_and_actions_names: list[str],
    model: Model,
) -> pd.Series[bool]:
    """Determine which states and actions enter the concurrent valuation.

    The concurrent valuation is the evaluation of the Q_and_F function. Hence, all
    variables that (directly or indirectly) influence the "utility" (Q) or the
    constraints (F), count as relevant for the concurrent valuation.

    Special variables such as the "_period" or parameters will be ignored.

    """
    enters_Q_and_F_fn_names = [
        "utility",
        *list(model.constraints),
    ]
    user_functions = get_all_user_functions(model)
    ancestors = get_ancestors(
        user_functions,
        targets=enters_Q_and_F_fn_names,
        include_targets=False,
    )
    return pd.Series(
        [var in ancestors for var in states_and_actions_names],
        index=states_and_actions_names,
    )


def _indicator_enters_transition(
    states_and_actions_names: list[str],
    model: Model,
) -> pd.Series[bool]:
    """Determine which states and actions enter the transition.

    Transition functions correspond to the "next_" functions in the model. This function
    returns all state and action variables that occur as inputs to these functions.

    Special variables such as the "_period" or parameters will be ignored.

    """
    next_fn_names = list(model.transitions)
    user_functions = get_all_user_functions(model)
    ancestors = get_ancestors(
        user_functions,
        targets=next_fn_names,
        include_targets=False,
    )
    return pd.Series(
        [var in ancestors for var in states_and_actions_names],
        index=states_and_actions_names,
    )


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


def get_all_user_functions(model: Model) -> dict[str, UserFunction]:
    return (
        {"utility": model.utility}
        | model.functions
        | model.transitions
        | model.constraints
    )
