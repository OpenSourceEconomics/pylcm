from types import MappingProxyType

import pandas as pd
from dags import get_ancestors
from dags.tree import QNAME_DELIMITER
from jax import Array

from lcm.grids import ContinuousGrid, Grid
from lcm.regime import Regime
from lcm.shocks import _ShockGrid
from lcm.typing import UserFunction


def get_variable_info(
    regime: Regime,
    *,
    user_functions: dict[str, UserFunction],
) -> pd.DataFrame:
    """Derive information about all variables in the regime.

    Args:
        regime: The regime as provided by the user.
        user_functions: Flat mapping of all function names to callables for this regime.

    Returns:
        A table with information about all variables in the regime. The index contains
        the name of a regime variable. The columns are booleans that are True if the
        variable has the corresponding property. The columns are: is_state, is_action,
        is_continuous, is_discrete.

    """
    variables = dict(regime.states) | dict(regime.actions)

    info = pd.DataFrame(index=pd.Index(list(variables)))

    info["is_state"] = info.index.isin(regime.states)
    info["is_shock"] = [isinstance(spec, _ShockGrid) for spec in variables.values()]
    info["is_action"] = ~info["is_state"]

    info["is_continuous"] = [
        isinstance(spec, ContinuousGrid) and not isinstance(spec, _ShockGrid)
        for spec in variables.values()
    ]
    info["is_discrete"] = ~info["is_continuous"]

    info["enters_concurrent_valuation"] = _indicator_enters_concurrent_valuation(
        state_and_action_names=list(variables),
        regime=regime,
        user_functions=user_functions,
    )

    info["enters_transition"] = _indicator_enters_transition(
        state_and_action_names=list(variables),
        user_functions=user_functions,
    )

    order = info.query("is_discrete & is_state").index.tolist()
    order += info.query("is_discrete & is_action").index.tolist()
    order += info.query("is_continuous & is_state").index.tolist()
    order += info.query("is_continuous & is_action").index.tolist()

    if set(order) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[order]


def _indicator_enters_concurrent_valuation(
    *,
    state_and_action_names: list[str],
    regime: Regime,
    user_functions: dict[str, UserFunction],
) -> pd.Series[bool]:
    """Determine which states and actions enter the concurrent valuation.

    The concurrent valuation is the evaluation of the Q_and_F function. Hence, all
    variables that (directly or indirectly) influence the "utility" (Q) or the
    constraints (F), count as relevant for the concurrent valuation.

    Special variables such as the "period" or parameters will be ignored.

    """
    enters_Q_and_F_func_names = [
        "utility",
        *list(regime.constraints),
    ]
    # Filter out non-callable entries
    resolved = {
        name: func
        for name, func in user_functions.items()
        if func is not None and callable(func)
    }
    ancestors = get_ancestors(
        resolved,
        targets=enters_Q_and_F_func_names,
        include_targets=False,
    )
    return pd.Series(
        [var in ancestors for var in state_and_action_names],
        index=state_and_action_names,
    )


def _indicator_enters_transition(
    *,
    state_and_action_names: list[str],
    user_functions: dict[str, UserFunction],
) -> pd.Series[bool]:
    """Determine which states and actions enter the transition.

    Transition functions correspond to the "next_" functions in the regime (both
    state transitions from grid attributes and the regime transition). This function
    returns all state and action variables that occur as inputs to these functions.

    Special variables such as the "period" or parameters will be ignored.

    """
    # Filter out non-callable entries
    resolved = {
        name: func
        for name, func in user_functions.items()
        if func is not None and callable(func)
    }
    next_func_names = [
        name
        for name in resolved
        if name.split(QNAME_DELIMITER)[-1].startswith("next_")
        and not getattr(resolved[name], "_is_auto_identity", False)
    ]
    ancestors = get_ancestors(
        resolved,
        targets=next_func_names,
        include_targets=False,
    )
    return pd.Series(
        [var in ancestors for var in state_and_action_names],
        index=state_and_action_names,
    )


def get_gridspecs(
    regime: Regime,
    *,
    user_functions: dict[str, UserFunction],
) -> MappingProxyType[str, Grid]:
    """Create a dictionary of grid specifications for each variable in the regime.

    Args:
        regime: The regime as provided by the user.
        user_functions: Flat mapping of all function names to callables for this regime.

    Returns:
        Immutable dictionary containing all variables of the regime. The keys are the
        names of the variables. The values describe which values the variable can take.
        For discrete variables these are the codes. For continuous variables this is
        information about how to build the grids.

    """
    variable_info = get_variable_info(regime, user_functions=user_functions)

    raw_variables = dict(regime.states) | dict(regime.actions)
    order = variable_info.index.tolist()
    return MappingProxyType({k: raw_variables[k] for k in order})


def get_grids(
    regime: Regime,
    *,
    user_functions: dict[str, UserFunction],
) -> MappingProxyType[str, Array]:
    """Create a dictionary of array grids for each variable in the regime.

    Args:
        regime: The regime as provided by the user.
        user_functions: Flat mapping of all function names to callables for this regime.

    Returns:
        Immutable dictionary containing all variables of the regime. The keys are the
        names of the variables. The values are the grids.

    """
    variable_info = get_variable_info(regime, user_functions=user_functions)
    gridspecs = get_gridspecs(regime, user_functions=user_functions)

    grids = {name: spec.to_jax() for name, spec in gridspecs.items()}
    order = variable_info.index.tolist()
    return MappingProxyType({k: grids[k] for k in order})
