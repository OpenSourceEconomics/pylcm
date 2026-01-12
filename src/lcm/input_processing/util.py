from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from dags import get_ancestors

from lcm.grids import ContinuousGrid, Grid
from lcm.utils import flatten_regime_namespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lcm.regime import Regime
    from lcm.typing import Any


def is_stochastic_transition(fn: Callable[..., Any]) -> bool:
    return hasattr(fn, "_stochastic_info")


def get_variable_info(regime: Regime) -> pd.DataFrame:
    """Derive information about all variables in the regime.

    Args:
        regime: The regime as provided by the user.

    Returns:
        A table with information about all variables in the regime. The index contains
        the name of a regime variable. The columns are booleans that are True if the
        variable has the corresponding property. The columns are: is_state, is_action,
        is_continuous, is_discrete.

    """
    variables = regime.states | regime.actions

    info = pd.DataFrame(index=list(variables))

    info["is_state"] = info.index.isin(regime.states)
    info["is_action"] = ~info["is_state"]

    info["is_continuous"] = [
        isinstance(spec, ContinuousGrid) for spec in variables.values()
    ]
    info["is_discrete"] = ~info["is_continuous"]

    info["enters_concurrent_valuation"] = _indicator_enters_concurrent_valuation(
        states_and_actions_names=list(variables),
        regime=regime,
    )

    info["enters_transition"] = _indicator_enters_transition(
        states_and_actions_names=list(variables),
        regime=regime,
    )

    order = info.query("is_discrete & is_state").index.tolist()
    order += info.query("is_discrete & is_action").index.tolist()
    order += info.query("is_continuous & is_state").index.tolist()
    order += info.query("is_continuous & is_action").index.tolist()

    if set(order) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[order]


def get_transition_info(regime: Regime) -> pd.DataFrame:
    info = pd.DataFrame(index=list(regime.transitions))
    stochastic_transitions = [
        name
        for name, trans in regime.transitions.items()
        if is_stochastic_transition(trans)
    ]
    transition_type = [
        trans._stochastic_info.type if is_stochastic_transition(trans) else "none"
        for name, trans in regime.transitions.items()
    ]
    info["is_stochastic"] = info.index.isin(stochastic_transitions)
    info["type"] = transition_type
    return info


def _indicator_enters_concurrent_valuation(
    states_and_actions_names: list[str],
    regime: Regime,
) -> pd.Series[bool]:
    """Determine which states and actions enter the concurrent valuation.

    The concurrent valuation is the evaluation of the Q_and_F function. Hence, all
    variables that (directly or indirectly) influence the "utility" (Q) or the
    constraints (F), count as relevant for the concurrent valuation.

    Special variables such as the "period" or parameters will be ignored.

    """
    enters_Q_and_F_fn_names = [
        "utility",
        *list(regime.constraints),
    ]
    user_functions = regime.get_all_functions()
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
    regime: Regime,
) -> pd.Series[bool]:
    """Determine which states and actions enter the transition.

    Transition functions correspond to the "next_" functions in the regime. This
    function returns all state and action variables that occur as inputs to these
    functions.

    Special variables such as the "period" or parameters will be ignored.

    """
    next_fn_names = list(flatten_regime_namespace(regime.transitions))
    user_functions = regime.get_all_functions()
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
    regime: Regime,
) -> dict[str, Grid]:
    """Create a dictionary of grid specifications for each variable in the regime.

    Args:
        regime (dict): The regime as provided by the user.

    Returns:
        Dictionary containing all variables of the regime. The keys are the names of the
        variables. The values describe which values the variable can take. For discrete
        variables these are the codes. For continuous variables this is information
        about how to build the grids.

    """
    variable_info = get_variable_info(regime)

    raw_variables = regime.states | regime.actions
    order = variable_info.index.tolist()
    return {k: raw_variables[k] for k in order}


def get_grids(
    regime: Regime,
) -> dict[str, Array]:
    """Create a dictionary of array grids for each variable in the regime.

    Args:
        regime: The regime as provided by the user.

    Returns:
        Dictionary containing all variables of the regime. The keys are the names of the
        variables. The values are the grids.

    """
    variable_info = get_variable_info(regime)
    gridspecs = get_gridspecs(regime)

    grids = {name: spec.to_jax() for name, spec in gridspecs.items()}
    order = variable_info.index.tolist()
    return {k: grids[k] for k in order}
