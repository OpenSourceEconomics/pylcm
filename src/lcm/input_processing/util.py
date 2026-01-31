from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import pandas as pd
from dags import get_ancestors
from jax import Array

from lcm.exceptions import ModelInitializationError
from lcm.grids import ContinuousGrid, Grid
from lcm.regime import Regime
from lcm.utils import flatten_regime_namespace


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
    variables = dict(regime.states) | dict(regime.actions)

    info = pd.DataFrame(index=pd.Index(list(variables)))

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
    user_functions = dict(regime.get_all_functions())
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
    user_functions = dict(regime.get_all_functions())
    ancestors = get_ancestors(
        user_functions,
        targets=next_fn_names,
        include_targets=False,
    )
    return pd.Series(
        [var in ancestors for var in states_and_actions_names],
        index=states_and_actions_names,
    )


def check_all_variables_used(variable_info: pd.DataFrame, regime_name: str) -> None:
    """Check that all states and actions are used somewhere in the model.

    Each state or action must appear in at least one of:
    - The concurrent valuation (utility or constraints)
    - A transition function

    Args:
        variable_info: DataFrame with variable information including
            enters_concurrent_valuation and enters_transition columns.
        regime_name: Name of the regime for clearer error messages.

    Raises:
        ModelInitializationError: If any variable is not used.

    """
    is_used = (
        variable_info["enters_concurrent_valuation"]
        | variable_info["enters_transition"]
    )
    unused_variables = variable_info.index[~is_used].tolist()

    if unused_variables:
        unused_states = [
            v for v in unused_variables if variable_info.loc[v, "is_state"]
        ]
        unused_actions = [
            v for v in unused_variables if variable_info.loc[v, "is_action"]
        ]

        msg_parts = []
        if unused_states:
            state_word = "state" if len(unused_states) == 1 else "states"
            msg_parts.append(f"{state_word} {unused_states}")
        if unused_actions:
            action_word = "action" if len(unused_actions) == 1 else "actions"
            msg_parts.append(f"{action_word} {unused_actions}")

        msg = (
            f"The following variables are defined but never used in regime "
            f"'{regime_name}': {' and '.join(msg_parts)}. "
            f"Each state and action must be used in at least one of: "
            f"utility, constraints, or transition functions."
        )
        raise ModelInitializationError(msg)


def get_gridspecs(
    regime: Regime,
) -> MappingProxyType[str, Grid]:
    """Create a dictionary of grid specifications for each variable in the regime.

    Args:
        regime: The regime as provided by the user.

    Returns:
        Immutable dictionary containing all variables of the regime. The keys are the
        names of the variables. The values describe which values the variable can take.
        For discrete variables these are the codes. For continuous variables this is
        information about how to build the grids.

    """
    variable_info = get_variable_info(regime)

    raw_variables = dict(regime.states) | dict(regime.actions)
    order = variable_info.index.tolist()
    return MappingProxyType({k: raw_variables[k] for k in order})


def get_grids(
    regime: Regime,
) -> MappingProxyType[str, Array]:
    """Create a dictionary of array grids for each variable in the regime.

    Args:
        regime: The regime as provided by the user.

    Returns:
        Immutable dictionary containing all variables of the regime. The keys are the
        names of the variables. The values are the grids.

    """
    variable_info = get_variable_info(regime)
    gridspecs = get_gridspecs(regime)

    grids = {name: spec.to_jax() for name, spec in gridspecs.items()}
    order = variable_info.index.tolist()
    return MappingProxyType({k: grids[k] for k in order})
