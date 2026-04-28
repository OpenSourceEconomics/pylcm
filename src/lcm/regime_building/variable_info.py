import math
from types import MappingProxyType

import pandas as pd

from lcm.grids import ContinuousGrid, Grid
from lcm.regime import Regime
from lcm.shocks import _ShockGrid
from lcm.typing import StateOrActionName


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
    info["is_shock"] = [isinstance(spec, _ShockGrid) for spec in variables.values()]
    info["is_action"] = ~info["is_state"]

    info["is_continuous"] = [
        isinstance(spec, ContinuousGrid) and not isinstance(spec, _ShockGrid)
        for spec in variables.values()
    ]
    info["is_discrete"] = ~info["is_continuous"]

    ordered_discrete_states = sorted(
        info.query("is_discrete & is_state").index.tolist(),
        key=lambda x: (
            regime.states[x].batch_size
            if regime.states[x].batch_size != 0
            else math.inf
        ),
    )
    ordered_continuous_states = sorted(
        info.query("is_continuous & is_state").index.tolist(),
        key=lambda x: (
            regime.states[x].batch_size
            if regime.states[x].batch_size != 0
            else math.inf
        ),
    )
    ordered_states_and_actions = [
        *ordered_discrete_states,
        *ordered_continuous_states,
        *info.query("is_action").index.tolist(),
    ]

    if set(ordered_states_and_actions) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[ordered_states_and_actions]


def get_grids(
    regime: Regime,
) -> MappingProxyType[StateOrActionName, Grid]:
    """Create a mapping of grid objects for each variable in the regime.

    Args:
        regime: The regime as provided by the user.

    Returns:
        Immutable mapping of state and action variable names to their grid objects.
        The values describe which values the variable can take. For discrete variables
        these are the codes. For continuous variables this is information about how to
        build the grids.

    """
    variable_info = get_variable_info(regime)

    raw_variables = dict(regime.states) | dict(regime.actions)
    order = variable_info.index.tolist()
    return MappingProxyType({k: raw_variables[k] for k in order})
