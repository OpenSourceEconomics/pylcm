import math
from types import MappingProxyType

from lcm.grids import ContinuousGrid, Grid
from lcm.interfaces import VariableInfo, VariableInfoMapping
from lcm.regime import Regime
from lcm.shocks import _ShockGrid
from lcm.typing import StateOrActionName


def get_variable_info(regime: Regime) -> VariableInfoMapping:
    """Derive kind/topology/shock tags for every variable in the regime.

    Args:
        regime: The regime as provided by the user.

    Returns:
        Immutable mapping from variable name to its `VariableInfo`. Iteration
        order is: discrete states (by batch size), continuous states (by batch
        size), then actions.

    """
    variables = dict(regime.states) | dict(regime.actions)
    raw: dict[StateOrActionName, VariableInfo] = {}
    for name, spec in variables.items():
        is_state = name in regime.states
        is_shock = isinstance(spec, _ShockGrid)
        is_continuous = isinstance(spec, ContinuousGrid) and not is_shock
        raw[name] = VariableInfo(
            kind="state" if is_state else "action",
            topology="continuous" if is_continuous else "discrete",
            is_shock=is_shock,
        )

    ordered = _ordered_state_action_names(regime, raw)
    return MappingProxyType({name: raw[name] for name in ordered})


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
    return MappingProxyType({name: raw_variables[name] for name in variable_info})


def _ordered_state_action_names(
    regime: Regime,
    raw: dict[StateOrActionName, VariableInfo],
) -> list[StateOrActionName]:
    """Order variables: discrete states, continuous states, actions.

    States are sorted by `batch_size` within each topology group; batch size 0
    sorts last (treated as +inf).

    """

    def state_batch_size(name: StateOrActionName) -> float:
        batch_size = regime.states[name].batch_size
        return batch_size if batch_size != 0 else math.inf

    discrete_states = sorted(
        (
            name
            for name, info in raw.items()
            if info.kind == "state" and info.topology == "discrete"
        ),
        key=state_batch_size,
    )
    continuous_states = sorted(
        (
            name
            for name, info in raw.items()
            if info.kind == "state" and info.topology == "continuous"
        ),
        key=state_batch_size,
    )
    actions = [name for name, info in raw.items() if info.kind == "action"]

    ordered = [*discrete_states, *continuous_states, *actions]
    if set(ordered) != set(raw):
        raise ValueError("Order and index do not match.")
    return ordered
