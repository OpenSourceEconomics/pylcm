"""Factories that build `Variables` and grid mappings from a user regime.

The `Variables` and `VariableInfo` dataclasses live in `lcm.engine`. This
module is the factory side: turn a user-facing `Regime` into the canonical
`Variables` instance and accompanying grid mapping, ordering names so the
state-action space iteration is stable.

Iteration order: discrete states (sorted by `batch_size`), continuous states
(sorted by `batch_size`), then actions in declaration order. Within each
state-topology group, `batch_size == 0` sorts last (treated as +inf).

"""

import math
from types import MappingProxyType
from typing import TYPE_CHECKING

from lcm.engine import VariableInfo, Variables
from lcm.grids import ContinuousGrid, Grid
from lcm.shocks import _ShockGrid
from lcm.typing import StateOrActionName

if TYPE_CHECKING:
    from lcm.user_regime import Regime as UserRegime


def _bind_forward_refs(*, regime_cls: type) -> None:
    """Bind `UserRegime` into this module's globals.

    The package claw rewrites string annotations on `from_regime`,
    `get_grids`, and similar helpers into runtime forward references
    resolved against this module's globals. `lcm.__init__` calls this
    helper once the user-facing `Regime` is loaded so the refs resolve
    at call time without depending on an ad-hoc assignment from outside
    the module.
    """
    global UserRegime  # noqa: PLW0603
    UserRegime = regime_cls  # ty: ignore[invalid-assignment]


def from_regime(user_regime: UserRegime) -> Variables:
    """Build `Variables` from a regime, ordering names canonically.

    Order: discrete states (by `batch_size`), continuous states (by
    `batch_size`), then actions in declaration order. Within each topology
    group, `batch_size == 0` sorts last.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        A `Variables` instance whose iteration order matches the canonical
        ordering described above.

    """
    raw_info = _raw_variable_info(user_regime)
    ordered_names = _ordered_state_action_names(user_regime, raw_info)
    return Variables(
        info=MappingProxyType({name: raw_info[name] for name in ordered_names})
    )


def get_grids(
    user_regime: UserRegime,
) -> MappingProxyType[StateOrActionName, Grid]:
    """Create a mapping of grid objects for each variable in the regime.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        Immutable mapping of state and action variable names to their grid objects,
        in the canonical order used by `from_regime` (discrete states,
        continuous states, then actions).

    """
    variables = from_regime(user_regime)
    raw_variables = dict(user_regime.states) | dict(user_regime.actions)
    return MappingProxyType({name: raw_variables[name] for name in variables})


def _raw_variable_info(
    user_regime: UserRegime,
) -> dict[StateOrActionName, VariableInfo]:
    """Derive `VariableInfo` for every state and action variable."""
    variables = dict(user_regime.states) | dict(user_regime.actions)
    info: dict[StateOrActionName, VariableInfo] = {}
    for name, spec in variables.items():
        is_state = name in user_regime.states
        is_shock = isinstance(spec, _ShockGrid)
        is_continuous = isinstance(spec, ContinuousGrid) and not is_shock
        info[name] = VariableInfo(
            kind="state" if is_state else "action",
            topology="continuous" if is_continuous else "discrete",
            is_shock=is_shock,
        )
    return info


def _ordered_state_action_names(
    user_regime: UserRegime,
    info: dict[StateOrActionName, VariableInfo],
) -> list[StateOrActionName]:
    """Order variables: discrete states, continuous states, actions.

    States are sorted by `batch_size` within each topology group; batch size 0
    sorts last (treated as +inf). Actions keep declaration order.

    """

    def state_batch_size(name: StateOrActionName) -> float:
        batch_size = user_regime.states[name].batch_size
        return batch_size if batch_size != 0 else math.inf

    discrete_states = sorted(
        (
            name
            for name, var_info in info.items()
            if var_info.kind == "state" and var_info.topology == "discrete"
        ),
        key=state_batch_size,
    )
    continuous_states = sorted(
        (
            name
            for name, var_info in info.items()
            if var_info.kind == "state" and var_info.topology == "continuous"
        ),
        key=state_batch_size,
    )
    actions = [name for name, var_info in info.items() if var_info.kind == "action"]

    ordered = [*discrete_states, *continuous_states, *actions]
    if set(ordered) != set(info):
        raise ValueError("Order and index do not match.")
    return ordered
