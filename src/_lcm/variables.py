"""Factories that build `Variables` and grid mappings from a user regime.

The `Variables` and `VariableInfo` dataclasses live in `_lcm.engine`. This
module is the factory side: turn a user-facing `Regime` into the canonical
`Variables` instance and accompanying grid mapping, ordering names so the
state-action space iteration is stable.

Iteration order: discrete states, continuous states, then actions in
declaration order. Within each state group the sort key is
`(not distributed, batch_size)` — `distributed=True` states sort first
(outermost productmap axis, so the cross-device collective wraps the inner
per-device kernel); within each distributed / non-distributed slice, ties
break by `batch_size` with 0 last (treated as +∞).

"""

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from _lcm.engine import VariableInfo, Variables
from _lcm.grids import ContinuousGrid, Grid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.typing import StateName, StateOrActionName
from lcm.transition import SolveSimulateStatePair

if TYPE_CHECKING:
    from lcm.regime import Regime as UserRegime


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


def _grid_states(user_regime: UserRegime) -> dict[StateName, Grid]:
    """Return the regime's states that are plain grids, excluding state pairs.

    A `SolveSimulateStatePair` in `states` is a derived function in the solve
    phase, not a grid dimension, so it is omitted from the solve-phase state
    grids and variable info.
    """
    return {
        name: spec
        for name, spec in user_regime.states.items()
        if not isinstance(spec, SolveSimulateStatePair)
    }


def simulate_variables_from_regime(user_regime: UserRegime) -> Variables:
    """Build the simulate-phase `Variables`: solve variables plus pair states.

    Each `SolveSimulateStatePair` is appended after the solve-ordered
    variables as a genuine state (its simulate role). The resulting order is
    NOT a productmap order — it only fixes column order in simulation output.
    """
    solve_variables = from_regime(user_regime)
    pair_info = {
        name: VariableInfo(
            kind="state",
            topology="continuous" if isinstance(grid, ContinuousGrid) else "discrete",
            is_process=False,
        )
        for name, grid in state_pair_grids(user_regime).items()
    }
    return Variables(info=MappingProxyType({**solve_variables.info, **pair_info}))


def state_pair_grids(user_regime: UserRegime) -> dict[StateName, Grid]:
    """Return the simulate-phase grids of the regime's `SolveSimulateStatePair`s.

    These states are absent from the solve grid (they are derived functions
    there); their grid is the simulate-phase domain used to seed, classify, and
    validate the carried-forward value.
    """
    return {
        name: cast("Grid", spec.grid)
        for name, spec in user_regime.states.items()
        if isinstance(spec, SolveSimulateStatePair)
    }


def from_regime(user_regime: UserRegime) -> Variables:
    """Build `Variables` from a regime, ordering names canonically.

    Order: discrete states, continuous states, then actions in declaration
    order. Within each state topology group, the sort key is
    `(not distributed, batch_size)` — `distributed=True` states come first
    (sharded axes outermost in productmap), and `batch_size == 0` sorts
    last (treated as +∞).

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
    raw_variables = _grid_states(user_regime) | dict(user_regime.actions)
    return MappingProxyType({name: raw_variables[name] for name in variables})


def _raw_variable_info(
    user_regime: UserRegime,
) -> dict[StateOrActionName, VariableInfo]:
    """Derive `VariableInfo` for every state and action variable."""
    variables = _grid_states(user_regime) | dict(user_regime.actions)
    info: dict[StateOrActionName, VariableInfo] = {}
    for name, spec in variables.items():
        is_state = name in user_regime.states
        is_process = isinstance(spec, _ContinuousStochasticProcess)
        is_continuous = isinstance(spec, ContinuousGrid) and not is_process
        info[name] = VariableInfo(
            kind="state" if is_state else "action",
            topology="continuous" if is_continuous else "discrete",
            is_process=is_process,
        )
    return info


def _ordered_state_action_names(
    user_regime: UserRegime,
    info: dict[StateOrActionName, VariableInfo],
) -> list[StateOrActionName]:
    """Order variables: discrete states, continuous states, actions.

    Within each state topology group, the sort key is
    `(not distributed, batch_size)`. `distributed=True` sorts first so the
    sharded axis is the outermost productmap axis (the cross-device collective
    wraps the inner per-device kernel). Ties break by `batch_size`, with
    `batch_size == 0` last (treated as +inf). Actions keep declaration order.

    """

    grid_states = _grid_states(user_regime)

    def state_sort_key(name: StateOrActionName) -> tuple[bool, float]:
        grid = grid_states[name]
        batch_size = grid.batch_size
        return (not grid.distributed, batch_size if batch_size != 0 else math.inf)

    discrete_states = sorted(
        (
            name
            for name, var_info in info.items()
            if var_info.kind == "state" and var_info.topology == "discrete"
        ),
        key=state_sort_key,
    )
    continuous_states = sorted(
        (
            name
            for name, var_info in info.items()
            if var_info.kind == "state" and var_info.topology == "continuous"
        ),
        key=state_sort_key,
    )
    actions = [name for name, var_info in info.items() if var_info.kind == "action"]

    ordered = [*discrete_states, *continuous_states, *actions]
    if set(ordered) != set(info):
        raise ValueError("Order and index do not match.")
    return ordered
