"""Factories that build `Variables` and grid mappings from a user regime.

The `Variables` and `VariableInfo` dataclasses live in `_lcm.engine`. This
module is the factory side: turn a user-facing `Regime` into the canonical
`Variables` instance and accompanying grid mapping, ordering names so the
state-action space iteration is stable.

Iteration order: discrete states, continuous states, then actions in
declaration order. Within each state group explicitly sharded states sort first,
so the device axis wraps the inner per-device kernel. Declaration order is
preserved within the sharded and unsharded parts of each group.

"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from _lcm.engine import VariableInfo, Variables
from _lcm.grids import ContinuousGrid, Grid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.typing import StateName, StateOrActionName
from lcm.phased import Phased

if TYPE_CHECKING:
    from lcm.regime import Regime as UserRegime


def from_regime(
    *, user_regime: UserRegime, sharded_state_names: frozenset[StateName] = frozenset()
) -> Variables:
    """Build `Variables` from a regime, ordering names canonically.

    Order: discrete states, continuous states, then actions in declaration
    order. Within each state topology group, explicitly sharded states come
    first, preserving declaration order within each part.

    Args:
        user_regime: User-form `Regime` instance.
        sharded_state_names: States assigned a device axis by the model.

    Returns:
        A `Variables` instance whose iteration order matches the canonical
        ordering described above.

    """
    raw_info = _raw_variable_info(user_regime)
    ordered_names = _ordered_state_action_names(
        info=raw_info, sharded_state_names=sharded_state_names
    )
    return Variables(
        info=MappingProxyType({name: raw_info[name] for name in ordered_names})
    )


def get_grids(
    *,
    user_regime: UserRegime,
    sharded_state_names: frozenset[StateName] = frozenset(),
) -> MappingProxyType[StateOrActionName, Grid]:
    """Create a mapping of grid objects for each variable in the regime.

    Args:
        user_regime: User-form `Regime` instance.
        sharded_state_names: States assigned a device axis by the model.

    Returns:
        Immutable mapping of state and action variable names to their grid objects,
        in the canonical order used by `from_regime` (discrete states,
        continuous states, then actions).

    """
    variables = from_regime(
        user_regime=user_regime, sharded_state_names=sharded_state_names
    )
    raw_variables = _grid_states(user_regime) | dict(user_regime.actions)
    return MappingProxyType(
        {name: grid for name in variables if (grid := raw_variables[name]) is not None}
    )


def simulate_variables_from_regime(
    *, user_regime: UserRegime, sharded_state_names: frozenset[StateName] = frozenset()
) -> Variables:
    """Build the simulate-phase `Variables`: solve variables plus carried states.

    Each carried state is appended after the solve-ordered variables as a
    genuine state (its simulate role). The resulting order is NOT a productmap
    order — it only fixes column order in simulation output.
    """
    solve_variables = from_regime(
        user_regime=user_regime, sharded_state_names=sharded_state_names
    )
    carried_info = {
        name: VariableInfo(
            kind="state",
            topology="continuous" if isinstance(grid, ContinuousGrid) else "discrete",
            is_process=False,
        )
        for name, grid in carried_state_grids(user_regime).items()
    }
    return Variables(info=MappingProxyType({**solve_variables.info, **carried_info}))


def carried_state_grids(user_regime: UserRegime) -> dict[StateName, Grid]:
    """Return the simulate-phase grids of the regime's carried states.

    Carried states — declared via `Phased(solve=..., simulate=Grid)` — are
    absent from the solve grid (they are derived functions there); their grid
    is the simulate-phase domain used to seed, classify, and validate the
    carried-forward value.
    """
    return {
        name: cast("Grid", spec.simulate)
        for name, spec in user_regime.states.items()
        if isinstance(spec, Phased)
    }


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
    """Return the regime's states that are plain grids, excluding carried states.

    A carried state (declared via `Phased(solve=..., simulate=Grid)`) is a
    derived function in the solve phase, not a grid dimension, so it is
    omitted from the solve-phase state grids and variable info.
    """
    return {
        name: spec
        for name, spec in user_regime.states.items()
        if isinstance(spec, Grid)
    }


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
    *,
    info: dict[StateOrActionName, VariableInfo],
    sharded_state_names: frozenset[StateName],
) -> list[StateOrActionName]:
    """Order variables: discrete states, continuous states, actions.

    Each state topology group puts explicitly sharded states first and keeps
    declaration order within each part. Actions keep declaration order.

    """

    state_sort_key = _StateSortKey(sharded_state_names=sharded_state_names)

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


@dataclass(frozen=True, eq=False)
class _StateSortKey:
    """Stable sort key placing explicitly sharded states first."""

    sharded_state_names: frozenset[StateName]
    """States assigned a device axis by the model."""

    def __call__(self, name: StateOrActionName) -> bool:
        return name not in self.sharded_state_names
