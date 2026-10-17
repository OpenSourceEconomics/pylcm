"""Continuous actions that are active only under some values of discrete actions."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from _lcm.grids.continuous import ContinuousGrid, IrregSpacedGrid
from lcm.exceptions import GridInitializationError
from lcm.typing import (
    ActionName,
    Float1D,
    OffGrid,
    ScalarFloat,
    ScalarInt,
)

# A value a discrete action takes in user functions: a category code or a number.
type DiscreteValue = int | float

# A discrete-action value as a user writes it, e.g. the category code `Labor.yes`.
type UserDiscreteValue = DiscreteValue | ScalarInt | ScalarFloat

# One `active_when` entry: a single value, or several in a tuple, list or set.
type UserDiscreteValues = (
    UserDiscreteValue | Sequence[UserDiscreteValue] | AbstractSet[UserDiscreteValue]
)


@dataclass(frozen=True, kw_only=True, init=False)
class ConditionalAction(ContinuousGrid):
    """A continuous action that is active only under some discrete-action values.

    Where every condition in `active_when` holds, the action ranges over its grid.
    Elsewhere it takes `inactive_value`, and user functions see that value. The
    declaration means the same as a constraint pinning the action to its inactive
    node unless the conditions hold.

    Example:
        ```python
        "hours": ConditionalAction(
            grid=LinSpacedGrid(start=0.0, stop=40.0, n_points=5),
            active_when={"labor": Labor.yes},
            inactive_value=0.0,
        )
        ```

    """

    grid: ContinuousGrid
    """The grid the action ranges over where it is active."""

    active_when: MappingProxyType[ActionName, frozenset[DiscreteValue]]
    """Mapping of discrete action names to the values that activate the action.

    The action is active where every named discrete action takes one of its listed
    values. Values are what user functions see for that action: category codes for a
    categorical grid, numbers for a numeric one.
    """

    inactive_value: float
    """The node of `grid` the action takes where it is inactive."""

    inactive_index: int
    """Position of `inactive_value` among the nodes of `grid`."""

    def __init__(
        self,
        *,
        grid: ContinuousGrid,
        active_when: Mapping[ActionName, UserDiscreteValues],
        inactive_value: float,
    ) -> None:
        _fail_if_grid_is_not_plain(grid=grid)
        conditions = _normalize_conditions(active_when=active_when)
        inactive_index = _find_inactive_index(grid=grid, inactive_value=inactive_value)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "active_when", conditions)
        object.__setattr__(self, "inactive_value", float(inactive_value))
        object.__setattr__(self, "inactive_index", inactive_index)

    @property
    def off_grid(self) -> OffGrid:
        """Return the lookup policy of the wrapped grid."""
        return self.grid.off_grid

    def to_jax(self) -> Float1D:
        """Return the nodes of the wrapped grid."""
        return self.grid.to_jax()

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the coordinate of `value` on the wrapped grid."""
        return self.grid.get_coordinate(value)


def _fail_if_grid_is_not_plain(*, grid: ContinuousGrid) -> None:
    if isinstance(grid, ConditionalAction):
        raise GridInitializationError(
            "ConditionalAction must wrap a plain continuous grid such as "
            f"LinSpacedGrid; got {type(grid).__name__}."
        )
    if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
        raise GridInitializationError(
            "ConditionalAction cannot wrap an IrregSpacedGrid that receives its "
            "points at runtime: its inactive node must be known when the model is "
            "built."
        )


def _normalize_conditions(
    *,
    active_when: Mapping[ActionName, UserDiscreteValues],
) -> MappingProxyType[ActionName, frozenset[DiscreteValue]]:
    if not active_when:
        raise GridInitializationError(
            "The `active_when` of a ConditionalAction names no discrete action; an "
            "action without conditions is an ordinary grid."
        )
    conditions: dict[ActionName, frozenset[DiscreteValue]] = {}
    for name, values in active_when.items():
        listed = values if isinstance(values, Sequence | AbstractSet) else (values,)
        normalized = frozenset(_as_python_number(value) for value in listed)
        if not normalized:
            raise GridInitializationError(
                f"The condition on {name!r} lists no value, so the action would "
                "never be active."
            )
        conditions[name] = normalized
    return MappingProxyType(conditions)


def _as_python_number(value: UserDiscreteValue) -> DiscreteValue:
    number = np.asarray(value).item()
    if not isinstance(number, int | float):
        raise GridInitializationError(
            f"A condition lists {value!r}, which is not a number or category code."
        )
    return number


def _find_inactive_index(*, grid: ContinuousGrid, inactive_value: float) -> int:
    """Return the position of the node `inactive_value` names.

    A node matches when it equals `inactive_value` up to the rounding of the grid's
    dtype (two units in the last place), so `0.9` names the float32 node
    `0.9000000357` of `LinSpacedGrid(0, 1, 11)`. The nearest node is returned; model
    build refuses it if its value occurs at more than one node.
    """
    nodes = np.asarray(grid.to_jax())
    distance = np.abs(nodes.astype(np.float64) - inactive_value)
    nearest = int(np.argmin(distance))
    if distance[nearest] > 2 * np.spacing(np.abs(nodes[nearest])):
        raise GridInitializationError(
            f"The inactive value {inactive_value!r} is not a node of the grid; its "
            f"nodes are {[float(node) for node in nodes]}."
        )
    return nearest
