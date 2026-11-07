"""The machinery behind `lcm.grids.ConditionalAction`.

A conditional action is a continuous action that is active only under some values of
discrete actions; these helpers normalize its conditions and locate its inactive node.
"""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from types import MappingProxyType

import numpy as np

from _lcm.grids.continuous import ContinuousGrid
from lcm.exceptions import GridInitializationError
from lcm.typing import ActionName, DiscreteGridValue, ScalarFloat, ScalarInt

# A discrete-action value as a user writes it, e.g. the category code `Labor.yes`.
type UserDiscreteValue = DiscreteGridValue | ScalarInt | ScalarFloat

# One `active_when` entry: a single value, or several in a tuple, list or set.
type UserDiscreteValues = (
    UserDiscreteValue | Sequence[UserDiscreteValue] | AbstractSet[UserDiscreteValue]
)


def normalize_conditions(
    *,
    active_when: Mapping[ActionName, UserDiscreteValues],
) -> MappingProxyType[ActionName, frozenset[DiscreteGridValue]]:
    """Return each condition's admitted values as a frozenset of Python numbers."""
    if not active_when:
        raise GridInitializationError(
            "The `active_when` of a ConditionalAction names no discrete action; an "
            "action without conditions is an ordinary grid."
        )
    conditions: dict[ActionName, frozenset[DiscreteGridValue]] = {}
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


def _as_python_number(value: UserDiscreteValue) -> DiscreteGridValue:
    number = np.asarray(value).item()
    if not isinstance(number, int | float):
        raise GridInitializationError(
            f"A condition lists {value!r}, which is not a number or category code."
        )
    return number


def find_inactive_index(*, grid: ContinuousGrid, inactive_value: float) -> int:
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
