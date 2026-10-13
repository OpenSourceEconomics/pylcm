"""User-facing grid classes and the `@categorical` decorator.

Leaf classes users instantiate to declare state and action grids on a
`Regime`, plus `@categorical` for declaring the category enumeration whose
fields become a `DiscreteGrid`'s codes, `characteristics` and
`category_with` for categories that are combinations of characteristics, and the
`ContinuousGrid` base class a solver names when it accepts any continuous
grid, and `ConditionalAction` for a continuous action that is active only under
some discrete-action values. The internal `Grid` / `UniformContinuousGrid` ABCs
and the validators / coordinate helpers live in `_lcm.grids`.
"""

from _lcm.grids.categorical import categorical
from _lcm.grids.characteristics import category_with, characteristics
from _lcm.grids.conditional_action import ConditionalAction
from _lcm.grids.continuous import (
    ContinuousGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
)
from _lcm.grids.discrete import DiscreteGrid
from _lcm.grids.piecewise import (
    GridBreakpoint,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)

__all__ = [
    "ConditionalAction",
    "ContinuousGrid",
    "DiscreteGrid",
    "GridBreakpoint",
    "IrregSpacedGrid",
    "LinSpacedGrid",
    "LogSpacedGrid",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "categorical",
    "category_with",
    "characteristics",
]
