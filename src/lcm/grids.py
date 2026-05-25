"""User-facing grid classes and the `@categorical` decorator.

Leaf classes users instantiate to declare state and action grids on a
`Regime`, plus `@categorical` for declaring the category enumeration whose
fields become a `DiscreteGrid`'s codes. The internal `Grid` / `ContinuousGrid`
/ `UniformContinuousGrid` ABCs and the validators / coordinate helpers live in
`_lcm.grids`.
"""

from _lcm.grids.categorical import categorical
from _lcm.grids.continuous import (
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
)
from _lcm.grids.discrete import DiscreteGrid
from _lcm.grids.piecewise import (
    PiecewiseGridSegment,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)

__all__ = [
    "DiscreteGrid",
    "IrregSpacedGrid",
    "LinSpacedGrid",
    "LogSpacedGrid",
    "PiecewiseGridSegment",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "categorical",
]
