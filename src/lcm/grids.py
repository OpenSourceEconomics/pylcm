"""User-facing grid classes and the `@categorical` decorator.

Leaf classes users instantiate to declare state and action grids on a
`Regime`, plus `@categorical` for declaring the category enumeration whose
fields become a `DiscreteGrid`'s codes, `characteristic_columns` and
`find_code` for categories that are combinations of characteristics, and the
`ContinuousGrid` base class a solver names when it accepts any continuous
grid. The internal `Grid` / `UniformContinuousGrid` ABCs and the validators /
coordinate helpers live in `_lcm.grids`.
"""

from _lcm.grids.categorical import categorical
from _lcm.grids.characteristics import characteristic_columns, find_code
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
    "ContinuousGrid",
    "DiscreteGrid",
    "GridBreakpoint",
    "IrregSpacedGrid",
    "LinSpacedGrid",
    "LogSpacedGrid",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "categorical",
    "characteristic_columns",
    "find_code",
]
