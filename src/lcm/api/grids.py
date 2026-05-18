"""User-facing grid classes.

Leaf classes users instantiate to declare state and action grids on a
`Regime`. The internal `Grid` / `ContinuousGrid` / `UniformContinuousGrid`
ABCs and the validators / coordinate helpers live in `lcm.grids` (private
implementation detail today; will move under `lcm._grids` in a follow-up).
"""

from lcm.grids.continuous import (
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
)
from lcm.grids.discrete import DiscreteGrid
from lcm.grids.piecewise import (
    Piece,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)

__all__ = [
    "DiscreteGrid",
    "IrregSpacedGrid",
    "LinSpacedGrid",
    "LogSpacedGrid",
    "Piece",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
]
