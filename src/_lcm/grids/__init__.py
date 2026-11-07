"""Grid infrastructure behind the outcome-space classes `lcm.grids` exposes.

`base.py` holds the abstract `Grid`, `continuous.py` the abstract continuous
grids and their validators, `discrete.py` and `piecewise.py` the machinery the
public grids call, `categorical.py` the `@categorical` decorator, and
`coordinates.py` the coordinate lookups interpolation reads. Nothing here
imports `lcm.grids`, so the public module can build on this package.
"""

from _lcm.grids.base import Grid
from _lcm.grids.categorical import categorical, validate_category_class
from _lcm.grids.continuous import ContinuousGrid, UniformContinuousGrid

__all__ = [
    "ContinuousGrid",
    "Grid",
    "UniformContinuousGrid",
    "categorical",
    "validate_category_class",
]
