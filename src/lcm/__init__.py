import contextlib
from types import MappingProxyType

import jax

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

from lcm import shocks
from lcm.ages import AgeGrid
from lcm.grids import (
    DiscreteGrid,
    DiscreteMarkovGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Piece,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
from lcm.model import Model
from lcm.regime import MarkovRegimeTransition, Regime, RegimeTransition
from lcm.simulation.result import SimulationResult

# Register MappingProxyType as a JAX pytree so it can be used in JIT-traced functions.
# This allows regime transition probabilities to use immutable mappings.
jax.tree_util.register_pytree_node(
    MappingProxyType,
    lambda mp: (tuple(mp.values()), tuple(mp.keys())),
    lambda keys, values: MappingProxyType(dict(zip(keys, values, strict=True))),
)

__all__ = [
    "AgeGrid",
    "DiscreteGrid",
    "DiscreteMarkovGrid",
    "IrregSpacedGrid",
    "LinSpacedGrid",
    "LogSpacedGrid",
    "MarkovRegimeTransition",
    "Model",
    "Piece",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "Regime",
    "RegimeTransition",
    "SimulationResult",
    "categorical",
    "shocks",
]
