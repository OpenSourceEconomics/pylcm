from __future__ import annotations

from types import MappingProxyType

import jax

try:
    import pdbp  # noqa: F401
except ImportError:
    pass

from lcm import mark
from lcm.ages import AgeGrid
from lcm.grids import DiscreteGrid, LinspaceGrid, LogspaceGrid, categorical
from lcm.model import Model
from lcm.regime import Regime
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
    "LinspaceGrid",
    "LogspaceGrid",
    "Model",
    "Regime",
    "SimulationResult",
    "categorical",
    "mark",
]
