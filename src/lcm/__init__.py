import contextlib
from types import MappingProxyType

import jax

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

from lcm import shocks
from lcm.ages import AgeGrid
from lcm.grids import (
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Piece,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
from lcm.interfaces import SolveSimulateFunctionPair
from lcm.model import Model
from lcm.persistence import (
    SimulateSnapshot,
    SolveSnapshot,
    load_snapshot,
    load_solution,
    save_solution,
)
from lcm.regime import MarkovTransition, Regime
from lcm.simulation.result import SimulationResult
from lcm.utils.error_handling import validate_transition_probs

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
    "IrregSpacedGrid",
    "LinSpacedGrid",
    "LogSpacedGrid",
    "MarkovTransition",
    "Model",
    "Piece",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "Regime",
    "SimulateSnapshot",
    "SimulationResult",
    "SolveSimulateFunctionPair",
    "SolveSnapshot",
    "categorical",
    "load_snapshot",
    "load_solution",
    "save_solution",
    "shocks",
    "validate_transition_probs",
]
