import contextlib
from types import MappingProxyType

import jax

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

from lcm import shocks
from lcm.ages import AgeGrid
from lcm.error_handling import validate_transition_probs
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
from lcm.model import Model
from lcm.pandas_utils import (
    initial_conditions_from_dataframe,
    transition_probs_from_series,
)
from lcm.persistence import (
    SimulateSnapshot,
    SolveSnapshot,
    load_snapshot,
    load_solution,
    save_solution,
)
from lcm.regime import MarkovTransition, Regime
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
    "SolveSnapshot",
    "categorical",
    "initial_conditions_from_dataframe",
    "load_snapshot",
    "load_solution",
    "save_solution",
    "shocks",
    "transition_probs_from_series",
    "validate_transition_probs",
]
