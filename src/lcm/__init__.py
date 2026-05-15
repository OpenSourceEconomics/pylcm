import contextlib
import os
from pathlib import Path
from types import MappingProxyType

# Use on-demand GPU memory allocation instead of JAX's default of pre-allocating
# 75% of GPU memory. This plays nicely with other GPU processes, makes nvidia-smi
# reflect actual usage, and enables meaningful GPU memory benchmarks. Users can
# override by setting XLA_PYTHON_CLIENT_PREALLOCATE=true before importing lcm.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Enable persistent JIT compilation cache. Large models (many regimes/states) can
# take minutes to compile; the cache makes subsequent runs near-instant. Users can
# override by setting JAX_COMPILATION_CACHE_DIR before importing lcm. Skip the
# default when no home directory is available (some HPC containers) rather than
# crashing at import time.
if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    try:
        _default_cache_dir = str(Path.home() / ".cache" / "jax")
    except RuntimeError:
        _default_cache_dir = None
    if _default_cache_dir is not None:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = _default_cache_dir
    del _default_cache_dir

import jax

# Patch jaxtyping's `"..."` sentinel to survive pickling before any
# `jaxtyping`-subscripted type is created (see the module docstring).
from lcm import _jaxtyping_patch  # noqa: F401

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

# Install beartype's AST-rewriting claw on the entire `lcm` package
# before any of its submodules is imported. The claw transforms each
# matching module's AST at first import to insert runtime type checks;
# if it isn't registered before the import happens, the affected module
# loads uninstrumented and `sys.modules` caches the unchecked version
# for the rest of the process. The claw uses `INTERNAL_CONF`, which
# surfaces violations as beartype's own `BeartypeCallHintViolation`.
# User-facing constructors (`Model`, `Regime`, `MarkovTransition`,
# every grid and shock class, `@categorical`) carry their own explicit
# `@beartype(conf=...)` decorators that map violations to the relevant
# project exception (`ModelInitializationError`,
# `RegimeInitializationError`, `GridInitializationError`, etc.); those
# decorators stack on top of the claw and win at the user boundary.
# See `lcm._beartype_conf`.
from beartype.claw import beartype_package

from lcm._beartype_conf import INTERNAL_CONF

beartype_package("lcm", conf=INTERNAL_CONF)

# Several modules annotate signatures with forward references that are
# `TYPE_CHECKING`-only at definition time (to break import cycles). The
# package claw rewrites those annotations into runtime forward references
# resolved against the module's globals at call time. Inject the resolved
# names here, after every involved module is loaded, so beartype can
# resolve them.
from lcm import persistence as _persistence  # noqa: E402
from lcm import shocks  # noqa: E402
from lcm import variables as _variables  # noqa: E402
from lcm._version import __version__  # noqa: E402
from lcm.ages import AgeGrid  # noqa: E402
from lcm.grids import (  # noqa: E402
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Piece,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
from lcm.interfaces import SolveSimulateFunctionPair  # noqa: E402
from lcm.model import Model  # noqa: E402
from lcm.persistence import (  # noqa: E402
    SimulateSnapshot,
    SolveSnapshot,
    load_snapshot,
    load_solution,
    save_solution,
)
from lcm.regime import MarkovTransition, Regime  # noqa: E402
from lcm.simulation.result import SimulationResult  # noqa: E402
from lcm.utils.containers import invert_regime_ids  # noqa: E402
from lcm.utils.error_handling import validate_transition_probs  # noqa: E402

_variables.Regime = Regime
_persistence.Model = Model
_persistence.SimulationResult = SimulationResult
del _persistence, _variables

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
    "__version__",
    "categorical",
    "invert_regime_ids",
    "load_snapshot",
    "load_solution",
    "save_solution",
    "shocks",
    "validate_transition_probs",
]
