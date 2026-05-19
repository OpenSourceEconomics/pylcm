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

# Register beartype's package claw before any submodule import so every
# `lcm.*` module loads with runtime type checks installed via
# `INTERNAL_CONF`. User-facing constructors stack an explicit
# `@beartype(conf=...)` decorator that maps violations to the relevant
# project exception (see `lcm._beartype_conf`).
from beartype.claw import beartype_package

from lcm._beartype_conf import INTERNAL_CONF

beartype_package("lcm", conf=INTERNAL_CONF)

# Modules with TYPE_CHECKING-only forward references expose a
# `_bind_forward_refs` helper; calling it here makes the claw's
# rewritten string annotations resolve at call time.
from lcm._version import __version__  # noqa: E402
from lcm.api.ages import AgeGrid  # noqa: E402
from lcm.api.categorical import categorical  # noqa: E402
from lcm.api.grids import (  # noqa: E402
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Piece,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from lcm.api.model import Model  # noqa: E402
from lcm.api.persistence import (  # noqa: E402
    SimulateSnapshot,
    SolveSnapshot,
    load_snapshot,
    load_solution,
    save_solution,
)
from lcm.api.persistence import (  # noqa: E402
    _bind_forward_refs as _bind_persistence_forward_refs,
)
from lcm.api.processes import (  # noqa: E402
    LogNormalIIDProcess,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    RouwenhorstAR1Process,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
    UniformIIDProcess,
)
from lcm.api.regime import (  # noqa: E402
    MarkovTransition,
    Regime,
    SolveSimulateFunctionPair,
    validate_transition_probs,
)
from lcm.api.result import SimulationResult  # noqa: E402
from lcm.utils.containers import invert_regime_ids  # noqa: E402
from lcm.variables import (  # noqa: E402
    _bind_forward_refs as _bind_variables_forward_refs,
)

_bind_variables_forward_refs(regime_cls=Regime)
_bind_persistence_forward_refs(model_cls=Model, simulation_result_cls=SimulationResult)
del _bind_persistence_forward_refs, _bind_variables_forward_refs

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
    "LogNormalIIDProcess",
    "LogSpacedGrid",
    "MarkovTransition",
    "Model",
    "NormalIIDProcess",
    "NormalMixtureIIDProcess",
    "Piece",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "Regime",
    "RouwenhorstAR1Process",
    "SimulateSnapshot",
    "SimulationResult",
    "SolveSimulateFunctionPair",
    "SolveSnapshot",
    "TauchenAR1Process",
    "TauchenNormalMixtureAR1Process",
    "UniformIIDProcess",
    "__version__",
    "categorical",
    "invert_regime_ids",
    "load_snapshot",
    "load_solution",
    "save_solution",
    "validate_transition_probs",
]
