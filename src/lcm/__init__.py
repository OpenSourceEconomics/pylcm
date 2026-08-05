import contextlib
import os
from pathlib import Path
from types import MappingProxyType

from lcm._compilation_cache import compilation_cache_name

# Use on-demand GPU memory allocation instead of JAX's default of pre-allocating
# 75% of GPU memory. This plays nicely with other GPU processes, makes nvidia-smi
# reflect actual usage, and enables meaningful GPU memory benchmarks. Users can
# override by setting XLA_PYTHON_CLIENT_PREALLOCATE=true before importing lcm.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# Enable persistent JIT compilation cache. Large models (many regimes/states) can
# take minutes to compile; the cache makes subsequent runs near-instant. Skip the
# default when no home directory is available (some HPC containers) rather than
# crashing at import time.
#
# The cache is split per project rather than pooled in one directory. Entries are
# keyed by a hash of the computation, so splitting costs no reuse — what it
# separates is concurrent *writers*, and concurrent writers are what leave
# unreadable entries behind, reported later as a decompression warning after which
# the cache's work is silently lost.
#
# Two overrides, outermost first:
# - `JAX_COMPILATION_CACHE_DIR` names the directory outright and is used as given.
# - `LCM_COMPILATION_CACHE_NAME` names only the leaf under the shared root, for
#   repositories holding several independent models that run concurrently (one
#   package per paper, say), where the enclosing project is too coarse a split.
if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    try:
        _default_cache_dir = str(
            Path.home() / ".cache" / "jax" / compilation_cache_name()
        )
    except RuntimeError:
        _default_cache_dir = None
    if _default_cache_dir is not None:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = _default_cache_dir
    del _default_cache_dir

import jax

# The environment variable above only reaches JAX if `jax` has not been imported
# yet: JAX reads it once, while defining its config options. Anything that
# imports `jax` before `lcm` — a test suite's `conftest`, a notebook, another
# library — therefore leaves the cache silently disabled and pays a full compile
# in every process. Applying the same value through `jax.config` afterwards makes
# the setting hold whichever order the two were imported in.
if not jax.config.jax_compilation_cache_dir:
    _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if _cache_dir:
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
    del _cache_dir

# JAX only writes an executable to the persistent cache when its compile time
# exceeds `jax_persistent_cache_min_compile_time_secs` (default: 1 second). A
# pylcm model compiles as many small per-regime/per-period programs, most of
# which fall under that threshold — with the default, the cache stays empty and
# every fresh process recompiles the whole model. Cache everything instead.
# Applied via `jax.config` so it takes effect regardless of whether `jax` was
# imported before `lcm`; users can override by setting the environment variable
# JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS before importing jax.
if os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS") is None:
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

# Register beartype's package claw on both the private implementation package
# `_lcm` and the public `lcm` package before any of their submodules are
# imported, so every module loads with runtime type checks installed via
# `INTERNAL_CONF`. User-facing constructors stack an explicit
# `@beartype(conf=...)` decorator that maps violations to the relevant project
# exception (see `_lcm.beartype_conf`).
# beartype 0.22.9 ships a stray `print(f'Detecting C-based callable {func!r}
# isomorphism...')` at `beartype._util.func.utilfunctest:1083`. It fires once
# per imported pseudo-callable (jaxlib PjitFunction at every pylcm import),
# polluting every pytask / test invocation. Silence it before the claw runs.
import beartype._util.func.utilfunctest as _beartype_utilfunctest
from beartype.claw import beartype_package

from _lcm.beartype_conf import INTERNAL_CONF

_beartype_utilfunctest.print = lambda *_args, **_kwargs: None  # ty: ignore[unresolved-attribute]

beartype_package("_lcm", conf=INTERNAL_CONF)
beartype_package("lcm", conf=INTERNAL_CONF)

from _lcm.variables import (  # noqa: E402
    _bind_forward_refs as _bind_variables_forward_refs,
)
from _lcm.version import __version__  # noqa: E402
from lcm.ages import AgeGrid  # noqa: E402
from lcm.case_piece import (  # noqa: E402
    affine_breakpoint,
    boundary,
    case_boundary,
    piece,
    piecewise_affine,
    smooth_helper,
)
from lcm.certainty_equivalent import (  # noqa: E402
    CertaintyEquivalent,
    LinearExpectation,
    PowerMean,
    QuasiArithmeticMean,
)
from lcm.grids import (  # noqa: E402
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseGridSegment,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
from lcm.koopmans_aggregation import CESAggregator, LinearAggregator  # noqa: E402
from lcm.model import Model  # noqa: E402
from lcm.persistence import (  # noqa: E402
    SimulateSnapshot,
    SolveSnapshot,
    load_snapshot,
    load_solution,
    save_solution,
)
from lcm.persistence import (  # noqa: E402
    _bind_forward_refs as _bind_persistence_forward_refs,
)
from lcm.phased import Phased  # noqa: E402
from lcm.processes import (  # noqa: E402
    LogNormalIIDProcess,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    RouwenhorstAR1Process,
    StateConditioned,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
    UniformIIDProcess,
)
from lcm.regime import (  # noqa: E402
    MarkovTransition,
    Regime,
    SamePeriodRef,
)
from lcm.result import SimulationResult  # noqa: E402
from lcm.solvers import (  # noqa: E402
    DCEGM,
    EGM,
    NBEGM,
    NEGM,
    NNBEGM,
    AdaptiveOuterMesh,
    BranchAggregateResult,
    DeterministicOuterMaximum,
    FiniteOuterGrid,
    GridSearch,
    LegacyGoldenSection,
    OuterBranchAggregator,
    OuterSearch,
    TwoAssetEGM,
    UniformObservedFixedCost,
)
from lcm.taste_shocks import ExtremeValueTasteShocks  # noqa: E402
from lcm.transition import (  # noqa: E402
    AgeSpecializedFunction,
    AgeSpecializedGrid,
    fixed_transition,
)

# Modules with TYPE_CHECKING-only forward references expose a
# `_bind_forward_refs` helper; calling it here makes the claw's
# rewritten string annotations resolve at call time.
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
    "DCEGM",
    "EGM",
    "NBEGM",
    "NEGM",
    "NNBEGM",
    "AdaptiveOuterMesh",
    "AgeGrid",
    "AgeSpecializedFunction",
    "AgeSpecializedGrid",
    "BranchAggregateResult",
    "CESAggregator",
    "CertaintyEquivalent",
    "DeterministicOuterMaximum",
    "DiscreteGrid",
    "ExtremeValueTasteShocks",
    "FiniteOuterGrid",
    "GridSearch",
    "IrregSpacedGrid",
    "LegacyGoldenSection",
    "LinSpacedGrid",
    "LinearAggregator",
    "LinearExpectation",
    "LogNormalIIDProcess",
    "LogSpacedGrid",
    "MarkovTransition",
    "Model",
    "NormalIIDProcess",
    "NormalMixtureIIDProcess",
    "OuterBranchAggregator",
    "OuterSearch",
    "Phased",
    "PiecewiseGridSegment",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "PowerMean",
    "QuasiArithmeticMean",
    "Regime",
    "RouwenhorstAR1Process",
    "SamePeriodRef",
    "SimulateSnapshot",
    "SimulationResult",
    "SolveSnapshot",
    "StateConditioned",
    "TauchenAR1Process",
    "TauchenNormalMixtureAR1Process",
    "TwoAssetEGM",
    "UniformIIDProcess",
    "UniformObservedFixedCost",
    "__version__",
    "affine_breakpoint",
    "boundary",
    "case_boundary",
    "categorical",
    "fixed_transition",
    "load_snapshot",
    "load_solution",
    "piece",
    "piecewise_affine",
    "save_solution",
    "smooth_helper",
]
