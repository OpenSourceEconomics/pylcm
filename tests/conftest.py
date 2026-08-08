from collections.abc import Iterator, Mapping
from dataclasses import make_dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jax_config

from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.processing import (
    PreparedModelStructure,
    compute_active_periods_by_regime,
    prepare_model_structure,
)
from _lcm.typing import RegimeName
from lcm.ages import AgeGrid
from lcm.typing import ScalarInt

# Module-level precision settings (updated by pytest_configure based on --precision)
X64_ENABLED: bool = True
# 12 decimals (not 14): CI showed that 14 exceeded reproducible machine precision
# across platforms. 12 is well within float64 guarantees (~15 significant digits)
# while avoiding spurious failures. See commit cdd9ac3.
DECIMAL_PRECISION: int = 12

# Multiple of the working dtype's epsilon that a repartitioned reduction is
# allowed to move a published value by. `batch_size` splits a computation whose
# result does not depend on the split, but splitting reassociates the compiled
# arithmetic, so the value array is owed agreement to the working precision
# rather than bit identity. Measured across the batch-size suites, the worst
# relative departure from the unsplayed solve is under 3 eps at either
# precision; eight is that with headroom.
INVARIANCE_EPS_MULTIPLE: float = 8.0


def invariance_tolerances(reference: np.ndarray) -> tuple[float, float]:
    """Return the `(rtol, atol)` a batch-size-repartitioned reduction is owed.

    The absolute term is the relative one scaled by the magnitude actually
    being compared, so a cell whose value sits near zero is not held to a
    tighter standard than the arithmetic can deliver.

    Args:
        reference: Array of reference values; supplies both the dtype the
            tolerance is derived from and the scale of the absolute term.

    Returns:
        Tuple of the relative and absolute tolerances.

    """
    values = np.asarray(reference)
    rtol = INVARIANCE_EPS_MULTIPLE * float(np.finfo(values.dtype).eps)
    finite = values[np.isfinite(values)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    return rtol, rtol * scale


def pytest_addoption(parser):
    """Register the --precision option for controlling JAX floating point precision."""
    parser.addoption(
        "--precision",
        action="store",
        default="64",
        choices=["32", "64"],
        help="Floating point precision for JAX (32 or 64 bit, default: 64)",
    )
    parser.addoption(
        "--release-compiled-programs",
        action="store_true",
        help=(
            "Drop JAX's in-memory compiled-program cache whenever the test "
            "module changes, bounding a worker's resident memory."
        ),
    )


def pytest_configure(config):
    """Configure JAX precision based on the --precision flag."""
    global X64_ENABLED, DECIMAL_PRECISION  # noqa: PLW0603

    X64_ENABLED = config.getoption("--precision") == "64"
    DECIMAL_PRECISION = 12 if X64_ENABLED else 5

    jax_config.update("jax_enable_x64", val=X64_ENABLED)

    # `--precision` is meant to say what the suite runs at, and on a recent
    # NVIDIA GPU the default answer is quietly less than it claims: a float32
    # matmul is computed in TF32, whose significand is under half of float32's.
    # Results the suite checks against analytical or brute-force references then
    # miss by margins no correct implementation could close, so what fails is
    # the format rather than the code. Asking for the declared precision costs
    # some throughput on that hardware and nothing anywhere else.
    jax_config.update("jax_default_matmul_precision", "highest")


def pytest_collection_modifyitems(items):
    """Mark the whole `tests/solution/` battery `slow`.

    These tests AOT-compile heavy JAX models; four in parallel exhaust a small CI
    runner's RAM (the macOS and Windows runners). They carry the `slow` marker so
    a memory-constrained runner can deselect them with `-m "not slow"` — the
    platform-independent kernel stays covered on the larger Linux/GPU runners.

    `tests/solution/` is the solve/oracle battery in its entirety, so it is marked
    by directory. Solving tests elsewhere declare the marker themselves —
    `pytestmark` where a module solves throughout, `@pytest.mark.slow` per test
    where it shares a module with construction and validation checks. Those checks
    compile nothing, and they are exactly the platform surface the small runners
    exist to cover, so they must not be swept along with their neighbours.
    """
    slow = pytest.mark.slow
    for item in items:
        if "solution" in item.path.parts:
            item.add_marker(slow)


def pytest_runtest_teardown(item, nextitem):
    """Release compiled programs at module boundaries, when asked to.

    A worker's resident memory grows with every distinct program it has
    compiled, because JAX holds each one live in an in-memory cache. Across a
    module-spanning session that growth has no bound, which is what exhausts a
    small runner. Dropping the cache when the module changes bounds it to one
    module's programs. The persistent on-disk cache absorbs most of the cost:
    a program compiled again is a lookup rather than a fresh compile.
    """
    if not item.config.getoption("--release-compiled-programs"):
        return
    current = getattr(item, "module", None)
    upcoming = getattr(nextitem, "module", None) if nextitem is not None else None
    if upcoming is not None and upcoming is current:
        return
    jax.clear_caches()


def build_prepared_structure(
    *, user_regimes: Mapping[RegimeName, FinalizedUserRegime], ages: AgeGrid
) -> PreparedModelStructure:
    """Build the `PreparedModelStructure` `process_regimes` requires.

    Tests that call `process_regimes` directly (bypassing `Model`) build this
    the same way `Model.__init__` does, rather than `process_regimes` growing
    a test-only fallback for constructing one internally.
    """
    return prepare_model_structure(
        user_regimes=user_regimes,
        ages=ages,
        active_periods_by_regime=compute_active_periods_by_regime(
            ages=ages, user_regimes=user_regimes
        ),
    )


@pytest.fixture(scope="session")
def binary_category_class():
    cls = make_dataclass(
        "BinaryCategoryClass", [("cat0", ScalarInt), ("cat1", ScalarInt)]
    )
    type.__setattr__(cls, "cat0", jnp.int32(0))
    type.__setattr__(cls, "cat1", jnp.int32(1))
    return cls


@pytest.fixture(name="x64_disabled")
def _fixture_x64_disabled() -> Iterator[None]:
    """Run the test with `jax_enable_x64=False`, restoring afterwards."""
    previous = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", val=False)
    try:
        yield
    finally:
        jax_config.update("jax_enable_x64", val=previous)


@pytest.fixture(name="x64_enabled")
def _fixture_x64_enabled() -> Iterator[None]:
    """Run the test with `jax_enable_x64=True`, restoring afterwards."""
    previous = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", val=True)
    try:
        yield
    finally:
        jax_config.update("jax_enable_x64", val=previous)
