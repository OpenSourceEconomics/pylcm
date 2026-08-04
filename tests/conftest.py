from collections.abc import Iterator, Mapping
from dataclasses import make_dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jax_config
from numpy.typing import ArrayLike

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
            "module changes, and periodically within a long module, bounding a "
            "worker's resident memory and its mapping count."
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


def assert_agrees_to_ulp(
    got: ArrayLike, expected: ArrayLike, *, n_ulp: int, err_msg: str = ""
) -> None:
    """Assert two arrays name the same real number to within `n_ulp` of the format.

    The instrument for a knob that partitions a computation without changing it —
    a batch size, a block size. Such a knob changes the vmap width each block is
    compiled for, and XLA emits a differently vectorized kernel per width, so the
    two results can land on representable neighbours. Bounding the gap in units of
    the working format's spacing states exactly that, and states it once for both
    precisions: a partition-dependent *reduction*, the defect this guards against,
    moves a value by orders of magnitude more than a few ULP.

    Args:
        got: Result under the partitioned computation.
        expected: Result under the unpartitioned one.
        n_ulp: Largest tolerated gap, in units of the spacing at the compared
            magnitude.
        err_msg: Context appended to the failure message.

    """
    got_arr = np.asarray(got)
    expected_arr = np.asarray(expected)
    # Compare the non-finite entries as the exact values they are. ULP distance is
    # meaningless for them — `np.spacing(inf)` is NaN, so every comparison against
    # it is false and any mismatch would pass silently.
    finite = np.isfinite(expected_arr)
    np.testing.assert_array_equal(
        np.where(finite, 0.0, got_arr),
        np.where(finite, 0.0, expected_arr),
        err_msg=f"non-finite entries differ. {err_msg}",
    )
    np.testing.assert_array_equal(
        np.isfinite(got_arr), finite, err_msg=f"finiteness differs. {err_msg}"
    )
    gap = np.where(finite, np.abs(got_arr - expected_arr), 0.0)
    spacing = np.spacing(np.maximum(np.abs(got_arr), np.abs(expected_arr)))
    in_ulp = np.divide(
        gap, spacing, out=np.zeros(gap.shape, dtype=float), where=gap > 0.0
    )
    worst = float(in_ulp.max(initial=0.0))
    if worst > n_ulp:
        where = np.unravel_index(int(np.argmax(in_ulp)), in_ulp.shape)
        msg = (
            f"Values differ by up to {worst:.1f} ULP, above the {n_ulp} allowed; "
            f"worst at {where}: {got_arr[where]!r} vs {expected_arr[where]!r}. "
            f"{err_msg}"
        )
        raise AssertionError(msg)


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


# Tests run within one module before compiled programs are released anyway.
#
# A module boundary is not a tight enough bound once a single module is large.
# `tests/solution/test_envelope_query.py` is 349 tests, so releasing only at its
# edges means releasing never while it runs: one process reached 62.65 GB
# anon-RSS by 82% of that file and was OOM-killed, though every test in it uses a
# handful of grid points. The same retention exhausts `vm.max_map_count` on hosts
# that set it low, because LLVM holds an `mmap` per compiled program -- so one
# accumulation shows up as bytes on one machine and as an abort inside
# `releaseMappedMemory` on another.
#
# 64 is a compromise: small enough that the envelope file releases five times
# instead of never, large enough that an ordinary module still releases only at
# its boundary and pays nothing extra.
_TESTS_BETWEEN_RELEASES = 64


def pytest_runtest_teardown(item, nextitem):
    """Release compiled programs at module boundaries, and periodically within one.

    A worker's resident memory grows with every distinct program it has
    compiled, because JAX holds each one live in an in-memory cache. Across a
    module-spanning session that growth has no bound, which is what exhausts a
    small runner. Dropping the cache when the module changes bounds it to one
    module's programs. The persistent on-disk cache absorbs most of the cost:
    a program compiled again is a lookup rather than a fresh compile.

    Within a LARGE module that bound is still too loose -- see
    `_TESTS_BETWEEN_RELEASES` -- so the cache is released every so many tests as
    well, whether or not the module is about to change.
    """
    if not item.config.getoption("--release-compiled-programs"):
        return
    session = item.session
    seen = getattr(session, "_lcm_tests_since_release", 0) + 1
    current = getattr(item, "module", None)
    upcoming = getattr(nextitem, "module", None) if nextitem is not None else None
    if upcoming is not None and upcoming is current and seen < _TESTS_BETWEEN_RELEASES:
        session._lcm_tests_since_release = seen
        return
    session._lcm_tests_since_release = 0
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
