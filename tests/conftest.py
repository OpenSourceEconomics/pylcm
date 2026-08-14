import ctypes
import ctypes.util
import functools
import pathlib
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
            "Accepted for compatibility and has no effect: releasing is the "
            "default. Pass --keep-compiled-programs to turn it off."
        ),
    )
    parser.addoption(
        "--keep-compiled-programs",
        action="store_true",
        help=(
            "Keep JAX's in-memory compiled-program cache for the whole session "
            "instead of dropping it at module boundaries. Resident memory then "
            "grows with every distinct program compiled, so pass this only when "
            "measuring compile behaviour itself, where dropping the cache would "
            "change what is being measured."
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

    _apply_backend_skips(items=items)


# How far a worker may grow past its last release before the next one, in MiB.
#
# What a test costs is the programs it builds, which is not a property of it
# being one test: a module of a few dozen tests can add over a gibibyte in a
# single one of them, while a module of hundreds may never grow that far. So
# growth is measured rather than counted. That needs no per-module tuning and
# cannot be invalidated by a new module the way a count is -- and it is the only
# trigger needed inside a module, because a compiled program costs both resident
# bytes and an LLVM `mmap`, so bounding the bytes bounds the mappings that would
# otherwise abort in `releaseMappedMemory` on a host with a low
# `vm.max_map_count`.
#
# A worker's peak is then roughly its post-release size plus this allowance plus
# the largest single test, so 1 GiB leaves the two CI workers well inside a 16 GB
# runner while an ordinary module, which never grows this far between module
# boundaries, keeps releasing exactly where it did before and pays nothing.
_MIB_BETWEEN_RELEASES = 1024.0

_STATUS_PATH = pathlib.Path("/proc/self/status")


def resident_mebibytes() -> float | None:
    """This process's resident set size in MiB, or `None` where it is unavailable.

    Read straight from the kernel rather than from a high-water mark, so a
    release that returns memory is visible as a fall.
    """
    try:
        status = _STATUS_PATH.read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None


@functools.cache
def _malloc_trim():
    """The C library's arena-trimming entry point, or `None` if it has none."""
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    try:
        return ctypes.CDLL(name).malloc_trim
    except OSError, AttributeError:
        return None


def return_free_heap_to_os() -> bool:
    """Hand every arena's free heap back to the operating system.

    Report whether the allocator offers the operation at all; a platform without
    it simply keeps whatever it keeps.
    """
    trim = _malloc_trim()
    if trim is None:
        return False
    trim(0)
    return True


def should_release_compiled_programs(*, config) -> bool:
    """Return whether a teardown drops JAX's compiled-program cache.

    Releasing is the default, because the unbounded worker is the failure that
    actually happens and the bounded one costs almost nothing: a program needed
    again is a lookup in the persistent on-disk cache rather than a fresh
    compile. Leaving it off by default put the bound behind a flag that only CI
    passed, so every local run, every ad-hoc battery and every bundle grew
    without one.

    `--keep-compiled-programs` opts out for the one case that needs it —
    measuring compile behaviour, where dropping the cache changes the quantity
    being measured.
    """
    return not config.getoption("--keep-compiled-programs")


def pytest_runtest_teardown(item, nextitem):
    """Release compiled programs at module boundaries, and periodically within one.

    A worker's resident memory grows with every distinct program it has
    compiled, because JAX holds each one live in an in-memory cache. Across a
    module-spanning session that growth has no bound, which is what exhausts a
    small runner. Dropping the cache when the module changes bounds it to one
    module's programs. The persistent on-disk cache absorbs most of the cost:
    a program compiled again is a lookup rather than a fresh compile.

    Within a LARGE module that bound is still too loose, so a release also
    happens once a worker has grown `_MIB_BETWEEN_RELEASES` past its last one,
    whether or not the module is about to change.

    Releasing takes two steps, and neither is worth much alone. Dropping the
    cache marks the memory free but frees nothing the operating system can see;
    XLA:CPU builds each program through LLVM, whose intermediate representation
    is ordinary heap, and a C allocator holds a freed block in its arena rather
    than returning it. Compilation is multi-threaded and an arena belongs to a
    thread, so the next program allocates fresh instead of reusing what the last
    one left, and resident memory tracks the sum of every program ever built
    rather than the largest one live at a time. Measured on the certified query
    battery, a worker at 9304 MiB fell to 1472 MiB on dropping the cache and
    trimming, having given back under 60 MiB for a trim without a drop.
    """
    if not should_release_compiled_programs(config=item.config):
        return
    session = item.session
    resident = resident_mebibytes()
    if resident is not None and getattr(session, "_lcm_mib_at_last_release", 0) == 0:
        session._lcm_mib_at_last_release = resident
    since = getattr(session, "_lcm_mib_at_last_release", 0)
    grown = resident is not None and resident - since >= _MIB_BETWEEN_RELEASES
    current = getattr(item, "module", None)
    upcoming = getattr(nextitem, "module", None) if nextitem is not None else None
    if upcoming is not None and upcoming is current and not grown:
        return
    jax.clear_caches()
    return_free_heap_to_os()
    # Measured after the release, so a module whose memory is held by something a
    # release cannot reach -- a session fixture, say -- settles at its own level
    # instead of releasing on every test from then on.
    session._lcm_mib_at_last_release = resident_mebibytes() or 0


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


def _apply_backend_skips(*, items: list[pytest.Item]) -> None:
    """Apply the backend-keyed skip markers, both of which mean "not on CPU".

    Which backend a run gets is never passed to the tests: CI picks a pixi
    environment (`tests-cpu` or `tests-cuda*`), that decides which jaxlib is
    installed, and `jax.default_backend()` reads the consequence. A skip keyed
    on it is therefore correct in every CI leg with no change to any CI
    command, which a `-m` selector would not be — `-m` replaces the configured
    expression rather than intersecting it, so every invocation would have to
    restate it and any that forgot would silently include the test.

    This runs at collection rather than as an autouse fixture, because a
    higher-scoped fixture is instantiated before a function-scoped one: a
    module-scoped fixture that solves a GPU-scale model would run — and exhaust
    the box — before a fixture-based skip could fire.

    The two markers say different things and are kept apart on purpose:

    - `gpu` — the test *needs* a GPU. A permanent property of the test: it is
      too large for a CPU-only box, or its expected values were generated on
      one. Pass `reason=` for anything more specific than "requires GPU".
    - `skipif_cpu` — the test would run fine on CPU, but XLA:CPU's LLVM does
      not finish compiling the program. A defect in a dependency, not a
      property of the test, so it is expected to be retired: re-run the marked
      tests on CPU at both precisions, and if they complete, delete the marker
      instead of letting it decay into a permanent skip.
    """
    if jax.default_backend() != "cpu":
        return

    for item in items:
        requires_gpu = item.get_closest_marker("gpu")
        if requires_gpu:
            item.add_marker(
                pytest.mark.skip(
                    reason=requires_gpu.kwargs.get("reason", "requires GPU")
                )
            )
        elif item.get_closest_marker("skipif_cpu"):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "XLA:CPU does not finish compiling this program; it "
                        "compiles on GPU. Re-run the marked tests on CPU to "
                        "check whether this still holds."
                    )
                )
            )
