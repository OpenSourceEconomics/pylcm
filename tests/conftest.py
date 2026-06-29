from collections.abc import Iterator
from dataclasses import make_dataclass

import jax.numpy as jnp
import pytest
from jax import config as jax_config

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


def pytest_configure(config):
    """Configure JAX precision based on the --precision flag."""
    global X64_ENABLED, DECIMAL_PRECISION  # noqa: PLW0603

    X64_ENABLED = config.getoption("--precision") == "64"
    DECIMAL_PRECISION = 12 if X64_ENABLED else 5

    jax_config.update("jax_enable_x64", val=X64_ENABLED)


# DC-EGM-family module-name tokens outside `tests/solution/`. The whole
# `tests/solution/` tree is the solve/oracle battery and is matched by directory.
_SLOW_MODULE_TOKENS = (
    "dcegm",
    "negm",
    "ds_app",
    "ds2024",
    "ds_pension",
    "ds_housing",
    "taste_shock",
    "mahler_yum",
    "solvers",
)


def pytest_collection_modifyitems(items):
    """Mark the DC-EGM solve/simulate/oracle battery `slow`.

    These tests AOT-compile heavy JAX models; four in parallel exhaust a small
    CI runner's RAM (the macOS and GPU runners). They carry the `slow` marker so
    a memory-constrained runner can deselect them with `-m "not slow"` — the
    platform-independent kernel stays covered on the larger Linux/GPU runners.
    """
    slow = pytest.mark.slow
    for item in items:
        name = item.path.name
        if "solution" in item.path.parts or any(
            token in name for token in _SLOW_MODULE_TOKENS
        ):
            item.add_marker(slow)


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
