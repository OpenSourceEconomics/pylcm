# Beartype claw must instrument `lcm.grids`, `lcm.shocks`, and `lcm.params`
# before any submodule of those packages is imported. The registrations below
# install an import hook on `sys.meta_path` that transforms each matching
# module's AST at first import. Subsequent imports in this file must come
# after the registration, hence the `# noqa: E402` markers on them.

from beartype.claw import beartype_package

from lcm._beartype_conf import GRID_CONF, PARAMS_CONF

beartype_package("lcm.grids", conf=GRID_CONF)
beartype_package("lcm.shocks", conf=GRID_CONF)
beartype_package("lcm.params", conf=PARAMS_CONF)

from collections.abc import Iterator  # noqa: E402
from dataclasses import make_dataclass  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from jax import config as jax_config  # noqa: E402

from lcm.typing import ScalarInt  # noqa: E402

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
