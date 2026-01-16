from __future__ import annotations

from dataclasses import make_dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Literal


# Module-level precision settings (updated in pytest_configure based on --precision)
X64_ENABLED: bool = True
DECIMAL_PRECISION: int = 14
DECIMAL_PRECISION_RELAXED: int = 5  # For tests that need relaxed precision regardless


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

    precision = config.getoption("--precision")
    X64_ENABLED = precision == "64"
    DECIMAL_PRECISION = 14 if X64_ENABLED else 5

    from jax import config as jax_config  # noqa: PLC0415

    jax_config.update("jax_enable_x64", val=X64_ENABLED)


@pytest.fixture(scope="session")
def precision(request) -> Literal["32", "64"]:
    """Fixture exposing the precision setting."""
    return request.config.getoption("--precision")


@pytest.fixture(scope="session")
def x64_enabled(precision) -> bool:
    """Fixture indicating whether 64-bit precision is enabled."""
    return precision == "64"


@pytest.fixture(scope="session")
def decimal_precision(precision) -> int:
    """Fixture providing the decimal precision for assertions based on JAX precision."""
    return 14 if precision == "64" else 5


@pytest.fixture(scope="session")
def binary_category_class():
    return make_dataclass("BinaryCategoryClass", [("cat0", int, 0), ("cat1", int, 1)])
