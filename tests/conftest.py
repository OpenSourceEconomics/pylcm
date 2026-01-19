from dataclasses import make_dataclass

import pytest

# Module-level precision settings (updated by pytest_configure based on --precision)
X64_ENABLED: bool = True
DECIMAL_PRECISION: int = 14


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
    DECIMAL_PRECISION = 14 if X64_ENABLED else 5

    from jax import config as jax_config  # noqa: PLC0415

    jax_config.update("jax_enable_x64", val=X64_ENABLED)


@pytest.fixture(scope="session")
def binary_category_class():
    return make_dataclass("BinaryCategoryClass", [("cat0", int, 0), ("cat1", int, 1)])
