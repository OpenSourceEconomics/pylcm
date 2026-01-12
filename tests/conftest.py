from __future__ import annotations

import os
from dataclasses import make_dataclass

import pytest
from jax import config

# Check environment variable for x64 setting, default to True
X64_ENABLED = os.environ.get("JAX_ENABLE_X64", "1").lower() in ("1", "true", "yes")

# Precision settings for tests
DECIMAL_PRECISION = 14 if X64_ENABLED else 5
DECIMAL_PRECISION_RELAXED = 5  # For tests that need relaxed precision regardless


def pytest_sessionstart(session):  # noqa: ARG001
    config.update("jax_enable_x64", val=X64_ENABLED)


@pytest.fixture(scope="session")
def binary_category_class():
    return make_dataclass("BinaryCategoryClass", [("cat0", int, 0), ("cat1", int, 1)])
