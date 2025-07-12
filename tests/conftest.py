from __future__ import annotations

from dataclasses import make_dataclass

import pytest
from jax import config


def pytest_sessionstart(session):  # noqa: ARG001
    config.update("jax_enable_x64", val=True)


@pytest.fixture(scope="session")
def binary_category_class():
    return make_dataclass("BinaryCategoryClass", [("cat0", int, 0), ("cat1", int, 1)])
