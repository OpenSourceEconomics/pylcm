"""Fixtures shared by the initial-condition test modules."""

import pytest

from _lcm.params.processing import process_params
from _lcm.typing import FlatParams
from lcm import Model
from tests.simulation.initial_conditions._models import make_minimal_model


@pytest.fixture
def model() -> Model:
    """Minimal model with two states (wealth, health) for initial states tests."""
    return make_minimal_model()


@pytest.fixture
def flat_params(model: Model) -> FlatParams:
    """Process params for the minimal model."""
    return process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
