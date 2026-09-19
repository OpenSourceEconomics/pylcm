"""Validation of discrete state codes in initial conditions."""

import jax.numpy as jnp
import pytest

from _lcm.simulation.initial_conditions import validate_initial_conditions
from _lcm.typing import FlatParams
from lcm import Model
from lcm.exceptions import InvalidInitialConditionsError

_ACTIVE = jnp.int32(0)
_TERMINAL = jnp.int32(1)


def test_validate_initial_conditions_invalid_discrete_value(
    *, model: Model, flat_params: FlatParams
) -> None:
    """Invalid discrete state code should raise InvalidInitialConditionsError."""
    with pytest.raises(InvalidInitialConditionsError, match=r"Invalid values.*health"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0]),
                "wealth": jnp.array([10.0]),
                "health": jnp.array([5]),
                "regime_id": jnp.array([_ACTIVE]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )
