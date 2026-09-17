"""Structural validation of initial conditions: regimes, states, shapes and ages."""

import jax.numpy as jnp
import pytest

from _lcm.params.processing import process_params
from _lcm.simulation.initial_conditions import validate_initial_conditions
from _lcm.typing import FlatParams
from lcm import Model
from lcm.exceptions import InvalidInitialConditionsError
from tests.simulation.initial_conditions._models import make_asymmetric_state_model

_ACTIVE = jnp.int32(0)
_TERMINAL = jnp.int32(1)


def test_validate_initial_conditions_valid_input(
    *, model: Model, flat_params: FlatParams
) -> None:
    """Valid input should not raise."""
    validate_initial_conditions(
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([10.0, 50.0]),
            "health": jnp.array([0, 1]),
            "regime_id": jnp.array([_ACTIVE, _ACTIVE]),
        },
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
        ages=model.ages,
    )


def test_validate_initial_conditions_missing_state(
    *, model: Model, flat_params: FlatParams
) -> None:
    """Missing state should raise InvalidInitialConditionsError."""
    with pytest.raises(
        InvalidInitialConditionsError, match=r"Missing model states: \['health'\]"
    ):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0, 0.0]),
                "wealth": jnp.array([10.0, 50.0]),
                "regime_id": jnp.array([_ACTIVE, _ACTIVE]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_validate_initial_conditions_extra_state(
    *, model: Model, flat_params: FlatParams
) -> None:
    """Extra state should raise InvalidInitialConditionsError."""
    with pytest.raises(InvalidInitialConditionsError, match="Unknown initial states"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0]),
                "wealth": jnp.array([10.0]),
                "health": jnp.array([0]),
                "unknown": jnp.array([1.0]),
                "regime_id": jnp.array([_ACTIVE]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_validate_initial_conditions_inconsistent_lengths(
    *, model: Model, flat_params: FlatParams
) -> None:
    """Arrays with different lengths should raise InvalidInitialConditionsError."""
    with pytest.raises(InvalidInitialConditionsError, match="same length"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0, 0.0]),
                "wealth": jnp.array([10.0, 20.0]),
                "health": jnp.array([0]),
                "regime_id": jnp.array([_ACTIVE, _ACTIVE]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_validate_initial_conditions_invalid_regime_id(
    *, model: Model, flat_params: FlatParams
) -> None:
    """Invalid regime id should raise InvalidInitialConditionsError."""
    with pytest.raises(InvalidInitialConditionsError, match="Invalid regime"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0]),
                "wealth": jnp.array([10.0]),
                "health": jnp.array([0]),
                "regime_id": jnp.array([99]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_validate_initial_conditions_invalid_age_values(
    *, model: Model, flat_params: FlatParams
) -> None:
    """Age values not on the grid should raise InvalidInitialConditionsError."""
    with pytest.raises(InvalidInitialConditionsError, match="Invalid age values"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0, 99.0]),
                "wealth": jnp.array([10.0, 50.0]),
                "health": jnp.array([0, 1]),
                "regime_id": jnp.array([_ACTIVE, _ACTIVE]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_missing_age_error_message(*, model: Model, flat_params: FlatParams) -> None:
    """Missing 'age' in initial conditions should produce a helpful message."""
    with pytest.raises(
        InvalidInitialConditionsError,
        match="'age' must be provided in initial_states",
    ):
        validate_initial_conditions(
            initial_conditions={
                "wealth": jnp.array([10.0]),
                "health": jnp.array([0]),
                "regime_id": jnp.array([_ACTIVE]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_subject_in_inactive_regime_at_starting_age() -> None:
    """Subject starts in dead at age 0, but dead is only active for age >= 2."""
    model = make_asymmetric_state_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _dead = model.regime_names_to_ids["dead"]

    with pytest.raises(InvalidInitialConditionsError, match="not active"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0]),
                "wealth": jnp.array([10.0]),
                "regime_id": jnp.array([_dead]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_all_subjects_in_regime_with_fewer_states() -> None:
    """Both subjects start in dead, which only needs wealth — health is not required."""
    model = make_asymmetric_state_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _dead = model.regime_names_to_ids["dead"]

    validate_initial_conditions(
        initial_conditions={
            "age": jnp.array([2.0, 2.0]),
            "wealth": jnp.array([10.0, 50.0]),
            "regime_id": jnp.array([_dead, _dead]),
        },
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
        ages=model.ages,
    )


def test_mixed_regimes_all_union_states_provided() -> None:
    """One subject in alive (needs wealth + health), one in dead (needs wealth).

    Passing the full union of states (wealth + health + age) satisfies validation.
    """
    model = make_asymmetric_state_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _alive = model.regime_names_to_ids["alive"]
    _dead = model.regime_names_to_ids["dead"]

    validate_initial_conditions(
        initial_conditions={
            "age": jnp.array([0.0, 2.0]),
            "wealth": jnp.array([10.0, 50.0]),
            "health": jnp.array([0, 0]),
            "regime_id": jnp.array([_alive, _dead]),
        },
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
        ages=model.ages,
    )
