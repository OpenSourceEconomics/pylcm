from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.error_handling import _format_sum_violation, validate_regime_transition_probs
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.typing import DiscreteAction, FloatND
from lcm_examples.mortality import RegimeId as MortalityRegimeId
from lcm_examples.mortality import get_model, get_params


def test_valid_probs_all_active():
    """Valid probabilities with all regimes active pass validation."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.7, 0.6]),
            "retirement": jnp.array([0.3, 0.4]),
        }
    )
    validate_regime_transition_probs(
        regime_transition_probs=probs,
        active_regimes_next_period=("working_life", "retirement"),
        regime_name="working_life",
        age=25.0,
        next_age=26.0,
    )


def test_valid_probs_with_inactive_regime_at_zero():
    """Inactive regime with zero probability passes validation."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.7, 0.6]),
            "retirement": jnp.array([0.3, 0.4]),
            "dead": jnp.array([0.0, 0.0]),
        }
    )
    validate_regime_transition_probs(
        regime_transition_probs=probs,
        active_regimes_next_period=("working_life", "retirement"),
        regime_name="working_life",
        age=25.0,
        next_age=26.0,
    )


def test_raises_for_probs_not_summing_to_one():
    """Per-subject probabilities that don't sum to 1 show a DataFrame summary."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.5, 0.6, 0.7]),
            "retirement": jnp.array([0.3, 0.3, 0.3]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"2 of 3 probability vectors do not sum to 1\.0",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            age=25.0,
            next_age=26.0,
        )


def test_raises_for_positive_probability_on_inactive_regime():
    """Positive probability on an inactive regime raises an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.7, 0.6]),
            "retirement": jnp.array([0.1, 0.2]),
            "dead": jnp.array([0.2, 0.2]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"'dead' is inactive at age 26\.0",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            age=25.0,
            next_age=26.0,
        )


def test_raises_for_out_of_bounds_values():
    """Probabilities outside [0, 1] that sum to 1 raise an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([1.5, 0.8]),
            "retirement": jnp.array([-0.5, 0.2]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"values outside \[0, 1\]",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            age=25.0,
            next_age=26.0,
        )


def test_raises_for_nan_values():
    """NaN values in probabilities raise an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([jnp.nan, 0.5]),
            "retirement": jnp.array([jnp.nan, 0.5]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"Non-finite values.*between ages 25\.0 and 26\.0",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            age=25.0,
            next_age=26.0,
        )


def test_raises_for_inf_values():
    """Inf values in probabilities raise an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([jnp.inf, 0.5]),
            "retirement": jnp.array([0.0, 0.5]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match="Non-finite values",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            age=25.0,
            next_age=26.0,
        )


def test_format_sum_violation_with_scalar_input():
    """A 0-d array (scalar) input does not raise IndexError."""
    result = _format_sum_violation(jnp.array(0.5))
    assert "1 of 1 probability vectors do not sum to 1.0" in result


def test_format_sum_violation_with_scalar_input_and_state_action_values():
    """0-d inputs for both sum_all and state_action_values work correctly."""
    result = _format_sum_violation(
        jnp.array(0.5),
        state_action_values=MappingProxyType({"wealth": jnp.array(10.0)}),
    )
    assert "1 of 1 probability vectors do not sum to 1.0" in result
    assert "wealth" in result


@categorical(ordered=False)
class _Action:
    stay: int
    leave: int


@categorical(ordered=False)
class _RegimeId:
    active: int
    terminal: int


def _next_regime_only_fails_for_leave(action: DiscreteAction) -> FloatND:
    """Transition function that is valid for action=0 but invalid for action=1.

    When action=stay (0): returns [1.0, 0.0] — valid.
    When action=leave (1): returns [1.5, -0.5] — out of bounds.

    """
    return jnp.where(
        action == _Action.leave,
        jnp.array([1.5, -0.5]),
        jnp.array([1.0, 0.0]),
    )


def _build_action_dependent_model() -> tuple[Model, dict]:
    """Build a minimal model whose transition bug only shows for the second action."""
    active = Regime(
        transition=MarkovTransition(_next_regime_only_fails_for_leave),
        active=lambda age: age < 27,
        actions={
            "action": DiscreteGrid(_Action),
            "consumption": LinSpacedGrid(start=1, stop=10, n_points=5),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": lambda wealth, consumption: wealth - consumption},
        constraints={"budget": lambda consumption, wealth: consumption <= wealth},
        functions={"utility": lambda consumption: jnp.log(consumption)},  # noqa: PLW0108
    )
    terminal = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": lambda wealth: jnp.log(wealth)},  # noqa: PLW0108
    )
    model = Model(
        regimes={"active": active, "terminal": terminal},
        ages=AgeGrid(start=25, stop=27, step="Y"),
        regime_id_class=_RegimeId,
    )
    params: dict = {"discount_factor": 0.95}
    return model, params


def test_solve_catches_transition_bug_hidden_at_first_grid_point():
    """Pre-solve validation catches invalid probs even if first action value is ok."""
    model, params = _build_action_dependent_model()
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="outside"):
        model.solve(params=params)


N_PERIODS = 4


def _invalid_survival_probs(n_periods: int) -> jnp.ndarray:
    """Build survival probs with an out-of-bounds entry (2.0 → death prob = -1.0)."""
    return jnp.array([2.0] + [0.0] * (n_periods - 2))


def test_solve_raises_for_invalid_regime_transition_probs():
    """model.solve() raises for out-of-bounds regime transition probabilities."""
    model = get_model(N_PERIODS)
    params = get_params(N_PERIODS, survival_probs=_invalid_survival_probs(N_PERIODS))
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params=params)


def test_simulate_raises_for_invalid_regime_transition_probs():
    """model.simulate() raises for out-of-bounds regime transition probabilities."""
    model = get_model(N_PERIODS)
    good_params = get_params(N_PERIODS)
    period_to_regime_to_V_arr = model.solve(params=good_params)

    bad_params = get_params(
        N_PERIODS, survival_probs=_invalid_survival_probs(N_PERIODS)
    )
    initial_conditions = {
        "age": jnp.array([40.0]),
        "wealth": jnp.array([10.0]),
        "regime": jnp.array([MortalityRegimeId.working_life]),
    }
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.simulate(
            params=bad_params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        )


def test_simulate_with_solve_raises_for_invalid_regime_transition_probs():
    """model.simulate(period_to_regime_to_V_arr=None) raises for out-of-bounds probs."""
    model = get_model(N_PERIODS)
    params = get_params(N_PERIODS, survival_probs=_invalid_survival_probs(N_PERIODS))
    initial_conditions = {
        "age": jnp.array([40.0]),
        "wealth": jnp.array([10.0]),
        "regime": jnp.array([MortalityRegimeId.working_life]),
    }
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
        )
