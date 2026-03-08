from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm.error_handling import validate_regime_transition_probs
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm_examples.mortality import get_model, get_params

# ======================================================================================
# Tests for validate_regime_transition_probs
# ======================================================================================


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
        period=0,
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
        period=0,
    )


def test_raises_for_probs_not_summing_to_one():
    """Probabilities that don't sum to 1 raise an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.5, 0.6]),
            "retirement": jnp.array([0.3, 0.4]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"sum to .* instead of 1\.0",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            period=0,
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
        match="'dead' is inactive",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            period=0,
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
            period=0,
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
        match="Non-finite values",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            period=0,
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
            period=0,
        )


# ======================================================================================
# Integration tests via public Model methods
# ======================================================================================

N_PERIODS = 4


def _invalid_survival_probs(n_periods: int) -> jnp.ndarray:
    """Build survival probs with an out-of-bounds entry (2.0 → death prob = -1.0)."""
    return jnp.array([2.0] + [0.0] * (n_periods - 2))


def test_solve_raises_for_invalid_regime_transition_probs():
    """model.solve() raises for out-of-bounds regime transition probabilities."""
    model = get_model(N_PERIODS)
    params = get_params(N_PERIODS, survival_probs=_invalid_survival_probs(N_PERIODS))
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params)


def test_simulate_raises_for_invalid_regime_transition_probs():
    """model.simulate() raises for out-of-bounds regime transition probabilities."""
    model = get_model(N_PERIODS)
    good_params = get_params(N_PERIODS)
    V_arr_dict = model.solve(good_params)

    bad_params = get_params(
        N_PERIODS, survival_probs=_invalid_survival_probs(N_PERIODS)
    )
    initial_states = {"age": jnp.array([40.0]), "wealth": jnp.array([10.0])}
    initial_regimes = ["working_life"]
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.simulate(
            params=bad_params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            V_arr_dict=V_arr_dict,
        )


def test_solve_and_simulate_raises_for_invalid_regime_transition_probs():
    """model.solve_and_simulate() raises for out-of-bounds regime transition probs."""
    model = get_model(N_PERIODS)
    params = get_params(N_PERIODS, survival_probs=_invalid_survival_probs(N_PERIODS))
    initial_states = {"age": jnp.array([40.0]), "wealth": jnp.array([10.0])}
    initial_regimes = ["working_life"]
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve_and_simulate(
            params=params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
        )
