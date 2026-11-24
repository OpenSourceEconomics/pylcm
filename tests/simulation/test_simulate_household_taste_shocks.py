"""Spec for simulating a household in which exactly one member decides under EV1.

The decider's discrete action is drawn as a single's is: one mean-zero Gumbel
draw per discrete action, added to the decider's own conditional maxima over
the continuous actions, before the argmax. The household takes that action.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_models import taste_shocks_toy

pytestmark = pytest.mark.slow

_SCALE = 0.2
_DISCOUNT_FACTOR = 0.95
_INITIAL_WEALTH = 4.6  # third node of the 6-point wealth grid


def _reference_decider_work_probability(*, wealth: float) -> float:
    """Softmax probability that member `x`, the single's twin, works."""
    terminal_wealth = np.linspace(0.0, 12.0, 25)
    consumption = np.linspace(0.5, 5.0, 8)
    v_done = np.log(terminal_wealth + 1.0)
    qc = np.empty(2)
    for d in (0, 1):
        next_w = wealth - consumption + taste_shocks_toy.WAGE * d
        q = (
            np.log(consumption)
            - taste_shocks_toy.KAPPA * d
            + _DISCOUNT_FACTOR * np.interp(next_w, terminal_wealth, v_done)
        )
        qc[d] = np.max(np.where(consumption <= wealth, q, -np.inf))
    probabilities = np.exp((qc - qc.max()) / _SCALE)
    return float(probabilities[1] / probabilities.sum())


def _initial_conditions(n_subjects: int) -> dict[str, jnp.ndarray]:
    return {
        "age": jnp.full(n_subjects, 40.0),
        "wealth": jnp.full(n_subjects, _INITIAL_WEALTH),
        "regime_id": jnp.full(
            n_subjects, taste_shocks_toy.ToyRegimeId.alive, dtype=jnp.int32
        ),
    }


def test_simulated_work_frequency_of_the_decider_is_its_softmax_probability() -> None:
    """The household works as often as the decider's logit probability says.

    Member `x` decides and its problem is the single's. With 20k subjects at
    one wealth node the Monte Carlo standard error of the frequency is below
    0.005.
    """
    n_subjects = 20_000
    result = taste_shocks_toy.get_household_model(
        weights={"x": 1.0, "y": 0.0}
    ).simulate(
        params=taste_shocks_toy.get_household_params(
            scale=_SCALE, discount_factor=_DISCOUNT_FACTOR
        ),
        initial_conditions=_initial_conditions(n_subjects),
        log_level="debug",
        seed=5471,
    )

    work = result.to_dataframe(use_categorical_codes=True).query("period == 0")["work"]

    np.testing.assert_allclose(
        work.mean(),
        _reference_decider_work_probability(wealth=_INITIAL_WEALTH),
        atol=0.015,
    )


def test_household_of_one_decider_draws_the_singles_choices() -> None:
    """A household of one member with weight 1 draws exactly the single's actions."""
    n_subjects = 64
    single = taste_shocks_toy.get_model().simulate(
        params=taste_shocks_toy.get_params(
            scale=_SCALE, discount_factor=_DISCOUNT_FACTOR
        ),
        initial_conditions=_initial_conditions(n_subjects),
        log_level="debug",
        seed=5471,
    )
    household = taste_shocks_toy.get_household_model(weights={"x": 1.0}).simulate(
        params=taste_shocks_toy.get_household_params(
            scale=_SCALE, discount_factor=_DISCOUNT_FACTOR
        ),
        initial_conditions=_initial_conditions(n_subjects),
        log_level="debug",
        seed=5471,
    )

    columns = ["period", "work", "consumption"]
    np.testing.assert_array_equal(
        household.to_dataframe(use_categorical_codes=True)[columns].to_numpy(),
        single.to_dataframe(use_categorical_codes=True)[columns].to_numpy(),
    )
