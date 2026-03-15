"""Test analytical solution and simulation with only discrete actions (stochastic)."""

from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.typing import DiscreteState, FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.test_solution_on_toy_model_deterministic import (
    RegimeId,
    alive_deterministic,
    dead,
    dict_of_vectors_to_matrix,
    matrix_to_dict_of_vectors,
    next_wealth,
    policy_first_period_deterministic,
    value_first_period_deterministic,
)


# ======================================================================================
# Model specification
# ======================================================================================
@categorical(ordered=False)
class Health:
    bad: int
    good: int


def next_health(health: DiscreteState, probs_array: FloatND) -> FloatND:
    return probs_array[health]


alive_stochastic = alive_deterministic.replace(
    states=dict(alive_deterministic.states) | {"health": DiscreteGrid(Health)},
    state_transitions=dict(alive_deterministic.state_transitions)
    | {"health": MarkovTransition(next_health)},
)


# ======================================================================================
# Analytical solution and simulation (stochastic model)
# ======================================================================================
def value_second_period_stochastic(wealth, health):
    """Value function in the second (last) period. Computed using pen and paper."""
    consumption = np.minimum(1, np.floor(wealth))
    return np.log(1 + consumption) * health


def policy_second_period_stochastic(wealth, health):
    """Policy function in the second (last) period. Computed using pen and paper.

    First column corresponds to consumption choice, second to work choice.

    """
    policy = np.column_stack(
        (np.minimum(1, np.floor(wealth)) * health, np.zeros_like(wealth)),
    ).astype(int)
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "work"])


def value_first_period_stochastic(wealth, health, params):
    """Value function in the first period. Computed using pen and paper."""
    probs_array = params["next_health"]["probs_array"]

    index = (wealth < 1).astype(int)  # map wealth to indices 0 and 1

    _values = np.array(
        [
            params["discount_factor"] * probs_array[0, 1] * np.log(2),
            np.maximum(
                0, params["discount_factor"] * probs_array[0, 1] * np.log(2) - 0.5
            ),
        ],
    )
    value_health_0 = _values[index]

    new_discount_factor = (
        params["discount_factor"] * params["next_health"]["probs_array"][1, 1]
    )
    value_health_1 = value_first_period_deterministic(
        wealth, params={"discount_factor": new_discount_factor}
    )

    # Combined
    return np.where(health, value_health_1, value_health_0)


def policy_first_period_stochastic(wealth, health, params):
    """Policy function in the first period. Computed using pen and paper."""
    probs_array = params["next_health"]["probs_array"]

    index = (wealth < 1).astype(int)  # map wealth to indices 0 and 1
    _policies = np.array(
        [
            [0, 0],
            [
                0,
                np.argmax(
                    (
                        0,
                        params["discount_factor"] * probs_array[0, 1] * np.log(2) - 0.5,
                    ),
                ),
            ],
        ],
    )
    policy_health_0 = _policies[index]

    new_discount_factor = (
        params["discount_factor"] * params["next_health"]["probs_array"][1, 1]
    )
    _policy_health_1 = policy_first_period_deterministic(
        wealth,
        params={"discount_factor": new_discount_factor},
    )

    policy_health_1 = dict_of_vectors_to_matrix(_policy_health_1)

    _health = health.reshape(-1, 1)
    policies = _health * policy_health_1 + (1 - _health) * policy_health_0
    return matrix_to_dict_of_vectors(policies, col_names=["consumption", "work"])


def analytical_solve_stochastic(wealth_grid, health_grid, params):
    V_arr_0 = value_first_period_stochastic(
        wealth=wealth_grid,
        health=health_grid,
        params=params,
    )
    V_arr_1 = value_second_period_stochastic(wealth=wealth_grid, health=health_grid)
    return [V_arr_0, V_arr_1]


def analytical_simulate_stochastic(initial_wealth, initial_health, health_1, params):
    """Compute analytical simulation results in the same format as to_dataframe().

    Returns DataFrame with columns: period, subject_id, regime, value, health, wealth,
    consumption, working. Sorted by (subject_id, period).
    Uses categorical dtypes for discrete variables to match to_dataframe() output.
    """
    n_subjects = len(initial_wealth)

    # Period 0
    V_arr_0 = value_first_period_stochastic(
        initial_wealth, initial_health, params=params
    )
    policy_0 = policy_first_period_stochastic(
        initial_wealth, initial_health, params=params
    )

    # Period 1
    wealth_1 = next_wealth(initial_wealth, **policy_0)
    V_arr_1 = value_second_period_stochastic(wealth_1, health_1)
    policy_1 = policy_second_period_stochastic(wealth_1, health_1)

    # Build DataFrame in the same format as to_dataframe()
    # Sorted by (subject_id, period)
    health_codes = np.concatenate([initial_health, health_1]).astype(int)
    consumption_codes = np.concatenate(
        [policy_0["consumption"], policy_1["consumption"]]
    ).astype(int)
    work_codes = np.concatenate([policy_0["work"], policy_1["work"]]).astype(int)

    df = pd.DataFrame(
        {
            "period": np.concatenate(
                [np.zeros(n_subjects), np.ones(n_subjects)]
            ).astype(int),
            "subject_id": np.tile(np.arange(n_subjects), 2),
            "regime": pd.Categorical(
                ["alive"] * (2 * n_subjects), categories=["alive", "dead"]
            ),
            "value": np.concatenate([V_arr_0, V_arr_1]),
            "health": pd.Categorical.from_codes(
                health_codes.tolist(),
                categories=pd.Index(["bad", "good"]),
            ),
            "wealth": np.concatenate([initial_wealth, wealth_1]),
            "consumption": pd.Categorical.from_codes(
                consumption_codes.tolist(),
                categories=pd.Index(["low", "high"]),
            ),
            "work": pd.Categorical.from_codes(
                work_codes.tolist(),
                categories=pd.Index(["not_working", "working"]),
            ),
        }
    )
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


# ======================================================================================
# Tests
# ======================================================================================


HEALTH_TRANSITION = [
    jnp.array([[0.9, 0.1], [0.2, 0.8]]),
    jnp.array([[0.9, 0.1], [0, 1]]),
    jnp.array([[0.5, 0.5], [0.2, 0.8]]),
]


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
@pytest.mark.parametrize("probs_array", HEALTH_TRANSITION)
def test_stochastic_solve(discount_factor, n_wealth_points, probs_array):
    # Update model
    # ==================================================================================
    n_periods = 3
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    new_states = dict(alive_stochastic.states)
    new_states["wealth"] = cast("LinSpacedGrid", new_states["wealth"]).replace(
        n_points=n_wealth_points
    )
    model = Model(
        regimes={
            "alive": alive_stochastic.replace(
                states=new_states, active=lambda age: age < n_periods - 1
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    # Solve model using LCM
    # ==================================================================================
    params = {
        "next_health": {"probs_array": probs_array},
        "next_regime": {"final_age_alive": model.n_periods - 2},
    }
    got = model.solve(params={"discount_factor": discount_factor, "alive": params})

    # Compute analytical solution
    # ==================================================================================
    wealth_grid_class = cast("LinSpacedGrid", new_states["wealth"])
    _wealth_grid = np.linspace(
        start=wealth_grid_class.start,
        stop=wealth_grid_class.stop,
        num=wealth_grid_class.n_points,
    )
    _health_grid = np.array([0, 1])

    # Repeat arrays to evaluate on all combinations of wealth and health
    wealth_grid = np.tile(_wealth_grid, len(_health_grid))
    health_grid = np.repeat(_health_grid, len(_wealth_grid))

    analytical_params = {"discount_factor": discount_factor, **params}
    _expected = analytical_solve_stochastic(
        wealth_grid=wealth_grid,
        health_grid=health_grid,
        params=analytical_params,
    )
    expected = [
        arr.reshape((len(_health_grid), len(_wealth_grid))) for arr in _expected
    ]

    # Do not assert that in the first period, the arrays have the same values on the
    # first and last index: TODO (@timmens): THIS IS A BUG AND NEEDS TO BE INVESTIGATED.
    # ==================================================================================
    aaae(
        got[0]["alive"][:, slice(1, -1)],
        expected[0][:, slice(1, -1)],
        decimal=DECIMAL_PRECISION,
    )
    aaae(got[1]["alive"], expected[1], decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
@pytest.mark.parametrize("probs_array", HEALTH_TRANSITION)
def test_stochastic_simulate(discount_factor, n_wealth_points, probs_array):
    # Update model
    # ==================================================================================
    n_periods = 3
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    new_states = dict(alive_stochastic.states)
    new_states["wealth"] = cast("LinSpacedGrid", new_states["wealth"]).replace(
        n_points=n_wealth_points
    )
    model = Model(
        regimes={
            "alive": alive_stochastic.replace(
                states=new_states, active=lambda age: age < n_periods - 1
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    # Simulate model using LCM
    # ==================================================================================
    params_alive = {
        "next_health": {"probs_array": probs_array},
        "next_regime": {"final_age_alive": model.n_periods - 2},
    }
    initial_conditions = {
        "wealth": jnp.array([0.25, 0.75, 1.25, 1.75, 2.0]),
        "health": jnp.array([0, 1, 0, 1, 1]),
        "age": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "regime_id": jnp.array([RegimeId.alive] * 5),
    }
    result = model.solve_and_simulate(
        params={"discount_factor": discount_factor, "alive": params_alive},
        initial_conditions=initial_conditions,
    )
    # Filter to alive regime only (dead regime has trivial values)
    got = result.to_dataframe().query('regime == "alive"').reset_index(drop=True)

    # Need to use health of second period from LCM output, to assure that the same
    # stochastic draws are used in the analytical simulation.
    # Convert categorical health to codes for analytical function
    health_1 = got.query("period == 1")["health"].cat.codes.to_numpy()

    analytical_params = {"discount_factor": discount_factor, **params_alive}
    expected = analytical_simulate_stochastic(
        initial_wealth=initial_conditions["wealth"],
        initial_health=initial_conditions["health"],
        health_1=health_1,
        params=analytical_params,
    )

    # Drop rows with wealth at boundary (analytical solution has edge effects)
    # Also drop age column (analytical function doesn't include it)
    got = got.query("wealth != 2").drop(columns=["age"]).reset_index(drop=True)
    expected = expected.query("wealth != 2").reset_index(drop=True)

    assert_frame_equal(
        got, expected, check_like=True, check_dtype=False, check_categorical=False
    )
