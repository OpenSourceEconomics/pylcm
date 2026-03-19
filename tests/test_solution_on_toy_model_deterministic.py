"""Test analytical solution and simulation with only discrete, deterministic actions."""

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
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class DiscreteConsumption:
    low: int
    high: int


@categorical(ordered=False)
class LaborSupply:
    do_not_work: int
    work: int


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def utility(
    consumption: DiscreteAction,
    labor_supply: DiscreteAction,
    wealth: ContinuousState,  # noqa: ARG001
    health: DiscreteState,
) -> FloatND:
    return jnp.log(1.0 + health * consumption) - 0.5 * labor_supply


def next_wealth(
    wealth: ContinuousState, consumption: DiscreteAction, labor_supply: DiscreteAction
) -> ContinuousState:
    return wealth - consumption + labor_supply


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def borrowing_constraint(
    consumption: DiscreteAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


alive_deterministic = Regime(
    actions={
        "consumption": DiscreteGrid(DiscreteConsumption),
        "labor_supply": DiscreteGrid(LaborSupply),
    },
    states={
        "wealth": LinSpacedGrid(
            start=0,
            stop=2,
            n_points=1,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
    },
    functions={"utility": utility},
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transition=next_regime,
    active=lambda age: age < 1,  # n_periods=2, so active in period 0
)

dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
    active=lambda age: age >= 1,  # n_periods=2, so active in period 1
)


def value_second_period_deterministic(wealth):
    """Value function in the second (last) period. Computed using pen and paper."""
    consumption = np.minimum(1, np.floor(wealth))
    return np.log(1 + consumption)


def policy_second_period_deterministic(wealth):
    """Policy function in the second (last) period. Computed using pen and paper.

    First column corresponds to consumption choice, second to work choice.

    """
    policy = np.column_stack(
        (np.minimum(1, np.floor(wealth)), np.zeros_like(wealth)),
    ).astype(int)
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "labor_supply"])


def value_first_period_deterministic(wealth, params):
    """Value function in the first period. Computed using pen and paper."""
    index = np.floor(wealth).astype(int)  # map wealth to index 0, 1 and 2
    values = np.array(
        [
            np.maximum(0, params["discount_factor"] * np.log(2) - 0.5),
            np.maximum(0, params["discount_factor"] * np.log(2) - 0.5) + np.log(2),
            (1 + params["discount_factor"]) * np.log(2),
        ],
    )
    return values[index]


def policy_first_period_deterministic(wealth, params):
    """Policy function in the first period. Computed using pen and paper."""
    index = np.floor(wealth).astype(int)  # map wealth to indices 0, 1 and 2
    policies = np.array(
        [
            [0, np.argmax((0, params["discount_factor"] * np.log(2) - 0.5))],
            [1, np.argmax((0, params["discount_factor"] * np.log(2) - 0.5))],
            [1, 0],
        ],
        dtype=int,
    )
    policy = policies[index]
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "labor_supply"])


def analytical_solve_deterministic(wealth_grid, params):
    V_arr_0 = value_first_period_deterministic(wealth_grid, params=params)
    V_arr_1 = value_second_period_deterministic(wealth_grid)
    return [V_arr_0, V_arr_1]


def analytical_simulate_deterministic(initial_wealth, params):
    """Compute analytical simulation results in the same format as to_dataframe().

    Returns DataFrame with columns: period, subject_id, regime, value, wealth,
    consumption, working. Sorted by (subject_id, period).
    Uses categorical dtypes for discrete variables to match to_dataframe() output.
    """
    n_subjects = len(initial_wealth)

    # Period 0
    V_arr_0 = value_first_period_deterministic(initial_wealth, params=params)
    policy_0 = policy_first_period_deterministic(initial_wealth, params=params)

    # Period 1
    wealth_1 = next_wealth(initial_wealth, **policy_0)
    V_arr_1 = value_second_period_deterministic(wealth_1)
    policy_1 = policy_second_period_deterministic(wealth_1)

    # Build DataFrame in the same format as to_dataframe()
    # Sorted by (subject_id, period)
    consumption_codes = np.concatenate(
        [policy_0["consumption"], policy_1["consumption"]]
    ).astype(int)
    work_codes = np.concatenate(
        [policy_0["labor_supply"], policy_1["labor_supply"]]
    ).astype(int)

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
            "wealth": np.concatenate([initial_wealth, wealth_1]),
            "consumption": pd.Categorical.from_codes(
                consumption_codes.tolist(),
                categories=pd.Index(["low", "high"]),
            ),
            "labor_supply": pd.Categorical.from_codes(
                work_codes.tolist(),
                categories=pd.Index(["do_not_work", "work"]),
            ),
        }
    )
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


def matrix_to_dict_of_vectors(arr, col_names):
    """Transform a matrix into a dict of vectors."""
    if arr.ndim != 2:
        raise ValueError("arr must be a two-dimensional array (matrix).")
    return dict(zip(col_names, arr.transpose(), strict=True))


def dict_of_vectors_to_matrix(d):
    """Transform a dict of vectors into a matrix."""
    return np.column_stack(list(d.values()))


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
def test_deterministic_solve(discount_factor, n_wealth_points):
    n_periods = 3
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    new_states = dict(alive_deterministic.states)
    new_states["wealth"] = cast("LinSpacedGrid", new_states["wealth"]).replace(
        n_points=n_wealth_points
    )
    model = Model(
        regimes={
            "alive": alive_deterministic.replace(
                states=new_states, active=lambda age: age < n_periods - 1
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    params_alive = {
        "utility": {"health": 1},
        "next_regime": {"final_age_alive": model.n_periods - 2},
    }
    got = model.solve(
        params={"discount_factor": discount_factor, "alive": params_alive}
    )

    wealth_grid_class = cast("LinSpacedGrid", new_states["wealth"])
    wealth_grid = np.linspace(
        start=wealth_grid_class.start,
        stop=wealth_grid_class.stop,
        num=wealth_grid_class.n_points,
    )
    analytical_params = {"discount_factor": discount_factor, **params_alive}
    expected = analytical_solve_deterministic(wealth_grid, params=analytical_params)

    # Do not assert that in the first period, the arrays have the same values on the
    # first and last index: TODO (@timmens): THIS IS A BUG AND NEEDS TO BE INVESTIGATED.
    aaae(
        got[0]["alive"][slice(1, -1)],
        expected[0][slice(1, -1)],
        decimal=DECIMAL_PRECISION,
    )
    aaae(got[1]["alive"], expected[1], decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
def test_deterministic_simulate(discount_factor, n_wealth_points):
    n_periods = 3
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    new_states = dict(alive_deterministic.states)
    new_states["wealth"] = cast("LinSpacedGrid", new_states["wealth"]).replace(
        n_points=n_wealth_points
    )
    model = Model(
        regimes={
            "alive": alive_deterministic.replace(
                states=new_states, active=lambda age: age < n_periods - 1
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    params_alive = {
        "utility": {"health": 1},
        "next_regime": {"final_age_alive": model.n_periods - 2},
    }
    result = model.simulate(
        params={"discount_factor": discount_factor, "alive": params_alive},
        initial_conditions={
            "wealth": jnp.array([0.25, 0.75, 1.25, 1.75]),
            "age": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "regime": jnp.array([RegimeId.alive] * 4),
        },
        V_arr_dict=None,
    )
    # Filter to alive regime only (dead regime has trivial values)
    got = (
        result.to_dataframe()
        .query('regime == "alive"')
        .drop(columns=["age"])  # Analytical function doesn't include age
        .reset_index(drop=True)
    )

    analytical_params = {"discount_factor": discount_factor, **params_alive}
    expected = analytical_simulate_deterministic(
        initial_wealth=np.array([0.25, 0.75, 1.25, 1.75]),
        params=analytical_params,
    )

    assert_frame_equal(
        got, expected, check_like=True, check_dtype=False, check_categorical=False
    )
