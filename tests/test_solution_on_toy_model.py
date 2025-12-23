"""Test analytical solution and simulation with only discrete actions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousState,
        DiscreteAction,
        DiscreteState,
        FloatND,
        ScalarInt,
    )


# ======================================================================================
# Model specification
# ======================================================================================
@dataclass
class ConsumptionChoice:
    low: int = 0
    high: int = 1


@dataclass
class WorkingStatus:
    retired: int = 0
    working: int = 1


@dataclass
class HealthStatus:
    bad: int = 0
    good: int = 1


@dataclass
class RegimeId:
    alive: int = 0
    dead: int = 1


def utility(
    consumption: DiscreteAction,
    working: DiscreteAction,
    wealth: ContinuousState,  # noqa: ARG001
    health: DiscreteState,
) -> FloatND:
    return jnp.log(1 + health * consumption) - 0.5 * working


def next_wealth(
    wealth: ContinuousState, consumption: DiscreteAction, working: DiscreteAction
) -> ContinuousState:
    return wealth - consumption + working


def next_regime(period: int, n_periods: int) -> ScalarInt:
    death_condition = period >= n_periods - 2  # is dead in last period
    return jnp.where(death_condition, RegimeId.dead, RegimeId.alive)


def borrowing_constraint(
    consumption: DiscreteAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


alive_deterministic = Regime(
    name="alive",
    actions={
        "consumption": DiscreteGrid(ConsumptionChoice),
        "working": DiscreteGrid(WorkingStatus),
    },
    states={
        "wealth": LinspaceGrid(
            start=0,
            stop=2,
            n_points=1,
        ),
    },
    utility=utility,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "next_wealth": next_wealth,
        "next_regime": next_regime,
    },
    active=range(1),  # n_periods=2, so active in period 0
)

dead = Regime(
    name="dead",
    terminal=True,
    utility=lambda: 0.0,
    active=[1],  # n_periods=2, so active in period 1
)


@lcm.mark.stochastic
def next_health(health: DiscreteState, health_transition: FloatND) -> FloatND:
    return health_transition[health]


alive_stochastic = deepcopy(alive_deterministic)
alive_stochastic.transitions["next_health"] = next_health
alive_stochastic.states["health"] = DiscreteGrid(HealthStatus)

model_deterministic = Model(
    [alive_deterministic, dead], regime_id_cls=RegimeId, n_periods=2
)
model_stochastic = Model([alive_stochastic, dead], regime_id_cls=RegimeId, n_periods=2)


# ======================================================================================
# Analytical solution and simulation (deterministic model)
# ======================================================================================
def value_second_period_deterministic(wealth):
    """Value function in the second (last) period. Computed using pen and paper."""
    consumption = np.minimum(1, np.floor(wealth))
    return np.log(1 + consumption)


def policy_second_period_deterministic(wealth):
    """Policy function in the second (last) period. Computed using pen and paper.

    First column corresponds to consumption choice, second to working choice.

    """
    policy = np.column_stack(
        (np.minimum(1, np.floor(wealth)), np.zeros_like(wealth)),
    ).astype(int)
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "working"])


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
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "working"])


def analytical_solve_deterministic(wealth_grid, params):
    V_arr_0 = value_first_period_deterministic(wealth_grid, params=params)
    V_arr_1 = value_second_period_deterministic(wealth_grid)
    return [V_arr_0, V_arr_1]


def analytical_simulate_deterministic(initial_wealth, params):
    # Simulate
    # ==================================================================================
    V_arr_0 = value_first_period_deterministic(initial_wealth, params=params)
    policy_0 = policy_first_period_deterministic(initial_wealth, params=params)

    wealth_1 = next_wealth(initial_wealth, **policy_0)

    V_arr_1 = value_second_period_deterministic(wealth_1)
    policy_1 = policy_second_period_deterministic(wealth_1)

    policy_0_renamed = {f"{k}_0": v for k, v in policy_0.items()}
    policy_1_renamed = {f"{k}_1": v for k, v in policy_1.items()}

    # Transform data into format as expected by LCM
    # ==================================================================================
    data = (
        {
            "subject_id": jnp.arange(len(initial_wealth)),
            "wealth_0": initial_wealth,
            "wealth_1": wealth_1,
            "value_0": V_arr_0,
            "value_1": V_arr_1,
        }
        | policy_0_renamed
        | policy_1_renamed
    )

    raw = pd.DataFrame(data)
    raw_long = pd.wide_to_long(
        raw,
        stubnames=["value", "wealth", "consumption", "working"],
        i="subject_id",
        j="period",
        sep="_",
    )

    return raw_long.swaplevel().sort_index().reset_index()


def matrix_to_dict_of_vectors(arr, col_names):
    """Transform a matrix into a dict of vectors."""
    if arr.ndim != 2:
        raise ValueError("arr must be a two-dimensional array (matrix).")
    return dict(zip(col_names, arr.transpose(), strict=True))


def dict_of_vectors_to_matrix(d):
    """Transform a dict of vectors into a matrix."""
    return np.column_stack(list(d.values()))


# ======================================================================================
# Analytical solution and simulation (stochastic model)
# ======================================================================================
def value_second_period_stochastic(wealth, health):
    """Value function in the second (last) period. Computed using pen and paper."""
    consumption = np.minimum(1, np.floor(wealth))
    return np.log(1 + consumption) * health


def policy_second_period_stochastic(wealth, health):
    """Policy function in the second (last) period. Computed using pen and paper.

    First column corresponds to consumption choice, second to working choice.

    """
    policy = np.column_stack(
        (np.minimum(1, np.floor(wealth)) * health, np.zeros_like(wealth)),
    ).astype(int)
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "working"])


def value_first_period_stochastic(wealth, health, params):
    """Value function in the first period. Computed using pen and paper."""
    health_transition = params["next_health"]["health_transition"]

    index = (wealth < 1).astype(int)  # map wealth to indices 0 and 1

    _values = np.array(
        [
            params["discount_factor"] * health_transition[0, 1] * np.log(2),
            np.maximum(
                0, params["discount_factor"] * health_transition[0, 1] * np.log(2) - 0.5
            ),
        ],
    )
    value_health_0 = _values[index]

    new_discount_factor = (
        params["discount_factor"] * params["next_health"]["health_transition"][1, 1]
    )
    value_health_1 = value_first_period_deterministic(
        wealth, params={"discount_factor": new_discount_factor}
    )

    # Combined
    return np.where(health, value_health_1, value_health_0)


def policy_first_period_stochastic(wealth, health, params):
    """Policy function in the first period. Computed using pen and paper."""
    health_transition = params["next_health"]["health_transition"]

    index = (wealth < 1).astype(int)  # map wealth to indices 0 and 1
    _policies = np.array(
        [
            [0, 0],
            [
                0,
                np.argmax(
                    (
                        0,
                        params["discount_factor"] * health_transition[0, 1] * np.log(2)
                        - 0.5,
                    ),
                ),
            ],
        ],
    )
    policy_health_0 = _policies[index]

    new_discount_factor = (
        params["discount_factor"] * params["next_health"]["health_transition"][1, 1]
    )
    _policy_health_1 = policy_first_period_deterministic(
        wealth,
        params={"discount_factor": new_discount_factor},
    )

    policy_health_1 = dict_of_vectors_to_matrix(_policy_health_1)

    _health = health.reshape(-1, 1)
    policies = _health * policy_health_1 + (1 - _health) * policy_health_0
    return matrix_to_dict_of_vectors(policies, col_names=["consumption", "working"])


def analytical_solve_stochastic(wealth_grid, health_grid, params):
    V_arr_0 = value_first_period_stochastic(
        wealth=wealth_grid,
        health=health_grid,
        params=params,
    )
    V_arr_1 = value_second_period_stochastic(wealth=wealth_grid, health=health_grid)
    return [V_arr_0, V_arr_1]


def analytical_simulate_stochastic(initial_wealth, initial_health, health_1, params):
    # Simulate
    # ==================================================================================
    V_arr_0 = value_first_period_stochastic(
        initial_wealth,
        initial_health,
        params=params,
    )
    policy_0 = policy_first_period_stochastic(
        initial_wealth,
        initial_health,
        params=params,
    )

    wealth_1 = next_wealth(initial_wealth, **policy_0)

    V_arr_1 = value_second_period_stochastic(wealth_1, health_1)
    policy_1 = policy_second_period_stochastic(wealth_1, health_1)

    policy_0_renamed = {f"{k}_0": v for k, v in policy_0.items()}
    policy_1_renamed = {f"{k}_1": v for k, v in policy_1.items()}

    # Transform data into format as expected by LCM
    # ==================================================================================
    data = {
        "subject_id": jnp.arange(len(initial_wealth)),
        "wealth_0": initial_wealth,
        "wealth_1": wealth_1,
        "health_0": initial_health,
        "health_1": health_1,
        "value_0": V_arr_0,
        "value_1": V_arr_1,
        **policy_0_renamed,
        **policy_1_renamed,
    }

    raw = pd.DataFrame(data)
    raw_long = pd.wide_to_long(
        raw,
        stubnames=["value", "consumption", "working", "wealth", "health"],
        i="subject_id",
        j="period",
        sep="_",
    )

    return raw_long.swaplevel().sort_index().reset_index()


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
def test_deterministic_solve(discount_factor, n_wealth_points):
    # Update model
    # ==================================================================================
    n_periods = 3
    new_states = alive_deterministic.states
    new_states["wealth"] = new_states["wealth"].replace(n_points=n_wealth_points)  # type: ignore[attr-defined]
    model = Model(
        [
            alive_deterministic.replace(states=new_states, active=range(n_periods - 1)),
            dead.replace(active=[n_periods - 1]),
        ],
        regime_id_cls=RegimeId,
        n_periods=n_periods,
    )

    # Solve model using LCM
    # ==================================================================================
    params_alive = {
        "discount_factor": discount_factor,
        "utility": {"health": 1},
        "next_regime": {"n_periods": model.n_periods},
    }
    got = model.solve(params={"alive": params_alive, "dead": {}})

    # Compute analytical solution
    # ==================================================================================
    wealth_grid_class: LinspaceGrid = new_states["wealth"]  # type: ignore[assignment]
    wealth_grid = np.linspace(
        start=wealth_grid_class.start,
        stop=wealth_grid_class.stop,
        num=wealth_grid_class.n_points,
    )
    expected = analytical_solve_deterministic(wealth_grid, params=params_alive)

    # Do not assert that in the first period, the arrays have the same values on the
    # first and last index: TODO (@timmens): THIS IS A BUG AND NEEDS TO BE INVESTIGATED.
    # ==================================================================================
    aaae(got[0]["alive"][slice(1, -1)], expected[0][slice(1, -1)], decimal=12)
    aaae(got[1]["alive"], expected[1], decimal=12)


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
def test_deterministic_simulate(discount_factor, n_wealth_points):
    # Update model
    # ==================================================================================
    n_periods = 3
    new_states = alive_deterministic.states
    new_states["wealth"] = new_states["wealth"].replace(n_points=n_wealth_points)  # type: ignore[attr-defined]
    model = Model(
        [
            alive_deterministic.replace(states=new_states, active=range(n_periods - 1)),
            dead.replace(active=[n_periods - 1]),
        ],
        n_periods=n_periods,
        regime_id_cls=RegimeId,
    )

    # Simulate model using LCM
    # ==================================================================================
    params_alive = {
        "discount_factor": discount_factor,
        "utility": {"health": 1},
        "next_regime": {"n_periods": model.n_periods},
    }
    got: dict[str, pd.DataFrame] = model.solve_and_simulate(
        params={"alive": params_alive, "dead": {}},
        initial_states={"wealth": jnp.array([0.25, 0.75, 1.25, 1.75])},
        initial_regimes=["alive"] * 4,
    )

    # Compute analytical simulation
    # ==================================================================================
    expected = analytical_simulate_deterministic(
        initial_wealth=np.array([0.25, 0.75, 1.25, 1.75]),
        params=params_alive,
    )

    assert_frame_equal(got["alive"], expected, check_like=True, check_dtype=False)


HEALTH_TRANSITION = [
    jnp.array([[0.9, 0.1], [0.2, 0.8]]),
    jnp.array([[0.9, 0.1], [0, 1]]),
    jnp.array([[0.5, 0.5], [0.2, 0.8]]),
]


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
@pytest.mark.parametrize("health_transition", HEALTH_TRANSITION)
def test_stochastic_solve(discount_factor, n_wealth_points, health_transition):
    # Update model
    # ==================================================================================
    n_periods = 3
    new_states = alive_stochastic.states
    new_states["wealth"] = new_states["wealth"].replace(n_points=n_wealth_points)  # type: ignore[attr-defined]
    model = Model(
        [
            alive_stochastic.replace(states=new_states, active=range(n_periods - 1)),
            dead.replace(active=[n_periods - 1]),
        ],
        regime_id_cls=RegimeId,
        n_periods=n_periods,
    )

    # Solve model using LCM
    # ==================================================================================
    params = {
        "discount_factor": discount_factor,
        "next_health": {"health_transition": health_transition},
        "next_regime": {"n_periods": model.n_periods},
    }
    got = model.solve(params={"alive": params, "dead": {}})

    # Compute analytical solution
    # ==================================================================================
    wealth_grid_class: LinspaceGrid = new_states["wealth"]  # type: ignore[assignment]
    _wealth_grid = np.linspace(
        start=wealth_grid_class.start,
        stop=wealth_grid_class.stop,
        num=wealth_grid_class.n_points,
    )
    _health_grid = np.array([0, 1])

    # Repeat arrays to evaluate on all combinations of wealth and health
    wealth_grid = np.tile(_wealth_grid, len(_health_grid))
    health_grid = np.repeat(_health_grid, len(_wealth_grid))

    _expected = analytical_solve_stochastic(
        wealth_grid=wealth_grid,
        health_grid=health_grid,
        params=params,
    )
    expected = [
        arr.reshape((len(_health_grid), len(_wealth_grid))) for arr in _expected
    ]

    # Do not assert that in the first period, the arrays have the same values on the
    # first and last index: TODO (@timmens): THIS IS A BUG AND NEEDS TO BE INVESTIGATED.
    # ==================================================================================
    aaae(got[0]["alive"][:, slice(1, -1)], expected[0][:, slice(1, -1)], decimal=12)
    aaae(got[1]["alive"], expected[1], decimal=12)


@pytest.mark.parametrize("discount_factor", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
@pytest.mark.parametrize("health_transition", HEALTH_TRANSITION)
def test_stochastic_simulate(discount_factor, n_wealth_points, health_transition):
    # Update model
    # ==================================================================================
    n_periods = 3
    new_states = alive_stochastic.states
    new_states["wealth"] = new_states["wealth"].replace(n_points=n_wealth_points)  # type: ignore[attr-defined]
    model = Model(
        [
            alive_stochastic.replace(states=new_states, active=range(n_periods - 1)),
            dead.replace(active=[n_periods - 1]),
        ],
        n_periods=n_periods,
        regime_id_cls=RegimeId,
    )

    # Simulate model using LCM
    # ==================================================================================
    params_alive = {
        "discount_factor": discount_factor,
        "next_health": {"health_transition": health_transition},
        "next_regime": {"n_periods": model.n_periods},
    }
    initial_states = {
        "wealth": jnp.array([0.25, 0.75, 1.25, 1.75, 2.0]),
        "health": jnp.array([0, 1, 0, 1, 1]),
    }
    _got: pd.DataFrame = model.solve_and_simulate(
        params={"alive": params_alive, "dead": {}},
        initial_states=initial_states,
        initial_regimes=["alive"] * 5,
    )["alive"]

    # Compute analytical simulation
    # ==================================================================================
    # Need to use health of second period from LCM output, to assure that the same
    # stochastic draws are used in the analytical simulation.
    health_1 = _got.query("period == 1")["health"].to_numpy()

    _expected = analytical_simulate_stochastic(
        initial_wealth=initial_states["wealth"],
        initial_health=initial_states["health"],
        health_1=health_1,
        params=params_alive,
    )

    # Drop all rows that contain wealth levels at the boundary.
    got = _got.query("wealth != 2")
    expected = _expected.query("wealth != 2")
    assert_frame_equal(got, expected, check_like=True, check_dtype=False)
