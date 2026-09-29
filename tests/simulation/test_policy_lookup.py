"""The public per-period policy lookup on a solved model."""

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from lcm import PolicyLookup
from lcm.exceptions import InvalidSimulationInputError
from lcm_examples.mortality import LaborSupply
from tests.test_models.deterministic.discrete import get_model as get_discrete_model
from tests.test_models.deterministic.discrete import (
    get_params as get_discrete_params,
)
from tests.test_models.deterministic.regression import (
    DEFAULT_CONSUMPTION_GRID,
    RegimeId,
    get_model,
    get_params,
)
from tests.test_models.nbegm_multi_discrete_toy import build_model as build_shock_model
from tests.test_models.nbegm_multi_discrete_toy import (
    build_params as build_shock_params,
)

N_PERIODS = 5
LAST_ALIVE_PERIOD = N_PERIODS - 2
WEALTH = jnp.array([5.0, 20.0, 40.0, 70.0])


@pytest.fixture(scope="module")
def solved():
    model = get_model(n_periods=N_PERIODS)
    params = get_params(n_periods=N_PERIODS)
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def _lookup(solved, **kwargs):
    model, params, solution = solved
    return model.lookup_policy(
        params=params,
        solution=solution,
        **{"regime_name": "working_life", **kwargs},
    )


def test_lookup_policy_returns_a_policy_lookup(solved):
    got = _lookup(solved, period=0, states={"wealth": WEALTH})
    assert isinstance(got, PolicyLookup)


def test_lookup_policy_last_alive_period_consumes_largest_feasible_grid_point(solved):
    """With a zero continuation, retiring and eating all feasible wealth is best."""
    got = _lookup(solved, period=LAST_ALIVE_PERIOD, states={"wealth": WEALTH})
    grid = np.asarray(DEFAULT_CONSUMPTION_GRID.to_jax())
    expected_c = np.array([grid[grid <= w].max() for w in np.asarray(WEALTH)])
    np.testing.assert_allclose(got.actions["consumption"], expected_c, rtol=1e-6)


def test_lookup_policy_last_alive_period_value_is_log_consumption(solved):
    got = _lookup(solved, period=LAST_ALIVE_PERIOD, states={"wealth": WEALTH})
    grid = np.asarray(DEFAULT_CONSUMPTION_GRID.to_jax())
    expected_v = np.log([grid[grid <= w].max() for w in np.asarray(WEALTH)])
    np.testing.assert_allclose(got.value, expected_v, rtol=1e-5)


def test_lookup_policy_last_alive_period_retires(solved):
    got = _lookup(solved, period=LAST_ALIVE_PERIOD, states={"wealth": WEALTH})
    np.testing.assert_array_equal(got.actions["labor_supply"], LaborSupply.retire)


@pytest.mark.parametrize("column", ["consumption", "labor_supply", "value"])
def test_lookup_policy_equals_what_simulate_records(*, solved, column):
    """At every simulated (period, subject), the lookup reproduces simulate's row."""
    model, params, solution = solved
    df = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": WEALTH,
            "age": jnp.full(WEALTH.shape, 18.0),
            "regime_id": jnp.full(WEALTH.shape, RegimeId.working_life),
        },
        log_level="off",
        seed=0,
    ).to_dataframe(use_labels=False)
    alive = df.query("regime_name == 'working_life'")
    got, expected = [], []
    for period, rows in alive.groupby("period"):
        lookup = _lookup(
            solved,
            period=int(period),
            states={"wealth": jnp.asarray(rows["wealth"].to_numpy())},
        )
        values = lookup.value if column == "value" else lookup.actions[column]
        got.append(np.asarray(values, dtype=float))
        expected.append(rows[column].to_numpy(dtype=float))
    np.testing.assert_allclose(np.concatenate(got), np.concatenate(expected))


@pytest.mark.parametrize("branch", [LaborSupply.work, LaborSupply.retire])
def test_lookup_policy_restricted_action_grid_fixes_the_branch(*, solved, branch):
    got = _lookup(
        solved,
        period=0,
        states={"wealth": WEALTH},
        action_grids={"labor_supply": jnp.array([branch], dtype=jnp.int32)},
    )
    np.testing.assert_array_equal(got.actions["labor_supply"], branch)


def test_lookup_policy_max_over_branches_equals_unconditional_value(solved):
    branch_values = [
        _lookup(
            solved,
            period=0,
            states={"wealth": WEALTH},
            action_grids={"labor_supply": jnp.array([code], dtype=jnp.int32)},
        ).value
        for code in (LaborSupply.work, LaborSupply.retire)
    ]
    unconditional = _lookup(solved, period=0, states={"wealth": WEALTH}).value
    np.testing.assert_allclose(np.maximum(*branch_values), unconditional)


def test_lookup_policy_rejects_an_unknown_regime(solved):
    with pytest.raises(InvalidSimulationInputError, match="retired"):
        _lookup(solved, regime_name="retired", period=0, states={"wealth": WEALTH})


def test_lookup_policy_rejects_a_period_the_regime_is_not_active_in(solved):
    with pytest.raises(InvalidSimulationInputError, match="period 4"):
        _lookup(solved, period=N_PERIODS - 1, states={"wealth": WEALTH})


def test_lookup_policy_rejects_missing_state(solved):
    with pytest.raises(InvalidSimulationInputError, match="wealth"):
        _lookup(solved, period=0, states={})


def test_lookup_policy_rejects_a_discrete_state_code_off_the_grid():
    model = get_discrete_model(n_periods=4)
    params = get_discrete_params(n_periods=4)
    solution = model.solve(params=params, log_level="off")
    with pytest.raises(InvalidSimulationInputError, match="wealth"):
        model.lookup_policy(
            params=params,
            solution=solution,
            period=0,
            regime_name="working_life",
            states={"wealth": jnp.array([0, 7], dtype=jnp.int32)},
        )


def test_lookup_policy_rejects_an_unknown_action_grid(solved):
    with pytest.raises(InvalidSimulationInputError, match="leisure"):
        _lookup(
            solved,
            period=0,
            states={"wealth": WEALTH},
            action_grids={"leisure": jnp.array([0])},
        )


def test_lookup_policy_rejects_params_the_solution_was_not_solved_with(solved):
    model, _, solution = solved
    with pytest.raises(InvalidSimulationInputError, match="params_fingerprint"):
        model.lookup_policy(
            params=get_params(n_periods=N_PERIODS, discount_factor=0.5),
            solution=solution,
            period=0,
            regime_name="working_life",
            states={"wealth": WEALTH},
        )


def test_lookup_policy_above_range_continuous_state_equals_simulate(solved):
    """A continuous state above its grid is extrapolated as simulate does."""
    model, params, solution = solved
    wealth = jnp.array([450.0, 600.0])
    df = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": wealth,
            "age": jnp.full(wealth.shape, 18.0),
            "regime_id": jnp.full(wealth.shape, RegimeId.working_life),
        },
        log_level="off",
        seed=0,
    ).to_dataframe(use_labels=False)
    first = df.query("period == 0")
    got = _lookup(solved, period=0, states={"wealth": wealth})
    np.testing.assert_allclose(
        np.column_stack([got.actions["consumption"], got.value]),
        first[["consumption", "value"]].to_numpy(dtype=float),
    )


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_state_grid_scales_shock_nodes_with_runtime_sigma(sigma):
    """Gauss-Hermite income nodes are the unit-sigma nodes times the runtime sigma."""
    model = build_shock_model()
    params = build_shock_params()
    unit = model.state_grid(params=params, regime_name="alive", state_name="income")
    params["alive"]["income"] = {"mu": 0.0, "sigma": sigma}
    got = model.state_grid(params=params, regime_name="alive", state_name="income")
    np.testing.assert_allclose(got, sigma * np.asarray(unit), rtol=1e-6)


def test_state_grid_rejects_an_unknown_state(solved):
    model, params, _ = solved
    with pytest.raises(InvalidSimulationInputError, match="income"):
        model.state_grid(params=params, regime_name="working_life", state_name="income")


def test_solution_result_is_exported_from_lcm(solved):
    _, _, solution = solved
    assert isinstance(solution, lcm.SolutionResult)


def test_state_names_is_the_value_function_axis_order(solved):
    model, _, _ = solved
    assert model.state_names(regime_name="working_life") == ("wealth",)


def test_value_array_indexed_in_state_names_order_matches_lookup(solved):
    """V at grid node k of each state, in `state_names` order, is the lookup's V."""
    model, params, solution = solved
    nodes = {
        name: model.state_grid(
            params=params, regime_name="working_life", state_name=name
        )
        for name in model.state_names(regime_name="working_life")
    }
    index = (3,)
    V = solution.values[0]["working_life"]
    got = _lookup(
        solved,
        period=0,
        states={
            name: nodes[name][i : i + 1] for name, i in zip(nodes, index, strict=True)
        },
    )
    np.testing.assert_allclose(got.value, V[index], rtol=1e-6)


@pytest.fixture(scope="module")
def solved_two_states():
    model = build_shock_model()
    params = build_shock_params()
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def _nodes_in_state_names_order(*, model, params):
    return {
        name: model.state_grid(params=params, regime_name="alive", state_name=name)
        for name in model.state_names(regime_name="alive")
    }


def test_value_array_shape_follows_state_names_order(solved_two_states):
    """Two states on grids of different sizes: V's shape is their sizes in order."""
    model, params, solution = solved_two_states
    nodes = _nodes_in_state_names_order(model=model, params=params)
    shape = tuple(len(grid) for grid in nodes.values())
    assert solution.values[0]["alive"].shape == shape != shape[::-1]


def test_value_array_indexed_in_state_names_order_matches_lookup_two_states(
    solved_two_states,
):
    model, params, solution = solved_two_states
    nodes = _nodes_in_state_names_order(model=model, params=params)
    index = (2, 1)
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={
            name: nodes[name][i : i + 1] for name, i in zip(nodes, index, strict=True)
        },
    )
    V = solution.values[0]["alive"]
    np.testing.assert_allclose(got.value, V[index], rtol=1e-5)
    assert not np.isclose(V[index], V[index[::-1]], rtol=1e-5)
