"""The policy lookup's action values and feasibility, from its own decision program.

`Q` and `F` cover the queried (possibly restricted) action grid with one leading
row per queried state, one axis per action in the order of `actions`. `Q` holds
the action value as the decision program computes it, also where the action is
infeasible; `F` says which entries the maximization admits.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.exceptions import InvalidSimulationInputError
from lcm_examples.mortality import LaborSupply
from tests.regime_building.test_nonterminal_collective_solve import (
    RegimeId as CoupleRegimeId,
)
from tests.regime_building.test_nonterminal_collective_solve import (
    _make_couple_regimes,
)
from tests.test_models.deterministic.regression import (
    DEFAULT_CONSUMPTION_GRID,
    DEFAULT_WEALTH_GRID,
    START_AGE,
    RegimeId,
    dead,
    get_model,
    get_params,
    working_life,
    working_life_edges,
)

N_PERIODS = 5
LAST_ALIVE_PERIOD = N_PERIODS - 2
# The declared action grids of `working_life`.
ACTION_GRIDS = {
    "labor_supply": np.asarray(DiscreteGrid(category_class=LaborSupply).to_jax()),
    "consumption": np.asarray(DEFAULT_CONSUMPTION_GRID.to_jax()),
}


@pytest.fixture(scope="module")
def solved():
    model = get_model(n_periods=N_PERIODS)
    params = get_params(n_periods=N_PERIODS)
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def lookup(*, solved, period, states=None, **kwargs):
    """Look up `working_life` at every wealth grid node unless `states` says else."""
    model, params, solution = solved
    if states is None:
        wealth = model.state_grid(
            params=params, regime_name="working_life", state_name="wealth"
        )
        states = {"wealth": jnp.asarray(wealth)}
    return model.lookup_policy(
        params=params,
        solution=solution,
        period=period,
        regime_name="working_life",
        states=states,
        **kwargs,
    )


def masked_max_and_argmax(*, Q, F):
    """Max of `Q` over the entries `F` admits, and its first flat argmax, per row."""
    masked = np.where(F, Q, -np.inf).reshape(Q.shape[0], -1)
    return masked.max(axis=1), masked.argmax(axis=1)


def test_action_values_are_absent_unless_requested(solved):
    got = lookup(solved=solved, period=0)
    assert (got.Q, got.F) == (None, None)


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_action_values_cover_the_action_grid_with_a_leading_row_axis(*, solved, period):
    got = lookup(solved=solved, period=period, return_action_values=True)
    n_rows = len(got.value)
    action_shape = tuple(len(ACTION_GRIDS[name]) for name in got.actions)
    assert (got.Q.shape, got.F.shape, got.F.dtype) == (
        (n_rows, *action_shape),
        (n_rows, *action_shape),
        np.dtype(bool),
    )


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_value_is_the_max_of_feasible_action_values(*, solved, period):
    got = lookup(solved=solved, period=period, return_action_values=True)
    expected, _ = masked_max_and_argmax(Q=np.asarray(got.Q), F=np.asarray(got.F))
    np.testing.assert_array_equal(np.asarray(got.value), expected)


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_actions_are_the_argmax_of_feasible_action_values(*, solved, period):
    got = lookup(solved=solved, period=period, return_action_values=True)
    _, flat = masked_max_and_argmax(Q=np.asarray(got.Q), F=np.asarray(got.F))
    indices = np.unravel_index(flat, got.Q.shape[1:])
    chosen = {
        name: ACTION_GRIDS[name][indices[axis]] for axis, name in enumerate(got.actions)
    }
    np.testing.assert_array_equal(
        np.stack([np.asarray(got.actions[name]) for name in got.actions]),
        np.stack(list(chosen.values())),
    )


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_requesting_action_values_leaves_actions_and_value_unchanged(*, solved, period):
    plain = lookup(solved=solved, period=period)
    with_values = lookup(solved=solved, period=period, return_action_values=True)
    np.testing.assert_array_equal(
        np.stack(
            [
                *(np.asarray(with_values.actions[n]) for n in plain.actions),
                np.asarray(with_values.value),
            ]
        ),
        np.stack(
            [
                *(np.asarray(plain.actions[n]) for n in plain.actions),
                np.asarray(plain.value),
            ]
        ),
    )


def test_some_actions_are_infeasible(solved):
    got = lookup(solved=solved, period=LAST_ALIVE_PERIOD, return_action_values=True)
    assert not np.asarray(got.F).all()


def test_infeasible_action_with_larger_raw_value_is_not_chosen(solved):
    """Over several rows, an infeasible action's larger raw `Q` is never the value.

    Eating more than one's wealth is infeasible and worth more utility, so at the
    last alive period an infeasible action holds a larger raw `Q` than the chosen
    one in a row whose neighbours differ.
    """
    states = {"wealth": jnp.array([5.0, 20.0, 40.0, 70.0])}
    got = lookup(
        solved=solved,
        period=LAST_ALIVE_PERIOD,
        states=states,
        return_action_values=True,
    )
    Q, F = np.asarray(got.Q), np.asarray(got.F)
    infeasible_max = np.where(F, -np.inf, Q).reshape(len(Q), -1).max(axis=1)
    assert (infeasible_max > np.asarray(got.value)).any()
    expected, _ = masked_max_and_argmax(Q=Q, F=F)
    np.testing.assert_array_equal(np.asarray(got.value), expected)


@pytest.mark.parametrize("branch", [LaborSupply.work, LaborSupply.retire])
def test_restricted_action_grid_restricts_the_action_values(*, solved, branch):
    got = lookup(
        solved=solved,
        period=0,
        action_grids={"labor_supply": jnp.array([branch], dtype=jnp.int32)},
        return_action_values=True,
    )
    labor_axis = list(got.actions).index("labor_supply")
    expected, _ = masked_max_and_argmax(Q=np.asarray(got.Q), F=np.asarray(got.F))
    assert got.Q.shape[1 + labor_axis] == 1
    np.testing.assert_array_equal(np.asarray(got.value), expected)


def test_action_values_are_refused_for_a_collective_regime():
    """A collective regime's household argmax has no single action value to report."""
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    model = Model(
        regimes=_make_couple_regimes(),
        edges={"couple": {"couple_terminal": 0}},
        ages=ages,
        regime_id_class=CoupleRegimeId,
        initial_nodes={ages.exact_values[0]: "couple"},
    )
    params = {"discount_factor": 0.95}
    solution = model.solve(params=params, log_level="off")
    with pytest.raises(InvalidSimulationInputError, match="collective"):
        model.lookup_policy(
            params=params,
            solution=solution,
            period=0,
            regime_name="couple",
            states={"wage": jnp.array([8.0, 40.0])},
            return_action_values=True,
        )


def test_budgeted_lookup_returns_the_feasibility_of_each_action():
    """Under a device budget, consuming more than one's wealth is infeasible.

    At wealth 2 with consumption grid {1, 2, 3}, only consumption 3 is
    infeasible, for either labor supply choice.
    """
    grid = LinSpacedGrid(start=1, stop=3, n_points=3)
    model = get_model(
        n_periods=3,
        wealth_grid=grid,
        consumption_grid=grid,
        execution_config=ExecutionConfig(device_memory_bytes=2**30),
    )
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="off")
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=1,
        regime_name="working_life",
        states={"wealth": jnp.array([2.0])},
        return_action_values=True,
    )
    assert tuple(got.actions) == ("labor_supply", "consumption")
    np.testing.assert_array_equal(
        np.asarray(got.F), np.array([[[True, True, False], [True, True, False]]])
    )


def test_action_values_bind_the_models_fixed_params(solved):
    """Parameters fixed at model build reach the action values like the decision.

    Fixing `disutility_of_work` at its runtime value leaves `Q`, `F`, the
    actions and the value equal to those of the model that takes it at runtime.
    """
    _, params, _ = solved
    ages = AgeGrid(start=START_AGE, inclusive_stop=START_AGE + N_PERIODS - 1, step="Y")
    fixed_model = Model(
        regimes={
            "working_life": working_life.replace(
                states={"wealth": DEFAULT_WEALTH_GRID},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": DEFAULT_CONSUMPTION_GRID,
                },
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={START_AGE: "working_life"},
        edges=working_life_edges(ages),
        fixed_params={"working_life": {"utility": {"disutility_of_work": 0.5}}},
    )
    fixed_params = {
        **params,
        "working_life": {"next_wealth": params["working_life"]["next_wealth"]},
    }
    fixed_solution = fixed_model.solve(params=fixed_params, log_level="off")
    states = {"wealth": jnp.array([5.0, 20.0, 40.0, 70.0])}
    got = fixed_model.lookup_policy(
        params=fixed_params,
        solution=fixed_solution,
        period=LAST_ALIVE_PERIOD,
        regime_name="working_life",
        states=states,
        return_action_values=True,
    )
    expected = lookup(
        solved=solved,
        period=LAST_ALIVE_PERIOD,
        states=states,
        return_action_values=True,
    )
    np.testing.assert_array_equal(flatten_lookup(got), flatten_lookup(expected))


def flatten_lookup(got):
    """`Q`, `F` and the value of a lookup, raveled into one array."""
    return np.concatenate([np.asarray(x).ravel() for x in (got.Q, got.F, got.value)])
