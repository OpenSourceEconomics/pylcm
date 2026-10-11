"""The policy lookup's action values and their parts, from its own decision program.

`Q`, `F`, `U` and `CE` cover the queried (possibly restricted) action grid with one
leading row per queried state, one axis per action in the order of `actions`. `Q`
holds the action value as the decision program computes it, also where the action
is infeasible; `F` says which entries the maximization admits. `U` is the flow
utility and `CE` the undiscounted certainty-equivalent continuation that the
regime's Koopmans aggregator combines into `Q`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    PowerMean,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.typing import BoolND, ContinuousAction, FloatND
from lcm_examples import epstein_zin
from lcm_examples.mortality import LaborSupply, utility_working
from tests.conftest import DECIMAL_PRECISION
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


@pytest.mark.parametrize("period", [1, N_PERIODS - 1])
def test_action_values_of_a_regime_without_states_or_actions_have_one_row(
    *, solved, period
):
    """`dead` has no states and no actions: its empty query is one feasible row.

    Its utility is zero and it has no continuation, so `Q` is `[0.0]` and `F` is
    `[True]`, each with the one leading row of the lookup.
    """
    model, params, solution = solved
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=period,
        regime_name="dead",
        states={},
        return_action_values=True,
    )
    assert (
        got.Q.shape,
        np.asarray(got.Q).tolist(),
        got.F.shape,
        np.asarray(got.F).tolist(),
    ) == ((1,), [0.0], (1,), [True])


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


DISCOUNT_FACTOR = 0.95
DISUTILITY_OF_WORK = 0.5


def test_flow_utility_and_continuation_are_absent_unless_requested(solved):
    got = lookup(solved=solved, period=0)
    assert (got.U, got.CE) == (None, None)


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_flow_utility_and_continuation_have_the_shape_of_the_action_values(
    *, solved, period
):
    got = lookup(solved=solved, period=period, return_action_values=True)
    assert (got.U.shape, got.CE.shape) == (got.Q.shape, got.Q.shape)


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_linear_action_value_is_flow_utility_plus_discounted_continuation(
    *, solved, period
):
    """Under `LinearAggregator`, `Q == U + discount_factor * CE` at every entry.

    Extrapolated continuations reach magnitudes in the hundreds, so the check is
    relative.
    """
    got = lookup(solved=solved, period=period, return_action_values=True)
    np.testing.assert_allclose(
        np.asarray(got.Q),
        np.asarray(got.U) + DISCOUNT_FACTOR * np.asarray(got.CE),
        rtol=10.0**-DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_flow_utility_is_the_utility_function_on_the_action_grid(*, solved, period):
    """`U` is `log(consumption) - disutility_of_work * working`, in every row.

    The utility of `working_life` reads no state, so every queried row holds the
    same table over `(labor_supply, consumption)`.
    """
    got = lookup(solved=solved, period=period, return_action_values=True)
    working = ACTION_GRIDS["labor_supply"] == LaborSupply.work
    table = (
        np.log(ACTION_GRIDS["consumption"])[None, :]
        - DISUTILITY_OF_WORK * working[:, None]
    )
    np.testing.assert_array_almost_equal(
        np.asarray(got.U),
        np.broadcast_to(table, got.U.shape),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [1, N_PERIODS - 1])
def test_terminal_regime_without_states_has_zero_continuation(*, solved, period):
    """`dead` is terminal: `Q` is its flow utility and its continuation is zero."""
    model, params, solution = solved
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=period,
        regime_name="dead",
        states={},
        return_action_values=True,
    )
    assert (np.asarray(got.U).tolist(), np.asarray(got.CE).tolist()) == (
        [0.0],
        [0.0],
    )


def test_infeasible_actions_keep_a_finite_flow_utility_and_continuation(solved):
    """Where `Q` is finite, so are `U` and `CE`, feasible or not."""
    states = {"wealth": jnp.array([5.0, 20.0, 40.0, 70.0])}
    got = lookup(
        solved=solved,
        period=0,
        states=states,
        return_action_values=True,
    )
    Q, F = np.asarray(got.Q), np.asarray(got.F)
    finite = np.isfinite(Q)
    assert (~F & finite).any()
    assert (np.isfinite(np.asarray(got.U)) & np.isfinite(np.asarray(got.CE)))[
        finite
    ].all()


@pytest.mark.parametrize("branch", [LaborSupply.work, LaborSupply.retire])
def test_restricted_action_grid_restricts_flow_utility_and_continuation(
    *, solved, branch
):
    """A one-code labor grid returns that code's slice of the full `U` and `CE`.

    The restricted lookup compiles its own program, so the slice agrees to the
    working precision, relative to continuations that reach the hundreds.
    """
    full = lookup(solved=solved, period=0, return_action_values=True)
    restricted = lookup(
        solved=solved,
        period=0,
        action_grids={"labor_supply": jnp.array([branch], dtype=jnp.int32)},
        return_action_values=True,
    )
    labor_axis = 1 + list(full.actions).index("labor_supply")
    position = int(np.flatnonzero(ACTION_GRIDS["labor_supply"] == branch)[0])
    expected = np.stack(
        [
            np.take(np.asarray(full.U), [position], axis=labor_axis),
            np.take(np.asarray(full.CE), [position], axis=labor_axis),
        ]
    )
    np.testing.assert_allclose(
        np.stack([np.asarray(restricted.U), np.asarray(restricted.CE)]),
        expected,
        rtol=10.0**-DECIMAL_PRECISION,
    )


EZ_N_PERIODS = 4
EZ_DISCOUNT_FACTOR = 0.9
EZ_ELASTICITY = 2.0
EZ_HEALTH_COST = 0.25
EZ_RISK_AVERSION = 3.0


@pytest.fixture(scope="module", params=["linear", "power_mean"])
def solved_epstein_zin(request):
    """The Epstein-Zin example, once with each certainty equivalent."""
    risk_aversion = None if request.param == "linear" else EZ_RISK_AVERSION
    model = epstein_zin.get_model(
        certainty_equivalent=None if risk_aversion is None else PowerMean(),
        n_periods=EZ_N_PERIODS,
    )
    params = epstein_zin.get_params(
        risk_aversion=risk_aversion,
        discount_factor=EZ_DISCOUNT_FACTOR,
        intertemporal_elasticity_of_substitution=EZ_ELASTICITY,
        health_cost=EZ_HEALTH_COST,
    )
    solution = model.solve(params=params, log_level="off")
    return model, params, solution, risk_aversion


def lookup_alive_on_the_grid(*, solved_epstein_zin, period):
    """Look up `alive` at every `(wealth, health)` node, with its action values."""
    model, params, solution, _ = solved_epstein_zin
    wealth = np.asarray(
        model.state_grid(params=params, regime_name="alive", state_name="wealth")
    )
    health = np.array(
        [epstein_zin.HealthStatus.bad, epstein_zin.HealthStatus.good], dtype=np.int32
    )
    wealth_rows, health_rows = (x.ravel() for x in np.meshgrid(wealth, health))
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=period,
        regime_name="alive",
        states={
            "wealth": jnp.asarray(wealth_rows),
            "health": jnp.asarray(health_rows),
        },
        return_action_values=True,
    )
    return got, wealth_rows, health_rows


def epstein_zin_continuation_oracle(
    *, solved_epstein_zin, period, wealth_rows, health_rows
):
    """The certainty equivalent of next period's value, computed by hand.

    The joint lottery has one node per next health code under survival, weighted
    by the survival probability times the health transition, and one `dead` node
    weighted by the death probability. Next wealth stays inside both wealth grids,
    so each node's value is a plain linear interpolation of the solved values.
    """
    model, params, solution, risk_aversion = solved_epstein_zin
    consumption = np.asarray(epstein_zin.CONSUMPTION_GRID.to_jax())
    alive_wealth = np.asarray(
        model.state_grid(params=params, regime_name="alive", state_name="wealth")
    )
    dead_wealth = np.asarray(
        model.state_grid(params=params, regime_name="dead", state_name="wealth")
    )
    alive_axes = model.state_names(regime_name="alive")
    V_alive = np.moveaxis(
        np.asarray(solution.values[period + 1]["alive"]),
        [alive_axes.index("wealth"), alive_axes.index("health")],
        [0, 1],
    )
    V_dead = np.asarray(solution.values[period + 1]["dead"])
    health_transition = np.asarray(epstein_zin.HEALTH_TRANSITION)

    def power(*, values, weights):
        if risk_aversion is None:
            return (weights * values).sum(axis=-1)
        exponent = 1.0 - risk_aversion
        return ((weights * values**exponent).sum(axis=-1)) ** (1.0 / exponent)

    expected = np.empty((len(wealth_rows), len(consumption)))
    for row, (wealth, health) in enumerate(zip(wealth_rows, health_rows, strict=True)):
        is_bad = health == epstein_zin.HealthStatus.bad
        survival = epstein_zin.SURVIVAL_PROBS[period] * (
            epstein_zin.BAD_HEALTH_SURVIVAL_FACTOR if is_bad else 1.0
        )
        for column, choice in enumerate(consumption):
            next_wealth = np.clip(
                wealth
                - choice
                + epstein_zin.INCOME
                - (EZ_HEALTH_COST if is_bad else 0.0),
                epstein_zin.WEALTH_GRID.start,
                epstein_zin.WEALTH_GRID.stop,
            )
            values = np.array(
                [
                    np.interp(next_wealth, alive_wealth, V_alive[:, next_health])
                    for next_health in (0, 1)
                ]
                + [np.interp(next_wealth, dead_wealth, V_dead)]
            )
            weights = np.array(
                [survival * p for p in health_transition[int(health)]]
                + [1.0 - survival]
            )
            expected[row, column] = power(values=values, weights=weights)
    return expected


@pytest.mark.parametrize("period", [0, EZ_N_PERIODS - 3])
def test_continuation_is_the_certainty_equivalent_of_next_values(
    *, solved_epstein_zin, period
):
    got, wealth_rows, health_rows = lookup_alive_on_the_grid(
        solved_epstein_zin=solved_epstein_zin, period=period
    )
    expected = epstein_zin_continuation_oracle(
        solved_epstein_zin=solved_epstein_zin,
        period=period,
        wealth_rows=wealth_rows,
        health_rows=health_rows,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(got.CE), expected, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("period", [0, EZ_N_PERIODS - 3])
def test_ces_action_value_aggregates_flow_utility_and_continuation(
    *, solved_epstein_zin, period
):
    """`Q == ((1 - b) * U**r + b * CE**r) ** (1 / r)` with `r = 1 - 1 / psi`."""
    got, _, _ = lookup_alive_on_the_grid(
        solved_epstein_zin=solved_epstein_zin, period=period
    )
    U, CE = np.asarray(got.U), np.asarray(got.CE)
    exponent = 1.0 - 1.0 / EZ_ELASTICITY
    expected = (
        (1.0 - EZ_DISCOUNT_FACTOR) * U**exponent + EZ_DISCOUNT_FACTOR * CE**exponent
    ) ** (1.0 / exponent)
    np.testing.assert_array_almost_equal(
        np.asarray(got.Q), expected, decimal=DECIMAL_PRECISION
    )


def test_terminal_regime_with_states_has_its_utility_as_action_value(
    solved_epstein_zin,
):
    """`dead` reads wealth: `U` is the bequest, `CE` is zero and `Q` equals `U`."""
    model, params, solution, _ = solved_epstein_zin
    wealth = np.asarray(
        model.state_grid(params=params, regime_name="dead", state_name="wealth")
    )
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=EZ_N_PERIODS - 1,
        regime_name="dead",
        states={"wealth": jnp.asarray(wealth)},
        return_action_values=True,
    )
    np.testing.assert_array_almost_equal(
        np.stack([np.asarray(got.U), np.asarray(got.Q), np.asarray(got.CE)]),
        np.stack([np.sqrt(wealth), np.sqrt(wealth), np.zeros_like(wealth)]),
        decimal=DECIMAL_PRECISION,
    )


def test_action_value_lookup_evaluates_the_regime_kernel_once():
    """`Q`, `F`, `U`, `CE` and the decision come from one kernel evaluation.

    The utility records each time it is traced; compiling the lookup program
    with its action values traces it once.
    """
    traces = []

    def counting_utility(
        *, consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
    ) -> FloatND:
        traces.append(None)
        return utility_working(
            consumption=consumption,
            is_working=is_working,
            disutility_of_work=disutility_of_work,
        )

    ages = AgeGrid(start=START_AGE, inclusive_stop=START_AGE + 2, step="Y")
    model = Model(
        regimes={
            "working_life": working_life.replace(
                functions={**working_life.functions, "utility": counting_utility}
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={START_AGE: "working_life"},
        edges=working_life_edges(ages),
    )
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="off")
    traces.clear()
    model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working_life",
        states={"wealth": jnp.array([5.0, 20.0])},
        return_action_values=True,
    )
    assert len(traces) == 1
