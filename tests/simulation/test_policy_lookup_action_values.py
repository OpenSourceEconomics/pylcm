"""The policy lookup's action values and their parts, from its own decision program.

`Q`, `F`, `U` and `CE` cover the queried (possibly restricted) action grid with one
leading row per queried state, one axis per action in the order of `actions`. `Q`
holds the action value as the decision program computes it, also where the action
is infeasible; `F` says which entries the maximization admits. `U` is the flow
utility and `CE` the undiscounted certainty-equivalent continuation that the
regime's Koopmans aggregator combines into `Q`.
"""

import dataclasses
import gc
import itertools
import weakref

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest

import lcm.model
from _lcm.simulation.program_types import (
    ACTION_FEASIBILITY,
    ACTION_INDEX,
    ACTION_VALUES,
    CONTINUATION_CE,
    DECISION_VALUE,
    FLOW_UTILITY,
)
from _lcm.simulation.programs import with_kept_action_parts
from _lcm.simulation.runtime import SimulationRuntime
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    PowerMean,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.result import PolicyLookup
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    Intermediate,
    ScalarInt,
)
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
ALL_INTERMEDIATES: tuple[Intermediate, ...] = ("Q", "F", "U", "CE")
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


def test_intermediates_are_absent_unless_requested(solved):
    got = lookup(solved=solved, period=0)
    assert (got.Q, got.F, got.U, got.CE) == (None, None, None, None)


def test_no_intermediates_dispatch_the_decision_without_its_parts(
    *, solved, monkeypatch
):
    """Keeping nothing runs the plain decision, which may stream over actions."""

    def _refuse(**_kwargs):
        msg = "keeping no intermediates must not build a parts program"
        raise AssertionError(msg)

    expected = lookup(solved=solved, period=0, keep_intermediates=ALL_INTERMEDIATES)
    monkeypatch.setattr(lcm.model, "with_kept_action_parts", _refuse)
    got = lookup(solved=solved, period=0, keep_intermediates=())
    np.testing.assert_array_equal(np.asarray(got.value), np.asarray(expected.value))


@pytest.mark.parametrize(
    "keep_intermediates",
    [("Q",), ("F",), ("U",), ("CE",), ("CE", "Q"), ("F", "U", "CE")],
)
def test_lookup_keeps_exactly_the_requested_intermediates(
    *, solved, keep_intermediates
):
    """Each requested intermediate equals the full request's; the others are None."""
    full = lookup(solved=solved, period=0, keep_intermediates=ALL_INTERMEDIATES)
    got = lookup(solved=solved, period=0, keep_intermediates=keep_intermediates)
    np.testing.assert_array_equal(np.asarray(got.value), np.asarray(full.value))
    for name in ALL_INTERMEDIATES:
        if name in keep_intermediates:
            np.testing.assert_array_equal(
                np.asarray(getattr(got, name)), np.asarray(getattr(full, name))
            )
        else:
            assert getattr(got, name) is None


def test_kept_parts_program_publishes_only_the_kept_parts(solved):
    model, _, _ = solved
    program = model._regimes["working_life"].simulation.programs.action_values[0]
    kept = with_kept_action_parts(
        program=program, roles=frozenset({CONTINUATION_CE}), cache={}
    )
    assert kept.output_roles == (ACTION_INDEX, DECISION_VALUE, CONTINUATION_CE)


@pytest.mark.parametrize(
    ("keep_intermediates", "match"),
    [
        (("V",), "keep_intermediates"),
        (("Q", "V"), "keep_intermediates"),
        (("V", "Q"), "keep_intermediates"),
        (("Q", "F", "U", "CE", "V"), "keep_intermediates"),
        (("Q", "Q"), "at most once"),
    ],
)
def test_lookup_refuses_unknown_or_repeated_intermediates(
    *, solved, keep_intermediates, match
):
    with pytest.raises(InvalidSimulationInputError, match=match):
        lookup(solved=solved, period=0, keep_intermediates=keep_intermediates)


# The public type check samples one entry per call, so each input is refused
# many times to catch a call that reports a different error.
@pytest.mark.parametrize("keep_intermediates", [("Q", ["U"]), ("Q", 1)])
def test_lookup_refuses_non_string_intermediates_on_every_call(
    *, solved, keep_intermediates
):
    for _ in range(30):
        with pytest.raises(InvalidSimulationInputError, match="keep_intermediates"):
            lookup(solved=solved, period=0, keep_intermediates=keep_intermediates)


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_action_values_cover_the_action_grid_with_a_leading_row_axis(*, solved, period):
    got = lookup(solved=solved, period=period, keep_intermediates=ALL_INTERMEDIATES)
    n_rows = len(got.value)
    action_shape = tuple(len(ACTION_GRIDS[name]) for name in got.actions)
    assert (got.Q.shape, got.F.shape, got.F.dtype) == (
        (n_rows, *action_shape),
        (n_rows, *action_shape),
        np.dtype(bool),
    )


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_value_is_the_max_of_feasible_action_values(*, solved, period):
    got = lookup(solved=solved, period=period, keep_intermediates=ALL_INTERMEDIATES)
    expected, _ = masked_max_and_argmax(Q=np.asarray(got.Q), F=np.asarray(got.F))
    np.testing.assert_array_equal(np.asarray(got.value), expected)


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_actions_are_the_argmax_of_feasible_action_values(*, solved, period):
    got = lookup(solved=solved, period=period, keep_intermediates=ALL_INTERMEDIATES)
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
    with_values = lookup(
        solved=solved, period=period, keep_intermediates=ALL_INTERMEDIATES
    )
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
        keep_intermediates=ALL_INTERMEDIATES,
    )
    assert (
        got.Q.shape,
        np.asarray(got.Q).tolist(),
        got.F.shape,
        np.asarray(got.F).tolist(),
    ) == ((1,), [0.0], (1,), [True])


def test_some_actions_are_infeasible(solved):
    got = lookup(
        solved=solved, period=LAST_ALIVE_PERIOD, keep_intermediates=ALL_INTERMEDIATES
    )
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
        keep_intermediates=ALL_INTERMEDIATES,
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
        keep_intermediates=ALL_INTERMEDIATES,
    )
    labor_axis = list(got.actions).index("labor_supply")
    expected, _ = masked_max_and_argmax(Q=np.asarray(got.Q), F=np.asarray(got.F))
    assert got.Q.shape[1 + labor_axis] == 1
    np.testing.assert_array_equal(np.asarray(got.value), expected)


def test_intermediates_are_refused_for_a_collective_regime():
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
            keep_intermediates=("CE",),
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
        keep_intermediates=ALL_INTERMEDIATES,
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
        keep_intermediates=ALL_INTERMEDIATES,
    )
    expected = lookup(
        solved=solved,
        period=LAST_ALIVE_PERIOD,
        states=states,
        keep_intermediates=ALL_INTERMEDIATES,
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
    got = lookup(solved=solved, period=period, keep_intermediates=ALL_INTERMEDIATES)
    assert (got.U.shape, got.CE.shape) == (got.Q.shape, got.Q.shape)


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_linear_action_value_is_flow_utility_plus_discounted_continuation(
    *, solved, period
):
    """Under `LinearAggregator`, `Q == U + discount_factor * CE` at every entry.

    The two terms reach magnitudes in the hundreds and can cancel to near zero, so
    the rounding of the sum is bounded relative to `|U| + discount_factor * |CE|`.
    Where both terms are zero, `Q` must be zero itself.
    """
    got = lookup(solved=solved, period=period, keep_intermediates=ALL_INTERMEDIATES)
    U, CE = np.asarray(got.U), np.asarray(got.CE)
    scale = np.abs(U) + DISCOUNT_FACTOR * np.abs(CE)
    np.testing.assert_array_almost_equal(
        (np.asarray(got.Q) - (U + DISCOUNT_FACTOR * CE))
        / np.where(scale > 0, scale, 1),
        np.zeros_like(scale),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [0, LAST_ALIVE_PERIOD])
def test_flow_utility_is_the_utility_function_on_the_action_grid(*, solved, period):
    """`U` is `log(consumption) - disutility_of_work * working`, in every row.

    The utility of `working_life` reads no state, so every queried row holds the
    same table over `(labor_supply, consumption)`.
    """
    got = lookup(solved=solved, period=period, keep_intermediates=ALL_INTERMEDIATES)
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
        keep_intermediates=ALL_INTERMEDIATES,
    )
    np.testing.assert_array_almost_equal(
        np.stack([np.asarray(got.U), np.asarray(got.CE)]),
        np.zeros((2, 1)),
        decimal=DECIMAL_PRECISION,
    )


def test_infeasible_actions_keep_a_finite_flow_utility_and_continuation(solved):
    """Where `Q` is finite, so are `U` and `CE`, feasible or not."""
    states = {"wealth": jnp.array([5.0, 20.0, 40.0, 70.0])}
    got = lookup(
        solved=solved,
        period=0,
        states=states,
        keep_intermediates=ALL_INTERMEDIATES,
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
    full = lookup(solved=solved, period=0, keep_intermediates=ALL_INTERMEDIATES)
    restricted = lookup(
        solved=solved,
        period=0,
        action_grids={"labor_supply": jnp.array([branch], dtype=jnp.int32)},
        keep_intermediates=ALL_INTERMEDIATES,
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
        keep_intermediates=ALL_INTERMEDIATES,
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
        keep_intermediates=ALL_INTERMEDIATES,
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
        keep_intermediates=ALL_INTERMEDIATES,
    )
    assert len(traces) == 1


@categorical(ordered=False)
class TerminalOnlyRegimeId:
    terminal: ScalarInt


def terminal_utility(*, x: ContinuousState) -> FloatND:
    return x + 2.0


def nonnegative_x(*, x: ContinuousState) -> BoolND:
    return x >= 0.0


def lookup_actionless_terminal(
    *, keep_intermediates: tuple[Intermediate, ...]
) -> PolicyLookup:
    """Look up a stateful terminal regime without actions at `x = 0` and `x = -1`.

    Each row has one candidate: utility `x + 2`, feasible only at `x >= 0`. The
    solve grid is wholly feasible; the query at `x = -1` extrapolates off it, as
    lookup permits for continuous states.
    """
    model = Model(
        ages=AgeGrid(exact_values=(0,)),
        regimes={
            "terminal": Regime(
                states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
                functions={"utility": terminal_utility},
                constraints={"nonnegative_x": nonnegative_x},
            )
        },
        regime_id_class=TerminalOnlyRegimeId,
        initial_nodes={0: "terminal"},
        edges={},
    )
    solution = model.solve(params={}, log_level="off")
    return model.lookup_policy(
        params={},
        solution=solution,
        period=0,
        regime_name="terminal",
        states={"x": jnp.asarray([0.0, -1.0])},
        keep_intermediates=keep_intermediates,
    )


@pytest.mark.parametrize("keep_intermediates", [(), ALL_INTERMEDIATES])
def test_actionless_row_with_infeasible_state_has_minus_infinity_value(
    *, keep_intermediates: tuple[Intermediate, ...]
) -> None:
    """An infeasible state with no actions to choose has no admissible value."""
    got = lookup_actionless_terminal(keep_intermediates=keep_intermediates)
    np.testing.assert_array_equal(np.asarray(got.value), [2.0, -np.inf])


def test_actionless_row_with_infeasible_state_keeps_its_raw_parts() -> None:
    """At an infeasible actionless row, `Q`, `U` and `CE` stay raw and `F` is false."""
    got = lookup_actionless_terminal(keep_intermediates=ALL_INTERMEDIATES)
    np.testing.assert_array_equal(
        np.stack([np.asarray(x, dtype=float) for x in (got.Q, got.U, got.CE, got.F)]),
        np.array([[2.0, 1.0], [2.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
    )


# Lifetime of the derived lookup programs. Each `Model` owns the programs its
# lookups derive, so they live exactly as long as it does and a rebuilt model
# leaves nothing behind.
DERIVED_CACHES = {
    "kept_parts": "_kept_parts_programs",
    "action_extents": "_action_extent_programs",
}


def solve_small_model():
    grid = LinSpacedGrid(start=1, stop=5, n_points=5)
    model = get_model(n_periods=3, wealth_grid=grid, consumption_grid=grid)
    params = get_params(n_periods=3)
    return model, params, model.solve(params=params, log_level="off")


def lookup_deriving(*, solved, kind, reverse=False):
    """Look up period 0 with options that derive one program of `kind`."""
    model, params, solution = solved
    options = (
        {"keep_intermediates": ("F", "Q") if reverse else ("Q", "F")}
        if kind == "kept_parts"
        else {"action_grids": {"consumption": jnp.asarray([1.0, 2.0, 3.0])}}
    )
    return model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working_life",
        states={"wealth": jnp.asarray([4.0, 5.0])},
        **options,
    )


def only_derived_entry(*, model, kind):
    """The one `(source, derived)` pair the model's `kind` cache holds."""
    ((key, (source, derived)),) = getattr(model, DERIVED_CACHES[kind]).items()
    assert key[0] == id(source)
    assert derived is not source
    return source, derived


def weak_observers_of_one_model(*, kind):
    """Weak references to a looked-up model and its derived programs, nothing else."""
    solved = solve_small_model()
    lookup_deriving(solved=solved, kind=kind)
    source, derived = only_derived_entry(model=solved[0], kind=kind)
    return {
        "model": weakref.ref(solved[0]),
        "source": weakref.ref(source),
        "source_function": weakref.ref(source.function),
        "derived": weakref.ref(derived),
        "derived_function": weakref.ref(derived.function),
    }


@pytest.mark.parametrize("kind", DERIVED_CACHES)
def test_derived_lookup_programs_are_released_with_their_model(kind):
    refs = weak_observers_of_one_model(kind=kind)
    gc.collect()
    assert {name: ref() is None for name, ref in refs.items()} == dict.fromkeys(
        refs, True
    )


@pytest.mark.parametrize("kind", DERIVED_CACHES)
def test_repeated_lookups_of_a_live_model_reuse_its_derived_program(kind):
    solved = solve_small_model()
    first = lookup_deriving(solved=solved, kind=kind)
    source, derived = only_derived_entry(model=solved[0], kind=kind)
    second = lookup_deriving(solved=solved, kind=kind, reverse=True)
    assert only_derived_entry(model=solved[0], kind=kind) == (source, derived)
    np.testing.assert_array_equal(np.asarray(second.value), np.asarray(first.value))


@pytest.mark.parametrize("kind", DERIVED_CACHES)
def test_two_models_derive_their_own_programs(kind):
    one, two = solve_small_model(), solve_small_model()
    lookup_deriving(solved=one, kind=kind)
    lookup_deriving(solved=two, kind=kind)
    _, derived_one = only_derived_entry(model=one[0], kind=kind)
    _, derived_two = only_derived_entry(model=two[0], kind=kind)
    assert derived_one is not derived_two
    released = weakref.ref(derived_one)
    del one, derived_one
    gc.collect()
    assert released() is None
    assert only_derived_entry(model=two[0], kind=kind)[1] is derived_two


def test_programs_sharing_a_function_derive_distinct_programs():
    """The cache tells source programs apart by identity, not by their function.

    A stored source keeps its identity from being reused while its entry lives.
    """
    model, _, _ = solve_small_model()
    source = model._regimes["working_life"].simulation.programs.action_values[0]
    twin = dataclasses.replace(source, name=f"{source.name}_twin")
    roles = frozenset({ACTION_VALUES, ACTION_FEASIBILITY})
    cache = {}
    derived = with_kept_action_parts(program=source, roles=roles, cache=cache)
    derived_twin = with_kept_action_parts(program=twin, roles=roles, cache=cache)
    assert twin.function is source.function
    assert derived_twin is not derived
    assert derived_twin.name == twin.name
    twin_id = id(twin)
    del twin, derived_twin
    gc.collect()
    assert id(cache[(twin_id, roles)][0]) == twin_id


def test_each_proper_subset_of_parts_derives_one_program():
    model, _, _ = solve_small_model()
    source = model._regimes["working_life"].simulation.programs.action_values[0]
    parts = (ACTION_VALUES, ACTION_FEASIBILITY, FLOW_UTILITY, CONTINUATION_CE)
    cache = {}
    for size in (1, 2, 3):
        for subset in itertools.combinations(parts, size):
            roles = frozenset(subset)
            derived = with_kept_action_parts(program=source, roles=roles, cache=cache)
            again = with_kept_action_parts(program=source, roles=roles, cache=cache)
            assert again is derived
    assert len(cache) == 14
    every = with_kept_action_parts(program=source, roles=frozenset(parts), cache=cache)
    assert every is source
    assert len(cache) == 14


def test_a_restored_model_starts_with_empty_derived_caches():
    solved = solve_small_model()
    for kind in DERIVED_CACHES:
        lookup_deriving(solved=solved, kind=kind)
    model = solved[0]
    state = model.__getstate__()
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    for attribute in DERIVED_CACHES.values():
        assert attribute not in state
        assert getattr(model, attribute)
        assert getattr(restored, attribute) == {}


@pytest.mark.parametrize("kind", DERIVED_CACHES)
def test_derived_program_is_chosen_under_the_model_lock_and_dispatched_outside_it(
    *, kind, monkeypatch
):
    solved = solve_small_model()
    model = solved[0]
    name = "with_kept_action_parts" if kind == "kept_parts" else "_with_action_extents"
    derive = getattr(lcm.model, name)
    dispatch = SimulationRuntime.dispatch
    locked_when_derived, locked_when_dispatched = [], []

    def observed_derive(**kwargs):
        locked_when_derived.append(model._simulate_compile_lock.locked())
        return derive(**kwargs)

    def observed_dispatch(self, **kwargs):
        locked_when_dispatched.append(model._simulate_compile_lock.locked())
        return dispatch(self, **kwargs)

    monkeypatch.setattr(lcm.model, name, observed_derive)
    monkeypatch.setattr(SimulationRuntime, "dispatch", observed_dispatch)
    lookup_deriving(solved=solved, kind=kind)
    assert (locked_when_derived, set(locked_when_dispatched)) == ([True], {False})
