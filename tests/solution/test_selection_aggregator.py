"""A selection aggregator chooses the action; the Koopmans aggregator values it.

The model is small enough to solve by hand. Wealth takes the codes `0, 1, 2, 3`.
Spending yields utility `1.5` and lowers wealth by one, and is feasible only
with positive wealth. Saving yields nothing and raises wealth by one, capped at
three. The last period pays `V_2 = [0, 1, 2, 4]` by wealth, and `δ = 0.9`.

A present-biased agent values the future at `u + δ·CE` and the self that
chooses it at `u + β̃δ·CE`, with `β = 0.5` for today's self in simulation:

- time-consistent: `β = β̃ = 1`;
- naive: `β̃ = 1`, today's self selects with `β`;
- sophisticated: `β̃ = β`;
- partially naive: `β < β̃ < 1`.

The expected values come from an exhaustive NumPy enumeration
(`_enumerate_values`), which shares no code with the solve.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import lcm
from lcm.exceptions import ExecutionPlanningError, ModelInitializationError
from lcm.execution import ExecutionConfig
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserFunction,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")

DELTA = 0.9
BETA = 0.5
SPEND_UTILITY = 1.5
TERMINAL_VALUES = np.array([0.0, 1.0, 2.0, 4.0])


@lcm.categorical(ordered=False)
class RegimeId:
    decide: ScalarInt
    end: ScalarInt


@lcm.categorical(ordered=True)
class Wealth:
    zero: ScalarInt
    one: ScalarInt
    two: ScalarInt
    three: ScalarInt


@lcm.categorical(ordered=False)
class Choice:
    spend: ScalarInt
    save: ScalarInt


def utility(choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Choice.spend, SPEND_UTILITY, 0.0)


def can_afford(*, wealth: DiscreteState, choice: DiscreteAction) -> BoolND:
    return (choice == Choice.save) | (wealth >= Wealth.one)


def next_wealth(*, wealth: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(
        choice == Choice.spend, wealth - 1, jnp.minimum(wealth + 1, Wealth.three)
    )


def terminal_utility(wealth: DiscreteState) -> FloatND:
    return jnp.asarray(TERMINAL_VALUES)[wealth]


def long_run_value(*, utility: FloatND, CE: FloatND, delta: float) -> FloatND:
    return utility + delta * CE


def perceived_value(
    *, utility: FloatND, CE: FloatND, beta_tilde: float, delta: float
) -> FloatND:
    return utility + beta_tilde * delta * CE


def present_biased_value(
    *, utility: FloatND, CE: FloatND, beta: float, delta: float
) -> FloatND:
    return utility + beta * delta * CE


def indifferent(*, utility: FloatND, CE: FloatND) -> FloatND:
    return jnp.zeros_like(utility + CE)


# The future selves select with `β̃`, today's self in simulation with `β`.
PRESENT_BIASED_SELECTION = lcm.Phased(
    solve=perceived_value, simulate=present_biased_value
)


def _model(
    *,
    koopmans_aggregator: UserFunction | lcm.Phased = long_run_value,
    selection_aggregator: UserFunction | lcm.Phased | None = PRESENT_BIASED_SELECTION,
    taste_shocks: lcm.ExtremeValueTasteShocks | None = None,
) -> lcm.Model:
    wealth = lcm.DiscreteGrid(Wealth)
    return lcm.Model(
        n_periods=3,
        regimes={
            "decide": lcm.Regime(
                actions={"choice": lcm.DiscreteGrid(Choice)},
                states={"wealth": wealth},
                state_transitions={"wealth": next_wealth},
                constraints={"can_afford": can_afford},
                functions={"utility": utility},
                koopmans_aggregator=koopmans_aggregator,
                selection_aggregator=selection_aggregator,
                taste_shocks=taste_shocks,
            ),
            "end": lcm.Regime(
                states={"wealth": wealth}, functions={"utility": terminal_utility}
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
        edges={
            "decide": {
                "decide": lcm.PeriodRange(exclusive_stop=1),
                "end": lcm.Periods(values=(1,)),
            }
        },
    )


def _params(*, beta_tilde: float) -> UserParams:
    return {
        "decide": {
            "koopmans_aggregator": {"delta": DELTA},
            "selection_aggregator": {
                "beta_tilde": beta_tilde,
                "beta": BETA,
                "delta": DELTA,
            },
        }
    }


def _enumerate_values(*, beta_tilde: float) -> dict[int, np.ndarray]:
    """Solve the model by enumerating both choices at every wealth level."""
    values = {2: TERMINAL_VALUES}
    for period in (1, 0):
        V = np.empty(4)
        for w in range(4):
            choices = [(0.0, min(w + 1, 3))]
            if w >= 1:
                choices.insert(0, (SPEND_UTILITY, w - 1))
            perceived = [
                u + beta_tilde * DELTA * values[period + 1][n] for u, n in choices
            ]
            chosen = choices[int(np.argmax(perceived))]
            V[w] = chosen[0] + DELTA * values[period + 1][chosen[1]]
        values[period] = V
    return values


AGENTS = {
    "time-consistent": 1.0,
    "sophisticated": BETA,
    "partially naive": 0.7,
}


@pytest.mark.parametrize("period", [0, 1])
@pytest.mark.parametrize("agent", list(AGENTS))
def test_solve_reads_the_long_run_value_at_the_selected_action(
    *, agent: str, period: int
) -> None:
    """`V_t` is `u + δ·CE` at the action that maximizes `u + β̃δ·CE`."""
    beta_tilde = AGENTS[agent]
    solution = _model().solve(params=_params(beta_tilde=beta_tilde), log_level="off")
    assert_array_almost_equal(
        solution.value(period=period, regime="decide"),
        _enumerate_values(beta_tilde=beta_tilde)[period],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    ("beta_tilde", "period", "expected"),
    [
        (1.0, 1, [0.9, 1.8, 3.6, 3.6]),
        (1.0, 0, [1.62, 3.24, 3.24, 4.74]),
        (BETA, 1, [0.9, 1.5, 2.4, 3.3]),
        (BETA, 0, [1.35, 2.31, 2.85, 3.66]),
        (0.7, 1, [0.9, 1.5, 3.6, 3.3]),
        (0.7, 0, [1.35, 3.24, 2.85, 4.74]),
    ],
)
def test_enumeration_matches_the_hand_solution(
    *, beta_tilde: float, period: int, expected: list[float]
) -> None:
    """The enumeration oracle reproduces the values solved by hand."""
    assert_array_almost_equal(
        _enumerate_values(beta_tilde=beta_tilde)[period], expected, decimal=12
    )


def test_sophisticated_value_lies_below_the_best_long_run_value() -> None:
    """At wealth two the self spends, worth 2.85, although saving is worth 2.97."""
    solution = _model().solve(params=_params(beta_tilde=BETA), log_level="off")
    assert_array_almost_equal(
        solution.value(period=0, regime="decide")[Wealth.two],
        2.85,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [0, 1])
def test_identical_aggregators_solve_like_no_selection_aggregator(
    *, period: int
) -> None:
    """With `β̃ = 1` the selection aggregator equals the Koopmans aggregator."""
    with_selection = _model(selection_aggregator=perceived_value).solve(
        params={
            "decide": {
                "koopmans_aggregator": {"delta": DELTA},
                "selection_aggregator": {"beta_tilde": 1.0, "delta": DELTA},
            }
        },
        log_level="off",
    )
    without_selection = _model(selection_aggregator=None).solve(
        params={"decide": {"koopmans_aggregator": {"delta": DELTA}}},
        log_level="off",
    )
    assert_array_almost_equal(
        with_selection.value(period=period, regime="decide"),
        without_selection.value(period=period, regime="decide"),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [0, 1])
def test_tied_selection_takes_the_first_feasible_action(*, period: int) -> None:
    """A selection objective tied across actions picks spend wherever it is feasible."""
    solution = _model(selection_aggregator=indifferent).solve(
        params={"decide": {"koopmans_aggregator": {"delta": DELTA}}},
        log_level="off",
    )
    next_values = (
        TERMINAL_VALUES
        if period == 1
        else np.array([0.9, 1.5, 2.4, 3.3])  # always spend at period 1
    )
    expected = [DELTA * next_values[1]] + [
        SPEND_UTILITY + DELTA * next_values[w - 1] for w in (1, 2, 3)
    ]
    assert_array_almost_equal(
        solution.value(period=period, regime="decide"),
        expected,
        decimal=DECIMAL_PRECISION,
    )


def _simulate(*, model: lcm.Model, params: UserParams) -> lcm.SimulationResult:
    return model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([0, 1, 2, 3]),
            "regime_id": jnp.full(4, RegimeId.decide),
            "period": jnp.zeros(4, dtype=jnp.int32),
        },
        log_level="off",
    )


def test_naive_selection_simulates_the_phased_koopmans_naive_choices() -> None:
    """`β̃ = 1` with today's `β` chooses what the `Phased` Koopmans recipe chooses."""
    selection = _simulate(model=_model(), params=_params(beta_tilde=1.0)).to_dataframe()
    phased_koopmans = _simulate(
        model=_model(
            koopmans_aggregator=lcm.Phased(
                solve=long_run_value, simulate=present_biased_value
            ),
            selection_aggregator=None,
        ),
        params={"decide": {"koopmans_aggregator": {"delta": DELTA, "beta": BETA}}},
    ).to_dataframe()
    decisions = selection["regime_name"] == "decide"
    assert_array_equal(
        selection.loc[decisions, "choice"].to_numpy(),
        phased_koopmans.loc[decisions, "choice"].to_numpy(),
    )


def test_simulation_selects_with_the_simulate_variant() -> None:
    """Today's self selects with `β`: the naive agent spends at wealth 1, 2 and 3."""
    df = _simulate(model=_model(), params=_params(beta_tilde=1.0)).to_dataframe()
    first_period = df.loc[df["period"] == 0].sort_values("wealth")
    assert list(first_period["choice"]) == ["save", "spend", "spend", "spend"]


def test_simulation_publishes_the_long_run_value_of_the_choice() -> None:
    """The simulated value is `u + δ·CE` at the chosen action."""
    df = _simulate(model=_model(), params=_params(beta_tilde=1.0)).to_dataframe()
    first_period = df.loc[df["period"] == 0].sort_values("wealth")
    # Naive today's self: save at 0, spend at 1, 2, 3 against V_1 = [.9, 1.8, 3.6, 3.6].
    assert_array_almost_equal(
        first_period["value"].to_numpy(),
        [1.62, 1.5 + 0.81, 1.5 + 1.62, 1.5 + 3.24],
        decimal=DECIMAL_PRECISION,
    )


def test_terminal_regime_refuses_a_selection_aggregator() -> None:
    """A terminal regime has no continuation to select against."""
    with pytest.raises(ModelInitializationError, match="selection_aggregator"):
        lcm.Model(
            n_periods=2,
            regimes={
                "decide": lcm.Regime(
                    actions={"choice": lcm.DiscreteGrid(Choice)},
                    functions={"utility": utility},
                ),
                "end": lcm.Regime(
                    functions={"utility": lambda: jnp.asarray(0.0)},
                    selection_aggregator=perceived_value,
                ),
            },
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
            edges={"decide": {"end": lcm.Periods(values=(0,))}},
        )


def test_taste_shocks_refuse_a_selection_aggregator() -> None:
    """The smoothed maximum does not read out a second aggregator."""
    with pytest.raises(ModelInitializationError, match="selection_aggregator"):
        _model(taste_shocks=lcm.ExtremeValueTasteShocks())


@pytest.mark.parametrize("return_action_values", [False, True])
def test_lookup_selects_like_simulation(*, return_action_values: bool) -> None:
    """Today's sophisticated self spends wherever it can, at every wealth level."""
    model = _model()
    params = _params(beta_tilde=BETA)
    got = model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([0, 1, 2, 3])},
        return_action_values=return_action_values,
    )
    assert got.actions is not None
    assert_array_equal(
        got.actions["choice"],
        [Choice.save, Choice.spend, Choice.spend, Choice.spend],
    )


@pytest.mark.parametrize("return_action_values", [False, True])
def test_lookup_publishes_the_long_run_value_of_the_selection(
    *, return_action_values: bool
) -> None:
    """The looked-up value is `u + δ·CE` at today's selected action."""
    model = _model()
    params = _params(beta_tilde=BETA)
    got = model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([0, 1, 2, 3])},
        return_action_values=return_action_values,
    )
    # Sophisticated V_1 = [0.9, 1.5, 2.4, 3.3]; save at 0, spend elsewhere.
    assert_array_almost_equal(
        got.value,
        [DELTA * 1.5, 1.5 + DELTA * 0.9, 1.5 + DELTA * 1.5, 1.5 + DELTA * 2.4],
        decimal=DECIMAL_PRECISION,
    )


def test_other_solvers_refuse_a_selection_aggregator() -> None:
    """Only `GridSearch` selects on one aggregator and values with another."""
    from lcm_examples import iskhakov_et_al_2017 as iskhakov  # noqa: PLC0415

    ages = lcm.AgeGrid(start=40, inclusive_stop=60, step="10Y")
    with pytest.raises(ModelInitializationError, match="DCEGM solver does not"):
        lcm.Model(
            edges=iskhakov.get_edges(ages=ages),
            regimes={
                "working_life": iskhakov.dcegm_working_life,
                "retirement": iskhakov.dcegm_retirement.replace(
                    selection_aggregator=perceived_value
                ),
                "dead": iskhakov.dead,
            },
            ages=ages,
            regime_id_class=iskhakov.RegimeId,
            initial_nodes={40: "working_life"},
        )


def test_collective_regime_refuses_a_selection_aggregator() -> None:
    """A household selects with its Pareto-weighted objective."""
    shared_utility = lcm.CollectiveUtility(utilities={"f": utility, "m": utility})
    with pytest.raises(ModelInitializationError, match="collective regime cannot"):
        lcm.Model(
            n_periods=2,
            regimes={
                "decide": lcm.Regime(
                    actions={"choice": lcm.DiscreteGrid(Choice)},
                    functions={"utility": shared_utility},
                    selection_aggregator=perceived_value,
                ),
                "end": lcm.Regime(
                    actions={"choice": lcm.DiscreteGrid(Choice)},
                    functions={"utility": shared_utility},
                ),
            },
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
            edges={"decide": {"end": lcm.Periods(values=(0,))}},
        )


@lcm.categorical(ordered=False)
class Mood:
    patient: ScalarInt
    impatient: ScalarInt


def mood_dependent_value(
    *, utility: FloatND, CE: FloatND, mood: DiscreteState, delta: float
) -> FloatND:
    return utility + jnp.where(mood == Mood.patient, 1.0, BETA) * delta * CE


def _mood_model(
    *,
    koopmans_aggregator: UserFunction = long_run_value,
    selection_aggregator: UserFunction | None = mood_dependent_value,
    execution_config: ExecutionConfig = ExecutionConfig(),  # noqa: B008
) -> lcm.Model:
    return lcm.Model(
        n_periods=3,
        regimes={
            "decide": lcm.Regime(
                actions={"choice": lcm.DiscreteGrid(Choice)},
                states={
                    "wealth": lcm.DiscreteGrid(Wealth),
                    "mood": lcm.DiscreteGrid(Mood),
                },
                state_transitions={
                    "wealth": next_wealth,
                    "mood": lcm.fixed_transition("mood"),
                },
                constraints={"can_afford": can_afford},
                functions={"utility": utility},
                koopmans_aggregator=koopmans_aggregator,
                selection_aggregator=selection_aggregator,
            ),
            "end": lcm.Regime(
                states={"wealth": lcm.DiscreteGrid(Wealth)},
                functions={"utility": terminal_utility},
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
        edges={
            "decide": {
                "decide": lcm.PeriodRange(exclusive_stop=1),
                "end": lcm.Periods(values=(1,)),
            }
        },
        execution_config=execution_config,
    )


def _mood_params() -> UserParams:
    return {
        "decide": {
            "koopmans_aggregator": {"delta": DELTA},
            "selection_aggregator": {"delta": DELTA},
        }
    }


def test_a_state_read_only_by_the_selection_aggregator_is_used() -> None:
    """An impatient mood selects like the sophisticated agent, a patient one not."""
    model = _mood_model()
    solution = model.solve(params=_mood_params(), log_level="off")
    assert_array_almost_equal(
        solution.value(period=1, regime="decide"),
        np.stack(
            [
                _enumerate_values(beta_tilde=1.0)[1],
                _enumerate_values(beta_tilde=BETA)[1],
            ],
            axis=1,
        ),
        decimal=DECIMAL_PRECISION,
    )


def shocked_utility(*, choice: DiscreteAction, shock: FloatND) -> FloatND:
    return utility(choice) + shock


def test_folded_state_refuses_a_selection_aggregator() -> None:
    """The selection route keeps every state axis it solves on."""
    with pytest.raises(ModelInitializationError, match="selection_aggregator"):
        lcm.Model(
            n_periods=3,
            regimes={
                "decide": lcm.Regime(
                    actions={"choice": lcm.DiscreteGrid(Choice)},
                    states={
                        "wealth": lcm.DiscreteGrid(Wealth),
                        "shock": lcm.NormalIIDProcess(
                            n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=True
                        ),
                    },
                    state_transitions={"wealth": next_wealth},
                    constraints={"can_afford": can_afford},
                    functions={"utility": shocked_utility},
                    selection_aggregator=perceived_value,
                ),
                "end": lcm.Regime(
                    states={"wealth": lcm.DiscreteGrid(Wealth)},
                    functions={"utility": terminal_utility},
                ),
            },
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
            edges={
                "decide": {
                    "decide": lcm.PeriodRange(exclusive_stop=1),
                    "end": lcm.Periods(values=(1,)),
                }
            },
        )


def test_invariant_blocking_refuses_a_selection_aggregator() -> None:
    """Blocked programs read the reducers a selection aggregator does not use."""
    with pytest.raises(ExecutionPlanningError, match="selection aggregator"):
        _mood_model(
            execution_config=ExecutionConfig(
                devices=(0,),
                device_memory_bytes=None,
                invariant_block_widths={"mood": 1},
            )
        )


def test_a_state_read_only_by_the_koopmans_aggregator_is_used() -> None:
    """A patient mood values and selects like the time-consistent agent."""
    model = _mood_model(
        koopmans_aggregator=mood_dependent_value, selection_aggregator=None
    )
    solution = model.solve(
        params={"decide": {"koopmans_aggregator": {"delta": DELTA}}}, log_level="off"
    )
    assert_array_almost_equal(
        solution.value(period=1, regime="decide")[:, Mood.patient],
        _enumerate_values(beta_tilde=1.0)[1],
        decimal=DECIMAL_PRECISION,
    )


def test_phased_selection_needs_both_variants() -> None:
    """A `Phased` selection aggregator with an empty phase is refused by name."""
    with pytest.raises(ModelInitializationError, match="selection_aggregator"):
        _model(selection_aggregator=lcm.Phased(solve=perceived_value, simulate=None))
