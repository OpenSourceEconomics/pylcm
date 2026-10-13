"""`Decoupled` chooses with one aggregator and stores the value of another.

The model is small enough to solve by hand. Wealth takes the codes `0, 1, 2, 3`.
Spending yields utility `1.5` and lowers wealth by one, and is feasible only
with positive wealth. Saving yields nothing and raises wealth by one, capped at
three. The last period pays `V_2 = [0, 1, 2, 4]` by wealth, and `δ = 0.9`.

A present-biased agent stores the long-run value `u + δ·CE`. The selves the solve
anticipates choose with `u + β̃δ·CE`, and the acting self in simulation chooses
with `u + βδ·CE`, `β = 0.5`:

- time-consistent: `β = β̃ = 1`;
- naive: `β̃ = 1`;
- sophisticated: `β̃ = β`;
- partially naive: `β < β̃ < 1`.

The expected values come from an exhaustive NumPy enumeration
(`_enumerate_values`), which shares no code with the solve.
"""

from collections.abc import Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import lcm
from lcm.exceptions import (
    ModelInitializationError,
    NoFeasibleActionError,
    RegimeInitializationError,
)
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
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


def long_run(*, utility: FloatND, CE: FloatND, delta: float) -> FloatND:
    return utility + delta * CE


def perceived_future_choice(
    *, utility: FloatND, CE: FloatND, beta_tilde: float, delta: float
) -> FloatND:
    return utility + beta_tilde * delta * CE


def current_choice(
    *, utility: FloatND, CE: FloatND, beta: float, delta: float
) -> FloatND:
    return utility + beta * delta * CE


def indifferent(*, utility: FloatND, CE: FloatND) -> FloatND:
    return jnp.zeros_like(utility + CE)


# The future selves choose with `β̃`, today's self in simulation with `β`.
PRESENT_BIAS = lcm.Decoupled(
    objective=lcm.Phased(solve=perceived_future_choice, simulate=current_choice),
    value=long_run,
)


def _model(
    *,
    koopmans_aggregator: UserFunction | lcm.Phased | lcm.Decoupled = PRESENT_BIAS,
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
            "koopmans_aggregator": {
                "beta_tilde": beta_tilde,
                "beta": BETA,
                "delta": DELTA,
            }
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


@pytest.mark.parametrize("period", [0, 1])
@pytest.mark.parametrize("agent", list(AGENTS))
def test_solve_stores_the_value_at_the_objective(*, agent: str, period: int) -> None:
    """`V_t` is `u + δ·CE` at the action that maximizes `u + β̃δ·CE`."""
    beta_tilde = AGENTS[agent]
    solution = _model().solve(params=_params(beta_tilde=beta_tilde), log_level="off")
    assert_array_almost_equal(
        solution.value(period=period, regime="decide"),
        _enumerate_values(beta_tilde=beta_tilde)[period],
        decimal=DECIMAL_PRECISION,
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
def test_identical_objective_and_value_solve_like_the_plain_aggregator(
    *, period: int
) -> None:
    """`Decoupled(objective=W, value=W)` solves like `koopmans_aggregator=W`."""
    params: UserParams = {"decide": {"koopmans_aggregator": {"delta": DELTA}}}
    wrapped = _model(
        koopmans_aggregator=lcm.Decoupled(objective=long_run, value=long_run)
    ).solve(params=params, log_level="off")
    plain = _model(koopmans_aggregator=long_run).solve(params=params, log_level="off")
    assert_array_equal(
        wrapped.value(period=period, regime="decide"),
        plain.value(period=period, regime="decide"),
    )


@pytest.mark.parametrize("period", [0, 1])
def test_tied_objective_takes_the_first_feasible_action(*, period: int) -> None:
    """An objective tied across actions picks spend wherever it is feasible."""
    solution = _model(
        koopmans_aggregator=lcm.Decoupled(objective=indifferent, value=long_run)
    ).solve(
        params={"decide": {"koopmans_aggregator": {"delta": DELTA}}},
        log_level="off",
    )
    next_values = (
        TERMINAL_VALUES if period == 1 else np.array([0.9, 1.5, 2.4, 3.3])
    )  # always spend at period 1
    expected = [DELTA * next_values[1]] + [
        SPEND_UTILITY + DELTA * next_values[w - 1] for w in (1, 2, 3)
    ]
    assert_array_almost_equal(
        solution.value(period=period, regime="decide"),
        expected,
        decimal=DECIMAL_PRECISION,
    )


def _simulate(*, model: lcm.Model, params: UserParams) -> pd.DataFrame:
    return model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([0, 1, 2, 3]),
            "regime_id": jnp.full(4, RegimeId.decide),
            "period": jnp.zeros(4, dtype=jnp.int32),
        },
        log_level="off",
    ).to_dataframe()


def _first_period(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["period"] == 0].sort_values("wealth")


# A naive agent (`β̃ = 1`) whose acting self chooses with `β`: it saves at wealth
# zero and spends otherwise, against `V_1 = [0.9, 1.8, 3.6, 3.6]`.
NAIVE_CHOICES = ["save", "spend", "spend", "spend"]
NAIVE_LONG_RUN_VALUES = [
    DELTA * 1.8,
    SPEND_UTILITY + DELTA * 0.9,
    SPEND_UTILITY + DELTA * 1.8,
    SPEND_UTILITY + DELTA * 3.6,
]


def test_simulation_chooses_with_the_simulate_variant() -> None:
    """Today's self chooses with `β`: the naive agent spends at wealth 1, 2 and 3."""
    df = _first_period(_simulate(model=_model(), params=_params(beta_tilde=1.0)))
    assert list(df["choice"]) == NAIVE_CHOICES


def test_simulation_publishes_the_stored_value_of_the_choice() -> None:
    """The simulated value is `u + δ·CE` at the chosen action."""
    df = _first_period(_simulate(model=_model(), params=_params(beta_tilde=1.0)))
    assert_array_almost_equal(
        df["value"].to_numpy(), NAIVE_LONG_RUN_VALUES, decimal=DECIMAL_PRECISION
    )


def test_phased_koopmans_aggregator_makes_the_same_choices() -> None:
    """`Phased(solve=f, simulate=g)` chooses like `Decoupled(Phased(f, g), f)`."""
    phased = _first_period(
        _simulate(
            model=_model(
                koopmans_aggregator=lcm.Phased(solve=long_run, simulate=current_choice)
            ),
            params={"decide": {"koopmans_aggregator": {"delta": DELTA, "beta": BETA}}},
        )
    )
    assert list(phased["choice"]) == NAIVE_CHOICES


def test_phased_koopmans_aggregator_publishes_the_acting_selfs_value() -> None:
    """A `Phased` Koopmans aggregator publishes `u + βδ·CE` at the choice."""
    phased = _first_period(
        _simulate(
            model=_model(
                koopmans_aggregator=lcm.Phased(solve=long_run, simulate=current_choice)
            ),
            params={"decide": {"koopmans_aggregator": {"delta": DELTA, "beta": BETA}}},
        )
    )
    assert_array_almost_equal(
        phased["value"].to_numpy(),
        [
            BETA * DELTA * 1.8,
            1.5 + BETA * DELTA * 0.9,
            1.5 + BETA * DELTA * 1.8,
            1.5 + BETA * DELTA * 3.6,
        ],
        decimal=DECIMAL_PRECISION,
    )


def _lookup(*, return_action_values: bool) -> lcm.PolicyLookup:
    model = _model()
    params = _params(beta_tilde=1.0)
    return model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([0, 1, 2, 3])},
        return_action_values=return_action_values,
    )


@pytest.mark.parametrize("return_action_values", [False, True])
def test_lookup_chooses_with_the_simulate_variant(
    *, return_action_values: bool
) -> None:
    """Lookup chooses like simulation, with today's `β`, not the solve's `β̃`."""
    got = _lookup(return_action_values=return_action_values)
    actions = cast("Mapping[str, IntND]", got.actions)
    assert_array_equal(
        actions["choice"], [Choice.save, Choice.spend, Choice.spend, Choice.spend]
    )


@pytest.mark.parametrize("return_action_values", [False, True])
def test_lookup_publishes_the_stored_value_of_the_choice(
    *, return_action_values: bool
) -> None:
    """The looked-up value is `u + δ·CE` at today's choice."""
    got = _lookup(return_action_values=return_action_values)
    assert_array_almost_equal(
        got.value, NAIVE_LONG_RUN_VALUES, decimal=DECIMAL_PRECISION
    )


def test_lookup_publishes_the_objective_profile() -> None:
    """`Q_objective` is today's `u + βδ·CE` per action; `Q` is `u + δ·CE`."""
    got = _lookup(return_action_values=True)
    assert_array_almost_equal(
        cast("FloatND", got.Q_objective),
        cast("FloatND", got.U) + BETA * DELTA * cast("FloatND", got.CE),
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_without_decoupled_publishes_no_objective_profile() -> None:
    """A plain Koopmans aggregator leaves `Q_objective` empty."""
    model = _model(koopmans_aggregator=long_run)
    params: UserParams = {"decide": {"koopmans_aggregator": {"delta": DELTA}}}
    got = model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([0, 1])},
        return_action_values=True,
    )
    assert got.Q_objective is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: lcm.Decoupled(
            objective=long_run,
            value=lcm.Phased(solve=long_run, simulate=long_run),  # ty: ignore[invalid-argument-type]
        ),
        lambda: lcm.Decoupled(
            objective=lcm.Decoupled(objective=long_run, value=long_run),  # ty: ignore[invalid-argument-type]
            value=long_run,
        ),
        lambda: lcm.Decoupled(objective=long_run, value=2.0),  # ty: ignore[invalid-argument-type]
    ],
    ids=["phased-value", "nested", "not-callable"],
)
def test_decoupled_refuses_an_invalid_declaration(build) -> None:
    """Only `objective` may be `Phased`, and both parts must be aggregators."""
    with pytest.raises(RegimeInitializationError, match="Decoupled"):
        build()


def test_phased_koopmans_aggregator_refuses_a_decoupled_aggregator() -> None:
    """`Phased` around a `Decoupled` is refused; put `Phased` on `objective`."""
    with pytest.raises(ModelInitializationError, match="Decoupled"):
        _model(
            koopmans_aggregator=lcm.Phased(solve=PRESENT_BIAS, simulate=PRESENT_BIAS)
        )


def test_other_solvers_refuse_a_decoupled_aggregator() -> None:
    """Only `GridSearch` chooses with one aggregator and stores another."""
    from lcm_examples import iskhakov_et_al_2017 as iskhakov  # noqa: PLC0415

    ages = lcm.AgeGrid(start=40, inclusive_stop=60, step="10Y")
    decoupled = lcm.Decoupled(objective=long_run, value=long_run)
    with pytest.raises(ModelInitializationError, match="DCEGM solver does not"):
        lcm.Model(
            edges=iskhakov.get_edges(ages=ages),
            regimes={
                "working_life": iskhakov.dcegm_working_life.replace(
                    koopmans_aggregator=decoupled
                ),
                "retirement": iskhakov.dcegm_retirement.replace(
                    koopmans_aggregator=decoupled
                ),
                "dead": iskhakov.dead,
            },
            ages=ages,
            regime_id_class=iskhakov.RegimeId,
            initial_nodes={40: "working_life"},
        )


def test_collective_regime_refuses_a_decoupled_aggregator() -> None:
    """A household chooses with its Pareto-weighted objective."""
    shared_utility = lcm.CollectiveUtility(utilities={"f": utility, "m": utility})
    with pytest.raises(ModelInitializationError, match="collective regime cannot"):
        lcm.Model(
            n_periods=2,
            regimes={
                "decide": lcm.Regime(
                    actions={"choice": lcm.DiscreteGrid(Choice)},
                    functions={"utility": shared_utility},
                    koopmans_aggregator=lcm.Decoupled(
                        objective=long_run, value=long_run
                    ),
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


def mood_dependent_choice(
    *, utility: FloatND, CE: FloatND, mood: DiscreteState, delta: float
) -> FloatND:
    return utility + jnp.where(mood == Mood.patient, 1.0, BETA) * delta * CE


def _mood_model(
    *,
    koopmans_aggregator: UserFunction | lcm.Decoupled = lcm.Decoupled(  # noqa: B008
        objective=mood_dependent_choice, value=long_run
    ),
    execution_config: lcm.ExecutionConfig = lcm.ExecutionConfig(),  # noqa: B008
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


def test_a_state_read_only_by_the_objective_is_used() -> None:
    """An impatient mood chooses like the sophisticated agent, a patient one not."""
    solution = _mood_model().solve(
        params={"decide": {"koopmans_aggregator": {"delta": DELTA}}}, log_level="off"
    )
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


def test_a_state_read_only_by_the_koopmans_aggregator_is_used() -> None:
    """A patient mood values and chooses like the time-consistent agent."""
    solution = _mood_model(koopmans_aggregator=mood_dependent_choice).solve(
        params={"decide": {"koopmans_aggregator": {"delta": DELTA}}}, log_level="off"
    )
    assert_array_almost_equal(
        solution.value(period=1, regime="decide")[:, Mood.patient],
        _enumerate_values(beta_tilde=1.0)[1],
        decimal=DECIMAL_PRECISION,
    )


def test_a_labelled_objective_parameter_is_read_by_label() -> None:
    """A `pd.Series` indexed by mood binds each row to its label, not its position."""
    beta_by_mood = pd.Series(
        [BETA, 1.0],
        index=pd.Index(["impatient", "patient"], name="mood"),
    )

    def mood_indexed_choice(
        *,
        utility: FloatND,
        CE: FloatND,
        mood: DiscreteState,
        beta: FloatND,
        delta: float,
    ) -> FloatND:
        return utility + beta[mood] * delta * CE

    solution = _mood_model(
        koopmans_aggregator=lcm.Decoupled(objective=mood_indexed_choice, value=long_run)
    ).solve(
        params={
            "decide": {"koopmans_aggregator": {"delta": DELTA, "beta": beta_by_mood}}
        },
        log_level="off",
    )
    assert_array_almost_equal(
        solution.value(period=1, regime="decide")[:, Mood.impatient],
        _enumerate_values(beta_tilde=BETA)[1],
        decimal=DECIMAL_PRECISION,
    )


@lcm.categorical(ordered=True)
class Funds:
    empty: ScalarInt
    ready: ScalarInt


def first_choice_utility(*, choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Choice.spend, 1.0, 0.0)


def can_act(*, funds: DiscreteState) -> BoolND:
    return funds > Funds.empty


def perceived_next_funds(*, funds: DiscreteState) -> DiscreteState:
    return funds


def realized_next_funds(*, funds: DiscreteState) -> DiscreteState:
    return jnp.zeros_like(funds)


def additive(*, utility: FloatND, CE: FloatND) -> FloatND:
    return utility + CE


@pytest.mark.parametrize("enable_jit", [False, True])
def test_simulation_rejects_an_objective_with_no_feasible_action(
    *, enable_jit: bool
) -> None:
    """A feasible start lands in a state that admits no action in period one."""
    model = lcm.Model(
        n_periods=3,
        regimes={
            "decide": lcm.Regime(
                actions={"choice": lcm.DiscreteGrid(Choice)},
                states={"funds": lcm.DiscreteGrid(Funds)},
                state_transitions={
                    "funds": lcm.Phased(
                        solve=perceived_next_funds, simulate=realized_next_funds
                    )
                },
                constraints={"can_act": can_act},
                functions={"utility": first_choice_utility},
                koopmans_aggregator=lcm.Decoupled(objective=additive, value=additive),
            ),
            "end": lcm.Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
        edges={
            "decide": {
                "decide": lcm.Periods(values=(0,)),
                "end": lcm.Periods(values=(1,)),
            }
        },
        enable_jit=enable_jit,
    )
    with pytest.raises(NoFeasibleActionError, match=r"period 1.*no feasible action"):
        model.simulate(
            params={},
            initial_conditions={
                "funds": jnp.array([Funds.ready]),
                "regime_id": jnp.array([RegimeId.decide]),
                "period": jnp.zeros(1, dtype=jnp.int32),
            },
            log_level="off",
        )


def _readout_oracle(
    *, selection: np.ndarray, readout: np.ndarray, where: np.ndarray
) -> tuple[int | None, float]:
    """Scan the flat elements in order, as the contract states it.

    The index is `None` where the value is NaN: a poisoned cell has no choice.
    """
    flat_selection = np.ravel(selection)
    flat_readout = np.ravel(readout)
    flat_where = np.ravel(np.broadcast_to(where, np.shape(selection)))
    feasible = [k for k in range(flat_selection.size) if flat_where[k]]
    if not feasible:
        return -1, -np.inf
    if any(np.isnan(flat_selection[k]) or np.isnan(flat_readout[k]) for k in feasible):
        return None, np.nan
    best = feasible[0]
    for k in feasible[1:]:
        if flat_selection[k] > flat_selection[best]:
            best = k
    return best, float(flat_readout[best])


READOUT_CASES = {
    "scalar feasible": (2.0, 7.0, True),
    "scalar infeasible": (2.0, 7.0, False),
    "scalar infeasible NaN objective": (np.nan, 7.0, False),
    "all infeasible": ([2.0, 1.0], [7.0, 8.0], [False, False]),
    "first of two": ([2.0, 1.0], [7.0, 8.0], [True, True]),
    "tie picks first feasible": (
        [5.0, 5.0, 9.0],
        [101.0, 7.0, 99.0],
        [False, True, True],
    ),
    "only -inf feasible": ([-np.inf, -np.inf], [7.0, 8.0], [False, True]),
    "signed-zero tie": ([-0.0, +0.0], [7.0, 8.0], [True, True]),
    "2x2 grid": ([[3.0, 4.0], [4.0, 2.0]], [[1.0, 7.0], [8.0, 9.0]], [[True] * 2] * 2),
    "NaN value at a feasible unchosen action": (
        [2.0, 1.0],
        [7.0, np.nan],
        [True, True],
    ),
    "NaN value at an infeasible action": ([2.0, 1.0], [7.0, np.nan], [True, False]),
    "NaN objective at a feasible action": ([2.0, np.nan], [7.0, 8.0], [True, True]),
}


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("case", list(READOUT_CASES))
def test_readout_follows_the_empty_tie_and_nan_contract(
    *, case: str, enable_jit: bool
) -> None:
    """Index `-1` and `-inf` without a feasible action; a feasible NaN poisons."""
    import jax  # noqa: PLC0415

    from _lcm.regime_building.decoupled import argmax_and_readout  # noqa: PLC0415

    selection, readout, where = (np.asarray(part) for part in READOUT_CASES[case])
    function = jax.jit(argmax_and_readout) if enable_jit else argmax_and_readout
    index, value = function(
        selection=jnp.asarray(selection),
        readout=jnp.asarray(readout),
        where=jnp.asarray(where),
    )
    expected_index, expected_value = _readout_oracle(
        selection=selection, readout=readout, where=where
    )
    got = (
        None if np.isnan(value) else int(index),
        "nan" if np.isnan(value) else float(value),
    )
    assert got == (
        expected_index,
        "nan" if np.isnan(expected_value) else expected_value,
    )


def shocked_utility(*, choice: DiscreteAction, shock: FloatND) -> FloatND:
    return utility(choice) + shock


SHOCK_POINTS = 3


def _shocked_model(*, fold: bool) -> lcm.Model:
    return lcm.Model(
        n_periods=3,
        regimes={
            "decide": lcm.Regime(
                actions={"choice": lcm.DiscreteGrid(Choice)},
                states={
                    "wealth": lcm.DiscreteGrid(Wealth),
                    "shock": lcm.NormalIIDProcess(
                        n_points=SHOCK_POINTS,
                        gauss_hermite=True,
                        mu=0.0,
                        sigma=1.0,
                        fold=fold,
                    ),
                },
                state_transitions={"wealth": next_wealth},
                constraints={"can_afford": can_afford},
                functions={"utility": shocked_utility},
                koopmans_aggregator=PRESENT_BIAS,
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


@pytest.mark.parametrize("period", [0, 1])
def test_folded_state_averages_the_stored_value(*, period: int) -> None:
    """Folding the shock gives the unfolded stored value averaged over its nodes."""
    params = _params(beta_tilde=BETA)
    unfolded = _shocked_model(fold=False)
    folded = _shocked_model(fold=True)
    weights = np.asarray(
        cast(
            "lcm.NormalIIDProcess", unfolded.user_regimes["decide"].states["shock"]
        ).get_transition_probs()[0]
    )
    assert_array_almost_equal(
        folded.solve(params=params, log_level="off").value(
            period=period, regime="decide"
        ),
        np.asarray(
            unfolded.solve(params=params, log_level="off").value(
                period=period, regime="decide"
            )
        )
        @ weights,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [0, 1])
def test_invariant_blocking_solves_like_the_unblocked_model(*, period: int) -> None:
    """Solving one mood at a time gives the unblocked stored values."""
    params: UserParams = {"decide": {"koopmans_aggregator": {"delta": DELTA}}}
    blocked = _mood_model(
        execution_config=lcm.ExecutionConfig(
            devices=(0,), device_memory_bytes=None, invariant_block_widths={"mood": 1}
        )
    )
    assert_array_equal(
        blocked.solve(params=params, log_level="off").value(
            period=period, regime="decide"
        ),
        _mood_model()
        .solve(params=params, log_level="off")
        .value(period=period, regime="decide"),
    )


SCALE = 0.3


def _enumerate_ev1_values(*, beta_tilde: float) -> dict[int, np.ndarray]:
    """Solve the EV1 model by enumeration.

    The value is `scale * logsumexp(Q_dec / scale) + sum_k P_k (Q_k - Q_dec_k)`
    over the feasible choices, with `P = softmax(Q_dec / scale)`.
    """
    values = {2: TERMINAL_VALUES}
    for period in (1, 0):
        V = np.empty(4)
        for w in range(4):
            choices = [(0.0, min(w + 1, 3))]
            if w >= 1:
                choices.insert(0, (SPEND_UTILITY, w - 1))
            u = np.array([c[0] for c in choices])
            ce = np.array([values[period + 1][c[1]] for c in choices])
            perceived = u + beta_tilde * DELTA * ce
            long_run_q = u + DELTA * ce
            probabilities = np.exp(perceived / SCALE) / np.exp(perceived / SCALE).sum()
            V[w] = SCALE * np.log(np.exp(perceived / SCALE).sum()) + probabilities @ (
                long_run_q - perceived
            )
        values[period] = V
    return values


@pytest.mark.parametrize(
    ("beta_tilde", "expected"),
    [
        (1.0, [0.9, 1.893979, 3.605445, 3.693979]),
        (BETA, [0.9, 1.645361, 3.051903, 3.445361]),
        (0.7, [0.9, 1.778744, 3.498829, 3.578744]),
    ],
)
def test_ev1_enumeration_matches_the_hand_solution(
    *, beta_tilde: float, expected: list[float]
) -> None:
    """The EV1 oracle reproduces the values computed by hand at scale 0.3."""
    assert_array_almost_equal(
        _enumerate_ev1_values(beta_tilde=beta_tilde)[1], expected, decimal=6
    )


def _ev1_model(
    *, koopmans_aggregator: UserFunction | lcm.Decoupled = PRESENT_BIAS
) -> lcm.Model:
    model = _model(koopmans_aggregator=koopmans_aggregator)
    return lcm.Model(
        n_periods=3,
        regimes={
            "decide": model.user_regimes["decide"].replace(
                taste_shocks=lcm.ExtremeValueTasteShocks()
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


def _ev1_params(*, beta_tilde: float) -> UserParams:
    return {
        "decide": {
            "koopmans_aggregator": {
                "beta_tilde": beta_tilde,
                "beta": BETA,
                "delta": DELTA,
            },
            "taste_shocks": {"scale": SCALE},
        }
    }


@pytest.mark.parametrize("period", [0, 1])
@pytest.mark.parametrize("agent", list(AGENTS))
def test_ev1_solve_stores_the_expected_value_at_the_noisy_objective(
    *, agent: str, period: int
) -> None:
    """With taste shocks `V` is the expected `u + delta * CE` plus the chosen shock."""
    beta_tilde = AGENTS[agent]
    solution = _ev1_model().solve(
        params=_ev1_params(beta_tilde=beta_tilde), log_level="off"
    )
    assert_array_almost_equal(
        solution.value(period=period, regime="decide"),
        _enumerate_ev1_values(beta_tilde=beta_tilde)[period],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [0, 1])
def test_ev1_identical_objective_and_value_solve_like_the_plain_aggregator(
    *, period: int
) -> None:
    """With `objective = value` the EV1 solve equals the plain logsum."""
    params: UserParams = {
        "decide": {
            "koopmans_aggregator": {"delta": DELTA},
            "taste_shocks": {"scale": SCALE},
        }
    }
    wrapped = _ev1_model(
        koopmans_aggregator=lcm.Decoupled(objective=long_run, value=long_run)
    ).solve(params=params, log_level="off")
    plain = _ev1_model(koopmans_aggregator=long_run).solve(
        params=params, log_level="off"
    )
    assert_array_almost_equal(
        wrapped.value(period=period, regime="decide"),
        plain.value(period=period, regime="decide"),
        decimal=DECIMAL_PRECISION,
    )


def _naive_today_probabilities() -> np.ndarray:
    """Today's naive self at period 0 and wealth one: spend versus save."""
    V_1 = _enumerate_ev1_values(beta_tilde=1.0)[1]
    current = np.array([SPEND_UTILITY + BETA * DELTA * V_1[0], BETA * DELTA * V_1[2]])
    return np.exp(current / SCALE) / np.exp(current / SCALE).sum()


def test_ev1_lookup_publishes_the_acting_selfs_choice_probabilities() -> None:
    """Probabilities are `softmax(Q_objective / scale)` with today's beta."""
    model = _ev1_model()
    params = _ev1_params(beta_tilde=1.0)
    got = model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([Wealth.one])},
    )
    assert_array_almost_equal(
        cast("FloatND", got.choice_probabilities)[0],
        _naive_today_probabilities(),
        decimal=DECIMAL_PRECISION,
    )


def test_ev1_simulated_choices_follow_the_acting_selfs_probabilities() -> None:
    """Simulated spend frequency at wealth one is within four standard errors."""
    n_subjects = 4_000
    df = (
        _ev1_model()
        .simulate(
            params=_ev1_params(beta_tilde=1.0),
            initial_conditions={
                "wealth": jnp.full(n_subjects, Wealth.one),
                "regime_id": jnp.full(n_subjects, RegimeId.decide),
                "period": jnp.zeros(n_subjects, dtype=jnp.int32),
            },
            log_level="off",
            seed=11,
        )
        .to_dataframe()
    )
    spend = _naive_today_probabilities()[0]
    share = (df.loc[df["period"] == 0, "choice"] == "spend").mean()
    assert abs(share - spend) < 4 * np.sqrt(spend * (1 - spend) / n_subjects)
