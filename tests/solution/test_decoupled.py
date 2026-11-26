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
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import lcm
from lcm.exceptions import (
    InvalidValueFunctionError,
    ModelInitializationError,
    NoFeasibleActionError,
    RegimeInitializationError,
)
from lcm.solvers import GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Intermediate,
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


def zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def indifferent_utility(choice: DiscreteAction) -> FloatND:
    return jnp.zeros_like(jnp.asarray(choice, dtype=float))


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
        solution.value(step=period, regime="decide")[..., 0],
        _enumerate_values(beta_tilde=beta_tilde)[period],
        decimal=DECIMAL_PRECISION,
    )


def test_sophisticated_value_lies_below_the_best_long_run_value() -> None:
    """At wealth two the self spends, worth 2.85, although saving is worth 2.97."""
    solution = _model().solve(params=_params(beta_tilde=BETA), log_level="off")
    assert_array_almost_equal(
        solution.value(step=0, regime="decide")[Wealth.two],
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
        wrapped.value(step=period, regime="decide"),
        plain.value(step=period, regime="decide"),
    )


@pytest.mark.parametrize("period", [0, 1])
def test_tied_objective_takes_the_action_of_larger_stored_value(*, period: int) -> None:
    """An objective tied across actions picks the action whose stored value is larger.

    The household tie rule ranks tied actions by the member's own value, here
    `long_run`; spending needs wealth, saving adds one unit up to the top.
    """
    solution = _model(
        koopmans_aggregator=lcm.Decoupled(objective=indifferent, value=long_run)
    ).solve(
        params={"decide": {"koopmans_aggregator": {"delta": DELTA}}},
        log_level="off",
    )
    next_values = TERMINAL_VALUES if period == 1 else np.array([0.9, 1.8, 3.6, 3.6])
    save = [DELTA * next_values[min(w + 1, 3)] for w in range(4)]
    expected = [save[0]] + [
        max(SPEND_UTILITY + DELTA * next_values[w - 1], save[w]) for w in (1, 2, 3)
    ]
    assert_array_almost_equal(
        solution.value(step=period, regime="decide")[..., 0],
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


ALL_INTERMEDIATES: tuple[Intermediate, ...] = ("Q", "F", "U", "CE", "Q_objective")


def _lookup(*, keep_intermediates: tuple[Intermediate, ...]) -> lcm.PolicyLookup:
    model = _model()
    params = _params(beta_tilde=1.0)
    return model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([0, 1, 2, 3])},
        keep_intermediates=keep_intermediates,
    )


@pytest.mark.parametrize("keep_intermediates", [(), ALL_INTERMEDIATES])
def test_lookup_chooses_with_the_simulate_variant(
    *, keep_intermediates: tuple[Intermediate, ...]
) -> None:
    """Lookup chooses like simulation, with today's `β`, not the solve's `β̃`."""
    got = _lookup(keep_intermediates=keep_intermediates)
    actions = cast("Mapping[str, IntND]", got.actions)
    assert_array_equal(
        actions["choice"], [Choice.save, Choice.spend, Choice.spend, Choice.spend]
    )


@pytest.mark.parametrize("keep_intermediates", [(), ALL_INTERMEDIATES])
def test_lookup_publishes_the_stored_value_of_the_choice(
    *, keep_intermediates: tuple[Intermediate, ...]
) -> None:
    """The looked-up value is `u + δ·CE` at today's choice."""
    got = _lookup(keep_intermediates=keep_intermediates)
    assert_array_almost_equal(
        got.value[..., 0], NAIVE_LONG_RUN_VALUES, decimal=DECIMAL_PRECISION
    )


def test_lookup_publishes_the_objective_profile() -> None:
    """`Q_objective` is today's `u + βδ·CE` per action; `Q` is `u + δ·CE`."""
    got = _lookup(keep_intermediates=ALL_INTERMEDIATES)
    assert_array_almost_equal(
        cast("FloatND", got.Q_objective),
        cast("FloatND", got.U) + BETA * DELTA * cast("FloatND", got.CE),
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_keeps_the_objective_profile_only_when_asked() -> None:
    """`Q_objective` alone keeps no other part; the other parts alone omit it."""
    alone = _lookup(keep_intermediates=("Q_objective",))
    without = _lookup(keep_intermediates=("Q", "F", "U", "CE"))
    assert alone.Q_objective is not None
    assert alone.Q is None
    assert without.Q_objective is None
    assert_array_almost_equal(
        alone.Q_objective,
        cast("FloatND", _lookup(keep_intermediates=ALL_INTERMEDIATES).Q_objective),
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
        keep_intermediates=ALL_INTERMEDIATES,
    )
    assert got.Q_objective is None


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


def _household_model(single: lcm.Model) -> lcm.Model:
    """`single`'s decision regime as a household that `f` decides.

    `f` carries all the Pareto weight; `m` gains nothing from spending, so its
    value is the discounted terminal value its share of `f`'s choices lead to.
    """
    regime = single.user_regimes["decide"]
    return lcm.Model(
        n_periods=3,
        regimes={
            "decide": regime.replace(
                functions={
                    "utility": lcm.CollectiveUtility(
                        utilities={"f": utility, "m": indifferent_utility},
                        objective=lcm.ParetoObjective(weights={"f": 1.0, "m": 0.0}),
                    )
                }
            ),
            "end": lcm.Regime(
                states={"wealth": lcm.DiscreteGrid(Wealth)},
                functions={
                    "utility": lcm.CollectiveUtility(
                        utilities={"f": terminal_utility, "m": terminal_utility}
                    )
                },
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
def test_a_household_chooses_on_its_weighted_objectives(*, period: int) -> None:
    """The deciding member's choices and values are the single's."""
    single = _model()
    household = _household_model(single)
    params = _params(beta_tilde=0.7)
    values = household.solve(params=params, log_level="off").value(
        step=period, regime="decide"
    )
    assert_array_almost_equal(
        values[..., 0],
        single.solve(params=params, log_level="off").value(
            step=period, regime="decide"
        )[..., 0],
        decimal=DECIMAL_PRECISION,
    )
    df = _first_period(_simulate(model=household, params=params))
    assert list(df["choice"]) == list(
        _first_period(_simulate(model=single, params=params))["choice"]
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
        solution.value(step=1, regime="decide")[..., 0],
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
        solution.value(step=1, regime="decide")[:, Mood.patient, 0],
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
        solution.value(step=1, regime="decide")[:, Mood.impatient, 0],
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
            "end": lcm.Regime(functions={"utility": zero_utility}),
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
        # Device-memory admission needs compiled peaks, which an eager model lacks.
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
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

    A tie in the objective goes to the larger stored value, then to the first
    element. The index is `None` where the value is NaN: a poisoned cell has no
    choice.
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
        if (flat_selection[k], flat_readout[k]) > (
            flat_selection[best],
            flat_readout[best],
        ):
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
    """A single's readout: index `-1` and `-inf` without a feasible action, and a
    feasible NaN poisons."""
    import jax  # noqa: PLC0415

    from _lcm.regime_building.decoupled import (  # noqa: PLC0415
        household_argmax_and_readout,
    )

    selection, readout, where = (np.asarray(part) for part in READOUT_CASES[case])
    function = (
        jax.jit(household_argmax_and_readout)
        if enable_jit
        else household_argmax_and_readout
    )
    index, values = function(
        selection=jnp.asarray(selection)[..., jnp.newaxis],
        readout=jnp.asarray(readout)[..., jnp.newaxis],
        where=jnp.asarray(where),
        weights=jnp.ones(1),
    )
    value = values[0]
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


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 1, 0)])
def test_household_ties_in_the_objectives_go_to_the_stored_values(
    *, order: tuple[int, ...]
) -> None:
    """Actions 0 and 1 tie in the household objective; f's higher stored value decides.

    Specification, section 13.5: ties in the objective go to the members by their
    stored values, in member order. The members' own objectives do not enter the
    tie, so f's larger objective at action 0 does not decide.
    """
    from _lcm.regime_building.decoupled import (  # noqa: PLC0415
        household_argmax_and_readout,
    )

    selection = jnp.asarray([[2.0, 1.0], [1.0, 2.0], [0.0, 0.0]])[jnp.asarray(order)]
    readout = jnp.asarray([[3.0, 9.0], [4.0, 1.0], [8.0, 8.0]])[jnp.asarray(order)]
    _, value = household_argmax_and_readout(
        selection=selection,
        readout=readout,
        where=jnp.ones(3, dtype=bool),
        weights=jnp.asarray([0.5, 0.5]),
    )
    assert_array_equal(value, [4.0, 1.0])


@lcm.categorical(ordered=False)
class Alternative:
    first: ScalarInt
    second: ScalarInt


def no_utility(
    *, alternative: DiscreteAction, c: ContinuousAction, s: ContinuousState
) -> FloatND:
    return jnp.zeros_like(alternative + c + s)


def utility_falls_in_c(
    *, alternative: DiscreteAction, c: ContinuousAction, s: ContinuousState
) -> FloatND:
    return 1.0 - c + 0.0 * (alternative + s)


def lands_at_c(*, c: ContinuousAction) -> ContinuousState:
    return c


def end_zero(*, s: ContinuousState) -> FloatND:
    return 0.0 * s


def end_s(*, s: ContinuousState) -> FloatND:
    return s


def end_one_plus_s(*, s: ContinuousState) -> FloatND:
    return 1.0 + s


def end_one_minus_s(*, s: ContinuousState) -> FloatND:
    return 1.0 - s


def objective_is_utility(*, utility: FloatND, CE: FloatND) -> FloatND:
    return utility + 0.0 * CE


def value_is_CE(*, utility: FloatND, CE: FloatND) -> FloatND:
    return CE + 0.0 * utility


TIE_SCALE = 0.5

# Per case: the deciding regime's utilities, the terminal utilities, whether
# taste shocks are declared, the solved values at `s = 0` in member order, and
# the simulated `c`. The objective is the period utility and the stored value
# the continuation, which `c` moves to `s`; every objective ties across `c`.
TIE_CASES: dict[
    str,
    tuple[
        Mapping[str, UserFunction], Mapping[str, UserFunction], bool, list[float], float
    ],
] = {
    # Spec 14.5a, step 3: one member, taste shocks. Its stored values are 1 and
    # 2, so c = 1 in both alternatives, each with probability one half.
    "one member, taste shocks": (
        {"self": no_utility},
        {"self": end_one_plus_s},
        True,
        [2.0 + TIE_SCALE * np.log(2.0)],
        1.0,
    ),
    # Spec 8.5, 13.5: f decides; m's objective prefers c = 0, his stored
    # value c = 1. His stored value decides.
    "the other member's stored value": (
        {"f": no_utility, "m": utility_falls_in_c},
        {"f": end_zero, "m": end_s},
        False,
        [0.0, 1.0],
        1.0,
    ),
    "the other member's stored value, taste shocks": (
        {"f": no_utility, "m": utility_falls_in_c},
        {"f": end_zero, "m": end_s},
        True,
        [TIE_SCALE * np.log(2.0), 1.0],
        1.0,
    ),
    # Spec 14.5a: the other members come before the decider. m's stored value
    # prefers c = 0, f's own c = 1.
    "the other member before the decider": (
        {"f": no_utility, "m": no_utility},
        {"f": end_s, "m": end_one_minus_s},
        False,
        [0.0, 1.0],
        0.0,
    ),
}


def _tie_model(
    *,
    utilities: Mapping[str, UserFunction],
    end_utilities: Mapping[str, UserFunction],
    taste_shocks: bool,
) -> lcm.Model:
    """`decide` at period 0 moves `s` to its action `c`; `end` pays `s` by member."""
    s = lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    single = list(utilities) == ["self"]
    weights = {"f": 1.0, "m": 0.0}
    return lcm.Model(
        n_periods=2,
        regimes={
            "decide": lcm.Regime(
                actions={
                    "alternative": lcm.DiscreteGrid(Alternative),
                    "c": lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                },
                states={"s": s},
                state_transitions={"s": lands_at_c},
                functions={
                    "utility": utilities["self"]
                    if single
                    else lcm.CollectiveUtility(
                        utilities=dict(utilities),
                        objective=lcm.ParetoObjective(weights=weights),
                    )
                },
                koopmans_aggregator=lcm.Decoupled(
                    objective=objective_is_utility, value=value_is_CE
                ),
                taste_shocks=lcm.ExtremeValueTasteShocks() if taste_shocks else None,
            ),
            "end": lcm.Regime(
                states={"s": s},
                functions={
                    "utility": end_utilities["self"]
                    if single
                    else lcm.CollectiveUtility(utilities=dict(end_utilities))
                },
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
        edges={"decide": {"end": lcm.Periods(values=(0,))}},
    )


@pytest.mark.parametrize("case", list(TIE_CASES))
def test_objective_ties_go_to_the_other_members_then_to_the_decider(
    *, case: str
) -> None:
    """Ties in the objective go to the stored values: others first, the decider last.

    Specification, sections 8.5, 13.5 and 14.5a: among the actions with the
    highest objective of the deciding member, those with the highest stored
    value of each other member in member order, then those with the highest
    stored value of the deciding member. With taste shocks the rule picks the
    other actions inside each alternative. Solve and simulation agree.
    """
    utilities, end_utilities, taste_shocks, expected, expected_c = TIE_CASES[case]
    model = _tie_model(
        utilities=utilities, end_utilities=end_utilities, taste_shocks=taste_shocks
    )
    params: UserParams = (
        {"decide": {"taste_shocks": {"scale": TIE_SCALE}}} if taste_shocks else {}
    )
    solution = model.solve(params=params, log_level="off")
    members = tuple(utilities)
    conditions = {
        "s": jnp.zeros(len(members)),
        "regime_id": jnp.full(len(members), RegimeId.decide),
        "period": jnp.zeros(len(members), dtype=jnp.int32),
    }
    if members != ("self",):
        conditions["own_member"] = jnp.asarray(
            [model.member_names_to_ids[member] for member in members],
            dtype=jnp.int32,
        )
    df = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=conditions,
        log_level="off",
        seed=0,
    ).to_dataframe()

    assert_array_almost_equal(
        solution.value(step=0, regime="decide")[0],
        expected,
        decimal=DECIMAL_PRECISION,
    )
    assert_array_equal(df.loc[df["period"] == 0, "c"], expected_c)


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
            step=period, regime="decide"
        ),
        _mood_model()
        .solve(params=params, log_level="off")
        .value(step=period, regime="decide"),
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
        solution.value(step=period, regime="decide")[..., 0],
        _enumerate_ev1_values(beta_tilde=beta_tilde)[period],
        decimal=DECIMAL_PRECISION,
    )


def _enumerate_ev1_household_values(*, beta_tilde: float) -> dict[int, np.ndarray]:
    """Her EV1 values beside his, in member order.

    Hers are the single's. His flow utility is zero, so his value is his
    discounted continuation value, averaged under her choice probabilities.
    """
    hers = _enumerate_ev1_values(beta_tilde=beta_tilde)
    his = {2: TERMINAL_VALUES}
    for period in (1, 0):
        V = np.empty(4)
        for w in range(4):
            u = np.array([0.0] if w == 0 else [SPEND_UTILITY, 0.0])
            nexts = [min(w + 1, 3)] if w == 0 else [w - 1, min(w + 1, 3)]
            perceived = u + beta_tilde * DELTA * hers[period + 1][nexts]
            probabilities = np.exp(perceived / SCALE) / np.exp(perceived / SCALE).sum()
            V[w] = probabilities @ (DELTA * his[period + 1][nexts])
        his[period] = V
    return {period: np.stack([hers[period], his[period]], axis=-1) for period in (0, 1)}


@pytest.mark.parametrize("period", [0, 1])
def test_ev1_household_publishes_the_deciders_value_and_the_others_mean(
    *, period: int
) -> None:
    """Her value is the single's; his is his stored value under her probabilities."""
    params = _ev1_params(beta_tilde=0.7)

    values = (
        _household_model(_ev1_model())
        .solve(params=params, log_level="off")
        .value(step=period, regime="decide")
    )

    assert_array_almost_equal(
        values,
        _enumerate_ev1_household_values(beta_tilde=0.7)[period],
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
        wrapped.value(step=period, regime="decide"),
        plain.value(step=period, regime="decide"),
        decimal=DECIMAL_PRECISION,
    )


def _naive_today_probabilities() -> np.ndarray:
    """Today's naive self at period 0 and wealth one: spend versus save."""
    V_1 = _enumerate_ev1_values(beta_tilde=1.0)[1]
    current = np.array([SPEND_UTILITY + BETA * DELTA * V_1[0], BETA * DELTA * V_1[2]])
    return np.exp(current / SCALE) / np.exp(current / SCALE).sum()


def test_ev1_lookup_publishes_the_acting_selfs_choice_probabilities() -> None:
    """Probabilities are `softmax(Q_objective / scale)` with today's beta.

    `Qc` is the stored long-run value at each choice and `value` its expectation
    at the noisy choice plus `scale` times the entropy, which differs from a logsum
    of `Qc` because the objective and the stored value differ at both choices.
    """
    model = _ev1_model()
    params = _ev1_params(beta_tilde=1.0)
    got = model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([Wealth.one])},
    )
    continuation = _enumerate_ev1_values(beta_tilde=1.0)[1]
    expected_qc = np.array(
        [SPEND_UTILITY + DELTA * continuation[0], DELTA * continuation[2]]
    )
    probabilities = _naive_today_probabilities()
    expected_value = probabilities @ expected_qc - SCALE * np.sum(
        probabilities * np.log(probabilities)
    )
    assert_array_almost_equal(
        cast("FloatND", got.choice_probabilities)[0],
        probabilities,
        decimal=DECIMAL_PRECISION,
    )
    assert_array_almost_equal(
        cast("FloatND", got.Qc)[0, :, 0], expected_qc, decimal=DECIMAL_PRECISION
    )
    assert_array_almost_equal(
        got.value[..., 0], [expected_value], decimal=DECIMAL_PRECISION
    )


def test_ev1_simulated_choices_follow_the_acting_selfs_probabilities() -> None:
    """Simulated spend frequency at wealth one is within four standard errors."""
    n_agents = 4_000
    df = (
        _ev1_model()
        .simulate(
            params=_ev1_params(beta_tilde=1.0),
            initial_conditions={
                "wealth": jnp.full(n_agents, Wealth.one),
                "regime_id": jnp.full(n_agents, RegimeId.decide),
                "period": jnp.zeros(n_agents, dtype=jnp.int32),
            },
            log_level="off",
            seed=11,
        )
        .to_dataframe()
    )
    spend = _naive_today_probabilities()[0]
    share = (df.loc[df["period"] == 0, "choice"] == "spend").mean()
    assert abs(share - spend) < 4 * np.sqrt(spend * (1 - spend) / n_agents)


@lcm.categorical(ordered=False)
class Gate:
    blocked: ScalarInt
    allowed: ScalarInt


def gate_utility(choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Gate.blocked, 7.0, 1.0)


def gate_is_open(choice: DiscreteAction) -> BoolND:
    return choice == Gate.allowed


def translated_objective(*, utility: FloatND, CE: FloatND, offset: float) -> FloatND:
    """Every action's objective is `offset`: a translation that moves no choice."""
    return jnp.full_like(utility + CE, offset)


def _gate_model(*, objective: UserFunction, enable_jit: bool) -> lcm.Model:
    """One EV1 decision with one feasible action (`R = 1`), then nothing."""
    return lcm.Model(
        n_periods=2,
        regimes={
            "decide": lcm.Regime(
                actions={"choice": lcm.DiscreteGrid(Gate)},
                functions={"utility": gate_utility},
                constraints={"gate_is_open": gate_is_open},
                koopmans_aggregator=lcm.Decoupled(objective=objective, value=additive),
                taste_shocks=lcm.ExtremeValueTasteShocks(),
            ),
            "end": lcm.Regime(functions={"utility": zero_utility}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
        edges={"decide": {"end": lcm.Periods(values=(0,))}},
        enable_jit=enable_jit,
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )


def _large_offset() -> float:
    """An objective level whose unit in the last place exceeds the stored value."""
    return float(2 ** (26 if jnp.asarray(0.0).dtype == jnp.float32 else 55))


@pytest.mark.parametrize("keep_intermediates", [(), ALL_INTERMEDIATES])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_ev1_infeasible_combination_publishes_minus_infinity(
    *, enable_jit: bool, keep_intermediates: tuple[Intermediate, ...]
) -> None:
    """`Qc` is `-inf` where no action is feasible; raw `Q` stays unmasked."""
    model = _gate_model(objective=additive, enable_jit=enable_jit)
    params: UserParams = {"decide": {"taste_shocks": {"scale": 1.0}}}
    got = model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="decide",
        states={},
        keep_intermediates=keep_intermediates,
    )
    Qc = np.asarray(got.Qc)[0]
    assert np.isneginf(Qc[Gate.blocked])
    assert Qc[Gate.allowed] == 1.0
    assert_array_equal(np.asarray(got.choice_probabilities)[0], [0.0, 1.0])
    if keep_intermediates:
        assert np.asarray(got.Q)[0, Gate.blocked] == 7.0


@pytest.mark.parametrize("enable_jit", [False, True])
def test_ev1_stored_value_survives_a_large_objective_translation(
    *, enable_jit: bool
) -> None:
    """The one feasible action's value `1` survives an objective of `2**26`."""
    model = _gate_model(objective=translated_objective, enable_jit=enable_jit)
    params: UserParams = {
        "decide": {
            "taste_shocks": {"scale": 1.0},
            "koopmans_aggregator": {"offset": _large_offset()},
        }
    }
    solution = model.solve(params=params, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="decide",
        states={},
    )
    assert float(np.asarray(solution.value(step=0, regime="decide"))[..., 0]) == 1.0
    assert_array_equal(np.asarray(lookup.value)[..., 0], [1.0])


def _entropy_oracle(
    *, objective: np.ndarray, value: np.ndarray, feasible: np.ndarray, scale: float
) -> float:
    """`sum_k P_k R_k + scale * H(P)` in float64, `P = softmax(objective / scale)`."""
    best = np.where(feasible, objective, -np.inf)
    index = np.argmax(best, axis=-1)
    o = np.take_along_axis(best, index[..., None], axis=-1)[..., 0]
    r = np.take_along_axis(value, index[..., None], axis=-1)[..., 0]
    live = np.isfinite(o)
    z = np.where(live, (o - o[live].max()) / scale, -np.inf)
    p = np.exp(z) / np.exp(z).sum()
    log_p = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
    return float(np.sum(np.where(p > 0, p * r, 0.0)) - scale * np.sum(p * log_p))


@pytest.mark.parametrize("seed", range(8))
def test_ev1_stored_value_is_invariant_to_objective_translations(*, seed: int) -> None:
    """Generated menus: translating the objective leaves the stored value unchanged.

    Objectives sit on a quarter grid within `[-4, 4]`, so `objective + 2**20`
    is exact in float32 and the translated choice problem is the same one.
    """
    rng = np.random.default_rng(seed)
    shape = (4, 3)
    objective = rng.integers(-16, 17, size=shape) / 4.0
    value = rng.normal(size=shape) * 3.0
    feasible = rng.uniform(size=shape) > 0.3
    feasible[0, 0] = True
    scale = float(rng.choice([0.25, 1.0, 3.0]))
    expected = _entropy_oracle(
        objective=objective, value=value, feasible=feasible, scale=scale
    )
    tolerance = (
        50
        * float(jnp.finfo(jnp.asarray(0.0).dtype).eps)
        * (1.0 + np.abs(value).max() + scale * np.log(value.size))
    )
    for offset in (0.0, 2.0**20):
        got = _ev1_cell_value(
            objective=jnp.asarray(objective + offset),
            value=jnp.asarray(value),
            feasible=jnp.asarray(feasible),
            scale=scale,
        )
        assert abs(float(got) - expected) <= tolerance, (offset, float(got), expected)


def _ev1_cell_value(
    *, objective: FloatND, value: FloatND, feasible: BoolND, scale: float
) -> FloatND:
    """A single's stored EV1 value of a cell, its leading axis the discrete one."""
    from _lcm.regime_building.decoupled import (  # noqa: PLC0415
        smoothed_logsum,
        smoothed_profiles,
    )
    from _lcm.regime_building.Q_and_F import QAndFParts  # noqa: PLC0415

    parts = QAndFParts(
        U=value[..., jnp.newaxis],
        F=feasible,
        CE=jnp.zeros_like(value)[..., jnp.newaxis],
        Q=value[..., jnp.newaxis],
        Q_objective=objective[..., jnp.newaxis],
    )
    stored, _ = smoothed_logsum(
        smoothed=smoothed_profiles(
            parts=parts, weights=jnp.ones(1), n_discrete_action_axes=1
        ),
        scale=jnp.asarray(scale),
    )
    return stored[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([2.0, -np.inf], -np.inf),
        ([2.0, np.inf], np.inf),
        ([-np.inf, np.inf], np.nan),
    ],
    ids=["minus-inf", "plus-inf", "both"],
)
def test_ev1_underflowed_feasible_infinite_value_is_kept(
    *, value: list[float], expected: float
) -> None:
    """A feasible action has positive EV1 mass even when its probability underflows."""
    got = float(
        _ev1_cell_value(
            objective=jnp.asarray([0.0, -1000.0]),
            value=jnp.asarray(value),
            feasible=jnp.asarray([True, True]),
            scale=1.0,
        )
    )
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == expected


def _phase_aggregators(
    declaration: UserFunction | lcm.Phased | lcm.Decoupled | None,
) -> dict[
    str,
    tuple[
        UserFunction | MappingProxyType[str, UserFunction] | None, UserFunction | None
    ],
]:
    regime = lcm.Regime(
        functions={"utility": zero_utility},
        koopmans_aggregator=declaration,
    )
    return {
        phase: (
            regime.get_koopmans_aggregator(phase=phase),
            regime.get_objective_aggregator(phase=phase),
        )
        for phase in ("solve", "simulate")
    }


def test_aggregator_getters_resolve_every_declaration_form() -> None:
    """Each phase gets the value aggregator and, for `Decoupled`, its objective."""
    assert _phase_aggregators(None) == {
        "solve": (None, None),
        "simulate": (None, None),
    }
    assert _phase_aggregators(long_run) == {
        "solve": (long_run, None),
        "simulate": (long_run, None),
    }
    assert _phase_aggregators(
        lcm.Phased(solve=perceived_future_choice, simulate=current_choice)
    ) == {
        "solve": (perceived_future_choice, None),
        "simulate": (current_choice, None),
    }
    assert _phase_aggregators(PRESENT_BIAS) == {
        "solve": (long_run, perceived_future_choice),
        "simulate": (long_run, current_choice),
    }
    assert _phase_aggregators(
        lcm.Decoupled(objective=current_choice, value=long_run)
    ) == {
        "solve": (long_run, current_choice),
        "simulate": (long_run, current_choice),
    }


def value_with_own(
    *, utility: FloatND, CE: FloatND, delta: float, value_only: float
) -> FloatND:
    return utility + delta * CE + 0.0 * value_only


def solve_objective_with_own(
    *, utility: FloatND, CE: FloatND, delta: float, solve_only: float
) -> FloatND:
    return utility + solve_only * delta * CE


def simulate_objective_with_own(
    *, utility: FloatND, CE: FloatND, delta: float, simulate_only: float
) -> FloatND:
    return utility + simulate_only * delta * CE


def test_params_template_is_one_namespace_over_all_three_aggregators() -> None:
    """Names of the value and both objectives form one namespace; shared is one."""
    model = _model(
        koopmans_aggregator=lcm.Decoupled(
            objective=lcm.Phased(
                solve=solve_objective_with_own, simulate=simulate_objective_with_own
            ),
            value=value_with_own,
        )
    )
    assert set(model.get_params_template()["decide"]["koopmans_aggregator"]) == {
        "delta",
        "value_only",
        "solve_only",
        "simulate_only",
    }


@lcm.categorical(ordered=False)
class _CompositionActionRegimeId:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt
    end: ScalarInt
    end_f: ScalarInt
    end_m: ScalarInt


@lcm.categorical(ordered=True)
class _Weight:
    low: ScalarInt
    high: ScalarInt


def _her_utility(
    *, c: ContinuousAction, s: ContinuousState, mu: DiscreteState
) -> FloatND:
    return 1.0 - c + 0.0 * s + 0.1 * mu


def _his_utility(
    *, c: ContinuousAction, s: ContinuousState, mu: DiscreteState
) -> FloatND:
    return c + 0.0 * s - 0.1 * mu


def _single_utility(*, c: ContinuousAction, s: ContinuousState) -> FloatND:
    return 0.5 * c + 0.0 * s - 1.0


def _keep_s(*, s: ContinuousState) -> ContinuousState:
    return s


def _s_is_c(*, c: ContinuousAction) -> ContinuousState:
    return c


def _low_mu(*, s: ContinuousState) -> DiscreteState:
    return jnp.zeros_like(s, dtype=jnp.int32)


def _end_s_mu(*, s: ContinuousState, mu: DiscreteState) -> FloatND:
    return s + 0.0 * mu


def _half_continuation(*, utility: FloatND, CE: FloatND) -> FloatND:
    return utility + 0.5 * CE


def _paid(*, t: FloatND) -> FloatND:
    return t


def _no_payment(*, s: ContinuousState) -> FloatND:
    return 0.0 * s


def _rule_composition_action(rule: str) -> lcm.CompositionAction:
    """Return the composition action of `rule`; `single_f` declares a formation."""
    if rule == "formation":
        return lcm.CompositionAction(
            destinations={
                "self": lcm.Destination(
                    regime="couple", role="f", projection={"s": _keep_s, "mu": _low_mu}
                )
            },
            rule=lcm.Consent(
                counterparty=lcm.Counterparty(
                    regime="single_m", role="m", projection={"s": _keep_s}
                )
            ),
        )
    targets = {
        member: lcm.Destination(
            regime=f"single_{member}",
            projection={"s": _keep_s},
            utility_adjustment=_paid if rule == "transfer" and member == "f" else None,
        )
        for member in ("f", "m")
    }
    rules = {
        "unilateral": lcm.PCOnActions(),
        "renegotiation": lcm.PCOnWeight(state="mu"),
        "transfer": lcm.MutualConsent(
            transfer=lcm.Transfer(
                name="t",
                candidates=lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                default=_no_payment,
            )
        ),
    }
    return lcm.CompositionAction(rule=rules[rule], destinations=targets)


def _composition_action_model(
    *, rule: str, aggregator: UserFunction | lcm.Phased | lcm.Decoupled
) -> lcm.Model:
    """Two periods; the regime with the move of `rule` declares `aggregator`.

    `couple` (f, m) composition actions to `single_f` and `single_m`, except under a
    formation, where `single_f` forms `couple` with `single_m`.
    """
    s = lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    c = lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    forms = rule == "formation"

    def single(*, mover: bool) -> lcm.Regime:
        return lcm.Regime(
            actions={"c": c},
            states={"s": s},
            state_transitions={"s": _s_is_c},
            functions={"utility": _single_utility},
            koopmans_aggregator=aggregator if mover else additive,
            composition_action=_rule_composition_action(rule) if mover else None,
        )

    return lcm.Model(
        n_periods=2,
        regimes={
            "couple": lcm.Regime(
                actions={"c": c},
                states={"s": s, "mu": lcm.DiscreteGrid(_Weight)},
                state_transitions={"s": _s_is_c, "mu": lcm.fixed_transition("mu")},
                functions={
                    "utility": lcm.CollectiveUtility(
                        utilities={"f": _her_utility, "m": _his_utility}
                    )
                },
                koopmans_aggregator=additive if forms else aggregator,
                composition_action=None if forms else _rule_composition_action(rule),
            ),
            "single_f": single(mover=forms),
            "single_m": single(mover=False),
            "end": lcm.Regime(
                states={"s": s, "mu": lcm.DiscreteGrid(_Weight)},
                functions={
                    "utility": lcm.CollectiveUtility(
                        utilities={"f": _end_s_mu, "m": _end_s_mu}
                    )
                },
            ),
            "end_f": lcm.Regime(states={"s": s}, functions={"utility": end_s}),
            "end_m": lcm.Regime(states={"s": s}, functions={"utility": end_s}),
        },
        regime_id_class=_CompositionActionRegimeId,
        initial_nodes=lcm.InitialNodes(
            by_period={0: ("couple", "single_f", "single_m")}
        ),
        edges={
            "couple": {"end": lcm.Periods(values=(0,))},
            "single_f": {"end_f": lcm.Periods(values=(0,))},
            "single_m": {"end_m": lcm.Periods(values=(0,))},
        },
    )


_RULES = ("unilateral", "renegotiation", "formation", "transfer")


@pytest.mark.parametrize("rule", _RULES)
def test_a_separate_objective_is_refused_on_a_regime_with_its_own_composition_action(
    *, rule: str
) -> None:
    """A regime with collective rules of its own declares no separate objective.

    The rules are a composition action, a renegotiation, a transfer or a formation
    (specification 13.5, [A 20]).

    The refusal comes when the model is built and names the regime, the two
    declarations and the way out.
    """
    mover = "single_f" if rule == "formation" else "couple"
    with pytest.raises(
        RegimeInitializationError,
        match=(
            rf"Regime '{mover}' declares a separate objective \(`Decoupled`\) and "
            r"its own composition action or formation\. A separate objective is for "
            r"regimes without a composition action or a formation of their own: drop "
            r"the separate objective, or let the current self choose differently by "
            r"declaring "
            r"`koopmans_aggregator=Phased\(solve=\.\.\., simulate=\.\.\.\)`\."
        ),
    ):
        _composition_action_model(
            rule=rule,
            aggregator=lcm.Decoupled(objective=_half_continuation, value=additive),
        )


@pytest.mark.parametrize("rule", _RULES)
def test_a_current_self_solves_and_simulates_with_a_composition_action(
    *, rule: str
) -> None:
    """The current self (specification 13.2) is not limited to regimes without an
    composition action (specification 13.5, [A 20]).

    Its solve is that of the solve-phase aggregator, and the simulation runs.
    """
    current_self = _composition_action_model(
        rule=rule, aggregator=lcm.Phased(solve=additive, simulate=_half_continuation)
    )
    solution = current_self.solve(params={}, log_level="off")
    plain = _composition_action_model(rule=rule, aggregator=additive).solve(
        params={}, log_level="off"
    )
    for regime in ("couple", "single_f", "single_m"):
        assert_array_equal(
            solution.value(step=0, regime=regime),
            plain.value(step=0, regime=regime),
        )
    start = (
        _CompositionActionRegimeId.single_f
        if rule == "formation"
        else _CompositionActionRegimeId.couple
    )
    frame = current_self.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "s": jnp.zeros(2),
            "mu": jnp.zeros(2, dtype=jnp.int32),
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.full(2, start),
            "own_member": jnp.asarray(
                [current_self.member_names_to_ids[m] for m in ("f", "m")],
                dtype=jnp.int32,
            ),
        },
        log_level="off",
        seed=0,
    ).to_dataframe(use_categorical_codes=True)
    first = frame.loc[frame["period"] == 0]
    assert len(first) == 2
    assert np.isfinite(first[["value_f", "value_m"]].to_numpy()).all()


def _tied_objective_utility(
    *, c: ContinuousAction, alternative: DiscreteAction, s: ContinuousState
) -> FloatND:
    """With `s' = c`, the objective `u + s'/2` ties at `c = 0` and `c = 1/2` above
    `c = 1`; the stored `u + s'` is 0, 1/4 and 3/8 there."""
    return jnp.where(c > 0.75, -0.625, -0.5 * c) + 0.125 * alternative + 0.0 * s


def _member_utility(offset: float) -> UserFunction:
    def utility(
        *,
        c: ContinuousAction,
        alternative: DiscreteAction,
        s: ContinuousState,
        mu: DiscreteState,
    ) -> FloatND:
        return _tied_objective_utility(c=c, alternative=alternative, s=s) + (
            offset + 0.0 * mu
        )

    return utility


def _constant_utility(offset: float) -> UserFunction:
    def utility(
        *,
        c: ContinuousAction,
        alternative: DiscreteAction,
        s: ContinuousState,
        mu: DiscreteState,
    ) -> FloatND:
        return offset + 0.0 * (c + alternative + s + mu)

    return utility


def _end_zero_mu(*, s: ContinuousState, mu: DiscreteState) -> FloatND:
    return 0.0 * (s + mu)


# Per read: the rule and the regime that declares the separate objective.
_READS = {
    "target of a unilateral composition action": ("unilateral", "single_f"),
    "target of a renegotiation": ("renegotiation", "single_f"),
    "target of a transfer": ("transfer", "single_m"),
    "counterparty of a formation": ("formation", "single_m"),
    "couple a formation forms": ("formation", "couple"),
}

# m's flow utility in the formed couple when the counterparty is read: between
# single_m's value at its objective's choice and its best stored value, without
# and with taste shocks (3/8 and 1/2; 0.663 and 0.788 at scale 1/2).
_COUNTERPARTY_GAIN = {False: 0.4375, True: 0.75}


def _read_model(*, rule: str, read: str, taste_shocks: bool) -> lcm.Model:
    """`read` declares a separate objective; another regime's move only reads it.

    A couple that composition actions stays at -10, so both members leave. A formed
    couple gives f one more than she has alone and m a constant.
    """
    s = lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    c = lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=3)
    actions = {"c": c, "alternative": lcm.DiscreteGrid(Alternative)}
    forms = rule == "formation"

    def aggregator(name: str) -> UserFunction | lcm.Decoupled:
        if name != read:
            return additive
        return lcm.Decoupled(objective=_half_continuation, value=additive)

    def taste(name: str) -> lcm.ExtremeValueTasteShocks | None:
        return lcm.ExtremeValueTasteShocks() if taste_shocks and name == read else None

    def single(name: str) -> lcm.Regime:
        return lcm.Regime(
            actions=actions,
            states={"s": s},
            state_transitions={"s": _s_is_c},
            functions={"utility": _tied_objective_utility},
            composition_action=_rule_composition_action("formation")
            if forms and name == "single_f"
            else None,
            koopmans_aggregator=aggregator(name),
            taste_shocks=taste(name),
        )

    utilities = (
        {
            "f": _member_utility(1.0),
            "m": _constant_utility(
                _COUNTERPARTY_GAIN[taste_shocks] if read == "single_m" else 2.0
            ),
        }
        if forms
        else {"f": _member_utility(-10.0), "m": _member_utility(-10.0)}
    )
    return lcm.Model(
        n_periods=2,
        regimes={
            "couple": lcm.Regime(
                actions=actions,
                states={"s": s, "mu": lcm.DiscreteGrid(_Weight)},
                state_transitions={"s": _s_is_c, "mu": lcm.fixed_transition("mu")},
                functions={
                    "utility": lcm.CollectiveUtility(
                        utilities=utilities,
                        objective=lcm.ParetoObjective(weights={"f": 1.0, "m": 0.0}),
                    )
                },
                composition_action=None if forms else _read_composition_action(rule),
                koopmans_aggregator=aggregator("couple"),
                taste_shocks=taste("couple"),
            ),
            "single_f": single("single_f"),
            "single_m": single("single_m"),
            "end": lcm.Regime(
                states={"s": s, "mu": lcm.DiscreteGrid(_Weight)},
                functions={
                    "utility": lcm.CollectiveUtility(
                        utilities={
                            "f": _end_s_mu,
                            "m": _end_zero_mu if forms else _end_s_mu,
                        }
                    )
                },
            ),
            "end_f": lcm.Regime(states={"s": s}, functions={"utility": end_s}),
            "end_m": lcm.Regime(states={"s": s}, functions={"utility": end_s}),
        },
        regime_id_class=_CompositionActionRegimeId,
        initial_nodes=lcm.InitialNodes(
            by_period={0: ("couple", "single_f", "single_m")}
        ),
        edges={
            "couple": {"end": lcm.Periods(values=(0,))},
            "single_f": {"end_f": lcm.Periods(values=(0,))},
            "single_m": {"end_m": lcm.Periods(values=(0,))},
        },
    )


def _read_composition_action(rule: str) -> lcm.CompositionAction:
    """The couple's composition action under `rule`; a transfer adjusts m's payoff."""
    composition_action_ = _rule_composition_action(rule)
    if rule != "transfer":
        return composition_action_
    return lcm.CompositionAction(
        rule=composition_action_.rule,
        destinations={
            "f": lcm.Destination(regime="single_f", projection={"s": _keep_s}),
            "m": lcm.Destination(
                regime="single_m", projection={"s": _keep_s}, utility_adjustment=_paid
            ),
        },
    )


def _read_params(*, read: str, taste_shocks: bool) -> UserParams:
    return {read: {"taste_shocks": {"scale": 0.5}}} if taste_shocks else {}


@pytest.mark.parametrize("taste_shocks", [False, True])
@pytest.mark.parametrize("read", list(_READS))
def test_a_read_of_a_separate_objective_takes_the_stored_values_at_its_choice(
    *, read: str, taste_shocks: bool
) -> None:
    """A read of a regime with a separate objective uses that regime's own selection.

    The read comes from another regime's composition action or formation
    (specification 14.4, 13.5 and 14.5).

    The read value is the read regime's own solution value, bitwise: the stored
    values where its objective chooses, objective ties going to the stored
    values. Without taste shocks the objective ties at two actions and a third
    has the highest stored value, so the read is 1/4 above the alternative's
    1/8 there, not the 3/8 of the best stored value.
    """
    rule, regime = _READS[read]
    model = _read_model(rule=rule, read=regime, taste_shocks=taste_shocks)
    solution = model.solve(
        params=_read_params(read=regime, taste_shocks=taste_shocks), log_level="off"
    )
    couple = np.asarray(solution.value(step=0, regime="couple"))
    # A single regime keeps its one member on a trailing axis.
    single_f = np.asarray(solution.value(step=0, regime="single_f"))[..., 0]
    if rule != "formation":
        member = 0 if regime == "single_f" else 1
        target = np.asarray(solution.value(step=0, regime=regime))[..., 0]
        assert_array_equal(
            couple[..., member], np.broadcast_to(target[:, None], (2, 2))
        )
        if not taste_shocks:
            assert_array_equal(target, [0.375, 0.375])
        return
    # The formation happens, so f's value alone is hers in the formed couple at
    # the projected weight `low`.
    assert_array_equal(single_f, couple[:, 0, 0])
    if taste_shocks:
        return
    if regime == "couple":
        assert_array_equal(couple[:, 0, 0], [1.375, 1.375])
    else:
        assert_array_equal(
            np.asarray(solution.value(step=0, regime=regime))[..., 0], [0.375, 0.375]
        )


@pytest.mark.parametrize("read", list(_READS))
def test_a_simulated_move_into_a_separate_objective_takes_the_solved_value(
    *, read: str
) -> None:
    """A person who moves into, or forms a couple with, a regime with a separate
    objective has the value the solve publishes for that move (specification
    14.4 and 13.5)."""
    rule, regime = _READS[read]
    model = _read_model(rule=rule, read=regime, taste_shocks=False)
    solution = model.solve(params={}, log_level="off")
    forms = rule == "formation"
    start = (
        _CompositionActionRegimeId.single_f
        if forms
        else _CompositionActionRegimeId.couple
    )
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "s": jnp.zeros(2),
            "mu": jnp.zeros(2, dtype=jnp.int32),
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.full(2, start),
            "own_member": jnp.asarray(
                [model.member_names_to_ids[m] for m in ("f", "m")], dtype=jnp.int32
            ),
        },
        log_level="off",
        seed=0,
    ).to_dataframe(use_categorical_codes=True)
    first = frame.loc[frame["period"] == 0].set_index("agent_id")
    if forms:
        solved = np.asarray(solution.value(step=0, regime="single_f"))[0]
        assert first.loc[0, "regime_name"] == "couple"
        assert first.loc[0, "value_f"] == solved
        return
    solved = np.asarray(solution.value(step=0, regime="couple"))[0, 0]
    for agent, member in enumerate(("f", "m")):
        assert first.loc[agent, "regime_name"] == f"single_{member}"
        assert first.loc[agent, "value"] == solved[agent]


def _bounded_decoupled_model(*, taste_shocks: bool, invalid_income: float) -> lcm.Model:
    """Build a two-period liquid problem with a state-specific borrowing floor."""

    def flow(*, consumption: ContinuousAction, choice: DiscreteAction) -> FloatND:
        return consumption + choice

    def resources(*, wealth: ContinuousState) -> FloatND:
        return wealth

    def savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
        return resources - consumption

    def next_wealth(*, savings: FloatND) -> ContinuousState:
        return savings

    def terminal(*, wealth: ContinuousState, income: ContinuousState) -> FloatND:
        return 0.0 * (wealth + income)

    def floor(*, income: ContinuousState) -> FloatND:
        return jnp.where(income == invalid_income, jnp.nan, 0.25 + 0.25 * income)

    def objective(*, utility: FloatND, CE: FloatND) -> FloatND:
        return -utility + CE

    def value(*, utility: FloatND, CE: FloatND) -> FloatND:
        return utility + CE

    margin = lcm.LiquidMargin(
        state="wealth",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    # The temporal callable-floor fixture's state and liquid-action grids.
    states = {
        "wealth": lcm.LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
        "income": lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
    }
    decide = lcm.ConsumptionSavingsRegime(
        liquid=margin,
        states=states,
        actions={
            "choice": lcm.DiscreteGrid(Choice),
            "consumption": lcm.LinSpacedGrid(start=0.25, stop=2.0, n_points=8),
        },
        state_transitions={
            "wealth": next_wealth,
            "income": lcm.fixed_transition("income"),
        },
        functions={"utility": flow, "resources": resources, "savings": savings},
        constraints={
            "borrowing": lcm.post_decision_lower_bound(margin=margin, lower=floor)
        },
        koopmans_aggregator=lcm.Decoupled(objective=objective, value=value),
        taste_shocks=lcm.ExtremeValueTasteShocks() if taste_shocks else None,
        solver=GridSearch(),
    )
    return lcm.Model(
        n_periods=2,
        regimes={
            "decide": decide,
            "end": lcm.Regime(states=states, functions={"utility": terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "decide"}),
        edges={"decide": {"end": lcm.Periods(values=(0,))}},
    )


@pytest.mark.parametrize("taste_shocks", [False, True], ids=["hard", "ev1"])
def test_callable_floor_decoupled_keeps_objective_choice_and_stored_value(
    *, taste_shocks: bool
) -> None:
    """A callable floor constrains the objective's choice and stored readout."""
    model = _bounded_decoupled_model(taste_shocks=taste_shocks, invalid_income=0.5)
    params: UserParams = (
        {"decide": {"taste_shocks": {"scale": SCALE}}} if taste_shocks else {}
    )
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    probabilities = np.array([1.0, np.exp(-1.0 / SCALE)])
    probabilities /= probabilities.sum()
    expected_value = (
        probabilities @ np.array([0.25, 1.25])
        - SCALE * np.sum(probabilities * np.log(probabilities))
        if taste_shocks
        else 0.25
    )
    assert_array_almost_equal(
        solution.value(step=0, regime="decide")[..., 0],
        np.full((2, 2), expected_value),
        decimal=DECIMAL_PRECISION,
    )
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="decide",
        states={"wealth": jnp.array([1.0]), "income": jnp.array([1.0])},
        keep_intermediates=("Q", "F", "U", "CE", "Q_objective"),
    )
    consumption = np.arange(0.25, 2.25, 0.25)
    Q = np.stack((consumption, consumption + 1.0))[None, ...]
    F = np.broadcast_to(consumption <= 0.5, Q.shape)
    assert_array_almost_equal(
        got.value[..., 0], [expected_value], decimal=DECIMAL_PRECISION
    )
    # Every value intermediate carries the trailing member axis of `"self"`.
    assert_array_equal(np.asarray(got.Q)[..., 0], Q)
    assert_array_equal(np.asarray(got.U)[..., 0], Q)
    assert_array_equal(np.asarray(got.CE)[..., 0], np.zeros_like(Q))
    assert_array_equal(got.F, F)
    assert_array_equal(np.asarray(got.Q_objective)[..., 0], -Q)
    if taste_shocks:
        assert_array_almost_equal(
            cast("FloatND", got.choice_probabilities)[0],
            probabilities,
            decimal=DECIMAL_PRECISION,
        )
        assert_array_equal(cast("FloatND", got.Qc)[0, :, 0], [0.25, 1.25])
    else:
        assert got.actions is not None
        assert_array_equal(got.actions["choice"], [Choice.spend])
        assert_array_equal(got.actions["consumption"], [0.25])


@pytest.mark.parametrize("taste_shocks", [False, True], ids=["hard", "ev1"])
@pytest.mark.parametrize("entry", ["solve", "lookup"])
def test_callable_floor_decoupled_rejects_invalid_floor_with_no_feasible_action(
    *, taste_shocks: bool, entry: str
) -> None:
    """A nonfinite actual floor is mandatory even when every action is infeasible."""
    model = _bounded_decoupled_model(
        taste_shocks=taste_shocks, invalid_income=0.0 if entry == "solve" else 0.5
    )
    params: UserParams = (
        {"decide": {"taste_shocks": {"scale": SCALE}}} if taste_shocks else {}
    )
    if entry == "solve":
        with pytest.raises(InvalidValueFunctionError, match="floor"):
            model.solve(params=params, log_level="off", runtime_checks=False)
    else:
        solution = model.solve(params=params, log_level="off", runtime_checks=False)
        with pytest.raises(InvalidValueFunctionError, match="floor"):
            model.lookup_policy(
                params=params,
                solution=solution,
                period=0,
                regime_name="decide",
                states={"wealth": jnp.array([1.0]), "income": jnp.array([0.5])},
                keep_intermediates=("Q", "F", "U", "CE", "Q_objective"),
            )


def _objective_at_minus_infinity(*, utility: FloatND, CE: FloatND) -> FloatND:
    return jnp.full_like(utility + CE, -jnp.inf)


def test_an_objective_at_minus_infinity_leaves_the_choice_to_stored_values() -> None:
    """Every feasible action ties in an objective worth minus infinity, so the
    stored values decide (specification 13.5): the regime chooses and stores as
    under the stored aggregator alone, and the simulation acts."""
    separate = _model(
        koopmans_aggregator=lcm.Decoupled(
            objective=_objective_at_minus_infinity, value=additive
        )
    )
    plain = _model(koopmans_aggregator=additive)

    separate_values = separate.solve(params={}, log_level="off").values
    plain_values = plain.solve(params={}, log_level="off").values
    for period in (0, 1):
        assert_array_equal(
            separate_values[period]["decide"], plain_values[period]["decide"]
        )
    pd.testing.assert_frame_equal(
        _simulate(model=separate, params={}),
        _simulate(model=plain, params={}),
        check_exact=True,
    )


@lcm.categorical(ordered=False)
class _EntryRegimeId:
    source: ScalarInt
    target: ScalarInt
    source_end: ScalarInt
    target_end: ScalarInt


@lcm.categorical(ordered=False)
class _Draw:
    low: ScalarInt
    high: ScalarInt


def _cannot_stay(*, alternative: DiscreteAction) -> BoolND:
    return alternative < 0


def _source_utility(*, alternative: DiscreteAction) -> FloatND:
    return 0.0 * alternative


def _no_bonus() -> FloatND:
    return jnp.asarray(0.0)


def _bonus_of_c(*, c: ContinuousAction) -> FloatND:
    return 0.25 * c


def _bonus_of_c_and_draw(*, c: ContinuousAction, draw: DiscreteState) -> FloatND:
    return 0.25 * c + 0.0 * draw


def _deciders_utility(*, alternative: DiscreteAction, bonus: FloatND) -> FloatND:
    """2 at `first` and 3 at `second`, plus the bonus of the continuous action."""
    return 2.0 + alternative + bonus


def _deciders_utility_without_first(
    *, alternative: DiscreteAction, bonus: FloatND
) -> FloatND:
    return jnp.where(alternative == Alternative.first, -jnp.inf, 2.0 + bonus + 1.0)


def _deciders_utility_without_alternative(
    *, alternative: DiscreteAction, bonus: FloatND
) -> FloatND:
    return jnp.full_like(2.0 + alternative + bonus, -jnp.inf)


def _others_utility(*, alternative: DiscreteAction, bonus: FloatND) -> FloatND:
    """11 at `first` and 7 at `second`, plus twice the bonus."""
    return 11.0 - 4.0 * alternative + 2.0 * bonus


def _objective_without_first(*, utility: FloatND, CE: FloatND) -> FloatND:
    """The deciding member's objective is `-inf` at `first` (utility below 3)."""
    return jnp.where(utility + CE < 3.0, -jnp.inf, utility + CE)


# The deciding member's utility, without and with a separate objective, by the
# alternatives it values above minus infinity.
_DECIDERS_UTILITY = {
    "none": _deciders_utility_without_alternative,
    "second": _deciders_utility_without_first,
}
_SEPARATE_OBJECTIVE = {
    "none": _objective_at_minus_infinity,
    "second": _objective_without_first,
}


def _entry_model(
    *,
    alternatives: str,
    separate_objective: bool,
    members: tuple[str, ...],
    continuous: bool,
    draw: str,
    taste_shocks: bool,
    enable_jit: bool,
) -> lcm.Model:
    """`source` cannot stay, so its composition action moves every agent to `target`.

    There the deciding member `x` (the only member, or `x` beside `y`) chooses
    between `first` and `second`, with a continuous action `c` within each
    alternative if `continuous`, and a current shock `draw` that `target` reads
    and `source` observes (`"observed"`), does not observe (`"hidden"`), or
    that does not exist (`"none"`).
    """
    if separate_objective:
        deciders_utility: UserFunction = _deciders_utility
        aggregator: UserFunction | lcm.Decoupled = lcm.Decoupled(
            objective=_SEPARATE_OBJECTIVE[alternatives], value=additive
        )
    else:
        deciders_utility = _DECIDERS_UTILITY[alternatives]
        aggregator = additive
    if members:
        target_utility: UserFunction | lcm.CollectiveUtility = lcm.CollectiveUtility(
            utilities={
                name: deciders_utility if name == "x" else _others_utility
                for name in members
            },
            objective=lcm.ParetoObjective(
                weights={name: float(name == "x") for name in members}
            ),
        )
        end_utility: UserFunction | lcm.CollectiveUtility = lcm.CollectiveUtility(
            utilities=dict.fromkeys(members, zero_utility)
        )
    else:
        target_utility = deciders_utility
        end_utility = zero_utility
    actions: dict[str, lcm.DiscreteGrid | lcm.LinSpacedGrid] = {
        "alternative": lcm.DiscreteGrid(Alternative)
    }
    if continuous:
        actions["c"] = lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    bonus = {
        (False, "none"): _no_bonus,
        (True, "none"): _bonus_of_c,
        (True, "hidden"): _bonus_of_c_and_draw,
        (True, "observed"): _bonus_of_c_and_draw,
    }[continuous, draw]
    return lcm.Model(
        n_periods=2,
        regimes={
            "source": lcm.Regime(
                actions={"alternative": lcm.DiscreteGrid(Alternative)},
                functions={"utility": _source_utility},
                constraints={"cannot_stay": _cannot_stay},
                koopmans_aggregator=additive,
                composition_action=lcm.CompositionAction(
                    destinations={
                        "self": lcm.Destination(
                            regime="target",
                            projection={},
                            role="x" if members else None,
                        )
                    },
                    rule=lcm.PCOnActions(),
                    observes=("draw",) if draw == "observed" else (),
                ),
            ),
            "target": lcm.Regime(
                actions=actions,
                functions={"utility": target_utility, "bonus": bonus},
                koopmans_aggregator=aggregator,
                taste_shocks=lcm.ExtremeValueTasteShocks() if taste_shocks else None,
                shocks=("draws",) if draw != "none" else (),
            ),
            "source_end": lcm.Regime(functions={"utility": zero_utility}),
            "target_end": lcm.Regime(functions={"utility": end_utility}),
        },
        shocks=(
            {
                "draws": lcm.shocks.ShockGroup(
                    shocks={
                        "draw": lcm.shocks.Discrete(
                            categories=_Draw, probs=np.asarray([0.5, 0.5])
                        )
                    }
                )
            }
            if draw != "none"
            else {}
        ),
        regime_id_class=_EntryRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: ("source", "target")}),
        edges={
            "source": {"source_end": lcm.Periods(values=(0,))},
            "target": {"target_end": lcm.Periods(values=(0,))},
        },
        enable_jit=enable_jit,
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )


def _simulate_entry(*, model: lcm.Model, params: UserParams) -> pd.DataFrame:
    """Four agents start in `source` and take its composition action."""
    return model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions={
            "period": jnp.zeros(4, dtype=jnp.int32),
            "regime_id": jnp.full(4, _EntryRegimeId.source, dtype=jnp.int32),
        },
        seed=7,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)


# Per case: a separate objective or not, the members of `target` (the deciding
# member `x` first or second), a continuous action within each alternative, and
# the current shock of `target`.
_ENTRY_CASES = {
    "one member": (False, (), False, "none"),
    "one member, separate objective": (True, (), False, "none"),
    "one member, separate objective, c": (True, (), True, "none"),
    "x and y, c": (False, ("x", "y"), True, "none"),
    "y and x, separate objective": (True, ("y", "x"), False, "none"),
    "one member, c, hidden draw": (False, (), True, "hidden"),
    "x and y, c, observed draw": (False, ("x", "y"), True, "observed"),
}


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("alternatives", ["none", "second"])
@pytest.mark.parametrize("case", list(_ENTRY_CASES))
def test_a_composition_action_into_a_taste_node_without_a_finite_alternative_fails(
    *, case: str, alternatives: str, enable_jit: bool
) -> None:
    """A composition action that takes an agent to a node with taste shocks where
    every alternative is worth minus infinity to the deciding member fails, as a
    node without a feasible action does (specification 14.5a).

    That node has no choice probabilities and is worth minus infinity to every
    member, with or without a separate objective. Where `second` is worth more
    than minus infinity, the agent takes it, and every member publishes its
    value there: 3 to the deciding member and 7 to `y`, plus the bonus of the
    best `c`.
    """
    separate_objective, members, continuous, draw = _ENTRY_CASES[case]
    model = _entry_model(
        alternatives=alternatives,
        separate_objective=separate_objective,
        members=members,
        continuous=continuous,
        draw=draw,
        taste_shocks=True,
        enable_jit=enable_jit,
    )
    params: UserParams = {"target": {"taste_shocks": {"scale": 0.5}}}
    if alternatives == "none":
        solution = model.solve(params=params, log_level="off")
        assert np.isneginf(np.asarray(solution.value(step=0, regime="target"))).all()
        with pytest.raises(NoFeasibleActionError, match="no feasible action"):
            _simulate_entry(model=model, params=params)
        return
    frame = _simulate_entry(model=model, params=params)
    first = frame.loc[frame["period"] == 0]
    assert (first["regime_name"] == "target").all()
    assert (first["alternative"] == Alternative.second).all()
    bonus = 0.25 if continuous else 0.0
    if members:
        assert (first["value_x"] == 3.0 + bonus).all()
        assert (first["value_y"] == 7.0 + 2.0 * bonus).all()
    else:
        assert (first["value"] == 3.0 + bonus).all()


@pytest.mark.parametrize("members", [(), ("y", "x")])
def test_a_composition_action_into_a_hard_max_at_minus_infinity_takes_the_tie(
    *, members: tuple[str, ...]
) -> None:
    """Without taste shocks, a separate objective worth minus infinity at every
    action ties them all, and the stored values choose (specification 14.5a):
    the agent who arrives acts. `y` comes first in the tie order and takes
    `first` and `c = 1`, worth 11.5 to `y`; alone, `x` takes `second` and `c = 1`,
    worth 3.25."""
    model = _entry_model(
        alternatives="none",
        separate_objective=True,
        members=members,
        continuous=True,
        draw="none",
        taste_shocks=False,
        enable_jit=True,
    )
    frame = _simulate_entry(model=model, params={})
    first = frame.loc[frame["period"] == 0]
    assert (first["regime_name"] == "target").all()
    if members:
        assert (first["alternative"] == Alternative.first).all()
        assert (first["value_x"] == 2.25).all()
        assert (first["value_y"] == 11.5).all()
    else:
        assert (first["alternative"] == Alternative.second).all()
        assert (first["value"] == 3.25).all()
