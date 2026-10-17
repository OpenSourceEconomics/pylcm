"""Given paths replace realisations without replacing agents' expectations."""

import dataclasses
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.simulation.runtime import SimulationRuntime
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    LogNormalIIDProcess,
    Model,
    NormalIIDProcess,
    Phased,
    Regime,
    StochasticTransition,
    TauchenAR1Process,
    TimeVarying,
    UniformIIDProcess,
    categorical,
    fixed_transition,
    time_varying_params,
)
from lcm.exceptions import InvalidInitialConditionsError, InvalidSimulationInputError
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserFunction,
    UserParams,
)
from tests.simulation import test_law_reading_a_realized_draw

health_model = test_law_reading_a_realized_draw.model

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(*, shock: ContinuousState, choice: ScalarFloat) -> ScalarFloat:
    return -((choice - shock) ** 2)


def _zero() -> ScalarFloat:
    return jnp.asarray(0.0)


def _model(*, fold: bool = False, execution: ExecutionConfig | None = None) -> Model:
    return Model(
        ages=AgeGrid(start=20, inclusive_stop=23, step="Y"),
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig() if execution is None else execution,
        initial_nodes={(20, 21): "alive"},
        regimes={
            "alive": Regime(
                states={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=fold
                    )
                },
                actions={"choice": LinSpacedGrid(start=-2.0, stop=2.0, n_points=5)},
                functions={"utility": _utility},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        edges={"alive": {"alive": (20, 21), "dead": (22,)}},
    )


def _initial() -> dict[str, jax.Array]:
    return {
        "age": jnp.array([20, 21]),
        "shock": jnp.array([0.0, 0.0]),
        "regime_id": jnp.full(2, RegimeId.alive),
    }


def test_common_path_decisions_match_policy_lookup() -> None:
    model = _model()
    params = {"discount_factor": 0.9}
    solution = model.solve(params=params, log_level="off")
    path = jnp.array([1.0, -1.0, 2.0])
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=_initial(),
        shock_paths={"shock": path},
        seed=7,
        log_level="off",
    )
    df = result.to_dataframe(use_labels=False)
    for period in range(3):
        rows = df.query("regime_name == 'alive' and period == @period")
        expected = model.lookup_policy(
            params=params,
            solution=solution,
            period=int(period),
            regime_name="alive",
            states={"shock": jnp.full(len(rows), path[period])},
        )
        assert expected.actions is not None
        np.testing.assert_array_equal(rows["shock"], path[period])
        np.testing.assert_array_equal(rows["choice"], expected.actions["choice"])
    # Independent myopic oracle: no state or action feeds continuation.
    np.testing.assert_array_equal(
        df.query("regime_name == 'alive'")["choice"],
        df.query("regime_name == 'alive'")["shock"],
    )


@pytest.mark.parametrize("fold", [False, True])
def test_individual_paths_keep_original_subject_order_and_late_entry(
    *, fold: bool
) -> None:
    model = _model(fold=fold)
    paths = jnp.array([[1.0, -1.0, 2.0], [-1.0, 1.0, -2.0]])
    result = model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions=_initial(),
        shock_paths={"shock": paths},
        seed=7,
        log_level="off",
    )
    df = result.to_dataframe(use_labels=False).reset_index()
    for _, row in df.query("regime_name == 'alive'").iterrows():
        np.testing.assert_array_equal(
            row["shock"], paths[int(row["subject_id"]), int(row["period"])]
        )
        np.testing.assert_array_equal(row["choice"], row["shock"])


@pytest.mark.parametrize(
    ("path", "match"),
    [
        (jnp.array([0.0, 0.0]), "shock.*period 2"),
        (jnp.array([0.0, jnp.nan, 0.0]), "shock.*period 1"),
        (jnp.array([0.0, jnp.inf, 0.0]), "shock.*period 1"),
        (jnp.zeros((3, 3)), "shock.*subjects"),
        (jnp.zeros((2, 3, 1)), "shock.*shape"),
    ],
)
def test_invalid_paths_fail_loudly(*, path: jax.Array, match: str) -> None:
    with pytest.raises(InvalidSimulationInputError, match=match):
        _model().simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_initial(),
            shock_paths={"shock": path},
            seed=7,
            log_level="off",
            runtime_checks=False,
        )


def test_given_discrete_draw_feeds_a_dependent_transition(health_model: Model) -> None:
    """Given bad health replaces a good-health lottery before wealth reads it."""
    result = health_model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "age": jnp.array([20, 20]),
            "wealth": jnp.array([0.0, 4.0]),
            "health": jnp.array([0, 0], dtype=jnp.int32),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
        },
        shock_paths={"health": jnp.array([0, 0, 0], dtype=jnp.int32)},
        log_level="off",
        seed=0,
    )
    rows = result.to_dataframe(use_labels=False).query("period == 1")
    np.testing.assert_array_equal(rows["health"], 0)
    np.testing.assert_array_equal(rows["wealth"], 0.0)


def test_discrete_paths_reject_non_codes(health_model: Model) -> None:
    with pytest.raises(InvalidSimulationInputError, match=r"health.*period 1"):
        health_model.simulate(
            params={"discount_factor": 1.0},
            initial_conditions={
                "age": jnp.array([20]),
                "wealth": jnp.array([0.0]),
                "health": jnp.array([0], dtype=jnp.int32),
                "regime_id": jnp.array([0], dtype=jnp.int32),
            },
            shock_paths={"health": jnp.array([0.0, 0.5, 0.0])},
            log_level="off",
            seed=0,
        )


def _wealth_utility(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _next_wealth(*, wealth: ScalarFloat, next_cost: ContinuousState) -> ScalarFloat:
    return wealth - next_cost


def test_transition_only_fold_uses_given_destination_period_draw() -> None:
    """A folded cost enters terminal wealth without becoming a stored state."""
    model = Model(
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={20: "alive"},
        regimes={
            "alive": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0.0, stop=4.0, n_points=5),
                    "cost": NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=True
                    ),
                },
                state_transitions={"wealth": _next_wealth},
                functions={"utility": _wealth_utility},
            ),
            "dead": Regime(
                states={"wealth": LinSpacedGrid(start=0.0, stop=4.0, n_points=5)},
                functions={"utility": _wealth_utility},
            ),
        },
        edges={"alive": {"dead": (20,)}},
    )
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    initial = {
        "age": jnp.array([20, 20]),
        "wealth": jnp.array([2.0, 3.0]),
        "regime_id": jnp.full(2, RegimeId.alive),
    }
    df = model.simulate(
        params={"discount_factor": 1.0},
        solution=solution,
        initial_conditions=initial,
        shock_paths={"cost": jnp.array([0.0, 0.5])},
        seed=0,
        log_level="off",
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(df.query("period == 1")["wealth"], [1.5, 2.5])


@time_varying_params("schedule")
def _bonus(schedule: FloatND) -> ScalarFloat:
    return schedule


def _bonus_early() -> ScalarFloat:
    return jnp.asarray(0.1)


def _bonus_middle() -> ScalarFloat:
    return jnp.asarray(0.2)


def _bonus_late() -> ScalarFloat:
    return jnp.asarray(0.3)


def _bonus_at_age(age: float) -> UserFunction:
    return {20.0: _bonus_early, 21.0: _bonus_middle, 22.0: _bonus_late}[age]


def _bonus_age_signature(age: float) -> float:
    return age


def _wealth_shock_utility(
    *, wealth: ContinuousState, shock: ContinuousState
) -> ScalarFloat:
    return wealth + shock


def _next_wealth_with_bonus(
    *, wealth: ContinuousState, next_shock: ContinuousState, bonus: ScalarFloat
) -> ScalarFloat:
    return wealth + next_shock + bonus


@pytest.mark.parametrize("kind", ["fixed", "runtime", "by_age"])
@pytest.mark.parametrize("budget", [None, 100_000_000])
def test_given_paths_preserve_dated_transition_bindings(
    *, kind: str, budget: int | None
) -> None:
    """Rebuilding a lottery retains fixed operands and each age's own function."""
    schedule: UserParams = {
        "alive": {
            "bonus": {
                "schedule": TimeVarying(
                    values=jnp.array([0.1, 0.2, 0.3]), ages=(20, 21, 22)
                )
            }
        }
    }
    model = Model(
        ages=AgeGrid(start=20, inclusive_stop=23, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={(20, 21): "alive"},
        fixed_params=schedule if kind == "fixed" else {},
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3}, device_memory_bytes=budget
        ),
        regimes={
            "alive": Regime(
                states={
                    "wealth": LinSpacedGrid(start=-5.0, stop=8.0, n_points=5),
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0
                    ),
                },
                functions={
                    "utility": _wealth_shock_utility,
                    "bonus": (
                        AgeSpecializedFunction(
                            build=_bonus_at_age, signature=_bonus_age_signature
                        )
                        if kind == "by_age"
                        else _bonus
                    ),
                },
                state_transitions={"wealth": _next_wealth_with_bonus},
            ),
            "dead": Regime(
                states={"wealth": LinSpacedGrid(start=-5.0, stop=8.0, n_points=5)},
                functions={"utility": _wealth_utility},
            ),
        },
        edges={"alive": {"alive": (20, 21), "dead": (22,)}},
    )
    params: UserParams = {
        "discount_factor": 0.9,
        **(schedule if kind == "runtime" else {}),
    }
    solution = model.solve(params=params, log_level="off")
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=_initial() | {"wealth": jnp.array([1.0, 2.0])},
        shock_paths={"shock": jnp.array([1.0, -1.0, 2.0, 0.5])},
        seed=7,
        log_level="off",
    ).to_dataframe(use_labels=False)
    for subject, expected in ((0, [1.0, 0.1, 2.3, 3.1]), (1, [2.0, 4.2, 5.0])):
        np.testing.assert_allclose(
            frame.loc[frame["subject_id"] == subject, "wealth"], expected, rtol=1e-5
        )


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("budget", [None, 100_000_000])
def test_given_paths_survive_budgeted_subject_chunks(
    *, width: int, budget: int | None
) -> None:
    paths = np.array(
        [
            [1.0, -1.0, 2.0],
            [-1.0, 1.0, -2.0],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
    )
    initial = {
        "age": jnp.array([20, 21, 20, 21, 20]),
        "shock": jnp.zeros(5),
        "regime_id": jnp.full(5, RegimeId.alive),
    }
    model = _model(
        execution=ExecutionConfig(
            axis_widths={"subject": width}, device_memory_bytes=budget
        )
    )
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")
    for _ in range(2):
        df = (
            model.simulate(
                params={"discount_factor": 0.9},
                solution=solution,
                initial_conditions=initial,
                shock_paths={"shock": paths},
                seed=3,
                log_level="off",
            )
            .to_dataframe(use_labels=False)
            .reset_index()
        )
        rows = df.query("regime_name == 'alive'")
        expected = paths[rows.subject_id.to_numpy(), rows.period.to_numpy()]
        np.testing.assert_array_equal(rows.shock, expected)
        np.testing.assert_array_equal(rows.choice, expected)


def test_omitted_and_empty_paths_preserve_seeded_output() -> None:
    model = _model()
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")
    original = model.simulate(
        params={"discount_factor": 0.9},
        solution=solution,
        initial_conditions=_initial(),
        seed=19,
        log_level="off",
    ).to_dataframe(use_labels=False)
    empty = model.simulate(
        params={"discount_factor": 0.9},
        solution=solution,
        initial_conditions=_initial(),
        seed=19,
        log_level="off",
        shock_paths={},
    ).to_dataframe(use_labels=False)
    pd.testing.assert_frame_equal(original, empty, check_exact=True)


def test_unknown_path_is_rejected() -> None:
    with pytest.raises(
        InvalidSimulationInputError, match=r"unknown.*not a simulation shock"
    ):
        _model().simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_initial(),
            shock_paths={"unknown": np.zeros(3)},
            log_level="off",
        )


def _utility_with_noise(
    *, shock: ContinuousState, noise: ContinuousState, choice: ScalarFloat
) -> ScalarFloat:
    return _utility(shock=shock, choice=choice) + noise


def test_supplied_shock_keeps_unnamed_seeded_draws() -> None:
    base = _model()
    alive = base.user_regimes["alive"]
    model = Model(
        ages=base.ages,
        regime_id_class=RegimeId,
        initial_nodes={(20, 21): "alive"},
        edges=base.edges,
        regimes={
            "alive": dataclasses.replace(
                alive,
                states=dict(alive.states)
                | {
                    "noise": NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0
                    )
                },
                functions={"utility": _utility_with_noise},
            ),
            "dead": base.user_regimes["dead"],
        },
    )
    initial = _initial() | {"noise": jnp.zeros(2)}
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")
    original = model.simulate(
        params={"discount_factor": 0.9},
        solution=solution,
        initial_conditions=initial,
        seed=17,
        log_level="off",
    ).to_dataframe(use_labels=False)
    given = model.simulate(
        params={"discount_factor": 0.9},
        solution=solution,
        initial_conditions=initial,
        seed=17,
        log_level="off",
        shock_paths={"shock": np.array([1.0, -1.0, 2.0])},
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(given["noise"], original["noise"])
    np.testing.assert_array_equal(given["regime_name"], original["regime_name"])


@pytest.mark.parametrize(
    ("process", "bad"),
    [
        (LogNormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0), 0.0),
        (UniformIIDProcess(n_points=3, start=-1.0, stop=1.0), 2.0),
    ],
)
def test_continuous_support_rejects_inadmissible_values(
    *, process: LogNormalIIDProcess | UniformIIDProcess, bad: float
) -> None:
    base = _model()
    model = Model(
        ages=base.ages,
        regime_id_class=RegimeId,
        initial_nodes={(20, 21): "alive"},
        edges=base.edges,
        regimes={
            "alive": dataclasses.replace(
                base.user_regimes["alive"], states={"shock": process}
            ),
            "dead": base.user_regimes["dead"],
        },
    )
    with pytest.raises(InvalidSimulationInputError, match=r"shock.*period 1"):
        model.simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_initial(),
            shock_paths={"shock": np.array([1.0, bad, 1.0])},
            log_level="off",
        )


def test_ar1_accepts_realized_off_node_path_without_recursing_it() -> None:
    base = _model()
    model = Model(
        ages=base.ages,
        regime_id_class=RegimeId,
        initial_nodes={(20, 21): "alive"},
        edges=base.edges,
        regimes={
            "alive": dataclasses.replace(
                base.user_regimes["alive"],
                states={
                    "shock": TauchenAR1Process(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, rho=0.8
                    )
                },
            ),
            "dead": base.user_regimes["dead"],
        },
    )
    path = jnp.array([0.37, -0.81, 0.19])
    rows = model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions=_initial(),
        shock_paths={"shock": path},
        seed=2,
        log_level="off",
    ).to_dataframe(use_labels=False)
    alive = rows.query("regime_name == 'alive'")
    np.testing.assert_array_equal(alive.shock, path[alive["period"].to_numpy()])


def test_unrepresentable_path_reports_state_and_period() -> None:
    if np.finfo(np.longdouble).max <= jnp.finfo(canonical_float_dtype()).max:
        pytest.skip("No float on this platform is wider than the simulation dtype.")
    path = np.full(3, np.finfo(np.longdouble).max, dtype=np.longdouble)
    with pytest.raises(InvalidSimulationInputError, match=r"shock.*dtype at period 0"):
        _model().simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_initial(),
            shock_paths={"shock": path},
            log_level="off",
        )


def test_inactive_path_columns_are_ignored() -> None:
    df = (
        _model()
        .simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_initial(),
            shock_paths={"shock": np.array([1.0, -1.0, 2.0, np.nan])},
            seed=1,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
    )
    rows = df.query("regime_name == 'alive'")
    np.testing.assert_array_equal(rows.choice, rows.shock)


_LATE_AGE = float(Fraction(241, 12))


def _fractional_entry_model(
    *, fold: bool = False, enable_jit: bool = True, execution: ExecutionConfig
) -> Model:
    ages = (Fraction(20), Fraction(241, 12), Fraction(121, 6), Fraction(81, 4))
    return Model(
        ages=AgeGrid(exact_values=ages),
        regime_id_class=RegimeId,
        initial_nodes={ages[:2]: "alive"},
        enable_jit=enable_jit,
        execution_config=execution,
        regimes={
            "alive": Regime(
                states={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=fold
                    )
                },
                actions={"choice": LinSpacedGrid(start=-2.0, stop=2.0, n_points=5)},
                functions={"utility": _utility},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        edges={"alive": {"alive": ages[:2], "dead": (ages[2],)}},
    )


def _alive_records(df: pd.DataFrame) -> np.ndarray:
    """Rows of (subject, period, shock, choice) for the alive regime, sorted."""
    alive = (
        df.reset_index()
        .query("regime_name == 'alive'")
        .sort_values(["subject_id", "period"])
    )
    return np.column_stack(
        [
            alive["subject_id"].to_numpy(dtype=float),
            alive["period"].to_numpy(dtype=float),
            alive["shock"].to_numpy(dtype=float),
            alive["choice"].to_numpy(dtype=float),
        ]
    )


def _expected_alive_records(
    *, starts: np.ndarray, history: np.ndarray, n_periods: int
) -> np.ndarray:
    """Myopic oracle: the unique optimum on the action grid equals the shock."""
    rows = []
    for subject, start in enumerate(starts):
        for period in range(int(start), n_periods):
            level = history[period] if history.ndim == 1 else history[subject, period]
            rows.append((subject, period, level, level))
    return np.array(rows, dtype=float)


_ENTRY_HISTORY = np.array([[-1.0, 1.0, 2.0], [2.0, -2.0, 1.0], [1.0, -1.0, -2.0]])


@pytest.fixture(scope="module")
def fractional_entry_model() -> Model:
    return _fractional_entry_model(
        execution=ExecutionConfig(axis_widths={"subject": 2})
    )


@pytest.mark.parametrize("one_ulp_below", [False, True], ids=["exact", "one-ulp-below"])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
@pytest.mark.parametrize("individual", [False, True], ids=["common", "individual"])
def test_fractional_late_entry_starts_from_its_entry_period_column(
    *,
    fractional_entry_model: Model,
    one_ulp_below: bool,
    order: tuple[int, ...],
    individual: bool,
) -> None:
    """A late entrant at an off-binary grid age reads its entry-period column.

    The host age need not equal the working-precision grid value bit for bit: an
    age the matching rule accepts as period 1 also reads history column 1.
    """
    late = np.nextafter(_LATE_AGE, -np.inf) if one_ulp_below else _LATE_AGE
    permutation = np.array(order)
    starts = np.array([1, 0, 1])[permutation]
    history = _ENTRY_HISTORY[permutation] if individual else _ENTRY_HISTORY[0]
    df = fractional_entry_model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions={
            "age": np.array([late, 20.0, late])[permutation],
            "shock": np.zeros(3),
            "regime_id": np.full(3, RegimeId.alive, dtype=np.int32),
        },
        shock_paths={"shock": history},
        seed=7,
        log_level="off",
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        _alive_records(df),
        _expected_alive_records(starts=starts, history=history, n_periods=3),
    )


@pytest.mark.parametrize(
    ("fold", "enable_jit", "budget", "container"),
    [
        pytest.param(True, True, None, "jax", id="folded-jax"),
        pytest.param(True, True, 100_000_000, "numpy", id="budgeted"),
        pytest.param(False, False, None, "numpy", id="eager"),
        pytest.param(True, True, None, "dataframe", id="dataframe"),
    ],
)
def test_fractional_late_entry_column_holds_across_execution_routes(
    *, fold: bool, enable_jit: bool, budget: int | None, container: str
) -> None:
    """Folding, eager execution, budgets and input containers keep entry dating."""
    model = _fractional_entry_model(
        fold=fold,
        enable_jit=enable_jit,
        execution=ExecutionConfig(
            axis_widths={"subject": 2}, device_memory_bytes=budget
        ),
    )
    ages = np.array([_LATE_AGE, 20.0, _LATE_AGE])
    if container == "dataframe":
        initial = pd.DataFrame(
            {"age": ages, "shock": np.zeros(3), "regime_name": ["alive"] * 3}
        )
    else:
        initial = {
            "age": ages,
            "shock": np.zeros(3),
            "regime_id": np.full(3, RegimeId.alive, dtype=np.int32),
        }
        if container == "jax":
            initial = {name: jnp.asarray(value) for name, value in initial.items()}
    df = model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions=initial,
        shock_paths={"shock": _ENTRY_HISTORY},
        seed=7,
        log_level="off",
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        _alive_records(df),
        _expected_alive_records(
            starts=np.array([1, 0, 1]), history=_ENTRY_HISTORY, n_periods=3
        ),
    )


@pytest.mark.parametrize("age", [19.0, 20.05, 21.0])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_off_grid_age_reports_the_same_error_with_and_without_paths(
    *, fractional_entry_model: Model, age: float, runtime_checks: bool
) -> None:
    """Supplying paths leaves an invalid entry age to the ordinary age validator."""
    messages = []
    for shock_paths in (None, {"shock": np.array([-1.0, 1.0, 2.0])}):
        with pytest.raises(InvalidInitialConditionsError, match="Invalid age") as err:
            fractional_entry_model.simulate(
                params={"discount_factor": 0.9},
                initial_conditions={
                    "age": np.array([age]),
                    "shock": np.zeros(1),
                    "regime_id": np.full(1, RegimeId.alive, dtype=np.int32),
                },
                shock_paths=shock_paths,
                seed=7,
                log_level="off",
                runtime_checks=runtime_checks,
            )
        messages.append(str(err.value))
    assert messages[0] == messages[1]


@categorical(ordered=False)
class PhasedRegimeId:
    alive: ScalarInt
    believed: ScalarInt
    dead: ScalarInt


def _shock_utility(*, shock: ContinuousState) -> ScalarFloat:
    return shock


def _belief_only_utility(*, belief_only: ContinuousState) -> ScalarFloat:
    return belief_only


def _normal_shock(*, fold: bool = False) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=fold
    )


def _phased_model(
    *,
    believed: Regime | None = None,
    partial_visit: bool = False,
    enable_jit: bool = False,
    budget: int | None = None,
) -> Model:
    """Agents believe they move to `believed`; simulated subjects move to `dead`.

    With `partial_visit`, `alive` is also valued at age 21 although no simulated
    subject is alive then.
    """
    if partial_visit:
        ages = AgeGrid(start=20, inclusive_stop=22, step="Y")
        edges = Phased(
            solve={"alive": {"alive": (20,), "dead": (21,)}},
            simulate={"alive": {"dead": (20, 21)}},
        )
    else:
        ages = AgeGrid(start=20, inclusive_stop=21, step="Y")
        edges = Phased(
            solve={"alive": {"believed": (20,)}},
            simulate={"alive": {"dead": (20,)}},
        )
    return Model(
        ages=ages,
        regime_id_class=PhasedRegimeId,
        initial_nodes={20: "alive"},
        regimes={
            "alive": Regime(
                states={"shock": _normal_shock()},
                actions={"choice": LinSpacedGrid(start=-2.0, stop=2.0, n_points=5)},
                functions={"utility": _utility},
            ),
            "believed": Regime(functions={"utility": _zero})
            if believed is None
            else believed,
            "dead": Regime(functions={"utility": _zero}),
        },
        edges=edges,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3}, device_memory_bytes=budget
        ),
    )


def _phased_initial() -> dict[str, jax.Array]:
    return {
        "age": jnp.full(5, 20),
        "shock": jnp.zeros(5),
        "regime_id": jnp.full(5, PhasedRegimeId.alive, dtype=jnp.int32),
    }


def _phased_panel(df: pd.DataFrame) -> np.ndarray:
    """Rows of (subject, period, regime id, shock, choice), NaN where undeclared."""
    frame = df.reset_index().sort_values(["subject_id", "period"])
    return np.column_stack(
        [
            frame["subject_id"].to_numpy(dtype=float),
            frame["period"].to_numpy(dtype=float),
            frame["regime_name"]
            .astype(str)
            .map({"alive": 0, "believed": 1, "dead": 2})
            .to_numpy(dtype=float),
            frame["shock"].to_numpy(dtype=float),
            frame["choice"].to_numpy(dtype=float),
        ]
    )


def _expected_phased_panel(entry_shocks: np.ndarray) -> np.ndarray:
    """Alive at period 0 choosing the supplied shock, then dead at period 1."""
    rows = []
    for subject, level in enumerate(entry_shocks):
        rows.append((subject, 0, 0.0, level, level))
        rows.append((subject, 1, 2.0, np.nan, np.nan))
    return np.array(rows, dtype=float)


@pytest.mark.parametrize(
    ("enable_jit", "budget"),
    [(False, None), (True, None), (True, 100_000_000)],
    ids=["eager", "jit", "jit-budget"],
)
def test_phased_history_skips_regime_valued_only_in_solve(
    *, enable_jit: bool, budget: int | None
) -> None:
    """A regime agents only believe in needs no simulation program for paths."""
    df = (
        _phased_model(enable_jit=enable_jit, budget=budget)
        .simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_phased_initial(),
            shock_paths={"shock": np.array([-1.0])},
            seed=7,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
    )
    np.testing.assert_array_equal(
        _phased_panel(df), _expected_phased_panel(np.full(5, -1.0))
    )


def test_phased_individual_history_keeps_subject_rows() -> None:
    levels = np.array([-1.0, 1.0, -2.0, 2.0, 0.0])
    df = (
        _phased_model(enable_jit=True, budget=100_000_000)
        .simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_phased_initial(),
            shock_paths={"shock": levels[:, None]},
            seed=7,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
    )
    np.testing.assert_array_equal(_phased_panel(df), _expected_phased_panel(levels))


def test_phased_history_skips_periods_valued_only_in_solve() -> None:
    """A simulated regime's solve-only periods need no path column."""
    df = (
        _phased_model(partial_visit=True)
        .simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_phased_initial(),
            shock_paths={"shock": np.array([-1.0])},
            seed=7,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
    )
    np.testing.assert_array_equal(
        _phased_panel(df), _expected_phased_panel(np.full(5, -1.0))
    )


@pytest.fixture(scope="module")
def lognormal_belief_model() -> Model:
    believed = Regime(
        states={
            "shock": LogNormalIIDProcess(
                n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0
            )
        },
        functions={"utility": _shock_utility},
    )
    return _phased_model(believed=believed, enable_jit=True)


@pytest.mark.parametrize(
    "ignored",
    [None, np.nan, np.inf, -np.inf, -1.0, 1.0],
    ids=["missing", "nan", "inf", "-inf", "negative", "positive"],
)
@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4), (4, 2, 0, 3, 1)])
def test_column_used_only_by_a_solve_only_regime_is_not_validated(
    *, lognormal_belief_model: Model, ignored: float | None, order: tuple[int, ...]
) -> None:
    """A same-named shock in a regime no subject visits does not constrain paths."""
    levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[list(order)]
    path = (
        levels[:, None]
        if ignored is None
        else np.column_stack((levels, np.full(5, ignored)))
    )
    df = lognormal_belief_model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions=_phased_initial(),
        shock_paths={"shock": path},
        seed=7,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(_phased_panel(df), _expected_phased_panel(levels))


def test_shock_of_a_solve_only_regime_is_not_a_simulation_shock() -> None:
    believed = Regime(
        states={"belief_only": _normal_shock()},
        functions={"utility": _belief_only_utility},
    )
    with pytest.raises(
        InvalidSimulationInputError, match=r"belief_only.*not a simulation shock"
    ):
        _phased_model(believed=believed).simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_phased_initial(),
            shock_paths={"belief_only": np.array([0.0, 0.0])},
            seed=7,
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.parametrize("path", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_phased_history_still_validates_simulated_periods(path: np.ndarray) -> None:
    with pytest.raises(InvalidSimulationInputError, match=r"shock.*period 0"):
        _phased_model().simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_phased_initial(),
            shock_paths={"shock": path},
            seed=7,
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.parametrize("budget", [None, 100_000_000])
def test_phased_local_draw_is_dated_at_the_destination_period(
    *, budget: int | None
) -> None:
    """A folded transition-only cost from period 0 reads history column 1."""
    wealth_grid = LinSpacedGrid(start=0.0, stop=8.0, n_points=9)
    model = Model(
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        regime_id_class=PhasedRegimeId,
        initial_nodes={20: "alive"},
        regimes={
            "alive": Regime(
                states={"wealth": wealth_grid, "cost": _normal_shock(fold=True)},
                state_transitions={"wealth": _next_wealth},
                functions={"utility": _wealth_utility},
            ),
            "believed": Regime(
                states={"wealth": wealth_grid}, functions={"utility": _wealth_utility}
            ),
            "dead": Regime(
                states={"wealth": wealth_grid}, functions={"utility": _wealth_utility}
            ),
        },
        edges=Phased(
            solve={"alive": {"believed": (20,)}},
            simulate={"alive": {"dead": (20,)}},
        ),
        enable_jit=True,
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3}, device_memory_bytes=budget
        ),
    )
    wealth = np.arange(2.0, 7.0)
    df = model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions={
            "age": jnp.full(5, 20),
            "wealth": jnp.asarray(wealth),
            "regime_id": jnp.full(5, PhasedRegimeId.alive, dtype=jnp.int32),
        },
        shock_paths={"cost": np.array([0.0, 0.5])},
        seed=7,
        log_level="off",
    ).to_dataframe(use_labels=False)
    terminal = df.reset_index().query("period == 1").sort_values("subject_id")
    np.testing.assert_array_equal(
        terminal[["regime_name", "wealth"]].to_numpy(),
        np.column_stack((np.full(5, "dead", dtype=object), wealth - 0.5)),
    )


def test_visited_terminal_regime_accepts_and_seeds_a_history() -> None:
    """A simulated period without a transition program still takes its column."""
    model = Model(
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        regime_id_class=PhasedRegimeId,
        initial_nodes={20: "dead"},
        regimes={
            "alive": Regime(functions={"utility": _zero}),
            "believed": Regime(functions={"utility": _zero}),
            "dead": Regime(
                states={"shock": _normal_shock()}, functions={"utility": _shock_utility}
            ),
        },
        edges={},
    )
    df = model.simulate(
        params={},
        initial_conditions=_phased_initial()
        | {"regime_id": jnp.full(5, PhasedRegimeId.dead, dtype=jnp.int32)},
        shock_paths={"shock": np.array([-1.0])},
        seed=7,
        log_level="off",
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(df["shock"].to_numpy(dtype=float), np.full(5, -1.0))


@categorical(ordered=False)
class PairRegimeId:
    a: ScalarInt
    b: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class PairHealth:
    bad: ScalarInt
    good: ScalarInt


def _pair_model(*, a: Regime, b: Regime) -> Model:
    """Two regimes entered at age 20 that never exchange subjects."""
    return Model(
        ages=AgeGrid(start=20, inclusive_stop=22, step="Y"),
        regime_id_class=PairRegimeId,
        initial_nodes={20: ("a", "b")},
        regimes={"a": a, "b": b, "dead": Regime(functions={"utility": _zero})},
        edges={
            "a": {"a": (20,), "dead": (21,)},
            "b": {"b": (20,), "dead": (21,)},
        },
    )


def _pair_panel(*, df: pd.DataFrame, column: str) -> np.ndarray:
    """Return the `(subject, period)` panel of `column` in regimes `a` and `b`."""
    rows = (
        df.reset_index()
        .query("regime_name != 'dead'")
        .sort_values(["subject_id", "period"])
    )
    return rows[column].to_numpy().reshape(2, 2)


def _uniform_pair_model() -> Model:
    return _pair_model(
        a=Regime(
            states={"shock": UniformIIDProcess(n_points=3, start=-1.0, stop=1.0)},
            functions={"utility": _shock_utility},
        ),
        b=Regime(
            states={"shock": UniformIIDProcess(n_points=3, start=0.0, stop=3.0)},
            functions={"utility": _shock_utility},
        ),
    )


def _pair_initial(**states: jax.Array) -> dict[str, jax.Array]:
    return {
        "age": jnp.array([20.0, 20.0]),
        "regime_id": jnp.array([PairRegimeId.a, PairRegimeId.b], dtype=jnp.int32),
        **states,
    }


def test_paths_are_checked_against_each_subjects_own_regime_support() -> None:
    """A value outside one regime's support is admissible for a subject in another."""
    paths = np.array([[0.5, -0.5, 0.0], [2.5, 2.0, 0.0]])
    df = (
        _uniform_pair_model()
        .simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_pair_initial(shock=jnp.zeros(2)),
            shock_paths={"shock": paths},
            seed=3,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
    )
    np.testing.assert_array_equal(_pair_panel(df=df, column="shock"), paths[:, :2])


def test_value_read_outside_the_subjects_own_regime_support_is_rejected() -> None:
    with pytest.raises(
        InvalidSimulationInputError, match=r"shock.*period 1 in regime 'a'"
    ):
        _uniform_pair_model().simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_pair_initial(shock=jnp.zeros(2)),
            shock_paths={"shock": np.array([[0.5, 2.5, 0.0], [2.5, 2.0, 0.0]])},
            seed=3,
            log_level="off",
        )


def test_unread_cell_of_a_late_entrant_may_be_missing() -> None:
    """The period before a subject enters is never read, so NaN there is accepted."""
    paths = np.array([[1.0, -1.0, 2.0], [np.nan, 1.0, -2.0]])
    df = (
        _model()
        .simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_initial(),
            shock_paths={"shock": paths},
            seed=7,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
        .reset_index()
        .query("regime_name == 'alive'")
    )
    expected = paths[df["subject_id"].to_numpy(), df["period"].to_numpy()]
    np.testing.assert_array_equal(df["shock"], expected)


def _pair_health_probs(health: DiscreteState) -> FloatND:
    del health
    return jnp.array([0.5, 0.5])


def _health_utility(health: DiscreteState) -> ScalarFloat:
    return health.astype(jnp.float32)


def test_paths_seed_only_subjects_whose_regime_draws_the_state() -> None:
    """A subject whose regime carries health deterministically keeps its entry."""
    health = DiscreteGrid(category_class=PairHealth)
    model = _pair_model(
        a=Regime(
            states={"health": health},
            state_transitions={"health": fixed_transition("health")},
            functions={"utility": _health_utility},
        ),
        b=Regime(
            states={"health": health},
            state_transitions={"health": StochasticTransition(func=_pair_health_probs)},
            functions={"utility": _health_utility},
        ),
    )
    df = model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions=_pair_initial(
            health=jnp.full(2, PairHealth.bad, dtype=jnp.int32)
        ),
        shock_paths={
            "health": np.full(3, PairHealth.good, dtype=np.int32),
        },
        seed=3,
        log_level="off",
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        _pair_panel(df=df, column="health"),
        [[PairHealth.bad, PairHealth.bad], [PairHealth.good, PairHealth.good]],
    )


@pytest.mark.parametrize("missing", ["age", "regime_id"])
def test_missing_entry_column_reports_the_same_error_with_and_without_paths(
    missing: str,
) -> None:
    initial = {k: v for k, v in _initial().items() if k != missing}
    messages = []
    for shock_paths in (None, {"shock": np.array([1.0, -1.0, 2.0])}):
        with pytest.raises(InvalidInitialConditionsError) as err:
            _model().simulate(
                params={"discount_factor": 0.9},
                initial_conditions=initial,
                shock_paths=shock_paths,
                log_level="off",
            )
        messages.append(str(err.value))
    assert messages[0] == messages[1]


def _runtime_cache_sizes(model: Model) -> tuple[int, ...]:
    sizes = []
    for regimes in model._simulate_runtime_regimes.values():
        executor = next(iter(regimes.values())).simulation.programs.executor
        assert isinstance(executor, SimulationRuntime)
        sizes.append(len(executor.cache))
    return tuple(sizes)


def test_repeated_paths_reuse_compiled_transition_programs() -> None:
    """Calls supplying the same shocks reuse one bound program set."""
    model = _model()
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")
    sizes = []
    for scale in (1.0, 0.5, -1.0):
        model.simulate(
            params={"discount_factor": 0.9},
            solution=solution,
            initial_conditions=_initial(),
            shock_paths={"shock": scale * np.array([1.0, -1.0, 2.0])},
            seed=7,
            log_level="off",
        )
        sizes.append(_runtime_cache_sizes(model))
    assert sizes[1:] == [sizes[0], sizes[0]]


@categorical(ordered=False)
class TwoEntryRegimeId:
    short: ScalarInt
    long: ScalarInt
    dead: ScalarInt


def _shock_utility(*, shock: ContinuousState) -> ScalarFloat:
    return shock


def _two_entry_model(
    *, enable_jit: bool = False, width: int = 1, budget: int | None = None
) -> Model:
    """Return a model whose `short` entrants read one column and `long` ones two.

    `short` exits to `dead` after period 0; `long` stays through period 1.
    """

    def live() -> Regime:
        return Regime(
            states={
                "shock": NormalIIDProcess(
                    n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0
                )
            },
            functions={"utility": _shock_utility},
        )

    return Model(
        ages=AgeGrid(start=20, inclusive_stop=22, step="Y"),
        regime_id_class=TwoEntryRegimeId,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            axis_widths={"subject": width}, device_memory_bytes=budget
        ),
        initial_nodes={20: ("short", "long", "dead")},
        regimes={
            "short": live(),
            "long": live(),
            "dead": Regime(functions={"utility": _zero}),
        },
        edges={"short": {"dead": (20,)}, "long": {"long": (20,), "dead": (21,)}},
    )


def _two_entry_initial(names: tuple[str, ...]) -> dict[str, jax.Array]:
    return {
        "age": jnp.full(len(names), 20.0),
        "shock": jnp.zeros(len(names)),
        "regime_id": jnp.asarray(
            [getattr(TwoEntryRegimeId, name) for name in names], dtype=jnp.int32
        ),
    }


def _two_entry_expected_rows(
    *, names: tuple[str, ...], path: np.ndarray
) -> list[tuple[int, int, str, float | None]]:
    """Enumerate each subject's literal trajectory and the level it reads."""
    visits = {
        "short": ((0, "short"), (1, "dead")),
        "long": ((0, "long"), (1, "long"), (2, "dead")),
        "dead": ((0, "dead"),),
    }
    return [
        (
            subject,
            period,
            regime,
            None
            if regime == "dead"
            else float(path[period] if path.ndim == 1 else path[subject, period]),
        )
        for subject, name in enumerate(names)
        for period, regime in visits[name]
    ]


def _two_entry_observed_rows(
    *, model: Model, names: tuple[str, ...], path: np.ndarray
) -> list[tuple[int, int, str, float | None]]:
    frame = (
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions=_two_entry_initial(names),
            shock_paths={"shock": path},
            seed=7,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
        .reset_index()
        .sort_values(["subject_id", "period"])
    )
    return [
        (int(subject), int(period), regime, None if regime == "dead" else float(shock))
        for subject, period, regime, shock in zip(
            frame.subject_id, frame.period, frame.regime_name, frame.shock, strict=True
        )
    ]


@pytest.mark.parametrize("individual", [False, True], ids=["common", "individual"])
@pytest.mark.parametrize("route", ["eager", "jit", "tiled", "budgeted"])
def test_history_ending_where_the_population_stops_reading_simulates(
    *, individual: bool, route: str
) -> None:
    """A history may end where every actual subject stops reading the shock.

    Another regime the simulation graph allows at entry would read further, but
    nobody occupies it.
    """
    names = ("short", "short", "short")
    path = np.array([[0.5], [-1.0], [0.25]]) if individual else np.array([0.5])
    model = _two_entry_model(
        enable_jit=route != "eager",
        width=2 if route in ("tiled", "budgeted") else 1,
        budget=100_000_000 if route == "budgeted" else None,
    )
    assert _two_entry_observed_rows(
        model=model, names=names, path=path
    ) == _two_entry_expected_rows(names=names, path=path)


@pytest.mark.parametrize("names", [("short", "long"), ("long", "short")])
@pytest.mark.parametrize("individual", [False, True], ids=["common", "individual"])
def test_history_missing_a_column_an_occupied_regime_reads_is_rejected(
    *, names: tuple[str, ...], individual: bool
) -> None:
    path = np.full((2, 1), 0.5) if individual else np.array([0.5])
    with pytest.raises(InvalidSimulationInputError, match=r"shock.*period 1"):
        _two_entry_model().simulate(
            params={"discount_factor": 0.5},
            initial_conditions=_two_entry_initial(names),
            shock_paths={"shock": path},
            seed=7,
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_unread_history_tail_and_row_order_leave_observations_unchanged(
    order: tuple[int, ...],
) -> None:
    """Padding the unread tail with zeros or NaN changes nothing, call after call."""
    model = _two_entry_model(enable_jit=True, width=2)
    path = np.array([[0.5], [-1.0], [0.25]])[list(order)]
    names = ("short", "short", "short")
    observed = [
        _two_entry_observed_rows(
            model=model,
            names=names,
            path=path if tail is None else np.column_stack((path, np.full(3, tail))),
        )
        for tail in (None, 0.0, np.nan, None)
    ]
    assert observed == [_two_entry_expected_rows(names=names, path=path)] * 4
