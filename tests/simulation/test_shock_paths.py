"""Given paths replace realisations without replacing agents' expectations."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    ExecutionConfig,
    LinSpacedGrid,
    LogNormalIIDProcess,
    Model,
    NormalIIDProcess,
    Regime,
    TauchenAR1Process,
    UniformIIDProcess,
    categorical,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.typing import ContinuousState, ScalarFloat, ScalarInt
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


@pytest.mark.parametrize("budget", [None, 100_000_000])
def test_supplied_paths_on_subject_mesh(*, budget: int | None) -> None:
    devices = tuple(range(8))
    if len(jax.devices()) < len(devices):
        pytest.skip("Select this control in a fresh eight-device CPU process.")
    paths = np.tile(np.array([1.0, -1.0, 2.0]), (9, 1))
    paths[1::2] *= -1
    initial = {
        "age": jnp.array([20, 21, 20, 21, 20, 21, 20, 21, 20]),
        "shock": jnp.zeros(9),
        "regime_id": jnp.full(9, RegimeId.alive),
    }
    model = _model(
        execution=ExecutionConfig(
            devices=devices,
            simulation_sharding="subjects",
            axis_widths={"subject": 3},
            device_memory_bytes=budget,
        )
    )
    df = (
        model.simulate(
            params={"discount_factor": 0.9},
            initial_conditions=initial,
            shock_paths={"shock": paths},
            seed=0,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
        .reset_index()
    )
    rows = df.query("regime_name == 'alive'")
    expected = paths[rows.subject_id.to_numpy(), rows.period.to_numpy()]
    np.testing.assert_array_equal(rows.shock, expected)
    np.testing.assert_array_equal(rows.choice, expected)


def test_unrepresentable_path_reports_state_and_period() -> None:
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
