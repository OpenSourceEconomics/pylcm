"""Period coordinates have explicit meaning at every public time boundary."""

import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Literal, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import lcm
import lcm.exceptions
from _lcm.solution import period_replay
from _lcm.time import ModelTime, coordinate_at
from _lcm.typing import DataclassInstance
from _lcm.utils.namespace import flatten_regime_namespace
from _lcm_regime_finder.model_snapshot import read_model
from lcm import component_jobs
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.persistence import PeriodCapture, load_period_capture
from lcm.regime import RegimeReplacement
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.solvers import EGM
from lcm.transition import AgeCaseLaw, AgeSelector, PeriodSelector, TransitionLaw
from lcm.typing import (
    Age,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
    RegimeName,
    ScalarInt,
    StateName,
    UserInitialNodes,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@lcm.categorical(ordered=False)
class RegimeId:
    work: ScalarInt
    done: ScalarInt


def _flow(period: Period) -> FloatND:
    return jnp.asarray(period + 1, dtype=float)


def _terminal() -> FloatND:
    return jnp.asarray(8.0)


class ClockKwargs(TypedDict, total=False):
    ages: lcm.AgeGrid
    n_periods: int


class FixtureClockKwargs(TypedDict, total=False):
    ages: lcm.AgeGrid
    n_periods: int | float


class ModelDefinition(ClockKwargs):
    regimes: Mapping[RegimeName, lcm.Regime]
    regime_id_class: type[RegimeId]
    initial_nodes: UserInitialNodes
    fixed_params: UserParams


class InitialNodesKwargs(TypedDict, total=False):
    by_age: Mapping[int, RegimeName | Sequence[RegimeName]]
    by_period: Mapping[int, RegimeName | Sequence[RegimeName]]


def _model(
    *,
    initial_nodes: UserInitialNodes | None = None,
    **kwargs: Unpack[FixtureClockKwargs],
) -> lcm.Model:
    return lcm.Model(
        regimes={
            "work": lcm.Regime(functions={"utility": _flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"})
        if initial_nodes is None
        else initial_nodes,
        edges={
            "work": {
                "work": lcm.PeriodRange(exclusive_stop=1),
                "done": lcm.Periods(values=(1,)),
            }
        },
        fixed_params={"discount_factor": 0.5},
        **kwargs,  # ty: ignore[invalid-argument-type] - Fractional horizons exercise validation.
    )


def test_period_model_has_no_artificial_ages() -> None:
    model = _model(n_periods=3)
    assert model.ages is None
    assert model.n_periods == 3
    assert model.graph.reachability.nodes == frozenset(
        {(0, "work"), (1, "work"), (2, "done")}
    )
    solution = model.solve(params={}, log_level="off")
    # V_2 = 8; V_1 = 2 + 8/2 = 6; V_0 = 1 + 6/2 = 4.
    for period, name, expected in ((0, "work", 4), (1, "work", 6), (2, "done", 8)):
        np.testing.assert_allclose(solution.values[period][name], expected)


def _work_until_final_period(period: Period) -> ScalarInt:
    return jnp.where(period == 0, RegimeId.work, RegimeId.done)


@pytest.mark.parametrize(
    "targets",
    [
        {
            "work": lcm.PeriodRange(exclusive_stop=1),
            "done": lcm.PeriodRange(exclusive_stop=2),
        },
        {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(0, 1))},
    ],
)
def test_period_selectors_bound_an_explicit_transition_law(
    targets: dict[RegimeName, lcm.PeriodRange | lcm.Periods],
) -> None:
    model = lcm.Model(
        n_periods=3,
        regimes={
            "work": lcm.Regime(functions={"utility": _flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={"work": lcm.Transition(targets=targets, law=_work_until_final_period)},
        fixed_params={"discount_factor": 0.5},
    )
    assert model.graph.edges.solve["work"] == {
        "work": frozenset({0}),
        "done": frozenset({0, 1}),
    }
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_allclose(solution.value(step=0, regime="work"), 4.0)
    np.testing.assert_allclose(solution.value(step=1, regime="work"), 6.0)
    result = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "regime_id": jnp.array([RegimeId.work]),
            "period": jnp.array([0]),
        },
        seed=42,
        log_level="off",
    )
    frame = result.to_dataframe().reset_index()
    assert frame["period"].tolist() == [0, 1, 2]
    assert frame["regime_name"].tolist() == ["work", "work", "done"]


@pytest.mark.parametrize("n_periods", [0, -1, True])
def test_period_horizon_requires_positive_integer(n_periods: float) -> None:
    with pytest.raises(ModelInitializationError, match="n_periods"):
        _model(n_periods=n_periods)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"ages": lcm.AgeGrid(start=40, inclusive_stop=42, step="Y"), "n_periods": 3}],
)
def test_exactly_one_time_coordinate_is_required(kwargs: FixtureClockKwargs) -> None:
    with pytest.raises(ModelInitializationError, match=r"[Ee]xactly one"):
        _model(**kwargs)


def test_period_graph_rejects_ambiguous_bare_selectors() -> None:
    with pytest.raises(ModelInitializationError, match=r"PeriodRange|Periods"):
        lcm.Model(
            n_periods=3,
            regimes={"work": lcm.Regime(functions={"utility": _terminal})},
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
            edges={"work": {"work": (0, 1)}},
        )


def _needs_age(age: FloatND) -> FloatND:
    return age


def test_period_model_rejects_reserved_age_dependency() -> None:
    with pytest.raises(ModelInitializationError, match="age"):
        lcm.Model(
            n_periods=1,
            regimes={
                "work": lcm.Regime(functions={"utility": _needs_age}),
                "done": lcm.Regime(functions={"utility": _terminal}),
            },
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
            edges={},
        )


def test_by_period_until_selects_the_periods_before_its_stop() -> None:
    schedule = lcm.ByPeriod.until(
        start_period_inclusive=1,
        stop_period_exclusive=4,
        law="work",
        then="done",
    ).resolve(n_periods=5)
    assert schedule.periods == (1, 2, 3)
    assert [schedule.at(period) for period in schedule.periods] == [
        "work",
        "work",
        "done",
    ]


def test_by_period_refuses_nonperiod_selectors() -> None:
    with pytest.raises(lcm.exceptions.RegimeInitializationError, match=r"[Pp]eriod"):
        lcm.ByPeriod(cases={1.0: "work"})


@pytest.mark.parametrize("dataframe", [False, True])
def test_period_start_and_output_without_age(*, dataframe: bool) -> None:
    model = _model(n_periods=3, initial_nodes=lcm.InitialNodes(by_period={1: "work"}))
    initial = (
        pd.DataFrame({"period": [1], "regime_name": ["work"]})
        if dataframe
        else {"period": jnp.array([1]), "regime_id": jnp.array([RegimeId.work])}
    )
    model.validate_initial_conditions(initial_conditions=initial, params={})
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(initial_conditions=initial, params={}),
        [True],
    )
    result = model.simulate(
        params={}, initial_conditions=initial, log_level="off", seed=42
    )
    frame = result.to_dataframe(additional_targets=["utility"]).reset_index()
    assert "age" not in frame.columns
    assert frame["period"].tolist() == [1, 2]
    np.testing.assert_allclose(frame["utility"], [2, 8])
    assert model.ages is None


@pytest.mark.parametrize(
    "entry",
    [
        {"age": jnp.array([0])},
        {"age": jnp.array([0]), "period": jnp.array([0])},
        {"period": jnp.array([-1])},
        {"period": jnp.array([3])},
        {"period": jnp.array([0.5])},
    ],
)
@pytest.mark.parametrize(
    "method",
    ["simulate", "validate_initial_conditions", "initial_conditions_feasibility"],
)
def test_period_starts_are_checked_at_every_entry(
    *, entry: dict[str, jax.Array], method: str
) -> None:
    model = _model(n_periods=3)
    kwargs = {"log_level": "off"} if method == "simulate" else {}
    with pytest.raises(
        lcm.exceptions.InvalidInitialConditionsError, match=r"period|coordinate"
    ):
        getattr(model, method)(
            params={},
            initial_conditions={"regime_id": jnp.array([RegimeId.work]), **entry},
            **kwargs,
        )


def test_period_public_initial_nodes_roundtrip() -> None:
    model = _model(n_periods=3)
    assert model.initial_nodes == lcm.InitialNodes(by_period={0: "work"})
    assert (
        _model(n_periods=3, initial_nodes=model.initial_nodes).graph.nodes
        == model.graph.nodes
    )
    assert model.graph.coordinate_kind == "period"


def _specialized_flow(period: int) -> Callable[..., FloatND]:
    if type(period) is not int:
        raise TypeError("Specialization requires an integer period.")
    value = float(period + 1)

    def flow() -> FloatND:
        return jnp.asarray(value)

    return flow


def _specialized_model(
    marker: lcm.AgeSpecializedFunction | lcm.PeriodSpecializedFunction,
) -> lcm.Model:
    return lcm.Model(
        n_periods=3,
        regime_id_class=RegimeId,
        regimes={
            "work": lcm.Regime(functions={"utility": marker}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
        fixed_params={"discount_factor": 0.5},
    )


def test_period_specialization_receives_integer_period() -> None:
    model = _specialized_model(
        lcm.PeriodSpecializedFunction(build=_specialized_flow, signature=int)
    )
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_allclose(solution.values[0]["work"], 4.0)


def _use_payment(*, payment: FloatND) -> FloatND:
    return payment


def _next_unused(*, unused: FloatND) -> FloatND:
    return unused


def test_model_level_period_specialization_receives_integer_period() -> None:
    model = lcm.Model(
        n_periods=3,
        states={"unused": lcm.LinSpacedGrid(start=0, stop=1, n_points=2)},
        state_transitions={"unused": _next_unused},
        functions={
            "payment": lcm.PeriodSpecializedFunction(
                build=_specialized_flow, signature=int
            )
        },
        regimes={
            "work": lcm.Regime(functions={"utility": _use_payment}),
            "done": lcm.Regime(functions={"utility": _terminal, "payment": None}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
        fixed_params={"discount_factor": 0.5},
    )
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_allclose(solution.values[0]["work"], 4.0)


def test_period_declared_transitions_preserve_coordinate_kind() -> None:
    definition: ModelDefinition = {
        "n_periods": 2,
        "regimes": {
            "work": lcm.Regime(functions={"utility": _flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        "regime_id_class": RegimeId,
        "initial_nodes": lcm.InitialNodes(by_period={0: "work"}),
        "fixed_params": {"discount_factor": 0.5},
    }
    model = lcm.Model(**definition, edges={"work": lcm.Transition(law="done")})
    rebuilt = lcm.Model(**definition, edges=model.declared_transitions["solve"])
    assert rebuilt.graph.nodes == model.graph.nodes
    np.testing.assert_allclose(
        rebuilt.solve(params={}, log_level="off").values[0]["work"], 5.0
    )


def test_period_model_rejects_age_specialization() -> None:
    with pytest.raises(ModelInitializationError, match=r"[Aa]geSpecialized|coordinate"):
        _specialized_model(
            lcm.AgeSpecializedFunction(  # ty: ignore[invalid-argument-type] - A period builder is refused before age specialization.
                build=_specialized_flow,
                signature=int,
            )
        )


def test_period_saved_result_omits_age(tmp_path: Path) -> None:
    model = _model(n_periods=3)
    result = model.simulate(
        params={},
        initial_conditions={
            "period": jnp.array([0]),
            "regime_id": jnp.array([RegimeId.work]),
        },
        seed=3,
        log_level="off",
    )
    path = tmp_path / "result.pkl"
    result.save(directory=path)
    restored = lcm.SimulationResult.load(directory=path)
    pd.testing.assert_frame_equal(restored.to_dataframe(), result.to_dataframe())
    assert "age" not in restored.to_dataframe().columns


def _age_model(*, initial_nodes: UserInitialNodes | None = None) -> lcm.Model:
    return lcm.Model(
        ages=lcm.AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regimes={
            "work": lcm.Regime(functions={"utility": _flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=((0, "work"),) if initial_nodes is None else initial_nodes,
        edges={"work": {"work": (0,), "done": (1,)}},
        fixed_params={"discount_factor": 0.5},
    )


def test_age_and_period_models_have_distinct_compatibility_identity() -> None:
    period_model = _model(n_periods=3)
    age_model = _age_model()
    assert (
        period_model._model_structure_fingerprint
        != age_model._model_structure_fingerprint
    )


@pytest.mark.parametrize("declaration", [False, True])
def test_legacy_age_model_state_restores_an_explicit_clock(
    *, declaration: bool
) -> None:
    model = _age_model()
    state = model.__getstate__()
    # Age-only archives predate the clock and the by_period field.
    if declaration:
        starts = lcm.InitialNodes(by_age={0: "work"})
        object.__delattr__(starts, "by_period")
        state["initial_nodes"] = starts  # ty: ignore[invalid-key]
    else:
        state["initial_nodes"] = model.graph.initial_nodes  # ty: ignore[invalid-key]
    del state["_time"]
    state.pop("_post_decision_shocks", None)
    restored = lcm.Model.__new__(lcm.Model)
    restored.__setstate__(state)
    assert restored._model_structure_fingerprint == model._model_structure_fingerprint
    result = restored.simulate(
        params={},
        solution=model.solve(params={}, log_level="off"),
        initial_conditions={
            "age": jnp.array([0]),
            "regime_id": jnp.array([RegimeId.work]),
        },
        seed=3,
        log_level="off",
    )
    frame = result.to_dataframe()
    np.testing.assert_array_equal(frame["age"], [0, 1, 2])
    np.testing.assert_allclose(frame["value"], [4, 6, 8])


def test_saved_model_without_clock_or_age_grid_is_rejected() -> None:
    state = _model(n_periods=3).__getstate__()
    del state["_time"]
    restored = lcm.Model.__new__(lcm.Model)
    with pytest.raises(ModelInitializationError, match=r"archive.*clock"):
        restored.__setstate__(state)


def _grid_for_period(period: int) -> lcm.LinSpacedGrid:
    if type(period) is not int:
        raise TypeError("Specialization requires an integer period.")
    return lcm.LinSpacedGrid(start=period, stop=period + 2, n_points=3)


def _wealth_utility(*, wealth: FloatND) -> FloatND:
    return wealth


def test_period_grid_specialization_receives_integer_period() -> None:
    model = lcm.Model(
        n_periods=3,
        regimes={
            "work": lcm.Regime(
                functions={"utility": _wealth_utility},
                states={
                    "wealth": lcm.PeriodSpecializedGrid(
                        build=_grid_for_period, signature=int
                    )
                },
            ),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={1: "work"}),
        edges={},
    )
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_allclose(solution.values[1]["work"][..., 0], [1, 2, 3])


def _typed_flow(*, pref_type: DiscreteState, period: Period) -> FloatND:
    return jnp.asarray(1 + period + pref_type, dtype=float)


def test_period_component_job_roundtrip(tmp_path: Path) -> None:
    model = lcm.Model(
        n_periods=2,
        regimes={
            "work": lcm.Regime(
                states={"pref_type": lcm.DiscreteGrid(category_class=RegimeId)},
                state_transitions={"pref_type": lcm.fixed_transition("pref_type")},
                functions={"utility": _typed_flow},
            ),
            "done": lcm.Regime(
                states={"pref_type": lcm.DiscreteGrid(category_class=RegimeId)},
                functions={"utility": _typed_flow},
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={"work": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"discount_factor": 0.5},
        execution_config=lcm.ExecutionConfig(
            invariant_block_widths={"pref_type": 1},
            invariant_block_schedule=lcm.InvariantBlockSchedule.BLOCK_MAJOR,
            axis_widths={"agent": 1},
        ),
    )
    initial = {
        "period": jnp.array([0, 0]),
        "regime_id": jnp.array([0, 0]),
        "pref_type": jnp.array([0, 1]),
    }
    directory = tmp_path / "jobs"
    plan = component_jobs.plan_component_jobs(
        model=model,
        params={},
        directory=directory,
        n_jobs=1,
        initial_conditions=initial,
        seed=5,
    )
    for job in range(len(plan.jobs)):
        component_jobs.run_component_job(
            model=model,
            params={},
            directory=directory,
            job=job,
            initial_conditions=initial,
            log_level="off",
        )
    collected = component_jobs.collect_component_jobs(
        model=model,
        params={},
        directory=directory,
        log_level="off",
    )
    reference = model.simulate(
        params={}, initial_conditions=initial, seed=5, log_level="off"
    )
    assert collected.simulation is not None
    pd.testing.assert_frame_equal(
        collected.simulation.to_dataframe(),
        reference.to_dataframe(),
        check_exact=True,
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("monthly", [False, True])
def test_age_callback_coordinates_preserve_age_grid_scalars(
    *, wrapped: bool, monthly: bool
) -> None:
    ages = (
        lcm.AgeGrid(start=20, inclusive_stop=Fraction(241, 12), step="M")
        if monthly
        else lcm.AgeGrid(start=20, inclusive_stop=21, step="Y")
    )
    axis = ModelTime.from_inputs(ages=ages, n_periods=None) if wrapped else ages
    for period in range(ages.n_periods):
        actual = coordinate_at(ages=axis, step=period)
        expected = ages.period_to_age(period)
        assert actual == expected
        assert type(actual) is type(expected)


def test_period_schedule_inspection_rejects_noninteger_coordinates() -> None:
    schedule = lcm.ByPeriod(cases={1: "work"}).resolve(n_periods=3)
    with pytest.raises((TypeError, KeyError), match=r"period|integer"):
        schedule.at(1.0)


def test_period_schedule_has_no_covered_ages() -> None:
    schedule = lcm.ByPeriod(cases={1: "work"}).resolve(n_periods=3)
    assert schedule.periods == (1,)
    with pytest.raises(AttributeError, match="period"):
        _ = schedule.covered_ages


def test_period_capture_and_public_replay_report_no_age(tmp_path: Path) -> None:
    source = {"example": "period-clock-v1"}
    model = _model(n_periods=3)
    model.solve(
        params={},
        log_level="off",
        period_capture=PeriodCapture(
            directory=tmp_path, steps=(("work", 1),), source_identity=source
        ),
    )
    directory = tmp_path / "work@1"
    record = load_period_capture(directory=directory)
    assert record.metadata["period"] == 1
    assert record.metadata.get("age") is None
    assert record.reference is not None
    np.testing.assert_allclose(record.reference, 6.0)
    replay = _model(n_periods=3).replay_period(
        directory=directory, params={}, source_identity=source
    )
    assert replay.capture.metadata.get("age") is None
    assert replay.reference_matches
    assert replay.optimized_hlo_matches
    np.testing.assert_allclose(replay.value, 6.0)


@pytest.mark.parametrize("surface", ["logical", "layout"])
def test_period_inspection_reports_no_age(
    *, surface: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(n_periods=3)
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "work@1")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model.solve(params={}, log_level="off")
    directory = tmp_path / "work@1"
    replay = (
        period_replay.replay_period(directory=directory)
        if surface == "logical"
        else period_replay.replay_period_on_recorded_layout(
            directory=directory, devices=jax.devices()
        )
    )
    assert replay.period == 1
    assert replay.age is None
    np.testing.assert_allclose(replay.output.value, 6.0)


def test_period_solve_and_simulation_logs_use_period_labels(
    *, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    model = _model(n_periods=3, initial_nodes=lcm.InitialNodes(by_period={1: "work"}))
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        solution = model.solve(params={}, log_level="debug", log_path=tmp_path)
        model.simulate(
            params={},
            initial_conditions={
                "period": jnp.array([1]),
                "regime_id": jnp.array([RegimeId.work]),
            },
            solution=solution,
            seed=3,
            log_level="debug",
            log_path=tmp_path,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("[attr] work period 1:" in message for message in messages)
    assert any(
        "work  period 1" in message and "V min=" in message for message in messages
    )
    assert sum("Period 1 (1 regimes)" in message for message in messages) == 2
    assert not any("age " in message.lower() for message in messages)


@pytest.mark.parametrize("phase", ["solve", "simulate"])
def test_period_nan_report_uses_period_labels(*, phase: str, tmp_path: Path) -> None:
    def bad_value() -> FloatND:
        return jnp.asarray(jnp.nan)

    model = lcm.Model(
        n_periods=2,
        regimes={
            "work": lcm.Regime(
                functions={
                    "utility": lcm.Phased(
                        solve=bad_value if phase == "solve" else _terminal,
                        simulate=bad_value,
                    )
                }
            ),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={"work": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"discount_factor": 0.5},
    )

    def run() -> None:
        solution = model.solve(params={}, log_level="debug", log_path=tmp_path)
        if phase == "simulate":
            model.simulate(
                params={},
                initial_conditions={
                    "period": jnp.array([0]),
                    "regime_id": jnp.array([RegimeId.work]),
                },
                solution=solution,
                seed=3,
                log_level="debug",
                log_path=tmp_path,
            )

    with pytest.raises(
        lcm.exceptions.InvalidValueFunctionError, match="at period 0"
    ) as caught:
        run()
    if phase == "solve":
        assert isinstance(caught.value.diagnostics, dict)
        assert "age" not in caught.value.diagnostics
        assert caught.value.diagnostics["period"] == 0
        assert "at period 0" in "\n".join(caught.value.__notes__)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"by_age": {0: "work"}, "by_period": {0: "work"}},
        {"by_age": {}, "by_period": {0: "work"}},
        {"by_period": {}},
        {"by_period": {0: ()}},
        {"by_period": {0: ""}},
        {"by_period": {0: ("work", "")}},
    ],
)
def test_initial_nodes_requires_one_nonempty_coordinate_map(
    kwargs: InitialNodesKwargs,
) -> None:
    """Exactly one nonempty coordinate mapping declares the starts."""
    with pytest.raises(ModelInitializationError):
        lcm.InitialNodes(**kwargs)


def test_initial_nodes_refuses_period_keys_in_age_map() -> None:
    """An explicit period selector cannot be relabelled as an age."""
    with pytest.raises(ModelInitializationError, match=r"[Aa]ge"):
        lcm.InitialNodes(by_age={lcm.PeriodRange(start=0): "work"})


@pytest.mark.parametrize("selector", [-1, 3, (0, 3), lcm.Periods(values=(3,))])
def test_initial_nodes_checks_period_horizon(selector: PeriodSelector) -> None:
    """Exact period starts must belong to the declared horizon."""
    with pytest.raises(ModelInitializationError, match=r"[Pp]eriod"):
        _model(
            n_periods=3, initial_nodes=lcm.InitialNodes(by_period={selector: "work"})
        )


def test_period_initial_nodes_freezes_and_unions_selectors() -> None:
    """Owned selectors union their starts and round-trip without changing identity."""
    names = ["work", "done", "work"]
    starts = {lcm.PeriodRange(start=-2, exclusive_stop=2): "work", 1: names}
    declaration = lcm.InitialNodes(by_period=starts)
    names.append("unknown")
    starts[2] = "unknown"
    model = _model(n_periods=3, initial_nodes=declaration)
    assert model.initial_nodes.by_age is None
    assert model.initial_nodes.by_period == {0: ("work",), 1: ("done", "work")}
    assert model.graph.initial_nodes == frozenset(
        {(0, "work"), (1, "work"), (1, "done")}
    )
    rebuilt = _model(n_periods=3, initial_nodes=model.initial_nodes)
    assert rebuilt._model_structure_fingerprint == model._model_structure_fingerprint


@pytest.mark.parametrize("selector", [lcm.PeriodRange(start=3), ()])
def test_period_initial_nodes_refuses_empty_selection(selector: PeriodSelector) -> None:
    """A selector must admit at least one starting node."""
    with pytest.raises(ModelInitializationError, match=r"no period"):
        _model(
            n_periods=3, initial_nodes=lcm.InitialNodes(by_period={selector: "work"})
        )


def test_period_initial_nodes_requires_named_keyword() -> None:
    """Period declarations cannot silently reinterpret legacy age syntax."""
    with pytest.raises(ModelInitializationError, match=r"by_period"):
        _model(n_periods=3, initial_nodes={0: "work"})


def test_initial_nodes_rejects_clock_mismatch_both_directions() -> None:
    """The declaration clock must match the model clock."""
    with pytest.raises(ModelInitializationError, match="by_period"):
        _model(n_periods=3, initial_nodes=lcm.InitialNodes(by_age={0: "work"}))
    with pytest.raises(ModelInitializationError, match="by_age"):
        _age_model(initial_nodes=lcm.InitialNodes(by_period={0: "work"}))


@pytest.mark.parametrize("selector", [(0, 1), range(2), lcm.Periods(values=(0, 1))])
def test_period_initial_nodes_accepts_explicit_integer_collections(
    selector: PeriodSelector,
) -> None:
    """The by_period keyword labels every supported exact-selector form."""
    model = _model(
        n_periods=3, initial_nodes=lcm.InitialNodes(by_period={selector: "work"})
    )
    assert model.graph.initial_nodes == frozenset({(0, "work"), (1, "work")})


@pytest.mark.parametrize("selector", [(0, 3), range(4), lcm.Periods(values=(0, 3))])
def test_period_initial_nodes_does_not_clip_explicit_collections(
    selector: PeriodSelector,
) -> None:
    """Exact starts require grid membership even when another coordinate is valid."""
    with pytest.raises(ModelInitializationError, match=r"[Pp]eriod"):
        _model(
            n_periods=3, initial_nodes=lcm.InitialNodes(by_period={selector: "work"})
        )


def test_period_initial_nodes_is_read_only() -> None:
    """Published mappings and coordinate-mode fields cannot be edited in place."""
    declaration = lcm.InitialNodes(by_period={0: "work"})
    with pytest.raises(TypeError):
        cast("dict[int, tuple[str, ...]]", declaration.by_period)[1] = ("done",)
    with pytest.raises(AttributeError):
        declaration.by_period = {1: ("done",)}  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("wrapper", ["case", "default", "until", "mapped-until"])
def test_by_period_owns_nested_law_mappings(wrapper: str) -> None:
    """The period schedule inherits the same owned law boundary as ByAge."""
    probability = lcm.StochasticTransition(func=_certain)
    probabilities = {"work": probability}

    def to_probabilities(law: AgeCaseLaw) -> AgeCaseLaw:  # noqa: ARG001
        return probabilities

    schedules = {
        "case": lcm.ByPeriod(cases={0: probabilities}),
        "default": lcm.ByPeriod(cases={}, default=probabilities),
        "until": lcm.ByPeriod.until(
            stop_period_exclusive=2, law=probabilities, then="done"
        ),
        "mapped-until": lcm.ByPeriod.until(
            stop_period_exclusive=2, law="work", then="done"
        ).with_mapped_laws(func=to_probabilities),
    }
    probabilities.clear()
    law = cast(
        "dict[str, lcm.StochasticTransition]",
        schedules[wrapper].resolve(n_periods=3).at(0),
    )
    assert law["work"].func is probability.func
    with pytest.raises(TypeError):
        law["done"] = probability


@pytest.mark.parametrize("mapping", [False, True])
def test_age_initial_nodes_refuses_legacy_period_selectors(*, mapping: bool) -> None:
    """Legacy age syntax must not admit the explicitly period-labelled wrappers."""
    selector = lcm.PeriodRange(start=0)
    initial_nodes = {selector: "work"} if mapping else ((selector, "work"),)
    with pytest.raises(ModelInitializationError, match=r"[Aa]ge"):
        _age_model(initial_nodes=initial_nodes)  # ty: ignore[invalid-argument-type] - Period selectors deliberately enter the legacy age boundary.


def _joint_zero() -> FloatND:
    return jnp.asarray(0.0)


def _joint_terminal_utility(*, x: ContinuousState) -> FloatND:
    return x


def _joint_probabilities() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _joint_next_x(*, shock: FloatND) -> FloatND:
    return shock


def _age_support(*, age: Age) -> FloatND:
    return jnp.asarray([10.0, 20.0]) + age


def _age_support_truth(*, age: Age) -> FloatND:
    # A legal whole-kernel phase difference with the identical support schema.
    return jnp.asarray([10.0, 20.0]) + age + 4.0


def _period_support(*, period: Period) -> FloatND:
    return jnp.asarray([10.0, 20.0]) + period


def _period_support_truth(*, period: Period) -> FloatND:
    return jnp.asarray([10.0, 20.0]) + period + 4.0


def _clock_joint(*, support: Callable[..., FloatND]) -> lcm.JointTransition:
    return lcm.JointTransition(
        support_size=2,
        support=support,
        probabilities=_joint_probabilities,
        outputs={"x": _joint_next_x},
    )


def _joint_clock_model(
    *,
    clock: Literal["age", "period"],
    source_period: int,
    origin: int,
    joint: lcm.JointTransition | lcm.Phased,
) -> lcm.Model:
    # The only edge ends at the explicit final slot. Earlier slots are unused
    # for a late start; no missing transition or unreachable target is involved.
    n_periods = source_period + 2
    source_coordinate = origin + source_period
    clock_kwargs: ClockKwargs = (
        {"n_periods": n_periods}
        if clock == "period"
        else {
            "ages": lcm.AgeGrid(
                start=origin,
                inclusive_stop=origin + n_periods - 1,
                step="Y",
            )
        }
    )
    initial_nodes = (
        lcm.InitialNodes(by_period={source_period: "work"})
        if clock == "period"
        else lcm.InitialNodes(by_age={source_coordinate: "work"})
    )
    selector = (
        lcm.Periods(values=(source_period,))
        if clock == "period"
        else (source_coordinate,)
    )
    return lcm.Model(
        **clock_kwargs,
        regimes={
            "work": lcm.Regime(
                functions={"utility": _joint_zero},
                joint_transitions={"done": {"shock": joint}},
            ),
            "done": lcm.Regime(
                states={"x": lcm.LinSpacedGrid(start=0, stop=100, n_points=3)},
                functions={"utility": _joint_terminal_utility},
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=initial_nodes,
        edges={"work": {"done": selector}},
        fixed_params={"discount_factor": 1.0},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.parametrize("source_period", [0, 2])
@pytest.mark.parametrize("bad_role", ["ordinary", "solve", "simulate"])
def test_period_clock_rejects_age_support_at_construction(
    *, source_period: int, bad_role: str
) -> None:
    """Every reachable support provider must obey the declared clock kind.

    Validation precedes the choice of log level and execution entry point,
    covering both phases and every admissible starting period.
    """
    bad = _clock_joint(support=_age_support)
    good = _clock_joint(support=_period_support)
    joint = (
        bad
        if bad_role == "ordinary"
        else lcm.Phased(
            solve=bad if bad_role == "solve" else good,
            simulate=bad if bad_role == "simulate" else good,
        )
    )
    with pytest.raises(ModelInitializationError, match="age"):
        _joint_clock_model(
            clock="period",
            source_period=source_period,
            origin=0,
            joint=joint,
        )


@pytest.mark.parametrize("log_level", ["off", "debug"])
@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize(
    ("clock", "origin", "source_period", "coordinate", "expected_value"),
    [
        ("period", 0, 0, 0.0, 15.0),
        ("period", 0, 2, 2.0, 17.0),
        ("age", 0, 2, 2.0, 17.0),
        ("age", 40, 1, 41.0, 56.0),
        ("age", 70, 1, 71.0, 86.0),
    ],
)
def test_valid_clock_support_preserves_solve_and_simulation(
    *,
    log_level: Literal["off", "debug"],
    phased: bool,
    clock: Literal["age", "period"],
    origin: int,
    source_period: int,
    coordinate: float,
    expected_value: float,
) -> None:
    """Keep valid age/period support and legal perceived/realized differences.

    With zero flow, discount one, and terminal V(x)=x, the solve expectation is
    ((10 + coordinate) + (20 + coordinate)) / 2. Its reference above is literal.
    The legal Phased case adds four only to realized support. This checks the
    simulation phase independently without relying on random sample averages.
    """
    support = _age_support if clock == "age" else _period_support
    truth = _age_support_truth if clock == "age" else _period_support_truth
    joint = (
        lcm.Phased(
            solve=_clock_joint(support=support), simulate=_clock_joint(support=truth)
        )
        if phased
        else _clock_joint(support=support)
    )
    model = _joint_clock_model(
        clock=clock,
        source_period=source_period,
        origin=origin,
        joint=joint,
    )
    assert (model.ages is None) == (clock == "period")
    solution = model.solve(params={}, log_level=log_level)
    actual_value = np.asarray(solution.value(step=source_period, regime="work"))
    np.testing.assert_array_almost_equal(
        actual_value, expected_value, decimal=DECIMAL_PRECISION
    )

    n_agents = 3
    initial_time = (
        jnp.full(n_agents, source_period, dtype=jnp.int32)
        if clock == "period"
        else jnp.full(n_agents, coordinate)
    )
    simulation = model.simulate(
        params={},
        initial_conditions={
            clock: initial_time,
            "regime_id": jnp.full(n_agents, RegimeId.work, dtype=jnp.int32),
        },
        solution=solution,
        seed=123,
        log_level=log_level,
    )
    frame = simulation.to_dataframe(use_categorical_codes=True)
    assert len(frame) == 2 * n_agents
    np.testing.assert_array_equal(
        np.sort(frame["period"].to_numpy()),
        np.repeat([source_period, source_period + 1], n_agents),
    )
    assert ("age" in frame.columns) == (clock == "age")
    terminal = frame.loc[frame["period"] == source_period + 1]
    assert len(terminal) == n_agents
    offset = 4.0 if phased else 0.0
    expected_nodes = np.asarray(
        [10.0 + coordinate + offset, 20.0 + coordinate + offset]
    )
    assert np.isin(terminal["x"].to_numpy(), expected_nodes).all()
    np.testing.assert_array_equal(terminal["value"], terminal["x"])


@pytest.mark.parametrize("phase", ["solve", "simulate"])
def test_period_header_prints_a_sub_annual_age_with_two_decimals(
    *, phase: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A Fraction-valued age reads `0.33` in the period header of both phases."""
    model = lcm.Model(
        ages=lcm.AgeGrid(exact_values=(0, Fraction(1, 3), Fraction(2, 3))),
        regimes={
            "work": lcm.Regime(functions={"utility": _flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=((0, "work"),),
        edges={"work": {"work": (0,), "done": (Fraction(1, 3),)}},
        fixed_params={"discount_factor": 0.5},
    )
    solution = model.solve(params={}, log_level="off")
    with caplog.at_level(logging.INFO, logger="lcm"):
        if phase == "solve":
            model.solve(params={}, log_level="progress")
        else:
            model.simulate(
                params={},
                initial_conditions={
                    "age": jnp.array([0.0]),
                    "regime_id": jnp.array([RegimeId.work]),
                },
                solution=solution,
                seed=3,
                log_level="progress",
            )
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("Age 0.33 (") for message in messages)


@lcm.categorical(ordered=False)
class StagedId:
    working: ScalarInt
    spending: ScalarInt
    final: ScalarInt


@lcm.categorical(ordered=True)
class Labor:
    no: ScalarInt
    yes: ScalarInt


_N_ECONOMIC_PERIODS = 3
# Largest gap, in units of the spacing, between a model with a skipped stage and
# its equivalent that reads every value once. The spacing is that of the larger of
# the value and 1, the magnitude of the utilities a value near zero is a sum of.
_SKIP_ULP = 8


@lcm.time_varying_params("wage")
def _earnings(*, labor: DiscreteAction, wage: FloatND) -> FloatND:
    return labor * wage


def _work_utility(*, labor: DiscreteAction, period: Period) -> FloatND:
    return -0.3 * labor * (1.0 + period)


def _work_utility_by_age(*, labor: DiscreteAction, age: Age) -> FloatND:
    return -0.3 * labor * (1.0 + (age - 60))


def _next_wealth_after_work(
    *, wealth: ContinuousState, earnings: FloatND
) -> ContinuousState:
    return wealth + earnings


def _spend_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _next_wealth_after_spending(
    *, wealth: ContinuousState, consumption: ContinuousAction, period: Period
) -> ContinuousState:
    return 1.05 * (wealth - consumption) + 0.1 * period


def _next_wealth_after_spending_by_age(
    *, wealth: ContinuousState, consumption: ContinuousAction, age: Age
) -> ContinuousState:
    return 1.05 * (wealth - consumption) + 0.1 * (age - 60)


def _affordable(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _final_utility(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _hand_work_utility(*, labor: DiscreteAction, period: Period) -> FloatND:
    return _work_utility(labor=labor, period=period // 2)


def _hand_next_wealth_after_spending(
    *, wealth: ContinuousState, consumption: ContinuousAction, period: Period
) -> ContinuousState:
    return _next_wealth_after_spending(
        wealth=wealth, consumption=consumption, period=period // 2
    )


def _without_discounting(*, utility: FloatND, CE: FloatND) -> FloatND:
    return utility + CE


def _staged_regimes(
    *, clock: Literal["staged", "hand", "age"]
) -> dict[RegimeName, lcm.Regime]:
    """A work decision and a consumption decision in every period.

    `"hand"` encodes the two decisions as two periods each: its functions map
    the period to the economic period, and the work decision adds the
    consumption decision's value without discounting.
    """
    wealth = lcm.LinSpacedGrid(start=1.0, stop=6.0, n_points=6)
    by_age = clock == "age"
    return {
        "working": lcm.Regime(
            states={"wealth": wealth},
            actions={"labor": lcm.DiscreteGrid(Labor)},
            state_transitions={"wealth": _next_wealth_after_work},
            functions={
                "utility": _hand_work_utility
                if clock == "hand"
                else _work_utility_by_age
                if by_age
                else _work_utility,
                "earnings": _earnings,
            },
            koopmans_aggregator=_without_discounting if clock == "hand" else None,
        ),
        "spending": lcm.Regime(
            states={"wealth": wealth},
            actions={
                "consumption": lcm.LinSpacedGrid(start=0.5, stop=5.0, n_points=10)
            },
            state_transitions={
                "wealth": _hand_next_wealth_after_spending
                if clock == "hand"
                else _next_wealth_after_spending_by_age
                if by_age
                else _next_wealth_after_spending
            },
            functions={"utility": _spend_utility},
            constraints={"affordable": _affordable},
            koopmans_aggregator=lcm.LinearAggregator() if clock == "hand" else None,
        ),
        "final": lcm.Regime(
            states={"wealth": wealth}, functions={"utility": _final_utility}
        ),
    }


def _staged_model(
    *,
    clock: Literal["staged", "hand", "age"],
    stages: Mapping[str, tuple[RegimeName, ...]] | None = None,
    edges: Mapping[RegimeName, Mapping[RegimeName, PeriodSelector] | lcm.Transition]
    | None = None,
    regimes: Mapping[RegimeName, lcm.Regime] | None = None,
    regime_id_class: type[DataclassInstance] = StagedId,
    shocks: Mapping[str, ShockGroup] | None = None,
    composite_states: Mapping[str, lcm.CompositeState] | None = None,
    discount_factor: float = 0.9,
    start_period: int = 0,
) -> lcm.Model:
    last = _N_ECONOMIC_PERIODS - 1
    if clock == "hand":
        steps = 2 * _N_ECONOMIC_PERIODS
        return lcm.Model(
            n_periods=steps,
            regimes=_staged_regimes(clock=clock),
            regime_id_class=StagedId,
            initial_nodes=lcm.InitialNodes(by_period={0: "working"}),
            edges={
                "working": {
                    "spending": lcm.Periods(values=tuple(range(0, steps - 2, 2))),
                    "final": lcm.Periods(values=(steps - 2,)),
                },
                "spending": {
                    "working": lcm.Periods(values=tuple(range(1, steps - 2, 2)))
                },
            },
            fixed_params={"discount_factor": discount_factor},
        )
    by_age = clock == "age"
    selector = lcm.AgeRange if by_age else lcm.PeriodRange
    origin = 60 if by_age else 0
    horizon: ClockKwargs = (
        {"ages": lcm.AgeGrid(start=60, inclusive_stop=60 + last, step="Y")}
        if by_age
        else {"n_periods": _N_ECONOMIC_PERIODS}
    )
    return lcm.Model(
        **horizon,
        stages={"work": ("working",), "spend": ("spending", "final")}
        if stages is None
        else stages,
        regimes=_staged_regimes(clock=clock) if regimes is None else regimes,
        regime_id_class=regime_id_class,
        shocks={} if shocks is None else shocks,
        composite_states={} if composite_states is None else composite_states,
        initial_nodes=lcm.InitialNodes(by_age={60: "working"})
        if by_age
        else lcm.InitialNodes(by_period={start_period: "working"}),
        edges={
            "working": {
                "spending": selector(exclusive_stop=origin + last),
                "final": selector(start=origin + last),
            },
            "spending": {"working": selector(exclusive_stop=origin + last)},
        }
        if edges is None
        else edges,
        fixed_params={"discount_factor": discount_factor},
    )


def _wage(*, steps: bool) -> lcm.TimeVarying:
    wages = jnp.array([1.0, 1.5, 2.0])
    if steps:
        return lcm.TimeVarying(
            values=jnp.repeat(wages, 2), periods=tuple(range(2 * len(wages)))
        )
    return lcm.TimeVarying(values=wages, periods=(0, 1, 2))


def _staged_frame(*, model: lcm.Model, clock: str) -> pd.DataFrame:
    n_agents = 4
    initial_time = (
        {"age": jnp.full(n_agents, 60)}
        if clock == "age"
        else {"period": jnp.zeros(n_agents, dtype=jnp.int32)}
    )
    params = (
        {"wage": _wage(steps=clock == "hand")}
        if clock != "age"
        else {
            "wage": lcm.TimeVarying(
                values=jnp.array([1.0, 1.5, 2.0]), ages=(60, 61, 62)
            )
        }
    )
    return model.simulate(
        params=params,
        initial_conditions={
            **initial_time,
            "regime_id": jnp.full(n_agents, StagedId.working, dtype=jnp.int32),
            "wealth": jnp.array([1.0, 2.5, 4.0, 6.0]),
        },
        seed=7,
        log_level="off",
    ).to_dataframe()


def _transition_edges() -> dict[RegimeName, lcm.Transition]:
    """The staged edges as laws with targets; the work stage moves on at the end."""
    last = _N_ECONOMIC_PERIODS - 1
    before_last = lcm.PeriodRange(exclusive_stop=last)
    return {
        "working": lcm.Transition(
            targets={"spending": before_last, "final": lcm.PeriodRange(start=last)},
            law=lcm.ByPeriod(cases={before_last: "spending", last: "final"}),
        ),
        "spending": lcm.Transition(targets={"working": before_last}, law="working"),
    }


@pytest.mark.parametrize("declared", ["selectors", "transitions"])
def test_stages_solve_and_simulate_as_their_hand_encoding(declared: str) -> None:
    """Two decisions per period equal one period per decision, linked undiscounted.

    `period` reads the economic period in user functions and in the frame, a
    time-varying parameter is indexed by economic period, and agents start in
    the first stage. A law of the first stage moves on in the last period.
    """
    staged = _staged_model(
        clock="staged", edges=_transition_edges() if declared == "transitions" else None
    )
    hand = _staged_model(clock="hand")
    staged_values = staged.solve(
        params={"wage": _wage(steps=False)}, log_level="off"
    ).values
    hand_values = hand.solve(params={"wage": _wage(steps=True)}, log_level="off").values
    assert staged.n_steps == hand.n_steps == 2 * _N_ECONOMIC_PERIODS
    assert {step: set(regimes) for step, regimes in staged_values.items()} == {
        step: set(regimes) for step, regimes in hand_values.items()
    }
    for step, regimes in hand_values.items():
        for name, value in regimes.items():
            np.testing.assert_array_equal(staged_values[step][name], value)

    staged_frame = _staged_frame(model=staged, clock="staged")
    hand_frame = _staged_frame(model=hand, clock="hand")
    np.testing.assert_array_equal(staged_frame["period"], hand_frame["period"] // 2)
    assert staged_frame["stage"].tolist() == [
        "work" if step % 2 == 0 else "spend" for step in hand_frame["period"]
    ]
    assert "stage" not in hand_frame.columns
    for column in ("agent_id", "regime_name", "value", "wealth", "labor"):
        pd.testing.assert_series_equal(staged_frame[column], hand_frame[column])


def test_stages_of_an_age_model_read_the_age_of_the_period() -> None:
    """An age model's stages read the period's age where a period model reads it."""
    by_age = _staged_model(clock="age")
    by_period = _staged_model(clock="staged")
    age_values = by_age.solve(
        params={
            "wage": lcm.TimeVarying(
                values=jnp.array([1.0, 1.5, 2.0]), ages=(60, 61, 62)
            )
        },
        log_level="off",
    ).values
    period_values = by_period.solve(
        params={"wage": _wage(steps=False)}, log_level="off"
    ).values
    for step, regimes in period_values.items():
        for name, value in regimes.items():
            np.testing.assert_array_equal(age_values[step][name], value)
    frame = _staged_frame(model=by_age, clock="age")
    np.testing.assert_array_equal(frame["age"], 60 + frame["period"])
    assert frame["stage"].cat.categories.tolist() == ["work", "spend"]


class _StagedChange(TypedDict, total=False):
    stages: Mapping[str, tuple[RegimeName, ...]]
    edges: Mapping[RegimeName, Mapping[RegimeName, PeriodSelector]]
    regimes: Mapping[RegimeName, lcm.Regime]
    shocks: Mapping[str, ShockGroup]


def _wealth_resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _wealth_savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _to_spending() -> IntND:
    return jnp.asarray(StagedId.spending)


def _to_spending_transition() -> lcm.Transition:
    return lcm.Transition(
        targets={"spending": lcm.PeriodRange()},
        law=lcm.DeterministicTransition(func=_to_spending),
    )


def _wealth_kept(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _wealth_after_saving(*, savings: FloatND) -> ContinuousState:
    return savings


def _out_of_stage_order(case: str) -> _StagedChange:
    regimes = _staged_regimes(clock="staged")
    if case == "aggregator":
        regimes["working"] = regimes["working"].replace(
            koopmans_aggregator=lcm.LinearAggregator()
        )
    source, destination = (
        ("spending", "working")
        if case == "composition_action"
        else ("working", "spending")
    )
    if case in {"composition_action", "observes"}:
        regimes[source] = regimes[source].replace(
            composition_action=lcm.CompositionAction(
                destinations={
                    "self": lcm.Destination(
                        regime=destination, projection={"wealth": _wealth_kept}
                    )
                },
                rule=lcm.PCOnActions(),
                observes=("eps",) if case == "observes" else (),
            )
        )
    if case == "observes":
        regimes["spending"] = regimes["spending"].replace(
            functions={"utility": _spend_utility_with_shock}, shocks=("income",)
        )
    if case == "egm":
        regimes["working"] = ConsumptionSavingsRegime(
            states={"wealth": lcm.LinSpacedGrid(start=1.0, stop=6.0, n_points=6)},
            actions={
                "consumption": lcm.LinSpacedGrid(start=0.5, stop=5.0, n_points=10)
            },
            functions={
                "utility": _spend_utility,
                "resources": _wealth_resources,
                "savings": _wealth_savings,
            },
            state_transitions={"wealth": _wealth_after_saving},
            solver=EGM(savings_grid=lcm.LinSpacedGrid(start=0.0, stop=6.0, n_points=7)),
            liquid=LiquidMargin(
                state="wealth",
                action="consumption",
                resources="resources",
                post_decision_state="savings",
            ),
        )
    changes: dict[str, _StagedChange] = {  # ty: ignore[invalid-assignment]
        "missing": {"stages": {"work": ("working",), "spend": ("spending",)}},
        "repeated": {
            "stages": {"work": ("working", "final"), "spend": ("spending", "final")}
        },
        "composition_action": {"regimes": regimes},
        "observes": {
            "regimes": regimes,
            "shocks": {
                "income": ShockGroup(
                    shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                    nodes=Product(eps=GaussHermite(3)),
                )
            },
        },
        "aggregator": {"regimes": regimes},
        "egm": {"regimes": regimes},
        "process": {
            "regimes": {
                **regimes,
                "spending": regimes["spending"].replace(
                    states={
                        **regimes["spending"].states,
                        "income": lcm.NormalIIDProcess(
                            n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0
                        ),
                    }
                ),
            },
            "edges": {
                "working": {
                    "spending": lcm.PeriodRange(exclusive_stop=_N_ECONOMIC_PERIODS - 1),
                    "final": lcm.PeriodRange(start=_N_ECONOMIC_PERIODS - 1),
                },
                "spending": {
                    "spending": lcm.PeriodRange(exclusive_stop=_N_ECONOMIC_PERIODS - 2),
                    "final": lcm.Periods(values=(_N_ECONOMIC_PERIODS - 2,)),
                },
            },
        },
        "beyond": {
            "stages": {"work": ("working", "final"), "spend": ("spending",)},
            "edges": {
                "working": {
                    "working": lcm.PeriodRange(exclusive_stop=_N_ECONOMIC_PERIODS - 1),
                    "final": lcm.PeriodRange(start=_N_ECONOMIC_PERIODS - 1),
                },
                "spending": {
                    "spending": lcm.PeriodRange(exclusive_stop=_N_ECONOMIC_PERIODS - 1)
                },
            },
        },
        "pair": {
            "edges": {  # ty: ignore[invalid-argument-type]
                "working": lcm.Phased(
                    solve=_to_spending_transition(),
                    simulate=_to_spending_transition(),
                ),
                "spending": {"working": lcm.PeriodRange()},
            }
        },
        "final": {
            "edges": {
                "working": {
                    "spending": lcm.PeriodRange(exclusive_stop=_N_ECONOMIC_PERIODS - 1),
                    "final": lcm.PeriodRange(start=_N_ECONOMIC_PERIODS - 1),
                },
                "spending": {"working": lcm.Periods(values=(_N_ECONOMIC_PERIODS - 1,))},
            }
        },
        "mixed": {
            "edges": {  # ty: ignore[invalid-argument-type]
                "working": lcm.Transition(
                    targets={
                        "spending": lcm.PeriodRange(
                            exclusive_stop=_N_ECONOMIC_PERIODS - 1
                        ),
                        "working": lcm.Periods(values=(0,)),
                        "final": lcm.PeriodRange(start=_N_ECONOMIC_PERIODS - 1),
                    },
                    law=lcm.DeterministicTransition(func=_to_spending),
                ),
                "spending": {
                    "working": lcm.PeriodRange(exclusive_stop=_N_ECONOMIC_PERIODS - 1)
                },
            }
        },
    }
    return changes[case]


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", r"in none: \['final'\]"),
        ("repeated", r"in several stages: \['final'\]"),
        (
            "composition_action",
            (
                r"composition action of 'spending' in stage 'spend' reads 'working' "
                r"of the earlier"
            ),
        ),
        ("observes", r"observes \['eps'\] of 'spending', drawn in its later stage"),
        ("aggregator", r"'working' declares a Koopmans aggregator in stage 'work'"),
        ("egm", r"'working' uses an endogenous-grid solver in stage 'work'"),
        (
            "process",
            (
                r"'spending' sits out stage 'work' .* processes \['income'\] are "
                r"not drawn there; declare the pass-through of 'spending' in stage "
                r"'work' explicitly \(a regime with the same states, one action, no "
                r"shock, zero utility and the value handed through\)"
            ),
        ),
        ("beyond", r"lands in 'final' at its next occurrence, which lies beyond"),
        ("pair", r"'working' maps to a `Phased` pair of transitions. Phase"),
        ("final", r"'spending' → 'working' selects only the final age"),
        (
            "mixed",
            (
                r"at period 0, the edges of 'working' in stage 'work' land in its "
                r"period, into \['spending'\], and in the next period, into "
                r"\['working'\]; a source's edges at one age either all stay in its "
                r"period or all cross into the next one; declare the pass-through of "
                r"'working' in stage 'spend' explicitly"
            ),
        ),
    ],
)
def test_stages_refuse_declarations_out_of_stage_order(
    *, case: str, match: str
) -> None:
    """A refusal names the regimes and stages it concerns.

    Where an edge into a regime that sits out a stage is refused, the refusal
    asks for that regime's pass-through in the stage, which the modeller can
    declare explicitly (specification, [A 14]).
    """
    with pytest.raises(ModelInitializationError, match=match):
        _staged_model(clock="staged", **_out_of_stage_order(case))


@lcm.categorical(ordered=False)
class SearchId:
    search: ScalarInt
    trainee: ScalarInt
    employed: ScalarInt
    unemployed: ScalarInt


def _search_cost(effort: DiscreteAction) -> FloatND:
    return -0.3 * effort


def _offer_wage(effort: DiscreteAction) -> FloatND:
    return 0.5 + 1.0 * effort


def _trainee_utility() -> FloatND:
    return jnp.asarray(0.0)


def _employed_utility(*, wage: ContinuousState, hours: DiscreteAction) -> FloatND:
    return wage + 0.1 * hours


def _unemployed_utility(wage: ContinuousState) -> FloatND:
    return 0.5 * wage


def _offer_is_good(wage: ContinuousState) -> BoolND:
    return wage >= 1.0


def _benefit(wage: ContinuousState) -> FloatND:
    return 2.0 + 0.0 * wage


def _certain() -> FloatND:
    return jnp.asarray(1.0)


def test_a_gated_edge_that_skips_a_stage_asks_for_the_pass_through() -> None:
    """A gated edge from `search` into `consume` skips the stage `train`.

    The refusal names the landing regimes and asks for their pass-through in
    `train`, which the modeller can declare explicitly (specification, [A 14]).
    """
    wage = lcm.LinSpacedGrid(start=0.0, stop=3.0, n_points=7)
    with pytest.raises(
        ModelInitializationError,
        match=(
            r"the gated edge 'search' -> 'employed' lands in 'employed' of stage "
            r"'consume'; a gated edge and its fallbacks land in the next stage; "
            r"declare the pass-through of 'employed' in stage 'train' explicitly"
            r".*declare the pass-through of 'unemployed' in stage 'train' "
            r"explicitly"
        ),
    ):
        lcm.Model(
            n_periods=1,
            stages={
                "search": ("search",),
                "train": ("trainee",),
                "consume": ("employed", "unemployed"),
            },
            regimes={
                "search": lcm.Regime(
                    actions={"effort": lcm.DiscreteGrid(values=jnp.asarray([0, 1]))},
                    functions={"utility": _search_cost},
                    state_transitions={"wage": _offer_wage},
                ),
                "trainee": lcm.Regime(functions={"utility": _trainee_utility}),
                "employed": lcm.Regime(
                    states={"wage": wage},
                    actions={"hours": lcm.DiscreteGrid(values=jnp.asarray([0, 1]))},
                    functions={"utility": _employed_utility},
                ),
                "unemployed": lcm.Regime(
                    states={"wage": wage}, functions={"utility": _unemployed_utility}
                ),
            },
            regime_id_class=SearchId,
            initial_nodes=lcm.InitialNodes(by_period={0: ("search", "trainee")}),
            edges={
                "search": lcm.Transition(
                    law=lcm.ByPeriod(
                        cases={0: {"employed": lcm.StochasticTransition(func=_certain)}}
                    ),
                    gates={
                        "employed": lcm.Gate(
                            predicate=_offer_is_good,
                            routes={
                                "only": lcm.MemberRoute(
                                    fallback=lcm.ProjectedRegimeValue(
                                        regime="unemployed",
                                        projection={"wage": _benefit},
                                    )
                                )
                            },
                        )
                    },
                )
            },
        )


@lcm.categorical(ordered=False)
class GameId:
    a_moves: ScalarInt
    b_moves: ScalarInt
    b_last: ScalarInt


@lcm.categorical(ordered=True)
class Pot:
    empty: ScalarInt
    one: ScalarInt
    two: ScalarInt
    three: ScalarInt
    four: ScalarInt


@lcm.categorical(ordered=True)
class Take:
    none: ScalarInt
    one: ScalarInt
    two: ScalarInt


_GAME_PERIODS = 3
_GAME_DISCOUNT = 0.9


def _mover_utility(*, take: DiscreteAction) -> FloatND:
    return jnp.sqrt(take.astype(float))


def _bystander_utility() -> FloatND:
    return jnp.asarray(0.0)


def _available(*, take: DiscreteAction, pot: DiscreteState) -> BoolND:
    return take <= pot


def _pot_after_take(*, pot: DiscreteState, take: DiscreteAction) -> DiscreteState:
    return pot - take


def _pot_regrown(*, pot: DiscreteState, take: DiscreteAction) -> DiscreteState:
    return jnp.minimum(pot - take + 2, Pot.four)


def _player_regime(
    *,
    mover: Literal["a", "b"],
    next_pot: Callable[..., DiscreteState] | None,
    mover_weight: Callable[..., FloatND] | float,
) -> lcm.Regime:
    """The regime in which `mover` takes from the pot and the other player waits."""
    players = ("a", "b")
    return lcm.Regime(
        states={"pot": lcm.DiscreteGrid(Pot)},
        actions={"take": lcm.DiscreteGrid(Take)},
        state_transitions={} if next_pot is None else {"pot": next_pot},
        constraints={"available": _available},
        functions={
            "utility": lcm.CollectiveUtility(
                utilities={
                    player: _mover_utility if player == mover else _bystander_utility
                    for player in players
                },
                objective=lcm.ParetoObjective(
                    weights={
                        player: mover_weight if player == mover else 0.0
                        for player in players
                    }
                ),
            )
        },
    )


def _best_take(payoffs: Mapping[int, float]) -> int:
    """The take of the highest payoff, which beats the second highest clearly."""
    ranked = sorted(payoffs, key=payoffs.__getitem__)
    if len(ranked) > 1:
        assert payoffs[ranked[-1]] - payoffs[ranked[-2]] > 1e-3
    return ranked[-1]


def _backward_induction(
    *, b_rests: tuple[int, ...]
) -> dict[tuple[int, RegimeName], np.ndarray]:
    """Each player's value per pot before each move, by subgame-perfect induction.

    Rows are pots, columns the players a and b. Every mover's best take is clearly
    better than its second best, so the equilibrium does not depend on a tie rule.
    In a period of `b_rests` only a moves, and the pot does not regrow.
    """
    pots = range(len(Pot.__dataclass_fields__))
    values: dict[tuple[int, RegimeName], np.ndarray] = {}
    for period in reversed(range(_GAME_PERIODS)):
        final = period == _GAME_PERIODS - 1
        if period in b_rests:
            ahead = _GAME_DISCOUNT * values[(period + 1, "a_moves")]
            before_a = np.zeros((len(pots), 2))
            for pot in pots:
                takes = range(min(pot, Take.two) + 1)
                payoffs = {t: np.sqrt(t) + ahead[pot - t, 0] for t in takes}
                take = _best_take(payoffs)
                before_a[pot] = (payoffs[take], ahead[pot - take, 1])
            values[(period, "a_moves")] = before_a
            continue
        before_b = np.zeros((len(pots), 2))
        for pot in pots:
            takes = range(min(pot, Take.two) + 1)
            if final:
                take = _best_take({t: np.sqrt(t) for t in takes})
                before_b[pot] = (0.0, np.sqrt(take))
                continue
            ahead = values[(period + 1, "a_moves")]
            regrown = {t: min(pot - t + 2, Pot.four) for t in takes}
            payoffs = {
                t: np.sqrt(t) + _GAME_DISCOUNT * ahead[regrown[t], 1] for t in takes
            }
            take = _best_take(payoffs)
            before_b[pot] = (_GAME_DISCOUNT * ahead[regrown[take], 0], payoffs[take])
        values[(period, "b_last" if final else "b_moves")] = before_b
        before_a = np.zeros((len(pots), 2))
        for pot in pots:
            takes = range(min(pot, Take.two) + 1)
            payoffs = {t: np.sqrt(t) + before_b[pot - t, 0] for t in takes}
            take = _best_take(payoffs)
            before_a[pot] = (payoffs[take], before_b[pot - take, 1])
        values[(period, "a_moves")] = before_a
    return values


def _game(
    *, mover_weight: Callable[..., FloatND] | float, b_rests: tuple[int, ...] = ()
) -> lcm.Model:
    """Player a moves, then player b, in every period; each mover weighs itself.

    In the periods `b_rests`, a moves alone and its move lands at its next one.
    """
    last = _GAME_PERIODS - 1
    both_move = lcm.Periods(values=tuple(p for p in range(last) if p not in b_rests))
    return lcm.Model(
        n_periods=_GAME_PERIODS,
        stages={"a": ("a_moves",), "b": ("b_moves", "b_last")},
        regimes={
            "a_moves": _player_regime(
                mover="a", next_pot=_pot_after_take, mover_weight=mover_weight
            ),
            "b_moves": _player_regime(
                mover="b", next_pot=_pot_regrown, mover_weight=mover_weight
            ),
            "b_last": _player_regime(
                mover="b", next_pot=None, mover_weight=mover_weight
            ),
        },
        regime_id_class=GameId,
        initial_nodes=lcm.InitialNodes(by_period={0: "a_moves"}),
        edges={
            "a_moves": {
                "b_moves": both_move,
                "b_last": lcm.Periods(values=(last,)),
                **({"a_moves": lcm.Periods(values=b_rests)} if b_rests else {}),
            },
            "b_moves": {"a_moves": both_move},
        },
        fixed_params={"discount_factor": _GAME_DISCOUNT},
    )


@pytest.mark.parametrize("b_rests", [(), (1,)])
def test_two_players_moving_in_turn_reach_the_subgame_perfect_values(
    b_rests: tuple[int, ...],
) -> None:
    """A sequential-move game: within each period player a moves, then player b.

    Each mover maximizes its own value; the other player's value follows the
    mover's choice. The values equal an independent backward induction, also
    when b sits out a period and a moves twice in a row.
    """
    solved = (
        _game(mover_weight=1.0, b_rests=b_rests)
        .solve(params={}, log_level="off")
        .values
    )
    for (period, regime), expected in _backward_induction(b_rests=b_rests).items():
        step = 2 * period + (regime != "a_moves")
        np.testing.assert_array_almost_equal(
            solved[step][regime], expected, decimal=DECIMAL_PRECISION
        )


@lcm.categorical(ordered=False)
class MarriageId:
    couple_sep: ScalarInt
    couple: ScalarInt
    couple_final: ScalarInt
    single: ScalarInt
    single_final: ScalarInt


@lcm.categorical(ordered=False)
class HandMarriageId:
    couple_sep: ScalarInt
    couple: ScalarInt
    couple_final: ScalarInt
    single: ScalarInt
    single_final: ScalarInt
    single_wait: ScalarInt


def _match_quality(*, wealth: ContinuousState, love: FloatND) -> FloatND:
    return love - 0.05 * wealth


def _shared_consumption(*, wealth: ContinuousState, share: ContinuousAction) -> FloatND:
    return share * wealth


def _half_of_consumption_utility(*, consumption: FloatND) -> FloatND:
    return jnp.log(0.5 * consumption)


def _single_consumption_utility(*, consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def _next_couple_wealth(
    *, wealth: ContinuousState, consumption: FloatND
) -> ContinuousState:
    return jnp.minimum(1.05 * (wealth - consumption) + 1.0, 12.0)


def _next_single_wealth(
    *, wealth: ContinuousState, consumption: FloatND
) -> ContinuousState:
    return jnp.minimum(1.05 * (wealth - consumption) + 0.5, 6.0)


def _half_of_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return 0.5 * wealth


def _final_couple_utility(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(0.5 * wealth)


def _waiting() -> FloatND:
    return jnp.asarray(0.0)


def _single_next_regime(*, period: Period) -> IntND:
    return jnp.where(
        period < _N_ECONOMIC_PERIODS - 2, MarriageId.single, MarriageId.single_final
    )


def _single_regime_probabilities(*, period: Period) -> FloatND:
    return jnp.where(
        period < _N_ECONOMIC_PERIODS - 2,
        jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
        jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def _single_stays(*, stay: FloatND) -> FloatND:
    return stay


def _next_single_wealth_with_income(
    *, wealth: ContinuousState, consumption: FloatND, income: FloatND
) -> ContinuousState:
    return jnp.minimum(1.05 * (wealth - consumption) + income, 6.0)


def _single_edges(
    *, variant: str, targets: Mapping[RegimeName, AgeSelector | PeriodSelector]
) -> Mapping[RegimeName, AgeSelector | PeriodSelector] | lcm.Transition:
    """The single's edges: selectors, or a law of the form `variant` names."""
    law: TransitionLaw
    if variant == "recoded":
        law = lcm.DeterministicTransition(func=_single_next_regime)
    elif variant == "spread":
        law = lcm.StochasticTransition(func=_single_regime_probabilities)
    elif variant == "cells":
        law = lcm.ByPeriod(
            cases={
                targets["single"]: {
                    "single": lcm.StochasticTransition(func=_single_stays)
                },
                targets["single_final"]: "single_final",
            }
        )
    else:
        return targets
    return lcm.Transition(targets=targets, law=law)


def _marriage(*, hand: bool, variant: str = "plain") -> lcm.Model:
    """Couples decide to part, then spend; singles only spend.

    `hand` writes the single's wait in the parting stage as a regime of its own,
    with edges into and out of it. `variant` declares the same model by age, with
    `Phased` edges, or with a regime law or per-target cells and state laws for
    the single.
    """
    last = _N_ECONOMIC_PERIODS - 1
    couple_wealth = lcm.LinSpacedGrid(start=2.0, stop=12.0, n_points=6)
    single_wealth = lcm.LinSpacedGrid(start=1.0, stop=6.0, n_points=6)
    share = lcm.LinSpacedGrid(start=0.1, stop=1.0, n_points=10)
    single = "single_wait" if hand else "single"
    regimes = {
        "couple_sep": lcm.Regime(
            states={"wealth": couple_wealth},
            state_transitions={"wealth": _wealth_kept},
            functions={
                "utility": lcm.CollectiveUtility(
                    utilities={"f": _match_quality, "m": _match_quality}
                )
            },
            composition_action=lcm.CompositionAction(
                destinations={
                    member: lcm.Destination(
                        regime=single, projection={"wealth": _half_of_wealth}
                    )
                    for member in ("f", "m")
                },
                rule=lcm.PCOnActions(),
            ),
        ),
        "couple": lcm.Regime(
            states={"wealth": couple_wealth},
            actions={"share": share},
            state_transitions={"wealth": _next_couple_wealth},
            functions={
                "consumption": _shared_consumption,
                "utility": lcm.CollectiveUtility(
                    utilities={
                        "f": _half_of_consumption_utility,
                        "m": _half_of_consumption_utility,
                    }
                ),
            },
        ),
        "couple_final": lcm.Regime(
            states={"wealth": couple_wealth},
            functions={
                "utility": lcm.CollectiveUtility(
                    utilities={"f": _final_couple_utility, "m": _final_couple_utility}
                )
            },
        ),
        "single": lcm.Regime(
            states={"wealth": single_wealth},
            actions={"share": share},
            state_transitions={
                "wealth": {
                    "single": _next_single_wealth_with_income,
                    "single_final": _next_single_wealth,
                }
                if variant == "cells"
                else _next_single_wealth
            },
            functions={
                "consumption": _shared_consumption,
                "utility": _single_consumption_utility,
            },
        ),
    }
    regimes["single_final"] = lcm.Regime(
        states={"wealth": single_wealth}, functions={"utility": _final_utility}
    )
    if hand:
        regimes["single_wait"] = lcm.Regime(
            states={"wealth": single_wealth},
            state_transitions={"wealth": lcm.fixed_transition("wealth")},
            functions={"utility": _waiting},
        )
    by_age = variant == "age"
    always, before_end, at_end = (
        (
            lcm.AgeRange(),
            lcm.AgeRange(exclusive_stop=60 + last - 1),
            lcm.AgeRange(start=60 + last - 1, exclusive_stop=60 + last),
        )
        if by_age
        else (
            lcm.PeriodRange(),
            lcm.PeriodRange(exclusive_stop=last - 1),
            lcm.Periods(values=(last - 1,)),
        )
    )
    single_targets: dict[RegimeName, AgeSelector | PeriodSelector] = {
        single: before_end,
        "single_final": at_end,
    }
    edges: dict[
        RegimeName, Mapping[RegimeName, AgeSelector | PeriodSelector] | lcm.Transition
    ] = {
        "couple_sep": {"couple": always},
        "couple": {"couple_sep": before_end, "couple_final": at_end},
        "single": _single_edges(variant=variant, targets=single_targets),
    }
    if hand:
        edges["single_wait"] = {"single": always}
    clock: ClockKwargs = (
        {"ages": lcm.AgeGrid(start=60, inclusive_stop=60 + last, step="Y")}
        if by_age
        else {"n_periods": _N_ECONOMIC_PERIODS}
    )
    starts = ("couple_sep", single)
    return lcm.Model(
        **clock,
        stages={
            "part": ("couple_sep", "couple_final", "single_final")
            + (("single_wait",) if hand else ()),
            "spend": ("couple", "single"),
        },
        regimes=regimes,
        regime_id_class=HandMarriageId if hand else MarriageId,
        initial_nodes=lcm.InitialNodes(by_age={60: starts})
        if by_age
        else lcm.InitialNodes(by_period={0: starts}),
        edges=lcm.Phased(solve=edges, simulate=edges) if variant == "phased" else edges,
        fixed_params={"discount_factor": 0.9},
    )


def _marriage_frame(*, model: lcm.Model, hand: bool) -> pd.DataFrame:
    codes = model.regime_names_to_ids
    single = codes["single_wait" if hand else "single"]
    return model.simulate(
        params={"couple_sep": {"love": 0.2}},
        initial_conditions={
            "period": jnp.zeros(4, dtype=jnp.int32),
            "regime_id": jnp.array(
                [codes["couple_sep"], codes["couple_sep"], single, single],
                dtype=jnp.int32,
            ),
            "own_member": jnp.array(
                [model.member_names_to_ids[member] for member in ("f", "m", "f", "m")],
                dtype=jnp.int32,
            ),
            "wealth": jnp.array([4.0, 10.0, 2.0, 5.0]),
        },
        seed=5,
        log_level="off",
    ).to_dataframe()


def test_an_edge_lands_at_the_next_occurrence_of_its_target() -> None:
    """A single who spends once per period equals one who waits out the parting.

    The single's edge to itself skips the parting stage, the couple's composition action
    enters the single in a later stage, and singles start in their own stage; written
    with a hand-made waiting regime, the values and the simulated rows agree. The
    waiting regime interpolates the single's value once more on the way, so the two
    agree to within a few units in the last place.
    """
    params = {"couple_sep": {"love": 0.2}}
    staged = _marriage(hand=False)
    hand = _marriage(hand=True)
    staged_values = staged.solve(params=params, log_level="off").values
    hand_values = hand.solve(params=params, log_level="off").values
    for step, regimes in hand_values.items():
        for name, value in regimes.items():
            if name != "single_wait":
                assert_agrees_to_ulp(
                    got=staged_values[step][name],
                    expected=value,
                    n_ulp=_SKIP_ULP,
                    operand_magnitude=1.0,
                )
    staged_frame = _marriage_frame(model=staged, hand=False)
    hand_frame = _marriage_frame(model=hand, hand=True)
    hand_frame = hand_frame[hand_frame["regime_name"] != "single_wait"]
    assert "single" in set(staged_frame["regime_name"])
    assert set(staged_frame["regime_name"]) == set(hand_frame["regime_name"])
    staged_frame = staged_frame.reset_index(drop=True)
    hand_frame = hand_frame.reset_index(drop=True)
    floats = [
        column
        for column in staged_frame.columns
        if pd.api.types.is_float_dtype(staged_frame[column])
    ]
    pd.testing.assert_frame_equal(
        staged_frame.drop(columns=floats),
        hand_frame.drop(columns=floats),
        check_categorical=False,
    )
    for column in floats:
        assert_agrees_to_ulp(
            got=staged_frame[column],
            expected=hand_frame[column],
            n_ulp=_SKIP_ULP,
            operand_magnitude=1.0,
        )


def test_leaving_reads_the_destination_at_its_occurrence_later_in_the_period() -> None:
    """Without love every couple parts: its value is the single's value that period.

    A partner leaves for half the couple's wealth, a node of the single's grid, and
    gets the single's spending-stage value there, undiscounted.
    """
    values = (
        _marriage(hand=False)
        .solve(params={"couple_sep": {"love": -100.0}}, log_level="off")
        .values
    )
    for period in range(_N_ECONOMIC_PERIODS - 1):
        single = np.asarray(values[2 * period + 1]["single"])[..., 0]
        np.testing.assert_array_almost_equal(
            values[2 * period]["couple_sep"],
            np.stack([single, single], axis=-1),
            decimal=DECIMAL_PRECISION,
        )


@lcm.categorical(ordered=False)
class SaverId:
    single: ScalarInt
    single_final: ScalarInt


@lcm.categorical(ordered=False)
class IdleSaverId:
    single: ScalarInt
    single_final: ScalarInt
    idle: ScalarInt
    rest: ScalarInt


def _saver_grid(period: int) -> lcm.LinSpacedGrid:
    return lcm.LinSpacedGrid(start=1.0, stop=6.0 + period, n_points=6)


def _saver_grid_by_age(age: float) -> lcm.LinSpacedGrid:
    return _saver_grid(round(age) - 60)


def _saver(
    *,
    stages: Mapping[str, tuple[RegimeName, ...]] | None,
    grid_by_period: bool = False,
    discount_factor: float | None = 0.9,
    by_age: bool = False,
) -> lcm.Model:
    """The marriage model's single alone: one spending decision per period.

    Without `stages` the model declares no stages. With them, the regimes `idle`
    and `rest` fill the stages the single sits out. `discount_factor=None` leaves
    the discount factor to the parameters. `by_age` declares the clock by age,
    from 60, and a period-specific grid by age.
    """
    last = _N_ECONOMIC_PERIODS - 1
    wealth = (
        lcm.AgeSpecializedGrid(build=_saver_grid_by_age)
        if grid_by_period and by_age
        else lcm.PeriodSpecializedGrid(build=_saver_grid, signature=int)
        if grid_by_period
        else _saver_grid(0)
    )
    regimes = {
        "single": lcm.Regime(
            states={"wealth": wealth},
            actions={"share": lcm.LinSpacedGrid(start=0.1, stop=1.0, n_points=10)},
            state_transitions={"wealth": _next_single_wealth},
            functions={
                "consumption": _shared_consumption,
                "utility": _single_consumption_utility,
            },
        ),
        "single_final": lcm.Regime(
            states={"wealth": wealth}, functions={"utility": _final_utility}
        ),
    }
    idle = () if stages is None else ("idle", "rest")
    for name in idle:
        regimes[name] = lcm.Regime(functions={"utility": _waiting})
    clock: ClockKwargs = (
        {"ages": lcm.AgeGrid(start=60, inclusive_stop=60 + last, step="Y")}
        if by_age
        else {"n_periods": _N_ECONOMIC_PERIODS}
    )
    origin = 60 if by_age else 0
    selector = lcm.AgeRange if by_age else lcm.PeriodRange
    return lcm.Model(
        **clock,
        stages=stages,
        regimes=regimes,
        regime_id_class=SaverId if stages is None else IdleSaverId,
        initial_nodes=lcm.InitialNodes(by_age={60: ("single", *idle)})
        if by_age
        else lcm.InitialNodes(by_period={0: ("single", *idle)}),
        edges={
            "single": {
                "single": selector(exclusive_stop=origin + last - 1),
                "single_final": selector(
                    start=origin + last - 1, exclusive_stop=origin + last
                ),
            }
        },
        fixed_params={}
        if discount_factor is None
        else {"discount_factor": discount_factor},
    )


_SAVER_SITS_OUT_FIRST = {
    "wait": ("idle", "rest"),
    "spend": ("single", "single_final"),
}


def _hand_saver_values(*, landing: int) -> dict[int, np.ndarray]:
    """The saver's value on period-specific grids by backward recursion in NumPy.

    Each period interpolates the next period's value once, linearly, and
    discounts it once. `landing` is the number of periods ahead whose grid the
    value read is taken to live on: 1 in the model, 0 for a read that lands on
    the grid of the source's period.
    """
    last = _N_ECONOMIC_PERIODS - 1
    shares = np.linspace(0.1, 1.0, 10)
    nodes = {
        period: np.linspace(1.0, 6.0 + period, 6)
        for period in range(_N_ECONOMIC_PERIODS)
    }
    values = {last: np.log(nodes[last])}
    for period in reversed(range(last)):
        consumption = shares * nodes[period][:, None]
        next_wealth = np.minimum(
            1.05 * (nodes[period][:, None] - consumption) + 0.5, 6.0
        )
        ahead = nodes[period + landing]
        position = (next_wealth - ahead[0]) / (ahead[1] - ahead[0])
        lower = np.clip(np.floor(position).astype(int), 0, len(ahead) - 2)
        weight = position - lower
        read = values[period + 1]
        continuation = (1.0 - weight) * read[lower] + weight * read[lower + 1]
        values[period] = (np.log(consumption) + 0.9 * continuation).max(axis=1)
    return values


@pytest.mark.parametrize("by_age", [False, True])
def test_a_crossing_edge_reads_the_grid_of_the_period_it_lands_in(
    *, by_age: bool
) -> None:
    """A saver reads its next value on the grid of the period it lands in.

    The wealth grid widens with the period. By period, the saver sits out the
    first stage; by age, it sits out the last, so the step after its own lies in
    its own period, whose age and grid are the wrong ones to read. Oracle: a
    backward recursion in NumPy with one interpolation and one discount per
    period. The same recursion reading on the grid of the source's period is far
    from the model's values.
    """
    stages = (
        dict(reversed(_SAVER_SITS_OUT_FIRST.items()))
        if by_age
        else _SAVER_SITS_OUT_FIRST
    )
    values = (
        _saver(stages=stages, grid_by_period=True, by_age=by_age)
        .solve(params={}, log_level="off")
        .values
    )
    expected = _hand_saver_values(landing=1)
    wrong = _hand_saver_values(landing=0)
    for period in range(_N_ECONOMIC_PERIODS - 1):
        got = np.asarray(values[2 * period + (0 if by_age else 1)]["single"])[..., 0]
        assert_agrees_to_ulp(
            got=got,
            expected=expected[period].astype(got.dtype),
            n_ulp=_SKIP_ULP,
            operand_magnitude=1.0,
        )
        assert np.abs(got - wrong[period]).max() > 1e-2


@pytest.mark.parametrize("discount_factor", [1.0, 0.5])
def test_a_landing_in_the_same_period_is_not_discounted(
    discount_factor: float,
) -> None:
    """A work decision whose spending decision follows two stages later.

    Oracle: the model without stages, one period per decision, in which the work
    decision adds the spending decision's value undiscounted.
    """
    staged = _staged_model(
        clock="staged",
        stages={
            "work": ("working",),
            "rest": ("idle",),
            "spend": ("spending", "final"),
        },
        regimes={
            **_staged_regimes(clock="staged"),
            "idle": lcm.Regime(functions={"utility": _waiting}),
        },
        regime_id_class=IdleId,
        discount_factor=discount_factor,
    )
    staged_values = staged.solve(
        params={"wage": _wage(steps=False)}, log_level="off"
    ).values
    hand_values = (
        _staged_model(clock="hand", discount_factor=discount_factor)
        .solve(params={"wage": _wage(steps=True)}, log_level="off")
        .values
    )
    for step, regimes in hand_values.items():
        for name, value in regimes.items():
            assert_agrees_to_ulp(
                got=staged_values[3 * (step // 2) + 2 * (step % 2)][name],
                expected=value,
                n_ulp=_SKIP_ULP,
                operand_magnitude=1.0,
            )


@pytest.mark.parametrize("discount_factor", [0.5, None])
def test_a_crossing_edge_discounts_once_whatever_it_skips(
    discount_factor: float | None,
) -> None:
    """A saver in the first of three stages equals the saver without stages.

    Oracle: the model without stages, which interpolates and discounts once per
    period. Without a fixed discount factor, the factor is a parameter of the
    declared regime that discounts.
    """
    stages = {
        "spend": ("single", "single_final"),
        "wait": ("idle",),
        "rest": ("rest",),
    }
    params: UserParams = (
        {"single": {"koopmans_aggregator": {"discount_factor": 0.5}}}
        if discount_factor is None
        else {}
    )
    plain = _saver(stages=None, discount_factor=discount_factor)
    plain_values = plain.solve(params=params, log_level="off").values
    staged = _saver(stages=stages, discount_factor=discount_factor)
    values = staged.solve(params=params, log_level="off").values
    for period, regimes in plain_values.items():
        for name, value in regimes.items():
            assert_agrees_to_ulp(
                got=values[3 * period][name],
                expected=value,
                n_ulp=_SKIP_ULP,
                operand_magnitude=1.0,
            )
    assert (
        staged.get_params_template()["single"]
        == (plain.get_params_template()["single"])
    )


def test_leaving_to_a_later_stage_gets_the_value_of_staying_or_leaving() -> None:
    """A couple parts where a partner prefers the single's value of that period.

    Oracle: a hand calculation per wealth node. Leaving gives each partner the
    value of the saver without stages at half the couple's wealth, a node of the
    saver's grid; staying gives the match quality plus the couple's value in the
    spending stage. Both are undiscounted, and the couple parts where leaving is
    strictly better for a partner.
    """
    love = 0.25
    values = (
        _marriage(hand=False)
        .solve(params={"couple_sep": {"love": love}}, log_level="off")
        .values
    )
    single_values = _saver(stages=None).solve(params={}, log_level="off").values
    wealth = np.linspace(2.0, 12.0, 6)
    for period in range(_N_ECONOMIC_PERIODS - 1):
        couple = np.asarray(values[2 * period + 1]["couple"])
        outside = np.asarray(single_values[period]["single"])
        inside = (love - 0.05 * wealth[:, None]).astype(couple.dtype) + couple
        leaves = (outside > inside).any(axis=-1, keepdims=True)
        assert leaves.any()
        assert not leaves.all()
        assert_agrees_to_ulp(
            got=values[2 * period]["couple_sep"],
            expected=np.where(leaves, outside, inside),
            n_ulp=_SKIP_ULP,
            operand_magnitude=1.0,
        )


def test_an_agent_has_no_row_in_a_stage_its_regime_sits_out() -> None:
    """A saver in the second stage is simulated as the saver without stages.

    Its rows are those of its own stage, and its wealth and regime change only
    there. The model has no shocks, and the objective is strictly concave in
    the share consumed, so the paths depend on neither draws nor ties.
    """
    frames = {}
    for staged in (False, True):
        model = _saver(stages=_SAVER_SITS_OUT_FIRST if staged else None)
        frames[staged] = model.simulate(
            params={},
            initial_conditions={
                "period": jnp.zeros(4, dtype=jnp.int32),
                "regime_id": jnp.full(
                    4, model.regime_names_to_ids["single"], dtype=jnp.int32
                ),
                "wealth": jnp.array([1.3, 2.5, 4.0, 5.7]),
            },
            seed=3,
            log_level="off",
        ).to_dataframe()
    assert set(frames[True]["stage"]) == {"spend"}
    assert set(frames[False]["regime_name"]) == {"single", "single_final"}
    for column in ("agent_id", "period", "regime_name", "share"):
        np.testing.assert_array_equal(
            np.asarray(frames[True][column]), np.asarray(frames[False][column])
        )
    for column in ("wealth", "value"):
        assert_agrees_to_ulp(
            got=frames[True][column],
            expected=frames[False][column],
            n_ulp=_SKIP_ULP,
            operand_magnitude=1.0,
        )


@lcm.categorical(ordered=False)
class LotteryId:
    source: ScalarInt
    idle: ScalarInt
    high: ScalarInt
    low: ScalarInt


def _even_odds() -> FloatND:
    return jnp.asarray([0.0, 0.0, 0.5, 0.5])


@pytest.mark.parametrize("declared", ["model", "regime"])
def test_a_source_applies_its_certainty_equivalent_once_to_a_later_stage(
    declared: str,
) -> None:
    """A lottery over two regimes two stages later, valued by a power mean.

    Oracle: a closed form. The source gets 1 and then 3 or 1 with even odds; at
    a risk aversion of 2 the certainty equivalent is `1 / (0.5 / 3 + 0.5)`, and
    the landing in the same period adds it undiscounted. The source is the only
    regime with a continuation, so declaring the power mean on it or on the
    model is the same model.
    """
    always = lcm.PeriodRange()
    model = lcm.Model(
        n_periods=1,
        stages={"draw": ("source",), "wait": ("idle",), "land": ("high", "low")},
        regimes={
            "source": lcm.Regime(
                functions={"utility": _one},
                certainty_equivalent=lcm.PowerMean() if declared == "regime" else None,
            ),
            "idle": lcm.Regime(functions={"utility": _waiting}),
            "high": lcm.Regime(functions={"utility": _three}),
            "low": lcm.Regime(functions={"utility": _one}),
        },
        regime_id_class=LotteryId,
        initial_nodes=lcm.InitialNodes(by_period={0: ("source", "idle")}),
        edges={
            "source": lcm.Transition(
                targets={"high": always, "low": always},
                law=lcm.StochasticTransition(func=_even_odds),
            )
        },
        certainty_equivalent=lcm.PowerMean()
        if declared == "model"
        else lcm.LinearExpectation(),
    )
    values = model.solve(params={"risk_aversion": 2.0}, log_level="off").values
    got = np.asarray(values[0]["source"])
    assert_agrees_to_ulp(
        got=got, expected=np.full_like(got, 1.0 + 1.0 / (0.5 / 3.0 + 0.5)), n_ulp=4
    )


@lcm.categorical(ordered=False)
class FormationId:
    single: ScalarInt
    partner: ScalarInt
    couple: ScalarInt


def _forming_utility(*, labor: DiscreteAction) -> FloatND:
    return 2.0 * labor


def _three() -> FloatND:
    return jnp.asarray(3.0)


def _one() -> FloatND:
    return jnp.asarray(1.0)


def _formation() -> lcm.Model:
    """A single forms a couple of the later stage if a partner there consents."""
    return lcm.Model(
        n_periods=1,
        stages={"meet": ("single",), "live": ("partner", "couple")},
        regimes={
            "single": lcm.Regime(
                actions={"labor": lcm.DiscreteGrid(Labor)},
                functions={"utility": _forming_utility},
                composition_action=lcm.CompositionAction(
                    destinations={
                        "self": lcm.Destination(
                            regime="couple", role="f", projection={}
                        )
                    },
                    rule=lcm.Consent(
                        counterparty=lcm.Counterparty(
                            regime="partner", role="m", projection={}
                        )
                    ),
                ),
            ),
            "partner": lcm.Regime(functions={"utility": _waiting}),
            "couple": lcm.Regime(
                functions={
                    "utility": lcm.CollectiveUtility(utilities={"f": _three, "m": _one})
                }
            ),
        },
        regime_id_class=FormationId,
        initial_nodes=lcm.InitialNodes(by_period={0: "single"}),
        edges={"single": {"partner": lcm.PeriodRange()}},
    )


def _skipping_case(
    variant: str,
) -> tuple[lcm.Model, UserParams, pd.DataFrame, tuple[RegimeName, ...]]:
    """A model with a regime that sits out a stage, its params, starts and regimes."""
    if variant == "consent":
        starts = pd.DataFrame({"period": [0, 0], "regime_name": ["single", "single"]})
        return _formation(), {}, starts, tuple(FormationId.__dataclass_fields__)
    params: UserParams = {"couple_sep": {"love": 0.2}}
    if variant == "cells":
        params = {
            **params,
            "single": {"single": {"next_wealth": {"income": 0.5}}},
            "edges": {"single": {"single": {"stay": 1.0}}},
        }
    starts = pd.DataFrame(
        {
            "age" if variant == "age" else "period": [60 if variant == "age" else 0]
            * 4,
            "regime_name": ["couple_sep", "couple_sep", "single", "single"],
            "own_member": ["f", "m", "f", "m"],
            "wealth": [4.0, 10.0, 2.0, 5.0],
        }
    )
    model = _marriage(hand=False, variant=variant)
    return model, params, starts, tuple(MarriageId.__dataclass_fields__)


@pytest.mark.parametrize(
    "variant", ["plain", "age", "phased", "recoded", "spread", "cells", "consent"]
)
def test_a_model_whose_edges_skip_a_stage_names_only_declared_regimes(
    *, variant: str, tmp_path: Path
) -> None:
    """A regime that sits out a stage is known only by the name the user declared.

    The regime codes, the values of a solution and of its saved copy, and the
    raw and tabulated simulation and its saved copy name only declared regimes.
    """
    model, params, starts, declared = _skipping_case(variant)
    solution = model.solve(params=params, log_level="off")
    loaded = lcm.load_solution(path=solution.save(path=tmp_path / "solution.lcm"))
    result = model.simulate(
        params=params,
        initial_conditions=starts,
        solution=loaded,
        seed=5,
        log_level="off",
    )
    result.save(directory=tmp_path / "result")
    restored = lcm.SimulationResult.load(directory=tmp_path / "result")
    frame = restored.to_dataframe()
    pd.testing.assert_frame_equal(frame, result.to_dataframe())
    assert tuple(model.regime_names_to_ids) == declared
    assert tuple(frame["regime_name"].cat.categories) == declared
    assert restored.regime_names == declared
    for values in (solution.values, loaded.values):
        assert {name for regimes in values.values() for name in regimes} <= set(
            declared
        )
    assert set(result.raw_results) <= set(declared)
    assert set(restored.raw_results) <= set(declared)


# The periods the readers below were called with, filled from inside the program.
_PERIODS_READ: set[int] = set()


def _record_periods(read: jax.Array) -> None:
    _PERIODS_READ.update(np.asarray(read).ravel().tolist())


def _read(period: Period) -> None:
    jax.debug.callback(_record_periods, period)


def _affordable_reading_period(
    *, consumption: ContinuousAction, wealth: ContinuousState, period: Period
) -> BoolND:
    _read(period)
    return consumption <= wealth


def _aggregator_reading_period(
    *, utility: FloatND, CE: FloatND, discount_factor: FloatND, period: Period
) -> FloatND:
    _read(period)
    return utility + discount_factor * CE


def _certain_reading_period(*, period: Period) -> FloatND:
    _read(period)
    return jnp.asarray(1.0)


def _open_reading_period(*, period: Period) -> BoolND:
    _read(period)
    return period >= 0


def _available_reading_period(*, wealth: ContinuousState, period: Period) -> BoolND:
    _read(period)
    return wealth > 0.0


def _wealth_kept_reading_period(
    *, wealth: ContinuousState, period: Period
) -> ContinuousState:
    _read(period)
    return wealth


def _no_adjustment_reading_period(
    *, wealth: ContinuousState, period: Period
) -> FloatND:
    _read(period)
    return 0.0 * wealth


def _spend_utility_with_shock(
    *, consumption: ContinuousAction, eps: FloatND
) -> FloatND:
    return jnp.log(consumption) + 0.0 * eps


def _half_reading_period(*, period: Period) -> FloatND:
    _read(period)
    return jnp.asarray(0.5)


def _bits_utility(
    *, a: DiscreteState, b: DiscreteState, labor: DiscreteAction
) -> FloatND:
    return 0.0 + a + b + labor


def _bits_final_utility(*, a: DiscreteState, b: DiscreteState) -> FloatND:
    return 0.0 + a + b


def _labor_kept(*, labor: DiscreteAction) -> DiscreteState:
    return labor


def _bits_regimes() -> dict[RegimeName, lcm.Regime]:
    """Two binary members; `b` keeps the last labor choice, `a` never changes."""
    bits = {"a": lcm.DiscreteGrid(Labor), "b": lcm.DiscreteGrid(Labor)}
    deciding = lcm.Regime(
        states=bits,
        actions={"labor": lcm.DiscreteGrid(Labor)},
        state_transitions={"a": lcm.fixed_transition("a"), "b": _labor_kept},
        functions={"utility": _bits_utility},
    )
    return {
        "working": deciding,
        "spending": deciding,
        "final": lcm.Regime(states=bits, functions={"utility": _bits_final_utility}),
    }


@lcm.categorical(ordered=False)
class IdleId:
    working: ScalarInt
    idle: ScalarInt
    spending: ScalarInt
    final: ScalarInt


@lcm.categorical(ordered=False)
class SettleId:
    couple: ScalarInt
    single_a: ScalarInt
    single_b: ScalarInt
    later: ScalarInt
    end: ScalarInt


def _couple_utility() -> FloatND:
    return jnp.asarray(0.0)


def _paid(*, share: FloatND) -> FloatND:
    return 4.0 * share


def _settling_couple() -> lcm.Model:
    """A couple that may separate at a transfer in the first stage of a period."""
    household = lcm.CollectiveUtility(
        utilities={"a": _couple_utility, "b": _couple_utility}
    )
    return lcm.Model(
        n_periods=2,
        stages={
            "together": ("couple", "single_a", "single_b"),
            "after": ("later", "end"),
        },
        regimes={
            "couple": lcm.Regime(
                functions={"utility": household},
                composition_action=lcm.CompositionAction(
                    rule=lcm.MutualConsent(
                        transfer=lcm.Transfer(
                            name="share",
                            candidates=lcm.LinSpacedGrid(
                                start=0.0, stop=1.0, n_points=5
                            ),
                            default=_half_reading_period,
                        )
                    ),
                    destinations={
                        member: lcm.Destination(
                            regime=f"single_{member}",
                            projection={},
                            utility_adjustment=_paid,
                        )
                        for member in ("a", "b")
                    },
                ),
            ),
            "single_a": lcm.Regime(functions={"utility": _couple_utility}),
            "single_b": lcm.Regime(functions={"utility": _couple_utility}),
            "later": lcm.Regime(functions={"utility": household}),
            "end": lcm.Regime(functions={"utility": household}),
        },
        regime_id_class=SettleId,
        initial_nodes=lcm.InitialNodes(
            by_period={0: ("couple", "single_a", "single_b")}
        ),
        edges={
            "couple": {
                "later": lcm.PeriodRange(exclusive_stop=1),
                "end": lcm.PeriodRange(start=1),
            },
            "later": {"couple": lcm.PeriodRange(exclusive_stop=1)},
        },
        fixed_params={"discount_factor": 0.9},
    )


@lcm.categorical(ordered=False)
class LandingId:
    couple: ScalarInt
    idle: ScalarInt
    single_a: ScalarInt
    single_b: ScalarInt


@lcm.categorical(ordered=True)
class TwoWeights:
    low: ScalarInt
    high: ScalarInt


def _one() -> FloatND:
    return jnp.asarray(1.0)


def _seven() -> FloatND:
    return jnp.asarray(7.0)


def _seven_at_every_weight(*, mu: DiscreteState) -> FloatND:
    return jnp.where(mu >= 0, 7.0, 0.0)


def _worthless() -> FloatND:
    return jnp.asarray(-jnp.inf)


def _never() -> BoolND:
    return jnp.asarray(a=False)


def _staying_allowed(*, wealth: ContinuousState) -> BoolND:
    return (wealth >= 2.0) & (wealth <= 4.0)


def _inside_a(*, wealth: ContinuousState) -> FloatND:
    return jnp.where(wealth <= 2.0, -jnp.inf, 0.0)


def _landed_wealth(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _enough_wealth(*, wealth: ContinuousState) -> BoolND:
    return wealth >= 3.0


def _half_more_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth + 0.5


def _landing_settlement() -> lcm.MutualConsent:
    return lcm.MutualConsent(
        transfer=lcm.Transfer(
            name="share",
            candidates=lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
            default=_half,
        )
    )


def _half() -> FloatND:
    return jnp.asarray(0.5)


type LandingRule = lcm.PCOnActions | lcm.PCOnWeight | lcm.MutualConsent

_LANDING_TOPOLOGIES: dict[str, dict[str, tuple[RegimeName, ...]] | None] = {
    "adjacent": {"decide": ("couple", "idle"), "land": ("single_a", "single_b")},
    "skipped": {
        "decide": ("couple",),
        "wait": ("idle",),
        "land": ("single_a", "single_b"),
    },
    "split": {"decide": ("couple", "idle", "single_a"), "land": ("single_b",)},
    "unstaged": None,
}


def _landing_couple(
    *,
    stay: Mapping[str, Callable[..., BoolND]],
    inside_a: Callable[..., FloatND],
    destination_a: lcm.Regime,
    rule: LandingRule,
    topology: str = "adjacent",
    wealth: Mapping[StateName, Callable[..., ContinuousState]] | None = None,
    permuted: bool = False,
    enable_jit: bool = True,
    shocks: Mapping[str, ShockGroup] | None = None,
) -> lcm.Model:
    """A couple deciding on its composition, its destinations a stage later.

    Member b gets 7 if the couple stays and 1 in `single_b`; member a gets
    `inside_a` and its value in `destination_a`. `stay` are the couple's
    constraints. With `wealth`, the couple has wealth on the nodes 1 to 6, and
    `wealth` projects it into `single_a`. `topology` names the stages, and
    `permuted` declares regimes, destinations and members in reverse order.
    """
    states: dict[StateName, lcm.LinSpacedGrid | lcm.DiscreteGrid] = (
        {} if wealth is None else {"wealth": _couple_wealth_grid()}
    )
    if isinstance(rule, lcm.PCOnWeight) and rule.state is not None:
        states["mu"] = lcm.DiscreteGrid(TwoWeights)
    adjustment = _paid if isinstance(rule, lcm.MutualConsent) else None
    destinations = {
        "a": lcm.Destination(
            regime="single_a",
            projection=wealth or {},
            utility_adjustment=adjustment,
        ),
        "b": lcm.Destination(
            regime="single_b", projection={}, utility_adjustment=adjustment
        ),
    }
    utilities = {
        "a": inside_a,
        "b": _seven_at_every_weight if "mu" in states else _seven,
    }
    regimes = {
        "couple": lcm.Regime(
            states=states,
            functions={
                "utility": lcm.CollectiveUtility(
                    utilities=dict(reversed(utilities.items()))
                    if permuted
                    else utilities
                )
            },
            constraints=stay,
            composition_action=lcm.CompositionAction(
                destinations=dict(reversed(destinations.items()))
                if permuted
                else destinations,
                rule=rule,
            ),
        ),
        "idle": lcm.Regime(functions={"utility": _waiting}),
        "single_a": destination_a,
        "single_b": lcm.Regime(functions={"utility": _one}),
    }
    return lcm.Model(
        n_periods=1,
        stages=_LANDING_TOPOLOGIES[topology],
        regimes=dict(reversed(regimes.items())) if permuted else regimes,
        regime_id_class=LandingId,
        initial_nodes=lcm.InitialNodes(
            by_period={0: tuple(LandingId.__dataclass_fields__)}
        ),
        edges={},
        enable_jit=enable_jit,
        shocks={} if shocks is None else shocks,
    )


def _couple_wealth_grid() -> lcm.LinSpacedGrid:
    return lcm.LinSpacedGrid(start=1.0, stop=6.0, n_points=6)


def _couple_value(model: lcm.Model) -> np.ndarray:
    return np.asarray(model.solve(params={}, log_level="off").values[0]["couple"])


def _rule_id(rule: LandingRule) -> str:
    """A rule's test id: its kind, its weight state if any, and its tie."""
    state = getattr(rule, "state", None)
    return "-".join((type(rule).__name__, *((state,) if state else ()), rule.tie))


def _utility_with_landing_shock(*, eps: FloatND) -> FloatND:
    return 4.0 + eps


def _utility_of_landing_action(*, offer: ContinuousAction) -> FloatND:
    return 4.0 - offer


def _offer_covers_landing_shock(*, offer: ContinuousAction, eps: FloatND) -> BoolND:
    return offer >= eps


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("shock_in_constraint", [False, True])
def test_later_stage_destination_integrates_its_own_shock(
    *, enable_jit: bool, shock_in_constraint: bool
) -> None:
    """The couple compares against a value averaged over the destination's draw.

    With equally likely shocks of -1 and 1, the destination's utilities are 3
    and 5 if utility reads the shock. If its constraint reads the shock instead,
    the lowest feasible offers are -0.5 and 1.5 and yield 4.5 and 2.5. Member a
    prefers either expected outside value to 1 from staying and leaves.
    """
    if shock_in_constraint:
        destination = lcm.Regime(
            actions={"offer": lcm.LinSpacedGrid(start=-0.5, stop=1.5, n_points=2)},
            functions={"utility": _utility_of_landing_action},
            constraints={"affordable": _offer_covers_landing_shock},
            shocks=("offer_shock",),
        )
    else:
        destination = lcm.Regime(
            functions={"utility": _utility_with_landing_shock},
            shocks=("offer_shock",),
        )
    model = _landing_couple(
        stay={},
        inside_a=_one,
        destination_a=destination,
        rule=lcm.PCOnActions(),
        enable_jit=enable_jit,
        shocks={
            "offer_shock": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=Product(eps=GaussHermite(2)),
            )
        },
    )
    expected = (3.5 if shock_in_constraint else 4.0, 1.0)
    assert_agrees_to_ulp(
        got=_couple_value(model),
        expected=np.asarray(expected),
        n_ulp=_SKIP_ULP,
        operand_magnitude=5.0,
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_later_stage_destination_requires_feasibility_at_each_possible_shock(
    *, enable_jit: bool
) -> None:
    """An outside option with no feasible action at a possible draw is unavailable.

    Member a gets minus infinity by staying, and would leave on a tie if the
    destination were supported. Its highest offer is 0.5 and cannot cover a
    draw of 1, so the couple stays.
    """
    model = _landing_couple(
        stay={},
        inside_a=_worthless,
        destination_a=lcm.Regime(
            actions={"offer": lcm.LinSpacedGrid(start=-0.5, stop=0.5, n_points=2)},
            functions={"utility": _utility_of_landing_action},
            constraints={"affordable": _offer_covers_landing_shock},
            shocks=("offer_shock",),
        ),
        rule=lcm.PCOnActions(tie="leave"),
        enable_jit=enable_jit,
        shocks={
            "offer_shock": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=Product(eps=GaussHermite(2)),
            )
        },
    )
    rows = model.simulate(
        params={},
        initial_conditions=pd.DataFrame(
            {
                "period": [0, 0],
                "regime_name": ["couple", "couple"],
                "own_member": ["a", "b"],
            }
        ),
        seed=5,
        log_level="off",
    ).to_dataframe()
    assert rows["regime_name"].astype(str).tolist() == ["couple", "couple"]


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize(
    "rule",
    [
        lcm.PCOnActions(tie="leave"),
        lcm.PCOnActions(),
        lcm.PCOnWeight(tie="leave"),
        lcm.PCOnWeight(),
        lcm.PCOnWeight(state="mu", tie="leave"),
        lcm.PCOnWeight(state="mu"),
        _landing_settlement(),
    ],
    ids=_rule_id,
)
@pytest.mark.parametrize(
    ("stay_feasible", "inside_a", "a_feasible", "a_flow", "expected"),
    [
        # No feasible stay, and no feasible action for member a in single_a: the
        # couple has no feasible outcome.
        (False, _one, False, _one, (-np.inf, -np.inf)),
        # Member a, worth -inf either way, accepts staying because single_a has
        # no feasible action; member b prefers 7 to 1.
        (True, _worthless, False, _one, (-np.inf, 7.0)),
        # Control: a feasible action worth -inf in single_a is an outside option.
        (True, _worthless, True, _worthless, None),
    ],
    ids=["no_outcome", "infeasible_destination", "feasible_minus_infinity"],
)
def test_a_later_stage_destination_without_feasible_action_is_unsupported(
    *,
    stay_feasible: bool,
    inside_a: Callable[[], FloatND],
    a_feasible: bool,
    a_flow: Callable[[], FloatND],
    expected: tuple[float, float] | None,
    rule: LandingRule,
    enable_jit: bool,
) -> None:
    """A destination's support is its feasibility, whatever stage it decides in.

    Oracles: the hand values, and the same couple with every regime in one stage,
    whose destinations are read where they decide. A Pareto weight in a state
    changes no value here, since every weight values the members alike. With a
    feasible action worth `-inf` in `single_a`, member a leaves under
    `tie="leave"` and stays under `tie="stay"`; under mutual consent member b
    keeps the couple together.
    """
    model = functools.partial(
        _landing_couple,
        stay={} if stay_feasible else {"feasible": _never},
        inside_a=inside_a,
        destination_a=lcm.Regime(
            functions={"utility": a_flow},
            constraints={} if a_feasible else {"feasible": _never},
        ),
        rule=rule,
    )
    got = _couple_value(model(enable_jit=enable_jit))
    same_stage = _couple_value(model(topology="unstaged"))
    if expected is None:
        expected = (
            (-np.inf, 1.0)
            if rule.tie == "leave" and not isinstance(rule, lcm.MutualConsent)
            else (-np.inf, 7.0)
        )
    np.testing.assert_array_equal(
        same_stage, np.broadcast_to(np.asarray(expected, dtype=got.dtype), got.shape)
    )
    np.testing.assert_array_equal(got, same_stage)


def _hand_landing_values(*, offset: float) -> np.ndarray:
    """The couple's values at wealth 1 to 6 under `tie="leave"`, by hand.

    Member a lands at wealth plus `offset`, where `single_a` is feasible from 3
    on and worth the wealth. The couple may stay at wealth 2 to 4, where member
    a gets `-inf` up to wealth 2 and 0 above; member b always prefers to stay.
    """
    rows = []
    for wealth in range(1, 7):
        landing = wealth + offset
        supported = landing >= 3.0
        outside = landing if supported else -np.inf
        inside = -np.inf if wealth <= 2 else 0.0
        willing = not supported or inside > outside
        if 2 <= wealth <= 4 and willing:
            rows.append((inside, 7.0))
        elif supported:
            rows.append((outside, 1.0))
        else:
            rows.append((-np.inf, -np.inf))
    return np.asarray(rows)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize(
    "rule", [lcm.PCOnActions(tie="leave"), lcm.PCOnWeight(tie="leave")], ids=_rule_id
)
@pytest.mark.parametrize("permuted", [False, True], ids=["declared", "permuted"])
@pytest.mark.parametrize(
    ("offset", "projection"),
    [(0.0, _landed_wealth), (0.5, _half_more_wealth)],
    ids=["on_grid", "off_grid"],
)
@pytest.mark.parametrize("topology", ["adjacent", "skipped", "split"])
def test_a_destination_is_supported_where_it_has_a_feasible_action_at_the_landing(
    *,
    topology: str,
    offset: float,
    projection: Callable[..., ContinuousState],
    permuted: bool,
    rule: LandingRule,
    enable_jit: bool,
) -> None:
    """Support follows the destination's own constraint, cell by cell.

    `single_a` has wealth on the nodes 0 to 10 and is feasible from wealth 3 on,
    where it is worth its wealth, so a landing between two feasible nodes reads
    its value exactly. Oracles: the hand values, and the same couple with every
    regime in one stage. Declaring regimes, destinations and members in reverse
    order changes no value of a member.
    """
    model = functools.partial(
        _landing_couple,
        stay={"allowed": _staying_allowed},
        inside_a=_inside_a,
        destination_a=lcm.Regime(
            states={"wealth": lcm.LinSpacedGrid(start=0.0, stop=10.0, n_points=11)},
            functions={"utility": _landed_wealth},
            constraints={"enough": _enough_wealth},
        ),
        rule=rule,
        wealth={"wealth": projection},
        permuted=permuted,
    )
    got = _couple_value(model(topology=topology, enable_jit=enable_jit))
    same_stage = _couple_value(model(topology="unstaged"))
    expected = _hand_landing_values(offset=offset).astype(got.dtype)
    if permuted:
        expected = expected[:, ::-1]
    for value, reference in ((same_stage, expected), (got, same_stage)):
        assert_agrees_to_ulp(
            got=value, expected=reference, n_ulp=_SKIP_ULP, operand_magnitude=10.0
        )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_a_couple_stays_where_a_later_destination_has_no_feasible_action(
    *, enable_jit: bool
) -> None:
    """The simulated couple stays where member a's destination admits no action.

    At wealth 2, member a, worth `-inf` inside, accepts staying because
    `single_a` has no feasible action at the landing, and member b prefers to
    stay; at wealth 3, member a prefers `single_a`, and the members separate.
    """
    model = _landing_couple(
        stay={"allowed": _staying_allowed},
        inside_a=_inside_a,
        destination_a=lcm.Regime(
            states={"wealth": lcm.LinSpacedGrid(start=0.0, stop=10.0, n_points=11)},
            functions={"utility": _landed_wealth},
            constraints={"enough": _enough_wealth},
        ),
        rule=lcm.PCOnActions(tie="leave"),
        wealth={"wealth": _landed_wealth},
        enable_jit=enable_jit,
    )
    frame = model.simulate(
        params={},
        initial_conditions=pd.DataFrame(
            {
                "period": [0] * 4,
                "regime_name": ["couple"] * 4,
                "own_member": ["a", "b"] * 2,
                "wealth": [2.0, 2.0, 3.0, 3.0],
            }
        ),
        seed=5,
        log_level="off",
    ).to_dataframe()
    rows = frame.reset_index().sort_values("agent_id")
    assert rows["agent_id"].tolist() == [0, 1, 2, 3]
    assert rows["regime_name"].astype(str).tolist() == [
        "couple",
        "couple",
        "single_a",
        "single_b",
    ]


@lcm.categorical(ordered=False)
class SpendingLandingId:
    couple: ScalarInt
    single: ScalarInt
    final: ScalarInt


def test_a_later_stage_destination_solved_on_an_endogenous_grid_is_refused() -> None:
    """Its feasibility at a landing state is not evaluated, as in one stage.

    The refusal is the one a read of the destination's problem in its own stage
    meets, raised when the model is built.
    """
    wealth = lcm.LinSpacedGrid(start=1.0, stop=6.0, n_points=6)
    with pytest.raises(
        lcm.exceptions.UnsupportedOperationError,
        match="requires its declared GridSearch solver",
    ):
        lcm.Model(
            n_periods=2,
            stages={"decide": ("couple",), "spend": ("single", "final")},
            regimes={
                "couple": lcm.Regime(
                    states={"wealth": wealth},
                    functions={
                        "utility": lcm.CollectiveUtility(
                            utilities={"a": _landed_wealth, "b": _landed_wealth}
                        )
                    },
                    composition_action=lcm.CompositionAction(
                        destinations={
                            member: lcm.Destination(
                                regime="single", projection={"wealth": _wealth_kept}
                            )
                            for member in ("a", "b")
                        },
                        rule=lcm.PCOnActions(),
                    ),
                ),
                "single": ConsumptionSavingsRegime(
                    states={"wealth": wealth},
                    actions={
                        "consumption": lcm.LinSpacedGrid(
                            start=0.5, stop=5.0, n_points=10
                        )
                    },
                    functions={
                        "utility": _spend_utility,
                        "resources": _wealth_resources,
                        "savings": _wealth_savings,
                    },
                    state_transitions={"wealth": _wealth_after_saving},
                    solver=EGM(
                        savings_grid=lcm.LinSpacedGrid(start=0.0, stop=6.0, n_points=7)
                    ),
                    liquid=LiquidMargin(
                        state="wealth",
                        action="consumption",
                        resources="resources",
                        post_decision_state="savings",
                    ),
                ),
                "final": lcm.Regime(
                    states={"wealth": wealth}, functions={"utility": _landed_wealth}
                ),
            },
            regime_id_class=SpendingLandingId,
            initial_nodes=lcm.InitialNodes(by_period={0: ("couple", "single")}),
            edges={"single": {"final": lcm.PeriodRange()}},
            fixed_params={"discount_factor": 0.9},
        )


def _period_reader(kind: str) -> tuple[lcm.Model, UserParams, RegimeName, set[int]]:
    """A staged model in which `kind` reads `period`, and the periods it must read.

    A declaration that cannot report what it reads expects no period; a wrong
    period then makes the model infeasible or every value `-inf`.
    """
    regimes = _staged_regimes(clock="staged")
    spending = regimes["spending"]
    params: UserParams = {"wage": _wage(steps=False)}
    last = _N_ECONOMIC_PERIODS - 1
    changes: dict[str, RegimeReplacement] = {
        "constraint": {"constraints": {"affordable": _affordable_reading_period}},
        "condition": {
            "constraints": {
                "affordable": lcm.Condition.from_callable(_affordable_reading_period)
            }
        },
        "condition_reference": {
            "constraints": {
                "affordable": _affordable,
                "before_the_last": lcm.ref("period") <= float(last - 1),
            }
        },
        "aggregator": {"koopmans_aggregator": _aggregator_reading_period},
        "composition_action": {
            "composition_action": lcm.CompositionAction(
                destinations={
                    "self": lcm.Destination(
                        regime="final",
                        projection={"wealth": _wealth_kept_reading_period},
                        utility_adjustment=_no_adjustment_reading_period,
                    )
                },
                rule=lcm.PCOnActions(),
                available=_available_reading_period,
            )
        },
        "shock_law": {
            "functions": {"utility": _spend_utility_with_shock},
            "shocks": ("income",),
        },
    }
    if kind in changes:
        regimes["spending"] = spending.replace(**changes[kind])
        model = _staged_model(
            clock="staged",
            regimes=regimes,
            shocks={
                "income": ShockGroup(
                    shocks={"eps": Normal(mu=_half_reading_period, sigma=1.0)},
                    nodes=Product(eps=GaussHermite(3)),
                )
            }
            if kind == "shock_law"
            else None,
        )
        expected = set() if kind == "condition_reference" else {0, 1}
        return model, params, "working", expected
    moving_on = {
        "spending": lcm.PeriodRange(exclusive_stop=last),
        "final": lcm.PeriodRange(start=last),
    }
    if kind == "regime_law":
        edges = {
            "working": moving_on,
            "spending": lcm.Transition(
                law={"working": lcm.StochasticTransition(func=_certain_reading_period)}
            ),
        }
        return _staged_model(clock="staged", edges=edges), params, "working", {0, 1}
    if kind == "gate":
        # A gate reads the period its row lands in.
        regimes["idle"] = regimes["working"]
        model = _staged_model(
            clock="staged",
            stages={"work": ("working", "idle"), "spend": ("spending", "final")},
            regimes=regimes,
            regime_id_class=IdleId,
            edges={
                "working": moving_on,
                "idle": moving_on,
                "spending": lcm.Transition(
                    law={"working": lcm.StochasticTransition(func=_certain)},
                    gates={
                        "working": lcm.Gate(
                            predicate=_open_reading_period,
                            routes={
                                "only": lcm.MemberRoute(
                                    fallback=lcm.ProjectedRegimeValue(
                                        regime="idle",
                                        projection={"wealth": _wealth_kept},
                                    )
                                )
                            },
                        )
                    },
                ),
            },
        )
        return model, params, "working", {1, 2}
    if kind == "pareto_weight":
        return _game(mover_weight=_certain_reading_period), {}, "a_moves", {0, 1, 2}
    if kind == "transfer":
        return _settling_couple(), {}, "couple", {0, 1}
    # A composite state's generated constraint reads the period.
    model = _staged_model(
        clock="staged",
        regimes=_bits_regimes(),
        composite_states={
            "bits": lcm.CompositeState(
                members=("a", "b"), start_codes={"a": (0,), "b": (0,)}
            )
        },
    )
    return model, {}, "working", set()


@pytest.mark.parametrize(
    "kind",
    [
        "constraint",
        "condition",
        "condition_reference",
        "aggregator",
        "regime_law",
        "gate",
        "composition_action",
        "shock_law",
        "pareto_weight",
        "transfer",
        "composite_state",
    ],
)
def test_every_reader_of_period_reads_the_period_of_its_stage(kind: str) -> None:
    """Whatever a user declares that reads `period` reads the period, not the step."""
    model, params, start, expected = _period_reader(kind)
    _PERIODS_READ.clear()
    values = model.solve(params=params, log_level="off").values
    assert expected == _PERIODS_READ
    assert np.isfinite(values[0][start]).any()


def test_regime_finder_refuses_a_staged_model() -> None:
    with pytest.raises(ValueError, match="`stages`"):
        read_model(_staged_model(clock="staged"))


def _wealth_lost_in_the_second_period(
    *, wealth: ContinuousState, period: Period
) -> ContinuousState:
    return jnp.where(period == 1, 0.25, wealth)


def test_a_staged_simulation_reports_the_period_of_an_infeasible_state() -> None:
    """A state without a feasible action is reported in its period, not its step."""
    regimes = _staged_regimes(clock="staged")
    regimes["working"] = lcm.Regime(
        states={"wealth": lcm.LinSpacedGrid(start=1.0, stop=6.0, n_points=6)},
        actions={"labor": lcm.DiscreteGrid(Labor)},
        state_transitions={"wealth": _wealth_lost_in_the_second_period},
        functions={"utility": _work_utility},
    )
    model = _staged_model(clock="staged", regimes=regimes)
    with pytest.raises(
        lcm.exceptions.NoFeasibleActionError, match="Regime 'spending', period 1, "
    ):
        model.simulate(
            params={},
            initial_conditions={
                "period": jnp.zeros(2, dtype=jnp.int32),
                "regime_id": jnp.full(2, StagedId.working, dtype=jnp.int32),
                "wealth": jnp.array([2.0, 4.0]),
            },
            seed=7,
            log_level="off",
        )


def test_a_staged_model_counts_periods_and_its_solution_counts_steps(
    tmp_path: Path,
) -> None:
    """`n_periods` counts periods everywhere; the solution and `n_steps` count steps.

    `lookup_policy` and `InitialNodes(by_period=...)` take the economic period;
    `Model.reachability` is indexed by step.
    """
    model = _staged_model(clock="staged")
    params = {"wage": _wage(steps=False)}
    solution = model.solve(params=params, log_level="off")
    assert (model.n_periods, model.n_steps) == (_N_ECONOMIC_PERIODS, 6)
    assert model.step_of(period=1, regime_name="spending") == 3
    assert (model.period_of_step(step=3), model.stage_of_step(step=3)) == (1, "spend")
    assert model.reachability.solution.targets(step=4, source="working") == ("final",)
    with pytest.raises(ModelInitializationError, match="Period 3 in selector"):
        _staged_model(clock="staged", start_period=_N_ECONOMIC_PERIODS)
    result = model.simulate(
        params=params,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.full(2, StagedId.working, dtype=jnp.int32),
            "wealth": jnp.array([2.0, 4.0]),
        },
        solution=solution,
        seed=7,
        log_level="off",
    )
    assert result.n_periods == _N_ECONOMIC_PERIODS
    loaded = lcm.load_solution(path=solution.save(path=tmp_path / "solution.lcm"))
    assert loaded.metadata.n_steps == 6
    np.testing.assert_array_equal(
        loaded.value(step=3, regime="spending"),
        solution.value(step=3, regime="spending"),
    )
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=1,
        regime_name="spending",
        states={"wealth": jnp.linspace(1.0, 6.0, 6)},
    )
    # The lookup evaluates the six grid points at its own vmap width, so XLA may
    # vectorize differently: a few ULP, where a wrong step would differ outright.
    assert_agrees_to_ulp(
        got=lookup.value,
        expected=solution.value(step=3, regime="spending"),
        n_ulp=4,
    )


def test_a_staged_off_grid_check_reports_the_period_and_stage() -> None:
    """A landing check names the period and the stage, not the step."""
    regimes = _staged_regimes(clock="staged")
    regimes["working"] = regimes["working"].replace(
        state_transitions={"wealth": _wealth_lost_in_the_second_period},
        functions={"utility": _work_utility},
    )
    regimes["spending"] = regimes["spending"].replace(
        states={
            "wealth": lcm.LinSpacedGrid(
                start=1.0, stop=6.0, n_points=6, off_grid="raise"
            )
        }
    )
    model = _staged_model(clock="staged", regimes=regimes)
    with pytest.raises(
        lcm.exceptions.OffGridStateError, match=r"period=1, stage='work', "
    ):
        model.solve(params={}, log_level="off")


def _a_odds_broken_in_period_one(*, period: Period) -> FloatND:
    return jnp.where(period == 1, jnp.array([0.4, 0.4]), jnp.array([0.4, 0.6]))


def _a_off_its_grid_in_period_one(*, a: DiscreteState, period: Period) -> DiscreteState:
    return jnp.where(period == 1, 2, a)


def _a_from_draw(draw: dict[str, ScalarInt]) -> ScalarInt:
    return draw["a"]


def _back_to_work_broken_in_period_one(*, period: Period) -> FloatND:
    return jnp.where(period == 1, 0.5, 1.0)


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("state_law", lcm.exceptions.InvalidStateTransitionProbabilitiesError),
        ("discrete_law", lcm.exceptions.InvalidStateTransitionError),
        ("regime_law", lcm.exceptions.InvalidRegimeTransitionProbabilitiesError),
        ("joint", lcm.exceptions.InvalidStateTransitionProbabilitiesError),
    ],
)
def test_a_staged_transition_check_reads_the_period_of_its_step(
    *, kind: str, error: type[Exception]
) -> None:
    """`validate_transitions` evaluates each law at the period of its step.

    The law is broken only in period 1, whose steps are 2 and 3. A step read as
    the period would break it at step 1, in period 0, or never.
    """
    regimes = _bits_regimes()
    edges = None
    if kind == "state_law":
        regimes["working"] = regimes["working"].replace(
            state_transitions={
                "a": lcm.StochasticTransition(func=_a_odds_broken_in_period_one),
                "b": _labor_kept,
            }
        )
    elif kind == "discrete_law":
        regimes["spending"] = regimes["spending"].replace(
            state_transitions={"a": _a_off_its_grid_in_period_one, "b": _labor_kept}
        )
    elif kind == "regime_law":
        edges = {
            "working": {
                "spending": lcm.PeriodRange(exclusive_stop=_N_ECONOMIC_PERIODS - 1),
                "final": lcm.PeriodRange(start=_N_ECONOMIC_PERIODS - 1),
            },
            "spending": lcm.Transition(
                law={
                    "working": lcm.StochasticTransition(
                        func=_back_to_work_broken_in_period_one
                    )
                }
            ),
        }
    else:
        draw = lcm.JointTransition(
            support_size=2,
            support={"a": jnp.asarray([0, 1], dtype=jnp.int32)},
            probabilities=_a_odds_broken_in_period_one,
            outputs={"a": _a_from_draw},
        )
        regimes["working"] = regimes["working"].replace(
            state_transitions={"b": _labor_kept},
            joint_transitions={"spending": {"draw": draw}, "final": {"draw": draw}},
        )
    model = _staged_model(clock="staged", regimes=regimes, edges=edges)
    with pytest.raises(error, match=r"\bages? 1\b"):
        model.validate_transitions(params={})


@lcm.categorical(ordered=False)
class SkipId:
    choose: ScalarInt
    end: ScalarInt
    wait_a: ScalarInt
    wait_b: ScalarInt


def _eight_and_bits(*, a: DiscreteState, b: DiscreteState) -> FloatND:
    return 8.0 + a + b


def _b_is_the_period(*, period: Period) -> DiscreteState:
    return period


def _skip_model(
    *,
    n_stages: int,
    source_stage: int,
    target_stage: int,
    source_period: int = 0,
    next_b: Callable[..., DiscreteState] | None = None,
    composite: bool = False,
    by_age: bool = False,
    enable_jit: bool = True,
) -> lcm.Model:
    """`choose` decides once and lands in the terminal `end`; `wait_*` fill stages.

    `choose` holds two bits, `a` kept and `b` kept unless `next_b` is given; its
    utility is the labor code plus both bits and that of `end` eight plus both.
    """
    crossing = target_stage <= source_stage
    last = source_period + crossing
    stages: dict[str, list[RegimeName]] = {f"stage_{j}": [] for j in range(n_stages)}
    stages[f"stage_{source_stage}"].append("choose")
    stages[f"stage_{target_stage}"].append("end")
    for name in ("wait_a", "wait_b"):
        empty = next((s for s, names in stages.items() if not names), "stage_0")
        stages[empty].append(name)
    bits = {"a": lcm.DiscreteGrid(Labor), "b": lcm.DiscreteGrid(Labor)}
    origin = 60 + source_period if by_age else source_period
    horizon: ClockKwargs = (
        {"ages": lcm.AgeGrid(start=60, inclusive_stop=60 + last, step="Y")}
        if by_age
        else {"n_periods": last + 1}
    )
    starts = {origin: ("choose", "wait_a", "wait_b")}
    return lcm.Model(
        **horizon,
        stages={stage: tuple(names) for stage, names in stages.items()},
        regimes={
            "choose": lcm.Regime(
                states=bits,
                actions={"labor": lcm.DiscreteGrid(Labor)},
                state_transitions={
                    "a": lcm.fixed_transition("a"),
                    "b": lcm.fixed_transition("b") if next_b is None else next_b,
                },
                functions={"utility": _bits_utility},
            ),
            "end": lcm.Regime(states=bits, functions={"utility": _eight_and_bits}),
            "wait_a": lcm.Regime(functions={"utility": _waiting}),
            "wait_b": lcm.Regime(functions={"utility": _waiting}),
        },
        regime_id_class=SkipId,
        initial_nodes=lcm.InitialNodes(by_age=starts)
        if by_age
        else lcm.InitialNodes(by_period=starts),
        edges={
            "choose": {
                "end": (origin,) if by_age else lcm.Periods(values=(source_period,))
            }
        },
        fixed_params={"discount_factor": 0.5} if crossing else {},
        composite_states={
            "bits": lcm.CompositeState(
                members=("a", "b"), start_codes={"a": (0,), "b": (0,)}
            )
        }
        if composite
        else {},
        enable_jit=enable_jit,
    )


def _skip_values(*, crossing: bool, next_b: int) -> tuple[float, float]:
    """The action values of `choose` at bits (0, 0) when the law sets `b` to `next_b`.

    Labor 1 sets `next_b` and labor 0 leaves `b` at 0 unless the law ignores labor;
    `end` is worth eight plus its bits, discounted by one half when it is next period.
    """
    beta = 0.5 if crossing else 1.0
    return beta * 8.0, 1.0 + beta * (8.0 + next_b)


@pytest.mark.parametrize(
    ("n_stages", "source_stage", "target_stage"),
    [(3, 0, 2), (2, 1, 1), (3, 1, 0), (3, 0, 1)],
    ids=[
        "skips-a-stage",
        "next-period-same-stage",
        "skips-into-next-period",
        "adjacent",
    ],
)
@pytest.mark.parametrize("enable_jit", [True, False], ids=["jit", "eager"])
def test_a_lookup_reads_each_target_where_its_edge_lands(
    *, n_stages: int, source_stage: int, target_stage: int, enable_jit: bool
) -> None:
    """`lookup_policy` reads a target's value at its next occurrence, as the solve."""
    model = _skip_model(
        n_stages=n_stages,
        source_stage=source_stage,
        target_stage=target_stage,
        enable_jit=enable_jit,
    )
    solution = model.solve(params={}, log_level="off")
    Q = _skip_values(crossing=target_stage <= source_stage, next_b=0)
    for kept in ((), ("Q",)):
        result = model.lookup_policy(
            params={},
            solution=solution,
            period=0,
            regime_name="choose",
            states={
                "a": jnp.zeros(1, dtype=jnp.int32),
                "b": jnp.zeros(1, dtype=jnp.int32),
            },
            keep_intermediates=kept,
        )
        np.testing.assert_array_equal(np.asarray(result.value).ravel(), [Q[1]])
        assert result.actions is not None
        np.testing.assert_array_equal(np.asarray(result.actions["labor"]).ravel(), [1])
        if kept:
            np.testing.assert_array_equal(np.asarray(result.Q).ravel(), Q)


def _simulated_choose_and_end(
    *, model: lcm.Model, source_period: int, by_age: bool
) -> tuple[float, float, int]:
    """Simulate one agent from `choose` at bits (0, 0): both values, `end`'s period."""
    frame = model.simulate(
        params={},
        initial_conditions={
            "regime_id": jnp.asarray([SkipId.choose], dtype=jnp.int32),
            "a": jnp.zeros(1, dtype=jnp.int32),
            "b": jnp.zeros(1, dtype=jnp.int32),
            "age" if by_age else "period": jnp.asarray(
                [60 + source_period if by_age else source_period]
            ),
        },
        seed=11,
        log_level="off",
    ).to_dataframe()
    source = frame.loc[frame["regime_name"] == "choose"].iloc[0]
    end = frame.loc[frame["regime_name"] == "end"].iloc[0]
    return float(source["value"]), float(end["value"]), int(end["period"])


@pytest.mark.parametrize(
    ("n_stages", "source_stage", "target_stage", "by_age"),
    [
        (2, 1, 0, True),
        (3, 1, 2, False),
        (3, 2, 0, False),
        (3, 1, 1, False),
        (2, 0, 1, False),
    ],
    ids=["crossing-by-age", "later-stage", "last-stage", "same-stage", "first-stage"],
)
@pytest.mark.parametrize(
    "next_b", [None, _labor_kept], ids=["kept", "action-dependent"]
)
def test_a_composite_state_starts_in_the_stage_of_its_regime(
    *,
    n_stages: int,
    source_stage: int,
    target_stage: int,
    by_age: bool,
    next_b: Callable[..., DiscreteState] | None,
) -> None:
    """The start tuples are seeded where the starting regime decides, not at step 0."""
    crossing = target_stage <= source_stage
    model = _skip_model(
        n_stages=n_stages,
        source_stage=source_stage,
        target_stage=target_stage,
        next_b=next_b,
        composite=True,
        by_age=by_age,
    )
    b_next = 0 if next_b is None else 1
    assert _simulated_choose_and_end(model=model, source_period=0, by_age=by_age) == (
        _skip_values(crossing=crossing, next_b=b_next)[1],
        8.0 + b_next,
        int(crossing),
    )


@pytest.mark.parametrize(
    ("n_stages", "source_stage", "target_stage"),
    [(2, 0, 1), (3, 0, 2), (3, 1, 2), (1, 0, 0)],
    ids=["first-stage", "skips-a-stage", "later-stage", "one-stage"],
)
@pytest.mark.parametrize("by_age", [False, True], ids=["period", "age"])
@pytest.mark.parametrize("enable_jit", [True, False], ids=["jit", "eager"])
def test_a_composite_member_law_reads_the_period_once(
    *,
    n_stages: int,
    source_stage: int,
    target_stage: int,
    by_age: bool,
    enable_jit: bool,
) -> None:
    """A member law setting `b` to the period sets it to 1 in period 1, any stage."""
    crossing = target_stage <= source_stage
    model = _skip_model(
        n_stages=n_stages,
        source_stage=source_stage,
        target_stage=target_stage,
        source_period=1,
        next_b=_b_is_the_period,
        composite=True,
        by_age=by_age,
        enable_jit=enable_jit,
    )
    assert _simulated_choose_and_end(model=model, source_period=1, by_age=by_age) == (
        _skip_values(crossing=crossing, next_b=1)[1],
        9.0,
        1 + crossing,
    )


@lcm.categorical(ordered=False)
class OfferId:
    source: ScalarInt
    destination: ScalarInt
    idle_a: ScalarInt
    idle_b: ScalarInt


def _offer_utility() -> FloatND:
    return jnp.asarray(8.0)


def _offer_cost(*, penalty: FloatND) -> FloatND:
    return -penalty


def _open_in_period_zero(*, period: Period) -> BoolND:
    return period == 0


def _open_in_period_one(*, period: Period) -> BoolND:
    return period == 1


def _open_in_period_two(*, period: Period) -> BoolND:
    return period == 2


def _open_at_age_61(*, age: Age) -> BoolND:
    return age == 61


@pytest.mark.parametrize(
    ("n_stages", "source_stage", "period", "available", "open_"),
    [
        (2, 1, 1, _open_in_period_one, True),
        (3, 2, 2, _open_in_period_two, True),
        (2, 1, 1, _open_in_period_zero, False),
        (1, 0, 1, _open_in_period_one, True),
        (2, 1, 1, _open_at_age_61, True),
    ],
    ids=["open", "open-third-stage", "closed", "one-stage", "by-age"],
)
def test_a_time_only_availability_is_proved_at_the_period_of_its_step(
    *,
    n_stages: int,
    source_stage: int,
    period: int,
    available: Callable[..., BoolND],
    open_: bool,
) -> None:
    """A composition action open in its period keeps its parameters and is taken.

    One closed there is pruned: its cost's parameter leaves the template.
    """
    stages: dict[str, list[RegimeName]] = {f"stage_{j}": [] for j in range(n_stages)}
    stages[f"stage_{source_stage}"].extend(("source", "destination"))
    for name in ("idle_a", "idle_b"):
        empty = next((s for s, names in stages.items() if not names), "stage_0")
        stages[empty].append(name)
    model = lcm.Model(
        ages=lcm.AgeGrid(start=60, inclusive_stop=60 + period, step="Y"),
        stages={stage: tuple(names) for stage, names in stages.items()},
        regimes={
            "source": lcm.Regime(
                functions={"utility": _waiting},
                composition_action=lcm.CompositionAction(
                    destinations={
                        "self": lcm.Destination(
                            regime="destination",
                            projection={},
                            utility_adjustment=_offer_cost,
                        )
                    },
                    rule=lcm.PCOnActions(),
                    available=available,
                ),
            ),
            "destination": lcm.Regime(functions={"utility": _offer_utility}),
            "idle_a": lcm.Regime(functions={"utility": _waiting}),
            "idle_b": lcm.Regime(functions={"utility": _waiting}),
        },
        regime_id_class=OfferId,
        initial_nodes=lcm.InitialNodes(
            by_age={60 + period: ("source", "idle_a", "idle_b")}
        ),
        edges={},
    )
    template = flatten_regime_namespace(model.get_params_template())
    assert (
        "source__composition_action__destinations__self__utility_adjustment__penalty"
        in template
    ) == open_
    params: UserParams = (
        {
            "source": {
                "composition_action": {
                    "destinations": {"self": {"utility_adjustment": {"penalty": 1.0}}}
                }
            }
        }
        if open_
        else {}
    )
    values = model.solve(params=params, log_level="off").values
    step = model.step_of(period=period, regime_name="source")
    np.testing.assert_array_equal(
        np.asarray(values[step]["source"]).ravel(), [7.0 if open_ else 0.0]
    )
