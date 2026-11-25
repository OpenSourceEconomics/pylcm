"""Period coordinates have explicit meaning at every public time boundary."""

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
    UserInitialNodes,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION

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
    np.testing.assert_allclose(solution.value(period=0, regime="work"), 4.0)
    np.testing.assert_allclose(solution.value(period=1, regime="work"), 6.0)
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
            axis_widths={"subject": 1},
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
        actual = coordinate_at(ages=axis, period=period)
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
            directory=tmp_path, periods=(("work", 1),), source_identity=source
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
    actual_value = np.asarray(solution.value(period=source_period, regime="work"))
    np.testing.assert_array_almost_equal(
        actual_value, expected_value, decimal=DECIMAL_PRECISION
    )

    n_subjects = 3
    initial_time = (
        jnp.full(n_subjects, source_period, dtype=jnp.int32)
        if clock == "period"
        else jnp.full(n_subjects, coordinate)
    )
    simulation = model.simulate(
        params={},
        initial_conditions={
            clock: initial_time,
            "regime_id": jnp.full(n_subjects, RegimeId.work, dtype=jnp.int32),
        },
        solution=solution,
        seed=123,
        log_level=log_level,
    )
    frame = simulation.to_dataframe(use_categorical_codes=True)
    assert len(frame) == 2 * n_subjects
    np.testing.assert_array_equal(
        np.sort(frame["period"].to_numpy()),
        np.repeat([source_period, source_period + 1], n_subjects),
    )
    assert ("age" in frame.columns) == (clock == "age")
    terminal = frame.loc[frame["period"] == source_period + 1]
    assert len(terminal) == n_subjects
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
            fixed_params={"discount_factor": 0.9},
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
        else lcm.InitialNodes(by_period={0: "working"}),
        edges={
            "working": {
                "spending": selector(exclusive_stop=origin + last),
                "final": selector(start=origin + last),
            },
            "spending": {"working": selector(exclusive_stop=origin + last)},
        }
        if edges is None
        else edges,
        fixed_params={"discount_factor": 0.9},
    )


def _wage(*, steps: bool) -> lcm.TimeVarying:
    wages = jnp.array([1.0, 1.5, 2.0])
    if steps:
        return lcm.TimeVarying(
            values=jnp.repeat(wages, 2), periods=tuple(range(2 * len(wages)))
        )
    return lcm.TimeVarying(values=wages, periods=(0, 1, 2))


def _staged_frame(*, model: lcm.Model, clock: str) -> pd.DataFrame:
    n_subjects = 4
    initial_time = (
        {"age": jnp.full(n_subjects, 60)}
        if clock == "age"
        else {"period": jnp.zeros(n_subjects, dtype=jnp.int32)}
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
            "regime_id": jnp.full(n_subjects, StagedId.working, dtype=jnp.int32),
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
    time-varying parameter is indexed by economic period, and subjects start in
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
    assert staged.n_periods == hand.n_periods == 2 * _N_ECONOMIC_PERIODS
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
    for column in ("subject_id", "regime_name", "value", "wealth", "labor"):
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
        ("spending", "working") if case == "exit" else ("working", "spending")
    )
    if case in {"exit", "observes"}:
        regimes[source] = regimes[source].replace(
            exit=lcm.Exit(
                targets={
                    "self": lcm.ExitTarget(
                        regime=destination, projection={"wealth": _wealth_kept}
                    )
                },
                rule=lcm.Unilateral(),
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
        "exit": {"regimes": regimes},
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
        "beyond": {"stages": {"work": ("working", "final"), "spend": ("spending",)}},
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
    }
    return changes[case]


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", r"in none: \['final'\]"),
        ("repeated", r"in several stages: \['final'\]"),
        ("exit", r"exit of 'spending' in stage 'spend' reads 'working' of the earlier"),
        ("observes", r"observes \['eps'\] of 'spending', drawn in its later stage"),
        ("aggregator", r"'working' declares a Koopmans aggregator in stage 'work'"),
        ("egm", r"'working' uses an endogenous-grid solver in stage 'work'"),
        ("process", r"'spending' sits out stage 'work' .* processes \['income'\]"),
        ("beyond", r"requires 'final' sitting out stage 'spend' at age 2, which is"),
        ("pair", r"'working' maps to a `Phased` pair of transitions. Phase"),
        ("final", r"'spending' → 'working' selects only the final age"),
    ],
)
def test_stages_refuse_declarations_out_of_stage_order(
    *, case: str, match: str
) -> None:
    """A refusal names declared regimes and stages, never a generated regime."""
    with pytest.raises(ModelInitializationError, match=match):
        _staged_model(clock="staged", **_out_of_stage_order(case))


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
            exit=lcm.Exit(
                targets={
                    member: lcm.ExitTarget(
                        regime=single, projection={"wealth": _half_of_wealth}
                    )
                    for member in ("f", "m")
                },
                rule=lcm.Unilateral(),
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

    The single's edge to itself skips the parting stage, the couple's exit enters
    the single in a later stage, and singles start in their own stage; written
    with a hand-made waiting regime, the values and the simulated rows agree.
    """
    params = {"couple_sep": {"love": 0.2}}
    staged = _marriage(hand=False)
    hand = _marriage(hand=True)
    staged_values = staged.solve(params=params, log_level="off").values
    hand_values = hand.solve(params=params, log_level="off").values
    for step, regimes in hand_values.items():
        for name, value in regimes.items():
            if name != "single_wait":
                np.testing.assert_array_equal(staged_values[step][name], value)
    staged_frame = _marriage_frame(model=staged, hand=False)
    hand_frame = _marriage_frame(model=hand, hand=True)
    hand_frame = hand_frame[hand_frame["regime_name"] != "single_wait"]
    assert "single" in set(staged_frame["regime_name"])
    assert set(staged_frame["regime_name"]) == set(hand_frame["regime_name"])
    pd.testing.assert_frame_equal(
        staged_frame.reset_index(drop=True),
        hand_frame.reset_index(drop=True),
        check_categorical=False,
    )


def test_an_exit_reads_the_destination_at_its_occurrence_later_in_the_period() -> None:
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
                exit=lcm.Exit(
                    targets={
                        "self": lcm.ExitTarget(regime="couple", role="f", projection={})
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
def test_the_user_never_sees_a_pass_through(*, variant: str, tmp_path: Path) -> None:
    """A regime that sits out a stage is known only by the name the user declared.

    The parameter template, the parameters of an edge that skips a stage, the
    regime codes, the starts, a refusal of missing parameters, the values of a
    solution and of its saved copy, the raw and tabulated simulation and its
    saved copy name only declared regimes.
    """
    model, params, starts, declared = _skipping_case(variant)
    generated = model._time.pass_throughs
    assert generated
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
    seen = repr(
        (
            model.get_params_template(),
            model.initial_nodes,
            [list(regimes) for regimes in solution.values.values()],
            [list(regimes) for regimes in loaded.values.values()],
            list(result.raw_results),
            list(restored.raw_results),
        )
    )
    if params:
        with pytest.raises(lcm.exceptions.InvalidParamsError) as missing:
            model.validate_transitions(params={})
        seen += str(missing.value)
    assert [name for name in generated if name in seen] == []


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


def _certain() -> FloatND:
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
    """A couple that may separate at a settlement in the first stage of a period."""
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
                exit=lcm.Exit(
                    rule=lcm.MutualConsent(
                        settle=lcm.Settlement(
                            name="share",
                            candidates=lcm.LinSpacedGrid(
                                start=0.0, stop=1.0, n_points=5
                            ),
                            default=_half_reading_period,
                        )
                    ),
                    targets={
                        member: lcm.ExitTarget(
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
        "exit": {
            "exit": lcm.Exit(
                targets={
                    "self": lcm.ExitTarget(
                        regime="final",
                        projection={"wealth": _wealth_kept_reading_period},
                        utility_adjustment=_no_adjustment_reading_period,
                    )
                },
                rule=lcm.Unilateral(),
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
    if kind == "settlement":
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
        "exit",
        "shock_law",
        "pareto_weight",
        "settlement",
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
