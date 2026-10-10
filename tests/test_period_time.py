"""Period coordinates have explicit meaning at every public time boundary."""

import logging
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import lcm
import lcm.exceptions
from _lcm.solution import period_replay
from _lcm.time import ModelTime, coordinate_at
from lcm import component_jobs
from lcm.exceptions import ModelInitializationError
from lcm.persistence import PeriodCapture, load_period_capture
from lcm.typing import (
    Age,
    ContinuousState,
    DiscreteState,
    FloatND,
    Period,
    ScalarInt,
    UserInitialNodes,
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


def _model(
    *,
    initial_nodes: UserInitialNodes | None = None,
    **kwargs: Any,
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
        **kwargs,
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
    targets: dict[str, lcm.PeriodRange | lcm.Periods],
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


@pytest.mark.parametrize("n_periods", [0, -1, True, 2.5])
def test_period_horizon_requires_positive_integer(n_periods: object) -> None:
    with pytest.raises(ModelInitializationError, match="n_periods"):
        _model(n_periods=n_periods)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"ages": lcm.AgeGrid(start=40, inclusive_stop=42, step="Y"), "n_periods": 3}],
)
def test_exactly_one_time_coordinate_is_required(kwargs: dict[str, Any]) -> None:
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


def test_by_period_schedule_preserves_stage_boundaries() -> None:
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


@pytest.mark.parametrize("selector", [1.0, True, lcm.AgeRange(start=0)])
def test_by_period_refuses_nonperiod_selectors(
    selector: float | lcm.AgeRange,
) -> None:
    with pytest.raises(lcm.exceptions.RegimeInitializationError, match=r"[Pp]eriod"):
        lcm.ByPeriod(cases={selector: "work"})


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
        {"period": jnp.array([True])},
    ],
)
@pytest.mark.parametrize(
    "method",
    ["simulate", "validate_initial_conditions", "initial_conditions_feasibility"],
)
def test_period_starts_are_checked_at_every_entry(
    *, entry: dict[str, object], method: str
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


def _specialized_flow(period: int) -> Any:
    if type(period) is not int:
        raise TypeError("Specialization requires an integer period.")
    value = float(period + 1)

    def flow() -> FloatND:
        return jnp.asarray(value)

    return flow


def _specialized_model(marker: lcm.AgeSpecializedFunction) -> lcm.Model:
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
    definition: dict[str, Any] = {
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
            lcm.AgeSpecializedFunction(
                build=cast("Any", _specialized_flow), signature=int
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
    state.pop("_transition_only_shock_names", None)
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
    np.testing.assert_allclose(solution.values[1]["work"], [1, 2, 3])


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


@pytest.mark.parametrize("coordinate", [True, 1.0])
def test_period_schedule_inspection_rejects_noninteger_coordinates(
    coordinate: float,
) -> None:
    schedule = lcm.ByPeriod(cases={1: "work"}).resolve(n_periods=3)
    with pytest.raises((TypeError, KeyError), match=r"period|integer"):
        schedule.at(coordinate)


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


@pytest.mark.parametrize("surface", ["logical", "layout", "memory", "declared"])
def test_period_inspection_reports_no_age(
    *, surface: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(n_periods=3)
    if surface == "declared":
        records = model._compile_period_cores(
            params={}, regime_name="work", period=1, axis_widths={}
        )
        assert records
        for record in records:
            assert record.period == 1
            assert record.age is None
        return
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "work@1")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model.solve(params={}, log_level="off")
    directory = tmp_path / "work@1"
    if surface == "memory":
        analysis = period_replay.analyze_period_core_memory(directory=directory)
        assert analysis.period == 1
        assert analysis.age is None
        assert analysis.core_memory_bytes
    else:
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
def test_initial_nodes_requires_one_nonempty_coordinate_map(kwargs: Any) -> None:
    """Exactly one nonempty coordinate mapping declares the starts."""
    with pytest.raises(ModelInitializationError):
        lcm.InitialNodes(**kwargs)


@pytest.mark.parametrize(
    "selector",
    [True, 0.0, Fraction(0), (0, 1.0), lcm.AgeRange(start=0)],
)
def test_initial_nodes_refuses_nonperiod_keys(selector: Any) -> None:
    """Period coordinates must be genuinely integer-valued."""
    with pytest.raises(ModelInitializationError, match=r"[Pp]eriod"):
        lcm.InitialNodes(by_period={selector: "work"})


@pytest.mark.parametrize(
    "selector", [lcm.PeriodRange(start=0), lcm.Periods(values=(0,))]
)
def test_initial_nodes_refuses_period_keys_in_age_map(selector: Any) -> None:
    """An explicit period selector cannot be relabelled as an age."""
    with pytest.raises(ModelInitializationError, match=r"[Aa]ge"):
        lcm.InitialNodes(by_age={selector: "work"})


@pytest.mark.parametrize("selector", [-1, 3, (0, 3), lcm.Periods(values=(3,))])
def test_initial_nodes_checks_period_horizon(selector: Any) -> None:
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
def test_period_initial_nodes_refuses_empty_selection(selector: Any) -> None:
    """A selector must admit at least one starting node."""
    with pytest.raises(ModelInitializationError, match=r"no period"):
        _model(
            n_periods=3, initial_nodes=lcm.InitialNodes(by_period={selector: "work"})
        )


@pytest.mark.parametrize(
    "initial_nodes", [((0, "work"),), {0: "work"}, {lcm.Periods(values=(0,)): "work"}]
)
def test_period_initial_nodes_requires_named_keyword(initial_nodes: Any) -> None:
    """Period declarations cannot silently reinterpret legacy age syntax."""
    with pytest.raises(ModelInitializationError, match=r"by_period"):
        _model(n_periods=3, initial_nodes=initial_nodes)


def test_initial_nodes_rejects_clock_mismatch_both_directions() -> None:
    """The declaration clock must match the model clock."""
    with pytest.raises(ModelInitializationError, match="by_period"):
        _model(n_periods=3, initial_nodes=lcm.InitialNodes(by_age={0: "work"}))
    with pytest.raises(ModelInitializationError, match="by_age"):
        _age_model(initial_nodes=lcm.InitialNodes(by_period={0: "work"}))


@pytest.mark.parametrize("selector", [(0, 1), range(2), lcm.Periods(values=(0, 1))])
def test_period_initial_nodes_accepts_explicit_integer_collections(
    selector: Any,
) -> None:
    """The by_period keyword labels every supported exact-selector form."""
    model = _model(
        n_periods=3, initial_nodes=lcm.InitialNodes(by_period={selector: "work"})
    )
    assert model.graph.initial_nodes == frozenset({(0, "work"), (1, "work")})


@pytest.mark.parametrize("selector", [(0, 3), range(4), lcm.Periods(values=(0, 3))])
def test_period_initial_nodes_does_not_clip_explicit_collections(selector: Any) -> None:
    """Exact starts require grid membership even when another coordinate is valid."""
    with pytest.raises(ModelInitializationError, match=r"[Pp]eriod"):
        _model(
            n_periods=3, initial_nodes=lcm.InitialNodes(by_period={selector: "work"})
        )


@pytest.mark.parametrize(
    "value", [lcm.ByAge(cases={0: "work"}), lcm.ByPeriod(cases={0: "work"})]
)
def test_initial_nodes_refuses_transition_schedules_as_names(value: Any) -> None:
    """Initial-node values name regimes rather than laws selected by another clock."""
    with pytest.raises(ModelInitializationError):
        lcm.InitialNodes(by_period={0: value})


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
    probability = lcm.StochasticTransition(func=lambda: jnp.asarray(1.0))
    probabilities = {"work": probability}
    schedules = {
        "case": lcm.ByPeriod(cases={0: probabilities}),
        "default": lcm.ByPeriod(cases={}, default=probabilities),
        "until": lcm.ByPeriod.until(
            stop_period_exclusive=2, law=probabilities, then="done"
        ),
        "mapped-until": lcm.ByPeriod.until(
            stop_period_exclusive=2, law="work", then="done"
        ).with_mapped_laws(func=lambda _: probabilities),
    }
    probabilities.clear()
    law = cast(
        "dict[str, lcm.StochasticTransition]",
        schedules[wrapper].resolve(n_periods=3).at(0),
    )
    assert law["work"].func is probability.func
    with pytest.raises(TypeError):
        law["done"] = probability


@pytest.mark.parametrize(
    "selector", [lcm.PeriodRange(start=0), lcm.Periods(values=(0,))]
)
@pytest.mark.parametrize("mapping", [False, True])
def test_age_initial_nodes_refuses_legacy_period_selectors(
    *, selector: Any, mapping: bool
) -> None:
    """Legacy age syntax must not admit the explicitly period-labelled wrappers."""
    initial_nodes = {selector: "work"} if mapping else ((selector, "work"),)
    with pytest.raises(ModelInitializationError, match=r"[Aa]ge"):
        _age_model(initial_nodes=initial_nodes)


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
    clock_kwargs: dict[str, Any] = (
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
    frame = simulation.to_dataframe(use_labels=False)
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
