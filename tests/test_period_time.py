"""Period coordinates have explicit meaning at every public time boundary."""

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

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
from lcm.typing import DiscreteState, FloatND, Period, ScalarInt

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
    initial_nodes: tuple[lcm.InitialNode | tuple[object, str], ...] | None = None,
    **kwargs: Any,
) -> lcm.Model:
    return lcm.Model(
        regimes={
            "work": lcm.Regime(functions={"utility": _flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=(lcm.InitialNode(period=0, regime="work"),)
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
            initial_nodes=(lcm.InitialNode(period=0, regime="work"),),
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
            initial_nodes=(lcm.InitialNode(period=0, regime="work"),),
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
    model = _model(
        n_periods=3, initial_nodes=(lcm.InitialNode(period=1, regime="work"),)
    )
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
    assert model.initial_nodes == frozenset({lcm.InitialNode(period=0, regime="work")})
    assert (
        _model(n_periods=3, initial_nodes=tuple(model.initial_nodes)).graph.nodes
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
        initial_nodes=(lcm.InitialNode(period=0, regime="work"),),
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
        initial_nodes=(lcm.InitialNode(period=0, regime="work"),),
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
        "initial_nodes": (lcm.InitialNode(period=0, regime="work"),),
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


def _age_model() -> lcm.Model:
    return lcm.Model(
        ages=lcm.AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regimes={
            "work": lcm.Regime(functions={"utility": _flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=((0, "work"),),
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


def test_legacy_age_model_state_restores_an_explicit_clock() -> None:
    model = _age_model()
    state = model.__getstate__()
    # Age-only archives carry the grid and public start tuples, without a clock.
    del state["_time"]
    del state["_resolved_initial_nodes"]
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
        initial_nodes=(lcm.InitialNode(period=1, regime="work"),),
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
        initial_nodes=(lcm.InitialNode(period=0, regime="work"),),
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
    model = _model(
        n_periods=3, initial_nodes=(lcm.InitialNode(period=1, regime="work"),)
    )
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
        initial_nodes=(lcm.InitialNode(period=0, regime="work"),),
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
