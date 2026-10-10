"""Runtime validation and verbosity remain independent in solve and simulate."""

import logging
from pathlib import Path

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import _lcm.solution.validate_V as value_validation
from _lcm.utils.logging import (
    LogLevel,
    get_logger,
    validation_enabled,
    validation_raises,
)
from lcm import (
    AgeGrid,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidParamsError,
    InvalidSimulationInputError,
    InvalidValueFunctionError,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(*, consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return jnp.log(consumption + 1) + 0.01 * wealth


def _next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def _borrowing_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 1, RegimeId.dead, RegimeId.alive)


def _make_model() -> Model:
    alive = UserRegime(
        functions={"utility": _utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
    )
    dead = UserRegime(
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets={"alive": 0, "dead": (0, 1)},
                law=DeterministicTransition(func=_next_regime),
            )
        },
    )


_HEALTHY_PARAMS = {"discount_factor": 0.95, "interest_rate": 0.05}
_LOG_LEVELS = ("off", "warning", "progress", "debug")
_INITIAL_CONDITIONS = {
    "age": jnp.array([0.0]),
    "wealth": jnp.array([5.0]),
    "regime_id": jnp.array([RegimeId.alive]),
}


@pytest.mark.parametrize("log_level", _LOG_LEVELS)
def test_runtime_checks_solve_default_rejects_nan(log_level: LogLevel) -> None:
    """Reject NaN value functions by default at every verbosity."""
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    with pytest.raises(InvalidValueFunctionError, match="alive"):
        model.solve(params=params, log_level=log_level)


@pytest.mark.parametrize("log_level", _LOG_LEVELS)
def test_runtime_checks_solve_disabled_allows_nan(log_level: LogLevel) -> None:
    """Allow NaN value functions only when runtime checks are disabled."""
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    solution = model.solve(params=params, log_level=log_level, runtime_checks=False)
    assert any(
        bool(jnp.any(jnp.isnan(value)))
        for values in solution.values.values()
        for value in values.values()
    )


@pytest.mark.parametrize("log_level", _LOG_LEVELS)
@pytest.mark.parametrize("supplied_solution", [False, True])
def test_runtime_checks_simulate_default_rejects_nan(
    *, log_level: LogLevel, supplied_solution: bool
) -> None:
    """Reject NaNs from an automatic solve or a supplied solution by default."""
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    solution = (
        model.solve(params=params, log_level="off", runtime_checks=False)
        if supplied_solution
        else None
    )
    with pytest.raises(InvalidValueFunctionError, match="alive"):
        model.simulate(
            params=params,
            initial_conditions=_INITIAL_CONDITIONS,
            solution=solution,
            log_level=log_level,
            seed=0,
        )


@pytest.mark.parametrize("log_level", _LOG_LEVELS)
@pytest.mark.parametrize("supplied_solution", [False, True])
def test_runtime_checks_simulate_disabled_allows_nan(
    *, log_level: LogLevel, supplied_solution: bool
) -> None:
    """Disabling checks permits simulation with NaN continuation values."""
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    solution = (
        model.solve(params=params, log_level="off", runtime_checks=False)
        if supplied_solution
        else None
    )
    result = model.simulate(
        params=params,
        initial_conditions=_INITIAL_CONDITIONS,
        solution=solution,
        log_level=log_level,
        runtime_checks=False,
        seed=0,
    )
    assert any(
        bool(jnp.any(jnp.isnan(value)))
        for values in result.period_to_regime_to_V_arr.values()
        for value in values.values()
    )


@pytest.mark.parametrize("log_level", _LOG_LEVELS)
def test_runtime_checks_preserve_healthy_results(log_level: LogLevel) -> None:
    """Checking runtime validity leaves healthy values and trajectories unchanged."""
    model = _make_model()
    checked = model.solve(
        params=_HEALTHY_PARAMS, log_level=log_level, runtime_checks=True
    )
    unchecked = model.solve(
        params=_HEALTHY_PARAMS, log_level=log_level, runtime_checks=False
    )
    for period, values in checked.values.items():
        for regime, value in values.items():
            assert_array_equal(value, unchecked.values[period][regime])
    checked_simulation = model.simulate(
        params=_HEALTHY_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        solution=checked,
        log_level=log_level,
        runtime_checks=True,
        seed=0,
    )
    unchecked_simulation = model.simulate(
        params=_HEALTHY_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        solution=unchecked,
        log_level=log_level,
        runtime_checks=False,
        seed=0,
    )
    assert_frame_equal(
        checked_simulation.to_dataframe(), unchecked_simulation.to_dataframe()
    )


def test_runtime_checks_disabled_preserves_required_params() -> None:
    """Disabling runtime checks still requires the model's declared parameters."""
    model = _make_model()
    with pytest.raises(InvalidParamsError, match="interest_rate"):
        model.solve(
            params={"discount_factor": 0.95}, log_level="off", runtime_checks=False
        )


def test_runtime_checks_disabled_preserves_solution_identity() -> None:
    """Disabling runtime checks still rejects solutions with different parameters."""
    model = _make_model()
    solution = model.solve(
        params=_HEALTHY_PARAMS, log_level="off", runtime_checks=False
    )
    with pytest.raises(InvalidSimulationInputError, match="param"):
        model.simulate(
            params={**_HEALTHY_PARAMS, "interest_rate": 0.06},
            initial_conditions=_INITIAL_CONDITIONS,
            solution=solution,
            log_level="off",
            runtime_checks=False,
            seed=0,
        )


def test_runtime_checks_disabled_preserves_initial_conditions_structure() -> None:
    """Disabling runtime checks still rejects unknown initial regime IDs."""
    model = _make_model()
    with pytest.raises(InvalidInitialConditionsError, match="regime"):
        model.simulate(
            params=_HEALTHY_PARAMS,
            initial_conditions={
                **_INITIAL_CONDITIONS,
                "regime_id": jnp.array([99]),
            },
            log_level="off",
            runtime_checks=False,
            seed=0,
        )


def test_warning_level_solves_without_per_row_materialisation():
    """Happy-path solve at log_level="warning" returns finite V without
    entering the failure-path localisation."""
    model = _make_model()
    period_to_regime_to_V_arr = model.solve(params=_HEALTHY_PARAMS, log_level="warning")
    for regime_to_V in period_to_regime_to_V_arr.values.values():
        for V_arr in regime_to_V.values():
            assert not jnp.any(jnp.isnan(V_arr))
            assert not jnp.any(jnp.isinf(V_arr))


def test_nan_failure_raises_with_regime_and_age():
    """A NaN-producing parameter set raises with the offending (regime, age).

    `discount_factor=NaN` poisons the next-V contribution to Q on the
    first non-terminal period; the validator must surface the offending
    regime in the error message.
    """
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    with pytest.raises(InvalidValueFunctionError, match=r"alive"):
        model.solve(params=params, log_level="debug")


def test_nan_failure_raises_at_warning_level(
    caplog: pytest.LogCaptureFixture,
):
    """At warning verbosity a NaN value function still stops the solve."""
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    with (
        caplog.at_level(logging.WARNING, logger="lcm"),
        pytest.raises(InvalidValueFunctionError, match="alive"),
    ):
        model.solve(params=params, log_level="warning")


def test_off_level_solves_without_diagnostics(caplog: pytest.LogCaptureFixture):
    """Disabled checks at off verbosity permit NaNs without diagnostic output."""
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    with caplog.at_level(logging.DEBUG):
        period_to_regime_to_V_arr = model.solve(
            params=params, log_level="off", runtime_checks=False
        )
    assert period_to_regime_to_V_arr is not None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_debug_level_emits_per_period_stats(
    *, caplog: pytest.LogCaptureFixture, tmp_path: Path, runtime_checks: bool
):
    """log_level="debug" logs a min/max/mean line for every (regime, period)."""
    model = _make_model()
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        model.solve(
            params=_HEALTHY_PARAMS,
            log_level="debug",
            log_path=tmp_path,
            runtime_checks=runtime_checks,
        )
    debug_stat_lines = [
        r
        for r in caplog.records
        if "V min=" in r.getMessage() and "max=" in r.getMessage()
    ]
    assert len(debug_stat_lines) >= 1


@pytest.mark.parametrize("first_level", ["off", "debug"])
@pytest.mark.parametrize("first_checks", [False, True])
def test_runtime_checks_logger_calls_keep_independent_policies(
    *, first_level: LogLevel, first_checks: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Constructing another logger preserves the first call's output and checks."""
    first = get_logger(log_level=first_level, runtime_checks=first_checks)
    second = get_logger(
        log_level="debug" if first_level == "off" else "off",
        runtime_checks=not first_checks,
    )
    assert validation_enabled(first) is first_checks
    assert validation_raises(first) is first_checks
    assert validation_enabled(second) is not first_checks
    assert validation_raises(second) is not first_checks
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        first.debug("first call")
        second.debug("second call")
    assert [record.getMessage() for record in caplog.records] == [
        "second call" if first_level == "off" else "first call"
    ]


@pytest.mark.parametrize("log_level", ["off", "warning"])
def test_runtime_checks_enrichment_failure_respects_output_policy(
    *,
    log_level: LogLevel,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed enrichment preserves the NaN error and the call's console policy."""
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}

    def fail_enrichment(**_arguments: object) -> None:
        raise RuntimeError("enrichment sentinel")

    monkeypatch.setattr(value_validation, "_enrich_with_diagnostics", fail_enrichment)
    with (
        caplog.at_level(logging.DEBUG, logger="lcm"),
        pytest.raises(InvalidValueFunctionError, match="alive"),
    ):
        model.solve(params=params, log_level=log_level)
    enrichment_warnings = [
        record.getMessage()
        for record in caplog.records
        if "Diagnostic enrichment failed" in record.getMessage()
    ]
    assert enrichment_warnings == (
        ["Diagnostic enrichment failed; raising original NaN error"]
        if log_level == "warning"
        else []
    )
