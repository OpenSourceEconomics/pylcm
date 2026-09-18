"""Red-green tests for logging the resolved simulation execution plan.

`Model.simulate` reports, once per call, the resolved execution route
(legacy/subjects), the subject devices, the resolved axis widths, the outer
chunk admission and the budget mode -- see
`_lcm.simulation.plan_summary.SimulationPlanSummary`. This is diagnostic
only: it changes no numerical, RNG, ownership, or admission decision, so
several checks below also confirm the simulated output is unaffected by
whether the record is logged.

The subject-sharded (multi-device) route runs in a four-CPU-device child
process, following the idiom in `tests/test_sharded_state_pruned_from_regimes.py`:
the placement under test only exists where several devices do, and a device
topology must be pinned before JAX initializes a backend.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest

import tests.conftest
from _lcm.utils.logging import LogLevel
from lcm.execution import ExecutionConfig
from lcm.result import SimulationResult
from lcm.typing import UserInitialConditions
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_N_PERIODS = 3
_N_SUBJECTS = 4
_N_DEVICES = 4


def _initial_conditions(n_subjects: int = _N_SUBJECTS) -> UserInitialConditions:
    return {
        "wealth": jnp.linspace(10.0, 40.0, n_subjects),
        "age": jnp.full(n_subjects, 18.0),
        "regime_id": jnp.full(n_subjects, RegimeId.working_life, dtype=jnp.int32),
    }


class _Collector(logging.Handler):
    """Collect every emitted record, independently of the logger's own level."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_legacy(*, log_level: LogLevel) -> tuple[SimulationResult, list[str]]:
    """Run the single-device (legacy) model and return the result and log lines."""
    logger = logging.getLogger("lcm")
    collector = _Collector()
    logger.addHandler(collector)
    try:
        model = get_model(n_periods=_N_PERIODS)
        params = get_params(n_periods=_N_PERIODS)
        solution = model.solve(params=params, log_level="off")
        result = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(),
            solution=solution,
            log_level=log_level,
            seed=1,
        )
    finally:
        logger.removeHandler(collector)
    messages = [
        record.getMessage() for record in collector.records if record.name == "lcm"
    ]
    return result, messages


def test_legacy_run_logs_the_plan_summary_and_detail_at_debug() -> None:
    """A legacy (single-device) `log_level='debug'` run reports the resolved plan."""
    result, messages = _run_legacy(log_level="debug")
    summary_lines = [m for m in messages if m.startswith("Simulation plan:")]
    detail_lines = [m for m in messages if m.startswith("Simulation plan detail:")]
    assert len(summary_lines) == 1
    assert len(detail_lines) == 1
    assert "route=legacy" in summary_lines[0]
    assert "route=legacy" in detail_lines[0]
    assert "outer_chunk_count=1" in detail_lines[0]
    assert "budget_mode=unbudgeted" in detail_lines[0]
    assert "effective_device_memory_bytes=None" in detail_lines[0]

    assert result.plan_summary is not None
    assert result.plan_summary.route == "legacy"
    assert result.plan_summary.outer_chunk_count == 1
    assert result.plan_summary.admitted_chunk_widths == (_N_SUBJECTS,)
    assert result.plan_summary.budget_mode == "unbudgeted"
    assert result.plan_summary.effective_device_memory_bytes is None
    assert len(result.plan_summary.subject_device_ids) >= 1
    assert "working_life" in result.plan_summary.axis_widths_by_regime


def test_legacy_run_logs_only_the_one_liner_at_progress() -> None:
    """At `log_level='progress'` the one-liner shows, never the detail line."""
    _, messages = _run_legacy(log_level="progress")
    assert any(m.startswith("Simulation plan:") for m in messages)
    assert not any(m.startswith("Simulation plan detail:") for m in messages)


@pytest.mark.parametrize("log_level", ["warning", "off"])
def test_legacy_run_logs_nothing_above_its_chosen_level(log_level: LogLevel) -> None:
    """Neither the one-liner nor the detail line appears below the progress tier."""
    _, messages = _run_legacy(log_level=log_level)
    assert not any(m.startswith("Simulation plan") for m in messages)


def test_simulation_output_is_identical_with_logging_on_or_off() -> None:
    """The diagnostic record changes no simulated value."""
    result_debug, _ = _run_legacy(log_level="debug")
    result_off, _ = _run_legacy(log_level="off")
    pd.testing.assert_frame_equal(
        result_debug.to_dataframe(), result_off.to_dataframe()
    )


def test_budgeted_run_reports_the_effective_budget() -> None:
    """A budgeted run's plan summary carries the effective device-memory bytes."""
    budget = 1 << 30
    logger = logging.getLogger("lcm")
    collector = _Collector()
    logger.addHandler(collector)
    try:
        model = get_model(
            n_periods=_N_PERIODS,
            execution_config=ExecutionConfig(device_memory_bytes=budget),
        )
        params = get_params(n_periods=_N_PERIODS)
        solution = model.solve(params=params, log_level="off")
        result = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(),
            solution=solution,
            log_level="debug",
            seed=1,
        )
    finally:
        logger.removeHandler(collector)
    messages = [
        record.getMessage() for record in collector.records if record.name == "lcm"
    ]
    assert result.plan_summary is not None
    assert result.plan_summary.budget_mode == "budgeted"
    assert result.plan_summary.effective_device_memory_bytes is not None
    assert result.plan_summary.effective_device_memory_bytes <= budget
    detail_lines = [m for m in messages if m.startswith("Simulation plan detail:")]
    assert len(detail_lines) == 1
    assert "budget_mode=budgeted" in detail_lines[0]
    assert (
        f"effective_device_memory_bytes="
        f"{result.plan_summary.effective_device_memory_bytes}" in detail_lines[0]
    )


def test_a_saved_and_reloaded_result_carries_no_plan_summary(tmp_path: Path) -> None:
    """The plan summary is diagnostic only and does not enter the archive."""
    result, _ = _run_legacy(log_level="debug")
    assert result.plan_summary is not None
    result.save(directory=tmp_path / "result")
    reloaded = SimulationResult.load(directory=tmp_path / "result")
    assert reloaded.plan_summary is None
    pd.testing.assert_frame_equal(reloaded.to_dataframe(), result.to_dataframe())


def report_subject_sharded(*, decimal: int) -> dict[str, Any]:
    """Run the subject-sharded route on four CPU devices and report the plan.

    Module-level so the four-device child process can import and call it (see
    `_run_in_four_device_process`). `decimal` is unused here (accepted to match
    the reusable child-process harness signature) but keeps parity with the
    `test_sharded_state_pruned_from_regimes` idiom.
    """
    del decimal
    logger = logging.getLogger("lcm")
    collector = _Collector()
    logger.addHandler(collector)
    try:
        ids = tuple(device.id for device in __import__("jax").devices()[:_N_DEVICES])
        model = get_model(
            n_periods=_N_PERIODS,
            execution_config=ExecutionConfig(
                devices=ids,
                simulation_sharding="subjects",
                axis_widths={"subject": _N_DEVICES},
            ),
        )
        params = get_params(n_periods=_N_PERIODS)
        solution = model.solve(params=params, log_level="off")
        result = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(n_subjects=_N_DEVICES),
            solution=solution,
            log_level="debug",
            seed=1,
        )
    finally:
        logger.removeHandler(collector)
    messages = [
        record.getMessage() for record in collector.records if record.name == "lcm"
    ]
    summary = result.plan_summary
    assert summary is not None
    return {
        "messages": messages,
        "route": summary.route,
        "n_subject_devices": len(summary.subject_device_ids),
        "subject_backend": summary.subject_backend,
        "outer_chunk_count": summary.outer_chunk_count,
        "admitted_chunk_widths": list(summary.admitted_chunk_widths),
        "budget_mode": summary.budget_mode,
    }


def _run_in_four_device_process(*, entry_point: str) -> dict[str, Any]:
    """Run one module-level report function on four CPU devices and return it.

    The child carries this run's float policy -- `jax_enable_x64`, the matmul
    precision and the matching `DECIMAL_PRECISION` -- because a topology pin
    needs a fresh process and a fresh process reads none of pytest's options.
    """
    code = (
        "import json, sys; import jax; "
        f"jax.config.update('jax_num_cpu_devices', {_N_DEVICES}); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "jax.config.update('jax_default_matmul_precision', 'highest'); "
        "from tests.simulation.test_simulation_plan_summary import "
        f"{entry_point} as entry; "
        "sys.stdout.write('@@' + json.dumps("
        f"entry(decimal={tests.conftest.DECIMAL_PRECISION!r})) + '@@')"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    _, _, rest = completed.stdout.partition("@@")
    payload, _, _ = rest.partition("@@")
    assert payload, completed.stdout
    return json.loads(payload)


@pytest.fixture(scope="module")
def subject_sharded() -> dict[str, Any]:
    """Return the subject-sharded report from one four-device child process."""
    return _run_in_four_device_process(entry_point="report_subject_sharded")


def test_subjects_run_logs_the_route_and_device_count(
    subject_sharded: dict[str, Any],
) -> None:
    """The subject-sharded route reports `route=subjects` and every device."""
    assert subject_sharded["route"] == "subjects"
    assert subject_sharded["n_subject_devices"] == _N_DEVICES
    assert subject_sharded["subject_backend"] == "cpu"


def test_subjects_run_logs_at_debug(subject_sharded: dict[str, Any]) -> None:
    """A subjects run at `log_level='debug'` logs both the summary and detail."""
    summary_lines = [
        m for m in subject_sharded["messages"] if m.startswith("Simulation plan:")
    ]
    detail_lines = [
        m
        for m in subject_sharded["messages"]
        if m.startswith("Simulation plan detail:")
    ]
    assert len(summary_lines) == 1
    assert len(detail_lines) == 1
    assert "route=subjects" in summary_lines[0]
    assert "route=subjects" in detail_lines[0]


def test_subjects_run_reports_chunking(subject_sharded: dict[str, Any]) -> None:
    """The subjects run reports the outer chunk count and admitted widths."""
    assert subject_sharded["outer_chunk_count"] >= 1
    assert sum(subject_sharded["admitted_chunk_widths"]) >= _N_DEVICES
