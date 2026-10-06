"""The compact per-core plan record a debug solve writes after width selection.

At `log_level="debug"` every compiled solve core leaves one record naming its
logical and physical value shape, devices, selected widths, dense or streamed
dispatch per axis, and the bytes admission charged it. At any other level the
solve writes none and builds none.
"""

import json
import logging
from collections.abc import Mapping

import pytest

from _lcm.execution.execution_plan import (
    CorePlanRecord,
    count_hlo_collectives,
    summarize_transfer_costs,
)
from _lcm.execution.value_transfer import TransferCost, TransferOperationClass
from _lcm.utils.logging import LogLevel
from lcm import ExecutionConfig, Model
from lcm.solver_api import SolutionResult
from lcm_examples import tiny

_FULL = 2**30


@pytest.mark.parametrize(
    "opcode",
    [
        "all-gather",
        "all-reduce",
        "all-to-all",
        "reduce-scatter",
        "collective-permute",
        "collective-broadcast",
    ],
)
@pytest.mark.parametrize("result_type", ["f32[4]{0}", "(f32[4]{0}, f32[4]{0})"])
def test_collective_counts_include_array_and_tuple_results(
    *, opcode: str, result_type: str
) -> None:
    """Count each synchronous collective once regardless of its result type."""
    hlo_text = f"%operation.2 = {result_type} {opcode}(%argument), channel_id=1"

    assert dict(count_hlo_collectives(hlo_text=hlo_text)) == {opcode: 1}


def test_collective_counts_include_start_without_counting_done_twice() -> None:
    """An asynchronous tuple-result collective contributes one operation."""
    hlo_text = (
        "%started = (f32[4]{0}, f32[4]{0}) all-reduce-start(%argument)\n"
        "%finished = f32[4]{0} all-reduce-done(%started)\n"
        "%other = f32[4]{0} all-reduce(%argument)\n"
    )

    assert dict(count_hlo_collectives(hlo_text=hlo_text)) == {"all-reduce": 2}


@pytest.mark.parametrize(
    "hlo_text",
    [
        None,
        "",
        "%copy = f32[4]{0} copy(%argument)",
        "%gather = f32[4]{0} gather(%argument)",
    ],
)
def test_collective_counts_are_empty_without_collective_operations(
    hlo_text: str | None,
) -> None:
    """Unavailable text and modules with no collectives have empty counts."""
    assert dict(count_hlo_collectives(hlo_text=hlo_text)) == {}


class _PlanRecords(logging.Handler):
    """Collect the plan records a solve emits."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[CorePlanRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        plan_record = getattr(record, "core_plan_record", None)
        if plan_record is not None:
            self.records.append(plan_record)


def _model(*, execution_config: ExecutionConfig) -> Model:
    model = tiny.get_model()
    return Model(
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=tiny.RegimeId,
        initial_nodes={model.ages.exact_values[0]: "working_life"},
        edges=model.edges,
        execution_config=execution_config,
    )


def _solve(
    *,
    execution_config: ExecutionConfig,
    log_level: LogLevel = "debug",
) -> tuple[list[CorePlanRecord], SolutionResult]:
    model = _model(execution_config=execution_config)
    handler = _PlanRecords()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        solution = model.solve(params=tiny.get_params(), log_level=log_level)
    finally:
        logger.removeHandler(handler)
    return handler.records, solution


def _records_by_cell(
    records: list[CorePlanRecord],
) -> Mapping[tuple[str, int], CorePlanRecord]:
    return {(record.regime, record.period): record for record in records}


def test_a_debug_solve_writes_one_record_per_solved_regime_period() -> None:
    """Every regime-period the solve publishes a value for has exactly one record."""
    records, solution = _solve(execution_config=ExecutionConfig())

    expected = sorted(
        (regime, period)
        for period, by_regime in solution.values.items()
        for regime in by_regime
    )
    assert sorted((r.regime, r.period) for r in records) == expected


def test_a_progress_solve_writes_no_record() -> None:
    """Below debug the solve builds no plan record."""
    records, _ = _solve(execution_config=ExecutionConfig(), log_level="progress")

    assert records == []


def test_the_logical_value_shape_is_the_published_value_shape() -> None:
    """The record's logical value shape is the shape of the value it publishes."""
    records, solution = _solve(execution_config=ExecutionConfig())

    assert {(r.regime, r.period): r.logical_value_shape for r in records} == {
        (regime, period): tuple(value.shape)
        for period, by_regime in solution.values.items()
        for regime, value in by_regime.items()
    }


def test_the_state_extents_name_the_value_axes_in_order() -> None:
    """The working regime's value has one axis, `wealth`, of 25 grid points."""
    records, _ = _solve(execution_config=ExecutionConfig())

    record = _records_by_cell(records)[("working_life", 0)]
    assert dict(record.state_extents) == {"wealth": 25}


def test_one_device_holds_the_whole_value() -> None:
    """On one device the physical shard is the whole logical value."""
    records, _ = _solve(execution_config=ExecutionConfig())

    assert all(
        r.physical_value_shape == r.logical_value_shape and len(r.device_ids) == 1
        for r in records
    )


def test_full_widths_dispatch_every_axis_densely() -> None:
    """A width covering the whole axis maps it as one dense block."""
    records, _ = _solve(
        execution_config=ExecutionConfig(
            axis_widths={"action_product": _FULL, "cell": _FULL}
        )
    )

    record = _records_by_cell(records)[("working_life", 0)]
    assert set(record.dispatch.values()) == {"dense"}


def test_a_partial_action_width_streams_the_action_product() -> None:
    """An action width below the 200-point product streams it in blocks."""
    records, _ = _solve(
        execution_config=ExecutionConfig(axis_widths={"action_product": 8})
    )

    record = _records_by_cell(records)[("working_life", 0)]
    assert (
        record.widths["action_product"],
        record.axis_extents["action_product"],
        record.dispatch["action_product"],
    ) == (8, 200, "streamed")


def test_an_unbudgeted_solve_records_no_compiler_reservation() -> None:
    """Without a budget admission reserves nothing, and the record says so."""
    records, _ = _solve(execution_config=ExecutionConfig(device_memory_bytes=None))

    assert {r.compiler_reservation_bytes for r in records} == {None}


def test_a_budgeted_solve_records_a_reservation_at_least_its_peak() -> None:
    """The admitted reservation covers the compiler's own raw peak."""
    records, _ = _solve(execution_config=ExecutionConfig(device_memory_bytes=2**34))

    assert all(
        r.compiler_reservation_bytes is not None
        and r.compiler_peak_bytes is not None
        and r.compiler_reservation_bytes >= r.compiler_peak_bytes > 0
        for r in records
    )


def test_a_record_round_trips_through_json() -> None:
    """The rendered line is one JSON object carrying the record's cell."""
    records, _ = _solve(execution_config=ExecutionConfig())

    rendered = json.loads(records[0].to_json())
    assert (rendered["regime"], rendered["period"], rendered["core"]) == (
        records[0].regime,
        records[0].period,
        records[0].core,
    )


def _cost(
    *,
    operation_class: TransferOperationClass,
    logical_bytes: int,
    per_device_bytes: int,
    temporary_bytes: int,
) -> TransferCost:
    return TransferCost(
        operation_class=operation_class,
        logical_bytes=logical_bytes,
        per_device_bytes=per_device_bytes,
        temporary_bytes=temporary_bytes,
        devices=(0, 1),
        reused_by_several_consumers=False,
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("active_replica_bytes", 400 + 30),
        ("logical_gathered_bytes", 400),
        ("transfer_workspace_bytes", 7 + 5),
    ],
)
def test_transfer_costs_separate_replicas_gathers_and_workspace(
    *, field: str, expected: int
) -> None:
    """A local read adds nothing; a collective gather adds its whole value."""
    summary = summarize_transfer_costs(
        costs=(
            _cost(
                operation_class=TransferOperationClass.LOCAL,
                logical_bytes=1000,
                per_device_bytes=500,
                temporary_bytes=0,
            ),
            _cost(
                operation_class=TransferOperationClass.COLLECTIVE,
                logical_bytes=400,
                per_device_bytes=400,
                temporary_bytes=7,
            ),
            _cost(
                operation_class=TransferOperationClass.DEVICE_COPY,
                logical_bytes=60,
                per_device_bytes=30,
                temporary_bytes=5,
            ),
        )
    )

    assert summary[field] == expected
