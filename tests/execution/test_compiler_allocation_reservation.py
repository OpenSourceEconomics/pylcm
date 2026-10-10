"""Budget admission accounts for complete represented compiler allocations."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from _lcm.execution.workspace_planning import (
    CompilerMemoryReservation,
    compiler_memory_reservation,
    compiler_peak_bytes,
    plan_workspace,
)
from lcm.exceptions import ExecutionPlanningError


def memory_stats(*, peak: object, **fields: object) -> SimpleNamespace:
    """Build an explicit complete synthetic compiler report."""
    values = {
        "peak_memory_in_bytes": peak,
        "argument_size_in_bytes": 0,
        "output_size_in_bytes": 0,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 0,
        "generated_code_size_in_bytes": 0,
        "host_generated_code_size_in_bytes": 0,
        "host_argument_size_in_bytes": 0,
        "host_output_size_in_bytes": 0,
        "host_alias_size_in_bytes": 0,
        "host_temp_size_in_bytes": 0,
    }
    values.update(fields)
    return SimpleNamespace(**values)


class _Executable:
    def __init__(self, analysis: object) -> None:
        self.analysis = analysis

    def memory_analysis(self) -> object:
        return self.analysis

    def __call__(self) -> None:
        pytest.fail("Compilation-only admission executed a candidate")


def synthetic_memory(peak: int) -> CompilerMemoryReservation:
    """Provide complete controlled accounting whose raw peak dominates storage."""
    return compiler_memory_reservation(
        compiled=_Executable(memory_stats(peak=peak)), widths={}
    )


@pytest.mark.parametrize(
    ("peak", "argument", "output", "alias", "temporary", "expected"),
    [
        (90, 56, 2, 0, 131_072, 131_130),
        (200, 80, 50, 50, 20, 200),
        (10, 80, 50, 50, 20, 100),
        (0, 0, 0, 0, 0, 0),
    ],
    ids=["temporary-dominates", "peak-dominates", "aliased-output", "empty-storage"],
)
def test_reservation_preserves_raw_peak_and_accounts_for_allocations(
    *, peak: int, argument: int, output: int, alias: int, temporary: int, expected: int
) -> None:
    executable = _Executable(
        memory_stats(
            peak=peak,
            argument_size_in_bytes=argument,
            output_size_in_bytes=output,
            alias_size_in_bytes=alias,
            temp_size_in_bytes=temporary,
        )
    )
    memory = compiler_memory_reservation(compiled=executable, widths={})
    assert (memory.peak_bytes, memory.reservation_bytes) == (peak, expected)
    assert compiler_peak_bytes(compiled=executable, widths={}) == peak


@pytest.mark.parametrize("resident", [0, 17])
def test_reservation_and_external_owners_must_both_fit(resident: int) -> None:
    executable = _Executable(memory_stats(peak=1, temp_size_in_bytes=100))
    with pytest.raises(ExecutionPlanningError, match="compiler reservation"):
        plan_workspace(
            axes=(),
            compile_candidate=lambda _: executable,
            budget_bytes=99 + resident,
            resident_bytes=resident,
        )
    plan = plan_workspace(
        axes=(),
        compile_candidate=lambda _: executable,
        budget_bytes=100 + resident,
        resident_bytes=resident,
    )
    assert plan.compiled is executable


@pytest.mark.parametrize(
    "field",
    [
        "peak_memory_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    ],
)
@pytest.mark.parametrize("invalid", [None, True, -1, 1.0])
def test_malformed_allocation_counters_refuse(*, field: str, invalid: object) -> None:
    report = memory_stats(peak=1, **{field: invalid})
    with pytest.raises(ExecutionPlanningError, match="per-device reservation"):
        plan_workspace(
            axes=(),
            compile_candidate=lambda _: _Executable(report),
            budget_bytes=100,
        )


@pytest.mark.parametrize(
    "field",
    [
        "peak_memory_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    ],
)
def test_report_missing_a_counter_is_not_an_allocation_record(field: str) -> None:
    """A report lacking any counter pylcm reads is refused as a whole."""
    report = vars(memory_stats(peak=1))
    report.pop(field)
    with pytest.raises(ExecutionPlanningError, match="not an allocation record"):
        plan_workspace(
            axes=(),
            compile_candidate=lambda _: _Executable(SimpleNamespace(**report)),
            budget_bytes=100,
        )


@pytest.mark.parametrize(("argument", "output"), [(4, 8), (8, 4)])
def test_aliases_cannot_exceed_either_represented_category(
    *, argument: int, output: int
) -> None:
    executable = _Executable(
        memory_stats(
            peak=1,
            argument_size_in_bytes=argument,
            output_size_in_bytes=output,
            alias_size_in_bytes=5,
        )
    )
    with pytest.raises(ExecutionPlanningError, match="aliases exceed"):
        compiler_memory_reservation(compiled=executable, widths={})


@pytest.mark.parametrize(
    "field",
    [
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    ],
)
def test_host_allocation_space_is_explicitly_refused(field: str) -> None:
    executable = _Executable(memory_stats(peak=1, **{field: 1000}))
    with pytest.raises(ExecutionPlanningError, match="host/default allocation spaces"):
        compiler_memory_reservation(compiled=executable, widths={})


def test_generated_code_metadata_does_not_reclassify_allocation_space() -> None:
    executable = _Executable(
        memory_stats(
            peak=5,
            generated_code_size_in_bytes=1_000_000,
            host_generated_code_size_in_bytes=1_000_000,
        )
    )
    memory = compiler_memory_reservation(compiled=executable, widths={})
    assert memory.reservation_bytes == 5


@pytest.mark.parametrize(
    "analysis",
    [
        [],
        {},
        [memory_stats(peak=1)],
        {"device-0": memory_stats(peak=1)},
        vars(memory_stats(peak=1)),
    ],
    ids=(
        "empty-sequence",
        "empty-mapping",
        "record-sequence",
        "record-mapping",
        "string-keyed-record",
    ),
)
def test_reservation_refuses_what_is_not_an_allocation_record(
    analysis: object,
) -> None:
    """Only the attribute record JAX returns on every backend is read."""
    with pytest.raises(ExecutionPlanningError, match="not an allocation record"):
        compiler_memory_reservation(compiled=_Executable(analysis), widths={})


def test_columnar_peak_refuses_the_reservation() -> None:
    executable = _Executable(memory_stats(peak=[5, 10], temp_size_in_bytes=100))
    with pytest.raises(ExecutionPlanningError, match="per-device reservation"):
        compiler_memory_reservation(compiled=executable, widths={})


def test_columnar_peak_refuses_the_peak_reader() -> None:
    executable = _Executable(memory_stats(peak=[5, 10], temp_size_in_bytes=100))
    with pytest.raises(ExecutionPlanningError, match="per-device peak"):
        compiler_peak_bytes(compiled=executable, widths={})


def test_cached_lookup_requires_complete_reservation() -> None:
    def insufficient(_: object) -> Any:
        return 1

    with pytest.raises(ExecutionPlanningError, match="complete reservation"):
        plan_workspace(
            axes=(),
            compile_candidate=lambda _: object(),
            budget_bytes=100,
            memory_for=insufficient,
        )


def test_unbudgeted_execution_does_not_read_a_report() -> None:
    executable = object()

    def fail(_: object) -> Any:
        pytest.fail("Unbudgeted planning consulted compiler memory")

    def compile_candidate(_: Mapping[str, int]) -> object:
        return executable

    plan = plan_workspace(axes=(), compile_candidate=compile_candidate, memory_for=fail)
    assert (plan.compiled, plan.peak_bytes, plan.reservation_bytes) == (
        executable,
        None,
        None,
    )
