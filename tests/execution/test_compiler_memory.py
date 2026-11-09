"""The compiler's memory report is read into byte counts, or into no report.

A backend that exposes no memory analysis, or a report missing a counter pylcm reads,
yields `None` rather than a partial record.
"""

from dataclasses import asdict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.compiler_memory import compiler_memory_bytes


def test_compiler_memory_bytes_normalizes_real_and_unsupported_backends() -> None:
    """Backend-specific stats become integer-or-None fields, or no report."""
    compiled = jax.jit(lambda values: values + 1).lower(jnp.ones(2)).compile()

    report = compiler_memory_bytes(compiled=compiled)

    raw = compiled.memory_analysis()
    if raw is None:
        assert report is None
    else:
        assert report is not None
        assert all(
            value is None or isinstance(value, int) for value in asdict(report).values()
        )

    class UnsupportedExecutable:
        def __init__(self, *, raises: bool) -> None:
            self.raises = raises

        def memory_analysis(self) -> None:
            if self.raises:
                raise RuntimeError("backend has no memory analysis")

    assert compiler_memory_bytes(compiled=UnsupportedExecutable(raises=False)) is None
    assert compiler_memory_bytes(compiled=UnsupportedExecutable(raises=True)) is None


class _ReportingExecutable[Report]:
    def __init__(self, *, report: Report) -> None:
        self.report = report

    def memory_analysis(self) -> Report:
        """Return the supplied report without normalizing its structure."""
        return self.report


@pytest.mark.parametrize(
    "report",
    [
        SimpleNamespace(peak_memory_in_bytes=1, temp_size_in_bytes=2),
        {"peak_memory_in_bytes": 1},
        [SimpleNamespace(peak_memory_in_bytes=1)],
    ],
    ids=("missing-counters", "string-keyed-record", "record-sequence"),
)
def test_compiler_memory_bytes_gives_no_report_for_a_non_record(
    report: SimpleNamespace | dict[str, int] | list[SimpleNamespace],
) -> None:
    """Only a report exposing every counter pylcm reads yields byte counts."""
    executable = _ReportingExecutable(report=report)
    # Deliberately supply reports that violate the compiler-memory protocol.
    assert compiler_memory_bytes(compiled=executable) is None  # ty: ignore[invalid-argument-type]
