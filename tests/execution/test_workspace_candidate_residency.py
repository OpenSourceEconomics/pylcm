"""A candidate's actual external residency participates in width admission."""

import dataclasses
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from _lcm.execution.core_program import TiledOutputAxis
from _lcm.execution.workspace_planning import plan_workspace
from lcm.exceptions import ExecutionPlanningError
from tests.execution.test_compiler_allocation_reservation import memory_stats


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Candidate:
    width: int
    raw_peak: int
    resident: int

    def memory_analysis(self) -> SimpleNamespace:
        return memory_stats(peak=self.raw_peak)

    def __call__(self) -> None:
        raise AssertionError("Planning must not execute a candidate")


@dataclasses.dataclass(kw_only=True)
class _Compiler:
    candidates: Mapping[int, _Candidate]
    calls: list[_Candidate] = dataclasses.field(default_factory=list)

    def __call__(self, widths: Mapping[str, int]) -> _Candidate:
        candidate = self.candidates[widths["cell"]]
        self.calls.append(candidate)
        return candidate


def _axis() -> TiledOutputAxis:
    return TiledOutputAxis(
        name="cell", state_names=("state",), extent=8, width_keyword="cell_width"
    )


def test_candidate_residency_changes_the_winner_without_recompiling() -> None:
    """The widest raw peak fits, but its own retained inputs make it infeasible."""
    widest = _Candidate(width=8, raw_peak=20, resident=90)
    winner = _Candidate(width=4, raw_peak=50, resident=40)
    compiler = _Compiler(candidates={8: widest, 4: winner})
    measured: list[_Candidate] = []

    def residency(candidate: _Candidate) -> int:
        measured.append(candidate)
        return candidate.resident

    plan = plan_workspace(
        axes=(_axis(),),
        compile_candidate=compiler,
        budget_bytes=100,
        resident_bytes=10,
        resident_bytes_for=residency,
    )
    assert compiler.calls == [widest, winner]
    assert measured == [widest, winner]
    assert plan.compiled is winner
    assert dict(plan.widths) == {"cell": 4}
    assert plan.peak_bytes == 50


def test_unbudgeted_plan_does_not_request_candidate_residency() -> None:
    candidate = _Candidate(width=4, raw_peak=20, resident=10)
    compiler = _Compiler(candidates={4: candidate})

    def unavailable(_candidate: _Candidate) -> int:
        raise AssertionError("Unbudgeted execution must not inspect live residency")

    plan = plan_workspace(
        axes=(_axis(),),
        compile_candidate=compiler,
        resident_bytes_for=unavailable,
    )
    assert plan.compiled is candidate
    assert plan.peak_bytes is None
    assert compiler.calls == [candidate]


@pytest.mark.parametrize("invalid", [True, -1, 1.5, None])
def test_candidate_residency_requires_nonnegative_integer_bytes(invalid: Any) -> None:
    candidate = _Candidate(width=8, raw_peak=1, resident=0)
    compiler = _Compiler(candidates={8: candidate})
    with pytest.raises(ExecutionPlanningError, match="residen"):
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=compiler,
            budget_bytes=100,
            resident_bytes_for=lambda _candidate: invalid,
        )
    assert compiler.calls == [candidate]


def test_candidate_residency_cannot_undercut_the_known_lower_bound() -> None:
    candidate = _Candidate(width=8, raw_peak=1, resident=10)
    compiler = _Compiler(candidates={8: candidate})
    with pytest.raises(ExecutionPlanningError, match="residen"):
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=compiler,
            budget_bytes=100,
            resident_bytes=20,
            resident_bytes_for=lambda item: item.resident,
        )
    assert compiler.calls == [candidate]


def test_unavailable_candidate_residency_refuses_with_width_context() -> None:
    """An inventory failure cannot be treated as zero resident bytes."""
    candidate = _Candidate(width=8, raw_peak=1, resident=10)
    compiler = _Compiler(candidates={8: candidate})
    original = LookupError("owner metadata unavailable")

    def unavailable(_candidate: _Candidate) -> int:
        raise original

    with pytest.raises(ExecutionPlanningError, match="residen") as error:
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=compiler,
            budget_bytes=100,
            resident_bytes_for=unavailable,
        )
    assert error.value.__cause__ is original
    assert "'cell': 8" in str(error.value)
    assert compiler.calls == [candidate]


def test_failure_reports_the_smallest_total_not_the_smallest_raw_peak() -> None:
    """Different candidates minimize raw peak and total occupied memory."""
    candidates = {
        8: _Candidate(width=8, raw_peak=10, resident=120),
        4: _Candidate(width=4, raw_peak=4, resident=200),
        2: _Candidate(width=2, raw_peak=20, resident=150),
        1: _Candidate(width=1, raw_peak=25, resident=120),
    }
    compiler = _Compiler(candidates=candidates)
    with pytest.raises(
        ExecutionPlanningError, match="No workspace-width candidate"
    ) as error:
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=compiler,
            budget_bytes=100,
            resident_bytes=10,
            resident_bytes_for=lambda item: item.resident,
        )
    assert compiler.calls == list(candidates.values())
    assert "130" in str(error.value)
    assert "100" in str(error.value)


def test_fixed_width_failure_reports_its_actual_candidate_residency() -> None:
    candidate = _Candidate(width=4, raw_peak=30, resident=90)
    compiler = _Compiler(candidates={4: candidate})
    with pytest.raises(ExecutionPlanningError, match="explicitly requested") as error:
        plan_workspace(
            axes=(_axis(),),
            fixed_widths={"cell": 4},
            compile_candidate=compiler,
            budget_bytes=100,
            resident_bytes=10,
            resident_bytes_for=lambda item: item.resident,
        )
    assert compiler.calls == [candidate]
    assert "90" in str(error.value)
    assert "100" in str(error.value)
