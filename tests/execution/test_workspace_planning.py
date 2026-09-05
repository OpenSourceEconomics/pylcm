"""Tests for compile-only workspace-frontier selection."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from types import SimpleNamespace
from typing import cast

import pytest

from _lcm.execution.core_program import ReductionSemantics, StreamableProductAxis
from _lcm.execution.workspace_planning import WorkspacePlan, plan_workspace
from lcm.exceptions import ExecutionPlanningError


class _Reduction:
    @property
    def semantic_key(self) -> Hashable:
        return "test-reduction"


class _IntSubclass(int):
    pass


class _Executable:
    def __init__(
        self, *, analysis: object, analysis_error: Exception | None = None
    ) -> None:
        self.analysis = analysis
        self.analysis_error = analysis_error
        self.memory_analysis_calls = 0
        self.executed = False

    def memory_analysis(self) -> object:
        self.memory_analysis_calls += 1
        if self.analysis_error is not None:
            raise self.analysis_error
        return self.analysis

    def __call__(self) -> None:
        self.executed = True
        raise AssertionError("workspace planning must never execute a candidate")


class _Compiler:
    def __init__(self, analysis_for: Callable[[dict[str, int]], object]) -> None:
        self.analysis_for = analysis_for
        self.calls: list[tuple[dict[str, int], _Executable]] = []

    def __call__(self, widths: Mapping[str, int]) -> _Executable:
        snapshot = dict(widths)
        executable = _Executable(analysis=self.analysis_for(snapshot))
        self.calls.append((snapshot, executable))
        return executable


def _axis(
    *,
    name: str = "action",
    extent: int = 8,
    requested_width: int | None = None,
    coordinate_names: tuple[str, ...] | None = None,
    coordinate_extents: tuple[int, ...] | None = None,
) -> StreamableProductAxis:
    extents = (extent,) if coordinate_extents is None else coordinate_extents
    names = (
        tuple(f"{name}_{index}" for index in range(len(extents)))
        if coordinate_names is None
        else coordinate_names
    )
    return StreamableProductAxis(
        name=name,
        coordinate_names=names,
        coordinate_extents=extents,
        canonical_order="c",
        reduction=cast("ReductionSemantics", _Reduction()),
        width_keyword=f"_lcm_{name}_width",
        requested_width=requested_width,
    )


def _stats(peak: object) -> SimpleNamespace:
    return SimpleNamespace(peak_memory_in_bytes=peak)


def test_no_axes_compile_the_empty_width_mapping_once_without_a_budget() -> None:
    executable = _Executable(
        analysis_error=AssertionError("memory analysis must not be called"),
        analysis=None,
    )
    calls: list[Mapping[str, int]] = []

    def compile_candidate(widths: Mapping[str, int]) -> _Executable:
        calls.append(widths)
        return executable

    plan = plan_workspace(axes=(), compile_candidate=compile_candidate)

    assert plan.widths == {}
    assert calls == [{}]
    assert plan.peak_bytes is None
    assert plan.compiled is executable
    assert executable.memory_analysis_calls == 0
    assert executable.executed is False


def test_no_axes_are_one_budgeted_candidate() -> None:
    compiler = _Compiler(lambda _widths: _stats(1))

    plan = plan_workspace(axes=(), compile_candidate=compiler, budget_bytes=1)

    assert plan.widths == {}
    assert plan.peak_bytes == 1
    assert [widths for widths, _ in compiler.calls] == [{}]
    assert compiler.calls[0][1] is plan.compiled
    assert compiler.calls[0][1].memory_analysis_calls == 1


def test_no_budget_compiles_full_or_requested_widths_exactly_once() -> None:
    executable = _Executable(
        analysis_error=AssertionError("memory analysis must not be called"),
        analysis=None,
    )
    calls: list[dict[str, int]] = []

    def compile_candidate(widths: Mapping[str, int]) -> _Executable:
        calls.append(dict(widths))
        return executable

    plan = plan_workspace(
        axes=(
            _axis(name="outer", extent=5),
            _axis(name="inner", extent=7, requested_width=3),
        ),
        compile_candidate=compile_candidate,
    )

    assert calls == [{"outer": 5, "inner": 3}]
    assert tuple(plan.widths) == ("outer", "inner")
    assert plan.widths == {"outer": 5, "inner": 3}
    assert plan.peak_bytes is None
    assert plan.compiled is executable
    assert executable.memory_analysis_calls == 0


def test_budget_frontier_is_cartesian_in_axis_declaration_order() -> None:
    compiler = _Compiler(lambda _widths: _stats(0))

    plan = plan_workspace(
        axes=(
            _axis(name="outer", extent=5),
            _axis(name="inner", extent=3),
        ),
        compile_candidate=compiler,
        budget_bytes=1,
    )

    assert [widths for widths, _ in compiler.calls] == [
        {"outer": outer, "inner": inner}
        for outer in (1, 2, 4, 5)
        for inner in (1, 2, 3)
    ]
    assert plan.widths == {"outer": 5, "inner": 3}
    assert all(tuple(widths) == ("outer", "inner") for widths, _ in compiler.calls)


def test_power_of_two_extent_appears_only_once_in_the_frontier() -> None:
    compiler = _Compiler(lambda _widths: _stats(0))

    plan_workspace(
        axes=(_axis(extent=8),),
        compile_candidate=compiler,
        budget_bytes=1,
    )

    assert [widths["action"] for widths, _ in compiler.calls] == [1, 2, 4, 8]


def test_requested_axis_is_singleton_while_other_axes_keep_their_frontier() -> None:
    compiler = _Compiler(lambda _widths: _stats(0))

    plan = plan_workspace(
        axes=(
            _axis(name="requested", extent=8, requested_width=3),
            _axis(name="searched", extent=5),
        ),
        compile_candidate=compiler,
        budget_bytes=1,
    )

    assert [widths for widths, _ in compiler.calls] == [
        {"requested": 3, "searched": width} for width in (1, 2, 4, 5)
    ]
    assert plan.widths == {"requested": 3, "searched": 5}


def test_budgeted_planning_compiles_and_analyzes_every_candidate() -> None:
    compiler = _Compiler(lambda widths: _stats(widths["action"]))

    plan = plan_workspace(
        axes=(_axis(extent=9),),
        compile_candidate=compiler,
        budget_bytes=100,
    )

    assert [widths["action"] for widths, _ in compiler.calls] == [1, 2, 4, 8, 9]
    assert all(
        executable.memory_analysis_calls == 1 for _, executable in compiler.calls
    )
    assert plan.widths == {"action": 9}


def test_peak_equal_to_budget_is_feasible() -> None:
    compiler = _Compiler(lambda widths: _stats(widths["action"]))

    plan = plan_workspace(
        axes=(_axis(extent=8),),
        compile_candidate=compiler,
        budget_bytes=4,
    )

    assert plan.widths == {"action": 4}
    assert plan.peak_bytes == 4


def test_width_product_then_declaration_order_lexicographic_widths_rank_plans() -> None:
    def analysis_for(widths: dict[str, int]) -> object:
        return _stats(0 if widths["outer"] * widths["inner"] <= 8 else 2)

    compiler = _Compiler(analysis_for)

    plan = plan_workspace(
        axes=(
            _axis(name="outer", extent=4),
            _axis(name="inner", extent=4),
        ),
        compile_candidate=compiler,
        budget_bytes=1,
    )

    assert plan.widths == {"outer": 4, "inner": 2}
    assert plan.peak_bytes == 0


def test_per_device_peaks_are_maximized_not_summed() -> None:
    compiler = _Compiler(lambda _widths: [_stats(60), _stats(70)])

    plan = plan_workspace(
        axes=(_axis(extent=8, requested_width=3),),
        compile_candidate=compiler,
        budget_bytes=70,
    )

    assert plan.widths == {"action": 3}
    assert plan.peak_bytes == 70


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        (_stats(7), 7),
        ({"peak_memory_in_bytes": 8}, 8),
        (_stats([3, 9]), 9),
        ({"peak_memory_in_bytes": {"device-0": 7, "device-1": 4}}, 7),
        ([_stats(3), {"peak_memory_in_bytes": 9}], 9),
        (
            {
                "device-0": _stats(3),
                "device-1": {"peak_memory_in_bytes": 9},
            },
            9,
        ),
    ],
    ids=(
        "attribute-record",
        "mapping-record",
        "attribute-per-device-field",
        "mapping-per-device-field",
        "per-device-sequence",
        "per-device-mapping",
    ),
)
def test_strict_peak_normalization_accepts_jax_style_records(
    *, analysis: object, expected: int
) -> None:
    compiler = _Compiler(lambda _widths: analysis)

    plan = plan_workspace(
        axes=(_axis(requested_width=2),),
        compile_candidate=compiler,
        budget_bytes=expected,
    )

    assert plan.peak_bytes == expected


@pytest.mark.parametrize(
    "analysis",
    [
        None,
        7,
        [7, 8],
        {},
        [],
        SimpleNamespace(temp_size_in_bytes=7),
        {"temp_size_in_bytes": 7},
        _stats(None),
        _stats(peak=True),
        _stats(1.0),
        _stats(-1),
        _stats([]),
        [_stats(3), SimpleNamespace(temp_size_in_bytes=4)],
        {
            "device-0": _stats(3),
            "device-1": {"temp_size_in_bytes": 4},
        },
    ],
    ids=(
        "none",
        "bare-int",
        "bare-int-sequence",
        "empty-mapping",
        "empty-sequence",
        "missing-attribute",
        "missing-mapping-key",
        "none-peak",
        "bool-peak",
        "float-peak",
        "negative-peak",
        "empty-peak-collection",
        "malformed-device-sequence",
        "malformed-device-mapping",
    ),
)
def test_malformed_memory_analysis_fails_closed(analysis: object) -> None:
    compiler = _Compiler(lambda _widths: analysis)

    with pytest.raises(ExecutionPlanningError, match="no valid per-device peak"):
        plan_workspace(
            axes=(_axis(requested_width=2),),
            compile_candidate=compiler,
            budget_bytes=10,
        )


@pytest.mark.parametrize(
    "compiled",
    [object(), SimpleNamespace(memory_analysis=7)],
    ids=("missing-method", "non-callable-method"),
)
def test_missing_memory_analysis_fails_closed(compiled: object) -> None:
    with pytest.raises(ExecutionPlanningError, match="analysis is unavailable"):
        plan_workspace(
            axes=(_axis(requested_width=2),),
            compile_candidate=lambda _widths: compiled,
            budget_bytes=10,
        )


def test_failing_memory_analysis_is_wrapped_with_its_cause() -> None:
    failure = RuntimeError("backend analysis failed")
    executable = _Executable(analysis=None, analysis_error=failure)

    with pytest.raises(ExecutionPlanningError, match="analysis failed") as caught:
        plan_workspace(
            axes=(_axis(requested_width=2),),
            compile_candidate=lambda _widths: executable,
            budget_bytes=10,
        )

    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("budget_bytes", [None, 10])
def test_compile_exceptions_propagate_unchanged(budget_bytes: int | None) -> None:
    failure = RuntimeError("compiler refused the candidate")

    def fail_compile(_widths: Mapping[str, int]) -> object:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=fail_compile,
            budget_bytes=budget_bytes,
        )

    assert caught.value is failure


def test_all_requested_axes_compile_only_one_candidate_and_report_overbudget() -> None:
    compiler = _Compiler(lambda _widths: _stats(11))

    with pytest.raises(
        ExecutionPlanningError,
        match="explicitly requested workspace widths require 11 peak bytes",
    ):
        plan_workspace(
            axes=(
                _axis(name="outer", extent=8, requested_width=3),
                _axis(name="inner", extent=7, requested_width=5),
            ),
            compile_candidate=compiler,
            budget_bytes=10,
        )

    assert [widths for widths, _ in compiler.calls] == [{"outer": 3, "inner": 5}]
    assert compiler.calls[0][1].memory_analysis_calls == 1


def test_no_feasible_candidate_is_reported_after_the_entire_frontier() -> None:
    compiler = _Compiler(lambda widths: _stats(20 - widths["action"]))

    with pytest.raises(
        ExecutionPlanningError,
        match="smallest reported peak is 12 bytes",
    ):
        plan_workspace(
            axes=(_axis(extent=8),),
            compile_candidate=compiler,
            budget_bytes=10,
        )

    assert [widths["action"] for widths, _ in compiler.calls] == [1, 2, 4, 8]
    assert all(
        executable.memory_analysis_calls == 1 for _, executable in compiler.calls
    )


def test_selected_executable_is_never_executed_or_recompiled() -> None:
    compiler = _Compiler(lambda widths: _stats(widths["action"]))

    plan = plan_workspace(
        axes=(_axis(extent=8),),
        compile_candidate=compiler,
        budget_bytes=4,
    )

    selected = next(
        executable for widths, executable in compiler.calls if widths == {"action": 4}
    )
    assert plan.compiled is selected
    assert len({id(executable) for _, executable in compiler.calls}) == 4
    assert all(not executable.executed for _, executable in compiler.calls)
    assert all(
        executable.memory_analysis_calls == 1 for _, executable in compiler.calls
    )


def test_workspace_plan_owns_an_immutable_width_snapshot() -> None:
    source = {"action": 2}
    plan = WorkspacePlan(widths=source, peak_bytes=4, compiled=object())

    source["action"] = 8

    assert plan.widths == {"action": 2}
    with pytest.raises(TypeError):
        cast("dict[str, int]", plan.widths)["action"] = 4


def test_duplicate_axis_names_are_rejected_before_compilation() -> None:
    compile_calls = 0

    def compile_candidate(_widths: Mapping[str, int]) -> object:
        nonlocal compile_calls
        compile_calls += 1
        return object()

    with pytest.raises(ValueError, match="duplicate names"):
        plan_workspace(
            axes=(_axis(name="same"), _axis(name="same")),
            compile_candidate=compile_candidate,
        )

    assert compile_calls == 0


def test_non_axis_declaration_is_rejected_before_compilation() -> None:
    with pytest.raises(TypeError, match="StreamableProductAxis"):
        plan_workspace(
            axes=cast("tuple[StreamableProductAxis, ...]", (object(),)),
            compile_candidate=lambda _widths: object(),
        )


@pytest.mark.parametrize(
    ("axis", "error", "match"),
    [
        (_axis(name=""), TypeError, "non-empty string"),
        (
            _axis(
                coordinate_names=("only",),
                coordinate_extents=(2, 3),
            ),
            ValueError,
            "same length",
        ),
        (
            _axis(coordinate_extents=(), coordinate_names=()),
            ValueError,
            "declare coordinate extents",
        ),
        (
            _axis(coordinate_extents=cast("tuple[int, ...]", (True,))),
            TypeError,
            "extents must be integers",
        ),
        (
            _axis(coordinate_extents=cast("tuple[int, ...]", (2.0,))),
            TypeError,
            "extents must be integers",
        ),
        (
            _axis(coordinate_extents=(0,)),
            ValueError,
            "extents must be positive",
        ),
        (
            _axis(coordinate_extents=(-2,)),
            ValueError,
            "extents must be positive",
        ),
        (_axis(extent=1), ValueError, "extent greater than one"),
    ],
    ids=(
        "empty-name",
        "mismatched-coordinate-declaration",
        "empty-coordinate-product",
        "bool-extent",
        "float-extent",
        "zero-extent",
        "negative-extent",
        "singleton-product",
    ),
)
def test_invalid_axis_extent_assumptions_are_rejected_at_the_planner_seam(
    *, axis: StreamableProductAxis, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        plan_workspace(
            axes=(axis,),
            compile_candidate=lambda _widths: object(),
        )


@pytest.mark.parametrize(
    ("requested_width", "error", "match"),
    [
        (True, TypeError, "must be an integer"),
        (cast("int", 2.0), TypeError, "must be an integer"),
        (_IntSubclass(2), TypeError, "must be an integer"),
        (0, ValueError, "must be positive"),
        (-1, ValueError, "must be positive"),
        (9, ValueError, "exceeds its product extent 8"),
    ],
    ids=("bool", "float", "int-subclass", "zero", "negative", "above-extent"),
)
def test_invalid_requested_width_is_rejected_before_compilation(
    *, requested_width: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        plan_workspace(
            axes=(_axis(requested_width=cast("int", requested_width)),),
            compile_candidate=lambda _widths: object(),
        )


@pytest.mark.parametrize(
    ("budget", "error", "match"),
    [
        (True, TypeError, "integer number of bytes"),
        (cast("int", 1.0), TypeError, "integer number of bytes"),
        (_IntSubclass(1), TypeError, "integer number of bytes"),
        (0, ValueError, "positive"),
        (-1, ValueError, "positive"),
    ],
    ids=("bool", "float", "int-subclass", "zero", "negative"),
)
def test_invalid_budget_is_rejected_before_compilation(
    *, budget: object, error: type[Exception], match: str
) -> None:
    compile_calls = 0

    def compile_candidate(_widths: Mapping[str, int]) -> object:
        nonlocal compile_calls
        compile_calls += 1
        return object()

    with pytest.raises(error, match=match):
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=compile_candidate,
            budget_bytes=cast("int", budget),
        )

    assert compile_calls == 0


def test_non_callable_compiler_is_rejected() -> None:
    with pytest.raises(TypeError, match="compiler must be callable"):
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=cast("Callable[[Mapping[str, int]], object]", object()),
        )
