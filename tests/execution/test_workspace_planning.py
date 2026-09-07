"""Tests for compile-only workspace-frontier selection."""

from collections.abc import Callable, Hashable, Mapping
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from beartype.roar import BeartypeCallHintViolation

from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from _lcm.execution.reductions import ReductionDeclaration
from _lcm.execution.workspace_planning import (
    WorkspacePlan,
    plan_workspace,
    workspace_width_candidates,
)
from lcm.exceptions import ExecutionPlanningError


class _Reduction:
    @property
    def semantic_key(self) -> Hashable:
        return "test-reduction"

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: the fold is order independent."""
        return "exact"


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
    name: str = "action_product",
    extent: int = 8,
    coordinate_names: tuple[str, ...] | None = None,
    coordinate_extents: tuple[int, ...] | None = None,
    minimum_width: int = 1,
    alignment: int = 1,
) -> ReducedAxis:
    extents = (extent,) if coordinate_extents is None else coordinate_extents
    names = (
        tuple(f"{name}_{index}" for index in range(len(extents)))
        if coordinate_names is None
        else coordinate_names
    )
    return ReducedAxis(
        name=name,
        coordinate_names=names,
        coordinate_extents=extents,
        canonical_order="c",
        reduction=cast("ReductionDeclaration", _Reduction()),
        width_keyword=f"_lcm_{name}_width",
        minimum_width=minimum_width,
        alignment=alignment,
    )


def _tiled(
    *,
    name: str = "cell",
    extent: int = 8,
    minimum_width: int = 1,
    alignment: int = 1,
) -> TiledOutputAxis:
    return TiledOutputAxis(
        name=name,
        state_names=(f"{name}_state",),
        extent=extent,
        width_keyword=f"_lcm_{name}_width",
        minimum_width=minimum_width,
        alignment=alignment,
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


@pytest.mark.parametrize(
    ("extent", "expected"),
    [(2, 1), (3, 2), (6, 4), (64, 32), (65, 64), (1000, 64)],
)
def test_bootstrap_width_is_the_largest_power_of_two_below_the_extent_capped_at_64(
    *, extent: int, expected: int
) -> None:
    """Without a budget an axis streams below its extent, never the whole product."""
    candidates = workspace_width_candidates(axes=(_axis(name="a", extent=extent),))

    assert candidates == ({"a": expected},)


def test_no_budget_compiles_bootstrap_or_fixed_widths_exactly_once() -> None:
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
            _axis(name="inner", extent=7),
        ),
        fixed_widths={"inner": 3},
        compile_candidate=compile_candidate,
    )

    assert calls == [{"outer": 4, "inner": 3}]
    assert tuple(plan.widths) == ("outer", "inner")
    assert plan.widths == {"outer": 4, "inner": 3}
    assert plan.peak_bytes is None
    assert plan.compiled is executable
    assert executable.memory_analysis_calls == 0


def test_budget_frontier_is_cartesian_and_ranked_widest_first() -> None:
    """Every combination appears once, ordered by product then lexicographically."""
    candidates = workspace_width_candidates(
        axes=(
            _axis(name="outer", extent=5),
            _axis(name="inner", extent=3),
        ),
        budget_bytes=1,
    )

    cartesian = [
        {"outer": outer, "inner": inner}
        for outer in (1, 2, 4, 5)
        for inner in (1, 2, 3)
    ]
    assert sorted(map(dict, candidates), key=repr) == sorted(cartesian, key=repr)
    assert list(map(dict, candidates)) == sorted(
        cartesian,
        key=lambda widths: (widths["outer"] * widths["inner"], tuple(widths.values())),
        reverse=True,
    )
    assert all(tuple(widths) == ("outer", "inner") for widths in candidates)


def test_power_of_two_extent_appears_only_once_in_the_frontier() -> None:
    candidates = workspace_width_candidates(axes=(_axis(extent=8),), budget_bytes=1)

    assert [widths["action_product"] for widths in candidates] == [8, 4, 2, 1]


def test_fixed_axis_is_singleton_while_other_axes_keep_their_frontier() -> None:
    candidates = workspace_width_candidates(
        axes=(
            _axis(name="fixed", extent=8),
            _axis(name="searched", extent=5),
        ),
        fixed_widths={"fixed": 3},
        budget_bytes=1,
    )

    assert list(map(dict, candidates)) == [
        {"fixed": 3, "searched": width} for width in (5, 4, 2, 1)
    ]


def test_a_fixed_width_above_the_extent_is_clamped_to_the_extent() -> None:
    """A width larger than the axis extent selects the whole axis."""
    candidates = workspace_width_candidates(
        axes=(_axis(extent=8),),
        fixed_widths={"action_product": 99},
    )

    assert candidates == ({"action_product": 8},)


def test_a_fixed_width_declared_by_another_program_is_ignored() -> None:
    """A width for an axis this program does not declare leaves its frontier alone.

    Every program of a solve is planned against the same `axis_widths` mapping, so
    a name one program declares reaches the planner for programs that do not. A
    name no program of the solve declares is refused before planning starts.
    """
    candidates = workspace_width_candidates(
        axes=(_axis(extent=8),),
        fixed_widths={"interval": 2},
    )

    assert candidates == ({"action_product": 4},)


def test_a_tiled_axis_is_planned_from_its_declared_extent() -> None:
    """A tiled output axis streams at the bootstrap width of its own extent."""
    candidates = workspace_width_candidates(axes=(_tiled(extent=8),))

    assert candidates == ({"cell": 4},)


def test_a_tiled_axis_frontier_covers_its_extent() -> None:
    """Under a budget a tiled axis offers the same ranked frontier as a reduced one."""
    candidates = workspace_width_candidates(axes=(_tiled(extent=8),), budget_bytes=1)

    assert [widths["cell"] for widths in candidates] == [8, 4, 2, 1]


def test_budgeted_planning_stops_at_the_first_feasible_candidate() -> None:
    """The full extent fits, so no narrower candidate is ever compiled."""
    compiler = _Compiler(lambda widths: _stats(widths["action_product"]))

    plan = plan_workspace(
        axes=(_axis(extent=9),),
        compile_candidate=compiler,
        budget_bytes=100,
    )

    assert [widths["action_product"] for widths, _ in compiler.calls] == [9]
    assert compiler.calls[0][1].memory_analysis_calls == 1
    assert plan.widths == {"action_product": 9}


def test_budgeted_planning_descends_the_frontier_until_one_candidate_fits() -> None:
    """Wider candidates are compiled and rejected before the widest feasible one."""
    compiler = _Compiler(lambda widths: _stats(widths["action_product"]))

    plan = plan_workspace(
        axes=(_axis(extent=9),),
        compile_candidate=compiler,
        budget_bytes=4,
    )

    assert [widths["action_product"] for widths, _ in compiler.calls] == [9, 8, 4]
    assert all(
        executable.memory_analysis_calls == 1 for _, executable in compiler.calls
    )
    assert plan.widths == {"action_product": 4}


def test_peak_equal_to_budget_is_feasible() -> None:
    compiler = _Compiler(lambda widths: _stats(widths["action_product"]))

    plan = plan_workspace(
        axes=(_axis(extent=8),),
        compile_candidate=compiler,
        budget_bytes=4,
    )

    assert plan.widths == {"action_product": 4}
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
        axes=(_axis(extent=8),),
        fixed_widths={"action_product": 3},
        compile_candidate=compiler,
        budget_bytes=70,
    )

    assert plan.widths == {"action_product": 3}
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
        axes=(_axis(),),
        fixed_widths={"action_product": 2},
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
            axes=(_axis(),),
            fixed_widths={"action_product": 2},
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
            axes=(_axis(),),
            fixed_widths={"action_product": 2},
            compile_candidate=lambda _widths: compiled,
            budget_bytes=10,
        )


def test_failing_memory_analysis_is_wrapped_with_its_cause() -> None:
    failure = RuntimeError("backend analysis failed")
    executable = _Executable(analysis=None, analysis_error=failure)

    with pytest.raises(ExecutionPlanningError, match="analysis failed") as caught:
        plan_workspace(
            axes=(_axis(),),
            fixed_widths={"action_product": 2},
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


def test_all_fixed_axes_compile_only_one_candidate_and_report_overbudget() -> None:
    compiler = _Compiler(lambda _widths: _stats(11))

    with pytest.raises(
        ExecutionPlanningError,
        match="explicitly requested workspace widths require 11 peak bytes",
    ):
        plan_workspace(
            axes=(
                _axis(name="outer", extent=8),
                _axis(name="inner", extent=7),
            ),
            fixed_widths={"outer": 3, "inner": 5},
            compile_candidate=compiler,
            budget_bytes=10,
        )

    assert [widths for widths, _ in compiler.calls] == [{"outer": 3, "inner": 5}]
    assert compiler.calls[0][1].memory_analysis_calls == 1


def test_no_feasible_candidate_is_reported_after_the_entire_frontier() -> None:
    compiler = _Compiler(lambda widths: _stats(20 - widths["action_product"]))

    with pytest.raises(
        ExecutionPlanningError,
        match="smallest reported peak is 12 bytes",
    ):
        plan_workspace(
            axes=(_axis(extent=8),),
            compile_candidate=compiler,
            budget_bytes=10,
        )

    assert [widths["action_product"] for widths, _ in compiler.calls] == [8, 4, 2, 1]
    assert all(
        executable.memory_analysis_calls == 1 for _, executable in compiler.calls
    )


def test_selected_executable_is_never_executed_or_recompiled() -> None:
    compiler = _Compiler(lambda widths: _stats(widths["action_product"]))

    plan = plan_workspace(
        axes=(_axis(extent=8),),
        compile_candidate=compiler,
        budget_bytes=4,
    )

    selected = next(
        executable
        for widths, executable in compiler.calls
        if widths == {"action_product": 4}
    )
    assert plan.compiled is selected
    assert len({id(executable) for _, executable in compiler.calls}) == 2
    assert all(not executable.executed for _, executable in compiler.calls)
    assert all(
        executable.memory_analysis_calls == 1 for _, executable in compiler.calls
    )


def test_workspace_plan_owns_an_immutable_width_snapshot() -> None:
    source = {"action_product": 2}
    plan = WorkspacePlan(widths=source, peak_bytes=4, compiled=object())

    source["action_product"] = 8

    assert plan.widths == {"action_product": 2}
    with pytest.raises(TypeError):
        cast("dict[str, int]", plan.widths)["action_product"] = 4


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
    with pytest.raises(BeartypeCallHintViolation):
        plan_workspace(
            axes=cast("tuple[ReducedAxis, ...]", (object(),)),
            compile_candidate=lambda _widths: object(),
        )


@pytest.mark.parametrize(
    ("axis", "error", "match"),
    [
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
        "mismatched-coordinate-declaration",
        "empty-coordinate-product",
        "bool-extent",
        "zero-extent",
        "negative-extent",
        "singleton-product",
    ),
)
def test_invalid_axis_extent_assumptions_are_rejected_at_the_planner_seam(
    *, axis: ReducedAxis, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        plan_workspace(
            axes=(axis,),
            compile_candidate=lambda _widths: object(),
        )


@pytest.mark.parametrize(
    ("fixed_width", "error", "match"),
    [
        (True, TypeError, "must be an integer"),
        (cast("int", 2.0), BeartypeCallHintViolation, "fixed_widths"),
        (_IntSubclass(2), TypeError, "must be an integer"),
        (0, ValueError, "must be positive"),
        (-1, ValueError, "must be positive"),
    ],
    ids=("bool", "float", "int-subclass", "zero", "negative"),
)
def test_invalid_fixed_width_is_rejected_before_compilation(
    *, fixed_width: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        plan_workspace(
            axes=(_axis(),),
            fixed_widths={"action_product": cast("int", fixed_width)},
            compile_candidate=lambda _widths: object(),
        )


def test_an_empty_fixed_width_axis_name_is_rejected_before_compilation() -> None:
    """A fixed width must name an axis."""
    with pytest.raises(TypeError, match="non-empty axis name"):
        plan_workspace(
            axes=(_axis(),),
            fixed_widths={"": 2},
            compile_candidate=lambda _widths: object(),
        )


def test_an_axis_whose_name_was_emptied_is_rejected_at_the_planner_seam() -> None:
    """The planner refuses an axis that reaches it without a usable name.

    `ReducedAxis` refuses an empty name at construction, so the state is reached by
    writing the field on the frozen instance. The planner's own check is what keeps
    a name emptied after construction — by a solver, or by a future field default —
    from silently producing an unaddressable width entry.
    """
    axis = _axis()
    object.__setattr__(axis, "name", "")

    with pytest.raises(TypeError, match="non-empty string"):
        plan_workspace(axes=(axis,), compile_candidate=lambda _widths: object())


@pytest.mark.parametrize(
    ("budget", "error", "match"),
    [
        (True, TypeError, "integer number of bytes"),
        (cast("int", 1.0), BeartypeCallHintViolation, "budget_bytes"),
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
    with pytest.raises(BeartypeCallHintViolation):
        plan_workspace(
            axes=(_axis(),),
            compile_candidate=cast("Callable[[Mapping[str, int]], object]", object()),
        )


def test_resident_bytes_make_a_candidate_that_fits_alone_infeasible() -> None:
    """The ceiling binds peak plus what the plan keeps resident on the device."""
    compiler = _Compiler(lambda widths: _stats(peak=100 * widths["actions"]))

    plan = plan_workspace(
        axes=(_axis(name="actions", extent=8),),
        compile_candidate=compiler,
        budget_bytes=800,
        resident_bytes=1,
    )

    assert plan.widths == {"actions": 4}


def test_zero_resident_bytes_leave_the_selection_unchanged() -> None:
    """Without a resident term the widest candidate that fits its peak wins."""
    compiler = _Compiler(lambda widths: _stats(peak=100 * widths["actions"]))

    plan = plan_workspace(
        axes=(_axis(name="actions", extent=8),),
        compile_candidate=compiler,
        budget_bytes=800,
        resident_bytes=0,
    )

    assert plan.widths == {"actions": 8}


def test_the_refusal_names_the_resident_term() -> None:
    """A core that fits at no width reports the resident bytes it competed with."""
    compiler = _Compiler(lambda widths: _stats(peak=100 * widths["actions"]))

    with pytest.raises(ExecutionPlanningError, match="resident"):
        plan_workspace(
            axes=(_axis(name="actions", extent=8),),
            compile_candidate=compiler,
            budget_bytes=150,
            resident_bytes=60,
        )


def test_negative_resident_bytes_are_refused() -> None:
    """Resident bytes are a count."""
    with pytest.raises(ValueError, match="resident"):
        plan_workspace(
            axes=(_axis(name="actions", extent=8),),
            compile_candidate=_Compiler(lambda _widths: _stats(peak=1)),
            budget_bytes=100,
            resident_bytes=-1,
        )


def test_a_resident_term_filling_the_budget_is_refused_naming_both_numbers() -> None:
    """A position with no budget left over cannot be served at any width."""
    compiler = _Compiler(lambda _widths: _stats(peak=0))

    with pytest.raises(
        ExecutionPlanningError,
        match=r"keeps 100 bytes resident.*100-byte budget",
    ):
        plan_workspace(
            axes=(_axis(name="actions", extent=8),),
            compile_candidate=compiler,
            budget_bytes=100,
            resident_bytes=100,
        )


def test_a_resident_term_filling_the_budget_compiles_no_candidate() -> None:
    """The refusal is reached without paying for the frontier."""
    compiler = _Compiler(lambda _widths: _stats(peak=0))

    with pytest.raises(ExecutionPlanningError):
        plan_workspace(
            axes=(_axis(name="actions", extent=8),),
            compile_candidate=compiler,
            budget_bytes=100,
            resident_bytes=100,
        )

    assert compiler.calls == []


def test_resident_bytes_are_ignored_without_a_budget() -> None:
    """An unbudgeted solve compiles its bootstrap width whatever is resident."""
    compiler = _Compiler(lambda _widths: _stats(peak=0))

    plan = plan_workspace(
        axes=(_axis(name="actions", extent=8),),
        compile_candidate=compiler,
        resident_bytes=10**9,
    )

    assert plan.widths == {"actions": 4}


def test_a_width_below_the_axis_minimum_is_never_proposed() -> None:
    """The budgeted frontier holds no width under the axis's declared floor."""
    candidates = workspace_width_candidates(
        axes=(_axis(extent=8, minimum_width=4),),
        budget_bytes=1,
    )

    assert {widths["action_product"] for widths in candidates} == {4, 8}


def test_a_bootstrap_width_below_the_axis_minimum_is_lifted_to_it() -> None:
    """An unbudgeted solve streams at the floor when the bootstrap sits below it."""
    candidates = workspace_width_candidates(axes=(_axis(extent=8, minimum_width=6),))

    assert candidates == ({"action_product": 6},)


def test_a_frontier_width_is_rounded_down_to_the_axis_alignment() -> None:
    """Every proposed width below the extent is a multiple of the alignment."""
    candidates = workspace_width_candidates(
        axes=(_axis(extent=12, minimum_width=3, alignment=3),),
        budget_bytes=1,
    )

    assert {widths["action_product"] for widths in candidates} == {3, 6, 12}


def test_the_full_extent_stays_admissible_under_an_alignment_it_violates() -> None:
    """The extent itself is always a candidate, alignment notwithstanding."""
    candidates = workspace_width_candidates(
        axes=(_axis(extent=10, alignment=4),),
        budget_bytes=1,
    )

    assert 10 in {widths["action_product"] for widths in candidates}


def test_an_alignment_never_rounds_a_width_below_the_axis_minimum() -> None:
    """Alignment shortens a width; the floor is what stops it."""
    candidates = workspace_width_candidates(
        axes=(_axis(extent=16, minimum_width=3, alignment=8),),
        budget_bytes=1,
    )

    assert {widths["action_product"] for widths in candidates} == {3, 8, 16}


def test_a_fixed_width_is_rounded_down_to_the_axis_alignment() -> None:
    """A fixed width the alignment does not divide is shortened, never widened."""
    candidates = workspace_width_candidates(
        axes=(_axis(extent=12, alignment=4),),
        fixed_widths={"action_product": 7},
    )

    assert candidates == ({"action_product": 4},)


def test_a_tiled_axis_frontier_respects_its_minimum_width() -> None:
    """A tiled output axis honours its own floor like a reduced axis does."""
    candidates = workspace_width_candidates(
        axes=(_tiled(extent=8, minimum_width=4),),
        budget_bytes=1,
    )

    assert {widths["cell"] for widths in candidates} == {4, 8}
