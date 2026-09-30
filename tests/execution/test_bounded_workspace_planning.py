"""Tests for the bounded feasibility-first width search on synthetic records."""

import logging
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Literal

import pytest

from _lcm.execution.workspace_planning import (
    plan_workspace,
    plan_workspace_bounded,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import WidthSearch, WidthSearchPolicy
from tests.execution.test_compiler_allocation_reservation import memory_stats
from tests.execution.test_workspace_planning import _axis, _Compiler

# Budget every synthetic case plans against, in bytes.
_BUDGET = 100

# Reported peak of a candidate the budget admits, and of one it refuses.
_ADMITTED_PEAK = 10
_REFUSED_PEAK = 1000


def _peaks(
    *, admitted: frozenset[int]
) -> Callable[[Mapping[str, int]], SimpleNamespace]:
    """Build an analysis callback admitting exactly the named widths of axis `a`."""

    def analysis_for(widths: Mapping[str, int]) -> SimpleNamespace:
        peak = _ADMITTED_PEAK if widths["a"] in admitted else _REFUSED_PEAK
        return memory_stats(peak=peak)

    return analysis_for


def _refuse_everything(_widths: Mapping[str, int]) -> SimpleNamespace:
    """Report a peak no budget in this module admits."""
    return memory_stats(peak=_REFUSED_PEAK)


def _widths_evaluated(*, compiler: _Compiler) -> list[dict[str, int]]:
    """Return the width mapping of every compiler request, in order."""
    return [widths for widths, _ in compiler.calls]


def _bounded(
    *,
    seed: Literal["conservative", "widest"] = "conservative",
    max_evaluations: int = 24,
    refinement_share: int = 8,
) -> WidthSearchPolicy:
    """Build a bounded policy, overriding the fields a case varies."""
    return WidthSearchPolicy(
        kind=WidthSearch.BOUNDED,
        seed=seed,
        max_evaluations=max_evaluations,
        refinement_share=refinement_share,
    )


def test_plan_workspace_bounded_evaluates_an_admitted_seed_once() -> None:
    """An admitted conservative seed ends the search after one evaluation."""
    compiler = _Compiler(lambda _widths: memory_stats(peak=_ADMITTED_PEAK))

    plan_workspace_bounded(
        axes=(_axis(name="a", extent=8),),
        compile_candidate=compiler,
        budget_bytes=_BUDGET,
        policy=_bounded(),
    )

    assert _widths_evaluated(compiler=compiler) == [{"a": 4}]


def test_plan_workspace_bounded_returns_the_executable_it_admitted() -> None:
    """The plan carries the exact object compiled for the selected width."""
    compiler = _Compiler(lambda _widths: memory_stats(peak=_ADMITTED_PEAK))

    plan = plan_workspace_bounded(
        axes=(_axis(name="a", extent=8),),
        compile_candidate=compiler,
        budget_bytes=_BUDGET,
        policy=_bounded(),
    )

    assert plan.compiled is compiler.calls[0][1]


def test_plan_workspace_bounded_halves_the_widest_axis_aligned_on_refusal() -> None:
    """A refused seed is followed by the aligned half of the widest unfixed axis."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=9, alignment=3),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
        )

    assert _widths_evaluated(compiler=compiler)[:2] == [{"a": 6}, {"a": 3}]


def test_plan_workspace_bounded_widest_seed_starts_at_rank_zero() -> None:
    """The widest-seed opt-in evaluates the frontier's first-ranked candidate."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=8),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(seed="widest"),
        )

    assert _widths_evaluated(compiler=compiler)[0] == {"a": 8}


def test_plan_workspace_bounded_walks_a_nonmonotone_island_in_order() -> None:
    """Shrinking past a refused width and refining into it evaluates 8, 4, 2, 3."""
    compiler = _Compiler(_peaks(admitted=frozenset({2, 3})))

    plan_workspace_bounded(
        axes=(_axis(name="a", extent=8),),
        compile_candidate=compiler,
        budget_bytes=_BUDGET,
        policy=_bounded(seed="widest"),
    )

    assert _widths_evaluated(compiler=compiler) == [
        {"a": 8},
        {"a": 4},
        {"a": 2},
        {"a": 3},
    ]


def test_plan_workspace_bounded_keeps_the_widest_admitted_island_width() -> None:
    """Refinement into a nonmonotone island selects its widest admitted width."""
    compiler = _Compiler(_peaks(admitted=frozenset({2, 3})))

    plan = plan_workspace_bounded(
        axes=(_axis(name="a", extent=8),),
        compile_candidate=compiler,
        budget_bytes=_BUDGET,
        policy=_bounded(seed="widest"),
    )

    assert plan.widths == {"a": 3}


def test_plan_workspace_bounded_admits_a_fully_aliased_candidate() -> None:
    """Arguments and outputs that alias each other are counted once, not twice."""
    compiler = _Compiler(
        lambda _widths: memory_stats(
            peak=0,
            argument_size_in_bytes=120,
            output_size_in_bytes=120,
            alias_size_in_bytes=120,
        )
    )

    plan = plan_workspace_bounded(
        axes=(_axis(name="a", extent=8),),
        compile_candidate=compiler,
        budget_bytes=150,
        policy=_bounded(),
    )

    assert plan.reservation_bytes == 120


def test_plan_workspace_bounded_names_the_evaluation_budget_on_exhaustion() -> None:
    """Exhausting the budget without an admission names the budget it spent."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(
        ExecutionPlanningError,
        match="no admitted candidate within the evaluation budget of 4",
    ):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=1024),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(max_evaluations=4, refinement_share=1),
        )


def test_plan_workspace_bounded_says_the_frontier_was_not_exhausted() -> None:
    """The refusal states that the search, not the model, ran out."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(
        ExecutionPlanningError, match="not an exhaustive test of the frontier"
    ):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=1024),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(max_evaluations=4, refinement_share=1),
        )


def test_plan_workspace_bounded_never_moves_a_fixed_axis() -> None:
    """A fixed axis is proposed at its declared width and at no other."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=64), _axis(name="b", extent=64)),
            fixed_widths={"b": 4},
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
        )

    assert {widths["b"] for widths, _ in compiler.calls} == {4}


def test_plan_workspace_bounded_evaluates_each_mapping_once() -> None:
    """No width mapping is proposed to the compiler twice."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=64), _axis(name="b", extent=64)),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
        )

    evaluated = _widths_evaluated(compiler=compiler)
    assert len({tuple(sorted(widths.items())) for widths in evaluated}) == len(
        evaluated
    )


def test_plan_workspace_bounded_spends_an_evaluation_on_a_cached_profile() -> None:
    """A profile the caller already holds costs an evaluation and no compilation."""
    compiler = _Compiler(_refuse_everything)
    cached = {("a", 8): memory_stats(peak=_REFUSED_PEAK)}

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=8),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(seed="widest", max_evaluations=3, refinement_share=1),
            cached_analysis_for=lambda widths: cached.get(("a", widths["a"])),
        )

    assert _widths_evaluated(compiler=compiler) == [{"a": 4}, {"a": 2}]


def test_plan_workspace_bounded_repeats_its_sequence_across_runs() -> None:
    """Two runs of the same case propose the same widths in the same order."""
    sequences = []
    for _run in range(2):
        compiler = _Compiler(_peaks(admitted=frozenset({2, 3})))
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=8),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(seed="widest"),
        )
        sequences.append(_widths_evaluated(compiler=compiler))

    assert sequences[0] == sequences[1]


def test_plan_workspace_bounded_stops_at_the_evaluation_budget() -> None:
    """The compiler is asked for no more candidates than the budget allows."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=1024),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(seed="widest", max_evaluations=3, refinement_share=1),
        )

    assert len(compiler.calls) == 3


def test_plan_workspace_bounded_logs_an_incompatible_hint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hint no axis declaration admits is reported rather than raised."""
    compiler = _Compiler(_refuse_everything)

    with (
        caplog.at_level(logging.WARNING, logger="lcm"),
        pytest.raises(ExecutionPlanningError),
    ):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=8, alignment=2),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
            hint={"a": 5},
        )

    assert "hint incompatible" in caplog.text


def test_plan_workspace_bounded_falls_through_an_incompatible_hint() -> None:
    """An incompatible hint leaves the seed rule in charge of the first proposal."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=8, alignment=2),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
            hint={"a": 5},
        )

    assert _widths_evaluated(compiler=compiler)[0] == {"a": 4}


def test_plan_workspace_bounded_evaluates_a_compatible_hint_first() -> None:
    """A hint every axis declaration admits becomes the first evaluation."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=8),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
            hint={"a": 2},
        )

    assert _widths_evaluated(compiler=compiler)[0] == {"a": 2}


def test_plan_workspace_bounded_shrinks_from_a_refused_hint() -> None:
    """A refused hint is shrunk from, not abandoned for the seed."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=16, alignment=4),),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
            hint={"a": 12},
        )

    assert _widths_evaluated(compiler=compiler)[:2] == [{"a": 12}, {"a": 4}]


def test_plan_workspace_bounded_exhaustive_kind_matches_the_ranked_walk() -> None:
    """`EXHAUSTIVE` selects the width the ranked frontier walk selects."""
    axes = (_axis(name="a", extent=8),)
    reference = plan_workspace(
        axes=axes,
        compile_candidate=_Compiler(_peaks(admitted=frozenset({1, 2}))),
        budget_bytes=_BUDGET,
    )

    plan = plan_workspace_bounded(
        axes=axes,
        compile_candidate=_Compiler(_peaks(admitted=frozenset({1, 2}))),
        budget_bytes=_BUDGET,
        policy=WidthSearchPolicy(),
    )

    assert plan.widths == reference.widths


def test_plan_workspace_bounded_caps_refinement_at_its_share() -> None:
    """Refinement stops at `refinement_share` evaluations, admission included."""
    compiler = _Compiler(_peaks(admitted=frozenset(range(1, 33))))

    plan_workspace_bounded(
        axes=(_axis(name="a", extent=64),),
        compile_candidate=compiler,
        budget_bytes=_BUDGET,
        policy=_bounded(seed="widest", refinement_share=2),
    )

    assert len(compiler.calls) == 4


def test_plan_workspace_bounded_halves_the_axis_of_largest_declared_extent() -> None:
    """The axis halved is the one declaring the largest extent, not the widest block."""
    compiler = _Compiler(_refuse_everything)

    with pytest.raises(ExecutionPlanningError):
        plan_workspace_bounded(
            axes=(_axis(name="a", extent=1024), _axis(name="b", extent=16)),
            compile_candidate=compiler,
            budget_bytes=_BUDGET,
            policy=_bounded(),
            hint={"a": 8, "b": 16},
        )

    assert _widths_evaluated(compiler=compiler)[:3] == [
        {"a": 8, "b": 16},
        {"a": 4, "b": 16},
        {"a": 2, "b": 16},
    ]
