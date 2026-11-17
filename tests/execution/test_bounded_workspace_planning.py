"""Tests for the policy `BoundedWidthSelector` walks on synthetic verdicts.

Each case drives the selector the way the solve loop does — propose a width
mapping, record whether it was admitted — and asserts on the proposals, the
kept selection and the exhaustion message.
"""

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal

import pytest

from _lcm.execution.core_program import ReducedAxis
from _lcm.execution.workspace_planning import BoundedWidthSelector
from lcm.execution import WidthSearch, WidthSearchPolicy
from tests.execution.test_workspace_planning import _axis

# Budget every synthetic case plans against, in bytes.
_BUDGET = 100

# Reported reservation of a mapping the budget admits, and of one it refuses.
_ADMITTED_BYTES = 10
_REFUSED_BYTES = 1000


def _admitting(*, widths: frozenset[int]) -> Callable[[Mapping[str, int]], bool]:
    """Admit exactly the named widths of axis `a`."""
    return lambda mapping: mapping["a"] in widths


def _refuse_everything(_widths: Mapping[str, int]) -> bool:
    """Refuse every mapping."""
    return False


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


def _walk(
    *,
    axes: tuple[ReducedAxis, ...],
    admitted: Callable[[Mapping[str, int]], bool],
    policy: WidthSearchPolicy,
    fixed_widths: Mapping[str, int] = MappingProxyType({}),
    hint: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, int]], BoundedWidthSelector]:
    """Run one selector to completion; return its proposals and the selector."""
    selector = BoundedWidthSelector(
        axes=axes, fixed_widths=fixed_widths, policy=policy, hint=hint
    )
    proposals: list[dict[str, int]] = []
    while (widths := selector.propose()) is not None:
        proposals.append(dict(widths))
        verdict = admitted(widths)
        reservation = _ADMITTED_BYTES if verdict else _REFUSED_BYTES
        selector.record(
            widths=widths,
            reservation_bytes=reservation,
            resident_bytes=0,
            peak_bytes=reservation,
            admitted=verdict,
        )
    return proposals, selector


def test_bounded_width_selector_evaluates_an_admitted_seed_once() -> None:
    """An admitted conservative seed ends the search after one evaluation."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=8),),
        admitted=lambda _widths: True,
        policy=_bounded(),
    )

    assert proposals == [{"a": 4}]


def test_bounded_width_selector_halves_the_widest_axis_aligned_on_refusal() -> None:
    """A refused seed is followed by the aligned half of the widest unfixed axis."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=9, alignment=3),),
        admitted=_refuse_everything,
        policy=_bounded(),
    )

    assert proposals[:2] == [{"a": 6}, {"a": 3}]


def test_bounded_width_selector_widest_seed_starts_at_rank_zero() -> None:
    """The widest-seed opt-in proposes the frontier's first-ranked candidate."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=8),),
        admitted=_refuse_everything,
        policy=_bounded(seed="widest"),
    )

    assert proposals[0] == {"a": 8}


def test_bounded_width_selector_walks_a_nonmonotone_island_in_order() -> None:
    """Shrinking past a refused width and refining into it proposes 8, 4, 2, 3."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=8),),
        admitted=_admitting(widths=frozenset({2, 3})),
        policy=_bounded(seed="widest"),
    )

    assert proposals == [{"a": 8}, {"a": 4}, {"a": 2}, {"a": 3}]


def test_bounded_width_selector_keeps_the_widest_admitted_island_width() -> None:
    """Refinement into a nonmonotone island selects its widest admitted width."""
    _, selector = _walk(
        axes=(_axis(name="a", extent=8),),
        admitted=_admitting(widths=frozenset({2, 3})),
        policy=_bounded(seed="widest"),
    )

    assert selector.selected is not None
    assert selector.selected.widths == {"a": 3}


def test_bounded_width_selector_names_the_evaluation_budget_on_exhaustion() -> None:
    """Exhausting the budget without an admission names the budget it spent."""
    _, selector = _walk(
        axes=(_axis(name="a", extent=1024),),
        admitted=_refuse_everything,
        policy=_bounded(max_evaluations=4, refinement_share=1),
    )

    assert selector.selected is None
    assert (
        "no admitted candidate within the evaluation budget of 4"
        in selector.exhaustion_message(budget=_BUDGET)
    )


def test_bounded_width_selector_says_the_frontier_was_not_exhausted() -> None:
    """The exhaustion message states that the search, not the model, ran out."""
    _, selector = _walk(
        axes=(_axis(name="a", extent=1024),),
        admitted=_refuse_everything,
        policy=_bounded(max_evaluations=4, refinement_share=1),
    )

    assert "not an exhaustive test of the frontier" in selector.exhaustion_message(
        budget=_BUDGET
    )


def test_bounded_width_selector_never_moves_a_fixed_axis() -> None:
    """A fixed axis is proposed at its declared width and at no other."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=64), _axis(name="b", extent=64)),
        admitted=_refuse_everything,
        policy=_bounded(),
        fixed_widths={"b": 4},
    )

    assert {widths["b"] for widths in proposals} == {4}


def test_bounded_width_selector_proposes_each_mapping_once() -> None:
    """No width mapping is proposed twice."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=64), _axis(name="b", extent=64)),
        admitted=_refuse_everything,
        policy=_bounded(),
    )

    assert len({tuple(sorted(widths.items())) for widths in proposals}) == len(
        proposals
    )


def test_bounded_width_selector_repeats_its_sequence_across_runs() -> None:
    """Two runs of the same case propose the same widths in the same order."""
    sequences = [
        _walk(
            axes=(_axis(name="a", extent=8),),
            admitted=_admitting(widths=frozenset({2, 3})),
            policy=_bounded(seed="widest"),
        )[0]
        for _run in range(2)
    ]

    assert sequences[0] == sequences[1]


def test_bounded_width_selector_stops_at_the_evaluation_budget() -> None:
    """The selector proposes no more mappings than the budget allows."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=1024),),
        admitted=_refuse_everything,
        policy=_bounded(seed="widest", max_evaluations=3, refinement_share=1),
    )

    assert len(proposals) == 3


def test_bounded_width_selector_logs_an_incompatible_hint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hint no axis declaration admits is reported rather than raised."""
    with caplog.at_level(logging.WARNING, logger="lcm"):
        _walk(
            axes=(_axis(name="a", extent=8, alignment=2),),
            admitted=_refuse_everything,
            policy=_bounded(),
            hint={"a": 5},
        )

    assert "hint incompatible" in caplog.text


def test_bounded_width_selector_falls_through_an_incompatible_hint() -> None:
    """An incompatible hint leaves the seed rule in charge of the first proposal."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=8, alignment=2),),
        admitted=_refuse_everything,
        policy=_bounded(),
        hint={"a": 5},
    )

    assert proposals[0] == {"a": 4}


def test_bounded_width_selector_proposes_a_compatible_hint_first() -> None:
    """A hint every axis declaration admits becomes the first proposal."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=8),),
        admitted=_refuse_everything,
        policy=_bounded(),
        hint={"a": 2},
    )

    assert proposals[0] == {"a": 2}


def test_bounded_width_selector_shrinks_from_a_refused_hint() -> None:
    """A refused hint is shrunk from, not abandoned for the seed."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=16, alignment=4),),
        admitted=_refuse_everything,
        policy=_bounded(),
        hint={"a": 12},
    )

    assert proposals[:2] == [{"a": 12}, {"a": 4}]


def test_bounded_width_selector_caps_refinement_at_its_share() -> None:
    """Refinement stops at `refinement_share` evaluations, admission included."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=64),),
        admitted=_admitting(widths=frozenset(range(1, 33))),
        policy=_bounded(seed="widest", refinement_share=2),
    )

    assert len(proposals) == 4


def test_bounded_width_selector_halves_the_axis_of_largest_declared_extent() -> None:
    """The axis halved is the one declaring the largest extent, not the widest block."""
    proposals, _ = _walk(
        axes=(_axis(name="a", extent=1024), _axis(name="b", extent=16)),
        admitted=_refuse_everything,
        policy=_bounded(),
        hint={"a": 8, "b": 16},
    )

    assert proposals[:3] == [
        {"a": 8, "b": 16},
        {"a": 4, "b": 16},
        {"a": 2, "b": 16},
    ]
