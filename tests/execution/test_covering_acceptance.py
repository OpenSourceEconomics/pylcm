"""The covered-axis seed agrees with an independent legal-set projection.

`_expected_seed` defines the seed of an axis by enumerating the widths it admits
and projecting a proposal onto them, using only integer arithmetic and none of
the planner's rounding code. The planner must reproduce it over a boundary
family of extents, floors, alignments, ceilings, pins and covering choices,
refuse exactly the cases that leave no legal width, apply the same small-extent
rule to tiled output axes, leave the budgeted frontier untouched, and walk a
refused covered seed strictly down.
"""

from collections.abc import Iterator, Mapping
from itertools import product
from typing import Any, Literal

import pytest

from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from _lcm.execution.workspace_planning import (
    BoundedWidthSelector,
    WidthDecision,
    bootstrap_widths,
    workspace_width_candidates,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import WidthSearch, WidthSearchPolicy

_REDUCED_CAP = 64
_TILED_CAP = 1024


def _powers_below(*, extent: int, cap: int) -> list[int]:
    """Enumerate the powers of two strictly below `extent` and at most `cap`."""
    return [
        2**k for k in range(extent.bit_length() + 1) if 2**k < extent and 2**k <= cap
    ]


def _legal_widths(
    *, extent: int, minimum: int, alignment: int, ceiling: int | None
) -> set[int]:
    """Enumerate every width the axis admits: its extent, or an aligned width at
    or above its floor, never above the ceiling."""
    top = extent if ceiling is None else min(ceiling, extent)
    return {
        width
        for width in range(1, top + 1)
        if width == extent or (width >= minimum and width % alignment == 0)
    }


def _expected_seed(
    *,
    extent: int,
    minimum: int = 1,
    alignment: int = 1,
    ceiling: int | None = None,
    pin: int | None = None,
    covered: bool = True,
    cap: int = _REDUCED_CAP,
) -> int:
    """Return the widest legal width at or below the proposal, or the narrowest
    legal width when none lies below it.

    The proposal is the pin when one is given, else the full extent for a covered
    extent of at most 64 that is not a power of two, else the widest power of two
    below the extent and at most `cap`.
    """
    legal = _legal_widths(
        extent=extent, minimum=minimum, alignment=alignment, ceiling=ceiling
    )
    if pin is not None:
        proposal = min(pin, extent)
    elif covered and extent <= _REDUCED_CAP and extent not in {2**k for k in range(7)}:
        proposal = extent
    else:
        proposal = max(_powers_below(extent=extent, cap=cap))
    below = {width for width in legal if width <= proposal}
    return max(below) if below else min(legal)


def _cases() -> Iterator[dict[str, Any]]:
    """Enumerate a boundary family of extents, floors, alignments, ceilings and pins."""
    for extent, alignment, half_floor, capped, pinned, covered in product(
        (2, 3, 5, 8, 10, 20, 31, 32, 63, 64, 65, 100),
        (1, 3),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
    ):
        yield {
            "extent": extent,
            "alignment": alignment,
            "minimum": (extent + 1) // 2 if half_floor else 1,
            "ceiling": extent - 1 if capped else None,
            "pin": 2 if pinned else None,
            "covered": covered,
        }


def _has_legal_width(case: Mapping[str, Any]) -> bool:
    """Report whether a case leaves the axis at least one legal width."""
    return bool(
        _legal_widths(
            extent=case["extent"],
            minimum=case["minimum"],
            alignment=case["alignment"],
            ceiling=case["ceiling"],
        )
    )


_ADMITTED = tuple(case for case in _cases() if _has_legal_width(case))
_REFUSED = tuple(case for case in _cases() if not _has_legal_width(case))


def _case_id(case: Mapping[str, Any]) -> str:
    """Name a case by its fields."""
    return "-".join(f"{key}={value}" for key, value in case.items())


def _axis(*, extent: int, minimum: int = 1, alignment: int = 1) -> ReducedAxis:
    """Build a hard-max reduced axis named `branch`."""
    return ReducedAxis(
        name="branch",
        coordinate_names=("branch_coordinate",),
        coordinate_extents=(extent,),
        canonical_order="c",
        reduction=HARD_MAX_REDUCTION,
        width_keyword="_lcm_branch_width",
        minimum_width=minimum,
        alignment=alignment,
    )


def _bootstrap_case(case: Mapping[str, Any]) -> Mapping[str, int]:
    """Seed one reference case through the planner."""
    return bootstrap_widths(
        axes=(
            _axis(
                extent=case["extent"],
                minimum=case["minimum"],
                alignment=case["alignment"],
            ),
        ),
        covered_axes=("branch",) if case["covered"] else (),
        fixed_widths={} if case["pin"] is None else {"branch": case["pin"]},
        width_ceilings={} if case["ceiling"] is None else {"branch": case["ceiling"]},
    )


def test_boundary_family_holds_admitted_and_refused_cases() -> None:
    """The boundary family has 368 cases with a legal width and 16 without."""
    assert (len(_ADMITTED), len(_REFUSED)) == (368, 16)


@pytest.mark.parametrize("case", _ADMITTED, ids=_case_id)
def test_bootstrap_widths_matches_the_legal_set_projection(
    *, case: dict[str, Any]
) -> None:
    """The planner seeds every case at the reference's legal-set projection."""
    assert dict(_bootstrap_case(case)) == {"branch": _expected_seed(**case)}


@pytest.mark.parametrize("case", _REFUSED, ids=_case_id)
def test_bootstrap_widths_refuses_a_ceiling_that_leaves_no_legal_width(
    *, case: dict[str, Any]
) -> None:
    """A ceiling below every legal width is refused at planning."""
    with pytest.raises(ExecutionPlanningError):
        _bootstrap_case(case)


@pytest.mark.parametrize("extent", [3, 5, 20, 63, 64, 65])
def test_bootstrap_widths_covers_a_small_tiled_axis_like_a_reduced_one(
    *, extent: int
) -> None:
    """A covered tiled output axis follows the same small-extent rule, under the
    tiled bootstrap cap."""
    axis = TiledOutputAxis(
        name="cell", state_names=("s",), extent=extent, width_keyword="_lcm_cell_width"
    )

    assert dict(bootstrap_widths(axes=(axis,), covered_axes=("cell",))) == {
        "cell": _expected_seed(extent=extent, cap=_TILED_CAP)
    }


@pytest.mark.parametrize("extent", [3, 5, 10, 20, 63, 64, 65])
def test_workspace_width_candidates_budgeted_frontier_ignores_covering(
    *, extent: int
) -> None:
    """Under a budget the exhaustive frontier is the same with and without
    covering."""
    axes = (_axis(extent=extent),)

    assert workspace_width_candidates(
        axes=axes, budget_bytes=1
    ) == workspace_width_candidates(axes=axes, budget_bytes=1, covered_axes=("branch",))


def _refused_walk(*, extent: int) -> tuple[list[int], WidthDecision | None]:
    """Refuse every proposal of a bounded search over one covered axis."""
    selector = BoundedWidthSelector(
        axes=(_axis(extent=extent),),
        fixed_widths={},
        covered_axes=("branch",),
        policy=WidthSearchPolicy(
            kind=WidthSearch.BOUNDED, max_evaluations=16, refinement_share=0
        ),
    )
    walk: list[int] = []
    while (widths := selector.propose()) is not None:
        walk.append(widths["branch"])
        selector.record(
            widths=widths,
            reservation_bytes=2,
            resident_bytes=0,
            peak_bytes=2,
            admitted=False,
        )
    return walk, selector.selected


@pytest.mark.parametrize("extent", [3, 5, 10, 20, 31, 63])
def test_bounded_width_selector_refused_walk_shrinks_strictly_from_the_extent(
    *, extent: int
) -> None:
    """Refusing every proposal walks from the extent down every power of two."""
    walk, _ = _refused_walk(extent=extent)

    assert walk == [
        extent,
        *reversed(_powers_below(extent=extent, cap=_REDUCED_CAP)),
    ]


@pytest.mark.parametrize("extent", [3, 5, 10, 20, 31, 63])
def test_bounded_width_selector_selects_nothing_when_every_width_is_refused(
    *, extent: int
) -> None:
    """A search whose every proposal is refused selects no width map."""
    _, selected = _refused_walk(extent=extent)

    assert selected is None


@pytest.mark.parametrize(
    ("fixed_widths", "hint", "covered_axes", "seed", "expected"),
    [
        ({"branch": 3}, None, ("branch",), "conservative", 3),
        ({}, {"branch": 7}, ("branch",), "conservative", 7),
        ({}, None, (), "widest", 20),
    ],
    ids=["pin", "hint", "widest"],
)
def test_bounded_width_selector_lets_pins_hints_and_widest_seed_take_precedence(
    *,
    fixed_widths: dict[str, int],
    hint: dict[str, int] | None,
    covered_axes: tuple[str, ...],
    seed: Literal["conservative", "widest"],
    expected: int,
) -> None:
    """A pin, a valid hint and the widest seed each fix the first proposal."""
    selector = BoundedWidthSelector(
        axes=(_axis(extent=20),),
        fixed_widths=fixed_widths,
        covered_axes=covered_axes,
        hint=hint,
        policy=WidthSearchPolicy(kind=WidthSearch.BOUNDED, seed=seed),
    )

    assert dict(selector.propose() or {}) == {"branch": expected}
