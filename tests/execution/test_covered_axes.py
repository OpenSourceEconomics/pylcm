"""Covering a small reduced axis at its whole extent instead of a power of two.

An axis named in `ExecutionConfig.covered_axes` whose extent the conservative
power-of-two width does not divide is seeded at the full extent, so the map over
it needs no remainder program. Fixed widths and ceilings still bind, and a
bounded search refused at the covered width falls back to the power-of-two width.
"""

from collections.abc import Hashable, Mapping
from typing import Any, Literal, cast

import pytest
from beartype.roar import BeartypeCallHintViolation

from _lcm.execution.core_program import ReducedAxis
from _lcm.execution.reductions import ReductionDeclaration
from _lcm.execution.workspace_planning import (
    BoundedWidthSelector,
    bootstrap_widths,
)
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import WidthSearch, WidthSearchPolicy
from tests.solution.test_compilation_identity import _capture_lowering_keys, _model
from tests.test_models.deterministic.regression import get_params


class _Reduction:
    @property
    def semantic_key(self) -> Hashable:
        return "test-reduction"

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: the fold is order independent."""
        return "exact"


def _axis(*, extent: int, name: str = "cell") -> ReducedAxis:
    return ReducedAxis(
        name=name,
        coordinate_names=(f"{name}_0",),
        coordinate_extents=(extent,),
        canonical_order="c",
        reduction=cast("ReductionDeclaration", _Reduction()),
        width_keyword=f"_lcm_{name}_width",
    )


@pytest.mark.parametrize(
    ("extent", "expected"),
    [(5, 5), (10, 10), (20, 20), (8, 4), (64, 32), (100, 64)],
)
def test_bootstrap_widths_covers_a_named_axis_the_power_of_two_leaves_a_remainder_on(
    *, extent: int, expected: int
) -> None:
    """A covered axis seeds at its extent exactly when the power of two leaves a
    remainder and the extent is within the bootstrap cap."""
    widths = bootstrap_widths(axes=(_axis(extent=extent),), covered_axes=("cell",))

    assert widths["cell"] == expected


def test_bootstrap_widths_lets_a_ceiling_below_the_extent_win() -> None:
    """A ceiling below the extent bounds a covered axis at the ceiling."""
    widths = bootstrap_widths(
        axes=(_axis(extent=10),),
        width_ceilings={"cell": 6},
        covered_axes=("cell",),
    )

    assert widths["cell"] == 6


def test_bootstrap_widths_lets_a_fixed_width_win() -> None:
    """A fixed width binds a covered axis at that width."""
    widths = bootstrap_widths(
        axes=(_axis(extent=10),), fixed_widths={"cell": 2}, covered_axes=("cell",)
    )

    assert widths["cell"] == 2


def test_bootstrap_widths_leaves_an_unnamed_axis_at_its_power_of_two() -> None:
    """Only the axes `covered_axes` names are covered."""
    widths = bootstrap_widths(
        axes=(_axis(extent=10), _axis(extent=10, name="other")),
        covered_axes=("cell",),
    )

    assert dict(widths) == {"cell": 10, "other": 8}


def test_bootstrap_widths_without_covered_axes_keeps_the_power_of_two_seed() -> None:
    """An empty `covered_axes` seeds every extent at its largest power of two below."""
    extents = (2, 3, 5, 8, 10, 20, 63, 64, 65, 100)
    widths = tuple(
        bootstrap_widths(axes=(_axis(extent=extent),), covered_axes=())["cell"]
        for extent in extents
    )

    assert widths == (1, 2, 4, 4, 8, 16, 32, 32, 64, 64)


def _refusing_walk(*, extent: int) -> list[int]:
    """Refuse every proposal of a bounded search over one covered axis."""
    selector = BoundedWidthSelector(
        axes=(_axis(extent=extent),),
        fixed_widths={},
        covered_axes=("cell",),
        policy=WidthSearchPolicy(kind=WidthSearch.BOUNDED, max_evaluations=10),
    )
    walk: list[int] = []
    while (widths := selector.propose()) is not None:
        walk.append(widths["cell"])
        selector.record(
            widths=widths,
            reservation_bytes=1,
            resident_bytes=0,
            peak_bytes=1,
            admitted=False,
        )
    return walk


@pytest.mark.parametrize(
    ("extent", "expected"), [(5, [5, 4, 2, 1]), (10, [10, 8, 4, 2, 1])]
)
def test_bounded_width_selector_falls_back_to_the_power_of_two_after_the_cover(
    *, extent: int, expected: list[int]
) -> None:
    """A refused covered width shrinks to the power-of-two seed, then halves."""
    assert _refusing_walk(extent=extent) == expected


@pytest.mark.parametrize(
    ("covered_axes", "error", "match"),
    [
        (["cell"], BeartypeCallHintViolation, "covered_axes"),
        ((1,), BeartypeCallHintViolation, "covered_axes"),
        (("",), ValueError, r"ExecutionConfig\.covered_axes entries"),
        (("cell", "cell"), ValueError, r"ExecutionConfig\.covered_axes names"),
    ],
)
def test_execution_config_refuses_malformed_covered_axes(
    *, covered_axes: Any, error: type[Exception], match: str
) -> None:
    """`covered_axes` must be a tuple of distinct non-empty axis names."""
    with pytest.raises(error, match=match):
        ExecutionConfig(covered_axes=covered_axes)


def test_model_refuses_covered_axes_naming_an_undeclared_axis() -> None:
    """A covered axis no core program declares is refused at model build."""
    with pytest.raises(ExecutionPlanningError, match="covered_axes"):
        _model(execution_config=ExecutionConfig(covered_axes=("no_such_axis",)))


def _keys_by_triple(
    *, covered_axes: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> dict[Hashable, tuple[Mapping[str, int], set[Hashable]]]:
    """Solve the identity toy and group its lowering keys by core."""
    captured = _capture_lowering_keys(monkeypatch=monkeypatch)
    _model(execution_config=ExecutionConfig(covered_axes=covered_axes)).solve(
        params=get_params(n_periods=3), log_level="off"
    )
    grouped: dict[Hashable, tuple[Mapping[str, int], set[Hashable]]] = {}
    for candidate, key in captured[0].items():
        triple, widths = cast("tuple[Hashable, Any]", candidate)
        grouped.setdefault(triple, (dict(widths), set()))[1].add(key)
    return grouped


def test_covered_axes_separate_lowering_keys_only_of_cores_declaring_the_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A covered solve relowers the cores declaring the covered axis and reuses the
    keys of every other core."""
    plain = _keys_by_triple(covered_axes=(), monkeypatch=monkeypatch)
    covered = _keys_by_triple(covered_axes=("cell",), monkeypatch=monkeypatch)
    declaring = {triple for triple, (widths, _) in plain.items() if "cell" in widths}
    shared = {
        triple
        for triple in plain
        if plain[triple][1] & covered[triple][1] == plain[triple][1]
    }
    disjoint = {triple for triple in plain if not plain[triple][1] & covered[triple][1]}

    assert (len(declaring) > 0, disjoint, shared) == (
        True,
        declaring,
        set(plain) - declaring,
    )
