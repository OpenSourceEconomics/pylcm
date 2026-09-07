"""Reduced and tiled axes are the only planner-visible axis kinds."""

import pytest

from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    ReducedAxis,
    TiledOutputAxis,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from lcm.exceptions import ExecutionPlanningError


def _reduced(*, minimum_width: int = 1, alignment: int = 1) -> ReducedAxis:
    return ReducedAxis(
        name="action_product",
        coordinate_names=("consumption", "labor"),
        coordinate_extents=(7, 3),
        canonical_order="c",
        reduction=HARD_MAX_REDUCTION,
        width_keyword="_lcm_action_product_width",
        minimum_width=minimum_width,
        alignment=alignment,
    )


def _tiled(
    *,
    name: str = "cell",
    extent: int = 40,
    state_names: tuple[str, ...] = ("wealth",),
    minimum_width: int = 1,
    alignment: int = 1,
) -> TiledOutputAxis:
    return TiledOutputAxis(
        name=name,
        state_names=state_names,
        extent=extent,
        width_keyword=f"_lcm_{name}_width",
        minimum_width=minimum_width,
        alignment=alignment,
    )


def test_reduced_axis_extent_is_the_product_of_coordinate_extents() -> None:
    """The extent of a reduced axis is the product of its coordinate extents."""
    assert _reduced().extent == 21


def test_tiled_axis_extent_is_the_declared_extent() -> None:
    """A tiled output axis has exactly the extent it declares."""
    assert _tiled().extent == 40


def test_requirements_list_reduced_axes_before_tiled_axes() -> None:
    """`axes` yields reduced axes first, then tiled axes, in declaration order."""
    tiled = _tiled()
    requirements = CoreExecutionRequirements(
        reduced_axes=(_reduced(),), tiled_axes=(tiled,)
    )

    assert requirements.axis_names == ("action_product", "cell")


def test_requirements_reject_duplicate_axis_names_across_kinds() -> None:
    """One program may not declare a reduced and a tiled axis with the same name."""
    tiled = _tiled(name="action_product", extent=4)

    with pytest.raises(ValueError, match="duplicate axis name 'action_product'"):
        CoreExecutionRequirements(reduced_axes=(_reduced(),), tiled_axes=(tiled,))


def test_requirements_have_no_requested_width_field() -> None:
    """Programs cannot pin a width; widths belong to the execution plan."""
    with pytest.raises(TypeError, match="requested_width"):
        ReducedAxis(
            name="action_product",
            coordinate_names=("consumption",),
            coordinate_extents=(7,),
            canonical_order="c",
            reduction=HARD_MAX_REDUCTION,
            width_keyword="_lcm_w",
            requested_width=4,  # ty: ignore[unknown-argument]
        )


def test_a_reduced_axis_defaults_to_no_width_floor_and_no_alignment() -> None:
    """An axis that states neither field admits every width down to one."""
    axis = _reduced()

    assert (axis.minimum_width, axis.alignment) == (1, 1)


def test_a_tiled_axis_names_the_states_its_tiles_cover() -> None:
    """A tiled output axis records which output states its tiles run over."""
    assert _tiled(state_names=("wealth", "health")).state_names == (
        "wealth",
        "health",
    )


def test_a_tiled_axis_defaults_to_no_width_floor_and_no_alignment() -> None:
    """A tiled axis that states neither field admits every width down to one."""
    axis = _tiled()

    assert (axis.minimum_width, axis.alignment) == (1, 1)


@pytest.mark.parametrize("minimum_width", [0, -4])
def test_a_minimum_width_below_one_is_refused(*, minimum_width: int) -> None:
    """A width floor is a positive number of cells."""
    with pytest.raises(ExecutionPlanningError, match="action_product"):
        _reduced(minimum_width=minimum_width)


@pytest.mark.parametrize("alignment", [0, -2])
def test_an_alignment_below_one_is_refused(*, alignment: int) -> None:
    """An alignment is a positive number of cells."""
    with pytest.raises(ExecutionPlanningError, match="action_product"):
        _reduced(alignment=alignment)


def test_a_minimum_width_above_the_reduced_extent_is_refused() -> None:
    """A floor no width of the canonical product could reach is refused."""
    with pytest.raises(ExecutionPlanningError, match="action_product"):
        _reduced(minimum_width=22)


def test_a_minimum_width_above_the_tiled_extent_is_refused() -> None:
    """A floor no tile of the declared extent could reach is refused."""
    with pytest.raises(ExecutionPlanningError, match="cell"):
        _tiled(extent=40, minimum_width=41)


def test_a_minimum_width_equal_to_the_extent_is_admitted() -> None:
    """A floor at the full extent leaves exactly one admissible width."""
    assert _reduced(minimum_width=21).minimum_width == 21
