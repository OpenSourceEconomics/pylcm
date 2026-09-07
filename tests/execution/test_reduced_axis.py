"""Reduced and tiled axes are the only planner-visible axis kinds."""

import pytest

from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    ReducedAxis,
    TiledOutputAxis,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION


def _reduced() -> ReducedAxis:
    return ReducedAxis(
        name="action_product",
        coordinate_names=("consumption", "labor"),
        coordinate_extents=(7, 3),
        canonical_order="c",
        reduction=HARD_MAX_REDUCTION,
        width_keyword="_lcm_action_product_width",
    )


def test_reduced_axis_extent_is_the_product_of_coordinate_extents() -> None:
    """The extent of a reduced axis is the product of its coordinate extents."""
    assert _reduced().extent == 21


def test_tiled_axis_extent_is_the_declared_extent() -> None:
    """A tiled output axis has exactly the extent it declares."""
    axis = TiledOutputAxis(name="cell", extent=40, width_keyword="_lcm_cell_width")

    assert axis.extent == 40


def test_requirements_list_reduced_axes_before_tiled_axes() -> None:
    """`axes` yields reduced axes first, then tiled axes, in declaration order."""
    tiled = TiledOutputAxis(name="cell", extent=40, width_keyword="_lcm_cell_width")
    requirements = CoreExecutionRequirements(
        reduced_axes=(_reduced(),), tiled_axes=(tiled,)
    )

    assert requirements.axis_names == ("action_product", "cell")


def test_requirements_reject_duplicate_axis_names_across_kinds() -> None:
    """One program may not declare a reduced and a tiled axis with the same name."""
    tiled = TiledOutputAxis(name="action_product", extent=4, width_keyword="_lcm_w")

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
