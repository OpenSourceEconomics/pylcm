"""Tests for the unbudgeted bootstrap width policy of tiled output axes.

A tiled output axis concatenates its tiles into a result that is resident at its
full extent whatever the tile width, so an unbudgeted plan may lower it wider than
a reduced axis, whose block is pure temporary.  The product of all bootstrap
widths stays bounded by a fixed block cap, so the working set remains bounded on
every backend.
"""

from collections.abc import Hashable
from typing import Literal, cast

import pytest

from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from _lcm.execution.reductions import ReductionDeclaration
from _lcm.execution.workspace_planning import (
    BOOTSTRAP_BLOCK_CAP,
    BOOTSTRAP_TILE_WIDTH_CAP,
    BOOTSTRAP_WIDTH_CAP,
    workspace_width_candidates,
)

# Shapes of the precautionary-savings ASV fixture: 500 consumption points folded
# into one action product, 2 500 output state cells tiled.
_PRECAUTIONARY_ACTION_EXTENT = 500
_PRECAUTIONARY_CELL_EXTENT = 2500


class _Reduction:
    """Minimal exact reduction declaration for a planner-only axis."""

    @property
    def semantic_key(self) -> Hashable:
        """Return the key two candidates must share to fold alike."""
        return "test-reduction"

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: the fold is order independent."""
        return "exact"


def _reduced(*, name: str = "action_product", extent: int = 8) -> ReducedAxis:
    """Build one reduced axis over a single coordinate grid of `extent` points."""
    return ReducedAxis(
        name=name,
        coordinate_names=(f"{name}_0",),
        coordinate_extents=(extent,),
        canonical_order="c",
        reduction=cast("ReductionDeclaration", _Reduction()),
        width_keyword=f"_lcm_{name}_width",
    )


def _tiled(
    *,
    name: str = "cell",
    extent: int = 8,
    minimum_width: int = 1,
    alignment: int = 1,
) -> TiledOutputAxis:
    """Build one tiled output axis under the declared width policy."""
    return TiledOutputAxis(
        name=name,
        state_names=(f"{name}_state",),
        extent=extent,
        width_keyword=f"_lcm_{name}_width",
        minimum_width=minimum_width,
        alignment=alignment,
    )


def test_the_precautionary_core_no_longer_tiles_its_cells_in_blocks_of_64() -> None:
    """An unbudgeted solve of the ASV precautionary shapes tiles cells 1024 wide."""
    (candidate,) = workspace_width_candidates(
        axes=(
            _reduced(extent=_PRECAUTIONARY_ACTION_EXTENT),
            _tiled(extent=_PRECAUTIONARY_CELL_EXTENT),
        )
    )

    assert candidate == {"action_product": 64, "cell": 1024}


@pytest.mark.parametrize(
    ("extent", "expected"),
    [
        (2, 1),
        (3, 2),
        (6, 4),
        (64, 32),
        (65, 64),
        (512, 256),
        (513, 512),
        (1025, 1024),
        (5000, 1024),
    ],
)
def test_a_tiled_axis_bootstraps_at_the_largest_power_of_two_below_1024(
    *, extent: int, expected: int
) -> None:
    """A tiled output axis streams below its extent, capped at 1024 rather than 64."""
    (candidate,) = workspace_width_candidates(axes=(_tiled(extent=extent),))

    assert candidate == {"cell": expected}


def test_a_reduced_axis_keeps_the_64_bootstrap_cap() -> None:
    """The action product is still never lowered in blocks wider than 64."""
    (candidate,) = workspace_width_candidates(axes=(_reduced(extent=5000),))

    assert candidate == {"action_product": 64}


def test_the_bootstrap_block_stays_within_the_block_cap() -> None:
    """Three wide tiled axes share one bounded block rather than multiplying caps."""
    (candidate,) = workspace_width_candidates(
        axes=(
            _tiled(name="cell", extent=5000),
            _tiled(name="savings_point", extent=5000),
            _tiled(name="euler_point", extent=5000),
        )
    )

    assert candidate == {"cell": 1024, "savings_point": 64, "euler_point": 64}


def test_no_tiled_axis_bootstraps_narrower_than_the_reduced_cap() -> None:
    """The block cap lowers a tiled axis to the 64 floor, never below it."""
    (candidate,) = workspace_width_candidates(
        axes=(
            _reduced(name="action_product", extent=5000),
            _reduced(name="stochastic_node", extent=5000),
            _tiled(name="cell", extent=5000),
        )
    )

    assert candidate == {
        "action_product": 64,
        "stochastic_node": 64,
        "cell": BOOTSTRAP_WIDTH_CAP,
    }


def test_a_pinned_tiled_axis_keeps_its_pin_and_narrows_the_others() -> None:
    """A pin is the width, and counts at its pinned value against the block cap."""
    (candidate,) = workspace_width_candidates(
        axes=(
            _tiled(name="cell", extent=5000),
            _tiled(name="euler_point", extent=5000),
        ),
        fixed_widths={"cell": 4096},
    )

    assert candidate == {"cell": 4096, "euler_point": BOOTSTRAP_WIDTH_CAP}


def test_the_budgeted_tiled_frontier_is_unchanged() -> None:
    """A declared budget still enumerates the whole ladder up to the extent."""
    candidates = workspace_width_candidates(axes=(_tiled(extent=8),), budget_bytes=1)

    assert [widths["cell"] for widths in candidates] == [8, 4, 2, 1]


def test_a_tiled_bootstrap_respects_alignment_and_the_axis_minimum() -> None:
    """The wider cap is still rounded onto the widths the axis policy admits."""
    (aligned,) = workspace_width_candidates(axes=(_tiled(extent=5000, alignment=100),))
    (lifted,) = workspace_width_candidates(
        axes=(_tiled(extent=5000, minimum_width=600),)
    )

    assert aligned == {"cell": 1000}
    assert lifted == {"cell": 1024}


def test_the_tile_cap_stays_above_the_reduced_cap() -> None:
    """The policy constants keep the tiled bootstrap monotone against the old rule."""
    assert BOOTSTRAP_TILE_WIDTH_CAP >= BOOTSTRAP_WIDTH_CAP
    assert BOOTSTRAP_BLOCK_CAP == BOOTSTRAP_WIDTH_CAP * BOOTSTRAP_TILE_WIDTH_CAP
