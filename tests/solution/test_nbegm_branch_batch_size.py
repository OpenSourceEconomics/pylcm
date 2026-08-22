"""NBEGM distinguishes requested branch sizes from admitted commit strides."""

import pytest

from _lcm.egm.fixed_width_map import (
    admitted_block_size,
    max_block_size_for_axis,
)
from _lcm.solution.nbegm import _BRANCH_MAP_GEOMETRY


def _admitted(*, n_rows: int, requested: int) -> int:
    max_block_size = max_block_size_for_axis(
        n_rows=n_rows,
        geometry=_BRANCH_MAP_GEOMETRY,
    )
    return admitted_block_size(
        requested=requested,
        max_block_size=max_block_size,
        microtile_width=_BRANCH_MAP_GEOMETRY.microtile_width,
    )


def test_request_one_is_default_equivalent_for_a_binary_branch_axis() -> None:
    """A two-branch model admits one four-row microtile for requests zero and one."""
    assert _admitted(n_rows=2, requested=0) == 4
    assert _admitted(n_rows=2, requested=1) == 4


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0, 20), (1, 4), (4, 4), (5, 8), (8, 8)],
)
def test_wide_branch_axis_exposes_distinct_admitted_strides(
    requested: int, expected: int
) -> None:
    """A 20-branch axis can schedule more than one admitted four-row multiple."""
    assert _admitted(n_rows=20, requested=requested) == expected
