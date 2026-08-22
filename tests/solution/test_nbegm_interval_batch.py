"""Current NBEGM interval requests all admit the same 256-row ride stride."""

import pytest

from _lcm.egm.fixed_width_map import (
    admitted_block_size,
    max_block_size_for_axis,
)
from _lcm.solution.nbegm import _RIDE_MAP_GEOMETRY


@pytest.mark.parametrize("requested", [0, 1, 2, 64, 255, 256, 999])
def test_interval_request_is_operationally_inert(requested: int) -> None:
    """Every nonnegative request admits 256 under the current ride geometry."""
    max_block_size = max_block_size_for_axis(
        n_rows=11,
        geometry=_RIDE_MAP_GEOMETRY,
    )
    assert max_block_size == 256
    assert (
        admitted_block_size(
            requested=requested,
            max_block_size=max_block_size,
            microtile_width=_RIDE_MAP_GEOMETRY.microtile_width,
        )
        == 256
    )
