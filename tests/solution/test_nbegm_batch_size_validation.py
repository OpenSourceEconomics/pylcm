"""NBEGM execution widths are validated by their public planner configuration."""

import pytest

from lcm import ExecutionConfig
from lcm.solvers import BRANCH_AXIS, CELL_AXIS, INTERVAL_AXIS, STOCHASTIC_NODE_AXIS


@pytest.mark.parametrize(
    "axis", [STOCHASTIC_NODE_AXIS, INTERVAL_AXIS, BRANCH_AXIS, CELL_AXIS]
)
@pytest.mark.parametrize("width", [-1, 0])
def test_nbegm_planner_width_must_be_positive(*, axis: str, width: int) -> None:
    """Explicit zero and negative widths are refused at the execution boundary."""
    with pytest.raises(ValueError, match=axis):
        ExecutionConfig(axis_widths={axis: width})


def test_omitting_a_width_selects_the_planner_default() -> None:
    """Omission leaves widths to planning without a solver-specific sentinel."""
    assert not ExecutionConfig().axis_widths
