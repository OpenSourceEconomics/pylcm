"""NBEGM batching controls are validated where they are declared.

Zero selects either a dense one-shot path for true streaming controls or the
largest admitted stride for fixed-window controls. Every control rejects negative
values before they can reach numerical lowering.
"""

from typing import Any

import pytest

from lcm.exceptions import RegimeInitializationError
from lcm.grids import LinSpacedGrid
from lcm.solvers import NBEGM, FiniteOuterGrid

SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=5)


@pytest.mark.parametrize(
    "knob",
    [
        "stochastic_node_batch_size",
        "envelope_segment_block_size",
        "interval_batch_size",
        "cell_block_size",
        "branch_batch_size",
    ],
)
def test_a_negative_nbegm_batching_control_is_named_and_rejected(knob: str) -> None:
    """Each NBEGM batching control names itself when given a negative size."""
    negative: dict[str, Any] = {knob: -1}
    with pytest.raises(RegimeInitializationError, match=rf"NBEGM\.{knob}"):
        NBEGM(savings_grid=SAVINGS_GRID, **negative)


def test_zero_is_the_accepted_default_batching_setting() -> None:
    """`0` selects the documented default for either batching contract."""
    assert NBEGM(savings_grid=SAVINGS_GRID, branch_batch_size=0).branch_batch_size == 0


def test_a_negative_outer_batch_size_is_rejected_by_the_search_strategy() -> None:
    """The outer chunk size is the search strategy's knob, so it validates there.

    `NNBEGM` no longer carries an `outer_batch_size` of its own to name.
    """
    with pytest.raises(RegimeInitializationError, match="batch_size"):
        FiniteOuterGrid(
            grid=LinSpacedGrid(start=0.0, stop=5.0, n_points=4), batch_size=-1
        )
