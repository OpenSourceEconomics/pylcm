"""Memory knobs are validated where they are declared, not inside `lax.map`.

Every batch/block size is `0` for "the whole axis at once" and a positive count
otherwise. A negative value is truthy, so it slips past the `or`-defaults and
reaches `lax.map(batch_size=...)` as a nonsense chunk length.
"""

from typing import Any

import pytest

from lcm.exceptions import RegimeInitializationError
from lcm.grids import LinSpacedGrid
from lcm.solvers import NBEGM, NNBEGM

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
def test_a_negative_nbegm_memory_knob_is_named_and_rejected(knob: str) -> None:
    """Each NB-EGM streaming knob names itself when given a negative size."""
    negative: dict[str, Any] = {knob: -1}
    with pytest.raises(RegimeInitializationError, match=rf"NBEGM\.{knob}"):
        NBEGM(savings_grid=SAVINGS_GRID, **negative)


def test_zero_is_the_accepted_whole_axis_setting() -> None:
    """`0` is the documented "one vectorized pass" setting, not an error."""
    assert NBEGM(savings_grid=SAVINGS_GRID, branch_batch_size=0).branch_batch_size == 0


def test_a_negative_nnbegm_outer_batch_size_names_nnbegm() -> None:
    """The nested solver's own knob is reported under its own name."""
    with pytest.raises(RegimeInitializationError, match=r"NNBEGM\.outer_batch_size"):
        NNBEGM(
            inner=NBEGM(
                savings_grid=SAVINGS_GRID,
                continuous_state="liquid",
                post_decision_function="savings",
            ),
            outer_action="investment",
            outer_post_decision="next_durable",
            outer_grid=LinSpacedGrid(start=0.0, stop=5.0, n_points=4),
            outer_batch_size=-1,
        )
