"""Interval batching preserves the per-interval continuation read.

When a carry target's next-state law reads the current liquid state, the
continuation core evaluates the continuation DAG once per declared liquid
interval. `ExecutionConfig.axis_widths[INTERVAL_AXIS]` controls the width of
the streamed interval blocks. The merged value function must agree between
single-interval blocks and a block spanning this model's two intervals.
"""

import numpy as np
import pytest

from lcm import ExecutionConfig
from lcm.solvers import INTERVAL_AXIS
from tests.conftest import assert_agrees_to_ulp
from tests.test_models import nbegm_next_asset_cliff_toy as toy

# Chunking the interval axis leaves every operation and its operand order
# untouched; the two solves differ only by the vectorized kernel XLA emits per
# chunk width, a gap of a few spacings of the operands, orders of magnitude below
# a chunk-dependent reduction.
_PARTITION_ULP = 64

_ALIVE = "alive"


def _solve_v(interval_width: int) -> dict[int, np.ndarray]:
    model = toy.build_model(
        variant="nbegm",
        execution_config=ExecutionConfig(axis_widths={INTERVAL_AXIS: interval_width}),
    )
    solution = model.solve(params=toy.build_params(), log_level="debug").values
    return {
        period: np.asarray(regimes[_ALIVE])
        for period, regimes in solution.items()
        if _ALIVE in regimes
    }


@pytest.mark.parametrize("interval_width", [1, 2])
def test_interval_width_leaves_the_value_function_unchanged(
    interval_width: int,
) -> None:
    """`V` names the same values whether intervals solve vectorized or in chunks."""
    vectorized = _solve_v(2)
    chunked = _solve_v(interval_width)
    assert vectorized.keys() == chunked.keys()
    for period in vectorized:
        # Every entry of `V` is a flow utility plus a discounted continuation of
        # the array's own magnitude, so an entry near zero is a cancellation
        # whose rounding is that of its operands.
        assert_agrees_to_ulp(
            got=chunked[period],
            expected=vectorized[period],
            n_ulp=_PARTITION_ULP,
            err_msg=f"period={period}",
            operand_magnitude=float(np.abs(vectorized[period]).max()),
        )
