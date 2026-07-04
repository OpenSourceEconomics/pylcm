"""Interval batching preserves the per-interval continuation read.

When a carry target's next-state law reads the current liquid state, the
continuation core evaluates the continuation DAG once per declared liquid
interval. `BQSEGM.interval_batch_size` controls how those evaluations run —
all intervals in one vectorized pass (`0`), or in sequential chunks of the
given size — and the merged value function must not depend on the choice.
"""

import numpy as np
import pytest

from tests.test_models import bqsegm_next_asset_cliff_toy as toy

_ALIVE = "alive"


def _solve_v(interval_batch_size: int) -> dict[int, np.ndarray]:
    model = toy.build_model(
        variant="bqsegm",
        interval_batch_size=interval_batch_size,
    )
    solution = model.solve(params=toy.build_params(), log_level="off")
    return {
        period: np.asarray(regimes[_ALIVE])
        for period, regimes in solution.items()
        if _ALIVE in regimes
    }


@pytest.mark.parametrize("interval_batch_size", [1, 2])
def test_interval_batch_size_leaves_the_value_function_unchanged(
    interval_batch_size: int,
) -> None:
    """`V` is identical whether intervals solve vectorized or in chunks."""
    vectorized = _solve_v(0)
    chunked = _solve_v(interval_batch_size)
    assert vectorized.keys() == chunked.keys()
    for period in vectorized:
        np.testing.assert_allclose(
            chunked[period], vectorized[period], rtol=1e-12, atol=1e-12
        )
