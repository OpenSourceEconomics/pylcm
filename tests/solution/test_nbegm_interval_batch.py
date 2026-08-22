"""Interval batching preserves the per-interval continuation read.

When a carry target's next-state law reads the current liquid state, the
continuation core evaluates the continuation DAG once per declared liquid
interval. Every iteration evaluates the same static interval window.
`NBEGM.interval_batch_size` changes only the commit stride and iteration count,
and the merged value function must not depend on that scheduling choice.
"""

import numpy as np
import pytest

from tests.test_models import nbegm_next_asset_cliff_toy as toy

_ALIVE = "alive"


def _solve_v(interval_batch_size: int) -> dict[int, np.ndarray]:
    model = toy.build_model(
        variant="nbegm",
        interval_batch_size=interval_batch_size,
    )
    solution = model.solve(params=toy.build_params(), log_level="debug")
    return {
        period: np.asarray(regimes[_ALIVE])
        for period, regimes in solution.items()
        if _ALIVE in regimes
    }


@pytest.mark.parametrize("interval_batch_size", [1, 2])
def test_interval_batch_size_leaves_the_value_function_unchanged(
    interval_batch_size: int,
) -> None:
    """`V` is identical across admitted interval commit strides."""
    default = _solve_v(0)
    requested = _solve_v(interval_batch_size)
    assert default.keys() == requested.keys()
    for period in default:
        np.testing.assert_allclose(
            requested[period], default[period], rtol=1e-12, atol=1e-12
        )
