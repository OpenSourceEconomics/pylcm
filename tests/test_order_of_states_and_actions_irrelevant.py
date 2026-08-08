"""Declaration order of states and actions does not change what a model solves.

A regime's variables resolve into a canonical order — discrete states, then
continuous states, then actions — and within each state group by
`(not distributed, batch_size)`. That order fixes the axes of the published
value array, so reordering the declarations relabels the result; it must not
change its content.

Two orderings are covered, because they are not the same claim:

- **declaration order**, which the canonical sort absorbs entirely, so the
  published array is unchanged;
- **resolved order**, moved by `batch_size`, which genuinely permutes the axes,
  so the published array is the transpose.

The axis-order contract for the endogenous-grid solvers, whose kernels work in
a private role order of their own, lives in
`tests/solution/test_egm_continuation_axis_order.py`.
"""

import numpy as np
import pytest

from lcm import LinSpacedGrid
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.deterministic.ds_pension import get_model, get_params

_N_PERIODS = 5
_N_BOTH = 8

_PLAIN = LinSpacedGrid(start=0.0, stop=15.0, n_points=_N_BOTH)
_BATCHED = LinSpacedGrid(start=0.0, stop=15.0, n_points=_N_BOTH, batch_size=1)


def _solve(**overrides):
    model = get_model(
        n_periods=_N_PERIODS, n_liquid=_N_BOTH, n_pension=_N_BOTH, **overrides
    )
    return model.solve(params=get_params(), log_level="off")


def _periods_with(solution, regime):
    return [period for period, regimes in solution.items() if regime in regimes]


def test_declaring_the_pension_grid_explicitly_changes_nothing():
    """Passing the shared grid through the override reproduces the default model.

    The negative control for the cases below: an override naming the same grid
    the model would have built anyway must be invisible.
    """
    default = _solve()
    explicit = _solve(working_pension_grid=_PLAIN)
    periods = _periods_with(default, "working")
    assert periods
    for period in periods:
        np.testing.assert_array_equal(
            np.asarray(explicit[period]["working"]),
            np.asarray(default[period]["working"]),
        )


def test_a_batched_pension_grid_transposes_the_published_working_value():
    """`batch_size` moves the pension axis first, and only relabels the result.

    The resolved order is what the value array's axes mean, so a state sorting
    ahead of another takes the outer axis. The content is the same solve, up to
    the reduction order a partitioned maximization runs in — the default solver
    here is `GridSearch`, whose batched path sums and reduces in a different
    order, so this is a reported quantity and carries the precision's tolerance.
    The endogenous-grid kernels run one unpartitioned core either way, and their
    counterpart in `tests/solution/test_egm_continuation_axis_order.py` asserts
    exact equality.
    """
    plain = _solve(working_pension_grid=_PLAIN)
    batched = _solve(working_pension_grid=_BATCHED)
    periods = _periods_with(plain, "working")
    assert periods
    for period in periods:
        expected = np.asarray(plain[period]["working"])
        got = np.asarray(batched[period]["working"])
        assert got.shape == expected.shape[::-1]
        np.testing.assert_array_almost_equal(got, expected.T, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("regime", ["retired", "dead"])
def test_a_batched_working_pension_grid_leaves_the_other_regimes_untouched(regime):
    """Reordering one regime's axes does not disturb the regimes around it."""
    plain = _solve(working_pension_grid=_PLAIN)
    batched = _solve(working_pension_grid=_BATCHED)
    periods = _periods_with(plain, regime)
    assert periods
    for period in periods:
        np.testing.assert_array_equal(
            np.asarray(batched[period][regime]), np.asarray(plain[period][regime])
        )
