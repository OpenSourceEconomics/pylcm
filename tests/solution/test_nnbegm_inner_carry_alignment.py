"""NNBEGM refuses an inner solver whose carry rows sit off the state grid.

The bridged outer envelope folds candidates pointwise: it replaces `value` and
`marginal_utility` per candidate and reuses the keeper's `endog_grid`, which is
only correct when every candidate publishes rows at the same abscissae.
"""

import pytest

from _lcm.solution.dcegm import DCEGM
from _lcm.solution.nbegm import NBEGM
from _lcm.solution.nnbegm import _fail_if_inner_carry_rows_not_grid_aligned
from lcm.exceptions import RegimeInitializationError
from lcm.grids import LinSpacedGrid


def _savings_grid() -> LinSpacedGrid:
    return LinSpacedGrid(start=0.0, stop=10.0, n_points=5)


def test_a_grid_aligned_inner_solver_is_accepted() -> None:
    """`NBEGM` publishes carry rows on the shared liquid grid."""
    _fail_if_inner_carry_rows_not_grid_aligned(
        inner=NBEGM(savings_grid=_savings_grid())
    )


def test_an_inner_solver_with_off_grid_carry_rows_is_refused() -> None:
    """`DCEGM` publishes an endogenous grid, so the pointwise fold would mispair."""
    with pytest.raises(RegimeInitializationError, match="off the shared state grid"):
        _fail_if_inner_carry_rows_not_grid_aligned(
            inner=DCEGM(
                savings_grid=_savings_grid(),
                continuous_state="liquid",
                continuous_action="consumption",
                resources="resources",
                post_decision_function="savings",
            )
        )
