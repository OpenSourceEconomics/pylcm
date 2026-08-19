"""N-NB-EGM over a hard discrete branch on a declaration-free budget.

Both aggregations are hard maxima, so `max_j max_d = max_{(j,d)}` and the
nested solve needs no ordering convention between the outer candidate search
and the inner discrete upper envelope. The dense three-way grid search is the
agreement oracle.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest

from lcm import Model
from tests.test_models import n_nbegm_discrete_toy as toy
from tests.test_models import n_nbegm_toy as smooth

_PARAMS = {"discount_factor": 0.95, "alive": {"premium": 1.0}}
_NO_INSURANCE_PARAMS = {"discount_factor": 0.95, "alive": {"premium": 1e6}}
_SMOOTH_PARAMS = {"discount_factor": 0.95}


def _interior_gaps(
    *,
    build: Callable[..., Model],
    params: Mapping[str, Any],
    n_periods: int,
) -> dict[int, float]:
    """Largest relative nested-vs-brute gap per alive period, off the boundary.

    The poorest cells and the state-grid edges carry the two solvers'
    constrained-region and extrapolation conventions rather than the quantity
    under test, so they are excluded.
    """
    nested = build(variant="n_nbegm", n_periods=n_periods).solve(
        params=params, log_level="off"
    )
    brute = build(variant="brute", n_periods=n_periods).solve(
        params=params, log_level="off"
    )
    return {
        period: float(
            np.nanmax(
                np.abs(
                    np.asarray(nested[period]["alive"])
                    - np.asarray(brute[period]["alive"])
                )[2:-1, 1:-1]
                / np.abs(np.asarray(brute[period]["alive"]))[2:-1, 1:-1]
            )
        )
        for period in sorted(brute)
        if "alive" in brute[period]
    }


def test_a_discrete_branch_over_a_declaration_free_budget_builds() -> None:
    """A nested regime may carry a discrete action without a declared schedule.

    The outer durable is a ride-along co-state of the inner solve, so the
    branch envelope runs per outer candidate on the shared liquid axis.
    """
    toy.build_model(variant="n_nbegm", n_periods=2)


def test_the_discrete_branch_binds_somewhere_on_the_grid() -> None:
    """Insurance is bought in at least one cell at the priced premium.

    Guards the agreement tests below: at a premium no cell can afford, the
    branch is never taken and the comparison would degenerate to the smooth
    toy's.
    """
    priced = toy.build_model(variant="brute", n_periods=2).solve(
        params=_PARAMS, log_level="off"
    )
    never = toy.build_model(variant="brute", n_periods=2).solve(
        params=_NO_INSURANCE_PARAMS, log_level="off"
    )
    assert np.any(np.asarray(priced[0]["alive"]) > np.asarray(never[0]["alive"]) + 1e-8)


def test_the_nested_discrete_solve_tracks_dense_brute() -> None:
    """One alive period reading the terminal carry tracks the grid search.

    Isolates the branch envelope from the accumulation of outer candidate-set
    differences across periods: the two solvers enumerate different outer
    candidate sets, so their gap compounds with the horizon.
    """
    gaps = _interior_gaps(build=toy.build_model, params=_PARAMS, n_periods=2)
    assert max(gaps.values()) < 0.05


@pytest.mark.parametrize("n_periods", [2, 3])
def test_the_discrete_branch_costs_no_accuracy_against_the_smooth_nest(
    n_periods: int,
) -> None:
    """A discrete branch does not widen the nested solve's gap to brute.

    The same two solvers on the same grids without the branch are the control,
    so what remains after the comparison is the branch envelope's own error
    rather than the outer candidate-set mismatch both models share.
    """
    control = _interior_gaps(
        build=smooth.build_model, params=_SMOOTH_PARAMS, n_periods=n_periods
    )
    branched = _interior_gaps(
        build=toy.build_model, params=_PARAMS, n_periods=n_periods
    )
    for period, gap in branched.items():
        assert gap <= 1.15 * control[period] + 1e-3
