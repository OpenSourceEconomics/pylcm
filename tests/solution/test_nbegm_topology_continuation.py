"""Acceptance target for the topology-preserving continuation read.

A NBEGM regime whose child value carries a recurring jump (next period's
budget has the same cliff) must read that child value without averaging
across the cliff: a query on the cliff's bad side must see the bad side's
continuation. The side-faithful read delivers that; full agreement with
dense brute additionally needs the parent's candidate set to contain the
save-to-cliff action, whose optimum can fall strictly between savings
nodes.

Until that off-grid candidate lands this is an xfail (strict): the
remaining residual is savings-grid resolution at the cliff preimage,
documented in `test_nbegm_jump_ride_along_recurring_is_resolution_limited`.
When the candidate lands, this test flips to XPASS and that residual test
retires.
"""

import numpy as np
import pytest

from tests.test_models import nbegm_jump_ride_along_toy as toy

_N_LIQUID = 160
_LIQUID = np.linspace(0.1, 30.0, _N_LIQUID)
_CLIFF_BY_KIND = (15.0 - 1.0, 15.0 - 4.0)
_EDGE = (_LIQUID > 1.5) & (_LIQUID < 27.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the optimal save-to-cliff action can fall between savings nodes; "
        "closing this needs an explicit candidate at each child breakpoint "
        "preimage in savings space"
    ),
)
def test_recurring_jump_period_matches_brute_with_side_faithful_read():
    """Deep-period NBEGM equals dense brute across each slice's cliff."""
    brute_model = toy.build_model(
        variant="brute",
        n_liquid=_N_LIQUID,
        liquid_max=30.0,
        n_savings=220,
        savings_max=28.0,
        n_consumption=1800,
    )
    brute = brute_model.solve(params=toy.build_params(), log_level="off")
    nbegm_model = toy.build_model(
        variant="nbegm",
        n_liquid=_N_LIQUID,
        liquid_max=30.0,
        n_savings=220,
        savings_max=28.0,
        n_consumption=160,
    )
    nbegm = nbegm_model.solve(params=toy.build_params(), log_level="off")

    period = min(p for p in brute if "alive" in brute[p])
    brute_v = np.asarray(brute[period]["alive"])
    nbegm_v = np.asarray(nbegm[period]["alive"])
    for kind in range(brute_v.shape[0]):
        away_from_cliff = np.abs(_LIQUID - _CLIFF_BY_KIND[kind]) > 0.75
        interior = _EDGE & away_from_cliff
        np.testing.assert_allclose(
            nbegm_v[kind, interior],
            brute_v[kind, interior],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period} kind={kind}",
        )
