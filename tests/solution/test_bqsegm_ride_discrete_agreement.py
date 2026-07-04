"""BQSEGM's discrete envelope composes with a ride-along co-state.

With a stochastic income node riding along a kink budget and a binary discrete
insurance choice, BQSEGM must solve the continuous subproblem per ride cell and
per discrete branch, integrate the continuation over the income nodes, and take
the discrete choice by the upper envelope. The value function must match a dense
brute solve across the liquid interior in both `buy_private`-relevant regions.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_ride_discrete_toy as toy

_ALIVE = "alive"
_TAX_EXEMPTION = 12.0
_LIQUID = np.linspace(0.1, 30.0, 120)
_AWAY_FROM_KINK = (
    (_LIQUID > 1.5) & (_LIQUID < 27.0) & (np.abs(_LIQUID - _TAX_EXEMPTION) > 0.4)
)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_bqsegm_ride_along_discrete_envelope_matches_brute() -> None:
    """`V` agrees with a 1500-point brute across the liquid interior at the
    terminal-adjacent period, over the income nodes and the discrete choice."""
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1500)
    period = max(p for p in brute if _ALIVE in brute[p])
    bq_v = np.asarray(bqsegm[period][_ALIVE])
    brute_v = np.asarray(brute[period][_ALIVE])
    # Leading axis is the income node; compare each node's liquid profile.
    for node in range(brute_v.shape[0]):
        np.testing.assert_allclose(
            bq_v[node][_AWAY_FROM_KINK],
            brute_v[node][_AWAY_FROM_KINK],
            rtol=5e-3,
            atol=5e-3,
            err_msg=f"income node {node}",
        )
