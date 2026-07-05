"""BQSEGM carries a branch-specific continuation when an action feeds a co-state.

A binary discrete choice feeds a non-liquid co-state's law of motion (`next_streak`
reads `buy_private`), so each branch evolves to a different co-state and must read
the continuation at its own next-state coordinate — a branch-specific continuation,
not one shared across branches. BQSEGM's value function must match a dense brute
solve across the liquid interior, over the income ride nodes and the co-state grid.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_ride_discrete_toy as toy

_ALIVE = "alive"
_TAX_EXEMPTION = 12.0
_LIQUID = np.linspace(0.1, 30.0, 100)
_AWAY_FROM_KINK = (
    (_LIQUID > 1.5) & (_LIQUID < 27.0) & (np.abs(_LIQUID - _TAX_EXEMPTION) > 0.4)
)


def _solve(variant: str, *, n_consumption: int) -> Mapping[int, Mapping]:
    model = toy.build_model(
        variant=variant,
        n_liquid=100,
        liquid_max=30.0,
        n_savings=150,
        savings_max=28.0,
        n_consumption=n_consumption,
        action_in_costate=True,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_bqsegm_action_in_costate_matches_brute() -> None:
    """`V` agrees with a 1200-point brute across the liquid interior when the
    discrete action feeds the `streak` co-state, over the income nodes and the
    co-state grid — the branch-specific-continuation case."""
    bqsegm = _solve("bqsegm", n_consumption=100)
    brute = _solve("brute", n_consumption=1200)
    period = max(p for p in brute if _ALIVE in brute[p])
    bq_v = np.asarray(bqsegm[period][_ALIVE])
    brute_v = np.asarray(brute[period][_ALIVE])
    assert bq_v.shape == brute_v.shape
    # The liquid (Euler) axis is the trailing one; every leading cell (income
    # node × streak co-state) must agree across the liquid interior.
    np.testing.assert_allclose(
        bq_v[..., _AWAY_FROM_KINK],
        brute_v[..., _AWAY_FROM_KINK],
        rtol=5e-3,
        atol=5e-3,
    )
