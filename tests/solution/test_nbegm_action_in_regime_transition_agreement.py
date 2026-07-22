"""NBEGM branch-specific continuation for an action feeding the regime transition.

A binary discrete choice enters the regime transition (buying insurance raises the
survival probability), so each branch weights the alive-vs-dead continuation differently
and must read its own blended next-period value. NBEGM's value function must match a
dense brute solve across the liquid interior, over the income ride nodes.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_ride_discrete_toy as toy

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
        action_in_regime_transition=True,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_nbegm_action_in_regime_transition_matches_brute() -> None:
    """`V` agrees with a 1200-point brute across the liquid interior when the
    discrete action shifts the regime transition, over the income nodes — the
    branch-specific-continuation case."""
    nbegm = _solve("nbegm", n_consumption=100)
    brute = _solve("brute", n_consumption=1200)
    period = max(p for p in brute if _ALIVE in brute[p])
    bq_v = np.asarray(nbegm[period][_ALIVE])
    brute_v = np.asarray(brute[period][_ALIVE])
    assert bq_v.shape == brute_v.shape
    (liquid_axis,) = (
        axis for axis, size in enumerate(bq_v.shape) if size == _LIQUID.shape[0]
    )
    bq_interior = np.take(bq_v, np.flatnonzero(_AWAY_FROM_KINK), axis=liquid_axis)
    brute_interior = np.take(brute_v, np.flatnonzero(_AWAY_FROM_KINK), axis=liquid_axis)
    np.testing.assert_allclose(bq_interior, brute_interior, rtol=5e-3, atol=5e-3)
