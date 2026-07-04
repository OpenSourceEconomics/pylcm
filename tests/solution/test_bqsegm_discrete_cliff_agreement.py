"""BQSEGM's discrete envelope composes with a cliffed budget.

With a binary discrete insurance choice and a declared jump in the budget (a
lump tax above an exemption), BQSEGM must solve the continuous subproblem
inside each branch honouring the cliff, then envelope the discrete choice. The
value function must match a dense brute solve across the liquid interior, away
from the one cell straddling the cliff (where a continuous EGM policy and a
finite grid resolve the jump by alignment alone).
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_discrete_cliff_toy as toy

_ALIVE = "alive"
_TAX_EXEMPTION = 12.0
_LIQUID = np.linspace(0.1, 30.0, 120)
_AWAY_FROM_CLIFF = (
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


def test_bqsegm_discrete_envelope_matches_brute_over_a_cliffed_budget() -> None:
    """`V` agrees with a 1500-point brute across the liquid interior in the
    terminal-adjacent period, honouring both the discrete choice and the cliff."""
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1500)
    # Terminal-adjacent: the last period `alive` is active carries the smooth
    # terminal continuation, isolating the cliff and discrete envelope in-period.
    period = max(p for p in brute if _ALIVE in brute[p])
    bq_v = np.asarray(bqsegm[period][_ALIVE])
    brute_v = np.asarray(brute[period][_ALIVE])
    np.testing.assert_allclose(
        bq_v[_AWAY_FROM_CLIFF], brute_v[_AWAY_FROM_CLIFF], rtol=2e-3, atol=2e-3
    )


def test_bqsegm_discrete_envelope_matches_brute_through_a_recurring_jump() -> None:
    """`V` agrees with a 1500-point brute at an early period, where the child
    value itself carries the discrete-plus-cliff jump, so each branch must read
    the continuation jump-aware rather than bridging across it."""
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1500)
    period = min(p for p in brute if _ALIVE in brute[p])
    bq_v = np.asarray(bqsegm[period][_ALIVE])
    brute_v = np.asarray(brute[period][_ALIVE])
    np.testing.assert_allclose(
        bq_v[_AWAY_FROM_CLIFF], brute_v[_AWAY_FROM_CLIFF], rtol=5e-3, atol=5e-3
    )
