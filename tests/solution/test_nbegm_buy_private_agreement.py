"""NBEGM agreement on a binary discrete insurance choice (F-E discrete envelope).

The discrete buy-private choice shifts cash-on-hand; NBEGM solves the continuous
subproblem per branch and takes the choice by the discrete upper envelope. The
multi-period solve must reproduce the dense `GridSearch` value, which maximises over
both the discrete choice and consumption, at every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_buy_private_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 24.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the buy-private toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=24.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_buy_private_discrete_envelope_matches_brute_every_age():
    """The discrete-envelope solve equals brute over the insurance choice, every age."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(nbegm[period]["alive"])[_INTERIOR],
            np.asarray(brute[period]["alive"])[_INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )
