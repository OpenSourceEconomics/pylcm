"""NBEGM agreement on a cash-on-hand floor (hard-constraint schedule).

A means-tested transfer floors cash-on-hand, so it is flat below the crossing and
the value is constant where the floor binds. The multi-period solve must reproduce
the dense `GridSearch` value where the floor binds and above it, every working age
— including the recurring flat continuation, where the Euler inversion is
degenerate and the floor's optimum is read by a dense consumption search instead.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_floor_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 0.5) & (_LIQUID < 24.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the floor toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=24.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_floor_schedule_matches_brute_where_the_floor_binds_every_age():
    """The floor solve equals brute where it binds and above it, every age."""
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
