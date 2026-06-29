"""BQSEGM agreement on a subsidy cliff declared as a jump `piecewise_affine`.

A single jump-kind schedule threshold is the binary case the v1 step solves
exactly. The schedule path recovers the two cash-on-hand levels from the schedule
and routes them to that step, so the multi-period solve — including the recurring
jumped continuation — must reproduce the dense `GridSearch` value across the cliff
at every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_jump_schedule_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 2.0) & (_LIQUID < 22.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the jump-schedule toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=150,
        savings_max=22.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_jump_schedule_matches_brute_through_the_cliff_every_age():
    """The jump-schedule solve equals brute across the cliff at every working age."""
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(bqsegm[period]["alive"])[_INTERIOR],
            np.asarray(brute[period]["alive"])[_INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )
