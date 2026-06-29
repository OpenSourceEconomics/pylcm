"""BQSEGM agreement on a mixed jump-and-kink schedule (the unified step).

A single schedule declaring both a jump (subsidy cliff) and a continuous kink (tax
bracket) routes to the unified step: the jump partitions the liquid axis into
continuous cases solved by coh inversion, masked across the jump with a boundary-
targeting candidate. The multi-period solve must reproduce the dense `GridSearch`
value through both the cliff and the kink at every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_mixed_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 2.0) & (_LIQUID < 24.0)
_NEAR = (np.abs(_LIQUID - 6.0) < 0.4) | (np.abs(_LIQUID - 16.0) < 0.4)
_KEEP = _INTERIOR & ~_NEAR


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the mixed toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=24.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_mixed_schedule_matches_brute_through_cliff_and_kink_every_age():
    """The mixed solve equals brute through the cliff and the kink at every age."""
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(bqsegm[period]["alive"])[_KEEP],
            np.asarray(brute[period]["alive"])[_KEEP],
            atol=3e-2,
            rtol=8e-3,
            err_msg=f"period={period}",
        )
