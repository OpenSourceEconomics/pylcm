"""NBEGM agreement on two subsidy cliffs (recurring N-cliff jump step).

Two jump-kind schedule thresholds drive the recurring N-cliff step: each subsidy
level is a masked case whose EGM reads the continuation jump-aware at both cliffs
and competes a boundary-targeting candidate per cliff. The multi-period solve must
reproduce the dense `GridSearch` value across both cliffs at every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_two_cliff_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 2.0) & (_LIQUID < 22.0)
_NEAR_CLIFF = (np.abs(_LIQUID - 6.0) < 0.4) | (np.abs(_LIQUID - 14.0) < 0.4)
_KEEP = _INTERIOR & ~_NEAR_CLIFF


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the two-cliff toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=22.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_two_cliff_schedule_matches_brute_through_both_cliffs_every_age():
    """The recurring two-cliff solve equals brute across both cliffs at every age."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(nbegm[period]["alive"])[_KEEP],
            np.asarray(brute[period]["alive"])[_KEEP],
            atol=3e-2,
            rtol=8e-3,
            err_msg=f"period={period}",
        )
