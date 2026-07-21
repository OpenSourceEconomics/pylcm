"""NBEGM agreement with the brute oracle on the continuous tax-bracket toy.

The continuous-schedule NBEGM path solves a piecewise-affine, continuous budget
end to end: it reads the declared tax schedule, recovers the active affine
cash-on-hand segment per interval, and runs the multi-interval EGM step. Its value
function must reproduce the dense-grid `GridSearch` value across the asset interior
and through the bracket kink, every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_tax_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the tax toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_nbegm_schedule_matches_brute_through_the_tax_kink_every_age():
    """The continuous-schedule solve equals brute at every working age, kink and all."""
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
