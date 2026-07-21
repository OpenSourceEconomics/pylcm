"""NBEGM agreement with brute under a stochastic multi-target transition.

A living NBEGM regime whose lifecycle transition reaches two distinct living
regimes (plus dead) must take its continuation as the probability-weighted blend
of both living value functions. Its solved value must reproduce the dense-grid
`GridSearch` value across the asset interior, in both living regimes, in every
`kind` slice, and at every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_multi_target_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 110)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 140) -> Mapping[int, Mapping]:
    """Solve the multi-target tax toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=110,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_nbegm_matches_brute_under_stochastic_multi_target_transition():
    """The schedule solve equals brute in both living regimes and kind slices."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        for regime in ("alive_a", "alive_b"):
            if regime not in brute[period] or regime not in nbegm[period]:
                continue
            brute_v = np.asarray(brute[period][regime])
            nbegm_v = np.asarray(nbegm[period][regime])
            # Value is shaped (kind, liquid); compare the interior of each slice.
            for kind in range(brute_v.shape[0]):
                np.testing.assert_allclose(
                    nbegm_v[kind, _INTERIOR],
                    brute_v[kind, _INTERIOR],
                    atol=2e-2,
                    rtol=5e-3,
                    err_msg=f"period={period} regime={regime} kind={kind}",
                )
