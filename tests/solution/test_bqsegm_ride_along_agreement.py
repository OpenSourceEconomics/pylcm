"""BQSEGM agreement with brute when a deterministic co-state rides along.

The continuous-schedule BQSEGM path must solve the 1-D liquid problem once per
ride-along `kind` slice, each against that slice's own budget (`base_income`
depends on `kind`) and continuation value. Its value function must reproduce the
dense-grid `GridSearch` value across the asset interior and through the bracket
kink, in every `kind` slice and at every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_ride_along_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the ride-along tax toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_bqsegm_matches_brute_in_every_ride_along_slice_every_age():
    """The schedule solve equals brute in both `kind` slices, kink and all."""
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        brute_v = np.asarray(brute[period]["alive"])
        bqsegm_v = np.asarray(bqsegm[period]["alive"])
        # Value is shaped (kind, liquid); compare the interior of each kind slice.
        for kind in range(brute_v.shape[0]):
            np.testing.assert_allclose(
                bqsegm_v[kind, _INTERIOR],
                brute_v[kind, _INTERIOR],
                atol=2e-2,
                rtol=5e-3,
                err_msg=f"period={period} kind={kind}",
            )
