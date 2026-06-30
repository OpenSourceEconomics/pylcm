"""BQSEGM agrees with brute when the budget kink lives on a derived income var.

The tax bracket kinks on `gross_income = liquid + base_income[kind]`, so the
kink's asset-space location differs across the two `kind` slices. BQSEGM must map
the income threshold to its per-cell asset preimage and match the dense-grid
`GridSearch` value across the asset interior of each slice.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from tests.test_models import bqsegm_derived_var_toy as toy

_LIQUID = jnp.linspace(0.1, 30.0, 120)
# Stay clear of both per-kind kinks (preimages 11.0 and 14.0) and the grid edges.
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the derived-income tax toy on the shared comparison grids."""
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
    """The derived-income schedule solve equals brute in both `kind` slices."""
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
