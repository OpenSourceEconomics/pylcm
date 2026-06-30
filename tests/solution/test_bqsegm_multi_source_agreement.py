"""BQSEGM agrees with brute when the budget kinks on several derived income vars.

The budget nets two taxes that bracket on two distinct monotone income concepts,
each offset differently by the ride-along `kind`, so their asset-space breakpoints
sit at different liquid points and reorder between slices. BQSEGM must merge the
breakpoints declared across the two variables into one per-cell sorted partition
and match the dense-grid `GridSearch` value across the asset interior of each
slice.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from tests.test_models import bqsegm_multi_source_toy as toy

_LIQUID = jnp.linspace(0.1, 30.0, 120)
# Stay clear of the per-kind breakpoints (preimages 11/14 for `a`, 12 for `b`)
# only at the grid edges; the interior spans across the kinks.
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the two-derived-variable budget toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_bqsegm_merges_two_derived_variable_kinks_matching_brute() -> None:
    """A budget with kinks on two derived income vars equals brute in both slices."""
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
