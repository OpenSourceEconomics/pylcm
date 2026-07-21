"""NBEGM agrees with brute when the budget kink lives on a derived income var.

The tax bracket kinks on `gross_income = liquid + base_income[kind]`, so the
kink's asset-space location differs across the two `kind` slices. NBEGM must map
the income threshold to its per-cell asset preimage and match the dense-grid
`GridSearch` value across the asset interior of each slice.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_models import nbegm_derived_var_toy as toy

_LIQUID = jnp.linspace(0.1, 30.0, 120)
# Stay clear of both per-kind kinks (preimages 11.0 and 14.0) and the grid edges.
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(
    variant: str, *, n_consumption: int = 120, tax_kink: float = 15.0
) -> Mapping[int, Mapping]:
    """Solve the derived-income tax toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(tax_kink=tax_kink), log_level="off")


def test_nbegm_matches_brute_in_every_ride_along_slice_every_age():
    """The derived-income schedule solve equals brute in both `kind` slices."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        brute_v = np.asarray(brute[period]["alive"])
        nbegm_v = np.asarray(nbegm[period]["alive"])
        # Value is shaped (kind, liquid); compare the interior of each kind slice.
        for kind in range(brute_v.shape[0]):
            np.testing.assert_allclose(
                nbegm_v[kind, _INTERIOR],
                brute_v[kind, _INTERIOR],
                atol=2e-2,
                rtol=5e-3,
                err_msg=f"period={period} kind={kind}",
            )


@pytest.mark.parametrize(
    "tax_kink",
    [
        # kink above the grid for `lo` (preimage 31.0 > 30.0), inside for `hi`.
        32.0,
        # kink below the grid for `hi` (preimage 0.0 < 0.1), inside for `lo`.
        4.0,
    ],
)
def test_nbegm_matches_brute_when_kink_preimage_leaves_the_grid(
    tax_kink: float,
) -> None:
    """A threshold whose asset preimage falls outside the grid is a no-op there.

    When `gross_income == tax_kink` maps to a liquid value outside `[0.1, 30]` for
    one `kind`, that slice never crosses the kink within the grid and stays on the
    single below-kink segment — matching brute, where the kink is likewise never
    reached.
    """
    nbegm = _solve("nbegm", tax_kink=tax_kink)
    brute = _solve("brute", n_consumption=1500, tax_kink=tax_kink)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        brute_v = np.asarray(brute[period]["alive"])
        nbegm_v = np.asarray(nbegm[period]["alive"])
        for kind in range(brute_v.shape[0]):
            np.testing.assert_allclose(
                nbegm_v[kind, _INTERIOR],
                brute_v[kind, _INTERIOR],
                atol=2e-2,
                rtol=5e-3,
                err_msg=f"period={period} kind={kind}",
            )
