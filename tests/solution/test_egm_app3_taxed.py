"""DC-EGM/FUES for the DS-2026 App.3 with-tax model matches VFI.

Application 3's Table 5 adds a piecewise-linear capital-income tax to the
discrete-housing model. The tax `T(a) = B + tau_a*(a - a0)` carries three level
discontinuities — up at a=3.87 and a=15, and down by about 0.14 at a=6.97 where
the subsidy bracket resets the offset — plus several rate kinks, so the budget is
non-monotone in assets — which is exactly why Table 5 compares only FUES (which
resolves the jumps/kinks via the upper envelope) and VFI (grid search), dropping
MSS/LTM.

The oracle is the grid-search (VFI) twin of the same with-tax model. On the
asset interior the FUES and VFI value functions agree up to grid resolution, with
only a few cells at the tax-bracket boundaries (the jump/kink nodes) disagreeing
more — the standard DC-EGM unstable-node pattern.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import X64_ENABLED
from tests.test_models.ds_app3_discrete_housing import (
    build_model,
    build_params,
    piecewise_capital_income_tax,
)

# The bracket-schedule comparison is float-eps-limited at the active precision.
_TAX_ATOL = 1e-9 if X64_ENABLED else 1e-5

# Construction-scale grids: a single local solve stays fast. The asset interior
# excludes the constrained low-wealth nodes and the top edge-clamp nodes.
N_ASSETS = 60
N_WAGE_NODES = 3
N_PERIODS = 5
N_CONSUMPTION = 80
N_LOW_NODES = 14
N_HIGH_NODES = 14

MEAN_TOL = 0.02
CELL_TOL = 0.05
MIN_FRACTION_WITHIN = 0.95


@pytest.mark.parametrize(
    ("assets", "expected"),
    [
        (1.0, 0.0),  # tax-free first bracket
        (2.35, 0.0114 * (2.35 - 2.20)),  # second bracket, B=0
        (4.0, 0.11412 + 0.024 * (4.0 - 3.87)),  # post-jump bracket at a=3.87
        (16.0, 0.378968 + 0.0294 * (16.0 - 15.0)),  # post-jump bracket at a=15
        (25.0, 0.525968 + 0.0294 * (25.0 - 20.0)),  # top bracket
    ],
)
def test_piecewise_tax_matches_the_bracket_schedule(assets: float, expected: float):
    """`T(a)` returns `B + tau_a*(a - a0)` for the bracket containing `a`."""
    got = float(piecewise_capital_income_tax(jnp.asarray(assets)))
    np.testing.assert_allclose(got, expected, atol=_TAX_ATOL)


def test_tax_schedule_has_three_level_discontinuities():
    """`T(a)` jumps up at a=3.87 and a=15 and drops down at a=6.97.

    The level jumps at a=3.87 (+0.10) and a=15 (+0.15) raise the tax; the
    subsidy bracket `[6.97, 8.36)` resets the offset to 0.05, so `T` drops by
    about 0.14 at a=6.97. All three discontinuities make the budget non-monotone
    and force the FUES-vs-VFI-only comparison; a continuous schedule would admit
    MSS/LTM too.
    """
    below_first = float(piecewise_capital_income_tax(jnp.asarray(3.86)))
    at_first = float(piecewise_capital_income_tax(jnp.asarray(3.87)))
    below_subsidy = float(piecewise_capital_income_tax(jnp.asarray(6.96)))
    at_subsidy = float(piecewise_capital_income_tax(jnp.asarray(6.97)))
    below_second = float(piecewise_capital_income_tax(jnp.asarray(14.99)))
    at_second = float(piecewise_capital_income_tax(jnp.asarray(15.0)))
    assert at_first - below_first > 0.05
    assert at_second - below_second > 0.05
    assert at_subsidy - below_subsidy < -0.1


def test_app3_taxed_fues_matches_vfi_on_asset_interior():
    """With taxes, App.3 FUES agrees with its VFI twin on the asset interior.

    FUES resolves the tax jumps and kinks: the mean interior value difference is
    tight and the bulk of interior cells fall within a sub-percent tolerance,
    with only the tax-bracket-boundary nodes disagreeing more.
    """
    dcegm_model = build_model(
        variant="dcegm",
        n_assets=N_ASSETS,
        n_wage_nodes=N_WAGE_NODES,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
        use_taxes=True,
    )
    brute_model = build_model(
        variant="brute",
        n_assets=N_ASSETS,
        n_wage_nodes=N_WAGE_NODES,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
        use_taxes=True,
    )
    dcegm_solution = dcegm_model.solve(
        params=build_params(variant="dcegm", n_periods=N_PERIODS, use_taxes=True),
        log_level="debug",
    )
    brute_solution = brute_model.solve(
        params=build_params(variant="brute", n_periods=N_PERIODS, use_taxes=True),
        log_level="debug",
    )

    interior = slice(N_LOW_NODES, N_ASSETS - N_HIGH_NODES)
    differences = []
    for period in sorted(brute_solution)[:-1]:
        if "working" not in brute_solution[period]:
            continue
        brute_V = np.asarray(brute_solution[period]["working"])
        dcegm_V = np.asarray(dcegm_solution[period]["working"])
        assert brute_V.shape == dcegm_V.shape
        differences.append(
            np.abs(dcegm_V[..., interior] - brute_V[..., interior]).ravel()
        )
    difference = np.concatenate(differences)

    assert float(difference.mean()) < MEAN_TOL
    assert float((difference <= CELL_TOL).mean()) >= MIN_FRACTION_WITHIN
