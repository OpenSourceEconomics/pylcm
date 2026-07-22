"""DC-EGM for the DS-2026 App.3 discrete-housing model matches VFI.

Application 3 reaches its terminal `dead` regime by a choice-driven discrete
transition: `next_housing = housing_choice`, so the held housing next period is
the chosen housing code, not the parent's current housing. The terminal bequest
carry is dead's utility on its own (housing, wealth) grid; the DC-EGM parent
selects the carry row at the *chosen* next-housing by indexing dead's housing
axis with `housing_choice` — the same action-indexed gather the non-terminal
cross-regime read uses.

The oracle is the grid-search (VFI) twin of the same model (same wealth, wage,
and housing grids), which solves the identical Bellman problem by brute force.
The DC-EGM and VFI value functions must agree on the wealth interior, where both
solvers are well-defined (VFI undershoots the borrowing-constrained low-wealth
region and edge-clamps the wealth grid's ceiling).
"""

from typing import Literal

import numpy as np
import pytest

from tests.test_models.ds_app3_discrete_housing import build_model, build_params

# Small construction-scale grids: a single local solve stays fast. The wealth
# interior excludes the constrained low-wealth nodes (VFI undershoots) and the
# top edge-clamp nodes (VFI saturates at the grid ceiling) so the head-to-head
# lives where both solvers are well-defined.
N_ASSETS = 60
N_WAGE_NODES = 3
N_PERIODS = 5
N_CONSUMPTION = 80
N_LOW_NODES = 14
N_HIGH_NODES = 14


@pytest.mark.parametrize("upper_envelope", ["fues"])
def test_app3_dcegm_matches_vfi_on_wealth_interior(
    upper_envelope: Literal["fues", "mss", "ltm", "rfc"],
):
    """App.3 DC-EGM matches its VFI twin on the wealth interior, per wage and house.

    A correct action-indexed terminal carry reads dead's bequest at the chosen
    next-housing; a wrong (identity) alignment would read the bequest at the
    parent's held housing instead, shifting the continuation by the bequest gap
    between housing levels and breaking the comparison. Agreement is up to the
    VFI consumption-grid resolution.
    """
    dcegm_model = build_model(
        variant="dcegm",
        n_assets=N_ASSETS,
        n_wage_nodes=N_WAGE_NODES,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
        upper_envelope=upper_envelope,
    )
    brute_model = build_model(
        variant="brute",
        n_assets=N_ASSETS,
        n_wage_nodes=N_WAGE_NODES,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
    )
    dcegm_solution = dcegm_model.solve(
        params=build_params(variant="dcegm", n_periods=N_PERIODS), log_level="debug"
    )
    brute_solution = brute_model.solve(
        params=build_params(variant="brute", n_periods=N_PERIODS), log_level="debug"
    )

    interior = slice(N_LOW_NODES, N_ASSETS - N_HIGH_NODES)
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working"])
        dcegm_V = np.asarray(dcegm_solution[period]["working"])
        # V axes: (housing, wage, assets).
        assert brute_V.shape == dcegm_V.shape
        np.testing.assert_allclose(
            dcegm_V[..., interior],
            brute_V[..., interior],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )
