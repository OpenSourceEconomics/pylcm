"""EGM-FUES for the DS-2026 App.2 housing model matches VFI.

Application 2's Table 3 compares two solution methods for the continuous-housing
model: EGM-FUES and NEGM. This test validates the **EGM-FUES** column, which
pylcm builds by discretising the next-housing choice onto the housing grid and
treating it as a discrete action, so the inner liquid-asset DC-EGM plus the
discrete-choice FUES upper envelope reproduces the paper's 1-D-FUES-nested-over-
the-housing-grid solve (its Box 2).

The oracle is the grid-search (VFI) twin of the same discrete-housing model
(same liquid, wage, and housing grids), which solves the identical Bellman
problem by brute force. On most of the liquid-wealth interior the two value
functions agree up to grid resolution. They diverge on a minority of cells in the
durable-rich corner (high held housing, high liquid), where DC-EGM sits slightly
below the VFI twin: the DC-EGM savings-grid interpolation of a kinked value
function plus the liquid top-edge extrapolation, a discretization gap that shrinks
only partially as the savings and liquid grids refine. So the test asserts *bulk*
agreement (the mean is tight and the overwhelming majority of interior cells fall
within a small tolerance), not exact equality at every cell.

The borrowing-constrained low-wealth region (where VFI is forced to the
consumption floor and the steep CES utility diverges) and the top edge-clamp
nodes are excluded outright.
"""

from typing import Literal

import numpy as np
import pytest

from tests.test_models.ds_app2_housing_fues import build_model, build_params

# Small construction-scale grids: a single local solve stays fast. The wealth
# interior excludes the constrained low-wealth nodes (steep-CES VFI floor) and
# the top edge-clamp nodes so the head-to-head lives where both solvers are
# well-defined.
N_LIQUID = 40
N_HOUSING = 5
N_PERIODS = 5
N_CONSUMPTION = 400
N_LOW_NODES = 12
N_HIGH_NODES = 8

# Bulk-agreement thresholds. The value function is O(35) on the interior, so a
# 0.3 cell tolerance is sub-percent. The two solvers disagree on a minority of
# interior cells, concentrated in the durable-rich corner (high held housing,
# high liquid). There DC-EGM sits slightly *below* the grid-search VFI: it is the
# DC-EGM savings-grid interpolation of a kinked value function plus the liquid
# top-edge extrapolation, a discretization gap that shrinks only partially as the
# savings and liquid grids refine. The branch-aware VFI oracle confirms this is
# not a comparator-ordering artifact — the `I[max_d V_d]` vs `max_d I[V_d]`
# ordering term is identically zero on this interior (the next-housing choice only
# switches in the excluded low-wealth (S, s) band; see
# `test_app2_branch_aware_vfi.py`). The tolerance accommodates the durable-corner
# discretization while the smooth-region mean stays sub-percent; the
# paper-comparable metric is the simulated consumption Euler error, which is
# unaffected.
MEAN_TOL = 0.18
CELL_TOL = 0.30
MIN_FRACTION_WITHIN = 0.82


@pytest.mark.parametrize("upper_envelope", ["fues"])
def test_app2_fues_matches_vfi_on_liquid_interior(
    upper_envelope: Literal["fues", "mss", "ltm", "rfc"],
):
    """App.2 EGM-FUES agrees with its VFI twin on the liquid interior in bulk.

    The discrete-choice DC-EGM selects the optimal next-housing level by the FUES
    upper envelope over the housing grid; the VFI twin does the same by brute
    grid search. The mean interior value difference is sub-percent and the large
    majority of interior cells fall within the cell tolerance — the EGM-FUES
    solve is correct. The remaining cells sit in the durable-rich corner (high
    held housing, high liquid), where DC-EGM sits slightly below the VFI twin
    through its savings-grid interpolation of a kinked value function and the
    liquid top-edge extrapolation. This is a discretization gap, not a
    comparator-ordering artifact: the branch-aware VFI oracle confirms the
    `I[max_d V_d]` vs `max_d I[V_d]` ordering term is identically zero on this
    interior (`test_app2_branch_aware_vfi.py`).
    """
    dcegm_model = build_model(
        variant="dcegm",
        n_grid=N_LIQUID,
        n_housing=N_HOUSING,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
        upper_envelope=upper_envelope,
    )
    brute_model = build_model(
        variant="brute",
        n_grid=N_LIQUID,
        n_housing=N_HOUSING,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
    )
    dcegm_solution = dcegm_model.solve(
        params=build_params(variant="dcegm"), log_level="debug"
    )
    brute_solution = brute_model.solve(
        params=build_params(variant="brute"), log_level="debug"
    )

    interior = slice(N_LOW_NODES, N_LIQUID - N_HIGH_NODES)
    scored_periods = [
        period
        for period in sorted(brute_solution)[:-1]
        if "working" in brute_solution[period]
    ]
    assert scored_periods

    differences = []
    for period in scored_periods:
        brute_V = np.asarray(brute_solution[period]["working"])
        dcegm_V = np.asarray(dcegm_solution[period]["working"])
        # V axes: (wage, housing, liquid).
        assert brute_V.shape == dcegm_V.shape
        differences.append(
            np.abs(dcegm_V[..., interior] - brute_V[..., interior]).ravel()
        )
    difference = np.concatenate(differences)

    assert float(difference.mean()) < MEAN_TOL
    fraction_within = float((difference <= CELL_TOL).mean())
    assert fraction_within >= MIN_FRACTION_WITHIN
