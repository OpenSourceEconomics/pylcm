"""Cross-check against the independent `dcegm` implementation.

`tests/data/dcegm_reference/ijrs_taste_shocks_reference.csv` holds choice-
specific values of the IJRS consumption-retirement model with EV1 taste shocks
(scale 0.2), solved by the `dcegm` package (see the README there). The smoothed
value `scale * logsumexp([value_work, value_retire] / scale)` is compared with
pylcm's solved V on the twin model at the fixture's wealth nodes:

- the brute-force twin (taste shocks via logsumexp over the consumption-grid
  `Qc`) must agree up to consumption-grid resolution,
- the DC-EGM twin must agree tightly — same algorithm family, independent code.

Skips until `lcm.taste_shocks` and `lcm.solvers` exist; red until both
implementations are wired.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lcm.taste_shocks", reason="Taste shocks not yet implemented")
pytest.importorskip("lcm.solvers", reason="DC-EGM solver not yet implemented")

from _lcm.config import TEST_DATA
from tests.test_models import dcegm_paper_twin

SCALE = 0.2

REGIME_FOR_LAGGED_CHOICE = {0: "working_life", 1: "retirement"}


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    df = pd.read_csv(
        TEST_DATA.joinpath("dcegm_reference", "ijrs_taste_shocks_reference.csv")
    )
    values = df[["value_work", "value_retire"]].to_numpy()
    shifted = values - np.nanmax(values, axis=1, keepdims=True)
    df["emax"] = np.nanmax(values, axis=1) + SCALE * np.log(
        np.nansum(np.exp(shifted / SCALE), axis=1)
    )
    # The retiree rows at the lowest wealth node carry an upstream artifact:
    # the fixture's `value_retire` there contradicts the value implied by the
    # fixture's own `policy_retire` column (e.g. period 0, wealth 1: stored
    # value -66.0, but consuming the stored policy 0.124·wealth per period is
    # worth about -54), and a fine-grid value-iteration recursion of the same
    # model confirms the policy-implied value. Both pylcm solvers reproduce
    # the recursion, so these rows are excluded rather than the tolerances
    # loosened; every other row is asserted at full tolerance.
    return df.query("not (lagged_choice == 1 and wealth == 1.0)")


def _wealth_node_indices(wealth_points: np.ndarray) -> np.ndarray:
    grid = np.asarray(dcegm_paper_twin.WEALTH_GRID.to_jax())
    indices = np.searchsorted(grid, wealth_points)
    np.testing.assert_allclose(grid[indices], wealth_points, atol=1e-12)
    return indices


@pytest.mark.parametrize(
    ("solver", "rtol", "atol"),
    [
        ("brute_force", 2e-2, 2e-2),
        ("dcegm", 2e-3, 2e-3),
    ],
)
def test_twin_smoothed_value_matches_dcegm_reference(solver, rtol, atol, reference):
    """pylcm's smoothed V equals the independent `dcegm` solution.

    Tolerances are looser for brute force (consumption-grid resolution) than for
    DC-EGM (same algorithm family, exact policies on both sides).
    """
    model = dcegm_paper_twin.get_model(solver)
    params = dcegm_paper_twin.get_params(taste_shock_scale=SCALE)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    for (period, lagged_choice), group in reference.groupby(
        ["period", "lagged_choice"]
    ):
        regime = REGIME_FOR_LAGGED_CHOICE[lagged_choice]
        v_arr = np.asarray(period_to_regime_to_V_arr[period][regime])
        node_indices = _wealth_node_indices(group["wealth"].to_numpy())
        np.testing.assert_allclose(
            v_arr[node_indices],
            group["emax"].to_numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"period={period}, regime={regime}",
        )
