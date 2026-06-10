"""Spec for the concave DC-EGM step (no discrete actions, no taste shocks).

The oracle is the retired part of the Iskhakov et al. (2017) analytical solution,
anchored by `tests/solution/test_retirement_only_oracle.py`. The DC-EGM solution
must hit the analytical values on the full wealth grid — including the lowest
wealth levels, where the credit-constrained segment is exact and the brute-force
solver is unstable.

Skips until `lcm.solvers` exists; red until the EGM step is wired into the solve.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

pytest.importorskip("lcm.solvers", reason="DC-EGM solver not yet implemented")

from tests.solution.test_retirement_only_oracle import (
    ANALYTICAL_CASES,
    load_analytical_values_retired,
    stack_retirement_V,
)
from tests.test_models.deterministic.dcegm_variants import (
    get_retirement_only_model,
    get_retirement_only_params,
)


@pytest.mark.parametrize(("case", "n_periods"), ANALYTICAL_CASES.items())
def test_dcegm_matches_analytical_on_full_wealth_grid(case, n_periods):
    """DC-EGM V equals the analytical retired values on every wealth node.

    Tighter than the brute-force tolerance and with no low-wealth exclusion: the
    constrained segment makes EGM exact where grid search is unstable.
    """
    model = get_retirement_only_model("dcegm", n_periods)
    params = get_retirement_only_params(n_periods)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    numerical = stack_retirement_V(period_to_regime_to_V_arr)
    analytical = load_analytical_values_retired(case)
    mse = np.mean((analytical - numerical) ** 2, axis=0)
    aaae(mse, 0, decimal=3)


@pytest.mark.parametrize(("case", "n_periods"), ANALYTICAL_CASES.items())
def test_dcegm_error_not_much_worse_than_brute_force(case, n_periods):
    """Diagnostic with slack: DC-EGM should not lose badly to grid search anywhere.

    Pointwise EGM dominance is not a theorem (the publish step interpolates onto
    the wealth grid), so this is a guard against gross regressions, not a
    superiority claim. The hard accuracy requirement lives in
    `test_dcegm_matches_analytical_on_full_wealth_grid`.
    """
    analytical = load_analytical_values_retired(case)
    params = get_retirement_only_params(n_periods)

    errors = {}
    for solver in ["brute_force", "dcegm"]:
        model = get_retirement_only_model(solver, n_periods)
        got = model.solve(params=params, log_level="debug")
        numerical = stack_retirement_V(got)
        # Exclude the brute-force-unstable low-wealth nodes from the head-to-head
        # so the comparison is on territory where both solvers are well-defined.
        errors[solver] = np.mean((analytical[:, 2:] - numerical[:, 2:]) ** 2)

    assert errors["dcegm"] <= 2.0 * errors["brute_force"] + 1e-12


def test_dcegm_solution_has_standard_v_array_layout():
    """DC-EGM publishes V arrays with the same shape/keys as the brute solver."""
    n_periods = 4
    params = get_retirement_only_params(n_periods)

    brute = get_retirement_only_model("brute_force", n_periods).solve(
        params=params, log_level="debug"
    )
    dcegm = get_retirement_only_model("dcegm", n_periods).solve(
        params=params, log_level="debug"
    )

    assert sorted(brute) == sorted(dcegm)
    for period in brute:
        assert sorted(brute[period]) == sorted(dcegm[period])
        for regime in brute[period]:
            assert brute[period][regime].shape == dcegm[period][regime].shape
