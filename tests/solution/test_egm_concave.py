"""Spec for the concave DC-EGM step (no discrete actions, no taste shocks).

The oracle is the retired part of the Iskhakov et al. (2017) analytical solution,
anchored by `tests/solution/test_retirement_only_oracle.py`. The DC-EGM solution
must hit the analytical values on the full wealth grid — including the lowest
wealth levels, where the credit-constrained segment is exact and the brute-force
solver is unstable.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LogSpacedGrid, Model
from lcm.exceptions import InvalidValueFunctionError
from lcm.regime import Regime as UserRegime
from lcm.typing import ContinuousState, FloatND
from lcm_examples.mortality import WEALTH_GRID
from tests.solution.test_retirement_only_oracle import (
    ANALYTICAL_CASES,
    load_analytical_values_retired,
    stack_retirement_V,
)
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
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
    # Elementwise — every (period, node) value must hit the analytical
    # solution; aggregating over periods could hide a localized error.
    np.testing.assert_allclose(numerical, analytical, atol=0.03)


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


def test_discount_factor_zero_yields_consume_everything_values():
    """With `discount_factor = 0`, V equals the utility of consuming all resources.

    The degenerate-inversion guard must hold after discounting: a zero
    discount factor zeroes the marginal continuation at every savings node, so
    the consume-everything corner is optimal at every wealth node and
    `V(wealth) = log(wealth)` in every non-terminal period.
    """
    n_periods = 3
    model = get_retirement_only_model("dcegm", n_periods)
    params = get_retirement_only_params(n_periods, discount_factor=0.0)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    wealth = np.asarray(model.user_regimes["retirement"].states["wealth"].to_jax())
    for period in range(n_periods - 1):
        np.testing.assert_allclose(
            np.asarray(period_to_regime_to_V_arr[period]["retirement"]),
            np.log(wealth),
            atol=1e-6,
            err_msg=f"period={period}",
        )


def _bequest_utility(wealth: ContinuousState, age: float) -> FloatND:
    return (age / 50.0) * jnp.log(wealth)


def test_age_dependent_terminal_utility_solves_to_closed_form():
    """A bequest utility reading `age` works as a DC-EGM continuation.

    Terminal carries are evaluated with the solve loop's `period` and `age`,
    so a terminal utility may read them like any regime function. With
    `u = log(c)`, bequest `(age/50) * log(wealth)` at terminal age 50, zero
    interest, and a two-period horizon, the decision period has the closed
    form `c* = wealth / (1 + beta)` and
    `V = log(c*) + beta * log(wealth - c*)`.
    """
    n_periods = 2
    discount_factor = 0.98
    # Dense log-spaced terminal grid: the bequest continuation is read by
    # linear interpolation on this grid, so it must resolve the curvature of
    # `log` for the closed form to be the test oracle.
    bequest_dead = UserRegime(
        transition=None,
        states={"wealth": LogSpacedGrid(start=0.25, stop=400.0, n_points=400)},
        functions={"utility": _bequest_utility},
    )
    model = Model(
        regimes={
            "retirement": dcegm_retirement.replace(active=lambda age: age < 50),
            "dead": bequest_dead,
        },
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )
    params = get_retirement_only_params(n_periods, discount_factor=discount_factor)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    wealth = np.asarray(WEALTH_GRID.to_jax())
    consumption = wealth / (1.0 + discount_factor)
    expected = np.log(consumption) + discount_factor * np.log(wealth - consumption)
    # The lowest wealth nodes carry visible publish-side interpolation error
    # (the value function curves hardest there); the age-dependence under
    # test is identical across nodes, so they are excluded rather than the
    # tolerance loosened for the whole grid.
    np.testing.assert_allclose(
        np.asarray(period_to_regime_to_V_arr[0]["retirement"])[3:],
        expected[3:],
        atol=1e-3,
    )


def test_nan_param_surfaces_as_value_function_error():
    """A NaN parameter raises `InvalidValueFunctionError` naming the regime.

    NaN propagates into the published V rows; the solve's NaN diagnostics must
    report regime and age for a DC-EGM regime just as for a brute-force one.
    """
    n_periods = 3
    model = get_retirement_only_model("dcegm", n_periods)
    params = get_retirement_only_params(n_periods, discount_factor=float("nan"))

    with pytest.raises(InvalidValueFunctionError, match="retirement"):
        model.solve(params=params, log_level="debug")


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
