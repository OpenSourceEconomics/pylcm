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
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
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

    wealth = np.asarray(WEALTH_GRID.to_jax())
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


@categorical(ordered=False)
class _InterestRegimeId:
    retirement: ScalarInt
    dead: ScalarInt


def _log_consumption_value(wealth: np.ndarray, *, n_alive: int) -> np.ndarray:
    """Closed-form retired value `V(w) = B log(w) + A` with log utility.

    With gross return $R_g = 1 + r$, discount factor $\\beta$, and `n_alive`
    remaining periods of life, the Bellman recursion for $V(w) = B \\log w + A$
    starts from $A = B = 0$ and iterates `n_alive` times.
    """
    discount_factor = 0.95
    gross_return = 1.05
    intercept = 0.0
    slope = 0.0
    for _ in range(n_alive):
        slope_new = 1.0 + discount_factor * slope
        if slope_new > 1.0:
            intercept = -np.log(slope_new) + discount_factor * (
                slope * np.log(gross_return * (slope_new - 1.0) / slope_new) + intercept
            )
        else:
            intercept = discount_factor * intercept
        slope = slope_new
    return slope * np.log(wealth) + intercept


def test_dcegm_with_interest_matches_closed_form_on_dense_wealth_grid():
    """A 5%-interest retirement model hits the closed form on every wealth node.

    With log utility, $\\beta = 0.95$, gross return $1.05$, and a multi-period
    horizon, the retired value function is $V(w) = B \\log(w) + A$ in closed
    form. The published V must match it on a dense linear wealth grid at every
    period — across many backward-induction steps, so carry-read and
    publish-side interpolation errors of the concave value rows may not
    accumulate beyond the tolerance.
    """
    n_periods = 10

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def resources(wealth: ContinuousState) -> FloatND:
        return wealth

    def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
        return resources - consumption

    def next_wealth(savings: FloatND, interest_rate: float) -> ContinuousState:
        return (1.0 + interest_rate) * savings

    def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
        return 1.0 / marginal_continuation

    def next_regime(age: float, final_age_alive: float) -> ScalarInt:
        return jnp.where(
            age >= final_age_alive,
            _InterestRegimeId.dead,
            _InterestRegimeId.retirement,
        )

    ages = AgeGrid(start=40, stop=40 + n_periods - 1, step="Y")
    last_age = ages.exact_values[-1]
    retirement = UserRegime(
        transition=next_regime,
        actions={"consumption": LinSpacedGrid(start=1, stop=400, n_points=100)},
        states={"wealth": LinSpacedGrid(start=1, stop=400, n_points=1000)},
        state_transitions={"wealth": next_wealth},
        functions={
            "utility": utility,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            # Cubic node clustering toward the borrowing limit, as the value
            # function curves hardest there.
            savings_grid=IrregSpacedGrid(
                points=tuple(400.0 * (i / 199) ** 3 for i in range(200))
            ),
            n_constrained_points=64,
        ),
        active=lambda age: age < last_age,
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda _age: True,
    )
    model = Model(
        regimes={"retirement": retirement, "dead": dead},
        ages=ages,
        regime_id_class=_InterestRegimeId,
    )
    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "final_age_alive": 40 + n_periods - 2,
    }

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    wealth = np.linspace(1.0, 400.0, 1000)
    for period in range(n_periods - 1):
        np.testing.assert_allclose(
            np.asarray(period_to_regime_to_V_arr[period]["retirement"]),
            _log_consumption_value(wealth, n_alive=(n_periods - 1) - period),
            atol=3e-3,
            err_msg=f"period={period}",
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
