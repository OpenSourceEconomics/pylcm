"""DC-EGM terminal carry: a bequest utility reads a model-level fixed param.

A terminal regime's bequest utility may reach a model-level (shared) parameter
through a helper function — for example a consumption-equivalence scale applied
to the bequest. When that parameter is supplied through `fixed_params`, it is
partialled into the compiled functions at model build and dropped from the live
parameter template. The terminal carry producer the DC-EGM parent reads must
still evaluate the bequest with the fixed value, exactly as a free solve param
would be threaded — so the parent's continuation through the terminal regime is
identical whether the scale is fixed or free.

The DC-EGM solution must match a mathematically identical dense brute-force
spec, and the fixed and free parametrisations must agree with each other.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, GridSearch
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

N_PERIODS = 4
# The lowest wealth nodes are where grid search is least reliable and the
# DC-EGM constrained segment is exact; excluded from the head-to-head so the
# comparison lives on territory where both solvers are well-defined.
N_BRUTE_UNSTABLE_NODES = 20

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=400.0, n_points=120)
CONSUMPTION_GRID = LinSpacedGrid(start=1.0, stop=400.0, n_points=500)
# Dense terminal wealth grid: the bequest continuation is read by linear
# interpolation on this grid, so it must resolve the curvature of `log`.
BEQUEST_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=400.0, n_points=600)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(400.0 * (i / 199) ** 3 for i in range(200)))

# Consumption-equivalence scale the bequest utility reads through a helper
# function. Supplied via `fixed_params` in the fixed parametrisation; the
# terminal regime's template never carries it as a free leaf there.
AVERAGE_CONSUMPTION_EQUIV = 2.5


@categorical(ordered=False)
class RegimeId:
    retirement: ScalarInt
    dead: ScalarInt


def utility_scale_factor(average_consumption_equiv: float) -> FloatND:
    """Consumption-equivalence scale applied inside the bequest.

    A model-level helper whose single parameter is the load-bearing one: when
    fixed, its qualified name `utility_scale_factor__average_consumption_equiv`
    must reach the terminal carry producer for the bequest to evaluate.
    """
    return jnp.asarray(1.0 / average_consumption_equiv)


def bequest_utility(
    wealth: ContinuousState,
    utility_scale_factor: FloatND,
    bequest_weight: float,
) -> FloatND:
    return bequest_weight * jnp.log(wealth * utility_scale_factor + 1.0)


def utility_retirement(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings_post(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth_from_savings(
    savings_post: FloatND, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * savings_post


def next_wealth_brute(
    wealth: ContinuousState, consumption: ContinuousAction, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption)


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_regime_from_retirement(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.retirement)


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _make_dead_regime() -> UserRegime:
    return UserRegime(
        transition=None,
        states={"wealth": BEQUEST_WEALTH_GRID},
        functions={
            "utility": bequest_utility,
            "utility_scale_factor": utility_scale_factor,
        },
    )


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


@functools.cache
def _get_model(solver: str, *, scale_is_fixed: bool) -> Model:
    """Retirement model carrying into a bequest terminal reading a model param.

    The terminal `dead` regime's bequest reaches `average_consumption_equiv`
    through `utility_scale_factor`. When `scale_is_fixed` the value is supplied
    through `fixed_params` (partialled at model build, dropped from the live
    template); otherwise it is a free solve param of the terminal regime.
    """
    is_dcegm = solver == "dcegm"
    ages = _ages()
    last_age = ages.exact_values[-1]
    retirement = UserRegime(
        transition=next_regime_from_retirement,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_from_savings if is_dcegm else next_wealth_brute
        },
        constraints=(
            {} if is_dcegm else {"borrowing_constraint": borrowing_constraint}
        ),
        functions=(
            {
                "utility": utility_retirement,
                "resources": resources,
                "savings_post": savings_post,
                "inverse_marginal_utility": inverse_marginal_utility,
            }
            if is_dcegm
            else {"utility": utility_retirement}
        ),
        solver=(
            DCEGM(
                continuous_state="wealth",
                continuous_action="consumption",
                resources="resources",
                post_decision_function="savings_post",
                savings_grid=SAVINGS_GRID,
                n_constrained_points=64,
            )
            if is_dcegm
            else GridSearch()
        ),
        active=lambda age, la=last_age: age < la,
    )
    return Model(
        regimes={"retirement": retirement, "dead": _make_dead_regime()},
        ages=ages,
        regime_id_class=RegimeId,
        fixed_params=_fixed_scale() if scale_is_fixed else {},
    )


def _base_params() -> dict:
    final_age_alive = 40 + (N_PERIODS - 2) * 10
    return {
        "discount_factor": 0.98,
        "interest_rate": 0.0,
        "final_age_alive": final_age_alive,
        "bequest_weight": 1.5,
    }


def _free_scale_params() -> dict:
    params = _base_params()
    params["dead"] = {
        "utility_scale_factor": {"average_consumption_equiv": AVERAGE_CONSUMPTION_EQUIV}
    }
    return params


def _fixed_scale() -> dict:
    return {
        "dead": {
            "utility_scale_factor": {
                "average_consumption_equiv": AVERAGE_CONSUMPTION_EQUIV
            }
        }
    }


@pytest.mark.parametrize("scale_is_fixed", [True, False])
def test_dcegm_terminal_bequest_fixed_param_matches_brute_force(
    scale_is_fixed: bool,  # noqa: FBT001
):
    """A terminal bequest reading a model-level param matches brute force.

    The DC-EGM parent carries into a terminal regime whose bequest reaches the
    `average_consumption_equiv` scale through a helper function. Whether that
    scale is a fixed param (partialled into the terminal carry producer at model
    build, dropped from the live template) or a free solve param (threaded from
    the regime's live params), the parent's value function at every working-life
    period agrees with the dense-grid brute-force oracle.
    """
    params = _base_params() if scale_is_fixed else _free_scale_params()
    dcegm_solution = _get_model("dcegm", scale_is_fixed=scale_is_fixed).solve(
        params=params, log_level="debug"
    )
    brute_solution = _get_model("brute_force", scale_is_fixed=scale_is_fixed).solve(
        params=params, log_level="debug"
    )

    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["retirement"])
        dcegm_V = np.asarray(dcegm_solution[period]["retirement"])
        assert brute_V.shape == dcegm_V.shape
        np.testing.assert_allclose(
            dcegm_V[..., N_BRUTE_UNSTABLE_NODES:],
            brute_V[..., N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}, scale_is_fixed={scale_is_fixed}",
        )
