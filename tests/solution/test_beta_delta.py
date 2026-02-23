"""Test beta-delta (quasi-hyperbolic) discounting via a custom H function.

Verifies exponential, sophisticated, and naive beta-delta discounting against
analytical solutions in a 3-period consumption-savings model with log utility.

Analytical solution
-------------------
With log utility u(c) = log(c) and budget w' = w - c, optimal consumption is
c_t = w_t / D_t where D_t depends on the discounting type:

    Exponential (beta=1):   D_1 = 1 + delta,  D_0 = 1 + delta*(1 + delta)
    Sophisticated (beta<1): D_1 = 1 + bd,     D_0 = 1 + bd*(1 + bd)
    Naive (beta<1):         D_1 = 1 + bd,     D_0 = 1 + bd*(1 + delta)

where bd = beta * delta. At t=1, naive = sophisticated. They differ at t=0:
naive uses V^E (exponential continuation) while sophisticated uses V (biased).

"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from lcm import AgeGrid, LinSpacedGrid, Model, PhaseVariant, Regime, categorical
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@categorical
class RegimeId:
    working: int
    dead: int


# --------------------------------------------------------------------------------------
# Model functions
# --------------------------------------------------------------------------------------
def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def next_regime(age: float) -> ScalarInt:
    return jnp.where(age >= 1, RegimeId.dead, RegimeId.working)


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def exponential_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
) -> float:
    return utility + discount_factor * E_next_V


def beta_delta_H(
    utility: float,
    E_next_V: float,
    beta: float,
    delta: float,
) -> float:
    return utility + beta * delta * E_next_V


# --------------------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------------------
WEALTH_START = 0.5
WEALTH_STOP = 50.0
N_WEALTH = 200
N_CONSUMPTION = 500


def _make_model(*, H_func=beta_delta_H):
    working = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=WEALTH_START / N_CONSUMPTION,
                stop=WEALTH_STOP,
                n_points=N_CONSUMPTION,
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=WEALTH_START,
                stop=WEALTH_STOP,
                n_points=N_WEALTH,
                transition=next_wealth,
            ),
        },
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        functions={
            "utility": utility,
            "H": H_func,
        },
        active=lambda age: age <= 1,
    )

    dead = Regime(
        transition=None,
        states={
            "wealth": LinSpacedGrid(
                start=WEALTH_START,
                stop=WEALTH_STOP,
                n_points=N_WEALTH,
                transition=None,
            ),
        },
        functions={"utility": terminal_utility},
        active=lambda age: age > 1,
    )

    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


# --------------------------------------------------------------------------------------
# Analytical denominators
# --------------------------------------------------------------------------------------
def _denominators(beta, delta):
    """Return (D_0, D_1) for the given discounting type."""
    bd = beta * delta
    d1 = 1.0 + bd
    d0 = 1.0 + bd * d1
    return d0, d1


def _denominators_naive(beta, delta):
    """Return (D_0, D_1) for naive beta-delta agents."""
    bd = beta * delta
    d1 = 1.0 + bd  # same as sophisticated at t=1
    d0 = 1.0 + bd * (1.0 + delta)  # uses exponential V^E for future
    return d0, d1


# --------------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "beta", "delta"),
    [
        ("exponential", 1.0, 0.95),
        ("sophisticated", 0.7, 0.95),
        ("naive", 0.7, 0.95),
        ("naive_phase_variant", 0.7, 0.95),
    ],
)
def test_beta_delta_consumption(label, beta, delta):
    initial_wealth = jnp.array([20.0])
    initial_age = jnp.array([0.0])
    w0 = 20.0

    if label.startswith("naive"):
        d0_exp, d1_exp = _denominators_naive(beta, delta)
    else:
        d0_exp, d1_exp = _denominators(beta, delta)

    expected_c0 = w0 / d0_exp
    expected_c1 = (w0 - expected_c0) / d1_exp

    h_params = {"beta": beta, "delta": delta}

    if label == "naive_phase_variant":
        # Use PhaseVariant to solve with exponential H, simulate with beta-delta H
        model = _make_model(
            H_func=PhaseVariant(solve=exponential_H, simulate=beta_delta_H),
        )
        # Params are the union of both variants' params
        result = model.solve_and_simulate(
            params={
                "working": {
                    "H": {"discount_factor": delta, "beta": beta, "delta": delta},
                },
            },
            initial_states={"age": initial_age, "wealth": initial_wealth},
            initial_regimes=["working"],
            debug_mode=False,
        )
    elif label == "naive":
        model = _make_model()
        # Solve with exponential discounting (beta=1)
        solve_params = {"working": {"H": {"beta": 1.0, "delta": delta}}}
        V = model.solve(solve_params, debug_mode=False)

        # Simulate with present-biased params
        sim_params = {"working": {"H": h_params}}
        result = model.simulate(
            params=sim_params,
            initial_states={"age": initial_age, "wealth": initial_wealth},
            initial_regimes=["working"],
            V_arr_dict=V,
            debug_mode=False,
        )
    else:
        model = _make_model()
        result = model.solve_and_simulate(
            params={"working": {"H": h_params}},
            initial_states={"age": initial_age, "wealth": initial_wealth},
            initial_regimes=["working"],
            debug_mode=False,
        )

    df = result.to_dataframe().query('regime == "working"')
    got_c0 = df.loc[df["age"] == 0, "consumption"].iloc[0]
    got_c1 = df.loc[df["age"] == 1, "consumption"].iloc[0]

    # Tolerance accounts for grid discretization
    assert_allclose(got_c0, expected_c0, atol=0.15, rtol=0)
    assert_allclose(got_c1, expected_c1, atol=0.15, rtol=0)
