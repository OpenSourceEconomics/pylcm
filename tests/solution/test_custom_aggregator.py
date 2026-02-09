"""Test that a custom aggregation function H can be used in a model."""

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


# --------------------------------------------------------------------------------------
# Shared model components
# --------------------------------------------------------------------------------------
@categorical
class LaborSupply:
    work: int
    retire: int


@categorical
class RegimeId:
    working: int
    dead: int


def utility(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    # Multiplicative disutility ensures positive utility for CES aggregator
    work_factor = jnp.where(is_working, 1.0 / (1.0 + disutility_of_work), 1.0)
    return consumption * work_factor


def labor_income(is_working: BoolND) -> FloatND:
    return jnp.where(is_working, 1.5, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
) -> ContinuousState:
    return wealth - consumption + labor_income


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.working)


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


# --------------------------------------------------------------------------------------
# CES aggregator (Epstein-Zin style)
# --------------------------------------------------------------------------------------
def ces_H(
    utility: float,
    continuation_value: float,
    discount_factor: float,
    ies: float,
) -> float:
    rho = 1 - ies
    return (
        (1 - discount_factor) * utility**rho + discount_factor * continuation_value**rho
    ) ** (1 / rho)


START_AGE = 0
N_PERIODS = 4
FINAL_AGE_ALIVE = START_AGE + N_PERIODS - 2  # = 2


def _make_model(custom_H=None):
    """Create a simple model, optionally with a custom H."""
    functions = {
        "labor_income": labor_income,
        "is_working": is_working,
    }
    if custom_H is not None:
        functions["H"] = custom_H

    working_regime = Regime(
        actions={
            "labor_supply": DiscreteGrid(LaborSupply),
            "consumption": LinSpacedGrid(start=0.5, stop=10, n_points=50),
        },
        states={
            "wealth": LinSpacedGrid(start=0.5, stop=10, n_points=30),
        },
        utility=utility,
        constraints={"borrowing_constraint": borrowing_constraint},
        transitions={
            "next_wealth": next_wealth,
            "next_regime": next_regime,
        },
        functions=functions,
        active=lambda age: age <= FINAL_AGE_ALIVE,
    )

    dead_regime = Regime(
        terminal=True,
        utility=lambda: 0.0,
        active=lambda age: age > FINAL_AGE_ALIVE,
    )

    return Model(
        regimes={"working": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=START_AGE, stop=FINAL_AGE_ALIVE + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_custom_ces_aggregator_differs_from_default():
    """A CES aggregator with ies != 1 should produce different value functions."""
    model_default = _make_model()
    model_ces = _make_model(custom_H=ces_H)

    params_default = {
        "discount_factor": 0.95,
        "working": {
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
    }
    params_ces = {
        "working": {
            "H": {"discount_factor": 0.95, "ies": 0.5},
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
        "dead": {},
    }

    V_default = model_default.solve(params_default)
    V_ces = model_ces.solve(params_ces)

    # The value functions should differ because the aggregation rule differs
    has_difference = False
    for period in V_default:
        for regime_name in V_default[period]:
            if not jnp.allclose(
                V_default[period][regime_name], V_ces[period][regime_name]
            ):
                has_difference = True
                break

    assert has_difference, (
        "CES and default aggregator should produce different value functions"
    )


def test_default_H_injected_for_non_terminal():
    """The default H function should be in non-terminal Regime.functions."""
    r = Regime(
        utility=lambda: 0.0,
        transitions={"next_regime": lambda: {"a": 1.0}},
        active=lambda age: age < 1,
    )
    assert "H" in r.functions


def test_default_H_not_injected_for_terminal():
    """Terminal regimes should not have H injected."""
    r = Regime(
        terminal=True,
        utility=lambda: 0.0,
    )
    assert "H" not in r.functions


def test_custom_H_not_overwritten():
    """A user-provided H should not be replaced by the default."""

    def my_H(utility: float, continuation_value: float) -> float:
        return utility + continuation_value

    r = Regime(
        utility=lambda: 0.0,
        transitions={"next_regime": lambda: {"a": 1.0}},
        active=lambda age: age < 1,
        functions={"H": my_H},
    )
    assert r.functions["H"] is my_H


def test_params_template_includes_H():
    """The params template should include H's parameters."""
    model = _make_model()
    template = model.params_template
    # Default H has discount_factor parameter
    assert "H" in template["working"]
    assert "discount_factor" in template["working"]["H"]


def test_params_template_custom_H():
    """Custom H params should appear in the template."""
    model = _make_model(custom_H=ces_H)
    template = model.params_template
    assert "H" in template["working"]
    assert "discount_factor" in template["working"]["H"]
    assert "ies" in template["working"]["H"]


def test_terminal_regime_value_unchanged_by_H():
    """Terminal regimes don't use H, so different H should give same terminal values."""
    model_default = _make_model()
    model_ces = _make_model(custom_H=ces_H)

    params_default = {
        "discount_factor": 0.95,
        "working": {
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
    }
    params_ces = {
        "working": {
            "H": {"discount_factor": 0.95, "ies": 0.5},
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
        "dead": {},
    }

    V_default = model_default.solve(params_default)
    V_ces = model_ces.solve(params_ces)

    # Last period is terminal — value functions should be identical
    last_period = max(V_default.keys())
    assert_array_equal(
        V_default[last_period]["dead"],
        V_ces[last_period]["dead"],
    )
