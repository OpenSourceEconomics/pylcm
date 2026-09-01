"""Small executable models for learning specialized consumption-saving regimes."""

import jax.numpy as jnp

import lcm
from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.regime import Regime
from lcm.solvers import DCEGM, EGM, NBEGM, NEGM, LTMEnvelope
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


WEALTH_GRID = LinSpacedGrid(start=1.0, stop=20.0, n_points=20)
CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=25.0, n_points=30)
SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=5.0, n_points=8)
INVESTMENT_GRID = LinSpacedGrid(start=0.0, stop=2.0, n_points=15)


def utility(consumption: ContinuousAction) -> FloatND:
    """Log flow utility."""
    return jnp.log(consumption)


def terminal_utility(wealth: ContinuousState) -> FloatND:
    """Consume terminal liquid wealth."""
    return jnp.log1p(wealth)


def savings(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    """Liquid post-decision balance."""
    return wealth - consumption


def next_wealth(savings: ContinuousState) -> ContinuousState:
    """Carry liquid savings into the terminal period."""
    return savings


def next_regime() -> ScalarInt:
    """The examples contain one decision period followed by death."""
    return RegimeId.dead


ONE_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="wealth",
    post_decision_state="savings",
)


def build_one_margin_model(*, enable_jit: bool = True) -> Model:
    """Build a two-period, one-margin model solved by plain EGM."""
    working = ConsumptionSavingsRegime(
        transition=next_regime,
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": next_wealth},
        functions={"utility": utility, "savings": savings},
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=ONE_MARGIN,
                lower=0.0,
            )
        },
        liquid=ONE_MARGIN,
        solver=EGM(savings_grid=SAVINGS_GRID),
        active=lambda age: age == 0,
    )
    dead = Regime(
        transition=None,
        states={"wealth": WEALTH_GRID},
        functions={"utility": terminal_utility},
        active=lambda age: age == 1,
    )
    return Model(
        regimes={"working": working, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        enable_jit=enable_jit,
    )


@lcm.piecewise_affine(
    output="tax",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint(
            threshold="tax_exemption",
            kind="continuous_kink",
        ),
    ),
)
def tax(
    *,
    liquid: ContinuousState,
    tax_rate: float,
    tax_exemption: float,
) -> FloatND:
    """Tax only liquid wealth above the exemption."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def resources(
    *,
    liquid: ContinuousState,
    tax: FloatND,
    income: float,
) -> FloatND:
    """Cash on hand after the kinked tax schedule."""
    return liquid + income - tax


def savings_after_tax(
    *,
    resources: FloatND,
    consumption: ContinuousAction,
) -> ContinuousState:
    """Post-decision savings out of after-tax resources."""
    return resources - consumption


def next_liquid(savings: ContinuousState) -> ContinuousState:
    """Carry post-decision savings into the terminal period."""
    return savings


def tax_terminal_utility(liquid: ContinuousState) -> FloatND:
    """Consume terminal liquid wealth."""
    return jnp.log1p(liquid)


TAX_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


def build_kinked_tax_model(*, enable_jit: bool = True) -> Model:
    """Build the smallest NBEGM model with a continuous tax-bracket kink."""
    working = ConsumptionSavingsRegime(
        transition=next_regime,
        states={"liquid": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"liquid": next_liquid},
        functions={
            "utility": utility,
            "tax": tax,
            "resources": resources,
            "savings": savings_after_tax,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=TAX_MARGIN,
                lower=0.0,
            )
        },
        liquid=TAX_MARGIN,
        solver=NBEGM(savings_grid=SAVINGS_GRID),
        active=lambda age: age == 0,
    )
    dead = Regime(
        transition=None,
        states={"liquid": WEALTH_GRID},
        functions={"utility": tax_terminal_utility},
        active=lambda age: age == 1,
    )
    return Model(
        regimes={"working": working, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        enable_jit=enable_jit,
    )


def kinked_tax_params() -> dict:
    """Parameters for the kinked-tax example."""
    return {
        "working": {
            "koopmans_aggregator": {"discount_factor": 0.95},
            "tax": {"tax_rate": 0.2, "tax_exemption": 7.0},
            "resources": {"income": 2.0},
        },
        "dead": {},
    }


def kinked_tax_initial_conditions() -> dict:
    """Two subjects on opposite sides of the tax exemption."""
    return {
        "age": jnp.array([0, 0]),
        "liquid": jnp.array([5.0, 10.0]),
        "regime_id": jnp.array([RegimeId.working, RegimeId.working]),
    }


def new_illiquid(
    *, illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """Post-decision illiquid stock."""
    return illiquid + illiquid_investment


def adjustment_cost(
    *, illiquid: ContinuousState, new_illiquid: ContinuousState
) -> FloatND:
    """Liquid cost of changing the illiquid stock."""
    return new_illiquid - illiquid


def resources_before_cost(wealth: ContinuousState) -> FloatND:
    """Liquid wealth plus current income."""
    return wealth + 5.0


def liquid_savings(
    *, resources: FloatND, consumption: ContinuousAction
) -> ContinuousState:
    """Liquid post-decision balance after the outer adjustment."""
    return resources - consumption


def next_wealth_from_liquid_savings(
    liquid_savings: ContinuousState,
) -> ContinuousState:
    """Carry nested liquid savings into the terminal period."""
    return liquid_savings


def next_illiquid(new_illiquid: ContinuousState) -> ContinuousState:
    """Carry the chosen illiquid stock into the terminal period."""
    return new_illiquid


def nested_utility(
    *, consumption: ContinuousAction, illiquid: ContinuousState
) -> FloatND:
    """Utility from consumption and the current illiquid stock."""
    return jnp.log(consumption) + 0.05 * jnp.log1p(illiquid)


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    """Invert the marginal utility of log consumption."""
    return 1.0 / marginal_continuation


def nested_terminal_utility(
    *, wealth: ContinuousState, illiquid: ContinuousState
) -> FloatND:
    """Terminal value of both assets."""
    return jnp.log1p(wealth) + 0.5 * jnp.log1p(illiquid)


NESTED_LIQUID_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources=NetOfAdjustmentCost(
        output="resources",
        before_cost="resources_before_cost",
        cost="adjustment_cost",
    ),
    post_decision_state="liquid_savings",
)

OUTER_MARGIN = OuterContinuousMargin(
    state="illiquid",
    action="illiquid_investment",
    post_decision_state="new_illiquid",
    no_adjustment=outer_unchanged,
)


def build_nested_model(*, enable_jit: bool = True) -> Model:
    """Build a two-period, two-margin model solved by nested EGM."""
    working = NestedConsumptionSavingsRegime(
        transition=next_regime,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        actions={
            "consumption": CONSUMPTION_GRID,
            "illiquid_investment": INVESTMENT_GRID,
        },
        state_transitions={
            "wealth": next_wealth_from_liquid_savings,
            "illiquid": next_illiquid,
        },
        functions={
            "utility": nested_utility,
            "new_illiquid": new_illiquid,
            "adjustment_cost": adjustment_cost,
            "resources_before_cost": resources_before_cost,
            "liquid_savings": liquid_savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=NESTED_LIQUID_MARGIN,
                lower=0.0,
            )
        },
        liquid=NESTED_LIQUID_MARGIN,
        outer_continuous=OUTER_MARGIN,
        solver=NEGM(
            inner=DCEGM(savings_grid=SAVINGS_GRID, envelope=LTMEnvelope()),
            outer_grid=ILLIQUID_GRID,
        ),
        active=lambda age: age == 0,
    )
    dead = Regime(
        transition=None,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        functions={"utility": nested_terminal_utility},
        active=lambda age: age == 1,
    )
    return Model(
        regimes={"working": working, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        enable_jit=enable_jit,
    )


def example_params() -> dict:
    """Parameters shared by both examples."""
    return {
        "working": {"koopmans_aggregator": {"discount_factor": 0.95}},
        "dead": {},
    }


def example_initial_conditions(*, nested: bool = False) -> dict:
    """Two subjects at the start of either example."""
    conditions = {
        "age": jnp.array([0, 0]),
        "wealth": jnp.array([5.0, 10.0]),
        "regime_id": jnp.array([RegimeId.working, RegimeId.working]),
    }
    if nested:
        conditions["wealth"] = jnp.array([10.0, 2.0])
        conditions["illiquid"] = jnp.array([0.0, 5.0])
    return conditions


__all__ = [
    "CONSUMPTION_GRID",
    "ILLIQUID_GRID",
    "INVESTMENT_GRID",
    "NESTED_LIQUID_MARGIN",
    "ONE_MARGIN",
    "OUTER_MARGIN",
    "SAVINGS_GRID",
    "TAX_MARGIN",
    "WEALTH_GRID",
    "RegimeId",
    "adjustment_cost",
    "build_kinked_tax_model",
    "build_nested_model",
    "build_one_margin_model",
    "example_initial_conditions",
    "example_params",
    "inverse_marginal_utility",
    "kinked_tax_initial_conditions",
    "kinked_tax_params",
    "liquid_savings",
    "nested_terminal_utility",
    "nested_utility",
    "new_illiquid",
    "next_illiquid",
    "next_liquid",
    "next_regime",
    "next_wealth",
    "next_wealth_from_liquid_savings",
    "resources",
    "resources_before_cost",
    "savings",
    "savings_after_tax",
    "tax",
    "tax_terminal_utility",
    "terminal_utility",
    "utility",
]
