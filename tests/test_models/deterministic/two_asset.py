"""Minimal deterministic two-asset (liquid + illiquid pension) model.

A worker holds a liquid account `liquid` and an illiquid pension `pension` and each
period chooses consumption and a one-directional pension `deposit` ($\\ge 0$). The
liquid post-decision balance is `liquid - consumption - deposit`, kept non-negative
by a borrowing constraint; the pension post-decision balance is
`pension + deposit + chi*log(1 + deposit)`, a concave employer match. Liquid earns
gross return `1 + return_liquid`, pension earns the higher `1 + return_pension`. On
death the pension is paid out as a lump sum and added to liquid wealth.

This is the two-action-coupled-continuous-state structure the multidimensional EGM
foundation targets, written in brute-force-solvable form: it is the dense-grid
reference (oracle) the 2-D EGM kernel is validated against. Two continuous states
(`liquid`, `pension`) and two continuous actions (`consumption`, `deposit`), both
coupled through the budget.
"""

import jax.numpy as jnp

from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.regime import Regime
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _crra(consumption: FloatND, crra: float) -> FloatND:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )


def utility(consumption: ContinuousAction, crra: float) -> FloatND:
    return _crra(consumption, crra)


def bequest(
    liquid: ContinuousState,
    pension: ContinuousState,
    crra: float,
    pension_bequest_weight: float,
) -> FloatND:
    """Consume liquid wealth plus the lump-sum pension payout in the final period.

    `pension_bequest_weight` is 1.0 for the model proper (the full pension balance is
    paid out); a value below 1 makes pension wealth marginally less valuable than
    liquid, which is what produces an interior-deposit (unconstrained) region.
    """
    return _crra(liquid + pension_bequest_weight * pension, crra)


def next_liquid(
    liquid: ContinuousState,
    consumption: ContinuousAction,
    deposit: ContinuousAction,
    return_liquid: float,
    wage: float,
) -> ContinuousState:
    return (1.0 + return_liquid) * (liquid - consumption - deposit) + wage


def next_pension(
    pension: ContinuousState,
    deposit: ContinuousAction,
    return_pension: float,
    match_rate: float,
) -> ContinuousState:
    return (1.0 + return_pension) * (
        pension + deposit + match_rate * jnp.log(1.0 + deposit)
    )


def feasible(
    liquid: ContinuousState,
    consumption: ContinuousAction,
    deposit: ContinuousAction,
) -> BoolND:
    """Liquid borrowing constraint: the liquid post-decision balance stays >= 0."""
    return consumption + deposit <= liquid


def next_regime_from_working(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.working)


def get_model(
    *,
    n_periods: int = 3,
    n_liquid: int = 12,
    n_pension: int = 10,
    n_consumption: int = 14,
    n_deposit: int = 8,
    liquid_max: float = 100.0,
    pension_max: float = 50.0,
) -> Model:
    """Create the two-regime (working, dead) two-asset model.

    Grid sizes default to the small oracle scale; pass larger values for a finer
    reference solve.
    """
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=1.0, stop=liquid_max, n_points=n_liquid)
    pension_grid = LinSpacedGrid(start=0.0, stop=pension_max, n_points=n_pension)
    working = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=1.0, stop=liquid_max, n_points=n_consumption
            ),
            "deposit": LinSpacedGrid(start=0.0, stop=pension_max, n_points=n_deposit),
        },
        states={"liquid": liquid_grid, "pension": pension_grid},
        state_transitions={"liquid": next_liquid, "pension": next_pension},
        constraints={"feasible": feasible},
        transition=next_regime_from_working,
        functions={"utility": utility},
        active=lambda age, la=last_age: age < la,
    )
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid, "pension": pension_grid},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def get_params(
    *,
    n_periods: int = 3,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.02,
    return_pension: float = 0.06,
    match_rate: float = 1.0,
    wage: float = 10.0,
    pension_bequest_weight: float = 1.0,
) -> dict:
    """Get parameters for the two-asset model (pension return exceeds liquid return)."""
    final_age_alive = 40 + (n_periods - 2) * 10
    return {
        "discount_factor": discount_factor,
        "final_age_alive": final_age_alive,
        "working": {
            "utility": {"crra": crra},
            "next_liquid": {"return_liquid": return_liquid, "wage": wage},
            "next_pension": {
                "return_pension": return_pension,
                "match_rate": match_rate,
            },
        },
        "dead": {
            "utility": {
                "crra": crra,
                "pension_bequest_weight": pension_bequest_weight,
            }
        },
    }
