"""Phase-0 kinked 2-asset toy + brute oracle (the NEGM parity target).

The smallest model carrying the Laibson frictions a future NEGM prototype must
reproduce: liquid `X` + illiquid `Z`, a credit-card borrowing-rate kink at the
liquid post-state `a^X = 0`, an illiquid withdrawal penalty (kink at `Iz = 0`),
the hard `Z >= 0` floor, and the direct illiquid utility flow `u(C + iota*Z)`.

Solved with the existing `BruteForce` solver as the oracle. Prints `V` at a few
fixed grid coordinates so a future NEGM outer-search-over-`a^Z` + inner-1D-EGM
prototype can be checked for V-parity by concrete value, not eyeball.

Run on gpu-01: python kinked_toy_oracle.py
"""

# ruff: noqa: T201, INP001, PLR2004  (throwaway Phase-0 probe: prints results)

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

N_X = 12
N_Z = 12
N_C = 25
N_IZ = 25
N_PERIODS = 4

ILLIQUID_FLOW = 0.05  # iota
WITHDRAWAL_PENALTY = 0.10  # kappa on Iz < 0
BORROW_RATE = 0.12  # credit-card rate on a^X < 0
SAVE_RATE = 0.03  # rate on a^X >= 0
RISK_AVERSION = 2.0


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def liquid_savings(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    illiquid_investment: ContinuousAction,
) -> FloatND:
    """Liquid post-decision balance `a^X`, with the withdrawal penalty wedge."""
    credited = jnp.where(
        illiquid_investment < 0.0,
        (1.0 - WITHDRAWAL_PENALTY) * illiquid_investment,
        illiquid_investment,
    )
    return wealth + 5.0 - consumption - credited  # 5.0 = fixed labour income


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    """Liquid law with the credit-card rate kink at `a^X = 0`."""
    rate = jnp.where(liquid_savings < 0.0, BORROW_RATE, SAVE_RATE)
    return (1.0 + rate) * liquid_savings


def next_illiquid(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + illiquid_investment


def utility(consumption: ContinuousAction, illiquid: ContinuousState) -> FloatND:
    flow = consumption + ILLIQUID_FLOW * illiquid
    return flow ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)


def liquid_floor(liquid_savings: FloatND) -> BoolND:
    return liquid_savings >= -5.0  # a small credit limit


def illiquid_floor(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> BoolND:
    return illiquid + illiquid_investment >= 0.0


def positive_consumption(consumption: ContinuousAction) -> BoolND:
    return consumption > 0.05


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def build_model() -> Model:
    final_age_alive = 20 + (N_PERIODS - 2) * 5
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": LinSpacedGrid(start=0.0, stop=30.0, n_points=N_X),
            "illiquid": LinSpacedGrid(start=0.0, stop=30.0, n_points=N_Z),
        },
        state_transitions={"wealth": next_wealth, "illiquid": next_illiquid},
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=N_C),
            "illiquid_investment": LinSpacedGrid(start=-8.0, stop=8.0, n_points=N_IZ),
        },
        transition=next_regime,
        constraints={
            "liquid_floor": liquid_floor,
            "illiquid_floor": illiquid_floor,
            "positive_consumption": positive_consumption,
        },
        functions={"utility": utility, "liquid_savings": liquid_savings},
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


params = {"discount_factor": 0.95, "alive": {}}

solution = build_model().solve(params=params, log_level="off")

# V at period 0, regime alive, at a few (wealth_idx, illiquid_idx) coordinates.
v0 = solution[0]["alive"]
print(f"V[period 0, alive] shape = {v0.shape}", flush=True)
coords = [
    (0, 0),
    (0, N_Z // 2),
    (N_X // 2, 0),
    (N_X // 2, N_Z // 2),
    (N_X - 1, N_Z - 1),
]
for ix, iz in coords:
    print(
        f"  V[wealth_idx={ix}, illiquid_idx={iz}] = {float(v0[ix, iz]):.10f}",
        flush=True,
    )
print(
    f"PARITY-TARGET period0_alive sum={float(jnp.sum(v0)):.8f} "
    f"min={float(jnp.min(v0)):.8f} max={float(jnp.max(v0)):.8f}",
    flush=True,
)
