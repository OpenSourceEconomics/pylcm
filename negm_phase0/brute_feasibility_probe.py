"""Phase-0 brute-force feasibility probe for the 2-asset (Laibson-shaped) model.

Solves a two-continuous-state (liquid `X`, illiquid `Z`) + two-continuous-action
(consumption `C`, net illiquid investment `Iz`) + income-process lifecycle model
with the existing `BruteForce` solver, and reports wall-clock and GPU peak memory
at a given grid size. The brute search axis is the full product

    periods x income_nodes x N_X x N_Z x N_C x N_Iz

which is exactly the curse the NEGM/multidim build targets.

Usage: python brute_feasibility_probe.py N_X N_Z N_C N_Iz N_INCOME N_PERIODS

Run on gpu-01 (V100 16 GB) with `XLA_PYTHON_CLIENT_PREALLOCATE=false` so the
reported peak is the true high-water mark, not the preallocated pool.
"""

# ruff: noqa: T201, INP001, PLR2004  (throwaway Phase-0 probe: prints results)

import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Regime,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

N_X = int(sys.argv[1])
N_Z = int(sys.argv[2])
N_C = int(sys.argv[3])
N_IZ = int(sys.argv[4])
N_INCOME = int(sys.argv[5])
N_PERIODS = int(sys.argv[6])

ILLIQUID_FLOW = 0.05  # iota: utility flow from the illiquid stock
WITHDRAWAL_PENALTY = 0.10  # kappa on illiquid withdrawals (Iz < 0)
RISK_AVERSION = 2.0


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def liquid_savings(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    illiquid_investment: ContinuousAction,
    income: ContinuousState,
) -> FloatND:
    """End-of-period liquid balance after consumption and the illiquid transfer.

    A withdrawal (`illiquid_investment < 0`) is penalised: only a fraction
    `1 - kappa` of withdrawn funds reaches the liquid account.
    """
    credited = jnp.where(
        illiquid_investment < 0.0,
        (1.0 - WITHDRAWAL_PENALTY) * illiquid_investment,
        illiquid_investment,
    )
    return wealth + jnp.exp(income) - consumption - credited


def next_wealth(liquid_savings: FloatND, interest_rate: float) -> ContinuousState:
    return (1.0 + interest_rate) * liquid_savings


def next_illiquid(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + illiquid_investment


def utility(consumption: ContinuousAction, illiquid: ContinuousState) -> FloatND:
    flow = consumption + ILLIQUID_FLOW * illiquid
    return flow ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)


def liquid_floor(liquid_savings: FloatND) -> BoolND:
    return liquid_savings >= 0.0


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
            "wealth": LinSpacedGrid(start=0.0, stop=40.0, n_points=N_X),
            "illiquid": LinSpacedGrid(start=0.0, stop=40.0, n_points=N_Z),
            "income": RouwenhorstAR1Process(n_points=N_INCOME),
        },
        state_transitions={
            "wealth": next_wealth,
            "illiquid": next_illiquid,
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=30.0, n_points=N_C),
            "illiquid_investment": LinSpacedGrid(start=-10.0, stop=10.0, n_points=N_IZ),
        },
        transition=next_regime,
        constraints={
            "liquid_floor": liquid_floor,
            "illiquid_floor": illiquid_floor,
            "positive_consumption": positive_consumption,
        },
        functions={
            "utility": utility,
            "liquid_savings": liquid_savings,
        },
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


params = {
    "discount_factor": 0.95,
    "alive": {
        "income": {"mu": 0.0, "sigma": 0.2, "rho": 0.9},
        "next_wealth": {"interest_rate": 0.03},
    },
}

search_width = N_INCOME * N_X * N_Z * N_C * N_IZ
print(
    f"grids: N_X={N_X} N_Z={N_Z} N_C={N_C} N_Iz={N_IZ} N_income={N_INCOME} "
    f"T={N_PERIODS} | per-period brute width = {search_width:,}",
    flush=True,
)

t0 = time.perf_counter()
model = build_model()
t_build = time.perf_counter() - t0

t0 = time.perf_counter()
solution = model.solve(params=params, log_level="off")
jax.block_until_ready(solution)
t_solve = time.perf_counter() - t0

stats = jax.devices()[0].memory_stats() or {}
peak = stats.get("peak_bytes_in_use", 0)
print(
    f"OK build={t_build:.2f}s solve={t_solve:.2f}s "
    f"gpu_peak={peak / 1e6:.1f} MB periods_solved={len(solution)}",
    flush=True,
)
print(f"  mem_stats_keys={sorted(stats)}", flush=True)
