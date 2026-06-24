"""Phase-0 kinked 2-asset toy + brute oracle (the NEGM parity target).

The smallest model carrying the Laibson frictions a NEGM prototype must
reproduce: liquid `X` + illiquid `Z`, a credit-card borrowing-rate kink at the
liquid post-state `a^X = 0`, an illiquid withdrawal penalty (kink at `Iz = 0`),
the hard `Z >= 0` floor, and the direct illiquid utility flow `u(C + iota*Z)`.

Solved with the `BruteForce` solver as the oracle. The liquid `wealth` grid
covers the reachable negative-wealth range: the liquid floor `a^X >= -5` and the
borrow rate `0.12` put the minimum reachable `next_wealth` at `1.12 * -5 = -5.6`,
so a grid starting at `0` would force the brute V-interpolation to extrapolate
below its own support at every low-wealth cell and inflate the value there. The
grid starts at `-6` so every reachable `next_wealth` lands inside the support.

Prints `V` at a few fixed grid coordinates so the NEGM solve can be checked for
V-parity by concrete value, plus a provenance block (lcm build, git SHA, x64
flag, grid shapes and checksums, the corner cell's independently recomputed
maximizing `(c, Iz, a^X, next_wealth, next_illiquid)`, and the interpolation
mode) so a future reader can reproduce the pinned numbers exactly.

Run: python kinked_toy_oracle.py
"""

# ruff: noqa: T201, INP001, PLR2004, S607  (Phase-0 probe: prints, calls git)

import os
import subprocess
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

import lcm
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

WEALTH_MIN = -6.0  # covers the reachable next_wealth floor 1.12 * -5 = -5.6
WEALTH_MAX = 30.0
ILLIQUID_MAX = 30.0
LABOUR_INCOME = 5.0
LIQUID_CREDIT_LIMIT = -5.0  # a^X >= -5

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
    return wealth + LABOUR_INCOME - consumption - credited


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
    return liquid_savings >= LIQUID_CREDIT_LIMIT


def illiquid_floor(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> BoolND:
    return illiquid + illiquid_investment >= 0.0


def positive_consumption(consumption: ContinuousAction) -> BoolND:
    return consumption > 0.05


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


WEALTH_GRID = LinSpacedGrid(start=WEALTH_MIN, stop=WEALTH_MAX, n_points=N_X)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=ILLIQUID_MAX, n_points=N_Z)
CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=N_C)
ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=-8.0, stop=8.0, n_points=N_IZ)


def build_model() -> Model:
    final_age_alive = 20 + (N_PERIODS - 2) * 5
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        state_transitions={"wealth": next_wealth, "illiquid": next_illiquid},
        actions={
            "consumption": CONSUMPTION_GRID,
            "illiquid_investment": ILLIQUID_INVESTMENT_GRID,
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


PARAMS = {"discount_factor": 0.95, "alive": {}}
DISCOUNT_FACTOR = 0.95

_CORNER = (0, 0)
_REPORT_COORDS = [
    (0, 0),
    (0, N_Z // 2),
    (N_X // 2, 0),
    (N_X // 2, N_Z // 2),
    (N_X - 1, N_Z - 1),
]


def _checksum(values: FloatND) -> float:
    """Order-stable scalar fingerprint of a 1-D grid."""
    return float(jnp.sum(jnp.asarray(values) * jnp.arange(1, values.shape[0] + 1)))


def _git_sha() -> str:
    """Short git SHA of the working tree, or `unknown` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def _interp_next_v(
    v_next: FloatND, wealth_query: FloatND, illiquid_query: FloatND
) -> FloatND:
    """Bilinear interpolation of the next-period `alive` V at a query point.

    Mirrors the brute solver's `map_coordinates` (order-1, edge-clamped)
    V-interpolation: fractional coordinates on each state grid, clamped to the
    grid range, then a bilinear blend. Used to recompute the corner cell's
    Bellman maximizer independently of the solver's own search.
    """

    def _coord(query: FloatND, lo: float, hi: float, n: int) -> FloatND:
        frac = (query - lo) / (hi - lo) * (n - 1)
        return jnp.clip(frac, 0.0, n - 1.0)

    cw = _coord(wealth_query, WEALTH_MIN, WEALTH_MAX, N_X)
    ci = _coord(illiquid_query, 0.0, ILLIQUID_MAX, N_Z)
    w0 = jnp.floor(cw).astype(jnp.int32)
    i0 = jnp.floor(ci).astype(jnp.int32)
    w1 = jnp.clip(w0 + 1, 0, N_X - 1)
    i1 = jnp.clip(i0 + 1, 0, N_Z - 1)
    fw = cw - w0
    fi = ci - i0
    return (
        v_next[w0, i0] * (1 - fw) * (1 - fi)
        + v_next[w1, i0] * fw * (1 - fi)
        + v_next[w0, i1] * (1 - fw) * fi
        + v_next[w1, i1] * fw * fi
    )


def _recompute_corner_maximizer(v_next_alive: FloatND) -> dict[str, float]:
    """Brute-search the period-0 corner-cell Bellman against the solved V[period 1].

    Recomputes `V[corner]` and its maximizing action by an explicit grid search
    over `(consumption, illiquid_investment)` — interpolating the next-period
    `alive` value with the brute solver's order-1 edge-clamped scheme — so the
    pinned corner value carries an independent argmax record, not just the
    solver's scalar.
    """
    ix, iz = _CORNER
    wealth = WEALTH_MIN + (WEALTH_MAX - WEALTH_MIN) * ix / (N_X - 1)
    illiquid = ILLIQUID_MAX * iz / (N_Z - 1)
    consumption = jnp.asarray(CONSUMPTION_GRID.to_jax())
    investment = jnp.asarray(ILLIQUID_INVESTMENT_GRID.to_jax())
    c_mesh, iz_mesh = jnp.meshgrid(consumption, investment, indexing="ij")
    credited = jnp.where(iz_mesh < 0.0, (1.0 - WITHDRAWAL_PENALTY) * iz_mesh, iz_mesh)
    savings = wealth + LABOUR_INCOME - c_mesh - credited
    next_w = jnp.where(savings < 0.0, 1.0 + BORROW_RATE, 1.0 + SAVE_RATE) * savings
    next_z = illiquid + iz_mesh
    flow = c_mesh + ILLIQUID_FLOW * illiquid
    flow_utility = flow ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)
    continuation = _interp_next_v(v_next_alive, next_w, next_z)
    feasible = (savings >= LIQUID_CREDIT_LIMIT) & (next_z >= 0.0) & (c_mesh > 0.05)
    objective = jnp.where(
        feasible, flow_utility + DISCOUNT_FACTOR * continuation, -jnp.inf
    )
    flat = int(jnp.argmax(objective))
    best_c_idx, best_iz_idx = divmod(flat, investment.shape[0])
    return {
        "V": float(objective.reshape(-1)[flat]),
        "consumption": float(consumption[best_c_idx]),
        "illiquid_investment": float(investment[best_iz_idx]),
        "liquid_savings": float(savings[best_c_idx, best_iz_idx]),
        "next_wealth": float(next_w[best_c_idx, best_iz_idx]),
        "next_illiquid": float(next_z[best_c_idx, best_iz_idx]),
    }


def main() -> None:
    """Solve the oracle, print its pinned values and the provenance block."""
    solution = build_model().solve(params=PARAMS, log_level="off")
    v0 = solution[0]["alive"]
    v1 = solution[1]["alive"]

    print(f"lcm.__file__ = {lcm.__file__}", flush=True)
    print(f"git SHA = {_git_sha()}", flush=True)
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}", flush=True)
    print("interpolation = order-1 (linear) edge-clamped map_coordinates", flush=True)
    wealth_grid = jnp.asarray(WEALTH_GRID.to_jax())
    illiquid_grid = jnp.asarray(ILLIQUID_GRID.to_jax())
    print(
        f"wealth grid: shape={wealth_grid.shape} "
        f"[{float(wealth_grid[0])}, {float(wealth_grid[-1])}] "
        f"checksum={_checksum(wealth_grid):.6f}",
        flush=True,
    )
    print(
        f"illiquid grid: shape={illiquid_grid.shape} "
        f"[{float(illiquid_grid[0])}, {float(illiquid_grid[-1])}] "
        f"checksum={_checksum(illiquid_grid):.6f}",
        flush=True,
    )
    print(
        f"consumption grid: shape=({N_C},) checksum="
        f"{_checksum(jnp.asarray(CONSUMPTION_GRID.to_jax())):.6f}",
        flush=True,
    )
    print(
        f"illiquid_investment grid: shape=({N_IZ},) checksum="
        f"{_checksum(jnp.asarray(ILLIQUID_INVESTMENT_GRID.to_jax())):.6f}",
        flush=True,
    )

    print(f"V[period 0, alive] shape = {v0.shape}", flush=True)
    for ix, iz in _REPORT_COORDS:
        print(
            f"  V[wealth_idx={ix}, illiquid_idx={iz}] = {float(v0[ix, iz]):.10f}",
            flush=True,
        )
    print(
        f"PARITY-TARGET period0_alive sum={float(jnp.sum(v0)):.8f} "
        f"min={float(jnp.min(v0)):.8f} max={float(jnp.max(v0)):.8f}",
        flush=True,
    )

    maximizer = _recompute_corner_maximizer(v1)
    print(f"corner cell {_CORNER} independent Bellman recomputation:", flush=True)
    print(f"  solver V = {float(v0[_CORNER]):.10f}", flush=True)
    print(f"  recomputed V = {maximizer['V']:.10f}", flush=True)
    print(
        "  argmax (c, Iz, a^X, next_wealth, next_illiquid) = "
        f"({maximizer['consumption']:.4f}, "
        f"{maximizer['illiquid_investment']:.4f}, "
        f"{maximizer['liquid_savings']:.4f}, "
        f"{maximizer['next_wealth']:.4f}, "
        f"{maximizer['next_illiquid']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
