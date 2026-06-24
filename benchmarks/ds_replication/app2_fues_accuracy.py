"""Euler-error + timing harness for the DS-2026 App.2 EGM-FUES column.

Application 2's Table 3 compares EGM-FUES and NEGM on the continuous-housing
model. This module scores the **EGM-FUES** column, which pylcm builds as a
discrete-choice DC-EGM over the housing grid (`ds_app2_housing_fues.py`): it
solves the model, simulates a sample path plus a per-period on-grid policy
panel, and scores the consumption Euler equation along the path with the Judd
(1992) metric, plus a compile-vs-runtime timing split. The companion NEGM column
lives in `app2_housing_accuracy.py`.

## The stochastic Euler equation

Utility is the App.2 separable CES, so the consumption marginal utility is
`u'(c) = alpha*c^{-gamma_C}` and the housing-service term drops. The wage is a
Markov (Tauchen AR1) process, so the conditional expectation is a
transition-probability-weighted sum over next-period wage nodes,

    E_t[u'(c_{t+1})]
        = sum_j P[wage_t, j] * alpha*c_{t+1}(a', H', wage_j)^{-gamma_C},

evaluated at the chosen post-decision liquid assets `a'` and next housing level
`H'` held fixed, with `c_{t+1}(., H', wage_j)` the period-(t+1) consumption
policy of the `(H', wage_j)` cell interpolated in liquid assets. The implied
consumption is `c_euler_t = (beta*(1+r)*E_t[u'(c_{t+1})] / alpha)^{-1/gamma_C}`
and the metric is `mean(log10(|c_euler_t / c_t - 1|))`.

## Valid points

A path point is scored only where the continuous consumption Euler equation
governs the optimum:

- the subject is in the working regime this period and the next;
- next-period liquid assets are strictly positive (the no-borrowing constraint
  is slack);
- the discrete housing level is not adjusted this period and not adjusted next
  period (`housing == housing_choice` at both `t` and `t+1`) — the housing
  switch is a value-function kink the smooth consumption Euler equation skips.

pylcm's simulate is grid-restricted for all solvers, so the magnitude at coarse
grids is coarse-grid, not a methodological ceiling; the paper grids (NG in
{250...1000}) exceed local memory and require a GPU sweep.
"""

import functools
import logging
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from lcm import TauchenAR1Process
from tests.test_models import ds_app2_housing_fues as fues
from tests.test_models.ds_app2_housing_fues import HousingFuesRegimeId

_logger = logging.getLogger("lcm")

# Paper Table 3 calibration (default no-tax column).
INTEREST_RATE = 0.04
DISCOUNT_FACTOR = 0.94
ALPHA = 0.70
GAMMA_C = 3.5
RHO_W = 0.82
SIGMA_W = 0.11
MU_W = 0.0
N_PERIODS = 8


def _wage_nodes_and_transition(
    *,
    rho: float = RHO_W,
    sigma: float = SIGMA_W,
    mu: float = MU_W,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Markov wage node values and the transition-probability matrix.

    Constructed from the same `TauchenAR1Process` the model's working regime
    uses; the Tauchen build-time grid is a NaN placeholder resolved from the
    runtime `(rho, sigma, mu)` parameters.
    """
    process = TauchenAR1Process(n_points=fues.N_WAGE_NODES, gauss_hermite=True)
    nodes = np.asarray(
        process.compute_gridpoints(
            rho=jnp.asarray(rho), sigma=jnp.asarray(sigma), mu=jnp.asarray(mu)
        )
    )
    transition = np.asarray(
        process.compute_transition_probs(
            rho=jnp.asarray(rho), sigma=jnp.asarray(sigma), mu=jnp.asarray(mu)
        )
    )
    return nodes, transition


def _nearest_wage_index(wage_values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Map each (possibly rounded) path wage value to its wage-node index."""
    return np.abs(wage_values[:, None] - nodes[None, :]).argmin(axis=1)


# Cap on the liquid-seed resolution of the policy reconstruction; bounds the
# panel size so the Euler scales to paper-scale NG (see `_policy_lookup`).
_MAX_POLICY_SEED_LIQUID = 24


def _subsample(grid: np.ndarray, max_points: int) -> np.ndarray:
    """Return at most `max_points` evenly-spaced nodes of `grid`, ends included."""
    if len(grid) <= max_points:
        return grid
    index = np.unique(np.linspace(0, len(grid) - 1, max_points).round().astype(int))
    return grid[index]


@functools.cache
def _build_solved(
    *,
    n_grid: int,
    n_housing: int,
    n_periods: int,
    n_consumption: int,
    tau: float,
):
    """Build and solve the EGM-FUES App.2 model, caching the solution."""
    model = fues.build_model(
        variant="dcegm",
        n_grid=n_grid,
        n_housing=n_housing,
        n_periods=n_periods,
        n_consumption=n_consumption,
    )
    params = fues.build_params(variant="dcegm", tau=tau)
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def _liquid_grid(model) -> np.ndarray:
    """Return the working regime's liquid-asset grid node array."""
    return np.asarray(model.user_regimes["working"].states["liquid"].to_jax())


def _full_grid_initial_conditions(
    *,
    liquid_grid: np.ndarray,
    n_housing: int,
    wage_nodes: np.ndarray,
    age: float,
) -> dict:
    """Seed one subject at every `(liquid, housing, wage)` product-grid cell."""
    housing_codes = np.arange(n_housing)
    liquid = np.repeat(liquid_grid, len(housing_codes) * len(wage_nodes))
    housing = np.tile(np.repeat(housing_codes, len(wage_nodes)), len(liquid_grid))
    wage = np.tile(wage_nodes, len(liquid_grid) * len(housing_codes))
    n_subjects = liquid.size
    return {
        "liquid": jnp.asarray(liquid),
        "housing": jnp.asarray(housing, dtype=jnp.int32),
        "wage": jnp.asarray(wage),
        "age": jnp.full(n_subjects, age),
        "regime_id": jnp.full(n_subjects, HousingFuesRegimeId.working, dtype=jnp.int32),
    }


def _policy_lookup(
    *,
    model,
    params,
    solution,
    liquid_grid: np.ndarray,
    n_housing: int,
    wage_nodes: np.ndarray,
    n_periods: int,
    seed: int,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """Per-`(period, housing, wage)` sorted `(liquid, consumption)` policy.

    The period-`p` policy is read off a fresh simulation seeded over the full
    `(liquid, housing, wage)` product grid at period `p`'s age, so the first
    simulated period lands on the regular grid; each `(housing, wage)` cell maps
    to the sorted liquid samples and the chosen consumption, ready for linear
    interpolation in liquid at an arbitrary `a'`.
    """
    # Subsample the liquid seed so the panel (liquid x n_housing x wage subjects,
    # each argmaxing over consumption x housing_choice) stays within device
    # memory at paper-scale NG; consumption is smooth in liquid, so a coarser
    # interpolation grid is adequate.
    liquid_grid = _subsample(liquid_grid, _MAX_POLICY_SEED_LIQUID)
    ages = model.ages.exact_values
    lookup: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for period in range(n_periods):
        initial_conditions = _full_grid_initial_conditions(
            liquid_grid=liquid_grid,
            n_housing=n_housing,
            wage_nodes=wage_nodes,
            age=float(ages[period]),
        )
        panel = (
            model.simulate(
                params=params,
                initial_conditions=initial_conditions,
                period_to_regime_to_V_arr=solution,
                log_level="off",
                seed=seed,
            )
            .to_dataframe()
            .query(f"period == {period} and regime_name == 'working'")
        )
        if panel.empty:
            continue
        panel = panel.copy()
        panel["wage_index"] = _nearest_wage_index(panel["wage"].to_numpy(), wage_nodes)
        panel["housing_code"] = panel["housing"].map(_housing_code).to_numpy()
        grouped = panel.groupby(["housing_code", "wage_index"], observed=True)
        for (housing_code, wage_index), cell in grouped:
            order = np.argsort(cell["liquid"].to_numpy())
            liquid = cell["liquid"].to_numpy()[order]
            consumption = cell["consumption"].to_numpy()[order]
            lookup[(int(period), int(housing_code), int(wage_index))] = (
                liquid,
                consumption,
            )
    return lookup


def _housing_code(label) -> int:
    """Map a housing level label `h<i>` to its integer code `i`."""
    return int(str(label).removeprefix("h"))


def _interp_policy(
    *,
    lookup: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    period: int,
    housing_code: int,
    wage_index: int,
    liquid: float,
) -> float:
    """Linearly interpolate the consumption policy of a cell at `liquid`.

    Returns NaN when the cell has no policy samples, so the point is dropped
    rather than scored against a fabricated value.
    """
    key = (period, housing_code, wage_index)
    if key not in lookup:
        return float("nan")
    liquid_samples, consumption_samples = lookup[key]
    if liquid_samples.size == 0:
        return float("nan")
    return float(np.interp(liquid, liquid_samples, consumption_samples))


def _expected_next_marginal_utility(
    *,
    lookup: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    wage_transition: np.ndarray,
    source_wage_index: int,
    next_period: int,
    next_housing_code: int,
    next_liquid: float,
    n_wage_nodes: int,
    alpha: float,
    gamma_c: float,
) -> float:
    """Transition-weighted expected next-period consumption marginal utility."""
    expected_marginal = 0.0
    for next_wage_index in range(n_wage_nodes):
        probability = wage_transition[source_wage_index, next_wage_index]
        if probability <= 0.0:
            continue
        next_consumption = _interp_policy(
            lookup=lookup,
            period=next_period,
            housing_code=next_housing_code,
            wage_index=next_wage_index,
            liquid=next_liquid,
        )
        if not np.isfinite(next_consumption) or next_consumption <= 0.0:
            return float("nan")
        expected_marginal += probability * (alpha * next_consumption ** (-gamma_c))
    return expected_marginal


def sample_path_euler_error(
    *,
    sample_panel: pd.DataFrame,
    lookup: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    wage_nodes: np.ndarray,
    wage_transition: np.ndarray,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    alpha: float = ALPHA,
    gamma_c: float = GAMMA_C,
) -> float:
    """Mean log10 stochastic consumption Euler error along a simulated path."""
    working = sample_panel.query("regime_name == 'working'").copy()
    working = working.sort_values(["subject_id", "period"]).reset_index(drop=True)

    subject = working["subject_id"].to_numpy()
    period = working["period"].to_numpy()
    consumption = working["consumption"].to_numpy()
    liquid = working["liquid"].to_numpy()
    housing_code = working["housing"].map(_housing_code).to_numpy()
    choice_code = working["housing_choice"].map(_housing_code).to_numpy()
    wage_value = working["wage"].to_numpy()
    wage_index = _nearest_wage_index(wage_value, wage_nodes)

    same_subject = subject[:-1] == subject[1:]
    consecutive = period[1:] == period[:-1] + 1
    has_next = np.zeros(consumption.shape, dtype=bool)
    has_next[:-1] = same_subject & consecutive

    errors: list[float] = []
    for t in range(len(working) - 1):
        if not has_next[t]:
            continue
        if working["regime_name"].iloc[t + 1] != "working":
            continue
        next_liquid = liquid[t + 1]
        if next_liquid <= 1e-4 or consumption[t] <= 0.0:
            continue
        # No housing switch this period or next: the discrete housing-adjustment
        # margin is a value-function kink the smooth Euler equation skips.
        if housing_code[t] != choice_code[t]:
            continue
        if housing_code[t + 1] != choice_code[t + 1]:
            continue

        expected_marginal = _expected_next_marginal_utility(
            lookup=lookup,
            wage_transition=wage_transition,
            source_wage_index=int(wage_index[t]),
            next_period=int(period[t + 1]),
            next_housing_code=int(choice_code[t]),
            next_liquid=float(next_liquid),
            n_wage_nodes=wage_nodes.size,
            alpha=alpha,
            gamma_c=gamma_c,
        )
        if not np.isfinite(expected_marginal) or expected_marginal <= 0.0:
            continue

        consumption_euler = (
            discount_factor * (1.0 + interest_rate) * expected_marginal / alpha
        ) ** (-1.0 / gamma_c)
        relative_error = abs(consumption_euler / consumption[t] - 1.0)
        if relative_error > 0.0:
            errors.append(np.log10(relative_error))

    if not errors:
        msg = "No valid interior working-regime points to score the Euler error."
        raise ValueError(msg)
    return float(np.mean(errors))


def app2_fues_euler_error(
    *,
    n_grid: int,
    n_housing: int = 5,
    n_periods: int = N_PERIODS,
    n_consumption: int = 400,
    n_subjects: int = 400,
    tau: float = 0.07,
    seed: int = 0,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    alpha: float = ALPHA,
    gamma_c: float = GAMMA_C,
) -> float:
    """Solve the EGM-FUES App.2 model and score the consumption Euler error."""
    model, params, solution = _build_solved(
        n_grid=n_grid,
        n_housing=n_housing,
        n_periods=n_periods,
        n_consumption=n_consumption,
        tau=tau,
    )
    wage_nodes, wage_transition = _wage_nodes_and_transition()
    liquid_grid = _liquid_grid(model)

    sample_panel = _simulate_sample_path(
        model=model,
        params=params,
        solution=solution,
        n_subjects=n_subjects,
        liquid_grid=liquid_grid,
        n_housing=n_housing,
        wage_nodes=wage_nodes,
        seed=seed,
    )
    lookup = _policy_lookup(
        model=model,
        params=params,
        solution=solution,
        liquid_grid=liquid_grid,
        n_housing=n_housing,
        wage_nodes=wage_nodes,
        n_periods=n_periods,
        seed=seed,
    )
    error = sample_path_euler_error(
        sample_panel=sample_panel,
        lookup=lookup,
        wage_nodes=wage_nodes,
        wage_transition=wage_transition,
        interest_rate=interest_rate,
        discount_factor=discount_factor,
        alpha=alpha,
        gamma_c=gamma_c,
    )
    _logger.info("DS App.2 EGM-FUES Euler error: n_grid=%d -> %.3f", n_grid, error)
    return error


def _simulate_sample_path(
    *,
    model,
    params,
    solution,
    n_subjects: int,
    liquid_grid: np.ndarray,
    n_housing: int,
    wage_nodes: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Simulate a spread of working subjects at the median wage node."""
    rng = np.random.default_rng(seed)
    median_wage = float(wage_nodes[len(wage_nodes) // 2])
    initial_conditions = {
        "liquid": jnp.asarray(
            rng.uniform(2.0, float(liquid_grid[-1]) * 0.6, n_subjects)
        ),
        "housing": jnp.asarray(rng.integers(1, n_housing, n_subjects), dtype=jnp.int32),
        "wage": jnp.full(n_subjects, median_wage),
        "age": jnp.full(n_subjects, float(model.ages.exact_values[0])),
        "regime_id": jnp.full(n_subjects, HousingFuesRegimeId.working, dtype=jnp.int32),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=seed,
    )
    return result.to_dataframe()


def app2_fues_timing(
    *,
    n_grid: int,
    n_housing: int = 5,
    n_periods: int = N_PERIODS,
    n_consumption: int = 400,
    tau: float = 0.07,
    n_runs: int = 3,
) -> dict[str, float]:
    """Measure compile and steady-state run time of one EGM-FUES solve.

    The first solve times compile-plus-run; later solves of the same model reuse
    the cached executable and time pure execution, so the compile cost is the
    difference. Every solve is forced to completion with `block_until_ready`.
    """
    model = fues.build_model(
        variant="dcegm",
        n_grid=n_grid,
        n_housing=n_housing,
        n_periods=n_periods,
        n_consumption=n_consumption,
    )
    params = fues.build_params(variant="dcegm", tau=tau)

    jax.clear_caches()
    start = time.perf_counter()
    jax.block_until_ready(model.solve(params=params, log_level="off"))
    compile_plus_run = time.perf_counter() - start

    runtimes = []
    for _ in range(n_runs):
        start = time.perf_counter()
        jax.block_until_ready(model.solve(params=params, log_level="off"))
        runtimes.append(time.perf_counter() - start)
    runtime = min(runtimes)

    result = {"compile_time": compile_plus_run - runtime, "runtime": runtime}
    _logger.info(
        "DS App.2 EGM-FUES timing: n_grid=%d -> compile=%.3fs run=%.3fs",
        n_grid,
        result["compile_time"],
        result["runtime"],
    )
    return result


def app2_fues_accuracy_table(
    *,
    n_grids: Sequence[int] = (12, 18),
    n_housing: int = 5,
    n_periods: int = N_PERIODS,
    n_consumption: int = 400,
    n_subjects: int = 400,
    tau: float = 0.07,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the EGM-FUES Euler-error sweep across liquid-grid sizes.

    One solve+simulate per grid size. The paper grids (NG in {250...1000}) exceed
    local memory and require a GPU sweep; pass smaller `n_grids` for a local run.
    """
    rows = [
        {
            "n_grid": n_grid,
            "fues_euler_error": app2_fues_euler_error(
                n_grid=n_grid,
                n_housing=n_housing,
                n_periods=n_periods,
                n_consumption=n_consumption,
                n_subjects=n_subjects,
                tau=tau,
                seed=seed,
            ),
        }
        for n_grid in n_grids
    ]
    return pd.DataFrame(rows)
