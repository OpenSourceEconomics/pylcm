"""Euler-error accuracy harness for Dobrescu & Shanker (2026) Application 2.

Application 2 of the FUES paper is the continuous-housing model of DS Section 2.2:
a finite-horizon problem with a liquid financial asset, an illiquid housing stock
carrying a proportional transaction cost, a **stochastic (Markov) wage**, and a
discrete adjust/keep choice. It maps onto pylcm's nested-EGM (NEGM) solver — an
inner DC-EGM on the liquid consumption-savings margin nested over an outer grid
search on the next housing stock. This module scores the **NEGM** consumption
Euler equation along a simulated sample path, the same Judd (1992) metric the
Application 1 and Application 3 harnesses use.

## The stochastic Euler equation

Utility is additively separable CES over consumption and the serviced house,
`u(c, H') = alpha*(c^{1-gamma_C} - 1)/(1 - gamma_C)
+ (1-alpha)*kappa*(H'^{1-gamma_H} - 1)/(1 - gamma_H)`, so the consumption
marginal utility is `u'(c) = alpha*c^{-gamma_C}` and the housing service term
drops from the consumption Euler equation. The wage is Markov, so the
conditional expectation is a transition-probability-weighted sum over
next-period wage nodes,

    E_t[u'(c_{t+1})]
        = sum_j P[wage_t, j] * alpha*c_{t+1}(a', H', wage_j)^{-gamma_C},

evaluated at the chosen post-decision liquid assets `a'` and next housing `H'`
held fixed, with `c_{t+1}(., ., wage_j)` the period-(t+1) consumption policy
linearly interpolated in `(liquid, housing)` for that wage node. The implied
consumption is

    c_euler_t = ( beta*(1+r)*E_t[u'(c_{t+1})] / alpha )^{-1/gamma_C},

and the metric is `mean(log10(|c_euler_t / c_t - 1|))` over the valid points.

## Valid points

A path point is scored only where the continuous consumption Euler equation
governs the optimum:

- the subject is in the working regime this period and the next (the retirement
  switch and the terminal bequest regime are excluded);
- next-period liquid assets are strictly positive, so chosen consumption did not
  exhaust resources and the no-borrowing constraint is slack (the constrained
  margin holds with a multiplier, not the plain Euler equation).

The housing margin is not a kink to exclude here: the housing service flow is
additively separable, so the consumption Euler equation holds at every interior
unconstrained point regardless of whether the house is adjusted.

## The on-grid consumption policy

`E_t[u'(c_{t+1})]` needs the period-(t+1) consumption policy at the
*counterfactual* wage nodes the agent did not realise, so the realised sample
path alone is not enough. The policy is reconstructed from a second simulation
seeded over the full `(liquid, housing, wage)` product grid: each subject's
period-`p` row records its chosen consumption, giving per-`(period, wage)`
samples that are linearly interpolated in `(liquid, housing)` to evaluate
`c_{t+1}(a', H', wage_j)`.

pylcm's simulate is grid-restricted for **all** solvers (so is the App.1 FUES
column), so the magnitude at coarse grids is *coarse-grid*, not a methodological
ceiling — it falls monotonically as the grid refines. The full paper grids (NG in
{250...1000}, many wage nodes) exceed local memory and require a GPU/CI sweep,
exactly like App.1's `{1000...10000}`; the small grids here (n_grid <= ~20, <= 5
wage nodes) run locally because the lifecycle is short.
"""

import functools
import logging
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from lcm import TauchenAR1Process
from tests.test_models import ds_app2_housing as app2
from tests.test_models.ds_app2_housing import HousingRegimeId

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
    uses (`app2.N_WAGE_NODES` Gauss-Hermite nodes), so the harness and the solve
    discretise the AR1 wage identically. The Tauchen build-time grid is a NaN
    placeholder resolved from the runtime `(rho, sigma, mu)` parameters.
    """
    process = TauchenAR1Process(n_points=app2.N_WAGE_NODES, gauss_hermite=True)
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


@functools.cache
def _build_solved(
    *,
    n_grid: int,
    n_periods: int,
    n_consumption: int,
    tau: float,
    liquid_batch_size: int = 0,
):
    """Build and solve the App.2 NEGM housing model, caching the solution."""
    model = app2.build_model(
        n_grid=n_grid,
        n_periods=n_periods,
        n_consumption=n_consumption,
        liquid_batch_size=liquid_batch_size,
    )
    params = app2.build_params(tau=tau)
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def app2_negm_timing(
    *,
    n_grid: int,
    n_periods: int = N_PERIODS,
    n_consumption: int = 400,
    tau: float = 0.07,
    n_runs: int = 3,
    liquid_batch_size: int = 0,
) -> dict[str, float]:
    """Measure compile and steady-state run time of one NEGM solve.

    The first solve times compile-plus-run; later solves of the same model reuse
    the cached executable and time pure execution, so the compile cost is the
    difference. Every solve is forced to completion with `block_until_ready`.

    `liquid_batch_size > 0` chunks the liquid Euler grid to bound peak device
    memory at large `n_grid`; it leaves the solved value function unchanged.
    """
    model = app2.build_model(
        n_grid=n_grid,
        n_periods=n_periods,
        n_consumption=n_consumption,
        liquid_batch_size=liquid_batch_size,
    )
    params = app2.build_params(tau=tau)

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
        "DS App.2 NEGM timing: n_grid=%d -> compile=%.3fs run=%.3fs",
        n_grid,
        result["compile_time"],
        result["runtime"],
    )
    return result


def _liquid_housing_grids(model) -> tuple[np.ndarray, np.ndarray]:
    """Return the working regime's liquid and housing grid node arrays."""
    working = model.user_regimes["working"]
    liquid = np.asarray(working.states["liquid"].to_jax())
    housing = np.asarray(working.states["housing"].to_jax())
    return liquid, housing


# Cap on the per-axis resolution of the policy-reconstruction seed grid. The
# panel simulate argmaxes over (consumption x housing_investment) per seeded
# subject, so a full liquid x housing seed at paper NG (NG^2 subjects) blows up
# device memory; a capped seed grid keeps the panel small while still resolving
# the consumption policy for linear interpolation.
_MAX_POLICY_SEED_POINTS = 16


def _subsample(grid: np.ndarray, max_points: int) -> np.ndarray:
    """Return at most `max_points` evenly-spaced nodes of `grid`, ends included."""
    if len(grid) <= max_points:
        return grid
    index = np.unique(np.linspace(0, len(grid) - 1, max_points).round().astype(int))
    return grid[index]


def _policy_interpolators(
    *,
    model,
    params,
    solution,
    liquid_grid: np.ndarray,
    housing_grid: np.ndarray,
    wage_nodes: np.ndarray,
    n_periods: int,
    seed: int,
) -> dict[tuple[int, int], RegularGridInterpolator]:
    """Build a per-`(period, wage)` consumption interpolator over `(liquid, housing)`.

    The period-`p` consumption policy is read off a fresh simulation seeded over a
    `(liquid, housing, wage)` *regular* grid at period `p`'s age: the first
    simulated period then lands exactly on that regular grid, so its chosen
    consumption reshapes into a `RegularGridInterpolator` on `(liquid, housing)`
    per wage node. Seeding once at age 0 and reading later periods would not work
    — the agent evolves off the regular grid after period 0. The seed grid is
    subsampled to `_MAX_POLICY_SEED_POINTS` per axis so the panel stays within
    device memory at paper-scale NG; the policy is smooth enough that the coarser
    interpolation grid is adequate.
    """
    liquid_grid = _subsample(liquid_grid, _MAX_POLICY_SEED_POINTS)
    housing_grid = _subsample(housing_grid, _MAX_POLICY_SEED_POINTS)
    liquid = np.repeat(liquid_grid, len(housing_grid) * len(wage_nodes))
    housing = np.tile(np.repeat(housing_grid, len(wage_nodes)), len(liquid_grid))
    wage = np.tile(wage_nodes, len(liquid_grid) * len(housing_grid))
    n_subjects = liquid.size
    expected = len(liquid_grid) * len(housing_grid)
    ages = model.ages.exact_values

    interpolators: dict[tuple[int, int], RegularGridInterpolator] = {}
    for period in range(n_periods):
        initial_conditions = {
            "liquid": jnp.asarray(liquid),
            "housing": jnp.asarray(housing),
            "wage": jnp.asarray(wage),
            "age": jnp.full(n_subjects, float(ages[period])),
            "regime_id": jnp.full(n_subjects, HousingRegimeId.working, dtype=jnp.int32),
        }
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
        for wage_index, cell in panel.groupby("wage_index", observed=True):
            if len(cell) != expected:
                continue
            grid = (
                cell.sort_values(["liquid", "housing"])["consumption"]
                .to_numpy()
                .reshape(len(liquid_grid), len(housing_grid))
            )
            interpolators[(int(period), int(wage_index))] = RegularGridInterpolator(
                (liquid_grid, housing_grid),
                grid,
                bounds_error=False,
                fill_value=None,
            )
    return interpolators


def _expected_next_marginal_utility(
    *,
    interpolators: dict[tuple[int, int], RegularGridInterpolator],
    wage_transition: np.ndarray,
    source_wage_index: int,
    next_period: int,
    next_liquid: float,
    next_housing: float,
    n_wage_nodes: int,
    alpha: float,
    gamma_c: float,
) -> float:
    """Transition-weighted expected next-period consumption marginal utility.

    Sums `P[source_wage, j] * alpha*c_{next}(a', H', j)^{-gamma_C}` over the next-period
    wage nodes `j`. Returns NaN when any positive-probability next-wage cell has
    no interpolator (so the path point is dropped, not scored against a
    fabricated value).
    """
    expected_marginal = 0.0
    for next_wage_index in range(n_wage_nodes):
        probability = wage_transition[source_wage_index, next_wage_index]
        if probability <= 0.0:
            continue
        interpolator = interpolators.get((next_period, next_wage_index))
        if interpolator is None:
            return float("nan")
        next_consumption = float(interpolator([[next_liquid, next_housing]])[0])
        if not np.isfinite(next_consumption) or next_consumption <= 0.0:
            return float("nan")
        expected_marginal += probability * (alpha * next_consumption ** (-gamma_c))
    return expected_marginal


def sample_path_euler_error(
    *,
    sample_panel: pd.DataFrame,
    interpolators: dict[tuple[int, int], RegularGridInterpolator],
    wage_nodes: np.ndarray,
    wage_transition: np.ndarray,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    alpha: float = ALPHA,
    gamma_c: float = GAMMA_C,
) -> float:
    """Mean log10 stochastic consumption Euler error along a simulated path.

    For each interior, unconstrained working-to-working transition the conditional
    expectation of next-period marginal utility is summed over the next-period
    wage nodes weighted by the wage transition probabilities,

        E_t[u'(c_{t+1})]
            = sum_j P[wage_t, j] * alpha*c_{t+1}(a', H', wage_j)^{-gamma_C},

    with `c_{t+1}` the period-(t+1) consumption policy interpolated in
    `(liquid, housing)`. The implied consumption is
    `c_euler_t = (beta*(1+r)*E_t[u'(c_{t+1})] / alpha)^{-1/gamma_C}` and the metric is
    `mean(log10(|c_euler_t / c_t - 1|))`.

    Args:
        sample_panel: Long-format simulation panel of the scored sample path, with
            columns `subject_id`, `period`, `regime_name`, `liquid`, `housing`,
            `wage`, and `consumption`.
        interpolators: Mapping of `(period, wage_index)` to the period's
            consumption-policy `RegularGridInterpolator` over `(liquid, housing)`,
            from `_policy_interpolators`.
        wage_nodes: Array of discretised log-wage node values.
        wage_transition: Wage transition-probability matrix `P[i, j]`.
        interest_rate: Net liquid return `r`.
        discount_factor: Discount factor `beta`.
        alpha: Consumption weight in the separable CES utility.
        gamma_c: Consumption CRRA curvature `gamma_C` (so `u'(c) = alpha*c^{-gamma_C}`).

    Returns:
        The mean base-10 log relative consumption Euler error over the valid
        working-regime sample-path points.
    """
    working = sample_panel.query("regime_name == 'working'").copy()
    working = working.sort_values(["subject_id", "period"]).reset_index(drop=True)

    subject = working["subject_id"].to_numpy()
    period = working["period"].to_numpy()
    consumption = working["consumption"].to_numpy()
    liquid = working["liquid"].to_numpy()
    housing = working["housing"].to_numpy()
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
        next_housing = housing[t + 1]
        if next_liquid <= 1e-4 or consumption[t] <= 0.0:
            continue

        expected_marginal = _expected_next_marginal_utility(
            interpolators=interpolators,
            wage_transition=wage_transition,
            source_wage_index=int(wage_index[t]),
            next_period=int(period[t + 1]),
            next_liquid=float(next_liquid),
            next_housing=float(next_housing),
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


def app2_negm_euler_error(
    *,
    n_grid: int,
    n_periods: int = N_PERIODS,
    n_consumption: int = 400,
    n_subjects: int = 400,
    tau: float = 0.07,
    seed: int = 0,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    alpha: float = ALPHA,
    gamma_c: float = GAMMA_C,
    liquid_batch_size: int = 0,
) -> float:
    """Solve the App.2 NEGM housing model and score the consumption Euler error.

    Solves the continuous-housing model with the nested-EGM solver, simulates a
    sample path and a full-grid policy panel, and returns the mean log10
    stochastic consumption Euler error along the sample path. The wage being
    Markov, the Euler expectation is a transition-probability-weighted sum over
    next-period wage nodes (see `sample_path_euler_error`).

    Args:
        n_grid: Number of liquid/housing/outer grid points per axis.
        n_periods: Number of model periods.
        n_consumption: Number of inner consumption-grid points.
        n_subjects: Number of simulated sample paths.
        tau: Proportional housing-transaction cost.
        seed: Simulation seed.
        interest_rate: Net liquid return `r`.
        discount_factor: Discount factor `beta`.
        alpha: Consumption weight in the separable CES utility.
        gamma_c: Consumption CRRA curvature `gamma_C`.

    Returns:
        The mean log10 consumption Euler error along the simulated sample path.
    """
    model, params, solution = _build_solved(
        n_grid=n_grid,
        n_periods=n_periods,
        n_consumption=n_consumption,
        tau=tau,
        liquid_batch_size=liquid_batch_size,
    )
    wage_nodes, wage_transition = _wage_nodes_and_transition()
    liquid_grid, housing_grid = _liquid_housing_grids(model)

    sample_panel = _simulate_sample_path(
        model=model,
        params=params,
        solution=solution,
        n_subjects=n_subjects,
        liquid_grid=liquid_grid,
        housing_grid=housing_grid,
        wage_nodes=wage_nodes,
        seed=seed,
    )
    interpolators = _policy_interpolators(
        model=model,
        params=params,
        solution=solution,
        liquid_grid=liquid_grid,
        housing_grid=housing_grid,
        wage_nodes=wage_nodes,
        n_periods=n_periods,
        seed=seed,
    )

    error = sample_path_euler_error(
        sample_panel=sample_panel,
        interpolators=interpolators,
        wage_nodes=wage_nodes,
        wage_transition=wage_transition,
        interest_rate=interest_rate,
        discount_factor=discount_factor,
        alpha=alpha,
        gamma_c=gamma_c,
    )
    _logger.info(
        "DS App.2 NEGM Euler error: n_grid=%d -> %.3f",
        n_grid,
        error,
    )
    return error


def _simulate_sample_path(
    *,
    model,
    params,
    solution,
    n_subjects: int,
    liquid_grid: np.ndarray,
    housing_grid: np.ndarray,
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
        "housing": jnp.asarray(rng.choice(housing_grid[1:], n_subjects)),
        "wage": jnp.full(n_subjects, median_wage),
        "age": jnp.full(n_subjects, float(model.ages.exact_values[0])),
        "regime_id": jnp.full(n_subjects, HousingRegimeId.working, dtype=jnp.int32),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=seed,
    )
    return result.to_dataframe()


def app2_negm_accuracy_table(
    *,
    n_grids: Sequence[int] = (12, 18),
    n_periods: int = N_PERIODS,
    n_consumption: int = 400,
    n_subjects: int = 400,
    tau: float = 0.07,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the NEGM Euler-error sweep across per-axis grid sizes.

    One solve+simulate per grid size. The paper grids (NG in {250...1000}) exceed
    local memory and require a GPU sweep; pass smaller `n_grids` for a local run.

    Args:
        n_grids: Per-axis grid sizes to sweep.
        n_periods: Number of model periods.
        n_consumption: Number of inner consumption-grid points.
        n_subjects: Number of simulated sample paths per cell.
        tau: Proportional housing-transaction cost.
        seed: Simulation seed.

    Returns:
        Long-format DataFrame with columns `n_grid` and `negm_euler_error`.
    """
    rows = [
        {
            "n_grid": n_grid,
            "negm_euler_error": app2_negm_euler_error(
                n_grid=n_grid,
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
