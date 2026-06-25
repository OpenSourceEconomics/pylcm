"""Euler-error accuracy harness for Dobrescu & Shanker (2026) Application 3.

Application 3 of the FUES paper is the discrete-housing model of DS Section 2.3
(an extended Fella 2014): a finite-horizon consumption-savings problem with a
*discrete* housing stock, an own-vs-rent choice, a proportional
housing-adjustment cost, and a **stochastic (Markov) wage**. This module scores
the **VFI / grid-search (GridSearch)** column of the paper's no-tax accuracy
tables (Table 4 / Table 7): it builds the brute variant of
`tests.test_models.ds_app3_discrete_housing`, solves it by backward induction,
simulates a sample path, and scores the consumption Euler equation along it.

## The stochastic Euler equation

The interior, unconstrained consumption Euler equation is

    u'(c_t) = beta * (1 + r) * E_t[u'(c_{t+1})],

with `u(c, h) = alpha * log(c) + (1 - alpha) * log(kappa * (h + iota))`, so the
consumption marginal utility is `u'(c) = alpha / c` and the housing-service term
drops out. The **key difference from Application 1** is the wage: it is a Markov
process, so the conditional expectation `E_t[u'(c_{t+1})]` is *not* the realized
next-period marginal utility — it is a sum over next-period wage nodes weighted
by the wage transition probabilities,

    E_t[u'(c_{t+1})] = sum_j P[wage_node_t, j] * u'( c_{t+1}(a', H', wage_j) ),

evaluated at the chosen post-decision financial assets `a'` and next-period
housing `H' = housing_choice` held fixed, with `c_{t+1}(., H', wage_j)` the
period-(t+1) consumption policy on the asset grid for that `(housing, wage)`
cell. The implied consumption is then

    c_euler_t = alpha / ( beta * (1 + r) * E_t[u'(c_{t+1})] ),

and the metric is `mean(log10(|c_euler_t / c_t - 1|))` over the valid points
(Judd 1992).

## Valid points

Mirroring Application 1, a path point is scored only where the continuous
consumption Euler equation governs the optimum:

- the subject is in the working regime this period and the next (the terminal
  bequest regime is excluded);
- consumption is interior and unconstrained — chosen consumption leaves strictly
  positive financial savings, so the no-borrowing constraint does not bind;
- housing is *not* adjusted this period and is *not* adjusted next period
  (`housing == housing_choice` at both `t` and `t+1`) — the discrete
  housing-switch margin is a kink in the value function, exactly like
  Application 1 excludes the work/retire switch.

## The on-grid consumption policy

`E_t[u'(c_{t+1})]` needs the period-(t+1) consumption policy evaluated at the
*counterfactual* wage nodes the agent did not realise, so the realized sample
path alone is not enough. The policy is reconstructed from a second simulation
seeded at the full `(assets, housing, wage)` product grid: each subject's
period-`p` row records `c_p` at its period-`p` state, giving per-period
`(housing, wage) -> [(assets, consumption)]` samples that are linearly
interpolated in assets to evaluate `c_{t+1}(a', H', wage_j)`.

The full paper grids ({2000, 4000, ...} asset points, 7 wage nodes, T=20) are a
GPU/CI sweep; the small grids here (asset points <= ~80, wage nodes <= 5) are
local-safe because T=20 is short. The FUES/MSS/LTM columns of Table 4 are
kernel-gated on `feat/dcegm` and are *not* produced here — this module produces
the VFI column only.
"""

import functools
import logging
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from lcm import RouwenhorstAR1Process
from tests.test_models import ds_app3_discrete_housing as app3
from tests.test_models.ds_app3_discrete_housing import (
    DiscreteHousingRegimeId,
    Housing,
)

_logger = logging.getLogger("lcm")

# Paper Table 4/7 calibration (no tax).
INTEREST_RATE = 0.06
DISCOUNT_FACTOR = 0.93
ALPHA = 0.77
RHO_W = 0.977
SIGMA_W = 0.063
MU_W = 0.0
ASSET_MAX = 40.0
N_PERIODS = 20


def _wage_nodes_and_transition(
    *,
    n_wage_nodes: int,
    rho: float = RHO_W,
    sigma: float = SIGMA_W,
    mu: float = MU_W,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Markov wage node values and the transition-probability matrix.

    The node values are the discretised log-wage grid and the matrix `P[i, j]`
    is the probability of moving from wage node `i` to node `j`, both from the
    same Rouwenhorst discretisation the model's wage process uses.
    """
    process = RouwenhorstAR1Process(n_points=n_wage_nodes, rho=rho, sigma=sigma, mu=mu)
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


def _full_grid_initial_conditions(
    *,
    n_assets: int,
    asset_max: float,
    wage_nodes: np.ndarray,
) -> dict:
    """Seed one subject at every `(assets, housing, wage)` product-grid cell.

    Simulating from this seeding records, at each period, the consumption policy
    at every discrete `(housing, wage)` cell over a spread of asset values — the
    on-grid policy lookup the stochastic Euler expectation interpolates.
    """
    asset_nodes = np.linspace(0.0, asset_max, n_assets)
    housing_codes = np.arange(len(Housing.__annotations__))
    assets = np.repeat(asset_nodes, len(housing_codes) * len(wage_nodes))
    housing = np.tile(np.repeat(housing_codes, len(wage_nodes)), len(asset_nodes))
    wage = np.tile(wage_nodes, len(asset_nodes) * len(housing_codes))
    n_subjects = assets.size
    return {
        "assets": jnp.asarray(assets),
        "housing": jnp.asarray(housing, dtype=jnp.int32),
        "wage": jnp.asarray(wage),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(
            n_subjects, DiscreteHousingRegimeId.working, dtype=jnp.int32
        ),
    }


def _policy_lookup(
    *,
    policy_panel: pd.DataFrame,
    wage_nodes: np.ndarray,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """Build a per-`(period, housing, wage)` sorted `(assets, consumption)` policy.

    Each cell maps to the sorted financial-asset samples and the consumption the
    policy chose there, ready for linear interpolation at an arbitrary `a'`.
    """
    working = policy_panel.query("regime_name == 'working'").copy()
    housing_codes = {name: idx for idx, name in enumerate(Housing.__annotations__)}
    working["housing_code"] = working["housing"].map(housing_codes).to_numpy()
    working["wage_index"] = _nearest_wage_index(working["wage"].to_numpy(), wage_nodes)

    lookup: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    grouped = working.groupby(["period", "housing_code", "wage_index"], observed=True)
    for (period, housing_code, wage_index), cell in grouped:
        order = np.argsort(cell["assets"].to_numpy())
        assets = cell["assets"].to_numpy()[order]
        consumption = cell["consumption"].to_numpy()[order]
        lookup[(int(period), int(housing_code), int(wage_index))] = (
            assets,
            consumption,
        )
    return lookup


def _interp_policy(
    *,
    lookup: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    period: int,
    housing_code: int,
    wage_index: int,
    assets: float,
) -> float:
    """Linearly interpolate the consumption policy of a cell at `assets`.

    Returns NaN when the cell has no policy samples (the agent never visits that
    `(period, housing, wage)` cell in the full-grid simulation), so the point is
    dropped from the Euler score rather than scored against a fabricated value.
    """
    key = (period, housing_code, wage_index)
    if key not in lookup:
        return float("nan")
    asset_samples, consumption_samples = lookup[key]
    if asset_samples.size == 0:
        return float("nan")
    return float(np.interp(assets, asset_samples, consumption_samples))


def _expected_next_marginal_utility(
    *,
    lookup: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    wage_transition: np.ndarray,
    source_wage_index: int,
    next_period: int,
    next_housing_code: int,
    next_assets: float,
    n_wage_nodes: int,
    alpha: float,
) -> float:
    """Transition-weighted expected next-period consumption marginal utility.

    Sums `P[source_wage, j] * alpha / c_{next}(next_assets, next_housing, j)`
    over the next-period wage nodes `j`. Returns NaN when any positive-probability
    next-wage cell has no policy sample (so the path point is dropped rather than
    scored against a fabricated value).
    """
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
            assets=next_assets,
        )
        if not np.isfinite(next_consumption) or next_consumption <= 0.0:
            return float("nan")
        expected_marginal += probability * (alpha / next_consumption)
    return expected_marginal


def _euler_marginal_return(
    *, next_assets: float, interest_rate: float, use_taxes: bool
) -> float:
    """Marginal return on a unit of saving, `1 + r - T'(a')`.

    Without taxes the gross return `1 + r`. With the piecewise-linear capital-income
    tax `T`, a unit of saving nets the realized bracket marginal rate
    `tau_k = T'(a')`, so the interior consumption Euler equation discounts by
    `1 + r - tau_k`. The bracket of `a'` is selected left-closed, matching
    `piecewise_capital_income_tax`.
    """
    if not use_taxes:
        return 1.0 + interest_rate
    bracket = min(
        max(
            int(np.searchsorted(app3.TAX_BRACKET_LOWER, next_assets, side="right")) - 1,
            0,
        ),
        len(app3.TAX_BRACKET_LOWER) - 1,
    )
    return 1.0 + interest_rate - app3.TAX_BRACKET_RATE[bracket]


def _is_near_tax_notch(*, next_assets: float, tol: float) -> bool:
    """Whether `next_assets` sits within `tol` of a tax-bracket boundary.

    At a bracket boundary the capital-income tax's level jump makes `T'` ill-defined
    (a one-sided derivative against a discontinuity), so the smooth Euler equality
    does not apply and the point is excluded from the with-tax score.
    """
    return any(abs(next_assets - lower) <= tol for lower in app3.TAX_BRACKET_LOWER[1:])


def sample_path_euler_error(  # noqa: C901
    *,
    sample_panel: pd.DataFrame,
    policy_panel: pd.DataFrame,
    wage_nodes: np.ndarray,
    wage_transition: np.ndarray,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    alpha: float = ALPHA,
    use_taxes: bool = False,
    tax_notch_tol: float = 1e-3,
) -> float:
    """Mean log10 stochastic consumption Euler error along a simulated path.

    For each interior, unconstrained, non-housing-switch working-to-working
    transition the conditional expectation of next-period marginal utility is
    summed over the next-period wage nodes weighted by the wage transition
    probabilities,

        E_t[u'(c_{t+1})] = sum_j P[wage_t, j] * alpha / c_{t+1}(a', H', wage_j),

    with `c_{t+1}(., H', wage_j)` the period-(t+1) consumption policy of the
    `(H', wage_j)` cell interpolated at the chosen post-decision financial
    assets `a' = assets_t - consumption_t + (1 + r) * (...)` read off the path as
    the next period's `assets`. The implied consumption is
    `c_euler_t = alpha / (beta * (1 + r - T'(a')) * E_t[u'(c_{t+1})])` and the metric
    is `mean(log10(|c_euler_t / c_t - 1|))`. Without taxes `T' = 0` and the return is
    the gross `1 + r`; with taxes the realized bracket marginal rate enters and points
    whose `a'` sits on a bracket boundary are excluded.

    Args:
        sample_panel: Long-format simulation panel of the scored sample path,
            with columns `subject_id`, `period`, `regime_name`, `assets`,
            `housing`, `wage`, `consumption`, and `housing_choice`.
        policy_panel: Long-format simulation panel seeded over the full
            `(assets, housing, wage)` grid, used to reconstruct the per-period
            on-grid consumption policy.
        wage_nodes: Array of discretised log-wage node values.
        wage_transition: Wage transition-probability matrix `P[i, j]`.
        interest_rate: Net financial return `r`.
        discount_factor: Discount factor `beta`.
        alpha: Consumption weight in the Cobb-Douglas log utility (so
            `u'(c) = alpha / c`).
        use_taxes: Whether the model carries the piecewise capital-income tax. When
            `True`, the Euler return is `1 + r - T'(a')` (the realized bracket
            marginal rate) and points whose `a'` lands on a bracket boundary are
            dropped; when `False`, the gross return `1 + r` is used (Table 4/7 path).
        tax_notch_tol: Half-width of the excluded neighborhood around each tax-bracket
            boundary, in asset units. Only consulted when `use_taxes` is `True`.

    Returns:
        The mean base-10 log relative consumption Euler error over the valid
        working-regime sample-path points.
    """
    lookup = _policy_lookup(policy_panel=policy_panel, wage_nodes=wage_nodes)
    housing_codes = {name: idx for idx, name in enumerate(Housing.__annotations__)}

    working = sample_panel.query("regime_name == 'working'").copy()
    working = working.sort_values(["subject_id", "period"]).reset_index(drop=True)

    subject = working["subject_id"].to_numpy()
    period = working["period"].to_numpy()
    consumption = working["consumption"].to_numpy()
    assets = working["assets"].to_numpy()
    wage_value = working["wage"].to_numpy()
    housing = working["housing"].to_numpy()
    housing_choice = working["housing_choice"].to_numpy()
    wage_index = _nearest_wage_index(wage_value, wage_nodes)

    # The next row is the same subject's next period when it is consecutive and
    # the same subject.
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
        # Interior, unconstrained: the post-decision financial savings carried
        # forward as next period's assets are strictly positive, so chosen
        # consumption did not exhaust resources and the no-borrowing constraint
        # is slack. The constrained margin holds with a multiplier, not the plain
        # Euler equation, so it is dropped.
        next_assets = assets[t + 1]
        if next_assets <= 1e-6:
            continue
        # At a tax-bracket boundary `T'` is ill-defined (one-sided against the level
        # jump), so the smooth Euler equality does not apply there.
        if use_taxes and _is_near_tax_notch(
            next_assets=float(next_assets), tol=tax_notch_tol
        ):
            continue
        # No housing switch this period and next: the discrete housing-adjustment
        # margin is a value-function kink the smooth Euler equation skips.
        if housing[t] != housing_choice[t]:
            continue
        if housing[t + 1] != housing_choice[t + 1]:
            continue

        expected_marginal = _expected_next_marginal_utility(
            lookup=lookup,
            wage_transition=wage_transition,
            source_wage_index=int(wage_index[t]),
            next_period=int(period[t + 1]),
            next_housing_code=housing_codes[housing_choice[t]],
            next_assets=float(next_assets),
            n_wage_nodes=wage_nodes.size,
            alpha=alpha,
        )
        if not np.isfinite(expected_marginal) or expected_marginal <= 0.0:
            continue

        marginal_return = _euler_marginal_return(
            next_assets=float(next_assets),
            interest_rate=interest_rate,
            use_taxes=use_taxes,
        )
        consumption_euler = alpha / (
            discount_factor * marginal_return * expected_marginal
        )
        relative_error = abs(consumption_euler / consumption[t] - 1.0)
        if relative_error > 0.0:
            errors.append(np.log10(relative_error))

    if not errors:
        msg = "No valid interior working-regime points to score the Euler error."
        raise ValueError(msg)
    return float(np.mean(errors))


@functools.cache
def _build_solved(
    *,
    n_assets: int,
    n_wage_nodes: int,
    n_periods: int,
    n_consumption: int,
    asset_max: float,
    tau: float,
    variant: str = "brute",
    upper_envelope: str = "fues",
    use_taxes: bool = False,
):
    """Build and solve the App.3 model (brute or DC-EGM), caching the solution."""
    model = app3.build_model(
        variant=variant,
        n_assets=n_assets,
        n_wage_nodes=n_wage_nodes,
        n_periods=n_periods,
        n_consumption=n_consumption,
        asset_max=asset_max,
        upper_envelope=upper_envelope,
        use_taxes=use_taxes,
    )
    params = app3.build_params(
        variant=variant, n_periods=n_periods, tau=tau, use_taxes=use_taxes
    )
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def app3_timing(
    *,
    n_assets: int,
    n_wage_nodes: int = 5,
    n_periods: int = N_PERIODS,
    n_consumption: int = 60,
    asset_max: float = ASSET_MAX,
    tau: float = 0.07,
    variant: str = "dcegm",
    upper_envelope: str = "fues",
    use_taxes: bool = False,
    n_runs: int = 3,
) -> dict[str, float]:
    """Measure compile and steady-state run time of one App.3 solve.

    The first solve times compile-plus-run; later solves of the same model reuse
    the cached executable and time pure execution, so the compile cost is the
    difference. Every solve is forced to completion with `block_until_ready`.
    """
    model = app3.build_model(
        variant=variant,
        n_assets=n_assets,
        n_wage_nodes=n_wage_nodes,
        n_periods=n_periods,
        n_consumption=n_consumption,
        asset_max=asset_max,
        upper_envelope=upper_envelope,
        use_taxes=use_taxes,
    )
    params = app3.build_params(
        variant=variant, n_periods=n_periods, tau=tau, use_taxes=use_taxes
    )

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
        "DS App.3 %s timing (use_taxes=%s): n_assets=%d -> compile=%.3fs run=%.3fs",
        upper_envelope.upper() if variant == "dcegm" else "VFI",
        use_taxes,
        n_assets,
        result["compile_time"],
        result["runtime"],
    )
    return result


def app3_vfi_euler_error(
    *,
    n_assets: int,
    n_wage_nodes: int = 5,
    n_periods: int = N_PERIODS,
    n_consumption: int = 60,
    n_subjects: int = 400,
    asset_max: float = ASSET_MAX,
    tau: float = 0.07,
    seed: int = 0,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    alpha: float = ALPHA,
    variant: str = "brute",
    upper_envelope: str = "fues",
    use_taxes: bool = False,
) -> float:
    """Solve the App.3 model (brute or DC-EGM) and score the consumption Euler error.

    Solves the discrete-housing model by grid search (VFI), simulates a sample
    path and a full-grid policy panel, and returns the mean log10 stochastic
    consumption Euler error along the sample path. The wage being Markov, the
    Euler expectation is a transition-probability-weighted sum over next-period
    wage nodes (see `sample_path_euler_error`).

    Args:
        n_assets: Number of financial-asset grid points.
        n_wage_nodes: Number of Markov wage discretisation nodes.
        n_periods: Number of model periods (the paper uses `T = 20`).
        n_consumption: Number of consumption-grid points the grid search scans.
        n_subjects: Number of simulated sample paths.
        asset_max: Upper bound of the financial-asset grid.
        tau: Proportional housing-adjustment cost.
        seed: Simulation seed.
        interest_rate: Net financial return `r`.
        discount_factor: Discount factor `beta`.
        alpha: Consumption weight in the Cobb-Douglas log utility.

    Returns:
        The mean log10 consumption Euler error along the simulated sample path.
    """
    model, params, solution = _build_solved(
        n_assets=n_assets,
        n_wage_nodes=n_wage_nodes,
        n_periods=n_periods,
        n_consumption=n_consumption,
        asset_max=asset_max,
        tau=tau,
        variant=variant,
        upper_envelope=upper_envelope,
        use_taxes=use_taxes,
    )
    wage_nodes, wage_transition = _wage_nodes_and_transition(n_wage_nodes=n_wage_nodes)

    sample_panel = _simulate_sample_path(
        model=model,
        params=params,
        solution=solution,
        n_subjects=n_subjects,
        asset_max=asset_max,
        wage_nodes=wage_nodes,
        seed=seed,
    )
    policy_panel = _simulate_policy_panel(
        model=model,
        params=params,
        solution=solution,
        n_assets=n_assets,
        asset_max=asset_max,
        wage_nodes=wage_nodes,
        seed=seed,
    )

    error = sample_path_euler_error(
        sample_panel=sample_panel,
        policy_panel=policy_panel,
        wage_nodes=wage_nodes,
        wage_transition=wage_transition,
        interest_rate=interest_rate,
        discount_factor=discount_factor,
        alpha=alpha,
        use_taxes=use_taxes,
    )
    _logger.info(
        "DS App.3 VFI Euler error: n_assets=%d n_wage=%d -> %.3f",
        n_assets,
        n_wage_nodes,
        error,
    )
    return error


def _simulate_sample_path(
    *,
    model,
    params,
    solution,
    n_subjects: int,
    asset_max: float,
    wage_nodes: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Simulate a spread of subjects starting as renters at the median wage."""
    median_wage = float(wage_nodes[len(wage_nodes) // 2])
    initial_conditions = {
        "assets": jnp.linspace(0.5, asset_max / 4.0, n_subjects),
        "housing": jnp.full(n_subjects, Housing.rent, dtype=jnp.int32),
        "wage": jnp.full(n_subjects, median_wage),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(
            n_subjects, DiscreteHousingRegimeId.working, dtype=jnp.int32
        ),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=seed,
    )
    return result.to_dataframe()


def _simulate_policy_panel(
    *,
    model,
    params,
    solution,
    n_assets: int,
    asset_max: float,
    wage_nodes: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Simulate from the full product grid to reconstruct the on-grid policy."""
    initial_conditions = _full_grid_initial_conditions(
        n_assets=n_assets, asset_max=asset_max, wage_nodes=wage_nodes
    )
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=seed,
    )
    return result.to_dataframe()


def app3_vfi_accuracy_table(
    *,
    n_assets_grid: Sequence[int] = (40, 60, 80),
    n_wage_nodes: int = 5,
    n_periods: int = N_PERIODS,
    n_consumption: int = 60,
    n_subjects: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the VFI Euler-error sweep across financial-asset grid sizes.

    One solve+simulate per asset-grid size. The columns mirror the paper's
    accuracy tables but report the VFI column only — the FUES/MSS/LTM columns are
    kernel-gated on `feat/dcegm`.

    Args:
        n_assets_grid: Financial-asset grid sizes to sweep.
        n_wage_nodes: Number of Markov wage discretisation nodes.
        n_periods: Number of model periods.
        n_consumption: Number of consumption-grid points the grid search scans.
        n_subjects: Number of simulated sample paths per cell.
        seed: Simulation seed.

    Returns:
        Long-format DataFrame with columns `n_assets`, `n_wage_nodes`, and
        `vfi_euler_error`.
    """
    rows = [
        {
            "n_assets": n_assets,
            "n_wage_nodes": n_wage_nodes,
            "vfi_euler_error": app3_vfi_euler_error(
                n_assets=n_assets,
                n_wage_nodes=n_wage_nodes,
                n_periods=n_periods,
                n_consumption=n_consumption,
                n_subjects=n_subjects,
                seed=seed,
            ),
        }
        for n_assets in n_assets_grid
    ]
    return pd.DataFrame(rows)
