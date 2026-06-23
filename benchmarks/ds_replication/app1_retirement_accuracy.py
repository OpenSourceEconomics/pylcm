"""Euler-error accuracy harness for Dobrescu & Shanker (2026) Application 1.

Application 1 of the FUES paper is the deterministic discrete-retirement model
"à la Iskhakov et al. (2017)": a worker chooses consumption and whether to keep
working or retire (retirement absorbing), with log utility, a per-period work
disutility `tau`, deterministic wage income while working, and a constant gross
return. This module recalibrates pylcm's Iskhakov-et-al.-2017 building blocks to
the paper's Table 2 parameters (`r = 0.02`, `beta = 0.96`, `T = 50`, `y = 20`,
`A_max = 500`) and reproduces the paper's FUES accuracy column.

FUES is pylcm's default DC-EGM upper envelope, so the accuracy column needs no
new solver: build the model, solve with DC-EGM, simulate a sample path, and
score the consumption Euler equation along it.

The Euler-error metric (Judd 1992) is the mean over a simulated sample path of
`log10` of the relative deviation between chosen consumption and the consumption
implied by the Euler equation. For log utility `u'(c) = 1/c` and the
deterministic baseline the conditional expectation collapses to the realized
next-period consumption, so

    c_euler_t = c_{t+1} / (beta * (1 + r)),   deviation_t = |c_euler_t / c_t - 1|,

and the metric is `mean(log10(deviation_t))` over the valid points. A point is
valid only where the agent works this period and next (the continuous Euler
equation governs the working regime; the retirement-switch period and the
retiree problem are excluded) and where consumption is interior and
unconstrained (the constraint margin holds with a multiplier, not the plain
Euler equation).

Use `app1_euler_error` for a single `(tau, n_grid)` cell and `app1_accuracy_table`
for the paper's `tau`-by-grid sweep. The full grids `{1000, 2000, 3000, 6000,
10000}` are GPU/CI-scale; pass smaller grids for a local run.
"""

import functools
import logging
from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
)
from lcm.regime import Regime
from lcm.solvers import DCEGM
from lcm_examples.iskhakov_et_al_2017 import (
    LaborSupply,
    RegimeId,
    dead,
    inverse_marginal_utility,
    is_working,
    labor_income,
    next_regime_from_retirement,
    next_regime_from_working,
    next_wealth_from_savings,
    resources,
    savings,
    utility_retirement,
    utility_working,
)

_logger = logging.getLogger("lcm")

# Paper Table 2 calibration (deterministic baseline, no taste shock).
ASSET_MAX = 500.0
INTEREST_RATE = 0.02
DISCOUNT_FACTOR = 0.96
WAGE = 20.0
N_PERIODS = 50

# The paper's reported FUES grids for Application 1.
PAPER_GRIDS = (1000, 2000, 3000, 6000, 10000)
PAPER_TAUS = (0.25, 0.50, 1.00, 2.00)


@functools.cache
def build_app1_model(
    *,
    n_grid: int,
    n_periods: int = N_PERIODS,
    asset_max: float = ASSET_MAX,
    upper_envelope: Literal["fues", "rfc"] = "fues",
) -> Model:
    """Build the DS-2026 Application 1 model solved by DC-EGM/FUES.

    Args:
        n_grid: Number of financial-asset (wealth) grid points; also the
            number of clustered exogenous savings nodes the DC-EGM solver scans.
        n_periods: Number of model periods. The final period is the terminal
            `dead` regime, so the number of decision periods is `n_periods - 1`.
        asset_max: Upper bound of the asset grid `a in [0, asset_max]`.

    Returns:
        A configured `Model` with a worker, an absorbing retiree, and a terminal
        dead regime, all DC-EGM regimes scanning the same savings grid.

    """
    wealth_grid = LinSpacedGrid(start=1.0, stop=asset_max, n_points=n_grid)
    # Consumption never enters the DC-EGM solve (Euler inversion replaces the
    # grid search); the action grid only bounds the simulated consumption draw,
    # so it tracks the wealth resolution.
    consumption_grid = LinSpacedGrid(start=1.0, stop=asset_max, n_points=n_grid)
    # Cubically clustered savings nodes toward the borrowing limit: the value
    # function curves hardest where the constraint starts to bind, and a
    # uniform grid under-resolves the lowest wealth nodes by orders of
    # magnitude. The lower bound (savings >= 0) encodes `consumption <= wealth`.
    savings_grid = IrregSpacedGrid(
        points=tuple(asset_max * (i / (n_grid - 1)) ** 3 for i in range(n_grid))
    )
    solver = DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=savings_grid,
        upper_envelope=upper_envelope,
        # The final decision period consumes everything, so its carry in the
        # queried resources range is constrained-segment points only; a small
        # count keeps the geometric spacing ratio and the carry interpolation
        # error small.
        n_constrained_points=64,
    )

    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    last_age = ages.exact_values[-1]

    working_life = Regime(
        transition=next_regime_from_working,
        actions={
            "labor_supply": DiscreteGrid(LaborSupply),
            "consumption": consumption_grid,
        },
        states={"wealth": wealth_grid},
        state_transitions={"wealth": next_wealth_from_savings},
        functions={
            "utility": utility_working,
            "labor_income": labor_income,
            "is_working": is_working,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=solver,
        active=lambda age, la=last_age: age < la,
    )
    retirement = Regime(
        transition=next_regime_from_retirement,
        actions={"consumption": consumption_grid},
        states={"wealth": wealth_grid},
        state_transitions={"wealth": next_wealth_from_savings},
        functions={
            "utility": utility_retirement,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=solver,
        active=lambda age, la=last_age: age < la,
    )

    return Model(
        regimes={
            "working_life": working_life,
            "retirement": retirement,
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )


def build_app1_params(
    *,
    tau: float,
    n_periods: int = N_PERIODS,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    wage: float = WAGE,
) -> dict:
    """Build the parameter dict for the Application 1 model.

    Args:
        tau: Per-period utility cost of working (`u(c) = log(c) - tau`).
        n_periods: Number of model periods (must match `build_app1_model`).
        interest_rate: Net asset return `r`.
        discount_factor: Discount factor `beta`.
        wage: Deterministic per-period wage while working.

    Returns:
        Parameter dict ready for `model.solve()` / `model.simulate()`.

    """
    # All but the final period is a decision period; death arrives at the last
    # decision age, so `final_age_alive` is the second-to-last age.
    final_age_alive = n_periods - 2
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        "final_age_alive": final_age_alive,
        "working_life": {
            "utility": {"disutility_of_work": tau},
            "labor_income": {"wage": wage},
        },
        "retirement": {
            "next_wealth": {"labor_income": 0.0},
        },
    }


def _initial_conditions(*, n_subjects: int, asset_max: float):
    """Spread initial wealth so the sample path spans early and late retirement."""
    return {
        "wealth": jnp.linspace(1.0, asset_max / 5.0, n_subjects),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, RegimeId.working_life, dtype=jnp.int32),
    }


def sample_path_euler_error(
    *,
    panel: pd.DataFrame,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
) -> float:
    """Mean log10 consumption Euler error along a simulated working-regime path.

    For log utility the Euler equation `u'(c_t) = beta*(1+r)*u'(c_{t+1})` implies
    `c_euler_t = c_{t+1} / (beta*(1+r))`, and the metric is the mean over valid
    points of `log10(|c_euler_t / c_t - 1|)`. A point is valid only where the
    subject works this period and the next (the working-regime continuous Euler
    equation; the retirement switch and the retiree problem are excluded) and
    where consumption is interior and unconstrained. The borrowing-constrained
    points — where consumption exhausts the period's resources — are dropped
    because the Euler equation there holds with a constraint multiplier.

    Args:
        panel: Long-format simulation panel with columns `subject_id`, `period`,
            `regime_name`, `labor_supply`, `consumption`, and `wealth`.
        interest_rate: Net asset return `r`.
        discount_factor: Discount factor `beta`.

    Returns:
        The mean base-10 log relative consumption Euler error over the valid
        working-regime sample-path points.

    """
    working = panel.query("regime_name == 'working_life'").copy()
    working = working.sort_values(["subject_id", "period"])

    consumption = working["consumption"].to_numpy()
    wealth = working["wealth"].to_numpy()
    labor = working["labor_supply"].to_numpy()
    subject = working["subject_id"].to_numpy()
    period = working["period"].to_numpy()

    next_consumption = np.full_like(consumption, np.nan)
    same_subject = subject[:-1] == subject[1:]
    consecutive = period[1:] == period[:-1] + 1
    has_next = np.zeros(consumption.shape, dtype=bool)
    has_next[:-1] = same_subject & consecutive
    next_consumption[:-1] = np.where(has_next[:-1], consumption[1:], np.nan)

    works_this = labor == "work"
    works_next = np.zeros(consumption.shape, dtype=bool)
    works_next[:-1] = has_next[:-1] & (labor[1:] == "work")

    # Unconstrained interior: consumption leaves strictly positive savings, so
    # the borrowing constraint does not bind. A tiny absolute floor guards the
    # finite-grid edge where chosen consumption equals available resources.
    unconstrained = consumption < wealth - 1e-6

    valid = works_this & works_next & has_next & unconstrained
    if not valid.any():
        msg = "No valid interior working-regime points to score the Euler error."
        raise ValueError(msg)

    consumption_valid = consumption[valid]
    next_consumption_valid = next_consumption[valid]
    consumption_euler = next_consumption_valid / (
        discount_factor * (1.0 + interest_rate)
    )
    relative_error = np.abs(consumption_euler / consumption_valid - 1.0)
    log10_error = np.log10(relative_error)
    finite = np.isfinite(log10_error)
    return float(np.mean(log10_error[finite]))


def app1_euler_error(
    *,
    tau: float,
    n_grid: int,
    n_periods: int = N_PERIODS,
    n_subjects: int = 1000,
    seed: int = 0,
    asset_max: float = ASSET_MAX,
    interest_rate: float = INTEREST_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    wage: float = WAGE,
    upper_envelope: Literal["fues", "rfc"] = "fues",
) -> float:
    """Solve, simulate, and score the FUES Euler error for one `(tau, n_grid)`.

    Args:
        tau: Per-period utility cost of working.
        n_grid: Number of financial-asset grid points.
        n_periods: Number of model periods.
        n_subjects: Number of simulated sample paths.
        seed: Simulation seed.
        asset_max: Upper bound of the asset grid.
        interest_rate: Net asset return `r`.
        discount_factor: Discount factor `beta`.
        wage: Deterministic per-period wage while working.

    Returns:
        The mean log10 consumption Euler error along the simulated sample path.

    """
    model = build_app1_model(
        n_grid=n_grid,
        n_periods=n_periods,
        asset_max=asset_max,
        upper_envelope=upper_envelope,
    )
    params = build_app1_params(
        tau=tau,
        n_periods=n_periods,
        interest_rate=interest_rate,
        discount_factor=discount_factor,
        wage=wage,
    )
    result = model.simulate(
        params=params,
        initial_conditions=_initial_conditions(
            n_subjects=n_subjects, asset_max=asset_max
        ),
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=seed,
    )
    panel = result.to_dataframe()
    euler_error = sample_path_euler_error(
        panel=panel,
        interest_rate=interest_rate,
        discount_factor=discount_factor,
    )
    _logger.info(
        "DS App.1 %s Euler error: tau=%.2f n_grid=%d -> %.3f",
        upper_envelope.upper(),
        tau,
        n_grid,
        euler_error,
    )
    return euler_error


def app1_accuracy_table(
    *,
    taus: Sequence[float] = PAPER_TAUS,
    n_grids: Sequence[int] = PAPER_GRIDS,
    n_periods: int = N_PERIODS,
    n_subjects: int = 1000,
    seed: int = 0,
    upper_envelope: Literal["fues", "rfc"] = "fues",
) -> pd.DataFrame:
    """Run the `tau`-by-grid Euler-error sweep, reproducing the FUES column.

    One solve+simulate per `(tau, n_grid)` cell. The full paper grids are
    GPU/CI-scale; pass smaller `n_grids` for a local run.

    Args:
        taus: Work-cost values to sweep.
        n_grids: Asset-grid sizes to sweep.
        n_periods: Number of model periods.
        n_subjects: Number of simulated sample paths per cell.
        seed: Simulation seed.

    Returns:
        Long-format DataFrame with columns `tau`, `n_grid`, and
        `fues_euler_error`.

    """
    rows = [
        {
            "tau": tau,
            "n_grid": n_grid,
            f"{upper_envelope}_euler_error": app1_euler_error(
                tau=tau,
                n_grid=n_grid,
                n_periods=n_periods,
                n_subjects=n_subjects,
                seed=seed,
                upper_envelope=upper_envelope,
            ),
        }
        for tau in taus
        for n_grid in n_grids
    ]
    return pd.DataFrame(rows)
