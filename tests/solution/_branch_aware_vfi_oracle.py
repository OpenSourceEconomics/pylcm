"""Branch-aware VFI oracle for the DS App.2 discrete-housing model.

A self-contained host-side NumPy value-function iteration for the discrete-housing
EGM-FUES model (`tests.test_models.ds_app2_housing_fues`). It exists to isolate the
*comparator-ordering* difference between the two ways a discrete-choice dynamic
program can read its continuation value across periods:

- **standard VFI** (`branch_aware=False`) stores each period's value as the hard
  max over the next-housing choice and linearly interpolates that already-maximized
  array in the liquid Euler state — the operator `I[max_d V_d]`. This is exactly the
  grid-search (brute) twin pylcm runs for the EGM-FUES column.
- **branch-aware VFI** (`branch_aware=True`) keeps the per-next-housing-choice value
  rows, interpolates each in the liquid state, and takes the hard max *after*
  interpolating — the operator `max_d I[V_d]`. This is exactly the discrete-choice
  DC-EGM continuation.

Because a linear interpolant of a pointwise maximum dominates the pointwise maximum
of the interpolants, `max_d I[V_d] <= I[max_d V_d]` everywhere, with strict
inequality only where the winning next-housing choice switches inside a liquid
bracket — the DS (S, s) inaction band. So the standard VFI sits at or above the
branch-aware VFI, and the gap is concentrated exactly at the (S, s) switch cells.

The oracle mirrors the model's economics in NumPy and pulls the wage discretisation
straight from the same `TauchenAR1Process` the model uses, so the standard mode
reproduces pylcm's brute solve to floating-point precision. Differencing the two
modes therefore isolates the pure comparator-ordering term. In the App.2
calibration the next-housing choice only switches in the low-wealth (S, s) band, so
that term is identically zero on the liquid interior the Table 3 score uses — the
interior dcegm-vs-brute disagreement is a separate, DC-EGM-side discretization
effect, not comparator ordering. It is a test tool only — never a registered solver.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from lcm import TauchenAR1Process
from tests.test_models.ds_app2_housing_fues import (
    N_WAGE_NODES,
    RETIREMENT_AGE,
    START_AGE,
    TERMINAL_AGE,
)


@dataclass(frozen=True)
class _Schedule:
    """Per-period regime layout derived from the model's lifecycle anchors."""

    regime: tuple[str, ...]
    """Regime active at each period (`"working"`, `"retired"`, or `"dead"`)."""
    target: tuple[str | None, ...]
    """Regime the period's decision transitions into (`None` for the terminal)."""


def _build_schedule(*, n_periods: int | None) -> _Schedule:
    """Derive the per-period regime and transition target from the lifecycle.

    Replicates the age thresholds `build_model` computes so the oracle's backward
    induction visits the same regimes in the same order as the pylcm solve.
    """
    if n_periods is None:
        n_ages = TERMINAL_AGE - START_AGE + 1
        retirement_age = RETIREMENT_AGE
        final_age = TERMINAL_AGE
    else:
        n_ages = n_periods + 1
        retirement_age = START_AGE + max(1, n_periods // 2)
        final_age = START_AGE + n_periods

    regime: list[str] = []
    target: list[str | None] = []
    for period in range(n_ages):
        age = START_AGE + period
        if age < retirement_age:
            regime.append("working")
            target.append("retired" if age + 1 >= retirement_age else "working")
        elif age < final_age:
            regime.append("retired")
            target.append("dead" if age + 1 >= final_age else "retired")
        else:
            regime.append("dead")
            target.append(None)
    return _Schedule(regime=tuple(regime), target=tuple(target))


def _stock_levels(*, n_housing: int, housing_max: float) -> np.ndarray:
    """Discrete housing-stock levels, matching `build_model`'s spacing."""
    housing_min = housing_max / (2.0 * n_housing)
    return np.array(
        [
            housing_min + (housing_max - housing_min) * i / (n_housing - 1)
            for i in range(n_housing)
        ]
    )


def _wage_grid_and_transition(
    *, rho: float, sigma: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wage nodes and transition matrix from the model's own Tauchen process."""
    process = TauchenAR1Process(n_points=N_WAGE_NODES, gauss_hermite=True)
    kwargs = {
        "rho": jnp.asarray(rho),
        "sigma": jnp.asarray(sigma),
        "mu": jnp.asarray(mu),
    }
    nodes = np.asarray(process.compute_gridpoints(**kwargs))
    transition = np.asarray(process.compute_transition_probs(**kwargs))
    return nodes, transition


def _interp_extrap_linspace(
    *, y: np.ndarray, start: float, stop: float, n: int, xq: np.ndarray
) -> np.ndarray:
    """Linearly interpolate `y` on a linspace grid, extrapolating off-range.

    Mirrors pylcm's `get_linspace_coordinate`: locate the query in the linear grid,
    clip the upper node index to the last segment, and use that segment's slope to
    extrapolate beyond either end. `y` carries the grid along its last axis; the
    result has the shape of `xq`.
    """
    step = (stop - start) / (n - 1)
    coord = (xq - start) / step
    i_upper = np.clip(np.floor(coord).astype(np.int64) + 1, 1, n - 1)
    i_lower = i_upper - 1
    weight = coord - i_lower
    return y[i_lower] * (1.0 - weight) + y[i_upper] * weight


@dataclass(frozen=True)
class BranchAwareVfiResult:
    """Value functions from one VFI pass plus the per-branch values.

    Attributes index periods then regimes. `value` holds the envelope (max over the
    next-housing choice) shaped `(wage, housing, liquid)` for the working regime and
    `(housing, liquid)` for the retired/dead regimes — the same axis order as the
    pylcm solve. `branch_value` keeps the pre-max per-next-housing-choice rows with an
    extra `housing_choice` axis, the raw material the branch-aware continuation reads.
    """

    value: dict[int, dict[str, np.ndarray]]
    """Mapping of period to regime to the envelope value array."""
    branch_value: dict[int, dict[str, np.ndarray]]
    """Mapping of period to regime to the per-next-housing-choice value array."""


def solve_branch_aware_vfi(
    *,
    branch_aware: bool,
    n_grid: int,
    n_housing: int = 5,
    n_consumption: int = 400,
    n_periods: int | None = 5,
    liquid_max: float = 50.0,
    housing_max: float = 20.0,
    tau: float = 0.07,
    discount_factor: float = 0.94,
    gamma_c: float = 3.5,
    gamma_h: float = 1.5,
    alpha: float = 0.70,
    kappa: float = 1.0,
    theta_bar: float = 1.0,
    return_liquid: float = 0.04,
    return_housing: float = 0.0,
    retirement_pension: float = 0.3,
    rho_w: float = 0.82,
    sigma_w: float = 0.11,
    mu_w: float = 0.0,
) -> BranchAwareVfiResult:
    """Solve the DS App.2 discrete-housing model by host-side VFI.

    With `branch_aware=False` the continuation reads the already-maximized next value
    array (`I[max_d V_d]`, the brute/VFI comparator); with `branch_aware=True` it
    interpolates each next-housing-choice branch and maxes after (`max_d I[V_d]`, the
    DC-EGM comparator). Every other step is identical, so differencing the two passes
    isolates the comparator-ordering gap.

    Args:
        branch_aware: Select the branch-aware (`True`) or standard (`False`)
            continuation operator.
        n_grid: Number of liquid-asset grid points.
        n_housing: Number of discrete housing levels.
        n_consumption: Number of consumption grid-search points.
        n_periods: Shortened horizon (`None` uses the full lifecycle), matching
            `build_model`'s convention.
        liquid_max: Upper bound of the liquid grid.
        housing_max: Upper bound of the housing-level grid.
        tau: Proportional housing round-trip cost.
        discount_factor: Discount factor `beta`.
        gamma_c: Consumption CRRA.
        gamma_h: Housing CES curvature.
        alpha: Consumption weight in the separable CES utility.
        kappa: Housing-utility scale.
        theta_bar: Terminal bequest weight.
        return_liquid: Net liquid return.
        return_housing: Net housing return.
        retirement_pension: Fixed retirement income.
        rho_w: AR(1) wage persistence.
        sigma_w: AR(1) wage innovation std.
        mu_w: AR(1) wage drift.

    Returns:
        The envelope and per-branch value functions per period and regime.
    """
    schedule = _build_schedule(n_periods=n_periods)
    n_total_periods = len(schedule.regime)

    stock = _stock_levels(n_housing=n_housing, housing_max=housing_max)
    liquid = np.linspace(0.0, liquid_max, n_grid)
    consumption = np.linspace(0.05, liquid_max, n_consumption)
    wage_nodes, wage_transition = _wage_grid_and_transition(
        rho=rho_w, sigma=sigma_w, mu=mu_w
    )
    working_income = np.exp(wage_nodes)

    def utility(*, c: np.ndarray, serviced: float) -> np.ndarray:
        consumption_term = (c ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)
        housing_term = (serviced ** (1.0 - gamma_h) - 1.0) / (1.0 - gamma_h)
        return alpha * consumption_term + (1.0 - alpha) * kappa * housing_term

    def housing_cost(*, h: int, h_choice: int) -> float:
        if h_choice == h:
            return 0.0
        return (1.0 + tau) * stock[h_choice] - (1.0 + return_housing) * stock[h]

    def interp_liquid(*, row: np.ndarray, xq: np.ndarray) -> np.ndarray:
        return _interp_extrap_linspace(
            y=row, start=0.0, stop=liquid_max, n=n_grid, xq=xq
        )

    value: dict[int, dict[str, np.ndarray]] = {}
    branch_value: dict[int, dict[str, np.ndarray]] = {}

    # Backward induction from the terminal period.
    for period in range(n_total_periods - 1, -1, -1):
        regime = schedule.regime[period]
        target = schedule.target[period]
        has_wage = regime == "working"

        if regime == "dead":
            estate = (1.0 + return_liquid) * liquid[None, :] + stock[:, None]
            bequest = theta_bar * estate  # (housing, liquid)
            value[period] = {regime: bequest}
            branch_value[period] = {regime: bequest[:, None, :]}
            continue

        assert target is not None  # only the terminal "dead" regime has no target
        vbranch = _period_branch_values(
            has_wage=has_wage,
            n_housing=n_housing,
            n_grid=n_grid,
            stock=stock,
            liquid=liquid,
            consumption=consumption,
            working_income=working_income,
            retirement_pension=retirement_pension,
            return_liquid=return_liquid,
            discount_factor=discount_factor,
            wage_transition=wage_transition,
            target_value=value[period + 1][target],
            target_branch=branch_value[period + 1][target],
            target_has_wage=target == "working",
            branch_aware=branch_aware,
            utility=utility,
            housing_cost=housing_cost,
            interp_liquid=interp_liquid,
        )
        envelope = vbranch.max(axis=2)  # (wage, housing, liquid)
        if has_wage:
            value[period] = {regime: envelope}
            branch_value[period] = {regime: vbranch}
        else:
            value[period] = {regime: envelope[0]}
            branch_value[period] = {regime: vbranch[0]}

    return BranchAwareVfiResult(value=value, branch_value=branch_value)


def _period_branch_values(
    *,
    has_wage: bool,
    n_housing: int,
    n_grid: int,
    stock: np.ndarray,
    liquid: np.ndarray,
    consumption: np.ndarray,
    working_income: np.ndarray,
    retirement_pension: float,
    return_liquid: float,
    discount_factor: float,
    wage_transition: np.ndarray,
    target_value: np.ndarray,
    target_branch: np.ndarray,
    target_has_wage: bool,
    branch_aware: bool,
    utility: Callable[..., np.ndarray],
    housing_cost: Callable[..., float],
    interp_liquid: Callable[..., np.ndarray],
) -> np.ndarray:
    """Per-next-housing-choice value array for one non-terminal period.

    Grid-searches consumption for every (source wage, held housing, next-housing
    choice) cell and maxes over consumption, returning the array shaped
    `(wage, housing, housing_choice, liquid)` (a singleton wage axis when the regime
    carries no wage). The envelope over the `housing_choice` axis is the period value.
    """
    n_wage_axis = working_income.shape[0] if has_wage else 1
    vbranch = np.full((n_wage_axis, n_housing, n_housing, n_grid), -np.inf)

    for source in range(n_wage_axis):
        this_income = working_income[source] if has_wage else retirement_pension
        for h in range(n_housing):
            for h_choice in range(n_housing):
                cost = housing_cost(h=h, h_choice=h_choice)
                resources = (1.0 + return_liquid) * liquid + this_income - cost
                next_liquid = resources[:, None] - consumption[None, :]
                feasible = next_liquid >= 0.0  # (liquid, consumption)
                flow = utility(c=consumption, serviced=float(stock[h_choice]))

                with np.errstate(invalid="ignore"):
                    e_next = _continuation(
                        next_liquid=next_liquid,
                        h_choice=h_choice,
                        target_has_wage=target_has_wage,
                        wage_transition_row=(
                            wage_transition[source] if target_has_wage else None
                        ),
                        target_value=target_value,
                        target_branch=target_branch,
                        branch_aware=branch_aware,
                        interp_liquid=interp_liquid,
                    )  # (liquid, consumption)

                q = flow[None, :] + discount_factor * e_next
                # A next-state read that brackets an infeasible (-inf) node is itself
                # undefined; treat it as infeasible so it never wins the max.
                feasible = feasible & np.isfinite(e_next)
                q = np.where(feasible, q, -np.inf)
                vbranch[source, h, h_choice, :] = q.max(axis=1)

    return vbranch


def _continuation(
    *,
    next_liquid: np.ndarray,
    h_choice: int,
    target_has_wage: bool,
    wage_transition_row: np.ndarray | None,
    target_value: np.ndarray,
    target_branch: np.ndarray,
    branch_aware: bool,
    interp_liquid: Callable[..., np.ndarray],
) -> np.ndarray:
    """Expected continuation value at the next-period states.

    The next-housing state equals the chosen `h_choice`, so the continuation reads
    the `h_choice`-slice of the target value. Under `branch_aware` each next-period
    housing-choice branch is interpolated in liquid and maxed afterwards; otherwise
    the already-maximized envelope is interpolated directly. When the target carries
    a wage axis the result is averaged over next-wage nodes with the transition row.
    """

    def read(*, wage_index: int | None) -> np.ndarray:
        if branch_aware:
            if wage_index is None:
                branches = target_branch[h_choice]  # (housing_choice, liquid)
            else:
                branches = target_branch[wage_index, h_choice]
            interpolated = np.stack(
                [
                    interp_liquid(row=branches[hc], xq=next_liquid)
                    for hc in range(branches.shape[0])
                ],
                axis=0,
            )
            return interpolated.max(axis=0)
        if wage_index is None:
            row = target_value[h_choice]
        else:
            row = target_value[wage_index, h_choice]
        return interp_liquid(row=row, xq=next_liquid)

    if not target_has_wage:
        return read(wage_index=None)

    assert wage_transition_row is not None  # a wage target carries a transition row
    contributions = [
        wage_transition_row[j] * read(wage_index=j)
        for j in range(wage_transition_row.shape[0])
    ]
    return np.sum(contributions, axis=0)
