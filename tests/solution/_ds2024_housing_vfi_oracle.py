"""Dense host VFI oracle for the DS-2024 housing NEGM model.

A self-contained NumPy value-function iteration for the DS-2024 housing model
(`tests.test_models.ds2024_housing`), used as the ground truth for the keeper
depreciation. It mirrors the NEGM nest's economics exactly:

- the **adjuster** searches the next house `H'` over the outer house grid, paying
  the round-trip `(1 + tau) H' - (1 + r_H) h(1 - delta)`;
- the **keeper** holds the house at the depreciated level `H' = h(1 - delta)` for
  free — a candidate that lands *off* the house grid when `delta > 0`.

Because the free-keep level is off-grid at `delta > 0`, the model's brute twin
(which searches `next_housing` on the grid) cannot represent it; this oracle
includes it explicitly, so it is the only valid reference for the `delta > 0`
keeper. The continuation is the standard VFI operator — the next period's envelope
value (already maxed over keep/adjust) read bilinearly in the liquid Euler state
and the durable house, matching the NEGM keeper's passive housing read. A dense
consumption grid keeps the inner search near-exact, so differencing against the
pylcm NEGM solve isolates the shared house/savings discretisation, not the oracle.

It is a test tool only — never a registered solver.
"""

from dataclasses import dataclass

import numpy as np

from tests.test_models.ds2024_housing import (
    _LOG_INCOME_BASE,
    INCOME_HIGH,
    INCOME_LOW,
    INCOME_PI,
    STATIONARY_AGE,
)


@dataclass(frozen=True)
class _Schedule:
    """Per-period regime layout derived from the model's lifecycle anchors."""

    regime: tuple[str, ...]
    """Regime active at each period (`"alive"` or `"dead"`)."""


def _build_schedule(*, n_periods: int) -> _Schedule:
    """Derive the per-period regime, matching `build_model`'s age thresholds.

    Ages run `STATIONARY_AGE .. STATIONARY_AGE + n_periods - 1`; the alive regime
    is active below the final age and the terminal bequest at it, so the last
    period is `"dead"` and the rest are `"alive"`.
    """
    final_age = STATIONARY_AGE + n_periods - 1
    regime = [
        "alive" if STATIONARY_AGE + period < final_age else "dead"
        for period in range(n_periods)
    ]
    return _Schedule(regime=tuple(regime))


def _interp_linspace_1d(
    *, y: np.ndarray, start: float, stop: float, n: int, xq: np.ndarray
) -> np.ndarray:
    """Linearly interpolate `y` on a linspace grid, extrapolating off-range.

    Mirrors pylcm's `get_linspace_coordinate`: locate the query in the linear grid,
    clip the upper node index to the last segment, and use that segment's slope to
    extrapolate beyond either end. `y` is 1-D over the grid; the result has the
    shape of `xq`.
    """
    step = (stop - start) / (n - 1)
    coord = (xq - start) / step
    i_upper = np.clip(np.floor(coord).astype(np.int64) + 1, 1, n - 1)
    i_lower = i_upper - 1
    weight = coord - i_lower
    return y[i_lower] * (1.0 - weight) + y[i_upper] * weight


@dataclass(frozen=True)
class _Calibration:
    """Resolved grids and parameters for one oracle solve."""

    liquid: np.ndarray
    house: np.ndarray
    consumption: np.ndarray
    income_value: np.ndarray
    n_grid: int
    housing_min: float
    housing_max: float
    liquid_max: float
    delta: float
    tau: float
    return_liquid: float
    return_housing: float
    discount_factor: float
    gamma_c: float
    alpha: float
    theta: float
    bequest_shift: float


def _bilinear_continuation(
    *,
    value_house_liquid: np.ndarray,
    next_house: float,
    next_liquid: np.ndarray,
    cal: _Calibration,
) -> np.ndarray:
    """Read a value array `V(house, liquid)` at `(next_house, next_liquid)`.

    Interpolates first along the house axis at the scalar `next_house`, then along
    the liquid axis at the `next_liquid` query — the bilinear read the NEGM keeper
    performs when the kept stock lands off the house grid.
    """
    step_h = (cal.housing_max - cal.housing_min) / (cal.n_grid - 1)
    coord = (next_house - cal.housing_min) / step_h
    i_upper = int(np.clip(np.floor(coord) + 1, 1, cal.n_grid - 1))
    i_lower = i_upper - 1
    weight = coord - i_lower
    row = (
        value_house_liquid[i_lower] * (1.0 - weight)
        + value_house_liquid[i_upper] * weight
    )
    return _interp_linspace_1d(
        y=row, start=cal.housing_min, stop=cal.liquid_max, n=cal.n_grid, xq=next_liquid
    )


def _alive_value(
    *, next_value: np.ndarray, next_is_alive: bool, cal: _Calibration
) -> np.ndarray:
    """One backward-induction step for the alive regime.

    Returns the envelope value `V(income, house, liquid)`. For each income node and
    held house, the candidate next houses are the free-keep level `h(1 - delta)`
    plus every outer-grid house; consumption is grid-searched and the standard-VFI
    continuation read bilinearly from `next_value`.
    """
    n_income = cal.income_value.shape[0]
    n_grid = cal.n_grid
    value = np.full((n_income, n_grid, n_grid), -np.inf)

    for income_index in range(n_income):
        income = cal.income_value[income_index]
        transition = np.asarray(INCOME_PI)[income_index]
        for house_index in range(n_grid):
            held = cal.house[house_index]
            depreciated = held * (1.0 - cal.delta)
            keep_cost = 0.0
            adjust_costs = (1.0 + cal.tau) * cal.house - (
                1.0 + cal.return_housing
            ) * depreciated
            next_houses = np.concatenate([[depreciated], cal.house])
            costs = np.concatenate([[keep_cost], adjust_costs])

            best = np.full(n_grid, -np.inf)
            for next_house, cost in zip(next_houses, costs, strict=True):
                resources = (1.0 + cal.return_liquid) * cal.liquid + income - cost
                next_liquid = resources[:, None] - cal.consumption[None, :]
                feasible = next_liquid >= cal.housing_min
                consumption_utility = (cal.consumption ** (1.0 - cal.gamma_c) - 1.0) / (
                    1.0 - cal.gamma_c
                )
                flow = consumption_utility + cal.alpha * np.log(next_house)

                if next_is_alive:
                    expected = np.zeros_like(next_liquid)
                    for next_income_index in range(n_income):
                        expected += transition[
                            next_income_index
                        ] * _bilinear_continuation(
                            value_house_liquid=next_value[next_income_index],
                            next_house=float(next_house),
                            next_liquid=next_liquid,
                            cal=cal,
                        )
                else:
                    expected = _bilinear_continuation(
                        value_house_liquid=next_value,
                        next_house=float(next_house),
                        next_liquid=next_liquid,
                        cal=cal,
                    )

                q = flow[None, :] + cal.discount_factor * expected
                q = np.where(feasible & np.isfinite(expected), q, -np.inf)
                best = np.maximum(best, q.max(axis=1))
            value[income_index, house_index, :] = best
    return value


def solve_ds2024_housing_vfi(
    *,
    n_grid: int,
    n_periods: int = 4,
    n_consumption: int = 400,
    liquid_max: float = 50.0,
    housing_max: float = 50.0,
    housing_min: float = 0.01,
    consumption_max: float = 50.0,
    delta: float = 0.0,
    tau: float = 0.20,
    discount_factor: float = 0.945,
    gamma_c: float = 1.458,
    alpha: float = 0.66,
    return_liquid: float = 0.024,
    return_housing: float = 0.10,
    theta: float = 2.0,
    bequest_shift: float = 200.0,
) -> dict[int, np.ndarray]:
    """Solve the DS-2024 housing model by host-side VFI with the free-keep candidate.

    Mirrors `build_model`/`build_params` defaults. The free-keep level `h(1 - delta)`
    is a search candidate alongside the outer house grid, so the solve is faithful
    at any `delta` — unlike the on-grid brute twin, which only represents free keeping
    at `delta = 0`.

    Args:
        n_grid: Number of liquid and house grid points (matches the model).
        n_periods: Number of model periods (the last is the terminal bequest).
        n_consumption: Number of consumption grid-search points (dense for accuracy).
        liquid_max: Upper bound of the liquid grid.
        housing_max: Upper bound of the house grid.
        housing_min: Lower bound of the grids and the borrowing limit.
        consumption_max: Upper bound of the consumption search grid.
        delta: House depreciation rate (the keeper holds at `h(1 - delta)`).
        tau: Proportional house round-trip cost.
        discount_factor: Discount factor `beta`.
        gamma_c: Consumption CRRA.
        alpha: Housing-service weight.
        return_liquid: Net liquid return.
        return_housing: Net house return.
        theta: Terminal bequest weight.
        bequest_shift: Terminal bequest shift `K`.

    Returns:
        Mapping of period to the alive envelope value `V(income, house, liquid)`;
        the terminal period maps to the bequest value `V(house, liquid)`.
    """
    liquid = np.linspace(housing_min, liquid_max, n_grid)
    house = np.linspace(housing_min, housing_max, n_grid)
    consumption = np.linspace(0.05, consumption_max, n_consumption)
    income_value = np.exp(_LOG_INCOME_BASE + np.array([INCOME_LOW, INCOME_HIGH])) * 1e-5

    cal = _Calibration(
        liquid=liquid,
        house=house,
        consumption=consumption,
        income_value=income_value,
        n_grid=n_grid,
        housing_min=housing_min,
        housing_max=housing_max,
        liquid_max=liquid_max,
        delta=delta,
        tau=tau,
        return_liquid=return_liquid,
        return_housing=return_housing,
        discount_factor=discount_factor,
        gamma_c=gamma_c,
        alpha=alpha,
        theta=theta,
        bequest_shift=bequest_shift,
    )

    schedule = _build_schedule(n_periods=n_periods)
    value: dict[int, np.ndarray] = {}

    for period in range(n_periods - 1, -1, -1):
        if schedule.regime[period] == "dead":
            estate = (
                bequest_shift + (1.0 + return_liquid) * liquid[None, :] + house[:, None]
            )
            value[period] = theta * (estate ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)
            continue
        next_is_alive = schedule.regime[period + 1] == "alive"
        value[period] = _alive_value(
            next_value=value[period + 1], next_is_alive=next_is_alive, cal=cal
        )

    return value
