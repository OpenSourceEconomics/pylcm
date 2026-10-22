"""The DS-2024 housing NEGM model builds, solves, and matches its VFI oracles.

The DS-2024 housing model (`tests.test_models.ds2024_housing`) is the NEGM column
of the RFC-vs-NEGM comparison — a faithful reproduction of the source economics
(CRRA-plus-log utility, Markov income, proportional house-trade cost). The keeper
holds the house at the depreciated level `H' = h(1 - delta)` for free.

Two oracles validate it on the liquid-housing interior:

- at `delta = 0` the free-keep level is `H' = h`, on the house grid, so the
  grid-search brute twin solves the identical Bellman problem and the two value
  functions agree up to grid resolution;
- at `delta = 0.10` the free-keep level is off the grid, so the brute twin cannot
  represent it; a dense host VFI oracle that includes the free-keep candidate
  explicitly (`tests.solution._ds2024_housing_vfi_oracle`) is the reference, and the
  NEGM keeper's depreciated hold reproduces it up to grid resolution.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest

from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution._ds2024_housing_vfi_oracle import solve_ds2024_housing_vfi
from tests.test_models.ds2024_housing import (
    _LOG_INCOME_BASE,
    INCOME_HIGH,
    INCOME_LOW,
    INCOME_PI,
    build_model,
    build_params,
)

_NEEDS_KERNEL = pytest.mark.requires_exact_affine_kernel(
    reason=EXACT_KERNEL_SKIP_REASON
)

pytestmark = pytest.mark.slow

N_GRID = 10
N_CONSUMPTION = 200
N_PERIODS = 4

# Pooled-interior thresholds. The value function is O(1)-O(10) on the interior.
# Comparisons against the brute grid-search twin are limited by the brute
# consumption grid's undershoot, so they agree in bulk up to grid resolution
# rather than exactly at every cell. The NEGM-vs-VFI-oracle comparison is much
# tighter: with the exact query-side outer envelope the observed disagreement
# is dominated by the shared state/liquid grid resolution (an empirical
# statement at these grids, not a bound), and the depreciating keeper matches
# the identity keeper's agreement.
BRUTE_MEAN_TOL = 0.20
BRUTE_CELL_TOL = 0.60
ORACLE_MEAN_TOL = 0.06
ORACLE_CELL_TOL = 0.30
MIN_FRACTION_WITHIN = 0.95


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "negm",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
        "brute",
    ],
)
def test_ds2024_housing_builds_and_solves(variant: Literal["negm", "brute"]):
    """Both DS-2024 housing variants construct and solve with a finite interior."""
    model = build_model(
        variant=variant,
        n_grid=6,
        n_periods=3,
        n_consumption=40,
    )
    solution = model.solve(
        params=build_params(variant=variant, delta=0.0), log_level="debug"
    ).values

    assert sorted(solution) == [0, 1, 2]
    alive_periods = [p for p in solution if "alive" in solution[p]]
    assert alive_periods
    for period in alive_periods:
        interior = np.asarray(solution[period]["alive"])[..., 2:-1, 2:-1]
        assert np.isfinite(interior).all()


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_ds2024_housing_negm_matches_brute_on_interior():
    """The NEGM solve agrees with its grid-search twin on the liquid-housing interior.

    At zero depreciation the keeper holds the house, so the NEGM nest and the
    brute grid search solve the identical Bellman problem. The pooled interior
    value difference is sub-unit in the mean and the overwhelming majority of
    cells fall within the cell tolerance — the NEGM model is correct up to grid
    resolution.
    """
    brute = (
        build_model(
            variant="brute",
            n_grid=N_GRID,
            n_periods=N_PERIODS,
            n_consumption=N_CONSUMPTION,
        )
        .solve(params=build_params(variant="brute", delta=0.0), log_level="debug")
        .values
    )
    negm = (
        build_model(
            variant="negm",
            n_grid=N_GRID,
            n_periods=N_PERIODS,
            n_consumption=N_CONSUMPTION,
        )
        .solve(params=build_params(variant="negm", delta=0.0), log_level="debug")
        .values
    )

    differences = []
    for period in sorted(brute):
        if "alive" not in brute[period]:
            continue
        brute_V = np.asarray(brute[period]["alive"])
        negm_V = np.asarray(negm[period]["alive"])
        assert brute_V.shape == negm_V.shape
        interior = (Ellipsis, slice(3, N_GRID - 2), slice(3, N_GRID - 2))
        finite = np.isfinite(brute_V[interior]) & np.isfinite(negm_V[interior])
        differences.append(np.abs(brute_V[interior] - negm_V[interior])[finite])
    difference = np.concatenate(differences)

    assert float(difference.mean()) < BRUTE_MEAN_TOL
    assert float((difference <= BRUTE_CELL_TOL).mean()) >= MIN_FRACTION_WITHIN


def _pooled_interior_difference(*, pylcm_solution, oracle):
    """Pooled absolute interior `V` difference between a pylcm solve and the oracle.

    The oracle stores the alive value as `(income, house, liquid)` and the terminal
    as `(house, liquid)`, the same axis order as the pylcm solve, so the interior
    slice and finite mask apply directly.
    """
    differences = []
    for period in sorted(oracle):
        if period not in pylcm_solution or "alive" not in pylcm_solution[period]:
            continue
        pylcm_V = np.asarray(pylcm_solution[period]["alive"])
        oracle_V = np.asarray(oracle[period])
        assert pylcm_V.shape == oracle_V.shape
        interior = (Ellipsis, slice(3, N_GRID - 2), slice(3, N_GRID - 2))
        finite = np.isfinite(pylcm_V[interior]) & np.isfinite(oracle_V[interior])
        differences.append(np.abs(pylcm_V[interior] - oracle_V[interior])[finite])
    return np.concatenate(differences)


def test_ds2024_housing_vfi_oracle_matches_brute_at_zero_delta():
    """The host VFI oracle reproduces the brute solve at zero depreciation.

    The oracle is a from-scratch NumPy value-function iteration; at `delta = 0` it
    solves the same economics as the model's grid-search twin, so the two agree on
    the liquid-housing interior up to grid resolution. This pins the oracle's
    economics before it is used as the `delta > 0` reference.
    """
    brute = (
        build_model(
            variant="brute",
            n_grid=N_GRID,
            n_periods=N_PERIODS,
            n_consumption=N_CONSUMPTION,
        )
        .solve(params=build_params(variant="brute", delta=0.0), log_level="debug")
        .values
    )
    oracle = solve_ds2024_housing_vfi(
        n_grid=N_GRID, n_periods=N_PERIODS, n_consumption=400, delta=0.0
    )

    difference = _pooled_interior_difference(pylcm_solution=brute, oracle=oracle)

    assert float(difference.mean()) < BRUTE_MEAN_TOL
    assert float((difference <= BRUTE_CELL_TOL).mean()) >= MIN_FRACTION_WITHIN


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_ds2024_housing_negm_keeper_depreciation_matches_vfi_oracle():
    """At `delta = 0.10` the NEGM keeper's depreciated hold matches the VFI oracle.

    The keeper holds the house at the depreciated level `H' = h(1 - delta)`, which
    lands off the house grid. The dense VFI oracle includes that free-keep candidate
    explicitly, so it is the valid reference where the on-grid brute twin is not. The
    NEGM solve reproduces it on the liquid-housing interior up to grid resolution —
    the keeper depreciation is faithful.
    """
    negm = (
        build_model(
            variant="negm",
            n_grid=N_GRID,
            n_periods=N_PERIODS,
            n_consumption=N_CONSUMPTION,
            delta=0.10,
        )
        .solve(params=build_params(variant="negm", delta=0.10), log_level="debug")
        .values
    )
    oracle = solve_ds2024_housing_vfi(
        n_grid=N_GRID,
        n_periods=N_PERIODS,
        n_consumption=400,
        delta=0.10,
        read="clamp",
    )

    difference = _pooled_interior_difference(pylcm_solution=negm, oracle=oracle)

    assert float(difference.mean()) < ORACLE_MEAN_TOL
    assert float((difference <= ORACLE_CELL_TOL).mean()) >= MIN_FRACTION_WITHIN


@pytest.fixture(scope="module")
def outer_refinement_mean() -> Callable[[int], float]:
    """Pooled interior NEGM-vs-oracle difference at one nested outer grid.

    Exactness is to the *declared outer grid*: enlarging the candidate set on
    nested grids (5 ⊂ 9 ⊂ 17 linspace points) can only add outer options, so
    the solve improves toward the continuous-outer problem. The reference is
    the VFI oracle with a dense outer search (129 points plus the free-keep
    candidate) and its continuation reads clamp-matched to the NEGM carry
    read; the state and liquid grids are fixed throughout, so the comparison
    isolates the outer-grid refinement.

    Beyond this range the difference sits at the fixed state-grid floor
    (roughly half the finest value here), where further outer refinement no
    longer moves it monotonically — an unresolved interaction of the remaining
    approximations shared by both sides. Candidate count alone cannot grow a
    uniform value-error bound (the max is 1-Lipschitz in the sup norm), so the
    floor is diagnosed, not explained, by these tests.

    The oracle and each solve are computed once per module and shared by the
    refinement tests. In file order each test adds at most one new solve, so
    no single test carries the oracle and all three solves.
    """
    oracle: dict[int, np.ndarray] = {}
    means: dict[int, float] = {}

    def mean_at(n_outer_grid: int) -> float:
        if not oracle:
            oracle.update(
                solve_ds2024_housing_vfi(
                    n_grid=N_GRID,
                    n_periods=N_PERIODS,
                    n_consumption=400,
                    delta=0.10,
                    n_outer_grid=129,
                    read="clamp",
                )
            )
        if n_outer_grid not in means:
            negm = (
                build_model(
                    variant="negm",
                    n_grid=N_GRID,
                    n_periods=N_PERIODS,
                    n_consumption=N_CONSUMPTION,
                    delta=0.10,
                    n_outer_grid=n_outer_grid,
                )
                .solve(
                    params=build_params(variant="negm", delta=0.10), log_level="debug"
                )
                .values
            )
            means[n_outer_grid] = float(
                _pooled_interior_difference(pylcm_solution=negm, oracle=oracle).mean()
            )
        return means[n_outer_grid]

    return mean_at


@_NEEDS_KERNEL
def test_ds2024_housing_negm_finest_outer_grid_is_within_the_oracle_tolerance(
    outer_refinement_mean: Callable[[int], float],
):
    """At 17 outer nodes the pooled interior difference is within tolerance."""
    assert outer_refinement_mean(17) < ORACLE_MEAN_TOL


@_NEEDS_KERNEL
def test_ds2024_housing_negm_improves_from_middle_to_finest_outer_grid(
    outer_refinement_mean: Callable[[int], float],
):
    """Refining the outer house grid from 9 to 17 nodes moves NEGM toward the truth."""
    assert outer_refinement_mean(9) > outer_refinement_mean(17)


@_NEEDS_KERNEL
def test_ds2024_housing_negm_improves_from_coarsest_to_middle_outer_grid(
    outer_refinement_mean: Callable[[int], float],
):
    """Refining the outer house grid from 5 to 9 nodes moves NEGM toward the truth."""
    assert outer_refinement_mean(5) > outer_refinement_mean(9)


@_NEEDS_KERNEL
def test_ds2024_housing_negm_finest_outer_grid_halves_the_coarsest_difference(
    outer_refinement_mean: Callable[[int], float],
):
    """The 17-node outer grid at least halves the 5-node difference to the truth."""
    assert outer_refinement_mean(17) < outer_refinement_mean(5) / 2


def _exact_state_first_period_value(
    *,
    n_grid: int,
    n_outer_grid: int,
    n_consumption: int,
    n_fine_liquid: int,
    delta: float,
    tau: float = 0.20,
    beta: float = 0.945,
    gamma_c: float = 1.458,
    alpha: float = 0.66,
    return_liquid: float = 0.024,
    return_housing: float = 0.10,
    theta: float = 2.0,
    bequest_shift: float = 200.0,
) -> np.ndarray:
    """First-period alive value of the three-period model with exact next states.

    A literal grid search per state: every next house `H'` (the free keep
    `h(1 - delta)` and every outer-grid node) is solved at its own level, and the
    last alive period is tabulated on a dense liquid grid, so the first-period
    continuation is read at the actual next state rather than blended over the
    house grid. Only the terminal bequest is read on the declared state grids,
    clamped, as the NEGM solve reads it. The calibration defaults are
    `build_params`'s. Returned in the pylcm layout `(income, liquid, house)`.
    """
    low, high = 0.01, 50.0
    liquid = np.linspace(low, high, n_grid)
    house = np.linspace(low, high, n_grid)
    outer = np.linspace(low, high, n_outer_grid)
    fine = np.linspace(low, high, n_fine_liquid)
    consumption = np.linspace(0.05, 50.0, n_consumption)
    income = np.exp(_LOG_INCOME_BASE + np.array([INCOME_LOW, INCOME_HIGH])) * 1e-5
    transition = np.asarray(INCOME_PI)
    consumption_utility = (consumption ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)
    estate = bequest_shift + (1.0 + return_liquid) * liquid[None, :] + house[:, None]
    terminal = theta * (estate ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)

    def read_terminal(*, next_house: float, next_liquid: np.ndarray) -> np.ndarray:
        step = (high - low) / (n_grid - 1)
        coordinate = np.clip((next_house - low) / step, 0.0, n_grid - 1)
        upper = int(np.clip(np.floor(coordinate) + 1, 1, n_grid - 1))
        weight = coordinate - (upper - 1)
        row = terminal[upper - 1] * (1.0 - weight) + terminal[upper] * weight
        return np.interp(np.clip(next_liquid, low, high), liquid, row)

    def best(*, held: float, cash: np.ndarray, continuation) -> np.ndarray:
        depreciated = held * (1.0 - delta)
        candidates = np.concatenate([[depreciated], outer])
        costs = np.concatenate(
            [
                [0.0],
                (1.0 + tau) * outer - (1.0 + return_housing) * depreciated,
            ]
        )
        value = np.full(cash.shape, -np.inf)
        for next_house, outer_cost in zip(candidates, costs, strict=True):
            next_liquid = (cash - outer_cost)[:, None] - consumption[None, :]
            q = (
                consumption_utility[None, :]
                + alpha * np.log(next_house)
                + beta * continuation(next_house=next_house, next_liquid=next_liquid)
            )
            q = np.where(next_liquid >= low, q, -np.inf)
            value = np.maximum(value, q.max(axis=1))
        return value

    next_houses = np.concatenate([house * (1.0 - delta), outer])
    last_alive = {
        (income_index, float(next_house)): best(
            held=next_house,
            cash=(1.0 + return_liquid) * fine + income[income_index],
            continuation=read_terminal,
        )
        for income_index in range(2)
        for next_house in next_houses
    }
    value = np.empty((2, n_grid, n_grid))
    for income_index in range(2):

        def expected(*, next_house: float, next_liquid: np.ndarray) -> np.ndarray:
            clamped = np.clip(next_liquid, low, high)
            return sum(
                transition[income_index, next_income]  # noqa: B023
                * np.interp(clamped, fine, last_alive[(next_income, next_house)])
                for next_income in range(2)
            )

        for house_index, held in enumerate(house):
            value[income_index, :, house_index] = best(
                held=held,
                cash=(1.0 + return_liquid) * liquid + income[income_index],
                continuation=expected,
            )
    return value


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_ds2024_housing_negm_does_not_exceed_exact_next_state_optimum():
    """The NEGM first-period value never exceeds the exact next-state optimum.

    The reference solves the next period at the actual next house of every
    candidate, so its continuation is at least as good as the NEGM read, which
    blends the next value over the house grid. Up to the reference's own
    consumption-grid undershoot, no interior NEGM value may lie above it: a
    higher value credits a continuation no next state attains.
    """
    n_grid, n_outer_grid, delta = 8, 5, 0.10
    negm = (
        build_model(
            variant="negm",
            n_grid=n_grid,
            n_periods=3,
            n_consumption=N_CONSUMPTION,
            delta=delta,
            n_outer_grid=n_outer_grid,
        )
        .solve(params=build_params(variant="negm", delta=delta), log_level="debug")
        .values
    )
    reference = _exact_state_first_period_value(
        n_grid=n_grid,
        n_outer_grid=n_outer_grid,
        n_consumption=1600,
        n_fine_liquid=351,
        delta=delta,
    )
    interior = (Ellipsis, slice(2, n_grid - 1), slice(2, n_grid - 1))
    excess = np.asarray(negm[0]["alive"])[interior] - reference[interior]

    assert float(excess.max()) < 0.01
