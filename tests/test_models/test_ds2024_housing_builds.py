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

from typing import Literal

import numpy as np
import pytest

from tests.solution._ds2024_housing_vfi_oracle import solve_ds2024_housing_vfi
from tests.test_models.ds2024_housing import build_model, build_params

N_GRID = 10
N_CONSUMPTION = 200
N_PERIODS = 4

# Pooled-interior thresholds. The value function is O(1)-O(10) on the interior;
# the brute consumption grid undershoots and the DC-EGM carries a savings-grid
# interpolation error, so the two agree in bulk up to grid resolution rather than
# exactly at every cell.
MEAN_TOL = 0.20
CELL_TOL = 0.60
MIN_FRACTION_WITHIN = 0.95


@pytest.mark.parametrize("variant", ["negm", "brute"])
def test_ds2024_housing_builds_and_solves(variant: Literal["negm", "brute"]):
    """Both DS-2024 housing variants construct and solve with a finite interior."""
    model = build_model(
        variant=variant,
        n_grid=6,
        n_periods=3,
        n_consumption=40,
    )
    solution = model.solve(
        params=build_params(variant=variant, delta=0.0), log_level="off"
    )

    assert sorted(solution) == [0, 1, 2]
    alive_periods = [p for p in solution if "alive" in solution[p]]
    assert alive_periods
    for period in alive_periods:
        interior = np.asarray(solution[period]["alive"])[..., 2:-1, 2:-1]
        assert np.isfinite(interior).all()


def test_ds2024_housing_negm_matches_brute_on_interior():
    """The NEGM solve agrees with its grid-search twin on the liquid-housing interior.

    At zero depreciation the keeper holds the house, so the NEGM nest and the
    brute grid search solve the identical Bellman problem. The pooled interior
    value difference is sub-unit in the mean and the overwhelming majority of
    cells fall within the cell tolerance — the NEGM model is correct up to grid
    resolution.
    """
    brute = build_model(
        variant="brute",
        n_grid=N_GRID,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
    ).solve(params=build_params(variant="brute", delta=0.0), log_level="off")
    negm = build_model(
        variant="negm",
        n_grid=N_GRID,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
    ).solve(params=build_params(variant="negm", delta=0.0), log_level="off")

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

    assert float(difference.mean()) < MEAN_TOL
    assert float((difference <= CELL_TOL).mean()) >= MIN_FRACTION_WITHIN


def _pooled_interior_difference(pylcm_solution, oracle):
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
    brute = build_model(
        variant="brute",
        n_grid=N_GRID,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
    ).solve(params=build_params(variant="brute", delta=0.0), log_level="off")
    oracle = solve_ds2024_housing_vfi(
        n_grid=N_GRID, n_periods=N_PERIODS, n_consumption=400, delta=0.0
    )

    difference = _pooled_interior_difference(brute, oracle)

    assert float(difference.mean()) < MEAN_TOL
    assert float((difference <= CELL_TOL).mean()) >= MIN_FRACTION_WITHIN


def test_ds2024_housing_negm_keeper_depreciation_matches_vfi_oracle():
    """At `delta = 0.10` the NEGM keeper's depreciated hold matches the VFI oracle.

    The keeper holds the house at the depreciated level `H' = h(1 - delta)`, which
    lands off the house grid. The dense VFI oracle includes that free-keep candidate
    explicitly, so it is the valid reference where the on-grid brute twin is not. The
    NEGM solve reproduces it on the liquid-housing interior up to grid resolution —
    the keeper depreciation is faithful.
    """
    negm = build_model(
        variant="negm",
        n_grid=N_GRID,
        n_periods=N_PERIODS,
        n_consumption=N_CONSUMPTION,
        delta=0.10,
    ).solve(params=build_params(variant="negm", delta=0.10), log_level="off")
    oracle = solve_ds2024_housing_vfi(
        n_grid=N_GRID, n_periods=N_PERIODS, n_consumption=400, delta=0.10
    )

    difference = _pooled_interior_difference(negm, oracle)

    assert float(difference.mean()) < MEAN_TOL
    assert float((difference <= CELL_TOL).mean()) >= MIN_FRACTION_WITHIN
