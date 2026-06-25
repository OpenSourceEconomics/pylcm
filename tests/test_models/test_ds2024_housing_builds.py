"""The DS-2024 housing NEGM model builds, solves, and matches its brute twin.

The DS-2024 housing model (`tests.test_models.ds2024_housing`) is the NEGM column
of the RFC-vs-NEGM comparison. At zero depreciation the keeper holds the house
(`H' = H`), so the model fits pylcm's NEGM keeper kernel and is a faithful
reproduction of the source economics (CRRA-plus-log utility, Markov income,
proportional house-trade cost). Its grid-search twin solves the same Bellman
problem by brute force; on the liquid-housing interior the two value functions
agree up to grid resolution — the standard NEGM/VFI oracle pair.

The paper's `delta = 0.10` keeper depreciates the held stock, which pylcm's
current NEGM keeper (a strict `H' = H` hold) cannot express; that path awaits the
NEGM keeper depreciation extension and is meanwhile validated by the brute twin.
"""

from typing import Literal

import numpy as np
import pytest

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
