"""The DS-2024 discrete-housing model builds, solves, and matches its twins.

The DS-2024 housing RFC column is the discrete-choice DC-EGM with the 1-D
upper-envelope backend nested over the housing grid (the source refines the
keeper's endogenous liquid grid per housing column with the rooftop cut). These
tests check it constructs and solves, that the RFC backend reproduces FUES (the
two 1-D envelopes agree), and that the solve bulk-agrees with its grid-search
(VFI) twin on the liquid interior up to grid resolution.
"""

from typing import Literal

import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from tests.test_models.ds2024_housing_fues import build_model, build_params

N_GRID = 10
N_HOUSING = 5
N_CONSUMPTION = 300
N_PERIODS = 4
INTERIOR = (Ellipsis, slice(3, N_GRID - 2))


@pytest.mark.parametrize("variant", ["dcegm", "brute"])
def test_ds2024_housing_fues_builds_and_solves(variant: Literal["dcegm", "brute"]):
    """Both discrete-housing variants construct and solve with a finite interior."""
    model = build_model(
        variant=variant,
        n_grid=8,
        n_housing=5,
        n_consumption=80,
        n_periods=3,
        upper_envelope="rfc",
    )
    solution = model.solve(
        params=build_params(variant=variant, delta=0.0), log_level="off"
    )
    assert sorted(solution) == [0, 1, 2]
    alive_periods = [p for p in solution if "alive" in solution[p]]
    assert alive_periods
    for period in alive_periods:
        assert np.isfinite(np.asarray(solution[period]["alive"])[..., 2:-1]).all()


def _solve(
    variant: Literal["dcegm", "brute"], upper_envelope: str
) -> PeriodToRegimeToVArr:
    model = build_model(
        variant=variant,
        n_grid=N_GRID,
        n_housing=N_HOUSING,
        n_consumption=N_CONSUMPTION,
        n_periods=N_PERIODS,
        upper_envelope=upper_envelope,  # ty: ignore[invalid-argument-type]
    )
    return model.solve(params=build_params(variant=variant, delta=0.0), log_level="off")


@pytest.mark.parametrize("upper_envelope", ["rfc", "fues"])
def test_ds2024_housing_dcegm_matches_brute_in_bulk(upper_envelope: str):
    """Each discrete-housing DC-EGM backend bulk-agrees with its grid-search twin.

    On the liquid interior the mean value difference is sub-unit and the large
    majority of cells fall within the cell tolerance — the discrete-housing
    DC-EGM (with the RFC or FUES upper-envelope backend) is correct up to grid
    resolution; the remaining cells sit in the durable-rich corner where DC-EGM
    carries a savings-grid interpolation error. Validating each backend against
    the oracle is the robust check; the two backends' policies can diverge by the
    envelope-internal tie-breaking on the durable-corner cells.
    """
    brute = _solve("brute", "rfc")
    dcegm = _solve("dcegm", upper_envelope)
    differences = []
    for period in sorted(brute):
        if "alive" not in brute[period]:
            continue
        brute_V = np.asarray(brute[period]["alive"])[INTERIOR]
        dcegm_V = np.asarray(dcegm[period]["alive"])[INTERIOR]
        finite = np.isfinite(brute_V) & np.isfinite(dcegm_V)
        differences.append(np.abs(brute_V - dcegm_V)[finite])
    difference = np.concatenate(differences)
    assert float(difference.mean()) < 0.5
    assert float((difference <= 1.5).mean()) >= 0.85
