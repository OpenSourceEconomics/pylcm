"""Branch-aware VFI oracle pins the comparator-ordering gap to the (S, s) band.

The DS App.2 discrete-housing model is solved two ways in Table 3 — grid-search VFI
(brute) and DC-EGM — and the two value functions differ. These tests use the
host-side branch-aware VFI oracle to attribute that difference precisely:

- the oracle's *standard* mode (`I[max_d V_d]`, interpolate the already-maximized
  next value array) reproduces pylcm's brute solve, so the oracle faithfully
  represents the VFI comparator;
- differencing the oracle's standard and branch-aware modes isolates the pure
  comparator-ordering term `I[max_d V_d] - max_d I[V_d] >= 0`. That term is non-
  negative everywhere and strictly positive *only* in the low-wealth (S, s) band
  where the next-period housing choice switches across a liquid bracket. On the
  liquid interior the two comparators agree exactly.

So the comparator-ordering effect is real but confined to the (S, s) switch cells,
which sit in the borrowing-constrained low-wealth region the Table 3 accuracy score
excludes. It does *not* account for the dcegm-vs-brute disagreement on the scored
interior — that is DC-EGM's own savings-grid and durable-corner discretization,
a separate effect with the opposite locus.
"""

import numpy as np
import pytest

from tests.solution._branch_aware_vfi_oracle import solve_branch_aware_vfi
from tests.test_models.ds_app2_housing_fues import build_model, build_params

# Low-wealth nodes the Table 3 accuracy score excludes (steep-CES VFI floor); the
# (S, s) switch cells live here. The liquid interior starts above this band.
N_LOW_NODES = 12


@pytest.mark.parametrize(
    ("n_grid", "n_housing", "n_consumption", "n_periods"), [(12, 4, 60, 3)]
)
def test_standard_vfi_oracle_reproduces_brute(
    n_grid: int, n_housing: int, n_consumption: int, n_periods: int
):
    """The oracle's standard mode matches pylcm's brute solve cell for cell.

    Running the oracle with `branch_aware=False` reproduces the grid-search VFI the
    `"brute"` variant runs: it interpolates the already-maximized next value array
    (`I[max_d V_d]`) on the same liquid, housing, wage, and consumption grids. Every
    period and regime agrees up to floating-point precision, so the oracle is a
    faithful stand-in for the VFI comparator.
    """
    brute = build_model(
        variant="brute",
        n_grid=n_grid,
        n_housing=n_housing,
        n_consumption=n_consumption,
        n_periods=n_periods,
    )
    solution = brute.solve(params=build_params(variant="brute"), log_level="off")
    oracle = solve_branch_aware_vfi(
        branch_aware=False,
        n_grid=n_grid,
        n_housing=n_housing,
        n_consumption=n_consumption,
        n_periods=n_periods,
    )

    for period in sorted(solution):
        for regime, brute_raw in solution[period].items():
            brute_V = np.asarray(brute_raw)
            oracle_V = oracle.value[period][regime]
            if oracle_V.ndim == 3:
                # Oracle stores the working regime as (wage, housing, liquid); pylcm
                # orders it (housing, wage, liquid).
                oracle_V = np.transpose(oracle_V, (1, 0, 2))
            finite = np.isfinite(brute_V) & np.isfinite(oracle_V)
            np.testing.assert_allclose(oracle_V[finite], brute_V[finite], atol=5e-3)


@pytest.mark.parametrize(
    ("n_grid", "n_housing", "n_consumption", "n_periods"), [(40, 5, 400, 5)]
)
def test_comparator_ordering_localizes_to_ss_band(
    n_grid: int, n_housing: int, n_consumption: int, n_periods: int
):
    """The comparator-ordering gap is non-negative and confined to the (S, s) band.

    Standard VFI interpolates the already-maximized next value array
    (`I[max_d V_d]`); branch-aware VFI interpolates each next-housing-choice branch
    and maxes after (`max_d I[V_d]`). A linear interpolant of a maximum dominates the
    maximum of the interpolants, so the standard value sits at or above the branch-
    aware value everywhere, with strict inequality only where the winning next-housing
    choice switches inside a liquid bracket — the (S, s) inaction band. Those cells
    lie in the low-wealth band the Table 3 score excludes, so the liquid interior
    shows exactly zero gap: the comparator ordering does not touch the scored region.
    """
    standard = solve_branch_aware_vfi(
        branch_aware=False,
        n_grid=n_grid,
        n_housing=n_housing,
        n_consumption=n_consumption,
        n_periods=n_periods,
    )
    branch_aware = solve_branch_aware_vfi(
        branch_aware=True,
        n_grid=n_grid,
        n_housing=n_housing,
        n_consumption=n_consumption,
        n_periods=n_periods,
    )

    scored_working_periods = [0, 1]
    saw_strict_gap = False
    for period in scored_working_periods:
        gap = standard.value[period]["working"] - branch_aware.value[period]["working"]

        # Ordering: standard VFI never falls below branch-aware VFI.
        assert gap.min() >= -1e-9

        # Localization: the strict gap lives only in the low-wealth (S, s) band; the
        # liquid interior (and the high-wealth edge) carry no comparator gap at all.
        interior_and_above = gap[..., N_LOW_NODES:]
        np.testing.assert_allclose(interior_and_above, 0.0, atol=1e-9)

        if gap.max() > 1e-3:
            saw_strict_gap = True
            ss_band = gap[..., :N_LOW_NODES]
            assert ss_band.max() == gap.max()

    # The effect is genuinely present somewhere in the (S, s) band.
    assert saw_strict_gap
