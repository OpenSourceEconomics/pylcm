"""DS-2026 Application 1 FUES Euler-error accuracy harness.

Application 1 of Dobrescu & Shanker (2026) is the deterministic discrete-retirement
model: log utility with a per-period work cost `tau`, a deterministic wage while
working, an absorbing retirement choice, and a constant gross return. The paper's
Table 2 reports the FUES accuracy column as the mean `log10` consumption Euler
error along a simulated sample path (Judd 1992). FUES is pylcm's default DC-EGM
upper envelope, so the harness solves the model with DC-EGM, simulates, and scores
the Euler equation along the working-regime path. The same harness scores the RFC
column (the paper's fourth method) by passing `upper_envelope="rfc"`.

These tests run a single small solve at a time (asset grid <= 1000, shortened
horizon) so they stay local-safe; the full paper grids {1000..10000} at T=50 are a
GPU/CI sweep.
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks.ds_replication.app1_retirement_accuracy import (
    app1_accuracy_table,
    app1_euler_error,
    sample_path_euler_error,
)

# Local-safe horizon: shorter than the paper's T=50 so a single solve+simulate is
# fast, but long enough that workers retire mid-path and the working-regime Euler
# equation has interior, non-switch points to score.
_LOCAL_N_PERIODS = 20
_LOCAL_N_SUBJECTS = 300


def test_app1_euler_error_is_finite_and_in_paper_ballpark():
    """The FUES Euler error at tau=1, n_grid=1000 is finite, negative, and sane.

    The mean log10 consumption Euler error is a negative number (more negative =
    more accurate); the paper reports roughly -1.6 for tau=1 at the full grids,
    so at a coarser local grid the metric sits in the -1.0 to -4.0 band.
    """
    error = app1_euler_error(
        tau=1.0,
        n_grid=1000,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    assert np.isfinite(error)
    assert -4.0 < error < -1.0


def test_app1_euler_error_improves_under_grid_refinement():
    """A finer asset grid yields a more accurate (more negative) FUES Euler error.

    The endogenous grid method nulls the Euler residual at the endogenous nodes;
    interpolating the policy back onto the coarse exogenous grid reintroduces it,
    so refining the grid drives the residual down. Going from 300 to 1000 asset
    points must shrink the mean log10 error by a clear margin.
    """
    coarse = app1_euler_error(
        tau=1.0,
        n_grid=300,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    fine = app1_euler_error(
        tau=1.0,
        n_grid=1000,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    assert -4.0 < coarse < -1.0
    assert -4.0 < fine < -1.0
    assert fine < coarse - 0.5


def test_app1_accuracy_table_has_one_row_per_cell():
    """The sweep returns one FUES Euler-error row per `(tau, n_grid)` cell."""
    table = app1_accuracy_table(
        taus=(1.0,),
        n_grids=(300, 1000),
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    assert list(table.columns) == ["tau", "n_grid", "fues_euler_error"]
    assert len(table) == 2
    assert table["fues_euler_error"].between(-4.0, -1.0).all()


def test_sample_path_euler_error_recovers_a_planted_residual():
    """A hand-built two-period working path reproduces its analytic Euler error.

    With `c_t` chosen so that `c_euler_t = c_{t+1} / (beta*(1+r))` overshoots `c_t`
    by exactly 10%, the relative deviation is 0.1 and the metric is `log10(0.1)`.
    The retirement-switch and constrained points are dropped, leaving only the
    single interior working-to-working transition.
    """
    beta, r = 0.96, 0.02
    c_next = 8.0
    c_euler = c_next / (beta * (1.0 + r))
    c_t = c_euler / 1.1  # so c_euler / c_t - 1 = 0.1
    panel = pd.DataFrame(
        {
            "subject_id": [0, 0, 0],
            "period": [0, 1, 2],
            "regime_name": ["working_life", "working_life", "retirement"],
            "labor_supply": ["work", "work", "retire"],
            "consumption": [c_t, c_next, 5.0],
            # Interior: consumption leaves strictly positive savings.
            "wealth": [c_t + 50.0, c_next + 50.0, 30.0],
        }
    )
    error = sample_path_euler_error(panel=panel, interest_rate=r, discount_factor=beta)
    assert error == pytest.approx(np.log10(0.1), abs=1e-9)


def test_app1_rfc_euler_error_is_in_the_same_regime_as_fues():
    """RFC-1D reproduces the FUES accuracy regime on Application 1.

    The rooftop-cut (`upper_envelope="rfc"`) and the FUES scan are both exact at
    the endogenous nodes and reintroduce the residual only through the policy
    interpolation onto the coarse grid, so on this retirement model they land in
    the same mean log10 Euler-error band.
    """
    config = {
        "tau": 1.0,
        "n_grid": 1000,
        "n_periods": _LOCAL_N_PERIODS,
        "n_subjects": _LOCAL_N_SUBJECTS,
        "seed": 0,
    }
    fues = app1_euler_error(upper_envelope="fues", **config)
    rfc = app1_euler_error(upper_envelope="rfc", **config)
    assert np.isfinite(rfc)
    assert -4.0 < rfc < -1.0
    assert abs(rfc - fues) < 1.0
