"""DS-2026 Application 1 FUES Euler-error accuracy harness.

Application 1 of Dobrescu & Shanker (2026) is the deterministic discrete-retirement
model: log utility with a per-period work cost `tau`, a deterministic wage while
working, an absorbing retirement choice, and a constant gross return. The paper's
Table 2 reports the FUES accuracy column as the mean `log10` consumption Euler
error along a simulated sample path (Judd 1992). FUES is pylcm's default DC-EGM
upper envelope, so the harness solves the model with DC-EGM, simulates, and scores
the Euler equation along the working-regime path. The same harness scores the RFC
column (the paper's fourth method) by passing `upper_envelope="rfc"`.

The backend choice moves the metric by orders of magnitude, because it decides
what simulation may read back: FUES rows carry the envelope crossings and so
qualify for the off-grid policy read, while RFC rows do not and keep the
grid-argmax path. The scored residual is therefore policy-interpolation error
under FUES and action-grid quantization error under RFC — two different
quantities, not two estimates of one.

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
    app1_timing,
    sample_path_euler_error,
)

# Local-safe horizon: shorter than the paper's T=50 so a single solve+simulate is
# fast, but long enough that workers retire mid-path and the working-regime Euler
# equation has interior, non-switch points to score.
_LOCAL_N_PERIODS = 20
_LOCAL_N_SUBJECTS = 300


def test_app1_euler_error_is_finite_and_below_the_grid_quantization_floor():
    """The FUES Euler error at tau=1, n_grid=1000 clears the action-grid floor.

    The mean log10 consumption Euler error is a negative number (more negative =
    more accurate). The working regime qualifies for the off-grid policy read:
    FUES publishes crossing-complete rows, so simulation re-decides the
    retirement branch at the subject's own resources and interpolates the
    winning branch's consumption off the action grid. The metric then measures
    the policy row's interpolation error rather than the action-grid spacing,
    and sits well below the -1.0 to -4.0 quantization band a grid argmax floors
    at.
    """
    error = app1_euler_error(
        tau=1.0,
        n_grid=1000,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    assert np.isfinite(error)
    assert error < -4.0


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
    assert coarse < -4.0
    assert fine < -4.0
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
    assert (table["fues_euler_error"] < -4.0).all()


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


def test_app1_rfc_euler_error_sits_at_the_action_grid_floor_above_fues():
    """RFC-1D scores the action-grid floor; FUES clears it by orders of magnitude.

    The rooftop-cut (`upper_envelope="rfc"`) leaves each envelope switch between
    two retained nodes, so its rows carry no crossing topology and the regime
    keeps the grid-argmax simulate path — its consumption is quantized to the
    action grid and the Euler residual floors at that spacing. FUES publishes
    the crossings, qualifies for the off-grid read, and lands in a different
    accuracy regime entirely. The two backends agree on the solved value; they
    differ in what simulation is allowed to read back.
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
    assert fues < rfc - 1.0


def test_app1_timing_separates_compile_from_runtime():
    """The first solve times compile-plus-run; later solves time pure execution.

    JAX caches the compiled solve, so the compile cost (first-call time minus the
    steady-state runtime) is strictly positive, and the steady-state runtime is a
    finite, positive number.
    """
    timing = app1_timing(tau=1.0, n_grid=300, n_periods=_LOCAL_N_PERIODS, n_runs=2)
    assert np.isfinite(timing["compile_time"])
    assert np.isfinite(timing["runtime"])
    assert timing["runtime"] > 0.0
    assert timing["compile_time"] > 0.0


def test_app1_timing_measures_compile_after_a_warm_cache():
    """Compile time stays positive even if the same config was solved earlier.

    `app1_timing` clears the JAX compilation cache before its first solve, so a
    prior solve of the same `(method, shape)` does not warm-start the compile
    measurement into noise.
    """
    app1_euler_error(
        tau=1.0,
        n_grid=300,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    timing = app1_timing(tau=1.0, n_grid=300, n_periods=_LOCAL_N_PERIODS, n_runs=2)
    assert timing["compile_time"] > 0.0


def test_app1_taste_shock_variant_is_in_paper_ballpark():
    """The taste-shock retirement variant (Table 6, scale 0.05) scores a sane error.

    EV1 taste shocks leave the continuous consumption Euler equation intact but
    disqualify the regime from the off-grid policy read: the realized discrete
    draw perturbs the branch the solve conditioned on. Simulated consumption
    therefore keeps the grid-argmax path and the metric floors at the
    action-grid spacing, in the -1.0 to -4.0 band.
    """
    error = app1_euler_error(
        tau=1.0,
        n_grid=1000,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
        taste_shock_scale=0.05,
    )
    assert np.isfinite(error)
    assert -4.0 < error < -1.0
