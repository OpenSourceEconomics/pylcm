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
what simulation may read back: MSS rows certify every envelope crossing and so
qualify for the off-grid policy read, while FUES rows do not (segment identity
is a slope-threshold heuristic) and keep the grid-argmax path, like RFC/LTM.
The scored residual is therefore policy-interpolation error under MSS and
action-grid quantization error under the other backends — two different
quantities, not two estimates of one. The MSS-versus-FUES comparison below
verifies the two backends solve to the same value arrays before attributing
the accuracy gap to the read.

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
    build_app1_model,
    build_app1_params,
    sample_path_euler_error,
)

# The replication scores float64 accuracy floors (log10 Euler errors below the
# float32 quantization level), so the module runs with x64 enabled regardless
# of the suite's `--precision` flag.
pytestmark = pytest.mark.usefixtures("x64_enabled")

# Local-safe horizon: shorter than the paper's T=50 so a single solve+simulate is
# fast, but long enough that workers retire mid-path and the working-regime Euler
# equation has interior, non-switch points to score.
_LOCAL_N_PERIODS = 20
_LOCAL_N_SUBJECTS = 300


def test_app1_fues_euler_error_sits_at_the_action_grid_floor():
    """The FUES Euler error at tau=1, n_grid=1000 is the action-grid floor.

    The mean log10 consumption Euler error is a negative number (more negative =
    more accurate); the paper reports roughly -1.6 for tau=1 at the full grids.
    FUES rows are not certified crossing-complete (segment identity is a
    slope-threshold heuristic), so the regime keeps the grid-argmax simulate
    path and the metric sits at the action-grid quantization floor, in the
    -1.0 to -4.0 band.
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


def test_app1_mss_euler_error_is_below_the_grid_quantization_floor():
    """The MSS Euler error at tau=1, n_grid=1000 clears the action-grid floor.

    MSS rows certify every envelope crossing, so the working regime qualifies
    for the off-grid policy read: simulation re-decides the retirement branch
    at the subject's own resources and interpolates the winning branch's
    consumption off the action grid. The metric then measures the policy row's
    interpolation error rather than the action-grid spacing, and sits well
    below the -1.0 to -4.0 quantization band a grid argmax floors at.
    """
    error = app1_euler_error(
        tau=1.0,
        n_grid=1000,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
        upper_envelope="mss",
    )
    assert np.isfinite(error)
    assert error < -4.0


def test_app1_mss_euler_error_improves_under_grid_refinement():
    """A finer asset grid yields a more accurate (more negative) MSS Euler error.

    The endogenous grid method nulls the Euler residual at the endogenous nodes;
    the off-grid read reintroduces it only through interpolation between the
    refined nodes, so refining the grid drives the residual down. Going from
    300 to 1000 asset points must shrink the mean log10 error by a clear
    margin.
    """
    coarse = app1_euler_error(
        tau=1.0,
        n_grid=300,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
        upper_envelope="mss",
    )
    fine = app1_euler_error(
        tau=1.0,
        n_grid=1000,
        n_periods=_LOCAL_N_PERIODS,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
        upper_envelope="mss",
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


def test_app1_mss_read_gap_dwarfs_the_measured_solve_disagreement():
    """The MSS-FUES Euler-error gap is orders of magnitude beyond the solve gap.

    Each backend builds and solves its own model, so backend differences enter
    through the solve as well as through the read. This comparison measures
    both: the two backends' solved value arrays share the same finite mask and
    agree within `1e-3` relative on it (they refine envelopes differently at
    isolated kink nodes; most entries are identical), while the Euler-error
    metric differs by more than a full log10 unit — MSS qualifies for the
    off-grid policy read and scores policy-interpolation error, FUES keeps the
    grid argmax and floors at the action-grid spacing. The comparison is joint
    (backend plus simulation path): the measured value agreement is consistent
    with a large read-quantization component in the metric gap, but a
    value-level bound does not by itself bound policy or branch differences
    near ties, so the design does not isolate the read's contribution.
    """
    config = {
        "tau": 1.0,
        "n_grid": 1000,
        "n_periods": _LOCAL_N_PERIODS,
        "n_subjects": _LOCAL_N_SUBJECTS,
        "seed": 0,
    }
    params = build_app1_params(tau=1.0)
    v_by_backend = {}
    for backend in ("fues", "mss"):
        model = build_app1_model(
            n_grid=1000, n_periods=_LOCAL_N_PERIODS, upper_envelope=backend
        )
        v_by_backend[backend] = model.solve(params=params, log_level="off")
    for period, regime_to_v in v_by_backend["fues"].items():
        for regime_name, v_fues in regime_to_v.items():
            v_mss = v_by_backend["mss"][period][regime_name]
            finite = np.isfinite(np.asarray(v_fues))
            np.testing.assert_array_equal(
                finite,
                np.isfinite(np.asarray(v_mss)),
                err_msg=f"finite masks differ at period {period}, {regime_name}",
            )
            np.testing.assert_allclose(
                np.asarray(v_fues)[finite],
                np.asarray(v_mss)[finite],
                rtol=1e-3,
                err_msg=f"solved V differs at period {period}, {regime_name}",
            )

    fues = app1_euler_error(upper_envelope="fues", **config)
    mss = app1_euler_error(upper_envelope="mss", **config)
    assert np.isfinite(fues)
    assert -4.0 < fues < -1.0
    assert mss < fues - 1.0


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
