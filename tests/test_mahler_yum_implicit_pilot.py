"""PR-12's Mahler pilot gate: implicit AD vs FD on the real model.

The pilot objective is the paper-mode period-36 adjuster node solve itself
(see `lcm_examples.mahler_yum_2024.implicit_pilot`). Plan section 19.3 is
applied per cell, with the branch structure the pilot itself uncovered:

- a cell whose outer optimum is a genuine smooth interior stationary point
  is RESOLVED — its AD tangent must agree with the Richardson-extrapolated
  central difference within the FD method's own uncertainty (AD is rejected
  on disagreement, not FD);
- a cell whose outer optimum sits at a KINK is a *diagnosed* failure of the
  local-normal calculus: `Q_f(f*)` is sign-definite and material, the
  stationarity screen must flag it UNRESOLVED, and no AD-vs-FD agreement is
  required — but the guarded implicit tangent must still be finite so a
  vectorized caller is not poisoned.

On the real paper-mode model the consumption floor makes the value
non-smooth in effort, so the interior optima found at this period are
floor-induced kinks; the gate therefore exercises the second branch on the
real model and the first branch is covered analytically in
`test_outer_implicit_derivative.py`. The full AD-vs-FD agreement on a
resolved *moment* is the GPU-scale gate that lands with the estimation
pipeline.
"""

import jax
import numpy as np
import pytest

from lcm_examples.mahler_yum_2024.implicit_pilot import (
    PilotReport,
    capture_pilot_problem,
    run_pilot,
    select_pilot_cells,
)

_N_MESH = 7
_POLISH = 24
_RELATIVE_STEP = 1e-2
_N_CELLS = 1


@pytest.fixture(scope="module")
def pilot_report() -> PilotReport:
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    problem = capture_pilot_problem()
    cells = select_pilot_cells(problem, n_cells=_N_CELLS)
    return run_pilot(
        problem,
        cells,
        n_mesh=_N_MESH,
        polish_iterations=_POLISH,
        relative_step=_RELATIVE_STEP,
    )


@pytest.mark.slow
def test_pilot_optimum_is_interior_and_finite(pilot_report: PilotReport) -> None:
    assert np.isfinite(pilot_report.f_star).all()
    assert (pilot_report.f_star > 0.0).all()
    assert (pilot_report.f_star < 1.0).all()
    # The guarded implicit tangent must be finite even on an unresolved cell.
    assert np.isfinite(pilot_report.ad_tangent).all()


@pytest.mark.slow
def test_each_cell_is_either_resolved_and_agrees_or_a_diagnosed_kink(
    pilot_report: PilotReport,
) -> None:
    """Section 19.3, per cell: resolved => AD ~ FD; unresolved => diagnosed.

    The band is the FD method's own uncertainty (the Richardson error proxy
    plus the argmax-quantization floor). A resolved cell must agree; an
    unresolved cell must be a genuinely diagnosed failure — here the
    stationarity screen must be the flag that fired (the real-model kink),
    not a silent miss.
    """
    h = _RELATIVE_STEP * max(1.0, abs(pilot_report.theta_baseline))
    polish_width = (1.0 / (_N_MESH - 1)) * 0.618**_POLISH
    quantization_floor = polish_width / h
    band = 5.0 * (pilot_report.fd_error_estimate + quantization_floor)
    gap = np.abs(pilot_report.ad_tangent - pilot_report.fd_richardson)

    diag = pilot_report.diagnostics
    resolved = ~np.asarray(diag.unresolved)
    nonstationary = np.asarray(diag.nonstationary)
    for i in range(pilot_report.f_star.shape[0]):
        if resolved[i]:
            assert gap[i] <= band[i], {
                "cell": pilot_report.cell_indices[i],
                "ad": pilot_report.ad_tangent[i],
                "richardson": pilot_report.fd_richardson[i],
                "gap": gap[i],
                "band": band[i],
            }
        else:
            # The only local-normal failure the real model exposes at this
            # period is the floor-induced kink; it must be what flagged it.
            assert nonstationary[i], {
                "cell": pilot_report.cell_indices[i],
                "q_f": pilot_report.q_f[i],
                "flags": {
                    "at_lower": bool(np.asarray(diag.at_lower_bound)[i]),
                    "at_upper": bool(np.asarray(diag.at_upper_bound)[i]),
                    "flat": bool(np.asarray(diag.flat_curvature)[i]),
                    "tie": bool(np.asarray(diag.basin_tie)[i]),
                },
            }


@pytest.mark.slow
def test_real_model_kink_is_materially_nonstationary(
    pilot_report: PilotReport,
) -> None:
    """Where the screen fires, Q_f(f*) is genuinely far from zero.

    Guards against the screen firing on rounding noise: an unresolved-by-
    stationarity cell must carry a first-order residual that dwarfs the
    residual a smooth optimum would leave, `|Q_ff| * bracket_width`.
    """
    diag = pilot_report.diagnostics
    nonstationary = np.asarray(diag.nonstationary)
    if not nonstationary.any():
        pytest.skip("no kink cell in this pilot sample")
    width = (1.0 / (_N_MESH - 1)) * 0.618**_POLISH
    smooth_residual = np.abs(pilot_report.q_ff) * width
    flagged = nonstationary
    assert (np.abs(pilot_report.q_f)[flagged] > 10.0 * smooth_residual[flagged]).all()
