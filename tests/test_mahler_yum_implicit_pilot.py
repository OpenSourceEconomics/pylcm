"""Implicit AD versus finite differences on the Mahler-Yum model.

The pilot objective is the paper-mode period-36 adjuster node solve itself
(see `lcm_examples.mahler_yum_2024.implicit_pilot`). Each cell follows one of
two numerically meaningful cases:

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
floor-induced kinks, while `test_outer_implicit_derivative.py` covers smooth
stationary points analytically.

What the pilot demonstrates today is narrower than that split suggests.
`capture_pilot_problem` builds no owner-provenance witness, and `run_pilot`
requires one, so every cell comes back `owner_missing` and the resolved
AD-versus-FD half of the contract has nothing to run on. The per-cell test
below skips with that reason rather than reporting a pass, so a green run is
never mistaken for real-model evidence of a certified implicit derivative.
Only `test_real_model_kink_is_materially_nonstationary`, whose screen does not
consult the witness, still carries real-model content.

Excluded from CI and run on request:

```
pytest tests/test_mahler_yum_implicit_pilot.py -m manual
```

The cost is not `_N_CELLS` or `_N_MESH` — those govern the pilot stage,
which runs only after the capture below. It is the capture itself: the
pilot builds the model with `enable_jit=False`, because the objective has
to stay traceable for `vmap` and the forward-mode JVP, and solving the
paper-scale model eagerly does not finish a single non-terminal period
within minutes. Shrinking the model is the lever here, not the knobs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.optimization.implicit_outer_derivative import ImplicitOptimumDiagnostics
from lcm_examples.mahler_yum_2024.implicit_pilot import (
    PilotReport,
    capture_pilot_problem,
    no_valid_derivative_reasons,
    run_pilot,
    select_pilot_cells,
)

_N_MESH = 7
_POLISH = 24
_RELATIVE_STEP = 1e-2
_N_CELLS = 1


def test_missing_complete_owner_witness_has_explicit_per_cell_reason() -> None:
    flags = jnp.asarray([False, False])
    missing = jnp.asarray([True, True])
    diagnostics = ImplicitOptimumDiagnostics(
        at_lower_bound=flags,
        at_upper_bound=flags,
        flat_curvature=flags,
        basin_tie=flags,
        nonstationary=flags,
        branch_certified=flags,
        owner_missing=missing,
        owner_incomplete=flags,
        owner_unresolved=flags,
        owner_primary_tie=flags,
        owner_changed=flags,
        unresolved=missing,
    )
    assert no_valid_derivative_reasons(diagnostics) == (
        "complete owner provenance is unavailable",
        "complete owner provenance is unavailable",
    )


@pytest.fixture(scope="module")
def pilot_report() -> PilotReport:
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    problem = capture_pilot_problem()
    cells = select_pilot_cells(problem=problem, n_cells=_N_CELLS)
    return run_pilot(
        problem=problem,
        cell_indices=cells,
        n_mesh=_N_MESH,
        polish_iterations=_POLISH,
        relative_step=_RELATIVE_STEP,
    )


@pytest.mark.slow
@pytest.mark.manual
def test_pilot_optimum_is_interior_and_finite(pilot_report: PilotReport) -> None:
    assert np.isfinite(pilot_report.f_star).all()
    assert (pilot_report.f_star > 0.0).all()
    assert (pilot_report.f_star < 1.0).all()
    # The guarded implicit tangent must be finite even on an unresolved cell.
    assert np.isfinite(pilot_report.ad_tangent).all()


@pytest.mark.slow
@pytest.mark.manual
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
    diag = pilot_report.diagnostics
    if np.all(np.asarray(diag.owner_missing)):
        pytest.skip(
            "unsupported provenance: the captured problem carries no "
            "owner-provenance witness, so every cell fails closed as "
            "owner_missing. Neither half of this contract is under test — no "
            "cell can be resolved, and the kink assertion below is reached "
            "only for a cell that failed on something other than the missing "
            "witness. Supply `PilotProblem.owner_provenance` to exercise it."
        )

    h = _RELATIVE_STEP * max(1.0, abs(pilot_report.theta_baseline))
    polish_width = (1.0 / (_N_MESH - 1)) * 0.618**_POLISH
    quantization_floor = polish_width / h
    band = 5.0 * (pilot_report.fd_error_estimate + quantization_floor)
    gap = np.abs(pilot_report.ad_tangent - pilot_report.fd_richardson)

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
            assert pilot_report.no_valid_derivative_reason[i], {
                "cell": pilot_report.cell_indices[i],
                "reason": pilot_report.no_valid_derivative_reason[i],
            }
            if not np.asarray(diag.owner_missing)[i]:
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
@pytest.mark.manual
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
