"""End-to-end NNBEGM solve with the adaptive continuous outer search.

The direct collapse (the last alive period) competes the exact initial-mesh
nodes plus refined candidates against the keeper, so with the initial mesh
equal to the finite grid it can never lose a cell and should improve
somewhere whenever the true outer optimum is off-grid. Earlier periods
consume the changed (higher-resolution) carry through their own inner EGM
re-solve, so their per-cell values move both ways by approximation error —
there the tests assert bounded losses and winners outnumbering losers.
Solver diagnostics must arrive on every alive period's kernel result with a
converged mesh.
"""

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jax_config

import _lcm.solution.solvers as solvers_mod
from lcm import AdaptiveOuterMesh
from tests.test_models import n_nbegm_toy as toy

if TYPE_CHECKING:
    from _lcm.solution.contract import KernelResult
    from _lcm.typing import PeriodToRegimeToVArr

_PARAMS = {"discount_factor": 0.95}
_N_PERIODS = 3
_ALIVE_PERIODS = (0, 1)
# Tolerances sized to the toy: its 120 cells spread their optimum basins
# over the whole outer axis, so every basin-flanking interval must validate
# — at 1e-6 that needs more nodes than the budget affords.
_MESH = AdaptiveOuterMesh(
    initial_grid=toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)


def _solve(
    *,
    outer_search: AdaptiveOuterMesh | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PeriodToRegimeToVArr, dict[int, KernelResult]]:
    recorded: dict[int, KernelResult] = {}
    original_call = solvers_mod._NNBEGMPeriodKernel.__call__

    def recording_call(
        self: solvers_mod._NNBEGMPeriodKernel,
        **kwargs: object,
    ) -> KernelResult:
        result = original_call(self, **kwargs)  # ty: ignore[invalid-argument-type]
        recorded[cast("int", kwargs["period"])] = result
        return result

    monkeypatch.setattr(solvers_mod._NNBEGMPeriodKernel, "__call__", recording_call)
    solution = toy.build_model(
        variant="n_nbegm",
        n_periods=_N_PERIODS,
        outer_search=outer_search,
    ).solve(params=_PARAMS, log_level="off")
    return solution, recorded


def test_continuous_outer_never_loses_to_the_finite_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Domination where it is guaranteed; bounded reshuffling upstream.

    The *direct collapse* (the last alive period, which reads the shared
    terminal carry) competes every finite-grid node plus refined candidates,
    so it can never lose a cell and should gain somewhere. Earlier periods
    consume a *different, higher-resolution carry*, so their inner EGM
    re-solve moves per-cell values in both directions by approximation
    error (largest at the constrained low-wealth corner); there the honest
    claims are that winners dominate losers and losses stay bounded.
    """
    if not jax_config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    finite_solution, _ = _solve(outer_search=None, monkeypatch=monkeypatch)
    continuous_solution, recorded = _solve(outer_search=_MESH, monkeypatch=monkeypatch)
    for period in _ALIVE_PERIODS:
        v_finite = np.asarray(finite_solution[period]["alive"])
        v_continuous = np.asarray(continuous_solution[period]["alive"])
        both_finite = np.isfinite(v_finite) & np.isfinite(v_continuous)
        # Finite-feasible cells stay feasible.
        assert bool(np.all(np.isfinite(v_continuous[np.isfinite(v_finite)])))
        gain = v_continuous[both_finite] - v_finite[both_finite]
        if period == _ALIVE_PERIODS[-1]:
            assert float(gain.min()) >= -1e-10
            assert float(gain.max()) > 1e-8, "refinement never improved any cell"
        else:
            assert float(gain.min()) > -0.2
            n_winners = int((gain > 1e-10).sum())
            n_losers = int((gain < -1e-10).sum())
            assert n_winners > n_losers
    assert set(recorded) == set(_ALIVE_PERIODS)


def test_continuous_solve_publishes_converged_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not jax_config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    _, recorded = _solve(outer_search=_MESH, monkeypatch=monkeypatch)
    for period in _ALIVE_PERIODS:
        diagnostics = recorded[period].diagnostics
        assert diagnostics is not None
        assert int(diagnostics.outer_nodes_used) >= toy.N_OUTER
        assert not bool(diagnostics.unresolved_mask)
        assert float(diagnostics.max_outer_bracket_width) >= 0.0
        assert bool(
            jnp.all(
                ~(diagnostics.outer_at_lower_bound & diagnostics.outer_at_upper_bound)
            )
        )
