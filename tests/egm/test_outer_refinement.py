"""Safeguarded continuous argmax and the adaptive shared outer mesh.

The PR-4 refinement battery: analytic unimodal surfaces, multimodal surfaces
where a single full-domain golden search would risk the wrong basin,
boundary optima kept exactly, the exact-node safeguard against a broken
surrogate, deterministic flat-tie resolution, invalid cells, adaptive-mesh
convergence with frozen-mesh replay, and candidate-budget failure modes.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import (
    refine_outer_mesh,
    safeguarded_continuous_argmax,
)
from _lcm.egm.outer_search import AdaptiveOuterMesh
from lcm import LinSpacedGrid
from lcm.exceptions import OuterSearchConvergenceError
from lcm.typing import Float1D, FloatND

_GOLDEN_ITERATIONS = 40


# ---------------------------------------------------------------------------
# Safeguarded continuous argmax
# ---------------------------------------------------------------------------


def test_unimodal_interior_maxima_beat_the_coarse_grid() -> None:
    centers = jnp.array([0.31, 0.57, 0.82])

    def objective(x: FloatND) -> FloatND:
        return -((x - centers) ** 2)

    nodes = jnp.linspace(0.0, 1.0, 7)
    result = safeguarded_continuous_argmax(
        objective,
        nodes=nodes,
        node_values=objective(nodes[:, None]),
        golden_iterations=_GOLDEN_ITERATIONS,
    )
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(centers), atol=1e-6)
    assert bool(jnp.all(result.refinement_gain >= 0.0))
    assert bool(jnp.all(result.value >= result.coarse_value))


def test_multimodal_surface_selects_the_global_basin() -> None:
    """A tall narrow peak next to a wide low one — the case that breaks a
    single full-domain golden search."""

    def objective(x: FloatND) -> FloatND:
        tall = 1.0 * jnp.exp(-(((x - 0.2) / 0.03) ** 2))
        wide = 0.6 * jnp.exp(-(((x - 0.75) / 0.2) ** 2))
        return tall + wide

    nodes = jnp.linspace(0.0, 1.0, 21)
    result = safeguarded_continuous_argmax(
        objective,
        nodes=nodes,
        node_values=objective(nodes[:, None]),
        golden_iterations=_GOLDEN_ITERATIONS,
    )
    np.testing.assert_allclose(float(result.x[0]), 0.2, atol=1e-4)
    # The margin diagnostic is ordered and finite.
    assert float(result.second_best_value[0]) <= float(result.value[0])
    assert np.isfinite(float(result.second_best_value[0]))


def test_boundary_maximum_is_kept_exactly() -> None:
    def objective(x: FloatND) -> FloatND:
        return 2.0 * x

    nodes = jnp.linspace(0.0, 1.0, 5)
    result = safeguarded_continuous_argmax(
        objective,
        nodes=nodes,
        node_values=objective(nodes[:, None]),
        golden_iterations=_GOLDEN_ITERATIONS,
    )
    assert float(result.x[0]) == 1.0
    assert float(result.value[0]) == 2.0
    assert bool(result.at_upper_bound[0])
    assert not bool(result.at_lower_bound[0])


def test_flat_surface_ties_break_to_the_smallest_abscissa() -> None:
    nodes = jnp.linspace(2.0, 5.0, 9)
    result = safeguarded_continuous_argmax(
        jnp.zeros_like,
        nodes=nodes,
        node_values=jnp.zeros((9, 1)),
        golden_iterations=17,
    )
    assert float(result.x[0]) == 2.0
    assert bool(result.at_lower_bound[0])


def test_all_invalid_cell_is_reported_invalid() -> None:
    nodes = jnp.linspace(0.0, 1.0, 5)
    values = jnp.stack([jnp.full(5, -jnp.inf), jnp.linspace(0.0, 1.0, 5)], axis=1)
    result = safeguarded_continuous_argmax(
        lambda x: jnp.broadcast_to(x, x.shape),
        nodes=nodes,
        node_values=values,
        golden_iterations=10,
    )
    assert not bool(result.valid[0])
    assert float(result.value[0]) == -jnp.inf
    assert bool(result.valid[1])


def test_broken_surrogate_cannot_lose_the_exact_winner() -> None:
    """With a surrogate that returns -inf everywhere, the selection falls
    back to the best exact node — the global safeguard in action."""
    nodes = jnp.linspace(0.0, 1.0, 11)
    exact = -((nodes[:, None] - 0.42) ** 2)

    def broken(x: FloatND) -> FloatND:
        return jnp.full(jnp.broadcast_shapes(x.shape, (1,)), -jnp.inf)

    result = safeguarded_continuous_argmax(
        broken,
        nodes=nodes,
        node_values=exact,
        golden_iterations=_GOLDEN_ITERATIONS,
    )
    assert float(result.x[0]) == pytest.approx(0.4)  # nearest node to 0.42
    assert float(result.value[0]) == pytest.approx(float(jnp.max(exact)))
    assert float(result.refinement_gain[0]) == 0.0
    assert float(result.bracket_width[0]) == 0.0


# ---------------------------------------------------------------------------
# Adaptive shared outer mesh
# ---------------------------------------------------------------------------

_MESH_CONFIG = AdaptiveOuterMesh(
    initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=9),
    max_nodes=513,
    max_refinement_rounds=12,
    value_atol=1e-7,
    value_rtol=1e-7,
)


def _smooth_surface(nodes: Float1D) -> FloatND:
    centers = jnp.array([0.3, 0.6])
    return jnp.sin(3.0 * nodes[:, None]) - (nodes[:, None] - centers) ** 2


def test_adaptive_mesh_converges_on_a_smooth_surface() -> None:
    result = refine_outer_mesh(
        initial_nodes=jnp.linspace(0.0, 1.0, 9),
        solve_at=_smooth_surface,
        config=_MESH_CONFIG,
    )
    assert not result.unresolved
    assert result.rounds_used >= 1
    assert result.nodes.shape[0] == result.node_values.shape[0]
    # The final interpolant reproduces exact solves at fresh probes.
    probes = jnp.array([0.111, 0.333, 0.555, 0.777, 0.999])
    read = LocalCubicOuterInterpolant().evaluate(
        nodes=result.nodes,
        values=result.node_values,
        query=probes[:, None],
    )
    np.testing.assert_allclose(
        np.asarray(read), np.asarray(_smooth_surface(probes)), atol=1e-5
    )


def test_frozen_mesh_replay_needs_no_further_refinement() -> None:
    """Re-running on the converged mesh inserts nothing — the frozen-mesh
    replay property derivative batches rely on."""
    first = refine_outer_mesh(
        initial_nodes=jnp.linspace(0.0, 1.0, 9),
        solve_at=_smooth_surface,
        config=_MESH_CONFIG,
    )
    replay = refine_outer_mesh(
        initial_nodes=first.nodes,
        solve_at=_smooth_surface,
        config=_MESH_CONFIG,
    )
    assert replay.rounds_used == 0
    np.testing.assert_array_equal(np.asarray(replay.nodes), np.asarray(first.nodes))
    np.testing.assert_array_equal(
        np.asarray(replay.node_values), np.asarray(first.node_values)
    )


def test_node_budget_exhaustion_fails_closed() -> None:
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
        max_nodes=8,
        max_refinement_rounds=12,
        value_atol=1e-12,
        value_rtol=1e-12,
    )
    with pytest.raises(OuterSearchConvergenceError, match="node budget"):
        refine_outer_mesh(
            initial_nodes=jnp.linspace(0.0, 1.0, 5),
            solve_at=lambda nodes: jnp.sin(9.0 * nodes[:, None]),
            config=config,
        )


def test_round_budget_exhaustion_can_return_unresolved_in_dev_mode() -> None:
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
        max_nodes=513,
        max_refinement_rounds=1,
        value_atol=1e-12,
        value_rtol=1e-12,
    )
    result = refine_outer_mesh(
        initial_nodes=jnp.linspace(0.0, 1.0, 5),
        solve_at=lambda nodes: jnp.sin(9.0 * nodes[:, None]),
        config=config,
        fail_closed=False,
    )
    assert result.unresolved
    assert result.max_validation_error > 1.0


def test_infeasible_cells_are_counted_not_fatal() -> None:
    def surface(nodes: Float1D) -> FloatND:
        good = -((nodes[:, None] - 0.5) ** 2)
        return jnp.concatenate([good, jnp.full_like(good, -jnp.inf)], axis=1)

    result = refine_outer_mesh(
        initial_nodes=jnp.linspace(0.0, 1.0, 9),
        solve_at=surface,
        config=_MESH_CONFIG,
    )
    assert result.n_cells_all_invalid == 1
    assert not result.unresolved
