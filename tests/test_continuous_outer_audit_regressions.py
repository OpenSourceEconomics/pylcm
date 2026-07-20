"""Regressions for the continuous-outer external audit (bundle-contouter-r1).

One runnable case per confirmed finding, each RED before its fix and GREEN
after. Named to merge with the reviewer's own ``reusable_tests`` (RT*) when
those are recovered.

F6 — the adaptive mesh could silently discard a known global maximum. Mesh
validation scored the exact-vs-interpolated midpoint error ONLY on intervals
flanking a node-local maximum (the golden-section brackets). A peak sitting
between two nodes that are both on a monotone ramp (neither a local maximum)
was therefore never scored, never marked, never inserted, and never bracketed
— so the search returned the best *node* and missed the true optimum. The fix
also refines any interval whose EXACT midpoint value beats the cell's best
node, because that is an unsampled incumbent regardless of the interpolant.
"""

import jax
import jax.numpy as jnp
import pytest

from _lcm.egm.outer_refinement import _consider, refine_outer_mesh
from _lcm.egm.outer_search import AdaptiveOuterMesh
from _lcm.optimization.implicit_outer_derivative import implicit_optimum_diagnostics
from lcm.exceptions import OuterSearchConvergenceError
from lcm.grids import LinSpacedGrid


def _hidden_peak_objective(x: jnp.ndarray) -> jnp.ndarray:
    """A monotone ramp with a narrow bump hidden between two ramp nodes.

    On the initial nodes ``[0, .25, .5, .75, 1]`` the values are the ramp
    ``4x`` (bump ~0), so only the top node is a local maximum. The bump peaks
    at ``x = 0.375`` — the midpoint of ``[.25, .5]``, whose endpoints are both
    on the rising ramp — with height ``ramp + 10 ~= 11.5``.
    """
    ramp = 4.0 * x
    bump = 10.0 * jnp.exp(-(((x - 0.375) / 0.05) ** 2))
    return ramp + bump


def test_hidden_interior_peak_is_not_discarded() -> None:
    initial_nodes = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
    )

    # The bump is invisible at the initial nodes: the best node is the ramp top.
    node_values = _hidden_peak_objective(initial_nodes)
    assert float(jnp.max(node_values)) < 4.1

    result = refine_outer_mesh(
        initial_nodes=initial_nodes,
        solve_at=_hidden_peak_objective,
        config=config,
        fail_closed=False,
    )

    # The refiner sampled the hidden peak (a node landed on it, value ~11.5),
    # rather than validating a mesh that never saw anything above the ramp top.
    assert float(jnp.max(result.node_values)) > 10.0
    peak_node = result.nodes[int(jnp.argmax(result.node_values))]
    assert abs(float(peak_node) - 0.375) < 0.05


def test_feasibility_crash_does_not_exhaust_the_node_budget() -> None:
    """A steep, far-from-optimal value crash must not trigger the new rule.

    The beats-best refinement fires only on an EXACT midpoint that exceeds the
    best node, so a monotone decline toward a feasibility edge (whose midpoints
    are never new incumbents) leaves the mesh compact.
    """
    initial_nodes = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
    )

    def crashing(x: jnp.ndarray) -> jnp.ndarray:
        # Concave, single interior max at 0.0 then a sharp decline; every
        # midpoint lies below its flanking nodes on the declining stretch.
        return -100.0 * x**2

    result = refine_outer_mesh(
        initial_nodes=initial_nodes,
        solve_at=crashing,
        config=config,
        fail_closed=False,
    )
    assert result.nodes.shape[0] < 33


def test_runner_up_is_never_a_duplicate_of_the_winner() -> None:
    """F8: a candidate at the winner's own abscissa must not become second-best.

    The objective is a function of the abscissa, so such a candidate carries
    the winner's value; installing it as the runner-up would report a zero
    best/second margin (a phantom branch tie). A genuinely DISTINCT candidate
    that ties in value still becomes a legitimate runner-up.
    """
    best_x = jnp.array(1.0)
    best_v = jnp.array(10.0)
    second_x = jnp.array(0.0)
    second_v = jnp.array(-jnp.inf)
    width = jnp.array(0.0)

    # A duplicate of the winner (same x, hence same value) must be rejected.
    _, _, _, dup_second_v, _ = _consider(
        cand_x=jnp.array(1.0),
        cand_v=jnp.array(10.0),
        cand_width=jnp.array(0.0),
        best_x=best_x,
        best_v=best_v,
        second_x=second_x,
        second_v=second_v,
        width=width,
    )
    assert not jnp.isfinite(dup_second_v)  # runner-up stays empty, no phantom tie

    # A distinct abscissa that ties in value is a real runner-up (margin 0).
    _, _, dist_second_x, dist_second_v, _ = _consider(
        cand_x=jnp.array(2.0),
        cand_v=jnp.array(10.0),
        cand_width=jnp.array(0.0),
        best_x=best_x,
        best_v=best_v,
        second_x=second_x,
        second_v=second_v,
        width=width,
    )
    assert float(dist_second_x) == 2.0
    assert float(dist_second_v) == 10.0


def test_exact_kink_optimum_is_flagged_though_ad_reads_stationary() -> None:
    """F5: the stationarity screen must not trust a single forward-mode Q_f.

    A symmetric tent written through ``maximum(f-c, c-f)`` with the winner
    exactly on the peak: ``jax.jvp`` AVERAGES the two operand tangents (+1, -1)
    at the tie to 0, so the single forward-mode Q_f sees the point as
    stationary and would ship the invalid implicit tangent. Real curvature from
    an added quadratic keeps it off the flat screen, so only the two-sided
    slope probe — which sees the 2*theta jump — flags it.
    """
    bounds = (jnp.array([0.0]), jnp.array([1.0]))

    def kinked(f, theta):
        return -theta * jnp.maximum(f - 0.3, 0.3 - f) - 0.5 * (f - 0.3) ** 2

    theta = jnp.array(1.0)
    f_star = jnp.array([0.3])  # exactly on the kink

    # The single forward-mode Q_f the old screen relied on reads ~0 here.
    q_f = jax.jvp(lambda f: kinked(f, theta), (f_star,), (jnp.ones_like(f_star),))[1]
    assert abs(float(q_f[0])) < 1e-6

    diag = implicit_optimum_diagnostics(
        kinked,
        theta=theta,
        f_star=f_star,
        basin_margin=jnp.array([1.0]),  # not basin-tied
        bounds=bounds,
    )
    assert not bool(diag.flat_curvature[0])  # real curvature; not the flat screen
    assert bool(diag.nonstationary[0])
    assert bool(diag.unresolved[0])
    assert not bool(diag.at_lower_bound[0])
    assert not bool(diag.at_upper_bound[0])


def _per_cell_cusp_solve(nodes: jnp.ndarray) -> jnp.ndarray:
    """A per-cell kinked outer surface: cell i peaks at a cusp `c_i`.

    `-|x - c_i|` with the cusp location varying by cell is the KV two-asset
    borrowing-price kink in miniature — the cubic interpolant misses every
    near-cusp midpoint and the `beats_best` incumbent rule keeps marking a
    fresh near-cusp interval each round, so the mesh cannot validate to
    tolerance within a small budget (the creeping front).
    """
    cusps = jnp.linspace(0.2, 0.8, 8)
    return -jnp.abs(nodes[:, None] - cusps[None, :])


def test_fail_closed_true_raises_on_a_non_convergent_kinked_surface() -> None:
    """The inference-grade default still fails hard when the mesh cannot validate."""
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
        max_nodes=17,
        max_refinement_rounds=3,
    )
    with pytest.raises(OuterSearchConvergenceError):
        refine_outer_mesh(
            initial_nodes=jnp.linspace(0.0, 1.0, 5),
            solve_at=_per_cell_cusp_solve,
            config=config,
            fail_closed=True,
        )


def test_fail_closed_false_returns_best_effort_with_residual_surfaced() -> None:
    """Development mode degrades to a best-effort mesh instead of raising.

    The handoff's KV Stage-2 need: a good-enough outer optimum on a kinked
    surface, with the residual validation error REPORTED (not hidden) so the
    caller decides whether it is good enough — exactly `unresolved=True` plus a
    surfaced `max_validation_error`.
    """
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
        max_nodes=17,
        max_refinement_rounds=3,
        fail_closed=False,
    )
    assert config.fail_closed is False
    result = refine_outer_mesh(
        initial_nodes=jnp.linspace(0.0, 1.0, 5),
        solve_at=_per_cell_cusp_solve,
        config=config,
        fail_closed=config.fail_closed,
    )
    assert result.unresolved is True
    assert result.max_validation_error > 1.0  # residual measured and surfaced
    assert result.nodes.shape[0] >= 5  # a real (refined) best-effort mesh


def test_fail_closed_defaults_to_true() -> None:
    """The field defaults to the inference-grade behaviour."""
    assert (
        AdaptiveOuterMesh(
            initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5)
        ).fail_closed
        is True
    )
