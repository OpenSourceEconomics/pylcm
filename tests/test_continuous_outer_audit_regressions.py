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
import numpy as np
import pytest

from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import (
    _consider,
    refine_outer_mesh,
    safeguarded_continuous_argmax,
)
from _lcm.egm.outer_search import AdaptiveOuterMesh
from _lcm.optimization.implicit_outer_derivative import (
    _continuous_outer_optimum_primal,
    implicit_optimum_diagnostics,
)
from lcm.exceptions import OuterSearchConvergenceError, RegimeInitializationError
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


# --- Rank robustness (KV 2014 two-asset EZ collaboration) -----------------
#
# The continuous-outer collapse runs *eagerly* (the host-side refinement and
# candidate-bank collapse have Python control flow), so every interpolant read
# dispatches primitive-by-primitive. The four-point slope estimate gathers
# `values[stencil]`, lifting the working arrays by two axes; the safeguarded
# argmax then adds a leading golden-section *bracket* axis. On a rank-4 model
# state (KV: two assets + a shock + a discrete regime) those combine to a
# rank-7 eager gather, which SIGSEGVs XLA's CPU runtime. The fix flattens the
# state/broadcast axes to one inside the interpolant (a pure reshape), capping
# every working array at rank 3. Mahler's rank-3 state never reached it; KV's
# rank-4 state did, and only through `fail_closed=False` (the inference default
# raises during refinement before the collapse ever runs).


def _cusp_surface(nodes: jnp.ndarray, state_shape: tuple[int, ...]) -> jnp.ndarray:
    """A per-cell kinked outer surface stacked on the shared mesh."""
    cusps = jnp.reshape(jnp.linspace(0.2, 0.8, int(np.prod(state_shape))), state_shape)
    lead = (slice(None),) + (None,) * len(state_shape)
    return -jnp.abs(nodes[lead] - cusps[None])


def test_high_rank_state_does_not_segfault_the_eager_outer_search() -> None:
    """A rank-4 state through the safeguarded argmax must not crash XLA/CPU.

    Pre-fix this segfaulted the interpreter (a rank-7 eager gather); the guard
    is that it now returns finite per-cell optima. A reintroduced regression
    would kill the test process outright, which is the loud signal we want.
    """
    nodes = jnp.linspace(0.0, 4.0, 129, dtype=jnp.float32)
    state_shape = (2, 2, 2, 2)  # rank 4 — the KV two-asset+shock+regime shape
    values = _cusp_surface(nodes, state_shape).astype(jnp.float32)
    interp = LocalCubicOuterInterpolant()

    result = safeguarded_continuous_argmax(
        lambda q: interp.evaluate(nodes=nodes, values=values, query=q),
        nodes=nodes,
        node_values=values,
        golden_iterations=16,
    )
    assert result.x.shape == state_shape
    assert bool(jnp.all(jnp.isfinite(result.x)))
    assert bool(jnp.all(jnp.isfinite(result.value)))


def test_worst_cell_dump_is_gated_and_changes_no_result(capsys, monkeypatch) -> None:
    """The `LCM_OUTER_DUMP_WORST_CELL` locator emits only when set, never else.

    It is a pure stderr read of data `_mark_intervals` already computes (to let a
    caller localize an outer-variable discontinuity without patching the package),
    so it must leave the refined mesh bit-identical whether or not it fires.
    """
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
        max_nodes=17,
        max_refinement_rounds=3,
        fail_closed=False,
    )
    nodes0 = jnp.linspace(0.0, 1.0, 5)

    monkeypatch.delenv("LCM_OUTER_DUMP_WORST_CELL", raising=False)
    off = refine_outer_mesh(
        initial_nodes=nodes0,
        solve_at=_per_cell_cusp_solve,
        config=config,
        fail_closed=False,
    )
    assert "LCM_OUTER_DUMP_WORST_CELL" not in capsys.readouterr().err

    monkeypatch.setenv("LCM_OUTER_DUMP_WORST_CELL", "1")
    on = refine_outer_mesh(
        initial_nodes=nodes0,
        solve_at=_per_cell_cusp_solve,
        config=config,
        fail_closed=False,
    )
    err = capsys.readouterr().err
    assert "LCM_OUTER_DUMP_WORST_CELL" in err
    assert "node_abscissa" in err
    assert "mid_exact_val" in err
    assert jnp.array_equal(off.nodes, on.nodes)
    assert off.max_validation_error == on.max_validation_error


def test_interpolant_read_is_invariant_to_state_axis_flattening() -> None:
    """The internal flatten must be a pure reshape: same reads at any rank.

    Reading the identical surface shaped `(N, 6)` versus `(N, 2, 3)` must give
    identical values and derivatives once reshaped back — this locks the fix's
    bit-for-bit-equivalence claim so a future refactor cannot silently perturb
    the interpolant while flattening.
    """
    interp = LocalCubicOuterInterpolant()
    nodes = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float64)
    flat_values = jnp.asarray(
        np.linspace(-1.0, 2.0, 9 * 6).reshape(9, 6), dtype=jnp.float64
    )
    query_flat = jnp.linspace(0.05, 0.95, 6, dtype=jnp.float64)

    v_flat, d_flat = interp.evaluate_with_derivative(
        nodes=nodes, values=flat_values, query=query_flat
    )
    v_nd, d_nd = interp.evaluate_with_derivative(
        nodes=nodes,
        values=flat_values.reshape(9, 2, 3),
        query=query_flat.reshape(2, 3),
    )
    assert jnp.array_equal(v_flat, v_nd.reshape(6))
    assert jnp.array_equal(d_flat, d_nd.reshape(6))


# --- F4 (round-2 audit): rounded ties must not flip the discrete outer action ---
#
# Two coupled defects let backend/precision rounding pick a different outer
# action despite the declared value tolerance: (a) the four-point slope
# reduction cancelled, so a mathematically constant surface grew a one-ULP
# interpolation peak; (b) the candidate fold compared values with strict `>` /
# `==` and no tie band, so a sub-ULP difference displaced the canonical smaller
# action. The fix centers the stencil (constant column -> exactly-zero slopes)
# and folds candidates within a dtype-aware ULP band by the smaller-abscissa
# rule (DC-1 / DC-2). The band is a rounding floor (`_TIE_BAND_ULPS` ULPs of the
# value dtype), NOT the mesh validation tolerance: a wider band would swallow the
# genuine off-node optima the continuous search exists to find.


def test_constant_surface_does_not_flip_the_outer_action() -> None:
    interp = LocalCubicOuterInterpolant()
    nodes = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.full((4,), 10.0)

    # Centering collapses the cancellation: the residual interpolation ripple
    # is at most a few ULP of the value scale, not an O(1e-1) spurious peak.
    dense = interp.evaluate(nodes=nodes, values=values, query=jnp.linspace(0, 3, 601))
    assert float(jnp.max(dense) - 10.0) < 1e-12

    # ... and the tie band resolves whatever ripple remains to the smaller node.
    result = safeguarded_continuous_argmax(
        lambda z: interp.evaluate(nodes=nodes, values=values, query=z),
        nodes=nodes,
        node_values=values,
        golden_iterations=16,
    )
    assert float(result.x) == 0.0


def test_below_tolerance_value_difference_resolves_to_the_smaller_action() -> None:
    one = np.float64(1.0)
    up = np.nextafter(one, np.inf)  # 2.22e-16 above 1, far below any real band

    # Within the band -> the smaller abscissa (action 0) wins, not the sub-ULP
    # larger value at action 1.
    tied = _consider(
        cand_x=jnp.array(1.0),
        cand_v=jnp.array(up),
        cand_width=jnp.array(0.0),
        best_x=jnp.array(0.0),
        best_v=jnp.array(one),
        second_x=jnp.array(1.0),
        second_v=jnp.array(-jnp.inf),
        width=jnp.array(0.0),
    )
    assert float(tied[0]) == 0.0

    # A genuine improvement beyond the band still wins ...
    real = _consider(
        cand_x=jnp.array(1.0),
        cand_v=jnp.array(2.0),
        cand_width=jnp.array(0.0),
        best_x=jnp.array(0.0),
        best_v=jnp.array(1.0),
        second_x=jnp.array(1.0),
        second_v=jnp.array(-jnp.inf),
        width=jnp.array(0.0),
    )
    assert float(real[0]) == 1.0

    # ... and a finite candidate always beats a -inf incumbent regardless of x
    # (the band is taken only where both values are finite).
    over_inf = _consider(
        cand_x=jnp.array(1.0),
        cand_v=jnp.array(5.0),
        cand_width=jnp.array(0.0),
        best_x=jnp.array(0.0),
        best_v=jnp.array(-jnp.inf),
        second_x=jnp.array(0.0),
        second_v=jnp.array(-jnp.inf),
        width=jnp.array(0.0),
    )
    assert float(over_inf[0]) == 1.0


def test_tie_band_does_not_swallow_a_genuine_off_node_advantage() -> None:
    """The band must be a rounding floor, not the mesh tolerance.

    A genuine off-node optimum beats its neighbouring node by ~h^2*curvature —
    on a refined outer mesh this is O(1e-4), vastly above the ULP band. Wiring
    the mesh validation tolerance (often ~1e-3) as the band folded every such
    advantage back to the node (0% off-node selections in the Mahler-Yum e2e
    solve). A candidate 1e-4 above the incumbent, at a LARGER abscissa, must
    still win — the search may not snap it to the smaller node.
    """
    won = _consider(
        cand_x=jnp.array(0.5),
        cand_v=jnp.array(np.float64(1.0) + 1e-4),
        cand_width=jnp.array(0.0),
        best_x=jnp.array(0.0),
        best_v=jnp.array(1.0),
        second_x=jnp.array(0.0),
        second_v=jnp.array(-jnp.inf),
        width=jnp.array(0.0),
    )
    assert float(won[0]) == 0.5


# --- F3 -----------------------------------------------------------------
# Nonfinite-sentinel arithmetic in the mesh validator discarded a finite
# feasible island and reported a zero residual over a genuinely-unresolved
# interval. Two reachable cases, each RED before the `_mark_intervals` fix.


def test_finite_off_node_island_is_not_discarded() -> None:
    """A cell feasible only *between* nodes must be refined, not dropped.

    Cell 1 is ``-inf`` at every initial node ``[0,.25,.5,.75,1]`` but carries a
    narrow feasible bump peaking at ``x=0.375`` (the midpoint of ``[.25,.5]``).
    The old incumbent threshold ``-inf + rtol*inf`` was ``NaN``, so ``beats_best``
    read False and the island was silently discarded (``n_cells_all_invalid=1``,
    no finite node ever inserted). The ``has_finite_incumbent`` gate now samples
    it.
    """

    def solve_at(x: jnp.ndarray) -> jnp.ndarray:
        ramp = 4.0 * x  # cell 0: ordinary rising ramp, feasible throughout
        island = jnp.where(  # cell 1: feasible only near x = 0.375
            jnp.abs(x - 0.375) < 0.1,
            5.0 - 50.0 * (x - 0.375) ** 2,
            -jnp.inf,
        )
        return jnp.stack([ramp, island], axis=-1)  # (M, 2)

    initial_nodes = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
    )

    # Precondition: cell 1 is infeasible at every initial node.
    assert bool(jnp.all(~jnp.isfinite(solve_at(initial_nodes)[:, 1])))

    result = refine_outer_mesh(
        initial_nodes=initial_nodes,
        solve_at=solve_at,
        config=config,
        fail_closed=False,
    )

    # The island is now captured: cell 1 has a finite node near its peak and is
    # no longer counted as an all-invalid cell.
    assert result.n_cells_all_invalid == 0
    cell1 = result.node_values[:, 1]
    assert bool(jnp.any(jnp.isfinite(cell1)))
    assert float(jnp.max(jnp.where(jnp.isfinite(cell1), cell1, -jnp.inf))) > 4.9


def test_unresolved_beats_best_interval_reports_infinite_residual() -> None:
    """A marked ``beats_best`` interval the interpolant can't read is not error 0.

    Node 0 is finite, node 1 is ``-inf``, and the midpoint value beats node 0 —
    so the interval is marked, but the cubic across a ``-inf`` endpoint reads
    ``-inf``. With the round budget spent and ``fail_closed=False`` the solve is
    honestly ``unresolved``; the old validator forced the residual to ``0`` (the
    "resolved to tolerance" reading) because ``interp_finite`` was False. It must
    read ``inf``.
    """

    def solve_at(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0.75, -jnp.inf, 4.0 * x + jnp.where(x > 0.1, 6.0, 0.0))

    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
        max_refinement_rounds=0,
    )
    result = refine_outer_mesh(
        initial_nodes=jnp.array([0.0, 1.0]),
        solve_at=solve_at,
        config=config,
        fail_closed=False,
    )
    assert result.unresolved is True
    assert not np.isfinite(result.max_validation_error)


def test_fail_closed_raises_on_unresolved_beats_best_interval() -> None:
    """The same configuration must fail closed under inference mode."""

    def solve_at(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0.75, -jnp.inf, 4.0 * x + jnp.where(x > 0.1, 6.0, 0.0))

    config = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
        max_refinement_rounds=0,
    )
    with pytest.raises(OuterSearchConvergenceError):
        refine_outer_mesh(
            initial_nodes=jnp.array([0.0, 1.0]),
            solve_at=solve_at,
            config=config,
            fail_closed=True,
        )


# --- F1 -----------------------------------------------------------------
# The fixed-radius two-sided slope probe accepted small but derivative-critical
# kinks: its threshold `rtol*|Q_ff|*delta` scales with the probe radius, so a
# kink whose slope jump is below it passes as stationary while the IFT tangent
# is order-one wrong. The multi-radius contraction test is amplitude-independent.


def _pinned_kink_objective(kink_coef: float):
    """Pro's F1 surface: a kink at f*=0.3 pins the optimum for |theta|<coef.

    ``Q(f, theta) = -coef*|f-0.3| - 0.5*(f-0.3)^2 + theta*(f-0.3)``. The
    maximizer stays at 0.3 for |theta| < coef, so ``df*/dtheta = 0``, yet the
    IFT tangent ``-Q_ftheta/Q_ff = -1/-1 = 1``. ``coef`` can be shrunk toward
    zero while that tangent error stays 1.
    """

    def objective(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return -kink_coef * jnp.abs(f - 0.3) - 0.5 * (f - 0.3) ** 2 + theta * (f - 0.3)

    return objective


@pytest.mark.parametrize("kink_coef", [0.004, 0.0005])
def test_small_amplitude_kink_is_flagged_nonstationary(kink_coef: float) -> None:
    objective = _pinned_kink_objective(kink_coef)
    theta = jnp.asarray(0.0)
    bounds = (jnp.asarray(0.0), jnp.asarray(1.0))
    f_star, _value, basin_margin = _continuous_outer_optimum_primal(
        objective, theta, bounds
    )
    # The primal pins the optimum at the kink, where the IFT tangent is wrong.
    assert abs(float(f_star) - 0.3) < 1e-3

    diag = implicit_optimum_diagnostics(
        objective,
        theta=theta,
        f_star=f_star,
        basin_margin=basin_margin,
        bounds=bounds,
    )
    # The single fixed-radius screen accepted this (jump ~2*coef below the
    # radius-scaled threshold); the contraction screen rejects it at any coef.
    assert bool(diag.nonstationary)
    assert bool(diag.unresolved)


def test_genuine_smooth_optimum_stays_resolved() -> None:
    """The contraction screen must not over-flag a smooth interior optimum."""

    def objective(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * (f - 0.3) ** 2 + theta * (f - 0.3)

    theta = jnp.asarray(0.0)
    bounds = (jnp.asarray(0.0), jnp.asarray(1.0))
    f_star, _value, basin_margin = _continuous_outer_optimum_primal(
        objective, theta, bounds
    )
    diag = implicit_optimum_diagnostics(
        objective,
        theta=theta,
        f_star=f_star,
        basin_margin=basin_margin,
        bounds=bounds,
    )
    assert not bool(diag.nonstationary)
    assert not bool(diag.unresolved)


# --- F2 -----------------------------------------------------------------
# Midpoint-only validation is mesh-relative, not a global safeguard: a peak
# narrower than the mesh that misses every sampled midpoint is neither seen nor
# bounded. An opt-in Lipschitz constant upgrades refinement to a certified
# branch-and-bound (Piyavskii-Shubert interval upper bounds).


def _narrow_peak_surface(nodes: jnp.ndarray) -> jnp.ndarray:
    """A ramp `4x` with a narrow spike at 0.3125 that peaks ABOVE the ramp top.

    On [0,.25,.5,.75,1] and their first-pass midpoints the spike is invisible
    (its width 0.04 falls between the samples), so the best node is the ramp top
    (value 4) while the true global max is `Q(0.3125) = 1.25 + 10 = 11.25`.
    """
    ramp = 4.0 * nodes
    bump = 10.0 * jnp.exp(-(((nodes - 0.3125) / 0.04) ** 2))
    return ramp + bump


def _f2_config(**overrides) -> AdaptiveOuterMesh:
    return AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
        value_atol=1e-2,
        value_rtol=1e-2,
        max_nodes=129,
        max_refinement_rounds=10,
        **overrides,
    )


def test_midpoint_validation_alone_misses_a_sub_mesh_peak() -> None:
    initial = jnp.linspace(0.0, 1.0, 5)
    # The spike is invisible at the initial midpoints too.
    assert float(_narrow_peak_surface(jnp.array([0.375]))[0]) < 4.0

    result = refine_outer_mesh(
        initial_nodes=initial,
        solve_at=_narrow_peak_surface,
        config=_f2_config(),
        fail_closed=False,
    )
    # Mesh-relative validation certifies a mesh that never saw the spike.
    assert float(jnp.max(result.node_values)) < 5.0


def test_certified_lipschitz_bound_captures_a_sub_mesh_peak() -> None:
    initial = jnp.linspace(0.0, 1.0, 5)
    # True Lipschitz constant of the surface is ~215; 250 is a safe upper bound.
    result = refine_outer_mesh(
        initial_nodes=initial,
        solve_at=_narrow_peak_surface,
        config=_f2_config(outer_lipschitz_bound=250.0),
        fail_closed=False,
    )
    # The certified branch-and-bound refines every interval whose Lipschitz upper
    # bound could beat the incumbent, so it samples the spike (peak ~11.25).
    assert float(jnp.max(result.node_values)) > 10.0
    peak_node = result.nodes[int(jnp.argmax(result.node_values))]
    assert abs(float(peak_node) - 0.3125) < 0.05


def test_outer_lipschitz_bound_must_be_positive() -> None:
    with pytest.raises(RegimeInitializationError, match="outer_lipschitz_bound"):
        _f2_config(outer_lipschitz_bound=0.0)
