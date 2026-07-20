"""PR-8 merge gates for the paper-mode Mahler & Yum configuration.

Fast gates check the paper-mode equations against the brute-force module's
(the shared functions are imported, so only the *replaced* ones can drift)
and the solver wiring against the plan's target interface. The slow gates
run the terminal period's NNBEGM kernel once per mesh and assert the
continuous-outer contract on the real model: keeper dominance under the
closed-form fixed-cost fold, off-node (continuous) next-habit selections,
and coarse/fine outer-mesh convergence.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.solution.solvers as _solvers
from lcm import AdaptiveOuterMesh, LinSpacedGrid, UniformObservedFixedCost
from lcm_examples.mahler_yum_2024 import (
    START_PARAMS,
    create_inputs,
)
from lcm_examples.mahler_yum_2024 import (
    consumption as brute_consumption,
)
from lcm_examples.mahler_yum_2024.paper import (
    adapt_params_to_paper_mode,
    build_paper_solver,
    cash_on_hand,
    create_mahler_yum_model,
    dead_utility,
    effort_value,
    keep_effort,
    next_lagged_effort,
    raw_cash_on_hand,
    utility,
)

_CAPTURE_PERIOD = 36


# ---------------------------------------------------------------------------
# Fast gates: paper-mode equation regressions
# ---------------------------------------------------------------------------


def test_floored_budget_matches_brute_consumption_off_the_floor() -> None:
    """Off the floor, the declared schedule reproduces the brute budget.

    Brute: `c = max(net_income + R*wealth - saving, min_consumption)`.
    Paper: `c = cash_on_hand - saving` with
    `cash_on_hand = max(net_income + R*wealth, min_consumption)`.
    Wherever the brute clamp is slack the two coincide exactly; the
    deviation on the floor is the documented Fortran-semantics difference.
    """
    rng = np.random.default_rng(0)
    net_income = jnp.asarray(rng.uniform(0.0, 2.0, size=200))
    wealth = jnp.asarray(rng.uniform(0.0, 10.0, size=200))
    saving = jnp.asarray(rng.uniform(0.0, 5.0, size=200))
    rate = jnp.asarray(1.04**2.0)
    floor = jnp.asarray(0.35)

    brute = brute_consumption(
        net_income=net_income,
        wealth=wealth,
        saving=saving,
        gross_interest_rate=rate,
        min_consumption=floor,
    )
    raw = raw_cash_on_hand(
        net_income=net_income, wealth=wealth, gross_interest_rate=rate
    )
    paper = cash_on_hand(raw_cash_on_hand=raw, min_consumption=floor) - saving

    slack = np.asarray(brute) > floor
    assert slack.any()
    assert not slack.all()
    np.testing.assert_array_equal(np.asarray(paper)[slack], np.asarray(brute)[slack])
    # On the floor the schedule never yields MORE consumption than the
    # Fortran top-up, and both budgets share the floored total resources.
    np.testing.assert_array_less(np.asarray(paper)[~slack] - 1e-12, floor)


def test_effort_identities_hold() -> None:
    """The outer wiring is the identity chain the NNBEGM contract expects."""
    values = jnp.linspace(0.0, 1.0, 7)
    np.testing.assert_array_equal(next_lagged_effort(effort=values), values)
    np.testing.assert_array_equal(effort_value(next_lagged_effort=values), values)
    np.testing.assert_array_equal(keep_effort(lagged_effort=values), values)


def test_utility_drops_only_the_adjustment_penalty() -> None:
    """Paper flow utility is the brute utility net of the folded penalty."""
    out = utility(
        effort_cost=jnp.asarray(0.3),
        work_disutility=jnp.asarray(0.2),
        consumption_utility=jnp.asarray(1.5),
    )
    np.testing.assert_allclose(np.asarray(out), 1.0)


def test_dead_utility_is_identically_zero_on_the_wealth_axis() -> None:
    wealth = jnp.linspace(0.0, 30.0, 12)
    out = dead_utility(wealth=wealth, discount_type=jnp.asarray(0))
    np.testing.assert_array_equal(np.asarray(out), np.zeros(12))


def test_params_adapter_keeps_the_floor_and_renames_the_penalty() -> None:
    model_params, _ = create_inputs(
        seed=0, n_simulation_subjects=10, params=START_PARAMS
    )
    adapted = adapt_params_to_paper_mode(model_params)
    assert "min_consumption" in adapted
    assert adapted["min_consumption"] == model_params["min_consumption"]
    assert "adjustment_cost_penalty" not in adapted
    np.testing.assert_array_equal(
        np.asarray(adapted["adjustment_cost_scale"]["adjustment_cost_envelope"]),
        np.asarray(model_params["adjustment_cost_penalty"]["adjustment_cost_envelope"]),
    )


def test_paper_solver_wiring_matches_the_plan_interface() -> None:
    solver = build_paper_solver()
    assert solver.outer_action == "effort"
    assert solver.outer_post_decision == "next_lagged_effort"
    assert solver.outer_no_adjustment_candidate == "keep_effort"
    aggregator = solver.branch_aggregator
    assert isinstance(aggregator, UniformObservedFixedCost)
    assert aggregator.scale_function == "adjustment_cost_scale"
    assert (aggregator.lower, aggregator.upper) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Slow gates: terminal-period kernel on the real model
# ---------------------------------------------------------------------------


class _StopAfterCaptureError(Exception):
    pass


def _mesh(
    *, initial: int, max_nodes: int, rounds: int, tol: float
) -> AdaptiveOuterMesh:
    return AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=initial),
        max_nodes=max_nodes,
        max_refinement_rounds=rounds,
        value_atol=tol,
        value_rtol=tol,
        golden_iterations=24,
    )


def _solve_and_capture(mesh: AdaptiveOuterMesh) -> dict:
    """Solve periods 37 and 36 once, capturing keeper and collapse at 36."""
    captured: dict = {}
    original_keeper = _solvers._NNBEGMPeriodKernel._solve_keeper
    original_call = _solvers._NNBEGMPeriodKernel.__call__
    original_collapse = _solvers.collapse_continuous_candidate_bank

    def capturing_keeper(self, **kw):
        result = original_keeper(self, **kw)
        if kw["period"] == _CAPTURE_PERIOD:
            captured["keeper_V"] = np.asarray(result.V_arr)
        return result

    def capturing_collapse(**kw):
        collapse = original_collapse(**kw)
        search = collapse.value_search
        captured["search_x"] = np.asarray(search.x)
        captured["search_valid"] = np.asarray(search.valid)
        captured["refinement_gain"] = np.asarray(search.refinement_gain)
        return collapse

    def capturing_call(self, **kw):
        result = original_call(self, **kw)
        if kw["period"] == _CAPTURE_PERIOD:
            captured["V"] = np.asarray(result.V_arr)
            diagnostics = result.diagnostics
            assert diagnostics is not None
            assert diagnostics.adjustment_probability is not None
            captured["p_adjust"] = np.asarray(diagnostics.adjustment_probability)
            raise _StopAfterCaptureError
        return result

    model = create_mahler_yum_model(implementation="paper", outer_search=mesh)
    model_params, _ = create_inputs(
        seed=0, n_simulation_subjects=10, params=START_PARAMS
    )
    params = adapt_params_to_paper_mode(model_params)
    patcher = pytest.MonkeyPatch()
    patcher.setattr(_solvers._NNBEGMPeriodKernel, "_solve_keeper", capturing_keeper)
    patcher.setattr(_solvers._NNBEGMPeriodKernel, "__call__", capturing_call)
    patcher.setattr(_solvers, "collapse_continuous_candidate_bank", capturing_collapse)
    try:
        with pytest.raises(_StopAfterCaptureError):
            model.solve(params={"alive": params}, log_level="off")
    finally:
        patcher.undo()
    return captured


@pytest.fixture(scope="module")
def coarse_capture() -> dict:
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    return _solve_and_capture(_mesh(initial=9, max_nodes=33, rounds=3, tol=1e-3))


@pytest.fixture(scope="module")
def fine_capture() -> dict:
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    return _solve_and_capture(_mesh(initial=17, max_nodes=65, rounds=4, tol=1e-4))


@pytest.mark.slow
def test_captured_period_is_fully_finite(coarse_capture: dict) -> None:
    for name in ("keeper_V", "V", "p_adjust"):
        assert np.isfinite(coarse_capture[name]).all(), name


@pytest.mark.slow
def test_exact_keeper_dominance_under_the_fixed_cost_fold(
    coarse_capture: dict,
) -> None:
    """The closed-form fold can only add option value over the exact keeper.

    `E_chi[max(V_K, W* - B*chi)] >= V_K` pointwise, and where the analytic
    adjustment probability is zero the collapse equals the separately
    evaluated keeper exactly (plan rule: the keeper is never read through
    the adjuster surrogate).
    """
    keeper_V = coarse_capture["keeper_V"]
    collapsed = coarse_capture["V"]
    assert (collapsed >= keeper_V - 1e-10).all()
    never_adjust = coarse_capture["p_adjust"] == 0.0
    if never_adjust.any():
        np.testing.assert_allclose(
            collapsed[never_adjust], keeper_V[never_adjust], rtol=0, atol=0
        )


@pytest.mark.slow
def test_coarse_fine_outer_convergence(
    coarse_capture: dict, fine_capture: dict
) -> None:
    """Refining the outer mesh barely moves the captured value surface.

    No one-sided bound is asserted: period 36 re-solves on period 37's
    *changed* carry, so a finer mesh can shift values in either direction
    (a grid ladder is a refinement path, not a pointwise bound). The gate
    is two-sided closeness at solver accuracy — measured deviation between
    the 9->33 and 17->65 meshes is ~3e-4 in value units.
    """
    coarse_V = coarse_capture["V"]
    fine_V = fine_capture["V"]
    np.testing.assert_allclose(fine_V, coarse_V, atol=2e-3, rtol=1e-5)


@pytest.mark.slow
def test_next_habit_is_continuous_not_grid_snapped(coarse_capture: dict) -> None:
    """A material share of cells selects an off-node continuous next habit.

    The safeguarded argmax refines between exact mesh nodes; if every
    selected effort sat on a node the outer choice would still be the old
    grid search. Golden-section abscissae are irrational combinations, so
    distance-to-node is a robust off-grid witness; refinement must also
    have strictly improved the value somewhere.
    """
    valid = coarse_capture["search_valid"]
    x = coarse_capture["search_x"][valid]
    # Every exact mesh node is dyadic: initial linspace(0, 1, 9) plus at most
    # three rounds of interval bisection lands on the 1/64 lattice.
    nodes = np.linspace(0.0, 1.0, 65)
    distance_to_node = np.abs(x[..., None] - nodes[None, :]).min(axis=-1)
    off_node_share = (distance_to_node > 1e-8).mean()
    assert off_node_share > 0.05, f"only {off_node_share:.1%} off-node selections"
    gain = coarse_capture["refinement_gain"][valid]
    assert (gain > 0.0).any(), "continuous refinement never improved a value"
