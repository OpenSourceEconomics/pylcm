"""Adaptive outer interpolation keeps each inner action's conditional profile."""

from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import (
    SafeguardedSearchResult,
    safeguarded_continuous_argmax,
)
from _lcm.solution.nnbegm import _BoundNNBEGM
from lcm import Model, post_decision_lower_bound
from lcm.consumption_savings_regime import NestedConsumptionSavingsRegime
from lcm.grids import ContinuousGrid, DiscreteGrid
from lcm.solvers import NBEGM, NNBEGM, AdaptiveOuterMesh
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.test_models import n_nbegm_discrete_toy as discrete_toy
from tests.test_models import n_nbegm_toy as smooth_toy


def _search(*, profiles: FloatND) -> SafeguardedSearchResult:
    """Optimize every trailing conditional cell on the same four exact nodes."""
    nodes = jnp.asarray((0.0, 1.0, 2.0, 3.0))
    interpolant = LocalCubicOuterInterpolant()
    return safeguarded_continuous_argmax(
        objective=lambda query: interpolant.evaluate(
            nodes=nodes, values=profiles, query=query
        ),
        nodes=nodes,
        node_values=profiles,
        golden_iterations=40,
    )


def _profiles() -> FloatND:
    """Two quadratic action profiles have different continuous outer maxima."""
    return jnp.asarray(((0.0, 2.0), (2.0, 0.0), (2.0, 0.0), (0.0, 2.0)))


def test_each_action_profile_reaches_its_own_continuous_maximum() -> None:
    """An interior action peak exceeds the interpolated meshwise action maximum."""
    profiles = _profiles()
    np.testing.assert_array_equal(jnp.max(profiles, axis=1), (2.0,) * 4)
    result = _search(profiles=profiles)
    np.testing.assert_array_almost_equal(
        result.value, (2.25, 2.0), decimal=DECIMAL_PRECISION
    )
    # Smooth optimizer abscissae have square-root-epsilon error, as in the
    # established continuous outer-carry conditional-marginal controls.
    eps = float(np.finfo(np.asarray(profiles).dtype).eps)
    np.testing.assert_allclose(
        result.x[0], 1.5, rtol=0.0, atol=max(1e-4, 16.0 * eps**0.5)
    )
    assert int(jnp.argmax(result.value)) == 0
    assert int(jnp.argmax(profiles[0])) == 1
    early = _search(profiles=jnp.max(profiles, axis=1))
    np.testing.assert_array_almost_equal(early.value, 2.0, decimal=DECIMAL_PRECISION)
    np.testing.assert_array_almost_equal(
        jnp.max(result.value) - early.value, 0.25, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize(
    ("weights", "expected", "chosen"),
    [((0.5, 0.5), (2.125, 2.125), 0), ((-0.5, 1.5), (1.875, 2.375), 1)],
    ids=("interpolation", "signed-extrapolation"),
)
def test_passive_blending_precedes_the_inner_action_maximum(
    *, weights: tuple[float, float], expected: tuple[float, float], chosen: int
) -> None:
    """Nodewise outer maxima retain action labels through signed passive blending."""
    profiles = _profiles()
    both_nodes = jnp.stack((profiles, profiles[:, ::-1]), axis=1)
    result = _search(profiles=both_nodes)
    np.testing.assert_array_almost_equal(
        result.value, ((2.25, 2.0), (2.0, 2.25)), decimal=DECIMAL_PRECISION
    )
    blended = jnp.sum(jnp.asarray(weights)[:, None] * result.value, axis=0)
    np.testing.assert_array_almost_equal(blended, expected, decimal=DECIMAL_PRECISION)
    assert int(jnp.argmax(blended)) == chosen
    premature_action_max = jnp.sum(jnp.asarray(weights) * jnp.max(result.value, axis=1))
    np.testing.assert_array_almost_equal(
        premature_action_max, 2.25, decimal=DECIMAL_PRECISION
    )


def _constant_floor() -> FloatND:
    """A callable lower bound has the same numerical level in every fixed cell."""
    return jnp.asarray(0.0)


def _adaptive_model(*, callable_floor: bool) -> Model:
    """Reuse all declared grids/economics of the existing hard-action nested toy."""
    baseline = discrete_toy.build_model(variant="n_nbegm", n_periods=2)
    original = cast("NestedConsumptionSavingsRegime", baseline.user_regimes["alive"])
    alive = replace(
        original,
        solver=NNBEGM(
            inner=NBEGM(savings_grid=smooth_toy.SAVINGS_GRID),
            outer_search=AdaptiveOuterMesh(
                initial_grid=smooth_toy.OUTER_GRID,
                max_nodes=513,
                max_refinement_rounds=10,
                value_atol=1e-4,
                value_rtol=1e-4,
                golden_iterations=40,
            ),
        ),
        constraints={
            "borrowing": post_decision_lower_bound(
                margin=original.liquid,
                lower=_constant_floor if callable_floor else 0.0,
            )
        },
    )
    return Model(
        regimes={"alive": alive, "dead": baseline.user_regimes["dead"]},
        regime_id_class=smooth_toy.RegimeId,
        ages=baseline.ages,
        edges=baseline.edges,
        fixed_params=baseline.fixed_params,
        initial_nodes=baseline.initial_nodes,
    )


def test_adaptive_hard_action_model_accepts_a_constant_callable_floor() -> None:
    """A fixed-cell scalar floor preserves the admitted nested action domain."""
    literal = _adaptive_model(callable_floor=False)
    control = cast("NestedConsumptionSavingsRegime", literal.user_regimes["alive"])
    assert isinstance(control.solver, _BoundNNBEGM)
    assert isinstance(control.solver.outer_search, AdaptiveOuterMesh)
    control_action_grid = control.actions["buy_private"]
    assert isinstance(control_action_grid, DiscreteGrid)
    control_wealth_grid = control.states["wealth"]
    assert isinstance(control_wealth_grid, ContinuousGrid)
    np.testing.assert_array_equal(control_action_grid.to_jax(), (0, 1))

    np.testing.assert_array_equal(
        control_wealth_grid.to_jax(), smooth_toy.WEALTH_GRID.to_jax()
    )
    np.testing.assert_array_equal(
        control.solver.inner.savings_grid.to_jax(), smooth_toy.SAVINGS_GRID.to_jax()
    )
    feature = _adaptive_model(callable_floor=True)
    actual = cast("NestedConsumptionSavingsRegime", feature.user_regimes["alive"])
    assert isinstance(actual.solver, _BoundNNBEGM)
    actual_action_grid = actual.actions["buy_private"]
    assert isinstance(actual_action_grid, DiscreteGrid)
    np.testing.assert_array_equal(
        actual_action_grid.to_jax(), control_action_grid.to_jax()
    )
    np.testing.assert_array_equal(
        actual.solver.inner.savings_grid.to_jax(),
        control.solver.inner.savings_grid.to_jax(),
    )
    continuation = feature._regimes["alive"].solution.continuation_spec
    assert isinstance(continuation, EGMContinuationSpec)
    query = continuation.layout.query_spec
    assert query is not None
    assert query.action_names == ("buy_private",)
    assert query.adaptive_outer is not None
    assert query.adaptive_outer.adjuster_capacity == 513
    assert query.adaptive_outer.golden_iterations == 40
    assert query.adaptive_outer.outer_state == "illiquid"
    assert query.adaptive_outer.outer_action == "illiquid_investment"
