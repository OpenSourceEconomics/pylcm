"""The branch re-decision value read follows the solve's Hermite convention.

The solve publishes conditional values by reading the refined row with the
cubic Hermite interpolant, using the marginal-utility row as exact node slopes
(envelope theorem). Simulation compares conditional branch values with the
same interpolant, so the ranking the re-decision sees is the ranking the solve
convention implies — a linear read of the same row can rank two close branches
differently.
"""

from types import MappingProxyType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import _lcm.simulation.simulate as simulation_module
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import safeguarded_continuous_argmax
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import Regime
from _lcm.simulation.simulate import (
    _interp_rows_with_support,
    _replace_continuous_action_with_policy_read,
)
from tests.conftest import DECIMAL_PRECISION


class _StubRegime(Regime):
    """Engine regime carrying only the fields the policy read reaches.

    Subclasses the real class so the runtime-typed perimeter accepts it, and
    writes the one field directly rather than building a whole compiled regime.
    """

    def __init__(self, *, simulation: object) -> None:
        object.__setattr__(self, "simulation", simulation)


def test_value_read_is_cubic_hermite_with_the_marginal_slopes():
    """On nodes `(1, 0)`, `(2, log 2)` with slopes `(1, 1/2)`, the read at `1.5`
    is the cubic Hermite value `0.40907`, not the linear chord `0.34657`."""
    sim_policy = EGMSimPolicy(
        endog_grid=jnp.array([1.0, 2.0]),
        policy=jnp.array([0.5, 1.0]),
        value=jnp.array([0.0, jnp.log(2.0)]),
        marginal_utility=jnp.array([1.0, 0.5]),
    )
    value, in_support = _interp_rows_with_support(
        sim_policy=sim_policy,
        field="value",
        index=(),
        resources=jnp.array([1.5]),
        n_subjects=1,
    )
    assert bool(in_support[0])
    assert float(value[0]) == pytest.approx(0.4090736, abs=1e-6)


def test_policy_read_stays_piecewise_linear():
    """The policy read is the linear chord: `0.75` midway between `0.5` and `1.0`.

    Only the value row carries exact node slopes (the marginal-utility row via
    the envelope theorem); the policy row has no slope data, so its read is
    piecewise linear.
    """
    sim_policy = EGMSimPolicy(
        endog_grid=jnp.array([1.0, 2.0]),
        policy=jnp.array([0.5, 1.0]),
        value=jnp.array([0.0, jnp.log(2.0)]),
        marginal_utility=jnp.array([1.0, 0.5]),
    )
    policy, _ = _interp_rows_with_support(
        sim_policy=sim_policy,
        field="policy",
        index=(),
        resources=jnp.array([1.5]),
        n_subjects=1,
    )
    assert float(policy[0]) == pytest.approx(0.75, abs=1e-9)


def test_policy_replacement_reports_the_value_of_the_emitted_action(monkeypatch):
    """An accepted off-grid action and its canonical value are published together."""
    read = SimpleNamespace(
        action_name="consumption",
        savings_lower_bound=0.0,
    )
    regime = _StubRegime(
        simulation=SimpleNamespace(
            egm_policy_read=read,
            grids={},
        )
    )
    sim_policy = EGMSimPolicy(
        endog_grid=jnp.array([0.0, 1.0]),
        policy=jnp.array([0.2, 0.8]),
        value=jnp.array([1.0, 2.0]),
        marginal_utility=jnp.array([1.0, 0.5]),
    )
    monkeypatch.setattr(
        simulation_module,
        "_resources_at_subjects",
        lambda **_kwargs: jnp.array([1.0]),
    )
    monkeypatch.setattr(
        simulation_module,
        "_interp_rows_with_support",
        lambda **_kwargs: (jnp.array([0.6]), jnp.array([True])),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonical_Q_at_actions",
        lambda **_kwargs: (jnp.array([7.0]), jnp.array([True])),
    )

    _, reported_value, _ = _replace_continuous_action_with_policy_read(
        optimal_actions=MappingProxyType({"consumption": jnp.array([0.5])}),
        regime=regime,
        sim_policy=sim_policy,
        states=MappingProxyType({"wealth": jnp.array([1.0])}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.array([1.0])}),
        action_names=("consumption",),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=jnp.array([5.0]),
    )

    assert float(reported_value[0]) == 7.0


def test_nested_policy_replacement_keeps_only_canonically_better_pairs(
    monkeypatch,
) -> None:
    """Nested actions and values move together only for feasible improvements."""
    regime = _StubRegime(simulation=SimpleNamespace())
    sim_policy = object.__new__(NestedEGMSimPolicy)
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.array([0.5, 0.7, 0.6, 0.65]),
            "investment": jnp.array([0.1, 0.2, 0.3, 0.4]),
        }
    )
    nested_actions = MappingProxyType(
        {
            "consumption": jnp.array([0.4, 0.8, 0.3, 0.9]),
            "investment": jnp.array([0.15, 0.25, 0.35, 0.45]),
        }
    )
    intrinsic_fallback = jnp.array([False, False, False, True])
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (nested_actions, intrinsic_fallback),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonically_refine_nested_inner_action",
        lambda **_kwargs: (
            nested_actions,
            jnp.array([7.0, 4.0, 9.0, 8.0]),
            jnp.array([True, True, False, True]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda **_kwargs: jnp.ones(4, dtype=bool),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_grid_baseline",
        lambda **_kwargs: (
            grid_actions,
            jnp.full(4, 5.0),
            jnp.ones(4, dtype=bool),
        ),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=regime,
        sim_policy=sim_policy,
        states=MappingProxyType({"wealth": jnp.ones(4)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(4)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=jnp.full(4, 5.0),
    )

    aaae(
        np.asarray(actions["consumption"]),
        np.array([0.4, 0.7, 0.6, 0.65]),
        decimal=DECIMAL_PRECISION,
    )
    aaae(
        np.asarray(actions["investment"]),
        np.array([0.15, 0.2, 0.3, 0.4]),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(np.asarray(reported_value), np.array([7, 5, 5, 5]))
    np.testing.assert_array_equal(
        np.asarray(fallback),
        np.array([False, True, True, True]),
    )


def test_nested_policy_replacement_never_emits_a_degraded_interpolated_pair(
    monkeypatch,
) -> None:
    """Canonical scoring rejects every degraded pair in a 135-case family."""
    nodes = jnp.asarray([0.0, 0.5, 1.0])
    interpolant = LocalCubicOuterInterpolant()
    proposed_outer = []
    proposed_consumption = []
    proposed_values = []
    grid_values = []

    for peak in (0.18, 0.31, 0.37, 0.63, 0.82):
        for curvature in (0.5, 1.0, 2.0):
            node_values = 1.0 - curvature * (nodes - peak) ** 2
            search = safeguarded_continuous_argmax(
                objective=lambda x, node_values=node_values: interpolant.evaluate(
                    nodes=nodes,
                    values=node_values,
                    query=x,
                ),
                nodes=nodes,
                node_values=node_values,
                golden_iterations=48,
            )
            outer = float(search.x)
            for beta in (0.2, 0.6, 1.2):
                policies = 0.05 + beta * nodes**2
                consumption = float(
                    simulation_module._interp_across_outer_axis(
                        nodes=nodes,
                        values=policies[:, None],
                        query=jnp.asarray([search.x]),
                    )[0]
                )
                for penalty in (10.0, 50.0, 200.0):

                    def objective(
                        x,
                        action,
                        curvature=curvature,
                        peak=peak,
                        penalty=penalty,
                        beta=beta,
                    ):
                        return (
                            1.0
                            - curvature * (x - peak) ** 2
                            - penalty * (action - (0.05 + beta * x**2)) ** 2
                        )

                    proposed_outer.append(outer)
                    proposed_consumption.append(consumption)
                    proposed_values.append(objective(outer, consumption))
                    grid_values.append(
                        max(
                            objective(float(x), 0.05 + beta * float(x) ** 2)
                            for x in nodes
                        )
                    )

    proposed_values_arr = jnp.asarray(proposed_values)
    grid_values_arr = jnp.asarray(grid_values)
    degraded = proposed_values_arr < grid_values_arr
    assert int(jnp.sum(degraded)) == 74

    nested_actions = MappingProxyType(
        {
            "consumption": jnp.asarray(proposed_consumption),
            "investment": jnp.asarray(proposed_outer),
        }
    )
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.zeros(len(grid_values)),
            "investment": jnp.zeros(len(grid_values)),
        }
    )
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            nested_actions,
            jnp.zeros(len(grid_values), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonically_refine_nested_inner_action",
        lambda **_kwargs: (
            nested_actions,
            proposed_values_arr,
            jnp.ones(len(grid_values), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda **_kwargs: jnp.ones(len(grid_values), dtype=bool),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_grid_baseline",
        lambda **_kwargs: (
            grid_actions,
            grid_values_arr,
            jnp.ones(len(grid_values), dtype=bool),
        ),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=_StubRegime(simulation=SimpleNamespace()),
        sim_policy=object.__new__(NestedEGMSimPolicy),
        states=MappingProxyType({"wealth": jnp.ones(len(grid_values))}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(len(grid_values))}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=grid_values_arr,
    )

    assert fallback is not None
    fallback_arr = jnp.asarray(fallback)
    np.testing.assert_array_equal(np.asarray(fallback_arr), np.asarray(degraded))
    assert bool(jnp.all(reported_value >= grid_values_arr))
    assert bool(jnp.all(~fallback_arr | (jnp.asarray(actions["consumption"]) == 0.0)))
    assert bool(jnp.all(~fallback_arr | (jnp.asarray(actions["investment"]) == 0.0)))


def test_nested_policy_does_not_compare_against_an_out_of_domain_grid_pair(
    monkeypatch,
) -> None:
    """A valid nested pair replaces an inadmissible extrapolated grid winner."""
    regime = _StubRegime(simulation=SimpleNamespace())
    sim_policy = object.__new__(NestedEGMSimPolicy)
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.5]),
            "investment": jnp.asarray([-20.0]),
        }
    )
    nested_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([1.25]),
            "investment": jnp.asarray([-1.15]),
        }
    )
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (nested_actions, jnp.asarray([False])),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonically_refine_nested_inner_action",
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([-10.75]),
            jnp.asarray([True]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda **_kwargs: jnp.asarray([True]),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_grid_baseline",
        lambda **_kwargs: (
            grid_actions,
            jnp.asarray([-7.46]),
            jnp.asarray([False]),
        ),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=regime,
        sim_policy=sim_policy,
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        # The extrapolated grid pair appears better numerically, but it is not
        # a feasible baseline because its outer post-decision is off-domain.
        grid_values=jnp.asarray([-7.46]),
    )

    aaae(np.asarray(actions["consumption"]), [1.25], decimal=DECIMAL_PRECISION)
    aaae(np.asarray(actions["investment"]), [-1.15], decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(np.asarray(reported_value), np.asarray([-10.75]))
    np.testing.assert_array_equal(np.asarray(fallback), np.asarray([False]))


def test_nested_policy_emits_a_safe_baseline_when_the_policy_read_falls_back(
    monkeypatch,
) -> None:
    """A rejected nested read emits its projected and canonically scored baseline."""
    regime = _StubRegime(simulation=SimpleNamespace())
    sim_policy = object.__new__(NestedEGMSimPolicy)
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.5]),
            "investment": jnp.asarray([20.0]),
        }
    )
    proposed_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.6]),
            "investment": jnp.asarray([18.0]),
        }
    )
    safe_baseline = MappingProxyType(
        {
            "consumption": jnp.asarray([0.5]),
            "investment": jnp.asarray([11.0]),
        }
    )
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (proposed_actions, jnp.asarray([True])),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonically_refine_nested_inner_action",
        lambda **_kwargs: (
            proposed_actions,
            jnp.asarray([-8.0]),
            jnp.asarray([True]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda **_kwargs: jnp.asarray([True]),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_grid_baseline",
        lambda **_kwargs: (
            safe_baseline,
            jnp.asarray([-9.0]),
            jnp.asarray([True]),
        ),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=regime,
        sim_policy=sim_policy,
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=jnp.asarray([-7.0]),
    )

    np.testing.assert_array_equal(np.asarray(actions["consumption"]), np.asarray([0.5]))
    np.testing.assert_array_equal(np.asarray(actions["investment"]), np.asarray([11.0]))
    np.testing.assert_array_equal(np.asarray(reported_value), np.asarray([-9.0]))
    np.testing.assert_array_equal(np.asarray(fallback), np.asarray([True]))


def test_nested_inner_action_is_refined_under_canonical_q(monkeypatch) -> None:
    """The inner action maximizes canonical Q at the proposed outer action."""
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "inner_action_name", "consumption")
    object.__setattr__(payload, "golden_iterations", 48)
    regime = _StubRegime(
        simulation=SimpleNamespace(
            grids={
                "consumption": SimpleNamespace(
                    to_jax=lambda: jnp.asarray([0.0, 0.5, 1.0])
                )
            }
        )
    )
    targets = jnp.asarray([0.27, 0.73])
    proposed_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.1, 0.9]),
            "investment": jnp.asarray([0.2, 0.4]),
        }
    )

    def score(*, candidate_actions, canonical_states, **_kwargs):
        consumption = jnp.asarray(candidate_actions["consumption"])
        assert consumption.shape == jnp.asarray(candidate_actions["investment"]).shape
        assert consumption.shape == jnp.asarray(canonical_states["wealth"]).shape
        values = 2.0 - (consumption - targets) ** 2
        return values, jnp.ones_like(values, dtype=bool)

    monkeypatch.setattr(simulation_module, "_canonical_Q_at_actions", score)

    actions, values, feasible = (
        simulation_module._canonically_refine_nested_inner_action(
            payload=payload,
            proposed_actions=proposed_actions,
            regime=regime,
            canonical_states=MappingProxyType({"wealth": jnp.ones(2)}),
            action_names=("consumption", "investment"),
            next_regime_to_V_arr=MappingProxyType({}),
            flat_params=MappingProxyType({}),
            period=0,
            age=jnp.asarray(40.0),
        )
    )

    dtype = np.asarray(actions["consumption"]).dtype
    location_atol = 8 * np.sqrt(np.finfo(dtype).eps)
    np.testing.assert_allclose(
        np.asarray(actions["consumption"]),
        np.asarray(targets),
        atol=location_atol,
    )
    np.testing.assert_array_equal(
        np.asarray(actions["investment"]),
        np.asarray(proposed_actions["investment"]),
    )
    np.testing.assert_allclose(np.asarray(values), np.full(2, 2.0))
    assert bool(jnp.all(feasible))
