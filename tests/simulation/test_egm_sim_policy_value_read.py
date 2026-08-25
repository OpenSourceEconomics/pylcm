"""The branch re-decision value read follows the solve's Hermite convention.

The solve publishes conditional values by reading the refined row with the
cubic Hermite interpolant, using the marginal-utility row as exact node slopes
(envelope theorem). Simulation compares conditional branch values with the
same interpolant, so the ranking the re-decision sees is the ranking the solve
convention implies — a linear read of the same row can rank two close branches
differently.
"""

import itertools
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
from lcm.exceptions import InvalidSimulationInputError
from tests.conftest import DECIMAL_PRECISION


class _StubRegime(Regime):
    """Engine regime carrying only the fields the policy read reaches.

    Subclasses the real class so the runtime-typed perimeter accepts it, and
    writes the one field directly rather than building a whole compiled regime.
    """

    def __init__(self, *, simulation: object) -> None:
        object.__setattr__(self, "simulation", simulation)


@pytest.mark.parametrize("coefficient", [1.0, 1.001])
def test_outer_transition_unit_slope_check_respects_configured_precision(
    coefficient: float,
) -> None:
    """A unit affine law passes while a materially different slope is refused."""

    def outer_post_decision(outer_state, investment):
        return outer_state + coefficient * investment

    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "outer_action_name", "investment")
    object.__setattr__(
        payload,
        "outer_post_decision_name",
        "outer_post_decision",
    )
    regime = _StubRegime(
        simulation=SimpleNamespace(
            functions={"outer_post_decision": outer_post_decision},
        )
    )
    result = simulation_module._outer_transition_offset_and_slope(
        payload=payload,
        regime=regime,
        states=MappingProxyType({"outer_state": jnp.asarray([1.37])}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        n_subjects=1,
    )

    assert result is not None
    _, slope_is_unit, _ = result
    assert bool(slope_is_unit[0]) is (coefficient == 1.0)


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
        in_regime=jnp.ones(1, dtype=bool),
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
        lambda **_kwargs: (
            nested_actions,
            intrinsic_fallback,
            jnp.array([7.0, 4.0, 9.0, 8.0]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
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
        in_regime=jnp.ones(4, dtype=bool),
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


def test_nested_policy_replay_scores_the_published_pair_without_reoptimizing(
    monkeypatch,
) -> None:
    """Simulation scores the published pair and keeps it when it beats the baseline."""
    proposed_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.4]),
            "investment": jnp.asarray([0.5]),
        }
    )
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.2]),
            "investment": jnp.asarray([0.0]),
        }
    )
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "inner_action_name", "consumption")
    object.__setattr__(payload, "golden_iterations", 24)
    object.__setattr__(payload, "value_atol", 0.1)
    object.__setattr__(payload, "value_rtol", 0.0)
    regime = _StubRegime(
        simulation=SimpleNamespace(
            grids={
                "consumption": SimpleNamespace(
                    to_jax=lambda: jnp.asarray([0.0, 0.5, 1.0])
                )
            }
        )
    )

    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            proposed_actions,
            jnp.asarray([False]),
            jnp.asarray([2.0]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonical_Q_at_actions",
        lambda *, candidate_actions, **_kwargs: (
            2.0 - (jnp.asarray(candidate_actions["consumption"]) - 0.7) ** 2,
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
        lambda **_kwargs: (grid_actions, jnp.asarray([1.0]), jnp.asarray([True])),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=regime,
        sim_policy=payload,
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=jnp.asarray([1.0]),
        in_regime=jnp.asarray([True]),
    )

    np.testing.assert_array_equal(
        np.asarray(actions["consumption"]),
        np.asarray(proposed_actions["consumption"]),
    )
    aaae(
        np.asarray(reported_value),
        np.asarray([1.91]),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(np.asarray(fallback), np.asarray([False]))


def test_nested_policy_rejects_an_inner_action_below_its_surrogate_value(
    monkeypatch,
) -> None:
    """A material conditional policy loss is controlled by the fallback gate."""
    nested_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([1.856252064]),
            "investment": jnp.asarray([4.592250281]),
        }
    )
    baseline_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([1.5]),
            "investment": jnp.asarray([4.0]),
        }
    )
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "value_atol", 1e-4)
    object.__setattr__(payload, "value_rtol", 1e-4)
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([False]),
            jnp.asarray([10.0]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([10.0 - 0.007090397]),
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
            baseline_actions,
            jnp.asarray([9.0]),
            jnp.asarray([True]),
        ),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=baseline_actions,
        regime=_StubRegime(simulation=SimpleNamespace()),
        sim_policy=payload,
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=jnp.asarray([9.0]),
        in_regime=jnp.asarray([True]),
    )

    np.testing.assert_array_equal(
        actions["consumption"], baseline_actions["consumption"]
    )
    np.testing.assert_array_equal(reported_value, jnp.asarray([9.0]))
    np.testing.assert_array_equal(fallback, jnp.asarray([True]))


def test_nested_policy_acceptance_truth_table(monkeypatch) -> None:
    """Every accepted pair is safe.

    Otherwise a safe baseline is used or simulation stops.
    """
    cell = {}
    nested_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.7]),
            "investment": jnp.asarray([0.8]),
        }
    )
    baseline_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.2]),
            "investment": jnp.asarray([0.3]),
        }
    )
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.1]),
            "investment": jnp.asarray([0.1]),
        }
    )

    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([cell["nested_fallback"]]),
            jnp.asarray([cell["nested_value"]]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([cell["nested_value"]]),
            jnp.asarray([cell["nested_feasible"]]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda **_kwargs: jnp.asarray([cell["nested_admissible"]]),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_grid_baseline",
        lambda **_kwargs: (
            baseline_actions,
            jnp.asarray([1.0]),
            jnp.asarray([cell["baseline_admissible"]]),
        ),
    )

    common = {
        "optimal_actions": grid_actions,
        "regime": _StubRegime(simulation=SimpleNamespace()),
        "sim_policy": object.__new__(NestedEGMSimPolicy),
        "states": MappingProxyType({"wealth": jnp.ones(1)}),
        "flat_params": MappingProxyType({}),
        "period": 0,
        "age": jnp.asarray(40.0),
        "canonical_states": MappingProxyType({"wealth": jnp.ones(1)}),
        "action_names": ("consumption", "investment"),
        "next_regime_to_V_arr": MappingProxyType({}),
        "grid_values": jnp.asarray([1.0]),
        "in_regime": jnp.ones(1, dtype=bool),
    }

    for (
        nested_fallback,
        nested_feasible,
        nested_admissible,
        finite,
        baseline_admissible,
        nested_ge_baseline,
    ) in itertools.product((False, True), repeat=6):
        nested_value = (2.0 if nested_ge_baseline else 0.5) if finite else jnp.nan
        cell.update(
            nested_fallback=nested_fallback,
            nested_feasible=nested_feasible,
            nested_admissible=nested_admissible,
            nested_value=nested_value,
            baseline_admissible=baseline_admissible,
        )
        should_accept = (
            not nested_fallback
            and nested_feasible
            and nested_admissible
            and finite
            and (not baseline_admissible or nested_ge_baseline)
        )
        if not should_accept and not baseline_admissible:
            with pytest.raises(InvalidSimulationInputError):
                _replace_continuous_action_with_policy_read(**common)
            continue

        actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
            **common
        )
        assert fallback is not None
        assert bool(fallback[0]) == (not should_accept)
        if should_accept:
            np.testing.assert_array_equal(
                np.asarray(actions["consumption"]),
                np.asarray(nested_actions["consumption"]),
            )
            assert float(reported_value[0]) == float(nested_value)
        else:
            np.testing.assert_array_equal(
                np.asarray(actions["consumption"]),
                np.asarray(baseline_actions["consumption"]),
            )
            assert float(reported_value[0]) == 1.0


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
            proposed_values_arr,
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
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
        in_regime=jnp.ones(len(grid_values), dtype=bool),
    )

    assert fallback is not None
    fallback_arr = jnp.asarray(fallback)
    np.testing.assert_array_equal(np.asarray(fallback_arr), np.asarray(degraded))
    assert bool(jnp.all(reported_value >= grid_values_arr))
    assert bool(jnp.all(~fallback_arr | (jnp.asarray(actions["consumption"]) == 0.0)))
    assert bool(jnp.all(~fallback_arr | (jnp.asarray(actions["investment"]) == 0.0)))


def test_nested_policy_accepts_improvement_over_a_canonically_unsafe_grid_pair(
    monkeypatch,
) -> None:
    """A valid nested pair replaces a grid decision canonical Q rejects."""
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
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([False]),
            jnp.asarray([-10.75]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
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
        # The grid pair is canonically unsafe, so it cannot be the baseline.
        grid_values=jnp.asarray([-7.46]),
        in_regime=jnp.ones(1, dtype=bool),
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
        lambda **_kwargs: (
            proposed_actions,
            jnp.asarray([True]),
            jnp.asarray([-8.0]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
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
        in_regime=jnp.ones(1, dtype=bool),
    )

    np.testing.assert_array_equal(np.asarray(actions["consumption"]), np.asarray([0.5]))
    np.testing.assert_array_equal(np.asarray(actions["investment"]), np.asarray([11.0]))
    np.testing.assert_array_equal(np.asarray(reported_value), np.asarray([-9.0]))
    np.testing.assert_array_equal(np.asarray(fallback), np.asarray([True]))


@pytest.mark.parametrize(
    "raw_outer",
    np.concatenate((np.linspace(-5.0, -0.1, 25), np.linspace(1.1, 6.0, 25))),
)
def test_nested_policy_rejection_projects_an_out_of_domain_grid_pair(
    monkeypatch, raw_outer
) -> None:
    """Canonical Q alone cannot admit a pair outside the solver's domain."""
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "outer_action_name", "investment")
    object.__setattr__(payload, "inner_action_name", "consumption")
    object.__setattr__(payload, "savings_lower_bound", 0.0)
    object.__setattr__(
        payload,
        "adjuster",
        SimpleNamespace(outer_nodes=jnp.asarray([0.0, 1.0])),
    )
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.25]),
            "investment": jnp.asarray([raw_outer]),
        }
    )

    def canonical_q(*, candidate_actions, **_kwargs):
        investment = jnp.asarray(candidate_actions["investment"])
        consumption = jnp.asarray(candidate_actions["consumption"])
        resources = jnp.abs(investment) + 2.0
        feasible = (consumption > 0.0) & (consumption <= resources)
        value = jnp.where(feasible, 10.0 - (investment - raw_outer) ** 2, -jnp.inf)
        return value, feasible

    monkeypatch.setattr(
        simulation_module,
        "_outer_transition_offset_and_slope",
        lambda **_kwargs: (
            jnp.zeros(1),
            jnp.ones(1, dtype=bool),
            jnp.asarray,
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_kwargs: jnp.abs(jnp.asarray(outer_action)) + 2.0,
    )
    monkeypatch.setattr(simulation_module, "_canonical_Q_at_actions", canonical_q)
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            grid_actions,
            jnp.asarray([True]),
            jnp.asarray([-jnp.inf]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
        lambda **_kwargs: (
            grid_actions,
            jnp.asarray([-jnp.inf]),
            jnp.asarray([False]),
        ),
    )

    grid_value, grid_feasible = canonical_q(candidate_actions=grid_actions)
    assert bool(grid_feasible[0])
    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=_StubRegime(simulation=SimpleNamespace()),
        sim_policy=payload,
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=grid_value,
        in_regime=jnp.ones(1, dtype=bool),
    )

    expected_actions = MappingProxyType(
        {
            "consumption": grid_actions["consumption"],
            "investment": jnp.clip(jnp.asarray([raw_outer]), 0.0, 1.0),
        }
    )
    expected_value, expected_feasible = canonical_q(candidate_actions=expected_actions)
    assert bool(expected_feasible[0])
    aaae(
        np.asarray(actions["consumption"]),
        np.asarray(expected_actions["consumption"]),
        decimal=DECIMAL_PRECISION,
    )
    aaae(
        np.asarray(actions["investment"]),
        np.asarray(expected_actions["investment"]),
        decimal=DECIMAL_PRECISION,
    )
    aaae(
        np.asarray(reported_value),
        np.asarray(expected_value),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(np.asarray(fallback), np.asarray([True]))


@pytest.mark.parametrize("raw_outer", np.linspace(-0.75, -5.0, 25))
def test_nested_policy_rejection_uses_the_safe_opposite_endpoint(
    monkeypatch,
    raw_outer,
) -> None:
    """A rejected replay uses the safe endpoint when the nearest one is infeasible."""
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "outer_action_name", "investment")
    object.__setattr__(payload, "inner_action_name", "consumption")
    object.__setattr__(payload, "savings_lower_bound", 0.0)
    object.__setattr__(
        payload,
        "adjuster",
        SimpleNamespace(outer_nodes=jnp.asarray([0.0, 1.0])),
    )
    raw_resources = raw_outer**2 - 0.25
    raw_inner = raw_resources + 0.1
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([raw_inner]),
            "investment": jnp.asarray([raw_outer]),
        }
    )

    def canonical_q(*, candidate_actions, **_kwargs):
        investment = jnp.asarray(candidate_actions["investment"])
        consumption = jnp.asarray(candidate_actions["consumption"])
        resources = investment**2 - 0.25
        feasible = (consumption > 0.0) & (consumption <= resources)
        value = jnp.where(
            feasible,
            1.0 - (investment + 1.0) ** 2 - (consumption - 0.25) ** 2,
            -jnp.inf,
        )
        return value, feasible

    monkeypatch.setattr(
        simulation_module,
        "_outer_transition_offset_and_slope",
        lambda **_kwargs: (
            jnp.zeros(1),
            jnp.ones(1, dtype=bool),
            jnp.asarray,
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_kwargs: jnp.asarray(outer_action) ** 2 - 0.25,
    )
    monkeypatch.setattr(simulation_module, "_canonical_Q_at_actions", canonical_q)
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            grid_actions,
            jnp.asarray([True]),
            jnp.asarray([-jnp.inf]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
        lambda **_kwargs: (
            grid_actions,
            jnp.asarray([-jnp.inf]),
            jnp.asarray([False]),
        ),
    )

    grid_value, grid_feasible = canonical_q(candidate_actions=grid_actions)
    assert not bool(grid_feasible[0])
    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=_StubRegime(simulation=SimpleNamespace()),
        sim_policy=payload,
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=grid_value,
        in_regime=jnp.ones(1, dtype=bool),
    )

    expected_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([min(raw_inner, 0.75)]),
            "investment": jnp.asarray([1.0]),
        }
    )
    expected_value, expected_feasible = canonical_q(candidate_actions=expected_actions)
    assert bool(expected_feasible[0])
    aaae(
        np.asarray(actions["consumption"]),
        np.asarray(expected_actions["consumption"]),
        decimal=DECIMAL_PRECISION,
    )
    aaae(
        np.asarray(actions["investment"]),
        np.asarray(expected_actions["investment"]),
        decimal=DECIMAL_PRECISION,
    )
    aaae(
        np.asarray(reported_value),
        np.asarray(expected_value),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(np.asarray(fallback), np.asarray([True]))
    _, emitted_feasible = canonical_q(candidate_actions=actions)
    assert bool(emitted_feasible[0])


def test_nested_grid_baseline_chooses_the_best_safe_endpoint(monkeypatch) -> None:
    """An out-of-domain grid pair uses the higher-valued feasible endpoint."""
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "outer_action_name", "investment")
    object.__setattr__(payload, "inner_action_name", "consumption")
    object.__setattr__(payload, "savings_lower_bound", 0.0)
    object.__setattr__(
        payload,
        "adjuster",
        SimpleNamespace(outer_nodes=jnp.asarray([0.0, 1.0])),
    )
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([1.5]),
            "investment": jnp.asarray([-1.0]),
        }
    )

    monkeypatch.setattr(
        simulation_module,
        "_outer_transition_offset_and_slope",
        lambda **_kwargs: (
            jnp.zeros(1),
            jnp.ones(1, dtype=bool),
            jnp.asarray,
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_kwargs: jnp.asarray(outer_action) + 2.0,
    )

    def canonical_q(*, candidate_actions, **_kwargs):
        investment = jnp.asarray(candidate_actions["investment"])
        consumption = jnp.asarray(candidate_actions["consumption"])
        resources = investment + 2.0
        feasible = (consumption > 0.0) & (consumption <= resources)
        return jnp.where(feasible, resources, -jnp.inf), feasible

    monkeypatch.setattr(simulation_module, "_canonical_Q_at_actions", canonical_q)

    actions, value, admissible = simulation_module._nested_grid_baseline(
        payload=payload,
        grid_actions=grid_actions,
        regime=_StubRegime(simulation=SimpleNamespace()),
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
    )

    np.testing.assert_array_equal(
        np.asarray(actions["consumption"]),
        np.asarray([1.5]),
    )
    np.testing.assert_array_equal(
        np.asarray(actions["investment"]),
        np.asarray([1.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(value),
        np.asarray([3.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(admissible),
        np.asarray([True]),
    )


def test_nested_grid_baseline_enforces_the_solver_owned_budget(monkeypatch) -> None:
    """A grid pair above resources is unsafe even when canonical Q admits it."""
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "outer_action_name", "investment")
    object.__setattr__(payload, "inner_action_name", "consumption")
    object.__setattr__(payload, "savings_lower_bound", 0.0)
    object.__setattr__(
        payload,
        "adjuster",
        SimpleNamespace(outer_nodes=jnp.asarray([0.0, 1.0])),
    )
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([3.0]),
            "investment": jnp.asarray([0.5]),
        }
    )

    monkeypatch.setattr(
        simulation_module,
        "_outer_transition_offset_and_slope",
        lambda **_kwargs: (
            jnp.zeros(1),
            jnp.ones(1, dtype=bool),
            jnp.asarray,
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_kwargs: jnp.asarray(outer_action) + 2.0,
    )

    def canonical_q(*, candidate_actions, **_kwargs):
        investment = jnp.asarray(candidate_actions["investment"])
        return investment, jnp.ones_like(investment, dtype=bool)

    monkeypatch.setattr(simulation_module, "_canonical_Q_at_actions", canonical_q)

    actions, value, admissible = simulation_module._nested_grid_baseline(
        payload=payload,
        grid_actions=grid_actions,
        regime=_StubRegime(simulation=SimpleNamespace()),
        states=MappingProxyType({"wealth": jnp.ones(1)}),
        canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
    )

    np.testing.assert_array_equal(np.asarray(actions["consumption"]), [3.0])
    np.testing.assert_array_equal(np.asarray(actions["investment"]), [1.0])
    np.testing.assert_array_equal(np.asarray(value), [1.0])
    np.testing.assert_array_equal(np.asarray(admissible), [True])


def test_nested_policy_rejection_fails_when_no_candidate_is_safe(
    monkeypatch,
) -> None:
    """Simulation stops rather than publishing an infeasible nested action pair."""
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.25]),
            "investment": jnp.asarray([-1.0]),
        }
    )
    nested_actions = MappingProxyType(
        {
            "consumption": jnp.asarray([0.5]),
            "investment": jnp.asarray([0.5]),
        }
    )
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([True]),
            jnp.asarray([-jnp.inf]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
        lambda **_kwargs: (
            nested_actions,
            jnp.asarray([-jnp.inf]),
            jnp.asarray([False]),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda **_kwargs: jnp.asarray([False]),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_grid_baseline",
        lambda **_kwargs: (
            grid_actions,
            jnp.asarray([-jnp.inf]),
            jnp.asarray([False]),
        ),
    )

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"neither.*nested.*nor.*grid",
    ):
        _replace_continuous_action_with_policy_read(
            optimal_actions=grid_actions,
            regime=_StubRegime(simulation=SimpleNamespace()),
            sim_policy=object.__new__(NestedEGMSimPolicy),
            states=MappingProxyType({"wealth": jnp.ones(1)}),
            flat_params=MappingProxyType({}),
            period=0,
            age=jnp.asarray(40.0),
            canonical_states=MappingProxyType({"wealth": jnp.ones(1)}),
            action_names=("consumption", "investment"),
            next_regime_to_V_arr=MappingProxyType({}),
            grid_values=jnp.asarray([-jnp.inf]),
            in_regime=jnp.ones(1, dtype=bool),
        )


@pytest.mark.parametrize("n_placeholders", range(1, 34))
def test_nested_policy_failure_ignores_out_of_regime_placeholders(
    monkeypatch,
    n_placeholders,
) -> None:
    """Only subjects assigned to the regime can make nested replay fail loud."""
    n_subjects = n_placeholders + 1
    in_regime = jnp.arange(n_subjects) == 0
    grid_actions = MappingProxyType(
        {
            "consumption": jnp.where(in_regime, 0.25, 0.0),
            "investment": jnp.zeros(n_subjects),
        }
    )
    nested_actions = MappingProxyType(
        {
            "consumption": jnp.where(in_regime, 0.25, 0.0),
            "investment": jnp.full(n_subjects, 0.5),
        }
    )
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (
            nested_actions,
            ~in_regime,
            jnp.where(in_regime, 2.0, -jnp.inf),
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_score_nested_action_pair",
        lambda **_kwargs: (
            nested_actions,
            jnp.where(in_regime, 2.0, -jnp.inf),
            in_regime,
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda **_kwargs: in_regime,
    )
    monkeypatch.setattr(
        simulation_module,
        "_nested_grid_baseline",
        lambda **_kwargs: (
            grid_actions,
            jnp.where(in_regime, 1.0, -jnp.inf),
            in_regime,
        ),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        regime=_StubRegime(simulation=SimpleNamespace()),
        sim_policy=object.__new__(NestedEGMSimPolicy),
        states=MappingProxyType({"wealth": jnp.ones(n_subjects)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        canonical_states=MappingProxyType({"wealth": jnp.ones(n_subjects)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        grid_values=jnp.where(in_regime, 1.0, -jnp.inf),
        in_regime=in_regime,
    )

    assert fallback is not None
    np.testing.assert_array_equal(
        np.asarray(actions["investment"])[0],
        np.asarray(nested_actions["investment"])[0],
    )
    np.testing.assert_array_equal(
        np.asarray(reported_value)[0],
        np.asarray([2.0])[0],
    )
    np.testing.assert_array_equal(
        np.asarray(fallback)[0],
        np.asarray([False])[0],
    )


def test_nested_action_pair_is_scored_without_changing_actions(monkeypatch) -> None:
    """Canonical scoring leaves the published nested action pair unchanged."""
    regime = _StubRegime(simulation=SimpleNamespace())
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

    actions, values, feasible = simulation_module._score_nested_action_pair(
        proposed_actions=proposed_actions,
        regime=regime,
        canonical_states=MappingProxyType({"wealth": jnp.ones(2)}),
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
    )

    np.testing.assert_array_equal(
        np.asarray(actions["consumption"]),
        np.asarray(proposed_actions["consumption"]),
    )
    np.testing.assert_array_equal(
        np.asarray(actions["investment"]),
        np.asarray(proposed_actions["investment"]),
    )
    expected_values = 2.0 - (np.asarray([0.1, 0.9]) - np.asarray(targets)) ** 2
    aaae(
        np.asarray(values),
        expected_values,
        decimal=DECIMAL_PRECISION,
    )
    assert bool(jnp.all(feasible))
