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
import pytest

import _lcm.simulation.simulate as simulation_module
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import Regime
from _lcm.simulation.simulate import (
    _interp_rows_with_support,
    _replace_continuous_action_with_policy_read,
)


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

    _, reported_value = _replace_continuous_action_with_policy_read(
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
