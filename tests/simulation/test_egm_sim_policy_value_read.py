"""The branch re-decision value read follows the solve's Hermite convention.

The solve publishes conditional values by reading the refined row with the
cubic Hermite interpolant, using the marginal-utility row as exact node slopes
(envelope theorem). Simulation compares conditional branch values with the
same interpolant, so the ranking the re-decision sees is the ranking the solve
convention implies — a linear read of the same row can rank two close branches
differently.
"""

import itertools
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import _lcm.simulation.simulate as simulation_module
from _lcm.egm import outer_affine_structure, outer_inversion
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy, OuterPolicyBank
from _lcm.egm.outer_inversion import DeclaredOuterInverse
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import EGMPolicyRead, Regime
from _lcm.simulation.simulate import (
    _interp_rows_with_support,
    _read_egm_policy_first,
    _replace_continuous_action_with_policy_read,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.typing import ActionName, RegimeName
from tests.conftest import DECIMAL_PRECISION


class _StubRegime(Regime):
    """Engine regime carrying only the fields the policy read reaches.

    Subclasses the real class so the runtime-typed perimeter accepts it, and
    writes the one field directly rather than building a whole compiled regime.
    """

    def __init__(self, *, simulation: SimpleNamespace) -> None:
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


def test_policy_first_reports_the_read_value_of_the_emitted_action(monkeypatch):
    """The policy-first decision reports the read value of the action it emits.

    On nodes `(1, 0)`, `(2, log 2)` with slopes `(1, 1/2)` the subject at
    resources `1.5` consumes the linear policy read `0.75` and reports the
    cubic Hermite value read `0.40907`; the subject at `3.0` is off the row's
    support and is rejected for the grid argmax.
    """
    read = EGMPolicyRead(
        action_name="consumption",
        resources_target="resources",
        savings_lower_bound=0.0,
        row_discrete_state_names=(),
        row_passive_state_names=(),
        row_discrete_action_names=(),
        row_axis_lengths_by_period=MappingProxyType({0: ()}),
        float_dtype="float64",
    )
    regime = _StubRegime(
        simulation=SimpleNamespace(
            egm_policy_read=read, grids={}, Q_and_F={0: lambda **_kwargs: None}
        )
    )
    sim_policy = EGMSimPolicy(
        endog_grid=jnp.array([1.0, 2.0]),
        policy=jnp.array([0.5, 1.0]),
        value=jnp.array([0.0, jnp.log(2.0)]),
        marginal_utility=jnp.array([1.0, 0.5]),
    )
    resources = jnp.array([1.5, 3.0])
    monkeypatch.setattr(
        simulation_module,
        "_compute_targets",
        lambda **_kwargs: {"resources": resources},
    )
    monkeypatch.setattr(
        simulation_module,
        "_evaluate_canonical_Q_at_actions",
        lambda **_kwargs: (jnp.full(2, jnp.nan), jnp.ones(2, dtype=bool)),
    )

    actions, value, accepted = _read_egm_policy_first(
        regime=regime,
        sim_policy=sim_policy,
        states=MappingProxyType({"wealth": resources}),
        canonical_states=MappingProxyType({"wealth": resources}),
        action_names=("consumption",),
        action_dtypes={"consumption": resources.dtype},
        next_regime_to_V_arr=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
        in_regime=jnp.ones(2, dtype=bool),
    )

    assert float(actions["consumption"][0]) == pytest.approx(0.75, abs=1e-9)
    assert float(value[0]) == pytest.approx(0.4090736, abs=1e-6)
    assert np.asarray(accepted).tolist() == [True, False]


class _NestedDispatchInputs(TypedDict):
    regime: Regime
    sim_policy: NestedEGMSimPolicy
    states: MappingProxyType[str, jax.Array]
    flat_params: MappingProxyType[str, jax.Array]
    period: int
    age: jax.Array
    canonical_states: MappingProxyType[str, jax.Array]
    action_names: tuple[ActionName, ...]
    next_regime_to_V_arr: MappingProxyType[RegimeName, jax.Array]


def _nested_dispatch_inputs(*, n_subjects: int) -> _NestedDispatchInputs:
    """Arguments of the nested policy-read dispatch, with stubbed reads."""
    return {
        "regime": _StubRegime(simulation=SimpleNamespace(post_decision_bounds=())),
        "sim_policy": object.__new__(NestedEGMSimPolicy),
        "states": MappingProxyType({"wealth": jnp.ones(n_subjects)}),
        "flat_params": MappingProxyType({}),
        "period": 0,
        "age": jnp.asarray(40.0),
        "canonical_states": MappingProxyType({"wealth": jnp.ones(n_subjects)}),
        "action_names": ("consumption", "investment"),
        "next_regime_to_V_arr": MappingProxyType({}),
    }


def test_nested_policy_replay_emits_the_read_winner_over_a_better_grid_pair(
    monkeypatch,
) -> None:
    """The read's pair is emitted with its canonical value, never compared to grid.

    The grid pair reports the higher value here; the nested read still owns the
    decision, and the reported value is canonical Q at the emitted pair.
    """
    proposed_actions = MappingProxyType(
        {"consumption": jnp.asarray([0.4]), "investment": jnp.asarray([0.5])}
    )
    grid_actions = MappingProxyType(
        {"consumption": jnp.asarray([0.2]), "investment": jnp.asarray([0.0])}
    )
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (proposed_actions, jnp.asarray([False])),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonical_Q_at_actions",
        lambda *, candidate_actions, **_kwargs: (
            2.0 - (jnp.asarray(candidate_actions["consumption"]) - 0.7) ** 2,
            jnp.asarray([True]),
        ),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        grid_values=jnp.asarray([5.0]),
        **_nested_dispatch_inputs(n_subjects=1),
    )

    for name in ("consumption", "investment"):
        np.testing.assert_array_equal(
            np.asarray(actions[name]), np.asarray(proposed_actions[name])
        )
    aaae(np.asarray(reported_value), np.asarray([1.91]), decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(np.asarray(fallback), np.asarray([False]))


@pytest.mark.parametrize(
    ("read_fallback", "feasible", "finite"),
    list(itertools.product((False, True), repeat=3)),
)
def test_nested_policy_acceptance_truth_table(
    *, monkeypatch, read_fallback, feasible, finite
) -> None:
    """A supported, feasible, finitely scored read is emitted; else the grid pair."""
    nested_actions = MappingProxyType(
        {"consumption": jnp.asarray([0.7]), "investment": jnp.asarray([0.8])}
    )
    grid_actions = MappingProxyType(
        {"consumption": jnp.asarray([0.1]), "investment": jnp.asarray([0.1])}
    )
    nested_value = 2.0 if finite else jnp.nan
    monkeypatch.setattr(
        simulation_module,
        "_read_nested_policy",
        lambda **_kwargs: (nested_actions, jnp.asarray([read_fallback])),
    )
    monkeypatch.setattr(
        simulation_module,
        "_canonical_Q_at_actions",
        lambda **_kwargs: (jnp.asarray([nested_value]), jnp.asarray([feasible])),
    )

    actions, reported_value, fallback = _replace_continuous_action_with_policy_read(
        optimal_actions=grid_actions,
        grid_values=jnp.asarray([1.0]),
        **_nested_dispatch_inputs(n_subjects=1),
    )

    accept = not read_fallback and feasible and finite
    expected = nested_actions if accept else grid_actions
    for name in ("consumption", "investment"):
        np.testing.assert_array_equal(
            np.asarray(actions[name]), np.asarray(expected[name])
        )
    assert float(reported_value[0]) == (2.0 if accept else 1.0)
    assert fallback is not None
    assert bool(fallback[0]) == (not accept)


_LIQUID_GRID = jnp.array([1.0, 2.0, 3.0])


def _keeper_rows() -> EGMSimPolicy:
    """One keeper branch on the shared liquid grid, with no row axes."""
    return EGMSimPolicy(
        endog_grid=_LIQUID_GRID,
        policy=jnp.array([0.5, 1.0, 1.5]),
        value=jnp.array([0.0, 0.7, 1.1]),
        marginal_utility=jnp.array([1.0, 0.5, 0.3]),
    )


def _adjuster_bank() -> OuterPolicyBank:
    """Two conditional adjuster branches on a two-node outer mesh."""
    stacked = jnp.stack([_LIQUID_GRID, _LIQUID_GRID])
    return OuterPolicyBank(
        outer_nodes=jnp.array([0.0, 1.0]),
        policies=EGMSimPolicy(
            endog_grid=stacked,
            policy=jnp.stack([jnp.array([0.4, 0.9, 1.4])] * 2),
            value=jnp.stack([jnp.array([0.0, 0.6, 1.0]), jnp.array([0.1, 0.8, 1.2])]),
            marginal_utility=jnp.stack([jnp.array([1.0, 0.5, 0.3])] * 2),
        ),
    )


def _replay_capability() -> OuterReplayCapability:
    """The settled verdict a published continuous-outer payload carries."""
    return OuterReplayCapability(
        inverse=DeclaredOuterInverse(coefficient=Fraction(1), low=0.0, high=20.0),
        undeclared_functions=(),
        unbindable_functions=(),
        unavailable_keeper_states=(),
        unaddressable_passive_states=(),
        unaddressable_discrete_actions=(),
    )


def _nested_payload(*, no_adjustment_name: str | None = None) -> NestedEGMSimPolicy:
    """A replayable continuous-outer payload over `_keeper_rows`/`_adjuster_bank`."""
    return NestedEGMSimPolicy(
        keeper=_keeper_rows(),
        adjuster=_adjuster_bank(),
        outer_action_name="investment",
        outer_state_name="illiquid",
        outer_post_decision_name="new_illiquid",
        inner_action_name="consumption",
        liquid_state_name="wealth",
        outer_no_adjustment_name=no_adjustment_name,
        resources_target_name="resources",
        savings_lower_bound=0.0,
        golden_iterations=4,
        replay_capability=_replay_capability(),
    )


def _new_illiquid(*, illiquid, investment):
    """`s' = Z + Iz`: the outer post-decision the replay inverts."""
    return illiquid + investment


def _resources(wealth):
    """Liquid resources the recovered pair is checked against."""
    return wealth + 5.0


def _read_with(*, payload, functions):
    """Replay `payload` for one subject against the given simulate function pool."""
    return simulation_module._read_nested_policy(
        payload=payload,
        optimal_actions=MappingProxyType(
            {"consumption": jnp.array([1.0]), "investment": jnp.array([0.0])}
        ),
        regime=_StubRegime(
            simulation=SimpleNamespace(
                grids={},
                functions=functions,
                constraints={},
                compute_regime_transition_probs=None,
                age_specialized_function_names=frozenset(),
                post_decision_bounds=(),
            )
        ),
        states=MappingProxyType(
            {"wealth": jnp.array([2.0]), "illiquid": jnp.array([1.0])}
        ),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
    )


def test_nested_read_replays_the_decision_when_every_function_resolves():
    """A payload whose declared functions all resolve replays without falling back."""
    _, fallback = _read_with(
        payload=_nested_payload(),
        functions={"new_illiquid": _new_illiquid, "resources": _resources},
    )

    assert not bool(fallback[0])


def test_nested_read_does_not_certify_the_declared_outer_map_itself():
    """The read consumes the published inversion verdict instead of re-deriving it.

    The solve settles the declared map's coefficient once, before publishing,
    and the payload carries that answer. A read that certified the map again
    could reach a verdict the solve it is replaying did not, and would then
    publish a decision the solve never ranked. Making every certifier fatal for
    the duration of the read is what establishes that none is reached.
    """

    def refuse(**_kwargs):
        raise AssertionError("the read re-certified the declared outer map")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(outer_affine_structure, "certify_outer_coefficient", refuse)
        patch.setattr(outer_inversion, "certify_declared_outer_inverse", refuse)
        _, fallback = _read_with(
            payload=_nested_payload(),
            functions={"new_illiquid": _new_illiquid, "resources": _resources},
        )

    assert not bool(fallback[0])


def test_nested_read_refuses_a_payload_whose_capability_is_unsupported():
    """A payload the solve would not have published stops the read, loudly.

    Structural support is settled before publication, so the read never has to
    decide it. A payload arriving with an unsupported verdict did not come from
    a publication that honoured that gate, and replaying it would emit the
    action-grid winner under the name of the refined method.
    """
    doubled = replace(
        _replay_capability(),
        inverse=DeclaredOuterInverse(coefficient=Fraction(2), low=0.0, high=20.0),
    )

    with pytest.raises(InvalidSimulationInputError, match="replay capability"):
        _read_with(
            payload=replace(_nested_payload(), replay_capability=doubled),
            functions={"new_illiquid": _new_illiquid, "resources": _resources},
        )
