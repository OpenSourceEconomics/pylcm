"""Canonical complete reselection for adaptive N-NB-EGM fallback."""

from types import MappingProxyType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.simulate as simulation_module
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.engine import Regime


class _StubRegime(Regime):
    """Engine regime carrying only the fields the fallback selector reaches."""

    def __init__(self, *, simulation: object) -> None:
        object.__setattr__(self, "simulation", simulation)


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def enable_x64(request: pytest.FixtureRequest):
    """Run each contract witness at both supported precisions.

    The suite selects one precision globally via `--precision`, so this fixture
    restores whatever was configured on the way out. Without that, a module
    that flips x64 leaks the last value it set into every later file on the
    same xdist worker.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    yield request.param
    jax.config.update("jax_enable_x64", previous)


def _payload(*, nodes=(1.0, 2.0), low=0.0, high=3.0) -> NestedEGMSimPolicy:
    """Build the minimal nested payload the selector and the baseline read.

    The runtime-typed perimeter checks the declared class, so this writes the
    fields it reaches onto a real instance rather than standing in a namespace.
    """
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(
        payload, "adjuster", SimpleNamespace(outer_nodes=jnp.asarray(nodes))
    )
    object.__setattr__(
        payload,
        "replay_capability",
        SimpleNamespace(inverse=SimpleNamespace(low=low, high=high)),
    )
    object.__setattr__(payload, "savings_lower_bound", 0.0)
    object.__setattr__(payload, "outer_action_name", "outer")
    object.__setattr__(payload, "inner_action_name", "inner")
    return payload


def _replay_bank(
    *,
    payload,
    values,
    feasible_keeper=False,
    keeper_value=-jnp.inf,
    keeper_target=0.5,
    offset=0.0,
    inner_actions=None,
):
    values = jnp.asarray(values)
    _n_nodes, n_subjects = values.shape
    if inner_actions is None:
        inner_actions = jnp.ones_like(values)
    return simulation_module._best_admissible_replay_candidate(
        payload=payload,
        candidate_values=values,
        candidate_support=jnp.ones_like(values, dtype=bool),
        candidate_actions=jnp.asarray(inner_actions),
        keeper_value=jnp.full((n_subjects,), keeper_value),
        keeper_support=jnp.full((n_subjects,), feasible_keeper),
        keeper_action=jnp.full((n_subjects,), 1.0),
        keeper_post_decision=jnp.full((n_subjects,), keeper_target),
        offset=jnp.full((n_subjects,), offset),
        transition_at=lambda action: action + jnp.asarray(offset),
        regime=_StubRegime(simulation=SimpleNamespace()),
        states=MappingProxyType({"state": jnp.zeros(n_subjects)}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(0.0),
        n_subjects=n_subjects,
    )


def _baseline(*, monkeypatch, payload, replay, canonical_q, state=0.0):
    monkeypatch.setattr(simulation_module, "_canonical_Q_at_actions", canonical_q)
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda *, actions, **_: jnp.asarray(actions["outer"]) != 99.0,
    )
    return simulation_module._nested_grid_baseline(
        payload=payload,
        grid_actions=MappingProxyType(
            {"outer": jnp.asarray([99.0]), "inner": jnp.asarray([1.0])}
        ),
        regime=_StubRegime(simulation=SimpleNamespace()),
        states={"state": jnp.asarray([state])},
        canonical_states={"state": jnp.asarray([state])},
        action_names=("outer", "inner"),
        next_regime_to_V_arr=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(0.0),
        replay_candidate=replay,
    )


def test_lower_solve_value_canonical_feasible_branch_survives(
    monkeypatch, enable_x64
) -> None:
    """Feasibility is applied branchwise before canonical-Q ranking."""
    del enable_x64
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_: jnp.full_like(outer_action, 10.0),
    )
    payload = _payload()
    replay = _replay_bank(payload=payload, values=[[10.0], [5.0]])

    def canonical_q(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        feasible = outer == 2.0
        return jnp.where(feasible, 5.0, 10.0), feasible

    actions, value, admissible = _baseline(
        monkeypatch=monkeypatch,
        payload=payload,
        replay=replay,
        canonical_q=canonical_q,
    )
    assert bool(admissible[0])
    assert float(actions["outer"][0]) == 2.0
    assert float(value[0]) == 5.0


def test_canonical_q_not_solve_value_selects_the_branch(
    monkeypatch, enable_x64
) -> None:
    """A lower published value wins when its canonical Q is higher."""
    del enable_x64
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_: jnp.full_like(outer_action, 10.0),
    )
    payload = _payload()
    replay = _replay_bank(payload=payload, values=[[10.0], [5.0]])

    def canonical_q(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        return outer, jnp.ones_like(outer, dtype=bool)

    actions, value, admissible = _baseline(
        monkeypatch=monkeypatch,
        payload=payload,
        replay=replay,
        canonical_q=canonical_q,
    )
    assert bool(admissible[0])
    assert float(actions["outer"][0]) == 2.0
    assert float(value[0]) == 2.0


def test_keeper_then_ascending_node_tie_order(monkeypatch, enable_x64) -> None:
    """First-maximum order is keeper, then ascending published node index."""
    del enable_x64
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_: jnp.full_like(outer_action, 10.0),
    )
    payload = _payload(nodes=(0.0, 2.0), low=0.0, high=2.0)

    def canonical_q(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        return jnp.full_like(outer, 5.0), jnp.ones_like(outer, dtype=bool)

    keeper_replay = _replay_bank(
        payload=payload,
        values=[[1.0], [1.0]],
        feasible_keeper=True,
        keeper_value=1.0,
        keeper_target=1.0,
    )
    actions, _, admissible = _baseline(
        monkeypatch=monkeypatch,
        payload=payload,
        replay=keeper_replay,
        canonical_q=canonical_q,
    )
    assert bool(admissible[0])
    assert float(actions["outer"][0]) == 1.0

    node_replay = _replay_bank(
        payload=payload,
        values=[[1.0], [1.0]],
        feasible_keeper=False,
    )
    actions, _, admissible = _baseline(
        monkeypatch=monkeypatch,
        payload=payload,
        replay=node_replay,
        canonical_q=canonical_q,
    )
    assert bool(admissible[0])
    assert float(actions["outer"][0]) == 0.0


def test_all_infeasible_branches_leave_no_winner(monkeypatch, enable_x64) -> None:
    """An all-invalid bank is reported unavailable instead of emitting row zero."""
    del enable_x64
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_: jnp.full_like(outer_action, 10.0),
    )
    payload = _payload()
    replay = _replay_bank(payload=payload, values=[[10.0], [5.0]])

    def canonical_q(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        return jnp.full_like(outer, -jnp.inf), jnp.zeros_like(outer, dtype=bool)

    _, value, admissible = _baseline(
        monkeypatch=monkeypatch,
        payload=payload,
        replay=replay,
        canonical_q=canonical_q,
    )
    assert not bool(admissible[0])
    assert np.isneginf(float(value[0]))


def test_branch_keeps_its_conditional_inner_action(monkeypatch, enable_x64) -> None:
    """Canonical ranking cannot pair a node with another node's inner policy."""
    del enable_x64
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_: jnp.full_like(outer_action, 10.0),
    )
    payload = _payload()
    replay = _replay_bank(
        payload=payload,
        values=[[10.0], [5.0]],
        inner_actions=[[1.0], [2.5]],
    )

    def canonical_q(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        inner = jnp.asarray(candidate_actions["inner"])
        feasible = inner == 2.5
        return jnp.where(feasible, outer + inner, -jnp.inf), feasible

    actions, _, admissible = _baseline(
        monkeypatch=monkeypatch,
        payload=payload,
        replay=replay,
        canonical_q=canonical_q,
    )
    assert bool(admissible[0])
    assert float(actions["outer"][0]) == 2.0
    assert float(actions["inner"][0]) == 2.5


def test_narrow_mesh_uses_one_domain_before_ranking(monkeypatch, enable_x64) -> None:
    """A keeper outside the mesh cannot suppress its reachable first node."""
    del enable_x64
    monkeypatch.setattr(
        simulation_module,
        "_nested_resources",
        lambda *, outer_action, **_: jnp.full_like(outer_action, 10.0),
    )
    payload = _payload(nodes=(2.0, 18.0), low=0.0, high=20.0)
    replay = _replay_bank(
        payload=payload,
        values=[[5.0], [1.0]],
        feasible_keeper=True,
        keeper_value=10.0,
        keeper_target=1.0,
        offset=1.0,
    )

    def canonical_q(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        return jnp.zeros_like(outer), jnp.ones_like(outer, dtype=bool)

    actions, _, admissible = _baseline(
        monkeypatch=monkeypatch,
        payload=payload,
        replay=replay,
        canonical_q=canonical_q,
        state=1.0,
    )
    assert bool(admissible[0])
    assert float(actions["outer"][0]) == 1.0
