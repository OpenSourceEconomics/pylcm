"""A gated edge's `Wbar` topology is the target's, so it is built per target.

`Wbar` lands on the target regime's state grid, and both the params-completed
state-action space and the sharding plan that shape it belong to the target
alone. Several sources reaching one target therefore describe the same
topology, and completing the target's runtime grids once per source repeats
work whose result cannot differ.

The count is the observable here, not a duration: a memo either fires or it
does not, and a timing on a build step this small would measure the box.
"""

import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.engine import Regime, StateActionSpace
from _lcm.regime_building.gated_edges import EdgeChannels
from _lcm.solution import backward_induction
from _lcm.solution.backward_induction import _iter_edge_topologies


@dataclasses.dataclass(frozen=True)
class _MockSolutionPhase:
    """The solve-phase attributes `_iter_edge_topologies` reads."""

    states: MappingProxyType[str, jnp.ndarray]
    """State grids, sized to give the target's `Wbar` its axes."""

    grids: MappingProxyType[str, object] = MappingProxyType({})
    """Grid objects the sharding plan is built from; empty means unsharded."""

    def state_action_space(self, regime_params):  # noqa: ARG002
        return StateActionSpace(
            discrete_actions=MappingProxyType({}),
            continuous_actions=MappingProxyType({}),
            states=self.states,
            state_and_discrete_action_names=tuple(self.states),
        )


class _MockRegime(Regime):
    """A `Regime` carrying only what the topology iterator reads."""

    def __init__(self, *, solution, gated_edges, stakeholders=None) -> None:
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "gated_edges", gated_edges)
        object.__setattr__(self, "stakeholders", stakeholders)


@dataclasses.dataclass(frozen=True)
class _MockEdge:
    """A gated edge carrying only the channel layout the iterator reads."""

    channels: EdgeChannels


def _edge(*, n_components: int = 1) -> _MockEdge:
    """An edge whose continuation stacks `n_components` channels."""
    return _MockEdge(
        channels=EdgeChannels(
            component_names=tuple(f"V_target_{i}" for i in range(n_components)),
            has_dissolution=False,
        )
    )


def _two_sources_one_target() -> MappingProxyType[str, Regime]:
    """Two source regimes whose gated edges both land on `shared_target`."""
    target_states = MappingProxyType({"wealth": jnp.arange(4.0)})
    stateless = MappingProxyType({})
    return MappingProxyType(
        {
            "source_a": _MockRegime(
                solution=_MockSolutionPhase(states=stateless),
                gated_edges=MappingProxyType({"shared_target": _edge()}),
            ),
            "source_b": _MockRegime(
                solution=_MockSolutionPhase(states=stateless),
                gated_edges=MappingProxyType({"shared_target": _edge()}),
            ),
            "shared_target": _MockRegime(
                solution=_MockSolutionPhase(states=target_states),
                gated_edges=MappingProxyType({}),
            ),
        }
    )


@pytest.fixture
def sharding_plan_calls(monkeypatch):
    """Record the target names `_iter_edge_topologies` builds a plan for."""
    calls: list[object] = []
    original = backward_induction._build_regime_sharding

    def counting(*, grids, n_devices):
        calls.append(grids)
        return original(grids=grids, n_devices=n_devices)

    monkeypatch.setattr(backward_induction, "_build_regime_sharding", counting)
    return calls


def test_one_target_reached_twice_is_described_once(sharding_plan_calls):
    """Two sources into one target build that target's topology a single time."""
    regimes = _two_sources_one_target()
    flat_params = MappingProxyType(dict.fromkeys(regimes, MappingProxyType({})))

    list(_iter_edge_topologies(regimes=regimes, flat_params=flat_params))

    assert len(sharding_plan_calls) == 1


def test_both_edges_still_get_their_topology(sharding_plan_calls):  # noqa: ARG001
    """Sharing the work does not drop an edge: both sources are still yielded."""
    regimes = _two_sources_one_target()
    flat_params = MappingProxyType(dict.fromkeys(regimes, MappingProxyType({})))

    got = list(_iter_edge_topologies(regimes=regimes, flat_params=flat_params))

    assert [(source, target) for source, target, _ in got] == [
        ("source_a", "shared_target"),
        ("source_b", "shared_target"),
    ]


def test_a_shared_target_gives_both_edges_the_same_shape(sharding_plan_calls):  # noqa: ARG001
    """The shared topology is the target's grid plus each edge's channel axis.

    Both edges here stack one target component, so both read `(4, 1)`: the
    target's four nodes, and one operand per node. A leg fallback is not an
    operand here — it projects onto another regime's grid and is read where the
    source lands, so it never joins the target-grid stack.
    """
    regimes = _two_sources_one_target()
    flat_params = MappingProxyType(dict.fromkeys(regimes, MappingProxyType({})))

    shapes = {
        topology.shape
        for _, _, topology in _iter_edge_topologies(
            regimes=regimes, flat_params=flat_params
        )
    }

    assert shapes == {(4, 1)}


def test_the_trailing_axis_is_the_edges_own_channel_count(sharding_plan_calls):  # noqa: ARG001
    """Each edge's trailing axis counts ITS operands, not the source's roles.

    The continuation array holds the operands the gate and the branches are
    built from, so the source reads them at its landing point and gates them
    there. A collective source's two roles are produced by that gate, not
    stored: the edge below stacks two target components and so reads `(4, 2)`,
    while the singleton edge beside it stays at `(4, 1)`.
    """
    regimes = dict(_two_sources_one_target())
    regimes["source_a"] = _MockRegime(
        solution=_MockSolutionPhase(states=MappingProxyType({})),
        gated_edges=MappingProxyType({"shared_target": _edge(n_components=2)}),
        stakeholders=("f", "m"),
    )
    frozen = MappingProxyType(regimes)
    flat_params = MappingProxyType(dict.fromkeys(frozen, MappingProxyType({})))

    shapes = {
        source: topology.shape
        for source, _, topology in _iter_edge_topologies(
            regimes=frozen, flat_params=flat_params
        )
    }

    assert shapes == {"source_a": (4, 2), "source_b": (4, 1)}
