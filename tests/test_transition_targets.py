"""A `Transition` with a per-target law reads its destinations off the law.

The destinations of a per-target law are the keys of its cells, plus the route
fallbacks of its gates; each destination's source ages are the ages whose case
names it, and a gate fallback's are those of its gated target. Supplying
`targets` as well is allowed, but only when it says the same thing.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    CollectiveUtility,
    Gate,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime
from lcm.transition import ModelEdges, PhaseEdges
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt

# Four ages: transitions happen out of ages 0, 1 and 2.
_AGES = AgeGrid(start=0, inclusive_stop=3, step="Y")

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    couple: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


# Leave `alive` for one of the two terminal singleton regimes, half and half.
_EXIT_LAW = {
    "retired": StochasticTransition(func=lambda age: 0.5 * jnp.ones_like(age)),
    "dead": StochasticTransition(func=lambda age: 0.5 * jnp.ones_like(age)),
}


def test_per_target_law_without_targets_derives_each_destination_and_its_ages():
    """Each target is reached at exactly the ages whose case names it."""
    model = _model(
        Transition(
            law=ByAge(
                cases={
                    (0, 1): {
                        "alive": StochasticTransition(func=_survive),
                        "dead": StochasticTransition(func=_die),
                    },
                    2: {"dead": StochasticTransition(func=_certain)},
                }
            )
        )
    )
    assert dict(model.graph.edges.solve["alive"]) == {
        "alive": frozenset({0, 1}),
        "dead": frozenset({0, 1, 2}),
    }


def test_plain_per_target_law_reaches_its_targets_at_every_non_final_age():
    """A law with no age schedule names its targets out of every non-final age."""
    model = _model(Transition(law=_EXIT_LAW))
    assert dict(model.graph.edges.simulate["alive"]) == {
        "retired": frozenset({0, 1, 2}),
        "dead": frozenset({0, 1, 2}),
    }


def test_supplied_targets_equal_to_the_derived_ones_are_accepted():
    """Spelling out the derived destinations changes nothing."""
    model = _model(
        Transition(targets={"retired": (0, 1, 2), "dead": (0, 1, 2)}, law=_EXIT_LAW)
    )
    assert dict(model.graph.edges.solve["alive"]) == {
        "retired": frozenset({0, 1, 2}),
        "dead": frozenset({0, 1, 2}),
    }


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(
            {"retired": AgeRange(start=0), "dead": AgeRange(start=0)},
            id="open-ended-range",
        ),
        pytest.param({"retired": (0, 1, 2, 3), "dead": (0, 1, 2, 3)}, id="final-age"),
    ],
)
def test_supplied_targets_that_also_select_the_final_age_are_accepted(targets):
    """No transition leaves the final age, so selecting it changes no destination."""
    model = _model(Transition(targets=targets, law=_EXIT_LAW))
    assert dict(model.graph.edges.solve["alive"]) == {
        "retired": frozenset({0, 1, 2}),
        "dead": frozenset({0, 1, 2}),
    }


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param({"retired": (0, 1, 2), "dead": (0, 1)}, id="fewer-ages"),
        pytest.param(
            {"retired": (0, 1, 2, 3), "dead": (0, 1, 3)},
            id="fewer-ages-beside-the-final-age",
        ),
        pytest.param({"retired": (0, 1, 2)}, id="missing-target"),
        pytest.param(
            {"retired": (0, 1, 2), "dead": (0, 1, 2), "alive": 0}, id="extra-target"
        ),
    ],
)
def test_supplied_targets_that_differ_from_the_derived_ones_are_refused(targets):
    """The error names both the supplied and the derived destinations."""
    with pytest.raises(
        ModelInitializationError, match=r"(?s)'alive'.*supplied.*derived"
    ):
        _model(Transition(targets=targets, law=_EXIT_LAW))


def test_age_schedule_names_each_target_at_the_ages_its_edge_fires():
    """A target active only from age 2 is named only by the case for age 1."""
    model = _model_with_edges(
        edges={
            "alive": Transition(law=_retirement_law("by-age")),
            "retired": {"dead": (2,)},
        }
    )
    assert dict(model.graph.edges.solve["alive"]) == {
        "alive": frozenset({0}),
        "retired": frozenset({1}),
        "dead": frozenset({0, 1}),
    }


@pytest.mark.parametrize(
    "case",
    [
        "source-active-at-fewer-ages",
        "target-active-from-a-later-age",
        "target-entered-before-it-is-active",
        "gate-fallback-at-fewer-ages",
    ],
)
def test_supplied_targets_narrower_than_the_law_are_refused(case):
    """A law with no age schedule names its targets out of every non-final age.

    Supplied targets that name fewer ages disagree with it, whether or not the
    edge could fire there; an age schedule says where each target is reached.
    """
    with pytest.raises(
        ModelInitializationError, match=r"(?s)'alive'.*supplied.*derived"
    ):
        _model_with_edges(edges=_narrowed_edges(case))


def test_law_over_all_targets_still_requires_targets():
    """A function returning a regime code names no target, so it needs `targets`."""
    with pytest.raises(RegimeInitializationError, match="targets"):
        Transition(law=_always_dead)


def test_gate_fallbacks_join_the_derived_targets_at_the_gated_ages():
    """A route's fallback regime is reached wherever its gated target is."""
    model = _model(
        Transition(
            law=ByAge(
                cases={
                    0: {
                        "couple": StochasticTransition(func=_survive),
                        "dead": StochasticTransition(func=_die),
                    },
                    (1, 2): {"dead": StochasticTransition(func=_certain)},
                }
            ),
            gates={"couple": _consent_gate()},
        )
    )
    assert dict(model.graph.edges.solve["alive"]) == {
        "couple": frozenset({0}),
        "dead": frozenset({0, 1, 2}),
        "alive": frozenset({0}),
    }


def test_gate_on_a_target_the_law_never_reaches_is_refused():
    """A gate belongs to a destination of its own `Transition`."""
    with pytest.raises(ModelInitializationError, match="'couple'"):
        _model(
            Transition(
                law={"dead": StochasticTransition(func=_certain)},
                gates={"couple": _consent_gate()},
            )
        )


def test_gate_declared_in_one_phase_only_is_refused():
    """A target reached in both phases is gated in both or in neither."""
    law = {
        "couple": StochasticTransition(func=_survive),
        "dead": StochasticTransition(func=_die),
    }
    with pytest.raises(ModelInitializationError, match="'couple'"):
        _model(
            Phased(
                solve={"alive": Transition(law=law, gates={"couple": _consent_gate()})},
                simulate={"alive": Transition(law=law)},
            )
        )


def _model(alive_edges: Transition | Phased[PhaseEdges, PhaseEdges]) -> Model:
    """Build a four-regime model whose `alive` source has `alive_edges`."""
    return _model_with_edges(
        edges=alive_edges if isinstance(alive_edges, Phased) else {"alive": alive_edges}
    )


def _model_with_edges(*, edges: ModelEdges) -> Model:
    """Build the four-regime model with `edges`."""
    return Model(
        regimes={
            "alive": Regime(
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _keep_wealth},
                functions={"utility": _utility},
            ),
            "couple": Regime(
                states={"wealth": _WEALTH},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _utility, "m": _utility}
                    )
                },
            ),
            "retired": Regime(
                states={"wealth": _WEALTH},
                state_transitions=(
                    {"wealth": _keep_wealth}
                    if isinstance(edges, Mapping) and "retired" in edges
                    else {}
                ),
                functions={"utility": _utility},
            ),
            "dead": Regime(states={"wealth": _WEALTH}, functions={"utility": _utility}),
        },
        ages=_AGES,
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges=edges,
        enable_jit=False,
    )


def _narrowed_edges(case: str) -> dict[str, Transition | dict[str, tuple[int]]]:
    """Edges whose supplied targets name fewer ages than their plain law."""
    if case == "source-active-at-fewer-ages":
        return {
            "alive": Transition(
                targets={"retired": (0, 1), "dead": (0, 1)}, law=_EXIT_LAW
            )
        }
    if case == "gate-fallback-at-fewer-ages":
        return {
            "alive": Transition(
                targets={"couple": (0, 1), "dead": (0, 1), "retired": (0, 1)},
                law={
                    "couple": StochasticTransition(func=_survive),
                    "dead": StochasticTransition(func=_die),
                },
                gates={"couple": _consent_gate(fallback="retired")},
            )
        }
    retired_ages = (1,) if case == "target-active-from-a-later-age" else (0, 1)
    return {
        "alive": Transition(
            targets={"alive": (0,), "retired": retired_ages, "dead": (0, 1)},
            law=_retirement_law("plain"),
        ),
        "retired": {"dead": (2,)},
    }


def _retirement_law(form: str) -> dict[str, StochasticTransition] | ByAge:
    """A law out of `alive` that names `retired` from age 1 on.

    `"plain"` is one per-target mapping over every age; `"by-age"` names
    `alive` at age 0 and `retired` at age 1.
    """
    if form == "plain":
        return {
            "alive": StochasticTransition(func=_stay_young),
            "retired": StochasticTransition(func=_retire),
            "dead": StochasticTransition(func=_die),
        }
    return ByAge(
        cases={
            0: {
                "alive": StochasticTransition(func=_survive),
                "dead": StochasticTransition(func=_die),
            },
            1: {
                "retired": StochasticTransition(func=_survive),
                "dead": StochasticTransition(func=_die),
            },
        }
    )


def _consent_gate(fallback: str = "alive") -> Gate:
    """A gate into the couple that sends a refused single to `fallback`."""
    return Gate(
        predicate=_couple_is_better,
        routes={
            "her": StakeholderRoute(
                target_stakeholder="f",
                fallback=ProjectedRegimeValue(
                    regime=fallback, projection={"wealth": _keep_wealth}
                ),
            )
        },
    )


def _couple_is_better(*, V_target_f: FloatND, V_target_m: FloatND) -> BoolND:
    return V_target_f >= V_target_m


def _keep_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _survive(age: FloatND) -> FloatND:
    return 0.5 * jnp.ones_like(age, dtype=float)


def _die(age: FloatND) -> FloatND:
    return 0.5 * jnp.ones_like(age, dtype=float)


def _stay_young(age: FloatND) -> FloatND:
    return jnp.where(age < 1, 0.5, 0.0)


def _retire(age: FloatND) -> FloatND:
    return jnp.where(age < 1, 0.0, 0.5)


def _certain(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _always_dead() -> ScalarInt:
    return _RegimeId.dead
