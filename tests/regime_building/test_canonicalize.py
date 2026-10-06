"""`canonicalize_regimes` — the model-level canonicalization stage.

Per regime and phase, every `state_transitions` value is expanded into the
canonical target-granular form `Mapping[RegimeName, law]`:

- a bare law broadcasts over the reachable targets that carry the state
- a user per-target dict passes through restricted to its named targets
- a `fixed_transition` entry desugars into per-target identity laws

The regime transition itself is canonicalized the same way: a coarse form
(bare callable or `StochasticTransition`) becomes a `Mapping[RegimeName, cell]`
over all regimes whose cells share one underlying transition object (one
evaluation, indexed per target), a user per-target dict passes through, and
a terminal regime keeps `None`.

The engine reads only this canonical form; reachability is resolved here,
not during function compilation.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from _lcm.reachability import build_phase_reachability
from _lcm.regime_building.canonicalize import (
    canonicalize_phased_regimes,
    canonicalize_regimes,
)
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.phases import normalize_all_regime_phases
from _lcm.regime_law import RegimeLaws
from lcm import (
    DiscreteGrid,
    LinearAggregator,
    LinearExpectation,
    LinSpacedGrid,
    Phased,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt
from tests.conftest import bind_laws


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(*, wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_regime(age: float) -> ScalarInt:  # noqa: ARG001
    return jnp.asarray(0, dtype=jnp.int32)


def _health_probs(*, health: int, probs_array: FloatND) -> FloatND:
    return probs_array[health]


def _wealth_grid() -> LinSpacedGrid:
    return LinSpacedGrid(start=1.0, stop=100.0, n_points=10)


def _base_regime_kwargs() -> dict[str, Any]:
    return {
        "states": {"wealth": _wealth_grid()},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility},
    }


# A regime together with the law `Model(edges=...)` would bind for it.
type _Declared = tuple[UserRegime, object]


def _regime(*, law: object = _next_regime, **overrides: Any) -> _Declared:
    spec: dict[str, Any] = _base_regime_kwargs()
    spec.update(overrides)
    return UserRegime(**spec), law


def _dead() -> _Declared:
    return UserRegime(functions={"utility": lambda: 0.0}), None


def _split(
    declared: Mapping[str, _Declared],
) -> tuple[dict[str, UserRegime], RegimeLaws]:
    """Separate the regimes from their laws, binding each law."""
    regimes = {name: regime for name, (regime, _) in declared.items()}
    laws = bind_laws({name: law for name, (_, law) in declared.items()})
    return regimes, laws


def _canonicalize(declared: dict[str, _Declared]) -> Mapping:
    regimes, laws = _split(declared)
    return canonicalize_regimes(
        user_regimes=finalize_regimes(
            user_regimes=regimes,
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
            laws=laws,
        ),
        laws=laws,
    )


def _two_regime_model_specs(work_overrides: dict[str, Any]) -> Mapping:
    retire = _regime(state_transitions={"wealth": _next_wealth})
    dead = _dead()
    return _canonicalize(
        {"work": _regime(**work_overrides), "retire": retire, "dead": dead}
    )


def test_bare_law_broadcasts_over_carrying_targets() -> None:
    """A bare law expands to one entry per reachable target carrying the state."""
    specs = _two_regime_model_specs({"state_transitions": {"wealth": _next_wealth}})
    canonical = specs["work"].solution.state_transitions["wealth"]
    assert isinstance(canonical, Mapping)
    assert set(canonical) == {"work", "retire"}
    assert all(law is _next_wealth for law in canonical.values())


def test_per_target_dict_is_restricted_to_named_targets() -> None:
    """A user per-target dict passes through with exactly its named targets.

    The per-target regime transition declares "retire" and "dead" reachable;
    "work" carries `wealth` but is structurally unreachable, so the law
    toward it is neither required nor produced.
    """
    specs = _two_regime_model_specs(
        {
            "law": {
                "retire": StochasticTransition(func=lambda age: jnp.asarray(0.6)),  # noqa: ARG005
                "dead": StochasticTransition(func=lambda age: jnp.asarray(0.4)),  # noqa: ARG005
            },
            "state_transitions": {"wealth": {"retire": _next_wealth}},
        }
    )
    canonical = specs["work"].solution.state_transitions["wealth"]
    assert set(canonical) == {"retire"}
    assert canonical["retire"] is _next_wealth


def test_fixed_transition_desugars_to_per_target_identities() -> None:
    """A `fixed_transition` entry becomes identity laws toward each carrier."""
    overrides: dict[str, Any] = {
        "states": {
            "wealth": _wealth_grid(),
            "health": DiscreteGrid(category_class=_Health),
        },
        "state_transitions": {
            "wealth": _next_wealth,
            "health": fixed_transition("health"),
        },
        "functions": {
            "utility": lambda consumption, health: jnp.log(consumption)  # noqa: ARG005
        },
    }
    dead = _dead()
    specs = _canonicalize(
        {"work": _regime(**overrides), "retire": _regime(**overrides), "dead": dead}
    )
    canonical = specs["work"].solution.state_transitions["health"]
    assert set(canonical) == {"work", "retire"}
    for law in canonical.values():
        assert law(health=jnp.asarray(1, dtype=jnp.int32)) == 1


def test_markov_law_broadcasts_as_markov() -> None:
    """A stochastic law stays `StochasticTransition`-wrapped in every cell."""
    overrides: dict[str, Any] = {
        "states": {
            "wealth": _wealth_grid(),
            "health": DiscreteGrid(category_class=_Health),
        },
        "state_transitions": {
            "wealth": _next_wealth,
            "health": StochasticTransition(func=_health_probs),
        },
        "functions": {
            "utility": lambda consumption, health: jnp.log(consumption)  # noqa: ARG005
        },
    }
    dead = _dead()
    specs = _canonicalize(
        {"work": _regime(**overrides), "retire": _regime(**overrides), "dead": dead}
    )
    canonical = specs["work"].solution.state_transitions["health"]
    assert all(isinstance(law, StochasticTransition) for law in canonical.values())


def test_carried_state_law_lives_only_in_the_simulation_slice() -> None:
    """A carried state's law targets carriers in simulation; solve has no entry."""

    def _impute(wealth: float) -> float:
        return wealth * 0.1

    def _evolve(pension_wealth: float) -> float:
        return pension_wealth * 1.03

    overrides: dict[str, Any] = {
        "states": {
            "wealth": _wealth_grid(),
            "pension_wealth": Phased(
                solve=_impute,
                simulate=LinSpacedGrid(start=0.0, stop=5.0, n_points=2),
            ),
        },
        "state_transitions": {"wealth": _next_wealth, "pension_wealth": _evolve},
    }
    dead = _dead()
    specs = _canonicalize(
        {"work": _regime(**overrides), "retire": _regime(**overrides), "dead": dead}
    )
    assert "pension_wealth" not in specs["work"].solution.state_transitions
    simulate_canonical = specs["work"].simulation.state_transitions["pension_wealth"]
    assert set(simulate_canonical) == {"work", "retire"}
    assert all(law is _evolve for law in simulate_canonical.values())


def test_coarse_markov_regime_transition_canonicalizes_to_shared_cells() -> None:
    """A coarse `StochasticTransition` regime transition becomes a per-target mapping.

    The canonical mapping covers all regimes (a coarse form declares every
    regime reachable) and every cell references the same underlying
    transition object, so the engine evaluates it once and indexes per
    target.
    """
    transition = StochasticTransition(func=lambda age: jnp.asarray([0.5, 0.3, 0.2]))  # noqa: ARG005
    specs = _two_regime_model_specs(
        {
            "law": transition,
            "state_transitions": {"wealth": _next_wealth},
        }
    )
    canonical = specs["work"].solution.regime_transition
    assert isinstance(canonical, Mapping)
    assert set(canonical) == {"work", "retire", "dead"}
    assert all(cell.underlying is transition for cell in canonical.values())


def test_coarse_deterministic_regime_transition_canonicalizes_to_shared_cells() -> None:
    """A coarse deterministic regime transition becomes a per-target mapping.

    The canonical mapping covers all regimes and every cell references the
    same underlying callable.
    """
    specs = _two_regime_model_specs({"state_transitions": {"wealth": _next_wealth}})
    canonical = specs["work"].solution.regime_transition
    assert isinstance(canonical, Mapping)
    assert set(canonical) == {"work", "retire", "dead"}
    assert all(cell.underlying is _next_regime for cell in canonical.values())


def test_temporal_graph_limits_canonical_transition_bundles() -> None:
    """Only the graph's declared targets create canonical transition bundles."""
    regimes, laws = _split(
        {
            "work": _regime(state_transitions={"wealth": _next_wealth}),
            "retire": _regime(state_transitions={"wealth": _next_wealth}),
            "dead": _dead(),
        }
    )
    finalized = finalize_regimes(
        user_regimes=regimes,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
        laws=laws,
    )
    raw_specs = normalize_all_regime_phases(user_regimes=finalized, laws=laws)
    graph = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={"work": {0}, "retire": {1}, "dead": {1}},
        support_by_period={"work": {0: ("retire", "dead")}},
        terminal_regimes={"dead"},
    )

    specs = canonicalize_phased_regimes(
        raw_specs=raw_specs,
        all_regime_names=frozenset(regimes),
        solution_reachability=graph,
        simulation_reachability=graph,
    )

    wealth_transitions = specs["work"].solution.state_transitions["wealth"]
    regime_transition = specs["work"].solution.regime_transition
    assert isinstance(wealth_transitions, Mapping)
    assert isinstance(regime_transition, Mapping)
    assert set(wealth_transitions) == {"retire"}
    assert set(regime_transition) == {"dead", "retire"}


def test_per_target_regime_transition_passes_through() -> None:
    """A user per-target regime transition stays a mapping of exactly its cells."""
    to_retire = StochasticTransition(func=lambda age: jnp.asarray(0.6))  # noqa: ARG005
    to_dead = StochasticTransition(func=lambda age: jnp.asarray(0.4))  # noqa: ARG005
    specs = _two_regime_model_specs(
        {
            "law": {"retire": to_retire, "dead": to_dead},
            "state_transitions": {"wealth": {"retire": _next_wealth}},
        }
    )
    canonical = specs["work"].solution.regime_transition
    assert isinstance(canonical, Mapping)
    assert dict(canonical) == {"retire": to_retire, "dead": to_dead}


def test_terminal_regime_has_empty_canonical_transitions() -> None:
    """A terminal regime canonicalizes to no laws and no regime transition."""
    specs = _two_regime_model_specs({"state_transitions": {"wealth": _next_wealth}})
    assert specs["dead"].solution.state_transitions == {}
    assert specs["dead"].solution.regime_transition is None


def test_two_step_seam_matches_wrapper() -> None:
    """`canonicalize_phased_regimes` over `normalize_all_regime_phases` equals
    the `canonicalize_regimes` wrapper.

    The main model path splits the wrapper into an explicit phase-normalization
    step and a canonicalization step so that model-level age normalization can
    sit between them. The split must not change the end result.
    """
    regimes, laws = _split(
        {
            "work": _regime(state_transitions={"wealth": _next_wealth}),
            "retire": _regime(state_transitions={"wealth": _next_wealth}),
            "dead": _dead(),
        }
    )
    finalized = finalize_regimes(
        user_regimes=regimes,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
        laws=laws,
    )

    wrapper = canonicalize_regimes(user_regimes=finalized, laws=laws)

    raw_specs = normalize_all_regime_phases(user_regimes=finalized, laws=laws)
    two_step = canonicalize_phased_regimes(
        raw_specs=raw_specs,
        all_regime_names=frozenset(finalized),
    )

    assert set(wrapper) == set(two_step)
    for regime_name in wrapper:
        for phase in ("solution", "simulation"):
            wrapped_slice = getattr(wrapper[regime_name], phase)
            two_step_slice = getattr(two_step[regime_name], phase)
            assert dict(wrapped_slice.functions) == dict(two_step_slice.functions)
            assert dict(wrapped_slice.constraints) == dict(two_step_slice.constraints)
            assert dict(wrapped_slice.grid_states) == dict(two_step_slice.grid_states)
            assert dict(wrapped_slice.state_transitions) == dict(
                two_step_slice.state_transitions
            )
            assert wrapped_slice.regime_transition == two_step_slice.regime_transition
            assert (
                wrapped_slice.stochastic_regime_transition
                == two_step_slice.stochastic_regime_transition
            )
