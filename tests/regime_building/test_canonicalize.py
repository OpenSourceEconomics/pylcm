"""`canonicalize_regimes` — the model-level canonicalization stage.

Per regime and phase, every `state_transitions` value is expanded into the
canonical target-granular form `Mapping[RegimeName, law]`:

- a bare law broadcasts over the reachable targets that carry the state
- a user per-target dict passes through restricted to its named targets
- a `fixed_transition` entry desugars into per-target identity laws

The engine reads only this canonical form; reachability is resolved here,
not during function compilation.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from _lcm.regime_building.canonicalize import canonicalize_regimes
from _lcm.regime_building.effective import build_effective_regimes
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Phased,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_regime(age: float) -> ScalarInt:
    return jnp.asarray(0, dtype=jnp.int32)


def _health_probs(health: int, probs_array: FloatND) -> FloatND:
    return probs_array[health]


def _wealth_grid() -> LinSpacedGrid:
    return LinSpacedGrid(start=1.0, stop=100.0, n_points=10)


def _base_regime_kwargs() -> dict[str, Any]:
    return {
        "transition": _next_regime,
        "states": {"wealth": _wealth_grid()},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility},
    }


def _regime(**overrides: Any) -> UserRegime:
    spec: dict[str, Any] = _base_regime_kwargs()
    spec.update(overrides)
    return UserRegime(**spec)


def _canonicalize(regimes: dict[str, UserRegime]) -> Mapping:
    return canonicalize_regimes(
        user_regimes=build_effective_regimes(
            user_regimes=regimes, derived_categoricals={}
        )
    )


def _two_regime_model_specs(work_overrides: dict[str, Any]) -> Mapping:
    retire = _regime(state_transitions={"wealth": _next_wealth})
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
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
    """A user per-target dict passes through with exactly its named targets."""
    specs = _two_regime_model_specs(
        {"state_transitions": {"wealth": {"retire": _next_wealth}}}
    )
    canonical = specs["work"].solution.state_transitions["wealth"]
    assert set(canonical) == {"retire"}
    assert canonical["retire"] is _next_wealth


def test_fixed_transition_desugars_to_per_target_identities() -> None:
    """A `fixed_transition` entry becomes identity laws toward each carrier."""
    overrides: dict[str, Any] = {
        "states": {"wealth": _wealth_grid(), "health": DiscreteGrid(_Health)},
        "state_transitions": {
            "wealth": _next_wealth,
            "health": fixed_transition("health"),
        },
        "functions": {"utility": lambda consumption, health: jnp.log(consumption)},
    }
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    specs = _canonicalize(
        {"work": _regime(**overrides), "retire": _regime(**overrides), "dead": dead}
    )
    canonical = specs["work"].solution.state_transitions["health"]
    assert set(canonical) == {"work", "retire"}
    for law in canonical.values():
        assert law(health=jnp.asarray(1, dtype=jnp.int32)) == 1


def test_markov_law_broadcasts_as_markov() -> None:
    """A stochastic law stays `MarkovTransition`-wrapped in every cell."""
    overrides: dict[str, Any] = {
        "states": {"wealth": _wealth_grid(), "health": DiscreteGrid(_Health)},
        "state_transitions": {
            "wealth": _next_wealth,
            "health": MarkovTransition(_health_probs),
        },
        "functions": {"utility": lambda consumption, health: jnp.log(consumption)},
    }
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    specs = _canonicalize(
        {"work": _regime(**overrides), "retire": _regime(**overrides), "dead": dead}
    )
    canonical = specs["work"].solution.state_transitions["health"]
    assert all(isinstance(law, MarkovTransition) for law in canonical.values())


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
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    specs = _canonicalize(
        {"work": _regime(**overrides), "retire": _regime(**overrides), "dead": dead}
    )
    assert "pension_wealth" not in specs["work"].solution.state_transitions
    simulate_canonical = specs["work"].simulation.state_transitions["pension_wealth"]
    assert set(simulate_canonical) == {"work", "retire"}
    assert all(law is _evolve for law in simulate_canonical.values())


def test_terminal_regime_has_empty_canonical_transitions() -> None:
    """A terminal regime canonicalizes to no laws and no regime transition."""
    specs = _two_regime_model_specs({"state_transitions": {"wealth": _next_wealth}})
    assert specs["dead"].solution.state_transitions == {}
    assert specs["dead"].solution.regime_transition is None
    assert specs["dead"].terminal
