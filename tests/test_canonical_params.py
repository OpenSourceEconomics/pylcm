"""Canonical params: the target level is a genuine tree level.

Per-target transition parameters live under nested template paths
`template[regime][target][transition_func][param]` — the target is a regime
name, not a mangled `to_<target>_…` prefix — mirroring the canonical
transition bundles, so param qnames parallel engine function qnames.
"""

from typing import Any

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_wealth_taxed(wealth: float, consumption: float, exit_tax: float) -> float:
    return (wealth - consumption) * (1.0 - exit_tax)


def _prob_dead(age: float, hazard: float) -> FloatND:
    return jnp.clip(hazard * age, 0.0, 1.0)


def _work_regime(**overrides: Any) -> UserRegime:
    spec: dict[str, Any] = {
        "transition": {
            "retired": MarkovTransition(
                lambda age, hazard: 1.0 - _prob_dead(age, hazard)
            ),
            "dead": MarkovTransition(_prob_dead),
        },
        "active": lambda age: age < 2,
        "states": {"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        "state_transitions": {
            "wealth": {
                "retired": _next_wealth_taxed,
            },
        },
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility},
    }
    spec.update(overrides)
    return UserRegime(**spec)


def _retired_regime() -> UserRegime:
    return UserRegime(
        transition={
            "dead": MarkovTransition(lambda age: jnp.where(age >= 1, 1.0, 0.0)),
        },
        active=lambda age: age < 2,
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        functions={"utility": _utility},
    )


def _build_model(work: UserRegime) -> Model:
    return Model(
        regimes={
            "work": work,
            "retired": _retired_regime(),
            "dead": UserRegime(transition=None, functions={"utility": lambda: 0.0}),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_per_target_state_law_params_nest_under_the_target() -> None:
    """A per-target law's params live at `template[regime][target][func]`."""
    model = _build_model(_work_regime())
    template = model.get_params_template()
    assert "exit_tax" in template["work"]["retired"]["next_wealth"]


def test_per_target_regime_transition_params_nest_under_the_target() -> None:
    """A granular cell's params live at `template[regime][target]["next_regime"]`."""
    model = _build_model(_work_regime())
    template = model.get_params_template()
    assert "hazard" in template["work"]["retired"]["next_regime"]
    assert "hazard" in template["work"]["dead"]["next_regime"]


def test_broadcast_law_params_stay_coarse_in_the_template() -> None:
    """The template mirrors the user's coarseness: a bare law keeps one
    unnested `next_<state>` key."""
    model = _build_model(_work_regime())
    template = model.get_params_template()
    assert "next_wealth" in template["retired"]


def test_per_target_params_solve_and_bind_per_target() -> None:
    """Per-target param values reach their target's law: solve succeeds with
    target-nested params and a coarse spelling for the same model errors
    nowhere else."""
    model = _build_model(_work_regime())
    params = {
        "work": {
            "discount_factor": 0.95,
            "retired": {
                "next_wealth": {"exit_tax": 0.1},
                "next_regime": {"hazard": 0.01},
            },
            "dead": {"next_regime": {"hazard": 0.01}},
        },
        "retired": {"discount_factor": 0.95},
    }
    regime_to_v = model.solve(params=params, log_level="off")
    assert set(regime_to_v[0]) >= {"work", "retired"}


def test_old_mangled_spelling_is_gone() -> None:
    """The `to_<target>_…` template spelling does not exist."""
    model = _build_model(_work_regime())
    template = model.get_params_template()
    assert not any(key.startswith("to_") for key in template["work"])
