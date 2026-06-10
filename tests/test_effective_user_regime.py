"""`EffectiveUserRegime` — the regime as the model runs it.

A user `Regime` validates only local, value-shape properties at construction;
completeness (a `utility` entry, state-transition coverage, default-`H`
injection, state/action overlap) is validated when the model builds its
effective regimes. `model.user_regimes` exposes the effective form: complete,
immutable, still in user vocabulary.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from _lcm.regime_building.effective import EffectiveUserRegime
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    dead: ScalarInt


def _utility(consumption: float, wealth: float) -> FloatND:
    return jnp.log(consumption) + wealth


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age >= 1, _RegimeId.dead, _RegimeId.work)


def _build_work_regime(**overrides: Any) -> UserRegime:
    spec: dict[str, Any] = {
        "transition": _next_regime,
        "active": lambda age: age < 2,
        "states": {"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility},
    }
    spec.update(overrides)
    return UserRegime(**spec)


def _build_model(work: UserRegime) -> Model:
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 1,
    )
    return Model(
        regimes={"work": work, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_regime_without_utility_constructs() -> None:
    """A regime may omit `utility` at construction; completeness is a model concern."""
    regime = _build_work_regime(functions={})
    assert "utility" not in regime.functions


def test_model_without_utility_raises_naming_the_regime() -> None:
    """Building a model from a utility-less regime fails, naming the regime."""
    work = _build_work_regime(functions={})
    with pytest.raises(RegimeInitializationError, match=r"work.*utility|utility"):
        _build_model(work)


def test_regime_with_uncovered_state_constructs() -> None:
    """A regime may omit a state's transition entry at construction."""
    regime = _build_work_regime(state_transitions={})
    assert "wealth" not in regime.state_transitions


def test_model_with_uncovered_state_raises() -> None:
    """Building a model from a coverage-incomplete regime fails."""
    work = _build_work_regime(state_transitions={})
    with pytest.raises(RegimeInitializationError, match=r"wealth"):
        _build_model(work)


def test_user_regimes_are_effective_with_default_h() -> None:
    """`model.user_regimes` exposes effective regimes: `H` injected, raw untouched."""
    work = _build_work_regime()
    model = _build_model(work)
    effective = model.user_regimes["work"]
    assert isinstance(effective, EffectiveUserRegime)
    assert "H" in effective.functions
    assert "H" not in work.functions


def test_state_action_overlap_raises_at_model_time() -> None:
    """A name used as both state and action fails when the model builds."""
    work = _build_work_regime(
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5),
        },
        state_transitions={
            "wealth": _next_wealth,
            "consumption": _next_wealth,
        },
    )
    with pytest.raises(RegimeInitializationError, match=r"consumption"):
        _build_model(work)


def test_effective_regime_is_a_user_regime() -> None:
    """Engine code typed against the user `Regime` accepts the effective form."""
    model = _build_model(_build_work_regime())
    assert isinstance(model.user_regimes["work"], UserRegime)


def test_model_level_derived_categoricals_are_merged() -> None:
    """Model-level `derived_categoricals` appear on every effective regime."""

    @categorical(ordered=False)
    class _Flag:
        off: ScalarInt
        on: ScalarInt

    work = _build_work_regime()
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 1,
    )
    model = Model(
        regimes={"work": work, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        derived_categoricals={"flag": DiscreteGrid(_Flag)},
    )
    assert "flag" in model.user_regimes["work"].derived_categoricals
    assert "flag" in model.user_regimes["dead"].derived_categoricals
