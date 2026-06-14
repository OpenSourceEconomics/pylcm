"""`fixed_transition` — the explicit law of motion for fixed states.

`fixed_transition(state_name)` returns an ordinary deterministic law that
keeps the state at its current value each period. It is the only spelling for
fixed states (`None` is rejected) and is legal wherever a law of motion is:
as a bare `state_transitions` entry, inside a `Phased` side, and inside a
per-target dict. The factory argument must match the `state_transitions` key
it is assigned to.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    dead: ScalarInt


def _utility(consumption: float, health: float) -> FloatND:
    return jnp.log(consumption) + health


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age >= 1, _RegimeId.dead, _RegimeId.work)


def _build_regime(**overrides: Any) -> UserRegime:
    """A small valid regime with a fixed health state; tests override slots."""
    spec: dict[str, Any] = {
        "transition": _next_regime,
        "active": lambda age: age < 2,
        "states": {
            "health": DiscreteGrid(_Health),
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
        },
        "state_transitions": {
            "health": fixed_transition("health"),
            "wealth": _next_wealth,
        },
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


def test_fixed_transition_is_the_identity_law() -> None:
    """The returned law maps the state's current value to itself."""
    law = fixed_transition("health")
    assert law(health=jnp.asarray(1, dtype=jnp.int32)) == 1


def test_fixed_state_keeps_its_value_in_simulation() -> None:
    """A `fixed_transition` state carries its seeded value through every period."""
    model = _build_model(_build_regime())
    result = model.simulate(
        params={"work": {"discount_factor": 0.95}},
        initial_conditions={
            "age": jnp.asarray([0.0, 0.0]),
            "health": jnp.asarray([_Health.bad, _Health.good]),
            "wealth": jnp.asarray([50.0, 80.0]),
            "regime_id": jnp.asarray([_RegimeId.work, _RegimeId.work]),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=42,
    )
    df = result.to_dataframe(use_labels=False).sort_values(["subject_id", "period"])
    work_rows = df.loc[df["regime_name"] == "work"]
    for subject_id, expected in ((0, _Health.bad), (1, _Health.good)):
        values = work_rows.loc[work_rows["subject_id"] == subject_id, "health"]
        assert (values == int(expected)).all()


def test_name_mismatch_raises() -> None:
    """The factory argument must match the `state_transitions` key."""
    with pytest.raises(RegimeInitializationError, match=r"helth.*health|health.*helth"):
        _build_regime(
            state_transitions={
                "health": fixed_transition("helth"),
                "wealth": _next_wealth,
            }
        )


def test_name_mismatch_inside_phased_raises() -> None:
    """The key check also covers `Phased` variants."""
    with pytest.raises(RegimeInitializationError, match=r"wrong.*health|health.*wrong"):
        _build_regime(
            state_transitions={
                "health": Phased(
                    solve=fixed_transition("wrong"),
                    simulate=fixed_transition("health"),
                ),
                "wealth": _next_wealth,
            }
        )


def test_name_mismatch_inside_per_target_dict_raises() -> None:
    """The key check also covers per-target dict cells."""
    with pytest.raises(RegimeInitializationError, match=r"wrong.*health|health.*wrong"):
        _build_regime(
            state_transitions={
                "health": {"work": fixed_transition("wrong")},
                "wealth": _next_wealth,
            }
        )


def test_unbound_none_state_transition_is_rejected() -> None:
    """`None` masks a model-level law; unbound, it errors at model build."""
    work = _build_regime(
        state_transitions={"health": None, "wealth": _next_wealth},
    )
    with pytest.raises(ModelInitializationError, match=r"nothing to mask"):
        _build_model(work)


def test_fixed_transition_for_state_not_in_regime_raises() -> None:
    """A fixed state must exist in the regime declaring its law.

    Coverage against the state set is a completeness property, validated when
    the model finalizes its regimes.
    """
    work = _build_regime(
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": _next_wealth,
            "pet_count": fixed_transition("pet_count"),
        }
    )
    with pytest.raises(RegimeInitializationError, match=r"pet_count"):
        _build_model(work)
