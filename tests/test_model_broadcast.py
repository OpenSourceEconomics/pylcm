"""Model-level regime slots: broadcast, masking, and DAG pruning.

`Model(functions=..., constraints=..., states=..., state_transitions=...,
actions=...)` merges each entry into every regime under the exactly-one-level
rule (a name is defined at model level or regime level, never both). A
regime-level `None` masks the model entry for that regime. Broadcast states
and actions are pruned per regime by DAG reachability; regime-level
declarations are never pruned.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _Skill:
    low: ScalarInt
    high: ScalarInt


def _utility_with_skill(consumption: float, skill: int) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * skill)


def _utility_plain(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _work_transition() -> dict[str, MarkovTransition]:
    return {
        "retired": MarkovTransition(lambda age: jnp.where(age >= 1, 0.0, 1.0)),
        "dead": MarkovTransition(lambda age: jnp.where(age >= 1, 1.0, 0.0)),
    }


def _retired_transition() -> dict[str, MarkovTransition]:
    return {
        "retired": MarkovTransition(lambda age: jnp.where(age >= 1, 0.0, 1.0)),
        "dead": MarkovTransition(lambda age: jnp.where(age >= 1, 1.0, 0.0)),
    }


def _work_regime(**overrides: Any) -> UserRegime:
    spec: dict[str, Any] = {
        "transition": _work_transition(),
        "active": lambda age: age < 2,
        "states": {"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility_with_skill},
    }
    spec.update(overrides)
    return UserRegime(**spec)


def _retired_regime(**overrides: Any) -> UserRegime:
    spec: dict[str, Any] = {
        "transition": _retired_transition(),
        "active": lambda age: age < 2,
        "states": {"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility_plain},
    }
    spec.update(overrides)
    return UserRegime(**spec)


def _build_model(**model_slots: Any) -> Model:
    regimes = model_slots.pop(
        "regimes",
        {
            "work": _work_regime(),
            "retired": _retired_regime(),
            "dead": UserRegime(transition=None, functions={"utility": lambda: 0.0}),
        },
    )
    return Model(
        regimes=regimes,
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        **model_slots,
    )


def test_model_functions_broadcast_to_every_regime() -> None:
    """A model-level function is part of every effective regime."""

    def _bonus(wealth: float) -> FloatND:
        return jnp.asarray(wealth * 0.01)

    model = _build_model(functions={"bonus": _bonus})
    assert all("bonus" in regime.functions for regime in model.user_regimes.values())


def test_model_level_utility_satisfies_completeness() -> None:
    """`utility` may arrive via the model-level broadcast.

    Under the exactly-one-level rule this requires every regime — including
    the terminal one — to share the broadcast utility.
    """

    def _bequest_compatible_utility(wealth: float) -> FloatND:
        return jnp.log(wealth)

    model = _build_model(
        regimes={
            "work": _work_regime(functions={}),
            "retired": _retired_regime(functions={}),
            "dead": UserRegime(
                transition=None,
                states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
                functions={},
            ),
        },
        functions={"utility": _bequest_compatible_utility},
    )
    assert all("utility" in regime.functions for regime in model.user_regimes.values())


def test_same_key_at_both_levels_raises() -> None:
    """A name defined at the model AND regime level is ambiguous."""
    with pytest.raises(ModelInitializationError, match=r"[Aa]mbiguous.*utility"):
        _build_model(functions={"utility": _utility_plain})


def test_none_masks_the_model_entry() -> None:
    """A regime-level `None` removes the broadcast entry for that regime."""

    def _bonus(wealth: float) -> FloatND:
        return jnp.asarray(wealth * 0.01)

    model = _build_model(
        regimes={
            "work": _work_regime(),
            "retired": _retired_regime(
                functions={"utility": _utility_plain, "bonus": None}
            ),
            "dead": UserRegime(transition=None, functions={"utility": lambda: 0.0}),
        },
        functions={"bonus": _bonus},
    )
    assert "bonus" in model.user_regimes["work"].functions
    assert "bonus" not in model.user_regimes["retired"].functions


def test_mask_without_model_entry_raises() -> None:
    """Masking a name no model-level slot provides is an error."""
    with pytest.raises(ModelInitializationError, match=r"nothing to mask|no model"):
        _build_model(
            regimes={
                "work": _work_regime(functions={"bonus": None}),
                "retired": _retired_regime(),
                "dead": UserRegime(transition=None, functions={"utility": lambda: 0.0}),
            },
        )


def test_broadcast_state_prunes_where_unused() -> None:
    """A broadcast state survives only in regimes whose DAG reads it."""
    model = _build_model(
        states={"skill": DiscreteGrid(_Skill)},
        state_transitions={"skill": fixed_transition("skill")},
    )
    assert "skill" in model.user_regimes["work"].states
    assert "skill" not in model.user_regimes["retired"].states
    assert model.pruned_variables["retired"] == frozenset({"skill"})
    assert model.pruned_variables["dead"] == frozenset({"skill"})


def test_cross_regime_rescue_keeps_handover_state() -> None:
    """A state unused in a regime's own DAG survives when a reachable target
    keeps it and the law toward that target reads it."""
    model = _build_model(
        regimes={
            "work": _work_regime(functions={"utility": _utility_plain}),
            "retired": _retired_regime(functions={"utility": _utility_with_skill}),
            "dead": UserRegime(transition=None, functions={"utility": lambda: 0.0}),
        },
        states={"skill": DiscreteGrid(_Skill)},
        state_transitions={"skill": fixed_transition("skill")},
    )
    assert "skill" in model.user_regimes["retired"].states
    assert "skill" in model.user_regimes["work"].states
    assert "skill" not in model.user_regimes["dead"].states


def test_broadcast_action_prunes_where_unused() -> None:
    """A broadcast action survives only in regimes whose DAG reads it."""

    def _utility_with_effort(consumption: float, effort: int) -> FloatND:
        return jnp.log(consumption) - 0.1 * effort

    model = _build_model(
        regimes={
            "work": _work_regime(functions={"utility": _utility_with_effort}),
            "retired": _retired_regime(),
            "dead": UserRegime(transition=None, functions={"utility": lambda: 0.0}),
        },
        actions={"effort": DiscreteGrid(_Skill)},
    )
    assert "effort" in model.user_regimes["work"].actions
    assert "effort" not in model.user_regimes["retired"].actions
    assert "effort" in model.pruned_variables["retired"]


def test_sharded_state_pruned_anywhere_raises() -> None:
    """A model-level `distributed=True` state must survive pruning in every
    non-terminal regime."""
    with pytest.raises(ModelInitializationError, match=r"skill.*pruned|pruned.*skill"):
        _build_model(
            states={
                "skill": DiscreteGrid(_Skill, distributed=True),
            },
            state_transitions={"skill": fixed_transition("skill")},
            regimes={
                "work": _work_regime(),
                "retired": _retired_regime(),  # does not read skill
                "dead": UserRegime(transition=None, functions={"utility": lambda: 0.0}),
            },
        )


def test_model_broadcast_solves_and_simulates() -> None:
    """A model assembled from broadcast slots solves and simulates."""
    model = _build_model(
        states={"skill": DiscreteGrid(_Skill)},
        state_transitions={"skill": fixed_transition("skill")},
    )
    result = model.simulate(
        params={
            "work": {"discount_factor": 0.95},
            "retired": {"discount_factor": 0.95},
        },
        initial_conditions={
            "age": jnp.zeros(4),
            "wealth": jnp.full(4, 50.0),
            "skill": jnp.asarray([0, 1, 0, 1]),
            "regime_id": jnp.full(4, _RegimeId.work),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=3,
    )
    df = result.to_dataframe(use_labels=False)
    work_rows = df.loc[df["regime_name"] == "work"].sort_values(
        ["subject_id", "period"]
    )
    values = work_rows.loc[work_rows["period"] == 0, "skill"]
    assert values.tolist() == [0, 1, 0, 1]
