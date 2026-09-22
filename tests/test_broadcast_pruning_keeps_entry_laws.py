"""Pruning a broadcast state keeps the entry laws it carries toward other regimes.

A regime that never reads a state can still be the source of an edge into a regime
that does, and it supplies the entered value through a target-keyed law in
`state_transitions`. Declaring the state at model level rather than on the target
regime makes it a pruning candidate in the source, but it must not change which
entry laws the model has: promoting a state to model level is a declaration move,
not a change of the transition structure.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=10)
_CONSUMPTION_GRID = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
_PARAMS = {"working": {"discount_factor": 0.95}}
_INITIAL_CONDITIONS = {
    "age": jnp.zeros(4),
    "wealth": jnp.asarray([20.0, 40.0, 60.0, 80.0]),
    "regime_id": jnp.full(4, _RegimeId.working),
}


def _utility_without_health(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _bequest_utility(wealth: float) -> FloatND:
    return jnp.log(wealth)


def _utility_with_health(*, wealth: float, health: int) -> FloatND:
    return jnp.log(wealth) * (1.0 + 0.1 * health)


def _next_wealth(*, wealth: float, consumption: float) -> float:
    return wealth - consumption


def _always_retire() -> FloatND:
    return jnp.asarray(1.0)


def _entry_health(wealth: float) -> FloatND:
    """Probabilities over `_Health`, richer agents arriving healthier."""
    good = jnp.clip(wealth / 100.0, 0.0, 1.0)
    return jnp.stack([1.0 - good, good], axis=-1)


def _working_regime(**overrides: Any) -> Regime:
    spec: dict[str, Any] = {
        "transition": {"retired": MarkovTransition(_always_retire)},
        "active": lambda age: age < 1,
        "states": {"wealth": _WEALTH_GRID},
        "state_transitions": {
            "wealth": _next_wealth,
            "health": {"retired": MarkovTransition(_entry_health)},
        },
        "actions": {"consumption": _CONSUMPTION_GRID},
        "functions": {"utility": _utility_without_health},
    }
    spec.update(overrides)
    return Regime(**spec)


def _retired_regime(**overrides: Any) -> Regime:
    spec: dict[str, Any] = {
        "transition": None,
        "active": lambda age: age >= 1,
        "states": {"wealth": _WEALTH_GRID},
        "functions": {"utility": _utility_with_health},
    }
    spec.update(overrides)
    return Regime(**spec)


def _entry_targets(*, regime: Regime, state_name: str) -> set[str]:
    """Return the targets a regime's entry law for `state_name` names."""
    law = regime.state_transitions[state_name]
    assert isinstance(law, Mapping)
    return set(law)


def _build(*, regimes: dict[str, Regime], **model_slots: Any) -> Model:
    return Model(
        regimes=regimes,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        **model_slots,
    )


def _regime_level_model() -> Model:
    """`health` declared on the regime that reads it."""
    return _build(
        regimes={
            "working": _working_regime(),
            "retired": _retired_regime(
                states={
                    "wealth": _WEALTH_GRID,
                    "health": DiscreteGrid(category_class=_Health),
                }
            ),
        }
    )


def _model_level_model() -> Model:
    """The same model with `health` promoted to a model-level state."""
    return _build(
        regimes={"working": _working_regime(), "retired": _retired_regime()},
        states={"health": DiscreteGrid(category_class=_Health)},
    )


def test_model_level_state_entered_through_a_law_builds() -> None:
    """A state only the target reads builds when declared at model level."""
    model = _model_level_model()
    assert "health" in model.user_regimes["retired"].states
    assert "health" not in model.user_regimes["working"].states


def test_promoting_the_state_leaves_the_solution_unchanged() -> None:
    """Model-level and regime-level declarations solve to the same value arrays."""
    regime_level = _regime_level_model().solve(params=_PARAMS, log_level="off")
    model_level = _model_level_model().solve(params=_PARAMS, log_level="off")

    for period in (0, 1):
        expected_period = regime_level.values[period]
        assert set(model_level.values[period]) == set(expected_period)
        for regime_name, expected in expected_period.items():
            np.testing.assert_array_almost_equal(
                np.asarray(model_level.values[period][regime_name]),
                np.asarray(expected),
                decimal=DECIMAL_PRECISION,
                err_msg=f"{regime_name}, period {period}",
            )


def test_promoting_the_state_leaves_the_simulation_unchanged() -> None:
    """Model-level and regime-level declarations simulate identically."""
    regime_level = _regime_level_model().simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        log_level="off",
        seed=7,
    )
    model_level = _model_level_model().simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        log_level="off",
        seed=7,
    )

    expected = regime_level.to_dataframe(use_labels=False).sort_values(
        ["subject_id", "period"]
    )
    got = model_level.to_dataframe(use_labels=False).sort_values(
        ["subject_id", "period"]
    )
    assert list(got.columns) == list(expected.columns)
    np.testing.assert_array_equal(
        got["health"].to_numpy(dtype=float),
        expected["health"].to_numpy(dtype=float),
    )
    np.testing.assert_array_almost_equal(
        got["wealth"].to_numpy(dtype=float),
        expected["wealth"].to_numpy(dtype=float),
        decimal=DECIMAL_PRECISION,
    )


def test_pruned_source_keeps_the_target_keyed_entry_law() -> None:
    """The state is pruned from the source, its law toward the target is not."""
    model = _model_level_model()
    working = model.user_regimes["working"]

    assert model.pruned_variables["working"] == frozenset({"health"})
    assert "health" not in working.states
    assert "health" in working.state_transitions
    assert _entry_targets(regime=working, state_name="health") == {"retired"}


def test_an_unkeyed_entry_law_survives_toward_a_retaining_target() -> None:
    """An unkeyed law enters every reachable target, so pruning keeps it."""
    model = _build(
        regimes={
            "working": _working_regime(
                state_transitions={
                    "wealth": _next_wealth,
                    "health": MarkovTransition(_entry_health),
                }
            ),
            "retired": _retired_regime(),
        },
        states={"health": DiscreteGrid(category_class=_Health)},
    )
    working = model.user_regimes["working"]

    assert model.pruned_variables["working"] == frozenset({"health"})
    assert isinstance(working.state_transitions["health"], MarkovTransition)


def test_an_identity_law_goes_with_the_state_it_fixes() -> None:
    """A law fixing a state the regime no longer carries is dropped with it."""
    model = _build(
        regimes={
            "working": _working_regime(
                state_transitions={
                    "wealth": _next_wealth,
                    "health": fixed_transition("health"),
                }
            ),
            "retired": _retired_regime(functions={"utility": _bequest_utility}),
        },
        states={"health": DiscreteGrid(category_class=_Health)},
    )
    working = model.user_regimes["working"]

    assert model.pruned_variables["working"] == frozenset({"health"})
    assert model.pruned_variables["retired"] == frozenset({"health"})
    assert "health" not in working.state_transitions


def test_a_keyed_law_is_dropped_when_no_target_retains_the_state() -> None:
    """With the state pruned on the target too, the entry law has nothing to enter."""
    model = _build(
        regimes={
            "working": _working_regime(),
            "retired": _retired_regime(functions={"utility": _bequest_utility}),
        },
        states={"health": DiscreteGrid(category_class=_Health)},
    )
    working = model.user_regimes["working"]

    assert model.pruned_variables["working"] == frozenset({"health"})
    assert model.pruned_variables["retired"] == frozenset({"health"})
    assert "health" not in working.state_transitions


def test_a_kept_entry_law_leaves_no_dangling_reference() -> None:
    """Everything a retained entry law reads survives pruning in the source."""

    def _entry_health_from_endowment(endowment: float) -> FloatND:
        good = jnp.clip(endowment, 0.0, 1.0)
        return jnp.stack([1.0 - good, good], axis=-1)

    model = _build(
        regimes={
            "working": _working_regime(
                state_transitions={
                    "wealth": _next_wealth,
                    "health": {
                        "retired": MarkovTransition(_entry_health_from_endowment)
                    },
                    "endowment": fixed_transition("endowment"),
                }
            ),
            "retired": _retired_regime(),
        },
        states={
            "health": DiscreteGrid(category_class=_Health),
            "endowment": LinSpacedGrid(start=0.0, stop=1.0, n_points=3),
        },
    )
    working = model.user_regimes["working"]

    assert "endowment" in working.states
    assert "endowment" not in model.pruned_variables["working"]
    assert _entry_targets(regime=working, state_name="health") == {"retired"}


def test_an_entry_law_reading_the_state_itself_keeps_the_state() -> None:
    """A law that reads the state it enters forces the source to carry it."""

    def _persistent_health(health: int) -> FloatND:
        return jnp.where(
            health == _Health.good,
            jnp.asarray([0.2, 0.8]),
            jnp.asarray([0.8, 0.2]),
        )

    model = _build(
        regimes={
            "working": _working_regime(
                state_transitions={
                    "wealth": _next_wealth,
                    "health": {"retired": MarkovTransition(_persistent_health)},
                }
            ),
            "retired": _retired_regime(),
        },
        states={"health": DiscreteGrid(category_class=_Health)},
    )

    assert model.pruned_variables["working"] == frozenset()
    assert "health" in model.user_regimes["working"].states


@pytest.mark.parametrize("declaration", ["regime-level", "model-level"])
def test_both_declarations_report_the_same_entry_law(*, declaration: str) -> None:
    """The source's entry law toward the target is the same either way."""
    model = (
        _regime_level_model() if declaration == "regime-level" else _model_level_model()
    )
    working = model.user_regimes["working"]

    assert _entry_targets(regime=working, state_name="health") == {"retired"}
