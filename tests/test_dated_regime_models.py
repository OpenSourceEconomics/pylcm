"""Models built from dated regime declarations: coverage, support and values."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    Choose,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.typing import DiscreteState, FloatND, Period, ScalarInt

AGES = AgeGrid(start=25, stop=75, step="10Y")


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _stay(health: DiscreteState) -> FloatND:
    return jnp.where(health == Health.good, 0.9, 0.7)


def _die(health: DiscreteState) -> FloatND:
    return 1 - _stay(health)


def _utility(*, wealth: FloatND, health: DiscreteState) -> FloatND:
    return wealth + health


def _regime(*, transition: Any, **kwargs: Any) -> Regime:
    return Regime(
        transition=transition,
        states={
            "health": DiscreteGrid(category_class=Health),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": lambda wealth: wealth,
        },
        functions={"utility": _utility},
        **kwargs,
    )


DEAD = Regime(transition=None, functions={"utility": lambda: 0.0})


def _dated_model(**overrides: Regime) -> Model:
    regimes = {
        "working": _regime(
            transition=ByAge.until(
                65,
                law={
                    "working": MarkovTransition(_stay),
                    "dead": MarkovTransition(_die),
                },
                then="retirement",
            )
        ),
        "retirement": _regime(transition=ByAge({AgeRange(start=65, stop=75): "dead"})),
        "dead": DEAD,
    }
    return Model(regimes=regimes | overrides, ages=AGES, regime_id_class=RegimeId)


def _legacy_model() -> Model:
    def stay(*, period: Period, health: DiscreteState) -> FloatND:
        return jnp.where(period < 3, _stay(health), 0.0)

    def die(*, period: Period, health: DiscreteState) -> FloatND:
        return jnp.where(period < 3, _die(health), 0.0)

    def retire(period: Period) -> FloatND:
        return jnp.where(period == 3, 1.0, 0.0)

    return Model(
        regimes={
            "working": _regime(
                transition={
                    "working": MarkovTransition(stay),
                    "dead": MarkovTransition(die),
                    "retirement": MarkovTransition(retire),
                },
                active=lambda age: age < 65,
            ),
            "retirement": _regime(
                transition=lambda: RegimeId.dead,
                active=lambda age: 65 <= age < 75,
            ),
            "dead": DEAD,
        },
        ages=AGES,
        regime_id_class=RegimeId,
    )


def test_dated_model_values_equal_the_hand_masked_legacy_model() -> None:
    """A schedule solves to exactly the values of its hand-written masked law."""
    params = {"discount_factor": 0.95}
    dated = _dated_model().solve(params=params, log_level="off").values
    legacy = _legacy_model().solve(params=params, log_level="off").values
    for period, by_regime in legacy.items():
        for regime, values in by_regime.items():
            np.testing.assert_array_equal(dated[period][regime], values)


def test_dated_model_covers_exactly_the_scheduled_ages() -> None:
    """Coverage is read off the schedules; terminal regimes cover every age."""
    reachability = _dated_model().reachability.solution
    assert reachability.active_regimes_by_period == (
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"retirement", "dead"}),
        frozenset({"dead"}),
    )


@pytest.mark.parametrize("period", [0, 3])
def test_dated_model_edges_are_the_declared_support_at_each_period(
    period: int,
) -> None:
    """The exit age reaches only `then`; earlier ages only the ordinary law."""
    expected = {0: ("dead", "working"), 3: ("retirement",)}[period]
    targets = _dated_model().reachability.solution.targets_by_period[period]
    assert targets["working"] == expected


@pytest.mark.parametrize(
    "override",
    [
        {"retirement": _regime(transition="dead", active=lambda age: age >= 65)},
        {"retirement": _regime(transition=lambda: RegimeId.dead)},
        {
            "retirement": _regime(
                transition=MarkovTransition(lambda: jnp.array([0.0, 0.0, 1.0]))
            )
        },
    ],
    ids=["active", "bare-callable", "targetless-vector"],
)
def test_dated_model_rejects_legacy_declarations(override: dict) -> None:
    """A dated model reads coverage and support only from the declarations."""
    with pytest.raises(ModelInitializationError):
        _dated_model(**override)


def test_dated_model_rejects_a_target_not_covered_at_the_next_age() -> None:
    """Declared support must be solved at the next age; nothing is dropped."""
    with pytest.raises(ModelInitializationError, match="retirement"):
        _dated_model(
            retirement=_regime(transition=ByAge({AgeRange(start=55, stop=65): "dead"}))
        )


def test_choose_routes_to_the_returned_regime_code() -> None:
    """A `Choose` selector's regime code picks the deterministic target."""

    def retire_if_healthy(health: DiscreteState) -> ScalarInt:
        return jnp.where(health == Health.good, RegimeId.retirement, RegimeId.dead)

    model = _dated_model(
        working=_regime(
            transition=ByAge(
                {
                    AgeRange(stop=55): "working",
                    55: Choose(retire_if_healthy, targets=("retirement", "dead")),
                }
            )
        )
    )
    targets = model.reachability.solution.targets_by_period[3]
    assert targets["working"] == ("dead", "retirement")
