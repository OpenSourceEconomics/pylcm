"""Solving a model with an `AgeSpecializedFunction` reflects the per-age closure.

The driving end-to-end contract for period specialization: a function wrapped in
`AgeSpecializedFunction` is resolved to its concrete per-age closure at build
time. Binding
a bonus to `age` at build time must therefore produce the exact same value function
as the age-invariant baseline that reads pylcm's runtime `age` argument directly.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidAdditionalTargetsError
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
    ScalarInt,
    UserFunction,
)


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    dead: ScalarInt


def _next_regime(period: int) -> ScalarInt:
    # Transition to the terminal regime at the last working-life period (age 65).
    return jnp.where(period >= 4, RegimeId.dead, RegimeId.working_life)


def _make_model(policy_bonus: UserFunction) -> Model:
    working_life = UserRegime(
        transition=_next_regime,
        active=lambda age: age < 75,
        states={
            "health": DiscreteGrid(Health),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=6),
        },
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": lambda wealth: wealth,
        },
        functions={
            "utility": lambda wealth, health, policy_bonus: (
                wealth + health + policy_bonus
            ),
            "policy_bonus": policy_bonus,
        },
    )
    dead = UserRegime(
        transition=None,
        active=lambda age: age >= 75,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working_life, "dead": dead},
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=RegimeId,
    )


def _bonus_of_age(age: float) -> Callable[[], float]:
    """Return the age's concrete policy-bonus function (an additive constant)."""

    def policy_bonus():
        return float(age)

    return policy_bonus


def _make_next_state_model(policy_bonus: UserFunction) -> Model:
    """A model whose law of motion `next_wealth = wealth + policy_bonus` reads a fn."""
    working_life = UserRegime(
        transition=_next_regime,
        active=lambda age: age < 75,
        states={"wealth": LinSpacedGrid(start=0, stop=2000, n_points=11)},
        state_transitions={
            "wealth": lambda wealth, policy_bonus: wealth + policy_bonus,
        },
        functions={"utility": lambda wealth: wealth, "policy_bonus": policy_bonus},
    )
    dead = UserRegime(
        transition=None,
        active=lambda age: age >= 75,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working_life, "dead": dead},
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=RegimeId,
    )


def test_age_specialized_next_state_matches_runtime_age_baseline():
    """A specialized function feeding `next_wealth` matches the runtime-`age` baseline.

    `next_wealth = wealth + policy_bonus` and `policy_bonus == age` both ways, so the
    simulated wealth trajectory is identical whether the bonus is bound per age at
    build time (through the periodized next-state) or read from pylcm's runtime `age`.
    """
    params = {"discount_factor": 0.95}
    initial_conditions = {
        "age": jnp.full(3, 25.0),
        "wealth": jnp.array([0.0, 100.0, 500.0]),
        "regime_id": jnp.full(3, RegimeId.working_life),
    }

    def _simulated_wealth(policy_bonus: UserFunction) -> np.ndarray:
        result = _make_next_state_model(policy_bonus).simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="debug",
        )
        return result.to_dataframe()["wealth"].to_numpy()

    baseline = _simulated_wealth(lambda age: age)
    specialized = _simulated_wealth(
        AgeSpecializedFunction(build=_bonus_of_age, signature=lambda age: age)
    )
    np.testing.assert_allclose(specialized, baseline, atol=1e-6)


def test_age_specialized_bonus_matches_runtime_age_baseline():
    """An `AgeSpecializedFunction` bonus bound to `age` equals the runtime baseline.

    Both regimes give `policy_bonus == age`: the baseline reads pylcm's runtime
    `age`; the specialized model binds it per period at build time. The two value
    functions must agree at every period and regime.
    """
    params = {"discount_factor": 0.95}
    baseline = _make_model(lambda age: age).solve(params=params, log_level="debug")
    specialized = _make_model(
        AgeSpecializedFunction(build=_bonus_of_age, signature=lambda age: age)
    ).solve(params=params, log_level="debug")

    assert baseline.keys() == specialized.keys()
    for period in baseline:
        for regime_name, V_arr in baseline[period].items():
            np.testing.assert_allclose(
                np.asarray(specialized[period][regime_name]),
                np.asarray(V_arr),
                atol=1e-6,
            )


def test_additional_target_depending_on_age_specialized_function_is_rejected():
    """`to_dataframe(additional_targets=[...])` rejects policy-specialized targets.

    Published simulation functions hold one representative-age closure, so a
    period-specific additional target that reads an `AgeSpecializedFunction` function
    (directly or through the DAG) would silently be computed under the wrong
    age's policy. It must raise instead.
    """
    model = _make_next_state_model(
        AgeSpecializedFunction(build=_bonus_of_age, signature=lambda age: age)
    )
    result = model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(3, 25.0),
            "wealth": jnp.array([0.0, 100.0, 500.0]),
            "regime_id": jnp.full(3, RegimeId.working_life),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )

    with pytest.raises(InvalidAdditionalTargetsError, match="policy-specialized"):
        result.to_dataframe(additional_targets=["policy_bonus"])


@categorical(ordered=True)
class _Capital:
    c0: ScalarInt
    c1: ScalarInt
    c2: ScalarInt


@categorical(ordered=True)
class _Invest:
    no: ScalarInt
    yes: ScalarInt


def _f1_next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 2, RegimeId.dead, RegimeId.working_life)


def _f1_next_capital(
    capital: DiscreteState, invest: DiscreteAction, boost: IntND
) -> DiscreteState:
    return jnp.clip(capital + invest * boost, 0, 2)


def _f1_utility(capital: DiscreteState, invest: DiscreteAction) -> FloatND:
    return capital * 1.0 - invest * 0.5


def _f1_boost_runtime(age: float) -> IntND:
    return jnp.where(age < 30.0, 0, 1)


def _f1_boost_of_age(age: float):
    value = 0 if age < 30.0 else 1

    def boost() -> IntND:
        return jnp.asarray(value, dtype=jnp.int32)

    return boost


def _f1_make_model(boost: UserFunction) -> Model:
    working = UserRegime(
        transition=_f1_next_regime,
        active=lambda age: age < 55,
        states={"capital": DiscreteGrid(_Capital)},
        actions={"invest": DiscreteGrid(_Invest)},
        state_transitions={"capital": _f1_next_capital},
        functions={"utility": _f1_utility, "boost": boost},
    )
    dead = UserRegime(
        transition=None,
        active=lambda age: age >= 55,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=AgeGrid(start=25, stop=55, step="10Y"),
        regime_id_class=RegimeId,
    )


def test_simulation_continuation_resolves_age_specialized_helper_per_age():
    """Round-11 F1: the simulation continuation resolves a solve-side
    `AgeSpecializedFunction` helper at each period's age, not frozen at the regime's
    representative (first active) age.

    `next_capital = clip(capital + invest*boost, 0, 2)` reads `boost` (0 at age < 30,
    1 at age >= 30). Investing at age 35 raises next capital only under the correct
    age-35 boost, so the age-35 continuation (V at age 45) makes investing worth its
    cost -> invest. A pool frozen at the first active age (25, boost=0) would make
    investing useless -> a reversed no-invest argmax. The specialized model must match
    the runtime-`age` baseline, which never reads the frozen pool.
    """

    def _invest_at_age35(boost: UserFunction) -> str:
        result = _f1_make_model(boost).simulate(
            params={"discount_factor": 0.95},
            initial_conditions={
                "age": jnp.array([35.0]),
                "capital": jnp.array([0]),
                "regime_id": jnp.array([RegimeId.working_life]),
            },
            period_to_regime_to_V_arr=None,
            log_level="debug",
        )
        return str(result.to_dataframe()["invest"].to_numpy()[0])

    baseline = _invest_at_age35(_f1_boost_runtime)
    specialized = _invest_at_age35(
        AgeSpecializedFunction(build=_f1_boost_of_age, signature=lambda age: age < 30.0)
    )
    assert baseline == "yes"
    assert specialized == baseline
