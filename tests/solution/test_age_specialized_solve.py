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
from lcm.typing import FloatND, ScalarInt, UserFunction


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


def _utility_of_consumption(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _feasible_consumption(consumption: float, wealth: float) -> bool:
    return consumption <= wealth


def _next_wealth_spend(wealth: float, consumption: float) -> float:
    return wealth - consumption + 1.0


def _cap_of_age(age: float) -> Callable[..., bool]:
    """Return the age's concrete feasibility constraint (an `AgeSpecializedFunction`).

    `wealth_cap` is slack for every grid cell (age >= 60, wealth <= 100), so the
    feasible set is unchanged and the model solves; the point is only that a
    specialized *constraint* node exists in the target pool.
    """

    def wealth_cap(consumption: float, wealth: float) -> bool:
        return consumption <= wealth + age

    return wealth_cap


def _make_specialized_constraint_model(wealth_cap: UserFunction) -> Model:
    working_life = UserRegime(
        transition=_next_regime,
        active=lambda age: age < 75,
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=8)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        state_transitions={"wealth": _next_wealth_spend},
        constraints={
            "feasible_consumption": _feasible_consumption,
            "wealth_cap": wealth_cap,
        },
        functions={"utility": _utility_of_consumption},
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


def _simulate_specialized_constraint_model():
    model = _make_specialized_constraint_model(
        AgeSpecializedFunction(build=_cap_of_age, signature=lambda age: age)
    )
    return model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(3, 25.0),
            "wealth": jnp.array([10.0, 50.0, 100.0]),
            "regime_id": jnp.full(3, RegimeId.working_life),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )


def test_additional_target_of_age_specialized_constraint_is_rejected():
    """Round-11 F3: a specialized *constraint* requested as a target is rejected.

    A constraint carries its own namespace: `_process_regime_core` excludes
    constraint names from `functions`, so an `AgeSpecializedFunction` constraint
    was omitted from `age_specialized_function_names` and escaped the guard, even
    though the additional-target pool re-merges constraints and advertises them as
    targets. Requesting the specialized constraint by name must raise, not reach
    target construction as an unresolved representative-age marker.
    """
    result = _simulate_specialized_constraint_model()
    with pytest.raises(InvalidAdditionalTargetsError, match="policy-specialized"):
        result.to_dataframe(additional_targets=["wealth_cap"])


def test_additional_targets_all_rejects_age_specialized_constraint():
    """Round-11 F3: `additional_targets='all'` rejects a specialized constraint.

    `'all'` expands to every advertised target, which includes the specialized
    constraint; the guard must reject the batch rather than silently compute it at
    the wrong age's policy closure.
    """
    result = _simulate_specialized_constraint_model()
    with pytest.raises(InvalidAdditionalTargetsError, match="policy-specialized"):
        result.to_dataframe(additional_targets="all")
