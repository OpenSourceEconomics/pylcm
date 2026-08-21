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
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidAdditionalTargetsError, InvalidInitialConditionsError
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction
from lcm.typing import FloatND, ScalarInt, UserFunction
from tests.conftest import DECIMAL_PRECISION


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
    """A specialized *constraint* requested as a target is rejected.

    A constraint carries its own namespace: `_process_regime_core` excludes
    constraint names from `functions`, so the guard that detects age-specialized
    targets must scan `constraints` too, not just `functions` — even though the
    additional-target pool re-merges constraints and advertises them as targets.
    Requesting the specialized constraint by name must raise, not reach target
    construction as an unresolved representative-age marker.
    """
    result = _simulate_specialized_constraint_model()
    with pytest.raises(InvalidAdditionalTargetsError, match="policy-specialized"):
        result.to_dataframe(additional_targets=["wealth_cap"])


def test_additional_targets_all_rejects_age_specialized_constraint():
    """`additional_targets='all'` rejects a specialized constraint.

    `'all'` expands to every advertised target, which includes the specialized
    constraint; the guard must reject the batch rather than silently compute it at
    the wrong age's policy closure.
    """
    result = _simulate_specialized_constraint_model()
    with pytest.raises(InvalidAdditionalTargetsError, match="policy-specialized"):
        result.to_dataframe(additional_targets="all")


def test_initial_conditions_feasibility_check_rejects_age_specialized_constraint():
    """Feasibility validation rejects a specialized constraint for subjects starting
    away from the regime's representative age.

    `_check_regime_feasibility` builds its feasibility function from the published
    `regime.simulation.constraints`, which hold an `AgeSpecializedFunction` resolved
    at the regime's representative (first active) age only. Checking a subject who
    starts at a later age against that closure would silently apply the wrong age's
    policy, so it must raise instead.
    """
    model = _make_specialized_constraint_model(
        AgeSpecializedFunction(build=_cap_of_age, signature=lambda age: age)
    )
    with pytest.raises(InvalidInitialConditionsError, match="policy-specialized"):
        model.simulate(
            params={"discount_factor": 0.95},
            initial_conditions={
                "age": jnp.array([25.0, 35.0, 45.0]),
                "wealth": jnp.array([10.0, 50.0, 100.0]),
                "regime_id": jnp.full(3, RegimeId.working_life),
            },
            period_to_regime_to_V_arr=None,
            log_level="debug",
        )


def _cap_binding_at_age(age: float) -> Callable[..., bool]:
    """Return the age's concrete consumption cap, chosen to bind on the grid.

    The cap rises from the lowest action node at the first active age to the
    highest at the last, so the feasible set genuinely differs period by period
    rather than only being labelled per period.
    """
    cap = 1.0 + (age - 25.0) * 0.225

    def wealth_cap(consumption: float) -> bool:
        return consumption <= cap

    return wealth_cap


def _solve_with_cap(wealth_cap: UserFunction):
    model = _make_specialized_constraint_model(wealth_cap)
    return model.solve(params={"discount_factor": 0.95}, log_level="off")


def _solve_specialized():
    return _solve_with_cap(
        AgeSpecializedFunction(build=_cap_binding_at_age, signature=lambda age: age)
    )


def _last_active_period(solution) -> int:
    """The last period in which the working-life regime is solved."""
    return max(
        period
        for period, regime_to_V in solution.items()
        if "working_life" in regime_to_V
    )


def _age_of_period(period: int) -> float:
    return 25.0 + 10.0 * period


def _last_period_value(solution):
    period = _last_active_period(solution)
    return solution[period]["working_life"]


def test_the_final_period_is_solved_under_its_own_ages_constraint() -> None:
    """Each period is restricted by the constraint built for that period's age.

    The regime's last active period is the one period whose value depends on no
    other period's rule — its continuation is terminal — so it is where a
    period can be attributed to an age at all. Solving it under a uniform cap
    fixed at that same age must therefore reproduce the specialized solve
    exactly there. Both a specialization frozen at the first active age and one
    that permuted which period got which closure would put a different cap on
    this period, and neither raises: the model still solves, and the only trace
    is a value function restricted by the wrong rule.

    The attribution reaches that one period only. An earlier period's value
    also depends on every later period's cap, so no uniform-cap solve is a
    reference for it, and a permutation confined to the earlier periods would
    pass here.
    """
    specialized = _solve_specialized()
    last_age = _age_of_period(_last_active_period(specialized))
    uniform_at_that_age = _solve_with_cap(_cap_binding_at_age(last_age))

    aaae(
        _last_period_value(specialized),
        _last_period_value(uniform_at_that_age),
        decimal=DECIMAL_PRECISION,
    )


def test_another_ages_constraint_would_have_given_that_period_a_different_value() -> (
    None
):
    """The reference the previous test matches is not one every age produces.

    Without this, an equality against a uniform-cap solve would hold for every
    age whose cap happened to yield the same value function, and the
    attribution it appears to make would be empty. The first active age's cap
    admits only the lowest action node, so it is the reference that must *not*
    match.
    """
    specialized = _solve_specialized()
    uniform_at_the_first_age = _solve_with_cap(_cap_binding_at_age(25.0))

    gap = jnp.max(
        jnp.abs(
            _last_period_value(specialized)
            - _last_period_value(uniform_at_the_first_age)
        )
    )

    assert float(gap) > 0.0
