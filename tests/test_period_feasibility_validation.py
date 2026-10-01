"""Regime-probability validation masks rows by the law period's own feasibility.

At age 1 the budget is `consumption <= wealth`, and the regime law is defined
only there (NaN elsewhere). At age 0 every action is feasible and the law is a
deterministic exit. Validating the age-1 law must use the age-1 constraint,
whether the age dependence enters through a specialized helper or through the
constraint itself, and whatever earlier root sets a representative age.
"""

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ByAge,
    ExecutionConfig,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.phased import Phased
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt, UserFunction


@categorical(ordered=False)
class ProbabilityId:
    working: ScalarInt
    left: ScalarInt
    right: ScalarInt


def _ten() -> FloatND:
    return jnp.asarray(10.0)


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _consumption(*, consumption: ContinuousState) -> FloatND:
    return consumption


def _valid_left(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    return jnp.where(consumption <= wealth, 0.5, jnp.nan)


def _valid_right(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    return 1.0 - _valid_left(wealth=wealth, consumption=consumption)


def _bad_left(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    # (wealth, consumption) = (1, 0) is feasible at every age.
    return jnp.where((wealth == 1.0) & (consumption == 0.0), 1.5, 0.5)


def _bad_right(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    return 1.0 - _bad_left(wealth=wealth, consumption=consumption)


def _limit_factory(age: float) -> UserFunction:
    def limit(*, wealth: ContinuousState) -> ContinuousState:
        return wealth if age >= 1.0 else jnp.ones_like(wealth)

    return limit


def _constraint_factory(age: float) -> UserFunction:
    def feasible(*, wealth: ContinuousState, consumption: ContinuousState) -> BoolND:
        limit = wealth if age >= 1.0 else jnp.ones_like(wealth)
        return consumption <= limit

    return feasible


def _age_signature(age: float) -> float:
    return age


def _uses_limit(
    *, consumption: ContinuousState, spending_limit: ContinuousState
) -> BoolND:
    return consumption <= spending_limit


def _feasibility_model(
    *,
    representation: Literal["helper", "constraint"],
    earlier_root: bool,
    enable_jit: bool = True,
    n_points: int = 2,
    bad_feasible: bool = False,
    law_phase: Literal["both", "solve", "simulate"] = "both",
) -> Model:
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=n_points)
    functions: dict[str, Any] = {"utility": _consumption}
    if representation == "helper":
        functions["spending_limit"] = AgeSpecializedFunction(
            build=_limit_factory, signature=_age_signature
        )
        constraints: dict[str, Any] = {"budget": _uses_limit}
    else:
        constraints = {
            "budget": AgeSpecializedFunction(
                build=_constraint_factory, signature=_age_signature
            )
        }
    left, right = (
        (_bad_left, _bad_right) if bad_feasible else (_valid_left, _valid_right)
    )
    checked_law = {
        "left": MarkovTransition(func=left),
        "right": MarkovTransition(func=right),
    }
    constant_law = {
        "left": MarkovTransition(func=_half),
        "right": MarkovTransition(func=_half),
    }
    late_law = (
        checked_law
        if law_phase == "both"
        else Phased(
            solve=checked_law if law_phase == "solve" else constant_law,
            simulate=checked_law if law_phase == "simulate" else constant_law,
        )
    )
    return Model(
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=ProbabilityId,
        initial_regimes={(0, 1): "working"} if earlier_root else {1: "working"},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        regimes={
            "working": Regime(
                regime_transitions=ByAge(cases={0: "left", 1: late_law}),
                states={"wealth": grid},
                actions={"consumption": grid},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions=functions,
                constraints=constraints,
            ),
            "left": Regime(regime_transitions=None, functions={"utility": _ten}),
            "right": Regime(regime_transitions=None, functions={"utility": _zero}),
        },
    )


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("earlier_root", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("n_points", [2, 3])
@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
def test_valid_late_law_solves_to_wealth_plus_five_halves(
    *,
    representation: Literal["helper", "constraint"],
    earlier_root: bool,
    enable_jit: bool,
    n_points: int,
    log_level: Literal["off", "warning", "debug"],
) -> None:
    """`V_1(w) = max_{c <= w} (c + 10 / 4) = w + 2.5`."""
    model = _feasibility_model(
        representation=representation,
        earlier_root=earlier_root,
        enable_jit=enable_jit,
        n_points=n_points,
    )
    values = model.solve(params={"discount_factor": 0.5}, log_level=log_level).values
    np.testing.assert_array_equal(
        np.asarray(values[1]["working"]), np.linspace(0.0, 1.0, n_points) + 2.5
    )


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("earlier_root", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
def test_bad_probability_at_a_feasible_row_raises(
    *,
    representation: Literal["helper", "constraint"],
    earlier_root: bool,
    enable_jit: bool,
    log_level: Literal["off", "warning", "debug"],
) -> None:
    """Feasibility masking never excuses an invalid law at a feasible row."""
    model = _feasibility_model(
        representation=representation,
        earlier_root=earlier_root,
        enable_jit=enable_jit,
        bad_feasible=True,
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="outside "):
        model.solve(params={"discount_factor": 0.5}, log_level=log_level)


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("earlier_root", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("law_phase", ["solve", "simulate"])
def test_each_phase_validates_with_its_own_period_feasibility(
    *,
    representation: Literal["helper", "constraint"],
    earlier_root: bool,
    enable_jit: bool,
    law_phase: Literal["solve", "simulate"],
) -> None:
    """A valid law confined to one phase solves to `[2.5, 3.5]`."""
    model = _feasibility_model(
        representation=representation,
        earlier_root=earlier_root,
        enable_jit=enable_jit,
        law_phase=law_phase,
    )
    values = model.solve(params={"discount_factor": 0.5}, log_level="off").values
    np.testing.assert_array_equal(np.asarray(values[1]["working"]), [2.5, 3.5])


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("earlier_root", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("law_phase", ["solve", "simulate"])
def test_each_phase_rejects_a_bad_feasible_row(
    *,
    representation: Literal["helper", "constraint"],
    earlier_root: bool,
    enable_jit: bool,
    law_phase: Literal["solve", "simulate"],
) -> None:
    """An invalid law confined to one phase still raises at a feasible row."""
    model = _feasibility_model(
        representation=representation,
        earlier_root=earlier_root,
        enable_jit=enable_jit,
        law_phase=law_phase,
        bad_feasible=True,
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="outside "):
        model.solve(params={"discount_factor": 0.5}, log_level="off")
