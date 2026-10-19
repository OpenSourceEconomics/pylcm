"""User functions read `<name>_code`, the position of a discrete variable's node.

The wage model indexes a parameter Series by the code of its experience state:
`wage_premium[experience_code]`. The Series is indexed by the grid's values for a
value grid and by the category labels for a category class, written in a shuffled
order, so only an alignment to grid order gives the reference wages. The reference
solves the model by hand, by backward induction over the three nodes, and reads the
premium by label, so it shares no indexing path with pylcm.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    Model,
    Transition,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import (
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserFunction,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION

_BASE_WAGE = 2.0
_DISUTILITY = 2.5
_DISCOUNT = 0.9
_LAST_WORKING_AGE = 2
_PREMIUM = {0: 0.0, 5: 0.1, 10: 0.25}
_LABEL_PREMIUM = {"low": 0.0, "medium": 0.1, "high": 0.25}


@categorical(ordered=False)
class _RegimeId:
    working_life: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _Work:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=True)
class _Level:
    low: ScalarInt
    medium: ScalarInt
    high: ScalarInt


def _wage(
    *,
    experience: DiscreteState,
    experience_code: DiscreteState,
    wage_premium: FloatND,
    base_wage: float,
) -> FloatND:
    return base_wage * (1 + 0.02 * experience + wage_premium[experience_code])


def _label_wage(
    *, experience_code: DiscreteState, wage_premium: FloatND, base_wage: float
) -> FloatND:
    return base_wage * (1 + wage_premium[experience_code])


def _utility(
    *, wage: FloatND, work: DiscreteAction, disutility_of_work: float
) -> FloatND:
    return (wage - disutility_of_work) * work


def _next_experience(
    *, experience: DiscreteState, work: DiscreteAction
) -> DiscreteState:
    return jnp.minimum(experience + 5 * work, 10)


def _next_level(
    *, experience_code: DiscreteState, work: DiscreteAction
) -> DiscreteState:
    return jnp.minimum(experience_code + work, _Level.high)


def _next_regime(*, age: float) -> ScalarInt:
    return jnp.where(age >= _LAST_WORKING_AGE, _RegimeId.dead, _RegimeId.working_life)


def _model(*, experience: DiscreteGrid, wage, next_experience) -> Model:
    ages = AgeGrid(start=0, inclusive_stop=_LAST_WORKING_AGE + 1, step="Y")
    working_life = Regime(
        actions={"work": DiscreteGrid(category_class=_Work)},
        states={"experience": experience},
        state_transitions={"experience": next_experience},
        functions={"utility": _utility, "wage": wage},
    )
    return Model(
        regimes={
            "working_life": working_life,
            "dead": Regime(functions={"utility": lambda: 0.0}),
        },
        ages=ages,
        regime_id_class=_RegimeId,
        initial_nodes={0: "working_life"},
        edges={
            "working_life": Transition(
                targets={
                    "dead": tuple(ages.exact_values[:-1]),
                    "working_life": tuple(ages.exact_values[:-2]),
                },
                law=DeterministicTransition(func=_next_regime),
            )
        },
    )


def _params(premium: pd.Series) -> UserParams:
    return {
        "discount_factor": _DISCOUNT,
        "working_life": {
            "utility": {"disutility_of_work": _DISUTILITY},
            "wage": {"wage_premium": premium, "base_wage": _BASE_WAGE},
        },
    }


def _value_case() -> tuple[Model, UserParams, tuple[float, ...]]:
    model = _model(
        experience=DiscreteGrid(values=jnp.array([0, 5, 10])),
        wage=_wage,
        next_experience=_next_experience,
    )
    premium = pd.Series(
        [_PREMIUM[10], _PREMIUM[0], _PREMIUM[5]],
        index=pd.Index([10, 0, 5], name="experience"),
    )
    wages = tuple(
        _BASE_WAGE * (1 + 0.02 * value + _PREMIUM[value]) for value in (0, 5, 10)
    )
    return model, _params(premium), wages


def _label_case() -> tuple[Model, UserParams, tuple[float, ...]]:
    model = _model(
        experience=DiscreteGrid(category_class=_Level),
        wage=_label_wage,
        next_experience=_next_level,
    )
    premium = pd.Series(
        [_LABEL_PREMIUM["high"], _LABEL_PREMIUM["low"], _LABEL_PREMIUM["medium"]],
        index=pd.Index(["high", "low", "medium"], name="experience"),
    )
    wages = tuple(
        _BASE_WAGE * (1 + _LABEL_PREMIUM[label]) for label in ("low", "medium", "high")
    )
    return model, _params(premium), wages


_CASES = {"value-grid": _value_case, "category-class": _label_case}


def _reference_solution(
    wages: tuple[float, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Solve the three-node model by backward induction; return V and policy."""
    values: dict[int, np.ndarray] = {}
    policies: dict[int, np.ndarray] = {}
    continuation = np.zeros(3)
    for period in range(_LAST_WORKING_AGE, -1, -1):
        is_last = period == _LAST_WORKING_AGE
        choices = np.array(
            [
                [
                    (wages[code] - _DISUTILITY) * work
                    + (
                        0.0
                        if is_last
                        else _DISCOUNT * continuation[min(code + work, 2)]
                    )
                    for work in (0, 1)
                ]
                for code in range(3)
            ]
        )
        values[period] = choices.max(axis=1)
        policies[period] = choices.argmax(axis=1)
        continuation = values[period]
    return values, policies


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("period", range(_LAST_WORKING_AGE + 1))
def test_code_indexed_series_solves_to_the_reference_value_function(*, case, period):
    """`wage_premium[experience_code]` reads the premium of each node's label."""
    model, params, wages = _CASES[case]()
    got = model.solve(params=params, log_level="off").values[period]["working_life"]
    aaae(np.asarray(got), _reference_solution(wages)[0][period], DECIMAL_PRECISION)


@pytest.mark.parametrize("case", list(_CASES))
def test_code_indexed_series_simulates_the_reference_choices(case):
    """Every subject works exactly where the reference policy works."""
    model, params, wages = _CASES[case]()
    frame = (
        model.simulate(
            params=params,
            initial_conditions={
                # A value grid starts from its values, a category class from codes.
                "experience": jnp.array(
                    [0, 5, 10] if case == "value-grid" else [0, 1, 2]
                ),
                "age": jnp.zeros(3),
                "regime_id": jnp.array([_RegimeId.working_life] * 3),
            },
            seed=0,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .query("regime_name == 'working_life'")
        .sort_values(["subject_id", "period"])
    )
    policies = _reference_solution(wages)[1]
    expected = [
        int(policies[period][int(code)])
        for period, code in zip(frame["period"], frame["experience"], strict=True)
    ]
    assert frame["work"].astype(int).tolist() == expected


def _wage_by_value(
    *, experience: DiscreteState, wage_premium: FloatND, base_wage: float
) -> FloatND:
    return base_wage * (1 + wage_premium[experience])


def test_indexing_a_param_by_the_value_of_a_value_grid_is_refused():
    """`table[experience]` on a value grid names `experience_code` as the fix."""
    with pytest.raises(RegimeInitializationError, match="experience_code"):
        _model(
            experience=DiscreteGrid(values=jnp.array([0, 5, 10])),
            wage=_wage_by_value,
            next_experience=_next_experience,
        )


def _experience_code(*, experience: DiscreteState) -> DiscreteState:
    return experience


def _stay_experience_code(*, experience_code: DiscreteState) -> DiscreteState:
    return experience_code


def _regime_with_a_code_named_function() -> Regime:
    return Regime(
        actions={"work": DiscreteGrid(category_class=_Work)},
        states={"experience": DiscreteGrid(values=jnp.array([0, 5, 10]))},
        state_transitions={"experience": _next_experience},
        functions={
            "utility": _utility,
            "wage": _wage,
            "experience_code": _experience_code,
        },
    )


def _regime_with_a_code_named_state() -> Regime:
    return Regime(
        actions={"work": DiscreteGrid(category_class=_Work)},
        states={
            "experience": DiscreteGrid(values=jnp.array([0, 5, 10])),
            "experience_code": DiscreteGrid(values=jnp.array([0, 1])),
        },
        state_transitions={
            "experience": _next_experience,
            "experience_code": _stay_experience_code,
        },
        functions={"utility": _utility, "wage": _wage},
    )


@pytest.mark.parametrize(
    "make_regime", [_regime_with_a_code_named_function, _regime_with_a_code_named_state]
)
def test_a_variable_named_like_the_code_of_a_discrete_variable_is_refused(
    make_regime,
):
    """`experience_code` collides with the code of the discrete `experience`."""
    with pytest.raises(RegimeInitializationError, match="experience_code"):
        Model(
            regimes={
                "working_life": make_regime(),
                "dead": Regime(functions={"utility": lambda: 0.0}),
            },
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=_RegimeId,
            initial_nodes={0: "working_life"},
            edges={
                "working_life": Transition(
                    targets={"dead": (0, 1), "working_life": (0,)},
                    law=DeterministicTransition(func=_next_regime),
                )
            },
        )


def _no_work_at_the_top_by_code(
    *, experience_code: DiscreteState, work: DiscreteAction
) -> FloatND:
    return jnp.logical_not((experience_code == 2) & (work == _Work.yes))


def _no_work_at_the_top_by_value(
    *, experience: DiscreteState, work: DiscreteAction
) -> FloatND:
    return jnp.logical_not((experience == 10) & (work == _Work.yes))


def _next_regime_by_code(*, age: float, experience_code: DiscreteState) -> ScalarInt:
    return jnp.where(
        (age >= _LAST_WORKING_AGE) | (experience_code < 0),
        _RegimeId.dead,
        _RegimeId.working_life,
    )


def _constrained_values(
    *, constraint: UserFunction, regime_law: UserFunction
) -> Mapping[int, Mapping[str, FloatND]]:
    model, params, _ = _value_case()
    regime = model.user_regimes["working_life"]
    regime = regime.replace(constraints={"no_work_at_the_top": constraint})
    ages = AgeGrid(start=0, inclusive_stop=_LAST_WORKING_AGE + 1, step="Y")
    constrained = Model(
        regimes={"working_life": regime, "dead": model.user_regimes["dead"]},
        ages=ages,
        regime_id_class=_RegimeId,
        initial_nodes={0: "working_life"},
        edges={
            "working_life": Transition(
                targets={
                    "dead": tuple(ages.exact_values[:-1]),
                    "working_life": tuple(ages.exact_values[:-2]),
                },
                law=DeterministicTransition(func=regime_law),
            )
        },
    )
    return constrained.solve(params=params, log_level="off").values


@pytest.mark.parametrize("period", range(_LAST_WORKING_AGE + 1))
def test_code_reads_in_a_constraint_and_a_regime_law_match_value_reads(period):
    """A constraint and a regime law reading `experience_code` act as value reads."""
    by_code = _constrained_values(
        constraint=_no_work_at_the_top_by_code, regime_law=_next_regime_by_code
    )
    by_value = _constrained_values(
        constraint=_no_work_at_the_top_by_value, regime_law=_next_regime
    )
    np.testing.assert_array_equal(
        np.asarray(by_code[period]["working_life"]),
        np.asarray(by_value[period]["working_life"]),
    )
