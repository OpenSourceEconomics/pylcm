"""Column contract of `SimulationResult.to_dataframe`.

The published frame names its columns in a fixed order: the columns that
identify a row, then a collective regime's per-stakeholder values in the order
the regime declares its stakeholders, then states, actions, and everything else.
Every claim here is about which name a column carries and where it sits, so the
assertions are exact rather than toleranced.
"""

import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.simulation.result_dataframe import _reorder_columns
from lcm import CollectiveUtility, DiscreteGrid, Model, Regime, categorical
from lcm.exceptions import PyLCMError
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.collective_fixtures import (
    AGES,
    DISCOUNT_FACTOR,
    WAGE_GRID,
    CoupleRegimeId,
    Work,
    make_couple_initial_conditions,
)


@categorical(ordered=False)
class SoloRegimeId:
    """Regime ids of the single-decision-maker model in this module."""

    working: ScalarInt  # code 0
    retired: ScalarInt  # code 1


@categorical(ordered=False)
class MixedRegimeId:
    """Regime ids of the model pairing a collective household with a singleton."""

    couple: ScalarInt  # code 0
    solo: ScalarInt  # code 1
    couple_terminal: ScalarInt  # code 2
    solo_terminal: ScalarInt  # code 3


# Params of the collective models here, whose regimes are `couple` /
# `couple_terminal` like every collective model in `tests.collective_fixtures`.
COUPLE_PARAMS = {
    "couple": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
    "couple_terminal": {},
}

# Params of the single-decision-maker model, whose regimes are `working` /
# `retired`.
SOLO_PARAMS = {
    "working": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
    "retired": {},
}


def test_to_dataframe_orders_stakeholder_columns_by_declared_stakeholders():
    """Per-stakeholder value columns follow the regime's `stakeholders` order.

    The household declares `stakeholders=("wife", "husband")`, which fixes the
    order of its value function's trailing axis, so the frame carries
    `value_wife` before `value_husband` — the household's own order, not the
    alphabetical order of the two names. `own_stakeholder` precedes them: it
    says which role the row occupies, so it identifies the row rather than
    reporting a value of it.
    """
    model = _make_reverse_alphabetical_collective_model()
    result = model.simulate(
        params=COUPLE_PARAMS,
        initial_conditions=make_couple_initial_conditions(n_subjects=2),
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    df = result.to_dataframe()
    assert list(df.columns) == [
        "subject_id",
        "period",
        "regime_name",
        "own_stakeholder",
        "value_wife",
        "value_husband",
        "wage",
        "work",
        "age",
    ]


def test_to_dataframe_places_additional_target_after_states_and_actions():
    """An additional target sits in the trailing block, whatever it is called.

    A computed target is not a value column: `value_of_leisure` is one of the
    regime's own functions, requested through `additional_targets`, and it
    belongs after the states and actions like any other computed column. The
    model has a single decision maker and therefore no per-stakeholder values at
    all, so nothing may be placed in front of `wage`.
    """
    model = _make_solo_model_with_value_prefixed_target()
    result = model.simulate(
        params=SOLO_PARAMS,
        initial_conditions={
            "wage": jnp.asarray([8.0, 40.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, SoloRegimeId.working, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    df = result.to_dataframe(additional_targets=["value_of_leisure"])
    assert list(df.columns) == [
        "subject_id",
        "period",
        "regime_name",
        "value",
        "wage",
        "work",
        "age",
        "value_of_leisure",
    ]


def test_model_rejects_a_state_named_like_a_stakeholder_value_column():
    """`value_<stakeholder>` is reserved for the stakeholder's published value.

    A collective regime with stakeholders `("f", "m")` publishes `value_f` and
    `value_m`, so a state of either name is refused when the model is built. The
    name it collides with is in the message, because the alternative is a frame
    whose `value_f` column silently holds the wife's value instead of the state
    the model author declared.
    """
    with pytest.raises(PyLCMError, match="value_f"):
        _make_collective_model_with_colliding_state()


def test_model_rejects_a_singleton_state_named_like_another_regimes_value_column():
    """`value_<stakeholder>` is reserved across the model, not within one regime.

    The published frame is a single table over every regime, so a singleton
    regime's state named `value_f` claims the column a collective regime already
    publishes for its wife. Neither regime is wrong on its own, which is why the
    name is refused against the model's stakeholders rather than the declaring
    regime's.
    """
    with pytest.raises(PyLCMError, match="value_f"):
        _make_mixed_model_with_a_singleton_state_shadowing_a_value_column()


@pytest.mark.parametrize("missing_column", ["subject_id", "period", "regime_name"])
def test_reorder_columns_requires_the_row_identifying_columns(missing_column):
    """Every published frame identifies its rows, so those columns are required.

    `subject_id`, `period` and `regime_name` are present in every result and are
    named unconditionally, so a frame that lacks one is refused rather than
    silently published one column narrower. Only the scalar `value` column is
    optional: an all-collective result publishes `value_<stakeholder>` columns
    in its place.
    """
    frame = pd.DataFrame(
        {
            "subject_id": [0, 1],
            "period": [0, 0],
            "regime_name": ["couple", "couple"],
            "value": [1.0, 2.0],
            "wage": [8.0, 40.0],
        }
    ).drop(columns=[missing_column])

    with pytest.raises(KeyError, match=missing_column):
        _reorder_columns(df=frame, state_names=["wage"], action_names=[])


def _make_reverse_alphabetical_collective_model() -> Model:
    """Build a two-stakeholder model whose stakeholders are not in name order.

    The economics are those of `tests.collective_fixtures`: a household choosing
    between work and leisure over a two-point wage grid, where the wife values
    her own leisure and the husband values household consumption. The
    stakeholders are declared `("wife", "husband")`, so the declared order and
    the alphabetical order of the names disagree.
    """
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        states={"wage": WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"wife": _utility_wife, "husband": _utility_husband}
            )
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"wife": _utility_wife, "husband": _utility_husband}
            )
        },
    )
    return Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
    )


def _make_solo_model_with_value_prefixed_target() -> Model:
    """Build a single-decision-maker model with a `value_`-prefixed function.

    `value_of_leisure` is an ordinary intermediate of the utility DAG, so it can
    be requested as an additional target. Its name shares the `value_` prefix
    with a collective regime's published stakeholder columns, and this model
    declares no stakeholders at all.
    """
    working = Regime(
        transition=_next_solo_regime,
        active=lambda age: age < 1,
        states={"wage": WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _solo_utility, "value_of_leisure": _value_of_leisure},
    )
    retired = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _solo_utility, "value_of_leisure": _value_of_leisure},
    )
    return Model(
        regimes={"working": working, "retired": retired},
        ages=AGES,
        regime_id_class=SoloRegimeId,
    )


def _make_collective_model_with_colliding_state() -> Model:
    """Build a collective model whose state is named `value_f`.

    Everything else is the two-stakeholder household of
    `tests.collective_fixtures`: the state is the household's wage, carrying the
    name of the column the wife's published value claims.
    """
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        states={"value_f": WAGE_GRID},
        state_transitions={"value_f": _next_colliding_state},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _colliding_utility_f, "m": _colliding_utility_m}
            )
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value_f": WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _colliding_utility_f, "m": _colliding_utility_m}
            )
        },
    )
    return Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
    )


def _make_mixed_model_with_a_singleton_state_shadowing_a_value_column() -> Model:
    """Build a model pairing a collective household with an unrelated singleton.

    The couple is the two-stakeholder household of `tests.collective_fixtures`
    over its wage state; the singleton lives alongside it and calls its own state
    `value_f`. Each regime transitions only into its own terminal, so the two
    never meet — they share nothing but the published frame.
    """
    couple = Regime(
        transition={"couple_terminal": MarkovTransition(_probability_one)},
        active=lambda age: age < 1,
        states={"wage": WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _utility_wife, "m": _utility_husband}
            )
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _utility_wife, "m": _utility_husband}
            )
        },
    )
    solo = Regime(
        transition={"solo_terminal": MarkovTransition(_probability_one)},
        active=lambda age: age < 1,
        states={"value_f": WAGE_GRID},
        state_transitions={"value_f": _next_colliding_state},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _colliding_solo_utility},
    )
    solo_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value_f": WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _colliding_solo_utility},
    )
    return Model(
        regimes={
            "couple": couple,
            "solo": solo,
            "couple_terminal": couple_terminal,
            "solo_terminal": solo_terminal,
        },
        ages=AGES,
        regime_id_class=MixedRegimeId,
    )


def _probability_one(age: FloatND) -> FloatND:
    """Regime transition: the declared target is reached with probability one."""
    return jnp.ones_like(age, dtype=float)


def _colliding_solo_utility(
    *, value_f: ContinuousState, work: DiscreteAction
) -> FloatND:
    """The singleton's earnings from the state its author called `value_f`."""
    return value_f * work


def _utility_wife(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife: values her own leisure highly, also sees household consumption."""
    return wage * work + 30.0 * (1.0 - work)


def _utility_husband(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband: values household consumption, indifferent to leisure."""
    return 2.0 * (wage * work)


def _solo_utility(
    *, wage: ContinuousState, work: DiscreteAction, value_of_leisure: FloatND
) -> FloatND:
    """Earnings when working, the value of leisure otherwise."""
    return wage * work + value_of_leisure


def _value_of_leisure(work: DiscreteAction) -> FloatND:
    """Time not sold on the labour market, valued at a constant rate."""
    return 30.0 * (1.0 - work)


def _colliding_utility_f(*, value_f: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife: her earnings from the state the model author called `value_f`."""
    return value_f * work + 30.0 * (1.0 - work)


def _colliding_utility_m(*, value_f: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband: household consumption out of the same state."""
    return 2.0 * (value_f * work)


def _next_colliding_state(work: DiscreteAction) -> ContinuousState:
    """Same wage law as `_next_wage`, for the state named `value_f`."""
    return 40.0 * work + 8.0 * (1.0 - work)


def _next_wage(work: DiscreteAction) -> ContinuousState:
    """Deterministic wage law: working today yields the high wage tomorrow."""
    return 40.0 * work + 8.0 * (1.0 - work)


def _next_couple_regime() -> ScalarInt:
    """Regime transition: `couple` becomes `couple_terminal` with probability one."""
    return CoupleRegimeId.couple_terminal


def _next_solo_regime() -> ScalarInt:
    """Regime transition: `working` becomes `retired` with probability one."""
    return SoloRegimeId.retired
