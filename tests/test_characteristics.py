"""Discrete variables whose categories are combinations of characteristics."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.simulation.initial_conditions import MISSING_CAT_CODE
from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Transition,
    categorical,
    characteristic_columns,
    find_code,
)
from lcm.exceptions import CategoricalDefinitionError
from lcm.regime import Regime
from lcm.typing import DiscreteAction, DiscreteState, FloatND, IntND, ScalarInt


@categorical(ordered=False)
class Status:
    employed: ScalarInt
    unemployed: ScalarInt
    di_1: ScalarInt
    di_2: ScalarInt


STATUS = characteristic_columns(
    category_class=Status,
    cases={
        "employed": {"has_job": True, "on_di": False, "di_years": 0},
        "unemployed": {"has_job": False, "on_di": False, "di_years": 0},
        "di_1": {"has_job": False, "on_di": True, "di_years": 1},
        "di_2": {"has_job": False, "on_di": True, "di_years": 2},
    },
)


def test_characteristic_columns_hold_each_characteristic_by_code():
    """`columns[name][code]` is the characteristic of the category with `code`."""
    np.testing.assert_array_equal(STATUS["di_years"], np.array([0, 0, 1, 2]))


@pytest.mark.parametrize(
    ("name", "dtype"),
    [("has_job", np.dtype(bool)), ("di_years", np.dtype(np.int32))],
)
def test_characteristic_columns_use_bool_and_int32_dtypes(*, name, dtype):
    """Bool characteristics stay boolean; integer ones share the code dtype."""
    assert STATUS[name].dtype == dtype


def test_characteristic_columns_list_characteristics_in_row_order():
    """The columns appear in the order the first row lists them."""
    assert tuple(STATUS) == ("has_job", "on_di", "di_years")


def test_characteristic_columns_are_read_only():
    """Neither the mapping nor its arrays can be changed after the check."""
    with pytest.raises(ValueError, match="read-only"):
        STATUS["di_years"][0] = 5


@categorical(ordered=False)
class _Two:
    a: ScalarInt
    b: ScalarInt


def _columns_missing_category():
    characteristic_columns(category_class=_Two, cases={"a": {"x": 1}})


def _columns_extra_category():
    characteristic_columns(
        category_class=_Two, cases={"a": {"x": 1}, "b": {"x": 2}, "c": {"x": 3}}
    )


def _columns_in_wrong_order():
    characteristic_columns(category_class=_Two, cases={"b": {"x": 1}, "a": {"x": 2}})


def _columns_with_differing_characteristics():
    characteristic_columns(category_class=_Two, cases={"a": {"x": 1}, "b": {"y": 2}})


def _columns_without_characteristics():
    characteristic_columns(category_class=_Two, cases={"a": {}, "b": {}})


def _columns_with_a_float():
    characteristic_columns(
        category_class=_Two,
        cases={"a": {"x": 0.5}, "b": {"x": 1.0}},  # ty: ignore[invalid-argument-type]
    )


def _columns_mixing_bool_and_int():
    characteristic_columns(category_class=_Two, cases={"a": {"x": True}, "b": {"x": 2}})


def _columns_outside_int32():
    characteristic_columns(
        category_class=_Two, cases={"a": {"x": 0}, "b": {"x": 2**31}}
    )


def _columns_with_identical_rows():
    characteristic_columns(
        category_class=_Two, cases={"a": {"x": 1, "y": 0}, "b": {"x": 1, "y": 0}}
    )


def _columns_for_a_plain_class():
    class _NotCategorical:
        a: int

    characteristic_columns(
        category_class=_NotCategorical,  # ty: ignore[invalid-argument-type]
        cases={"a": {"x": 1}},
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (_columns_missing_category, r"missing.*\['b'\]"),
        (_columns_extra_category, r"not categories.*\['c'\]"),
        (_columns_in_wrong_order, "declaration order"),
        (_columns_with_differing_characteristics, r"`b`.*\['y'\].*\['x'\]"),
        (_columns_without_characteristics, "at least one characteristic"),
        (_columns_with_a_float, r"0\.5"),
        (_columns_mixing_bool_and_int, r"`x`.*bool.*int"),
        (_columns_outside_int32, r"`b\.x`.*int32"),
        (_columns_with_identical_rows, r"`a`, `b`.*same characteristics"),
        (_columns_for_a_plain_class, "DataclassInstance"),
    ],
)
def test_characteristic_columns_reject_inconsistent_tables(*, build, message):
    """Every inconsistency between the class and its table raises."""
    with pytest.raises(CategoricalDefinitionError, match=message):
        build()


@pytest.mark.parametrize(
    ("characteristics", "code"),
    [
        ({"has_job": True, "on_di": False, "di_years": 0}, 0),
        ({"has_job": False, "on_di": False, "di_years": 0}, 1),
        ({"has_job": False, "on_di": True, "di_years": 2}, 3),
    ],
)
def test_find_code_returns_the_code_of_the_matching_category(*, characteristics, code):
    """A full row maps back to its category's code."""
    assert int(find_code(STATUS, **characteristics)) == code


def test_find_code_accepts_a_subset_that_tells_all_categories_apart():
    """A subset suffices when no two categories share it."""
    columns = characteristic_columns(
        category_class=_Two, cases={"a": {"x": 1, "y": 0}, "b": {"x": 2, "y": 0}}
    )

    assert int(find_code(columns, x=2)) == 1


def test_find_code_returns_the_missing_code_for_an_unlisted_combination():
    """A combination no category has maps to the missing-category code."""
    found = find_code(STATUS, has_job=True, on_di=True, di_years=1)

    assert int(found) == MISSING_CAT_CODE


def test_find_code_returns_int32():
    """Codes come back in the dtype of category codes."""
    assert find_code(STATUS, has_job=True, on_di=False, di_years=0).dtype == jnp.int32


def test_find_code_under_jit_matches_eager():
    """Characteristics computed inside `jit` map to the same codes."""

    @jax.jit
    def next_status(years):
        return find_code(STATUS, has_job=False, on_di=True, di_years=years + 1)

    assert int(next_status(jnp.int32(1))) == int(Status.di_2)


def test_find_code_under_vmap_maps_each_combination():
    """Batched characteristics map to one code each, misses included."""
    years = jnp.array([1, 2, 3], dtype=jnp.int32)

    found = jax.vmap(
        lambda y: find_code(STATUS, has_job=False, on_di=True, di_years=y)
    )(years)

    np.testing.assert_array_equal(found, np.array([2, 3, MISSING_CAT_CODE]))


def test_find_code_broadcasts_array_arguments():
    """Array characteristics broadcast against each other and against scalars."""
    found = find_code(STATUS, has_job=jnp.array([True, False]), on_di=False, di_years=0)

    np.testing.assert_array_equal(found, np.array([0, 1]))


def test_find_code_with_a_subset_shared_by_several_categories_raises():
    """A subset that leaves categories tied raises before reading any value."""
    with pytest.raises(CategoricalDefinitionError, match=r"codes \[0, 1\]"):
        find_code(STATUS, on_di=False, di_years=0)


def test_find_code_without_characteristics_raises():
    """`find_code` needs at least one characteristic."""
    with pytest.raises(CategoricalDefinitionError, match="at least one"):
        find_code(STATUS)


def test_find_code_with_an_unknown_characteristic_raises():
    """Characteristic names must be columns of the table."""
    with pytest.raises(CategoricalDefinitionError, match="wage"):
        find_code(STATUS, has_job=True, on_di=False, di_years=0, wage=1)


@categorical(ordered=False)
class _Apply:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    active: ScalarInt
    dead: ScalarInt


_HAS_JOB = np.array([True, False, False, False])
_ON_DI = np.array([False, False, True, True])
_DI_YEARS = np.array([0, 0, 1, 2], dtype=np.int32)


def status_has_job(status: DiscreteState) -> IntND:
    return jnp.asarray(STATUS["has_job"])[status]


def status_on_di(status: DiscreteState) -> IntND:
    return jnp.asarray(STATUS["on_di"])[status]


def status_di_years(status: DiscreteState) -> IntND:
    return jnp.asarray(STATUS["di_years"])[status]


def _hand_income(*, status: DiscreteState) -> FloatND:
    on_di = jnp.asarray(_ON_DI)[status]
    years = jnp.asarray(_DI_YEARS)[status]
    return 1.0 * jnp.asarray(_HAS_JOB)[status] + on_di * (0.6 - 0.1 * years)


def _pattern_income(
    *, status_has_job: IntND, status_on_di: IntND, status_di_years: IntND
) -> FloatND:
    return 1.0 * status_has_job + status_on_di * (0.6 - 0.1 * status_di_years)


def _next_hand_status(*, status: DiscreteState, apply: DiscreteAction) -> IntND:
    has_job = jnp.asarray(_HAS_JOB)[status]
    on_di = jnp.asarray(_ON_DI)[status]
    years = jnp.where(on_di, jnp.minimum(jnp.asarray(_DI_YEARS)[status] + 1, 2), 1)
    to_di = on_di | (apply == _Apply.yes)
    di_code = jnp.where(years == 1, Status.di_1, Status.di_2)
    return jnp.where(
        has_job, Status.employed, jnp.where(to_di, di_code, Status.unemployed)
    )


def _next_pattern_status(
    *,
    status_has_job: IntND,
    status_on_di: IntND,
    status_di_years: IntND,
    apply: DiscreteAction,
) -> IntND:
    years = jnp.where(status_on_di, jnp.minimum(status_di_years + 1, 2), 1)
    to_di = status_on_di | (apply == _Apply.yes)
    on_di = find_code(STATUS, has_job=False, on_di=True, di_years=years)
    return jnp.where(
        status_has_job, Status.employed, jnp.where(to_di, on_di, Status.unemployed)
    )


def _next_wealth(*, wealth: FloatND, consumption: FloatND, income: FloatND):
    return wealth - consumption + income


def _utility(
    *, consumption: FloatND, wealth: FloatND, apply: DiscreteAction
) -> FloatND:
    return jnp.log(consumption) + 0.1 * wealth - 0.1 * (apply == _Apply.yes)


def _next_regime(*, age: ScalarInt) -> ScalarInt:
    return jnp.where(age >= 2, _RegimeId.dead, _RegimeId.active)


def _labour_model(*, use_pattern: bool) -> Model:
    if use_pattern:
        functions = {
            "utility": _utility,
            "income": _pattern_income,
            "status_has_job": status_has_job,
            "status_on_di": status_on_di,
            "status_di_years": status_di_years,
        }
        next_status = _next_pattern_status
    else:
        functions = {"utility": _utility, "income": _hand_income}
        next_status = _next_hand_status
    active = Regime(
        actions={
            "apply": DiscreteGrid(category_class=_Apply),
            "consumption": LinSpacedGrid(start=0.5, stop=2.0, n_points=4),
        },
        states={
            "status": DiscreteGrid(category_class=Status),
            "wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=5),
        },
        state_transitions={"status": next_status, "wealth": _next_wealth},
        functions=functions,
    )
    dead = Regime(functions={"utility": lambda: jnp.array(0.0)})
    return Model(
        regimes={"active": active, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "active"},
        edges={
            "active": Transition(
                targets={"active": (0, 1), "dead": (0, 1, 2)},
                law=DeterministicTransition(func=_next_regime),
            )
        },
    )


_PARAMS = {"discount_factor": 0.95}
_INITIAL_CONDITIONS = {
    "age": jnp.zeros(4),
    "status": jnp.arange(4, dtype=jnp.int32),
    "wealth": jnp.array([1.0, 2.0, 3.0, 4.0]),
    "regime_id": jnp.full(4, _RegimeId.active),
}


def _solve(*, use_pattern: bool) -> np.ndarray:
    values = (
        _labour_model(use_pattern=use_pattern)
        .solve(params=_PARAMS, log_level="off")
        .values
    )
    return np.stack([values[period]["active"] for period in (0, 1, 2)])


def _simulate(*, use_pattern: bool) -> pd.DataFrame:
    return (
        _labour_model(use_pattern=use_pattern)
        .simulate(
            params=_PARAMS, initial_conditions=_INITIAL_CONDITIONS, log_level="off"
        )
        .to_dataframe()
    )


def test_characteristic_pattern_solves_to_the_hand_table_values():
    """Published characteristics and `find_code` reproduce the hand-table V."""
    np.testing.assert_array_equal(_solve(use_pattern=True), _solve(use_pattern=False))


def test_characteristic_pattern_simulates_the_hand_table_paths():
    """Published characteristics and `find_code` reproduce the simulated paths."""
    pd.testing.assert_frame_equal(
        _simulate(use_pattern=True), _simulate(use_pattern=False)
    )


def test_published_characteristics_appear_as_additional_targets():
    """A published characteristic is an output column like any other function."""
    frame = (
        _labour_model(use_pattern=True)
        .simulate(
            params=_PARAMS, initial_conditions=_INITIAL_CONDITIONS, log_level="off"
        )
        .to_dataframe(additional_targets=["status_di_years"])
        .query("regime_name == 'active' and period == 0")
        .sort_values("subject_id")
    )

    np.testing.assert_array_equal(frame["status_di_years"], np.array([0, 0, 1, 2]))
