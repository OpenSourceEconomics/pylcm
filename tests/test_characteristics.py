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
    category_with,
    characteristics,
)
from lcm.exceptions import CategoricalDefinitionError
from lcm.regime import Regime
from lcm.typing import DiscreteAction, DiscreteState, FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED


@categorical(ordered=False)
class Status:
    employed: ScalarInt
    unemployed: ScalarInt
    di_1: ScalarInt
    di_2: ScalarInt


STATUS = characteristics(
    category_class=Status,
    by_category={
        "employed": {"has_job": True, "on_di": False, "di_years": 0},
        "unemployed": {"has_job": False, "on_di": False, "di_years": 0},
        "di_1": {"has_job": False, "on_di": True, "di_years": 1},
        "di_2": {"has_job": False, "on_di": True, "di_years": 2},
    },
)


def test_characteristics_hold_each_characteristic_by_code():
    """`columns[name][code]` is the characteristic of the category with `code`."""
    np.testing.assert_array_equal(STATUS["di_years"], np.array([0, 0, 1, 2]))


@pytest.mark.parametrize(
    ("name", "dtype"),
    [("has_job", np.dtype(bool)), ("di_years", np.dtype(np.int32))],
)
def test_characteristics_use_bool_and_int32_dtypes(*, name, dtype):
    """Bool characteristics stay boolean; integer ones share the code dtype."""
    assert STATUS[name].dtype == dtype


def test_characteristics_list_characteristics_in_row_order():
    """The columns appear in the order the first row lists them."""
    assert tuple(STATUS) == ("has_job", "on_di", "di_years")


@pytest.mark.parametrize(("name", "value"), [("di_years", 5), ("has_job", False)])
def test_characteristic_arrays_are_read_only(*, name, value):
    """Neither an integer nor a bool column can be changed after the check."""
    with pytest.raises(ValueError, match="read-only"):
        STATUS[name][0] = value


def test_characteristics_mapping_is_read_only():
    """No column of the mapping can be replaced or added after the check."""
    with pytest.raises(TypeError, match="does not support item assignment"):
        STATUS["di_years"] = np.array([0, 0, 0, 0], dtype=np.int32)  # ty: ignore[invalid-assignment]


@categorical(ordered=False)
class _Two:
    a: ScalarInt
    b: ScalarInt


def _columns_missing_category():
    characteristics(category_class=_Two, by_category={"a": {"x": 1}})


def _columns_extra_category():
    characteristics(
        category_class=_Two, by_category={"a": {"x": 1}, "b": {"x": 2}, "c": {"x": 3}}
    )


def _columns_in_wrong_order():
    characteristics(category_class=_Two, by_category={"b": {"x": 1}, "a": {"x": 2}})


def _columns_with_differing_characteristics():
    characteristics(category_class=_Two, by_category={"a": {"x": 1}, "b": {"y": 2}})


def _columns_without_characteristics():
    characteristics(category_class=_Two, by_category={"a": {}, "b": {}})


def _columns_with_a_float_in_a_later_row():
    characteristics(
        category_class=_Two,
        by_category={"a": {"x": 0}, "b": {"x": 1.5}},
    )


def _columns_with_a_string_in_a_later_row():
    characteristics(
        category_class=_Two,
        by_category={"a": {"x": 0}, "b": {"x": "q"}},  # ty: ignore[invalid-argument-type]
    )


def _columns_mixing_bool_and_int():
    characteristics(category_class=_Two, by_category={"a": {"x": True}, "b": {"x": 2}})


def _columns_outside_int32():
    characteristics(category_class=_Two, by_category={"a": {"x": 0}, "b": {"x": 2**31}})


def _columns_with_identical_rows():
    characteristics(
        category_class=_Two, by_category={"a": {"x": 1, "y": 0}, "b": {"x": 1, "y": 0}}
    )


def _columns_with_identical_rows_in_another_key_order():
    characteristics(
        category_class=_Two, by_category={"a": {"x": 0, "y": 1}, "b": {"y": 1, "x": 0}}
    )


def _columns_below_int32():
    characteristics(
        category_class=_Two, by_category={"a": {"x": 0}, "b": {"x": -(2**31) - 1}}
    )


def _columns_holding_the_missing_code():
    characteristics(
        category_class=_Two, by_category={"a": {"x": 0}, "b": {"x": -(2**31)}}
    )


def _columns_for_a_plain_class():
    class _NotCategorical:
        a: int

    characteristics(
        category_class=_NotCategorical,  # ty: ignore[invalid-argument-type]
        by_category={"a": {"x": 1}},
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (_columns_missing_category, r"exactly the categories.*lists \['a'\]"),
        (_columns_extra_category, r"lists \['a', 'b', 'c'\]"),
        (_columns_in_wrong_order, r"declaration order, but lists \['b', 'a'\]"),
        (_columns_with_differing_characteristics, r"`b`.*\['y'\].*\['x'\]"),
        (_columns_without_characteristics, "at least one characteristic"),
        (_columns_with_a_float_in_a_later_row, r"`x`.*int.*float"),
        (_columns_with_a_string_in_a_later_row, r"`b\.x` is 'q'"),
        (_columns_mixing_bool_and_int, r"`x`.*bool.*int"),
        (_columns_outside_int32, r"`b\.x`.*int32"),
        (_columns_with_identical_rows, r"`a`, `b`.*same characteristics"),
        (
            _columns_with_identical_rows_in_another_key_order,
            "have the same characteristics",
        ),
        (_columns_below_int32, r"`b\.x`.*int32"),
        (_columns_holding_the_missing_code, r"`b\.x`.*missing-category code"),
        (_columns_for_a_plain_class, "DataclassInstance"),
    ],
)
def test_characteristics_reject_inconsistent_tables(*, build, message):
    """Every inconsistency between the class and its table raises."""
    with pytest.raises(CategoricalDefinitionError, match=message):
        build()


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ({"has_job": True, "on_di": False, "di_years": 0}, 0),
        ({"has_job": False, "on_di": False, "di_years": 0}, 1),
        ({"has_job": False, "on_di": True, "di_years": 2}, 3),
    ],
)
def test_category_with_returns_the_code_of_the_matching_category(*, row, code):
    """A full row maps back to its category's code."""
    assert int(category_with(STATUS, **row)) == code


def test_category_with_accepts_a_subset_that_tells_all_categories_apart():
    """A subset suffices when no two categories share it."""
    columns = characteristics(
        category_class=_Two, by_category={"a": {"x": 1, "y": 0}, "b": {"x": 2, "y": 0}}
    )

    assert int(category_with(columns, x=2)) == 1


def test_category_with_returns_the_missing_code_for_an_unlisted_combination():
    """A combination no category has maps to the missing-category code."""
    found = category_with(STATUS, has_job=True, on_di=True, di_years=1)

    assert int(found) == MISSING_CAT_CODE


def test_category_with_returns_int32():
    """Codes come back in the dtype of category codes."""
    assert (
        category_with(STATUS, has_job=True, on_di=False, di_years=0).dtype == jnp.int32
    )


def test_category_with_under_jit_matches_eager():
    """Characteristics computed inside `jit` map to the same codes."""

    @jax.jit
    def next_status(years):
        return category_with(STATUS, has_job=False, on_di=True, di_years=years + 1)

    assert int(next_status(jnp.int32(1))) == int(Status.di_2)


def test_category_with_under_vmap_maps_each_combination():
    """Batched characteristics map to one code each, misses included."""
    years = jnp.array([1, 2, 3], dtype=jnp.int32)

    found = jax.vmap(
        lambda y: category_with(STATUS, has_job=False, on_di=True, di_years=y)
    )(years)

    np.testing.assert_array_equal(found, np.array([2, 3, MISSING_CAT_CODE]))


def test_category_with_broadcasts_array_arguments():
    """Array characteristics broadcast against each other and against scalars."""
    found = category_with(
        STATUS, has_job=jnp.array([True, False]), on_di=False, di_years=0
    )

    np.testing.assert_array_equal(found, np.array([0, 1]))


def test_category_with_a_subset_shared_by_several_categories_raises():
    """A subset that leaves categories tied raises before reading any value."""
    with pytest.raises(CategoricalDefinitionError, match=r"codes \[0, 1\]"):
        category_with(STATUS, on_di=False, di_years=0)


def test_category_with_without_characteristics_raises():
    """`category_with` needs at least one characteristic."""
    with pytest.raises(CategoricalDefinitionError, match="at least one"):
        category_with(STATUS)


def test_category_with_an_unknown_characteristic_raises():
    """Characteristic names must be columns of the table."""
    with pytest.raises(CategoricalDefinitionError, match="wage"):
        category_with(STATUS, has_job=True, on_di=False, di_years=0, wage=1)


def test_characteristics_match_row_entries_by_name_in_any_order():
    """A row lists its characteristics in any order; columns follow the first row."""
    columns = characteristics(
        category_class=_Two, by_category={"a": {"x": 1, "y": 0}, "b": {"y": 0, "x": 2}}
    )

    assert {name: columns[name].tolist() for name in columns} == {
        "x": [1, 2],
        "y": [0, 0],
    }


def test_characteristics_accept_the_int32_range_above_the_missing_code():
    """Every int32 above the missing-category code is a valid characteristic."""
    columns = characteristics(
        category_class=_Two,
        by_category={"a": {"x": -(2**31) + 1}, "b": {"x": 2**31 - 1}},
    )

    np.testing.assert_array_equal(columns["x"], np.array([-(2**31) + 1, 2**31 - 1]))


@categorical(ordered=False)
class _Signed:
    minimum: ScalarInt
    negative: ScalarInt
    zero: ScalarInt
    maximum: ScalarInt


SIGNED = characteristics(
    category_class=_Signed,
    by_category={
        "minimum": {"x": -(2**31) + 1},
        "negative": {"x": -1},
        "zero": {"x": 0},
        "maximum": {"x": 2**31 - 1},
    },
)


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_category_with_finds_int32_extremes_exactly(*, mode):
    """Signed queries at and next to the int32 endpoints find their own rows."""
    queries = jnp.array([-(2**31), -(2**31) + 1, -1, 0, 1, 2**31 - 1], dtype=jnp.int32)

    def lookup(values):
        return category_with(SIGNED, x=values)

    found = jax.jit(lookup)(queries) if mode == "jit" else lookup(queries)

    np.testing.assert_array_equal(
        found, np.array([MISSING_CAT_CODE, 0, 1, 2, MISSING_CAT_CODE, 3])
    )


@pytest.mark.skipif(not X64_ENABLED, reason="int64 queries need x64")
def test_category_with_treats_int64_values_outside_int32_as_missing():
    """Wide signed queries keep their value: beyond int32 nothing matches."""
    queries = jnp.array([-(2**31) - 1, -1, 2**31, 2**32 - 1], dtype=jnp.int64)

    found = category_with(SIGNED, x=queries)

    np.testing.assert_array_equal(
        found, np.array([MISSING_CAT_CODE, 1, MISSING_CAT_CODE, MISSING_CAT_CODE])
    )


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("value", [-(2**31) - 1, 2**31, 2**32 - 1, 2**64])
def test_category_with_treats_python_ints_outside_int32_as_missing(*, mode, value):
    """No category has a characteristic outside int32, so such a value misses."""

    def lookup():
        return category_with(SIGNED, x=value)

    found = jax.jit(lookup)() if mode == "jit" else lookup()

    assert int(found) == MISSING_CAT_CODE


@pytest.mark.parametrize(
    "value",
    [
        jnp.asarray(2**32 - 1, dtype=jnp.uint32),
        jnp.asarray(1, dtype=jnp.uint8),
        jnp.asarray(1.0),
    ],
)
def test_category_with_rejects_values_that_are_not_bool_or_signed_integers(*, value):
    """Unsigned and float queries raise before any comparison."""
    with pytest.raises(CategoricalDefinitionError, match="bool or signed integer"):
        category_with(SIGNED, x=value)


def test_category_with_a_globally_ambiguous_subset_raises_for_a_unique_value():
    """A subset is judged on all categories, also where the queried value is unique."""
    with pytest.raises(CategoricalDefinitionError, match="do not tell all categories"):
        category_with(STATUS, on_di=True, di_years=1)


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_category_with_broadcasts_two_array_arguments(*, mode):
    """Two array characteristics broadcast against each other into a grid of codes."""

    def lookup(*, has_job, on_di):
        return category_with(STATUS, has_job=has_job, on_di=on_di, di_years=0)

    has_job = jnp.array([[True], [False]])
    on_di = jnp.array([[False, True]])
    found = (
        jax.jit(lookup)(has_job=has_job, on_di=on_di)
        if mode == "jit"
        else lookup(has_job=has_job, on_di=on_di)
    )

    np.testing.assert_array_equal(
        found, np.array([[0, MISSING_CAT_CODE], [1, MISSING_CAT_CODE]])
    )


def _exact_signed_codes(values):
    """Return the code of each value in `SIGNED`, compared as Python integers."""
    stored = {int(value): code for code, value in enumerate(SIGNED["x"].tolist())}
    return np.array([stored.get(int(v), MISSING_CAT_CODE) for v in values], np.int32)


def _signed_lookup(values):
    return category_with(SIGNED, x=values)


_TRANSFORMS = {
    "eager": _signed_lookup,
    "jit": jax.jit(_signed_lookup),
    "vmap": jax.vmap(_signed_lookup),
    "jit_vmap": jax.jit(jax.vmap(_signed_lookup)),
}


def _weak_batch(values):
    """Stack weakly typed integer scalars into one weakly typed array."""
    batch = jnp.stack([jnp.asarray(int(value)) for value in values])
    assert batch.weak_type
    return batch


@pytest.mark.skipif(not X64_ENABLED, reason="64-bit queries need x64")
@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("representation", ["python", "weak", "strong"])
def test_category_with_a_wide_integer_cannot_alias_a_negative_characteristic(
    *, mode, representation
):
    """`2**32 - 1` misses although its low 32 bits equal the stored -1."""
    value = 2**32 - 1
    if representation == "weak":
        value = jnp.asarray(value)
        assert value.weak_type
    elif representation == "strong":
        value = jnp.asarray(value, dtype=jnp.int64)

    found = _signed_lookup(value) if mode == "eager" else jax.jit(_signed_lookup)(value)

    assert (found.dtype, int(found)) == (jnp.int32, MISSING_CAT_CODE)


@pytest.mark.skipif(not X64_ENABLED, reason="64-bit queries need x64")
@pytest.mark.parametrize("mode", ["eager", "jit", "vmap", "jit_vmap"])
@pytest.mark.parametrize("representation", ["weak", "strong"])
def test_category_with_int64_neighbourhoods_of_every_characteristic(
    *, mode, representation
):
    """Each stored value, its neighbours and its 32-bit aliases map exactly."""
    values = sorted(
        {
            stored + k * 2**32 + offset
            for stored in SIGNED["x"].tolist()
            for k in (-2, -1, 0, 1, 2)
            for offset in (-1, 0, 1)
        }
    )
    queries = (
        _weak_batch(values)
        if representation == "weak"
        else jnp.asarray(values, dtype=jnp.int64)
    )

    found = _TRANSFORMS[mode](queries)

    np.testing.assert_array_equal(found, _exact_signed_codes(values))


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap", "jit_vmap"])
def test_category_with_small_weak_integers_map_exactly(*, mode):
    """Weakly typed small integers find their rows at either precision."""
    values = (-1, 0, 1, 2)

    found = _TRANSFORMS[mode](_weak_batch(values))

    np.testing.assert_array_equal(found, _exact_signed_codes(values))


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap", "jit_vmap"])
@pytest.mark.parametrize(
    "dtype", [jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.bool_]
)
def test_category_with_accepts_every_signed_width_and_bool(*, mode, dtype):
    """Strong signed integers of any width, and bools, compare by value."""
    if dtype == jnp.int64 and not X64_ENABLED:
        pytest.skip("int64 needs x64")
    values = (False, True) if dtype == jnp.bool_ else (-1, 0, 1, 2)

    found = _TRANSFORMS[mode](jnp.asarray(values, dtype=dtype))

    np.testing.assert_array_equal(found, _exact_signed_codes(values))


@categorical(ordered=False)
class _Flag:
    off: ScalarInt
    on: ScalarInt


FLAG = characteristics(
    category_class=_Flag,
    by_category={"off": {"flag": False}, "on": {"flag": True}},
)

# Signed queries of a bool column: only 0 and 1 name a category.
_FLAG_QUERIES = (-2, -1, 0, 1, 2)
_FLAG_CODES = np.array(
    [MISSING_CAT_CODE, MISSING_CAT_CODE, 0, 1, MISSING_CAT_CODE], dtype=np.int32
)


def _flag_lookup(values):
    return category_with(FLAG, flag=values)


_FLAG_TRANSFORMS = {
    "eager": _flag_lookup,
    "jit": jax.jit(_flag_lookup),
    "vmap": jax.vmap(_flag_lookup),
    "jit_vmap": jax.jit(jax.vmap(_flag_lookup)),
}


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_category_with_python_ints_other_than_0_and_1_miss_a_bool_column(*, mode):
    """A Python int names a bool category only as 0 or 1; 2 is not `True`."""
    lookup = _FLAG_TRANSFORMS[mode]

    found = [int(lookup(value)) for value in _FLAG_QUERIES]

    np.testing.assert_array_equal(found, _FLAG_CODES)


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap", "jit_vmap"])
@pytest.mark.parametrize("representation", ["weak", "int8", "int16", "int32", "int64"])
def test_category_with_signed_arrays_other_than_0_and_1_miss_a_bool_column(
    *, mode, representation
):
    """Signed arrays of any width name a bool category only as 0 or 1."""
    if representation == "int64" and not X64_ENABLED:
        pytest.skip("int64 needs x64")
    queries = (
        _weak_batch(_FLAG_QUERIES)
        if representation == "weak"
        else jnp.asarray(_FLAG_QUERIES, dtype=representation)
    )

    found = _FLAG_TRANSFORMS[mode](queries)

    assert found.dtype == jnp.int32
    np.testing.assert_array_equal(found, _FLAG_CODES)


@pytest.mark.skipif(not X64_ENABLED, reason="64-bit queries need x64")
@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_category_with_wide_weak_values_broadcast_with_bool_flags(*, mode):
    """A wide alias misses at every flag it broadcasts against."""
    columns = characteristics(
        category_class=_Signed,
        by_category={
            "minimum": {"x": -1, "flag": False},
            "negative": {"x": -1, "flag": True},
            "zero": {"x": 0, "flag": False},
            "maximum": {"x": 0, "flag": True},
        },
    )

    def lookup(*, values, flags):
        return category_with(columns, x=values, flag=flags)

    values = _weak_batch((-1, 2**32 - 1, 0)).reshape(3, 1)
    flags = jnp.array([[False, True]])
    found = (
        jax.jit(lookup)(values=values, flags=flags)
        if mode == "jit"
        else lookup(values=values, flags=flags)
    )

    np.testing.assert_array_equal(
        found, np.array([[0, 1], [MISSING_CAT_CODE, MISSING_CAT_CODE], [2, 3]])
    )


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
    on_di = category_with(STATUS, has_job=False, on_di=True, di_years=years)
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
    """Published characteristics and `category_with` reproduce the hand-table V."""
    np.testing.assert_array_equal(_solve(use_pattern=True), _solve(use_pattern=False))


def test_characteristic_pattern_simulates_the_hand_table_paths():
    """Published characteristics and `category_with` reproduce the simulated paths."""
    pd.testing.assert_frame_equal(
        _simulate(use_pattern=True), _simulate(use_pattern=False), check_exact=True
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


def _float_characteristics():
    return characteristics(
        category_class=_Two,
        by_category={
            "a": {"index": 0, "wage": 0.5, "employed": False},
            "b": {"index": 1, "wage": 1.25, "employed": True},
        },
    )


def test_characteristics_store_float_values_as_float64():
    """Floating properties retain their values in a read-only float64 column."""
    column = _float_characteristics()["wage"]
    assert (column.dtype, column.tolist(), column.flags.writeable) == (
        np.dtype(np.float64),
        [0.5, 1.25],
        False,
    )


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap", "jit_vmap"])
def test_float_characteristics_are_read_by_category_code(*, mode):
    """Category codes select numerical properties under JAX transformations."""
    columns = _float_characteristics()

    def wage(code):
        return jnp.asarray(columns["wage"])[code]

    transforms = {
        "eager": wage,
        "jit": jax.jit(wage),
        "vmap": jax.vmap(wage),
        "jit_vmap": jax.jit(jax.vmap(wage)),
    }
    found = transforms[mode](jnp.array([1, 0, 1], dtype=jnp.int32))
    np.testing.assert_array_almost_equal(
        found, np.array([1.25, 0.5, 1.25]), decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_characteristics_reject_nonfinite_float_properties(*, value):
    """A numerical category property must be finite, including later rows."""
    with pytest.raises(CategoricalDefinitionError, match="finite"):
        characteristics(
            category_class=_Two, by_category={"a": {"wage": 0.5}, "b": {"wage": value}}
        )


@pytest.mark.parametrize("first", [False, 0])
def test_characteristics_reject_mixed_float_column_types(*, first):
    """Each characteristic has one Python scalar type across its categories."""
    with pytest.raises(CategoricalDefinitionError, match="mixes"):
        characteristics(
            category_class=_Two, by_category={"a": {"wage": first}, "b": {"wage": 1.25}}
        )


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_category_with_ignores_unqueried_float_properties(*, mode):
    """Integer identifying fields work in tables that also contain floats."""
    columns = _float_characteristics()

    def lookup(index):
        return category_with(columns, index=index)

    found = jax.jit(lookup)(jnp.int32(1)) if mode == "jit" else lookup(jnp.int32(1))
    assert int(found) == 1


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_category_with_rejects_float_columns_even_for_integer_queries(*, mode):
    """A float property cannot identify a category, even at an exact integer."""
    columns = _float_characteristics()

    def lookup(index):
        return category_with(columns, index=index, wage=1)

    with pytest.raises(CategoricalDefinitionError, match=r"float.*wage|wage.*float"):
        jax.jit(lookup)(jnp.int32(1)) if mode == "jit" else lookup(jnp.int32(1))


def test_category_with_rejects_python_float_queries():
    """A float query cannot match an integer identifying field."""
    with pytest.raises(CategoricalDefinitionError):
        category_with(STATUS, di_years=1.0)  # ty: ignore[invalid-argument-type]


def test_characteristics_reject_duplicate_float_rows():
    """Float properties do not remove the requirement that category rows differ."""
    with pytest.raises(CategoricalDefinitionError, match="same characteristics"):
        characteristics(
            category_class=_Two, by_category={"a": {"wage": 0.5}, "b": {"wage": 0.5}}
        )


def test_category_with_tie_only_floats_break_names_the_float_columns():
    """A tie that only float properties break says that no query can resolve it."""
    columns = characteristics(
        category_class=_Two,
        by_category={"a": {"hours": 0, "wage": 0.0}, "b": {"hours": 0, "wage": 0.8}},
    )
    with pytest.raises(
        CategoricalDefinitionError, match=r"\['wage'\] are float properties"
    ):
        category_with(columns, hours=0)


def test_category_with_tie_other_columns_break_asks_for_more_characteristics():
    """A tie that an unqueried integer or bool column breaks asks for more of them."""
    with pytest.raises(CategoricalDefinitionError, match="Pass more characteristics"):
        category_with(STATUS, has_job=False)
