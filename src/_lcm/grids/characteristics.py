"""Characteristics of the categories of a discrete variable.

Some discrete variables have categories that are combinations of characteristics:
a labour-market status is "employed", "on disability insurance for one year", and
so on, and each status fixes whether the person has a job, receives benefits and
for how long. `characteristics` checks a table that lists every category's
characteristics against its categorical class and returns one array per
characteristic, indexed by code. `category_with` maps characteristics back to a code.
"""

from collections.abc import Mapping
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from beartype import beartype

from _lcm.beartype_conf import CATEGORICAL_CONF
from _lcm.grids.categorical import get_category_codes, validate_category_class
from _lcm.typing import DataclassInstance
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import CategoricalDefinitionError
from lcm.typing import IntND, ValueND

# The value of one characteristic of one category, as written in the table.
type CharacteristicValue = int | bool

# The missing-category code; `initial_conditions` imports the grids, so the
# value is restated here and a test pins it to the engine's sentinel.
_MISSING_CODE = np.iinfo(np.int32).min

_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)


@beartype(conf=CATEGORICAL_CONF)
def characteristics(
    *,
    category_class: type[DataclassInstance],
    by_category: Mapping[str, Mapping[str, CharacteristicValue]],
) -> MappingProxyType[str, np.ndarray]:
    """Check a table of category characteristics and return it by code.

    Example:
        @categorical(ordered=False)
        class Status:
            employed: ScalarInt
            di_1: ScalarInt
            di_2: ScalarInt

        STATUS = characteristics(
            category_class=Status,
            by_category={
                "employed": {"has_job": True, "di_years": 0},
                "di_1": {"has_job": False, "di_years": 1},
                "di_2": {"has_job": False, "di_years": 2},
            },
        )
        STATUS["di_years"]  # array([0, 1, 2], dtype=int32)

    Args:
        category_class: The categorical class whose categories the table
            describes.
        by_category: Mapping of every category name, in declaration order, to its
            characteristics. Every category sets the same characteristics,
            in any order, each to an `int` or a `bool`, and no two categories
            set all of them to the same values.

    Returns:
        A read-only mapping from each characteristic, in the first row's order,
        to a read-only 1-d array indexed by category code: `bool` for bool
        characteristics, `int32` otherwise.

    Raises:
        CategoricalDefinitionError: If the table does not describe exactly the
            categories of `category_class` in order, the categories set
            different characteristics, a value is no `int` or `bool` or leaves
            the int32 range, a characteristic mixes `bool` and `int`, or two
            categories share all characteristics.

    """
    class_errors = validate_category_class(category_class)
    if class_errors:
        raise CategoricalDefinitionError(format_messages(class_errors))

    error_messages = _check_case_names(
        category_names=tuple(get_category_codes(category_class)),
        case_names=tuple(by_category),
    )
    if not error_messages:
        error_messages = _check_rows(by_category=by_category)
    if error_messages:
        raise CategoricalDefinitionError(format_messages(error_messages))

    names = tuple(next(iter(by_category.values())))
    columns = {}
    for name in names:
        values = [row[name] for row in by_category.values()]
        dtype = bool if isinstance(values[0], bool) else np.int32
        column = np.asarray(values, dtype=dtype)
        column.flags.writeable = False
        columns[name] = column
    return MappingProxyType(columns)


@beartype(conf=CATEGORICAL_CONF)
def category_with(
    columns: Mapping[str, np.ndarray], /, **query: ValueND | int
) -> IntND:
    """Return the code of the category with the given characteristics.

    The given characteristics must tell all categories apart; this is checked
    from `columns` before any value is read, so it holds for every value and
    raises at trace time. Values are Python ints or bools, or bool or
    signed-integer arrays, traced or not, and broadcast against each other.

    Example:
        category_with(STATUS, has_job=False, di_years=years + 1)

    Args:
        columns: The table `characteristics` returns.
        **query: Values of some or all characteristics.

    Returns:
        The int32 code of the matching category at every broadcast position,
        or the missing-category code (the int32 minimum) where no category has
        these characteristics, including every Python int outside int32.

    Raises:
        CategoricalDefinitionError: If no characteristic is given, one is not a
            column, two categories share the given characteristics, or a value
            is an array of neither bool nor signed-integer dtype.

    """
    names = _get_identifying_names(columns=columns, given=tuple(query))
    values = {name: _prepare_value(name=name, value=query[name]) for name in names}
    shape = jnp.broadcast_shapes(
        *(value.shape for value in values.values() if value is not None)
    )
    n_categories = len(columns[names[0]])
    category_axis = (n_categories,) + (1,) * len(shape)
    matches = jnp.ones((n_categories, *shape), dtype=bool)
    for name, value in values.items():
        if value is None:
            matches = jnp.zeros_like(matches)
            continue
        column = jnp.asarray(columns[name]).reshape(category_axis)
        matches = matches & (column == value)
    return jnp.where(
        jnp.any(matches, axis=0),
        jnp.argmax(matches, axis=0).astype(jnp.int32),
        jnp.int32(_MISSING_CODE),
    )


def _prepare_value(*, name: str, value: ValueND | int) -> ValueND | None:
    """Return `value` as an array that compares by its integer value.

    A Python int outside int32 matches no characteristic and returns `None`.
    Arrays must be bool or signed integers: JAX
    compares an unsigned array with an int32 column in int32 when x64 is off,
    which would let a large unsigned value match a negative characteristic.
    """
    if isinstance(value, int):
        if _INT32_MIN <= value <= _INT32_MAX:
            return jnp.asarray(value)
        return None
    if value.dtype.kind not in {"b", "i"}:
        raise CategoricalDefinitionError(
            f"category_with takes bool or signed integer values, but `{name}` has "
            f"dtype {value.dtype}."
        )
    return value


def _check_case_names(
    *, category_names: tuple[str, ...], case_names: tuple[str, ...]
) -> list[str]:
    """Return a message unless the table lists the categories in order."""
    if case_names == category_names:
        return []
    return [
        (
            f"The table must list exactly the categories {list(category_names)}, "
            f"in declaration order, but lists {list(case_names)}."
        )
    ]


def _check_rows(
    *, by_category: Mapping[str, Mapping[str, CharacteristicValue]]
) -> list[str]:
    """Return every inconsistency among the rows of the table."""
    first_name, first_row = next(iter(by_category.items()))
    names = list(first_row)
    if not names:
        return ["Every category needs at least one characteristic."]

    error_messages = [
        f"Category `{case}` sets the characteristics {list(row)}, but `{first_name}` "
        f"sets {names}; every category sets the same characteristics."
        for case, row in by_category.items()
        if set(row) != set(names)
    ]
    if error_messages:
        return error_messages

    for name in names:
        values = {case: row[name] for case, row in by_category.items()}
        error_messages.extend(_check_column(name=name, values=values))
    if error_messages:
        return error_messages

    groups: dict[tuple[CharacteristicValue, ...], list[str]] = {}
    for case, row in by_category.items():
        groups.setdefault(tuple(row[name] for name in names), []).append(case)
    error_messages.extend(
        f"Categories {', '.join(f'`{case}`' for case in group)} have the same "
        f"characteristics; every category needs a combination of its own."
        for group in groups.values()
        if len(group) > 1
    )
    return error_messages


def _check_column(*, name: str, values: Mapping[str, CharacteristicValue]) -> list[str]:
    """Return the problems of one characteristic's values."""
    error_messages = [
        f"`{case}.{name}` is {value!r}; characteristics take an int or a bool."
        for case, value in values.items()
        if type(value) not in {int, bool}
    ]
    if error_messages:
        return error_messages
    if len({type(value) for value in values.values()}) > 1:
        return [f"Characteristic `{name}` mixes bool and int values."]
    return [
        f"`{case}.{name}` is {value}, outside the int32 range of category codes."
        for case, value in values.items()
        if not _INT32_MIN <= value <= _INT32_MAX
    ]


def _get_identifying_names(
    *, columns: Mapping[str, np.ndarray], given: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the given characteristics if they tell all categories apart."""
    if not given:
        raise CategoricalDefinitionError(
            f"category_with needs at least one characteristic; the columns are "
            f"{list(columns)}."
        )
    unknown = [name for name in given if name not in columns]
    if unknown:
        raise CategoricalDefinitionError(
            f"category_with got characteristics {unknown} that are not columns; the "
            f"columns are {list(columns)}."
        )
    names = tuple(name for name in columns if name in given)
    groups: dict[tuple[CharacteristicValue, ...], list[int]] = {}
    rows = zip(*(np.asarray(columns[name]).tolist() for name in names), strict=True)
    for code, row in enumerate(rows):
        groups.setdefault(row, []).append(code)
    ties = [codes for codes in groups.values() if len(codes) > 1]
    if ties:
        details = "; ".join(f"codes {codes}" for codes in ties)
        raise CategoricalDefinitionError(
            f"The characteristics {list(names)} do not tell all categories apart: "
            f"{details} share them. Pass more characteristics, or use the class "
            f"constant."
        )
    return names
