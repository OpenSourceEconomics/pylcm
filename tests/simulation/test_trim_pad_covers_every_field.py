"""`trim_pad_from_raw_results` must trim EVERY per-subject field, not a fixed list.

The simulate dispatch pads the subject axis up to a multiple of the batch size and
trims the pad rows off once, at the end. That trim used to enumerate the fields it
knew about — `V_arr`, `actions`, `states`, `in_regime` — which is a second, silent
copy of `PeriodRegimeSimulationData`'s definition. When `nested_policy_fallback` was
added to the structure, the enumeration was not, so `dataclasses.replace` carried the
untrimmed field through: a leaf padded to 24 rows against a 21-row `_in_regime` mask,
and `to_dataframe` died on `column[mask]` with an `IndexError`.

These tests are written against the structure's own field list, so a field added
tomorrow is covered without editing this file. That is the point — the defect was not
"one field was forgotten" but "the list can be forgotten".
"""

import dataclasses
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import pytest

from _lcm.engine import PeriodRegimeSimulationData
from _lcm.simulation.initial_conditions import trim_pad_from_raw_results

#: Fields holding `name -> per-subject array` mappings rather than a bare array.
_MAPPING_FIELDS = frozenset({"actions", "states"})

_N_REAL = 7
_N_PADDED = 8


def _column(field: dataclasses.Field, n_rows: int):
    """A per-subject column whose dtype satisfies `field`'s annotation.

    The dtype is read off the annotation rather than fixed, because these fields are
    beartype-checked: a `Bool1D` handed an int array is rejected at construction.
    """
    annotation = str(field.type)
    if "Bool" in annotation:
        return jnp.arange(n_rows) % 2 == 0
    if "Int" in annotation:
        return jnp.arange(n_rows, dtype=jnp.int32)
    return jnp.arange(n_rows, dtype=jnp.float64)


def _period_data(n_rows: int) -> PeriodRegimeSimulationData:
    """One period's data with EVERY field at `n_rows` rows.

    Built field-by-field from the dataclass so a newly added field is populated here
    automatically; `kwargs` is `Any`-valued because the per-field type is only known
    at runtime, and beartype checks it at construction anyway.
    """
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(PeriodRegimeSimulationData):
        column = _column(field, n_rows)
        if field.name in _MAPPING_FIELDS:
            kwargs[field.name] = MappingProxyType({"a": column, "b": column})
        else:
            kwargs[field.name] = column
    return PeriodRegimeSimulationData(**kwargs)


def _padded_period_data() -> PeriodRegimeSimulationData:
    """One period's data with EVERY field padded to `_N_PADDED` rows."""
    return _period_data(_N_PADDED)


def _widths(data: PeriodRegimeSimulationData) -> dict[str, int]:
    """Leading-axis length of every per-subject array, mappings flattened."""
    widths: dict[str, int] = {}
    for field in dataclasses.fields(PeriodRegimeSimulationData):
        value = getattr(data, field.name)
        if field.name in _MAPPING_FIELDS:
            for key, arr in value.items():
                widths[f"{field.name}[{key}]"] = int(arr.shape[0])
        else:
            widths[field.name] = int(value.shape[0])
    return widths


def test_the_mapping_fields_are_still_the_mapping_fields():
    """Guard the assumption `_MAPPING_FIELDS` encodes, so this file cannot go stale.

    If the structure grows a new mapping-valued field, the builder above would hand it
    a bare array and the coverage test would silently stop exercising the mapping path.
    """
    data = _padded_period_data()
    for field in dataclasses.fields(PeriodRegimeSimulationData):
        value = getattr(data, field.name)
        is_mapping = hasattr(value, "items")
        assert is_mapping == (field.name in _MAPPING_FIELDS), (
            f"{field.name}: mapping-ness disagrees with _MAPPING_FIELDS -- update it"
        )


def test_every_field_of_the_structure_is_trimmed():
    """No field may survive the trim at its padded width."""
    raw = MappingProxyType({"work": MappingProxyType({0: _padded_period_data()})})
    trimmed = trim_pad_from_raw_results(raw_results=raw, original_n_subjects=_N_REAL)
    widths = _widths(trimmed["work"][0])

    # The test is worthless if it inspects nothing.
    assert widths, "no fields inspected -- the probe is broken, not the code clean"
    untrimmed = {name: w for name, w in widths.items() if w != _N_REAL}
    assert not untrimmed, (
        f"fields left at their padded width: {untrimmed} "
        f"(expected every field at {_N_REAL})"
    )


def test_an_already_trimmed_period_is_returned_untouched():
    """The no-op path must stay a no-op -- it is keyed on `V_arr`'s width."""
    data = _period_data(_N_REAL)
    raw = MappingProxyType({"work": MappingProxyType({0: data})})

    trimmed = trim_pad_from_raw_results(raw_results=raw, original_n_subjects=_N_REAL)
    assert trimmed["work"][0] is data


@pytest.mark.parametrize("skipped", sorted(_MAPPING_FIELDS | {"in_regime"}))
def test_the_coverage_check_rejects_a_field_left_behind(skipped: str):
    """The invariant discriminates: reintroduce the defect and it must be caught.

    Rather than trust that `test_every_field_of_the_structure_is_trimmed` would notice,
    hand it a result in which one field really was left at its padded width.
    """
    trimmed_data = trim_pad_from_raw_results(
        raw_results=MappingProxyType(
            {"work": MappingProxyType({0: _padded_period_data()})}
        ),
        original_n_subjects=_N_REAL,
    )["work"][0]

    field = next(
        f for f in dataclasses.fields(PeriodRegimeSimulationData) if f.name == skipped
    )
    padded_column = _column(field, _N_PADDED)
    regression = dataclasses.replace(
        trimmed_data,
        **{
            skipped: MappingProxyType({"a": padded_column, "b": padded_column})
            if skipped in _MAPPING_FIELDS
            else padded_column
        },
    )
    widths = _widths(regression)
    assert any(w != _N_REAL for w in widths.values()), (
        f"leaving {skipped} untrimmed was not observable -- the check is blind"
    )
