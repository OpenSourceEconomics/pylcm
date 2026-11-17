"""Class regressions for bounded callable capture; no native Model or DP solve."""

import pytest
from dataclasses import dataclass
import functools
import json
from _lcm_regime_finder.callable_identity import (
    capture_callable,
    capture_session,
    primitive_operation,
)
from _lcm_regime_finder.model_snapshot import normalize, source_function

_REPR_DIRECTION = 1


@dataclass(frozen=True)
class ReprDependentCategory:
    value: int

    def __repr__(self):
        return str(_REPR_DIRECTION * self.value)


def independent_helper(history):
    return history + 1


def custom_where(condition, left, right):
    return left


def local_callback(history):
    return independent_helper(history)


def factory(cutoff):
    def callback(history, age):
        return history if age < cutoff else 0

    return callback


def _get_numpy():
    try:
        import numpy
    except ImportError:
        pytest.skip("NumPy is optional; no environment change requested")
    return numpy


def test_known_primitive_stops_before_library_implementation_graph() -> None:
    meta = source_function(_get_numpy().where)
    assert (meta["kind"]) == ("primitive")
    assert (meta["canonical"]) == ("numpy.where")
    assert (meta["resolution"]) == ("actual_object_identity")
    assert (meta["call_targets"]) == ({})
    assert (meta["referenced_functions"]) == ({})
    assert ("bound_self") not in (meta)
    assert ("source") not in (meta)


def test_naked_primitive_has_no_invented_named_input_contract() -> None:
    meta = source_function(_get_numpy().logical_and)
    assert meta["variadic"]
    assert (meta["arguments"]) == ([])


def test_callsite_rule_survives_terminal_capture() -> None:
    import ast

    choose = _get_numpy().where

    def callback(history, age):
        return choose(age < 2, history, 0)

    meta = source_function(callback)
    assert (primitive_operation(meta, ast.Name(id="choose"))) == ("select")


def test_identical_name_does_not_authenticate_an_unrelated_callable() -> None:
    meta = source_function(custom_where)
    assert (meta["kind"]) == ("callable")
    assert ("canonical") not in (meta)


def test_repeated_capture_shares_record_within_one_frozen_transaction() -> None:
    count = 10000
    record = normalize(
        {
            "callbacks": [local_callback] * count,
            "production_domain": {"n_points": 10**18},
        }
    )
    callbacks = record["callbacks"]
    assert (len(callbacks)) == (count)
    assert all(value is callbacks[0] for value in callbacks)
    assert (record["production_domain"]["n_points"]) == (10**18)


def test_capture_cache_does_not_survive_the_transaction() -> None:
    fn = functools.partial(independent_helper, history=1)
    first = source_function(fn)
    fn.keywords["history"] = 2
    second = source_function(fn)
    assert (first["binding_sha256"]) != (second["binding_sha256"])
    assert (first["partial_bindings"]["history"]) == (1)
    assert (second["partial_bindings"]["history"]) == (2)


def test_normalizer_is_part_of_the_capture_cache_key() -> None:
    fn = functools.partial(independent_helper, history=1)

    def labelled(value, **kwargs):
        return {"normalizer": "labelled", "value": normalize(value, **kwargs)}

    with capture_session():
        first = capture_callable(fn, normalize)
        second = capture_callable(fn, labelled)
    assert (first["binding_sha256"]) != (second["binding_sha256"])
    assert (first["partial_bindings"]) != (second["partial_bindings"])


def test_depth_does_not_truncate_or_duplicate_callable_identity() -> None:
    fn = factory(2)
    with capture_session():
        deep = source_function(fn, depth=800)
        full = source_function(fn, depth=0)
    assert (deep) is (full)
    assert (full["binding_sha256"]) is not None
    assert ("source") in (full)
    assert (full["capture_status"]) == ("COMPLETE")


def test_set_identity_uses_captured_values_not_process_representation() -> None:
    global _REPR_DIRECTION
    values = {ReprDependentCategory(1), ReprDependentCategory(2)}
    try:
        _REPR_DIRECTION = 1
        first = normalize(values)
        _REPR_DIRECTION = -1
        second = normalize(values)
    finally:
        _REPR_DIRECTION = 1
    assert (first) == (second)
    assert ([x["fields"]["value"] for x in first]) == ([1, 2])
    assert (json.dumps(first, sort_keys=True)) == (json.dumps(second, sort_keys=True))
