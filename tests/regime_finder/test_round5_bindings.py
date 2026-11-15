"""Source-backed binding regressions and independent Python call oracles.

No Model, solve, simulation, or Bellman evaluation occurs. The independent
oracle invokes only the explicit tiny scalar fixtures below. Native numerical
integration still requires the pinned production environment.
"""

import pytest
import typing
import lcm.typing as native_typing
from beartype import BeartypeConf, BeartypeStrategy, FrozenDict, beartype
from lcm.typing import FloatND
from typing import Protocol, TypeAliasType
from dataclasses import dataclass
import functools
import inspect
import builtins
from abc import ABCMeta
from collections.abc import Callable
from unittest.mock import patch
from types import FunctionType

import numpy as np
import jax.numpy as jnp
from jax import Array
from numpy.typing import NDArray

from _lcm_regime_finder.callable_identity import resolve_callable
from _lcm_regime_finder.declarations import (
    DeclarationError,
    ScalarSource,
    normalize_declarations,
)
from _lcm_regime_finder.expressions import partial as evaluate_expression
from _lcm_regime_finder.model_snapshot import read_declarations, source_function
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder.semantic_check import SourceProof, _Abstract, _UNKNOWN
from tests.regime_finder.test_model_adapter import snapshot


_OPTIONAL_BINDING = None


def optional_global(history):
    return history if _OPTIONAL_BINDING is None else 0


def optional_factory(marker):
    def callback(history):
        return history if marker is None else marker

    return callback


def identity(history):
    return history


def difference(left, right):
    return left - right


def triple(first, second, third=3):
    return first + second - third


def default_helper(value=9):
    return value


def call_default(history):
    del history
    return default_helper()


def positional_callback(helper):
    def callback(history):
        return helper(history)

    return callback


def named_callback(helper):
    def callback(history):
        return helper(history=history)

    return callback


def keyword_override(helper, replacement):
    def callback(history):
        return helper(left=history, right=replacement)

    return callback


def keyword_default(helper):
    def callback(history):
        return helper(left=history)

    return callback


def keyword_only(*, history=7):
    return history


def variadic(**kwargs):
    return kwargs["history"]


@dataclass
class Receiver:
    optional: int | None

    def callback(self, history):
        return history if self.optional is None else 0


def direct_oracle(callback, **arguments):
    """Python's actual invocation semantics; no production AST is reused."""
    bound = inspect.signature(callback).bind(**arguments)
    return callback(*bound.args, **bound.kwargs)


def test_none_global_is_a_value_not_an_absent_bound_receiver() -> None:
    captured = read_declarations({"callback": optional_global})["callback"]
    assert (captured["bindings"]["_OPTIONAL_BINDING"]) is None
    assert ("bound_self") not in (captured)
    assert not (
        captured["source_unavailable"] if "source_unavailable" in captured else False
    )


def test_none_closure_and_other_scalar_bindings(*, subtests: pytest.Subtests) -> None:
    identities = set()
    for marker in (None, False, 0, 1, -1, "marker"):
        with subtests.test(marker=marker):
            captured = source_function(optional_factory(marker))
            assert (captured["bindings"]["marker"]) == (marker)
            assert ("bound_self") not in (captured)
            identities.add(captured["binding_sha256"])
    assert (len(identities)) == (6)


def test_bound_receiver_containing_none_remains_a_receiver() -> None:
    captured = source_function(Receiver(None).callback)
    assert (captured["bound_self"]["fields"]["optional"]) is None
    assert (captured["bindings"]["self"]) == (captured["bound_self"])


def _check_scalar(callback, history):
    captured = source_function(callback)
    expected = direct_oracle(callback, history=history)
    expression = ScalarSource(captured).expression()
    translated = evaluate_expression(expression, {"history": history})
    independently_checked = SourceProof({}, static={"history": history}).callback(
        captured
    )
    assert translated.known
    assert independently_checked.proved
    assert (translated.value) == (expected)
    assert (independently_checked.value) == (expected)
    return captured


def test_exposed_partial_keyword_is_not_frozen(*, subtests: pytest.Subtests) -> None:
    for frozen in (-3, 0, 2):
        callback = functools.partial(identity, history=frozen)
        captured = source_function(callback)
        assert (captured["parameter_kinds"]["history"]) == ("KEYWORD_ONLY")
        assert (ScalarSource(captured).expression()) == ("history")
        assert (SourceProof({}).callback(captured).reads) == (frozenset({"history"}))
        for history in (-4, -1, 0, 1, 5):
            with subtests.test(frozen=frozen, history=history):
                _check_scalar(callback, history)


def test_public_normalization_retains_overridable_current_input() -> None:
    original = snapshot()
    original["user_regimes"]["original"]["fields"]["functions"]["utility"] = (
        source_function(functools.partial(identity, history=0))
    )
    normalized = normalize_declarations(original)
    for cell in normalized["cells"]:
        callback = normalized["callbacks"][cell["functions"]["utility"]]
        assert (callback["reads"]) == (["history"])
        assert (callback["expression"]) == ("history")


def test_public_production_cannot_adopt_deletion_of_overridable_input() -> None:
    original = snapshot()
    original["user_regimes"]["original"]["fields"]["functions"]["utility"] = (
        source_function(functools.partial(identity, history=0))
    )
    original["semantic_contract"] = {
        "roles": {
            "original": {
                "actor": "one",
                "information": ["history", "x"],
                "commitments": [],
            }
        },
        "stages": {"original": "decision"},
        "allocation": {
            "kind": "per_block",
            "status": "known",
            "identity": "toy-declared-allocation",
        },
        "production_metadata": {
            "status": "complete",
            "scope": "explicit tiny declaration witness",
        },
    }
    result = analyze_production(original, limits={"seconds": 10}, search_mode="exact")
    assert (result["semantic_check"]["status"]) == ("PASS")
    assert all(
        "history" in axes for axes in result["analysis"]["dependency_closure"].values()
    )


def test_positional_partial_helper_uses_effective_signature() -> None:
    callback = positional_callback(functools.partial(difference, -3))
    _check_scalar(callback, 7)


def test_partial_keyword_default_and_explicit_override() -> None:
    helper = functools.partial(difference, right=-2)
    _check_scalar(keyword_default(helper), 6)
    _check_scalar(keyword_override(helper, 5), 6)


def test_mixed_nested_partials_keep_positional_and_default_roles() -> None:
    helper = functools.partial(functools.partial(triple, 5), third=-3)
    _check_scalar(positional_callback(helper), 4)


def test_empty_helper_call_uses_the_captured_effective_default() -> None:
    _check_scalar(call_default, history=2)


def test_keyword_only_binding_works_by_name() -> None:
    _check_scalar(named_callback(keyword_only), history=11)


def test_wrong_positional_binding_is_not_certified() -> None:
    captured = source_function(positional_callback(keyword_only))
    with pytest.raises(DeclarationError):
        ScalarSource(captured).expression()
    checked = SourceProof({}, static={"history": 1}).callback(captured)
    assert not (checked.proved)


def test_variadic_input_is_still_an_obligation() -> None:
    captured = source_function(variadic)
    assert captured["variadic"]
    checked = SourceProof({}, static={"history": 1}).callback(captured)
    assert not (checked.proved)


def _builtin_int_check(x: int) -> bool:
    return isinstance(x, int)


_INT_CHECK_ALIAS = builtins.isinstance
_INT_TYPE_ALIAS = int


def _aliased_builtin_int_check(x: int) -> bool:
    return _INT_CHECK_ALIAS(x, _INT_TYPE_ALIAS)


def _type_guard_dispatch(regime: int, x: int, y: int) -> int | NDArray[np.int64]:
    if _builtin_int_check(regime):
        if regime == 0:
            return x + 2
        return y * 3
    else:
        return np.where(regime == 0, x + 2, y * 3)


def _unsupported_float_type_check(x: int) -> bool:
    return isinstance(x, float)


def _unsupported_bool_type_check(x: int) -> bool:
    return isinstance(x, bool)


def _unsupported_tuple_type_check(x: int) -> bool:
    return isinstance(x, (int, float))


class _TypeGuardMeta(ABCMeta):
    def __instancecheck__(cls, instance: int) -> bool:
        raise AssertionError("Source capture must not execute a metaclass hook")


class _TypeGuardCustom(metaclass=_TypeGuardMeta):
    pass


def _custom_metaclass_type_check(x: int) -> bool:
    return isinstance(x, _TypeGuardCustom)


class _TypeGuardLookalike:
    pass


_TypeGuardLookalike.__module__ = "builtins"
_TypeGuardLookalike.__name__ = "int"
_TypeGuardLookalike.__qualname__ = "int"


def _qualified_name_spoof_type_check(x: int) -> bool:
    return isinstance(x, _TypeGuardLookalike)


def _shadowed_type_check(x: int, isinstance: Callable[[int, type[int]], bool]) -> bool:
    return isinstance(x, int)


def _lookalike_builtin_call(x: int, classinfo: type[int]) -> bool:
    raise AssertionError("A name-spoofed callback must remain unexecuted")


_lookalike_builtin_call.__module__ = "builtins"
_lookalike_builtin_call.__name__ = "isinstance"
_lookalike_builtin_call.__qualname__ = "isinstance"


def _rebound_type_check(x: int) -> bool:
    return _lookalike_builtin_call(x, int)


@pytest.mark.parametrize("callback", (_builtin_int_check, _aliased_builtin_int_check))
def test_builtin_int_guard_proves_only_its_input_read(
    callback: Callable[[int], bool],
) -> None:
    captured = source_function(callback)
    answer = SourceProof({}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset({"x"})
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize("value", (0, 1, -2, 0.5, False))
def test_builtin_int_guard_never_fixes_a_boolean_from_numeric_metadata(
    value: int | float,
) -> None:
    answer = SourceProof({}, static={"x": value}).callback(
        source_function(_builtin_int_check)
    )
    assert answer.proved
    assert answer.reads == frozenset()
    assert answer.value is _UNKNOWN


def test_builtin_int_type_capture_requires_actual_object_identity() -> None:
    actual = source_function(_builtin_int_check)
    spoofed = source_function(_qualified_name_spoof_type_check)
    real_type = resolve_callable(actual["bindings"]["int"], actual["callable_records"])
    fake_type = resolve_callable(
        spoofed["bindings"]["_TypeGuardLookalike"], spoofed["callable_records"]
    )
    assert real_type["kind"] == "primitive"
    assert real_type["canonical"] == "builtins.int"
    assert real_type["resolution"] == "actual_object_identity"
    assert fake_type != real_type
    assert not SourceProof({}).callback(spoofed).proved


@pytest.mark.parametrize(("regime", "coordinate"), ((0, "x"), (1, "y")))
def test_builtin_int_guard_dispatch_has_only_the_selected_branch_read(
    regime: int, coordinate: str
) -> None:
    answer = SourceProof({}, static={"regime": regime}).callback(
        source_function(_type_guard_dispatch)
    )
    assert answer.proved
    assert answer.reads == frozenset({coordinate})


def test_builtin_int_guard_dispatch_keeps_a_free_selector() -> None:
    answer = SourceProof({}).callback(source_function(_type_guard_dispatch))
    assert answer.proved
    assert answer.reads == frozenset({"regime", "x", "y"})


def _guard_assignment_fallthrough(flag: bool, x: int, y: int) -> int:
    if flag:
        selected = x + 2
    else:
        selected = y * 3
    return selected


def _guard_unsupported_branch(flag: bool, x: int) -> int:
    if flag:
        raise RuntimeError("Unsupported selected branch must not become fallthrough")
    return x


def test_builtin_int_guard_unknown_if_preserves_assignments_into_continuation() -> None:
    answer = SourceProof({}).callback(source_function(_guard_assignment_fallthrough))
    assert answer.proved
    assert answer.reads == frozenset({"flag", "x", "y"})


@pytest.mark.parametrize("static", ({}, {"flag": True}))
def test_builtin_int_guard_unsupported_branch_cannot_become_fallthrough(
    static: dict[str, bool],
) -> None:
    answer = SourceProof({}, static=static).callback(
        source_function(_guard_unsupported_branch)
    )
    assert not answer.proved


@pytest.mark.parametrize(
    "callback",
    (
        _unsupported_float_type_check,
        _unsupported_bool_type_check,
        _unsupported_tuple_type_check,
        _custom_metaclass_type_check,
        _qualified_name_spoof_type_check,
        _shadowed_type_check,
        _rebound_type_check,
    ),
)
def test_unsupported_type_guards_remain_conservative(
    callback: Callable[[int], bool]
    | Callable[[int, Callable[[int, type[int]], bool]], bool],
) -> None:
    answer = SourceProof({}).callback(source_function(callback))
    assert not answer.proved
    assert "x" in answer.reads


class _OpaqueTypeGuardOperand:
    @property
    def payload(self) -> int:
        raise AssertionError("A source proof must not execute an opaque property")


_OPAQUE_TYPE_GUARD_OPERAND = _OpaqueTypeGuardOperand()


def _opaque_operand_type_check() -> bool:
    return isinstance(_OPAQUE_TYPE_GUARD_OPERAND.payload, int)


def test_builtin_int_guard_does_not_certify_an_unproved_operand() -> None:
    answer = SourceProof({}).callback(source_function(_opaque_operand_type_check))
    assert not answer.proved


def test_builtin_int_guard_rejects_a_marker_dictionary_as_a_type() -> None:
    original = source_function(_builtin_int_check)
    marker = dict(
        resolve_callable(original["bindings"]["int"], original["callable_records"])
    )
    with patch.dict(
        _aliased_builtin_int_check.__globals__, {"_INT_TYPE_ALIAS": marker}
    ):
        captured = source_function(_aliased_builtin_int_check)
    reference = captured["bindings"]["_INT_TYPE_ALIAS"]
    assert (
        captured["callable_records"][reference["capture_ref"]]["kind"]
        == "binding-record"
    )
    answer = SourceProof({}).callback(captured)
    assert not answer.proved
    assert "x" in answer.reads


def _builtin_int_type_as_value() -> type[int]:
    return int


def _builtin_int_cast(x: int) -> int:
    return int(x)


def test_builtin_int_guard_does_not_admit_type_values_or_casts() -> None:
    assert (
        not SourceProof({}).callback(source_function(_builtin_int_type_as_value)).proved
    )
    assert not SourceProof({}).callback(source_function(_builtin_int_cast)).proved


def _formal_class_type_check(x: int, int: type[int]) -> bool:
    return isinstance(x, int)


def _default_class_type_check(x: int, int: type[int] = int) -> bool:
    return isinstance(x, int)


def _inline_complex_type_check(x: int, flag: bool) -> bool:
    if flag:
        return isinstance(x, int)
    return False


def _local_class_type_check(x: int, flag: bool) -> bool:
    if flag:
        int = _INT_TYPE_ALIAS
    return isinstance(x, int)


def test_unsupported_type_guards_reject_class_formals_defaults_and_partials() -> None:
    for callback in (
        _formal_class_type_check,
        _default_class_type_check,
        functools.partial(_formal_class_type_check, int=int),
    ):
        answer = SourceProof({}).callback(source_function(callback))
        assert not answer.proved
        assert "x" in answer.reads


def test_unsupported_type_guards_reject_inline_and_local_binding_bodies() -> None:
    for callback in (_inline_complex_type_check, _local_class_type_check):
        answer = SourceProof({}, static={"flag": True}).callback(
            source_function(callback)
        )
        assert not answer.proved
        assert "x" in answer.reads


# French saved C902 -> C466 -> C467 -> C427 -> C56 and Fan's _true_constraint:
# genuine asarray(True) must be read without invoking any fixture body.
_ARRAY_CONSTRUCTOR_ALIAS = jnp.asarray


def _french_true_array_literal() -> Array:
    return jnp.asarray(True)


def _french_true_array_leaf() -> Array:
    return _french_true_array_literal()


def _french_true_array_branch() -> Array:
    return _french_true_array_leaf()


def _array_constructor_alias(value):
    return _ARRAY_CONSTRUCTOR_ALIAS(value)


def _array_with_dtype(value):
    return _ARRAY_CONSTRUCTOR_ALIAS(value, dtype=np.float32)


def _array_keyword_operand(value):
    return _ARRAY_CONSTRUCTOR_ALIAS(a=value)


def _array_positional_dtype(value):
    return _ARRAY_CONSTRUCTOR_ALIAS(value, np.float32)


def _asarray_spoof(value):
    raise AssertionError("A forged array converter must never run")


_asarray_spoof.__module__ = "jax.numpy"
_asarray_spoof.__name__ = "asarray"
_asarray_spoof.__qualname__ = "asarray"


class _AsarrayCallableSpoof:
    __module__ = "jax.numpy"
    __name__ = "asarray"
    __qualname__ = "asarray"

    def __call__(self, value):
        raise AssertionError("A name-spoofed callable object must never run")


def test_asarray_french_no_arg_helper_chain_proves_empty_read_bound_only() -> None:
    captured = source_function(_french_true_array_branch)
    answer = SourceProof({}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset()
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    ("constructor", "canonical"),
    ((np.asarray, "numpy.asarray"), (jnp.asarray, "jax.numpy.asarray")),
)
def test_asarray_actual_alias_retains_input_reads_without_exact_values(
    constructor, canonical: str
) -> None:
    with patch.dict(
        _array_constructor_alias.__globals__, {"_ARRAY_CONSTRUCTOR_ALIAS": constructor}
    ):
        captured = source_function(_array_constructor_alias)
    reference = captured["call_targets"]["_ARRAY_CONSTRUCTOR_ALIAS"]
    target = resolve_callable(reference, captured["callable_records"])
    assert target["kind"] == "primitive"
    assert target["canonical"] == canonical
    assert target["operation"] == "asarray"
    assert target["resolution"] == "actual_object_identity"
    answer = SourceProof({}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset({"value"})
    assert answer.value is _UNKNOWN
    for value in (True, False):
        known = SourceProof({}, static={"value": value}).callback(captured)
        assert known.proved
        assert known.reads == frozenset()
        assert known.value is _UNKNOWN
    unproved = SourceProof({}).callback(
        captured, {"value": _Abstract(frozenset({"unresolved"}), proved=False)}
    )
    assert not unproved.proved
    assert unproved.reads == frozenset({"unresolved"})
    assert unproved.value is _UNKNOWN


@pytest.mark.parametrize("constructor", (np.asarray, jnp.asarray))
@pytest.mark.parametrize(
    "value", (0.1, 16_777_217.0, 2**80, -(2**80), 0, 1, [True, False])
)
def test_asarray_numeric_or_nonscalar_metadata_never_gains_exact_identity(
    constructor, value
) -> None:
    with patch.dict(
        _array_constructor_alias.__globals__, {"_ARRAY_CONSTRUCTOR_ALIAS": constructor}
    ):
        captured = source_function(_array_constructor_alias)
    answer = SourceProof({}, static={"value": value}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset()
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize("callback", (_array_keyword_operand, _array_positional_dtype))
def test_asarray_unsupported_argument_shapes_remain_unproved(callback) -> None:
    answer = SourceProof({}).callback(source_function(callback))
    assert not answer.proved
    assert "value" in answer.reads


@pytest.mark.parametrize("fake", (_asarray_spoof, _AsarrayCallableSpoof()))
def test_asarray_rebound_module_lookalike_does_not_gain_primitive_authority(
    fake,
) -> None:
    with (
        patch.dict(vars(jnp), {"asarray": fake}),
        patch.dict(
            _array_constructor_alias.__globals__, {"_ARRAY_CONSTRUCTOR_ALIAS": fake}
        ),
    ):
        captured = source_function(_array_constructor_alias)
    target = resolve_callable(
        captured["call_targets"]["_ARRAY_CONSTRUCTOR_ALIAS"],
        captured["callable_records"],
    )
    assert target.get("kind") != "primitive"
    assert not SourceProof({}).callback(captured).proved


def test_asarray_marker_dictionary_is_data_not_an_authenticated_callee() -> None:
    marker = {
        "kind": "primitive",
        "operation": "asarray",
        "canonical": "jax.numpy.asarray",
        "resolution": "actual_object_identity",
    }
    with patch.dict(
        _array_constructor_alias.__globals__, {"_ARRAY_CONSTRUCTOR_ALIAS": marker}
    ):
        captured = source_function(_array_constructor_alias)
    assert not SourceProof({}, static={"value": True}).callback(captured).proved


def _array_boolean_arithmetic_dependency(value):
    if jnp.asarray(True) + jnp.asarray(True) == 2:
        return 0
    return value


def test_asarray_boolean_arithmetic_cannot_erase_a_later_input() -> None:
    # Boolean-array addition need not equal Python's True + True. The source
    # reader must not choose that branch using an invented Python scalar value.
    answer = SourceProof({}).callback(
        source_function(_array_boolean_arithmetic_dependency)
    )
    assert answer.proved
    assert answer.reads == frozenset({"value"})
    assert answer.value is _UNKNOWN


_NUMERIC_ALIAS: Callable[..., Array] = jnp.maximum
_ARRAY_DTYPE = jnp.int32


def _numeric_unary(value):
    return _NUMERIC_ALIAS(value)


def _numeric_binary(left, right):
    return _NUMERIC_ALIAS(left, right)


def _numeric_clip(value, lower, upper):
    return _NUMERIC_ALIAS(value, lower, upper)


def _numeric_stack(left, right):
    return _NUMERIC_ALIAS((left, right))


def _numeric_sum_slice(values):
    return _NUMERIC_ALIAS(values[2:])


def _numeric_unpacked(left, right, options):
    return _NUMERIC_ALIAS(left, right, **options)


def _cast_from_asarray(value):
    return jnp.asarray(value).astype(jnp.int32)


def _cast_from_where(flag, left, right):
    return jnp.where(flag, left, right).astype(jnp.int32)


def _cast_from_unknown_receiver(value):
    return value.astype(jnp.int32)


def _cast_with_formal_dtype(value, dtype):
    return jnp.asarray(value).astype(dtype)


def _cast_with_bound_dtype(value):
    return jnp.asarray(value).astype(_ARRAY_DTYPE)


def _cast_shadowed_module(value, jnp):
    return value.astype(jnp.int32)


def _numeric_spoof(value):
    raise AssertionError("A spoofed numeric primitive must not execute")


_numeric_spoof.__module__ = "jax.numpy"
_numeric_spoof.__name__ = "exp"
_numeric_spoof.__qualname__ = "exp"


@pytest.mark.parametrize(
    ("constructor", "callback", "reads"),
    (
        (jnp.maximum, _numeric_binary, {"left", "right"}),
        (np.maximum, _numeric_binary, {"left", "right"}),
        (jnp.exp, _numeric_unary, {"value"}),
        (jnp.log, _numeric_unary, {"value"}),
        (jnp.sqrt, _numeric_unary, {"value"}),
        (jnp.clip, _numeric_clip, {"value", "lower", "upper"}),
        (jnp.stack, _numeric_stack, {"left", "right"}),
        (jnp.sum, _numeric_sum_slice, {"values"}),
    ),
)
def test_french_numeric_primitives_keep_all_operand_reads_unknown(
    constructor, callback, reads
) -> None:
    with patch.dict(callback.__globals__, {"_NUMERIC_ALIAS": constructor}):
        captured = source_function(callback)
    answer = SourceProof({}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset(reads)
    assert answer.value is _UNKNOWN


def test_numeric_primitive_does_not_discard_unpacked_keyword_dependencies() -> None:
    captured = source_function(_numeric_unpacked)
    answer = SourceProof({}).callback(captured)
    assert not answer.proved
    assert {"left", "right", "options"} <= answer.reads


def test_array_dtype_only_keyword_is_authenticated_and_remains_unknown() -> None:
    captured = source_function(_array_with_dtype)
    assert "np.float32" in captured["dtype_references"]
    answer = SourceProof({}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset({"value"})
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    ("callback", "reads"),
    (
        (_cast_from_asarray, {"value"}),
        (_cast_from_where, {"flag", "left", "right"}),
        (_cast_with_bound_dtype, {"value"}),
    ),
)
def test_astype_requires_a_captured_native_array_producer(callback, reads) -> None:
    answer = SourceProof({}).callback(source_function(callback))
    assert answer.proved
    assert answer.reads == frozenset(reads)
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    ("callback", "reads"),
    (
        (_cast_from_unknown_receiver, {"value"}),
        (_cast_with_formal_dtype, {"value", "dtype"}),
        (_cast_shadowed_module, {"value", "jnp"}),
    ),
)
def test_astype_unknown_receiver_or_dtype_stays_unproved_and_retains_reads(
    callback, reads
) -> None:
    answer = SourceProof({}).callback(source_function(callback))
    assert not answer.proved
    assert frozenset(reads) <= answer.reads


def test_astype_forged_dtype_marker_is_not_a_captured_class() -> None:
    marker = {
        "kind": "primitive",
        "operation": "array_dtype",
        "canonical": "numpy.int32",
        "resolution": "actual_object_identity",
        "implementation_seal": "captured-native-numeric-object-code/v1",
    }
    with patch.dict(_cast_with_bound_dtype.__globals__, {"_ARRAY_DTYPE": marker}):
        captured = source_function(_cast_with_bound_dtype)
    assert not SourceProof({}).callback(captured).proved


def test_numeric_primitive_identity_does_not_follow_a_rebound_library_name() -> None:
    with (
        patch.dict(vars(jnp), {"exp": _numeric_spoof}),
        patch.dict(_numeric_unary.__globals__, {"_NUMERIC_ALIAS": _numeric_spoof}),
    ):
        captured = source_function(_numeric_unary)
    assert not SourceProof({}).callback(captured).proved


def test_numeric_pjit_same_object_changed_executable_code_is_not_a_primitive() -> None:
    target = jnp.exp
    implementation = vars(target)["_fun"]
    original_code = implementation.__code__
    assert implementation.__closure__ is None
    replacement = _numeric_spoof.__code__.replace(
        co_filename=original_code.co_filename,
        co_firstlineno=original_code.co_firstlineno,
        co_qualname=original_code.co_qualname,
    )
    try:
        implementation.__code__ = replacement
        with patch.dict(_numeric_unary.__globals__, {"_NUMERIC_ALIAS": target}):
            captured = source_function(_numeric_unary)
    finally:
        implementation.__code__ = original_code
    assert not SourceProof({}).callback(captured).proved


def test_numeric_ufunc_same_method_changed_code_is_not_a_primitive() -> None:
    target = jnp.maximum
    method = inspect.getattr_static(type(target), "__call__")
    original_code = method.__code__
    assert method.__closure__ is None
    replacement = _numeric_spoof.__code__.replace(
        co_filename=original_code.co_filename,
        co_firstlineno=original_code.co_firstlineno,
        co_qualname=original_code.co_qualname,
    )
    try:
        method.__code__ = replacement
        with patch.dict(_numeric_binary.__globals__, {"_NUMERIC_ALIAS": target}):
            captured = source_function(_numeric_binary)
    finally:
        method.__code__ = original_code
    assert not SourceProof({}).callback(captured).proved


def _spread_leaf(value):
    return value


def _helper_dict_spread(history):
    return _spread_leaf(**{"value": history})


def _helper_positional_spread(history):
    return _spread_leaf(*(history,))


def _helper_dynamic_spread(history, options):
    return _spread_leaf(**options)


def _numeric_positional_spread(left, right):
    return _NUMERIC_ALIAS(*(left, right))


@pytest.mark.parametrize(
    ("callback", "reads"),
    (
        (_helper_dict_spread, {"history"}),
        (_helper_positional_spread, {"history"}),
        (_helper_dynamic_spread, {"history", "options"}),
        (_numeric_positional_spread, {"left", "right"}),
    ),
)
def test_argument_spreads_remain_unproved_before_recognized_call_paths(
    callback, reads
) -> None:
    proof = SourceProof({})
    answer = proof.callback(source_function(callback))
    assert not answer.proved
    assert frozenset(reads) <= answer.reads


# These source-proof fixtures are inspected, never economically evaluated.


class _SizeReceiver(Protocol):
    @property
    def size(self) -> int: ...


class _NeverReadSize:
    @property
    def size(self) -> int:
        raise AssertionError("The source checker must not evaluate size properties")


@beartype
def _checked_floatnd_size(table: FloatND) -> int:
    return table[:, 0].size


@beartype(conf=BeartypeConf(strategy=BeartypeStrategy.On, is_pep484_tower=True))
def _checked_floatnd_sum_size(table: FloatND) -> Array:
    return jnp.sum(table[:, 0], axis=-1) + table[:, 0].size


def _plain_calls_checked_tax_helper(table: FloatND) -> Array:
    return _checked_floatnd_sum_size(table)


def _plain_floatnd_size(table: FloatND) -> int:
    return table[:, 0].size


def _arbitrary_size(receiver: _SizeReceiver) -> int:
    return receiver.size


def _array_identity(value: Array) -> Array:
    return value


@beartype
def _checked_floatnd_unknown_result_size(table: FloatND) -> int:
    return _array_identity(table).size


@beartype
def _checked_floatnd_reassigned_size(table: FloatND) -> int:
    table = _array_identity(table)
    return table.size


@beartype
def _checked_floatnd_alias_size(table: FloatND) -> int:
    alias = table
    return alias.size


@beartype(conf=BeartypeConf(strategy=BeartypeStrategy.O0))
def _unchecked_floatnd_size(table: FloatND) -> int:
    return table[:, 0].size


@beartype(conf=BeartypeConf(violation_param_type=UserWarning))
def _warning_floatnd_size(table: FloatND) -> int:
    return table[:, 0].size


@beartype(conf=BeartypeConf(hint_overrides=FrozenDict({FloatND: object})))
def _overridden_floatnd_size(table: FloatND) -> int:
    return table[:, 0].size


@pytest.mark.parametrize(
    "callback",
    (_checked_floatnd_size, _checked_floatnd_sum_size, _plain_calls_checked_tax_helper),
)
def test_authenticated_floatnd_input_supports_size_bound_not_value(
    callback: Callable[..., int | Array],
) -> None:
    captured = source_function(callback)
    answer = SourceProof({}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset({"table"})
    assert answer.value is _UNKNOWN
    unproved = SourceProof({}).callback(
        captured, {"table": _Abstract(frozenset({"unresolved"}), proved=False)}
    )
    assert not unproved.proved
    assert unproved.reads == frozenset({"unresolved"})
    assert unproved.value is _UNKNOWN


@pytest.mark.parametrize(
    "callback",
    (
        _plain_floatnd_size,
        _unchecked_floatnd_size,
        _warning_floatnd_size,
        _overridden_floatnd_size,
        _checked_floatnd_unknown_result_size,
        _checked_floatnd_reassigned_size,
        _checked_floatnd_alias_size,
    ),
)
def test_array_input_fact_is_not_an_unchecked_annotation_or_result_guess(
    callback: Callable[..., int],
) -> None:
    answer = SourceProof({}).callback(source_function(callback))
    assert not answer.proved
    assert "table" in answer.reads
    assert answer.value is _UNKNOWN


def test_arbitrary_size_property_is_not_evaluated_or_certified() -> None:
    answer = SourceProof({}, static={"receiver": _NeverReadSize()}).callback(
        source_function(_arbitrary_size)
    )
    assert not answer.proved
    assert answer.value is _UNKNOWN


def test_rebound_native_floatnd_alias_does_not_grant_array_inputs() -> None:
    constructor: Callable[[str, object], TypeAliasType] = vars(typing)["TypeAliasType"]  # noqa: PAN001 - runtime alias values are arbitrary expressions
    assert constructor is TypeAliasType
    replacement: TypeAliasType = constructor("FloatND", object)
    with patch.dict(vars(native_typing), {"FloatND": replacement}):
        captured = source_function(_checked_floatnd_size)
    assert not SourceProof({}).callback(captured).proved


def test_mutated_cached_floatnd_class_does_not_grant_array_inputs() -> None:
    alias: TypeAliasType = vars(native_typing)["FloatND"]
    with patch.object(alias.__value__, "array_type", object):
        captured = source_function(_checked_floatnd_size)
    assert not SourceProof({}).callback(captured).proved


def _array_input_resealed(captured, mutation: str):
    """Reuse the existing genuine-wire rehasher after a scoped metadata mutation."""
    from copy import deepcopy
    from _lcm_regime_finder.semantic_check import _CallableRecords
    from tests.regime_finder.test_fibre_source import _native_wire_retyped_reference

    table = deepcopy(captured["callable_records"])
    wrapper = table[captured["capture_ref"]]
    target = table[wrapper["wrapped_target"]["capture_ref"]]
    alias_ref = target["parameter_annotation_refs"]["table"]
    alias = table[alias_ref["capture_ref"]]
    if mutation == "old_capture":
        for record in table.values():
            for name in (
                "native_array_annotation",
                "parameter_annotation_refs",
                "annotation_bindings",
                "wrapped_parameter_annotation_refs",
                "wrapped_input_config",
                "beartype_input_configuration",
            ):
                record.pop(name, None)
    elif mutation == "wrong_alias_authority":
        alias["native_array_annotation"]["array_type"] = "builtins.object"
    elif mutation == "wrong_binding_type":
        alias["binding_type"] = "unsupported.ArrayAlias"
    elif mutation == "missing_source_binding":
        target["annotation_bindings"] = {}
    elif mutation == "wrong_effective_argument":
        wrapper["wrapped_parameter_annotation_refs"] = {"other": alias_ref}
    elif mutation == "missing_target_annotation":
        target["parameter_annotation_refs"] = {}
    elif mutation == "config_none":
        table[wrapper["wrapped_input_config"]["capture_ref"]][
            "beartype_input_configuration"
        ] = None
    elif mutation == "config_list":
        table[wrapper["wrapped_input_config"]["capture_ref"]][
            "beartype_input_configuration"
        ] = []
    elif mutation == "defaults_none":
        wrapper["defaults"] = None
    elif mutation == "defaults_list":
        wrapper["defaults"] = []
    elif mutation == "missing_target":
        wrapper.pop("wrapped_target")
    elif mutation.startswith("tower_") or mutation == "old_array_config":
        settings = table[wrapper["wrapped_input_config"]["capture_ref"]][
            "beartype_input_configuration"
        ]
        if mutation == "tower_changed_union":
            settings["hint_overrides"][0][1] = ["builtins.object"]
        elif mutation == "tower_extra_override":
            settings["hint_overrides"].append(
                ["lcm.typing.FloatND", ["builtins.object"]]
            )
        elif mutation == "tower_reordered_union":
            settings["hint_overrides"][0][1] = ["builtins.int", "builtins.float"]
        elif mutation == "tower_nonstring_component":
            settings["hint_overrides"][0][1][1] = True
        elif mutation == "tower_false_flag":
            settings["numeric_tower"] = False
        elif mutation == "tower_wrong_source":
            settings["tower_source_sha256"] = "0" * 64
        elif mutation == "old_array_config":
            settings.pop("numeric_tower")
            settings.pop("tower_source_sha256")
        else:
            raise AssertionError("Unknown tower metadata control")
    else:
        raise AssertionError("Unknown metadata-control mutation")
    records, _ = _native_wire_retyped_reference(
        _CallableRecords(table), alias_ref, alias["binding_type"]
    )
    return records, records.records[captured["capture_ref"]]


@pytest.mark.parametrize(
    "mutation",
    (
        "old_capture",
        "wrong_alias_authority",
        "wrong_binding_type",
        "missing_source_binding",
        "wrong_effective_argument",
        "missing_target_annotation",
        "config_none",
        "config_list",
        "defaults_none",
        "defaults_list",
        "missing_target",
    ),
)
def test_array_input_facts_require_the_actual_resealed_annotation_and_config(
    mutation: str,
) -> None:
    records, original = _array_input_resealed(
        source_function(_checked_floatnd_size), mutation
    )
    assert not records.errors
    answer = SourceProof({}, callable_records=records).callback(original)
    assert not answer.proved
    assert answer.value is _UNKNOWN


@beartype
def _checked_floatnd_default_none(
    table: FloatND = None,  # ty: ignore[invalid-parameter-default] - deliberate unchecked-default counterexample
) -> int:
    return table.size


def _calls_checked_with_omitted_default() -> int:
    return _checked_floatnd_default_none()


def test_array_fact_is_not_granted_to_beartypes_unchecked_omitted_default() -> None:
    answer = SourceProof({}).callback(
        source_function(_calls_checked_with_omitted_default)
    )
    assert not answer.proved
    assert answer.value is _UNKNOWN


def _sum_with_axis(value: Array, axis: int | tuple[int, ...] | None) -> Array:
    return jnp.sum(value, axis=axis)


def _sum_with_unsupported_keepdims(value: Array) -> Array:
    return jnp.sum(value, axis=-1, keepdims=True)


@pytest.mark.parametrize("axis", (-1, 0, (0, -1), (), None))
def test_sum_exact_axis_metadata_retains_reads_and_unknown_result(
    axis: int | tuple[int, ...] | None,
) -> None:
    captured = source_function(_sum_with_axis)
    answer = SourceProof({}, static={"axis": axis}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset({"value"})
    assert answer.value is _UNKNOWN
    unproved = SourceProof({}, static={"axis": axis}).callback(
        captured, {"value": _Abstract(frozenset({"unresolved"}), proved=False)}
    )
    assert not unproved.proved
    assert unproved.reads == frozenset({"unresolved"})
    assert unproved.value is _UNKNOWN


@pytest.mark.parametrize("axis", (True, False, 0.0, (0, True), (0, 1.0), [0]))
def test_sum_rejects_nonexact_or_bool_axis_metadata(
    axis: int | float | tuple[int | float, ...] | list[int],
) -> None:
    answer = SourceProof({}, static={"axis": axis}).callback(
        source_function(_sum_with_axis)
    )
    assert not answer.proved
    assert "value" in answer.reads
    assert answer.value is _UNKNOWN


def test_sum_unknown_axis_and_unsupported_keywords_stay_unproved() -> None:
    dynamic = SourceProof({}).callback(source_function(_sum_with_axis))
    assert not dynamic.proved
    assert dynamic.reads == frozenset({"value", "axis"})
    assert dynamic.value is _UNKNOWN
    unsupported = SourceProof({}).callback(
        source_function(_sum_with_unsupported_keepdims)
    )
    assert not unsupported.proved
    assert "value" in unsupported.reads
    assert unsupported.value is _UNKNOWN


@beartype(
    conf=BeartypeConf(
        strategy=BeartypeStrategy.On,
        is_pep484_tower=True,
        hint_overrides=FrozenDict({FloatND: object}),
    )
)
def _tower_with_floatnd_override(table: FloatND) -> int:
    return table[:, 0].size


@beartype(
    conf=BeartypeConf(
        strategy=BeartypeStrategy.On,
        is_pep484_tower=True,
        hint_overrides=FrozenDict({Array: object}),
    )
)
def _tower_with_array_override(table: FloatND) -> int:
    return table[:, 0].size


@pytest.mark.parametrize(
    "callback", (_tower_with_floatnd_override, _tower_with_array_override)
)
def test_canonical_numeric_tower_does_not_allow_additional_array_overrides(
    callback: Callable[[FloatND], int],
) -> None:
    answer = SourceProof({}).callback(source_function(callback))
    assert not answer.proved
    assert "table" in answer.reads
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    "mutation",
    (
        "tower_changed_union",
        "tower_extra_override",
        "tower_reordered_union",
        "tower_nonstring_component",
        "tower_false_flag",
        "tower_wrong_source",
    ),
)
def test_canonical_numeric_tower_requires_exact_resealed_builtin_facts(
    mutation: str,
) -> None:
    records, original = _array_input_resealed(
        source_function(_checked_floatnd_sum_size), mutation
    )
    assert not records.errors
    answer = SourceProof({}, callable_records=records).callback(original)
    assert not answer.proved
    assert answer.value is _UNKNOWN


def test_old_array_config_without_numeric_tower_facts_is_not_upgraded() -> None:
    records, original = _array_input_resealed(
        source_function(_checked_floatnd_size), "old_array_config"
    )
    assert not records.errors
    assert not SourceProof({}, callable_records=records).callback(original).proved


class _OverrideEqualityPoison:
    def __hash__(self) -> int:
        return hash(float)

    def __eq__(self, other: object) -> bool:
        raise AssertionError(
            "Numeric-tower metadata must not call application equality"
        )


def test_numeric_tower_override_observation_rejects_unknown_keys_without_equality() -> (
    None
):
    from beartype.roar import BeartypeCallHintParamViolation
    from _lcm_regime_finder.callable_identity import _beartype_input_config

    # A malformed configuration object isolates the collector from BeartypeConf's
    # own interning/sanitization; it is never attached to a callable or granted authority.
    conf = object.__new__(BeartypeConf)
    object.__setattr__(
        conf,
        "_hint_overrides",
        FrozenDict(
            {
                _OverrideEqualityPoison(): float | int,
                complex: complex | float | int,
            }
        ),
    )
    object.__setattr__(conf, "_is_pep484_tower", True)
    object.__setattr__(conf, "_strategy", BeartypeStrategy.On)
    object.__setattr__(conf, "_violation_param_type", BeartypeCallHintParamViolation)
    object.__setattr__(conf, "_is_violation_param_warn", False)
    observed = _beartype_input_config(conf)
    assert observed is not None
    assert observed["hint_overrides"] is None


def _minimum_direct(left: Array, right: Array) -> Array:
    return jnp.minimum(left, right)


def _minimum_known_operands() -> Array:
    return jnp.minimum(1.25, 2.5)


def _minimum_branch(history: Array) -> Array | float:
    if jnp.minimum(0, 1) == 0:
        return 0.0
    return history


def _minimum_cast(left: Array, right: Array) -> Array:
    return jnp.minimum(left, right).astype(jnp.int32)


def _minimum_keyword(left: Array, right: Array, options: Array) -> Array:
    return _NUMERIC_ALIAS(left, right, out=options)


def _minimum_where(left: Array, right: Array, options: Array) -> Array:
    return _NUMERIC_ALIAS(left, right, where=options)


def _minimum_spoof(left: Array, right: Array) -> Array:
    raise AssertionError("A spoofed minimum must not execute")


_minimum_spoof.__module__ = "jax.numpy"
_minimum_spoof.__name__ = "minimum"
_minimum_spoof.__qualname__ = "minimum"


@pytest.mark.parametrize("callback", [_minimum_direct, _numeric_binary])
def test_minimum_fresh_capture_keeps_both_operands_and_unknown_result(
    callback: FunctionType,
) -> None:
    with patch.dict(callback.__globals__, {"_NUMERIC_ALIAS": jnp.minimum}):
        captured = source_function(callback)
    callee = "jnp.minimum" if callback is _minimum_direct else "_NUMERIC_ALIAS"
    target = resolve_callable(
        captured["call_targets"][callee], captured["callable_records"]
    )
    assert (
        target["kind"],
        target["operation"],
        target["canonical"],
        target["resolution"],
        target["implementation_seal"],
    ) == (
        "primitive",
        "minimum",
        "jax.numpy.minimum",
        "actual_object_identity",
        "captured-native-numeric-object-code/v1",
    )
    answer = SourceProof({}).callback(captured)
    assert answer.proved
    assert answer.reads == frozenset({"left", "right"})
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    "callback", [_minimum_known_operands, _minimum_branch, _minimum_cast]
)
def test_minimum_never_grants_a_python_value_or_selects_a_branch(
    callback: FunctionType,
) -> None:
    answer = SourceProof({}).callback(source_function(callback))
    assert answer.proved
    assert answer.reads == (
        frozenset()
        if callback is _minimum_known_operands
        else frozenset({"history"})
        if callback is _minimum_branch
        else frozenset({"left", "right"})
    )
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    ("callback", "reads"),
    [
        (_numeric_unary, frozenset({"value"})),
        (_numeric_clip, frozenset({"value", "lower", "upper"})),
        (_minimum_keyword, frozenset({"left", "right", "options"})),
        (_minimum_where, frozenset({"left", "right", "options"})),
        (_numeric_unpacked, frozenset({"left", "right", "options"})),
    ],
)
def test_minimum_unsupported_arity_options_or_unpacking_retains_all_reads(
    callback: FunctionType,
    reads: frozenset[str],
) -> None:
    with patch.dict(callback.__globals__, {"_NUMERIC_ALIAS": jnp.minimum}):
        captured = source_function(callback)
    answer = SourceProof({}).callback(captured)
    assert not answer.proved
    assert frozenset(reads) <= answer.reads
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize("forgery", ["rebound_function", "marker_dictionary"])
def test_minimum_capture_rejects_rebound_function_or_forged_primitive_marker(
    forgery: typing.Literal["rebound_function", "marker_dictionary"],
) -> None:
    marker = {
        "kind": "primitive",
        "operation": "minimum",
        "canonical": "jax.numpy.minimum",
        "resolution": "actual_object_identity",
        "implementation_seal": "captured-native-numeric-object-code/v1",
    }
    replacement = _minimum_spoof if forgery == "rebound_function" else marker
    with (
        patch.dict(vars(jnp), {"minimum": replacement}),
        patch.dict(_numeric_binary.__globals__, {"_NUMERIC_ALIAS": replacement}),
    ):
        captured = source_function(_numeric_binary)
    answer = SourceProof({}).callback(captured)
    assert not answer.proved
    assert {"left", "right"} <= answer.reads
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    "field", ["operation", "canonical", "resolution", "implementation_seal"]
)
def test_minimum_changed_captured_primitive_marker_is_not_authority(
    field: typing.Literal[
        "operation", "canonical", "resolution", "implementation_seal"
    ],
) -> None:
    from copy import deepcopy

    captured = deepcopy(source_function(_minimum_direct))
    reference = captured["call_targets"]["jnp.minimum"]
    target = captured["callable_records"][reference["capture_ref"]]
    assert target["kind"] == "primitive" and target["operation"] == "minimum"
    target[field] = "unverified"
    answer = SourceProof({}).callback(captured)
    assert not answer.proved


def test_minimum_numpy_uses_exactly_the_existing_maximum_authority_boundary() -> None:
    from _lcm_regime_finder.callable_identity import _primitive

    maximum = _primitive(np.maximum)
    minimum = _primitive(np.minimum)
    assert (minimum is None) is (maximum is None)
    with patch.dict(_numeric_binary.__globals__, {"_NUMERIC_ALIAS": np.minimum}):
        captured = source_function(_numeric_binary)
    answer = SourceProof({}).callback(captured)
    if maximum is None:
        assert not answer.proved
        assert {"left", "right"} <= answer.reads
    else:
        assert minimum is not None
        assert minimum["operation"] == "minimum"
        assert minimum["canonical"] == "numpy.minimum"
        assert answer.proved and answer.reads == frozenset({"left", "right"})
    assert answer.value is _UNKNOWN


def test_minimum_rejects_changed_same_object_ufunc_method_code() -> None:
    target = jnp.minimum
    method = inspect.getattr_static(type(target), "__call__")
    assert isinstance(method, FunctionType)
    original_code = method.__code__
    assert method.__closure__ is None
    replacement = _minimum_spoof.__code__.replace(
        co_filename=original_code.co_filename,
        co_firstlineno=original_code.co_firstlineno,
        co_qualname=original_code.co_qualname,
    )
    try:
        method.__code__ = replacement
        captured = source_function(_minimum_direct)
    finally:
        method.__code__ = original_code
    assert method.__code__ is original_code
    answer = SourceProof({}).callback(captured)
    assert not answer.proved
    assert {"left", "right"} <= answer.reads


@pytest.mark.parametrize("behavior", ["equal", "raise"])
def test_native_numeric_defaults_reject_hostile_type_without_equality(
    behavior: typing.Literal["equal", "raise"],
) -> None:
    from _lcm.solution.fingerprint import _is_captured_jax_python_function
    from _lcm_regime_finder.callable_identity import _primitive

    calls: list[str] = []

    class HostileMeta(type):
        def __eq__(cls, other: object) -> bool:
            calls.append("equality")
            if behavior == "raise":
                raise AssertionError("Opaque numeric-default type equality must not run")
            return True

        __hash__ = type.__hash__

    class OpaqueDefault(metaclass=HostileMeta):
        pass

    target = jnp.asarray
    assert isinstance(target, FunctionType)
    assert _is_captured_jax_python_function(target)
    before = _primitive(target)
    assert before is not None and before["operation"] == "asarray"
    original_defaults = target.__defaults__
    original_kwdefaults = target.__kwdefaults__
    assert original_defaults
    try:
        target.__defaults__ = (OpaqueDefault(), *original_defaults[1:])
        assert _is_captured_jax_python_function(target)
        assert _primitive(target) is None
        assert calls == []
    finally:
        target.__defaults__ = original_defaults
        target.__kwdefaults__ = original_kwdefaults
        assert target.__defaults__ is original_defaults
        assert target.__kwdefaults__ is original_kwdefaults
        restored = _primitive(target)
        assert restored is not None and restored["operation"] == "asarray"


@pytest.mark.parametrize("behavior", ["equal", "raise"])
def test_native_numeric_static_property_rejects_hostile_type_without_equality(
    behavior: typing.Literal["equal", "raise"],
) -> None:
    from _lcm_regime_finder.callable_identity import _primitive

    calls: list[str] = []

    class HostileMeta(type):
        def __eq__(cls, other: object) -> bool:
            calls.append("equality")
            if behavior == "raise":
                raise AssertionError("Opaque numeric-property type equality must not run")
            return True

        __hash__ = type.__hash__

    class OpaqueProperty(metaclass=HostileMeta):
        pass

    target = jnp.minimum
    before = _primitive(target)
    assert before is not None and before["operation"] == "minimum"
    properties = vars(target)["_ufunc__static_props"]
    assert type(properties) is dict
    original_identity = properties["identity"]
    assert original_identity is None
    try:
        properties["identity"] = OpaqueProperty()
        assert _primitive(target) is None
        assert calls == []
    finally:
        properties["identity"] = original_identity
        assert vars(target)["_ufunc__static_props"] is properties
        assert properties["identity"] is original_identity
        restored = _primitive(target)
        assert restored is not None and restored["operation"] == "minimum"
