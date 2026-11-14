"""Source-backed binding regressions and independent Python call oracles.

No Model, solve, simulation, or Bellman evaluation occurs. The independent
oracle invokes only the explicit tiny scalar fixtures below. Native numerical
integration still requires the pinned production environment.
"""

import pytest
from dataclasses import dataclass
import functools
import inspect
import builtins
from abc import ABCMeta
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
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
from _lcm_regime_finder.semantic_check import SourceProof, _UNKNOWN
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


def test_positional_partial_helper_uses_effective_signature(
    *, subtests: pytest.Subtests
) -> None:
    for fixed in (-3, 0, 10):
        callback = positional_callback(functools.partial(difference, fixed))
        for history in (-2, 0, 7):
            with subtests.test(fixed=fixed, history=history):
                _check_scalar(callback, history)


def test_partial_keyword_default_and_explicit_override(
    *, subtests: pytest.Subtests
) -> None:
    for default in (-2, 0, 4):
        helper = functools.partial(difference, right=default)
        callbacks = [keyword_default(helper)]
        callbacks += [keyword_override(helper, override) for override in (-1, 0, 5)]
        for callback in callbacks:
            for history in (-3, 0, 6):
                with subtests.test(default=default, history=history, callback=callback):
                    _check_scalar(callback, history)


def test_mixed_nested_partials_keep_positional_and_default_roles(
    *, subtests: pytest.Subtests
) -> None:
    for first in (-2, 0, 5):
        for third in (-3, 2):
            helper = functools.partial(functools.partial(triple, first), third=third)
            callback = positional_callback(helper)
            for history in (-1, 0, 4):
                with subtests.test(first=first, third=third, history=history):
                    _check_scalar(callback, history)


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
