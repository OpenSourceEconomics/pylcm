"""Source-backed binding regressions and independent Python call oracles.

No Model, solve, simulation, or Bellman evaluation occurs. The independent
oracle invokes only the explicit tiny scalar fixtures below. Native numerical
integration still requires the pinned production environment.
"""

import pytest
from dataclasses import dataclass
import functools
import inspect

from _lcm_regime_finder.declarations import (
    DeclarationError,
    ScalarSource,
    normalize_declarations,
)
from _lcm_regime_finder.expressions import partial as evaluate_expression
from _lcm_regime_finder.model_snapshot import read_declarations, source_function
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder.semantic_check import SourceProof
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
