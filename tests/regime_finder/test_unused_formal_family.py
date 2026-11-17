"""Unknown helper bodies retain actual inputs, not unused outer formals."""

import pytest
from copy import deepcopy
from dataclasses import dataclass
import functools
import math
from collections.abc import Callable

from _lcm_regime_finder import find_regimes as _find_regimes
from functools import partial

# These regressions exercise the optional stronger semantic audit, not the
# Model-graph contract whose primitives retain their declared dependencies.
find_regimes = partial(_find_regimes, analysis_scope="full_semantics")
from _lcm_regime_finder.declarations import normalize_declarations
from tests.regime_finder.test_round5_local_obligations import native_fixture


def opaque_conversion(a: float, dtype: str | None = None) -> float:
    if dtype is None:
        return a
    return a


def unused_outer(history: float) -> float:
    return opaque_conversion(0.0)


def actual_outer(history: float) -> float:
    return opaque_conversion(history)


def transitive_outer(history: float) -> float:
    derived = history + 2
    return opaque_conversion(derived)


def unused_builtin(history: float) -> float:
    return math.sqrt(0.0)


def nested_helper(a: float) -> float:
    return opaque_conversion(a)


def nested_outer(history: float) -> float:
    return nested_helper(history)


def closed_helper(offset: int) -> Callable[[float], float]:
    def convert(a: float) -> float:
        if offset:
            return a + offset
        return a

    return convert


WITH_CLOSURE = closed_helper(3)
WITH_PARTIAL_DEFAULT = functools.partial(opaque_conversion, dtype=None)


def closure_outer(history: float) -> float:
    return WITH_CLOSURE(history)


def partial_outer(history: float) -> float:
    return WITH_PARTIAL_DEFAULT(history)


@dataclass
class Receiver:
    offset: int

    def convert(self, a: float) -> float:
        if self.offset:
            return a + self.offset
        return a


RECEIVER = Receiver(3)


def receiver_outer(history: float) -> float:
    return RECEIVER.convert(history)


def variadic_helper(*arguments: float) -> float:
    return arguments[0]


def variadic_outer(history: float) -> float:
    return variadic_helper(0)


def test_unused_outer_formal_does_not_create_a_false_family_disagreement() -> None:
    # Independent literal oracle: this fixture returns zero for every input.
    assert ([unused_outer(value) for value in (-3, 0, 9)]) == ([0.0, 0.0, 0.0])
    report = find_regimes(
        native_fixture(callback=unused_outer),
        facts_backend="metadata",
    )
    assert (report["semantic_check"]["errors"]) == ([])
    assert (report["status"]) == ("UNRESOLVED")
    callback = next(
        value
        for value in report["analysis"]["declarations"]["callbacks"].values()
        if (value.get("source") or "").startswith("def unused_outer")
    )
    assert (callback["reads"]) == ([])
    assert (callback["status"]) == ("opaque")
    assert any(
        row["kind"] == "callback_interpretation"
        for row in report["analysis"]["obligations"]
    )


def test_actual_and_transitive_inputs_stay_in_the_family(
    *, subtests: pytest.Subtests
) -> None:
    for function, expected in (
        (actual_outer, [-3, 0, 9]),
        (transitive_outer, [-1, 2, 11]),
    ):
        with subtests.test(function=function.__name__):
            assert ([function(value) for value in (-3, 0, 9)]) == (expected)
            report = find_regimes(
                native_fixture(callback=function),
                facts_backend="metadata",
            )
            assert (report["semantic_check"]["errors"]) == ([])
            callback = next(
                value
                for value in report["analysis"]["declarations"]["callbacks"].values()
                if (value.get("source") or "").startswith("def " + function.__name__)
            )
            assert (callback["reads"]) == (["history"])


def test_captured_signature_with_unavailable_body_keeps_actual_input_scope() -> None:
    assert ([unused_builtin(value) for value in (-3, 0, 9)]) == ([0.0, 0.0, 0.0])
    report = find_regimes(
        native_fixture(callback=unused_builtin),
        facts_backend="metadata",
    )
    assert (report["semantic_check"]["errors"]) == ([])
    assert (report["status"]) == ("UNRESOLVED")


def test_nested_closure_receiver_and_partial_inputs_remain_dependencies(
    *, subtests: pytest.Subtests
) -> None:
    for function, expected in (
        (nested_outer, 7),
        (closure_outer, 10),
        (receiver_outer, 10),
        (partial_outer, 7),
    ):
        with subtests.test(function=function.__name__):
            assert (function(7)) == (expected)
            report = find_regimes(
                native_fixture(callback=function),
                facts_backend="metadata",
            )
            assert (report["semantic_check"]["errors"]) == ([])
            callback = next(
                value
                for value in report["analysis"]["declarations"]["callbacks"].values()
                if (value.get("source") or "").startswith("def " + function.__name__)
            )
            assert (callback["reads"]) == (["history"])
            assert (callback["status"]) == ("opaque")


def test_variadic_helper_retains_the_conservative_outer_scope() -> None:
    report = find_regimes(
        native_fixture(callback=variadic_outer),
        facts_backend="metadata",
    )
    assert (report["semantic_check"]["errors"]) == ([])
    callback = next(
        value
        for value in report["analysis"]["declarations"]["callbacks"].values()
        if (value.get("source") or "").startswith("def variadic_outer")
    )
    assert (callback["reads"]) == (["history"])
