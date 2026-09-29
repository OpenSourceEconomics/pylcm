"""Required `initial_regimes`, keyword-only declarations, and `ByAge` availability."""

import inspect
from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    Choose,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarInt

AGES = AgeGrid(start=25, stop=75, step="10Y")


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _utility(*, wealth: FloatND) -> FloatND:
    return wealth


def _regime(*, regime_transitions: Any) -> Regime:
    return Regime(
        regime_transitions=regime_transitions,
        states={"wealth": LinSpacedGrid(start=0, stop=100, n_points=3)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility": _utility},
    )


def _stay() -> FloatND:
    return jnp.asarray(0.9)


def _die() -> FloatND:
    return jnp.asarray(0.1)


def _regimes() -> dict[str, Regime]:
    return {
        "working": _regime(
            regime_transitions=ByAge.until(
                stop_age_exclusive=65,
                law={
                    "working": MarkovTransition(func=_stay),
                    "dead": MarkovTransition(func=_die),
                },
                then="retirement",
            )
        ),
        "retirement": _regime(
            regime_transitions=ByAge(cases={AgeRange(start=65, stop=75): "dead"})
        ),
        "dead": Regime(regime_transitions=None, functions={"utility": lambda: 0.0}),
    }


def _model(**kwargs: Any) -> Model:
    return Model(regimes=_regimes(), ages=AGES, regime_id_class=RegimeId, **kwargs)


def test_model_without_initial_regimes_is_a_signature_error() -> None:
    """`initial_regimes` is a required keyword with no default."""
    with pytest.raises(TypeError, match="initial_regimes"):
        _model()


def test_model_initial_regimes_has_no_default() -> None:
    """The signature itself carries no fallback root universe."""
    parameter = inspect.signature(Model.__init__).parameters["initial_regimes"]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "initial_regimes",
    [None, {}, "working", ("working", "retirement"), {25: ()}, {25: "unknown"}],
    ids=["none", "empty", "bare-name", "bare-sequence", "empty-rule", "unknown"],
)
def test_model_rejects_malformed_initial_regimes(initial_regimes: Any) -> None:
    """`None`, empty, bare-name and unknown-name roots all fail at construction."""
    with pytest.raises(ModelInitializationError):
        _model(initial_regimes=initial_regimes)


@pytest.mark.parametrize(
    "initial_regimes",
    [{61: "working"}, {True: "working"}, {"25": "working"}],
    ids=["off-grid", "boolean-age", "string-age"],
)
def test_model_rejects_malformed_root_ages(initial_regimes: Any) -> None:
    """Root ages are exact grid coordinates; nothing is rounded onto the clock."""
    with pytest.raises(ModelInitializationError):
        _model(initial_regimes=initial_regimes)


@pytest.mark.parametrize(
    ("initial_regimes", "expected"),
    [
        ({25: "working"}, frozenset({(25, "working")})),
        (
            {(25, 45): "working", AgeRange(start=65, stop=75): ("retirement",)},
            frozenset({(25, "working"), (45, "working"), (65, "retirement")}),
        ),
        (
            {25: "working", range(25, 36, 10): "working"},
            frozenset({(25, "working"), (35, "working")}),
        ),
    ],
    ids=["one-pair", "tuple-and-range", "overlapping-rules-union"],
)
def test_initial_nodes_are_the_cartesian_union_of_rules(
    *, initial_regimes: Any, expected: frozenset
) -> None:
    """Each rule contributes ages times names; rules are unioned."""
    assert _model(initial_regimes=initial_regimes).initial_nodes == expected


def test_initial_nodes_are_immutable() -> None:
    """`model.initial_nodes` is an immutable snapshot of the admissible starts."""
    assert isinstance(_model(initial_regimes={25: "working"}).initial_nodes, frozenset)


@pytest.mark.parametrize(
    ("call", "kwargs"),
    [
        (MarkovTransition, {"func": _stay}),
        (Choose, {"func": _stay, "targets": ("dead",)}),
        (ByAge, {"cases": {25: "dead"}}),
        (AgeRange, {"start": 25, "stop": 35}),
        (ByAge.until, {"stop_age_exclusive": 65, "law": "dead", "then": "dead"}),
    ],
    ids=["MarkovTransition", "Choose", "ByAge", "AgeRange", "ByAge.until"],
)
def test_declarations_reject_positional_arguments(*, call: Any, kwargs: dict) -> None:
    """Declaration constructors take keyword arguments only."""
    first, *rest = kwargs.values()
    with pytest.raises(TypeError, match="positional"):
        call(first, **dict(zip(list(kwargs)[1:], rest, strict=True)))


def _until_periods(*, ages: AgeGrid, stop: Any, start: Any = None) -> dict[Any, object]:
    schedule = ByAge.until(
        start_age_inclusive=start, stop_age_exclusive=stop, law="law", then="then"
    )
    return {
        ages.exact_values[period]: law
        for period, law in schedule.resolve(ages).law_by_period.items()
    }


@pytest.mark.parametrize(
    ("ages", "start", "expected"),
    [
        (
            AgeGrid(start=58, stop=64, step="Y"),
            59,
            {59: "law", 60: "law", 61: "then"},
        ),
        (
            AgeGrid(start=61, stop=63, step="Q"),
            Fraction(245, 4),
            {
                Fraction(245, 4): "law",
                Fraction(246, 4): "law",
                Fraction(247, 4): "then",
            },
        ),
        (
            AgeGrid(exact_values=(58, 60, Fraction(123, 2), 62, 64)),
            58,
            {58: "law", 60: "law", Fraction(123, 2): "then"},
        ),
        (AgeGrid(start=58, stop=64, step="Y"), 61, {61: "then"}),
    ],
    ids=["annual", "quarterly", "irregular", "one-source-interval"],
)
def test_until_selects_law_from_start_and_then_at_the_stop_predecessor(
    *, ages: AgeGrid, start: Any, expected: dict
) -> None:
    """`[start, stop)`: `then` at the last source before 62, nothing at or after 62."""
    assert _until_periods(ages=ages, start=start, stop=62) == expected


def test_until_without_start_begins_at_the_first_age() -> None:
    """An omitted `start_age_inclusive` selects the first clock coordinate."""
    ages = AgeGrid(start=59, stop=64, step="Y")
    assert _until_periods(ages=ages, stop=62) == {59: "law", 60: "law", 61: "then"}


def test_by_age_default_is_available_at_the_final_age() -> None:
    """An explicit fallback law may be selected at the last clock position."""
    ages = AgeGrid(start=25, stop=45, step="10Y")
    resolved = ByAge(cases={}, default="dead").resolve(ages)
    assert tuple(resolved.law_by_period) == (0, 1, 2)


def test_unused_final_age_law_does_not_fail_model_construction() -> None:
    """A law declared at the last age is legal while no start requires it there."""
    regimes = _regimes()
    regimes["retirement"] = _regime(
        regime_transitions=ByAge(cases={AgeRange(start=65): "dead"})
    )
    model = Model(
        regimes=regimes,
        ages=AGES,
        regime_id_class=RegimeId,
        initial_regimes={25: "working"},
    )
    assert model.initial_nodes == frozenset({(25, "working")})


def test_root_at_final_age_of_a_nonterminal_regime_fails() -> None:
    """The same final-age law fails once a start requires that nonterminal problem."""
    regimes = _regimes()
    regimes["retirement"] = _regime(
        regime_transitions=ByAge(cases={AgeRange(start=65): "dead"})
    )
    with pytest.raises(ModelInitializationError, match="75"):
        Model(
            regimes=regimes,
            ages=AGES,
            regime_id_class=RegimeId,
            initial_regimes={75: "retirement"},
        )
