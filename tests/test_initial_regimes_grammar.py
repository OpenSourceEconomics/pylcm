"""Required `initial_nodes`, keyword-only declarations, and `ByAge` availability."""

import inspect
from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import pytest

import lcm
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarInt

AGES = AgeGrid(start=25, inclusive_stop=75, step="10Y")


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _utility(*, wealth: FloatND) -> FloatND:
    return wealth


def _regime() -> Regime:
    return Regime(
        states={"wealth": LinSpacedGrid(start=0, stop=100, n_points=3)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility": _utility},
    )


def _stay() -> FloatND:
    return jnp.asarray(0.9)


def _die() -> FloatND:
    return jnp.asarray(0.1)


EDGES = {
    "working": Transition(
        targets={"working": (25, 35, 45), "dead": (25, 35, 45), "retirement": 55},
        law=ByAge.until(
            stop_age_exclusive=65,
            law={
                "working": StochasticTransition(func=_stay),
                "dead": StochasticTransition(func=_die),
            },
            then="retirement",
        ),
    ),
    "retirement": {"dead": 65},
}


def _regimes() -> dict[str, Regime]:
    return {
        "working": _regime(),
        "retirement": _regime(),
        "dead": Regime(functions={"utility": lambda: 0.0}),
    }


def _model(**kwargs: Any) -> Model:
    return Model(
        regimes=_regimes(),
        ages=AGES,
        regime_id_class=RegimeId,
        edges=EDGES,
        **kwargs,
    )


def test_model_without_initial_nodes_is_a_signature_error() -> None:
    """`initial_nodes` is a required keyword with no default."""
    with pytest.raises(TypeError, match="initial_nodes"):
        _model()


def test_model_initial_nodes_has_no_default() -> None:
    """The signature itself carries no fallback root universe."""
    parameter = inspect.signature(Model.__init__).parameters["initial_nodes"]
    assert parameter.default is inspect.Parameter.empty


_NOT_A_MAPPING = r"parameter initial_nodes=.* violates type hint UserInitialNodes"


@pytest.mark.parametrize(
    ("initial_nodes", "match"),
    [
        (None, _NOT_A_MAPPING),
        ({}, r"^`initial_nodes` must be a nonempty mapping .*; got \{\}\.$"),
        ("working", _NOT_A_MAPPING),
        (("working", "retirement"), _NOT_A_MAPPING),
        ({25: ()}, r"a nonempty sequence of names; got \(\)\.$"),
        ({25: "unknown"}, r"names unknown regime\(s\) \['unknown'\]"),
    ],
    ids=["none", "empty", "bare-name", "bare-sequence", "empty-rule", "unknown"],
)
def test_model_rejects_malformed_initial_nodes(
    *, initial_nodes: Any, match: str
) -> None:
    """`None`, empty, bare-name and unknown-name roots all fail at construction."""
    with pytest.raises(ModelInitializationError, match=match):
        _model(initial_nodes=initial_nodes)


@pytest.mark.parametrize(
    ("initial_nodes", "match"),
    [
        ({61: "working"}, r"^Age 61 in selector 61 is not an age of the model"),
        ({True: "working"}, r"^Age selector True must name numeric ages"),
        ({"25": "working"}, r"^Age selector '25' must name numeric ages"),
    ],
    ids=["off-grid", "boolean-age", "string-age"],
)
def test_model_rejects_malformed_root_ages(*, initial_nodes: Any, match: str) -> None:
    """Root ages are exact grid coordinates; nothing is rounded onto the clock."""
    with pytest.raises(ModelInitializationError, match=match):
        _model(initial_nodes=initial_nodes)


@pytest.mark.parametrize(
    ("initial_nodes", "expected"),
    [
        ({25: "working"}, frozenset({(25, "working")})),
        (
            {
                (25, 45): "working",
                AgeRange(start=65, exclusive_stop=75): ("retirement",),
            },
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
    *, initial_nodes: Any, expected: frozenset
) -> None:
    """Each rule contributes ages times names; rules are unioned."""
    assert _model(initial_nodes=initial_nodes).graph.initial_nodes == expected


def test_initial_nodes_are_immutable() -> None:
    """`model.initial_nodes` is an immutable snapshot of the admissible starts."""
    assert isinstance(
        _model(initial_nodes={25: "working"}).initial_nodes, lcm.InitialNodes
    )


def test_explicit_initial_nodes_normalizes_and_unions_selectors() -> None:
    """Publish exact ages and sorted, unique regimes after binding the grid."""
    model = _model(
        initial_nodes=lcm.InitialNodes(
            by_age={
                AgeRange(start=24, exclusive_stop=36): ["working", "dead"],
                25: ["working", "working"],
            }
        )
    )
    assert model.initial_nodes.by_age == {
        25: ("dead", "working"),
        35: ("dead", "working"),
    }


def test_explicit_initial_nodes_owns_regime_collections() -> None:
    """Editing the source mapping or its lists leaves a declaration unchanged."""
    names = ["working"]
    by_age = {25: names}
    declaration = lcm.InitialNodes(by_age=by_age)
    names.append("dead")
    by_age[35] = ["dead"]
    assert declaration.by_age == {25: ("working",)}


@pytest.mark.parametrize(
    "by_age",
    [
        {},
        {25: ()},
        {25: ""},
        {25: ("working", "")},
        {True: "working"},
        {float("nan"): "working"},
    ],
)
def test_explicit_initial_nodes_rejects_invalid_declarations(by_age: Any) -> None:
    """Refuse empty declarations, invalid regime names and invalid selectors."""
    with pytest.raises(ModelInitializationError):
        lcm.InitialNodes(by_age=by_age)


@pytest.mark.parametrize(
    "by_age", [{26: "working"}, {25: "unknown"}, {AgeRange(start=80): "dead"}]
)
def test_explicit_initial_nodes_checks_model_coordinates(by_age: Any) -> None:
    """Binding refuses off-grid starts, unknown regimes and empty selections."""
    with pytest.raises(ModelInitializationError):
        _model(initial_nodes=lcm.InitialNodes(by_age=by_age))


def test_explicit_initial_nodes_round_trip_preserves_fingerprint() -> None:
    """Equivalent declarations and reconstruction keep the model identity."""
    legacy = _model(initial_nodes=((25, "working"), (35, "working")))
    explicit = _model(
        initial_nodes=lcm.InitialNodes(by_age={range(25, 36, 10): "working"})
    )
    rebuilt = _model(initial_nodes=explicit.initial_nodes)
    assert (
        legacy._model_structure_fingerprint
        == explicit._model_structure_fingerprint
        == rebuilt._model_structure_fingerprint
    )


def test_explicit_initial_nodes_mapping_is_read_only() -> None:
    """A published initial-node mapping cannot be changed in place."""
    nodes = lcm.InitialNodes(by_age={25: "working"})
    with pytest.raises(TypeError):
        nodes.by_age[25] = ("dead",)  # ty: ignore[invalid-assignment]


def test_explicit_initial_nodes_field_is_frozen() -> None:
    """A declaration cannot switch its coordinate mapping after construction."""
    nodes = lcm.InitialNodes(by_age={25: "working"})
    with pytest.raises(AttributeError):
        nodes.by_age = {35: ("working",)}  # ty: ignore[invalid-assignment]


def test_explicit_initial_nodes_normalizes_legacy_pickle_state() -> None:
    """Restoring a model with stored pairs publishes the explicit declaration."""
    model = _model(initial_nodes={25: "working"})
    state = model.__getstate__()
    state["initial_nodes"] = model.graph.initial_nodes  # ty: ignore[invalid-key]
    restored = object.__new__(Model)
    restored.__setstate__(state)
    assert restored.initial_nodes == lcm.InitialNodes(by_age={25: "working"})


@pytest.mark.parametrize(
    ("call", "kwargs"),
    [
        (StochasticTransition, {"func": _stay}),
        (DeterministicTransition, {"func": _stay}),
        (ByAge, {"cases": {25: "dead"}}),
        (AgeRange, {"start": 25, "exclusive_stop": 35}),
        (ByAge.until, {"stop_age_exclusive": 65, "law": "dead", "then": "dead"}),
    ],
    ids=[
        "StochasticTransition",
        "DeterministicTransition",
        "ByAge",
        "AgeRange",
        "ByAge.until",
    ],
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
            AgeGrid(start=58, inclusive_stop=64, step="Y"),
            59,
            {59: "law", 60: "law", 61: "then"},
        ),
        (
            AgeGrid(start=61, inclusive_stop=63, step="Q"),
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
        (AgeGrid(start=58, inclusive_stop=64, step="Y"), 61, {61: "then"}),
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
    ages = AgeGrid(start=59, inclusive_stop=64, step="Y")
    assert _until_periods(ages=ages, stop=62) == {59: "law", 60: "law", 61: "then"}


def test_by_age_default_is_available_at_the_final_age() -> None:
    """An explicit fallback law may be selected at the last clock position."""
    ages = AgeGrid(start=25, inclusive_stop=45, step="10Y")
    resolved = ByAge(cases={}, default="dead").resolve(ages)
    assert tuple(resolved.law_by_period) == (0, 1, 2)


def test_unused_final_age_law_does_not_fail_model_construction() -> None:
    """A law declared at the last age is legal while no start requires it there.

    The `then` branch of the working law covers every age from 65 on,
    including the last age 75.
    """
    model = Model(
        regimes=_regimes(),
        edges=EDGES,
        ages=AGES,
        regime_id_class=RegimeId,
        initial_nodes={25: "working"},
    )
    assert model.graph.initial_nodes == frozenset({(25, "working")})


def test_root_at_final_age_of_a_nonterminal_regime_fails() -> None:
    """A start that requires a nonterminal problem at the last age fails."""
    with pytest.raises(ModelInitializationError, match="75"):
        Model(
            regimes=_regimes(),
            edges=EDGES,
            ages=AGES,
            regime_id_class=RegimeId,
            initial_nodes={75: "retirement"},
        )
