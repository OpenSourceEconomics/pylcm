"""Exact starts on fractional clocks, empty selectors, and the `ByAge.until` bounds.

Starts are exact grid coordinates on any clock, a selector that picks no grid
age is an error rather than an empty rule, and `ByAge.until` takes only its
named keyword bounds, with a start strictly before an on-grid stop.
"""

from fractions import Fraction
from typing import Any

import pytest

from lcm import AgeGrid, AgeRange, ByAge, LinSpacedGrid, Model, categorical
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _model(*, ages: AgeGrid, initial_regimes: Any) -> Model:
    wealth = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    return Model(
        regimes={
            "working": Regime(
                regime_transitions=ByAge.until(
                    stop_age_exclusive=ages.exact_values[-1], law="working", then="dead"
                ),
                states={"wealth": wealth},
                state_transitions={"wealth": lambda wealth: wealth},
                functions={"utility": _utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": wealth},
                functions={"utility": _utility},
            ),
        },
        ages=ages,
        regime_id_class=_RegimeId,
        initial_regimes=initial_regimes,
    )


_QUARTERLY = AgeGrid(start=61, stop=63, step="Q")
_IRREGULAR = AgeGrid(exact_values=(58, 60, Fraction(123, 2), 62, 64))


@pytest.mark.parametrize(
    ("ages", "initial_regimes", "expected"),
    [
        (
            _QUARTERLY,
            {Fraction(245, 4): "working"},
            {Fraction(245, 4)},
        ),
        (
            _QUARTERLY,
            {AgeRange(start=61, stop=Fraction(247, 4)): "working"},
            {61, Fraction(245, 4), Fraction(246, 4)},
        ),
        (
            _IRREGULAR,
            {(60, Fraction(123, 2)): "working"},
            {60, Fraction(123, 2)},
        ),
        (
            _IRREGULAR,
            {AgeRange(start=59, stop=62): "working"},
            {60, Fraction(123, 2)},
        ),
        (
            _IRREGULAR,
            {range(58, 63, 2): "working"},
            {58, 60, 62},
        ),
    ],
    ids=[
        "quarterly-exact-age",
        "quarterly-half-open-range",
        "irregular-tuple",
        "irregular-half-open-range",
        "irregular-integer-range",
    ],
)
def test_initial_nodes_are_exact_on_fractional_clocks(
    *, ages: AgeGrid, initial_regimes: Any, expected: set
) -> None:
    """Starts land on the exact grid coordinates the selector names."""
    model = _model(ages=ages, initial_regimes=initial_regimes)
    assert model.initial_nodes == frozenset((age, "working") for age in expected)


@pytest.mark.parametrize(
    ("initial_regimes", "match"),
    [
        (
            {AgeRange(start=30, stop=31): "working"},
            r"AgeRange\(start=30, stop=31\).*selects no age",
        ),
        ({Fraction(245, 4): "working"}, r"61\.25|245/4"),
    ],
    ids=["age-range-between-grid-points", "fraction-off-an-annual-grid"],
)
def test_a_root_selector_without_a_grid_age_fails(
    *, initial_regimes: Any, match: str
) -> None:
    """A root rule whose selector picks no grid age is refused by name."""
    with pytest.raises(ModelInitializationError, match=match):
        _model(
            ages=AgeGrid(start=25, stop=45, step="10Y"),
            initial_regimes=initial_regimes,
        )


def test_a_by_age_case_without_a_grid_age_fails() -> None:
    """A schedule case whose selector picks no grid age is refused by name."""
    with pytest.raises(
        RegimeInitializationError, match=r"AgeRange\(start=30, stop=31\).*no age"
    ):
        ByAge(cases={AgeRange(start=30, stop=31): "dead"}).resolve(
            AgeGrid(start=25, stop=45, step="10Y")
        )


@pytest.mark.parametrize(
    ("start", "match"),
    [
        (60.5, r"start_age_inclusive 60\.5 is not an age of the model"),
        (62, r"start_age_inclusive 62 is not before stop_age_exclusive 62"),
        (63, r"start_age_inclusive 63 is not before stop_age_exclusive 62"),
    ],
    ids=["off-grid-start", "start-at-stop", "start-after-stop"],
)
def test_until_rejects_a_start_not_before_an_on_grid_stop(
    *, start: float, match: str
) -> None:
    """The first source must be a grid age strictly before the stop."""
    with pytest.raises(RegimeInitializationError, match=match):
        ByAge.until(
            start_age_inclusive=start, stop_age_exclusive=62, law="law", then="then"
        ).resolve(AgeGrid(start=58, stop=64, step="Y"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"boundary": 62, "law": "law", "then": "then"}, "boundary"),
        (
            {"stop_age_exclusive": 62, "start": 58, "law": "law", "then": "then"},
            "start",
        ),
    ],
    ids=["boundary-keyword", "start-keyword"],
)
def test_until_rejects_obsolete_keywords(*, kwargs: dict, match: str) -> None:
    """Only `start_age_inclusive` and `stop_age_exclusive` name the bounds."""
    with pytest.raises(TypeError, match=match):
        ByAge.until(**kwargs)


def test_until_rejects_a_positional_stop() -> None:
    """The stop is a keyword argument, never positional."""
    with pytest.raises(TypeError, match="positional"):
        ByAge.until(62, law="law", then="then")  # ty: ignore[missing-argument, too-many-positional-arguments]
