"""Dated regime declarations: exact age selectors, schedules and declared support."""

from fractions import Fraction

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, AgeRange, ByAge, Choose, MarkovTransition, Phased
from lcm.exceptions import RegimeInitializationError


def _probs() -> jnp.ndarray:
    return jnp.array([0.5, 0.5])


def _code() -> int:
    return 1


ANNUAL = AgeGrid(start=60, stop=65, step="Y")
QUARTERLY = AgeGrid(start=61, stop=63, step="Q")


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (61, (61,)),
        ((60, 62, 62), (60, 62)),
        (range(61, 63), (61, 62)),
        (AgeRange(start=62, stop=65), (62, 63, 64)),
        (AgeRange(stop=62), (60, 61)),
        (AgeRange(start=60.5, stop=63), (61, 62)),
    ],
)
def test_by_age_selects_exact_grid_coordinates(*, selector, expected) -> None:
    """Each selector covers exactly the existing grid points it names."""
    schedule = ByAge(cases={selector: "b"}).resolve(ANNUAL)
    assert schedule.covered_ages == expected


def test_range_selects_integers_not_intervening_quarters() -> None:
    """A Python `range` names integer ages only, never the quarters between them."""
    schedule = ByAge(cases={range(61, 63): "b"}).resolve(QUARTERLY)
    assert schedule.covered_ages == (61, 62)


def test_age_range_selects_quarterly_points_in_half_open_interval() -> None:
    """An interval selects every existing grid point in `[start, stop)`."""
    schedule = ByAge(cases={AgeRange(start=62, stop=Fraction(125, 2)): "b"}).resolve(
        QUARTERLY
    )
    assert schedule.covered_ages == (62, Fraction(249, 4))


def test_default_fills_every_remaining_age() -> None:
    """`default` covers each age no explicit case selects, the final age included."""
    schedule = ByAge(cases={61: "a"}, default="b").resolve(ANNUAL)
    assert [schedule.at(age) for age in schedule.covered_ages] == [
        "b",
        "a",
        "b",
        "b",
        "b",
        "b",
    ]


@pytest.mark.parametrize(
    ("grid", "expected_exit"),
    [(ANNUAL, 61), (QUARTERLY, Fraction(247, 4))],
)
def test_until_exits_on_the_grid_predecessor_of_the_stop_age(
    *, grid: AgeGrid, expected_exit: int | Fraction
) -> None:
    """`until` applies `then` on the grid predecessor of the stop age, `law` before."""
    schedule = ByAge.until(
        stop_age_exclusive=62, law="work", then="retire", start_age_inclusive=61
    ).resolve(grid)
    assert schedule.at(expected_exit) == "retire"


def test_until_covers_start_up_to_the_stop_age() -> None:
    """`until` covers `start_age_inclusive <= age < stop_age_exclusive`."""
    schedule = ByAge.until(
        stop_age_exclusive=63, law="work", then="retire", start_age_inclusive=61
    ).resolve(ANNUAL)
    assert schedule.covered_ages == (61, 62)


def test_until_with_only_the_exit_position_leaves_the_ordinary_leg_unused() -> None:
    """A start equal to the predecessor covers the exit age alone."""
    schedule = ByAge.until(
        stop_age_exclusive=62, law="work", then="retire", start_age_inclusive=61
    ).resolve(ANNUAL)
    assert [schedule.at(age) for age in schedule.covered_ages] == ["retire"]


def test_resolved_schedule_rejects_an_uncovered_age() -> None:
    """Inspecting an age outside the coverage raises instead of guessing a law."""
    schedule = ByAge(cases={61: "a"}).resolve(ANNUAL)
    with pytest.raises(KeyError, match="60"):
        schedule.at(60)


@pytest.mark.parametrize(
    ("cases", "match"),
    [
        ({63.5: "a"}, r"Age 63\.5 in selector 63\.5 is not an age of the model"),
        ({(61, 61.25): "a"}, r"Age 61\.25 in selector \(61, 61\.25\) is not an age"),
        ({True: "a"}, r"Age selector True must name numeric ages"),
        ({float("nan"): "a"}, r"Age selector nan contains a nonfinite age"),
        ({AgeRange(start=63, stop=62): "a"}, r"start 63 must be below stop 62"),
        (
            {AgeRange(start=60.1, stop=60.9): "a"},
            r"selector AgeRange\(start=60\.1, stop=60\.9\) selects no age",
        ),
        (
            {AgeRange(stop=62): "a", 61: "b"},
            r"selector 61 overlaps another case at age\(s\) \[61\]",
        ),
    ],
    ids=[
        "off-grid",
        "off-grid-tuple",
        "boolean",
        "nan",
        "reversed",
        "empty",
        "overlap",
    ],
)
def test_invalid_selectors_are_rejected_at_resolution(*, cases, match: str) -> None:
    """Off-grid, boolean, nonfinite, reversed, empty and overlapping selections
    raise instead of being rounded, dropped or truncated."""
    with pytest.raises(RegimeInitializationError, match=match):
        ByAge(cases=cases).resolve(ANNUAL)


@pytest.mark.parametrize(
    ("cases", "expected_ages"),
    [({65: "a"}, (65,)), ({AgeRange(start=62): "a"}, (62, 63, 64, 65))],
)
def test_a_law_at_the_final_age_resolves_as_available(
    *, cases, expected_ages: tuple[int, ...]
) -> None:
    """A selection reaching the final age marks a law available there."""
    assert ByAge(cases=cases).resolve(ANNUAL).covered_ages == expected_ages


_NONE_INSIDE_A_SCHEDULE = r"^`None` marks a terminal regime only as the top-level"


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: ByAge(cases={61: None}), _NONE_INSIDE_A_SCHEDULE),
        (lambda: ByAge(cases={61: "a"}, default=None), _NONE_INSIDE_A_SCHEDULE),
        (
            lambda: ByAge.until(stop_age_exclusive=62, law="a", then=None),
            _NONE_INSIDE_A_SCHEDULE,
        ),
        (
            lambda: ByAge(cases={61: Phased(solve=None, simulate="a")}),
            _NONE_INSIDE_A_SCHEDULE,
        ),
        (
            lambda: ByAge(cases={61: ByAge(cases={61: "a"})}),
            r"^`ByAge` cannot be nested inside `ByAge` or `Phased`",
        ),
        (lambda: ByAge(cases={}), r"^`ByAge` needs at least one case\.$"),
    ],
    ids=["case", "default", "until-then", "phased-half", "nested", "empty"],
)
def test_none_and_nested_schedules_are_rejected_inside_a_schedule(
    *, build, match: str
) -> None:
    """Only a top-level `regime_transitions=None` is terminal; wrappers cannot be."""
    with pytest.raises(RegimeInitializationError, match=match):
        build()


def test_until_rejects_a_stop_age_without_a_predecessor() -> None:
    """The first grid age has no predecessor on which to exit."""
    with pytest.raises(RegimeInitializationError, match="predecessor"):
        ByAge.until(stop_age_exclusive=60, law="a", then="b").resolve(ANNUAL)


def test_until_rejects_an_off_grid_stop_age() -> None:
    """`until` ages are exact grid coordinates."""
    with pytest.raises(RegimeInitializationError, match=r"61\.5"):
        ByAge.until(stop_age_exclusive=61.5, law="a", then="b").resolve(ANNUAL)


def test_markov_transition_records_declared_targets() -> None:
    """A vector regime law carries its declared support as a tuple of names."""
    law = MarkovTransition(func=_probs, targets=["a", "b"])
    assert law.targets == ("a", "b")


@pytest.mark.parametrize(
    ("targets", "match"),
    [
        ((), r"^`MarkovTransition\.targets` must name a regime\.$"),
        (("a", "a"), r"names a regime more than once: \['a', 'a'\]"),
    ],
    ids=["empty", "duplicate"],
)
def test_markov_transition_rejects_empty_or_duplicate_targets(
    *, targets, match: str
) -> None:
    """Declared support is nonempty and names each target once."""
    with pytest.raises(RegimeInitializationError, match=match):
        MarkovTransition(func=_probs, targets=targets)


def test_markov_transition_without_targets_keeps_state_law_semantics() -> None:
    """A state law declares no regime support."""
    assert MarkovTransition(func=_probs).targets is None


def test_choose_records_declared_targets() -> None:
    """A deterministic selector names its support."""
    assert Choose(func=_code, targets=("a", "b")).targets == ("a", "b")


def test_choose_rejects_empty_targets() -> None:
    """A deterministic selector must name at least one target."""
    with pytest.raises(
        RegimeInitializationError, match=r"^`Choose\.targets` must name a regime\.$"
    ):
        Choose(func=_code, targets=())


def test_choose_exposes_the_wrapped_signature() -> None:
    """The DAG machinery reads the selector's own argument names."""

    def select(work: int) -> int:
        return work

    assert Choose(func=select, targets=("a",)).__wrapped__ is select  # ty: ignore[unresolved-attribute]
