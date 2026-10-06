"""Dated regime declarations: exact age selectors, schedules and declared support."""

import inspect
from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import pytest

import lcm
from _lcm.regime_law import bind_regime_law
from _lcm.user_regime_validation import validate_regimes
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DeterministicTransition,
    Phased,
    StochasticTransition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime


def _probs() -> jnp.ndarray:
    return jnp.array([0.5, 0.5])


def _code() -> int:
    return 1


ANNUAL = AgeGrid(start=60, inclusive_stop=65, step="Y")
QUARTERLY = AgeGrid(start=61, inclusive_stop=63, step="Q")


def test_age_range_exclusive_stop_excludes_the_boundary_age() -> None:
    """The exclusive endpoint is excluded from a selected age interval."""
    schedule = ByAge(cases={AgeRange(start=60, exclusive_stop=62): "a"}).resolve(ANNUAL)
    assert schedule.covered_ages == (60, 61)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (61, (61,)),
        ((60, 62, 62), (60, 62)),
        (range(61, 63), (61, 62)),
        (AgeRange(start=62, exclusive_stop=65), (62, 63, 64)),
        (AgeRange(exclusive_stop=62), (60, 61)),
        (AgeRange(start=60.5, exclusive_stop=63), (61, 62)),
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
    schedule = ByAge(
        cases={AgeRange(start=62, exclusive_stop=Fraction(125, 2)): "b"}
    ).resolve(QUARTERLY)
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
    """`until` selects `then` at the grid predecessor of `stop_age_exclusive`."""
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
        ({63.5: "a"}, r"Age 63\.5 in selector .* is not an age of the model"),
        ({(61, 61.25): "a"}, r"Age 61\.25 in selector .* is not an age"),
        ({True: "a"}, "must name numeric ages, not True"),
        ({float("nan"): "a"}, "contains a nonfinite age"),
        (
            {AgeRange(start=63, exclusive_stop=62): "a"},
            "start 63 must be below exclusive_stop 62",
        ),
        (
            {AgeRange(start=60.1, exclusive_stop=60.9): "a"},
            "selects no age of the model",
        ),
        (
            {AgeRange(exclusive_stop=62): "a", 61: "b"},
            r"overlaps another case at age\(s\) \[61\]",
        ),
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


_TERMINAL_INSIDE = "marks a terminal regime only as the top-level"


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: ByAge(cases={61: None}), _TERMINAL_INSIDE),
        (lambda: ByAge(cases={61: "a"}, default=None), _TERMINAL_INSIDE),
        (
            lambda: ByAge.until(stop_age_exclusive=62, law="a", then=None),
            _TERMINAL_INSIDE,
        ),
        (
            lambda: ByAge(cases={61: Phased(solve=None, simulate="a")}),
            _TERMINAL_INSIDE,
        ),
        (lambda: ByAge(cases={61: ByAge(cases={61: "a"})}), "cannot be nested"),
        (lambda: ByAge(cases={}), "needs at least one case"),
    ],
)
def test_none_and_nested_schedules_are_rejected_inside_a_schedule(
    *, build, match: str
) -> None:
    """A schedule cannot mark a regime terminal at some ages, nor nest schedules."""
    with pytest.raises(RegimeInitializationError, match=match):
        build()


_LAWS = {
    "regime_name": lambda: "retired",
    "choose": lambda: DeterministicTransition(func=_code),
    "markov_transition": lambda: StochasticTransition(func=_probs),
    "per_target_dict": lambda: {
        "working": StochasticTransition(func=_probs),
        "retired": StochasticTransition(func=_probs),
    },
}


def _phased(*, schedule_side: str, law: object) -> Phased:
    schedule = ByAge.until(stop_age_exclusive=63, law=law, then="retired")
    sides = {"solve": "working", "simulate": "working"}
    for side in ("solve", "simulate"):
        if schedule_side in (side, "both"):
            sides[side] = schedule
    return Phased(**sides)


@pytest.mark.parametrize("law_form", list(_LAWS))
@pytest.mark.parametrize("schedule_side", ["solve", "simulate", "both"])
def test_law_rejects_a_schedule_inside_a_top_level_phased(
    *, schedule_side: str, law_form: str
) -> None:
    """A top-level `Phased` may not wrap a `ByAge` on either side."""
    transition = _phased(schedule_side=schedule_side, law=_LAWS[law_form]())
    side_pattern = {
        "solve": "`solve`",
        "simulate": "`simulate`",
        "both": "`solve`.*`simulate`",
    }[schedule_side]
    with pytest.raises(
        RegimeInitializationError,
        match=rf"`ByAge` cannot be nested inside `ByAge` or `Phased`.*{side_pattern}",
    ):
        bind_regime_law(transition)


@pytest.mark.parametrize("law_form", list(_LAWS))
def test_law_accepts_a_top_level_phased_of_plain_laws(*, law_form: str) -> None:
    """A top-level `Phased` whose sides are plain laws binds and validates."""
    transition = Phased(solve=_LAWS[law_form](), simulate=_LAWS[law_form]())
    law = bind_regime_law(transition)
    validate_regimes(
        regimes={"regime": Regime(functions={"utility": lambda: 0.0})},
        laws={"regime": law},
    )
    assert law.regime_transitions is transition


def test_until_rejects_a_stop_age_without_a_predecessor() -> None:
    """The first grid age has no predecessor on which to exit."""
    with pytest.raises(RegimeInitializationError, match="predecessor"):
        ByAge.until(stop_age_exclusive=60, law="a", then="b").resolve(ANNUAL)


def test_until_rejects_an_off_grid_stop_age() -> None:
    """`until` bounds are exact grid coordinates."""

    with pytest.raises(RegimeInitializationError, match=r"61\.5"):
        ByAge.until(stop_age_exclusive=61.5, law="a", then="b").resolve(ANNUAL)


@pytest.mark.parametrize("wrapper", [DeterministicTransition, StochasticTransition])
@pytest.mark.parametrize("targets", [("a", "b"), (), ("a", "a")])
def test_transition_kernels_reject_topology_metadata(*, wrapper, targets) -> None:
    """The model graph is the sole public owner of regime support."""
    with pytest.raises(RegimeInitializationError, match=r"Model\(edges=\.\.\.\)"):
        wrapper(
            func=_code if wrapper is DeterministicTransition else _probs,
            targets=targets,
        )


def test_stochastic_kernel_has_no_topology_field() -> None:
    """State and regime probability kernels expose the same targetless API."""
    assert not hasattr(StochasticTransition(func=_probs), "targets")


def test_choose_exposes_the_wrapped_signature() -> None:
    """The DAG machinery reads the selector's own argument names."""

    def select(work: int) -> int:
        return work

    assert DeterministicTransition(func=select).__wrapped__ is select  # ty: ignore[unresolved-attribute]


def test_by_age_constructor_takes_only_cases_and_default() -> None:
    """The public constructor exposes `cases` and `default`, nothing private."""
    assert tuple(inspect.signature(ByAge).parameters) == ("cases", "default")


@pytest.mark.parametrize("wrapper", [DeterministicTransition, StochasticTransition])
@pytest.mark.parametrize("restriction", [AgeRange(exclusive_stop=64), True])
def test_transition_rejects_embedded_age_restrictions(*, wrapper, restriction) -> None:
    """Source-age support belongs to Model.edges, including invalid metadata."""
    with pytest.raises(RegimeInitializationError, match=r"Model\(edges=\.\.\.\)"):
        wrapper(
            func=_code if wrapper is DeterministicTransition else _probs,
            targets={"working": restriction},
        )


@pytest.mark.parametrize(
    ("name", "wrapper"),
    [
        ("deterministic_transition", DeterministicTransition),
        ("stochastic_transition", StochasticTransition),
    ],
)
def test_transition_decorator_preserves_function_contract(
    *, name: str, wrapper
) -> None:
    """A public decorator preserves its targetless DAG signature."""
    decorator = getattr(lcm, name, None)
    assert decorator is not None, f"The public {name} decorator must be available."

    def transition(state: int) -> jnp.ndarray:
        return jnp.asarray(float(state))

    law = decorator()(transition)
    assert isinstance(law, wrapper)
    assert law.__wrapped__ is transition
    assert inspect.signature(law) == inspect.signature(transition)
    assert int(law(state=2)) == 2
    assert not hasattr(law, "targets")


def test_stochastic_state_decorator_keeps_fixed_component() -> None:
    """A stochastic state decorator retains its fixed categorical groups."""
    decorator = getattr(lcm, "stochastic_transition", None)
    assert decorator is not None, "The public stochastic decorator must be available."
    law = decorator(fixed_component=(0, 0))(_probs)
    assert isinstance(law, StochasticTransition)
    assert not hasattr(law, "targets")
    assert law.fixed_component == (0, 0)


def test_deterministic_state_decorator_declares_no_regime_targets() -> None:
    """Explicit deterministic state laws need no regime support declaration."""
    decorator = getattr(lcm, "deterministic_transition", None)
    assert decorator is not None, (
        "The public deterministic decorator must be available."
    )
    law = decorator()(_code)
    assert isinstance(law, DeterministicTransition)
    assert not hasattr(law, "targets")
    assert law() == 1


def test_deterministic_state_wrapper_declares_no_regime_targets() -> None:
    """A deterministic wrapper can mark a state law without regime targets."""
    law = DeterministicTransition(func=_code)
    assert not hasattr(law, "targets")


def test_until_schedule_resolves_like_its_declaration() -> None:
    """A schedule built by `until` resolves to its legs without a constructor
    argument of its own."""
    schedule = ByAge.until(stop_age_exclusive=62, law="work", then="retire")
    assert dict(schedule.resolve(ANNUAL).law_by_period) == {0: "work", 1: "retire"}


def test_with_mapped_laws_calls_func_once_per_law() -> None:
    """Mapping the laws evaluates `func` exactly once for each declared law."""
    calls: list[object] = []

    def rename(law: object) -> object:
        calls.append(law)
        return f"{law}_x"

    ByAge(cases={61: "a", 62: "b"}, default="c").with_mapped_laws(func=rename)
    assert calls == ["a", "b", "c"]


def test_with_mapped_laws_returns_self_when_no_law_changes() -> None:
    """An identity mapping keeps the schedule object itself."""
    schedule = ByAge(cases={61: "a"})
    assert schedule.with_mapped_laws(func=lambda law: law) is schedule


@pytest.mark.parametrize(
    ("grid", "age", "expected"),
    [(ANNUAL, 61.0, "a"), (QUARTERLY, 61.25, "a"), (QUARTERLY, Fraction(5, 4), None)],
)
def test_resolved_schedule_at_matches_exact_ages(
    *, grid: AgeGrid, age: Any, expected: str | None
) -> None:
    """`at` finds a law at an age equal to a grid age and raises for any other."""
    schedule = ByAge(cases={AgeRange(start=61, exclusive_stop=62): "a"}).resolve(grid)
    if expected is None:
        with pytest.raises(KeyError):
            schedule.at(age)
    else:
        assert schedule.at(age) == expected
