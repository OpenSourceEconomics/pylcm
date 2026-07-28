"""Model-level age normalization: build-time resolution of age markers.

`normalize_age_specialization` is the single early boundary that resolves every
`AgeSpecializedFunction` / `AgeSpecializedGrid` into concrete build-time objects.
These tests pin the contract the downstream pipeline relies on:

- every factory is called exactly once per active period (never at inactive ages,
  never twice for a marker broadcast to both phases);
- the representative objects come from the first active period;
- no public marker survives into the representative regime or the phase specs;
- a regime active at no age, or a grid marker with runtime-supplied points, is
  rejected loudly.
"""

import jax.numpy as jnp
import pytest

from _lcm.regime_building.age_normalization import (
    PeriodizedUserFunction,
    normalize_age_specialization,
)
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.phases import normalize_all_regime_phases
from lcm import IrregSpacedGrid, LinSpacedGrid
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction, AgeSpecializedGrid


def _ages() -> AgeGrid:
    # Ages 20..24 -> periods 0..4.
    return AgeGrid(start=20, stop=24, step="Y")


def _utility(consumption: float, extra: float) -> float:
    return consumption + extra


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_regime(age: float):  # noqa: ARG001
    return jnp.asarray(0, dtype=jnp.int32)


def _normalized(regimes: dict[str, UserRegime], ages: AgeGrid):
    finalized = finalize_regimes(user_regimes=regimes, derived_categoricals={})
    phased = normalize_all_regime_phases(user_regimes=finalized)
    return finalized, normalize_age_specialization(
        user_regimes=finalized, phased_specs=phased, ages=ages
    )


def _dead() -> UserRegime:
    return UserRegime(transition=None, functions={"utility": lambda: 0.0})


def test_no_markers_passes_through_unchanged() -> None:
    """A model with no age markers normalizes to exactly its input, schedule None."""
    alive = UserRegime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={"wealth": _next_wealth},
    )
    finalized, result = _normalized({"work": alive, "dead": _dead()}, _ages())

    assert result.grid_schedule is None
    assert result.representative_user_regimes["work"] is finalized["work"]


def test_grid_builder_called_once_per_active_period() -> None:
    """The grid factory runs exactly once per active age, never at inactive ages."""
    calls: list[float] = []

    def build(age: float) -> LinSpacedGrid:
        calls.append(age)
        return LinSpacedGrid(start=float(age), stop=float(age) + 10.0, n_points=5)

    alive = UserRegime(
        transition=_next_regime,
        active=lambda age: age in {20, 22},
        states={
            "wealth": AgeSpecializedGrid(build=build, signature=lambda age: age),
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={"wealth": _next_wealth},
    )
    _, result = _normalized({"work": alive, "dead": _dead()}, _ages())

    assert sorted(calls) == [20, 22]
    assert tuple(sorted(result.grid_schedule.by_period)) == (0, 2)
    assert result.grid_schedule.specialized_states_by_regime["work"] == frozenset(
        {"wealth"}
    )


def test_function_builder_called_once_per_active_period() -> None:
    """A broadcast function marker builds once per active age, not once per phase."""
    calls: list[float] = []

    def build(age: float):
        calls.append(age)
        return lambda consumption: consumption + age

    alive = UserRegime(
        transition=_next_regime,
        active=lambda age: age in {20, 22},
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={
            "utility": AgeSpecializedFunction(build=build, signature=lambda age: age),
        },
        state_transitions={"wealth": _next_wealth},
    )
    _, result = _normalized({"work": alive, "dead": _dead()}, _ages())

    # Broadcast to solve AND simulate, yet built once per active age (2), not 4x.
    assert sorted(calls) == [20, 22]
    spec = result.phased_specs["work"]
    for phase in ("solution", "simulation"):
        func = getattr(spec, phase).functions["utility"]
        assert isinstance(func, PeriodizedUserFunction)
        assert tuple(sorted(func.concrete_by_period)) == (0, 2)


def test_representative_objects_come_from_first_active_period() -> None:
    """Representative function/grid are the first-active-period concrete objects."""

    def build_grid(age: float) -> LinSpacedGrid:
        return LinSpacedGrid(start=float(age), stop=float(age) + 10.0, n_points=5)

    alive = UserRegime(
        transition=_next_regime,
        active=lambda age: age in {21, 23},
        states={
            "wealth": AgeSpecializedGrid(build=build_grid, signature=lambda age: age),
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={"wealth": _next_wealth},
    )
    _, result = _normalized({"work": alive, "dead": _dead()}, _ages())

    rep_grid = result.representative_user_regimes["work"].states["wealth"]
    assert not isinstance(rep_grid, AgeSpecializedGrid)
    # First active age is 21 -> start == 21.
    assert float(rep_grid.start) == 21.0


def test_no_public_markers_remain_after_normalization() -> None:
    """Neither the representative regime nor the phase specs keep public markers."""
    alive = UserRegime(
        transition=_next_regime,
        active=lambda age: age in {20, 22},
        states={
            "wealth": AgeSpecializedGrid(
                build=lambda age: LinSpacedGrid(
                    start=float(age), stop=float(age) + 10.0, n_points=5
                ),
                signature=lambda age: age,
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={
            "utility": AgeSpecializedFunction(
                build=lambda age: lambda consumption: consumption + age,
                signature=lambda age: age,
            ),
        },
        state_transitions={"wealth": _next_wealth},
    )
    _, result = _normalized({"work": alive, "dead": _dead()}, _ages())

    rep = result.representative_user_regimes["work"]
    assert not any(
        isinstance(v, AgeSpecializedFunction) for v in rep.functions.values()
    )
    assert not any(isinstance(v, AgeSpecializedGrid) for v in rep.states.values())
    for phase in ("solution", "simulation"):
        slice_ = getattr(result.phased_specs["work"], phase)
        assert not any(
            isinstance(v, AgeSpecializedGrid) for v in slice_.grid_states.values()
        )
        assert not any(
            isinstance(v, AgeSpecializedFunction) for v in slice_.functions.values()
        )


def test_never_active_specialized_regime_is_rejected() -> None:
    """A regime with a marker but active at no age is a modelling error."""
    alive = UserRegime(
        transition=_next_regime,
        active=lambda age: False,  # noqa: ARG005
        states={
            "wealth": AgeSpecializedGrid(
                build=lambda age: LinSpacedGrid(
                    start=float(age), stop=float(age) + 10.0, n_points=5
                ),
                signature=lambda age: age,
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={"wealth": _next_wealth},
    )
    with pytest.raises(RegimeInitializationError, match="active at no model age"):
        _normalized({"work": alive, "dead": _dead()}, _ages())


def test_age_specialized_runtime_points_grid_is_rejected() -> None:
    """A grid marker resolving to runtime-supplied points is rejected."""
    alive = UserRegime(
        transition=_next_regime,
        states={
            "wealth": AgeSpecializedGrid(
                build=lambda age: IrregSpacedGrid(n_points=5, points=None),  # noqa: ARG005
                signature=lambda age: age,
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={"wealth": _next_wealth},
    )
    with pytest.raises(RegimeInitializationError, match="supplied at"):
        _normalized({"work": alive, "dead": _dead()}, _ages())
