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
from lcm import (
    IrregSpacedGrid,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Phased,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction, AgeSpecializedGrid
from lcm.typing import ScalarInt


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
    finalized = finalize_regimes(
        user_regimes=regimes,
        derived_categoricals={},
        certainty_equivalent=LinearExpectation(),
    )
    phased = normalize_all_regime_phases(user_regimes=finalized)
    active_periods_by_regime = {
        regime_name: tuple(ages.get_periods_where(regime.active))
        for regime_name, regime in finalized.items()
    }
    return finalized, normalize_age_specialization(
        user_regimes=finalized,
        phased_specs=phased,
        ages=ages,
        active_periods_by_regime=active_periods_by_regime,
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


def test_function_marker_build_receives_float_age_on_integer_age_grid() -> None:
    """`build(age)`/`signature(age)` always see a `float`, even on a whole-year grid.

    `AgeGrid.period_to_age` returns a Python `int` for a whole-year grid, but the
    `AgeSpecializedFunction`/`AgeSpecializedGrid` contract is typed
    `Callable[[float], ...]`. Every resolution site must cast to `float` before
    calling the user's factory, so the same age is never seen as two different
    Python types across call sites.
    """
    age_types: list[type] = []

    def build(age: float):
        age_types.append(type(age))
        return lambda consumption: consumption + age

    def signature(age: float) -> float:
        age_types.append(type(age))
        return age

    alive = UserRegime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={
            "utility": AgeSpecializedFunction(build=build, signature=signature),
        },
        state_transitions={"wealth": _next_wealth},
    )
    _normalized({"work": alive, "dead": _dead()}, _ages())

    assert age_types
    assert all(t is float for t in age_types)


def test_grid_marker_build_receives_float_age_on_integer_age_grid() -> None:
    """`AgeSpecializedGrid.build(age)`/`signature(age)` always see a `float`."""
    age_types: list[type] = []

    def build(age: float) -> LinSpacedGrid:
        age_types.append(type(age))
        return LinSpacedGrid(start=float(age), stop=float(age) + 10.0, n_points=5)

    def signature(age: float) -> float:
        age_types.append(type(age))
        return age

    alive = UserRegime(
        transition=_next_regime,
        states={"wealth": AgeSpecializedGrid(build=build, signature=signature)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={"wealth": _next_wealth},
    )
    _normalized({"work": alive, "dead": _dead()}, _ages())

    assert age_types
    assert all(t is float for t in age_types)


def test_function_marker_with_varying_parameter_names_is_rejected() -> None:
    """`AgeSpecializedFunction.build(age)` closures must share one parameter set.

    A later period's closure gaining or dropping a parameter compared to the
    representative (first-active) period's closure would leave that parameter
    dangling and unwired at qname-renaming time, so it is rejected at model-build
    time rather than surfacing as a confusing missing-argument error at solve().
    """

    def build(age: float):
        if age < 22:
            return lambda consumption: consumption
        return lambda consumption, region=0.0: consumption + region

    alive = UserRegime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={
            "utility": AgeSpecializedFunction(build=build, signature=lambda age: age),
        },
        state_transitions={"wealth": _next_wealth},
    )
    with pytest.raises(RegimeInitializationError, match="parameter"):
        _normalized({"work": alive, "dead": _dead()}, _ages())


def test_age_specialized_function_in_carried_state_solve_is_resolved() -> None:
    """An `AgeSpecializedFunction` inside a carried state's `Phased.solve` resolves.

    A carried state's solve-phase imputation is a first-class regime function
    elsewhere in the pipeline (params template, DAG discovery), so an age-specialized
    marker placed there must be detected and periodized like any other function
    marker -- not silently pass through unresolved because marker-detection only
    scans `functions`/`constraints`.
    """
    calls: list[float] = []

    def build(age: float):
        calls.append(age)
        return lambda aime: aime * 0.1 + age

    alive = UserRegime(
        transition=_next_regime,
        active=lambda age: age in {20, 22},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=5),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": Phased(
                solve=AgeSpecializedFunction(build=build, signature=lambda age: age),
                simulate=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={
            "wealth": _next_wealth,
            "aime": lambda aime: aime,
            "pension_wealth": lambda pension_wealth: pension_wealth,
        },
    )
    _, result = _normalized({"work": alive, "dead": _dead()}, _ages())

    # Built once per active age, not left as an unresolved factory.
    assert sorted(calls) == [20, 22]
    rep_state = result.representative_user_regimes["work"].states["pension_wealth"]
    assert isinstance(rep_state, Phased)
    assert not isinstance(rep_state.solve, AgeSpecializedFunction)
    # Solve-phase: pension_wealth is the periodized imputation, not a grid axis.
    solve_func = result.phased_specs["work"].solution.functions["pension_wealth"]
    assert isinstance(solve_func, PeriodizedUserFunction)
    assert tuple(sorted(solve_func.concrete_by_period)) == (0, 2)
    # Simulate-phase: pension_wealth is a genuine grid state, not a function.
    assert "pension_wealth" not in result.phased_specs["work"].simulation.functions
    assert "pension_wealth" in result.phased_specs["work"].simulation.grid_states


def test_grid_marker_resolution_calls_to_jax_once_per_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_resolve_grid_marker` must not call a grid's `to_jax()` twice per period.

    `to_jax()` has no caching at any level; a custom `ContinuousGrid` whose
    `to_jax()` does real work (e.g. loading calibration data) would otherwise pay
    for it twice per active period for no benefit.
    """
    calls: list[None] = []
    original_to_jax = LinSpacedGrid.to_jax

    def counting_to_jax(self):
        calls.append(None)
        return original_to_jax(self)

    monkeypatch.setattr(LinSpacedGrid, "to_jax", counting_to_jax)

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
        functions={"utility": lambda consumption: consumption},
        state_transitions={"wealth": _next_wealth},
    )
    _normalized({"work": alive, "dead": _dead()}, _ages())

    assert len(calls) == 2


def test_age_specialized_process_state_grid_is_rejected() -> None:
    """`AgeSpecializedGrid.build(age)` resolving to a stochastic process is rejected.

    v1 only supports age-varying continuous states, not process states: a process
    state's intrinsic transition wiring is resolved once at the representative age,
    so an age-varying process would silently freeze its transition kernel while its
    axis nodes kept varying per period.
    """
    alive = UserRegime(
        transition=_next_regime,
        states={
            "shock": AgeSpecializedGrid(
                build=lambda age: NormalIIDProcess(
                    n_points=5,
                    gauss_hermite=False,
                    mu=0.0,
                    sigma=0.1 + age * 0.01,
                    n_std=2.0,
                ),
                signature=lambda age: age,
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        functions={"utility": lambda consumption: consumption},
        state_transitions={"shock": fixed_transition("shock")},
    )
    with pytest.raises(RegimeInitializationError, match="process"):
        _normalized({"work": alive, "dead": _dead()}, _ages())


def test_never_active_specialized_regime_reports_the_real_cause() -> None:
    """A regime active nowhere with a specialized function reports the real cause.

    `Model()` construction validates variable usage before age-normalization runs;
    that earlier check must resolve `AgeSpecializedFunction` markers even when the
    regime has zero active periods, so it sees the marker's real dependencies and
    does not misreport a used action as unused -- masking the accurate
    "active at no model age" error that fires moments later in the pipeline.
    """

    @categorical(ordered=False)
    class _RegimeId:
        work: ScalarInt
        dead: ScalarInt

    def _next_regime(age: float) -> ScalarInt:  # noqa: ARG001
        return jnp.asarray(0, dtype=jnp.int32)

    work = UserRegime(
        transition=_next_regime,
        active=lambda age: False,  # noqa: ARG005
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        actions={
            "consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=4),
            # Read only inside the marker, so its reachability depends entirely on
            # the marker being resolved before `get_ancestors` runs.
            "bonus": LinSpacedGrid(start=0.0, stop=1.0, n_points=3),
        },
        functions={
            "utility": AgeSpecializedFunction(
                build=lambda age: lambda consumption, bonus: consumption + bonus + age,
                signature=lambda age: age,
            ),
        },
        state_transitions={"wealth": _next_wealth},
    )
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})

    with pytest.raises(RegimeInitializationError, match="active at no model age"):
        Model(
            regimes={"work": work, "dead": dead},
            ages=_ages(),
            regime_id_class=_RegimeId,
        )


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
