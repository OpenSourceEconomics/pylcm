"""Models built from dated regime declarations: coverage, support and values."""

from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
    _SupportedStochasticTransition,
)
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DeterministicTransition,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidRegimeTransitionProbabilitiesError,
    ModelInitializationError,
)
from lcm.regime import Regime
from lcm.typing import BoolND, DiscreteState, FloatND, Period, ScalarInt
from tests.test_models.graph import with_fixture_graph

AGES = AgeGrid(start=25, inclusive_stop=75, step="10Y")
ROOTS: dict[object, str] = {25: "working"}


def _choose_age_restricted_target(period: Period) -> ScalarInt:
    return jnp.where(period < 3, RegimeId.working, RegimeId.retirement)


def _age_restricted_probabilities(period: Period) -> FloatND:
    return jnp.asarray(
        [jnp.where(period < 3, 1.0, 0.0), jnp.where(period == 3, 1.0, 0.0), 0.0]
    )


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _stay(health: DiscreteState) -> FloatND:
    return jnp.where(health == Health.good, 0.9, 0.7)


def _die(health: DiscreteState) -> FloatND:
    return 1 - _stay(health)


def _utility(*, wealth: FloatND, health: DiscreteState) -> FloatND:
    return wealth + health


def _regime(*, regime_transitions: Any, **kwargs: Any) -> Regime:
    return Regime(
        regime_transitions=regime_transitions,
        states={
            "health": DiscreteGrid(category_class=Health),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": lambda wealth: wealth,
        },
        functions={"utility": _utility},
        **kwargs,
    )


DEAD = Regime(regime_transitions=None, functions={"utility": lambda: 0.0})


def _dated_model(*, edges: object | None = None, **overrides: Regime) -> Model:
    regimes = {
        "working": _regime(
            regime_transitions=ByAge.until(
                stop_age_exclusive=65,
                law={
                    "working": StochasticTransition(func=_stay),
                    "dead": StochasticTransition(func=_die),
                },
                then="retirement",
            )
        ),
        "retirement": _regime(
            regime_transitions=ByAge(
                cases={AgeRange(start=65, exclusive_stop=75): "dead"}
            )
        ),
        "dead": DEAD,
    }
    if edges is not None:
        return Model(
            regimes=regimes | overrides,
            ages=AGES,
            regime_id_class=RegimeId,
            initial_nodes=((25, "working"),),
            edges=edges,
        )
    return with_fixture_graph(
        regimes=regimes | overrides,
        ages=AGES,
        regime_id_class=RegimeId,
        initial_nodes=ROOTS,
    )


def _hand_masked_model() -> Model:
    def stay(*, period: Period, health: DiscreteState) -> FloatND:
        return jnp.where(period < 3, _stay(health), 0.0)

    def die(*, period: Period, health: DiscreteState) -> FloatND:
        return jnp.where(period < 3, _die(health), 0.0)

    def retire(period: Period) -> FloatND:
        return jnp.where(period == 3, 1.0, 0.0)

    return with_fixture_graph(
        regimes={
            "working": _regime(
                regime_transitions=ByAge.until(
                    stop_age_exclusive=65,
                    law={
                        "working": StochasticTransition(func=stay),
                        "dead": StochasticTransition(func=die),
                    },
                    then={
                        "dead": StochasticTransition(func=die),
                        "retirement": StochasticTransition(func=retire),
                    },
                ),
            ),
            "retirement": _regime(
                regime_transitions=ByAge(
                    cases={
                        AgeRange(
                            start=65, exclusive_stop=75
                        ): _SupportedDeterministicTransition(
                            func=lambda: RegimeId.dead, targets=("dead",)
                        )
                    }
                ),
            ),
            "dead": DEAD,
        },
        ages=AGES,
        regime_id_class=RegimeId,
        initial_nodes=ROOTS,
    )


def _solve_dated_and_hand_masked() -> tuple[Any, Any]:
    params = {"discount_factor": 0.95}
    dated = _dated_model().solve(params=params, log_level="off").values
    masked = _hand_masked_model().solve(params=params, log_level="off").values
    return dated, masked


def test_hand_masked_model_additionally_solves_only_the_dead_problem_at_65() -> None:
    """The masked law names death as a zero-mass target of the exit at 55.

    Only that zero-mass target demands the dead problem at 65, so it is the one
    (period, regime) pair the masked model solves beyond the dated model.
    """
    dated, masked = _solve_dated_and_hand_masked()
    masked_keys = {(p, r) for p, by_regime in masked.items() for r in by_regime}
    dated_keys = {(p, r) for p, by_regime in dated.items() for r in by_regime}
    assert masked_keys - dated_keys == {(4, "dead")}


def test_dated_model_values_equal_the_hand_masked_model() -> None:
    """A schedule solves to exactly the values of its hand-written masked law."""
    dated, masked = _solve_dated_and_hand_masked()
    np.testing.assert_equal(
        {(p, r): np.asarray(v) for p, by in dated.items() for r, v in by.items()},
        {(p, r): np.asarray(masked[p][r]) for p, by in dated.items() for r in by},
    )


def test_dated_model_solves_exactly_the_problems_demanded_from_its_root() -> None:
    """Schedules say where a law is available; the root decides what is solved.

    From working at 25, death is first reached at 35. The exit at 55 leads only
    to retirement, so nobody is dead at 65 and death is next reached at 75.
    """
    reachability = _dated_model().reachability.solution
    assert reachability.active_regimes_by_period == (
        frozenset({"working"}),
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"retirement"}),
        frozenset({"dead"}),
    )


@pytest.mark.parametrize("period", [0, 3])
def test_dated_model_edges_are_the_declared_support_at_each_period(
    period: int,
) -> None:
    """The exit age reaches only `then`; earlier ages only the ordinary law."""
    expected = {0: ("dead", "working"), 3: ("retirement",)}[period]
    targets = _dated_model().reachability.solution.targets_by_period[period]
    assert targets["working"] == expected


@pytest.mark.parametrize(
    ("wrapper", "func"),
    [
        (DeterministicTransition, _choose_age_restricted_target),
        (StochasticTransition, _age_restricted_probabilities),
    ],
)
def test_target_age_mapping_declares_edges_at_source_age(*, wrapper, func) -> None:
    """A destination's selector restricts source ages and next-age value reads."""
    model = _dated_model(
        working=_regime(regime_transitions=wrapper(func=func)),
        edges={
            "working": {"working": AgeRange(exclusive_stop=55), "retirement": 55},
            "retirement": {"dead": 65},
        },
    )
    targets = model.reachability.solution.targets_by_period
    assert targets[0]["working"] == ("working",)
    assert targets[3]["working"] == ("retirement",)
    assert model.reachability.solution.active_regimes_by_period == (
        frozenset({"working"}),
        frozenset({"working"}),
        frozenset({"working"}),
        frozenset({"working"}),
        frozenset({"retirement"}),
        frozenset({"dead"}),
    )


@pytest.mark.parametrize(
    ("wrapper", "func"),
    [
        (DeterministicTransition, lambda: RegimeId.working),
        (StochasticTransition, lambda: jnp.asarray([1.0, 0.0, 0.0])),
    ],
)
def test_target_age_mapping_rejects_mass_outside_current_support(
    *, wrapper, func
) -> None:
    """A target allowed at another age cannot receive mass at the current age."""
    model = _dated_model(
        working=_regime(regime_transitions=wrapper(func=func)),
        edges={
            "working": {"working": AgeRange(exclusive_stop=55), "retirement": 55},
            "retirement": {"dead": 65},
        },
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params={"discount_factor": 0.95}, log_level="off")


@pytest.mark.parametrize(
    "transition",
    [
        lambda: RegimeId.dead,
        StochasticTransition(func=lambda: jnp.array([0.0, 0.0, 1.0])),
    ],
    ids=["bare-callable", "targetless-vector"],
)
def test_dated_model_accepts_targetless_kernels_with_explicit_graph(
    *, transition
) -> None:
    """Numerical kernels get their structural support solely from Model.edges."""
    model = _dated_model(
        edges={
            "working": {
                "working": (25, 35, 45),
                "dead": (25, 35, 45),
                "retirement": 55,
            },
            "retirement": {"dead": 65},
        },
        retirement=_regime(regime_transitions=transition),
    )
    assert model.reachability.solution.targets_by_period[4]["retirement"] == ("dead",)


def test_dated_model_rejects_a_target_not_covered_at_the_next_age() -> None:
    """Declared support must be solved at the next age; nothing is dropped."""
    with pytest.raises(ModelInitializationError, match="retirement"):
        _dated_model(
            retirement=_regime(
                regime_transitions=ByAge(
                    cases={AgeRange(start=55, exclusive_stop=65): "dead"}
                )
            )
        )


@pytest.mark.parametrize(
    ("wrapper", "func"),
    [
        (DeterministicTransition, lambda: RegimeId.dead),
        (StochasticTransition, lambda: jnp.array([0.0, 0.0, 1.0])),
    ],
)
def test_target_age_mapping_rejects_unknown_target_only_named_at_final_age(
    *, wrapper: Any, func: Any
) -> None:
    """Validate every destination name even when its age has no outgoing edge."""
    with pytest.raises(ModelInitializationError, match="typo"):
        _dated_model(
            working=_regime(regime_transitions=wrapper(func=func)),
            edges={"working": {"dead": 25, "typo": 75}},
        )


def test_choose_routes_to_the_returned_regime_code() -> None:
    """A deterministic selector's regime code picks the target."""

    def retire_if_healthy(health: DiscreteState) -> ScalarInt:
        return jnp.where(health == Health.good, RegimeId.retirement, RegimeId.dead)

    model = _dated_model(
        working=_regime(
            regime_transitions=ByAge(
                cases={
                    AgeRange(exclusive_stop=55): "working",
                    55: _SupportedDeterministicTransition(
                        func=retire_if_healthy, targets=("retirement", "dead")
                    ),
                }
            )
        )
    )
    targets = model.reachability.solution.targets_by_period[3]
    assert targets["working"] == ("dead", "retirement")


def _model_with_entries(initial_nodes: Any) -> Model:
    return with_fixture_graph(
        regimes={
            "working": _regime(
                regime_transitions=ByAge.until(
                    stop_age_exclusive=65,
                    law={
                        "working": StochasticTransition(func=_stay),
                        "dead": StochasticTransition(func=_die),
                    },
                    then="retirement",
                )
            ),
            "retirement": _regime(
                regime_transitions=ByAge(
                    cases={AgeRange(start=65, exclusive_stop=75): "dead"}
                )
            ),
            "dead": DEAD,
        },
        ages=AGES,
        regime_id_class=RegimeId,
        initial_nodes=initial_nodes,
    )


def test_reachability_nodes_are_the_exact_demanded_age_regime_pairs() -> None:
    """Every demanded problem appears once, keyed by its exact grid age."""
    expected = frozenset(
        {(age, "working") for age in (25, 35, 45, 55)}
        | {(65, "retirement")}
        | {(age, "dead") for age in (35, 45, 55, 75)}
    )
    assert _dated_model().reachability.nodes == expected


@pytest.mark.parametrize(
    ("initial_nodes", "expected"),
    [
        (
            {
                25: "working",
                AgeRange(start=65, exclusive_stop=75): ("retirement", "dead"),
            },
            frozenset({(25, "working"), (65, "retirement"), (65, "dead")}),
        ),
        (
            {25: "working", (25, 35): "working"},
            frozenset({(25, "working"), (35, "working")}),
        ),
    ],
    ids=["rules", "overlapping-rules-union"],
)
def test_initial_nodes_are_the_permitted_covered_pairs(
    *, initial_nodes: Any, expected: frozenset
) -> None:
    """Entry rules select exact covered pairs and union across rules."""
    assert _model_with_entries(initial_nodes).initial_nodes == expected


@pytest.mark.parametrize(
    ("initial_nodes", "match"),
    [
        (
            {25: "retirement"},
            (
                r"requires 'retirement' at age 25, where `edges` declares no edge "
                r"out of 'retirement' at that age"
            ),
        ),
        ({25: "unknown"}, r"names unknown regime\(s\) \['unknown'\]"),
        ({61: "working"}, r"^Age 61 in selector 61 is not an age of the model"),
        (
            {AgeRange(start=80): "dead"},
            (
                r"selector AgeRange\(start=80, exclusive_stop=None\) "
                r"selects no age of the model"
            ),
        ),
    ],
    ids=["uncovered-pair", "unknown-regime", "off-grid-age", "empty-selector"],
)
def test_initial_nodes_rejects_pairs_that_are_not_declared_problems(
    *, initial_nodes: Any, match: str
) -> None:
    """Entry rules name covered pairs; nothing is filtered away."""
    with pytest.raises(ModelInitializationError, match=match):
        _model_with_entries(initial_nodes)


def test_simulation_input_outside_the_entry_permissions_raises() -> None:
    """A covered pair that is not an admissible entry is rejected before evaluation."""
    model = _model_with_entries({25: "working"})
    with pytest.raises(InvalidInitialConditionsError, match="retirement"):
        model.validate_initial_conditions(
            initial_conditions={
                "regime_id": jnp.array([RegimeId.working, RegimeId.retirement]),
                "age": jnp.array([25.0, 65.0]),
                "health": jnp.array([Health.good, Health.good]),
                "wealth": jnp.array([10.0, 10.0]),
            },
            params={"discount_factor": 0.95},
        )


_ALREADY_VISITED_ROOTS = {25: "working", 65: "retirement"}


def test_a_root_already_visited_leaves_solved_values_unchanged() -> None:
    """Admitting a start the base root already reaches changes no solved value."""
    params = {"discount_factor": 0.95}
    wide = (
        _model_with_entries(_ALREADY_VISITED_ROOTS)
        .solve(params=params, log_level="off")
        .values
    )
    base = _dated_model().solve(params=params, log_level="off").values
    assert {p: set(r) for p, r in wide.items()} == {p: set(r) for p, r in base.items()}
    for period, by_regime in base.items():
        for regime, values in by_regime.items():
            np.testing.assert_array_equal(wide[period][regime], values)


def test_a_root_already_visited_leaves_the_model_identity_unchanged() -> None:
    """Models that differ only in an already-visited start share their identity."""
    assert (
        _model_with_entries(_ALREADY_VISITED_ROOTS)._model_structure_fingerprint
        == _dated_model()._model_structure_fingerprint
    )


def _choose_working(health: DiscreteState) -> ScalarInt:
    return jnp.where(health == Health.good, RegimeId.working, RegimeId.dead)


def _leaky_vector(health: DiscreteState) -> FloatND:
    return jnp.array([0.1, 0.0, 0.9]) + 0.0 * health


def _short_mass(health: DiscreteState) -> FloatND:
    return 0.8 * _stay(health)


@pytest.mark.parametrize(
    "transition",
    [
        _SupportedDeterministicTransition(
            func=_choose_working, targets=("retirement", "dead")
        ),
        _SupportedStochasticTransition(
            func=_leaky_vector, targets=("retirement", "dead")
        ),
        {
            "retirement": StochasticTransition(func=_short_mass),
            "dead": StochasticTransition(func=_die),
        },
    ],
    ids=["choose-outside-support", "vector-mass-outside-support", "short-mass"],
)
@pytest.mark.parametrize("log_level", ["off", "warning"])
def test_invalid_regime_selection_raises_at_every_log_level(
    *, transition: Any, log_level: LogLevel
) -> None:
    """Regime-selection validity does not depend on verbosity in a dated model."""
    model = _dated_model(
        working=_regime(
            regime_transitions=ByAge(
                cases={AgeRange(exclusive_stop=55): "working", 55: transition},
            )
        )
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params={"discount_factor": 0.95}, log_level=log_level)


def _certain() -> FloatND:
    return jnp.asarray(1.0)


def test_shared_state_law_may_name_targets_outside_the_declared_support() -> None:
    """Support comes from the transition; extra per-target state laws go unused."""
    shared_wealth_law = {
        "working": lambda wealth: wealth,
        "retirement": lambda wealth: wealth,
    }
    retirement = Regime(
        regime_transitions=ByAge(
            cases={
                AgeRange(start=65, exclusive_stop=75): {
                    "dead": StochasticTransition(func=_certain)
                }
            }
        ),
        states={
            "health": DiscreteGrid(category_class=Health),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": shared_wealth_law,
        },
        functions={"utility": _utility},
    )
    model = _dated_model(retirement=retirement)
    assert model.reachability.solution.targets_by_period[4] == {"retirement": ("dead",)}


def _is_healthy(health: DiscreteState) -> BoolND:
    return health == Health.good


def _stay_if_healthy(is_healthy: BoolND) -> FloatND:
    return jnp.where(is_healthy, 0.9, 0.7)


def _die_if_frail(is_healthy: BoolND) -> FloatND:
    return 1 - _stay_if_healthy(is_healthy)


def test_scheduled_cells_keep_the_annotations_of_the_laws_they_wrap() -> None:
    """A period-masked cell reading an annotated DAG output builds and solves."""
    working = Regime(
        regime_transitions=ByAge.until(
            stop_age_exclusive=65,
            law={
                "working": StochasticTransition(func=_stay_if_healthy),
                "dead": StochasticTransition(func=_die_if_frail),
            },
            then="retirement",
        ),
        states={
            "health": DiscreteGrid(category_class=Health),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": lambda wealth: wealth,
        },
        functions={"utility": _utility, "is_healthy": _is_healthy},
    )
    params = {"discount_factor": 0.95}
    values = _dated_model(working=working).solve(params=params, log_level="off")
    expected = _dated_model().solve(params=params, log_level="off")
    np.testing.assert_array_equal(
        values.values[0]["working"], expected.values[0]["working"]
    )


def test_user_regimes_keep_the_dated_declaration() -> None:
    """A model publishes each regime's transition exactly as declared."""
    declared = ByAge(cases={AgeRange(start=65, exclusive_stop=75): "dead"})
    model = _dated_model(retirement=_regime(regime_transitions=declared))
    assert model.user_regimes["retirement"].regime_transitions is declared


_EARLY_STAGES = (AgeRange(start=25, exclusive_stop=55),)


def _stay_by_stage(health: DiscreteState) -> FloatND:
    stage = _EARLY_STAGES[0]
    return _stay(health) * (stage.start is not None)


def test_laws_may_read_age_ranges_from_module_constants() -> None:
    """An `AgeRange` held in a module constant is part of the model identity."""
    working = _regime(
        regime_transitions=ByAge.until(
            stop_age_exclusive=65,
            law={
                "working": StochasticTransition(func=_stay_by_stage),
                "dead": StochasticTransition(func=_die),
            },
            then="retirement",
        )
    )
    assert _dated_model(working=working).reachability.nodes == (
        _dated_model().reachability.nodes
    )


@categorical(ordered=False)
class _VectorRegimeId:
    alive: ScalarInt
    done: ScalarInt


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _support_only_vector() -> FloatND:
    """One entry per declared target, not one per regime id."""
    return jnp.asarray([1.0])


def test_a_vector_law_shorter_than_the_regime_ids_is_refused() -> None:
    """A vector `StochasticTransition` returns one entry per regime id.

    A vector holding only its declared targets' entries has no position for the
    other regimes; reading them anyway would repeat the last entry and double the
    mass, so the solve refuses the law and names both lengths.
    """
    model = with_fixture_graph(
        regimes={
            "alive": Regime(
                regime_transitions=_SupportedStochasticTransition(
                    func=_support_only_vector, targets=("done",)
                ),
                functions={"utility": _zero_utility},
            ),
            "done": Regime(
                regime_transitions=None, functions={"utility": _zero_utility}
            ),
        },
        regime_id_class=_VectorRegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "alive"},
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=r"1 entries.*2 regime"
    ):
        model.solve(params={"alive": {"discount_factor": 0.9}}, log_level="off")


MONTHLY_AGES = AgeGrid(start=0, inclusive_stop=Fraction(1, 4), step="M")


@categorical(ordered=False)
class _MonthlyRegimeId:
    end: ScalarInt


def _monthly_model(initial_nodes: dict) -> Model:
    return with_fixture_graph(
        ages=MONTHLY_AGES,
        regime_id_class=_MonthlyRegimeId,
        initial_nodes=initial_nodes,
        regimes={
            "end": Regime(regime_transitions=None, functions={"utility": _zero_utility})
        },
    )


def _monthly_start(*, age: FloatND, log_level: LogLevel, initial_nodes: dict) -> Any:
    return _monthly_model(initial_nodes).simulate(
        params={},
        initial_conditions={
            "age": age,
            "regime_id": jnp.array([_MonthlyRegimeId.end]),
        },
        seed=0,
        log_level=log_level,
    )


def _first_month(age_source: str) -> FloatND:
    """The age one month in, read off the grid or written as a Python float."""
    return MONTHLY_AGES.values[1:2] if age_source == "grid" else jnp.array([1 / 12])


@pytest.mark.parametrize("age_source", ["grid", "literal"])
@pytest.mark.parametrize("log_level", ["off", "warning"])
def test_undeclared_monthly_start_raises_at_every_log_level(
    *, age_source: str, log_level: LogLevel
) -> None:
    """A start one month in is refused when only age 0 is an admissible entry."""
    with pytest.raises(InvalidInitialConditionsError, match=r"\(1/12, 'end'\)"):
        _monthly_start(
            age=_first_month(age_source),
            log_level=log_level,
            initial_nodes={0: "end"},
        )


@pytest.mark.parametrize("age_source", ["grid", "literal"])
@pytest.mark.parametrize("log_level", ["off", "warning"])
def test_declared_monthly_start_simulates_from_its_period(
    *, age_source: str, log_level: LogLevel
) -> None:
    """A subject admitted one month in to a terminal regime is recorded at period 1."""
    result = _monthly_start(
        age=_first_month(age_source),
        log_level=log_level,
        initial_nodes={0: "end", Fraction(1, 12): "end"},
    )
    assert result.to_dataframe()["period"].tolist() == [1]


@pytest.mark.parametrize("log_level", ["off", "warning"])
def test_off_grid_monthly_start_raises_at_every_log_level(
    *, log_level: LogLevel
) -> None:
    """An age between two grid months is reported as off the age grid."""
    with pytest.raises(ValueError, match="not valid age grid points"):
        _monthly_start(
            age=jnp.array([0.05]),
            log_level=log_level,
            initial_nodes={0: "end", Fraction(1, 12): "end"},
        )


@pytest.mark.parametrize(
    "wrapper",
    [_SupportedDeterministicTransition, _SupportedStochasticTransition],
    ids=lambda w: w.__name__,
)
def test_a_model_refuses_a_regime_law_carrying_its_targets(*, wrapper: Any) -> None:
    """A regime law tagged with destinations is refused; the graph declares them."""
    with pytest.raises(ModelInitializationError, match=r"Model\(edges=\.\.\.\)"):
        _dated_model(
            retirement=_regime(
                regime_transitions=wrapper(
                    func=lambda: RegimeId.dead, targets=("dead",)
                )
            ),
            edges={
                "working": {"working": (25, 35, 45), "retirement": 55},
                "retirement": {"dead": 65},
            },
        )
