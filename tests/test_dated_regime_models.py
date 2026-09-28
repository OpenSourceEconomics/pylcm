"""Models built from dated regime declarations: coverage, support and values."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    Choose,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
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

AGES = AgeGrid(start=25, stop=75, step="10Y")


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


def _dated_model(**overrides: Regime) -> Model:
    regimes = {
        "working": _regime(
            regime_transitions=ByAge.until(
                65,
                law={
                    "working": MarkovTransition(_stay),
                    "dead": MarkovTransition(_die),
                },
                then="retirement",
            )
        ),
        "retirement": _regime(
            regime_transitions=ByAge({AgeRange(start=65, stop=75): "dead"})
        ),
        "dead": DEAD,
    }
    return Model(regimes=regimes | overrides, ages=AGES, regime_id_class=RegimeId)


def _legacy_model() -> Model:
    def stay(*, period: Period, health: DiscreteState) -> FloatND:
        return jnp.where(period < 3, _stay(health), 0.0)

    def die(*, period: Period, health: DiscreteState) -> FloatND:
        return jnp.where(period < 3, _die(health), 0.0)

    def retire(period: Period) -> FloatND:
        return jnp.where(period == 3, 1.0, 0.0)

    return Model(
        regimes={
            "working": _regime(
                regime_transitions=ByAge(
                    {
                        AgeRange(stop=65): {
                            "working": MarkovTransition(stay),
                            "dead": MarkovTransition(die),
                            "retirement": MarkovTransition(retire),
                        }
                    }
                ),
            ),
            "retirement": _regime(
                regime_transitions=ByAge(
                    {
                        AgeRange(start=65, stop=75): Choose(
                            lambda: RegimeId.dead, targets=("dead",)
                        )
                    }
                ),
            ),
            "dead": DEAD,
        },
        ages=AGES,
        regime_id_class=RegimeId,
    )


def test_dated_model_values_equal_the_hand_masked_legacy_model() -> None:
    """A schedule solves to exactly the values of its hand-written masked law."""
    params = {"discount_factor": 0.95}
    dated = _dated_model().solve(params=params, log_level="off").values
    legacy = _legacy_model().solve(params=params, log_level="off").values
    for period, by_regime in legacy.items():
        for regime, values in by_regime.items():
            np.testing.assert_array_equal(dated[period][regime], values)


def test_dated_model_covers_exactly_the_scheduled_ages() -> None:
    """Coverage is read off the schedules; terminal regimes cover every age."""
    reachability = _dated_model().reachability.solution
    assert reachability.active_regimes_by_period == (
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"working", "dead"}),
        frozenset({"retirement", "dead"}),
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
    "override",
    [
        {"retirement": _regime(regime_transitions=lambda: RegimeId.dead)},
        {
            "retirement": _regime(
                regime_transitions=MarkovTransition(lambda: jnp.array([0.0, 0.0, 1.0]))
            )
        },
    ],
    ids=["bare-callable", "targetless-vector"],
)
def test_dated_model_rejects_legacy_declarations(override: dict) -> None:
    """A dated model reads coverage and support only from the declarations."""
    with pytest.raises(ModelInitializationError):
        _dated_model(**override)


def test_dated_model_rejects_a_target_not_covered_at_the_next_age() -> None:
    """Declared support must be solved at the next age; nothing is dropped."""
    with pytest.raises(ModelInitializationError, match="retirement"):
        _dated_model(
            retirement=_regime(
                regime_transitions=ByAge({AgeRange(start=55, stop=65): "dead"})
            )
        )


def test_choose_routes_to_the_returned_regime_code() -> None:
    """A `Choose` selector's regime code picks the deterministic target."""

    def retire_if_healthy(health: DiscreteState) -> ScalarInt:
        return jnp.where(health == Health.good, RegimeId.retirement, RegimeId.dead)

    model = _dated_model(
        working=_regime(
            regime_transitions=ByAge(
                {
                    AgeRange(stop=55): "working",
                    55: Choose(retire_if_healthy, targets=("retirement", "dead")),
                }
            )
        )
    )
    targets = model.reachability.solution.targets_by_period[3]
    assert targets["working"] == ("dead", "retirement")


def _model_with_entries(initial_regimes: Any) -> Model:
    return Model(
        regimes={
            "working": _regime(
                regime_transitions=ByAge.until(
                    65,
                    law={
                        "working": MarkovTransition(_stay),
                        "dead": MarkovTransition(_die),
                    },
                    then="retirement",
                )
            ),
            "retirement": _regime(
                regime_transitions=ByAge({AgeRange(start=65, stop=75): "dead"})
            ),
            "dead": DEAD,
        },
        ages=AGES,
        regime_id_class=RegimeId,
        initial_regimes=initial_regimes,
    )


def test_reachability_nodes_are_the_exact_covered_age_regime_pairs() -> None:
    """Every declared problem appears once, keyed by its exact grid age."""
    expected = frozenset(
        {(age, "working") for age in (25, 35, 45, 55)}
        | {(65, "retirement")}
        | {(age, "dead") for age in (25, 35, 45, 55, 65, 75)}
    )
    assert _dated_model().reachability.nodes == expected


@pytest.mark.parametrize(
    ("initial_regimes", "expected"),
    [
        ({}, frozenset()),
        ("retirement", frozenset({(65, "retirement")})),
        (
            {25: "working", AgeRange(start=65, stop=75): ("retirement", "dead")},
            frozenset({(25, "working"), (65, "retirement"), (65, "dead")}),
        ),
        (
            {25: "working", (25, 35): "working"},
            frozenset({(25, "working"), (35, "working")}),
        ),
    ],
    ids=["empty", "bare-name", "rules", "overlapping-rules-union"],
)
def test_initial_nodes_are_the_permitted_covered_pairs(
    *, initial_regimes: Any, expected: frozenset
) -> None:
    """Entry rules select exact covered pairs and union across rules."""
    assert _model_with_entries(initial_regimes).initial_nodes == expected


def test_initial_nodes_default_to_every_covered_pair() -> None:
    """Without `initial_regimes`, every declared problem admits external entry."""
    model = _dated_model()
    assert model.initial_nodes == model.reachability.nodes


@pytest.mark.parametrize(
    "initial_regimes",
    [{25: "retirement"}, "unknown", {61: "working"}, {AgeRange(start=80): "dead"}],
    ids=["uncovered-pair", "unknown-regime", "off-grid-age", "empty-selector"],
)
def test_initial_regimes_rejects_pairs_that_are_not_declared_problems(
    initial_regimes: Any,
) -> None:
    """Entry rules name covered pairs; nothing is filtered away."""
    with pytest.raises(ModelInitializationError):
        _model_with_entries(initial_regimes)


def test_simulation_input_outside_the_entry_permissions_raises() -> None:
    """A covered pair that is not a permitted entry is rejected before evaluation."""
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


def test_entry_permissions_leave_solved_values_unchanged() -> None:
    """Permissions are entry metadata, not a change to the economic problem."""
    params = {"discount_factor": 0.95}
    narrow = _model_with_entries({}).solve(params=params, log_level="off").values
    wide = _dated_model().solve(params=params, log_level="off").values
    for period, by_regime in wide.items():
        for regime, values in by_regime.items():
            np.testing.assert_array_equal(narrow[period][regime], values)


def test_entry_permissions_leave_the_model_identity_unchanged() -> None:
    """Solutions stay compatible across models differing only in permissions."""
    assert (
        _model_with_entries({})._model_structure_fingerprint
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
        Choose(_choose_working, targets=("retirement", "dead")),
        MarkovTransition(_leaky_vector, targets=("retirement", "dead")),
        {"retirement": MarkovTransition(_short_mass), "dead": MarkovTransition(_die)},
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
                {AgeRange(stop=55): "working", 55: transition},
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
            {AgeRange(start=65, stop=75): {"dead": MarkovTransition(_certain)}}
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
            65,
            law={
                "working": MarkovTransition(_stay_if_healthy),
                "dead": MarkovTransition(_die_if_frail),
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
    declared = ByAge({AgeRange(start=65, stop=75): "dead"})
    model = _dated_model(retirement=_regime(regime_transitions=declared))
    assert model.user_regimes["retirement"].regime_transitions is declared


_EARLY_STAGES = (AgeRange(start=25, stop=55),)


def _stay_by_stage(health: DiscreteState) -> FloatND:
    stage = _EARLY_STAGES[0]
    return _stay(health) * (stage.start is not None)


def test_laws_may_read_age_ranges_from_module_constants() -> None:
    """An `AgeRange` held in a module constant is part of the model identity."""
    working = _regime(
        regime_transitions=ByAge.until(
            65,
            law={
                "working": MarkovTransition(_stay_by_stage),
                "dead": MarkovTransition(_die),
            },
            then="retirement",
        )
    )
    assert _dated_model(working=working).reachability.nodes == (
        _dated_model().reachability.nodes
    )
