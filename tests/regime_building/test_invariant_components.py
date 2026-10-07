"""Invariant-component analysis reads preservation off declarations alone.

A state is a candidate when a declaration gives it the identity law
(`fixed_transition`, or the group state generated for a declared
`fixed_component`). It is eligible in a phase only when every reachable edge out of
a carrier either preserves it by that identity law or leaves it for a regime that
does not carry it, no type-free regime enters a carrier, and no other value channel
reads across its codes. The analysis never changes how a model solves.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.invariant_components import (
    InvariantComponent,
    RegimeEdge,
    analyze_invariant_components,
    fail_if_invariant_blocking_is_unsafe,
)
from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.regime_building.test_carried_state_through_gated_self_loop import (
    _make_regimes as _make_gated_self_loop_regimes,
)
from tests.regime_building.test_carried_state_through_gated_self_loop import (
    _src_transition,
)
from tests.regime_building.test_same_period_ref_period_axes import (
    _make_model as _make_same_period_ref_model,
)
from tests.solution.test_fixed_component_markov import (
    _interleaved_kind_health,
)
from tests.solution.test_fixed_component_markov import (
    _model as _fixed_component_model,
)


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _EntryRegimeId:
    young: ScalarInt
    work: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _ReentryRegimeId:
    young: ScalarInt
    work: ScalarInt
    gap: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _PrefType:
    patient: ScalarInt
    average: ScalarInt
    impatient: ScalarInt


@categorical(ordered=False)
class _Health:
    bad: ScalarInt
    good: ScalarInt


_N_TYPES = 3


def _work_utility(
    *, consumption: ContinuousAction, pref_type: DiscreteState
) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * pref_type)


def _typed_health_utility(
    *,
    consumption: ContinuousAction,
    pref_type: DiscreteState,
    health: DiscreteState,
) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * pref_type) + 0.2 * health


def _typed_bequest(*, wealth: ContinuousState, pref_type: DiscreteState) -> FloatND:
    return jnp.log(wealth) * (1.0 + pref_type)


def _type_free_bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _feasible(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2.0


def _next_health(*, health: DiscreteState, pref_type: DiscreteState) -> FloatND:
    stay = 0.6 + 0.1 * pref_type
    return jnp.where(jnp.arange(2) == health, stay, 1.0 - stay)


def _reset_pref_type(pref_type: DiscreteState) -> DiscreteState:
    return jnp.zeros_like(pref_type)


def _rotate_pref_type(pref_type: DiscreteState) -> DiscreteState:
    return (pref_type + 1) % _N_TYPES


def _stay_with_certainty(pref_type: DiscreteState) -> FloatND:
    return jnp.eye(_N_TYPES)[pref_type]


def _draw_pref_type() -> FloatND:
    return jnp.array([0.2, 0.3, 0.5])


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 2, _RegimeId.work, _RegimeId.dead)


def _model(
    *,
    pref_law: object = None,
    terminal_reads_type: bool = True,
    typed_health: bool = False,
    sharded_states: tuple[str, ...] = (),
) -> Model:
    """Build the two-regime life cycle with a model-level preference type.

    Args:
        pref_law: The model-level law of `pref_type`; `None` declares the identity.
        terminal_reads_type: Whether the bequest of `dead` depends on the type,
            so that `dead` keeps the state rather than pruning it.
        typed_health: Whether `work` also carries a health state whose
            transition probabilities depend on the type.
        sharded_states: The states carrying a device axis.

    Returns:
        The model.

    """
    work_states: dict[str, DiscreteGrid] = {}
    work_laws: dict[str, StochasticTransition] = {}
    if typed_health:
        work_states["health"] = DiscreteGrid(_Health)
        work_laws["health"] = StochasticTransition(func=_next_health)
    return Model(
        edges={
            "work": Transition(
                targets={"work": (0, 1), "dead": (0, 1, 2)},
                law=DeterministicTransition(func=_next_regime),
            )
        },
        regimes={
            "work": Regime(
                states=work_states,
                actions={"consumption": LinSpacedGrid(start=1, stop=3, n_points=3)},
                functions={
                    "utility": _typed_health_utility if typed_health else _work_utility
                },
                constraints={"feasible": _feasible},
                state_transitions=work_laws,
            ),
            "dead": Regime(
                functions={
                    "utility": _typed_bequest
                    if terminal_reads_type
                    else _type_free_bequest
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=4, step="Y"),
        regime_id_class=_RegimeId,
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
            "pref_type": DiscreteGrid(_PrefType),
        },
        state_transitions={
            "wealth": _next_wealth,
            "pref_type": fixed_transition("pref_type")
            if pref_law is None
            else pref_law,
        },
        execution_config=ExecutionConfig(sharded_states=sharded_states),
        initial_nodes={0: ("work", "dead")},
    )


def _entry_model(*, drop_and_reenter: bool) -> Model:
    """Build a model where a regime without the type transitions into one with it.

    Args:
        drop_and_reenter: Whether `work` first leaves the type in `gap` and
            re-enters from there; otherwise a type-free `young` enters `work`.

    Returns:
        The model.

    """
    consumption = {"consumption": LinSpacedGrid(start=1, stop=3, n_points=3)}
    entry_law = StochasticTransition(func=_draw_pref_type)
    regime_id = _ReentryRegimeId if drop_and_reenter else _EntryRegimeId
    next_work_regime = _work_to_gap if drop_and_reenter else _work_to_work
    edges: dict[str, dict[str, int | tuple[int, ...]] | Transition] = {
        "young": {"work": 0},
        "work": Transition(
            targets={"work": 1, "gap": 1, "dead": (1, 2)}
            if drop_and_reenter
            else {"work": 1, "dead": (1, 2)},
            law=DeterministicTransition(func=next_work_regime),
        ),
    }
    if drop_and_reenter:
        edges["gap"] = {"dead": 2}
    regimes = {
        "young": Regime(
            actions=consumption,
            functions={"utility": _type_free_young_utility},
            constraints={"feasible": _feasible},
            state_transitions={"wealth": _next_wealth, "pref_type": entry_law},
        ),
        "work": Regime(
            states={"pref_type": DiscreteGrid(_PrefType)},
            actions=consumption,
            functions={"utility": _work_utility},
            constraints={"feasible": _feasible},
            state_transitions={
                "wealth": _next_wealth,
                "pref_type": fixed_transition("pref_type"),
            },
        ),
        "gap": Regime(
            actions=consumption,
            functions={"utility": _type_free_young_utility},
            constraints={"feasible": _feasible},
            state_transitions={"wealth": _next_wealth, "pref_type": entry_law},
        ),
        "dead": Regime(functions={"utility": _type_free_bequest}),
    }
    if not drop_and_reenter:
        del regimes["gap"]
    return Model(
        regimes=regimes,
        edges=edges,
        ages=AgeGrid(start=0, inclusive_stop=4, step="Y"),
        regime_id_class=regime_id,
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        initial_nodes={0: "young"},
    )


def _type_free_young_utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _work_to_work(age: float) -> ScalarInt:
    return jnp.where(age < 2, _EntryRegimeId.work, _EntryRegimeId.dead)


def _work_to_gap(age: float) -> ScalarInt:
    return jnp.where(age < 2, _ReentryRegimeId.gap, _ReentryRegimeId.dead)


def _components(model: Model) -> dict[str, InvariantComponent]:
    return dict(
        analyze_invariant_components(
            user_regimes=model.user_regimes,
            laws=model.graph.laws,
            regimes=model._regimes,
            reachability=model.reachability,
            initial_nodes=model.initial_nodes,
            ages=model.ages,
            fixed_component_splits=model._fixed_component_splits,
        )
    )


def _pref_type(model: Model) -> InvariantComponent:
    return _components(model)["pref_type"]


@pytest.mark.parametrize("phase", ["solve", "simulate"])
def test_identity_law_with_typed_terminal_is_eligible(phase):
    """A type preserved on every edge, terminal included, is eligible in each phase."""
    assert getattr(_pref_type(_model()), phase).eligible


def test_identity_chain_records_every_preserving_edge():
    """Each solved `work` period hands the type on to both of its targets."""
    assert _pref_type(_model()).solve.preserving_edges == (
        RegimeEdge(period=0, source="work", target="dead"),
        RegimeEdge(period=0, source="work", target="work"),
        RegimeEdge(period=1, source="work", target="dead"),
        RegimeEdge(period=1, source="work", target="work"),
        RegimeEdge(period=2, source="work", target="dead"),
    )


def test_carrying_periods_follow_the_regime_coverage():
    """`work` carries the type at ages 0-2, and the typed `dead` at every age."""
    assert dict(_pref_type(_model()).solve.carrying_periods) == {
        "dead": (0, 1, 2, 3),
        "work": (0, 1, 2),
    }


def test_canonical_code_mapping_keeps_declared_labels():
    """Codes and labels are the state grid's, in declaration order."""
    component = _pref_type(_model())
    assert (component.codes, component.labels) == (
        (0, 1, 2),
        ("patient", "average", "impatient"),
    )


def test_type_free_terminal_is_one_shared_dependency():
    """A bequest that ignores the type keeps `dead` as one shared, untyped node."""
    component = _pref_type(_model(terminal_reads_type=False))
    assert (
        component.solve.eligible,
        "dead" in component.solve.carrying_periods,
        component.solve.shared_dependencies,
    ) == (
        True,
        False,
        (
            RegimeEdge(period=0, source="work", target="dead"),
            RegimeEdge(period=1, source="work", target="dead"),
            RegimeEdge(period=2, source="work", target="dead"),
        ),
    )


def test_typed_terminal_keeps_its_type_specific_view():
    """A type-dependent `dead` is a carrier read through preserving edges."""
    component = _pref_type(_model())
    assert (
        component.solve.shared_dependencies,
        "dead" in component.solve.carrying_periods,
    ) == (
        (),
        True,
    )


def test_type_dependent_preferences_and_other_transitions_stay_eligible():
    """Type-dependent utility and health transitions do not couple the types."""
    assert _pref_type(_model(typed_health=True)).solve.eligible


_PER_TARGET_RESET = {"work": _reset_pref_type, "dead": fixed_transition("pref_type")}


@pytest.mark.parametrize(
    "pref_law",
    [
        pytest.param(_reset_pref_type, id="reset"),
        pytest.param(_rotate_pref_type, id="cross-type"),
        pytest.param(
            StochasticTransition(func=_stay_with_certainty),
            id="zero-probability-off-diagonal",
        ),
    ],
)
def test_state_without_identity_law_is_no_candidate(pref_law):
    """Values never establish invariance, not even a probability-one diagonal."""
    assert "pref_type" not in _components(_model(pref_law=pref_law))


@pytest.mark.parametrize(
    "work_law",
    [
        pytest.param(_reset_pref_type, id="reset"),
        pytest.param(_rotate_pref_type, id="cross-type"),
    ],
)
def test_identity_law_on_some_edges_only_is_refused(work_law):
    """A reset or cross-type read on one carrier edge refuses the identity elsewhere."""
    pref_law = {"work": work_law, "dead": fixed_transition("pref_type")}
    refusals = _pref_type(_model(pref_law=pref_law)).solve.refusals
    assert any(
        "work -> work" in refusal and "not the identity" in refusal
        for refusal in refusals
    )


@pytest.mark.parametrize("drop_and_reenter", [False, True])
def test_entry_into_a_carrier_without_binding_is_refused(drop_and_reenter):
    """A type-free regime entering a carrier would read every type's value."""
    refusals = _pref_type(
        _entry_model(drop_and_reenter=drop_and_reenter)
    ).solve.refusals
    assert any("enters" in refusal for refusal in refusals)


def test_phase_mismatch_is_analysed_per_phase():
    """An identity solve law with a resetting simulate law is solve-only eligible."""
    component = _pref_type(
        _model(
            pref_law=Phased(
                solve=fixed_transition("pref_type"), simulate=_reset_pref_type
            )
        )
    )
    assert (component.solve.eligible, component.simulate.eligible) == (True, False)


def test_initial_conditions_roots_are_preserved():
    """Every admissible start whose regime carries the type is a simulate root."""
    assert _pref_type(_model()).initial_nodes == ((0, "dead"), (0, "work"))


def test_generated_fixed_component_maps_groups_to_original_codes():
    """The interleaved group 1 is original codes 1 and 3, not a renumbered slice."""
    model = _fixed_component_model(
        factored=True, fixed_component=(0, 1, 0, 1), law=_interleaved_kind_health
    )
    component = _components(model)["kind_health_fixed"]
    assert (
        component.original_state_name,
        component.original_codes_by_code,
        component.solve.eligible,
    ) == ("kind_health", ((0, 2), (1, 3)), True)


def test_eligibility_does_not_depend_on_sharded_states():
    """Sharding the type changes placement, not the structural analysis."""
    assert _pref_type(_model(sharded_states=("pref_type",))) == _pref_type(_model())


def test_same_period_reference_is_an_unsupported_channel():
    """A same-period value reference touching a carrier refuses the coordinate."""
    model = _make_same_period_ref_model(later_ceiling=10.0, initial_nodes={0: "couple"})
    refusals = _components(model)["wealth"].solve.refusals
    assert any("same-period reference" in refusal for refusal in refusals)


def test_gated_edge_is_an_unsupported_channel():
    """A gated edge touching a carrier refuses the coordinate."""
    model = Model(
        regimes=_make_gated_self_loop_regimes(),
        edges={"src": _src_transition()},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_GatedRegimeId,
        initial_nodes={0: "src"},
    )
    refusals = _components(model)["wage"].solve.refusals
    assert any("gated edge" in refusal for refusal in refusals)


@categorical(ordered=False)
class _GatedRegimeId:
    src: ScalarInt
    src_exit: ScalarInt
    src_fallback: ScalarInt


def test_continuous_fixed_state_has_no_code_mapping():
    """Blocking needs codes, so a continuous identity-law state is refused."""
    model = _make_same_period_ref_model(later_ceiling=10.0, initial_nodes={0: "couple"})
    refusals = _components(model)["wealth"].solve.refusals
    assert any("discrete grid" in refusal for refusal in refusals)


def test_eligible_explicit_blocking_passes():
    """A width-one block of an eligible type is accepted before dispatch."""
    fail_if_invariant_blocking_is_unsafe(
        components=_components(_model()), block_widths={"pref_type": 1}, phase="solve"
    )


@pytest.mark.parametrize(
    ("model_factory", "block_widths", "fragment"),
    [
        pytest.param(
            lambda: _model(pref_law=_PER_TARGET_RESET),
            {"pref_type": 1},
            "not the identity",
            id="ineligible",
        ),
        pytest.param(_model, {"wealth": 1}, "no identity law", id="not-a-candidate"),
        pytest.param(_model, {"pref_type": 0}, "positive integer", id="zero-width"),
        pytest.param(_model, {"pref_type": True}, "positive integer", id="bool-width"),
        pytest.param(_model, {"pref_type": 4}, "exceeds", id="wider-than-codes"),
    ],
)
def test_unsafe_explicit_blocking_fails_before_dispatch(
    *, model_factory: Callable[[], Model], block_widths, fragment
):
    """An unsafe request raises naming the failed condition."""
    with pytest.raises(ExecutionPlanningError, match=fragment):
        fail_if_invariant_blocking_is_unsafe(
            components=_components(model_factory()),
            block_widths=block_widths,
            phase="solve",
        )


def test_unsafe_explicit_blocking_names_a_remedy():
    """The refusal tells the caller how to proceed without blocking."""
    with pytest.raises(ExecutionPlanningError, match="Remove"):
        fail_if_invariant_blocking_is_unsafe(
            components=_components(_model(pref_law=_PER_TARGET_RESET)),
            block_widths={"pref_type": 1},
            phase="solve",
        )


def test_analysis_leaves_the_solution_unchanged():
    """Analysing a model does not alter what it solves to."""
    params = {"discount_factor": 0.95}
    model = _model()
    before = model.solve(params=params, log_level="off").values
    _components(model)
    after = model.solve(params=params, log_level="off").values
    assert all(
        np.array_equal(
            np.asarray(before[period][name]), np.asarray(after[period][name])
        )
        for period in before
        for name in before[period]
    )
