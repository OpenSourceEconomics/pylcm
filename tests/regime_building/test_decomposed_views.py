"""What the engine reads when a regime is written in the declaration vocabulary.

A regime's `functions`, `constraints` and `transition` are what the author
wrote — declarations included. The three decomposed views are what the engine
runs: one utility per stakeholder, the ordinary constraints alone, and a
per-target probability cell for every target. The transformations are tested on
raw mappings, because that is the input they exist to handle.
"""

from collections.abc import Mapping
from typing import cast

import jax.numpy as jnp
import pytest

from lcm import (
    CollectiveUtility,
    Phased,
    ProjectedRegimeValue,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
)
from lcm.regime import (
    decompose_constraints,
    decompose_functions,
    decompose_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousState, FloatND


def _u_f(wealth: ContinuousState) -> FloatND:
    """The first stakeholder's flow utility."""
    return jnp.log(wealth)


def _u_m(wealth: ContinuousState) -> FloatND:
    """The second stakeholder's flow utility."""
    return 0.5 * jnp.log(wealth)


def _u_m_simulate(wealth: ContinuousState) -> FloatND:
    """The second stakeholder's flow utility as simulation realizes it."""
    return 0.25 * jnp.log(wealth)


def _budget(wealth: ContinuousState) -> FloatND:
    """An ordinary feasibility constraint."""
    return wealth > 0.0


def _ir_f(*, Q_f: FloatND, V_alone_f: FloatND) -> FloatND:
    """The first stakeholder's participation constraint."""
    return Q_f >= V_alone_f


def _identity_wealth(wealth: ContinuousState) -> FloatND:
    """The projection carrying wealth into the reference regime unchanged."""
    return wealth


def _prob_one() -> FloatND:
    """A degenerate selection probability."""
    return jnp.asarray(1.0)


def _gate(V_target_f: FloatND) -> FloatND:
    """A gate that is always open."""
    return V_target_f > -jnp.inf


_ROUTE_F = StakeholderRoute(
    target_stakeholder="f",
    fallback=ProjectedRegimeValue(
        regime="single_f", projection={"wealth": _identity_wealth}
    ),
)

_REFERENCE = ProjectedRegimeValue(
    regime="single_f", projection={"wealth": _identity_wealth}
)


def _cells(transition: object) -> Mapping[str, MarkovTransition]:
    """The per-target cells a decomposed per-target transition holds."""
    return cast("Mapping[str, MarkovTransition]", decompose_transition(transition))


def _phases(transition: object) -> Phased:
    """The two phases a decomposed `Phased` transition holds."""
    return cast("Phased", decompose_transition(transition))


def test_a_collective_utility_becomes_one_entry_per_stakeholder():
    """Each stakeholder's body lands under the name the engine reads."""
    raw = {"utility": CollectiveUtility(utilities={"f": _u_f, "m": _u_m})}

    assert dict(decompose_functions(raw)) == {"utility_f": _u_f, "utility_m": _u_m}


def test_the_declaration_object_never_reaches_the_engine():
    """Nothing under `"utility"` survives decomposition."""
    raw = {"utility": CollectiveUtility(utilities={"f": _u_f, "m": _u_m})}

    assert "utility" not in decompose_functions(raw)


def test_a_delegated_body_is_taken_from_the_entry_it_delegates_to():
    """A `None` body means the regime's own `utility_<s>` is that utility."""
    raw = {
        "utility": CollectiveUtility(utilities={"f": None, "m": _u_m}),
        "utility_f": _u_f,
    }

    assert decompose_functions(raw)["utility_f"] is _u_f


def test_stakeholder_entries_follow_the_declaration_not_the_mapping():
    """The household's own order survives however the entries arrived."""
    raw = {
        "utility_m": _u_m,
        "utility": CollectiveUtility(utilities={"f": _u_f, "m": None}),
    }

    assert list(decompose_functions(raw)) == ["utility_f", "utility_m"]


def test_a_regime_declaring_no_household_is_left_alone():
    """A singleton regime's functions pass through unchanged."""
    raw = {"utility": _u_f, "helper": _budget}

    assert decompose_functions(raw) is raw


def test_decomposing_functions_twice_changes_nothing():
    """The transformation is idempotent, so a second engine stage is safe."""
    raw = {"utility": CollectiveUtility(utilities={"f": _u_f, "m": _u_m})}
    once = decompose_functions(raw)

    assert dict(decompose_functions(once)) == dict(once)


def test_a_phased_stakeholder_body_reaches_the_engine_whole():
    """The phase split happens downstream, so the container survives intact."""
    body = Phased(solve=_u_m, simulate=_u_m_simulate)
    raw = {"utility": CollectiveUtility(utilities={"m": body, "f": _u_f})}

    assert decompose_functions(raw)["utility_m"] is body


def test_a_value_dependent_constraint_leaves_the_ordinary_constraints():
    """What stays is what is evaluated before the action values exist."""
    raw = {
        "budget": _budget,
        "ir_f": ValueDependentConstraint(
            predicate=_ir_f, references={"V_alone_f": _REFERENCE}
        ),
    }

    assert dict(decompose_constraints(raw)) == {"budget": _budget}


def test_decomposing_constraints_twice_changes_nothing():
    """The transformation is idempotent."""
    raw = {
        "budget": _budget,
        "ir_f": ValueDependentConstraint(
            predicate=_ir_f, references={"V_alone_f": _REFERENCE}
        ),
    }
    once = decompose_constraints(raw)

    assert dict(decompose_constraints(once)) == dict(once)


def test_a_value_dependent_transition_leaves_its_selection_probability():
    """The cell the canonical pipeline reads is the declared probability."""
    probability = MarkovTransition(_prob_one)
    raw = {
        "couple": ValueDependentTransition(
            probability=probability, gate=_gate, routes={"f": _ROUTE_F}
        )
    }

    assert _cells(raw)["couple"] is probability


def test_a_bare_probability_is_wrapped_into_the_cell_grammar():
    """A per-target cell takes a `MarkovTransition`, so a callable is wrapped."""
    raw = {
        "couple": ValueDependentTransition(
            probability=_prob_one, gate=_gate, routes={"f": _ROUTE_F}
        )
    }

    assert _cells(raw)["couple"].func is _prob_one


def test_an_ordinary_target_cell_passes_through_untouched():
    """Only the value-dependent cells are rewritten."""
    cell = MarkovTransition(_prob_one)

    assert _cells({"couple": cell})["couple"] is cell


def test_a_terminal_transition_stays_terminal():
    """`None` is not a mapping and means the regime ends."""
    assert decompose_transition(None) is None


def test_a_coarse_transition_passes_through_untouched():
    """A callable naming no target has no cell to rewrite."""
    assert decompose_transition(_prob_one) is _prob_one


def test_each_phase_of_a_phased_transition_is_decomposed_on_its_own():
    """A value-dependent transition may be declared inside `Phased`."""
    solve_probability = MarkovTransition(_prob_one)
    simulate_probability = MarkovTransition(_prob_one)
    raw = Phased(
        solve={
            "couple": ValueDependentTransition(
                probability=solve_probability, gate=_gate, routes={"f": _ROUTE_F}
            )
        },
        simulate={
            "couple": ValueDependentTransition(
                probability=simulate_probability, gate=_gate, routes={"f": _ROUTE_F}
            )
        },
    )
    decomposed = _phases(raw)

    assert (decomposed.solve["couple"], decomposed.simulate["couple"]) == (
        solve_probability,
        simulate_probability,
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"couple": MarkovTransition(_prob_one)},
        None,
        _prob_one,
    ],
)
def test_decomposing_a_transition_twice_changes_nothing(raw):
    """The transformation is idempotent on every form the slot takes."""
    once = decompose_transition(raw)

    assert decompose_transition(once) == once


def test_a_transition_with_nothing_to_decompose_is_the_very_same_object():
    """Identity survives, so a phase-variation scan can still compare by `is`.

    A regime declares one phase variance by writing one object into both
    phases. A view that rebuilt the mapping on every read would make every
    per-target transition look phase-varying, so the view returns its input
    untouched when there is no declaration in it.
    """
    raw = {"couple": MarkovTransition(_prob_one)}

    assert decompose_transition(raw) is raw


def test_one_object_written_into_both_phases_still_reads_as_one():
    """The two phases of an undecomposed `Phased` stay identical objects."""
    shared = {"couple": MarkovTransition(_prob_one)}
    raw = Phased(solve=shared, simulate=shared)
    decomposed = _phases(raw)

    assert decomposed.solve is decomposed.simulate
