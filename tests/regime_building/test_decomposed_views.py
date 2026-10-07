"""What the engine reads when a regime is written in the declaration vocabulary.

A regime's `functions` and `constraints` are what the author wrote —
declarations included. The two decomposed views are what the engine runs: one
utility per stakeholder, and the ordinary constraints alone. The
transformations are tested on raw mappings, because that is the input they
exist to handle.
"""

import jax.numpy as jnp

from lcm import (
    CollectiveUtility,
    Phased,
    ProjectedRegimeValue,
    ValueDependentConstraint,
)
from lcm.regime import decompose_constraints, decompose_functions
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


_REFERENCE = ProjectedRegimeValue(
    regime="single_f", projection={"wealth": _identity_wealth}
)


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
