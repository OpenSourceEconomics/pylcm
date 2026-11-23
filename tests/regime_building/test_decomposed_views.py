"""What the engine reads when a regime is written in the declaration vocabulary.

A regime's `functions` and `constraints` are what the author wrote —
declarations included. The decomposed function view is what the engine runs: one
utility per member. The transformations are tested on raw mappings, because that
is the input they exist to handle.
"""

import jax.numpy as jnp

from lcm import CollectiveUtility, Phased
from lcm.regime import decompose_constraints, decompose_functions
from lcm.typing import ContinuousState, FloatND


def _u_f(wealth: ContinuousState) -> FloatND:
    """The first member's flow utility."""
    return jnp.log(wealth)


def _u_m(wealth: ContinuousState) -> FloatND:
    """The second member's flow utility."""
    return 0.5 * jnp.log(wealth)


def _u_m_simulate(wealth: ContinuousState) -> FloatND:
    """The second member's flow utility as simulation realizes it."""
    return 0.25 * jnp.log(wealth)


def _budget(wealth: ContinuousState) -> FloatND:
    """An ordinary feasibility constraint."""
    return wealth > 0.0


def test_a_collective_utility_becomes_one_entry_per_member() -> None:
    """Each member's body lands under the name the engine reads."""
    raw = {"utility": CollectiveUtility(utilities={"f": _u_f, "m": _u_m})}

    assert dict(decompose_functions(raw)) == {"utility_f": _u_f, "utility_m": _u_m}


def test_the_declaration_object_never_reaches_the_engine() -> None:
    """Nothing under `"utility"` survives decomposition."""
    raw = {"utility": CollectiveUtility(utilities={"f": _u_f, "m": _u_m})}

    assert "utility" not in decompose_functions(raw)


def test_a_delegated_body_is_taken_from_the_entry_it_delegates_to() -> None:
    """A `None` body means the regime's own `utility_<s>` is that utility."""
    raw = {
        "utility": CollectiveUtility(utilities={"f": None, "m": _u_m}),
        "utility_f": _u_f,
    }

    assert decompose_functions(raw)["utility_f"] is _u_f


def test_member_entries_follow_the_declaration_not_the_mapping() -> None:
    """The household's own order survives however the entries arrived."""
    raw = {
        "utility_m": _u_m,
        "utility": CollectiveUtility(utilities={"f": _u_f, "m": None}),
    }

    assert list(decompose_functions(raw)) == ["utility_f", "utility_m"]


def test_a_regime_declaring_no_household_is_left_alone() -> None:
    """A singleton regime's functions pass through unchanged."""
    raw = {"utility": _u_f, "helper": _budget}

    assert decompose_functions(raw) is raw


def test_decomposing_functions_twice_changes_nothing() -> None:
    """The transformation is idempotent, so a second engine stage is safe."""
    raw = {"utility": CollectiveUtility(utilities={"f": _u_f, "m": _u_m})}
    once = decompose_functions(raw)

    assert dict(decompose_functions(once)) == dict(once)


def test_a_phased_member_body_reaches_the_engine_whole() -> None:
    """The phase split happens downstream, so the container survives intact."""
    body = Phased(solve=_u_m, simulate=_u_m_simulate)
    raw = {"utility": CollectiveUtility(utilities={"m": body, "f": _u_f})}

    assert decompose_functions(raw)["utility_m"] is body


def test_decomposing_constraints_twice_changes_nothing() -> None:
    """The transformation is idempotent."""
    raw = {"budget": _budget}
    once = decompose_constraints(raw)

    assert dict(decompose_constraints(once)) == dict(once)
