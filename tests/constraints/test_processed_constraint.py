"""Every constraint reaches a solver in one normalized form.

Whatever a user writes — a structured condition, a declared post-decision
bound, or a bare predicate — model processing turns it into a
`ProcessedConstraint` carrying the same four things: what it is called, what it
says, how to evaluate it, and which names it reads. A solver therefore never
has to ask what kind of constraint it is holding before it can evaluate it, and
never has to guess which names it needs available.

Structure is what varies. A bare predicate normalizes to an opaque condition,
so a solver that must reason about a constraint can tell it apart from one it
can prove, and refuse it rather than accepting it and ignoring what it says.
"""

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from _lcm.constraints.processed import normalize_constraints
from lcm import ref
from lcm.regime import LiquidMargin, post_decision_lower_bound
from lcm.typing import FloatND

_LIQUID = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


def _affordable(consumption: FloatND, wealth: FloatND) -> FloatND:
    return consumption <= wealth


def test_a_bare_predicate_keeps_its_name() -> None:
    """The key it was declared under is the name it is reported by."""
    processed = normalize_constraints(constraints={"affordable": _affordable})

    assert processed["affordable"].name == "affordable"


def test_a_bare_predicate_reports_its_signature_as_dependencies() -> None:
    """Its parameter names are what a solver must supply."""
    processed = normalize_constraints(constraints={"affordable": _affordable})

    assert processed["affordable"].dependencies == frozenset({"consumption", "wealth"})


def test_a_bare_predicate_evaluates_as_it_did() -> None:
    """Normalizing does not change what the predicate admits."""
    processed = normalize_constraints(constraints={"affordable": _affordable})

    got = processed["affordable"].evaluate(
        consumption=jnp.array([1.0, 3.0]), wealth=jnp.array([2.0, 2.0])
    )

    assert_array_equal(got, jnp.array([True, False]))


def test_a_bare_predicate_is_opaque() -> None:
    """It offers a solver nothing to reason about."""
    processed = normalize_constraints(constraints={"affordable": _affordable})

    assert processed["affordable"].is_opaque


def test_a_declared_condition_is_not_opaque() -> None:
    """A condition built from references carries structure a solver can read."""
    processed = normalize_constraints(
        constraints={"borrowing": ref("savings") >= 0.0},
    )

    assert not processed["borrowing"].is_opaque


def test_a_declared_condition_reports_the_names_it_reads() -> None:
    """Dependencies come from the expression, not from a signature."""
    processed = normalize_constraints(
        constraints={"borrowing": ref("savings") >= 0.0},
    )

    assert processed["borrowing"].dependencies == frozenset({"savings"})


def test_a_post_decision_lower_bound_normalizes_to_a_readable_comparison() -> None:
    """The declared bound reaches a solver as structure, not as a callable.

    A solver proves this constraint by comparing the declared bound with its
    savings grid, which it can only do if the bound survives normalization as a
    comparison rather than as an opaque predicate.
    """
    processed = normalize_constraints(
        constraints={"borrowing": post_decision_lower_bound(margin=_LIQUID, lower=0.0)},
    )

    assert str(processed["borrowing"].condition) == "savings >= 0.0"


def test_a_post_decision_lower_bound_still_evaluates_as_a_predicate() -> None:
    """Structure does not cost it its behaviour under a solver that evaluates."""
    processed = normalize_constraints(
        constraints={"borrowing": post_decision_lower_bound(margin=_LIQUID, lower=0.0)},
    )

    got = processed["borrowing"].evaluate(savings=jnp.array([-1.0, 0.0, 1.0]))

    assert_array_equal(got, jnp.array([False, True, True]))


def test_normalizing_nothing_yields_nothing() -> None:
    """A regime declaring no constraints normalizes to an empty mapping."""
    assert dict(normalize_constraints(constraints={})) == {}
