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

import inspect

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from _lcm.constraints.ir import describe
from _lcm.constraints.processed import normalize_constraints
from lcm import implies, ref
from lcm.condition import Condition
from lcm.consumption_savings_regime import LiquidMargin, post_decision_lower_bound
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


def test_a_condition_is_callable_like_any_other_constraint() -> None:
    """A condition can be called directly, so existing consumers still work.

    Every consumer of a constraint — the DAG builder, the params template, the
    feasibility mask — reaches it through a signature and a call. A condition
    carries both, so declaring one in `constraints` needs no rewiring of the
    machinery that already consumes predicates.
    """
    condition = ref("savings") >= 0.0

    got = condition(savings=jnp.array([-1.0, 1.0]))

    assert_array_equal(got, jnp.array([False, True]))


def test_a_condition_reports_its_argument_names() -> None:
    """Its argument names are the names it reads, in a stable order."""
    assert (ref("wealth") >= ref("floor")).arg_names == ("floor", "wealth")


def test_an_opaque_condition_keeps_the_wrapped_signature() -> None:
    """Wrapping a predicate does not reorder or rename its arguments."""
    condition = Condition.from_callable(_affordable)

    assert condition.arg_names == ("consumption", "wealth")


def test_an_opaque_condition_keeps_the_wrapped_annotations() -> None:
    """A wrapped predicate's annotations survive, so DAG composition still types.

    The DAG builder rejects an unannotated parameter, so a wrapper that lost
    the annotations would make a previously composable predicate uncomposable.
    """
    condition = Condition.from_callable(_affordable)

    assert inspect.signature(condition).parameters["wealth"].annotation is FloatND


def test_a_single_comparison_is_one_boundary_surface() -> None:
    """The comparison a solver splits its grid on is the one that was written."""
    processed = normalize_constraints(constraints={"borrowing": ref("savings") >= 0.0})

    surfaces = processed["borrowing"].boundary_surfaces

    assert [describe(surface) for surface in surfaces or ()] == ["savings >= 0.0"]


def test_a_conjunction_decomposes_into_one_surface_per_comparison() -> None:
    """Both halves of an `and` bound the admitted region, so both survive."""
    processed = normalize_constraints(
        constraints={"band": (ref("savings") >= 0.0) & (ref("savings") <= 4.0)},
    )

    surfaces = processed["band"].boundary_surfaces

    assert [describe(surface) for surface in surfaces or ()] == [
        "savings >= 0.0",
        "savings <= 4.0",
    ]


@pytest.mark.parametrize(
    "condition",
    [
        (ref("savings") >= 0.0) | (ref("wealth") >= 0.0),
        ~(ref("savings") >= 0.0),
        implies(premise=ref("working") == 1, consequent=ref("savings") >= 0.0),
    ],
)
def test_a_condition_that_is_not_a_conjunction_offers_no_surfaces(
    condition: Condition,
) -> None:
    """Only a conjunction of comparisons decomposes into boundaries.

    Under an `or` the admitted region is a union, and under a negation the
    admitted side of each comparison flips, so neither hands a solver a set of
    surfaces it could split a grid on. Answering `None` says the condition does
    not decompose, which a solver must not read as 'there are no boundaries'.
    """
    processed = normalize_constraints(constraints={"c": condition})

    assert processed["c"].boundary_surfaces is None


def test_an_opaque_predicate_offers_no_surfaces() -> None:
    """A bare callable carries no structure, so there is nothing to decompose."""
    processed = normalize_constraints(constraints={"affordable": _affordable})

    assert processed["affordable"].boundary_surfaces is None


def test_a_declared_bound_arrives_as_the_surface_it_stands_for() -> None:
    """The convenience constructor decomposes exactly as the comparison does.

    A solver proves a declared bound by reading the surface it bounds; were the
    sugar to arrive as something other than the comparison an author could have
    written by hand, the two spellings would need two proofs.
    """
    processed = normalize_constraints(
        constraints={
            "borrowing": post_decision_lower_bound(margin=_LIQUID, lower=-2.0)
        },
    )

    surfaces = processed["borrowing"].boundary_surfaces

    assert [describe(surface) for surface in surfaces or ()] == ["savings >= -2.0"]


def test_the_declaration_the_user_wrote_is_kept() -> None:
    """Normalization adds a reading of the declaration without replacing it.

    Age specialization and the pruning walk both recognise a declaration by its
    own type, so the object the user handed over has to remain reachable.
    """
    declaration = post_decision_lower_bound(margin=_LIQUID, lower=-2.0)

    processed = normalize_constraints(constraints={"borrowing": declaration})

    assert processed["borrowing"].declaration is declaration


def test_a_constraint_reports_the_argument_names_it_is_called_with() -> None:
    """Its argument names are what a caller must supply, in a stable order."""
    processed = normalize_constraints(
        constraints={"solvent": ref("wealth") >= ref("floor")},
    )

    assert processed["solvent"].arg_names == ("floor", "wealth")


def test_a_constraint_materializes_into_a_callable_that_evaluates_it() -> None:
    """The function handed to the DAG admits exactly what the constraint says."""
    processed = normalize_constraints(constraints={"borrowing": ref("savings") >= 0.0})

    built = processed["borrowing"].as_function()

    assert_array_equal(built(savings=jnp.array([-1.0, 1.0])), jnp.array([False, True]))


def test_a_materialized_constraint_keeps_the_name_it_was_declared_under() -> None:
    """The DAG identifies a constraint by name, so the name has to travel."""
    processed = normalize_constraints(constraints={"borrowing": ref("savings") >= 0.0})

    built = processed["borrowing"].as_function()

    assert getattr(built, "__name__", None) == "borrowing"
