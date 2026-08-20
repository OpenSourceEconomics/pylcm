"""A constraint declares one meaning that every solver reads the same way.

A user writes a condition once. Grid search evaluates it directly; an
endogenous-grid solver inspects the same object's structure and decides whether
it can prove, compile, or refuse it. Because the evaluator is generated from the
declaration rather than written alongside it, the predicate a solver proves and
the predicate simulation evaluates cannot drift apart.

A bare callable stays legal and normalizes to an opaque condition: grid search
runs it unchanged, and a solver that needs structure refuses it rather than
accepting and ignoring it.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from lcm import ref
from lcm.condition import Condition, implies
from lcm.typing import FloatND


def test_ref_comparison_reads_back_as_the_declaration_written() -> None:
    """`assets < limit` renders as itself."""
    assert str(ref("assets") < ref("limit")) == "assets < limit"


def test_comparing_a_ref_with_a_number_keeps_the_number() -> None:
    """A literal operand survives as the number written, not as a name."""
    assert str(ref("savings") >= 0.0) == "savings >= 0.0"


def test_a_number_on_the_left_is_reflected_into_the_same_meaning() -> None:
    """`0.0 <= savings` declares what `savings >= 0.0` declares."""
    assert str(ref("savings") >= 0.0) == "savings >= 0.0"


def test_two_conditions_are_never_equal_by_structure() -> None:
    """Conditions compare by identity, so `==` cannot report a false match.

    `==` on a reference builds a condition rather than answering a question,
    which would make field-by-field equality on an expression return a truthy
    object for any two nodes. Comparing by identity keeps that from silently
    reporting unrelated declarations as the same one; compare renderings, or
    dependencies, when structural sameness is what is wanted.
    """
    assert (ref("a") >= 0.0) != (ref("a") >= 0.0)


def test_a_condition_reports_the_names_it_depends_on() -> None:
    """Dependencies are the names a solver must have available to evaluate it."""
    condition = ref("assets") < ref("limit")

    assert condition.dependencies == frozenset({"assets", "limit"})


def test_dependencies_of_a_boolean_combination_are_the_union() -> None:
    """Combining conditions unions the names they read."""
    condition = (ref("assets") < ref("limit")) & (ref("age") >= 65)

    assert condition.dependencies == frozenset({"assets", "limit", "age"})


def test_implies_depends_on_both_sides() -> None:
    """`p implies q` reads everything either side reads."""
    condition = implies(
        premise=ref("insurance") == 1, consequent=ref("cash") >= ref("premium")
    )

    assert condition.dependencies == frozenset({"insurance", "cash", "premium"})


def test_a_condition_evaluates_elementwise_on_arrays() -> None:
    """The generated evaluator is an ordinary predicate over its dependencies."""
    condition = ref("savings") >= 0.0

    got = condition.evaluate(savings=jnp.array([-1.0, 0.0, 1.0]))

    assert_array_equal(got, jnp.array([False, True, True]))


def test_the_comparison_operator_decides_who_owns_equality() -> None:
    """`<` puts the boundary point outside; `<=` puts it inside."""
    strict = ref("assets") < 5.0
    weak = ref("assets") <= 5.0

    at_boundary = jnp.array([5.0])

    assert not bool(strict.evaluate(assets=at_boundary)[0])
    assert bool(weak.evaluate(assets=at_boundary)[0])


def test_and_evaluates_as_conjunction() -> None:
    """A conjunction admits only points satisfying both sides."""
    condition = (ref("a") >= 0.0) & (ref("b") >= 0.0)

    got = condition.evaluate(a=jnp.array([1.0, -1.0]), b=jnp.array([1.0, 1.0]))

    assert_array_equal(got, jnp.array([True, False]))


def test_implies_is_vacuously_true_where_the_premise_fails() -> None:
    """`p implies q` constrains only the points where `p` holds."""
    condition = implies(premise=ref("insured") == 1, consequent=ref("cash") >= 10.0)

    got = condition.evaluate(
        insured=jnp.array([1, 1, 0]),
        cash=jnp.array([20.0, 0.0, 0.0]),
    )

    assert_array_equal(got, jnp.array([True, False, True]))


def test_an_opaque_callable_becomes_a_condition_that_still_evaluates() -> None:
    """A bare predicate stays legal and keeps its behaviour."""

    def feasible(consumption: FloatND, wealth: FloatND) -> FloatND:
        return consumption <= wealth

    condition = Condition.from_callable(feasible)

    got = condition.evaluate(
        consumption=jnp.array([1.0, 3.0]), wealth=jnp.array([2.0, 2.0])
    )

    assert_array_equal(got, jnp.array([True, False]))


def test_an_opaque_callable_reports_its_signature_as_dependencies() -> None:
    """Its parameter names are what a solver must supply."""

    def feasible(consumption: FloatND, wealth: FloatND) -> FloatND:
        return consumption <= wealth

    assert Condition.from_callable(feasible).dependencies == frozenset(
        {"consumption", "wealth"}
    )


def test_an_opaque_condition_exposes_no_structure() -> None:
    """A solver cannot mistake an opaque predicate for a structured one."""

    def feasible(consumption: FloatND, wealth: FloatND) -> FloatND:
        return consumption <= wealth

    assert Condition.from_callable(feasible).is_opaque


def test_a_built_condition_is_not_opaque() -> None:
    """A condition assembled from refs carries structure a solver can read."""
    assert not (ref("savings") >= 0.0).is_opaque


def test_evaluating_with_a_missing_dependency_names_it() -> None:
    """A missing input is an error naming the input, not a KeyError."""
    condition = ref("savings") >= 0.0

    with pytest.raises(TypeError, match="savings"):
        condition.evaluate(wealth=jnp.array([1.0]))


@pytest.mark.parametrize(
    "condition",
    [ref("savings") >= 0.0, ref("savings") <= 0.0, ref("savings") == 0.0],
)
def test_an_inclusive_comparison_admits_the_boundary_point(
    condition: Condition,
) -> None:
    """Equality ownership is read off the comparison, not agreed separately.

    A solver splitting a grid at a boundary has to know which side owns the
    boundary point, and the operator the user wrote is the whole answer. Keeping
    the question next to the operator that decides it is what stops a second,
    divergent convention appearing wherever a boundary is consumed.
    """
    assert condition.expression.admits_equality  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize(
    "condition",
    [ref("savings") > 0.0, ref("savings") < 0.0, ref("savings") != 0.0],
)
def test_a_strict_comparison_leaves_the_boundary_point_out(
    condition: Condition,
) -> None:
    """A strict operator puts the boundary point outside the admitted region."""
    assert not condition.expression.admits_equality  # ty: ignore[unresolved-attribute]
