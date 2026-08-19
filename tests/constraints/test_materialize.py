"""A constraint adopts its regime's annotations rather than asserting its own.

The DAG requires every consumer of a value to annotate it the way its producer
returns it. A constraint therefore cannot annotate itself: the same name is a
continuous value in one regime and an integer-coded state in another, and only
the surrounding functions know which. Reading the annotation off the regime's
own function pool is what lets a constraint over a discrete state compose at
all, rather than failing the DAG's consistency check.
"""

import inspect

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from _lcm.constraints.materialize import annotation_of, as_constraint_function
from _lcm.constraints.processed import normalize_constraints
from lcm import ref
from lcm.typing import DiscreteState, FloatND


def _savings(wealth: FloatND) -> FloatND:
    return wealth


def _health(age: FloatND) -> DiscreteState:
    return jnp.int32(age > 0)


def _uses_health(health: DiscreteState) -> FloatND:
    return jnp.float32(health)


def test_a_name_takes_the_return_annotation_of_the_function_producing_it() -> None:
    """A computed name is annotated as its producer returns it."""
    assert annotation_of(pool={"savings": _savings}, name="savings") == "FloatND"


def test_a_discrete_name_keeps_its_discrete_annotation() -> None:
    """An integer-coded state is not silently reported as continuous."""
    assert annotation_of(pool={"health": _health}, name="health") == "DiscreteState"


def test_a_name_only_consumed_takes_the_annotation_its_consumer_declares() -> None:
    """A name nothing produces is annotated as the first consumer declares it."""
    assert annotation_of(pool={"uses": _uses_health}, name="health") == "DiscreteState"


def test_a_name_the_pool_says_nothing_about_falls_back_to_continuous() -> None:
    """A name supplied at runtime rather than computed is a continuous value."""
    assert annotation_of(pool={"savings": _savings}, name="wage") == "FloatND"


def test_the_built_function_adopts_the_pool_annotation() -> None:
    """A condition over a discrete state composes with that state's producer.

    Annotating it continuously would make the DAG reject the pair, so this is
    what lets a discrete-only constraint be declared at all.
    """
    processed = normalize_constraints(constraints={"hale": ref("health") == 1})

    built = as_constraint_function(
        constraint=processed["hale"], pool={"health": _health}
    )

    annotation = inspect.signature(built).parameters["health"].annotation
    assert annotation == "DiscreteState"


def test_the_built_function_keeps_the_constraint_name() -> None:
    """It is reported under the name the constraint was declared with.

    The constraint's identity comes from its mapping key, so carrying that key
    onto the function is what makes a traceback or a DAG diagnostic name the
    constraint the author wrote rather than a generic wrapper.
    """
    processed = normalize_constraints(constraints={"hale": ref("health") == 1})

    built = as_constraint_function(
        constraint=processed["hale"], pool={"health": _health}
    )

    # `ConstraintFunction` is a Protocol and does not declare `__name__`,
    # though every function carries one.
    assert getattr(built, "__name__", None) == "hale"


def test_the_built_function_evaluates_as_the_constraint_does() -> None:
    """Adopting an annotation does not change what the constraint admits."""
    processed = normalize_constraints(constraints={"solvent": ref("savings") >= 0.0})

    built = as_constraint_function(
        constraint=processed["solvent"], pool={"savings": _savings}
    )

    assert_array_equal(built(savings=jnp.array([-1.0, 1.0])), jnp.array([False, True]))
