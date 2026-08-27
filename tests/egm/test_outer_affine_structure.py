"""The declared outer coefficient must be exact, and refuse what it cannot certify.

N-NB-EGM inverts the outer post-decision map to recover the action that reaches a
retained target. Recovering the map's slope by differencing two rounded probes is
what made that inversion wrong: for `target = state + action` the true slope is
exactly one, but the two-point secant returns a value one ULP away, and dividing
by it carries the reassembled stock off its declared domain.

The slope is therefore read off the *structure* of the traced function rather than
measured. A forward walk over the jaxpr seeds the outer action with coefficient one
and propagates exact rational arithmetic through the affine primitives; anything
else touching the action refuses the map instead of approximating it.

Automatic differentiation is not an acceptable instrument for this. `stop_gradient`
is invisible to it, so a map whose slope is three linearizes as one — the walk has
to see the arithmetic, not its derivative.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

from _lcm.egm.outer_affine_structure import certify_outer_coefficient


def _certify(func):
    """Certify `func(state, action)` in the outer action, at scalar fills."""
    return certify_outer_coefficient(
        func=func,
        outer_action_name="action",
        abstract_args=(jnp.float32(1.7), jnp.float32(0.25)),
        arg_names=("state", "action"),
    )


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        pytest.param(lambda state, action: state + action, Fraction(1), id="unit"),
        pytest.param(
            lambda state, action: state + 2 * action,
            Fraction(2),
            id="dyadic-2",
        ),
        pytest.param(
            lambda state, action: state + 0.5 * action,
            Fraction(1, 2),
            id="dyadic-half",
        ),
        pytest.param(
            lambda state, action: state + 3 * action,
            Fraction(3),
            id="non-dyadic-3",
        ),
        pytest.param(
            lambda state, action: state - (-action),
            Fraction(1),
            id="double-negation",
        ),
        pytest.param(
            lambda state, action: state + action - action,
            Fraction(0),
            id="cancels-to-zero",
        ),
        pytest.param(
            lambda state, action: state - action,
            Fraction(-1),
            id="negative-unit",
        ),
    ],
)
def test_an_affine_map_certifies_its_exact_coefficient(func, expected) -> None:
    """An affine dependence on the outer action reports its coefficient exactly."""
    assert _certify(func).coefficient == expected


def test_a_stop_gradient_contribution_is_counted_not_ignored() -> None:
    """`stop_gradient` hides a term from AD but not from the map itself.

    The map below moves three units of stock per unit of action. Linearizing it
    reports one, so an AD-based certificate would admit it as unit-slope and the
    recovered action would be three times too small.
    """
    certificate = _certify(
        lambda state, action: state + action + jax.lax.stop_gradient(2 * action)
    )
    assert certificate.coefficient == Fraction(3)


def test_the_secant_and_the_walk_disagree_where_it_matters() -> None:
    """The two-point probe misreads a slope the walk reads exactly.

    This is the defect being removed, stated as a comparison: for the unit law the
    secant is not one, while the certified coefficient is exactly one.
    """
    law = lambda state, action: state + action  # noqa: E731
    # Not every state exposes the secant error -- at `1.7` the probe is exact.
    # This is the measured witness, where it is one ULP low.
    state = jnp.float32(3.8263208866119385)
    secant = law(state, jnp.float32(1.0)) - law(state, jnp.float32(0.0))

    assert float(secant) != 1.0
    assert _certify(law).coefficient == Fraction(1)


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(
            lambda state, action: state + jnp.clip(action, 0.0, 5.0),
            id="clip",
        ),
        pytest.param(lambda state, action: state + action**2, id="power"),
        pytest.param(
            lambda state, action: state + action / state,
            id="state-dependent-slope",
        ),
        pytest.param(
            lambda state, action: state + jnp.where(action > 0, action, 0.0),
            id="where",
        ),
        pytest.param(lambda state, action: state + jnp.exp(action), id="exp"),
    ],
)
def test_a_map_that_is_not_affine_in_the_outer_action_is_refused(func) -> None:
    """A non-affine dependence reports no coefficient and names what it saw."""
    certificate = _certify(func)
    assert certificate.coefficient is None
    assert certificate.violation is not None


def test_a_map_ignoring_the_outer_action_certifies_a_zero_coefficient() -> None:
    """A map that never reads the action has coefficient zero, not a refusal.

    Zero is certified rather than refused because the refusal belongs to the
    caller: a zero slope is exactly invertible nowhere, which is a different
    complaint from an unreadable structure.
    """
    assert _certify(lambda state, _action: state * 2.0).coefficient == Fraction(0)


def test_a_state_dependent_term_not_touching_the_action_is_ignored() -> None:
    """Arbitrary nonlinearity is fine as long as it never reaches the action."""
    certificate = _certify(
        lambda state, action: jnp.exp(state) + jnp.sin(state) + action
    )
    assert certificate.coefficient == Fraction(1)


def test_a_jit_wrapped_law_is_certified_through_its_sub_jaxpr() -> None:
    """A law behind `jax.jit` is still read; the walk descends into the call."""
    inner = jax.jit(lambda value: value)
    assert _certify(
        lambda state, action: state + inner(action)
    ).coefficient == Fraction(1)


def test_an_untraceable_map_is_refused_rather_than_assumed() -> None:
    """A map that will not trace is refused; an unverified slope is not a slope."""

    def hostile(_state, _action):
        raise RuntimeError("cannot trace")

    certificate = _certify(hostile)
    assert certificate.coefficient is None
    assert certificate.violation is not None
