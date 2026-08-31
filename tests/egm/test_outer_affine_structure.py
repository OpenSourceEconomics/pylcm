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

from _lcm.egm.outer_affine_structure import (
    _PASS_THROUGH_PRIMS,
    certify_outer_coefficient,
)


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


@pytest.mark.parametrize(
    "dtype",
    [jnp.float16, jnp.bfloat16, jnp.int32],
    ids=["float16", "bfloat16", "int32"],
)
def test_a_lossy_action_conversion_is_refused(dtype) -> None:
    """A dtype conversion that quantizes the action is not an affine identity.

    `convert_element_type` preserves array shape while discarding action bits, so
    treating it as coefficient-preserving would certify a quantized transition as
    slope one. The recovered action's forward image then differs from the target
    by far more than a rounding: a float16 round trip moves a target near ten by
    3e-3, and an int32 round trip by 3e-1. Containment alone cannot catch that,
    because the wrong image is still inside the declared domain.
    """
    certificate = _certify(
        lambda state, action: state + action.astype(dtype).astype(state.dtype)
    )

    assert certificate.coefficient is None


@pytest.mark.parametrize(
    "dtype",
    [jnp.float16, jnp.bfloat16],
    ids=["float16", "bfloat16"],
)
def test_a_widening_action_conversion_is_refused_too(dtype) -> None:
    """A conversion is refused for being a conversion, not for losing bits here.

    Widening the action back out of a narrow dtype preserves every value it
    still holds, so this direction is arithmetically safe. It is refused anyway:
    the walk certifies from structure, and a rule admitting conversions it can
    prove exact would have to carry a per-dtype exactness table and be right
    about every pair. No supported map converts the action, so the narrow
    refusal costs nothing a model needs, and a map that genuinely needs one is
    solved by `GridSearch`.
    """
    certificate = _certify(lambda state, action: state + action.astype(dtype))

    assert certificate.coefficient is None


def test_a_conversion_hidden_inside_a_jit_wrapper_is_still_refused() -> None:
    """A nested call cannot launder a conversion past the certificate.

    The walk descends into sub-jaxprs, so a quantizing hop wrapped in `jit` has
    to be refused exactly as the bare one is. If descent stopped at the call
    primitive the wrapper would be an opaque box certifying slope one, which is
    the same false certificate with one more layer around it.
    """

    @jax.jit
    def quantize(action):
        return action.astype(jnp.float16).astype(jnp.float32)

    certificate = _certify(lambda state, action: state + quantize(action))

    assert certificate.coefficient is None


def test_a_four_fold_dyadic_coefficient_still_certifies_exactly() -> None:
    """Refusing conversions leaves the supported dyadic coefficients untouched."""
    certificate = _certify(lambda state, action: state + 4 * action)

    assert certificate.coefficient == Fraction(4)


def test_the_conversion_primitive_is_not_a_pass_through() -> None:
    """`convert_element_type` must stay out of the pass-through set.

    Named directly so that restoring it fails on the rule itself and not only
    through the behaviour tests above. A pass-through entry asserts that a
    primitive preserves the action's value, and this one preserves only its
    shape.
    """
    assert "convert_element_type" not in _PASS_THROUGH_PRIMS
