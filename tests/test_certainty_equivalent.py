"""Tests for nonlinear certainty equivalents over the continuation value."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import PowerCertaintyEquivalent, TransformedExpectation
from lcm.exceptions import RegimeInitializationError
from lcm.typing import FloatND


def test_power_certainty_equivalent_transform_and_inverse_are_inverses():
    """`inverse(transform(x)) == x` for positive values."""
    ce = PowerCertaintyEquivalent()
    x = jnp.array([0.5, 1.0, 2.0, 7.5])
    roundtrip = ce.inverse(
        value=ce.transform(value=x, risk_aversion=jnp.asarray(0.5)),
        risk_aversion=jnp.asarray(0.5),
    )
    np.testing.assert_allclose(roundtrip, x, rtol=1e-6)


def test_power_certainty_equivalent_param_names():
    """The power CE declares exactly the `risk_aversion` runtime param."""
    assert PowerCertaintyEquivalent().param_names == frozenset({"risk_aversion"})


def test_transformed_expectation_param_names_union_over_both_callables():
    """`param_names` is the union of transform and inverse args minus `value`."""

    def g(value: FloatND, theta: FloatND) -> FloatND:
        return value * theta

    def g_inv(value: FloatND, theta: FloatND, offset: FloatND) -> FloatND:
        return value / theta + offset

    ce = TransformedExpectation(transform=g, inverse=g_inv)
    assert ce.param_names == frozenset({"theta", "offset"})


def test_transformed_expectation_rejects_callable_without_value_arg():
    """Both callables must take the value array via an argument named `value`."""

    def g(v: FloatND) -> FloatND:
        return v

    def g_inv(value: FloatND) -> FloatND:
        return value

    with pytest.raises(RegimeInitializationError, match="value"):
        TransformedExpectation(transform=g, inverse=g_inv)
