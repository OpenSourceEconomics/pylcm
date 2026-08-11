"""The Euler inversion treats its inverse-marginal map as a unary callable.

`invert_euler` receives the regime's inverse-marginal-utility map with every
parameter but the marginal continuation already bound, so what reaches it is a
one-argument function. Its parameter *name* is an implementation detail of
whoever built it — the lifted form produced for grid-valued steps names it after
the array it maps, an analytic inverse names it after the economics — and the
inversion must not depend on which.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from _lcm.egm.euler import invert_euler
from lcm.typing import ScalarFloat
from tests.conftest import DECIMAL_PRECISION


def _crra_inverse(any_parameter_name: ScalarFloat) -> ScalarFloat:
    """Inverse marginal utility of $u(c) = \\log c$, i.e. $c = 1 / u'$."""
    return 1.0 / any_parameter_name


def test_inverse_marginal_map_need_not_name_its_argument() -> None:
    """A unary map whose parameter carries any name inverts the Euler equation."""
    action = invert_euler(
        expected_marginal_continuation=jnp.asarray(0.5),
        discount_factor=jnp.asarray(0.96),
        inverse_marginal_utility=_crra_inverse,
    )
    assert_allclose(action, 1.0 / (0.96 * 0.5), rtol=10.0**-DECIMAL_PRECISION)


def test_a_marginal_continuation_of_zero_yields_a_finite_action() -> None:
    """The degenerate inversion is clamped, so no infinite endogenous point appears."""
    action = invert_euler(
        expected_marginal_continuation=jnp.asarray(0.0),
        discount_factor=jnp.asarray(0.96),
        inverse_marginal_utility=_crra_inverse,
    )
    assert jnp.isfinite(action)
    assert action == pytest.approx(1.0 / jnp.finfo(action.dtype).eps, rel=1e-6)
