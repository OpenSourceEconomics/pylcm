"""CRRA felicity: log limit, power branch, and finite derivatives at the limit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.crra import crra_utility


def test_crra_utility_is_the_log_at_the_unit_coefficient() -> None:
    """`crra == 1` returns `log(c)`."""
    np.testing.assert_allclose(crra_utility(jnp.array(2.0), 1.0), np.log(2.0))


def test_crra_utility_is_the_power_form_away_from_the_unit_coefficient() -> None:
    """`crra != 1` returns `c ** (1 - crra) / (1 - crra)`."""
    np.testing.assert_allclose(crra_utility(jnp.array(2.0), 3.0), 2.0**-2 / -2.0)


def test_crra_utility_consumption_derivative_is_finite_at_the_log_limit() -> None:
    """`d/dc log(c)` is `1 / c` at `crra == 1`, not NaN."""
    slope = jax.grad(crra_utility)(jnp.array(2.0), 1.0)
    np.testing.assert_allclose(slope, 0.5)


def test_crra_utility_coefficient_derivative_is_finite_at_the_log_limit() -> None:
    """The derivative with respect to `crra` is zero at the log limit, not NaN."""
    slope = jax.grad(crra_utility, argnums=1)(jnp.array(2.0), 1.0)
    np.testing.assert_allclose(slope, 0.0)


@pytest.mark.parametrize("crra", [0.5, 2.0])
def test_crra_utility_consumption_derivative_matches_the_power_rule(
    crra: float,
) -> None:
    """Away from the limit the consumption derivative is `c ** (-crra)`."""
    slope = jax.grad(crra_utility)(jnp.array(2.0), crra)
    np.testing.assert_allclose(slope, 2.0 ** (-crra), rtol=1e-12)
