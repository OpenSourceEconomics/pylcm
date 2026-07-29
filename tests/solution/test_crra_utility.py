"""Shared CRRA felicity is differentiable at the log case.

`crra_utility` selects between the log branch and the power branch on an exact
`crra == 1.0` test. Both branches are evaluated whichever is selected, so the
*unselected* one has to stay finite: an infinite value there is discarded from
the primal but reaches the derivative as `0 * inf`, and poisons it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.crra import crra_utility


@pytest.mark.parametrize("crra", [1.0, 2.0, 0.5])
def test_crra_utility_matches_its_closed_form(crra: float) -> None:
    """Felicity is `log(c)` at `crra == 1` and `c^(1-crra)/(1-crra)` elsewhere."""
    consumption = 2.0
    expected = (
        np.log(consumption)
        if crra == 1.0
        else consumption ** (1.0 - crra) / (1.0 - crra)
    )
    got = crra_utility(jnp.asarray(consumption), crra)
    np.testing.assert_allclose(float(got), expected, rtol=1e-6)


@pytest.mark.parametrize("crra", [1.0, 2.0, 0.5])
def test_crra_marginal_utility_is_finite_and_equals_c_to_the_minus_crra(
    crra: float,
) -> None:
    """`d/dc` is `c^(-crra)` at every `crra`, including the log case.

    At `crra == 1` the power branch is `c^0 / 0`, which is infinite. It is not
    selected, but `jnp.where` evaluates it regardless, so an implementation that
    leaves it infinite returns `nan` here rather than `1 / c`.
    """
    consumption = 2.0
    got = jax.grad(lambda c: crra_utility(c, crra))(jnp.asarray(consumption))
    np.testing.assert_allclose(float(got), consumption**-crra, rtol=1e-6)


def test_crra_coefficient_derivative_is_finite_at_the_log_case() -> None:
    """`d/d(crra)` exists at `crra == 1`, where the felicity is `log(c)`.

    The log branch does not depend on `crra`, so the derivative is zero. An
    unselected power branch left at `c^0 / 0` makes it `nan` instead — the
    same poisoning as for the consumption derivative, reached through the
    other argument.
    """
    consumption = 2.0
    got = jax.grad(crra_utility, argnums=1)(jnp.asarray(consumption), 1.0)
    np.testing.assert_allclose(float(got), 0.0, atol=1e-12)
