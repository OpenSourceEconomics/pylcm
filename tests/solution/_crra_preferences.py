"""A closed-form CRRA `Preferences` bundle for the EGM step unit tests.

The steps take felicity, its marginal, and its inverse marginal as callables, so
a unit test that drives one directly supplies them itself. CRRA is the family the
DS pension benchmark and the G2EGM reference are calibrated in, so the three
closed forms live here once rather than in every step test.
"""

import jax.numpy as jnp

from _lcm.egm.preferences import Preferences
from lcm.typing import FloatND


def crra_preferences(crra: float, *, disutility: float = 0.0) -> Preferences:
    """Build the CRRA bundle `u(c) = c**(1-crra) / (1-crra) - disutility`.

    Args:
        crra: Coefficient of relative risk aversion. `1.0` selects the log
            limit, whose `1 / (1 - crra)` level offset this form omits.
        disutility: Additive level shift on the felicity — an agent's flow cost
            of the discrete state the step is solving in. It leaves the marginal
            and its inverse untouched.

    Returns:
        The felicity, marginal felicity, and inverse marginal felicity, each a
        callable of one array.

    """

    def utility(consumption: FloatND) -> FloatND:
        # `jnp.where` evaluates both branches, so the unselected exponent is
        # replaced by one: at `crra == 1` the power branch would be `c**0 / 0`,
        # whose infinity reaches the derivative as `0 * inf`.
        is_log = crra == 1.0
        safe_exponent = jnp.where(is_log, 1.0, 1.0 - crra)
        felicity = jnp.where(
            is_log,
            jnp.log(consumption),
            consumption**safe_exponent / safe_exponent,
        )
        return felicity - disutility

    def marginal_utility(consumption: FloatND) -> FloatND:
        return consumption ** (-crra)

    def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
        return marginal_continuation ** (-1.0 / crra)

    return Preferences(
        utility=utility,
        marginal_utility=marginal_utility,
        inverse_marginal_utility=inverse_marginal_utility,
    )
