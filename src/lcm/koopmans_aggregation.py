"""Built-in Koopmans aggregators.

A regime's Koopmans aggregator `W` combines current period utility with the
certainty-equivalent continuation value into the state-action value; its
functional form is where time preference enters. These are the two standard
specifications; pass one via `koopmans_aggregator=...` on the `Regime` or the
`Model` (the model-level default is `W_linear`).
"""

import jax.numpy as jnp

from lcm.typing import FloatND

__all__ = ["W_epstein_zin", "W_linear"]


def W_linear(utility: FloatND, CE: FloatND, discount_factor: FloatND) -> FloatND:
    """Aggregate as `U + β · CE` — expected utility, and the default aggregator."""
    return utility + discount_factor * CE


def W_epstein_zin(
    utility: FloatND,
    CE: FloatND,
    discount_factor: FloatND,
    intertemporal_elasticity_of_substitution: FloatND,
) -> FloatND:
    """Aggregate as `((1-beta)*U^rho + beta*CE^rho)^(1/rho)` — the Epstein-Zin form.

    The runtime parameter is the intertemporal elasticity of substitution
    `psi`; the aggregator curvature is `rho = 1 - 1/psi`. `psi = 1` is the
    Cobb-Douglas (log) limit `U^(1-beta) * CE^beta`. Pair with
    `certainty_equivalent=PowerMean()` for the full Epstein-Zin recursion;
    both `U` and `CE` must be strictly positive.
    """
    rho = 1.0 - 1.0 / intertemporal_elasticity_of_substitution
    # The unselected CES branch must not divide by zero at `ψ = 1`.
    safe_rho = jnp.where(rho == 0.0, 1.0, rho)
    cobb_douglas = utility ** (1.0 - discount_factor) * CE**discount_factor
    ces = (
        (1.0 - discount_factor) * utility**safe_rho + discount_factor * CE**safe_rho
    ) ** (1.0 / safe_rho)
    return jnp.where(rho == 0.0, cobb_douglas, ces)
