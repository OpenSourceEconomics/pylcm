"""Built-in Koopmans aggregators.

A regime's Koopmans aggregator `W` combines current period utility with the
certainty-equivalent continuation value into the state-action value; its
functional form is where time preference enters. These are the two standard
specifications; pass one via `koopmans_aggregator=...` on the `Regime` or the
`Model` (the model-level default is `W_linear`).

`W_epstein_zin` is a weighted power mean, the same object as the `PowerMean`
certainty equivalent it is usually paired with — one averages `(utility, CE)`
at exponent `1 - 1/psi`, the other the continuation lottery at exponent
`1 - risk_aversion` — so both route through one stable evaluation.
"""

from _lcm.power_mean import weighted_power_mean_of_pair
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

    This is the weighted power mean of `(U, CE)` at weights
    `(1 - beta, beta)` and exponent `rho`, so it is the same object as the
    `PowerMean` certainty equivalent one level in and shares its evaluation.
    The runtime parameter is the intertemporal elasticity of substitution
    `psi`; the aggregator curvature is `rho = 1 - 1/psi`. `psi = 1` is the
    Cobb-Douglas (log) limit `U^(1-beta) * CE^beta`, approached smoothly from
    either side. Pair with `certainty_equivalent=PowerMean()` for the full
    Epstein-Zin recursion.

    Args:
        utility: Strictly positive current-period utility.
        CE: Strictly positive certainty equivalent of the continuation value.
        discount_factor: `beta`, in `[0, 1]`.
        intertemporal_elasticity_of_substitution: `psi`, strictly positive.

    Returns:
        The aggregated state-action value, in the same units as its inputs.

    """
    return weighted_power_mean_of_pair(
        first=utility,
        second=CE,
        first_weight=1.0 - discount_factor,
        second_weight=discount_factor,
        exponent=1.0 - 1.0 / intertemporal_elasticity_of_substitution,
    )
