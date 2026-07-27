"""Built-in Koopmans aggregators.

A regime's `H` combines current period utility with the certainty-equivalent
continuation value into the state-action value; its functional form is where
time preference enters. These are the two standard specifications; pass them
via `functions={"H": ...}` (a regime without an explicit `H` gets `H_linear`).
"""

from _lcm.egm.ez_kernel import ez_period_value
from lcm.typing import FloatND

__all__ = ["H_epstein_zin", "H_linear"]


def H_linear(utility: FloatND, E_next_V: FloatND, discount_factor: FloatND) -> FloatND:
    """Aggregate as `U + β · CE` — the expected-utility form, and the default `H`."""
    return utility + discount_factor * E_next_V


def H_epstein_zin(
    utility: FloatND,
    E_next_V: FloatND,
    discount_factor: FloatND,
    intertemporal_elasticity_of_substitution: FloatND,
) -> FloatND:
    """Aggregate as `((1-beta)*U^rho + beta*CE^rho)^(1/rho)` — the Epstein-Zin form.

    The runtime parameter is the intertemporal elasticity of substitution
    `psi`; the aggregator curvature is `rho = 1 - 1/psi`. `psi = 1` is the
    Cobb-Douglas (log) limit `U^(1-beta) * CE^beta`. Pair with
    `certainty_equivalent=PowerMean()` for the full Epstein-Zin recursion;
    both `U` and `CE` must be strictly positive.

    Every solver evaluates the aggregation through one implementation —
    `ez_period_value`, which computes the CES in the log domain — so the
    brute-force and endogenous-grid solvers publish identical cardinal values,
    and the aggregation stays exact where raw CES powers would leave the
    dtype's range.
    """
    return ez_period_value(
        flow=utility,
        nu=E_next_V,
        discount_factor=discount_factor,
        inverse_eis=1.0 / intertemporal_elasticity_of_substitution,
    )
