"""Built-in Koopmans aggregators.

A regime's Koopmans aggregator `W` combines current period utility with the
certainty-equivalent continuation value into the state-action value; its
functional form is where time preference enters. These are the two standard
specifications; pass one via `koopmans_aggregator=...` on the `Regime` or the
`Model` (the model-level default is `LinearAggregator()`).

They are specification objects, matching the certainty-equivalent slot, and
they name functional forms: `LinearAggregator` is the `rho = 1` member of the
family `CESAggregator` parameterizes, up to the normalization that rescales
the value function without touching policies. Neither name states a preference
class — `CESAggregator()` is Epstein-Zin only alongside a `PowerMean()`
certainty equivalent, and `LinearAggregator()` gives time-additive preferences
only alongside `LinearExpectation()`.

The slot also takes any callable, so a form neither class covers stays a
plain function of `utility`, `CE`, and its own parameters.

`CESAggregator` is a weighted power mean, the same object as the `PowerMean`
certainty equivalent it is usually paired with — one averages `(utility, CE)`
at exponent `1 - 1/psi`, the other the continuation lottery at exponent
`1 - risk_aversion` — so both route through one stable evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from _lcm.power_mean import weighted_power_mean_of_pair
from lcm.typing import FloatND

__all__ = ["CESAggregator", "KoopmansAggregator", "LinearAggregator"]


class KoopmansAggregator(ABC):
    """Base class for Koopmans-aggregator specifications.

    Declared on a non-terminal `Regime` via `koopmans_aggregator=...`. The
    shipped implementations are `LinearAggregator` (the default) and
    `CESAggregator`.

    An aggregator is *called*, not dispatched on: `utility` and `CE` are wired
    directly by the Bellman step, and every further parameter of `__call__` is
    resolved by name — from the params template under the pseudo-function key
    `koopmans_aggregator`, or, where the name matches a regime function, from
    that function's output. The signature is therefore the sole declaration of
    what an aggregator consumes, which is why this class does not carry a
    parameter-name property the way `CertaintyEquivalent` does. Subclassing is
    a convenience: the slot accepts any callable with the same convention.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> FloatND:  # noqa: ANN401
        """Return the state-action value from utility and the continuation.

        Called with `utility=...`, `CE=...`, and every further parameter the
        concrete signature declares. Subclasses name those parameters
        explicitly; this declaration only fixes the calling convention.

        Returns:
            The aggregated state-action value, in the same units as its
            inputs.

        """


@dataclass(frozen=True, kw_only=True)
class LinearAggregator(KoopmansAggregator):
    """Aggregate as `U + beta * CE` — the default aggregator.

    Linear in `(utility, CE)`, which is what makes the recursion collapse to
    discounted time-additive utility when the certainty equivalent is
    `LinearExpectation()`. Paired with a curved certainty equivalent it is a
    Kreps-Porteus specification instead, and not time-additive.

    Up to a positive rescaling of the value function this is `CESAggregator`
    at `psi` of infinity: the CES form weights `(1 - beta, beta)` where this
    one weights `(1, beta)`, and `(1 - beta) * V` satisfies the one recursion
    exactly when `V` satisfies the other. Policies are identical; levels
    differ by that factor.
    """

    def __call__(
        self, utility: FloatND, CE: FloatND, discount_factor: FloatND
    ) -> FloatND:
        """Return `utility + discount_factor * CE`.

        Args:
            utility: Current-period utility.
            CE: Certainty equivalent of the continuation value.
            discount_factor: `beta`, in `[0, 1]`.

        Returns:
            The aggregated state-action value.

        """
        return utility + discount_factor * CE


@dataclass(frozen=True, kw_only=True)
class CESAggregator(KoopmansAggregator):
    """Aggregate as `((1-beta)*U^rho + beta*CE^rho)^(1/rho)` — the CES form.

    This is the weighted power mean of `(utility, CE)` at weights
    `(1 - beta, beta)` and exponent `rho`, so it is the same object as the
    `PowerMean` certainty equivalent one level in and shares its evaluation.
    The runtime parameter is the intertemporal elasticity of substitution
    `psi`; the aggregator curvature is `rho = 1 - 1/psi`. `psi = 1` is the
    Cobb-Douglas (log) limit `U^(1-beta) * CE^beta`, approached smoothly from
    either side.

    The form alone does not make a preference class: pair it with
    `certainty_equivalent=PowerMean()` for the Epstein-Zin recursion, which
    collapses to expected CRRA utility when `risk_aversion = 1 / psi`.
    """

    def __call__(
        self,
        utility: FloatND,
        CE: FloatND,
        discount_factor: FloatND,
        intertemporal_elasticity_of_substitution: FloatND,
    ) -> FloatND:
        """Return the weighted power mean of `(utility, CE)` at exponent `rho`.

        Args:
            utility: Strictly positive current-period utility.
            CE: Strictly positive certainty equivalent of the continuation
                value.
            discount_factor: `beta`, in `[0, 1]`.
            intertemporal_elasticity_of_substitution: `psi`, strictly
                positive.

        Returns:
            The aggregated state-action value, in the same units as its
            inputs.

        """
        return weighted_power_mean_of_pair(
            first=utility,
            second=CE,
            first_weight=1.0 - discount_factor,
            second_weight=discount_factor,
            exponent=1.0 - 1.0 / intertemporal_elasticity_of_substitution,
        )
