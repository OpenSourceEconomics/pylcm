from abc import abstractmethod
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf

from lcm.shocks._base import _ShockGrid
from lcm.typing import Float1D, FloatND


@dataclass(frozen=True, kw_only=True)
class _ShockGridIID(_ShockGrid):
    """Base for iid shocks — draw does not depend on previous value."""

    @abstractmethod
    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
    ) -> Float1D: ...


@dataclass(frozen=True, kw_only=True)
class Uniform(_ShockGridIID):
    r"""Discretized iid uniform shock: :math:`U(\text{start}, \text{stop})`.

    The continuous distribution is discretized into ``n_points`` equally spaced
    points between ``start`` and ``stop``.

    Attributes:
        start: Lower bound of the uniform distribution.
        stop: Upper bound of the uniform distribution.

    """

    start: float | None = None
    stop: float | None = None

    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        return jnp.linspace(start=kwargs["start"], stop=kwargs["stop"], num=n_points)

    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:  # noqa: ARG002
        return jnp.full((n_points, n_points), fill_value=1 / n_points)

    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
    ) -> Float1D:
        return jax.random.uniform(
            key=key, minval=params["start"], maxval=params["stop"]
        )


@dataclass(frozen=True, kw_only=True)
class Normal(_ShockGridIID):
    r"""Discretized iid normal shock: :math:`N(\mu_\varepsilon, \sigma_\varepsilon^2)`.

    The continuous distribution is discretized into ``n_points`` equally spaced
    points spanning :math:`\mu_\varepsilon \pm n_\text{std} \cdot \sigma_\varepsilon`.

    Attributes:
        mu: Mean of the shock distribution.
        sigma: Standard deviation of the shock distribution.
        n_std: Number of standard deviations from the mean to the grid boundary.

    """

    mu: float | None = None
    sigma: float | None = None
    n_std: float | None = None

    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        mu, sigma, n_std = kwargs["mu"], kwargs["sigma"], kwargs["n_std"]
        x_min = mu - n_std * sigma
        x_max = mu + n_std * sigma
        return jnp.linspace(start=x_min, stop=x_max, num=n_points)

    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        mu, sigma, n_std = kwargs["mu"], kwargs["sigma"], kwargs["n_std"]
        x_min = mu - n_std * sigma
        x_max = mu + n_std * sigma
        x, stepsize = jnp.linspace(start=x_min, stop=x_max, num=n_points, retstep=True)
        P = jnp.zeros(n_points)
        P = P.at[1:].set(jnp.diff(cdf(x + 0.5 * stepsize, loc=mu, scale=sigma)))
        P = P.at[0].set(cdf(x_min + 0.5 * stepsize, loc=mu, scale=sigma))
        P = P.at[-1].set(1 - cdf(x_max - 0.5 * stepsize, loc=mu, scale=sigma))
        return jnp.full((n_points, n_points), fill_value=P)

    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
    ) -> Float1D:
        return params["mu"] + params["sigma"] * jax.random.normal(key=key)
