from abc import abstractmethod
from dataclasses import dataclass
from math import comb
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf

from lcm.shocks._base import _ShockGrid
from lcm.typing import Float1D, FloatND


@dataclass(frozen=True, kw_only=True)
class _ShockGridAR1(_ShockGrid):
    """Base for AR(1) shocks — draw depends on previous value."""

    @abstractmethod
    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D: ...


@dataclass(frozen=True, kw_only=True)
class Tauchen(_ShockGridAR1):
    r"""AR(1) shock discretized via Tauchen (1986).

    The process is
    :math:`y_t = \mu + \rho \, y_{t-1} + \varepsilon_t`,
    where :math:`\varepsilon_t \sim N(0, \sigma_\varepsilon^2)`.

    Attributes:
        rho: Persistence parameter of the AR(1) process.
        sigma: Standard deviation of the innovation.
        mu: Intercept (drift) of the AR(1) process.
        n_std: Number of standard deviations for the grid boundary.

    Original implementation follows `QuantEcon
    <https://quanteconpy.readthedocs.io/en/latest/markov/approximation.html#quantecon.markov.approximation.tauchen>`_.

    """

    rho: float | None = None
    sigma: float | None = None
    mu: float | None = None
    n_std: float | None = None

    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        rho, sigma = kwargs["rho"], kwargs["sigma"]
        mu, n_std = kwargs["mu"], kwargs["n_std"]
        std_y = jnp.sqrt(sigma**2 / (1 - rho**2))
        x_max = n_std * std_y
        x = jnp.linspace(-x_max, x_max, n_points)
        return x + mu / (1 - rho)

    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        rho, sigma = kwargs["rho"], kwargs["sigma"]
        n_std = kwargs["n_std"]
        std_y = jnp.sqrt(sigma**2 / (1 - rho**2))
        x_max = n_std * std_y
        x = jnp.linspace(-x_max, x_max, n_points)
        step = (2 * x_max) / (n_points - 1)
        half_step = 0.5 * step
        P = jnp.empty((n_points, n_points))
        for i in range(n_points):
            P = P.at[i, 0].set(cdf((x[0] - rho * x[i] + half_step) / sigma))
            P = P.at[i, -1].set(
                1 - cdf((x[n_points - 1] - rho * x[i] - half_step) / sigma)
            )
            for j in range(1, n_points - 1):
                z = x[j] - rho * x[i]
                P = P.at[i, j].set(
                    cdf((z + half_step) / sigma) - cdf((z - half_step) / sigma)
                )
        return P

    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D:
        return (
            params["mu"]
            + params["rho"] * current_value
            + params["sigma"] * jax.random.normal(key=key)
        )


@dataclass(frozen=True, kw_only=True)
class Rouwenhorst(_ShockGridAR1):
    r"""AR(1) shock discretized via Rouwenhorst (1995).

    The process is
    :math:`y_t = \mu + \rho \, y_{t-1} + \varepsilon_t`,
    where :math:`\varepsilon_t \sim N(0, \sigma_\varepsilon^2)`.

    Attributes:
        rho: Persistence parameter of the AR(1) process.
        sigma: Standard deviation of the innovation.
        mu: Intercept (drift) of the AR(1) process.

    Original implementation follows `QuantEcon
    <https://quanteconpy.readthedocs.io/en/latest/markov/approximation.html#quantecon.markov.approximation.rouwenhorst>`_.

    """

    rho: float | None = None
    sigma: float | None = None
    mu: float | None = None

    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        rho, sigma, mu = kwargs["rho"], kwargs["sigma"], kwargs["mu"]
        nu = jnp.sqrt((n_points - 1) / (1 - rho**2)) * sigma
        long_run_mean = mu / (1.0 - rho)
        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)

    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        rho = kwargs["rho"]
        q = (rho + 1) / 2

        # Binomial coefficient lookup table
        C = jnp.array([[comb(n, k) for k in range(n_points)] for n in range(n_points)])

        i = jnp.arange(n_points)[:, None, None]
        j = jnp.arange(n_points)[None, :, None]
        k = jnp.arange(n_points)[None, None, :]

        # P[i,j] = sum_k C(i,k) C(n-1-i,j-k) q^(n-1-i-j+2k) (1-q)^(i+j-2k)
        valid = (k >= jnp.maximum(0, i + j - n_points + 1)) & (k <= jnp.minimum(i, j))
        k_s = jnp.where(valid, k, 0)
        jmk = jnp.where(valid, j - k, 0)

        terms = (
            C[i, k_s]
            * C[n_points - 1 - i, jmk]
            * q ** (n_points - 1 - i - j + 2 * k_s)
            * (1 - q) ** (i + j - 2 * k_s)
        )
        return jnp.where(valid, terms, 0.0).sum(axis=-1)

    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D:
        return (
            params["mu"]
            + params["rho"] * current_value
            + params["sigma"] * jax.random.normal(key=key)
        )
