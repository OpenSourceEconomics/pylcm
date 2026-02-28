from abc import abstractmethod
from dataclasses import dataclass, fields
from math import comb
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats.norm import cdf, pdf

from lcm.exceptions import GridInitializationError
from lcm.shocks._base import _gauss_hermite_normal, _ShockGrid
from lcm.typing import Float1D, FloatND


@dataclass(frozen=True, kw_only=True)
class _ShockGridAR1(_ShockGrid):
    """Base for AR(1) shocks — draw depends on previous value."""

    @abstractmethod
    def draw_shock(
        self,
        params: MappingProxyType[str, float | FloatND],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D: ...


@dataclass(frozen=True, kw_only=True)
class Tauchen(_ShockGridAR1):
    r"""AR(1) shock discretized via Tauchen (1986).

    The process is
    $y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$,
    where $\varepsilon_t \sim N(0, \sigma_\varepsilon^2)$.

    When `gauss_hermite=True`, the grid uses Gauss-Hermite quadrature nodes
    with importance-sampling weights following
    [Tauchen & Hussey (1991)](https://doi.org/10.2307/2938229).
    When `gauss_hermite=False`, it uses equally spaced points spanning
    $\pm n_\text{std}$ unconditional standard deviations, following
    [QuantEcon](https://quanteconpy.readthedocs.io/en/latest/markov/approximation.html#quantecon.markov.approximation.tauchen).

    """

    gauss_hermite: bool
    """Use Gauss-Hermite quadrature nodes and weights."""

    rho: float | None = None
    """Persistence parameter of the AR(1) process."""

    sigma: float | None = None
    """Standard deviation of the innovation."""

    mu: float | None = None
    """Intercept (drift) of the AR(1) process."""

    n_std: float | None = None
    """Number of standard deviations for the grid boundary."""

    def __post_init__(self) -> None:
        if self.n_points % 2 == 0:
            msg = (
                f"n_points must be odd (got {self.n_points}). Odd n guarantees a"
                " quadrature node at the mean (Abramowitz & Stegun, 1972,"
                " Table 25.10)."
            )
            raise GridInitializationError(msg)
        if self.gauss_hermite and self.n_std is not None:
            msg = "gauss_hermite=True and n_std are mutually exclusive."
            raise GridInitializationError(msg)

    @property
    def _param_field_names(self) -> tuple[str, ...]:
        exclude = {"n_points", "gauss_hermite"}
        if self.gauss_hermite:
            exclude.add("n_std")
        return tuple(f.name for f in fields(self) if f.name not in exclude)

    def compute_gridpoints(self, n_points: int, **kwargs: float | Array) -> Float1D:
        rho, sigma, mu = kwargs["rho"], kwargs["sigma"], kwargs["mu"]
        if self.gauss_hermite:
            return _gauss_hermite_normal(n_points, mu / (1 - rho), sigma)[0]
        n_std = kwargs["n_std"]
        std_y = jnp.sqrt(sigma**2 / (1 - rho**2))
        x_max = n_std * std_y
        x = jnp.linspace(-x_max, x_max, n_points)
        return x + mu / (1 - rho)

    def compute_transition_probs(
        self, n_points: int, **kwargs: float | Array
    ) -> FloatND:
        rho, sigma = kwargs["rho"], kwargs["sigma"]
        if self.gauss_hermite:
            nodes, weights = _gauss_hermite_normal(n_points, 0.0, sigma)
            f_cond = pdf(nodes[None, :], loc=rho * nodes[:, None], scale=sigma)
            g_prop = pdf(nodes, loc=0.0, scale=sigma)
            raw = weights * f_cond / g_prop
            return raw / raw.sum(axis=1, keepdims=True)
        n_std = kwargs["n_std"]
        std_y = jnp.sqrt(sigma**2 / (1 - rho**2))
        x_max = n_std * std_y
        x = jnp.linspace(-x_max, x_max, n_points)
        step = (2 * x_max) / (n_points - 1)
        half_step = 0.5 * step
        # z[i, j] = x[j] - rho * x[i]: (n_points, n_points)
        z = x[None, :] - rho * x[:, None]
        upper = cdf((z + half_step) / sigma)
        lower = cdf((z - half_step) / sigma)
        P = upper - lower
        P = P.at[:, 0].set(upper[:, 0])
        return P.at[:, -1].set(1 - lower[:, -1])

    def draw_shock(
        self,
        params: MappingProxyType[str, float | FloatND],
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
    $y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$,
    where $\varepsilon_t \sim N(0, \sigma_\varepsilon^2)$.

    Implementation based on [Kopecky & Suen (2010)](https://doi.org/10.1016/j.red.2010.02.002).

    """

    rho: float | None = None
    """Persistence parameter of the AR(1) process."""

    sigma: float | None = None
    """Standard deviation of the innovation."""

    mu: float | None = None
    """Intercept (drift) of the AR(1) process."""

    def __post_init__(self) -> None:
        if self.n_points % 2 == 0:
            msg = (
                f"n_points must be odd (got {self.n_points}). Odd n guarantees a"
                " quadrature node at the mean (Abramowitz & Stegun, 1972,"
                " Table 25.10)."
            )
            raise GridInitializationError(msg)

    def compute_gridpoints(self, n_points: int, **kwargs: float | Array) -> Float1D:
        rho, sigma, mu = kwargs["rho"], kwargs["sigma"], kwargs["mu"]
        nu = jnp.sqrt((n_points - 1) / (1 - rho**2)) * sigma
        long_run_mean = mu / (1.0 - rho)
        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)

    def compute_transition_probs(
        self, n_points: int, **kwargs: float | Array
    ) -> FloatND:
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
        params: MappingProxyType[str, float | FloatND],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D:
        return (
            params["mu"]
            + params["rho"] * current_value
            + params["sigma"] * jax.random.normal(key=key)
        )
