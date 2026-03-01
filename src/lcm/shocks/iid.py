from abc import abstractmethod
from dataclasses import dataclass, fields
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats.norm import cdf

from lcm.exceptions import GridInitializationError
from lcm.shocks._base import _gauss_hermite_normal, _ShockGrid
from lcm.typing import Float1D, FloatND


@dataclass(frozen=True, kw_only=True)
class _ShockGridIID(_ShockGrid):
    """Base for iid shocks — draw does not depend on previous value."""

    @abstractmethod
    def draw_shock(
        self,
        params: MappingProxyType[str, float | FloatND],
        key: FloatND,
    ) -> Float1D: ...


@dataclass(frozen=True, kw_only=True)
class Uniform(_ShockGridIID):
    r"""Discretized iid uniform shock: $U(\text{start}, \text{stop})$.

    The continuous distribution is discretized into `n_points` equally spaced
    points between `start` and `stop`.

    """

    start: float | None = None
    """Lower bound of the uniform distribution."""

    stop: float | None = None
    """Upper bound of the uniform distribution."""

    def compute_gridpoints(self, n_points: int, **kwargs: float | Array) -> Float1D:
        return jnp.linspace(start=kwargs["start"], stop=kwargs["stop"], num=n_points)

    def compute_transition_probs(
        self,
        n_points: int,
        **kwargs: float | Array,  # noqa: ARG002
    ) -> FloatND:
        return jnp.full((n_points, n_points), fill_value=1 / n_points)

    def draw_shock(
        self,
        params: MappingProxyType[str, float | FloatND],
        key: FloatND,
    ) -> Float1D:
        return jax.random.uniform(
            key=key, minval=params["start"], maxval=params["stop"]
        )


@dataclass(frozen=True, kw_only=True)
class Normal(_ShockGridIID):
    r"""Discretized iid normal shock: $N(\mu_\varepsilon, \sigma_\varepsilon^2)$.

    When `gauss_hermite=True`, the distribution is discretized using
    Gauss-Hermite quadrature nodes and weights.  When `gauss_hermite=False`,
    it uses `n_points` equally spaced points spanning
    $\mu_\varepsilon \pm n_\text{std} \cdot \sigma_\varepsilon$.

    """

    gauss_hermite: bool
    """Use Gauss-Hermite quadrature nodes and weights."""

    mu: float | None = None
    """Mean of the shock distribution."""

    sigma: float | None = None
    """Standard deviation of the shock distribution."""

    n_std: float | None = None
    """Number of standard deviations from the mean to the grid boundary."""

    def __post_init__(self) -> None:
        if self.n_points % 2 == 0:
            if self.gauss_hermite:
                msg = (
                    f"n_points must be odd (got {self.n_points}). Odd n guarantees"
                    " a quadrature node at the mean (Abramowitz & Stegun, 1972,"
                    " Table 25.10)."
                )
            else:
                msg = (
                    f"n_points must be odd (got {self.n_points}). Odd n guarantees"
                    " a grid point exactly at the mean."
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
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            nodes, _weights = _gauss_hermite_normal(n_points, mu, sigma)
            return nodes
        n_std = kwargs["n_std"]
        x_min = mu - n_std * sigma
        x_max = mu + n_std * sigma
        return jnp.linspace(start=x_min, stop=x_max, num=n_points)

    def compute_transition_probs(
        self, n_points: int, **kwargs: float | Array
    ) -> FloatND:
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            _nodes, weights = _gauss_hermite_normal(n_points, mu, sigma)
            return jnp.full((n_points, n_points), fill_value=weights)
        n_std = kwargs["n_std"]
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
        params: MappingProxyType[str, float | FloatND],
        key: FloatND,
    ) -> Float1D:
        return params["mu"] + params["sigma"] * jax.random.normal(key=key)


@dataclass(frozen=True, kw_only=True)
class LogNormal(_ShockGridIID):
    r"""Discretized iid log-normal shock: $\ln X \sim N(\mu, \sigma^2)$."""

    gauss_hermite: bool
    """Use Gauss-Hermite quadrature nodes and weights."""

    mu: float | None = None
    """Mean of the underlying normal distribution ($E[\\ln X]$)."""

    sigma: float | None = None
    """Standard deviation of the underlying normal distribution."""

    n_std: float | None = None
    """Number of standard deviations in log-space for the grid boundary."""

    def __post_init__(self) -> None:
        if self.n_points % 2 == 0:
            if self.gauss_hermite:
                msg = (
                    f"n_points must be odd (got {self.n_points}). Odd n guarantees"
                    " a quadrature node at the mean (Abramowitz & Stegun, 1972,"
                    " Table 25.10)."
                )
            else:
                msg = (
                    f"n_points must be odd (got {self.n_points}). Odd n guarantees"
                    " a grid point exactly at the mean."
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
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            nodes, _weights = _gauss_hermite_normal(n_points, mu, sigma)
            return jnp.exp(nodes)
        n_std = kwargs["n_std"]
        return jnp.exp(jnp.linspace(mu - n_std * sigma, mu + n_std * sigma, n_points))

    def compute_transition_probs(
        self, n_points: int, **kwargs: float | Array
    ) -> FloatND:
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            _nodes, weights = _gauss_hermite_normal(n_points, mu, sigma)
            return jnp.full((n_points, n_points), fill_value=weights)
        n_std = kwargs["n_std"]
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
        params: MappingProxyType[str, float | FloatND],
        key: FloatND,
    ) -> Float1D:
        return jnp.exp(params["mu"] + params["sigma"] * jax.random.normal(key=key))
