from abc import abstractmethod
from dataclasses import dataclass, fields
from math import comb
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf

from lcm import grid_helpers
from lcm.exceptions import GridInitializationError
from lcm.grids import ContinuousGrid
from lcm.typing import Float1D, FloatND, ScalarFloat

# ======================================================================================
# ShockGrid base class and subclasses
# ======================================================================================


@dataclass(frozen=True, kw_only=True)
class ShockGrid(ContinuousGrid):
    """Base class for discretized continuous shock grids.

    Subclasses define distribution-specific parameters as dataclass fields.
    Parameters set to ``None`` must be supplied at runtime via ``params``.

    The class *is* the distribution type — no ``distribution_type`` string needed.

    Attributes:
        n_points: The number of points for the discretization of the shock.

    """

    n_points: int

    @property
    def _param_field_names(self) -> tuple[str, ...]:
        """Names of distribution-specific parameters."""
        return tuple(f.name for f in fields(self) if f.name != "n_points")

    @property
    def params(self) -> MappingProxyType[str, float]:
        """Mapping of the distribution's parameters' names to their specified values."""
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self._param_field_names
                if getattr(self, name) is not None
            }
        )

    @property
    def params_to_pass_at_runtime(self) -> tuple[str, ...]:
        """Return names of distribution params that are not yet specified."""
        return tuple(
            name for name in self._param_field_names if getattr(self, name) is None
        )

    @property
    def is_fully_specified(self) -> bool:
        """Whether all required distribution params are present."""
        return not self.params_to_pass_at_runtime

    @abstractmethod
    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        """Compute discretized gridpoints for the shock distribution."""

    @abstractmethod
    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        """Compute transition probability matrix for the shock distribution."""

    def get_gridpoints(self) -> Float1D:
        """Get the gridpoints used for discretization.

        Returns NaN of the correct shape when required params are missing.

        """
        if not self.is_fully_specified:
            return jnp.full(self.n_points, jnp.nan)
        return self.compute_gridpoints(self.n_points, **self.params)

    def get_transition_probs(self) -> FloatND:
        """Get the transition probabilities at the gridpoints.

        Returns uniform probabilities when required params are missing.

        """
        if not self.is_fully_specified:
            return jnp.full(
                (self.n_points, self.n_points), fill_value=1 / self.n_points
            )
        return self.compute_transition_probs(self.n_points, **self.params)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return self.get_gridpoints()

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""
        if not self.is_fully_specified:
            raise GridInitializationError(
                "Cannot compute coordinate for a ShockGrid without all shock params."
            )
        return grid_helpers.get_irreg_coordinate(value, self.to_jax())


@dataclass(frozen=True, kw_only=True)
class ShockGridIID(ShockGrid):
    """Base for iid shocks — draw does not depend on previous value."""

    @abstractmethod
    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
    ) -> Float1D: ...


@dataclass(frozen=True, kw_only=True)
class ShockGridAR1(ShockGrid):
    """Base for AR(1) shocks — draw depends on previous value."""

    @abstractmethod
    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D: ...


@dataclass(frozen=True, kw_only=True)
class ShockGridIIDUniform(ShockGridIID):
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
class ShockGridIIDNormal(ShockGridIID):
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


@dataclass(frozen=True, kw_only=True)
class ShockGridAR1Tauchen(ShockGridAR1):
    r"""AR(1) shock discretized via Tauchen (1986).

    The process is
    :math:`y_t = \mu + \rho \, y_{t-1} + \varepsilon_t`,
    where :math:`\varepsilon_t \sim N(0, \sigma_\varepsilon^2)`.

    Attributes:
        rho: Persistence parameter of the AR(1) process.
        sigma: Standard deviation of the innovation.
        mu: Intercept (drift) of the AR(1) process.
        n_std: Number of standard deviations for the grid boundary.

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
class ShockGridAR1Rouwenhorst(ShockGridAR1):
    r"""AR(1) shock discretized via Rouwenhorst (1995).

    The process is
    :math:`y_t = \mu + \rho \, y_{t-1} + \varepsilon_t`,
    where :math:`\varepsilon_t \sim N(0, \sigma_\varepsilon^2)`.

    Attributes:
        rho: Persistence parameter of the AR(1) process.
        sigma: Standard deviation of the innovation.
        mu: Intercept (drift) of the AR(1) process.

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
        P = jnp.where(valid, terms, 0.0).sum(axis=-1)
        return P / P.sum(axis=1, keepdims=True)

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
