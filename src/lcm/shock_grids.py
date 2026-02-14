from abc import abstractmethod
from dataclasses import dataclass, fields
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
        mean: Mean of the shock distribution.
        std: Standard deviation of the shock distribution.
        n_std: Number of standard deviations from the mean to the grid boundary.

    """

    mean: float | None = None
    std: float | None = None
    n_std: float | None = None

    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        mean, std, n_std = (
            kwargs["mean"],
            kwargs["std"],
            kwargs["n_std"],
        )
        x_min = mean - n_std * std
        x_max = mean + n_std * std
        return jnp.linspace(start=x_min, stop=x_max, num=n_points)

    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        mean, std, n_std = (
            kwargs["mean"],
            kwargs["std"],
            kwargs["n_std"],
        )
        x_min = mean - n_std * std
        x_max = mean + n_std * std
        x, stepsize = jnp.linspace(start=x_min, stop=x_max, num=n_points, retstep=True)
        P = jnp.zeros(n_points)
        P = P.at[1:].set(jnp.diff(cdf(x + 0.5 * stepsize, loc=mean, scale=std)))
        P = P.at[0].set(cdf(x_min + 0.5 * stepsize, loc=mean, scale=std))
        P = P.at[-1].set(1 - cdf(x_max - 0.5 * stepsize, loc=mean, scale=std))
        return jnp.full((n_points, n_points), fill_value=P)

    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
    ) -> Float1D:
        return params["mean"] + params["std"] * jax.random.normal(key=key)


@dataclass(frozen=True, kw_only=True)
class ShockGridAR1Tauchen(ShockGridAR1):
    r"""AR(1) shock discretized via Tauchen (1986).

    The process is
    :math:`y_t = \mu_\varepsilon + \rho \, (y_{t-1} - \mu_\varepsilon) + \varepsilon_t`,
    where :math:`\varepsilon_t \sim N(0, \sigma_\varepsilon^2)`.

    Attributes:
        ar1_coeff: Persistence parameter of the AR(1) process.
        std: Standard deviation of the innovation.
        mean: Unconditional (long-run) mean shift.
        n_std: Number of standard deviations for the grid boundary.

    """

    ar1_coeff: float | None = None
    std: float | None = None
    mean: float | None = None
    n_std: float | None = None

    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        ar1_coeff, std = kwargs["ar1_coeff"], kwargs["std"]
        mean, n_std = kwargs["mean"], kwargs["n_std"]
        std_y = jnp.sqrt(std**2 / (1 - ar1_coeff**2))
        x_max = n_std * std_y
        x = jnp.linspace(-x_max, x_max, n_points)
        return x + mean / (1 - ar1_coeff)

    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        ar1_coeff, std = kwargs["ar1_coeff"], kwargs["std"]
        n_std = kwargs["n_std"]
        std_y = jnp.sqrt(std**2 / (1 - ar1_coeff**2))
        x_max = n_std * std_y
        x = jnp.linspace(-x_max, x_max, n_points)
        step = (2 * x_max) / (n_points - 1)
        half_step = 0.5 * step
        P = jnp.empty((n_points, n_points))
        for i in range(n_points):
            P = P.at[i, 0].set(cdf((x[0] - ar1_coeff * x[i] + half_step) / std))
            P = P.at[i, -1].set(
                1 - cdf((x[n_points - 1] - ar1_coeff * x[i] - half_step) / std)
            )
            for j in range(1, n_points - 1):
                z = x[j] - ar1_coeff * x[i]
                P = P.at[i, j].set(
                    cdf((z + half_step) / std) - cdf((z - half_step) / std)
                )
        return P

    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D:
        mu = params["mean"]
        return (
            mu
            + params["ar1_coeff"] * (current_value - mu)
            + params["std"] * jax.random.normal(key=key)
        )


@dataclass(frozen=True, kw_only=True)
class ShockGridAR1Rouwenhorst(ShockGridAR1):
    r"""AR(1) shock discretized via Rouwenhorst (1995).

    The process is
    :math:`y_t = \mu_\varepsilon + \rho \, (y_{t-1} - \mu_\varepsilon) + \varepsilon_t`,
    where :math:`\varepsilon_t \sim N(0, \sigma_\varepsilon^2)`.

    Attributes:
        ar1_coeff: Persistence parameter of the AR(1) process.
        std: Standard deviation of the innovation.
        mean: Unconditional (long-run) mean shift.

    """

    ar1_coeff: float | None = None
    std: float | None = None
    mean: float | None = None

    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        ar1_coeff, std, mean = kwargs["ar1_coeff"], kwargs["std"], kwargs["mean"]
        nu = jnp.sqrt((n_points - 1) / (1 - ar1_coeff**2)) * std
        long_run_mean = mean / (1.0 - ar1_coeff)
        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)

    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        ar1_coeff = kwargs["ar1_coeff"]
        q = (ar1_coeff + 1) / 2
        P = jnp.zeros((n_points, n_points))
        P = P.at[0, 0].set(q)
        P = P.at[0, 1].set(1 - q)
        P = P.at[1, 0].set(1 - q)
        P = P.at[1, 1].set(q)
        for i in range(2, n_points):
            P11 = jnp.zeros((i + 1, i + 1))
            P12 = jnp.zeros((i + 1, i + 1))
            P21 = jnp.zeros((i + 1, i + 1))
            P22 = jnp.zeros((i + 1, i + 1))
            P11 = P11.at[0:i, 0:i].set(P[0:i, 0:i])
            P12 = P12.at[0:i, 1 : i + 1].set(P[0:i, 0:i])
            P21 = P21.at[1 : i + 1, 0:i].set(P[0:i, 0:i])
            P22 = P22.at[1 : i + 1, 1 : i + 1].set(P[0:i, 0:i])
            P = P.at[0 : i + 1, 0 : i + 1].set(
                q * P11 + (1 - q) * P12 + (1 - q) * P21 + q * P22
            )
            P = P.at[1:i, :].set(P[1:i, :] / 2)
        return P

    def draw_shock(
        self,
        params: MappingProxyType[str, float],
        key: FloatND,
        current_value: Float1D,
    ) -> Float1D:
        mu = params["mean"]
        return (
            mu
            + params["ar1_coeff"] * (current_value - mu)
            + params["std"] * jax.random.normal(key=key)
        )
