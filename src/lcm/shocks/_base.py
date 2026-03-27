from abc import abstractmethod
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import overload

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats.norm import cdf

from lcm.exceptions import GridInitializationError
from lcm.grids import ContinuousGrid
from lcm.grids import coordinates as grid_coordinates
from lcm.typing import Float1D, FloatND, ScalarFloat


def _gauss_hermite_normal(
    *, n_points: int, mu: float | Float1D, sigma: float | Float1D
) -> tuple[Float1D, Float1D]:
    """Compute Gauss-Hermite quadrature nodes and weights for $N(\\mu, \\sigma^2)$.

    The raw Hermite nodes/weights are computed via numpy (requires only the
    concrete `n_points`).  The affine transform to $N(\\mu, \\sigma^2)$ uses
    JAX so that `mu` and `sigma` may be JAX tracers inside JIT.

    """
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(n_points)
    nodes = mu + sigma * np.sqrt(2) * jnp.asarray(raw_nodes)
    weights = jnp.asarray(raw_weights / np.sqrt(np.pi))
    return nodes, weights


@dataclass(frozen=True, kw_only=True)
class _ShockGrid(ContinuousGrid):
    """Base class for discretized continuous shock grids.

    Subclasses define distribution-specific parameters as dataclass fields.
    Parameters set to `None` must be supplied at runtime via `params`.

    """

    n_points: int
    """The number of points for the discretization of the shock."""

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
    def compute_gridpoints(self, **kwargs: float) -> Float1D:
        """Compute discretized gridpoints for the shock distribution."""

    @abstractmethod
    def compute_transition_probs(self, **kwargs: float) -> FloatND:
        """Compute transition probability matrix for the shock distribution."""

    def get_gridpoints(self) -> Float1D:
        """Get the gridpoints used for discretization.

        Returns NaN of the correct shape when required params are missing (i.e., will
        only be passed at runtime).

        """
        if not self.is_fully_specified:
            return jnp.full(self.n_points, jnp.nan)
        return self.compute_gridpoints(**self.params)

    def get_transition_probs(self) -> FloatND:
        """Get the transition probabilities at the gridpoints.

        Returns NaN of the correct shape when required params are missing (i.e., will
        only be passed at runtime).

        """
        if not self.is_fully_specified:
            return jnp.full((self.n_points, self.n_points), jnp.nan)
        return self.compute_transition_probs(**self.params)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return self.get_gridpoints()

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
        """Return the generalized coordinate of a value in the grid."""
        if not self.is_fully_specified:
            raise GridInitializationError(
                "Cannot compute coordinate for a ShockGrid without all shock params."
            )
        return grid_coordinates.get_irreg_coordinate(value=value, points=self.to_jax())


def _validate_gauss_hermite_grid(
    *,
    n_points: int,
    gauss_hermite: bool,
    n_std: float | None,
) -> None:
    """Validate `n_points` / `gauss_hermite` / `n_std` consistency."""
    if gauss_hermite and n_points % 2 == 0:
        msg = (
            f"n_points must be odd (got {n_points}). Odd n guarantees"
            " a quadrature node at the mean (Abramowitz & Stegun, 1972,"
            " Table 25.10)."
        )
        raise GridInitializationError(msg)
    if gauss_hermite and n_std is not None:
        msg = "gauss_hermite=True and n_std are mutually exclusive."
        raise GridInitializationError(msg)


def _mixture_cdf(
    *,
    x: FloatND,
    p1: float,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> FloatND:
    """Evaluate the CDF of a two-component normal mixture.

    $F(x) = p_1 \\, \\Phi\\!\\left(\\frac{x - \\mu_1}{\\sigma_1}\\right)
           + (1 - p_1) \\, \\Phi\\!\\left(\\frac{x - \\mu_2}{\\sigma_2}\\right)$

    """
    return p1 * cdf((x - mu1) / sigma1) + (1 - p1) * cdf((x - mu2) / sigma2)
