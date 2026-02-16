from abc import abstractmethod
from dataclasses import dataclass, fields
from types import MappingProxyType

import jax.numpy as jnp

from lcm import grid_helpers
from lcm.exceptions import GridInitializationError
from lcm.grids import ContinuousGrid
from lcm.typing import Float1D, FloatND, ScalarFloat


@dataclass(frozen=True, kw_only=True)
class _ShockGrid(ContinuousGrid):
    """Base class for discretized continuous shock grids.

    Subclasses define distribution-specific parameters as dataclass fields.
    Parameters set to ``None`` must be supplied at runtime via ``params``.

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
    def compute_gridpoints(self, n_points: int, **kwargs: float) -> Float1D:
        """Compute discretized gridpoints for the shock distribution."""

    @abstractmethod
    def compute_transition_probs(self, n_points: int, **kwargs: float) -> FloatND:
        """Compute transition probability matrix for the shock distribution."""

    def get_gridpoints(self) -> Float1D:
        """Get the gridpoints used for discretization.

        Returns NaN of the correct shape when required params are missing (i.e., will
        only be passed at runtime).

        """
        if not self.is_fully_specified:
            return jnp.full(self.n_points, jnp.nan)
        return self.compute_gridpoints(self.n_points, **self.params)

    def get_transition_probs(self) -> FloatND:
        """Get the transition probabilities at the gridpoints.

        Returns NaN of the correct shape when required params are missing (i.e., will
        only be passed at runtime).

        """
        if not self.is_fully_specified:
            return jnp.full((self.n_points, self.n_points), jnp.nan)
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
