import math
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats.norm import cdf

from _lcm.grids import ContinuousGrid
from _lcm.grids import coordinates as grid_coordinates
from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, FloatND, ScalarFloat, ScalarInt, StateName


@dataclass(frozen=True, kw_only=True)
class StateConditioned:
    r"""A shock parameter conditioned on a discrete state of the same regime.

    Write it in place of the parameter it conditions. Use it when the size of a shock
    depends on where the subject currently is — the variance of the earnings innovation
    differing by employment status, with
    $\sigma_\text{employed} < \sigma_\text{unemployed}$:

        NormalIIDProcess(
            n_points=7,
            gauss_hermite=False,
            mu=0.0,
            n_std=3.0,
            sigma=StateConditioned(
                on="employment_status",
                by={"employed": 0.2, "unemployed": 0.5},
            ),
        )

    Writing $s_t$ for the time-$t$ value of `on` and $\sigma_{s_t}$ for `by[s_t]`, an
    AR(1) process transitions as

    ```{math}
    y_{t+1} \mid y_t, s_t \sim N(\mu + \rho y_t,\ \sigma_{s_t}^2),
    ```

    an IID process dropping the $\rho y_t$ term. The row is binned on one set of nodes,
    placed from the widest value in `by` — only $\sigma_{s_t}$ varies with $s_t$.
    Standing where the scalar would go, it says which parameter is conditioned and
    leaves no way to give that parameter twice.

    The conditioning value is dated $t$, so the variance of the innovation realized
    between $t$ and $t+1$ is set by where the subject is at $t$. The mapping it is read
    against is the *target* regime's declaration, so a process declared with different
    values in the source and target regimes takes the target's.

    `on` must name a `DiscreteGrid` state the regime carries. Build that grid from the
    same `@categorical` class in every regime that carries it: the per-category value is
    selected by the category's integer code, so regimes that encode the same categories
    in a different order would draw each other's values.

    All of it is fixed at build time. Neither `by` nor the process's own parameters
    reach the params template, so none of them can be estimated.

    Defined in this leaf module so the process base class can annotate the field without
    an import cycle; re-exported from `_lcm.processes.state_conditioned`.
    """

    on: StateName
    """Name of the `DiscreteGrid` state whose time-$t$ value selects the parameter."""

    by: Mapping[str, float]
    """Mapping of that state's category names to the value used for each."""


def _gauss_hermite_normal(
    *,
    n_points: int,
    mu: ScalarFloat,
    sigma: ScalarFloat,
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
class _ContinuousStochasticProcess(ContinuousGrid):
    """Base class for discretized continuous stochastic processes.

    Subclasses define distribution-specific parameters as dataclass fields.
    Parameters set to `None` must be supplied at runtime via `params`.

    """

    n_points: int
    """The number of points for the discretization of the process."""

    state_conditioned: StateConditioned | None = field(init=False, default=None)
    """The conditioning declaration, when `sigma` was given as one.

    Not an argument of its own: conditioning is expressed on the parameter it
    applies to, as `sigma=StateConditioned(...)`, which is what makes a scalar and
    a conditioned `sigma` mutually exclusive by construction. `__post_init__` moves
    the declaration here and leaves `sigma` holding the scalar that places the
    nodes.

    The per-category values enter only the transition CDF, evaluated directly at
    the time-$t$ value. Available for the CDF-binned IID normal and Tauchen AR(1)
    processes, whose transition CDFs carry `sigma`. A Rouwenhorst transition depends
    on `rho` alone, so fixing the nodes would leave `sigma` no channel at all.
    """

    _NON_PARAM_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"n_points", "batch_size", "distributed", "state_conditioned"}
    )
    """Dataclass field names that are not distribution parameters.

    Subclasses extend this via `cls._NON_PARAM_FIELDS | {...}` when they
    introduce further non-parameter fields (e.g. `gauss_hermite`).
    """

    def __post_init__(self) -> None:
        self._resolve_conditioned_sigma()

    def _resolve_conditioned_sigma(self) -> None:
        """Split a conditioned `sigma` into its declaration and its node-placing value.

        A conditioned process bins every category on one set of nodes, so the axis has
        to cover the widest of them. That value is not free information, which is why
        it is read off `by` rather than asked for separately.
        """
        declaration = getattr(self, "sigma", None)
        if not isinstance(declaration, StateConditioned):
            return
        values = tuple(declaration.by.values())
        unusable = {
            category: value
            for category, value in declaration.by.items()
            if not isinstance(value, float | int)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
        }
        if not values or unusable:
            msg = (
                f"a conditioned `sigma` needs finite positive values for every "
                f"category, but got {unusable or declaration.by}."
            )
            raise GridInitializationError(msg)
        object.__setattr__(self, "state_conditioned", declaration)
        object.__setattr__(self, "sigma", max(values))

    @property
    def _param_field_names(self) -> tuple[str, ...]:
        """Names of distribution-specific parameters."""
        return tuple(
            f.name for f in fields(self) if f.name not in self._NON_PARAM_FIELDS
        )

    @property
    def params(self) -> MappingProxyType[str, ScalarFloat | ScalarInt]:
        """Distribution parameters as canonical 0-d JAX scalars.

        Boundary cast: dataclass fields supplied by the user as Python
        `bool` / `int` / `float` are returned as `ScalarInt` / `ScalarFloat`
        so every consumer downstream — `compute_gridpoints`,
        `compute_transition_probs`, the regime-building runtime closures —
        sees the canonical dtype.

        """
        out: dict[str, ScalarFloat | ScalarInt] = {}
        for name in self._param_field_names:
            value = getattr(self, name)
            if value is None:
                continue
            # `bool` before `int` — `True` is a Python `int` subclass.
            if isinstance(value, bool | int):
                out[name] = jnp.int32(value)
            else:
                out[name] = jnp.asarray(value)
        return MappingProxyType(out)

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
    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        """Compute discretized gridpoints for the process."""

    @abstractmethod
    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        """Compute transition probability matrix for the process."""

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

        Raises `GridInitializationError` for a state-conditioned process because this
        method has no conditioning-state input.

        Returns NaN of the correct shape when required params are missing (i.e., will
        only be passed at runtime).

        """
        if self.state_conditioned is not None:
            msg = (
                "`get_transition_probs()` cannot return a transition matrix for a "
                "state-conditioned process because its transition law depends on "
                f"'{self.state_conditioned.on}', but the method has no "
                "conditioning-state input."
            )
            raise GridInitializationError(msg)
        if not self.is_fully_specified:
            return jnp.full((self.n_points, self.n_points), jnp.nan)
        return self.compute_transition_probs(**self.params)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return self.get_gridpoints()

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the generalized coordinate of a value in the grid."""
        if not self.is_fully_specified:
            raise GridInitializationError(
                "Cannot compute coordinate for a continuous stochastic process "
                "without all "
                "distribution params."
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
    p1: ScalarFloat,
    mu1: ScalarFloat,
    sigma1: ScalarFloat,
    mu2: ScalarFloat,
    sigma2: ScalarFloat,
) -> FloatND:
    """Evaluate the CDF of a two-component normal mixture.

    $F(x) = p_1 \\, \\Phi\\!\\left(\\frac{x - \\mu_1}{\\sigma_1}\\right)
           + (1 - p_1) \\, \\Phi\\!\\left(\\frac{x - \\mu_2}{\\sigma_2}\\right)$

    """
    return p1 * cdf((x - mu1) / sigma1) + (1 - p1) * cdf((x - mu2) / sigma2)
