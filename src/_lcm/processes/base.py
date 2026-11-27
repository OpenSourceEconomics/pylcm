"""The base every stochastic process extends, and the quadrature they share.

`_ContinuousStochasticProcess` fixes what a process owes the engine — a node
grid, and the weights of moving onto it — while the Gauss-Hermite and
normal-mixture helpers build those nodes and weights for the leaf classes.
`lcm.state_conditioned.StateConditioned` marks a process parameter that varies with
another state.
"""

import math
from abc import abstractmethod
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import ClassVar, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats.norm import cdf

from _lcm.grids import ContinuousGrid
from _lcm.grids import coordinates as grid_coordinates
from _lcm.grids.continuous import _validate_off_grid
from _lcm.typing import PRNGKeyND
from lcm.exceptions import GridInitializationError
from lcm.state_conditioned import StateConditioned
from lcm.typing import Float1D, FloatND, OffGrid, ScalarFloat


def standard_normal(key: PRNGKeyND) -> ScalarFloat:
    """Draw one standard normal that a constant scale cannot be folded into.

    `jax.random.normal` scales its internal draw by `sqrt(2)`. Unsealed, XLA may
    fold a constant `sigma` multiplying the result into that scaling, depending
    on how the surrounding program is partitioned and fused.
    """
    return sealed(jax.random.normal(key=key))


def sealed(value: FloatND) -> FloatND:
    """Round `value` on its own, before any consumer sees it.

    Shock draws seal each product so that it is neither folded with constants
    nor contracted into a fused multiply-add with the sum it feeds. A draw then
    rounds exactly as written in every compiled program, so simulating the same
    agents partitioned over devices or not gives bitwise-equal states.
    """
    return jax.lax.optimization_barrier(value)


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


SIMULATE_DRAWS_RENAMED = (
    "The `simulate_draws=` argument was renamed. Pass `simulated_draws=` instead, "
    'with the same values "continuous" or "nodes".'
)


@dataclass(frozen=True, kw_only=True)
class _ContinuousStochasticProcess(ContinuousGrid):
    """Base class for discretized continuous stochastic processes.

    Subclasses define distribution-specific parameters as dataclass fields.
    Parameters set to `None` must be supplied at runtime via `params`.

    """

    def __new__(
        cls,
        *args: object,  # noqa: ARG004, PAN001 - refuses removed keywords first
        **kwargs: object,  # noqa: PAN001 - refuses removed keywords before any check
    ) -> Self:
        if "simulate_draws" in kwargs:
            raise TypeError(SIMULATE_DRAWS_RENAMED)
        return super().__new__(cls)

    n_points: int
    """The number of points for the discretization of the process."""

    simulated_draws: Literal["continuous", "nodes"] = "continuous"
    """Draw from the continuous law, or sample the solve nodes with their weights."""

    off_grid: OffGrid = field(
        default="extrapolate", metadata={"fingerprint_omit_if_default": True}
    )
    """How a value beyond the outer nodes is read, as on a continuous grid.

    - `"extrapolate"`: Tauchen evaluates its transition law at the off-grid
      value. Rouwenhorst uses the edge node's transition row and extends the
      continuation along the edge segment. Interpolated policies also extend
      along the edge segment.
    - `"clip_lookup"`: those reads use the nearest edge node; the state keeps
      its value.
    - `"clip_state"`: a simulated value is additionally clipped to the outer
      nodes.
    - `"raise"`: a simulated value beyond the outer nodes is refused.

    A `simulated_draws="nodes"` draw beyond the nodes samples the nearest edge
    node's transition row when the off-grid rule permits the state.
    """

    state_conditioned: StateConditioned | None = field(init=False, default=None)
    """The first conditioning declaration, naming the state the process reads.

    Not an argument of its own: conditioning is expressed on the parameter it
    applies to, as `sigma=StateConditioned(...)`, which is what makes a scalar and
    a conditioned value of one parameter mutually exclusive by construction. Every
    conditioned parameter of a process names the same state and categories, so this
    declaration's `on` and categories hold for all of them.
    """

    conditioned_params: tuple[tuple[str, StateConditioned], ...] = field(
        init=False, default=(), metadata={"fingerprint_omit_if_default": True}
    )
    """Each conditioned parameter's name and declaration, in field order.

    Each such parameter itself holds its declaration, so the process carries no
    scalar for it; `law_of` gives one category's law. The nodes are common to all
    categories:
    they span every category's own nodes. The per-category values enter the
    transition row, evaluated directly at the time-$t$ value, and the simulated
    draw. Available for the processes whose nodes are spaced evenly or supplied
    (the CDF-binned IID normal, Tauchen AR(1) and Tauchen normal-mixture AR(1));
    Gauss-Hermite nodes and a Rouwenhorst transition belong to one law.
    """

    _NON_PARAM_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "n_points",
            "state_conditioned",
            "conditioned_params",
            "simulated_draws",
            "off_grid",
        }
    )
    """Dataclass field names that are not distribution parameters.

    Subclasses extend this via `cls._NON_PARAM_FIELDS | {...}` when they
    introduce further non-parameter fields (e.g. `gauss_hermite`).
    """

    @property
    def constructor_name(self) -> str:
        """Name of the public call that builds this process, for error messages."""
        return type(self).__name__

    def __post_init__(self) -> None:
        if self.simulated_draws not in {"continuous", "nodes"}:
            msg = "simulated_draws must be 'continuous' or 'nodes'."
            raise GridInitializationError(msg)
        _validate_off_grid(self.off_grid)
        self._store_law_values_as_floats()
        self._resolve_conditioned_params()

    def has_same_law_as(self, other: _ContinuousStochasticProcess) -> bool:
        """Return whether `other` declares this law, whatever its draw mode.

        `simulated_draws` only chooses how a simulation realizes the law, so two
        declarations differing in it alone describe one process.
        """
        return replace(self, simulated_draws="continuous") == replace(
            other, simulated_draws="continuous"
        )

    def _store_law_values_as_floats(self) -> None:
        """Store every fixed law value as a Python `float`, however it was written.

        A law written with `1` describes the law written with `1.0`, so both store
        one value and share one identity. A Boolean is refused: it names no number.
        """
        law_fields = (
            spec.name
            for spec in fields(self)
            if spec.name not in self._NON_PARAM_FIELDS | {"gauss_hermite"}
        )
        for name in law_fields:
            value = getattr(self, name)
            if value is None or isinstance(value, StateConditioned):
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = (
                    f"{self.constructor_name}: `{name}` must be a real number, "
                    f"but got {value!r}."
                )
                raise GridInitializationError(msg)
            object.__setattr__(self, name, float(value))

    def _resolve_conditioned_params(self) -> None:
        """List every conditioned parameter's declaration in `conditioned_params`.

        Every category's law is then built on its own, which applies the checks a
        scalar law gets.
        """
        declared = tuple(
            (name, value)
            for name in self._param_field_names
            if isinstance(value := getattr(self, name), StateConditioned)
        )
        if not declared:
            return
        first = declared[0][1]
        for name, declaration in declared:
            positive = name.startswith("sigma")
            unusable = {
                category: value
                for category, value in declaration.by.items()
                if not math.isfinite(value) or (positive and value <= 0.0)
            }
            if not declaration.by or unusable:
                msg = (
                    f"a conditioned `{name}` needs finite "
                    f"{'positive ' if positive else ''}values for every category, "
                    f"but got {unusable or declaration.by}."
                )
                raise GridInitializationError(msg)
            if declaration.on != first.on:
                msg = (
                    "the conditioned parameters of one process must name the same "
                    f"state, but `{declared[0][0]}` is conditioned on '{first.on}' "
                    f"and `{name}` on '{declaration.on}'."
                )
                raise GridInitializationError(msg)
            if set(declaration.by) != set(first.by):
                msg = (
                    "the conditioned parameters of one process must name the same "
                    f"categories, but `{declared[0][0]}` names {sorted(first.by)} "
                    f"and `{name}` names {sorted(declaration.by)}."
                )
                raise GridInitializationError(msg)
        object.__setattr__(self, "state_conditioned", first)
        object.__setattr__(self, "conditioned_params", declared)
        for category in first.by:
            self.law_of(category)

    def law_of(self, category: str) -> Self:
        """Return the unconditioned process holding `category`'s parameter values."""
        return replace(
            self,
            **{
                name: declaration.by[category]
                for name, declaration in self.conditioned_params
            },
        )

    @property
    def _param_field_names(self) -> tuple[str, ...]:
        """Names of distribution-specific parameters."""
        return tuple(
            f.name for f in fields(self) if f.name not in self._NON_PARAM_FIELDS
        )

    @property
    def params(self) -> MappingProxyType[str, ScalarFloat]:
        """Fixed distribution parameters as 0-d JAX floats of the active precision.

        Every fixed law value is a Python `float`, so each consumer downstream —
        `compute_gridpoints`, `compute_transition_probs`, the regime-building
        runtime closures — receives a floating scalar. A conditioned parameter has
        no single value and is left out; `law_of` gives one category's parameters.

        """
        return MappingProxyType(
            {
                name: jnp.asarray(value)
                for name in self._param_field_names
                if isinstance(value := getattr(self, name), float)
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
    def compute_gridpoints(self, **kwargs: ScalarFloat) -> Float1D:
        """Compute discretized gridpoints for the process."""

    @abstractmethod
    def compute_transition_probs(self, **kwargs: ScalarFloat) -> FloatND:
        """Compute transition probability matrix for the process."""

    def get_gridpoints(self) -> Float1D:
        """Get the gridpoints used for discretization.

        Returns NaN of the correct shape when required params are missing (i.e., will
        only be passed at runtime).

        """
        if not self.is_fully_specified:
            return jnp.full(self.n_points, jnp.nan)
        if self.state_conditioned is None:
            return self.compute_gridpoints(**self.params)
        # ponytail: every call rebuilds each category's law; cache the nodes if
        # model building shows it.
        grids = jnp.stack(
            [
                self.law_of(category).get_gridpoints()
                for category in self.state_conditioned.by
            ]
        )
        low, high = grids[:, 0].min(), grids[:, -1].max()
        nodes = jnp.linspace(low, high, self.n_points)
        # A category whose own nodes already span the range supplies them exactly.
        for grid in grids:
            nodes = jnp.where((grid[0] == low) & (grid[-1] == high), grid, nodes)
        return nodes

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
    """Validate Gauss-Hermite node counts and mutually exclusive grid options."""
    if gauss_hermite and n_points < 1:
        msg = f"n_points must be positive (got {n_points})."
        raise GridInitializationError(msg)
    if gauss_hermite and n_std is not None:
        msg = "gauss_hermite=True and n_std are mutually exclusive."
        raise GridInitializationError(msg)


def _mixture_cdf(
    *,
    x: FloatND,
    p1: float | ScalarFloat,
    mu1: float | ScalarFloat,
    sigma1: float | ScalarFloat,
    mu2: float | ScalarFloat,
    sigma2: float | ScalarFloat,
) -> FloatND:
    """Evaluate the CDF of a two-component normal mixture.

    $F(x) = p_1 \\, \\Phi\\!\\left(\\frac{x - \\mu_1}{\\sigma_1}\\right)
           + (1 - p_1) \\, \\Phi\\!\\left(\\frac{x - \\mu_2}{\\sigma_2}\\right)$

    """
    return p1 * cdf((x - mu1) / sigma1) + (1 - p1) * cdf((x - mu2) / sigma2)
