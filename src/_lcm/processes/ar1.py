"""The discretized AR(1) processes a regime can declare as a state.

`TauchenAR1Process`, `RouwenhorstAR1Process` and
`TauchenNormalMixtureAR1Process` each pair a node grid with the transition
matrix its discretization implies.
"""

from abc import abstractmethod
from dataclasses import dataclass, fields
from functools import partial
from math import comb
from types import MappingProxyType
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax.scipy.stats.norm import cdf

from _lcm.beartype_conf import GRID_CONF
from _lcm.dtypes import canonical_float_dtype
from _lcm.grids import coordinates as grid_coordinates
from _lcm.grids.base import UserVector, fail_if_not_a_numeric_vector
from _lcm.processes.base import (
    StateConditioned,
    _ContinuousStochasticProcess,
    _gauss_hermite_normal,
    _mixture_cdf,
    _validate_gauss_hermite_grid,
    sealed,
    standard_normal,
)
from _lcm.processes.state_conditioned import tauchen_row
from _lcm.typing import PRNGKeyND
from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, FloatND, IntND, ScalarFloat, ScalarInt


@dataclass(frozen=True, kw_only=True)
class _AR1Process(_ContinuousStochasticProcess):
    """Base for AR(1) processes — draw depends on previous value."""

    rho: float | int | None = None
    """Persistence parameter of the stationary AR(1) process."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rho is not None:
            self.fail_if_nonstationary_rho(rho=self.rho)

    def fail_if_nonstationary_rho(self, *, rho: float | FloatND | IntND) -> None:
        """Reject persistence that cannot place a stationary grid at entry."""
        values = np.asarray(rho)
        if not np.all((values > -1) & (values < 1)):
            msg = (
                f"{type(self).__name__} with rho={rho} has no stationary distribution "
                "to place a grid on: unit-root or explosive persistence requires "
                "a process on user-supplied nodes."
            )
            raise GridInitializationError(msg)

    @abstractmethod
    def draw_shock(
        self,
        *,
        params: MappingProxyType[str, ScalarFloat | ScalarInt],
        key: PRNGKeyND,
        current_value: ScalarFloat,
    ) -> ScalarFloat: ...


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class TauchenAR1Process(_AR1Process):
    r"""AR(1) process discretized via Tauchen (1986).

    The process is
    $y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$,
    where $\varepsilon_t \sim N(0, \sigma_\varepsilon^2)$.

    When `gauss_hermite=True`, the grid uses Gauss-Hermite quadrature nodes
    with CDF-based transition probabilities computed at midpoints between nodes.
    Even and odd `n_points` are supported; only odd counts include the unconditional
    mean.

    When `gauss_hermite=False`, it uses equally spaced points spanning
    $\pm n_\text{std}$ unconditional standard deviations, following
    [QuantEcon](https://quanteconpy.readthedocs.io/en/latest/markov/approximation.html#quantecon.markov.approximation.tauchen).

    """

    gauss_hermite: bool
    """Use Gauss-Hermite quadrature nodes and weights."""

    sigma: float | int | StateConditioned | None = None
    """Standard deviation of the innovation.

    A `StateConditioned` here conditions the innovation scale on a discrete state
    of the same regime; `__post_init__` then leaves this holding the scalar that
    places the nodes.
    """

    mu: float | int | None = None
    """Intercept (drift) of the AR(1) process."""

    n_std: float | int | None = None
    """Number of standard deviations for the grid boundary."""

    @staticmethod
    def on_nodes(
        *,
        nodes: UserVector,
        simulate_draws: Literal["continuous", "nodes"] = "continuous",
        rho: float | None = None,
        sigma: float | StateConditioned | None = None,
        mu: float | None = 0.0,
        transition_rule: Literal[
            "midpoint", "innovation_quadrature", "density"
        ] = "midpoint",
        n_quadrature: int | None = None,
        innovation_nodes: UserVector | None = None,
        innovation_weights: UserVector | None = None,
    ) -> TauchenAR1Process:
        """Bin an AR(1) law on finite increasing physical nodes.

        The default rule integrates midpoint bins with open first and last tails.
        ``density`` normalizes normal densities at the nodes; it ignores cell widths.
        ``innovation_quadrature`` maps standardized innovations linearly to adjacent
        state nodes, clipping targets to the endpoints. Supply ``n_quadrature``
        (default 9 Gauss-Hermite points), or both innovation arrays. Weights must be
        finite, nonnegative, and have positive total mass; they are normalized.
        Continuous simulation still draws the normal law. Node simulation samples
        the selected transition matrix. Ordinary Tauchen construction is unchanged.
        The nodes do not depend on stationarity, so unit roots are supported.
        Unspecified law parameters are supplied at solve or simulation time.
        `nodes` is a one-dimensional NumPy or JAX array of integer or floating
        dtype; lists and tuples are refused.
        """
        fail_if_not_a_numeric_vector(
            value=nodes,
            argument="nodes",
            example="TauchenAR1Process.on_nodes(nodes=jnp.array([-1.0, 0.0, 1.0]))",
        )
        quadrature_nodes, quadrature_weights = _prepare_innovation_quadrature(
            transition_rule=transition_rule,
            n_quadrature=n_quadrature,
            innovation_nodes=innovation_nodes,
            innovation_weights=innovation_weights,
        )
        return _TauchenOnNodesProcess(
            transition_rule=transition_rule,
            innovation_nodes=quadrature_nodes,
            innovation_weights=quadrature_weights,
            nodes=tuple(float(node) for node in np.asarray(nodes).tolist()),
            simulate_draws=simulate_draws,
            n_points=len(nodes),
            gauss_hermite=False,
            rho=rho,
            sigma=sigma,
            mu=mu,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_gauss_hermite_grid(
            n_points=self.n_points,
            gauss_hermite=self.gauss_hermite,
            n_std=self.n_std,
        )

    @property
    def _param_field_names(self) -> tuple[str, ...]:
        exclude = self._NON_PARAM_FIELDS | {"gauss_hermite"}
        if self.gauss_hermite:
            exclude = exclude | {"n_std"}
        return tuple(f.name for f in fields(self) if f.name not in exclude)

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        n_points = self.n_points
        rho, sigma, mu = kwargs["rho"], kwargs["sigma"], kwargs["mu"]
        std_y = jnp.sqrt(sigma**2 / (1 - rho**2))
        if self.gauss_hermite:
            return _gauss_hermite_normal(
                n_points=n_points, mu=mu / (1 - rho), sigma=std_y
            )[0]
        n_std = kwargs["n_std"]
        x_max = n_std * std_y
        x = jnp.linspace(-x_max, x_max, n_points)
        return x + mu / (1 - rho)

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        n_points = self.n_points
        rho, sigma = kwargs["rho"], kwargs["sigma"]
        std_y = jnp.sqrt(sigma**2 / (1 - rho**2))

        if self.gauss_hermite:
            nodes, _weights = _gauss_hermite_normal(
                n_points=n_points, mu=jnp.asarray(0.0), sigma=std_y
            )
        else:
            n_std = kwargs["n_std"]
            x_max = n_std * std_y
            nodes = jnp.linspace(-x_max, x_max, n_points)

        # Midpoints between consecutive nodes: (n_points - 1,)
        midpoints = (nodes[:-1] + nodes[1:]) / 2

        # CDF at midpoints for each source state: (n_points, n_points - 1)
        # Denominator is sigma (innovation std), not std_y (unconditional std),
        # because the conditional distribution y_{t+1} | y_t has variance sigma^2.
        cdf_vals = cdf((midpoints[None, :] - rho * nodes[:, None]) / sigma)
        first_col = cdf_vals[:, :1]
        last_col = 1 - cdf_vals[:, -1:]
        return jnp.concatenate(
            [first_col, jnp.diff(cdf_vals, axis=1), last_col], axis=1
        )

    def draw_shock(
        self,
        *,
        params: MappingProxyType[str, ScalarFloat | ScalarInt],
        key: PRNGKeyND,
        current_value: ScalarFloat,
    ) -> ScalarFloat:
        return (
            params["mu"]
            + sealed(params["rho"] * current_value)
            + sealed(params["sigma"] * standard_normal(key))
        )


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class _TauchenOnNodesProcess(TauchenAR1Process):
    """AR(1) transition rules on a fixed physical axis independent of law parameters."""

    nodes: tuple[float, ...]
    transition_rule: Literal["midpoint", "innovation_quadrature", "density"] = (
        "midpoint"
    )
    innovation_nodes: tuple[float, ...] = ()
    innovation_weights: tuple[float, ...] = ()

    _NON_PARAM_FIELDS: ClassVar[frozenset[str]] = (
        TauchenAR1Process._NON_PARAM_FIELDS  # noqa: SLF001
        | {
            "nodes",
            "n_std",
            "transition_rule",
            "innovation_nodes",
            "innovation_weights",
        }
    )

    def __post_init__(self) -> None:
        points = np.asarray(self.nodes, dtype=canonical_float_dtype())
        if (
            len(points) < 2  # noqa: PLR2004
            or self.n_points != len(points)
            or not np.isfinite(points).all()
            or not np.all(points[1:] > points[:-1])
        ):
            raise GridInitializationError(
                "nodes must contain at least two finite strictly increasing values "
                "at the active precision."
            )
        super().__post_init__()

    @property
    def constructor_name(self) -> str:
        """Name of the public call that builds this process, for error messages."""
        return "TauchenAR1Process.on_nodes"

    def fail_if_nonstationary_rho(self, *, rho: float | FloatND | IntND) -> None:
        """Require finite persistence; supplied nodes need no stationary law."""
        if not np.isfinite(np.asarray(rho)).all():
            raise GridInitializationError("rho must be finite for supplied nodes.")

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        """Return the supplied nodes, independent of runtime law parameters."""
        del kwargs
        return jnp.asarray(self.nodes, dtype=canonical_float_dtype())

    def get_gridpoints(self) -> Float1D:
        """Return the physical axis even when law parameters are unspecified."""
        return self.compute_gridpoints()

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Interpolate a coordinate on the supplied physical axis."""
        return grid_coordinates.get_irreg_coordinate(value=value, points=self.to_jax())

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        """Evaluate the selected transition row at each supplied source node."""
        return jax.vmap(partial(self.compute_transition_row, **kwargs))(self.to_jax())

    # keyword-only-exempt: library-callback=jax.vmap
    def compute_transition_row(
        self,
        from_value: ScalarFloat,
        *,
        rho: ScalarFloat | ScalarInt,
        sigma: ScalarFloat | ScalarInt,
        mu: ScalarFloat | ScalarInt,
    ) -> Float1D:
        """Return destination probabilities at an actual physical source value."""
        nodes = self.compute_gridpoints()
        if self.transition_rule == "midpoint":
            return tauchen_row(
                nodes=nodes, from_value=from_value, rho=rho, sigma=sigma, mu=mu
            )
        mean = mu + sealed(rho * from_value)
        if self.transition_rule == "density":
            upper = jnp.clip(jnp.searchsorted(nodes, mean), 1, len(nodes) - 1)
            lower = upper - 1
            nearest = jnp.where(
                mean - nodes[lower] <= nodes[upper] - mean, lower, upper
            )
            center = nodes[nearest]
            # Subtract the nearest node's log density before squaring can overflow.
            half_difference = nodes * 0.5 - center * 0.5
            delta = 2 * (half_difference / sigma)
            centered_sum = (half_difference + (center - mean)) / sigma
            is_center = jnp.arange(len(nodes)) == nearest
            # Equal distances have zero log ratio, even when their scaled gap overflows.
            zero_ratio = is_center | (jnp.isinf(delta) & (centered_sum == 0))
            safe_delta = jnp.where(zero_ratio, 0.0, delta)
            safe_sum = jnp.where(zero_ratio, 0.0, centered_sum)
            return jax.nn.softmax(-safe_delta * safe_sum)
        innovations = jnp.asarray(self.innovation_nodes, dtype=nodes.dtype)
        weights = jnp.asarray(self.innovation_weights, dtype=nodes.dtype)
        targets = jnp.clip(mean + sigma * innovations, nodes[0], nodes[-1])
        lower = jnp.clip(
            jnp.searchsorted(nodes, targets, side="right") - 1, 0, len(nodes) - 2
        )
        upper_share = (targets - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
        row = jnp.zeros_like(nodes).at[lower].add(weights * (1 - upper_share))
        return row.at[lower + 1].add(weights * upper_share)


def _prepare_innovation_quadrature(
    *,
    transition_rule: Literal["midpoint", "innovation_quadrature", "density"],
    n_quadrature: int | None,
    innovation_nodes: UserVector | None,
    innovation_weights: UserVector | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate the rule and freeze a standardized innovation probability measure."""
    if transition_rule not in {"midpoint", "innovation_quadrature", "density"}:
        raise GridInitializationError("Unknown on_nodes transition_rule.")
    supplied = innovation_nodes is not None or innovation_weights is not None
    if transition_rule != "innovation_quadrature":
        if n_quadrature is not None or supplied:
            raise GridInitializationError(
                "Quadrature inputs require innovation_quadrature."
            )
        return (), ()
    if supplied:
        if (
            innovation_nodes is None
            or innovation_weights is None
            or n_quadrature is not None
        ):
            raise GridInitializationError(
                "Supply both innovation arrays, or n_quadrature."
            )
        for name, value in (
            ("innovation_nodes", innovation_nodes),
            ("innovation_weights", innovation_weights),
        ):
            fail_if_not_a_numeric_vector(
                value=value, argument=name, example=f"{name}=jnp.array([0.0])"
            )
        points = np.asarray(innovation_nodes, dtype=canonical_float_dtype())
        weights = np.asarray(innovation_weights, dtype=canonical_float_dtype())
    else:
        count = 9 if n_quadrature is None else n_quadrature
        if isinstance(count, bool) or count < 1:
            raise GridInitializationError("n_quadrature must be a positive integer.")
        points, weights = np.polynomial.hermite_e.hermegauss(count)
        points = points.astype(canonical_float_dtype())
        weights = weights.astype(canonical_float_dtype())
    if (
        not len(points)
        or points.shape != weights.shape
        or not np.isfinite(points).all()
        or not np.isfinite(weights).all()
        or not np.all(weights >= 0)
        or not np.any(weights > 0)
    ):
        raise GridInitializationError(
            "Innovation nodes and nonnegative weights must be finite, nonempty, "
            "equal-length vectors with positive total mass."
        )
    # Scaling first avoids overflow when finite relative weights have a large total.
    weights = weights / weights.max()
    weights = weights / weights.sum()
    return tuple(float(x) for x in points), tuple(float(x) for x in weights)


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class RouwenhorstAR1Process(_AR1Process):
    r"""AR(1) process discretized via Rouwenhorst (1995).

    The process is
    $y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$,
    where $\varepsilon_t \sim N(0, \sigma_\varepsilon^2)$.

    Implementation based on [Kopecky & Suen (2010)](https://doi.org/10.1016/j.red.2010.02.002).

    """

    sigma: float | int | StateConditioned | None = None
    """Standard deviation of the innovation.

    A `StateConditioned` is accepted so that declaring one is refused with an
    explanation when the model is built: a Rouwenhorst transition depends on `rho`
    alone, so fixing the nodes would leave a conditioned `sigma` no channel.
    """

    mu: float | int | None = None
    """Intercept (drift) of the AR(1) process."""

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        n_points = self.n_points
        rho, sigma, mu = kwargs["rho"], kwargs["sigma"], kwargs["mu"]
        nu = jnp.sqrt((n_points - 1) / (1 - rho**2)) * sigma
        long_run_mean = mu / (1.0 - rho)
        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        n_points = self.n_points
        rho = kwargs["rho"]
        q = (rho + 1) / 2

        # Binomial coefficient lookup table
        C = jnp.array(
            [[comb(nr, k) for k in range(n_points)] for nr in range(n_points)]
        )

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
        *,
        params: MappingProxyType[str, ScalarFloat | ScalarInt],
        key: PRNGKeyND,
        current_value: ScalarFloat,
    ) -> ScalarFloat:
        return (
            params["mu"]
            + sealed(params["rho"] * current_value)
            + sealed(params["sigma"] * standard_normal(key))
        )


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class TauchenNormalMixtureAR1Process(_AR1Process):
    r"""AR(1) process with mixture-of-normals innovations, discretized via Tauchen.

    The process is
    $y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$,
    where $\varepsilon_t \sim p_1 \, N(\mu_1, \sigma_1^2)
    + (1 - p_1) \, N(\mu_2, \sigma_2^2)$.

    Transition probabilities use the mixture CDF in place of the normal CDF,
    following [Fella, Gallipoli & Pan (2019)](https://doi.org/10.1016/j.red.2019.03.013),
    Section 4.3 / Eq. 21.

    """

    mu: float | int | None = None
    """Intercept (drift) of the AR(1) process."""

    n_std: float | int | None = None
    """Number of unconditional standard deviations for the grid boundary."""

    p1: float | int | None = None
    """Probability of the first mixture component."""

    mu1: float | int | None = None
    """Mean of the first mixture component."""

    sigma1: float | int | None = None
    """Standard deviation of the first mixture component."""

    mu2: float | int | None = None
    """Mean of the second mixture component."""

    sigma2: float | int | None = None
    """Standard deviation of the second mixture component."""

    @staticmethod
    def _innovation_variance(
        *,
        p1: ScalarFloat,
        mu1: ScalarFloat,
        sigma1: ScalarFloat,
        mu2: ScalarFloat,
        sigma2: ScalarFloat,
    ) -> ScalarFloat:
        """Compute the variance of the mixture innovation."""
        mean_eps = p1 * mu1 + (1 - p1) * mu2
        return p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - mean_eps**2

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        n_points = self.n_points
        rho, mu = kwargs["rho"], kwargs["mu"]
        n_std = kwargs["n_std"]
        p1, mu1, sigma1 = kwargs["p1"], kwargs["mu1"], kwargs["sigma1"]
        mu2, sigma2 = kwargs["mu2"], kwargs["sigma2"]

        sigma_eps_sq = self._innovation_variance(
            p1=p1, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2
        )
        std_y = jnp.sqrt(sigma_eps_sq / (1 - rho**2))
        mean_eps = p1 * mu1 + (1 - p1) * mu2
        long_run_mean = (mu + mean_eps) / (1 - rho)
        x_max = n_std * std_y
        return jnp.linspace(long_run_mean - x_max, long_run_mean + x_max, n_points)

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        n_points = self.n_points
        rho, mu = kwargs["rho"], kwargs["mu"]
        n_std = kwargs["n_std"]
        p1, mu1, sigma1 = kwargs["p1"], kwargs["mu1"], kwargs["sigma1"]
        mu2, sigma2 = kwargs["mu2"], kwargs["sigma2"]

        sigma_eps_sq = self._innovation_variance(
            p1=p1, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2
        )
        std_y = jnp.sqrt(sigma_eps_sq / (1 - rho**2))
        mean_eps = p1 * mu1 + (1 - p1) * mu2
        long_run_mean = (mu + mean_eps) / (1 - rho)
        x_max = n_std * std_y
        x = jnp.linspace(long_run_mean - x_max, long_run_mean + x_max, n_points)
        step = (2 * x_max) / (n_points - 1)
        half_step = 0.5 * step

        # z[i, j] = x[j] - mu - rho * x[i]: the innovation needed to reach x[j]
        # from x[i], shifted by -mu to center on the innovation distribution.
        z = x[None, :] - mu - rho * x[:, None]

        upper = _mixture_cdf(
            x=z + half_step, p1=p1, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2
        )
        lower = _mixture_cdf(
            x=z - half_step, p1=p1, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2
        )
        P = upper - lower
        P = P.at[:, 0].set(upper[:, 0])
        return P.at[:, -1].set(1 - lower[:, -1])

    def draw_shock(
        self,
        *,
        params: MappingProxyType[str, ScalarFloat | ScalarInt],
        key: PRNGKeyND,
        current_value: ScalarFloat,
    ) -> ScalarFloat:
        key1, key2 = jax.random.split(key=key)
        component = jax.random.bernoulli(key1, params["p1"])
        normal = standard_normal(key2)
        eps = jnp.where(
            component,
            params["mu1"] + sealed(params["sigma1"] * normal),
            params["mu2"] + sealed(params["sigma2"] * normal),
        )
        return params["mu"] + sealed(params["rho"] * current_value) + eps
