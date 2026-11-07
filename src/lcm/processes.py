"""User-facing stochastic process leaf classes.

Each class declares a **state grid AND** its transition mechanism. Unlike
ordinary grids (`LinSpacedGrid` etc.), which are pure outcome-space, a
process class bundles both the discretization nodes and the dynamics on
those nodes.

Place an instance of one of these in `Regime(states={...})` — *not* in
`state_transitions`. pylcm treats the discretization nodes as the realized
values of the state at each grid point; the transition mechanism is
invoked automatically.

Naming convention: `<Distribution><Kind>Process`.

- `*IIDProcess`: independent draws at each period.
- `*AR1Process`: AR(1) process with a chosen discretization scheme.

Process-law values keep the numeric kind the user wrote (`3` versus `3.0`):
the kind is part of a law's identity, so these classes never normalize it. The
shared bases, quadrature helpers and transition-row builders live in
`_lcm.processes`.
"""

import math
from dataclasses import dataclass, field, fields
from math import comb
from types import MappingProxyType
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax.scipy.stats.norm import cdf

from _lcm.beartype_conf import GRID_CONF
from _lcm.grids.base import UserVector, fail_if_not_a_numeric_vector
from _lcm.processes.ar1 import (
    TransitionRule,
    _AR1Process,
    _prepare_innovation_quadrature,
)
from _lcm.processes.base import (
    SIMULATE_DRAWS_RENAMED,
    _gauss_hermite_normal,
    _mixture_cdf,
    _validate_gauss_hermite_grid,
    sealed,
    standard_normal,
)
from _lcm.processes.iid import FOLD_REMOVED, _IIDProcess
from _lcm.processes.joint_normal import (
    HostNormalArray,
    NormalCovariance,
    NormalMean,
    _exact_zero_pivots,
    _rounded_zero_pivots,
    covariance_factor,
)
from _lcm.processes.state_conditioned import tauchen_row
from _lcm.typing import PRNGKeyND, ProcessLawScalar
from lcm.exceptions import GridInitializationError
from lcm.state_conditioned import StateConditioned
from lcm.typing import (
    Float1D,
    Float2D,
    FloatND,
    OffGrid,
    ScalarFloat,
    ScalarInt,
    StateName,
)

__all__ = [
    "JointNormalIIDProcess",
    "LogNormalIIDProcess",
    "NormalIIDProcess",
    "NormalMixtureIIDProcess",
    "RouwenhorstAR1Process",
    "StateConditioned",
    "TauchenAR1Process",
    "TauchenNormalMixtureAR1Process",
    "UniformIIDProcess",
]


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class UniformIIDProcess(_IIDProcess):
    r"""Discretized IID uniform process: $U(\text{start}, \text{stop})$.

    The continuous distribution is discretized into `n_points` equally spaced
    points between `start` and `stop`.

    """

    start: ProcessLawScalar | None = None
    """Lower bound of the uniform distribution."""

    stop: ProcessLawScalar | None = None
    """Upper bound of the uniform distribution."""

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        return jnp.linspace(
            start=kwargs["start"], stop=kwargs["stop"], num=self.n_points
        )

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:  # noqa: ARG002
        n_points = self.n_points
        return jnp.full((n_points, n_points), fill_value=1 / n_points)

    def draw_shock(
        self, *, params: MappingProxyType[str, ScalarFloat | ScalarInt], key: PRNGKeyND
    ) -> ScalarFloat:
        return jax.random.uniform(
            key=key, minval=params["start"], maxval=params["stop"]
        )


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class NormalIIDProcess(_IIDProcess):
    r"""Discretized IID normal process: $N(\mu_\varepsilon, \sigma_\varepsilon^2)$.

    When `gauss_hermite=True`, the distribution is discretized using
    Gauss-Hermite quadrature nodes and weights.  When `gauss_hermite=False`,
    it uses `n_points` equally spaced points spanning
    $\mu_\varepsilon \pm n_\text{std} \cdot \sigma_\varepsilon$.

    Gauss-Hermite supports even and odd `n_points`; only odd counts include the mean.

    """

    gauss_hermite: bool
    """Use Gauss-Hermite quadrature nodes and weights."""

    mu: ProcessLawScalar | None = None
    """Mean of the distribution."""

    sigma: ProcessLawScalar | StateConditioned | None = None
    """Standard deviation of the distribution.

    A `StateConditioned` here conditions the innovation scale on a discrete state
    of the same regime; `__post_init__` then leaves this holding the scalar that
    places the nodes.
    """

    n_std: ProcessLawScalar | None = None
    """Number of standard deviations from the mean to the grid boundary."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_gauss_hermite_grid(
            n_points=self.n_points, gauss_hermite=self.gauss_hermite, n_std=self.n_std
        )

    @property
    def _param_field_names(self) -> tuple[str, ...]:
        exclude = self._NON_PARAM_FIELDS | {"gauss_hermite"}
        if self.gauss_hermite:
            exclude = exclude | {"n_std"}
        return tuple(f.name for f in fields(self) if f.name not in exclude)

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        n_points = self.n_points
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            nodes, _weights = _gauss_hermite_normal(
                n_points=n_points, mu=mu, sigma=sigma
            )
            return nodes
        n_std = kwargs["n_std"]
        x_min = mu - n_std * sigma
        x_max = mu + n_std * sigma
        return jnp.linspace(start=x_min, stop=x_max, num=n_points)

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        n_points = self.n_points
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            _nodes, weights = _gauss_hermite_normal(
                n_points=n_points, mu=mu, sigma=sigma
            )
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
        self, *, params: MappingProxyType[str, ScalarFloat | ScalarInt], key: PRNGKeyND
    ) -> ScalarFloat:
        return params["mu"] + sealed(params["sigma"] * standard_normal(key))


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class LogNormalIIDProcess(_IIDProcess):
    r"""Discretized IID log-normal process: $\ln X \sim N(\mu, \sigma^2)$.

    Gauss-Hermite supports even and odd `n_points`; only odd counts include `exp(mu)`.
    """

    gauss_hermite: bool
    """Use Gauss-Hermite quadrature nodes and weights."""

    mu: ProcessLawScalar | None = None
    """Mean of the underlying normal distribution ($E[\\ln X]$)."""

    sigma: ProcessLawScalar | StateConditioned | None = None
    """Standard deviation of the underlying normal distribution.

    A `StateConditioned` is accepted so that declaring one is refused with an
    explanation when the model is built, rather than by the type checker alone.
    """

    n_std: ProcessLawScalar | None = None
    """Number of standard deviations in log-space for the grid boundary."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_gauss_hermite_grid(
            n_points=self.n_points, gauss_hermite=self.gauss_hermite, n_std=self.n_std
        )

    @property
    def _param_field_names(self) -> tuple[str, ...]:
        exclude = self._NON_PARAM_FIELDS | {"gauss_hermite"}
        if self.gauss_hermite:
            exclude = exclude | {"n_std"}
        return tuple(f.name for f in fields(self) if f.name not in exclude)

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        n_points = self.n_points
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            nodes, _weights = _gauss_hermite_normal(
                n_points=n_points, mu=mu, sigma=sigma
            )
            return jnp.exp(nodes)
        n_std = kwargs["n_std"]
        return jnp.exp(jnp.linspace(mu - n_std * sigma, mu + n_std * sigma, n_points))

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        n_points = self.n_points
        mu, sigma = kwargs["mu"], kwargs["sigma"]
        if self.gauss_hermite:
            _nodes, weights = _gauss_hermite_normal(
                n_points=n_points, mu=mu, sigma=sigma
            )
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
        self, *, params: MappingProxyType[str, ScalarFloat | ScalarInt], key: PRNGKeyND
    ) -> ScalarFloat:
        return jnp.exp(params["mu"] + sealed(params["sigma"] * standard_normal(key)))


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class NormalMixtureIIDProcess(_IIDProcess):
    r"""Discretized IID normal-mixture process.

    Each period's draw is taken from a two-component normal mixture:

    $\varepsilon \sim p_1 \, N(\mu_1, \sigma_1^2)
    + (1 - p_1) \, N(\mu_2, \sigma_2^2)$

    Grid points are equally spaced around the mixture mean $\pm n_\text{std}$
    mixture standard deviations. Transition probabilities use CDF binning with
    the mixture CDF in place of the normal CDF.

    """

    n_std: ProcessLawScalar | None = None
    """Number of mixture standard deviations for the grid boundary."""

    p1: ProcessLawScalar | None = None
    """Probability of the first mixture component."""

    mu1: ProcessLawScalar | None = None
    """Mean of the first mixture component."""

    sigma1: ProcessLawScalar | None = None
    """Standard deviation of the first mixture component."""

    mu2: ProcessLawScalar | None = None
    """Mean of the second mixture component."""

    sigma2: ProcessLawScalar | None = None
    """Standard deviation of the second mixture component."""

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        n_points = self.n_points
        n_std = kwargs["n_std"]
        p1, mu1, sigma1 = kwargs["p1"], kwargs["mu1"], kwargs["sigma1"]
        mu2, sigma2 = kwargs["mu2"], kwargs["sigma2"]

        mean_eps = p1 * mu1 + (1 - p1) * mu2
        var_eps = (
            p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - mean_eps**2
        )
        std_eps = jnp.sqrt(var_eps)
        return jnp.linspace(
            mean_eps - n_std * std_eps, mean_eps + n_std * std_eps, n_points
        )

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        n_points = self.n_points
        n_std = kwargs["n_std"]
        p1, mu1, sigma1 = kwargs["p1"], kwargs["mu1"], kwargs["sigma1"]
        mu2, sigma2 = kwargs["mu2"], kwargs["sigma2"]

        mean_eps = p1 * mu1 + (1 - p1) * mu2
        var_eps = (
            p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - mean_eps**2
        )
        std_eps = jnp.sqrt(var_eps)
        x = jnp.linspace(
            mean_eps - n_std * std_eps, mean_eps + n_std * std_eps, n_points
        )
        step = (2 * n_std * std_eps) / (n_points - 1)
        half_step = 0.5 * step

        upper = _mixture_cdf(
            x=x + half_step, p1=p1, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2
        )
        lower = _mixture_cdf(
            x=x - half_step, p1=p1, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2
        )
        w = upper - lower
        w = w.at[0].set(upper[0])
        w = w.at[-1].set(1 - lower[-1])
        return jnp.full((n_points, n_points), fill_value=w)

    def draw_shock(
        self, *, params: MappingProxyType[str, ScalarFloat | ScalarInt], key: PRNGKeyND
    ) -> ScalarFloat:
        key1, key2 = jax.random.split(key=key)
        component = jax.random.bernoulli(key1, params["p1"])
        normal = standard_normal(key2)
        return jnp.where(
            component,
            params["mu1"] + sealed(params["sigma1"] * normal),
            params["mu2"] + sealed(params["sigma2"] * normal),
        )


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

    sigma: ProcessLawScalar | StateConditioned | None = None
    """Standard deviation of the innovation.

    A `StateConditioned` here conditions the innovation scale on a discrete state
    of the same regime; `__post_init__` then leaves this holding the scalar that
    places the nodes.
    """

    mu: ProcessLawScalar | None = None
    """Intercept (drift) of the AR(1) process."""

    n_std: ProcessLawScalar | None = None
    """Number of standard deviations for the grid boundary."""

    @staticmethod
    def on_nodes(
        *,
        nodes: UserVector,
        simulated_draws: Literal["continuous", "nodes"] = "continuous",
        off_grid: OffGrid = "extrapolate",
        rho: float | None = None,
        sigma: float | StateConditioned | None = None,
        mu: float | None = 0.0,
        transition_rule: TransitionRule = "midpoint",
        n_quadrature: int | None = None,
        innovation_nodes: UserVector | None = None,
        innovation_weights: UserVector | None = None,
        **removed: object,  # noqa: PAN001 - refuses the renamed keyword
    ) -> TauchenAR1Process:
        """Bin an AR(1) law on finite increasing physical nodes.

        The default rule integrates midpoint bins with open first and last tails.
        ``density`` normalizes normal densities at the nodes; it ignores cell widths.
        Density derivatives use normalized scaled arithmetic without host checks.
        A test-suite certificate checks explicitly supplied rows;
        routine differentiation makes no componentwise certificate promise.
        ``innovation_quadrature`` maps standardized innovations linearly to adjacent
        state nodes, clipping targets to the endpoints. Supply ``n_quadrature``
        (default 9 Gauss-Hermite points), or both innovation arrays. Weights must be
        finite, nonnegative, and have positive total mass; they are normalized.
        Continuous simulation still draws the normal law. Node simulation samples
        the selected transition matrix. `off_grid` sets how a value beyond the
        first or last node is read, as on a continuous grid. Ordinary Tauchen
        construction is unchanged.
        The nodes do not depend on stationarity, so unit roots are supported.
        Unspecified law parameters are supplied at solve or simulation time.
        `nodes` is a one-dimensional NumPy or JAX array of integer or floating
        dtype; lists and tuples are refused.
        """
        if "simulate_draws" in removed:
            raise TypeError(SIMULATE_DRAWS_RENAMED)
        if removed:
            msg = f"Unexpected keyword arguments: {sorted(removed)}."
            raise TypeError(msg)
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
        from _lcm.processes.ar1_on_nodes import (  # noqa: PLC0415 - subclasses this class
            _TauchenOnNodesProcess,
        )

        return _TauchenOnNodesProcess(
            transition_rule=transition_rule,
            innovation_nodes=quadrature_nodes,
            innovation_weights=quadrature_weights,
            nodes=tuple(float(node) for node in np.asarray(nodes).tolist()),
            simulated_draws=simulated_draws,
            off_grid=off_grid,
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

    def transition_row_at(
        self, *, from_value: ScalarFloat, **kwargs: ScalarFloat | ScalarInt
    ) -> Float1D:
        """Return the law's transition row from any current value, on the nodes.

        The row bins the conditional law at the nodes' midpoints, exactly as
        `compute_transition_probs` does at each node, so it is a probability
        distribution wherever the current value lies.
        """
        return tauchen_row(
            nodes=self.compute_gridpoints(**kwargs),
            rho=kwargs["rho"],
            sigma=kwargs["sigma"],
            from_value=from_value,
            mu=kwargs["mu"],
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
class RouwenhorstAR1Process(_AR1Process):
    r"""AR(1) process discretized via Rouwenhorst (1995).

    The process is
    $y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$,
    where $\varepsilon_t \sim N(0, \sigma_\varepsilon^2)$.

    Implementation based on [Kopecky & Suen (2010)](https://doi.org/10.1016/j.red.2010.02.002).

    """

    sigma: ProcessLawScalar | StateConditioned | None = None
    """Standard deviation of the innovation.

    A `StateConditioned` is accepted so that declaring one is refused with an
    explanation when the model is built: a Rouwenhorst transition depends on `rho`
    alone, so fixing the nodes would leave a conditioned `sigma` no channel.
    """

    mu: ProcessLawScalar | None = None
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

    mu: ProcessLawScalar | None = None
    """Intercept (drift) of the AR(1) process."""

    n_std: ProcessLawScalar | None = None
    """Number of unconditional standard deviations for the grid boundary."""

    p1: ProcessLawScalar | None = None
    """Probability of the first mixture component."""

    mu1: ProcessLawScalar | None = None
    """Mean of the first mixture component."""

    sigma1: ProcessLawScalar | None = None
    """Standard deviation of the first mixture component."""

    mu2: ProcessLawScalar | None = None
    """Mean of the second mixture component."""

    sigma2: ProcessLawScalar | None = None
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

    def transition_row_at(
        self, *, from_value: ScalarFloat, **kwargs: ScalarFloat | ScalarInt
    ) -> Float1D:
        """Return the law's transition row from any current value, on the nodes.

        Each node's bin reaches half a step to either side, the first and last
        bins open, as in `compute_transition_probs`.
        """
        x = self.compute_gridpoints(**kwargs)
        mixture = {name: kwargs[name] for name in ("p1", "mu1", "sigma1", "mu2")}
        half_step = 0.5 * (x[-1] - x[0]) / (self.n_points - 1)
        z = x - kwargs["mu"] - kwargs["rho"] * from_value
        upper = _mixture_cdf(x=z + half_step, sigma2=kwargs["sigma2"], **mixture)
        lower = _mixture_cdf(x=z - half_step, sigma2=kwargs["sigma2"], **mixture)
        row = (upper - lower).at[0].set(upper[0])
        return row.at[-1].set(1 - lower[-1])

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


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class JointNormalIIDProcess:
    """Declare several named, jointly normal shocks independent over time.

    `n_points` is a count per dimension, or a tuple in `states` order. Quadrature
    is the product of scalar standard-normal rules, transformed by a lower-triangular
    covariance factor. Simulation uses the corresponding continuous Gaussian law.
    Singular positive-semidefinite covariance matrices are supported, including zero
    variances. Parameters set to `None` are supplied under the declaration's key in
    the regime's parameter mapping.

    The engine's value axes are independent normal coordinates: a correlated tensor
    does not have separate Cartesian grids in physical shock coordinates. User DAGs
    and simulation frames read the named physical shocks.
    """

    states: tuple[StateName, ...]
    """Physical shock names, in covariance row and column order."""
    n_points: int | tuple[int, ...]
    """Quadrature cardinalities in the same order as `states`."""
    gauss_hermite: bool
    """Use scalar Gauss-Hermite rules rather than normal-CDF bins."""
    mean: NormalMean | None = None
    """Joint mean; `None` leaves it as a runtime parameter."""
    cov: NormalCovariance | None = None
    """Symmetric positive-semidefinite covariance; `None` leaves it at runtime."""
    n_std: ProcessLawScalar | None = None
    """Equally spaced support halfwidth, required without Gauss-Hermite."""
    exact_zero_pivots: tuple[tuple[str, tuple[bool, ...] | None], ...] = field(
        init=False, default=(), repr=False, compare=False
    )
    """Exact zero pivots of the fixed covariance rounded to each float dtype name.

    `None` where the rounded covariance is not positive semidefinite."""

    def __new__(
        cls,
        *args: object,  # noqa: ARG004, PAN001 - refuses removed keywords first
        **kwargs: object,  # noqa: PAN001 - refuses removed keywords before any check
    ) -> Self:
        if "fold" in kwargs:
            raise TypeError(FOLD_REMOVED)
        return super().__new__(cls)

    def __post_init__(self) -> None:
        if not self.states or len(set(self.states)) != len(self.states):
            raise GridInitializationError(
                "Joint normal states must be nonempty and unique."
            )
        if any(
            not name.isidentifier() or name.startswith("next_") or "__" in name
            for name in self.states
        ):
            raise GridInitializationError(
                "Joint normal states must be ordinary state names."
            )
        counts = self.node_counts
        if len(counts) != len(self.states) or any(
            isinstance(count, bool) or count < 1 for count in counts
        ):
            raise GridInitializationError(
                "n_points must supply a positive count per state."
            )
        if not self.gauss_hermite and (
            self.n_std is None
            or not math.isfinite(self.n_std)
            or self.n_std <= 0
            or min(counts) < 2  # noqa: PLR2004 - scalar CDF bins need two endpoints
        ):
            raise GridInitializationError(
                "CDF bins require positive finite n_std and at least two nodes."
            )
        for count in counts:
            self.standard_normal_process(count)
        if self.mean is not None:
            self.validate_mean(self.mean)
            object.__setattr__(self, "mean", tuple(float(value) for value in self.mean))
        if self.cov is not None:
            self.validate_covariance(self.cov)
            object.__setattr__(
                self,
                "cov",
                tuple(tuple(float(value) for value in row) for row in self.cov),
            )
            object.__setattr__(
                self,
                "exact_zero_pivots",
                tuple(
                    (
                        np.dtype(dtype).name,
                        _rounded_zero_pivots(cov=self.cov, dtype=dtype),
                    )
                    for dtype in (np.float32, np.float64)
                ),
            )

    @property
    def node_counts(self) -> tuple[int, ...]:
        """Return one quadrature cardinality per named shock."""
        return (
            (self.n_points,) * len(self.states)
            if isinstance(self.n_points, int)
            else self.n_points
        )

    def standard_normal_process(self, n_points: int) -> NormalIIDProcess:
        """Return the scalar coordinate rule used by the joint tensor."""
        return NormalIIDProcess(
            n_points=n_points,
            gauss_hermite=self.gauss_hermite,
            mu=0.0,
            sigma=1.0,
            n_std=self.n_std,
        )

    def validate_mean(self, mean: NormalMean | FloatND | HostNormalArray) -> None:
        """Reject a nonfinite or incorrectly shaped joint mean."""
        values = np.asarray(mean)
        if values.shape != (len(self.states),) or not np.isfinite(values).all():
            raise GridInitializationError(
                "mean must be a finite vector in states order."
            )

    def validate_covariance(
        self, cov: NormalCovariance | FloatND | HostNormalArray
    ) -> tuple[bool, ...]:
        """Reject a covariance without assuming small negative pivots are zero.

        Exact rational LDL elimination classifies the supplied finite binary floats.
        Thus a genuinely indefinite matrix is never accepted using an eigenvalue
        tolerance, while exact singular matrices retain their zero pivots.

        Returns:
            Whether each exact pivot, in states order, is zero.
        """
        return _exact_zero_pivots(cov=np.asarray(cov), size=len(self.states))

    def fixed_zero_pivots(self) -> tuple[bool, ...] | None:
        """Return the exact zero pivots of the fixed covariance at active precision.

        The factor eliminates nothing on these pivots, so a fixed singular law keeps
        exactly its declared support. `None` for a runtime covariance.
        """
        if self.cov is None:
            return None
        pattern = dict(self.exact_zero_pivots)[jnp.asarray(self.cov).dtype.name]
        if pattern is None:
            raise GridInitializationError(
                "cov must be positive semidefinite at the active float precision."
            )
        return pattern

    def compute_gridpoints(
        self, *, mean: FloatND | None = None, cov: FloatND | None = None
    ) -> Float2D:
        """Transform the tensor of scalar nodes using already validated parameters."""
        axes = [
            self.standard_normal_process(count).get_gridpoints()
            for count in self.node_counts
        ]
        standard_nodes = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, len(self.states)
        )
        return jax.vmap(lambda node: self.transform(standard=node, mean=mean, cov=cov))(
            standard_nodes
        )

    def get_gridpoints(self) -> Float2D:
        """Return the fixed support, or NaNs when the law needs runtime parameters."""
        if self.mean is None or self.cov is None:
            return jnp.full((math.prod(self.node_counts), len(self.states)), jnp.nan)
        return self.compute_gridpoints()

    def compute_grid_bounds(
        self, *, mean: FloatND | None = None, cov: FloatND | None = None
    ) -> Float2D:
        """Return physical marginal extents without constructing the product tensor.

        These intervals describe each support column separately, not a Cartesian
        joint support or the unbounded continuous sampling distribution.
        """
        covariance = jnp.asarray(self.cov) if self.cov is not None else cov
        if covariance is None:
            raise GridInitializationError("Joint normal cov must be supplied.")
        factor = covariance_factor(cov=covariance, zero_pivots=self.fixed_zero_pivots())
        axes = [
            self.standard_normal_process(count).get_gridpoints()
            for count in self.node_counts
        ]
        low = jnp.stack([axis.min() for axis in axes])
        high = jnp.stack([axis.max() for axis in axes])
        bounds = []
        for row in range(len(self.states)):
            positive = factor[row] >= 0
            lower = self.transform(
                standard=jnp.where(positive, low, high), mean=mean, cov=cov
            )[row]
            upper = self.transform(
                standard=jnp.where(positive, high, low), mean=mean, cov=cov
            )[row]
            bounds.append(jnp.stack((lower, upper)))
        return jnp.stack(bounds)

    def get_weights(self) -> Float1D:
        """Return the product rule's parameter-independent quadrature weights."""
        weights = jnp.ones(1)
        for count in self.node_counts:
            scalar_weights = self.standard_normal_process(count).get_transition_probs()[
                0
            ]
            weights = (weights[:, None] * scalar_weights[None, :]).reshape(-1)
        return weights

    def transform(
        self,
        *,
        standard: Float1D,
        mean: FloatND | None = None,
        cov: FloatND | None = None,
    ) -> Float1D:
        """Map independent coordinates into the named physical shocks."""
        mean_array = jnp.asarray(self.mean) if self.mean is not None else mean
        cov_array = jnp.asarray(self.cov) if self.cov is not None else cov
        if mean_array is None or cov_array is None:
            raise GridInitializationError("Joint normal mean and cov must be supplied.")
        factor = covariance_factor(cov=cov_array, zero_pivots=self.fixed_zero_pivots())
        values = []
        for row in range(len(self.states)):
            value = mean_array[row]
            for column in range(row + 1):
                value = sealed(value + sealed(factor[row, column] * standard[column]))
            values.append(value)
        return jnp.stack(values)

    def draw_shock(
        self, *, key: PRNGKeyND, mean: FloatND | None = None, cov: FloatND | None = None
    ) -> Float1D:
        """Draw one correlated shock vector from independent sealed normal draws."""
        keys = jax.random.split(key, len(self.states))
        return self.transform(
            standard=jax.vmap(standard_normal)(keys), mean=mean, cov=cov
        )
