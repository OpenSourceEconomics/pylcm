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
from jax.experimental import io_callback
from jax.scipy.stats.norm import cdf
from jaxtyping import Int

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
from _lcm.processes.density_derivative_admission import _checked_density_coefficients
from _lcm.processes.state_conditioned import tauchen_row
from _lcm.typing import PRNGKeyND
from lcm.exceptions import GridInitializationError
from lcm.typing import Bool1D, Float1D, FloatND, IntND, ScalarFloat, ScalarInt


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
        Its first-order sensitivities use dimensionless absolute and relative
        error budgets of 1e-12 (float64) or 1e-5 (float32). Differentiation checks
        every parameter column and raises `DensityDerivativeAccuracyError` if
        any component cannot be certified. Checks run on the host during AD.
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
        if self.transition_rule == "density":
            return _normal_density_row(
                (
                    nodes,
                    jnp.asarray(mu, dtype=nodes.dtype),
                    jnp.asarray(rho, dtype=nodes.dtype),
                    jnp.asarray(from_value, dtype=nodes.dtype),
                    jnp.asarray(sigma, dtype=nodes.dtype),
                )
            )
        mean = mu + sealed(rho * from_value)
        innovations = jnp.asarray(self.innovation_nodes, dtype=nodes.dtype)
        weights = jnp.asarray(self.innovation_weights, dtype=nodes.dtype)
        targets = jnp.clip(mean + sigma * innovations, nodes[0], nodes[-1])
        lower = jnp.clip(
            jnp.searchsorted(nodes, targets, side="right") - 1, 0, len(nodes) - 2
        )
        left, right = nodes[lower], nodes[lower + 1]
        # Finite endpoints need not have a finite difference. Scale the operands
        # before subtraction, but retain the resolution of small/adjacent cells.
        wide = (
            (left < 0)
            & (right > 0)
            & (
                jnp.maximum(jnp.abs(left), jnp.abs(right))
                > jnp.finfo(nodes.dtype).max / 2
            )
        )
        scale = jnp.where(wide, 0.5, 1.0)
        upper_share = (targets * scale - left * scale) / (right * scale - left * scale)
        row = jnp.zeros_like(nodes).at[lower].add(weights * (1 - upper_share))
        return row.at[lower + 1].add(weights * upper_share)


def _rounded_sum_with_error(
    *, left: FloatND, right: FloatND
) -> tuple[FloatND, FloatND]:
    """Two-sum with explicit rounding boundaries, also under XLA reassociation."""
    total = sealed(left + right)
    right_part = sealed(total - left)
    left_error = sealed(left - sealed(total - right_part))
    right_error = sealed(right - right_part)
    return total, sealed(left_error + right_error)


type _ScaledFloat = tuple[FloatND, IntND]
# A normalized mantissa and an unrestricted binary exponent.


def _scaled_float(value: FloatND) -> _ScaledFloat:
    """Decompose stored finite bits without flushing subnormal inputs.

    Arithmetic acts only on normal mantissas. In particular, a product's low
    word need not be representable in the physical units of the state axis.
    """
    info = jnp.finfo(value.dtype)
    integer = jnp.dtype(f"uint{info.bits}")
    bits = jax.lax.bitcast_convert_type(value, integer)
    fraction = bits & integer.type(2**info.nmant - 1)
    exponent = (bits >> info.nmant) & integer.type(
        2 ** (info.bits - info.nmant - 1) - 1
    )
    negative = (bits >> (info.bits - 1)) != 0
    normal = 0.5 + fraction.astype(value.dtype) * 2.0 ** -(info.nmant + 1)
    subnormal, subnormal_exponent = jnp.frexp(fraction.astype(value.dtype))
    mantissa = jnp.where(exponent != 0, normal, subnormal)
    power = jnp.where(
        exponent != 0,
        exponent.astype(jnp.int32) - (info.maxexp - 2),
        subnormal_exponent + info.minexp - info.nmant,
    )
    mantissa = jnp.where(negative, -mantissa, mantissa)
    return mantissa, jnp.where(mantissa != 0, power, -100_000)


def _normalize_scaled(*, mantissa: FloatND, exponent: IntND) -> _ScaledFloat:
    """Normalize arithmetic on ordinary-sized mantissas, keeping true zeros."""
    normalized, shift = jnp.frexp(mantissa)
    return normalized, jnp.where(normalized != 0, exponent + shift, -100_000)


def _scaled_two_sum(
    *, left: _ScaledFloat, right: _ScaledFloat
) -> tuple[_ScaledFloat, _ScaledFloat]:
    """Return exact high/low words without scaling a small term out of range.

    Disjoint significands are already an expansion and stay separate. When
    they overlap, aligning by at most one precision plus two bits keeps every
    TwoSum intermediate normal; the returned words are normalized separately.
    """
    left_m, left_e = left
    right_m, right_e = right
    left_is_larger = left_e >= right_e
    high_m = jnp.where(left_is_larger, left_m, right_m)
    high_e = jnp.maximum(left_e, right_e)
    low_m = jnp.where(left_is_larger, right_m, left_m)
    low_e = jnp.minimum(left_e, right_e)
    limit = jnp.finfo(high_m.dtype).nmant + 3
    distance = high_e - low_e
    power = -jnp.minimum(distance, limit)
    info = jnp.finfo(high_m.dtype)
    integer = jnp.dtype(f"int{info.bits}")
    factor = jax.lax.bitcast_convert_type(
        (power.astype(integer) + info.maxexp - 1) << info.nmant, high_m.dtype
    )
    total, error = _rounded_sum_with_error(left=high_m, right=sealed(low_m * factor))
    total_m, total_e = _normalize_scaled(mantissa=total, exponent=high_e)
    error_m, error_e = _normalize_scaled(mantissa=error, exponent=high_e)
    separated = distance > limit
    return (
        (jnp.where(separated, high_m, total_m), jnp.where(separated, high_e, total_e)),
        (jnp.where(separated, low_m, error_m), jnp.where(separated, low_e, error_e)),
    )


def _scaled_sum(terms: tuple[_ScaledFloat, ...]) -> _ScaledFloat:
    """Grow an exact, fixed-length nonoverlapping floating expansion.

    Each insertion retains every TwoSum residual. Only after all cancellation
    do we round to its largest nonzero word. Its sign is the sign of the exact
    sum; the omitted nonoverlapping tail is below working relative precision.
    The number of terms is bounded by the affine formula, independent of grid
    size and exponent range. No integer accumulator or arbitrary precision is
    used in production.
    """
    expansion: tuple[_ScaledFloat, ...] = (terms[0],)
    for term in terms[1:]:
        carry = term
        residuals: tuple[_ScaledFloat, ...] = ()
        for component in expansion:
            carry, error = _scaled_two_sum(left=carry, right=component)
            residuals = (*residuals, error)
        expansion = (*residuals, carry)
    largest = expansion[0]
    for mantissa, exponent in expansion[1:]:
        largest = (
            jnp.where(mantissa != 0, mantissa, largest[0]),
            jnp.where(mantissa != 0, exponent, largest[1]),
        )
    return largest


def _scaled_product_words(
    *, left: FloatND, right: FloatND
) -> tuple[_ScaledFloat, _ScaledFloat]:
    """Retain the product residual on normalized mantissas (Dekker).

    All splitting and error products stay normal even when the physical
    product or its low word would overflow or underflow. Rounding barriers
    prevent contraction/reassociation from deleting the error word.
    """
    left_m, left_e = _scaled_float(left)
    right_m, right_e = _scaled_float(right)
    factor = 2.0 ** ((jnp.finfo(left.dtype).nmant + 2) // 2) + 1.0
    left_split = sealed(left_m * factor)
    left_high = sealed(left_split - sealed(left_split - left_m))
    left_low = sealed(left_m - left_high)
    right_split = sealed(right_m * factor)
    right_high = sealed(right_split - sealed(right_split - right_m))
    right_low = sealed(right_m - right_high)
    product = sealed(left_m * right_m)
    error = sealed(sealed(left_high * right_high) - product)
    error = sealed(error + sealed(left_high * right_low))
    error = sealed(error + sealed(left_low * right_high))
    error = sealed(error + sealed(left_low * right_low))
    return (
        _normalize_scaled(mantissa=product, exponent=left_e + right_e),
        _normalize_scaled(mantissa=error, exponent=left_e + right_e),
    )


def _scaled_to_float(value: _ScaledFloat) -> FloatND:
    """Round only the final result, including subnormals, from its IEEE bits.

    Integer round-to-nearest/even avoids an intermediate subnormal float being
    flushed before its last power-of-two scaling on an accelerator.
    """
    mantissa, exponent = value
    info = jnp.finfo(mantissa.dtype)
    integer = jnp.dtype(f"uint{info.bits}")
    bits = jax.lax.bitcast_convert_type(mantissa, integer)
    sign = bits & integer.type(1 << (info.bits - 1))
    fraction = bits & integer.type(2**info.nmant - 1)
    significand = fraction | integer.type(1 << info.nmant)
    shift = jnp.clip(info.minexp + 1 - exponent, 1, info.nmant + 1).astype(integer)
    truncated = significand >> shift
    mask = (integer.type(1) << shift) - integer.type(1)
    lost = significand & mask
    halfway = integer.type(1) << (shift - integer.type(1))
    rounded = truncated + (
        (lost > halfway) | ((lost == halfway) & ((truncated & 1) != 0))
    ).astype(integer)
    normal = (
        (
            jnp.clip(exponent, info.minexp + 1, info.maxexp).astype(integer)
            + integer.type(info.maxexp - 2)
        )
        << info.nmant
    ) | fraction
    magnitude = jnp.where(exponent <= info.minexp, rounded, normal)
    infinity = integer.type((2 ** (info.bits - info.nmant - 1) - 1) << info.nmant)
    magnitude = jnp.where(exponent > info.maxexp, infinity, magnitude)
    magnitude = jnp.where(
        (exponent < info.minexp - info.nmant) | (mantissa == 0),
        integer.type(0),
        magnitude,
    )
    return jax.lax.bitcast_convert_type(sign | magnitude, mantissa.dtype)


def _scaled_density_ratio_words(
    *, left: _ScaledFloat, right: _ScaledFloat, sigma: _ScaledFloat, factor: float
) -> _ScaledFloat:
    """Retain a coefficient in scaled form until its directional product."""
    left_m, left_e = left
    right_m, right_e = right
    sigma_m, sigma_e = sigma
    return _normalize_scaled(
        mantissa=factor * left_m * right_m / sigma_m**2,
        exponent=left_e + right_e - 2 * sigma_e,
    )


def _scaled_linear_product(*, coefficient: _ScaledFloat, tangent: FloatND) -> FloatND:
    """Multiply a tangent by finite power-of-two factors before any overflow.

    A partial derivative may exceed maxfloat while its directional product is
    finite, or its tangent is zero. Never mint that infinite coefficient.
    Each step is linear in the tangent and uses a finite normal multiplier,
    so reverse transposition also keeps zero cotangents zero. Six steps cover
    the exponent range of gap*source/sigma**2 for either supported IEEE format.
    """
    mantissa, exponent = coefficient
    info = jnp.finfo(mantissa.dtype)
    integer = jnp.dtype(f"int{info.bits}")
    growing = exponent > 0
    multiplier = jnp.where(growing, 2 * mantissa, mantissa)
    remaining = jnp.where(mantissa != 0, exponent - growing.astype(exponent.dtype), 0)
    result = multiplier * tangent
    for _ in range(6):
        shift = jnp.clip(remaining, info.minexp, info.maxexp - 1)
        factor = jax.lax.bitcast_convert_type(
            (shift.astype(integer) + info.maxexp - 1) << info.nmant, mantissa.dtype
        )
        result = result * factor
        remaining = remaining - shift
    return result


def _density_contrasts_and_gap(
    operands: tuple[Float1D, FloatND, FloatND, FloatND, FloatND],
) -> tuple[Float1D, _ScaledFloat, _ScaledFloat, Int[jax.Array, ""]]:
    """Center on the exact nearest node and retain the complete affine residual."""
    nodes, location, rho, source, sigma = operands
    product_high, product_low = _scaled_product_words(left=rho, right=source)
    mean_terms = (_scaled_float(location), product_high, product_low)
    negative_mean_terms = tuple((-m, e) for m, e in mean_terms)
    node_m, node_e = _scaled_float(nodes)
    offsets = _scaled_sum(((node_m, node_e), *negative_mean_terms))
    # Stored subnormal nodes can compare as zero in accelerator arithmetic.
    # Compare normal expansion signs, and keep the selected identity as an index.
    above = offsets[0] >= 0
    first_above = jnp.where(jnp.any(above), jnp.argmax(above), len(nodes) - 1)
    upper = jnp.clip(first_above, 1, len(nodes) - 1)
    lower = upper - 1
    low_m, low_e = _scaled_float(nodes[lower])
    high_m, high_e = _scaled_float(nodes[upper])
    midpoint_residual = _scaled_sum(
        ((low_m, low_e - 1), (high_m, high_e - 1), *negative_mean_terms)
    )
    center_index = jnp.where(midpoint_residual[0] < 0, upper, lower)
    center = nodes[center_index]
    center_m, center_e = _scaled_float(center)
    half_gap = _scaled_sum(((node_m, node_e - 1), (-center_m, center_e - 1)))
    residual = _scaled_sum(
        ((node_m, node_e - 1), (center_m, center_e - 1), *negative_mean_terms)
    )
    contrast_words = _scaled_density_ratio_words(
        left=half_gap, right=residual, sigma=_scaled_float(sigma), factor=-2.0
    )
    is_reference = jnp.arange(len(nodes)) == center_index
    contrast_words = (
        jnp.where(is_reference, 0.0, contrast_words[0]),
        contrast_words[1],
    )
    contrasts = _scaled_to_float(contrast_words)
    return contrasts, half_gap, contrast_words, center_index


type _ScaledPair = tuple[_ScaledFloat, _ScaledFloat]


def _scaled_multiply(*, left: _ScaledFloat, right: _ScaledFloat) -> _ScaledFloat:
    """Multiply normal mantissas without materializing physical operands."""
    return _normalize_scaled(
        mantissa=sealed(left[0] * right[0]), exponent=left[1] + right[1]
    )


def _scaled_exp(
    logarithm: Float1D,
) -> tuple[_ScaledFloat, tuple[Float1D, IntND, Float1D, Bool1D]]:
    """Keep recoverable normal-tail weights below physical exp underflow.

    The exponent floor is below any recoverable first derivative: finite IEEE
    operands give score exponents below 8*maxexp; even a maxfloat direction
    cannot rescue a weight below 16*maxexp. The reduction uses split ln(2),
    so its rounding error scales with the small residual, not the log weight.
    """
    info = jnp.finfo(logarithm.dtype)
    floor = -16 * info.maxexp
    active = jnp.isfinite(logarithm) & (logarithm >= floor * np.log(2.0))
    value = jnp.where(active, logarithm, 0.0)
    power = jnp.floor(value / np.log(2.0)).astype(jnp.int32)
    if logarithm.dtype == jnp.float64:
        high, low = 0.6931471803691238, 1.9082149292705877e-10
    else:
        high, low = 0.693359375, -0.00021219444005469058
    residual = sealed(sealed(value - sealed(power * high)) - sealed(power * low))
    selected_exp = jax.lax.exp(residual, accuracy=jax.lax.Tolerance(ulps=2))
    mantissa, exponent = _normalize_scaled(mantissa=selected_exp, exponent=power)
    weights = jnp.where(active, mantissa, 0.0), jnp.where(active, exponent, -100_000)
    return weights, (residual, power, selected_exp, active)


def _scaled_accumulate(*, total: _ScaledPair, term: _ScaledFloat) -> _ScaledPair:
    """Compensated twofold sum in exponent-separated physical units.

    This is a bounded floating reduction, not an exact arbitrary-grid sum.
    It retains the leading rounding residual; its standard accumulation error
    is O(n*u**2*sum(abs(terms))), plus input formation and final rounding.
    """
    high, error = _scaled_two_sum(left=total[0], right=term)
    low = _scaled_sum((total[1], error))
    return _scaled_two_sum(left=high, right=low)


# keyword-only-exempt: library-callback=jax.lax.scan
def _scaled_scan_step(
    carry: _ScaledPair, term: _ScaledFloat
) -> tuple[_ScaledPair, _ScaledPair]:
    updated = _scaled_accumulate(total=carry, term=term)
    return updated, updated


def _scaled_other_sums(values: _ScaledFloat) -> _ScaledPair:
    """Pool every element except its own using linear-work prefix/suffix scans.

    Exclusion is structural: the almost-unit component never computes 1-p or
    subtracts its own dominant score from a rounded pooled moment.
    """
    zero = (jnp.zeros_like(values[0][0]), jnp.asarray(-100_000, dtype=jnp.int32))
    _, prefix = jax.lax.scan(_scaled_scan_step, (zero, zero), values)
    _, suffix = jax.lax.scan(_scaled_scan_step, (zero, zero), values, reverse=True)

    def exclusive(*, values: _ScaledPair, reverse: bool) -> _ScaledPair:
        def shift(word: _ScaledFloat) -> _ScaledFloat:
            mantissa, exponent = word
            zeros = jnp.zeros_like(mantissa[:1])
            powers = jnp.full_like(exponent[:1], -100_000)
            if reverse:
                return jnp.concatenate((mantissa[1:], zeros)), jnp.concatenate(
                    (exponent[1:], powers)
                )
            return jnp.concatenate((zeros, mantissa[:-1])), jnp.concatenate(
                (powers, exponent[:-1])
            )

        return shift(values[0]), shift(values[1])

    left = exclusive(values=prefix, reverse=False)
    right = exclusive(values=suffix, reverse=True)
    result = _scaled_accumulate(total=left, term=right[0])
    return _scaled_accumulate(total=result, term=right[1])


def _normalized_density_slope(
    *,
    weights: _ScaledFloat,
    total: _ScaledFloat,
    other_weights: _ScaledPair,
    scores: _ScaledFloat,
) -> _ScaledFloat:
    """Form the normalized Jacobian column before its physical-range boundary.

    For w_i=exp(l_i), Z=sum(w), relative score d_i, the exact identity is
    dp_i = w_i/Z**2 * [sum(j!=i,w_j)*d_i - sum(j!=i,w_j*d_j)].
    Every product and sum stays scaled until that final coefficient. In
    particular, w_i*d_i can be finite when d_i overflows or w_i underflows.
    """
    weighted = _scaled_multiply(left=weights, right=scores)
    other_scores = _scaled_other_sums(weighted)
    first = _scaled_multiply(left=other_weights[0], right=scores)
    second = _scaled_multiply(left=other_weights[1], right=scores)
    centered = _scaled_sum(
        (
            first,
            second,
            (-other_scores[0][0], other_scores[0][1]),
            (-other_scores[1][0], other_scores[1][1]),
        )
    )
    numerator = _scaled_multiply(left=weights, right=centered)
    return _normalize_scaled(
        mantissa=numerator[0] / total[0] ** 2,
        exponent=numerator[1] - 2 * total[1],
    )


def _normal_density_row_impl(
    operands: tuple[Float1D, FloatND, FloatND, FloatND, FloatND],
) -> Float1D:
    """Evaluate the preserved represented-input density row."""
    return jax.nn.softmax(_density_contrasts_and_gap(operands)[0])


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _normal_density_row_jvp(
    primals: tuple[tuple[Float1D, FloatND, FloatND, FloatND, FloatND]],
    tangents: tuple[tuple[Float1D, FloatND, FloatND, FloatND, FloatND]],
) -> tuple[Float1D, Float1D]:
    """Normalize scaled score moments before materializing row sensitivities.

    Nodes are fixed constructor data; the actual-source path remains active.
    Coefficients depend only on primals. The final finite-factor operations
    are linear in tangents and therefore preserve reverse transposition.
    """
    (operands,), (dots,) = primals, tangents
    nodes, _location, rho, source, sigma = operands
    _nodes_dot, location_dot, rho_dot, source_dot, sigma_dot = dots
    contrasts, half_gap, contrast_words, reference = _density_contrasts_and_gap(
        operands
    )
    weights, exp_metadata = _scaled_exp(contrasts)
    other_weights = _scaled_other_sums(weights)
    total = _scaled_sum((weights, *other_weights))
    scale = _scaled_float(sigma)
    unit = _scaled_float(jnp.ones_like(nodes))
    location_score = _scaled_density_ratio_words(
        left=half_gap, right=unit, sigma=scale, factor=2.0
    )
    contrast_m, contrast_e = contrast_words
    sigma_score = _normalize_scaled(
        mantissa=-2 * contrast_m / scale[0], exponent=contrast_e - scale[1]
    )
    location_slope = _normalized_density_slope(
        weights=weights, total=total, other_weights=other_weights, scores=location_score
    )
    sigma_slope = _normalized_density_slope(
        weights=weights, total=total, other_weights=other_weights, scores=sigma_score
    )
    rho_slope = _scaled_multiply(left=location_slope, right=_scaled_float(source))
    source_slope = _scaled_multiply(left=location_slope, right=_scaled_float(rho))
    mantissas = jnp.stack(
        (location_slope[0], rho_slope[0], source_slope[0], sigma_slope[0])
    )
    exponents = jnp.stack(
        (location_slope[1], rho_slope[1], source_slope[1], sigma_slope[1])
    )
    checked_m, checked_e = io_callback(
        _checked_density_coefficients,
        (
            jax.ShapeDtypeStruct(mantissas.shape, mantissas.dtype),
            jax.ShapeDtypeStruct(exponents.shape, exponents.dtype),
        ),
        *operands,
        reference,
        *exp_metadata,
        mantissas,
        exponents,
        ordered=False,
    )
    location_slope = checked_m[0], checked_e[0]
    rho_slope = checked_m[1], checked_e[1]
    source_slope = checked_m[2], checked_e[2]
    sigma_slope = checked_m[3], checked_e[3]
    tangent = (
        _scaled_linear_product(coefficient=location_slope, tangent=location_dot)
        + _scaled_linear_product(coefficient=rho_slope, tangent=rho_dot)
        + _scaled_linear_product(coefficient=source_slope, tangent=source_dot)
        + _scaled_linear_product(coefficient=sigma_slope, tangent=sigma_dot)
    )
    return jax.nn.softmax(contrasts), tangent


_normal_density_row = jax.custom_jvp(_normal_density_row_impl)
_normal_density_row.defjvp(_normal_density_row_jvp)


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
