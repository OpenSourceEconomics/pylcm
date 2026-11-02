"""Correlated normal IID shocks on a tensor of independent normal coordinates."""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Float

from _lcm.beartype_conf import GRID_CONF
from _lcm.processes.base import sealed, standard_normal
from _lcm.processes.iid import FOLD_REMOVED, NormalIIDProcess
from _lcm.typing import PRNGKeyND
from lcm.exceptions import GridInitializationError
from lcm.typing import BoolND, Float1D, Float2D, FloatND, StateName

type NormalMean = tuple[float | int, ...] | Float1D
type NormalCovariance = tuple[tuple[float | int, ...], ...] | Float2D
type HostNormalArray = Float[np.ndarray, "..."]


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
    n_std: float | int | None = None
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


def covariance_factor(
    *, cov: FloatND, zero_pivots: tuple[bool, ...] | None = None
) -> Float2D:
    """Return a lower-triangular factor of an already validated PSD covariance.

    Elimination runs on the diagonal congruence `C = D^{-1} cov D^{-1}`, where the
    power-of-two scales `D` place every positive diagonal entry of `C` in `[1, 4)`
    and a zero diagonal keeps scale one. Power-of-two scaling is exact, and every
    entry of the PSD matrix `C` is below 4 in magnitude, so no product of two
    entries overflows and a product underflows only where it is negligible against
    the residual it updates. The factor of `cov` is `D` times the factor of `C`.

    A zero pivot contributes no remaining column. `zero_pivots`, the exact pattern
    of a fixed covariance, marks pivots that are zero although roundoff leaves a
    residual: their columns stay exactly zero. Clipping diagonal residuals only
    removes arithmetic roundoff after the exact host-side PSD validation.
    """
    cov = cov.astype(jnp.result_type(cov, 0.0))
    diagonal = jnp.diagonal(cov)
    _, exponent = jnp.frexp(diagonal)
    # A diagonal m * 2**e with m in [1/2, 1) has 4**shift <= diagonal < 4**(shift+1).
    shift = jnp.where(diagonal > 0, (exponent - 1) // 2, 0)
    residual = jnp.ldexp(cov, -(shift[:, None] + shift[None, :]))
    factor = jnp.zeros_like(cov)
    for column in range(cov.shape[0]):
        if zero_pivots is not None and zero_pivots[column]:
            continue
        pivot = jnp.maximum(residual[column, column], 0.0)
        denominator = jnp.where(pivot > 0, pivot, 1.0)
        for row in range(column, cov.shape[0]):
            value = jnp.where(
                pivot > 0, residual[row, column] / jnp.sqrt(denominator), 0.0
            )
            factor = factor.at[row, column].set(value)
        # Eliminate with the unrooted pivot: sqrt rounding must not create noise
        # in a perfectly correlated coordinate.
        for row in range(column + 1, cov.shape[0]):
            for other in range(column + 1, cov.shape[0]):
                correction = sealed(
                    sealed(residual[row, column] * residual[other, column])
                    / denominator
                )
                residual = residual.at[row, other].set(
                    sealed(residual[row, other] - correction)
                )
    return jnp.ldexp(factor, shift[:, None])


# Host consumers share the forward map's compiled factor, and so its rank.
compiled_covariance_factor = jax.jit(covariance_factor, static_argnames="zero_pivots")


def admitted_correlation_factor(
    *, correlation: Float2D, zero_pivots: BoolND
) -> tuple[Float2D, BoolND]:
    """Factor a unit-diagonal matrix with exact host-owned support as data.

    Host admission proves PSD for this invocation. On that fixed support stratum,
    active pivots still need resolved finite positive arithmetic. A rounded zero
    residual is not evidence of exact zero support. Safe operands precede each
    root and division, including inactive columns, so singular support has finite
    tangents along its admitted stratum.
    """
    residual = correlation
    factor = jnp.zeros_like(correlation)
    valid = jnp.all(jnp.isfinite(correlation))
    for column in range(correlation.shape[0]):
        zero = zero_pivots[column]
        pivot = residual[column, column]
        resolved = jnp.isfinite(pivot) & (pivot > 0)
        valid &= zero | resolved
        safe_pivot = jnp.where(zero | ~resolved, 1.0, pivot)
        for row in range(column, correlation.shape[0]):
            value = jnp.where(zero, 0.0, residual[row, column] / jnp.sqrt(safe_pivot))
            factor = factor.at[row, column].set(value)
        for row in range(column + 1, correlation.shape[0]):
            for other in range(column + 1, correlation.shape[0]):
                correction = jnp.where(
                    zero,
                    0.0,
                    sealed(
                        sealed(residual[row, column] * residual[other, column])
                        / safe_pivot
                    ),
                )
                residual = residual.at[row, other].set(
                    sealed(residual[row, other] - correction)
                )
    return factor, valid & jnp.all(jnp.isfinite(factor))


def _rounded_zero_pivots(
    *, cov: NormalCovariance, dtype: type[np.floating]
) -> tuple[bool, ...] | None:
    """Return the exact zero pivots of `cov` rounded to `dtype`, if it stays PSD."""
    with np.errstate(over="ignore"):
        values = np.asarray(cov, dtype=dtype)
    try:
        return _exact_zero_pivots(cov=values, size=len(cov))
    except GridInitializationError:
        return None


def _exact_zero_pivots(*, cov: HostNormalArray, size: int) -> tuple[bool, ...]:
    """Classify finite binary floats by exact LDL elimination, rejecting non-PSD."""
    if (
        cov.shape != (size, size)
        or not np.isfinite(cov).all()
        or not np.array_equal(cov, cov.T)
    ):
        raise GridInitializationError(
            "cov must be finite, symmetric and square in states order."
        )
    residual = [[Fraction(float(value)) for value in row] for row in cov]
    zero = []
    for column in range(size):
        pivot = residual[column][column]
        if pivot < 0 or (
            pivot == 0
            and any(residual[row][column] != 0 for row in range(column + 1, size))
        ):
            raise GridInitializationError("cov must be positive semidefinite.")
        zero.append(pivot == 0)
        if pivot == 0:
            continue
        for row in range(column + 1, size):
            for other in range(column + 1, size):
                residual[row][other] -= (
                    residual[row][column] * residual[other][column] / pivot
                )
    return tuple(zero)
