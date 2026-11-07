"""Covariance factorization behind `lcm.processes.JointNormalIIDProcess`.

Correlated normal IID shocks live on a tensor of independent normal coordinates;
the helpers here classify and factor the covariance that maps them.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from _lcm.processes.base import sealed
from _lcm.typing import ProcessLawScalar
from lcm.exceptions import GridInitializationError
from lcm.typing import BoolND, Float1D, Float2D, FloatND

type NormalMean = tuple[ProcessLawScalar, ...] | Float1D
type NormalCovariance = tuple[tuple[ProcessLawScalar, ...], ...] | Float2D
type HostNormalArray = Float[np.ndarray, "..."]


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
