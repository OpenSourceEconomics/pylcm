"""Independent scalar laws for represented supplied-node AR(1) inputs.

Only the Python standard library is used. Fraction arithmetic forms absolute
normal log densities and lottery targets exactly from the supplied binary floats.
Decimal evaluates exponentials and normalizes the density law at 120 digits.
These references do not reuse a production row builder, nearest-node centering,
searchsorted, scatter, or floating-point subtraction of physical endpoints.
"""

from collections.abc import Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise
from typing import Literal


def _fraction(value: float) -> Fraction:
    """Preserve the exact binary value, including a float32 promoted to float64."""
    return Fraction(float(value))


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _normal_log_densities(
    *, nodes: Sequence[float], source: float, rho: float, sigma: float, mu: float
) -> tuple[Fraction, ...]:
    mean = _fraction(mu) + _fraction(rho) * _fraction(source)
    scale = _fraction(sigma)
    if scale <= 0:
        raise ValueError("The innovation scale must be positive.")
    # The common normalizing factor cancels from the discrete probability law.
    return tuple(-((_fraction(x) - mean) ** 2) / (2 * scale**2) for x in nodes)


def _density_weights(logs: Sequence[Fraction]) -> tuple[Decimal, ...]:
    largest = max(logs)
    masses = tuple(_decimal(value - largest).exp() for value in logs)
    total = sum(masses, Decimal(0))
    return tuple(value / total for value in masses)


def density_row(
    *, nodes: Sequence[float], source: float, rho: float, sigma: float, mu: float
) -> tuple[float, ...]:
    """Normalize exact absolute normal log densities at the represented nodes."""
    logs = _normal_log_densities(
        nodes=nodes, source=source, rho=rho, sigma=sigma, mu=mu
    )
    with localcontext() as context:
        context.prec = 120
        return tuple(float(weight) for weight in _density_weights(logs))


def density_matrix(
    *, nodes: Sequence[float], rho: float, sigma: float, mu: float
) -> tuple[tuple[float, ...], ...]:
    """Evaluate the same conditional law at each physical source node."""
    return tuple(
        density_row(nodes=nodes, source=source, rho=rho, sigma=sigma, mu=mu)
        for source in nodes
    )


def density_derivative(
    *,
    nodes: Sequence[float],
    source: float,
    rho: float,
    sigma: float,
    mu: float,
    parameter: Literal["mu", "rho", "sigma"],
) -> tuple[float, ...]:
    """Differentiate the normalized law while holding its physical nodes fixed."""
    logs = _normal_log_densities(
        nodes=nodes, source=source, rho=rho, sigma=sigma, mu=mu
    )
    mean = _fraction(mu) + _fraction(rho) * _fraction(source)
    scale = _fraction(sigma)
    distances = tuple(_fraction(x) - mean for x in nodes)
    if parameter == "sigma":
        scores = tuple(distance**2 / scale**3 for distance in distances)
    elif parameter == "mu":
        scores = tuple(distance / scale**2 for distance in distances)
    elif parameter == "rho":
        scores = tuple(
            _fraction(source) * distance / scale**2 for distance in distances
        )
    else:
        raise ValueError(f"Unknown derivative parameter: {parameter}")
    # A common score cancels analytically. Remove it with exact rational arithmetic
    # before converting, so an enormous common score cannot erase its differences.
    offsets = tuple(score - scores[0] for score in scores)
    with localcontext() as context:
        context.prec = 120
        weights = _density_weights(logs)
        decimal_scores = tuple(_decimal(offset) for offset in offsets)
        weighted_score = sum(
            (
                weight * score
                for weight, score in zip(weights, decimal_scores, strict=True)
            ),
            Decimal(0),
        )
        return tuple(
            float(weight * (score - weighted_score))
            for weight, score in zip(weights, decimal_scores, strict=True)
        )


def quadrature_row(
    *,
    nodes: Sequence[float],
    source: float,
    rho: float,
    sigma: float,
    mu: float,
    innovations: Sequence[float],
    relative_weights: Sequence[float],
) -> tuple[float, ...]:
    """Clip each exact innovation target and allocate its mass by scalar loops."""
    points = tuple(_fraction(x) for x in nodes)
    weights = tuple(_fraction(value) for value in relative_weights)
    if len(points) < 2 or any(a >= b for a, b in pairwise(points)):
        raise ValueError("At least two strictly increasing nodes are required.")
    if len(innovations) != len(weights) or not weights:
        raise ValueError(
            "Innovation nodes and weights must have the same positive length."
        )
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be nonnegative and have positive total mass.")
    mean = _fraction(mu) + _fraction(rho) * _fraction(source)
    scale = _fraction(sigma)
    if scale <= 0:
        raise ValueError("The innovation scale must be positive.")
    total = sum(weights)
    row = [Fraction(0) for _ in points]
    for innovation, relative_weight in zip(innovations, weights, strict=True):
        target = mean + scale * _fraction(innovation)
        target = min(max(target, points[0]), points[-1])
        for lower, (left, right) in enumerate(pairwise(points)):
            if target <= right:
                upper_share = (target - left) / (right - left)
                mass = relative_weight / total
                row[lower] += mass * (1 - upper_share)
                row[lower + 1] += mass * upper_share
                break
    return tuple(float(value) for value in row)


def quadrature_matrix(
    *,
    nodes: Sequence[float],
    rho: float,
    sigma: float,
    mu: float,
    innovations: Sequence[float],
    relative_weights: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Evaluate the exact clipped lottery at each physical source node."""
    return tuple(
        quadrature_row(
            nodes=nodes,
            source=source,
            rho=rho,
            sigma=sigma,
            mu=mu,
            innovations=innovations,
            relative_weights=relative_weights,
        )
        for source in nodes
    )
