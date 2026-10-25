"""OR2: independent exact-input two-node density sensitivity oracle.

Only the Python standard library is used. Absolute log densities and score
differences are exact Fractions. Decimal evaluates the two-node normalization
and its derivative without subtracting two rounded, nearly equal scores.
This is a new oracle extension; the historical OR1 is not changed.

Run this file directly to print the full sign/neighborhood/scale witness matrix.
This calculation is not execution of pylcm, JAX, or a model of its implementation.
"""

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import NamedTuple


class Case(NamedTuple):
    precision: int
    neighbor: int
    scale_exponent: int
    sign: int
    nodes: tuple[float, float]
    rho: float
    mu: float
    sigma: float
    source_index: int


def make_case(*, precision: int, neighbor: int, scale_exponent: int, sign: int) -> Case:
    """Generalize RT2's residual (1-k*epsilon)*(1+k*epsilon)-1."""
    if precision not in (32, 64) or sign not in (-1, 1):
        raise ValueError("Use precision 32 or 64 and sign -1 or 1.")
    epsilon = math.ldexp(1.0, -23 if precision == 32 else -52)
    scale = math.ldexp(1.0, scale_exponent)
    radius = (1.0 + neighbor * epsilon) * scale
    return Case(
        precision=precision,
        neighbor=neighbor,
        scale_exponent=scale_exponent,
        sign=sign,
        nodes=(-radius, radius),
        rho=1.0 - neighbor * epsilon,
        mu=sign * scale,
        sigma=epsilon * scale,
        source_index=0 if sign > 0 else 1,
    )


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _exact_law(*, case: Case, parameter: str) -> tuple[Fraction, Fraction, Fraction]:
    points = tuple(Fraction(value) for value in case.nodes)
    source = points[case.source_index]
    mean = Fraction(case.mu) + Fraction(case.rho) * source
    scale = Fraction(case.sigma)
    distances = tuple(point - mean for point in points)
    logs = tuple(-(distance**2) / (2 * scale**2) for distance in distances)
    if parameter == "mu":
        scores = tuple(distance / scale**2 for distance in distances)
    elif parameter == "rho":
        scores = tuple(source * distance / scale**2 for distance in distances)
    elif parameter == "source":
        scores = tuple(
            Fraction(case.rho) * distance / scale**2 for distance in distances
        )
    elif parameter == "sigma":
        scores = tuple(distance**2 / scale**3 for distance in distances)
    else:
        raise ValueError("Unknown parameter: " + parameter)
    return mean, logs[1] - logs[0], scores[1] - scores[0]


def evaluate_case(
    *, case: Case, parameter: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return probabilities and their analytic normalized-law derivative.

    If t=exp(-abs(log f1-log f0)), then p0*p1=t/(1+t)^2. Keeping
    this product avoids loss of the derivative of the almost-unit component.
    In particular, p0*p1 is never computed by subtracting 1 from a rounded p.
    """
    _mean, log_odds, score_difference = _exact_law(case=case, parameter=parameter)
    with localcontext() as context:
        context.prec = 120
        tail = (-abs(_decimal(log_odds))).exp()
        total = Decimal(1) + tail
        small, large = tail / total, Decimal(1) / total
        row = (large, small) if log_odds <= 0 else (small, large)
        upper_derivative = tail / total**2 * _decimal(score_difference)
        return (
            (float(row[0]), float(row[1])),
            (-float(upper_derivative), float(upper_derivative)),
        )


def density_pairwise_derivative(
    *,
    nodes: tuple[float, ...],
    source: float,
    rho: float,
    mu: float,
    sigma: float,
    parameter: str,
) -> tuple[float, ...]:
    """Exact score differences and Decimal pair fluxes, without pooled centering."""
    points = tuple(Fraction(value) for value in nodes)
    mean = Fraction(mu) + Fraction(rho) * Fraction(source)
    scale = Fraction(sigma)
    distances = tuple(point - mean for point in points)
    logs = tuple(-(distance**2) / (2 * scale**2) for distance in distances)
    if parameter == "sigma":
        scores = tuple(distance**2 / scale**3 for distance in distances)
    else:
        factor = (
            Fraction(source)
            if parameter == "rho"
            else Fraction(rho)
            if parameter == "source"
            else Fraction(1)
        )
        scores = tuple(factor * distance / scale**2 for distance in distances)
    with localcontext() as context:
        context.prec = 120
        weights = tuple(_decimal(value - max(logs)).exp() for value in logs)
        total = sum(weights, Decimal(0))
        return tuple(
            float(
                weight
                * sum(
                    (
                        other_weight * _decimal(score - other_score)
                        for other_weight, other_score in zip(
                            weights, scores, strict=True
                        )
                    ),
                    Decimal(0),
                )
                / total**2
            )
            for weight, score in zip(weights, scores, strict=True)
        )
