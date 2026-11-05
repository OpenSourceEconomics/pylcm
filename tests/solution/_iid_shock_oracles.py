"""Literal references over declared finite primitives, independent of engine code.

These references import only the standard library. The finite model fixes two wealth
states, three IID nodes and two actions. Every continuation reads the next period;
selection precedes weighted storage, and collective channels use one common action.
They certify witness arithmetic, not a production lowering or a continuous optimum.
"""

import math
from fractions import Fraction
from itertools import accumulate, pairwise
from typing import Literal

type Scalar = Fraction | float
type FiniteValues = dict[tuple[int, int], tuple[Scalar, ...]]
type FinitePolicies = dict[tuple[int, int, int], int]


def finite_backward(
    *, mode: Literal["hard", "ev1", "collective"], periods: int = 3
) -> tuple[FiniteValues, FinitePolicies]:
    """Enumerate Q=wealth+2a+z(2a-1), wealth'=a, beta=1/2.

    The collective second channel is wealth+2(1-a)-z(2a-1), with equal Pareto
    weights. EV1 has unit scale and publishes the location-normalized logsum.
    Policies hold modal action IDs, with the lowest ID resolving exact ties.
    """
    values: FiniteValues = {}
    policies: FinitePolicies = {}
    weights = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
    channels = 2 if mode == "collective" else 1
    for period in reversed(range(periods)):
        for wealth in (0, 1):
            integrated: list[Scalar] = [Fraction(0) for _ in range(channels)]
            for node, (shock, weight) in enumerate(
                zip((-1, 0, 1), weights, strict=True)
            ):
                candidates = []
                for action in (0, 1):
                    flow = (
                        wealth + 2 * action + shock * (2 * action - 1),
                        wealth + 2 * (1 - action) - shock * (2 * action - 1),
                    )
                    continuation = values.get(
                        (period + 1, action), (Fraction(0),) * channels
                    )
                    candidates.append(
                        tuple(
                            Fraction(flow[stakeholder])
                            + Fraction(1, 2) * continuation[stakeholder]
                            for stakeholder in range(channels)
                        )
                    )
                objectives = [sum(candidate) for candidate in candidates]
                choice = max(range(2), key=objectives.__getitem__)
                policies[period, wealth, node] = choice
                if mode == "ev1":
                    maximum = max(objectives)
                    selected = (
                        float(maximum)
                        + math.log(
                            sum(math.exp(value - maximum) for value in objectives)
                        ),
                    )
                else:
                    selected = candidates[choice]
                for stakeholder in range(channels):
                    integrated[stakeholder] += weight * selected[stakeholder]
            values[period, wealth] = tuple(integrated)
    return values, policies


def categorical_cells(
    probabilities: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, tuple[int, ...]], ...]:
    """Refine the union of exact CDF cutpoints under one common uniform."""
    if not probabilities:
        raise ValueError("probabilities must contain a law")
    for law in probabilities:
        _validate_probabilities(law)
    breakpoints = sorted(
        {Fraction(0), Fraction(1)}
        | {point for law in probabilities for point in accumulate(law)}
    )
    return tuple(
        (
            right - left,
            tuple(
                categorical_draw(probabilities=law, uniform=left)
                for law in probabilities
            ),
        )
        for left, right in pairwise(breakpoints)
    )


def categorical_draw(*, probabilities: tuple[Fraction, ...], uniform: Fraction) -> int:
    """Select the half-open CDF cell, skipping all zero-mass categories."""
    _validate_probabilities(probabilities)
    if not 0 <= uniform < 1:
        raise ValueError("uniform must lie in [0, 1)")
    for category, cutpoint in enumerate(accumulate(probabilities)):
        if uniform < cutpoint:
            return category
    raise AssertionError("normalized probabilities must contain the uniform")


def signed_power_statistic(*, q: Fraction, high: Fraction) -> Fraction:
    """Return the exact exponent-minus-two statistic for weights (-q, 1+q)."""
    if q < 0 or high <= 1:
        raise ValueError("q must be nonnegative and high must exceed one")
    return -q + (1 + q) * high**-2


def resources_integral(*, wealth: Fraction) -> tuple[Fraction, Fraction]:
    """Integrate the row V(m)=m² and its economic marginal at m=wealth+z."""
    nodes = (Fraction(0), Fraction(2))
    weights = (Fraction(1, 4), Fraction(3, 4))
    return (
        sum(
            (
                weight * (wealth + node) ** 2
                for weight, node in zip(weights, nodes, strict=True)
            ),
            Fraction(0),
        ),
        sum(
            (
                weight * 2 * (wealth + node)
                for weight, node in zip(weights, nodes, strict=True)
            ),
            Fraction(0),
        ),
    )


def log_savings_candidate(*, resources: Fraction) -> Fraction:
    """Solve the equal-weight log Euler equation at savings resources/2."""
    if resources <= 0:
        raise ValueError("resources must be positive")
    return resources / 2


def log_savings_value(
    *, resources: Fraction, consumption: Fraction, shock: float = 0.0
) -> float:
    """Evaluate half-log consumption plus half-log saving plus a utility shock."""
    if not 0 < consumption < resources:
        raise ValueError("consumption must lie strictly between zero and resources")
    return math.log(consumption) / 2 + math.log(resources - consumption) / 2 + shock


def weighted_value_and_derivative(
    *,
    values: tuple[Scalar, ...],
    marginals: tuple[Scalar, ...],
    weights: tuple[Fraction, ...],
    weight_derivatives: tuple[Fraction, ...],
) -> tuple[Scalar, Scalar]:
    """Differentiate the complete weighted sum, including changing probabilities."""
    value: Scalar = Fraction(0)
    derivative: Scalar = Fraction(0)
    for node_value, marginal, weight, weight_derivative in zip(
        values, marginals, weights, weight_derivatives, strict=True
    ):
        if weight:
            value += weight * node_value
            derivative += weight * marginal
        if weight_derivative:
            derivative += weight_derivative * node_value
    return value, derivative


def base_moment(*, measure: Literal["normal", "uniform"], degree: int) -> Fraction:
    """Return exact standard-normal or unit-uniform raw moments."""
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if measure == "uniform":
        return Fraction(1, degree + 1)
    return Fraction(0) if degree % 2 else Fraction(math.prod(range(1, degree, 2)))


def _validate_probabilities(probabilities: tuple[Fraction, ...]) -> None:
    if (
        not probabilities
        or any(probability < 0 for probability in probabilities)
        or sum(probabilities) != 1
    ):
        raise ValueError("probabilities must be nonnegative and sum to one")
