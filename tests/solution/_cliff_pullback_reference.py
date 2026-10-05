"""Exact reference for save-to-cliff targets of a draw-conditioned child.

Standard library only: it imports neither pylcm, JAX nor NumPy, and shares no code
with the solver. Geometry is exact rational arithmetic; the two-period value is a
binary64 calculation of a closed form.

The contract. The source saves `s`; the next kind `j` is drawn; the child's liquid
state is `slope * s + offsets[j]`; child kind `j` loses a lump-sum subsidy at its
own liquid cutoff `cutoffs[j]`. A savings level just below the preimage
`(cutoffs[j] - offsets[j]) / slope` keeps the subsidy at node `j`, one just above
loses it, so each node contributes the preimage of its *own* child row's cutoff.
"""

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction


def child_cliff_preimages(
    *,
    cutoffs: Sequence[Fraction],
    slope: Fraction,
    offsets: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    """Return, per drawn child kind `j`, the savings landing on that kind's cutoff.

    Args:
        cutoffs: Child liquid cutoff of each child kind.
        slope: Positive savings slope of the liquid law.
        offsets: Liquid-law intercept of each drawn child kind.

    Returns:
        One exact preimage per child kind, in kind order.

    """
    if slope <= 0:
        msg = f"The liquid law must increase in savings; got slope {slope}."
        raise ValueError(msg)
    return tuple(
        (cutoff - offset) / slope
        for cutoff, offset in zip(cutoffs, offsets, strict=True)
    )


def two_period_log_value(
    *,
    current_resources: float,
    cutoffs: tuple[float, float],
    offsets: tuple[float, float],
    base_income: float,
    subsidy: float,
    discount_factor: float,
) -> float:
    """Return the supremum of the two-period log-utility objective over savings.

    The objective is
    `log(R - s) + beta / 2 * sum_j log(s + offsets[j] + base_income
    + subsidy * 1[s + offsets[j] < cutoffs[j]])`. Between consecutive preimages it
    is smooth and strictly concave, so each interval's supremum is its stationary
    point or a one-sided endpoint limit, found by derivative bisection.

    Args:
        current_resources: Cash-on-hand `R` of the source period.
        cutoffs: Child liquid cutoff of each child kind.
        offsets: Liquid-law intercept of each drawn child kind (unit slope).
        base_income: Child income added to the child's liquid state.
        subsidy: Lump-sum subsidy below the child cutoff.
        discount_factor: The discount factor `beta`.

    Returns:
        The supremum of the objective; it is an endpoint limit when a cliff binds.

    """
    preimages = sorted(c - o for c, o in zip(cutoffs, offsets, strict=True))
    edges = [0.0, *(p for p in preimages if 0.0 < p < current_resources)]
    edges.append(current_resources)

    def below_flags(midpoint: float) -> tuple[bool, ...]:
        return tuple(midpoint + o < c for c, o in zip(cutoffs, offsets, strict=True))

    def derivative(*, s: float, below: tuple[bool, ...]) -> float:
        return -1.0 / (current_resources - s) + 0.5 * discount_factor * sum(
            1.0 / (s + o + base_income + (subsidy if b else 0.0))
            for o, b in zip(offsets, below, strict=True)
        )

    def objective(*, s: float, below: tuple[bool, ...]) -> float:
        return math.log(current_resources - s) + 0.5 * discount_factor * sum(
            math.log(s + o + base_income + (subsidy if b else 0.0))
            for o, b in zip(offsets, below, strict=True)
        )

    best = -math.inf
    for lower, upper in itertools.pairwise(edges):
        below = below_flags(0.5 * (lower + upper))
        top = min(upper, current_resources * (1.0 - 1e-12))
        if derivative(s=lower, below=below) <= 0.0:
            optimum = lower
        elif derivative(s=top, below=below) >= 0.0:
            optimum = top
        else:
            low, high = lower, top
            for _ in range(200):
                middle = 0.5 * (low + high)
                if derivative(s=middle, below=below) > 0.0:
                    low = middle
                else:
                    high = middle
            optimum = 0.5 * (low + high)
        best = max(best, objective(s=optimum, below=below))
    return best
