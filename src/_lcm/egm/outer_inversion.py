"""Recover and admit the outer action, identically in solve and simulation.

N-NB-EGM searches the outer margin over post-decision targets and retains those,
so both phases have to recover the action that reaches a retained target. They
must recover it the same way: a solve that certifies a candidate and a simulation
that replays a different action for it disagree about which stock was chosen.

The recovery reads the map's coefficient from its structure rather than measuring
it (see `outer_affine_structure`), then divides only when the coefficient is not
one. Admission is tolerance-free and two-tier:

- every candidate's image must lie inside the outer state's declared domain,
  because a stock outside it has no value function; and
- a candidate whose target *is* a declared endpoint must reproduce that endpoint
  bit for bit, because the endpoint is where an ULP of error leaves the domain.

Interior candidates are admitted on containment alone. An interior image can sit
an ULP from its node without leaving the domain, and requiring exactness there
would drop most interior candidates for an error the value function never sees.

That choice fixes what a represented candidate *means*, and the weaker meaning is
the one in force here. A represented interior candidate is an inverse-produced
`(action, image)` pair, not a claim that the action reproduces its nominal target
bit for bit; an interior round-trip difference is an approximation diagnostic
rather than a broken exact-replay promise. Only endpoints carry the exact
identity, because only there does the difference leave the declared domain.
"""

from fractions import Fraction

import jax.numpy as jnp

from lcm.typing import BoolND, FloatND, ScalarFloat

__all__ = [
    "coefficient_is_exactly_invertible",
    "outer_candidate_is_admissible",
    "recover_outer_action",
]


def coefficient_is_exactly_invertible(coefficient: Fraction) -> bool:
    """Return whether dividing by `coefficient` is exact in binary floating point.

    A power of two scales the exponent and leaves the significand alone, so the
    quotient is exact. Any other factor rounds it however exactly the factor
    itself is known -- measured, a coefficient of three leaves hundreds of
    thousands of states off their endpoint even with a perfectly declared slope.
    Such a map is refused where it is declared rather than left to drop
    candidates one at a time.

    This is necessary structural support, never sufficient pointwise
    certification. Exact division by a power of two does not make
    `target - offset` followed by the forward addition exact for every
    representable offset -- on a signed domain it is not -- so the runtime
    endpoint and containment predicate stays active for a dyadic coefficient
    exactly as it does for any other.
    """
    if coefficient == 0:
        return False
    magnitude = abs(coefficient)
    numerator, denominator = magnitude.numerator, magnitude.denominator
    return _is_power_of_two(numerator) and _is_power_of_two(denominator)


def recover_outer_action(
    *, target: FloatND, at_zero: FloatND, coefficient: Fraction
) -> FloatND:
    """Return the outer action whose post-decision image is `target`.

    `at_zero` is the map evaluated at a zero outer action, so the affine map is
    `image = at_zero + coefficient * action`. At unit coefficient the inverse is
    a subtraction and no division is performed at all, which is what removes the
    rounding the estimated-slope route introduced.
    """
    difference = target - at_zero
    if coefficient == 1:
        return difference
    divisor = jnp.asarray(
        coefficient.numerator / coefficient.denominator, dtype=difference.dtype
    )
    return difference / divisor


def outer_candidate_is_admissible(
    *,
    image: FloatND,
    target: FloatND,
    low: ScalarFloat,
    high: ScalarFloat,
) -> BoolND:
    """Return which candidates may be published, deciding without a tolerance.

    Args:
        image: The declared map re-evaluated at the recovered action.
        target: The post-decision target the solve retained.
        low: The outer state's declared lower bound.
        high: The outer state's declared upper bound.

    Returns:
        A boolean array over candidates. Every comparison is exact, so no scale
        constant appears anywhere on the acceptance path and there is nothing an
        out-of-domain image can pass by being small enough.
    """
    finite = jnp.isfinite(image) & jnp.isfinite(target)
    inside = (image >= low) & (image <= high)
    at_endpoint = (target == low) | (target == high)
    reproduces_endpoint = image == target
    return finite & inside & (~at_endpoint | reproduces_endpoint)


def _is_power_of_two(value: int) -> bool:
    """Return whether a positive integer is a power of two."""
    return value > 0 and value & (value - 1) == 0
