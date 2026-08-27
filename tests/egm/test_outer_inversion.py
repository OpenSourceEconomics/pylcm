"""One inverse, shared by solve and simulation, admitted without a tolerance.

N-NB-EGM retains post-decision targets and recovers the outer action that reaches
one. Both phases must recover it the same way and admit it on the same predicate,
or a candidate the solve certified can be replayed as a different action.

Admissibility is two-tier, and deliberately not uniform. Every candidate's image
must lie inside the outer state's declared domain, because a stock outside it has
no value function. A candidate whose target *is* a declared endpoint must in
addition reproduce that endpoint bit for bit: the endpoint is where an ULP of
error leaves the domain, and it is the only place where exactness is both
necessary and achievable. Requiring bit-exactness at interior nodes would reject
most of them for an error that never leaves the domain.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.outer_inversion import (
    coefficient_is_exactly_invertible,
    outer_candidate_is_admissible,
    recover_outer_action,
)


@pytest.mark.parametrize(
    ("coefficient", "invertible"),
    [
        pytest.param(Fraction(1), True, id="unit"),
        pytest.param(Fraction(2), True, id="two"),
        pytest.param(Fraction(1, 2), True, id="half"),
        pytest.param(Fraction(4), True, id="four"),
        pytest.param(Fraction(1, 8), True, id="eighth"),
        pytest.param(Fraction(-2), True, id="negative-two"),
        pytest.param(Fraction(3), False, id="three"),
        pytest.param(Fraction(6, 5), False, id="six-fifths"),
        pytest.param(Fraction(0), False, id="zero"),
    ],
)
def test_only_a_power_of_two_coefficient_inverts_exactly(coefficient, invertible):
    """Dividing by a power of two is exact in binary; dividing by three is not.

    The coefficient decides what the inverse may attempt, and nothing more. A
    dyadic factor is necessary structural support, not sufficient pointwise
    certification: the runtime predicate still decides each candidate. A
    non-dyadic factor rounds the quotient however exactly the factor is known,
    so it is refused at declaration rather than left to fail per candidate.
    """
    assert coefficient_is_exactly_invertible(coefficient) is invertible


def test_the_unit_inverse_elides_its_division() -> None:
    """At unit slope the action is a subtraction, with no division to round."""
    target = jnp.asarray([0.0, 20.0, 7.5], dtype=jnp.float32)
    at_zero = jnp.asarray([3.8263209, 1.37, 2.25], dtype=jnp.float32)

    recovered = recover_outer_action(
        target=target, at_zero=at_zero, coefficient=Fraction(1)
    )

    np.testing.assert_array_equal(np.asarray(recovered), np.asarray(target - at_zero))


def test_the_recovered_action_reaches_the_endpoint_the_probe_route_missed() -> None:
    """The measured witness inverts exactly once the slope is not estimated.

    State `3.8263208866119385` at target `0.0` is the case the two-point probe
    got wrong: it recovered an action whose image was `-7.15e-07`, below the
    declared floor, and published it.
    """
    at_zero = jnp.asarray(3.8263208866119385, dtype=jnp.float32)
    target = jnp.asarray(0.0, dtype=jnp.float32)

    action = recover_outer_action(
        target=target, at_zero=at_zero, coefficient=Fraction(1)
    )

    assert float(at_zero + action) == 0.0


def test_a_candidate_landing_outside_the_domain_is_inadmissible() -> None:
    """An image below the floor or above the ceiling is refused, endpoint or not."""
    low, high = jnp.float32(0.0), jnp.float32(20.0)
    image = jnp.asarray([-7.152557e-07, 20.000002, 10.0], dtype=jnp.float32)
    target = jnp.asarray([0.0, 20.0, 10.0], dtype=jnp.float32)

    admissible = outer_candidate_is_admissible(
        image=image, target=target, low=low, high=high
    )

    np.testing.assert_array_equal(
        np.asarray(admissible), np.array([False, False, True])
    )


def test_an_endpoint_target_must_reproduce_its_endpoint_bit_for_bit() -> None:
    """An in-domain image at an endpoint target is still refused unless exact.

    The image below is inside `[0, 20]`, so containment alone would admit it. It
    is not the endpoint, so the durable stock it advances to is not the node the
    solve ranked.
    """
    low, high = jnp.float32(0.0), jnp.float32(20.0)
    # The magnitude the estimated-slope route actually produced, reflected to the
    # admissible side of the floor so containment alone would let it through.
    image = jnp.asarray([7.152557e-07], dtype=jnp.float32)
    target = jnp.asarray([0.0], dtype=jnp.float32)

    assert bool((image >= low)[0])
    assert bool((image <= high)[0])
    assert not bool(
        outer_candidate_is_admissible(image=image, target=target, low=low, high=high)[0]
    )


def test_a_subnormal_image_at_a_zero_endpoint_is_indistinguishable_from_it() -> None:
    """Exactness at a zero endpoint is only as fine as the backend's subnormals.

    XLA flushes subnormals, so the smallest representable positive float compares
    equal to zero and is admitted. The tolerance-free predicate therefore resolves
    the floor to the smallest normal magnitude, not to the last subnormal. This is
    a property of the arithmetic rather than of the predicate, and it is pinned
    here so a backend that stops flushing is noticed rather than assumed.
    """
    low, high = jnp.float32(0.0), jnp.float32(20.0)
    subnormal = jnp.asarray([1.4e-45], dtype=jnp.float32)
    target = jnp.asarray([0.0], dtype=jnp.float32)

    assert bool((subnormal == target)[0])
    assert bool(
        outer_candidate_is_admissible(
            image=subnormal, target=target, low=low, high=high
        )[0]
    )


def test_an_interior_target_is_admitted_on_containment_not_bit_exactness() -> None:
    """A represented interior candidate is an inverse-produced pair, not an identity.

    An interior node's image can be an ULP off without leaving the domain. It is
    admitted, and the difference is an approximation diagnostic -- deliberately
    *not* a claim that the replayed action reproduces its nominal target exactly.
    Only endpoints carry that identity, because only there does the difference
    leave the declared domain.
    """
    low, high = jnp.float32(0.0), jnp.float32(20.0)
    target = jnp.asarray([7.5], dtype=jnp.float32)
    image = jnp.nextafter(target, jnp.asarray([jnp.inf], dtype=jnp.float32))

    assert image[0] != target[0]
    assert bool(
        outer_candidate_is_admissible(image=image, target=target, low=low, high=high)[0]
    )


def test_a_non_finite_image_is_inadmissible() -> None:
    """NaN and infinity are refused rather than compared."""
    low, high = jnp.float32(0.0), jnp.float32(20.0)
    image = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float32)
    target = jnp.asarray([0.0, 20.0, 10.0], dtype=jnp.float32)

    admissible = outer_candidate_is_admissible(
        image=image, target=target, low=low, high=high
    )

    assert not bool(jnp.any(admissible))
