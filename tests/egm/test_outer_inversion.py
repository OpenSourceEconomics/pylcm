"""One inverse, shared by solve and simulation, admitted without a tolerance.

N-NB-EGM retains post-decision targets and recovers the outer action that reaches
one. Both phases must recover it the same way and admit it on the same predicate,
or a candidate the solve certified can be replayed as a different action.

Admissibility is two-tier, and deliberately not uniform. Every candidate's image
must lie inside its relevant domain. A candidate whose target *is* a domain
endpoint must in addition reproduce that endpoint bit for bit: the endpoint is
where an ULP of error leaves the domain. Requiring target-local reproduction at
interior nodes would reject ordinary candidate banks that remain inside it.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.outer_inversion import (
    abstract_like,
    certify_declared_outer_inverse,
    coefficient_is_exactly_invertible,
    invert_declared_outer_target,
    outer_candidate_is_admissible,
    recover_outer_action,
)
from lcm.exceptions import RegimeInitializationError


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
def test_only_a_power_of_two_coefficient_inverts_exactly(*, coefficient, invertible):
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
    """An interior represented candidate is an inverse-produced pair.

    A round trip may differ from the nominal target by more than its local ULP
    while remaining inside the represented domain. Interior admission therefore
    preserves containment; exact identity remains an endpoint requirement.
    """
    low, high = jnp.float32(0.0), jnp.float32(20.0)
    target = jnp.asarray([7.5], dtype=jnp.float32)
    one_step = jnp.nextafter(target, jnp.asarray([jnp.inf], dtype=jnp.float32))
    image = jnp.nextafter(one_step, jnp.asarray([jnp.inf], dtype=jnp.float32))

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


def _targets(**overrides):
    """Build a post-decision DAG returning the outer target by name."""
    coefficient = overrides.get("coefficient", 1.0)

    def func(*, illiquid, illiquid_investment):
        return {"new_illiquid": illiquid + coefficient * illiquid_investment}

    return func


def _certify(*, func, domain=(0.0, 20.0)):
    return certify_declared_outer_inverse(
        func=func,
        arg_names=("illiquid", "illiquid_investment"),
        abstract_args=abstract_like((jnp.float32(1.0), jnp.float32(0.0))),
        outer_action_name="illiquid_investment",
        outer_post_decision_name="new_illiquid",
        outer_state_domain=domain,
        regime_name="working",
    )


def test_the_certified_inverse_carries_the_coefficient_and_the_domain() -> None:
    """A unit affine map certifies coefficient one over the declared domain."""
    inverse = _certify(func=_targets())

    assert inverse.coefficient == Fraction(1)
    assert (inverse.low, inverse.high) == (0.0, 20.0)


def test_a_non_dyadic_coefficient_is_refused_where_it_is_declared() -> None:
    """A map moving three units of stock per unit of action is refused at build.

    Dividing by three rounds, so the recovered action reaches a stock away from
    the ranked node. The refusal names the regime, the coefficient, and how to
    declare a map that can be inverted.
    """
    with pytest.raises(RegimeInitializationError) as refusal:
        _certify(func=_targets(coefficient=3.0))

    assert "working" in str(refusal.value)
    assert "3" in str(refusal.value)
    assert "illiquid_investment" in str(refusal.value)


def test_a_map_ignoring_the_outer_action_is_refused_as_uninvertible() -> None:
    """A constant map retains nothing about the action that reached it."""
    with pytest.raises(RegimeInitializationError):
        _certify(func=lambda illiquid, illiquid_investment: {"new_illiquid": illiquid})  # noqa: ARG005


def test_a_non_affine_map_is_refused_rather_than_approximated() -> None:
    """A squared action has no single coefficient, so the map is refused."""

    def squared(*, illiquid, illiquid_investment):
        return {"new_illiquid": illiquid + illiquid_investment**2}

    with pytest.raises(RegimeInitializationError):
        _certify(func=squared)


def test_inverting_a_retained_endpoint_publishes_the_action_and_its_image() -> None:
    """The inversion returns the action, the stock it reaches, and its admission.

    At the measured witness state the recovered action reaches the floor exactly,
    so the candidate is admitted and its image is the endpoint itself.
    """
    inverse = _certify(func=_targets())
    at_zero = jnp.asarray([3.8263208866119385], dtype=jnp.float32)
    target = jnp.asarray([0.0], dtype=jnp.float32)

    inversion = invert_declared_outer_target(
        inverse=inverse,
        target=target,
        at_zero=at_zero,
        forward=lambda action: at_zero + action,
    )

    assert float(inversion.action[0]) == -3.8263208866119385
    assert float(inversion.image[0]) == 0.0
    assert bool(inversion.admissible[0])


def test_an_inversion_landing_off_the_domain_is_dropped_not_published() -> None:
    """A candidate whose image leaves the declared domain is refused pointwise.

    The forward map below overshoots the floor by the magnitude the estimated
    slope route produced. Nothing raises: the candidate is dropped and the rest
    of the bank is unaffected.
    """
    inverse = _certify(func=_targets())
    at_zero = jnp.asarray([3.8263209, 5.0], dtype=jnp.float32)
    target = jnp.asarray([0.0, 10.0], dtype=jnp.float32)
    overshoot = jnp.asarray([-7.152557e-07, 0.0], dtype=jnp.float32)

    inversion = invert_declared_outer_target(
        inverse=inverse,
        target=target,
        at_zero=at_zero,
        forward=lambda action: at_zero + action + overshoot,
    )

    np.testing.assert_array_equal(
        np.asarray(inversion.admissible), np.array([False, True])
    )


def _endpoint_failures(*, states, endpoint, route):
    """Count states whose recovered action misses `endpoint` under `route`.

    `route` maps `(at_zero, target)` to the recovered outer action, so the two
    inversions can be compared on exactly the same states and the same map.
    """
    target = jnp.full_like(states, endpoint)
    image = states + route(states, target)
    return int(jnp.sum(image != target))


def _certified_route(*, at_zero, target):
    return recover_outer_action(target=target, at_zero=at_zero, coefficient=Fraction(1))


def _secant_route(*, at_zero, target):
    """Recover the action by dividing through a slope read from two evaluations."""
    one = jnp.ones_like(at_zero)
    slope = (at_zero + one) - at_zero
    return (target - at_zero) / slope


@pytest.mark.parametrize("endpoint", [0.0, 20.0], ids=["floor", "ceiling"])
def test_no_float32_state_misses_an_endpoint_under_the_certified_inverse(
    endpoint,
) -> None:
    """For `new = old + action`, every state reaches either endpoint exactly.

    The sweep covers the toy's declared `[0.5, 20]` range at float32. The
    estimated-slope route is run over the identical states as a positive
    control: it misses the endpoint for thousands of them, which is what shows
    the sweep is able to detect a miss at all rather than reporting zero
    because it looks in the wrong place.
    """
    states = jnp.linspace(0.5, 20.0, 200_001, dtype=jnp.float32)
    assert states.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(states)))

    certified = _endpoint_failures(
        states=states, endpoint=endpoint, route=_certified_route
    )
    secant = _endpoint_failures(states=states, endpoint=endpoint, route=_secant_route)

    assert secant > 0, "control did not fire: the sweep cannot detect a missed endpoint"
    assert certified == 0


@pytest.mark.parametrize("endpoint", [0.0, 20.0], ids=["floor", "ceiling"])
def test_every_certified_endpoint_recovery_is_admitted(endpoint) -> None:
    """Reaching the endpoint exactly is what the admission predicate asks for.

    The sweep above establishes bit-exactness; this establishes that the
    predicate consuming it admits the whole sweep, so no state is dropped for
    an inversion that in fact succeeded.
    """
    states = jnp.linspace(0.5, 20.0, 20_001, dtype=jnp.float32)
    target = jnp.full_like(states, endpoint)
    action = _certified_route(at_zero=states, target=target)

    admissible = outer_candidate_is_admissible(
        image=states + action,
        target=target,
        low=jnp.float32(0.0),
        high=jnp.float32(20.0),
    )

    assert int(jnp.sum(~admissible)) == 0
