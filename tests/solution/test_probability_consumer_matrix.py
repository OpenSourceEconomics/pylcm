"""No consumer of a probability disagrees with another about what it is.

The engine asks three questions of every weight — is it the null event, is it
a valid probability, how much does it contribute — and the answers have to be
the same in the mass guard, in the node-neutralizing mask, in the linear
expectation, in `QuasiArithmeticMean`, in `PowerMean` and its pair form, and
in an `aggregate` a user wrote. A probability below the dtype's normal range
is where they can disagree: arithmetic reports it as zero, its bits do not.

The disagreements are not small. Under a nonlinear certainty equivalent the
transform of a near-zero value is large, so a rare node's contribution is of
order one and dropping it moves both the value and the discrete choice. A
node dropped as impossible loses the `-inf` it carried. A negative
probability read as `-0` passes a guard meant to refuse it.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import LinearExpectation, PowerMean, QuasiArithmeticMean
from _lcm.power_mean import weighted_power_mean, weighted_power_mean_of_pair
from _lcm.probability import is_negative
from _lcm.regime_building.Q_and_F import (
    _regime_mass_is_a_distribution,
    _values_without_impossible_nodes,
)


def _dtype() -> np.dtype:
    return np.dtype(jnp.zeros(()).dtype)


def _largest_subnormal() -> Any:
    dtype = _dtype()
    return np.nextafter(dtype.type(np.finfo(dtype).tiny), dtype.type(0.0), dtype=dtype)


def _smallest_subnormal() -> Any:
    dtype = _dtype()
    return np.nextafter(dtype.type(0.0), dtype.type(1.0), dtype=dtype)


def _log_domain_rtol() -> float:
    """The accuracy an anchored log-domain mean can have at this precision.

    The mean is recovered as `exp(a + log M / p)`, so an error of one ulp in
    the log costs `|a|` relative ulps in the result. These lotteries anchor at
    `log(tiny)`, which is the widest anchor the dtype admits — about 87 nats in
    single precision and 708 in double — and the tolerance follows from that
    rather than from a number chosen to pass.
    """
    dtype = _dtype()
    return 4.0 * float(np.finfo(dtype).eps) * abs(float(np.log(np.finfo(dtype).tiny)))


_RARE_WEIGHTS = [
    pytest.param(_smallest_subnormal, id="smallest_subnormal"),
    pytest.param(_largest_subnormal, id="largest_subnormal"),
]


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_a_positive_subnormal_remains_live_in_a_power_mean(*, compile_it: bool) -> None:
    """Omitting the rare node changes this finite harmonic mean by about one half."""
    dtype = _dtype()
    tiny = dtype.type(np.finfo(dtype).tiny)
    rare_weight = _largest_subnormal()
    values = jnp.asarray([1.0, tiny], dtype=dtype)
    weights = jnp.asarray([1.0, rare_weight], dtype=dtype)
    fn: Callable = jax.jit(weighted_power_mean) if compile_it else weighted_power_mean

    got = fn(values=values, weights=weights, exponent=jnp.asarray(-1.0, dtype=dtype))
    exact = (np.longdouble(1) + np.longdouble(rare_weight)) / (
        np.longdouble(1) + np.longdouble(rare_weight) / np.longdouble(tiny)
    )

    np.testing.assert_allclose(
        np.asarray(got), float(exact), rtol=_log_domain_rtol(), atol=0
    )


def test_a_positive_subnormal_on_minus_infinity_is_not_donor_replaced() -> None:
    """A represented nonzero event carrying -inf makes the continuation -inf."""
    dtype = _dtype()
    values = jnp.asarray([1.0, -jnp.inf], dtype=dtype)
    weights = jnp.asarray([1.0, _largest_subnormal()], dtype=dtype)

    got = _values_without_impossible_nodes(values=values, weights=weights)

    assert bool(jnp.isneginf(got[1]))


def test_a_negative_subnormal_fails_the_regime_distribution_guard() -> None:
    """A negative nonzero bit pattern is invalid even when arithmetic sees -0.

    The sign is read off each target's own probability, because a `jnp.minimum`
    over the targets would already have turned it into `-0`.
    """
    dtype = _dtype()
    negative = jnp.asarray(np.negative(_smallest_subnormal()), dtype=dtype)
    mass = jnp.asarray(1.0, dtype=dtype) + negative

    assert not bool(_regime_mass_is_a_distribution(mass, is_negative(negative)))


def test_a_lottery_of_valid_probabilities_passes_the_guard() -> None:
    """The guard refuses a sign, not a size: a subnormal probability is valid."""
    dtype = _dtype()
    rare = jnp.asarray(_smallest_subnormal(), dtype=dtype)
    mass = jnp.asarray(1.0, dtype=dtype) + rare

    assert bool(_regime_mass_is_a_distribution(mass, is_negative(rare)))


@pytest.mark.parametrize("rare_weight", _RARE_WEIGHTS)
def test_a_zero_weight_node_is_still_donor_replaced(rare_weight) -> None:
    """Neutralizing a node that cannot occur is what the mask is for."""
    dtype = _dtype()
    values = jnp.asarray([1.0, jnp.nan, -jnp.inf], dtype=dtype)
    weights = jnp.asarray([1.0, 0.0, rare_weight()], dtype=dtype)

    got = _values_without_impossible_nodes(values=values, weights=weights)

    assert float(np.asarray(got)[1]) == 1.0
    assert bool(jnp.isneginf(got[2]))


@pytest.mark.parametrize("rare_weight", _RARE_WEIGHTS)
@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_a_subnormal_weight_on_an_infinity_keeps_it_in_a_linear_expectation(
    rare_weight, *, compile_it: bool
) -> None:
    """A reachable state where no action is feasible is worth `-inf`."""
    dtype = _dtype()
    values = jnp.asarray([1.0, -jnp.inf], dtype=dtype)
    weights = jnp.asarray([1.0, rare_weight()], dtype=dtype)
    expectation = LinearExpectation()
    fn: Callable = (
        jax.jit(expectation.aggregate) if compile_it else expectation.aggregate
    )

    got = fn(values=values, weights=weights, params={})

    assert bool(jnp.isneginf(got))


@pytest.mark.parametrize("rare_weight", _RARE_WEIGHTS)
def test_a_subnormal_weight_on_a_finite_value_is_bounded_in_a_linear_expectation(
    rare_weight,
) -> None:
    """The linear route's rare node moves the mean by at most its own share."""
    dtype = _dtype()
    largest_finite = np.finfo(dtype).max / dtype.type(2.0)
    values = jnp.asarray([1.0, largest_finite], dtype=dtype)
    weights = jnp.asarray([1.0, rare_weight()], dtype=dtype)

    got = np.longdouble(
        np.asarray(
            LinearExpectation().aggregate(values=values, weights=weights, params={})
        )
    )
    exact = (
        np.longdouble(1.0)
        + np.longdouble(rare_weight()) * np.longdouble(largest_finite)
    ) / (np.longdouble(1.0) + np.longdouble(rare_weight()))

    np.testing.assert_allclose(float(got), float(exact), rtol=1e-6, atol=0)


@pytest.mark.parametrize("rare_weight", _RARE_WEIGHTS)
@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_a_subnormal_weight_is_priced_under_a_power_mean_certainty_equivalent(
    rare_weight, *, compile_it: bool
) -> None:
    """`PowerMean` prices the rare node at the share the transform gives it."""
    dtype = _dtype()
    tiny = dtype.type(np.finfo(dtype).tiny)
    rare = rare_weight()
    values = jnp.asarray([1.0, tiny], dtype=dtype)
    weights = jnp.asarray([1.0, rare], dtype=dtype)
    equivalent = PowerMean()
    params = {"risk_aversion": jnp.asarray(2.0, dtype=dtype)}
    fn: Callable = jax.jit(equivalent.aggregate) if compile_it else equivalent.aggregate

    got = fn(values=values, weights=weights, params=params)
    exact = (np.longdouble(1) + np.longdouble(rare)) / (
        np.longdouble(1) + np.longdouble(rare) / np.longdouble(tiny)
    )

    np.testing.assert_allclose(
        np.asarray(got), float(exact), rtol=_log_domain_rtol(), atol=0
    )


def test_a_subnormal_weight_does_not_reverse_a_discrete_choice() -> None:
    """The lottery is worth less than the safe alternative, and is not chosen."""
    dtype = _dtype()
    tiny = dtype.type(np.finfo(dtype).tiny)
    lottery = PowerMean().aggregate(
        values=jnp.asarray([1.0, tiny], dtype=dtype),
        weights=jnp.asarray([1.0, _largest_subnormal()], dtype=dtype),
        params={"risk_aversion": jnp.asarray(2.0, dtype=dtype)},
    )

    assert float(np.asarray(lottery)) < 0.75


def test_a_subnormal_weight_is_priced_under_a_user_written_transform() -> None:
    """A `QuasiArithmeticMean` a user assembled gets weights it can multiply."""
    dtype = _dtype()
    tiny = dtype.type(np.finfo(dtype).tiny)
    rare = _largest_subnormal()
    equivalent = QuasiArithmeticMean(
        transform=lambda value: 1.0 / value,
        inverse=lambda value: 1.0 / value,
    )

    got = equivalent.aggregate(
        values=jnp.asarray([1.0, tiny], dtype=dtype),
        weights=jnp.asarray([1.0, rare], dtype=dtype),
        params={},
    )
    exact = (np.longdouble(1) + np.longdouble(rare)) / (
        np.longdouble(1) + np.longdouble(rare) / np.longdouble(tiny)
    )

    np.testing.assert_allclose(
        np.asarray(got), float(exact), rtol=_log_domain_rtol(), atol=0
    )


@pytest.mark.parametrize("rare_weight", _RARE_WEIGHTS)
def test_a_subnormal_weight_is_priced_in_the_pair_power_mean(rare_weight) -> None:
    """The two-node form prices a rare branch as the general form does."""
    dtype = _dtype()
    tiny = dtype.type(np.finfo(dtype).tiny)
    rare = rare_weight()

    got = weighted_power_mean_of_pair(
        first=jnp.asarray(1.0, dtype=dtype),
        second=jnp.asarray(tiny, dtype=dtype),
        first_weight=jnp.asarray(1.0, dtype=dtype),
        second_weight=jnp.asarray(rare, dtype=dtype),
        exponent=jnp.asarray(-1.0, dtype=dtype),
    )
    general = weighted_power_mean(
        values=jnp.asarray([1.0, tiny], dtype=dtype),
        weights=jnp.asarray([1.0, rare], dtype=dtype),
        exponent=jnp.asarray(-1.0, dtype=dtype),
    )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(general), rtol=_log_domain_rtol(), atol=0
    )


@pytest.mark.parametrize(
    "aggregate",
    [
        pytest.param(
            lambda values, weights: PowerMean().aggregate(
                values=values,
                weights=weights,
                params={"risk_aversion": jnp.asarray(2.0, dtype=_dtype())},
            ),
            id="power_mean",
        ),
        pytest.param(
            lambda values, weights: QuasiArithmeticMean(
                transform=lambda value: value, inverse=lambda value: value
            ).aggregate(values=values, weights=weights, params={}),
            id="quasi_arithmetic",
        ),
    ],
)
def test_a_negative_subnormal_weight_is_not_read_as_a_dead_node(aggregate) -> None:
    """Negative mass reaches a transforming aggregate rather than dropping out.

    The linear expectation states no such rule — it multiplies whatever it is
    handed, and the engine refuses a signed transition before it gets there.
    """
    dtype = _dtype()
    values = jnp.asarray([1.0, 2.0], dtype=dtype)
    weights = jnp.asarray([1.0, np.negative(_smallest_subnormal())], dtype=dtype)

    assert bool(jnp.isnan(jnp.asarray(aggregate(values, weights))))


@pytest.mark.parametrize(
    "aggregate",
    [
        pytest.param(
            lambda values, weights: LinearExpectation().aggregate(
                values=values, weights=weights, params={}
            ),
            id="linear",
        ),
        pytest.param(
            lambda values, weights: PowerMean().aggregate(
                values=values,
                weights=weights,
                params={"risk_aversion": jnp.asarray(2.0, dtype=_dtype())},
            ),
            id="power_mean",
        ),
    ],
)
def test_a_nan_weight_stays_poison(aggregate) -> None:
    """A weight that is not a probability is not laundered into a dead node."""
    dtype = _dtype()
    values = jnp.asarray([1.0, 2.0], dtype=dtype)
    weights = jnp.asarray([1.0, np.nan], dtype=dtype)

    assert bool(jnp.isnan(jnp.asarray(aggregate(values, weights))))
