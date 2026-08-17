"""Regression matrix for probability weights near the normal/subnormal boundary.

A represented zero is the only unconditional null event. A nonzero subnormal may
be omitted only when doing so leaves the rounded result of the consuming
aggregate unchanged. The original largest-subnormal fixtures below are exact
omission controls for their particular finite values; the round-11 additions
exercise mantissas and continuations for which replacing the weight by zero or
by ``finfo.tiny`` changes the answer.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from _lcm.zero_safe import joint_weight, zero_safe_weighted_term
from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.typing import FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    common: ScalarInt
    rare: ScalarInt


def _active_dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _largest_subnormal() -> ScalarFloat:
    """The largest probability the dtype cannot hold as a normal number."""
    dtype = _active_dtype()
    tiny = np.asarray(np.finfo(dtype).tiny, dtype=dtype)
    return jnp.asarray(np.nextafter(tiny, np.asarray(0.0, dtype=dtype)), dtype=dtype)


def _smallest_subnormal() -> ScalarFloat:
    """The smallest strictly positive bit pattern of the active dtype."""
    dtype = _active_dtype()
    return jnp.asarray(
        np.nextafter(np.asarray(0.0, dtype=dtype), np.asarray(1.0, dtype=dtype)),
        dtype=dtype,
    )


def _smallest_normal() -> ScalarFloat:
    """The smallest probability the dtype holds as a normal number."""
    return jnp.asarray(np.finfo(_active_dtype()).tiny, dtype=_active_dtype())


def _certain() -> ScalarFloat:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _impossible() -> ScalarFloat:
    return jnp.asarray(0.0, dtype=_active_dtype())


def _no_utility() -> ScalarFloat:
    return jnp.asarray(0.0, dtype=_active_dtype())


def _common_payoff() -> ScalarFloat:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _finite_rare_payoff(shock: ScalarFloat) -> FloatND:
    """A payoff differing from the common target's, read at every shock node."""
    return jnp.asarray(5.0, dtype=_active_dtype()) + 0.0 * shock


def _infeasible_rare_payoff(shock: ScalarFloat) -> FloatND:
    """The ordinary value of a state at which no action is feasible."""
    return jnp.asarray(-jnp.inf, dtype=_active_dtype()) + 0.0 * shock


def _largest_finite_rare_payoff(shock: ScalarFloat) -> FloatND:
    """The largest finite value that still leaves room to average over the nodes.

    `finfo.max` itself cannot be used: the target's own weighted average over its
    process nodes has no headroom left there and overflows to `+inf` before any
    weight rule is reached, so that value measures the aggregation rather than
    the weight. Halving it restores the headroom and changes nothing else — the
    contribution of a subnormal weight against it is still the largest a finite
    continuation can produce.
    """
    largest_with_headroom = np.finfo(_active_dtype()).max / 2.0
    return jnp.asarray(largest_with_headroom, dtype=_active_dtype()) + 0.0 * shock


def _smallest_normal_rare_payoff(shock: ScalarFloat) -> FloatND:
    """A finite positive value for the nonlinear harmonic-mean boundary."""
    return (
        jnp.asarray(np.finfo(_active_dtype()).tiny, dtype=_active_dtype()) + 0.0 * shock
    )


def _model(
    rare_probability: Callable[[], ScalarFloat],
    *,
    rare_payoff: Callable[..., FloatND] = _finite_rare_payoff,
    certainty_equivalent: CertaintyEquivalent | None = None,
    rare_carries_a_process: bool = True,
) -> Model:
    """A source choosing between a common target and a rare one.

    The rare target carries a target-only IID process, so the witness runs
    through the feature under audit; `rare_carries_a_process=False` removes it
    to cover the route with no stochastic node to multiply against.
    """
    rare_states = (
        {"shock": NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)}
        if rare_carries_a_process
        else {}
    )
    rare_utility = rare_payoff if rare_carries_a_process else _common_payoff
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "common": MarkovTransition(_certain),
                    "rare": MarkovTransition(rare_probability),
                },
                active=lambda age: age < 21,
                functions={"utility": _no_utility},
                certainty_equivalent=certainty_equivalent,
            ),
            "common": Regime(transition=None, functions={"utility": _common_payoff}),
            "rare": Regime(
                transition=None,
                states=rare_states,
                functions={"utility": rare_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=_RegimeId,
    )


_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}


def _source_value(model: Model, log_level: LogLevel = "off") -> FloatND:
    return jnp.asarray(model.solve(params=_PARAMS, log_level=log_level)[0]["source"])


def test_a_subnormal_weight_on_a_finite_value_drops() -> None:
    """For this payoff, the true rare term rounds away and omission is exact."""
    np.testing.assert_allclose(
        np.asarray(_source_value(_model(_largest_subnormal))), 1.0
    )


def test_a_subnormal_weight_drops_for_a_target_without_stochastic_nodes() -> None:
    """The same exact-omission control holds without a stochastic target axis."""
    model = _model(_largest_subnormal, rare_carries_a_process=False)
    np.testing.assert_allclose(np.asarray(_source_value(model)), 1.0)


def test_a_subnormal_weight_drops_under_a_nonlinear_aggregator() -> None:
    """For this PowerMean fixture, the true rare branch also rounds away."""
    model = _model(_largest_subnormal, certainty_equivalent=PowerMean())
    params = {
        "source": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": 2.0},
        }
    }
    value = jnp.asarray(model.solve(params=params, log_level="off")[0]["source"])

    np.testing.assert_allclose(np.asarray(value), 1.0)


def test_a_subnormal_weight_on_an_infinite_value_keeps_the_infinity() -> None:
    """A reachable state at which no action is feasible is worth `-inf`."""
    model = _model(_largest_subnormal, rare_payoff=_infeasible_rare_payoff)

    assert bool(jnp.all(jnp.isneginf(_source_value(model))))


def test_a_normal_weight_on_an_infinite_value_keeps_the_infinity() -> None:
    """The infinite branch is a rule about weights, not about small weights."""
    model = _model(_smallest_normal, rare_payoff=_infeasible_rare_payoff)

    assert bool(jnp.all(jnp.isneginf(_source_value(model))))


def test_a_zero_weight_on_an_infinite_value_contributes_nothing() -> None:
    """A target that cannot be reached is not made infinite by what stands at it."""
    model = _model(_impossible, rare_payoff=_infeasible_rare_payoff)

    np.testing.assert_allclose(np.asarray(_source_value(model)), 1.0)


def test_a_smallest_normal_weight_is_priced() -> None:
    """One representable step above the subnormal range, nothing is special."""
    assert bool(jnp.all(jnp.isfinite(_source_value(_model(_smallest_normal)))))


def test_a_zero_weight_leaves_the_common_target_alone() -> None:
    """A target that cannot be reached contributes nothing, and poisons nothing."""
    np.testing.assert_allclose(np.asarray(_source_value(_model(_impossible))), 1.0)


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_weighted_term_classifies_by_bits_not_by_comparison(
    *, compile_it: bool
) -> None:
    """Zero-ness is read from the bits, so the verdict does not depend on the backend.

    A subnormal weight compares equal to zero where the backend flushes it and
    does not where it represents it. Against an infinite value the two would
    otherwise disagree: one drops the node, the other keeps the infinity.
    """
    dtype = _active_dtype()
    subnormal = _largest_subnormal()
    func = (
        jax.jit(zero_safe_weighted_term, static_argnames="subnormal_is_accounted_for")
        if compile_it
        else zero_safe_weighted_term
    )
    negative_infinity = jnp.asarray(-jnp.inf, dtype=dtype)

    def term(weight: ScalarFloat, value: ScalarFloat) -> FloatND:
        return func(weight=weight, value=value, subnormal_is_accounted_for=False)

    assert bool(jnp.isneginf(term(subnormal, negative_infinity)))
    assert float(term(jnp.asarray(0.0, dtype=dtype), negative_infinity)) == 0.0
    np.testing.assert_allclose(
        float(term(subnormal, jnp.asarray(5.0, dtype=dtype))), 0.0, atol=1e-30
    )


# A subnormal's mantissa is part of the model. A largest-subnormal fixture alone
# cannot distinguish preserving the weight from replacing every subnormal by the
# smallest normal magnitude, so the cases below vary the mantissa itself.


def test_the_smallest_subnormal_is_not_enlarged_in_a_linear_continuation() -> None:
    """The rarest event prices its target at no more than its true share.

    Promoting the weight to `tiny` would add about four units here. Omitting it
    costs at most `tiny * |V|`, which is the declared accepted approximation, so
    the assertion is one-sided: the answer must not exceed the exact one.

    The upper bound carries one representable step at the answer's magnitude.
    `exact` is computed in long double while the engine publishes at the active
    precision, so a backend that prices the node returns the nearest
    representable value to the exact answer — which lies above it half the time.
    That step is seven orders of magnitude below the overstatement this test
    exists to catch, so it cannot hide one.
    """
    dtype = _active_dtype()
    p = np.longdouble(np.asarray(_smallest_subnormal(), dtype=dtype))
    value = np.longdouble(np.finfo(dtype).max / 2.0)
    exact = (np.longdouble(1.0) + p * value) / (np.longdouble(1.0) + p)

    got = np.longdouble(
        np.asarray(
            _source_value(
                _model(_smallest_subnormal, rare_payoff=_largest_finite_rare_payoff)
            )
        )
    )

    rounding = np.longdouble(np.spacing(np.asarray(got, dtype=dtype)))
    assert got <= exact + rounding
    assert got >= exact - np.longdouble(np.finfo(dtype).tiny) * value


def test_the_smallest_subnormal_keeps_its_true_nonlinear_weight() -> None:
    """Finite PowerMean values distinguish true min-subnormal from `tiny`."""
    dtype = _active_dtype()
    p = np.longdouble(np.asarray(_smallest_subnormal(), dtype=dtype))
    tiny = np.longdouble(np.finfo(dtype).tiny)
    expected = (np.longdouble(1.0) + p) / (np.longdouble(1.0) + p / tiny)
    model = _model(
        _smallest_subnormal,
        rare_payoff=_smallest_normal_rare_payoff,
        certainty_equivalent=PowerMean(),
    )
    params = {
        "source": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": 2.0},
        }
    }

    got = jnp.asarray(model.solve(params=params, log_level="off")[0]["source"])

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected, dtype=dtype),
        rtol=5e-6 if dtype == np.dtype("float32") else 1e-13,
        atol=0.0,
    )


def test_a_negative_smallest_subnormal_does_not_bypass_the_distribution_guard() -> None:
    """Raw mass/minimum arithmetic must not see `-minsub` as represented `-0`."""
    model = _model(
        lambda: -_smallest_subnormal(), rare_payoff=_largest_finite_rare_payoff
    )
    value = _source_value(model, log_level="off")

    assert bool(jnp.all(jnp.isnan(value)))


def test_a_subnormal_joint_product_is_never_larger_than_its_true_size() -> None:
    """A product the backend cannot form stays nonzero and never overstates the node.

    Retaining the product's own mantissa would need subnormal arithmetic the
    backend does not perform: it flushes a subnormal result to zero, so the true
    value is unavailable at the point the product is formed. The substitute is
    the smallest representable magnitude, which no underflowed product can fall
    below — the error is one-sided by construction.
    """
    dtype = _active_dtype()
    if dtype == np.dtype("float32"):
        factor, n_factors = np.asarray(2.0**-22, dtype=dtype), 6
    else:
        factor, n_factors = np.asarray(2.0**-52, dtype=dtype), 20
    factors = jnp.full((n_factors,), factor, dtype=jnp.asarray(0.0).dtype)
    exact = np.longdouble(factor) ** n_factors

    got = np.longdouble(np.asarray(joint_weight(factors), dtype=dtype))

    assert got > 0.0
    assert got <= exact
