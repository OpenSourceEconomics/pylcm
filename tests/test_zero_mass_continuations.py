"""Continuation values treat an exactly-zero probability as a null event.

`-inf` is the ordinary value of a state at which every action is infeasible, and
an exactly-zero weight is equally ordinary — a Markov row with a zero entry, a
binned process with an empty tail bin, a regime that cannot be reached from
here. Where the two meet, the node contributes nothing: an event of probability
zero carries no information about the expectation. Naive floating-point
arithmetic disagrees, because `0.0 * -inf` is NaN rather than zero, and that NaN
then destroys every well-specified node beside it.

A positive probability is a different matter entirely: `-inf` there is the
answer, and nothing below may hide it. NaN weights stay poison, and negative
weights stay invalid rather than being quietly absorbed.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.Q_and_F import (
    _expectation_over_stochastic_nodes,
    _scalar_target_contribution,
)
from lcm.typing import Float1D, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


@dataclass(frozen=True, kw_only=True)
class _InheritedLinearExpectation(LinearExpectation):
    """A user subclass, which the engine routes through the joint lottery.

    `Q_and_F` selects the per-target route on the exact type, so a subclass
    reaches `aggregate` even when it overrides nothing — that is the point of
    testing against this class rather than `LinearExpectation` itself.
    """


def _float(value: float) -> ScalarFloat:
    """Return `value` at the precision the suite is running at."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return jnp.asarray(value, dtype=dtype)


def _array(values: list[float]) -> Float1D:
    """Return `values` at the precision the suite is running at."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return jnp.asarray(values, dtype=dtype)


def _scalar_lottery(
    *, live_V: float, dead_V: float, live_prob: float, dead_prob: float
) -> tuple[Float1D, Float1D, ScalarFloat]:
    """Collect two stateless targets into one joint lottery.

    Returns:
        Tuple of the flattened lottery values, their weights, and the
        probability mass the two targets represent.

    """
    _, values, weights, mass = _scalar_target_contribution(
        scalar_targets=("live", "dead"),
        next_regime_to_V_arr={"live": _float(live_V), "dead": _float(dead_V)},
        active_regime_probs={"live": _float(live_prob), "dead": _float(dead_prob)},
        as_lottery=True,
        zero=_float(0.0),
    )
    return jnp.concatenate(values), jnp.concatenate(weights), mass


def test_zero_probability_node_carrying_minus_inf_is_a_null_event() -> None:
    """A node of probability zero drops out, whatever value it carries."""
    got = _expectation_over_stochastic_nodes(
        values=_array([-jnp.inf, 1.0, 2.0]),
        weights=_array([0.0, 0.5, 0.5]),
    )
    np.testing.assert_array_almost_equal(got, 1.5, decimal=DECIMAL_PRECISION)


def test_zero_probability_node_under_jit_is_a_null_event() -> None:
    """The null-event rule survives compilation."""
    got = jax.jit(_expectation_over_stochastic_nodes)(
        values=_array([-jnp.inf, 1.0, 2.0]),
        weights=_array([0.0, 0.5, 0.5]),
    )
    np.testing.assert_array_almost_equal(got, 1.5, decimal=DECIMAL_PRECISION)


def test_zero_probability_node_does_not_reverse_the_action() -> None:
    """The action ranked best against a null-event continuation is the true one."""
    continuation = _expectation_over_stochastic_nodes(
        values=_array([-jnp.inf, 1.0, 2.0]),
        weights=_array([0.0, 0.5, 0.5]),
    )
    index, _ = argmax_and_max(jnp.stack([_float(1.4), continuation]))
    assert int(index) == 1


@pytest.mark.parametrize("n_nodes", [2, 3, 5, 9])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_zero_probability_node_is_null_at_every_position(
    n_nodes: int, sign: float
) -> None:
    """Wherever the null node sits, the live nodes state the expectation."""
    for dead in range(n_nodes):
        weights = np.arange(1.0, n_nodes + 1.0)
        weights[dead] = 0.0
        weights = weights / weights.sum()
        values = np.linspace(0.5, 3.5, n_nodes)
        values[dead] = sign * np.inf
        live = weights > 0.0
        expected = float(np.sum(weights[live] * values[live]))

        got = _expectation_over_stochastic_nodes(
            values=_array(values.tolist()),
            weights=_array(weights.tolist()),
        )

        np.testing.assert_array_almost_equal(got, expected, decimal=DECIMAL_PRECISION)


def test_positive_probability_minus_inf_survives() -> None:
    """A node that can happen keeps its value, infinite or not."""
    got = _expectation_over_stochastic_nodes(
        values=_array([-jnp.inf, 1.0]),
        weights=_array([0.5, 0.5]),
    )
    assert bool(jnp.isneginf(got))


def test_nan_weight_stays_poison() -> None:
    """A NaN weight is not a probability, and is not quietly absorbed."""
    got = _expectation_over_stochastic_nodes(
        values=_array([1.0, 2.0]),
        weights=_array([jnp.nan, 0.5]),
    )
    assert bool(jnp.isnan(got))


def test_target_with_no_mass_at_all_contributes_nothing() -> None:
    """A target whose nodes carry no mass contributes zero, not NaN."""
    got = _expectation_over_stochastic_nodes(
        values=_array([1.0, 2.0]),
        weights=_array([0.0, 0.0]),
    )
    np.testing.assert_array_almost_equal(got, 0.0, decimal=DECIMAL_PRECISION)


def test_target_with_no_mass_contributes_nothing_even_carrying_infinities() -> None:
    """No mass anywhere means no branch, whatever the dead nodes carry."""
    got = _expectation_over_stochastic_nodes(
        values=_array([-jnp.inf, -jnp.inf]),
        weights=_array([0.0, 0.0]),
    )
    np.testing.assert_array_almost_equal(got, 0.0, decimal=DECIMAL_PRECISION)


def test_a_null_node_leaves_both_cotangents_finite() -> None:
    """Differentiating through a null node produces no NaN in either channel.

    Nothing here claims a classical derivative of the expectation with respect
    to a probability at a zero-mass `-inf` boundary — any positive mass sends
    the expectation to `-inf`. What is claimed is that the structural mask
    keeps the computation differentiable instead of poisoning the tape.
    """

    def expectation(values: Float1D, weights: Float1D) -> ScalarFloat:
        return _expectation_over_stochastic_nodes(values=values, weights=weights)

    values = _array([-jnp.inf, 1.0, 2.0])
    weights = _array([0.0, 0.5, 0.5])

    d_values = jax.grad(expectation, argnums=0)(values, weights)
    d_weights = jax.grad(expectation, argnums=1)(values, weights)

    assert bool(jnp.all(jnp.isfinite(d_values)))
    assert bool(jnp.all(jnp.isfinite(d_weights)))


def test_zero_probability_stateless_target_is_a_null_event() -> None:
    """An unreachable stateless target does not contaminate the joint lottery."""
    values, weights, mass = _scalar_lottery(
        live_V=2.0, dead_V=-jnp.inf, live_prob=1.0, dead_prob=0.0
    )
    got = _InheritedLinearExpectation().aggregate(
        values=values, weights=weights, params={}
    )
    np.testing.assert_array_almost_equal(mass, 1.0, decimal=DECIMAL_PRECISION)
    np.testing.assert_array_almost_equal(got, 2.0, decimal=DECIMAL_PRECISION)


def test_zero_probability_stateless_target_does_not_reverse_the_action() -> None:
    """The action ranked best against a null stateless target is the true one."""
    values, weights, _ = _scalar_lottery(
        live_V=2.0, dead_V=-jnp.inf, live_prob=1.0, dead_prob=0.0
    )
    continuation = _InheritedLinearExpectation().aggregate(
        values=values, weights=weights, params={}
    )
    index, _ = argmax_and_max(jnp.stack([_float(1.9), continuation]))
    assert int(index) == 1


def test_linear_expectation_drops_a_zero_weight_node() -> None:
    """`LinearExpectation.aggregate` states the same null-event rule."""
    got = LinearExpectation().aggregate(
        values=_array([2.0, -jnp.inf]),
        weights=_array([1.0, 0.0]),
        params={},
    )
    np.testing.assert_array_almost_equal(got, 2.0, decimal=DECIMAL_PRECISION)


def test_linear_expectation_keeps_a_positive_weight_minus_inf() -> None:
    """A reachable `-inf` is the expectation, not something to be masked away."""
    got = LinearExpectation().aggregate(
        values=_array([2.0, -jnp.inf]),
        weights=_array([0.5, 0.5]),
        params={},
    )
    assert bool(jnp.isneginf(got))


def test_massless_joint_lottery_aggregates_to_nan() -> None:
    """A lottery with no mass anywhere is malformed, and says so."""
    got = LinearExpectation().aggregate(
        values=_array([2.0, 3.0]),
        weights=_array([0.0, 0.0]),
        params={},
    )
    assert bool(jnp.isnan(got))


def test_a_null_node_inside_a_certain_target_is_a_null_event() -> None:
    """A target reached with certainty still drops its zero-probability nodes.

    The stateful collector hands values and weights to the aggregate together,
    so a node that cannot occur has to be neutral before it is collected — the
    target's own probability says nothing about it.
    """
    values = _array([-jnp.inf, 1.0, 2.0])
    node_weights = _array([0.0, 0.5, 0.5])
    target_probability = _float(1.0)

    final_weights = target_probability * node_weights
    collected = jnp.where(final_weights == 0, _float(0.0), values)

    got = _InheritedLinearExpectation().aggregate(
        values=collected, weights=final_weights, params={}
    )

    np.testing.assert_array_almost_equal(got, 1.5, decimal=DECIMAL_PRECISION)
