"""Regression tests for the production scaled-probability reduction."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.Q_and_F import _expectation_over_stochastic_nodes
from lcm.typing import FloatND


def _maybe_jit(
    func: Callable[..., FloatND], *, compiled: bool
) -> Callable[..., FloatND]:
    return jax.jit(func) if compiled else func


def test_production_reduction_avoids_general_exponent_balancing() -> None:
    """Keep general exponent balancing off the target-node expectation hot path."""
    dtype = jnp.zeros(()).dtype
    traced = jax.make_jaxpr(_expectation_over_stochastic_nodes)(
        values=jnp.ones(5, dtype=dtype),
        weights=jnp.full(5, 0.2, dtype=dtype),
        shifts=jnp.zeros(5, dtype=jnp.int32),
    )
    primitives = {str(equation.primitive) for equation in traced.jaxpr.eqns}

    assert primitives.isdisjoint({"clz", "custom_jvp_call"})


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_a_large_rare_contribution_is_formed_before_its_scale_is_applied(
    *, compiled: bool
) -> None:
    """A probability below the common scale can still contribute one quarter.

    The scaled coefficient is the smallest normal number and therefore safe to
    multiply. Applying its residual scale before it meets the value would make
    the probability subnormal and let the backend flush a contribution of 1/4.
    """
    dtype = jnp.zeros(()).dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    large = jnp.asarray(1.0, dtype=dtype) / tiny
    reduce = _maybe_jit(_expectation_over_stochastic_nodes, compiled=compiled)

    got = reduce(
        values=jnp.asarray([0.0, large], dtype=dtype),
        weights=jnp.asarray([1.0, tiny], dtype=dtype),
        shifts=jnp.asarray([0, 2], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(0.25, dtype=dtype))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("kind", ["nan", "positive_infinity", "negative_infinity"])
def test_a_live_nonfinite_node_is_not_reclassified_as_null_by_its_scale(
    *, compiled: bool, kind: str
) -> None:
    """A live node stays non-finite even when its plain weight underflows."""
    reduce = _maybe_jit(_expectation_over_stochastic_nodes, compiled=compiled)
    spread = 300 if jnp.zeros(()).dtype == jnp.float32 else 2800
    nonfinite = {
        "nan": jnp.nan,
        "positive_infinity": jnp.inf,
        "negative_infinity": -jnp.inf,
    }[kind]

    got = reduce(
        values=jnp.asarray([1.0, nonfinite]),
        weights=jnp.ones(2),
        shifts=jnp.asarray([0, spread], dtype=jnp.int32),
    )

    if kind == "nan":
        assert bool(jnp.isnan(got))
    elif kind == "positive_infinity":
        assert bool(jnp.isposinf(got))
    else:
        assert bool(jnp.isneginf(got))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_a_represented_zero_remains_the_null_event_at_any_scale(
    *, compiled: bool
) -> None:
    """A genuine zero coefficient annihilates a non-finite value."""
    reduce = _maybe_jit(_expectation_over_stochastic_nodes, compiled=compiled)

    got = reduce(
        values=jnp.asarray([1.0, jnp.nan]),
        weights=jnp.asarray([1.0, 0.0]),
        shifts=jnp.asarray([0, 300], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(1.0))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_the_mass_is_read_on_the_same_common_scale(*, compiled: bool) -> None:
    """Weights `(1, 1/4)` price the second node at one fifth."""
    reduce = _maybe_jit(_expectation_over_stochastic_nodes, compiled=compiled)

    got = reduce(
        values=jnp.asarray([0.0, 1.0]),
        weights=jnp.ones(2),
        shifts=jnp.asarray([0, 2], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(0.2))
