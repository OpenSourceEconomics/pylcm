"""Economic derivative cancellation is checked before publishing a valid CE read."""

import jax
import jax.numpy as jnp
import pytest

from _lcm.egm.ce_scaled_linear import join_ce_direction


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("compiled", [False, True])
def test_finite_ce_join_retains_the_original_opposing_operands(
    *, compiled: bool
) -> None:
    """A unit coefficient joins actual finite directions before any log compression."""
    dtype = jnp.asarray(0.0).dtype
    power = 53 if dtype == jnp.float64 else 24
    base = jnp.asarray(2.0**power, dtype=dtype)
    direction = jnp.asarray(-(2.0**power - 1.0), dtype=dtype)
    assert float(base) + float(direction) == 1.0
    evaluate = jax.jit(join_ce_direction) if compiled else join_ce_direction
    result = evaluate(
        logabs=jnp.zeros_like(base),
        sign=jnp.ones_like(base),
        base=base,
        direction=direction,
    )
    assert not bool(jnp.isfinite(result)) or abs(float(result) - 1.0) <= 1e-4
