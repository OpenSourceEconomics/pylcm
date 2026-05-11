"""Pyproject's `filterwarnings` rule promotes the JAX truncation warning to an error.

The rule fires when source code asks for a dtype wider than the active
`jax_enable_x64` setting permits — under `--precision=32`, every stray
`jnp.int64` / `jnp.float64` request in `src/` becomes a test failure.
The dtype-invariant test modules opt back to `default` for the same
warning because they exist to *exercise* the cast at the barrier.
"""

import jax.numpy as jnp
import pytest


def test_float64_request_under_no_x64_raises(x64_disabled: None):
    """A `jnp.float64` literal under `jax_enable_x64=False` is promoted to an error."""
    with pytest.raises(UserWarning, match="Explicitly requested dtype float64"):
        jnp.asarray([1.0, 2.0], dtype=jnp.float64)


def test_int64_request_under_no_x64_raises(x64_disabled: None):
    """A `jnp.int64` literal under `jax_enable_x64=False` is promoted to an error."""
    with pytest.raises(UserWarning, match="Explicitly requested dtype int64"):
        jnp.asarray([1, 2, 3], dtype=jnp.int64)
