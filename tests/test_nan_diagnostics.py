"""Tests for lazy NaN diagnostic enrichment in validate_V.

These tests currently FAIL — they demonstrate bugs in the diagnostic path
that need to be fixed:
1. Diagnostic arrays have wrong shapes (not productmapped)
2. Diagnostic failure swallows the original InvalidValueFunctionError
3. GPU→CPU fallback catches too narrow an exception type
"""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm.exceptions import InvalidValueFunctionError
from lcm.interfaces import StateActionSpace
from lcm.utils.error_handling import validate_V


def _make_state_action_space(
    *,
    n_wealth: int = 3,
    n_consumption: int = 2,
) -> StateActionSpace:
    return StateActionSpace(
        states=MappingProxyType(
            {"wealth": jnp.linspace(1.0, 5.0, n_wealth)},
        ),
        discrete_actions=MappingProxyType({}),
        continuous_actions=MappingProxyType(
            {"consumption": jnp.linspace(0.1, 2.0, n_consumption)},
        ),
        state_and_discrete_action_names=("wealth",),
    )


def _make_nan_V(n_wealth: int = 3) -> jnp.ndarray:
    """Create a V array with NaN to trigger diagnostics."""
    return jnp.full(n_wealth, jnp.nan)


@pytest.mark.xfail(
    reason="compute_intermediates called with flat 1D arrays, not productmapped",
    strict=True,
)
def test_diagnostic_arrays_have_state_action_grid_shape():
    """Diagnostic by_dim breakdown must have entries for each state dimension.

    Currently fails because _enrich_with_diagnostics passes flat 1D grid
    arrays to compute_intermediates instead of a productmapped Cartesian
    product. The resulting arrays are 1D (from broadcasting), so
    _summarize_diagnostics maps axis 0 to the first state name but all
    other state dimensions are missing.
    """
    sas = _make_state_action_space(n_wealth=3, n_consumption=2)

    def mock_compute_intermediates(**kwargs: jnp.ndarray) -> tuple:
        wealth = jnp.asarray(kwargs["wealth"])
        consumption = jnp.asarray(kwargs["consumption"])
        U = jnp.log(consumption)
        F = wealth - consumption >= 0
        E_next_V = jnp.zeros_like(U)
        Q = U + 0.9 * E_next_V
        probs = MappingProxyType({"alive": jnp.array(1.0)})
        return U, F, E_next_V, Q, probs

    with pytest.raises(InvalidValueFunctionError) as exc_info:
        validate_V(
            V_arr=_make_nan_V(3),
            age=0.0,
            regime_name="alive",
            partial_solution=MappingProxyType({}),
            compute_intermediates=mock_compute_intermediates,
            state_action_space=sas,
            next_regime_to_V_arr=MappingProxyType(
                {"alive": jnp.zeros(3)},
            ),
            internal_params=MappingProxyType({}),
        )

    exc = exc_info.value
    assert exc.diagnostics is not None
    # The by_dim breakdown should have an entry for "wealth"
    diagnostics: dict = exc.diagnostics  # ty: ignore[invalid-assignment]
    u_by_dim = diagnostics["U_nan_fraction"]["by_dim"]
    assert "wealth" in u_by_dim, (
        f"Expected 'wealth' in by_dim breakdown, got: {u_by_dim}"
    )


@pytest.mark.xfail(
    reason="_enrich_with_diagnostics not wrapped in try/except",
    strict=True,
)
def test_diagnostic_failure_preserves_original_error():
    """If diagnostics crash, the original InvalidValueFunctionError must survive.

    Currently fails because _enrich_with_diagnostics is called without
    try/except in validate_V. When the diagnostic closure raises, its
    exception replaces the original NaN error.
    """
    sas = _make_state_action_space()

    def broken_compute_intermediates(**kwargs: jnp.ndarray) -> None:  # noqa: ARG001
        msg = "intentional diagnostic failure"
        raise RuntimeError(msg)

    with pytest.raises(InvalidValueFunctionError, match="NaN"):
        validate_V(
            V_arr=_make_nan_V(),
            age=0.0,
            regime_name="test",
            partial_solution=MappingProxyType({}),
            compute_intermediates=broken_compute_intermediates,
            state_action_space=sas,
            next_regime_to_V_arr=MappingProxyType(
                {"test": jnp.zeros(3)},
            ),
            internal_params=MappingProxyType({}),
        )


@pytest.mark.xfail(
    reason="GPU fallback catches JaxRuntimeError but closure runs eagerly",
    strict=True,
)
def test_gpu_fallback_catches_eager_runtime_errors():
    """CPU fallback must catch RuntimeError from eager (non-JIT) execution.

    Currently fails because _enrich_with_diagnostics catches only
    jax.errors.JaxRuntimeError, but the closure is not JIT-compiled.
    Eager execution raises plain RuntimeError on failure.
    """
    sas = _make_state_action_space()
    call_count = 0

    def flaky_compute_intermediates(**kwargs: jnp.ndarray) -> tuple:  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "simulated GPU OOM"
            raise RuntimeError(msg)
        # Second call (CPU fallback) succeeds
        U = jnp.zeros(3)
        F = jnp.ones(3, dtype=bool)
        E_next_V = jnp.zeros(3)
        Q = jnp.zeros(3)
        probs = MappingProxyType({"test": jnp.array(1.0)})
        return U, F, E_next_V, Q, probs

    with pytest.raises(InvalidValueFunctionError) as exc_info:
        validate_V(
            V_arr=_make_nan_V(),
            age=0.0,
            regime_name="test",
            partial_solution=MappingProxyType({}),
            compute_intermediates=flaky_compute_intermediates,
            state_action_space=sas,
            next_regime_to_V_arr=MappingProxyType(
                {"test": jnp.zeros(3)},
            ),
            internal_params=MappingProxyType({}),
        )

    # The fallback should have retried on CPU
    assert call_count == 2
    assert exc_info.value.diagnostics is not None
