"""Tests for lazy NaN diagnostic enrichment in validate_V."""

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


def test_diagnostic_arrays_have_state_action_grid_shape():
    """Diagnostic by_dim breakdown has entries for each state and action."""
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
    diagnostics: dict = exc.diagnostics  # ty: ignore[invalid-assignment]
    u_by_dim = diagnostics["U_nan_fraction"]["by_dim"]
    assert "wealth" in u_by_dim, f"Expected 'wealth' in by_dim, got: {u_by_dim}"
    assert "consumption" in u_by_dim, (
        f"Expected 'consumption' in by_dim, got: {u_by_dim}"
    )


def test_diagnostic_failure_preserves_original_error():
    """If diagnostics crash, the original InvalidValueFunctionError survives."""
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


def test_gpu_fallback_catches_eager_runtime_errors():
    """CPU fallback catches RuntimeError from eager (non-JIT) execution."""
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

    assert call_count == 2
    assert exc_info.value.diagnostics is not None
