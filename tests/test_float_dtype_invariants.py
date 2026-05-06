"""Float dtypes follow `canonical_float_dtype()` across pylcm boundaries."""

from collections.abc import Iterator
from types import MappingProxyType

import jax.numpy as jnp
import pytest
from jax import config as jax_config

from lcm.dtypes import canonical_float_dtype
from lcm.grids import IrregSpacedGrid, LinSpacedGrid, LogSpacedGrid
from lcm.params import MappingLeaf
from lcm.params.processing import process_params
from lcm.simulation.initial_conditions import build_initial_states
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)


@pytest.fixture(name="x64_disabled")
def _fixture_x64_disabled() -> Iterator[None]:
    previous = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", val=False)
    try:
        yield
    finally:
        jax_config.update("jax_enable_x64", val=previous)


def test_build_initial_states_continuous_state_cast_to_canonical_dtype(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """Continuous initial states land at `canonical_float_dtype()` for any input."""
    model = get_model(n_periods=3)
    # User passes float64 arrays under x64=False — should be cast to float32.
    initial_states = {
        "wealth": jnp.asarray([20.0, 50.0], dtype=jnp.float64),
        "age": jnp.asarray([18.0, 18.0], dtype=jnp.float64),
    }
    flat = build_initial_states(
        initial_states=initial_states,
        internal_regimes=model.internal_regimes,
    )
    target = canonical_float_dtype()
    for key, arr in flat.items():
        if arr.dtype.kind == "f":
            assert arr.dtype == target, (
                f"Initial state {key} has dtype {arr.dtype}, expected {target}."
            )


def test_build_initial_states_int_input_for_continuous_state_cast_to_canonical(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """Int initial-condition arrays for continuous states land at canonical float."""
    model = get_model(n_periods=3)
    initial_states = {
        "wealth": jnp.asarray([20, 50], dtype=jnp.int32),
        "age": jnp.asarray([18, 18], dtype=jnp.int32),
    }
    flat = build_initial_states(
        initial_states=initial_states,
        internal_regimes=model.internal_regimes,
    )
    target = canonical_float_dtype()
    # All non-discrete state entries should be canonical-float now.
    for key, arr in flat.items():
        if "wealth" in key or "age" in key:
            assert arr.dtype == target, (
                f"Continuous state {key} has dtype {arr.dtype}, expected {target}."
            )


def test_build_initial_states_missing_continuous_fallback_dtype_is_canonical(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """Missing continuous states fall back to `nan` at the canonical float dtype."""
    model = get_model(n_periods=3)
    initial_states = {
        "wealth": jnp.asarray([20.0, 50.0]),
        "age": jnp.asarray([18.0, 18.0]),
    }
    flat = build_initial_states(
        initial_states=initial_states,
        internal_regimes=model.internal_regimes,
    )
    # Find a fallback-NaN entry and check its dtype.
    target = canonical_float_dtype()
    nan_entries = [arr for arr in flat.values() if arr.dtype.kind == "f"]
    assert nan_entries  # sanity
    for arr in nan_entries:
        assert arr.dtype == target


def test_process_params_casts_float64_array_to_canonical_under_no_x64(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """A `float64` array param is downcast to `float32` under `jax_enable_x64=False`."""
    template = MappingProxyType({"regime_a": MappingProxyType({"schedule": "Array"})})
    user_params = {
        "regime_a": {"schedule": jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)}
    }

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    schedule = out["regime_a"]["schedule"]
    assert schedule.dtype == jnp.float32  # ty: ignore[unresolved-attribute]


def test_process_params_passes_python_float_through_for_jax_weak_typing(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """Python `float` params stay weak-typed so JAX promotes them per call site."""
    template = MappingProxyType(
        {"regime_a": MappingProxyType({"discount_factor": "float"})}
    )
    user_params = {"regime_a": {"discount_factor": 0.95}}

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    # Python float stays Python float; JAX weak-typing handles promotion at JIT.
    assert out["regime_a"]["discount_factor"] == 0.95
    assert isinstance(out["regime_a"]["discount_factor"], float)


def test_process_params_float_array_overflow_raises_with_qualified_name(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """An out-of-float32 float64 *array* raises naming the qualified leaf."""
    template = MappingProxyType({"regime_a": MappingProxyType({"schedule": "Array"})})
    user_params = {
        "regime_a": {"schedule": jnp.asarray([0.0, 1e40], dtype=jnp.float64)}
    }

    with pytest.raises(OverflowError, match="schedule"):
        process_params(
            params=user_params,
            params_template=template,  # ty: ignore[invalid-argument-type]
        )


def test_simulate_state_pool_dtype_stable_across_periods(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """A multi-period simulate keeps every state's dtype stable across periods."""
    n_periods = 4
    model = get_model(n_periods=n_periods)
    params = get_params(n_periods=n_periods)
    initial = {
        "wealth": jnp.asarray([20.0, 50.0, 80.0]),
        "age": jnp.asarray([18.0, 18.0, 18.0]),
        "regime": jnp.asarray([RegimeId.working_life] * 3),
    }

    result = model.simulate(
        params=params, period_to_regime_to_V_arr=None, initial_conditions=initial
    )

    # Build the per-period state-dtype matrix and assert stability.
    seen: dict[str, set] = {}
    for period_data in result.raw_results.values():
        for snap in period_data.values():
            for state_name, arr in snap.states.items():
                seen.setdefault(state_name, set()).add(arr.dtype)
    for state_name, dtypes in seen.items():
        assert len(dtypes) == 1, f"State {state_name} drifted across periods: {dtypes}"


def test_solve_v_arrays_at_canonical_float_dtype(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """Every V-array returned by `model.solve()` is at `canonical_float_dtype()`."""
    model = get_model(n_periods=3)
    period_to_regime_to_V_arr = model.solve(params=get_params(n_periods=3))
    target = canonical_float_dtype()
    for period_v in period_to_regime_to_V_arr.values():
        for regime_name, v_arr in period_v.items():
            assert v_arr.dtype == target, (
                f"V[{regime_name}] dtype is {v_arr.dtype}, expected {target}."
            )


def test_continuous_grid_to_jax_dtype_is_canonical(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """Continuous grid `to_jax()` returns canonical-float arrays."""
    target = canonical_float_dtype()
    assert LinSpacedGrid(start=0, stop=1, n_points=5).to_jax().dtype == target
    assert LogSpacedGrid(start=1, stop=10, n_points=5).to_jax().dtype == target
    assert IrregSpacedGrid(points=(0.0, 0.5, 1.0)).to_jax().dtype == target


def test_process_params_casts_float_array_inside_mapping_leaf_to_canonical(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """`MappingLeaf` float arrays land at `canonical_float_dtype()`."""
    template = MappingProxyType(
        {"regime_a": MappingProxyType({"sched": "MappingLeaf"})}
    )
    user_params = {
        "regime_a": {
            "sched": MappingLeaf(
                {
                    "low": jnp.asarray([0.1, 0.2], dtype=jnp.float64),
                    "high": jnp.asarray([0.5, 0.7], dtype=jnp.float64),
                }
            )
        }
    }

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    leaf = out["regime_a"]["sched"]
    assert leaf.data["low"].dtype == jnp.float32  # ty: ignore[unresolved-attribute]
    assert leaf.data["high"].dtype == jnp.float32  # ty: ignore[unresolved-attribute]
