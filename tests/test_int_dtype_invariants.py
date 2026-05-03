"""Integer dtypes are pinned to int32 across pylcm regardless of x64 mode."""

import jax.numpy as jnp

from lcm.simulation.initial_conditions import (
    MISSING_CAT_CODE,
    build_initial_states,
)
from tests.test_models.deterministic.regression import get_model


def test_discrete_grid_to_jax_is_int32() -> None:
    model = get_model(n_periods=3)
    for regime in model.regimes.values():
        for grid in {**regime.states, **regime.actions}.values():
            jax_arr = grid.to_jax()
            if jax_arr.dtype.kind == "i":
                assert jax_arr.dtype == jnp.int32, (
                    f"Discrete grid yielded {jax_arr.dtype}, expected int32."
                )


def test_build_initial_states_discrete_dtype_is_int32() -> None:
    model = get_model(n_periods=3)
    initial_states = {
        "wealth": jnp.array([20.0, 50.0]),
        "age": jnp.array([18.0, 18.0]),
    }
    flat = build_initial_states(
        initial_states=initial_states,
        internal_regimes=model.internal_regimes,
    )
    for key, arr in flat.items():
        if arr.dtype.kind == "i":
            assert arr.dtype == jnp.int32, (
                f"Initial state {key} has dtype {arr.dtype}, expected int32."
            )


def test_missing_cat_code_is_int32_minimum() -> None:
    assert jnp.iinfo(jnp.int32).min == MISSING_CAT_CODE
