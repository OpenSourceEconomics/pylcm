"""Integer dtypes are pinned to int32 across pylcm regardless of x64 mode."""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm import Model
from lcm.ages import AgeGrid
from lcm.params import MappingLeaf
from lcm.params.processing import process_params
from lcm.simulation.initial_conditions import (
    MISSING_CAT_CODE,
    build_initial_states,
)
from lcm.simulation.transitions import _update_states_for_subjects
from tests.test_models.deterministic.regression import (
    RegimeId,
    dead,
    get_model,
    get_params,
    working_life,
)


def test_discrete_grid_to_jax_is_int32() -> None:
    """Every `DiscreteGrid.to_jax()` in the model returns an `int32` array."""
    model = get_model(n_periods=3)
    for regime in model.regimes.values():
        for grid in {**regime.states, **regime.actions}.values():
            jax_arr = grid.to_jax()
            if jax_arr.dtype.kind == "i":
                assert jax_arr.dtype == jnp.int32, (
                    f"Discrete grid yielded {jax_arr.dtype}, expected int32."
                )


def test_build_initial_states_discrete_dtype_is_int32() -> None:
    """`build_initial_states` casts every discrete state array to `int32`."""
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
    """`MISSING_CAT_CODE` equals `iinfo(int32).min` — never a real category code."""
    assert jnp.iinfo(jnp.int32).min == MISSING_CAT_CODE


def test_update_states_for_subjects_preserves_storage_dtype() -> None:
    """A transition that returns int64 cannot promote the storage pool to int64."""
    all_states = MappingProxyType(
        {"work__health": jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)}
    )
    int64_next = jnp.asarray([1, 1, 1, 1], dtype=jnp.int64)
    computed = MappingProxyType({"work": MappingProxyType({"next_health": int64_next})})
    subjects = jnp.asarray([True, False, True, False])

    updated = _update_states_for_subjects(
        all_states=all_states,
        computed_next_states=computed,
        subject_indices=subjects,
    )

    assert updated["work__health"].dtype == jnp.int32


def test_process_params_passes_python_int_through_for_jax_weak_typing() -> None:
    """Python `int` params stay weak-typed so JAX promotes them per call site."""
    template = MappingProxyType({"regime_a": MappingProxyType({"final_age": "int"})})
    user_params = {"regime_a": {"final_age": 65}}

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    # Python int stays Python int; JAX weak-typing handles promotion at JIT.
    assert out["regime_a"]["final_age"] == 65
    assert isinstance(out["regime_a"]["final_age"], int)


def test_process_params_casts_int64_array_to_int32() -> None:
    """A `jnp.int64` array param leaf is normalised to `jnp.int32`."""
    template = MappingProxyType({"regime_a": MappingProxyType({"schedule": "Array"})})
    user_params = {"regime_a": {"schedule": jnp.asarray([0, 1, 2], dtype=jnp.int64)}}

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    schedule = out["regime_a"]["schedule"]
    assert schedule.dtype == jnp.int32  # ty: ignore[unresolved-attribute]


def test_process_params_int_array_overflow_raises_with_qualified_name() -> None:
    """An out-of-int32-range int array surfaces the param's qualified name."""
    template = MappingProxyType({"regime_a": MappingProxyType({"big_param": "Array"})})
    user_params = {"regime_a": {"big_param": jnp.asarray([0, 2**40], dtype=jnp.int64)}}

    with pytest.raises(ValueError, match="big_param"):
        process_params(
            params=user_params,
            params_template=template,  # ty: ignore[invalid-argument-type]
        )


def test_process_params_casts_int_array_inside_mapping_leaf_to_int32() -> None:
    """`MappingLeaf` int arrays land at `jnp.int32` after params processing."""
    template = MappingProxyType(
        {"regime_a": MappingProxyType({"sched": "MappingLeaf"})}
    )
    user_params = {
        "regime_a": {
            "sched": MappingLeaf(
                {
                    "low": jnp.asarray([0, 1], dtype=jnp.int64),
                    "high": jnp.asarray([10, 20], dtype=jnp.int64),
                }
            )
        }
    }

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    leaf = out["regime_a"]["sched"]
    assert leaf.data["low"].dtype == jnp.int32  # ty: ignore[unresolved-attribute]
    assert leaf.data["high"].dtype == jnp.int32  # ty: ignore[unresolved-attribute]


def test_simulate_accepts_int64_regime_initial_condition_and_round_trips() -> None:
    """`regime` as `jnp.int64` simulates the same as `jnp.int32`."""
    n_periods = 3
    final_age_alive = 18 + n_periods - 2
    model = Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=18, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )
    params = get_params(n_periods=n_periods)

    common = {
        "wealth": jnp.linspace(20.0, 80.0, num=4),
        "age": jnp.full((4,), 18.0),
    }
    initial_conditions_int32 = {
        **common,
        "regime": jnp.asarray([RegimeId.working_life] * 4, dtype=jnp.int32),
    }
    initial_conditions_int64 = {
        **common,
        "regime": jnp.asarray([RegimeId.working_life] * 4, dtype=jnp.int64),
    }

    df_int32 = model.simulate(
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=initial_conditions_int32,
    ).to_dataframe()
    df_int64 = model.simulate(
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=initial_conditions_int64,
    ).to_dataframe()

    assert df_int64["regime"].equals(df_int32["regime"])
