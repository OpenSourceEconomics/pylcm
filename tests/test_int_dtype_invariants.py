"""Integer dtypes are pinned to int32 across pylcm regardless of x64 mode."""

from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.grids import Grid
from _lcm.params.processing import process_params
from _lcm.params.sequence_leaf import SequenceLeaf
from _lcm.simulation.initial_conditions import (
    MISSING_CAT_CODE,
    build_initial_states,
)
from _lcm.simulation.transitions import _advance_states_for_subjects
from _lcm.typing import ParamsTemplate
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm import Model
from lcm.ages import AgeGrid
from lcm.params import MappingLeaf
from tests.test_models.deterministic.regression import (
    RegimeId,
    dead,
    get_model,
    get_params,
    working_life,
)


def _as_template(plain: dict) -> ParamsTemplate:
    """Deep-freeze a plain nested dict into a `ParamsTemplate` for tests."""
    return cast("ParamsTemplate", ensure_containers_are_immutable(plain))


# These tests deliberately pass `int64` inputs to verify the cast at
# the barrier. Re-allow the JAX truncation warning that the
# project-wide filter (see `pyproject.toml`) promotes to an error —
# the legitimate trigger lives here.
pytestmark = pytest.mark.filterwarnings(
    "default:Explicitly requested dtype.*:UserWarning"
)


def test_discrete_grid_to_jax_is_int32() -> None:
    """Every `DiscreteGrid.to_jax()` in the model returns an `int32` array."""
    model = get_model(n_periods=3)
    for regime in model.user_regimes.values():
        for grid in {**regime.states, **regime.actions}.values():
            if not isinstance(grid, Grid):
                continue
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
    states_per_regime = build_initial_states(
        initial_states=initial_states,
        regimes=model._regimes,
    )
    for regime_name, regime_states in states_per_regime.items():
        for state_name, arr in regime_states.items():
            if arr.dtype.kind == "i":
                assert arr.dtype == jnp.int32, (
                    f"Initial state {regime_name}.{state_name} has dtype "
                    f"{arr.dtype}, expected int32."
                )


def test_missing_cat_code_is_int32_minimum() -> None:
    """`MISSING_CAT_CODE` equals `iinfo(int32).min` — never a real category code."""
    assert jnp.iinfo(jnp.int32).min == MISSING_CAT_CODE


def test_advance_states_for_subjects_keeps_same_dtype_round_trip() -> None:
    """Canonical-dtype transition outputs round-trip through the state pool.

    With every input boundary pinned to the canonical dtype, a well-typed user
    transition returns canonical-dtype outputs and `_advance_states_for_subjects`
    writes them through `jnp.where` without dtype change. This test pins the
    contract for the int side; mixed-dtype inputs are out of scope — the
    function does not defend against transitions that violate the canonical-
    dtype invariant.
    """
    states_per_regime = MappingProxyType(
        {
            "work": MappingProxyType(
                {"health": jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)}
            )
        }
    )
    next_values = jnp.asarray([1, 1, 1, 1], dtype=jnp.int32)
    next_states_per_regime = MappingProxyType(
        {"work": MappingProxyType({"health": next_values})}
    )
    subjects = jnp.asarray([True, False, True, False])

    next_states = _advance_states_for_subjects(
        states_per_regime=states_per_regime,
        next_states_per_regime=next_states_per_regime,
        subject_indices=subjects,
    )

    assert next_states["work"]["health"].dtype == jnp.int32


def test_process_params_casts_python_int_to_int32() -> None:
    """A Python `int` param leaf is cast to `jnp.int32`."""
    template = _as_template({"regime_a": {"fun": {"final_age": "int"}}})
    user_params = {"regime_a": {"fun": {"final_age": 65}}}

    out = process_params(
        params=user_params,
        params_template=template,
    )

    final_age = out["regime_a"]["fun__final_age"]
    assert int(final_age) == 65  # ty: ignore[invalid-argument-type]
    assert final_age.dtype == jnp.int32  # ty: ignore[unresolved-attribute]


def test_process_params_casts_int64_array_to_int32() -> None:
    """An `int64` array param leaf is normalised to `jnp.int32`.

    Built with `np.asarray`: a JAX `int64` array is rejected at the beartype
    boundary (`_ParamsLeaf` pins JAX integer leaves to `int32`), so the
    realistic raw-data path for a wider-dtype input is a numpy array.
    """
    template = _as_template({"regime_a": {"fun": {"schedule": "Array"}}})
    user_params = {
        "regime_a": {"fun": {"schedule": np.asarray([0, 1, 2], dtype=np.int64)}}
    }

    out = process_params(
        params=user_params,
        params_template=template,
    )

    schedule = out["regime_a"]["fun__schedule"]
    assert schedule.dtype == jnp.int32  # ty: ignore[unresolved-attribute]


def test_process_params_int_array_overflow_raises_with_qualified_name() -> None:
    """An out-of-int32-range int array surfaces the param's qualified name."""
    template = _as_template({"regime_a": {"fun": {"big_param": "Array"}}})
    # Numpy here: under `jax_enable_x64=False`, `jnp.asarray(..., dtype=int64)`
    # of an out-of-int32 value raises before our helper sees it.
    user_params = {
        "regime_a": {"fun": {"big_param": np.asarray([0, 2**40], dtype=np.int64)}}
    }

    with pytest.raises(ValueError, match="big_param"):
        process_params(
            params=user_params,
            params_template=template,
        )


@pytest.mark.parametrize("key", ["low", "high"])
def test_process_params_casts_int_array_inside_mapping_leaf_to_int32(key: str) -> None:
    """`MappingLeaf` int arrays land at `jnp.int32` after params processing."""
    template = _as_template({"regime_a": {"fun": {"sched": "MappingLeaf"}}})
    user_params = {
        "regime_a": {
            "fun": {
                "sched": MappingLeaf(
                    {
                        "low": jnp.asarray([0, 1], dtype=jnp.int64),
                        "high": jnp.asarray([10, 20], dtype=jnp.int64),
                    }
                )
            }
        }
    }

    out = process_params(
        params=user_params,
        params_template=template,
    )

    assert (
        out["regime_a"]["fun__sched"].data[key].dtype  # ty: ignore[unresolved-attribute, invalid-argument-type]
        == jnp.int32
    )


@pytest.mark.parametrize("index", [0, 1])
def test_process_params_casts_int_array_inside_sequence_leaf_to_int32(
    index: int,
) -> None:
    """`SequenceLeaf` int arrays land at `jnp.int32` after params processing."""
    template = _as_template({"regime_a": {"fun": {"sched": "SequenceLeaf"}}})
    user_params = {
        "regime_a": {
            "fun": {
                "sched": SequenceLeaf(
                    [
                        jnp.asarray([0, 1], dtype=jnp.int64),
                        jnp.asarray([10, 20], dtype=jnp.int64),
                    ]
                )
            }
        }
    }

    out = process_params(
        params=user_params,
        params_template=template,
    )

    assert (
        out["regime_a"]["fun__sched"].data[index].dtype  # ty: ignore[unresolved-attribute, invalid-argument-type]
        == jnp.int32
    )


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
        "regime_id": jnp.asarray([RegimeId.working_life] * 4, dtype=jnp.int32),
    }
    initial_conditions_int64 = {
        **common,
        "regime_id": jnp.asarray([RegimeId.working_life] * 4, dtype=jnp.int64),
    }

    df_int32 = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=initial_conditions_int32,
    ).to_dataframe()
    df_int64 = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=initial_conditions_int64,
    ).to_dataframe()

    pd.testing.assert_frame_equal(df_int64, df_int32, check_dtype=False)
