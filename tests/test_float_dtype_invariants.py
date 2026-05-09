"""Float dtypes follow `canonical_float_dtype()` across pylcm boundaries."""

from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from lcm.dtypes import canonical_float_dtype
from lcm.grids import IrregSpacedGrid, LinSpacedGrid, LogSpacedGrid
from lcm.params import MappingLeaf
from lcm.params.processing import process_params
from lcm.params.sequence_leaf import SequenceLeaf
from lcm.simulation.initial_conditions import build_initial_states
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)


def test_build_initial_states_casts_user_float64_to_canonical(x64_disabled: None):
    """A float64 continuous initial state lands at `canonical_float_dtype()`."""
    model = get_model(n_periods=3)
    initial_states = {
        "wealth": np.asarray([20.0, 50.0], dtype=np.float64),
        "age": np.asarray([18.0, 18.0], dtype=np.float64),
    }
    flat = build_initial_states(
        initial_states=initial_states,  # ty: ignore[invalid-argument-type]
        internal_regimes=model.internal_regimes,
    )
    assert flat["working_life__wealth"].dtype == canonical_float_dtype()


def test_build_initial_states_casts_user_int_to_canonical(x64_disabled: None):
    """A continuous initial state given as int32 lands at `canonical_float_dtype()`."""
    model = get_model(n_periods=3)
    initial_states = {
        "wealth": jnp.asarray([20, 50], dtype=jnp.int32),
        "age": jnp.asarray([18, 18], dtype=jnp.int32),
    }
    flat = build_initial_states(
        initial_states=initial_states,
        internal_regimes=model.internal_regimes,
    )
    assert flat["working_life__wealth"].dtype == canonical_float_dtype()


def test_build_initial_states_missing_continuous_fallback_dtype_is_canonical(
    x64_disabled: None,
):
    """A missing continuous state falls back to a canonical-dtype array."""
    model = get_model(n_periods=3)
    # Supply a placeholder state to set n_subjects without touching `wealth`.
    flat = build_initial_states(
        initial_states={"placeholder": jnp.asarray([0.0, 0.0])},
        internal_regimes=model.internal_regimes,
    )
    assert flat["working_life__wealth"].dtype == canonical_float_dtype()


def test_build_initial_states_missing_continuous_fallback_values_are_nan(
    x64_disabled: None,
):
    """A missing continuous state falls back to an all-NaN array.

    Pinning only the dtype would let a regression that fills the fallback
    with zeros (or anything else representable) pass; assert the values.
    """
    model = get_model(n_periods=3)
    flat = build_initial_states(
        initial_states={"placeholder": jnp.asarray([0.0, 0.0])},
        internal_regimes=model.internal_regimes,
    )
    assert bool(jnp.all(jnp.isnan(flat["working_life__wealth"])))


def test_process_params_casts_float64_array_to_canonical_under_no_x64(
    x64_disabled: None,
):
    """A `float64` array param is downcast to `float32` under `jax_enable_x64=False`.

    Build with `np.asarray` rather than `jnp.asarray` — the JAX builder
    silently truncates to `float32` under no-x64 at construction time, so a
    JAX-built input would never reach the helper as `float64`.
    """
    template = MappingProxyType({"regime_a": MappingProxyType({"schedule": "Array"})})
    user_params = {
        "regime_a": {"schedule": np.asarray([0.1, 0.2, 0.3], dtype=np.float64)}
    }

    out = process_params(
        params=user_params,  # ty: ignore[invalid-argument-type]
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    schedule = out["regime_a"]["schedule"]
    assert schedule.dtype == jnp.float32


def test_process_params_casts_python_float_to_canonical(x64_disabled: None):
    """A Python `float` param leaf is cast to `canonical_float_dtype()`."""
    template = MappingProxyType(
        {"regime_a": MappingProxyType({"discount_factor": "float"})}
    )
    user_params = {"regime_a": {"discount_factor": 0.95}}

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    discount_factor = out["regime_a"]["discount_factor"]
    np.testing.assert_allclose(float(discount_factor), 0.95, rtol=1e-6)
    assert discount_factor.dtype == canonical_float_dtype()


def test_process_params_float_array_overflow_raises_with_qualified_name(
    x64_disabled: None,
):
    """An out-of-float32 float64 array raises naming the qualified leaf."""
    template = MappingProxyType({"regime_a": MappingProxyType({"schedule": "Array"})})
    user_params = {"regime_a": {"schedule": np.asarray([0.0, 1e40], dtype=np.float64)}}

    with pytest.raises(OverflowError, match="schedule"):
        process_params(
            params=user_params,  # ty: ignore[invalid-argument-type]
            params_template=template,  # ty: ignore[invalid-argument-type]
        )


def test_simulate_state_pool_dtype_stable_across_periods(x64_disabled: None):
    """A multi-period simulate keeps every state's dtype stable across periods.

    The intended invariant is per-state stability; failing on any single
    state still gives an actionable signal because the assertion message
    names the offending state and its observed dtypes.
    """
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

    seen: dict[str, set] = {}
    for period_data in result.raw_results.values():
        for snap in period_data.values():
            for state_name, arr in snap.states.items():
                seen.setdefault(state_name, set()).add(arr.dtype)
    drifted = {name: dtypes for name, dtypes in seen.items() if len(dtypes) != 1}
    assert not drifted, f"States drifted across periods: {drifted}"


def test_solve_v_arrays_at_canonical_float_dtype(x64_disabled: None):
    """Every V-array returned by `model.solve()` is at `canonical_float_dtype()`."""
    model = get_model(n_periods=3)
    period_to_regime_to_V_arr = model.solve(params=get_params(n_periods=3))
    target = canonical_float_dtype()
    wrong = {
        (period, regime_name): v_arr.dtype
        for period, period_v in period_to_regime_to_V_arr.items()
        for regime_name, v_arr in period_v.items()
        if v_arr.dtype != target
    }
    assert not wrong, f"V-arrays not at {target}: {wrong}"


@pytest.mark.parametrize(
    "make_grid",
    [
        lambda: LinSpacedGrid(start=0, stop=1, n_points=5),
        lambda: LogSpacedGrid(start=1, stop=10, n_points=5),
        lambda: IrregSpacedGrid(points=(0.0, 0.5, 1.0)),
    ],
    ids=["linspaced", "logspaced", "irregspaced"],
)
def test_continuous_grid_to_jax_dtype_is_canonical_under_no_x64(
    make_grid: Callable[[], LinSpacedGrid | LogSpacedGrid | IrregSpacedGrid],
    x64_disabled: None,
):
    """Continuous grid `to_jax()` materialises at `float32` under no-x64.

    Asserts the concrete target dtype rather than `canonical_float_dtype()`
    so the test fails if a future grid implementation hardcodes `float64`
    (which JAX would silently truncate to `float32` under no-x64; the
    helper-side comparison would mask that, the literal-side comparison
    surfaces it).

    Grids are constructed inside the test body so the `x64_disabled`
    fixture is in effect; grid dtype is now sticky to construction-time
    `jax_enable_x64`.
    """
    grid = make_grid()
    assert grid.to_jax().dtype == jnp.float32


@pytest.mark.parametrize("attr", ["start", "stop"])
def test_uniform_grid_stores_endpoints_as_canonical_jax_scalar(
    attr: str, x64_disabled: None
):
    """`LinSpacedGrid` stores `start`/`stop` as JAX scalars at canonical dtype."""
    grid = LinSpacedGrid(start=0.0, stop=100.0, n_points=10)
    value = getattr(grid, attr)
    assert isinstance(value, jnp.ndarray)
    assert value.dtype == canonical_float_dtype()


def test_irreg_grid_stores_points_as_canonical_jax_array(x64_disabled: None):
    """`IrregSpacedGrid` stores `points` as a JAX array at canonical dtype."""
    grid = IrregSpacedGrid(points=(0.0, 0.5, 1.0))
    assert isinstance(grid.points, jnp.ndarray)
    assert grid.points.dtype == canonical_float_dtype()


@pytest.mark.parametrize("key", ["low", "high"])
def test_process_params_casts_float_array_inside_mapping_leaf_to_canonical(
    key: str, x64_disabled: None
):
    """`MappingLeaf` float arrays land at `canonical_float_dtype()`."""
    template = MappingProxyType(
        {"regime_a": MappingProxyType({"sched": "MappingLeaf"})}
    )
    user_params = {
        "regime_a": {
            "sched": MappingLeaf(
                {
                    "low": np.asarray([0.1, 0.2], dtype=np.float64),
                    "high": np.asarray([0.5, 0.7], dtype=np.float64),
                }
            )
        }
    }

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    assert (
        out["regime_a"]["sched"].data[key].dtype  # ty: ignore[unresolved-attribute]
        == jnp.float32
    )


@pytest.mark.parametrize("index", [0, 1])
def test_process_params_casts_float_array_inside_sequence_leaf_to_canonical(
    index: int, x64_disabled: None
):
    """`SequenceLeaf` float arrays land at `canonical_float_dtype()`."""
    template = MappingProxyType(
        {"regime_a": MappingProxyType({"sched": "SequenceLeaf"})}
    )
    user_params = {
        "regime_a": {
            "sched": SequenceLeaf(
                [
                    np.asarray([0.1, 0.2], dtype=np.float64),
                    np.asarray([0.5, 0.7], dtype=np.float64),
                ]
            )
        }
    }

    out = process_params(
        params=user_params,
        params_template=template,  # ty: ignore[invalid-argument-type]
    )

    assert (
        out["regime_a"]["sched"].data[index].dtype  # ty: ignore[unresolved-attribute]
        == jnp.float32
    )
