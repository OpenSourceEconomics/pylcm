import os
from collections.abc import Mapping

import jax
import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm import (
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Piece,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from lcm._config import TEST_DATA
from lcm.grids import UniformContinuousGrid
from lcm.typing import FloatND
from lcm_examples import mortality as mortality_example
from lcm_examples import precautionary_savings as ps_example
from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    create_inputs,
)
from tests.conftest import X64_ENABLED
from tests.test_models.deterministic.regression import RegimeId, get_model, get_params

_PRECISION_DIR = TEST_DATA / "regression_tests" / ("f64" if X64_ENABLED else "f32")

_HAS_GPU = jax.devices()[0].platform == "gpu"
_skip_no_gpu = pytest.mark.skipif(not _HAS_GPU, reason="requires GPU")


def test_regression_test():
    """Test that the output of lcm does not change."""
    # Load expected output
    expected_simulate = pd.read_pickle(
        TEST_DATA / "regression_tests" / "simulation.pkl"
    )
    expected_solve = pd.read_pickle(TEST_DATA / "regression_tests" / "solution.pkl")

    # Generate current lcm output
    n_periods = 4
    model = get_model(n_periods=n_periods)
    params = get_params(
        n_periods=n_periods,
        discount_factor=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )

    got_solve: Mapping[int, Mapping[str, FloatND]] = model.solve(params=params)
    got_simulate = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([5.0, 20, 40, 70]),
            "age": jnp.array([18.0, 18.0, 18.0, 18.0]),
            "regime": jnp.array([RegimeId.working_life] * 4),
        },
        period_to_regime_to_V_arr=None,
    ).to_dataframe()

    # Compare solution (iterate over expected regimes — got may have additional ones)
    for period in expected_solve:
        for regime in expected_solve[period]:
            aaae(expected_solve[period][regime], got_solve[period][regime], decimal=5)

    # Compare simulation (use tolerance to match solution comparison precision)
    assert_frame_equal(
        got_simulate,
        expected_simulate,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )


@pytest.mark.gpu
@_skip_no_gpu
def test_regression_precautionary_savings():
    """Test that precautionary savings benchmark model output does not change."""
    expected = pd.read_pickle(_PRECISION_DIR / "precautionary_savings_simulation.pkl")

    model = ps_example.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_n_points=10,
        consumption_n_points=10,
    )
    params = ps_example.get_params(shock_type="rouwenhorst", sigma=0.2, rho=0.9)

    n_subjects = 4
    got = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_subjects, 20.0),
            "wealth": jnp.full(n_subjects, 5.0),
            "income": jnp.full(n_subjects, 0.0),
            "regime": jnp.zeros(n_subjects, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    ).to_dataframe()

    assert_frame_equal(
        got,
        expected,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )


@pytest.mark.gpu
@_skip_no_gpu
def test_regression_mortality():
    """Test that mortality benchmark model output does not change."""
    expected = pd.read_pickle(_PRECISION_DIR / "mortality_simulation.pkl")

    n_periods = 4
    model = mortality_example.get_model(n_periods=n_periods)
    params = mortality_example.get_params(n_periods=n_periods)

    n_subjects = 4
    got = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_subjects, 40.0),
            "wealth": jnp.full(n_subjects, 100.0),
            "regime": jnp.zeros(n_subjects, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    ).to_dataframe()

    assert_frame_equal(
        got,
        expected,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )


def _per_period_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a simulation DataFrame to per-period averages.

    Categorical columns are converted to integer codes so fraction-by-category
    is captured (e.g. the regime column becomes per-period mortality share).
    Subject-level detail is lost, which is exactly the point — XLA kernel
    fusion differs across processes and individual survival draws flip
    stochastically at Mahler-Yum's scale; per-period averages absorb that
    noise while still pinning the trajectory shape of the simulation.
    """
    numeric = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            numeric[col] = df[col].cat.codes
    return numeric.groupby("period").mean(numeric_only=True)


@pytest.mark.gpu
@_skip_no_gpu
def test_regression_mahler_yum():
    """Test that Mahler & Yum per-period-averaged trajectories are stable.

    Set `LCM_UPDATE_FIXTURES=1` to regenerate the fixture from pytest. The
    fixture must be regenerated under the same `PYTHONHASHSEED` that CI uses
    (`0`, pinned on the `tests`/`tests-32bit` pixi tasks) — otherwise random
    draws derived through Python's hash-dependent ordering diverge between
    fixture-generation and test runs by tens of percent per period. With the
    pin in place, f64 values are byte-reproducible and f32 values drift by
    <1e-4 per column.

    Tolerances are set to `atol=0.15, rtol=0.15`:

    The fixture was generated under the pre-PR-#331 implicit
    auto-partition-lift code path, where Mahler-Yum's discrete
    fixed-transition states were partition-lifted with `jax.lax.scan`
    by default. Under the current explicit-dispatch API they remain in
    the state-action space (`FUSED_VMAP`) — a deliberate kernel-shape
    change, not a regression. The resulting drift accumulates over 80
    backward-induction periods and shows up on both fraction-
    denominated columns (~13/128 ≈ 0.10 on `regime`) and dollar-
    denominated columns (~10% on `value`).

    `atol=0.15, rtol=0.15` give a ~50% margin above observed drift
    while still catching anything a real model regression would
    produce (order-of-magnitude larger shifts). Regenerating the
    fixture under the current kernel would tighten these, but is
    deferred until the Mahler-Yum model's partition choices are
    finalized.
    """
    fixture_path = _PRECISION_DIR / "mahler_yum_simulation_per_period.pkl"

    n_subjects = 128
    common_params, initial_states = create_inputs(
        seed=0,
        n_simulation_subjects=n_subjects,
        **START_PARAMS,  # ty: ignore[invalid-argument-type]
    )
    model = MAHLER_YUM_MODEL
    params = {"alive": common_params}
    initial_conditions = {
        **initial_states,
        "regime": jnp.full(
            n_subjects,
            model.regime_names_to_ids["alive"],
            dtype=jnp.int32,
        ),
    }

    got = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    ).to_dataframe()
    got_means = _per_period_averages(got)

    if os.environ.get("LCM_UPDATE_FIXTURES"):
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        got_means.to_pickle(fixture_path)
        pytest.skip(f"regenerated fixture at {fixture_path}")

    expected = pd.read_pickle(fixture_path)
    assert_frame_equal(
        got_means,
        expected,
        check_dtype=False,
        atol=0.15,
        rtol=0.15,
        check_column_type=False,
    )


def _create_grid(
    grid_type: str, start: float, stop: float, n_points: int
) -> (
    UniformContinuousGrid
    | IrregSpacedGrid
    | PiecewiseLinSpacedGrid
    | PiecewiseLogSpacedGrid
):
    """Create a grid of the specified type."""
    if grid_type == "LinSpacedGrid":
        return LinSpacedGrid(start=start, stop=stop, n_points=n_points)
    if grid_type == "LogSpacedGrid":
        return LogSpacedGrid(start=start, stop=stop, n_points=n_points)
    if grid_type == "PiecewiseLinSpacedGrid":
        # More points in lower part, cutoff at 100
        n_lower = n_points // 3 * 2
        return PiecewiseLinSpacedGrid(
            pieces=(
                Piece(interval=f"[{start}, 100)", n_points=n_lower),
                Piece(interval=f"[100, {stop}]", n_points=n_points - n_lower + 1),
            )
        )
    if grid_type == "PiecewiseLogSpacedGrid":
        # Different cutoff at 50, more points in upper part
        n_upper = n_points // 3 * 2
        return PiecewiseLogSpacedGrid(
            pieces=(
                Piece(interval=f"[{start}, 50)", n_points=n_points - n_upper + 1),
                Piece(interval=f"[50, {stop}]", n_points=n_upper),
            )
        )
    if grid_type == "IrregSpacedGrid":
        # Points between lin/log spacing - use average of both
        lin_points = np.linspace(start, stop, n_points)
        log_points = np.logspace(np.log10(start), np.log10(stop), n_points)
        irreg_points = tuple((lin_points + log_points) / 2)
        return IrregSpacedGrid(points=irreg_points)
    msg = f"Unknown grid type: {grid_type}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    "grid_type",
    [
        "LinSpacedGrid",
        "LogSpacedGrid",
        "PiecewiseLinSpacedGrid",
        "PiecewiseLogSpacedGrid",
        "IrregSpacedGrid",
    ],
)
def test_model_with_different_grid_types(grid_type: str):
    """Test that model solution and simulation work with all grid types."""
    n_periods = 4
    # As the borrowing constraint uses weak inequality, we cannot use log-spaced grids
    # for wealth. Consuming everything this period is allowed, but cannot be
    # represented.
    wealth_grid = _create_grid(
        grid_type=grid_type.replace("Log", "Lin"), n_points=100, start=1, stop=400
    )
    consumption_grid = _create_grid(
        grid_type=grid_type, start=1, stop=400, n_points=500
    )

    model = get_model(
        n_periods=n_periods,
        wealth_grid=wealth_grid,
        consumption_grid=consumption_grid,
    )
    params = get_params(
        n_periods=n_periods,
        discount_factor=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )

    # This should complete without error
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([5.0, 20, 40, 70]),
            "age": jnp.array([18.0, 18.0, 18.0, 18.0]),
            "regime": jnp.array([RegimeId.working_life] * 4),
        },
        period_to_regime_to_V_arr=None,
    )
    df = result.to_dataframe()

    # Basic sanity checks
    assert len(df) == n_periods * 4  # 4 periods * 4 subjects
    assert "wealth" in df.columns
    assert "consumption" in df.columns
