"""CPU semantic and GPU behavioral tests for the Mahler & Yum (2024) example.

The CPU tests cover the retirement structure, transition routing, and public input
factory. The GPU tests pin simulated lifecycle moments at
`seed=32`, `n=10000`, so a change in the example's economics appears as a change
in a readable quantity rather than an opaque pickle diff.

The pinned numbers are a joint property of the example's economics and the
engine's numerics, so a change to either moves them and they have to be
re-frozen against a run whose correctness has been argued separately. They are
for the explicit working, retirement, and dead regimes and are reproducible
across GPUs at float64 — a re-freeze needs a fresh run, not a matching device.

The structural invariants at the bottom of the module are the stable half: they
assert relations that hold for any correct solve, so they survive a re-freeze
and are what catches an economics change that a shifted pin would merely
absorb.
"""

from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    RETIREMENT_REGIME,
    START_PARAMS,
    WORKING_REGIME,
    ages,
    create_inputs,
    retirement_period,
    retirement_to_dead_probability,
    retirement_to_retirement_probability,
    working_to_dead_probability,
    working_to_retirement_probability,
    working_to_working_probability,
)
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED

# The full Mahler & Yum solve takes about 25 minutes on a 24 GB M4 Pro. Keep the
# behavioral moments on GPU and use small semantic checks on CPU.
#
# The behavioral-moment assertions below are calibrated to the paper's
# 64-bit replication; under 32-bit the simulation drifts past every band,
# so the moments lose signal.
_GPU_X64_MARKS = (
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="requires GPU"),
    pytest.mark.skipif(
        not X64_ENABLED, reason="moments calibrated for 64-bit precision"
    ),
)


def _gpu_x64(test):
    for mark in _GPU_X64_MARKS:
        test = mark(test)
    return test


_ADDITIONAL_TARGETS = [
    "utility",
    "effort_cost",
    "pension",
    "income",
    "consumption",
    "effort_value",
    "lagged_effort_value",
]


def test_survival_mass_moves_from_working_to_retirement_at_the_boundary():
    """Survivors enter retirement at 65 while death receives the complement."""
    survival_probs = jnp.full((retirement_period + 1, 2, 2), 0.8)
    education = jnp.asarray(1, dtype=jnp.int32)
    health = jnp.asarray(0, dtype=jnp.int32)
    before_period = jnp.asarray(retirement_period - 2, dtype=jnp.int32)
    boundary_period = jnp.asarray(retirement_period - 1, dtype=jnp.int32)
    retired_period = jnp.asarray(retirement_period, dtype=jnp.int32)

    before_retirement = np.array(
        [
            working_to_working_probability(
                period=before_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
            working_to_retirement_probability(
                period=before_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
            working_to_dead_probability(
                period=before_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
        ]
    )
    at_retirement = np.array(
        [
            working_to_working_probability(
                period=boundary_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
            working_to_retirement_probability(
                period=boundary_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
            working_to_dead_probability(
                period=boundary_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
        ]
    )
    in_retirement = np.array(
        [
            retirement_to_retirement_probability(
                period=retired_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
            retirement_to_dead_probability(
                period=retired_period,
                education=education,
                health=health,
                survival_probs=survival_probs,
            ),
        ]
    )

    aaae(before_retirement, [0.8, 0.0, 0.2], decimal=DECIMAL_PRECISION)
    aaae(at_retirement, [0.0, 0.8, 0.2], decimal=DECIMAL_PRECISION)
    aaae(in_retirement, [0.8, 0.2], decimal=DECIMAL_PRECISION)


def test_retirement_split_removes_exactly_the_work_only_dimensions():
    """Retirement drops labor supply and productivity from the choice problem."""
    working_dimensions = set(WORKING_REGIME.states) | set(WORKING_REGIME.actions)
    retirement_dimensions = set(RETIREMENT_REGIME.states) | set(
        RETIREMENT_REGIME.actions
    )

    assert working_dimensions - retirement_dimensions == {
        "labor_supply",
        "productivity",
        "productivity_shock",
    }
    assert retirement_dimensions <= working_dimensions
    working_transition = cast("Mapping[str, object]", WORKING_REGIME.transition)
    retirement_transition = cast("Mapping[str, object]", RETIREMENT_REGIME.transition)
    assert set(working_transition) == {"working", "retirement", "dead"}
    assert set(retirement_transition) == {"retirement", "dead"}


def test_living_regimes_partition_ages_at_65():
    """Every living age belongs to working life or retirement according to age 65."""
    activity = np.array(
        [
            [
                WORKING_REGIME.active(int(age)),
                RETIREMENT_REGIME.active(int(age)),
            ]
            for age in ages.values[:-1]
        ]
    )
    expected = np.array(
        [[age < 65, age >= 65] for age in ages.values[:-1]],
        dtype=bool,
    )

    np.testing.assert_array_equal(activity, expected)


def test_create_inputs_starts_subjects_in_working_regime():
    """The input factory returns shared runtime params and starts work at 25."""
    model_params, initial_conditions = create_inputs(
        seed=0,
        n_simulation_subjects=4,
        params=START_PARAMS,
    )

    assert set(model_params) == {
        "adjustment_cost_envelope",
        "avg_earnings",
        "benefit_rate",
        "discount_factor_by_type",
        "effort_cost_grid",
        "effort_elasticity",
        "gross_interest_rate",
        "health_consumption_penalty",
        "income_normalization",
        "labor_tax_rate",
        "min_consumption",
        "pension_base",
        "pension_replacement_rate",
        "productivity_shock_scale",
        "tax_scale",
        "utility_constant",
        "wagep",
        "work_disutility_grid",
        "y1",
        "yt_s",
        "yt_sq",
    }
    assert initial_conditions["regime_name"].tolist() == ["working"] * 4
    np.testing.assert_array_equal(initial_conditions["age"], np.full(4, 25))


@_gpu_x64
def test_model_solves_and_simulates():
    """Smoke test: model runs end-to-end with small n."""
    model_params, ic_df = create_inputs(
        seed=0, n_simulation_subjects=4, params=START_PARAMS
    )
    result = MAHLER_YUM_MODEL.simulate(
        params=model_params,
        initial_conditions=ic_df,
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="debug",
    )
    df = result.to_dataframe()
    assert len(df) > 0
    assert "period" in df.columns
    assert "wealth" in df.columns
    assert "labor_supply" in df.columns


@pytest.fixture(scope="module")
def simulation_result():
    """Full simulation with START_PARAMS (seed=32, n=10000)."""
    model_params, initial_conditions = create_inputs(
        seed=32, n_simulation_subjects=10000, params=START_PARAMS
    )
    result = MAHLER_YUM_MODEL.simulate(
        params=model_params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=42,
        log_level="debug",
    )
    res = result.to_dataframe(additional_targets=_ADDITIONAL_TARGETS)
    return res[res["regime_name"] != "dead"].copy()


@_gpu_x64
@pytest.mark.parametrize(
    ("period", "expected_retired", "expected_part_time", "expected_full_time"),
    [
        (0, 633, 6219, 3148),
        (1, 653, 7687, 1656),
        (2, 579, 5846, 3559),
        (3, 526, 6376, 3072),
        (4, 473, 5329, 4161),
    ],
)
def test_labor_supply_distribution(
    simulation_result,
    period,
    expected_retired,
    expected_part_time,
    expected_full_time,
):
    """Labor supply counts per period must match reference within tolerance."""
    p = simulation_result[simulation_result["period"] == period]
    vc = p["labor_supply"].value_counts()
    assert abs(vc.get("retired", 0) - expected_retired) <= 5
    assert abs(vc.get("part_time", 0) - expected_part_time) <= 5
    assert abs(vc.get("full_time", 0) - expected_full_time) <= 5


@_gpu_x64
@pytest.mark.parametrize(
    ("period", "expected_mean_wealth"),
    [
        (0, 0.0),
        (5, 0.3038),
        (10, 1.0975),
        (15, 2.3246),
        (20, 2.8767),
        (25, 1.9310),
        (30, 0.9374),
    ],
)
def test_mean_wealth_profile(simulation_result, period, expected_mean_wealth):
    """Mean wealth at key periods must match reference."""
    p = simulation_result[simulation_result["period"] == period]
    np.testing.assert_allclose(p["wealth"].mean(), expected_mean_wealth, atol=0.01)


@_gpu_x64
@pytest.mark.parametrize(
    ("period", "expected_good_frac"),
    [
        (0, 0.9219),
        (10, 0.9112),
        (20, 0.8509),
        (30, 0.7049),
    ],
)
def test_health_good_fraction(simulation_result, period, expected_good_frac):
    """Fraction in good health must decline with age as expected."""
    p = simulation_result[simulation_result["period"] == period]
    np.testing.assert_allclose(
        (p["health"] == "good").mean(), expected_good_frac, atol=0.005
    )


@_gpu_x64
@pytest.mark.parametrize(
    ("period", "expected_alive"),
    [
        (10, 9880),
        (20, 9113),
        (30, 5037),
        (37, 510),
    ],
)
def test_survival_counts(simulation_result, period, expected_alive):
    """Number of surviving agents must match reference."""
    n = len(simulation_result[simulation_result["period"] == period])
    assert abs(n - expected_alive) <= 5


@_gpu_x64
def test_effort_statistics(simulation_result):
    """Mean and std of effort_value across all periods must match reference."""
    np.testing.assert_allclose(
        simulation_result["effort_value"].mean(), 0.8804, atol=0.005
    )
    np.testing.assert_allclose(
        simulation_result["effort_value"].std(), 0.2040, atol=0.005
    )


@_gpu_x64
def test_consumption_by_health(simulation_result):
    """Consumption must be higher for good health than bad health."""
    cons = simulation_result.groupby("health")["consumption"].mean()
    np.testing.assert_allclose(cons.loc["good"], 0.6884, atol=0.005)
    np.testing.assert_allclose(cons.loc["bad"], 0.6080, atol=0.005)
    assert cons.loc["good"] > cons.loc["bad"]


@_gpu_x64
def test_income_by_education(simulation_result):
    """Mean income during working life must be higher for high education."""
    working = simulation_result[simulation_result["period"] < retirement_period]
    inc = working.groupby("education")["income"].mean()
    np.testing.assert_allclose(inc.loc["low"], 0.7995, atol=0.01)
    np.testing.assert_allclose(inc.loc["high"], 1.5526, atol=0.01)
    assert inc.loc["high"] > inc.loc["low"]


@_gpu_x64
def test_retirement_regime_starts_at_retirement_period(simulation_result):
    """Living agents enter the retirement regime at the mandatory age."""
    post_ret = simulation_result[simulation_result["period"] >= retirement_period]
    assert (post_ret["regime_name"] == "retirement").all()
    assert post_ret["labor_supply"].isna().all()
    assert post_ret["income"].isna().all()


@_gpu_x64
def test_total_living_rows(simulation_result):
    """Total number of living-regime rows must match reference."""
    assert abs(len(simulation_result) - 294706) <= 50


@_gpu_x64
def test_wealth_non_negative(simulation_result):
    """Wealth must be non-negative (borrowing constraint)."""
    assert (simulation_result["wealth"] >= -1e-6).all()


@_gpu_x64
def test_consumption_positive(simulation_result):
    """Consumption must be positive."""
    assert (simulation_result["consumption"] > 0).all()
