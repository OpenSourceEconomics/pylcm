"""Behavioral-moment tests for the Mahler & Yum (2024) example.

Each test pins a moment of the simulated lifecycle (labor-supply distribution,
wealth profile, health, survival, …) at `seed=32`, `n=10000`, so a change in
the example's economics is caught as a change in a readable quantity rather
than an opaque pickle diff.

Re-frozen on an A100 (2026-08-12, commit 39cd7a3c) after the four correctness fixes
(income normalizer, spline clipping, P(e) pension, nearest-habit rounding) landed on
top of main's latest engine numerics (age-specialization, probability/zero-safe
rework, PRs #398/#414/#416/#418). Every number here moved again; the structural
invariants at the bottom of the module did not.
"""

import jax
import numpy as np
import pytest

from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    create_inputs,
    retirement_period,
)
from tests.conftest import X64_ENABLED

# The Mahler & Yum state-action space is GPU-scale: the backward-induction solve
# exhausts system RAM on CPU regardless of `n_subjects`. Run only where a GPU is
# available.
#
# The behavioral-moment assertions below are calibrated to the paper's
# 64-bit replication; under 32-bit the simulation drifts past every band,
# so the moments lose signal. Skip the whole module at 32-bit precision.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="requires GPU"),
    pytest.mark.skipif(
        not X64_ENABLED, reason="moments calibrated for 64-bit precision"
    ),
]

_ADDITIONAL_TARGETS = [
    "utility",
    "effort_cost",
    "pension",
    "income",
    "consumption",
    "effort_value",
    "lagged_effort_value",
]


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


@pytest.mark.parametrize(
    ("period", "expected_retired", "expected_part_time", "expected_full_time"),
    [
        (0, 633, 6219, 3148),
        (1, 622, 7677, 1697),
        (2, 562, 5869, 3553),
        (3, 518, 6404, 3051),
        (4, 452, 5349, 4162),
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


@pytest.mark.parametrize(
    ("period", "expected_mean_wealth"),
    [
        (0, 0.0),
        (5, 0.3038),
        (10, 1.0975),
        (15, 2.3374),
        (20, 2.8916),
        (25, 1.9310),
        (30, 0.9374),
    ],
)
def test_mean_wealth_profile(simulation_result, period, expected_mean_wealth):
    """Mean wealth at key periods must match reference."""
    p = simulation_result[simulation_result["period"] == period]
    np.testing.assert_allclose(p["wealth"].mean(), expected_mean_wealth, atol=0.01)


@pytest.mark.parametrize(
    ("period", "expected_good_frac"),
    [
        (0, 0.9219),
        (10, 0.9112),
        (20, 0.8450),
        (30, 0.7049),
    ],
)
def test_health_good_fraction(simulation_result, period, expected_good_frac):
    """Fraction in good health must decline with age as expected."""
    p = simulation_result[simulation_result["period"] == period]
    np.testing.assert_allclose(
        (p["health"] == "good").mean(), expected_good_frac, atol=0.005
    )


@pytest.mark.parametrize(
    ("period", "expected_alive"),
    [
        (10, 9871),
        (20, 9140),
        (30, 5063),
        (37, 510),
    ],
)
def test_survival_counts(simulation_result, period, expected_alive):
    """Number of surviving agents must match reference."""
    n = len(simulation_result[simulation_result["period"] == period])
    assert abs(n - expected_alive) <= 5


def test_effort_statistics(simulation_result):
    """Mean and std of effort_value across all periods must match reference."""
    np.testing.assert_allclose(
        simulation_result["effort_value"].mean(), 0.8804, atol=0.005
    )
    np.testing.assert_allclose(
        simulation_result["effort_value"].std(), 0.2040, atol=0.005
    )


def test_consumption_by_health(simulation_result):
    """Consumption must be higher for good health than bad health."""
    cons = simulation_result.groupby("health")["consumption"].mean()
    np.testing.assert_allclose(cons.loc["good"], 0.6884, atol=0.005)
    np.testing.assert_allclose(cons.loc["bad"], 0.6080, atol=0.005)
    assert cons.loc["good"] > cons.loc["bad"]


def test_income_by_education(simulation_result):
    """Mean income during working life must be higher for high education."""
    working = simulation_result[simulation_result["period"] < retirement_period]
    inc = working.groupby("education")["income"].mean()
    np.testing.assert_allclose(inc.loc["low"], 0.7995, atol=0.01)
    np.testing.assert_allclose(inc.loc["high"], 1.5526, atol=0.01)
    assert inc.loc["high"] > inc.loc["low"]


def test_retirement_regime_starts_at_retirement_period(simulation_result):
    """Living agents enter the retirement regime at the mandatory age."""
    post_ret = simulation_result[simulation_result["period"] >= retirement_period]
    assert (post_ret["regime_name"] == "retirement").all()
    assert post_ret["labor_supply"].isna().all()
    assert post_ret["income"].isna().all()


def test_total_living_rows(simulation_result):
    """Total number of living-regime rows must match reference."""
    assert abs(len(simulation_result) - 294959) <= 50


def test_wealth_non_negative(simulation_result):
    """Wealth must be non-negative (borrowing constraint)."""
    assert (simulation_result["wealth"] >= -1e-6).all()


def test_consumption_positive(simulation_result):
    """Consumption must be positive."""
    assert (simulation_result["consumption"] > 0).all()
