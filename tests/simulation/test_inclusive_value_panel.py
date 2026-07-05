"""End-to-end test of the opt-in smoothed-choice-probability diagnostic.

Solves and simulates the deterministic regression model, then checks that
`SimulationResult.inclusive_value_panel` returns a valid distribution over the
`labor_supply` levels per subject-period and that, as the temperature falls, the
most-probable level matches the hard choice the simulation actually realized.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.simulation import inclusive_values
from lcm import LinSpacedGrid
from tests.test_models.deterministic.regression import (
    START_AGE,
    RegimeId,
    get_model,
    get_params,
)

_N_PERIODS = 4
_N_SUBJECTS = 8
_SEED = 42


@pytest.fixture
def simulation_result():
    model = get_model(
        _N_PERIODS,
        wealth_grid=LinSpacedGrid(start=1, stop=50, n_points=20),
        consumption_grid=LinSpacedGrid(start=1, stop=50, n_points=15),
    )
    params = get_params(_N_PERIODS)
    return model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "wealth": jnp.full(_N_SUBJECTS, 5.0),
            "age": jnp.full(_N_SUBJECTS, float(START_AGE)),
            "regime_id": jnp.array([RegimeId.working_life] * _N_SUBJECTS),
        },
        period_to_regime_to_V_arr=None,
        seed=_SEED,
    )


def test_panel_probabilities_form_a_distribution_per_subject_period(simulation_result):
    """Each subject-period's labor-supply probabilities sum to one."""
    panel = simulation_result.inclusive_value_panel(
        tau=0.1, choice_action="labor_supply"
    )

    totals = panel.groupby(["subject", "period"])["probability"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-6)


def test_panel_argmax_level_matches_hard_choice_as_tau_shrinks(simulation_result):
    """At a small temperature the top-probability level is the realized choice."""
    panel = simulation_result.inclusive_value_panel(
        tau=1e-3, choice_action="labor_supply"
    )
    realized = simulation_result.to_dataframe(use_labels=False)

    top_level = (
        panel.loc[panel.groupby(["subject", "period"])["probability"].idxmax()]
        .set_index(["subject", "period"])["level"]
        .sort_index()
    )
    realized_work = (
        realized[realized["regime_name"] == "working_life"]
        .set_index(["subject_id", "period"])["labor_supply"]
        .sort_index()
    )

    aligned = top_level.reindex(realized_work.index)
    np.testing.assert_array_equal(aligned.to_numpy(), realized_work.to_numpy())


def test_panel_is_invariant_to_subject_batch_size(simulation_result, monkeypatch):
    """Chunking the recompute over subjects gives the same panel as one batch."""
    full = simulation_result.inclusive_value_panel(
        tau=0.1, choice_action="labor_supply"
    )
    monkeypatch.setattr(inclusive_values, "_SUBJECT_BATCH_SIZE", 3)
    batched = simulation_result.inclusive_value_panel(
        tau=0.1, choice_action="labor_supply"
    )

    sort_keys = ["subject", "period", "level"]
    pd.testing.assert_frame_equal(
        full.sort_values(sort_keys).reset_index(drop=True),
        batched.sort_values(sort_keys).reset_index(drop=True),
    )


def test_panel_has_expected_columns(simulation_result):
    """The diagnostic panel is tidy: subject, period, regime, level, probability."""
    panel = simulation_result.inclusive_value_panel(
        tau=0.1, choice_action="labor_supply"
    )

    assert list(panel.columns) == [
        "subject",
        "period",
        "regime",
        "level",
        "probability",
    ]
