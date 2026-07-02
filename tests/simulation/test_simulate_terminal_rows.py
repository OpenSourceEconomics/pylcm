"""The `terminal_rows` knob on `SimulationResult.to_dataframe`.

A subject that enters a terminal regime keeps its frozen state and is
re-emitted in every later period the regime is active. `to_dataframe`
collapses those post-entry duplicates by default (`terminal_rows="first"`);
`"all"` emits the full absorbing representation.
"""

import jax.numpy as jnp
import pandas as pd
import pytest

from lcm import PowerMean
from lcm.result import SimulationResult
from lcm_examples.epstein_zin import EZRegimeId, HealthStatus, get_model, get_params

N_PERIODS = 4
N_SUBJECTS = 20


@pytest.fixture(scope="module")
def result() -> SimulationResult:
    model = get_model(certainty_equivalent=PowerMean())
    return model.simulate(
        params=get_params(risk_aversion=0.5),
        initial_conditions={
            "age": jnp.full(N_SUBJECTS, 60.0),
            "wealth": jnp.linspace(1.0, 10.0, N_SUBJECTS),
            "health": jnp.full(N_SUBJECTS, HealthStatus.good, dtype=jnp.int32),
            "regime_id": jnp.full(N_SUBJECTS, EZRegimeId.alive, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=42,
    )


def test_first_keeps_exactly_one_terminal_row_per_subject(result: SimulationResult):
    """Each subject that enters `dead` appears there exactly once."""
    df = result.to_dataframe(terminal_rows="first")
    dead_rows_per_subject = (
        df.query("regime_name == 'dead'").groupby("subject_id").size()
    )
    assert (dead_rows_per_subject == 1).all()
    assert len(dead_rows_per_subject) == N_SUBJECTS  # survival ends at 0.0


def test_first_is_all_minus_post_entry_terminal_rows(result: SimulationResult):
    """`"first"` equals `"all"` with each subject's post-entry `dead` rows dropped."""
    df_all = result.to_dataframe(terminal_rows="all")
    entry_period = df_all["subject_id"].map(
        df_all.query("regime_name == 'dead'").groupby("subject_id")["period"].min()
    )
    expected = df_all.loc[df_all["period"] <= entry_period].reset_index(drop=True)
    df_first = result.to_dataframe(terminal_rows="first")
    pd.testing.assert_frame_equal(df_first, expected)


def test_all_emits_every_active_period(result: SimulationResult):
    """`"all"` keeps the absorbing representation: one row per subject and period."""
    df = result.to_dataframe(terminal_rows="all")
    assert len(df) == N_SUBJECTS * N_PERIODS


def test_first_is_the_default(result: SimulationResult):
    """`to_dataframe()` collapses post-entry terminal rows by default."""
    pd.testing.assert_frame_equal(
        result.to_dataframe(), result.to_dataframe(terminal_rows="first")
    )


def test_first_composes_with_integer_codes(result: SimulationResult):
    """The filter applies identically under `use_labels=False`."""
    df = result.to_dataframe(use_labels=False, terminal_rows="first")
    dead_rows_per_subject = (
        df.loc[df["regime_name"] == int(EZRegimeId.dead)].groupby("subject_id").size()
    )
    assert (dead_rows_per_subject == 1).all()
