"""Simulating subjects in chunks must not change any simulated value.

`subject_batch_size` controls how many subjects are pushed through the forward
simulation at once. It is a pure memory knob: the `to_dataframe()` output must be
identical whether subjects run in a single pass (`None`) or in chunks. The model
used here has both a categorical `MarkovTransition` (health) and a continuous shock
process (income), so the per-subject RNG feeds both `jax.random.choice` and
`draw_shock` — the case that would silently diverge if a subject's draws depended on
the chunk it lands in.
"""

import jax
import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

from tests.conftest import DECIMAL_PRECISION
from tests.test_models.processes import (
    MultiRegimeId,
    get_multi_regime_model,
    get_multi_regime_params,
)

_INITIAL_CONDITIONS = {
    "health": jnp.array([0, 1, 0, 1, 0, 1, 0], dtype=jnp.int32),
    "income": jnp.array([0.0, 0.5, -0.3, 0.2, 0.1, -0.1, 0.4]),
    "wealth": jnp.array([1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 1.2]),
    "age": jnp.zeros(7),
    "regime_id": jnp.full(7, MultiRegimeId.work, dtype=jnp.int32),
}


def _simulate_df(*, subject_batch_size: int | None) -> pd.DataFrame:
    model = get_multi_regime_model(n_periods=6, distribution_type="normal")
    params = get_multi_regime_params("normal")
    result = model.simulate(
        log_level="off",
        params=params,
        initial_conditions=_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=None,
        seed=42,
        subject_batch_size=subject_batch_size,
    )
    return (
        result.to_dataframe()
        .sort_values(["subject_id", "period"])
        .reset_index(drop=True)
    )


def _assert_columns_invariant(baseline: pd.DataFrame, batched: pd.DataFrame) -> None:
    assert list(batched.columns) == list(baseline.columns)
    for column in baseline.columns:
        if pd.api.types.is_float_dtype(baseline[column]):
            # `aaae` treats matching NaN positions as equal, so dead-regime rows
            # (NaN states/actions) compare cleanly.
            aaae(
                batched[column].to_numpy(),
                baseline[column].to_numpy(),
                decimal=DECIMAL_PRECISION,
            )
        else:
            # NaN-aware exact comparison for discrete/label columns.
            pd.testing.assert_series_equal(batched[column], baseline[column])


@pytest.mark.parametrize("subject_batch_size", [2, 3, 100])
def test_simulation_output_is_invariant_to_subject_batch_size(
    subject_batch_size: int,
) -> None:
    """Chunked simulation reproduces the single-pass `to_dataframe()` exactly.

    Across an even split (2 over 7 subjects), an uneven one (3 → 3, 3, 1), and a
    chunk larger than the population (100 → single chunk), every discrete column
    matches the unbatched run exactly and every continuous column to
    `DECIMAL_PRECISION`.
    """
    baseline = _simulate_df(subject_batch_size=None)
    batched = _simulate_df(subject_batch_size=subject_batch_size)
    _assert_columns_invariant(baseline, batched)


def test_raw_results_are_host_resident_jax_arrays_when_batched() -> None:
    """With `subject_batch_size` set, `raw_results` leaves are host-backed jax.Arrays.

    Each chunk is offloaded to host as it completes, so the leaves stay `jax.Array`
    (not numpy) and live on the CPU device.
    """
    model = get_multi_regime_model(n_periods=6, distribution_type="normal")
    params = get_multi_regime_params("normal")
    result = model.simulate(
        log_level="off",
        params=params,
        initial_conditions=_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=None,
        seed=42,
        subject_batch_size=2,
    )

    v_arr = result.raw_results["work"][0].V_arr
    assert isinstance(v_arr, jax.Array)
    assert v_arr.devices() == {jax.devices("cpu")[0]}
