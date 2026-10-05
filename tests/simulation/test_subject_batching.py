"""Simulating subjects in chunks must not change any simulated value.

The ExecutionConfig subject width controls how many subjects enter the forward
simulation at once. It is a pure memory knob: the `to_dataframe()` output must be
structurally exact whether subjects run in one pass or in chunks. Only the published
`value` and derived targets, which each chunk's independently compiled kernel
recomputes from exact states and actions, receive eight ordered representable
steps. The model
used here has both a categorical `StochasticTransition` (health) and a continuous shock
process (income), so the per-subject RNG feeds both `jax.random.choice` and
`draw_shock` — the case that would silently diverge if a subject's draws depended on
the chunk it lands in.
"""

import jax
import pandas as pd
import pytest
from jax import numpy as jnp

from lcm import ExecutionConfig, Model
from tests.simulation._profile_comparison import assert_public_frames
from tests.test_models.initial_nodes import initial_nodes_of
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


def _simulate_df(
    *,
    subject_batch_size: int,
    additional_targets: list[str] | None = None,
    repeat: bool = False,
) -> pd.DataFrame:
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    model = Model(
        edges=base.graph.edges,
        regimes=dict(base.user_regimes),
        regime_id_class=MultiRegimeId,
        ages=base.ages,
        fixed_params=dict(base.fixed_params),
        execution_config=ExecutionConfig(
            axis_widths={}
            if subject_batch_size == 0
            else {"subject": subject_batch_size}
        ),
        initial_nodes=initial_nodes_of(model=base),
    )
    params = get_multi_regime_params("normal")
    result = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions=_INITIAL_CONDITIONS,
        seed=42,
    )
    if repeat:
        result = model.simulate(
            log_level="debug",
            params=params,
            initial_conditions=_INITIAL_CONDITIONS,
            seed=42,
        )
    # Compare delivered row identities/order, not a sorted/reset surrogate.
    return result.to_dataframe(additional_targets=additional_targets)


def _assert_columns_invariant(*, baseline: pd.DataFrame, batched: pd.DataFrame) -> None:
    """Separate exact structure from independently compiled published values."""
    assert_public_frames(
        got=batched,
        expected=baseline,
        mode="independently_compiled",
        n_ulp=8,
        value_columns=("value", "utility"),
    )


@pytest.mark.parametrize("subject_batch_size", [2, 3, 100])
def test_simulation_output_is_invariant_to_subject_batch_size(
    subject_batch_size: int,
) -> None:
    """Chunked simulation preserves structure and the published-value budget.

    Across an even split (2 over 7 subjects), an uneven one (3 → 3, 3, 1), and a
    chunk larger than the population (100 → single chunk), every discrete column
    matches the unbatched run exactly, as do continuous states/actions. Only the
    independently compiled published `value` column receives eight ordered steps.
    """
    baseline = _simulate_df(subject_batch_size=0)
    batched = _simulate_df(subject_batch_size=subject_batch_size)
    _assert_columns_invariant(baseline=baseline, batched=batched)


@pytest.mark.parametrize("subject_batch_size", [2, 3, 100])
def test_to_dataframe_targets_are_invariant_to_subject_batch_size(
    subject_batch_size: int,
) -> None:
    """Eagerly computed `additional_targets` are invariant to the chunk size.

    The target DAG (`utility`) is evaluated over the in-regime rows in chunks of
    `subject_batch_size` when set; every split reproduces the single-pass
    `utility` column for every subject-period within the value allowance, and
    the states and actions it is computed from exactly.
    """
    baseline = _simulate_df(subject_batch_size=0, additional_targets=["utility"])
    batched = _simulate_df(
        subject_batch_size=subject_batch_size, additional_targets=["utility"]
    )
    _assert_columns_invariant(baseline=baseline, batched=batched)


@pytest.mark.parametrize("subject_batch_size", [2, 3, 4])
def test_warm_simulation_is_invariant_to_subject_batch_size(
    subject_batch_size: int,
) -> None:
    """Warm chunked calls match the unbatched simulation for every subject-period."""
    baseline = _simulate_df(subject_batch_size=0)
    batched = _simulate_df(subject_batch_size=subject_batch_size, repeat=True)
    _assert_columns_invariant(baseline=baseline, batched=batched)


def test_raw_results_are_host_resident_jax_arrays_when_batched() -> None:
    """With `subject_batch_size` set, `raw_results` leaves are host-backed jax.Arrays.

    Each chunk is offloaded to host as it completes, so the leaves stay `jax.Array`
    (not numpy) and live on the CPU device.
    """
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    model = Model(
        edges=base.graph.edges,
        regimes=dict(base.user_regimes),
        regime_id_class=MultiRegimeId,
        ages=base.ages,
        fixed_params=dict(base.fixed_params),
        # Unbudgeted pinned chunks offload each one; a budget may admit one chunk.
        execution_config=ExecutionConfig(
            axis_widths={"subject": 2}, device_memory_bytes=None
        ),
        initial_nodes=initial_nodes_of(model=base),
    )
    params = get_multi_regime_params("normal")
    result = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions=_INITIAL_CONDITIONS,
        seed=42,
    )

    v_arr = result.raw_results["work"][0].V_arr
    assert isinstance(v_arr, jax.Array)
    assert v_arr.devices() == {jax.devices("cpu")[0]}


def test_additional_target_column_has_a_numeric_dtype() -> None:
    """An eagerly computed `additional_targets` column is float, not object.

    A regime whose target evaluates to one constant for every row contributes a
    scalar rather than a per-row array. It has to reach the frame as a numeric
    value, so the column stays float and supports arithmetic, aggregation and
    round-tripping through Arrow.
    """
    df = _simulate_df(subject_batch_size=0, additional_targets=["utility"])
    assert pd.api.types.is_float_dtype(df["utility"])
