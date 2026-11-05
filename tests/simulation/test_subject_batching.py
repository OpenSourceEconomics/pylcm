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

import gc
import weakref

import jax
import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp

from _lcm.params.edges import regime_kernel_params
from _lcm.simulation.additional_targets import _compute_targets
from lcm import ExecutionConfig, Model
from lcm.typing import BoolND, ContinuousState, FloatND, FunctionName, IntND
from tests.conftest import DECIMAL_PRECISION
from tests.simulation._profile_comparison import assert_public_frames
from tests.test_model import _output_model, _output_utility
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
    additional_targets: list[FunctionName] | None = None,
    repeat: bool = False,
) -> pd.DataFrame:
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    model = Model(
        edges=base.edges,
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
        edges=base.edges,
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


def _offset_report(*, wealth: ContinuousState, offset: FloatND) -> FloatND:
    return wealth + offset


def _constant_report() -> FloatND:
    return jnp.asarray(0.5)


def _log_report(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth - 0.5)


def _positive_report(*, wealth: ContinuousState) -> BoolND:
    return wealth > 1.25


def _integer_report(*, wealth: ContinuousState) -> IntND:
    return (wealth > 1.25).astype(jnp.int32)


def _deferred_target_model() -> Model:
    return _output_model(
        functions={
            "utility": _output_utility,
            "offset_report": _offset_report,
            "constant_report": _constant_report,
            "log_report": _log_report,
            "positive_report": _positive_report,
            "integer_report": _integer_report,
        },
        regime_output=(
            "offset_report",
            "constant_report",
            "log_report",
            "positive_report",
            "integer_report",
        ),
    )


@pytest.mark.parametrize(
    ("n_rows", "capacity"),
    [(0, 4), (1, 4), (3, 4), (4, 4), (5, 4), (8, 4), (9, 4), (3, None), (3, 0)],
)
@pytest.mark.parametrize("leaf_kind", ["numpy", "jax"])
def test_deferred_targets_preserve_real_rows_and_dtypes(
    *, n_rows: int, capacity: int | None, leaf_kind: str
) -> None:
    """Deferred targets preserve real values, domains and types at every split."""
    float_dtype = np.float64 if jax.config.x64_enabled else np.float32
    wealth = 1.0 + np.arange(n_rows, dtype=float_dtype) / 32.0
    source = wealth if leaf_kind == "numpy" else jnp.asarray(wealth)
    model = _deferred_target_model()
    params = model._canonicalize_params(params={"offset": 0.25})
    targets = tuple(model.user_regimes["alive"].output_only_functions)
    with jax.debug_nans(new_val=True):
        result = _compute_targets(
            data={"period": np.zeros(n_rows, dtype=np.int32), "wealth": source},
            targets=targets,
            regime=model._regimes["alive"],
            regime_params=regime_kernel_params(params, regime_name="alive"),
            subject_batch_size=capacity,
        )
    expected = {
        "offset_report": wealth + float_dtype(0.25),
        "constant_report": np.full(n_rows, 0.5, dtype=float_dtype),
        "positive_report": wealth > 1.25,
        "integer_report": (wealth > 1.25).astype(np.int32),
    }
    for name, values in expected.items():
        actual = np.asarray(result[name])
        np.testing.assert_array_equal(actual, values)
        assert actual.dtype == values.dtype
    log_values = np.asarray(result["log_report"])
    np.testing.assert_array_almost_equal(
        log_values, np.log(wealth - float_dtype(0.5)), decimal=DECIMAL_PRECISION
    )
    assert log_values.dtype == np.dtype(float_dtype)
    assert log_values.shape == (n_rows,)
    np.testing.assert_array_equal(np.asarray(source), wealth)


@pytest.mark.parametrize("persistent_compilation_cache", [False], indirect=True)
def test_deferred_targets_reuse_capacity_and_release_caller_storage(
    *, persistent_compilation_cache: bool
) -> None:
    """Current values reuse fixed-capacity code without retaining caller storage."""
    assert not persistent_compilation_cache
    float_dtype = np.float64 if jax.config.x64_enabled else np.float32
    model = _deferred_target_model()
    params = [
        regime_kernel_params(
            model._canonicalize_params(params={"offset": offset}), regime_name="alive"
        )
        for offset in (0.25, 0.5, 0.75, 0.125)
    ]
    inputs = [
        jnp.asarray(1.0 + np.arange(n_rows, dtype=float_dtype) / 32.0)
        for n_rows in (4, 3, 5, 9)
    ]
    references = [weakref.ref(value) for value in inputs]
    events: list[str] = []

    # keyword-only-exempt: library-callback=jax.monitoring
    def record_compile(
        event: str, start_time: float, end_time: float, **metadata: str | int
    ) -> None:
        del start_time, end_time
        if event == "/jax/core/compile/backend_compile_duration":
            events.append(str(metadata["fun_name"]))

    counts = []
    outputs = []
    jax.clear_caches()
    jax.monitoring.register_event_time_span_listener(record_compile)
    try:
        for source, flat_params, offset in zip(
            inputs, params, (0.25, 0.5, 0.75, 0.125), strict=True
        ):
            events.clear()
            output = _compute_targets(
                data={
                    "period": np.zeros(source.shape[0], dtype=np.int32),
                    "wealth": source,
                },
                targets=("offset_report",),
                regime=model._regimes["alive"],
                regime_params=flat_params,
                subject_batch_size=4,
            )["offset_report"]
            jax.block_until_ready(output)
            counts.append(len(events))
            assert isinstance(output, np.ndarray)
            assert output.base is None
            expected = 1.0 + np.arange(source.shape[0], dtype=float_dtype) / 32.0
            np.testing.assert_array_equal(output, expected + float_dtype(offset))
            assert output.nbytes == source.shape[0] * np.dtype(float_dtype).itemsize
            outputs.append(output)
    finally:
        jax.monitoring.unregister_event_time_span_listener(record_compile)
    del source, inputs
    gc.collect()
    assert all(reference() is None for reference in references)
    assert counts[0] > 0, "The actual target must exercise a cold compilation."
    assert counts[1:] == [0, 0, 0], counts
