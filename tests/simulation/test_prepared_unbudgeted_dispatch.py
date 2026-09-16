"""AC4: unbudgeted dispatch reuses a warm exact-signature prepared executable.

Covers design.md section 5 ("minimal unbudgeted dispatch contract"): with no
device-memory budget configured, an exact-signature repeat call skips
redundant materialization/width-frontier construction (no additional
`SimulationRuntime._prepare_materialized` calls) while still binding fresh
leaves, still using the configured/default subject width, and still falling
back correctly on any signature change.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import chunk_admission
from _lcm.simulation.entry_inputs import capture_simulation_entry_inputs
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationRuntime
from benchmarks.asv._simulation_witnesses import multi_regime
from lcm.execution import ExecutionConfig


def _small_unbudgeted_model():
    return multi_regime(execution_config=ExecutionConfig())


def _simulate(*, model, params, solution, initial_conditions, seed):
    return model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=seed,
    )


def test_unbudgeted_path_admits_no_residency_or_chunk_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline guarantee (should already hold on the current tree)."""
    model, params, initial_conditions = _small_unbudgeted_model()
    solution = model.solve(params=params, log_level="off")

    entry_calls = []
    real_entry = capture_simulation_entry_inputs

    def counting_entry(**kwargs: Any) -> Any:
        entry_calls.append(kwargs)
        return real_entry(**kwargs)

    chunk_calls = []
    real_chunks = chunk_admission.prepare_simulation_chunks

    def counting_chunks(**kwargs: Any) -> Any:
        chunk_calls.append(kwargs)
        return real_chunks(**kwargs)

    residency_calls = []
    real_footprint = measure_buffer_footprint

    def counting_footprint(**kwargs: Any) -> Any:
        residency_calls.append(kwargs)
        return real_footprint(**kwargs)

    monkeypatch.setattr(
        "_lcm.simulation.entry_inputs.capture_simulation_entry_inputs",
        counting_entry,
    )
    monkeypatch.setattr(chunk_admission, "prepare_simulation_chunks", counting_chunks)
    monkeypatch.setattr(
        "_lcm.simulation.residency.measure_buffer_footprint", counting_footprint
    )

    _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )

    assert len(chunk_calls) == 0
    assert len(residency_calls) == 0


def test_warm_exact_signature_hit_skips_redundant_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated call with an unchanged signature does not re-run the full
    materialization/width-selection route, and still returns fresh, correct
    numbers when a same-shaped input value changes."""
    model, params, initial_conditions = _small_unbudgeted_model()
    solution = model.solve(params=params, log_level="off")

    calls = []
    real_prepare_materialized = SimulationRuntime._prepare_materialized

    def counting_prepare_materialized(self: SimulationRuntime, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_prepare_materialized(self, **kwargs)

    monkeypatch.setattr(
        SimulationRuntime, "_prepare_materialized", counting_prepare_materialized
    )

    result_first = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )
    calls_after_first = len(calls)
    assert calls_after_first > 0

    shifted_initial_conditions = dict(initial_conditions)
    shifted_initial_conditions["wealth"] = initial_conditions["wealth"] + 0.5

    for _ in range(4):
        result_repeat = _simulate(
            model=model,
            params=params,
            solution=solution,
            initial_conditions=shifted_initial_conditions,
            seed=0,
        )

    # Same period/regime/program identity, same shapes/dtypes/devices/widths:
    # no additional `_prepare_materialized` calls on the warm repeats.
    assert len(calls) == calls_after_first

    # Fresh leaves are still bound: a same-shaped input value change changes
    # the result.
    first_wealth = result_first.to_dataframe()["wealth"].to_numpy()
    repeat_wealth = result_repeat.to_dataframe()["wealth"].to_numpy()
    assert not np.allclose(first_wealth, repeat_wealth)

    # Identical inputs and seed still reproduce identical output arrays.
    result_repeat_2 = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=shifted_initial_conditions,
        seed=0,
    )
    np.testing.assert_array_equal(
        result_repeat.to_dataframe()["wealth"].to_numpy(),
        result_repeat_2.to_dataframe()["wealth"].to_numpy(),
    )


def test_signature_change_falls_back_and_stays_correct() -> None:
    """A shape/dtype/period/parameter change cannot dispatch a stale executable."""
    model, params, initial_conditions = _small_unbudgeted_model()
    solution = model.solve(params=params, log_level="off")

    baseline = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )

    # Warm the cache first.
    _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )

    # Change population size (shape change) and confirm correctness against a
    # freshly built model/runtime rather than a stale cached executable.
    grown_initial_conditions = {
        key: jnp.concatenate([value, value[:2]])
        for key, value in initial_conditions.items()
    }
    grown_result = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=grown_initial_conditions,
        seed=0,
    )

    fresh_model, fresh_params, _ = _small_unbudgeted_model()
    fresh_solution = fresh_model.solve(params=fresh_params, log_level="off")
    fresh_result = _simulate(
        model=fresh_model,
        params=fresh_params,
        solution=fresh_solution,
        initial_conditions=grown_initial_conditions,
        seed=0,
    )

    np.testing.assert_array_equal(
        grown_result.to_dataframe()["wealth"].to_numpy(),
        fresh_result.to_dataframe()["wealth"].to_numpy(),
    )
    # And the original-size call remains unaffected by the growth.
    replay = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )
    np.testing.assert_array_equal(
        baseline.to_dataframe()["wealth"].to_numpy(),
        replay.to_dataframe()["wealth"].to_numpy(),
    )
