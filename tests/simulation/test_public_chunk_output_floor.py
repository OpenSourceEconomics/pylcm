"""A real public call cannot narrow away its complete retained CPU result bank."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import _lcm.simulation.simulate as simulation
from _lcm.simulation.chunk_admission import _ChunkProfiler
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


def test_public_chunk_selection_refuses_the_irreducible_retained_output_floor(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All candidate extents are profiled, and no chunk starts below the result bank."""
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.linspace(1, 3, 64),
        "age": jnp.zeros(64),
        "regime_id": jnp.full(64, _LifecycleRegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    full_chunks: list[int] = []
    run_chunk = simulation._simulate_subject_chunk

    def observe_chunk(**call: Any) -> object:
        full_chunks.append(call["initial_regime_ids"].shape[0])
        return run_chunk(**call)

    with monkeypatch.context() as capture:
        capture.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
        expected = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            seed=3,
            log_level="off",
        )
    assert full_chunks == [64]
    # Independent publication-slot oracle: every field of every retained record,
    # including rows outside a regime's current membership, is a future slot.
    output_slots = sum(
        leaf.nbytes
        for periods in expected.raw_results.values()
        for record in periods.values()
        for leaf in jax.tree.leaves(
            tuple(getattr(record, field.name) for field in dataclasses.fields(record))
        )
    )
    budgeted = Model(
        regimes=dict(model.user_regimes),
        ages=model.ages,
        regime_id_class=_LifecycleRegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=output_slots - 1),
    )
    budgeted_solution = budgeted.solve(params=params, log_level="off")
    candidates: list[int] = []
    profile = _ChunkProfiler.__call__

    # keyword-only-exempt: library-callback=_ChunkProfiler.__call__
    def observe_profile(self: _ChunkProfiler, *, n_subjects: int) -> object:
        candidates.append(n_subjects)
        return profile(self, n_subjects=n_subjects)

    def forbid_chunk(**_call: object) -> object:
        raise AssertionError("A chunk allocated before its retained output bank fit")

    monkeypatch.setattr(_ChunkProfiler, "__call__", observe_profile)
    monkeypatch.setattr(simulation, "_simulate_subject_chunk", forbid_chunk)
    with pytest.raises(ExecutionPlanningError, match="No declared simulation chunk"):
        budgeted.simulate(
            params=params,
            solution=budgeted_solution,
            initial_conditions=initial,
            seed=3,
            log_level="off",
        )
    assert candidates == [64, 32, 16, 8, 4, 2, 1]
