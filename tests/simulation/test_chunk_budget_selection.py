"""Budgeted public chunks are admitted abstractly before padding or execution."""

from typing import Any

import jax.numpy as jnp
import pytest

import _lcm.simulation.simulate as simulation
from _lcm.simulation.runtime import SimulationRuntime
from lcm import ExecutionConfig, Model
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("count", [1, 6])
def test_budgeted_public_chunks_prepare_without_real_prewarm_templates(
    *, monkeypatch: pytest.MonkeyPatch, prewarm: bool, count: int
) -> None:
    base = _stateful_target_model()
    model = Model(
        regimes=dict(base.user_regimes),
        ages=base.ages,
        regime_id_class=_LifecycleRegimeId,
        n_subjects=count if prewarm else None,
        execution_config=ExecutionConfig(
            axis_widths={"subject": 2}, device_memory_bytes=2**32
        ),
    )
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = model.solve(params=params, log_level="off")
    abstract_preparations: list[int] = []
    chunk_sizes: list[int] = []
    prepare = SimulationRuntime.prepare_abstract
    run_chunk = simulation._simulate_subject_chunk

    def observe_abstract(self: SimulationRuntime, **call: Any) -> object:
        abstract_preparations.append(call["n_subjects"])
        return prepare(self, **call)

    def observe_chunk(**call: Any) -> object:
        assert abstract_preparations, (
            "The whole chunk was dispatched before abstract admission"
        )
        chunk_sizes.append(len(call["initial_regime_ids"]))
        return run_chunk(**call)

    def forbid_real_templates(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Budgeted chunk planning entered real-template prewarm")

    monkeypatch.setattr(SimulationRuntime, "prepare_abstract", observe_abstract)
    monkeypatch.setattr(SimulationRuntime, "prepare", forbid_real_templates)
    monkeypatch.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.linspace(1, 3, count),
            "age": jnp.zeros(count),
            "regime_id": jnp.full(count, _LifecycleRegimeId.alive, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    )
    assert result.n_subjects == count
    assert chunk_sizes == ([1] if count == 1 else [2, 2, 2])
    assert set(abstract_preparations) == {min(count, 2)}
