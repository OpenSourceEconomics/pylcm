"""A supported-stack integration witness for one-profile independent admission."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import _lcm.simulation.chunk_admission as admission
import _lcm.simulation.simulate as simulation
from lcm import ExecutionConfig, Model
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


def test_full_cohort_is_profiled_once_before_dispatch_on_each_public_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _stateful_target_model()
    model = Model(
        regimes=dict(base.user_regimes),
        ages=base.ages,
        regime_id_class=_LifecycleRegimeId,
        execution_config=ExecutionConfig(
            devices=(jax.devices()[0].id,),
            axis_widths={"subject": 2},
            simulation_chunk_policy="independent",
            device_memory_bytes=2**32,
        ),
    )
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = model.solve(params=params, log_level="off")
    profiles: list[int] = []
    chunks: list[int] = []
    profile_widths = admission._ChunkProfiler.profile_widths
    run_chunk = simulation._simulate_subject_chunk

    def observe_profile(self: Any, **kwargs: Any) -> Any:
        profiles.append(kwargs["n_subjects"])
        return profile_widths(self, **kwargs)

    def observe_chunk(**kwargs: Any) -> Any:
        # A whole-shape profile must finish before this numerical execution starts.
        assert len(profiles) == len(chunks) + 1
        chunks.append(int(kwargs["initial_regime_ids"].shape[0]))
        return run_chunk(**kwargs)

    monkeypatch.setattr(admission._ChunkProfiler, "profile_widths", observe_profile)
    monkeypatch.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
    for _ in range(2):
        result = model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "wealth": jnp.linspace(1, 3, 7),
                "age": jnp.zeros(7),
                "regime_id": jnp.full(7, _LifecycleRegimeId.alive, dtype=jnp.int32),
            },
            seed=17,
            log_level="off",
        )
        assert result.n_subjects == 7
        del result
    # Compiled programs may be reused; current-call admission is never skipped.
    assert profiles == [7, 7]
    assert chunks == [7, 7]
