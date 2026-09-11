"""Execution widths bound the complete forward population dispatched at once."""

import inspect
from typing import Any

import jax.numpy as jnp
import pytest

import _lcm.simulation.simulate as simulation
from lcm import ExecutionConfig, LinSpacedGrid, Model
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)


def test_subject_batch_size_is_not_a_public_simulation_argument() -> None:
    """A model's execution configuration is the public subject-width control."""
    assert "subject_batch_size" not in inspect.signature(Model.simulate).parameters


def test_a_fixed_subject_width_bounds_complete_dispatched_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seven real rows execute in three chunks of three, with padding hidden."""
    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        execution_config=ExecutionConfig(axis_widths={"subject": 3}),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    chunks: list[int] = []
    windows: list[tuple[int, int]] = []
    run_chunk = simulation._simulate_subject_chunk

    def observe_chunk(**kwargs: Any) -> object:
        chunks.append(int(kwargs["initial_regime_ids"].shape[0]))
        window = kwargs["subject_slice"]
        windows.append((window.start, window.stop))
        assert kwargs["original_n_subjects"] == 7
        return run_chunk(**kwargs)

    monkeypatch.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.linspace(1.0, 3.0, 7),
            "age": jnp.full(7, 18.0),
            "regime_id": jnp.full(7, RegimeId.working_life, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    )
    assert result.n_subjects == 7
    assert chunks == [3, 3, 3]
    assert windows == [(0, 3), (3, 6), (6, 9)]
