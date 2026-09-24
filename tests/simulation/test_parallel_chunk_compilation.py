"""Chunk planning compiles forward programs on worker threads without changing them."""

import threading
from typing import Any, cast

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

import lcm.model as model_module
from _lcm.simulation.chunk_planning import SimulationChunkPlan
from _lcm.simulation.runtime import _SimulationCandidateCompiler
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)

_PARAMS = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}


def _simulate_and_observe(
    *, monkeypatch: pytest.MonkeyPatch, max_compilation_workers: int
) -> dict[str, Any]:
    """Simulate a fresh budgeted model, recording chunk-planning compiles and plan."""
    model = _stateful_target_model()
    initial = {
        "wealth": jnp.asarray([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, _LifecycleRegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params=_PARAMS, log_level="off")
    planning = [False]
    compiles: list[tuple[bool, str, tuple[tuple[str, int], ...], str | None]] = []
    plans: list[SimulationChunkPlan] = []
    lock = threading.Lock()
    compile_body = _SimulationCandidateCompiler.__call__
    prepare = model_module.prepare_simulation_chunks

    def observe_compile(self: _SimulationCandidateCompiler, *args: Any) -> Any:
        (widths,) = args
        compiled = compile_body(self, widths)
        if planning[0]:
            with lock:
                compiles.append(
                    (
                        threading.current_thread() is threading.main_thread(),
                        self.program.name,
                        tuple(sorted(widths.items())),
                        cast("jax.stages.Compiled", compiled.executable).as_text(),
                    )
                )
        return compiled

    def observe_prepare(**call: Any) -> Any:
        planning[0] = True
        try:
            prepared = prepare(**call)
        finally:
            planning[0] = False
        plans.append(prepared.plan)
        return prepared

    with monkeypatch.context() as patch:
        patch.setattr(_SimulationCandidateCompiler, "__call__", observe_compile)
        patch.setattr(model_module, "prepare_simulation_chunks", observe_prepare)
        result = model.simulate(
            params=_PARAMS,
            initial_conditions=initial,
            solution=solution,
            seed=17,
            log_level="off",
            max_compilation_workers=max_compilation_workers,
        )
    (plan,) = plans
    profile = plan.profile
    contract = (
        profile.n_subjects,
        profile.padded_population,
        tuple(
            (stage.name, stage.peak_bytes, stage.reservation_bytes)
            for stage in profile.stages
        ),
        dict(profile.fixed_reservation),
        dict(profile.output_reservation),
        dict(profile.setup_reservation),
        dict(plan.required_bytes),
    )
    return {
        "compiles": compiles,
        "contract": contract,
        "panel": result.to_dataframe(),
    }


@pytest.fixture(scope="module")
def serial_and_parallel() -> dict[int, dict[str, Any]]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        return {
            workers: _simulate_and_observe(
                monkeypatch=monkeypatch, max_compilation_workers=workers
            )
            for workers in (1, 2)
        }


def test_chunk_planning_compiles_every_forward_program_off_the_main_thread(
    serial_and_parallel: dict[int, dict[str, Any]],
) -> None:
    compiles = serial_and_parallel[2]["compiles"]
    assert compiles
    assert not any(on_main for on_main, *_ in compiles)


def test_serial_chunk_planning_compiles_on_the_main_thread(
    serial_and_parallel: dict[int, dict[str, Any]],
) -> None:
    compiles = serial_and_parallel[1]["compiles"]
    assert compiles
    assert all(on_main for on_main, *_ in compiles)


def test_parallel_chunk_planning_compiles_the_same_executables_as_serial(
    serial_and_parallel: dict[int, dict[str, Any]],
) -> None:
    def executables(workers: int) -> list[tuple[str, tuple, str]]:
        return sorted(entry[1:] for entry in serial_and_parallel[workers]["compiles"])

    assert executables(2) == executables(1)


def test_parallel_chunk_planning_admits_the_same_chunk_contract_as_serial(
    serial_and_parallel: dict[int, dict[str, Any]],
) -> None:
    assert serial_and_parallel[2]["contract"] == serial_and_parallel[1]["contract"]


def test_parallel_chunk_planning_simulates_the_same_panel_as_serial(
    serial_and_parallel: dict[int, dict[str, Any]],
) -> None:
    pd.testing.assert_frame_equal(
        serial_and_parallel[2]["panel"], serial_and_parallel[1]["panel"]
    )
