"""The complete public profile prepares the same stages actual chunks dispatch."""

import importlib
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

import _lcm.simulation.simulate as simulation
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.transitions import _advance_states_for_subjects
from _lcm.solution.artifacts import OwnedSolutionView
from lcm import DiscreteGrid, ExecutionConfig, Model, Regime, categorical
from lcm.ages import AgeGrid
from lcm.typing import ScalarInt
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


@categorical(ordered=False)
class _Flag:
    zero: ScalarInt
    one: ScalarInt


def _flag_utility(flag: jax.Array) -> jax.Array:
    return flag + 1.0


def _promote_flag(flag: jax.Array) -> jax.Array:
    return flag.astype(jnp.int64)


def _keep_flag(flag: jax.Array) -> jax.Array:
    return flag


def _finish_regime() -> ScalarInt:
    return _LifecycleRegimeId.done


def test_profile_preserves_same_kind_categorical_storage_dtype(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-kind integer updates retain the carrier's canonical integer storage."""
    model = Model(
        regimes={
            "alive": Regime(
                transition=_finish_regime,
                active=lambda age: age == 0,
                functions={"utility": _flag_utility},
            ),
            "done": Regime(transition=None, functions={"utility": _flag_utility}),
        },
        states={"flag": DiscreteGrid(_Flag)},
        state_transitions={
            "flag": _promote_flag if jax.config.x64_enabled else _keep_flag
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_LifecycleRegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=2**32),
    )
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off")
    start_counts: list[tuple[SimulationRuntime, int]] = []
    run_chunk = simulation._simulate_subject_chunk

    def observe_chunk(**call: Any) -> object:
        runtime = next(iter(call["regimes"].values())).simulation.programs.executor
        assert isinstance(runtime, SimulationRuntime)
        start_counts.append((runtime, len(runtime.cache)))
        return run_chunk(**call)

    monkeypatch.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "flag": jnp.asarray([0, 1], dtype=jnp.int32),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, _LifecycleRegimeId.alive, dtype=jnp.int32),
        },
        seed=3,
        log_level="off",
    )
    assert result.raw_results["done"][0].states["flag"].dtype == jnp.int32
    assert start_counts
    assert all(count == len(runtime.cache) for runtime, count in start_counts), (
        "Promoted consumers compiled only after the outer admission"
    )


def test_unit_profile_uses_real_merged_carrier_descriptors() -> None:
    """Metadata follows actual cross-kind merge outputs, without grid-dtype guesses."""

    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = model.solve(params=params, log_level="off")
    regimes = model._runtime_regimes_for_shape(compile_batch_size=2)
    runtime = next(iter(regimes.values())).simulation.programs.executor
    assert isinstance(runtime, SimulationRuntime)
    layout = jax.sharding.SingleDeviceSharding(runtime.subject_devices[0])
    before = MappingProxyType(
        {
            name: MappingProxyType(
                {"wealth": jax.ShapeDtypeStruct((2,), jnp.int32, sharding=layout)}
            )
            for name in regimes
        }
    )
    updates = MappingProxyType(
        {
            name: MappingProxyType(
                {"wealth": jax.ShapeDtypeStruct((2,), jnp.float32, sharding=layout)}
            )
            for name in regimes
        }
    )
    merge = runtime.operations.prepare_abstract(
        function=_advance_states_for_subjects,
        arguments={
            "states_per_regime": before,
            "next_states_per_regime": updates,
            "subject_indices": jax.ShapeDtypeStruct((2,), jnp.bool_, sharding=layout),
        },
        subject_arg_names=(
            "states_per_regime",
            "next_states_per_regime",
            "subject_indices",
        ),
        devices=runtime.subject_devices,
    )
    columns = merge.executable.out_info["alive"]
    assert columns["wealth"].dtype == jnp.float32
    flat_params = model._process_params(params)
    unit = getattr(
        importlib.import_module("_lcm.simulation.forward_program_profiles"),
        "profile_forward_unit",
        None,
    )
    assert callable(unit), (
        "Forward metadata needs a unit boundary consuming the actual current carrier"
    )
    profiles = unit(
        runtime=runtime,
        regime=regimes["alive"],
        name="alive",
        period=0,
        flat_params=flat_params,
        base=regimes["alive"].solution.state_action_space(
            regime_params=flat_params["alive"]
        ),
        values=cast("OwnedSolutionView", solution._engine_view).values,
        ages=model.ages,
        n_subjects=2,
        widths={"subject": 2},
        columns=columns,
        ordinary_key=jax.ShapeDtypeStruct((), jax.random.key(0).dtype, sharding=layout),
        taste_key=None,
    )
    wealth = profiles["decision"].arguments["wealth"]
    assert wealth.dtype == columns["wealth"].dtype
    assert wealth.shape == columns["wealth"].shape
    assert wealth.sharding == columns["wealth"].sharding


def test_profiled_public_chunks_need_no_additional_core_compilation(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _stateful_target_model()
    model = Model(
        regimes=dict(base.user_regimes),
        ages=base.ages,
        regime_id_class=_LifecycleRegimeId,
        execution_config=ExecutionConfig(
            axis_widths={"subject": 2}, device_memory_bytes=2**32
        ),
    )
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = model.solve(params=params, log_level="off")
    before: list[tuple[SimulationRuntime, int]] = []
    actual_chunk = simulation._simulate_subject_chunk

    def observe_chunk(**call: Any) -> object:
        runtime = next(iter(call["regimes"].values())).simulation.programs.executor
        assert isinstance(runtime, SimulationRuntime)
        before.append((runtime, len(runtime.cache)))
        return actual_chunk(**call)

    monkeypatch.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.linspace(1, 3, 5),
            "age": jnp.zeros(5),
            "regime_id": jnp.full(5, _LifecycleRegimeId.alive, dtype=jnp.int32),
        },
        seed=17,
        log_level="debug",
    )
    assert result.n_subjects == 5
    assert len(before) == 3
    assert all(count == len(runtime.cache) for runtime, count in before)


def test_nongated_chunk_does_not_allocate_an_unused_edge_age(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = model.solve(params=params, log_level="off")
    actual_chunk = simulation._simulate_subject_chunk
    active = [False]
    get_age = AgeGrid.period_to_age

    # keyword-only-exempt: library-callback=AgeGrid.period_to_age
    def guarded_age(self: AgeGrid, period: int) -> int | float:
        assert not active[0], (
            "Nongated chunk allocated the unused next-edge age outside admission"
        )
        return get_age(self, period)

    def observe_chunk(**call: Any) -> object:
        active[0] = True
        try:
            return actual_chunk(**call)
        finally:
            active[0] = False

    monkeypatch.setattr(AgeGrid, "period_to_age", guarded_age)
    monkeypatch.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray([1.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, _LifecycleRegimeId.alive, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    )
    assert result.n_subjects == 2
