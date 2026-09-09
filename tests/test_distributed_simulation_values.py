"""Simulation reads placed values without changing the owned solution.

Run this module alone so its four-CPU-device topology precedes JAX initialization.
"""

# Test-model declarations must run after the four-device configuration below.
# ruff: noqa: PLC0415

from functools import cache
from typing import cast

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.solution.artifacts import OwnedSolutionView
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.typing import ContinuousState, ScalarFloat, ScalarInt
from tests.conftest import assert_agrees_to_ulp

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_AVAILABLE = True
except RuntimeError:
    _TOPOLOGY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TOPOLOGY_AVAILABLE, reason="Requires a fresh four-CPU-device process"
)


@pytest.mark.parametrize("reverse_order", [False, True])
def test_abstract_read_preserves_required_ordered_device_identity(
    *, monkeypatch: pytest.MonkeyPatch, reverse_order: bool
) -> None:
    """Equal device sets cannot replace the exact required replica ordering."""
    from dataclasses import replace

    from _lcm.execution.core_program import resolve_core_program
    from _lcm.execution.value_transfer import ValueTransferKind, resolve_value_transfer
    from tests.execution.test_abstract_core_program import _inputs

    program, aligned = _inputs()
    devices = (jax.devices()[1], jax.devices()[3])
    required = jax.NamedSharding(
        jax.make_mesh((2,), ("copy",), devices=devices), jax.P()
    )
    observed = jax.NamedSharding(
        jax.make_mesh(
            (2,), ("copy",), devices=devices[::-1] if reverse_order else devices
        ),
        jax.P(),
    )
    transfer = resolve_value_transfer(
        target=aligned.target,
        source=aligned.source,
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_template=program.arguments["extra"],
        source_sharding=required,
    )
    candidate = replace(
        program,
        arguments={
            "next_regime_to_V_arr": {
                "future": jax.ShapeDtypeStruct((3,), jnp.float32, sharding=observed)
            },
            "extra": program.arguments["extra"],
        },
    )

    def forbid_copy(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Abstract metadata resolution allocated a copy")

    monkeypatch.setattr(jax, "device_put", forbid_copy)
    if reverse_order:
        with pytest.raises(ValueError, match="required-sharding"):
            resolve_core_program(
                program=candidate, input_transfer_plan=(transfer,), abstract_inputs=True
            )
    else:
        resolved = resolve_core_program(
            program=candidate, input_transfer_plan=(transfer,), abstract_inputs=True
        )
        assert resolved.input_transfer_plan == (transfer,)
        assert resolved.input_transfer_plan[0].source_sharding == required


def test_chunk_offload_admits_source_destination_overlap_on_selected_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import numpy as np

    from _lcm.simulation.host_operations import ProfiledSimulationOperations
    from _lcm.simulation.memory import SimulationMemory
    from _lcm.simulation.residency import measure_buffer_footprint
    from lcm.exceptions import ExecutionPlanningError
    from tests.simulation.test_population_allocation_budget import _forbid_concrete

    devices = (jax.devices()[1], jax.devices()[3])
    mesh = jax.make_mesh(
        (2,), ("subject",), (jax.sharding.AxisType.Auto,), devices=devices
    )
    original = jax.device_put(
        np.arange(8, dtype=np.int32), jax.NamedSharding(mesh, jax.P("subject"))
    )
    memory = SimulationMemory(
        budget_bytes=24,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=original),
    )
    module = importlib.import_module("_lcm.simulation.chunk_offload")
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_concrete)
        with pytest.raises(ExecutionPlanningError):
            module.offload_chunk(
                tree={"value": original}, host_device=devices[0], memory=memory
            )
    memory.budget_bytes = 1_000_000
    actual = module.offload_chunk(
        tree={"value": original}, host_device=devices[0], memory=memory
    )
    np.testing.assert_array_equal(actual["value"], np.arange(8))
    assert actual["value"].devices() == {devices[0]}
    assert original.devices() == set(devices)
    np.testing.assert_array_equal(original, np.arange(8))


def test_chunk_assembly_cannot_exempt_an_excluded_cpu_from_a_cpu_budget() -> None:
    import numpy as np

    from _lcm.simulation.assembly import concatenate_arrays
    from _lcm.simulation.host_operations import ProfiledSimulationOperations
    from _lcm.simulation.memory import SimulationMemory
    from _lcm.simulation.residency import measure_buffer_footprint
    from lcm.exceptions import ExecutionPlanningError

    source = jax.device_put(np.arange(4, dtype=np.int32), jax.devices()[0])
    devices = (jax.devices()[3],)
    memory = SimulationMemory(
        budget_bytes=1_000_000,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=source),
    )
    with pytest.raises(ExecutionPlanningError, match="budget omits"):
        concatenate_arrays(arrays=(source, source), memory=memory)
    np.testing.assert_array_equal(source, np.arange(4))


def test_chunk_stage_devices_must_match_actual_executable_placement() -> None:
    import numpy as np

    from _lcm.simulation.chunk_planning import SimulationStageProfile
    from lcm.exceptions import ExecutionPlanningError

    def actual_body(*, values: jax.Array) -> jax.Array:
        return values + 1

    placement = jax.sharding.SingleDeviceSharding(jax.devices()[3])
    abstract = jax.ShapeDtypeStruct((4,), np.dtype(np.int32), sharding=placement)
    compiled = jax.jit(actual_body).lower(values=abstract).compile()
    with pytest.raises(ExecutionPlanningError, match="compiled placement"):
        SimulationStageProfile(
            name="actual", executable=compiled, devices=(jax.devices()[1],)
        )
    valid = SimulationStageProfile(
        name="actual", executable=compiled, devices=(jax.devices()[3],)
    )
    assert valid.peak_bytes > 0


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class _ThreeTypes:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class _FourTypes:
    low: ScalarInt
    lower_middle: ScalarInt
    upper_middle: ScalarInt
    high: ScalarInt


def _utility(
    *, wealth: ContinuousState, consumption: ScalarFloat, preference: ScalarInt
) -> ScalarFloat:
    return jnp.log(consumption) + (preference + 1) * wealth * 0.001


def _next_wealth(*, wealth: ContinuousState, consumption: ScalarFloat) -> ScalarFloat:
    return wealth - consumption


def _transition(age: ScalarFloat) -> ScalarInt:
    return jnp.where(age >= 1, _RegimeId.retired, _RegimeId.working)


def _terminal_utility(wealth: ContinuousState) -> ScalarFloat:
    return wealth * 0.5


def _build_model(
    *,
    n_types: int,
    devices: tuple[int, ...],
    sharded: bool,
    prewarm: bool = False,
    subject_width: int | None = None,
    budget: int | None = None,
) -> Model:
    """Build a sharded preference axis beside an unsharded terminal regime."""
    wealth = LinSpacedGrid(start=1, stop=20, n_points=6)
    return Model(
        regimes={
            "working": Regime(
                transition=_transition,
                active=lambda age: age < 3,
                states={"wealth": wealth},
                state_transitions={"wealth": _next_wealth},
                actions={"consumption": LinSpacedGrid(start=1, stop=5, n_points=5)},
                functions={"utility": _utility},
            ),
            "retired": Regime(
                transition=None,
                states={"wealth": wealth},
                functions={"utility": _terminal_utility},
            ),
        },
        states={
            "preference": DiscreteGrid(_ThreeTypes if n_types == 3 else _FourTypes)
        },
        state_transitions={"preference": fixed_transition("preference")},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            sharded_states=("preference",) if sharded else (),
            devices=devices,
            axis_widths={} if subject_width is None else {"subject": subject_width},
            device_memory_bytes=budget,
        ),
        n_subjects=7 if prewarm else None,
    )


@cache
def _simulate(
    *, n_types: int, devices: tuple[int, ...], sharded: bool, prewarm: bool = False
) -> tuple[SolutionResult, SimulationResult, SimulationResult]:
    """Replay the same owned solution twice with a population requiring padding."""
    model = _build_model(
        n_types=n_types, devices=devices, sharded=sharded, prewarm=prewarm
    )
    params = {"discount_factor": 0.95}
    solution = model.solve(params=params, log_level="off")
    initial = {
        "wealth": jnp.full(7, 12.0),
        "age": jnp.zeros(7),
        "preference": jnp.arange(7, dtype=jnp.int32) % n_types,
        "regime_id": jnp.full(7, _RegimeId.working),
    }
    first = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=0,
        log_level="off",
    )
    second = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=0,
        log_level="off",
    )
    return solution, first, second


_PLACEMENTS = [(3, (0, 1, 2, 3)), (4, (0, 1, 2, 3)), (3, (1, 2, 3))]


@pytest.mark.parametrize(("n_types", "devices"), _PLACEMENTS)
@pytest.mark.parametrize("prewarm", [False, True])
def test_placed_solution_simulates_to_the_single_device_result(
    *, n_types: int, devices: tuple[int, ...], prewarm: bool
) -> None:
    """Full and proper submeshes preserve all seven subjects' simulated paths."""
    _, actual, _ = _simulate(
        n_types=n_types, devices=devices, sharded=True, prewarm=prewarm
    )
    _, expected, _ = _simulate(n_types=n_types, devices=(0,), sharded=False)
    got = actual.to_dataframe(use_labels=False)
    want = expected.to_dataframe(use_labels=False)
    pd.testing.assert_index_equal(got.index, want.index)
    pd.testing.assert_index_equal(got.columns, want.columns)
    for column in want:
        if column == "value":
            assert_agrees_to_ulp(
                got=got[column].to_numpy(),
                expected=want[column].to_numpy(),
                n_ulp=8,
                err_msg=f"{column=}, {n_types=}, {devices=}",
            )
        else:
            pd.testing.assert_series_equal(got[column], want[column], check_exact=True)


@pytest.mark.parametrize(("n_types", "devices"), _PLACEMENTS)
def test_simulation_retains_original_value_arrays_on_their_solve_placements(
    *, n_types: int, devices: tuple[int, ...]
) -> None:
    """Published value mappings keep the original arrays alive after repeated replay."""
    solution, first, second = _simulate(n_types=n_types, devices=devices, sharded=True)
    # Public ValueStore reads deliberately return detached copies. The engine's
    # immutable owned view is the original placement/lifetime contract being tested.
    owned = cast("OwnedSolutionView", solution._engine_view)
    for period, values in owned.values.items():
        for regime, value in values.items():
            assert not value.is_deleted()
            assert first.period_to_regime_to_V_arr[period][regime] is value
            assert second.period_to_regime_to_V_arr[period][regime] is value


@pytest.mark.parametrize(("n_types", "devices"), _PLACEMENTS)
def test_repeated_simulation_of_the_same_placed_solution_is_identical(
    *, n_types: int, devices: tuple[int, ...]
) -> None:
    """Temporary transfers from one replay never invalidate the next replay."""
    _, first, second = _simulate(n_types=n_types, devices=devices, sharded=True)
    pd.testing.assert_frame_equal(
        first.to_dataframe(), second.to_dataframe(), check_exact=True
    )


@pytest.mark.parametrize("devices", [(0, 1, 2, 3), (1, 3)])
def test_fixed_inner_subject_width_survives_outer_device_alignment(
    *,
    devices: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W=3 stays three while the outer chunk aligns to four on either mesh."""
    from typing import Any

    import numpy as np

    import _lcm.simulation.simulate as simulation
    from _lcm.simulation.runtime import SimulationRuntime

    model = _build_model(
        n_types=4, devices=devices, sharded=True, subject_width=3, budget=2**32
    )
    params = {"discount_factor": 0.95}
    solution = model.solve(params=params, log_level="off")
    initial = {
        "wealth": jnp.full(7, 12.0),
        "age": jnp.zeros(7),
        "preference": jnp.arange(7, dtype=jnp.int32) % 4,
        "regime_id": jnp.full(7, _RegimeId.working),
    }
    chunks: list[int] = []
    widths: list[int] = []
    run_chunk = simulation._simulate_subject_chunk
    prepare = SimulationRuntime._prepare_materialized

    def observe_chunk(**call: Any) -> object:
        chunks.append(call["initial_regime_ids"].shape[0])
        return run_chunk(**call)

    def observe_width(self: SimulationRuntime, **call: Any) -> object:
        selected = prepare(self, **call)
        if "subject" in selected.widths:
            widths.append(selected.widths["subject"])
        return selected

    with monkeypatch.context() as capture:
        capture.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
        capture.setattr(SimulationRuntime, "_prepare_materialized", observe_width)
        actual = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=0,
            log_level="off",
        )
    assert chunks == [4, 4]
    assert widths
    assert set(widths) == {3}
    assert actual.n_subjects == 7
    assert all(
        array.devices() == {jax.devices()[devices[0]]}
        for array in jax.tree.leaves(actual.raw_results)
    )
    _, expected, _ = _simulate(n_types=4, devices=(0,), sharded=False)
    got, want = (
        actual.to_dataframe(use_labels=False),
        expected.to_dataframe(use_labels=False),
    )
    for name in want:
        if name == "value":
            assert_agrees_to_ulp(
                got=got[name].to_numpy(), expected=want[name].to_numpy(), n_ulp=8
            )
        else:
            pd.testing.assert_series_equal(got[name], want[name], check_exact=True)
    np.testing.assert_array_equal(initial["preference"], np.arange(7) % 4)
