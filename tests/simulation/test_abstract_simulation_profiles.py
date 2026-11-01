"""Prewarming uses declared abstract operands without allocating a population."""

from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Never, Unpack, cast

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.policy_queries as query_module
import lcm.model as model_module
from _lcm.execution.core_program import CoreBuildContext, MaterializedCoreProgram
from _lcm.execution.workspace_planning import compiler_peak_bytes
from _lcm.simulation.chunk_profile_inventory import abstract_tree
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import (
    CompiledSimulationProgram,
    SimulationRuntime,
    _simulation_lowering_key,
)
from _lcm.simulation.simulate import _lookup_values_from_indices
from _lcm.typing import ArgumentTree, PytreeValue, ShapeDtypePytree
from lcm import ExecutionConfig, LinSpacedGrid
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ReferenceName
from lcm_examples import collective_household as household
from tests.simulation._callback_types import (
    ChunkPreparation,
    ChunkPreparationResult,
    RuntimeDispatch,
    RuntimePreparation,
)
from tests.test_models.deterministic.regression import RegimeId, get_model, get_params


class _ConcreteAllocationError(AssertionError):
    """Identify an allocating dispatch while only abstract operands are allowed."""


def _forbid_allocation[Ignored](*_args: Ignored, **_kwargs: Ignored) -> Never:
    raise _ConcreteAllocationError("An abstract profile tried to allocate an operand")


def _abstract[Leaf](leaf: Leaf) -> Leaf | jax.ShapeDtypeStruct:
    if isinstance(leaf, jax.Array):
        assert not leaf.is_deleted()
        return jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=leaf.sharding, weak_type=leaf.weak_type
        )
    return leaf


def test_an_actual_decision_compiles_from_abstract_arguments_without_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real continuation-reading decision profiles before any subject upload."""
    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        # The unbudgeted dispatch is the reference the abstract compile reproduces.
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3, "action_product": 2}, device_memory_bytes=None
        ),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    recorded: list[tuple[SimulationRuntime, RuntimeDispatch, PytreeValue]] = []
    dispatch = SimulationRuntime.dispatch

    def observe(
        self: SimulationRuntime, **kwargs: Unpack[RuntimeDispatch]
    ) -> PytreeValue:
        result = dispatch(self, **kwargs)
        if not recorded and kwargs["program"].requirements.value_reads:
            recorded.append((self, kwargs, result))
        return result

    with monkeypatch.context() as capture:
        capture.setattr(SimulationRuntime, "dispatch", observe)
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "wealth": jnp.asarray([1.0, 2.0, 3.0]),
                "age": jnp.full(3, 18.0),
                "regime_id": jnp.full(3, RegimeId.working_life, dtype=jnp.int32),
            },
            seed=17,
            log_level="off",
        )
    assert len(recorded) == 1
    observed_runtime, call, expected = recorded[0]
    runtime = SimulationRuntime(
        execution=observed_runtime.execution,
        enable_jit=True,
        subject_devices=observed_runtime.subject_devices,
    )
    arguments = call["arguments"]
    assert isinstance(arguments, Mapping)
    abstract_arguments = jax.tree.map(_abstract, arguments)
    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaf in jax.tree.leaves(abstract_arguments)
    )
    prepare = getattr(runtime, "prepare_abstract", None)
    assert callable(prepare), "The declared runtime needs an abstract-only compiler"
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_allocation)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_allocation)
        with pytest.raises(_ConcreteAllocationError):
            jnp.zeros(3)
        compiled = prepare(
            program=call["program"],
            arguments=abstract_arguments,
            period=call["period"],
            n_subjects=call["n_subjects"],
            widths={"action_product": 2, "subject": 3},
        )
    assert isinstance(compiled.executable, jax.stages.Compiled)
    assert compiler_peak_bytes(compiled=compiled.executable, widths=compiled.widths) > 0
    actual = compiled(**arguments)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    dispatched = runtime.dispatch(**call)
    assert len(runtime.cache) == 1
    assert next(iter(runtime.cache.values())) is compiled
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(dispatched), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_an_actual_host_operation_profiles_without_allocating_its_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action decoding shares its actual dispatch executable with abstract planning."""
    arguments = {
        "flat_indices": jnp.asarray([0, 2, 1], dtype=jnp.int32),
        "grids": MappingProxyType({"consumption": jnp.asarray([1.0, 3.0, 7.0])}),
    }
    expected = _lookup_values_from_indices(**arguments)
    operations = ProfiledSimulationOperations()
    prepare = getattr(operations, "prepare_abstract", None)
    assert callable(prepare), "Host operations need allocation-free preparation"
    devices = (jax.devices()[0],)
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_allocation)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_allocation)
        with pytest.raises(_ConcreteAllocationError):
            jnp.zeros(3)
        profile = prepare(
            function=_lookup_values_from_indices,
            arguments=jax.tree.map(_abstract, arguments),
            subject_arg_names=("flat_indices",),
            devices=devices,
        )
    assert profile.peak_bytes == compiler_peak_bytes(
        compiled=profile.executable, widths={}
    )
    assert profile.peak_bytes > 0
    actual = operations.dispatch(
        function=_lookup_values_from_indices,
        arguments=arguments,
        subject_arg_names=("flat_indices",),
        devices=devices,
        live_footprint=lambda: measure_buffer_footprint(tree=arguments),
        budget_devices=devices,
        budget_bytes=1_000_000,
    )
    assert len(operations.cache) == 1
    assert next(iter(operations.cache.values())) is profile
    assert isinstance(actual, dict | MappingProxyType)
    np.testing.assert_array_equal(actual["consumption"], expected["consumption"])


@pytest.mark.parametrize("invalid", ["concrete", "missing_layout", "wrong_layout"])
def test_abstract_host_operands_cannot_bypass_placement_guards(
    *, invalid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation refuses a leaf that physical dispatch would still need to place."""
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    value = (
        jnp.ones(3, dtype=jnp.int32)
        if invalid == "concrete"
        else jax.ShapeDtypeStruct(
            (3,),
            jnp.int32,
            sharding=(
                None
                if invalid == "missing_layout"
                else jax.NamedSharding(
                    jax.make_mesh((1,), ("X",), devices=(device,)), jax.P()
                )
            ),
        )
    )
    operations = ProfiledSimulationOperations()
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_allocation)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_allocation)
        with pytest.raises(ExecutionPlanningError, match="Abstract operation"):
            operations.prepare_abstract(
                function=_lookup_values_from_indices,
                arguments={
                    "flat_indices": value,
                    "grids": MappingProxyType(
                        {
                            "consumption": jax.ShapeDtypeStruct(
                                (3,), jnp.float32, sharding=sharding
                            )
                        }
                    ),
                },
                subject_arg_names=("flat_indices",),
                devices=(device,),
            )
    assert not operations.cache


def _entry_argument_metadata(value: ArgumentTree) -> Hashable:
    """Compare named leaf metadata separately from compiler container structure."""
    if isinstance(value, Mapping):
        return tuple(
            (name, _entry_argument_metadata(value[name])) for name in sorted(value)
        )
    if isinstance(value, tuple | list):
        return tuple(_entry_argument_metadata(item) for item in value)
    if value is None:
        return None
    assert isinstance(value, jax.Array | jax.ShapeDtypeStruct)
    return value.shape, str(value.dtype), value.weak_type, value.sharding


@pytest.mark.slow
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_profiled_participation_entry_reuses_actual_dispatch_key(  # noqa: PLR0915 - keep one real profile-to-dispatch witness together
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real entry dispatch reuses the exact profiled executable and report."""
    model = household.get_model(
        n_periods=3,
        wealth_n_points=3,
        consumption_n_points=2,
        execution_config=ExecutionConfig(
            axis_widths={"query_cell": 1, "subject": 4},
            device_memory_bytes=2**31,
        ),
    )
    assert all(
        model._regimes[name].simulation.exit_queries
        for name in ("couple", "single_f", "single_m")
    )
    params = household.get_params()
    initial = household.get_initial_conditions(n_subjects=4, model=model)
    solution = model.solve(params=params, log_level="off", max_compilation_workers=1)
    profiles: dict[
        tuple[int, int, tuple[tuple[str, int], ...]],
        tuple[tuple[Hashable, ...], CompiledSimulationProgram, Hashable],
    ] = {}
    planning = [False]
    checked: list[bool] = []
    produced_information: list[bool] = []
    produce_arguments = query_module._QueryArguments.__call__
    prepare_chunks = model_module.prepare_simulation_chunks
    prepare = SimulationRuntime.prepare_abstract
    compile_candidate = SimulationRuntime.compile_candidate

    # keyword-only-exempt: library-callback=_QueryArguments.__call__
    def observed_arguments(
        self: query_module._QueryArguments, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        arguments = produce_arguments(self, context)
        information = arguments.get("_lcm_query_information")
        if information is not None:
            assert isinstance(information, MappingProxyType)
            assert information
            assert all(
                isinstance(row, MappingProxyType) for row in information.values()
            )
            assert jax.tree.leaves(information)
            abstract = abstract_tree(
                tree=information,
                sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
            )
            assert jax.tree.structure(abstract) == jax.tree.structure(information)
            produced_information.append(True)
        return arguments

    def observed_chunks(**call: Unpack[ChunkPreparation]) -> ChunkPreparationResult:
        planning[0] = True
        try:
            return prepare_chunks(**call)
        finally:
            planning[0] = False

    def observed_profile(
        self: SimulationRuntime, **call: Unpack[RuntimePreparation]
    ) -> CompiledSimulationProgram:
        program = call["program"]
        if planning[0] and program.name.startswith("simulate_participation_entry:"):
            with monkeypatch.context() as guard:
                guard.setattr(jax, "device_put", _forbid_allocation)
                guard.setattr(
                    jax._src.core.EvalTrace, "process_primitive", _forbid_allocation
                )
                with pytest.raises(_ConcreteAllocationError):
                    jnp.zeros(3)
                compiled = prepare(self, **call)
        else:
            compiled = prepare(self, **call)
        if planning[0] and program.name.startswith("simulate_participation_entry:"):
            assert isinstance(compiled.executable, jax.stages.Compiled)
            assert compiled.memory is not None
            assert compiled.memory.records
            keys = [key for key, entry in self.cache.items() if entry is compiled]
            assert len(keys) == 1
            identity = id(self), id(program.function), tuple(call["widths"].items())
            record = (
                cast("tuple[Hashable, ...]", keys[0]),
                compiled,
                _entry_argument_metadata(call["arguments"]),
            )
            assert identity not in profiles or profiles[identity] == record
            profiles[identity] = record
        return compiled

    # keyword-only-exempt: library-callback=SimulationRuntime.compile_candidate
    def observed_candidate(
        self: SimulationRuntime,
        *,
        program: MaterializedCoreProgram,
        n_subjects: int,
        widths: Mapping[str, int],
        abstract_inputs: bool = False,
    ) -> CompiledSimulationProgram:
        if (
            not planning[0]
            and not checked
            and program.name.startswith("simulate_participation_entry:")
        ):
            assert profiles
            identity = id(self), id(program.function), tuple(widths.items())
            assert identity in profiles, (
                "Real entry needs its same-runtime/function/width profile"
            )
            profiled_key, profiled, metadata = profiles[identity]
            assert n_subjects == 4
            assert dict(widths) == {"subject": 4}
            holders = program.arguments["_holders"]
            assert isinstance(holders, jax.Array)
            assert holders.shape == (4,)
            assert holders.dtype == jnp.bool_
            states = program.arguments["states"]
            assert isinstance(states, Mapping)
            wealth = np.asarray(states["wealth"])
            finite = wealth[np.isfinite(wealth)]
            assert finite.size
            assert np.unique(finite).size > 1
            assert _entry_argument_metadata(program.arguments) == metadata
            key = cast(
                "tuple[Hashable, ...]",
                _simulation_lowering_key(
                    runtime=self, program=program, n_subjects=n_subjects, widths=widths
                ),
            )
            assert key[0] == profiled_key[0]
            assert key[2:] == profiled_key[2:]
            actual_arguments = dict(cast("tuple[tuple[str, Hashable], ...]", key[1]))
            profile_arguments = dict(
                cast("tuple[tuple[str, Hashable], ...]", profiled_key[1])
            )
            assert actual_arguments.keys() == profile_arguments.keys()
            differences = {
                name: (str(profile_arguments[name]), str(actual_arguments[name]))
                for name in actual_arguments
                if actual_arguments[name] != profile_arguments[name]
            }
            cached = self.cache.get(key)
            assert cached is profiled, (
                f"Profiled entry missing at actual dispatch: {program.name}; "
                f"holders={np.asarray(holders).tolist()}; "
                f"different argument trees={differences}"
            )
            reused = compile_candidate(
                self,
                program=program,
                n_subjects=n_subjects,
                widths=widths,
                abstract_inputs=abstract_inputs,
            )
            assert reused is profiled
            assert reused.executable is profiled.executable
            assert reused.memory is profiled.memory
            checked.append(True)
            return reused
        return compile_candidate(
            self,
            program=program,
            n_subjects=n_subjects,
            widths=widths,
            abstract_inputs=abstract_inputs,
        )

    with monkeypatch.context() as observe:
        observe.setattr(query_module._QueryArguments, "__call__", observed_arguments)
        observe.setattr(model_module, "prepare_simulation_chunks", observed_chunks)
        observe.setattr(SimulationRuntime, "prepare_abstract", observed_profile)
        observe.setattr(SimulationRuntime, "compile_candidate", observed_candidate)
        result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=17,
            log_level="off",
            max_compilation_workers=1,
        )
    assert checked == [True]
    assert produced_information
    frame = result.to_dataframe()
    assert len(frame.loc[frame["period"] == 2]) == 4
