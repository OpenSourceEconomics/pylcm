"""Real private Model construction/solve and original native-bank admission."""

import logging
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import lcm.model as model_module
from _lcm.continuation import ProjectedContinuationSpec
from _lcm.dtypes import canonical_float_dtype
from _lcm.execution.workspace_planning import (
    CompilerMemoryReservation,
    WorkspaceBudgetError,
)
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.host_operations import (
    ProfiledSimulationOperations,
    _operation_key,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    union_buffer_footprints,
)
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.block_major import RetainedComponentValues
from _lcm.solution.contract import BackwardInductionResult
from _lcm.solution.current_iid_case_r import (
    CASE_R_NODE_SELECTION,
    CASE_R_RESOURCES_REPLAY,
    CaseRProgramFunction,
    NativeCaseRPeriodKernel,
    _native_case_r_array_template,
    _zeros_for_template,
)
from _lcm.solution.feasibility import BoundedResult
from _lcm.solution.generation import canonical_generation_roots
from _lcm.solution.solve_phase_records import CallId
from _lcm.typing import FlatParams, PytreeValue, ShapeDtypePytree
from lcm import ExecutionConfig, InitialNodes, Model
from lcm._solver_api import authority as authority_module
from lcm.shocks import PhysicalValues
from lcm.solver_api import ArtifactRef, SolutionResult
from lcm.typing import ReferenceName
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_call_owned_generation import (
    _declarations,
    _params,
    _public_model_from_declarations,
)

# Tests of the native current-IID route itself, which named shocks of EGM regimes
# no longer take by default.
pytestmark = [
    pytest.mark.coverage(backends=("cpu",), precisions="both"),
    pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
    pytest.mark.usefixtures("native_current_iid"),
]


def test_budgeted_public_model_solves_real_r_and_keeps_evicted_generation() -> None:
    """Public budgeted solves retain the producing row through generation eviction."""
    declarations = replace(
        _declarations(runtime_child_grid=True),
        execution_config=ExecutionConfig(
            devices=(jax.devices("cpu")[0].id,), device_memory_bytes=1 << 28
        ),
    )
    params = _params(child_points=np.array([1.0, 12.5]))
    model = _public_model_from_declarations(declarations=declarations)
    result = model.solve(params=params, log_level="warning", max_compilation_workers=1)
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    first = view.generation
    assert first is not None
    assert first.execution is not model
    child_value = np.asarray(view.values[1]["child"])
    wealth = np.array([1.0, 12.5], dtype=child_value.dtype)
    # Both eps outcomes' exact log/log optima occur in the declared saving grid.
    expected = np.log(wealth / 2.0) + np.log((wealth + 1.0) / 2.0)
    tolerance = 64 * np.finfo(child_value.dtype).eps
    np.testing.assert_allclose(child_value, expected, rtol=tolerance, atol=tolerance)
    assert np.isfinite(np.asarray(view.values[0]["parent"])).all()
    model._generation_cache.capacity = 1
    _, second = model._prepare_call(
        params=_params(child_points=np.array([1.25, 12.75]))
    )
    assert second is not first
    assert second.execution._regimes is not first.execution._regimes
    for generation, endpoint in ((first, 1.0), (second, 1.25)):
        spec = generation.execution._regimes["child"].solution.continuation_spec
        assert isinstance(spec, ProjectedContinuationSpec)
        grid = spec.date_at(period=1).template.grid
        assert float(grid[0]) == endpoint
        assert any(
            leaf is grid for leaf in jax.tree.leaves(model._generation_cache.roots())
        )
    assert view.generation is first
    _, reacquired = model._prepare_call(params=params)
    assert reacquired is first


def test_public_case_r_replays_from_a_result_whose_generation_was_evicted() -> None:
    """A held result keeps its producing generation for physical replay."""
    params = _params(child_points=np.array([1.0, 12.5]))
    model = _public_model_from_declarations(
        declarations=_declarations(runtime_child_grid=True)
    )
    result = model.solve(params=params, log_level="warning", max_compilation_workers=1)
    initial_conditions = {
        "regime_id": np.array([0]),
        "age": np.array([0]),
        "wealth": np.array([4.0]),
    }
    before = model.simulate(
        params=params,
        solution=result,
        seed=19,
        log_level="warning",
        initial_conditions=initial_conditions,
    ).to_dataframe()
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    first = view.generation
    model._generation_cache.capacity = 1
    _, second = model._prepare_call(
        params=_params(child_points=np.array([1.25, 12.75]))
    )
    assert second is not first
    after = model.simulate(
        params=params,
        solution=result,
        seed=19,
        log_level="warning",
        initial_conditions=initial_conditions,
    ).to_dataframe()
    assert view.generation is first
    pd.testing.assert_frame_equal(after, before, check_exact=True)


@dataclass(frozen=True, kw_only=True)
class _RealProducerRecord:
    source: Callable[..., PytreeValue]
    key: Hashable
    signature: (
        tuple[jax.tree_util.PyTreeDef, tuple[jax.ShapeDtypeStruct | int, ...]] | None
    )
    resident_bytes: int
    memory: CompilerMemoryReservation

    @property
    def required_bytes(self) -> int:
        return self.resident_bytes + self.memory.reservation_bytes


def test_cold_model_native_output_refuses_using_its_actual_compiler_record(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeat the real constructor with an observed bank requirement minus one.

    Observers return original results unchanged. No executable/profile or output
    is fabricated, and the tightened second run receives its real public budget.
    """
    declarations = replace(
        _declarations(),
        execution_config=ExecutionConfig(
            devices=(jax.devices("cpu")[0].id,), device_memory_bytes=1 << 28
        ),
    )
    records: list[_RealProducerRecord] = []
    failures: list[tuple[_RealProducerRecord, int]] = []
    original_admit = ProfiledSimulationOperations.admit_producer

    # keyword-only-exempt: library-callback=monkeypatch.setattr
    def observe_admission(
        self: ProfiledSimulationOperations,
        *,
        function: Callable[..., PytreeValue],
        arguments: Mapping[ReferenceName, ShapeDtypePytree],
        devices: tuple[jax.Device, ...],
        output_sharding: jax.sharding.Sharding | None,
        budget_bytes: int,
        resident_bytes: int,
    ) -> jax.stages.Compiled:
        source = function.func if isinstance(function, partial) else function
        key = _operation_key(
            function=function,
            arguments=arguments,
            static_arguments=MappingProxyType({}),
            subject_outputs=False,
            devices=devices,
            output_sharding=output_sharding,
        )
        signature = None
        if source in (_native_case_r_array_template, _zeros_for_template):
            bound = (
                (function.args, function.keywords)
                if isinstance(function, partial)
                else ((), {})
            )
            leaves, structure = jax.tree.flatten((arguments, bound))
            descriptors: list[jax.ShapeDtypeStruct | int] = []
            for leaf in leaves:
                if isinstance(leaf, jax.ShapeDtypeStruct) or type(leaf) is int:
                    descriptors.append(leaf)
                else:
                    raise AssertionError(
                        "Native bank signature lost its original metadata."
                    )
            signature = (structure, tuple(descriptors))
        try:
            executable = original_admit(
                self,
                function=function,
                arguments=arguments,
                devices=devices,
                output_sharding=output_sharding,
                budget_bytes=budget_bytes,
                resident_bytes=resident_bytes,
            )
        except WorkspaceBudgetError:
            profile = self.cache[key]
            failures.append(
                (
                    _RealProducerRecord(
                        source=source,
                        key=key,
                        signature=signature,
                        resident_bytes=resident_bytes,
                        memory=profile.memory,
                    ),
                    budget_bytes,
                )
            )
            raise
        profile = self.cache[key]
        assert profile.executable is executable
        records.append(
            _RealProducerRecord(
                source=source,
                key=key,
                signature=signature,
                resident_bytes=resident_bytes,
                memory=profile.memory,
            )
        )
        return executable

    monkeypatch.setattr(
        ProfiledSimulationOperations, "admit_producer", observe_admission
    )
    Model._from_call_owned_params(declarations=declarations, params=_params())
    peak = 0
    bank: _RealProducerRecord | None = None
    for record in records:
        if (
            record.source in (_native_case_r_array_template, _zeros_for_template)
            and record.required_bytes > peak
        ):
            bank = record
        peak = max(peak, record.required_bytes)
    assert bank is not None, "The genuine native bank must determine a new prefix peak."
    assert any(record.output_bytes > 0 for record in bank.memory.records)
    tight_budget = bank.required_bytes - 1
    assert tight_budget >= bank.resident_bytes
    original_produce = SimulationEntryAllocations.produce
    refused_outputs: list[bool] = []

    # keyword-only-exempt: library-callback=monkeypatch.setattr
    def observe_outputs[T: PytreeValue](
        self: SimulationEntryAllocations,
        *,
        function: Callable[..., T],
        arguments: Mapping[str, PytreeValue],
    ) -> T:
        pending = tuple(self._pending)
        try:
            return original_produce(self, function=function, arguments=arguments)
        except WorkspaceBudgetError:
            assert len(pending) == len(self._pending)
            assert all(
                left is right
                for left, right in zip(pending, self._pending, strict=True)
            )
            refused_outputs.append(True)
            raise

    monkeypatch.setattr(SimulationEntryAllocations, "produce", observe_outputs)
    records.clear()
    with pytest.raises(WorkspaceBudgetError):
        Model._from_call_owned_params(
            declarations=replace(
                declarations,
                execution_config=replace(
                    declarations.execution_config, device_memory_bytes=tight_budget
                ),
            ),
            params=_params(),
        )
    assert len(failures) == 1
    failed, budget = failures[0]
    assert failed.source is bank.source
    # Keys can retain fresh date-schema objects by identity. Compare their actual
    # bound/input descriptors as well as every genuine compiler memory counter.
    assert failed.signature is not None
    assert failed.signature == bank.signature
    assert failed.memory == bank.memory
    assert failed.required_bytes > budget
    assert budget == tight_budget
    assert refused_outputs == [True]


def test_real_r_solution_leaf_tuple_replays_original_child_actions() -> None:
    """Actual construction, solve and forward reuse the original native row."""
    params = _params()
    declarations = replace(
        _declarations(), initial_nodes=InitialNodes(by_age={0: "parent", 1: "child"})
    )
    model = Model._from_call_owned_params(declarations=declarations, params=params)
    solution = model.solve(params=params, log_level="off", max_compilation_workers=1)
    ref = ArtifactRef(period=1, regime="child", key=CASE_R_RESOURCES_REPLAY)
    payload = solution.replay_artifacts[ref]
    assert type(payload) is tuple
    assert all(isinstance(leaf, jax.Array) for leaf in payload)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray([1.0, 12.5]),
            "age": jnp.ones((2,)),
            "regime_id": jnp.full(
                (2,), model.regime_names_to_ids["child"], dtype=jnp.int32
            ),
            "eps": PhysicalValues(
                values=np.asarray([0.0, 1.0], dtype=canonical_float_dtype())
            ),
        },
        log_level="off",
        seed=17,
    )
    child = result.raw_results["child"][1]
    np.testing.assert_array_equal(child.shocks["eps"], [0.0, 1.0])
    expected = np.asarray([0.5, 6.75])
    tolerance = 64 * np.finfo(np.asarray(child.actions["consumption"]).dtype).eps
    np.testing.assert_allclose(
        child.actions["consumption"], expected, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        result.raw_results["done"][2].states["wealth"],
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def test_balanced_r_retains_required_row_without_optional_selection_bank() -> None:
    """A genuine prepared producer declares exactly its retained original row."""
    params = _params()
    model = Model._from_call_owned_params(declarations=_declarations(), params=params)
    dated = model._dated_generation
    assert dated is not None
    kernels = dated.preparation.numerical.kernels["child"]
    kernel = kernels.period_kernels[1]
    assert isinstance(kernel, NativeCaseRPeriodKernel)
    replay = kernel.core_programs()["replay"]
    function = (
        replay.function.func
        if isinstance(replay.function, partial)
        else replay.function
    )
    assert isinstance(function, CaseRProgramFunction)
    assert function.core.balanced is not None
    assert replay.retained_artifact_keys == (CASE_R_RESOURCES_REPLAY,)
    assert replay.retained_artifact_payload_types == {CASE_R_RESOURCES_REPLAY: tuple}
    assert isinstance(replay.output_roles, BoundedResult)
    assert replay.output_roles.output[2] is not None
    assert replay.output_roles.output[5] is None
    assert kernels.artifact_authorities[CASE_R_RESOURCES_REPLAY].required
    assert CASE_R_NODE_SELECTION not in kernels.artifact_authorities


def test_public_budgeted_r_charges_actual_outputs_before_result_assembly(  # noqa: PLR0915 - keep the genuine solve and original copy observers together
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual compiled outputs are live before the original assembly can copy."""
    owners: list[SimulationEntryAllocations] = []
    finished: list[BackwardInductionResult] = []
    prepared_roots: dict[int, DeviceBufferFootprint] = {}
    assembly: list[tuple[SimulationEntryAllocations, DeviceBufferFootprint]] = []
    assembly_copies: list[bool] = []
    original_leaf_copy = authority_module._copy_artifact_array_leaf
    original_copy = SimulationEntryAllocations.copy_solution_leaf
    original_finish = Model._finish_solution

    # keyword-only-exempt: library-callback=copy_solution_leaf
    def observe_copy(
        self: SimulationEntryAllocations, *, leaf: jax.Array, label: str
    ) -> jax.Array:
        if id(self) not in prepared_roots:
            prepared_roots[id(self)] = measure_buffer_footprint(tree=self.model_roots)
        result = original_copy(self, leaf=leaf, label=label)
        owners.append(self)
        return result

    def require_covered(
        *, expected: DeviceBufferFootprint, observed: DeviceBufferFootprint
    ) -> None:
        assert (
            union_buffer_footprints(footprints=(observed, expected)).spans
            == observed.spans
        ), "Postsolve output storage is not charged"

    def observe_leaf_copy(
        *,
        leaf: jax.Array,
        label: str,
        array_copier: authority_module._ArrayCopier | None = None,
    ) -> jax.Array:
        if assembly:
            owner, required = assembly[-1]
            live = owner.snapshot()
            require_covered(expected=prepared_roots[id(owner)], observed=live)
            require_covered(expected=required, observed=live)
            assert array_copier == owner.copy_solution_leaf
            assembly_copies.append(True)
        return original_leaf_copy(leaf=leaf, label=label, array_copier=array_copier)

    # keyword-only-exempt: library-callback=Model._finish_solution
    def observe_finish(
        self: Model,
        *,
        preparation: model_module._SolutionPreparation,
        internal_result: BackwardInductionResult,
        flat_params: FlatParams,
        log: logging.Logger,
        call_id: CallId | None,
        process_grid_resolver: ProcessGridResolver | None = None,
        component_values: RetainedComponentValues | None = None,
        validation_materializer: SimulationEntryAllocations | None = None,
    ) -> SolutionResult:
        assert owners  # The original admitted copier actually completed in this run.
        owner = owners[-1]
        assert owner.original_inputs is not None
        assert owner.budget_bytes == 1 << 28
        assert preparation.generation.execution is self
        outputs = canonical_generation_roots(declarations=(internal_result,))
        assert outputs
        required = measure_buffer_footprint(tree=outputs)
        model_roots = prepared_roots[id(owner)]
        assert model_roots.spans
        # Prove the very same detector rejects actual nonempty output storage.
        with pytest.raises(AssertionError, match="Postsolve output storage"):
            require_covered(
                expected=required, observed=measure_buffer_footprint(tree=())
            )
        # These are real newly produced buffers, not aliases of prepared grids.
        assert (
            union_buffer_footprints(footprints=(model_roots, required)).spans
            != model_roots.spans
        )
        finished.append(internal_result)
        assembly.append((owner, required))
        try:
            return original_finish(
                self,
                preparation=preparation,
                internal_result=internal_result,
                flat_params=flat_params,
                log=log,
                call_id=call_id,
                process_grid_resolver=process_grid_resolver,
                component_values=component_values,
                validation_materializer=validation_materializer,
            )
        finally:
            assembly.pop()

    monkeypatch.setattr(SimulationEntryAllocations, "copy_solution_leaf", observe_copy)
    monkeypatch.setattr(Model, "_finish_solution", observe_finish)
    monkeypatch.setattr(
        authority_module, "_copy_artifact_array_leaf", observe_leaf_copy
    )
    model = _public_model_from_declarations(
        declarations=replace(
            _declarations(),
            execution_config=ExecutionConfig(device_memory_bytes=1 << 28),
        )
    )
    result = model.solve(
        params=_params(), log_level="warning", max_compilation_workers=1
    )
    assert len(finished) == 1
    assert assembly_copies
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    assert view.values[1]["child"] is finished[0].value_functions[1]["child"]
    child_value = np.asarray(view.values[1]["child"])
    wealth = np.asarray([1.0, 12.5], dtype=child_value.dtype)
    expected = np.log(wealth / 2.0) + np.log((wealth + 1.0) / 2.0)
    tolerance = 64 * np.finfo(child_value.dtype).eps
    np.testing.assert_allclose(child_value, expected, rtol=tolerance, atol=tolerance)
