"""Prewarming uses declared abstract operands without allocating a population."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import compiler_peak_bytes
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.simulate import _lookup_values_from_indices
from lcm import ExecutionConfig, LinSpacedGrid
from lcm.exceptions import ExecutionPlanningError
from tests.test_models.deterministic.regression import RegimeId, get_model, get_params


class _ConcreteAllocationError(AssertionError):
    """Identify an allocating dispatch while only abstract operands are allowed."""


def _forbid_allocation(*_args: object, **_kwargs: object) -> object:
    raise _ConcreteAllocationError("An abstract profile tried to allocate an operand")


def _abstract(leaf: object) -> object:
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
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3, "action_product": 2}
        ),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    recorded: list[tuple[SimulationRuntime, dict[str, Any], object]] = []
    dispatch = SimulationRuntime.dispatch

    def observe(self: SimulationRuntime, **kwargs: Any) -> object:
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
