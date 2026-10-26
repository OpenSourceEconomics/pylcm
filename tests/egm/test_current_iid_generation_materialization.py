"""Held real entry-owned bank/template materialization and compiler admission."""

import sys
from collections.abc import Callable
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ProjectedProducerDate
from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm.current_iid_declarations import (
    _materialize_projected_template,
    _projected_template,
)
from _lcm.egm.projected_continuation import ProjectedChildAuthority
from _lcm.execution.workspace_planning import WorkspaceBudgetError
from _lcm.shock_registry import ShockBinding
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.residency import (
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.solution.current_iid_case_r import native_case_r_date_template
from _lcm.solution.fingerprint import _semantic_fingerprint, _SemanticHasher
from _lcm.typing import PytreeValue
from lcm.exceptions import ExecutionPlanningError
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.typing import FloatND

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _owner(
    *,
    roots: tuple[PytreeValue, ...] = (),
    retained: Callable[[], tuple[PytreeValue, ...]] | None = None,
) -> SimulationEntryAllocations:
    return SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=()),
        solution=None,
        model_roots=roots,
        retained_generation_roots=retained,
        devices=(jax.devices("cpu")[0],),
        budget_bytes=2**20,
    )


def _binding(
    *,
    points: tuple[float, ...],
    weights: tuple[float, ...],
    weight_dtype: str = "float64",
    shock_name: str = "eps",
) -> ShockBinding:
    return ShockBinding(
        group=ShockGroup(
            shocks={shock_name: Normal(mu=0.0, sigma=1.0)},
            nodes=UserNodes(
                nodes=np.asarray(points)[:, None],
                weights=np.asarray(weights, dtype=weight_dtype),
            ),
        ),
        state_inputs=frozenset(),
        state_domains=MappingProxyType({}),
        categorical_domains=MappingProxyType({}),
        functions=MappingProxyType({}),
        params_template=MappingProxyType({}),
        callable_parameters=frozenset(),
    )


@pytest.mark.parametrize("weight_dtype", ["float16", "float64"])
def test_current_iid_banks_use_actual_upload_owner_and_keep_zero_mass_rows(
    *, weight_dtype: str
) -> None:
    bindings = {
        "first": _binding(
            points=(0.0, 1.0), weights=(1.0, 0.0), weight_dtype=weight_dtype
        ),
        "second": _binding(
            points=(-1.0, 0.0, 1.0),
            weights=(0.25, 0.5, 0.25),
            shock_name="eta",
        ),
    }
    original = CurrentIIDPlan.create(bindings=bindings, age=1, coordinate_kind="age")
    assert original is not None
    owner = _owner(roots=(original.indices, original.node_operands))
    try:
        actual = CurrentIIDPlan.create(
            bindings=bindings, age=1, coordinate_kind="age", array_writer=owner
        )
        assert actual is not None
        np.testing.assert_array_equal(actual.indices, original.indices)
        assert actual.indices.shape == (6, 2)
        held = jax.tree.leaves(owner.solve_input_roots())
        for (_, _, expected), (_, _, observed) in zip(
            original.bindings, actual.bindings, strict=True
        ):
            assert observed.fingerprint == expected.fingerprint
            for left, right in zip(
                jax.tree.leaves(expected), jax.tree.leaves(observed), strict=True
            ):
                assert np.asarray(left).tobytes() == np.asarray(right).tobytes()
                assert any(leaf is right for leaf in held)
        np.testing.assert_array_equal(actual.bindings[0][2].weights, [1.0, 0.0])
    finally:
        owner.close()


def test_native_templates_keep_original_bits_and_date_under_actual_producer() -> None:
    grid = jnp.asarray([-0.0, 2.0])
    original = _projected_template(grid=grid, capacity=8)
    date = ProjectedProducerDate(
        authority=ProjectedChildAuthority(
            producer="child", period=1, liquid_state="wealth", capacity=8
        ),
        template=original,
    )
    expected = native_case_r_date_template(
        date=date,
        liquid_shape=grid.shape,
        resources_capacity=8,
        child_slots=1,
        current_nodes=2,
    )
    owner = _owner(roots=(grid, original, expected))
    try:
        projected = _materialize_projected_template(
            grid=grid, capacity=8, materializer=owner
        )
        for left, right in zip(
            jax.tree.leaves(original), jax.tree.leaves(projected), strict=True
        ):
            assert (left.shape, left.dtype) == (right.shape, right.dtype)
            assert np.asarray(left).tobytes() == np.asarray(right).tobytes()
        actual = native_case_r_date_template(
            date=date,
            liquid_shape=grid.shape,
            resources_capacity=8,
            child_slots=1,
            current_nodes=2,
            materializer=owner,
        )
        assert actual.projected is date.template
        for left, right in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            assert (left.shape, left.dtype) == (right.shape, right.dtype)
            assert np.asarray(left).tobytes() == np.asarray(right).tobytes()
        assert owner.operations.cache
    finally:
        owner.close()


def test_template_refusal_uses_real_compiler_record_and_fresh_retained_roots() -> None:
    grid = jnp.asarray([0.0, 2.0])
    retained: list[FloatND] = []
    owner = _owner(roots=(grid,), retained=lambda: tuple(retained))
    factory = partial(_projected_template, capacity=8)
    try:
        owner.produce(function=factory, arguments={"grid": grid})
        compiled = next(iter(owner.operations.cache.values()))
        # A newly retained root must enter the next admission, even though the
        # original producer executable/profile is already cached.
        retained.append(jnp.ones((32,)))
        live = owner.snapshot()
        device = owner.devices[0]
        external = resident_bytes_by_device(
            live=live,
            arguments=measure_buffer_footprint(tree=grid),
            devices=owner.devices,
        )[device]
        owner.budget_bytes = external + compiled.memory.reservation_bytes - 1
        resident = sum(stop - start for start, stop in live.spans[device])
        assert resident <= owner.budget_bytes
        pending = tuple(owner._pending)
        with pytest.raises(WorkspaceBudgetError):
            owner.produce(function=factory, arguments={"grid": grid})
        assert all(
            left is right for left, right in zip(pending, owner._pending, strict=True)
        )
        np.testing.assert_array_equal(retained[0], np.ones(32))
    finally:
        owner.close()


def test_numeric_factory_capture_refuses_before_compilation() -> None:
    grid = jnp.asarray([0.0, 2.0])
    owner = _owner(roots=(grid,))
    try:
        with pytest.raises(ExecutionPlanningError, match="explicit numerical"):
            owner.produce(
                function=partial(_projected_template, grid=grid, capacity=8),
                arguments={},
            )
        assert not owner.operations.cache
    finally:
        owner.close()


def test_template_preflight_counts_an_excluded_source_device() -> None:
    """Source residency is checked even though the factory runs on entry CPU0."""
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("Run the source-device custody control with logical CPU devices.")
    source = jax.device_put(np.ones(64), device=devices[-1])
    grid = jax.device_put(np.asarray([0.0, 2.0]), device=devices[0])
    owner = _owner(roots=(grid, source))
    owner.budget_bytes = source.nbytes - 1
    try:
        with pytest.raises(ExecutionPlanningError, match="before allocation"):
            _materialize_projected_template(grid=grid, capacity=8, materializer=owner)
        assert not owner.operations.cache
        np.testing.assert_array_equal(source, np.ones(64))
    finally:
        owner.close()


_REFERENCED_DEVICE_BIAS: jax.Array | None = None
_UNRELATED_DEVICE_BANK: jax.Array | None = None


def _function_with_referenced_device_bias(*, value: FloatND) -> FloatND:
    assert _REFERENCED_DEVICE_BIAS is not None
    return value + _REFERENCED_DEVICE_BIAS


def test_semantic_array_observer_keeps_digest_and_exact_referenced_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The genuine hasher reports consumed buffers without a broad globals scan."""
    referenced = jnp.asarray(0.125)
    unrelated = jnp.ones((32,))
    monkeypatch.setattr(sys.modules[__name__], "_REFERENCED_DEVICE_BIAS", referenced)
    monkeypatch.setattr(sys.modules[__name__], "_UNRELATED_DEVICE_BANK", unrelated)
    roots: list[jax.Array] = []
    observed = _SemanticHasher(array_recorder=roots.append)
    observed.visit(value=_function_with_referenced_device_bias)
    assert observed.hexdigest() == _semantic_fingerprint(
        _function_with_referenced_device_bias
    )
    assert any(root is referenced for root in roots)
    assert all(root is not unrelated for root in roots)
