"""Shape-only arguments remain resident while a profiled operation allocates."""

import dataclasses
from collections.abc import Callable
from functools import partialmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import DeviceBufferFootprint, measure_buffer_footprint
from lcm.exceptions import ExecutionPlanningError

_SUBJECT_COUNT = 262_144
_MISSING_REGIME = -2_147_483_648


def _empty_membership(*, state: jax.Array) -> jax.Array:
    """Use only shape and dtype, so the numerical input is otherwise dead."""
    return jnp.full_like(state, _MISSING_REGIME, dtype=jnp.int32)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RetainedInput:
    state: jax.Array

    def __call__(self) -> DeviceBufferFootprint:
        self.state.block_until_ready()
        return measure_buffer_footprint(tree=self.state)


def _input() -> _RetainedInput:
    """Use one explicit actual device without pinning global topology."""
    return _RetainedInput(
        state=jax.device_put(
            np.arange(_SUBJECT_COUNT, dtype=np.int32), jax.devices()[0]
        ).block_until_ready()
    )


def _payload_bytes(*, tree: object, device: jax.Device) -> int:
    footprint = measure_buffer_footprint(tree=tree)
    return sum(stop - start for start, stop in footprint.spans.get(device, ()))


def _dispatch(
    *, owner: _RetainedInput, operations: ProfiledSimulationOperations, budget: int
) -> object:
    devices = tuple(owner.state.sharding.device_set)
    return operations.dispatch(
        function=_empty_membership,
        arguments={"state": owner.state},
        subject_arg_names=("state",),
        devices=devices,
        live_footprint=owner,
        budget_devices=devices,
        budget_bytes=budget,
    )


def test_shape_only_operation_preserves_distinct_retained_input_and_output() -> None:
    """A generous admission keeps both concrete payloads and the exact sentinel."""
    owner = _input()
    operations = ProfiledSimulationOperations()
    result = _dispatch(
        owner=owner, operations=operations, budget=4 * owner.state.nbytes
    )
    assert isinstance(result, jax.Array)
    assert result.dtype == np.dtype(np.int32)
    assert result.sharding.device_set == owner.state.sharding.device_set
    device = next(iter(owner.state.sharding.device_set))
    assert _payload_bytes(tree=(owner.state, result), device=device) == (
        owner.state.nbytes + result.nbytes
    )
    assert not owner.state.is_deleted()
    np.testing.assert_array_equal(
        owner.state, np.arange(_SUBJECT_COUNT, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        result, np.full(_SUBJECT_COUNT, _MISSING_REGIME, dtype=np.int32)
    )
    (profile,) = operations.cache.values()
    analysis = profile.executable.memory_analysis()
    assert analysis is not None
    assert analysis.output_size_in_bytes >= result.nbytes
    assert profile.peak_bytes >= result.nbytes


# keyword-only-exempt: library-callback=functools.partialmethod
def _refuse_operation_execution(
    self: jax.stages.Compiled,
    *args: Any,
    original: Callable[..., object],
    operations: ProfiledSimulationOperations,
    attempted: list[bool],
    **kwargs: Any,
) -> object:
    """Observe the real compiled boundary without fabricating a memory report."""
    if any(self is profile.executable for profile in operations.cache.values()):
        attempted.append(True)
        raise AssertionError("Over-budget shape-only executable was dispatched")
    return original(self, *args, **kwargs)


def test_unused_retained_input_is_charged_before_output_execution(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Input alone fits; input plus distinct output must be refused pre-execution."""
    owner = _input()
    operations = ProfiledSimulationOperations()
    budget = 3 * owner.state.nbytes // 2
    device = next(iter(owner.state.sharding.device_set))
    assert _payload_bytes(tree=owner.state, device=device) < budget
    attempted: list[bool] = []
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _refuse_operation_execution,
            original=jax.stages.Compiled.__call__,
            operations=operations,
            attempted=attempted,
        ),
    )
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        _dispatch(owner=owner, operations=operations, budget=budget)
    assert attempted == []
    (profile,) = operations.cache.values()
    analysis = profile.executable.memory_analysis()
    assert analysis is not None
    assert analysis.argument_size_in_bytes >= owner.state.nbytes
    assert profile.peak_bytes > budget
    np.testing.assert_array_equal(
        owner.state, np.arange(_SUBJECT_COUNT, dtype=np.int32)
    )
