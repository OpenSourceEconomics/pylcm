"""Budgeted host operations keep exact arithmetic and recheck live residency.

Run this file alone: its ordered-mesh control requires a fresh four-device process.
"""

import dataclasses
import gc
import weakref
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import DeviceBufferFootprint, measure_buffer_footprint
from _lcm.simulation.simulate import _lookup_values_from_indices
from _lcm.simulation.transitions import (
    _advance_states_for_subjects,
    _draw_random_regime_ids,
)
from lcm.exceptions import ExecutionPlanningError
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


class _DispatchArguments(TypedDict):
    function: Callable[..., object]
    arguments: Mapping[str, object]
    subject_arg_names: tuple[str, ...]
    devices: tuple[jax.Device, ...]
    live_footprint: Callable[[], DeviceBufferFootprint]
    budget_devices: tuple[jax.Device, ...]


class _BudgetedDispatchArguments(_DispatchArguments):
    budget_bytes: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class _OwnedInputs:
    arrays: list[object]

    def __call__(self) -> DeviceBufferFootprint:
        jax.block_until_ready(self.arrays)
        return measure_buffer_footprint(tree=self.arrays)


def _shift(*, state: jax.Array, offset: float = 1.0) -> jax.Array:
    return state + offset


def _typed_static_shift(*, state: jax.Array, selector: bool | int) -> jax.Array:
    return state + (1 if type(selector) is bool else 2)


def _static_zero_sign(*, state: jax.Array, selector: float) -> jax.Array:
    return jnp.full(state.shape, jnp.signbit(selector))


def _operation_case(
    *, name: str
) -> tuple[Callable[..., object], dict[str, object], tuple[str, ...]]:
    if name == "merge":
        return (
            _advance_states_for_subjects,
            {
                "states_per_regime": MappingProxyType(
                    {"target": MappingProxyType({"wealth": jnp.arange(6.0)})}
                ),
                "next_states_per_regime": MappingProxyType(
                    {"target": MappingProxyType({"wealth": jnp.arange(6.0) + 2})}
                ),
                "subject_indices": jnp.asarray([True, False, True, False, True, False]),
            },
            ("states_per_regime", "next_states_per_regime", "subject_indices"),
        )
    if name == "draw":
        return (
            _draw_random_regime_ids,
            {
                "keys": jax.random.split(jax.random.key(72), 6),
                "prob_rows": (jnp.full(6, 0.3), jnp.full(6, 0.7)),
                "regime_ids": jnp.asarray([2, 5], dtype=jnp.int32),
            },
            ("keys", "prob_rows"),
        )
    return (
        _lookup_values_from_indices,
        {
            "flat_indices": jnp.asarray([0, 3, 7, 12, 1, 14], dtype=jnp.int32),
            "grids": MappingProxyType(
                {"saving": jnp.arange(3.0), "labour": jnp.arange(5.0)}
            ),
        },
        ("flat_indices",),
    )


@pytest.mark.parametrize("operation", ["merge", "draw", "gather"])
def test_profiled_operations_preserve_original_arithmetic(*, operation: str) -> None:
    function, arguments, subject_names = _operation_case(name=operation)
    expected = function(**arguments)
    dispatcher = ProfiledSimulationOperations()
    actual = dispatcher.dispatch(
        function=function,
        arguments=arguments,
        subject_arg_names=subject_names,
        devices=(jax.devices()[0],),
        live_footprint=_OwnedInputs(arrays=[arguments]),
        budget_devices=(jax.devices()[0],),
        budget_bytes=1_000_000,
    )
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for got, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert got.dtype == wanted.dtype
        if np.issubdtype(got.dtype, np.floating):
            assert_agrees_to_ulp(
                got=np.asarray(got), expected=np.asarray(wanted), n_ulp=0
            )
        else:
            np.testing.assert_array_equal(got, wanted)
    assert len(dispatcher.cache) == 1
    assert next(iter(dispatcher.cache.values())).peak_bytes > 0


def test_retained_growth_rechecks_the_same_cached_executable() -> None:
    state = jnp.arange(4.0)
    provider = _OwnedInputs(arrays=[state])
    dispatcher = ProfiledSimulationOperations()
    kwargs: _DispatchArguments = {
        "function": _shift,
        "arguments": {"state": state},
        "subject_arg_names": ("state",),
        "devices": (jax.devices()[0],),
        "live_footprint": provider,
        "budget_devices": (jax.devices()[0],),
    }
    result = dispatcher.dispatch(**kwargs, budget_bytes=1_000_000)
    compiled = next(iter(dispatcher.cache.values()))
    budget = compiled.peak_bytes + 20
    del result
    provider.arrays.append(jnp.ones(16, dtype=jnp.uint8))
    result = dispatcher.dispatch(**kwargs, budget_bytes=budget)
    assert next(iter(dispatcher.cache.values())) is compiled
    del result
    provider.arrays.append(jnp.ones(16, dtype=jnp.uint8))
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        dispatcher.dispatch(**kwargs, budget_bytes=budget)
    assert next(iter(dispatcher.cache.values())) is compiled
    assert len(dispatcher.cache) == 1
    state_ref, provider_ref = weakref.ref(state), weakref.ref(provider)
    del kwargs, provider, state
    gc.collect()
    assert state_ref() is None
    assert provider_ref() is None


def test_impossible_inventory_refuses_before_compilation() -> None:
    state = jnp.arange(4.0)
    dispatcher = ProfiledSimulationOperations()
    with pytest.raises(ExecutionPlanningError, match="budget"):
        dispatcher.dispatch(
            function=_shift,
            arguments={"state": state},
            subject_arg_names=("state",),
            devices=(jax.devices()[0],),
            live_footprint=_OwnedInputs(arrays=[state]),
            budget_devices=(jax.devices()[0],),
            budget_bytes=1,
        )
    assert dispatcher.cache == {}


def test_static_bindings_include_types_and_reject_arrays() -> None:
    dispatcher = ProfiledSimulationOperations()
    state = jnp.arange(4.0)
    kwargs: _BudgetedDispatchArguments = {
        "function": _typed_static_shift,
        "arguments": {"state": state},
        "subject_arg_names": ("state",),
        "devices": (jax.devices()[0],),
        "live_footprint": _OwnedInputs(arrays=[state]),
        "budget_devices": (jax.devices()[0],),
        "budget_bytes": 1_000_000,
    }
    boolean = dispatcher.dispatch(**kwargs, static_arguments={"selector": True})
    integer = dispatcher.dispatch(**kwargs, static_arguments={"selector": 1})
    np.testing.assert_array_equal(boolean, np.arange(4.0) + 1)
    np.testing.assert_array_equal(integer, np.arange(4.0) + 2)
    assert len(dispatcher.cache) == 2
    with pytest.raises(ExecutionPlanningError, match="static"):
        dispatcher.dispatch(**kwargs, static_arguments={"selector": state})


def test_ordered_subject_meshes_have_distinct_compiled_placement() -> None:
    devices = jax.devices()
    dispatcher = ProfiledSimulationOperations()
    for ordered in (
        (devices[3], devices[1], devices[2]),
        (devices[2], devices[3], devices[1]),
    ):
        mesh = jax.make_mesh(
            (3,), ("X",), (jax.sharding.AxisType.Auto,), devices=ordered
        )
        state = jax.device_put(jnp.arange(6.0), jax.NamedSharding(mesh, jax.P("X")))
        result = dispatcher.dispatch(
            function=_shift,
            arguments={"state": state},
            subject_arg_names=("state",),
            devices=ordered,
            live_footprint=_OwnedInputs(arrays=[state]),
            budget_devices=tuple(devices),
            budget_bytes=1_000_000,
        )
        assert isinstance(result, jax.Array)
        assert isinstance(result.sharding, jax.NamedSharding)
        assert tuple(result.sharding.mesh.devices.flat) == ordered
        np.testing.assert_array_equal(result, np.arange(6.0) + 1)
    assert len(dispatcher.cache) == 2


def test_static_float_bindings_preserve_signed_zero() -> None:
    dispatcher = ProfiledSimulationOperations()
    state = jnp.arange(4.0)
    for selector in (0.0, -0.0):
        result = dispatcher.dispatch(
            function=_static_zero_sign,
            arguments={"state": state},
            subject_arg_names=("state",),
            devices=(jax.devices()[0],),
            live_footprint=_OwnedInputs(arrays=[state]),
            budget_devices=(jax.devices()[0],),
            budget_bytes=1_000_000,
            static_arguments={"selector": selector},
        )
        np.testing.assert_array_equal(result, np.full(4, np.signbit(selector)))
    assert len(dispatcher.cache) == 2
