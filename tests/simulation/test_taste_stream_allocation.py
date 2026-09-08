"""Addressed taste keys use admitted profiles without retaining call owners."""

import gc
import weakref
from collections.abc import Callable
from functools import partialmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import compiler_peak_bytes
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.random import generate_simulation_keys
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.taste_stream import (
    advance_simulation_taste_key,
    create_taste_shock_key,
    generate_taste_shock_keys,
)
from lcm.exceptions import ExecutionPlanningError


def _memory(
    *, roots: object = (), operations: ProfiledSimulationOperations | None = None
) -> SimulationMemory:
    devices = (jax.devices()[0],)
    return SimulationMemory(
        budget_bytes=2**24,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations() if operations is None else operations,
        inputs=measure_buffer_footprint(tree=roots),
    )


def _invoke(
    *, operation: str, key: jax.Array, memory: SimulationMemory | None
) -> jax.Array:
    if operation == "root":
        created = create_taste_shock_key(seed=2**40 + 17, memory=memory)
        assert isinstance(created, jax.Array)
        return created
    if operation == "carry":
        return advance_simulation_taste_key(
            key=key, original_n_subjects=11, memory=memory
        )
    return generate_taste_shock_keys(
        key=key,
        address_words=tuple(range(8)),
        subject_slice=slice(4, 10),
        original_n_subjects=8,
        memory=memory,
    )


# keyword-only-exempt: library-callback=functools.partialmethod
def _record_dispatch(
    self: jax.stages.Compiled,
    *,
    original: Callable[..., object],
    calls: list[jax.stages.Compiled],
    **arguments: Any,
) -> object:
    calls.append(self)
    return original(self, **arguments)


@pytest.mark.parametrize("operation", ["root", "keys", "carry"])
def test_taste_allocations_use_the_profile_and_refuse_tiny_budgets_before_dispatch(
    *, operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real reported peaks admit exact outputs, then prevent an impossible dispatch."""
    key = jax.random.key(31, impl="threefry2x32")
    expected = np.asarray(
        jax.random.key_data(_invoke(operation=operation, key=key, memory=None))
    )
    memory = _memory(roots=key)
    calls: list[jax.stages.Compiled] = []
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _record_dispatch, original=jax.stages.Compiled.__call__, calls=calls
        ),
    )

    actual = _invoke(operation=operation, key=key, memory=memory)
    np.testing.assert_array_equal(jax.random.key_data(actual), expected)
    assert len(calls) == 1
    profile = next(iter(memory.operations.cache.values()))
    assert profile.executable is calls[0]
    assert profile.peak_bytes == compiler_peak_bytes(compiled=calls[0], widths={})
    assert profile.peak_bytes > 0

    memory.close_unit()
    del actual
    memory.budget_bytes = 1
    with pytest.raises(ExecutionPlanningError, match="budget"):
        _invoke(operation=operation, key=key, memory=memory)
    assert len(calls) == 1
    assert next(iter(memory.operations.cache.values())) is profile
    np.testing.assert_array_equal(
        jax.random.key_data(key),
        jax.random.key_data(jax.random.key(31, impl="threefry2x32")),
    )


def test_growing_residency_rechecks_the_cached_taste_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid cached executable cannot authorize newly retained output payloads."""
    key = jax.random.key(11, impl="threefry2x32")
    memory = _memory(roots=key)
    result = _invoke(operation="keys", key=key, memory=memory)
    profile = next(iter(memory.operations.cache.values()))
    memory.close_unit()
    del result
    memory.budget_bytes = profile.peak_bytes + 16
    calls: list[jax.stages.Compiled] = []
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _record_dispatch, original=jax.stages.Compiled.__call__, calls=calls
        ),
    )
    accepted = _invoke(operation="keys", key=key, memory=memory)
    assert calls == [profile.executable]
    memory.close_unit()
    del accepted
    retained = jnp.ones(32, dtype=jnp.uint8)
    memory.hold(retained)
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        _invoke(operation="keys", key=key, memory=memory)
    assert calls == [profile.executable]
    assert next(iter(memory.operations.cache.values())) is profile


def test_dynamic_age_and_global_row_words_reuse_one_taste_executable() -> None:
    """Only chunk shape specializes the profile; age, population and row words vary."""
    key = jax.random.key(47, impl="threefry2x32")
    memory = _memory(roots=key)
    compiled = []
    host_results = []
    for address, start, population in (
        (tuple(range(8)), 0, 9),
        (tuple(range(10, 18)), 2**32 - 1, 2**32 + 3),
        (tuple(range(8)), 0, 9),
    ):
        result = generate_taste_shock_keys(
            key=key,
            address_words=address,
            subject_slice=slice(start, start + 6),
            original_n_subjects=population,
            memory=memory,
        )
        assert len(memory.operations.cache) == 1
        compiled.append(next(iter(memory.operations.cache.values())))
        host_results.append(np.array(jax.random.key_data(result), copy=True))
        memory.close_unit()
    assert compiled[0] is compiled[1] is compiled[2]
    np.testing.assert_array_equal(host_results[0], host_results[2])
    assert np.any(host_results[0] != host_results[1])
    np.testing.assert_array_equal(host_results[1][-1], host_results[1][-2])


def test_taste_profile_cache_retains_no_root_output_or_memory_owner() -> None:
    """Closing the call releases arrays while the reusable compiler cache survives."""
    operations = ProfiledSimulationOperations()
    memory = _memory(operations=operations)
    key = create_taste_shock_key(seed=71, memory=memory)
    assert isinstance(key, jax.Array)
    result = _invoke(operation="keys", key=key, memory=memory)
    references = (weakref.ref(key), weakref.ref(result), weakref.ref(memory))
    memory.close_unit()
    del key, result, memory
    gc.collect()
    assert len(operations.cache) == 2
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg"])
@pytest.mark.parametrize("budgeted", [False, True])
def test_independent_override_preserves_the_original_carry_in_each_partition_mode(
    *, implementation: str, budgeted: bool
) -> None:
    """The remaining ordinary stream matches its original population-wide split."""
    with jax.default_prng_impl(implementation):
        key = jax.random.key(2**40 + 31)
        memory = _memory(roots=key) if budgeted else None
        for partitionable in (False, True, False):
            with jax.threefry_partitionable(partitionable):
                expected, _ = generate_simulation_keys(
                    key=key,
                    names=["taste_shock"],
                    n_initial_states=12,
                    original_n_subjects=11,
                    subject_slice=slice(6, 12),
                )
                actual = advance_simulation_taste_key(
                    key=key, original_n_subjects=11, memory=memory
                )
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(
                jax.random.key_data(actual), jax.random.key_data(expected)
            )
            if memory is not None:
                memory.close_unit()
        if memory is not None:
            assert len(memory.operations.cache) == 2
