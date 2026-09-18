"""Random setup must reach compiler admission before creating device payloads."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import random as simulation_random
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.random import create_simulation_key, split_simulation_key
from _lcm.simulation.residency import measure_buffer_footprint
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


def _guard_random_allocation(
    *, original: Callable[..., Any], operation: str, **kwargs: Any
) -> Any:
    """Permit tracing, but reject the old eager device-array construction."""
    argument = kwargs["seed"] if operation == "seed" else kwargs["key"]
    is_period_split = operation == "seed" or kwargs.get("num") == 3
    if is_period_split and not isinstance(argument, jax.core.Tracer):
        raise AssertionError(f"Unprofiled simulation random allocation: {operation}")
    return original(**kwargs)


@pytest.mark.parametrize("operation", ["seed", "period_split"])
def test_budgeted_random_setup_allocates_only_inside_profiled_code(
    *, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A budgeted public simulation cannot eagerly allocate its random key arrays."""
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray([_LifecycleRegimeId.alive], dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    name = "key" if operation == "seed" else "split"
    original = getattr(jax.random, name)
    monkeypatch.setattr(
        jax.random,
        name,
        functools.partial(
            _guard_random_allocation, original=original, operation=operation
        ),
    )
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        seed=2**32 - 1,
    )
    assert result.n_subjects == 1


@pytest.mark.parametrize("seed", [0, -1, 2**31, 2**32 - 1, 2**40 + 17])
def test_budgeted_keys_preserve_seed_bits_and_split_streams(seed: int) -> None:
    """Signed and high-bit seeds produce exactly the existing JAX streams."""
    devices = (jax.devices()[0],)
    memory = SimulationMemory(
        budget_bytes=2**24,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=()),
    )
    original = jax.random.key(seed=seed)
    created = create_simulation_key(seed=seed, memory=memory)
    np.testing.assert_array_equal(
        jax.random.key_data(created), jax.random.key_data(original)
    )
    split = split_simulation_key(key=created, memory=memory)
    expected_split = jax.random.split(key=original, num=3)
    for actual, expected in zip(split, expected_split, strict=True):
        np.testing.assert_array_equal(
            jax.random.key_data(actual), jax.random.key_data(expected)
        )


@pytest.mark.parametrize(
    "implementations",
    [
        ("threefry2x32", "rbg", "threefry2x32"),
        ("rbg", "threefry2x32", "rbg"),
    ],
)
def test_cached_key_creation_follows_the_current_prng_implementation(
    implementations: tuple[str, ...],
) -> None:
    """One memory/cache owner must follow each public default-PRNG context."""
    devices = (jax.devices()[0],)
    memory = SimulationMemory(
        budget_bytes=2**24,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=()),
    )
    seed = 2**40 + 17
    for implementation in implementations:
        with jax.default_prng_impl(implementation):
            expected = jax.random.key(seed=seed)
            actual = create_simulation_key(seed=seed, memory=memory)
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(
                jax.random.key_data(actual), jax.random.key_data(expected)
            )
            for got, want in zip(
                split_simulation_key(key=actual, memory=memory),
                jax.random.split(key=expected, num=3),
                strict=True,
            ):
                assert got.dtype == want.dtype
                np.testing.assert_array_equal(
                    jax.random.key_data(got), jax.random.key_data(want)
                )


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg"])
def test_cached_key_creation_follows_the_current_seed_offset(
    implementation: str,
) -> None:
    """Seed-offset contexts cannot reuse an executable for another offset."""
    devices = (jax.devices()[0],)
    memory = SimulationMemory(
        budget_bytes=2**24,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=()),
    )
    seed = 2**40 + 17
    original_offset = jax.config.jax_random_seed_offset
    try:
        with jax.default_prng_impl(implementation):
            for offset in (0, 17, 0):
                jax.config.update("jax_random_seed_offset", offset)
                expected = jax.random.key(seed=seed)
                actual = create_simulation_key(seed=seed, memory=memory)
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(
                    jax.random.key_data(actual), jax.random.key_data(expected)
                )
    finally:
        jax.config.update("jax_random_seed_offset", original_offset)


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg"])
def test_cached_key_splitting_follows_the_current_partition_mode(
    implementation: str,
) -> None:
    """The same typed key/cache follows the selected Threefry splitting rule."""
    devices = (jax.devices()[0],)
    memory = SimulationMemory(
        budget_bytes=2**24,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=()),
    )
    key = jax.random.key(seed=17, impl=implementation)
    for partitionable in (False, True, False):
        with jax.threefry_partitionable(partitionable):
            expected = jax.random.split(key=key, num=3)
            actual = split_simulation_key(key=key, memory=memory)
            for got, want in zip(actual, expected, strict=True):
                assert got.dtype == want.dtype
                np.testing.assert_array_equal(
                    jax.random.key_data(got), jax.random.key_data(want)
                )


def _change_seed_offset_before_tracing(
    *, original: Callable[..., Any], **kwargs: Any
) -> Any:
    """Represent an external config change after the wrapper snapshots metadata."""
    initial_offset = jax.config.jax_random_seed_offset
    try:
        jax.config.update("jax_random_seed_offset", initial_offset + 1)
        return original(**kwargs)
    finally:
        jax.config.update("jax_random_seed_offset", initial_offset)


def test_key_creation_refuses_seed_offset_changed_before_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed trace context cannot be cached under the snapshotted offset."""
    devices = (jax.devices()[0],)
    memory = SimulationMemory(
        budget_bytes=2**24,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=()),
    )
    monkeypatch.setattr(
        simulation_random,
        "run_simulation_operation",
        functools.partial(
            _change_seed_offset_before_tracing,
            original=simulation_random.run_simulation_operation,
        ),
    )
    with pytest.raises(
        ExecutionPlanningError, match="seed offset changed before tracing"
    ):
        create_simulation_key(seed=17, memory=memory)
    assert not memory.operations.cache
