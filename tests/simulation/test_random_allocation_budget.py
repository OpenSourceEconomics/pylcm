"""Random setup must reach compiler admission before creating device payloads."""

from collections.abc import Callable
from typing import Literal

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import random as simulation_random
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.random import create_simulation_key, split_simulation_key
from _lcm.simulation.random_address import RandomAddress, Word, draw_row_keys
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.typing import PRNGKeyND
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import PRNGKey
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg", "unsafe_rbg"])
@pytest.mark.parametrize("compiled", [False, True])
def test_named_row_keys_preserve_scalar_fold_order(
    *,
    implementation: Literal["threefry2x32", "rbg", "unsafe_rbg"],
    compiled: bool,
) -> None:
    """Batched keys retain the scalar address and high/low/period word order."""
    with jax.default_prng_impl(implementation):
        key = jax.random.key(seed=29, impl=implementation)
        address_words = np.asarray(
            RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
        )
        row_high = np.asarray([0, 1, 0xFFFFFFFF], dtype=np.uint32)
        row_low = np.asarray([0xFFFFFFFF, 0, 7], dtype=np.uint32)
        period = np.uint32(0xFFFFFFFF)
        expected = []
        for high, low in zip(row_high, row_low, strict=True):
            cell = key
            for word in (*address_words, high, low, period):
                cell = jax.random.fold_in(cell, word)
            expected.append(np.asarray(jax.random.key_data(cell)))
        function = jax.jit(draw_row_keys) if compiled else draw_row_keys
        keys = function(
            key=key,
            address_words=address_words,
            row_high=row_high,
            row_low=row_low,
            period=period,
        )
        actual = jax.random.key_data(keys)
        scalar = np.stack(expected)
        if implementation != "threefry2x32":
            scalar = scalar[:, None]
        np.testing.assert_array_equal(actual, scalar)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("persistent_compilation_cache", [False], indirect=True)
@pytest.mark.parametrize("mapped_periods", [False, True])
def test_repeated_eager_named_keys_reuse_compiled_code(
    *, mapped_periods: bool, persistent_compilation_cache: bool
) -> None:
    """Changing the root reuses eager row-key code, including period batching."""
    assert persistent_compilation_cache is False
    address_words = np.asarray(
        RandomAddress(kind="shock", name="eager_reuse").words(), dtype=np.uint32
    )
    high = np.asarray([0, 1, 0, 1], dtype=np.uint32)
    low = np.asarray([0, 7, 0xFFFFFFFF, 0], dtype=np.uint32)
    periods = jnp.asarray([0, 1], dtype=jnp.uint32)
    roots = tuple(jax.random.key(seed, impl="threefry2x32") for seed in (41, 42))

    def draw(*, key: PRNGKey) -> PRNGKeyND:
        def at_period(period: Word) -> PRNGKeyND:
            return draw_row_keys(
                key=key,
                address_words=address_words,
                row_high=high,
                row_low=low,
                period=period,
            )

        return jax.vmap(at_period)(periods) if mapped_periods else at_period(periods[0])

    events: list[str] = []

    # keyword-only-exempt: library-callback=jax.monitoring
    def record_compile(
        event: str, start_time: float, end_time: float, **metadata: str | int
    ) -> None:
        del start_time, end_time
        if event == "/jax/core/compile/backend_compile_duration":
            events.append(str(metadata.get("fun_name", "")))

    jax.monitoring.register_event_time_span_listener(record_compile)
    try:
        first = jax.block_until_ready(draw(key=roots[0]))
        cold = events.copy()
        events.clear()
        second = jax.block_until_ready(draw(key=roots[1]))
        repeated = events.copy()
    finally:
        jax.monitoring.unregister_event_time_span_listener(record_compile)
    assert cold, "The first call must exercise the compile observer."
    assert not repeated, f"A new root recompiled eager row-key code: {repeated}"
    assert np.any(jax.random.key_data(first) != jax.random.key_data(second))


def _guard_random_allocation[**P, Result](
    *, original: Callable[P, Result], operation: str
) -> Callable[P, Result]:
    def call(*args: P.args, **kwargs: P.kwargs) -> Result:
        """Permit tracing, but reject the old eager device-array construction."""
        argument = kwargs["seed"] if operation == "seed" else kwargs["key"]
        is_period_split = operation == "seed" or kwargs.get("num") == 3
        if is_period_split and not isinstance(argument, jax.core.Tracer):
            raise AssertionError(
                f"Unprofiled simulation random allocation: {operation}"
            )
        return original(*args, **kwargs)

    return call


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
        _guard_random_allocation(original=original, operation=operation),
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
def test_cached_key_creation_is_threefry_under_every_default_prng(
    implementations: tuple[str, ...],
) -> None:
    """One memory/cache owner creates the same Threefry root under every default."""
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
            expected = jax.random.key(seed=seed, impl="threefry2x32")
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
                expected = jax.random.key(seed=seed, impl="threefry2x32")
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


def _change_seed_offset_before_tracing[**P, Result](
    *, original: Callable[P, Result]
) -> Callable[P, Result]:
    def call(*args: P.args, **kwargs: P.kwargs) -> Result:
        """Represent an external config change after the wrapper snapshots metadata."""
        initial_offset = jax.config.jax_random_seed_offset
        try:
            jax.config.update("jax_random_seed_offset", initial_offset + 1)
            return original(*args, **kwargs)
        finally:
            jax.config.update("jax_random_seed_offset", initial_offset)

    return call


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
        _change_seed_offset_before_tracing(
            original=simulation_random.run_simulation_operation,
        ),
    )
    with pytest.raises(
        ExecutionPlanningError, match="seed offset changed before tracing"
    ):
        create_simulation_key(seed=17, memory=memory)
    assert not memory.operations.cache
