"""Identity, admission and lifetime contract for the runtime-owned profile cache.

A bounded, versioned, abstract-only profile cache shared across a model's
`_simulate_runtime_regimes` runtimes. A hit reuses the immutable multi-stage
`SimulationChunkProfile`; it never skips the caller's admission formula and
never retains caller arrays, results or closures over the current call.
"""

import gc
import inspect
import threading
import weakref
from collections.abc import Callable, Iterator
from functools import partial
from types import CodeType, MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import _lcm.simulation.chunk_admission as admission
import _lcm.simulation.chunk_profile_inventory as chunk_profile_inventory_module
from _lcm.simulation.chunk_admission import (
    _simulation_chunk_profile_key,
)
from _lcm.simulation.chunk_inputs import SimulationCallInputs
from _lcm.simulation.chunk_planning import SimulationChunkProfile
from _lcm.simulation.chunk_profile_cache import (
    PROFILE_CACHE_MAX_ENTRIES,
    ChunkProfileCacheRegistry,
    ProfileCacheToken,
    profile_cache_registry,
)
from _lcm.simulation.runtime import SimulationRuntime
from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.execution import ExecutionConfig
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


def _utility(*, wealth: ContinuousState, saving: ContinuousAction) -> FloatND:
    return wealth + saving


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _next_wealth(*, wealth: ContinuousState, saving: ContinuousAction) -> FloatND:
    return wealth + saving


def _next_regime() -> ScalarInt:
    return _RegimeId.done


def _only_initial_age(age: float) -> bool:
    return age == 0


def _budgeted_model(*, device_memory_bytes: int = 2**32) -> Model:
    return Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=_only_initial_age,
                functions={"utility": _utility},
                actions={"saving": LinSpacedGrid(start=1, stop=2, n_points=2)},
            ),
            "done": Regime(transition=None, functions={"utility": _terminal_utility}),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=5, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=device_memory_bytes),
    )


def _initial(*, n: int, dtype: jnp.dtype = jnp.float32) -> dict[str, jax.Array]:
    return {
        "wealth": jnp.linspace(1.0, 2.0, n, dtype=dtype),
        "age": jnp.zeros(n),
        "regime_id": jnp.full(n, _RegimeId.alive, dtype=jnp.int32),
    }


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    """Give every test a clean, isolated aggregate cache."""
    profile_cache_registry().clear()
    yield
    profile_cache_registry().clear()


def _count_builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count real calls to `profile_simulation_chunk`, i.e. cache misses."""
    calls = [0]
    original = admission.profile_simulation_chunk

    def counted(**kwargs: Any) -> SimulationChunkProfile:
        calls[0] += 1
        return original(**kwargs)

    monkeypatch.setattr(admission, "profile_simulation_chunk", counted)
    return calls


def _simulate(*, model: Model, initial: dict[str, jax.Array]) -> object:
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = model.solve(params=params, log_level="off")
    return model.simulate(
        params=params, initial_conditions=initial, solution=solution, log_level="off"
    )


def test_repeated_call_same_metadata_reuses_profile_and_reruns_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shapes/dtypes but fresh numerical values: profile hits, admission reruns."""
    model = _budgeted_model()
    calls = _count_builds(monkeypatch)
    admission_calls: list[int] = []
    original_required = admission._required_bytes

    def counted_admission(**kwargs: Any) -> object:
        admission_calls.append(1)
        return original_required(**kwargs)

    monkeypatch.setattr(admission, "_required_bytes", counted_admission)

    _simulate(model=model, initial=_initial(n=3))
    first_builds = calls[0]
    first_admissions = len(admission_calls)
    assert first_builds > 0
    assert first_admissions > 0

    # Different numerical values, identical shapes/dtypes/widths: must hit.
    _simulate(
        model=model, initial=_initial(n=3) | {"wealth": jnp.asarray([9.0, 9.0, 9.0])}
    )
    assert calls[0] == first_builds, (
        "same-shaped fresh values must hit the profile cache"
    )
    assert len(admission_calls) > first_admissions, (
        "a profile-cache hit must never skip admission"
    )


def test_population_shape_change_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _budgeted_model()
    calls = _count_builds(monkeypatch)
    _simulate(model=model, initial=_initial(n=3))
    after_first = calls[0]
    _simulate(model=model, initial=_initial(n=4))
    assert calls[0] > after_first, "a different original/padded population must miss"


def _budgeted_runtime(*, model: Model) -> SimulationRuntime:
    """Return the one budgeted runtime this model builds for a three-subject batch."""
    regimes = model._runtime_regimes_for_shape(compile_batch_size=3)
    runtime = next(iter(regimes.values())).simulation.programs.executor
    assert isinstance(runtime, SimulationRuntime)
    return runtime


def _key_for_initial_conditions(
    *, model: Model, initial: dict[str, jax.Array]
) -> tuple:
    regimes = model._runtime_regimes_for_shape(compile_batch_size=3)
    runtime = _budgeted_runtime(model=model)
    return _simulation_chunk_profile_key(
        runtime=runtime,
        regimes=regimes,
        call_inputs=SimulationCallInputs(
            devices=runtime.subject_devices,
            flat_params=MappingProxyType({}),
            base_state_action_spaces=MappingProxyType({}),
        ),
        values={},
        flags={},
        policies=None,
        ages=model.ages,
        initial_conditions=initial,
        regime_names_to_ids=model.regime_names_to_ids,
        n_subjects=3,
        population=3,
        original_population=3,
        widths={"subject": 3},
        independent_taste=False,
        log_level="off",
    )


def test_dtype_change_misses() -> None:
    """A dtype change on a dynamic leaf (x64 is disabled) must change the key."""
    model = _budgeted_model()
    base = _initial(n=3)
    key_a = _key_for_initial_conditions(
        model=model, initial=base | {"age": jnp.zeros(3, dtype=jnp.float32)}
    )
    key_b = _key_for_initial_conditions(
        model=model, initial=base | {"age": jnp.zeros(3, dtype=jnp.int32)}
    )
    assert key_a != key_b, "a dtype change must produce a different profile key"


def test_weak_type_change_misses() -> None:
    """A weak-type change on a dynamic leaf must change the key."""
    model = _budgeted_model()
    base = _initial(n=3)
    strong = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)
    weak = jax.device_put(1.0) * jnp.ones(3, dtype=jnp.float32)
    key_a = _key_for_initial_conditions(model=model, initial=base | {"wealth": strong})
    key_b = _key_for_initial_conditions(model=model, initial=base | {"wealth": weak})
    if strong.weak_type == weak.weak_type:
        pytest.skip("this JAX build did not produce a weak-type leaf for the probe")
    assert key_a != key_b, "a weak-type change must produce a different profile key"


def test_log_level_change_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _budgeted_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = model.solve(params=params, log_level="off")
    calls = _count_builds(monkeypatch)
    model.simulate(
        params=params,
        initial_conditions=_initial(n=3),
        solution=solution,
        log_level="off",
    )
    after_first = calls[0]
    model.simulate(
        params=params,
        initial_conditions=_initial(n=3),
        solution=solution,
        log_level="warning",
    )
    assert calls[0] > after_first, "a changed diagnostic log level must miss"


def test_key_is_stable_and_versioned() -> None:
    """The canonical key is a plain hashable tuple carrying an explicit version."""
    model = _budgeted_model()
    regimes = model._runtime_regimes_for_shape(compile_batch_size=3)
    key_kwargs: dict[str, Any] = {
        "runtime": _budgeted_runtime(model=model),
        "regimes": regimes,
        "call_inputs": None,
        "values": {},
        "flags": {},
        "policies": None,
        "ages": model.ages,
        "initial_conditions": _initial(n=3),
        "regime_names_to_ids": model.regime_names_to_ids,
        "n_subjects": 3,
        "population": 3,
        "original_population": 3,
        "widths": {"subject": 3},
        "independent_taste": False,
        "log_level": "off",
    }

    key_kwargs["call_inputs"] = SimulationCallInputs(
        devices=(jax.devices()[0],),
        flat_params=MappingProxyType({}),
        base_state_action_spaces=MappingProxyType({}),
    )
    first = _simulation_chunk_profile_key(**key_kwargs)
    second = _simulation_chunk_profile_key(**key_kwargs)
    assert first == second
    assert hash(first) == hash(second)
    assert first[0] == ("chunk-profile-key", 1)


_RUNTIME_MARKER = "test-runtime-marker"


def test_concurrent_requests_share_one_immutable_build() -> None:
    """Only one thread actually builds; every waiter observes the same object."""
    registry = ChunkProfileCacheRegistry()
    started = threading.Event()
    release = threading.Event()
    build_calls = [0]
    sentinel = object()

    def slow_build() -> object:
        build_calls[0] += 1
        started.set()
        release.wait(timeout=5)
        return sentinel

    def fast_build() -> object:
        build_calls[0] += 1
        return sentinel

    results: list[object] = []

    def worker(build: Callable[[], object]) -> None:
        results.append(
            registry.get_or_build(runtime_token=_RUNTIME_MARKER, key="k", build=build)
        )

    first = threading.Thread(target=worker, args=(slow_build,))
    first.start()
    started.wait(timeout=5)
    second = threading.Thread(target=worker, args=(fast_build,))
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert build_calls[0] == 1
    assert results == [sentinel, sentinel]


def test_failed_build_publishes_nothing() -> None:
    registry = ChunkProfileCacheRegistry()

    class _BoomError(RuntimeError):
        pass

    def failing_build() -> object:
        raise _BoomError("build failed")

    with pytest.raises(_BoomError):
        registry.get_or_build(
            runtime_token=_RUNTIME_MARKER, key="k", build=failing_build
        )
    assert len(registry) == 0

    def succeeding_build() -> object:
        return object()

    result = registry.get_or_build(
        runtime_token=_RUNTIME_MARKER, key="k", build=succeeding_build
    )
    assert result is not None
    assert len(registry) == 1


def test_restore_resets_the_profile_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing runtime reset boundary builds a fresh runtime and token.

    `Model.__setstate__` (the documented restore/unpickle/rebind boundary,
    `lcm/model.py`) always rebuilds `_simulate_runtime_regimes` from scratch,
    so it always constructs a fresh `SimulationRuntime` with a fresh
    `profile_cache_token`, regardless of whether the surrounding state
    round-trips through `pickle` byte-for-byte. Drive that exact boundary
    directly rather than the full pickle machinery, which has a pre-existing,
    unrelated `mappingproxy` limitation for this model's `fixed_params`.
    """
    model = _budgeted_model()
    calls = _count_builds(monkeypatch)
    _simulate(model=model, initial=_initial(n=3))
    after_first = calls[0]

    state = model.__getstate__()
    model.__setstate__(state)
    _simulate(model=model, initial=_initial(n=3))
    assert calls[0] > after_first, (
        "the runtime reset boundary must not reuse the prior runtime's profiles"
    )


def test_no_caller_arrays_or_results_are_retained() -> None:
    """The cache never keeps a live caller array, result, or dispatch closure alive."""
    model = _budgeted_model()
    initial = _initial(n=3)
    wealth_ref = weakref.ref(initial["wealth"])
    result = _simulate(model=model, initial=initial)
    result_ref = weakref.ref(result)
    del initial, result
    gc.collect()
    assert wealth_ref() is None, "the profile cache retained a caller input array"
    assert result_ref() is None, "the profile cache retained a simulation result"


def test_bounded_aggregate_entries_across_heterogeneous_runtimes() -> None:
    """The aggregate LRU is bounded across many distinct runtimes, not per-runtime."""
    registry = ChunkProfileCacheRegistry()
    for shape in range(PROFILE_CACHE_MAX_ENTRIES + 20):
        registry.get_or_build(
            runtime_token=f"runtime-{shape}",
            key=f"key-{shape}",
            build=object,
        )
    assert len(registry) <= PROFILE_CACHE_MAX_ENTRIES


def test_warm_repeat_call_does_not_add_new_trace_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile-cache hit must not reissue the abstract host-operation traces.

    `abstract_tree` is only ever invoked from `ChunkProfileInventory.operation`,
    i.e. from inside `profile_simulation_chunk` (see
    `chunk_profile_inventory.py`). When the profile cache hits, that whole walk
    — and therefore every JAX trace request issued while building it — is
    skipped, so the count of `abstract_tree` calls on a warm same-shape call is
    zero, not merely no larger than on the cold call.
    """
    model = _budgeted_model()
    inventory_module = chunk_profile_inventory_module
    abstract_tree_calls = [0]
    original_abstract_tree = inventory_module.abstract_tree

    def counted(**kwargs: Any) -> object:
        abstract_tree_calls[0] += 1
        return original_abstract_tree(**kwargs)

    monkeypatch.setattr(inventory_module, "abstract_tree", counted)
    monkeypatch.setattr(
        "_lcm.simulation.chunk_profiles.abstract_tree", counted, raising=False
    )

    _simulate(model=model, initial=_initial(n=3))
    first_count = abstract_tree_calls[0]
    assert first_count > 0, "the cold call must build the profile via abstract_tree"

    abstract_tree_calls[0] = 0
    _simulate(
        model=model, initial=_initial(n=3) | {"wealth": jnp.asarray([9.0, 9.0, 9.0])}
    )
    assert abstract_tree_calls[0] == 0, (
        "a warm same-shape call must not retrace abstract host operations"
    )


def test_profile_widths_defines_no_nested_callable() -> None:
    """The per-call profiling path must not define a callable on every call.

    The package's beartype claw decorates a nested `def` anew on each call of
    its enclosing function, and such a callable would close over the profiler
    and so keep this call's arrays and compiled executables reachable from the
    retained wrapper. The builder handed to the cache is therefore a
    `functools.partial` over a module-level function, which carries no code
    object of its own.
    """
    body = inspect.unwrap(admission._ChunkProfiler.profile_widths)
    nested = [
        constant.co_name
        for constant in body.__code__.co_consts
        if isinstance(constant, CodeType)
    ]
    assert nested == [], "profile_widths must not define a nested callable"


def test_registry_does_not_retain_the_builder_or_its_bound_arguments() -> None:
    """Nothing reachable from the builder outlives the call that supplied it."""
    registry = ChunkProfileCacheRegistry()
    token = ProfileCacheToken()

    class _BoundArgument:
        """Stand in for a caller array or compiled executable bound to a builder."""

    argument = _BoundArgument()
    argument_ref = weakref.ref(argument)
    builder = partial(_returns_sentinel, argument=argument)
    registry.get_or_build(runtime_token=token, key="k", build=builder)

    del argument, builder
    gc.collect()
    assert argument_ref() is None, "the registry retained the builder's arguments"


def _returns_sentinel(*, argument: object) -> object:
    """Build a profile stand-in that does not reference the bound argument."""
    assert argument is not None
    return object()


def test_dropped_runtime_releases_its_cached_profiles() -> None:
    """A runtime's profiles are released with it, not held until LRU eviction."""
    registry = ChunkProfileCacheRegistry()
    token = ProfileCacheToken()
    registry.get_or_build(runtime_token=token, key="k", build=object)
    assert len(registry) == 1

    del token
    gc.collect()
    assert len(registry) == 0, "a dropped runtime's profiles were not released"


def test_dropped_runtime_release_frees_its_cached_profile_object() -> None:
    """The released entry drops its reference to the cached profile itself."""
    registry = ChunkProfileCacheRegistry()
    token = ProfileCacheToken()
    profile = registry.get_or_build(runtime_token=token, key="k", build=_Profile)
    profile_ref = weakref.ref(profile)

    del token, profile
    gc.collect()
    len(registry)
    gc.collect()
    assert profile_ref() is None, "a released entry still held its cached profile"


class _Profile:
    """Weak-referenceable stand-in for a cached profile holding executables."""
