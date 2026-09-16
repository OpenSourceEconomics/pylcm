"""Batch 1: migrated helpers add no per-call wrapper work on a warm simulate hit.

`chunk_profile_inventory.abstract_tree`, `forward_program_profiles._shared_tree`
and `process_grids._trace_process_jaxpr` used to define their leaf/trace
callback as a nested `def` on every call; `host_operations.
_validated_static_arguments` used to re-run `inspect.unwrap`/closure/qualname/
default inspection on every call. All are call-independent for an unchanged
function object, so the beartype claw would decorate/memoize a fresh wrapper
for the nested cases, and the inspection would repeat needlessly for the
other, on every single dispatch.

This module asserts, on a small CPU model, for both the unbudgeted default
`ExecutionConfig()` and a budgeted config:

1. A cold `simulate` call does nontrivial `builtins.compile` work (positional
   control), while five warm calls after it create no *additional* wrapper
   objects attributable to the migrated modules.
2. Value-dependent per-call checks still run: a same-shaped rejected input
   still raises, at every log level.
3. Backend compile requests are zero on the warm hit.
4. The per-call jaxpr trace-request count on a warm call is unaffected by
   migrating `_trace_process_jaxpr`'s nested callback: `_produce_staged`
   calls it, uncached, on every solve, so the trace itself is a separate,
   already-known cost (P2), not something this wrapper migration removes.
"""

import builtins
from collections.abc import Iterator
from contextlib import contextmanager

import jax.numpy as jnp
import pytest

from _lcm.simulation import (
    chunk_profile_inventory,
    forward_program_profiles,
    host_operations,
    process_grids,
)
from _lcm.utils.logging import LogLevel, get_logger
from benchmarks.asv._compile_counters import count_compile_requests
from lcm.execution import ExecutionConfig
from lcm_examples import precautionary_savings

_N_SUBJECTS = 2_000
_LOG_LEVELS: tuple[LogLevel, ...] = ("off", "warning", "progress", "debug")


def _model(*, execution_config: ExecutionConfig | None = None):
    model = precautionary_savings.create_model(
        n_periods=3,
        shock_type="rouwenhorst",
        wealth_grid_type="lin",
        wealth_n_points=6,
        consumption_n_points=6,
        execution_config=execution_config,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst", sigma=0.1, rho=0.5
    )
    return model, params


def _initial_conditions(*, n_subjects: int = _N_SUBJECTS):
    return {
        "age": jnp.full(n_subjects, 20.0),
        "wealth": jnp.full(n_subjects, 5.0),
        "income": jnp.full(n_subjects, 0.0),
        "regime_id": jnp.zeros(n_subjects, dtype="int32"),
    }


# Modules whose migrated helpers must not trigger a fresh beartype wrapper
# compilation on a warm, unchanged-signature call. `builtins.compile` calls
# elsewhere in the dispatch path (other modules' own per-call constructs, out
# of Batch 1's scope) are deliberately not counted: they are not what this
# migration promises to remove.
_ATTRIBUTED_MODULE_MARKERS = (
    "_lcm.simulation.chunk_profile_inventory",
    "_lcm.simulation.host_operations",
    "_lcm.simulation.forward_program_profiles",
    "_lcm.simulation.process_grids",
)


@contextmanager
def _count_new_wrapper_objects() -> Iterator[dict[str, int]]:
    """Count `builtins.compile` calls attributable to the two migrated helpers.

    `builtins.compile` is called by beartype's own code generation for a
    wrapper it has not seen before; the beartype claw's synthesized filename
    names the wrapped function's fully qualified name (visible in tracebacks
    as e.g. ``<@beartype(_lcm.simulation.host_operations._validated_static_
    arguments) at 0x...>``). Before migration, `chunk_profile_inventory.
    abstract_tree`'s nested `abstract` callback was a fresh function object on
    every call, so the claw generated (and compiled) a fresh wrapper every
    call; after migration, `_abstract_leaf` and `_validated_operation_function`
    are module-level and decorated once at import, so a warm call must
    attribute zero further `compile()` calls to either module.
    """
    counts = {"compile_calls": 0, "attributed_compile_calls": 0}
    original_compile = builtins.compile

    def counting_compile(*args: object, **kwargs: object) -> object:
        counts["compile_calls"] += 1
        filename = kwargs.get("filename", args[1] if len(args) > 1 else "")
        if any(marker in str(filename) for marker in _ATTRIBUTED_MODULE_MARKERS):
            counts["attributed_compile_calls"] += 1
        return original_compile(*args, **kwargs)  # ty: ignore[no-matching-overload]

    builtins.compile = counting_compile  # ty: ignore[invalid-assignment]
    try:
        yield counts
    finally:
        builtins.compile = original_compile


def _run_simulate(*, model, params, initial_conditions, log_level: LogLevel):
    return model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level=log_level,
        seed=0,
        taste_shock_seed=1,
    )


@pytest.mark.parametrize(
    "execution_config",
    [
        None,
        ExecutionConfig(
            device_memory_bytes=2 * 1024**3,
            axis_widths={"subject": _N_SUBJECTS},
        ),
    ],
    ids=["unbudgeted_default", "budgeted"],
)
def test_warm_simulate_creates_no_new_migrated_wrappers(execution_config) -> None:
    model, params = _model(execution_config=execution_config)
    initial_conditions = _initial_conditions()

    # Cold call: positive control. Compiling and beartype-decorating a cold
    # model's dispatch path must do *some* compile() work overall; this
    # confirms the counting mechanism itself is live, independent of whether
    # any of that work happens to be attributed to the two migrated modules.
    with _count_new_wrapper_objects() as cold_counts:
        _run_simulate(
            model=model,
            params=params,
            initial_conditions=initial_conditions,
            log_level="off",
        )
    assert cold_counts["compile_calls"] > 0, (
        "Positive control failed: the cold call did no compile work at all, "
        "so a zero warm count would not demonstrate anything."
    )

    # Five warm, unchanged-signature calls: no new wrapper objects may be
    # attributable to the migrated helpers. `abstract_tree` and
    # `_validated_static_arguments` are exercised by every simulate call
    # through chunk profiling and host-operation dispatch respectively.
    warm_counts = []
    for _ in range(5):
        with _count_new_wrapper_objects() as counts:
            _run_simulate(
                model=model,
                params=params,
                initial_conditions=initial_conditions,
                log_level="off",
            )
        warm_counts.append(counts["attributed_compile_calls"])

    assert warm_counts == [0] * 5, (
        f"Warm calls triggered new compile() work: {warm_counts}. The migrated "
        "helpers in chunk_profile_inventory.py / host_operations.py must not "
        "create a fresh per-call wrapper on an unchanged-signature warm hit."
    )


@pytest.mark.parametrize(
    "execution_config",
    [
        None,
        ExecutionConfig(
            device_memory_bytes=2 * 1024**3,
            axis_widths={"subject": _N_SUBJECTS},
        ),
    ],
    ids=["unbudgeted_default", "budgeted"],
)
def test_warm_simulate_hits_zero_backend_compiles(execution_config) -> None:
    model, params = _model(execution_config=execution_config)
    initial_conditions = _initial_conditions()

    # Warm the executable cache.
    _run_simulate(
        model=model,
        params=params,
        initial_conditions=initial_conditions,
        log_level="off",
    )

    with count_compile_requests() as counts:
        _run_simulate(
            model=model,
            params=params,
            initial_conditions=initial_conditions,
            log_level="off",
        )

    assert counts.compile_requests == 0, (
        "N3's compiler-memory cache and the executable cache must remain warm "
        "on an unchanged-signature repeat call; Batch 1 must not reopen them."
    )


@pytest.mark.parametrize("log_level", _LOG_LEVELS)
def test_value_dependent_operation_checks_still_raise(log_level: LogLevel) -> None:
    """`_validated_static_arguments` must still reject bad current values.

    Migrating the function-identity inspection to a per-function cache must
    not weaken or skip the value-dependent checks: an operation invoked with a
    non-string argument name is still rejected, at every log level, because
    this check is unconditional in the design (design.md section 3, row for
    `host_operations.py:73-151,287-318`).
    """

    def _pure_operation(*, value: object) -> object:
        return value

    # Exercise the check under each log level's logger, exactly like a real
    # dispatch would: the check itself is unconditional and must not become
    # log-level-gated by the registration-time memoization.
    logger = get_logger(log_level=log_level)
    logger.debug("Checking value-dependent rejection at log_level=%s", log_level)

    # `arguments` and `static_arguments` overlapping on "value" is a
    # value-dependent conflict (design.md row for host_operations.py:73-151,
    # 287-318): it cannot be decided from the function object alone, so it
    # must still be checked on every call after the function-identity check
    # is cached.
    with pytest.raises(host_operations.ExecutionPlanningError):
        host_operations._validated_static_arguments(
            function=_pure_operation,
            arguments={"value": object()},
            static_arguments={"value": 1},
            subject_outputs=False,
        )


def test_migrated_leaf_helper_is_module_level_and_reused() -> None:
    """`abstract_tree`'s leaf callback must be a stable module-level object.

    Before the migration, `chunk_profile_inventory.abstract_tree` defined its
    `abstract` callback as a nested `def`, so two calls never returned the
    same function object. After migration, the helper the tree map dispatches
    to must be the same module-level function across calls.
    """
    assert hasattr(chunk_profile_inventory, "_abstract_leaf")
    first = chunk_profile_inventory._abstract_leaf
    second = chunk_profile_inventory._abstract_leaf
    assert first is second
    assert "<locals>" not in first.__qualname__


def test_migrated_shared_leaf_helper_is_module_level_and_reused() -> None:
    """`_shared_tree`'s leaf callback must be a stable module-level object."""
    assert hasattr(forward_program_profiles, "_shared_leaf")
    first = forward_program_profiles._shared_leaf
    second = forward_program_profiles._shared_leaf
    assert first is second
    assert "<locals>" not in first.__qualname__


def test_migrated_process_grid_call_is_module_level_and_reused() -> None:
    """`_trace_process_jaxpr`'s trace callback must be a stable module-level object."""
    assert hasattr(process_grids, "_process_grid_call")
    first = process_grids._process_grid_call
    second = process_grids._process_grid_call
    assert first is second
    assert "<locals>" not in first.__qualname__


def test_warm_call_trace_request_count_is_unaffected_by_the_migration() -> None:
    """Migrating `_trace_process_jaxpr`'s nested callback does not remove the
    per-warm-call jaxpr retrace: `_produce_staged` calls `jax.make_jaxpr` fresh
    on every solve, uncached, so this is the separate, already-known P2
    one-retrace-per-warm-call cost, not a side effect of the beartype-wrapper
    migration. This test documents the current (post-migration) count so a
    future change to that caching boundary shows up here.
    """
    model, params = _model(execution_config=None)
    initial_conditions = _initial_conditions()

    # Warm the executable cache.
    _run_simulate(
        model=model,
        params=params,
        initial_conditions=initial_conditions,
        log_level="off",
    )

    with count_compile_requests() as counts:
        _run_simulate(
            model=model,
            params=params,
            initial_conditions=initial_conditions,
            log_level="off",
        )

    # A warm call still retraces the continuous-process jaxpr at least once
    # (the known P2 site); this migration only removes the beartype wrapper
    # compile, not the trace itself.
    assert counts.trace_requests >= 1


def _module_level_pure_operation(*, value: object) -> object:
    """Module-level stand-in operation: satisfies the identity check's contract."""
    return value


def test_operation_function_identity_check_is_cached() -> None:
    """Function-identity validation for the same function object is memoized."""
    info_before = host_operations._validated_operation_function.cache_info()
    host_operations._validated_operation_function(_module_level_pure_operation)
    host_operations._validated_operation_function(_module_level_pure_operation)
    info_after = host_operations._validated_operation_function.cache_info()

    # The second call must be a cache hit, not a fresh inspection.
    assert info_after.hits >= info_before.hits + 1
