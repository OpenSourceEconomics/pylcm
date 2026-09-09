"""A warm simulate call compiles nothing, and the simulation loop validates cheaply.

The compile-request pins say that repeating a simulate call at a subject width
the process has already seen asks JAX for no further compilation, at any log
level. The host-time rows say what runtime validation costs once nothing
compiles any more: the `progress` path must stay within half again the host
time of the validation-free `off` path.

The bar applies to both the forward-simulation loop and the whole
`Model.simulate` call, including its `validate_simulation_inputs` preflight.
"""

import contextlib
import logging
import statistics
import time
from collections.abc import Callable, Iterator

import jax
import jax._src.monitoring
import jax.numpy as jnp
import pytest

import lcm.model
from _lcm.utils.logging import LogLevel
from benchmarks.asv._compile_counters import COMPILE_EVENT, count_compile_requests
from benchmarks.asv._simulation_witnesses import (
    MULTI_INITIAL_CONDITIONS,
    WITNESSES,
)
from lcm import Model
from lcm.typing import FloatND, UserInitialConditions, UserParams
from tests.test_models.processes import (
    MultiRegimeId,
    get_multi_regime_model,
    get_multi_regime_params,
)

LOG_LEVELS: tuple[LogLevel, ...] = ("off", "warning", "progress", "debug")

# Host time of `progress` may exceed the validation-free `off` path by at most
# this factor.
HOST_TIME_BAR = 1.5

# Warm calls timed per log level before the median is taken. The two levels are
# timed alternately, so a background load that varies slowly over the run moves
# both medians together rather than the ratio.
HOST_TIME_REPEATS = 9

# The coordinator `Model.simulate` runs before the period loop. The recording
# stub must observe it, or the measurement no longer isolates the loop.
_PREFLIGHT_VALIDATORS = frozenset({"validate_simulation_inputs"})


def _double(x: FloatND) -> FloatND:
    """Return twice the input."""
    return 2.0 * x


def _warm_then_count(*, witness: str, log_level: LogLevel) -> tuple[int, int, int]:
    """Return the trace, lowering, and compile requests of a second simulate call."""
    model, params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=params, log_level="off")
    model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level=log_level,
        seed=0,
    )
    with count_compile_requests() as counts:
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level=log_level,
            seed=0,
        )
    return counts.trace_requests, counts.lowering_requests, counts.compile_requests


@pytest.mark.parametrize("witness", sorted(WITNESSES))
@pytest.mark.parametrize("log_level", LOG_LEVELS)
def test_second_simulate_call_requests_no_compilation(
    *, witness: str, log_level: LogLevel
) -> None:
    """After one warm call, a simulate call at any log level compiles nothing."""
    assert _warm_then_count(witness=witness, log_level=log_level) == (0, 0, 0)


def test_counters_report_one_of_each_for_a_freshly_compiled_function() -> None:
    """Compiling a function inside the block is reported once per stage."""
    argument = jnp.arange(3.0)
    with count_compile_requests() as counts:
        jax.jit(_double)(argument)
    assert (
        counts.trace_requests,
        counts.lowering_requests,
        counts.compile_requests,
    ) == (1, 1, 1)


class _BlockError(Exception):
    """Raised inside a counting block to see what its teardown does with it."""


def test_a_counting_block_leaves_no_listener_behind() -> None:
    """A finished counting block puts JAX's listener list back as it found it."""
    before = len(jax._src.monitoring.get_event_duration_listeners())
    with count_compile_requests():
        pass
    assert len(jax._src.monitoring.get_event_duration_listeners()) == before


def test_a_cleared_listener_list_does_not_mask_the_blocks_own_error() -> None:
    """A block whose listener is gone still reports the error the block raised.

    `jax.monitoring.clear_event_listeners()` drops every registered listener,
    so the teardown has nothing to unregister. Unregistering a listener that is
    already gone raises inside JAX, and the teardown runs in a `finally`, so a
    teardown that let that through would replace the block's own exception.
    """
    with pytest.raises(_BlockError):
        _raise_from_a_block_whose_listeners_were_cleared()


def _raise_from_a_block_whose_listeners_were_cleared() -> None:
    """Clear JAX's listener list inside a counting block, then raise."""
    with count_compile_requests():
        jax.monitoring.clear_event_listeners()
        raise _BlockError


def test_a_counting_block_stops_counting_once_its_listener_cannot_be_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listener the teardown could not remove contributes to no later block."""
    before = tuple(jax._src.monitoring.get_event_duration_listeners())
    with monkeypatch.context() as teardown:
        teardown.setattr(
            jax.monitoring,
            "unregister_event_duration_listener",
            _refuse_listener_removal,
        )
        with count_compile_requests() as stranded:
            pass
    retained = tuple(
        listener
        for listener in jax._src.monitoring.get_event_duration_listeners()
        if listener not in before
    )
    try:
        with count_compile_requests() as active:
            jax.monitoring.record_event_duration_secs(COMPILE_EVENT, 0.001)
        assert (len(retained), stranded.compile_requests, active.compile_requests) == (
            1,
            0,
            1,
        )
    finally:
        for listener in retained:
            jax.monitoring.unregister_event_duration_listener(listener)


def _refuse_listener_removal(_listener: Callable[..., None]) -> None:
    """Report that unregistering failed while keeping the listener registered."""
    raise AssertionError("listener remains registered")


def test_cold_simulate_call_compiles() -> None:
    """A simulate call on a freshly built model issues backend compilations."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    model = Model(
        regimes=dict(base.user_regimes),
        regime_id_class=MultiRegimeId,
        ages=base.ages,
        fixed_params=dict(base.fixed_params),
    )
    params = get_multi_regime_params("normal")
    solution = model.solve(params=params, log_level="off")
    with count_compile_requests() as counts:
        model.simulate(
            params=params,
            initial_conditions=MULTI_INITIAL_CONDITIONS,
            solution=solution,
            log_level="off",
            seed=0,
        )
    assert counts.compile_requests > 0


@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_a_subject_width_never_simulated_before_compiles_on_its_first_call(
    *, witness: str
) -> None:
    """A model asked for a subject width it has not seen before compiles for it."""
    model, params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=params, log_level="off")
    model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="debug",
        seed=0,
    )
    doubled = {name: jnp.tile(value, 2) for name, value in initial_conditions.items()}
    with count_compile_requests() as counts:
        model.simulate(
            params=params,
            initial_conditions=doubled,
            solution=solution,
            log_level="debug",
            seed=0,
        )
    assert counts.compile_requests > 0


@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_repeating_a_subject_width_at_debug_compiles_nothing(*, witness: str) -> None:
    """A subject width simulated twice at `debug` compiles only on its first call."""
    model, params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=params, log_level="off")
    doubled = {name: jnp.tile(value, 2) for name, value in initial_conditions.items()}
    model.simulate(
        params=params,
        initial_conditions=doubled,
        solution=solution,
        log_level="debug",
        seed=0,
    )
    with count_compile_requests() as counts:
        model.simulate(
            params=params,
            initial_conditions=doubled,
            solution=solution,
            log_level="debug",
            seed=0,
        )
    assert counts.compile_requests == 0


@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_simulation_loop_host_time_at_progress_is_within_the_bar_of_off(
    *,
    witness: str,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    """The simulation loop at `progress` costs at most half again its `off` host time.

    Measured with the pre-flight validators stubbed, so the ratio belongs to the
    period loop alone. The estimator is the median of `HOST_TIME_REPEATS` = 9
    warm calls per log level, the two levels timed alternately inside one
    process, both warmed at the timed subject width so no compilation enters
    either median, and every `lcm` record sent to a null handler so that log
    emission is held fixed while the level varies. The witnesses are small — 3
    subjects for `dissolution`, 7 for `multi_regime` — which is why pre-flight
    validation dominates the whole call at these widths but not the loop.

    Both legs still carry the level-independent part of `Model.simulate`
    (padding, batch-size resolution, dispatch), which pulls the ratio toward
    1.0, so the row is a lower bound on any growth in the loop's validation
    cost rather than a measurement of it in isolation. The two medians are
    recorded alongside the ratio so a reader can see how much room there is.
    """
    off_seconds, progress_seconds = _median_host_times(
        witness=witness,
        log_level="progress",
        repeats=HOST_TIME_REPEATS,
        stub_preflight=True,
    )
    ratio = progress_seconds / off_seconds
    record_testsuite_property(f"loop_off_ms[{witness}]", round(off_seconds * 1e3, 4))
    record_testsuite_property(
        f"loop_progress_ms[{witness}]", round(progress_seconds * 1e3, 4)
    )
    record_testsuite_property(f"loop_progress_over_off[{witness}]", round(ratio, 4))
    assert ratio <= HOST_TIME_BAR, (
        f"progress/off host time is {ratio:.3f}x "
        f"(off={off_seconds:.6f}s, progress={progress_seconds:.6f}s)"
    )


@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_simulate_host_time_at_progress_is_within_the_bar_of_off(
    *,
    witness: str,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    """A whole simulate call at `progress` costs at most half again its `off` host time.

    Same estimator as the loop row above, with nothing stubbed, so the ratio
    covers the complete `validate_simulation_inputs` preflight and period loop.
    """
    off_seconds, progress_seconds = _median_host_times(
        witness=witness,
        log_level="progress",
        repeats=HOST_TIME_REPEATS,
        stub_preflight=False,
    )
    ratio = progress_seconds / off_seconds
    record_testsuite_property(f"call_progress_over_off[{witness}]", round(ratio, 4))
    assert ratio <= HOST_TIME_BAR, (
        f"progress/off host time is {ratio:.3f}x "
        f"(off={off_seconds:.6f}s, progress={progress_seconds:.6f}s)"
    )


@contextlib.contextmanager
def _preflight_validation_stubbed() -> Iterator[list[str]]:
    """Hold out the complete preflight coordinator before the period loop.

    Its logger policy is the variable being timed, so a recording no-op at the
    actual Model-bound coordinator isolates loop cost without changing the
    separate, unstubbed whole-call timing rows.

    Yields the list of validator names the stub actually absorbed, so a caller
    can tell a seam that held from one that silently stopped holding --- which
    is what would happen if `lcm.model` ever reached a validator through a
    qualified path instead of the bare name.
    """
    absorbed: list[str] = []

    def stub(**kwargs: object) -> None:  # noqa: ARG001
        """Record entry into the full preflight, absorbing its arguments."""
        absorbed.append("validate_simulation_inputs")

    # Bound by name: a stub takes any keyword arguments, which is wider than
    # the coordinator declares, so a direct assignment would be off-signature.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lcm.model, "validate_simulation_inputs", stub)
        yield absorbed


@contextlib.contextmanager
def _lcm_log_output_held_fixed() -> Iterator[None]:
    """Send every `lcm` record to a null handler for the duration of the block.

    A host-time ratio across two log levels is a statement about runtime
    validation, so the one other thing the level changes — how many records
    reach a handler — is held fixed rather than measured. The handler list is
    replaced (not merely extended) because `get_logger` installs a stdout
    handler only on a logger that has none, and propagation is turned off so no
    ancestor handler, `caplog`'s included, formats the records either.
    """
    logger = logging.getLogger("lcm")
    saved_handlers = logger.handlers[:]
    saved_propagate = logger.propagate
    saved_level = logger.level
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers = saved_handlers
        logger.propagate = saved_propagate
        logger.setLevel(saved_level)


def _host_time(
    *,
    model: Model,
    params: UserParams,
    initial_conditions: UserInitialConditions,
    solution: object,
    log_level: LogLevel,
) -> float:
    """Return the wall time of one simulate call, device work included."""
    start = time.perf_counter()
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,  # ty: ignore[invalid-argument-type]
        log_level=log_level,
        seed=0,
    )
    jax.block_until_ready(result.raw_results)
    return time.perf_counter() - start


def _median_host_times(
    *, witness: str, log_level: LogLevel, repeats: int, stub_preflight: bool
) -> tuple[float, float]:
    """Return the median host time at `off` and at `log_level`, in seconds.

    Both levels are warmed at this witness's subject width before any call is
    timed, and the timed calls alternate between the levels.

    With `stub_preflight`, the seam is checked rather than trusted: if neither
    validator was absorbed by the stub the measurement did not hold anything
    out, and this raises instead of returning a ratio that quietly covers the
    whole call.
    """
    model, params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=params, log_level="off")
    levels: tuple[LogLevel, ...] = ("off", log_level)
    timings: dict[LogLevel, list[float]] = {level: [] for level in levels}
    absorbed: list[str] = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(_lcm_log_output_held_fixed())
        if stub_preflight:
            absorbed = stack.enter_context(_preflight_validation_stubbed())
        for warm_level in levels:
            _host_time(
                model=model,
                params=params,
                initial_conditions=initial_conditions,
                solution=solution,
                log_level=warm_level,
            )
        for _ in range(repeats):
            for measured_level in levels:
                timings[measured_level].append(
                    _host_time(
                        model=model,
                        params=params,
                        initial_conditions=initial_conditions,
                        solution=solution,
                        log_level=measured_level,
                    )
                )
        missing = _PREFLIGHT_VALIDATORS - set(absorbed)
        if stub_preflight and missing:
            msg = (
                f"the pre-flight seam held nothing out for {sorted(missing)}; "
                "`Model.simulate` no longer reaches those validators by the "
                "bare name `lcm.model` binds"
            )
            raise RuntimeError(msg)
    return statistics.median(timings["off"]), statistics.median(timings[log_level])
