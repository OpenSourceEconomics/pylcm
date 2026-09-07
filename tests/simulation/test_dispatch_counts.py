"""A warm simulate call compiles nothing, and runtime validation stays cheap.

The compile-request pins say that repeating a simulate call at a subject width
the process has already seen asks JAX for no further compilation, at any log
level. The host-time rows say what runtime validation costs once nothing
compiles any more: the `progress` path must stay within half again the host
time of the validation-free `off` path.
"""

import contextlib
import logging
import statistics
import time
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest

from _lcm.utils.logging import LogLevel
from benchmarks.asv._dispatch_counters import count_compile_requests
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
def test_a_fresh_subject_width_at_debug_compiles_on_its_first_call(
    *, witness: str
) -> None:
    """The first `debug` call at a subject width never seen before compiles."""
    model, params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=params, log_level="off")
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


@pytest.mark.parametrize(
    "witness",
    [
        "dissolution",
        pytest.param(
            "multi_regime",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "The forward-simulation loop is at parity between the two levels; "
                    "what remains above `off` is the pre-flight transition and "
                    "initial-condition validation `Model.simulate` runs once before "
                    "the loop, which this witness is small enough for to dominate."
                ),
            ),
        ),
    ],
)
def test_simulate_host_time_at_progress_is_within_the_bar_of_off(
    *, witness: str
) -> None:
    """Runtime validation at `progress` costs at most half again the `off` host time."""
    ratio = _median_host_time_ratio(
        witness=witness, log_level="progress", repeats=HOST_TIME_REPEATS
    )
    assert ratio <= HOST_TIME_BAR, f"progress/off host time is {ratio:.3f}x"


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


def _median_host_time_ratio(
    *, witness: str, log_level: LogLevel, repeats: int
) -> float:
    """Return median(host time at `log_level`) / median(host time at `off`).

    Both levels are warmed at this witness's subject width before any call is
    timed, so no compilation enters either median.
    """
    model, params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=params, log_level="off")
    with _lcm_log_output_held_fixed():
        timings: dict[LogLevel, list[float]] = {"off": [], log_level: []}
        for warm_level in ("off", log_level):
            _host_time(
                model=model,
                params=params,
                initial_conditions=initial_conditions,
                solution=solution,
                log_level=warm_level,
            )
        for _ in range(repeats):
            for measured_level in ("off", log_level):
                timings[measured_level].append(
                    _host_time(
                        model=model,
                        params=params,
                        initial_conditions=initial_conditions,
                        solution=solution,
                        log_level=measured_level,
                    )
                )
    return statistics.median(timings[log_level]) / statistics.median(timings["off"])
