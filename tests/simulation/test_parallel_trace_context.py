"""Parallel chunk planning honours the caller's scoped JAX trace settings.

JAX keeps settings such as dtype promotion, matmul precision and x64 in
thread-local state. A simulation that is valid under the caller's scoped settings
must simulate the same panel whether chunk planning compiles serially or on the
compilation worker pool, and no worker may trace under different settings.
"""

import contextvars
import threading
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from fractions import Fraction
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.simulation.chunk_profiles import _compile_forward_units_in_parallel
from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.execution import ExecutionConfig
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_PARAMS = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
_THREAD_TIMEOUT_SECONDS = 20


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


def _utility(*, wealth: ContinuousState, saving: ContinuousAction) -> FloatND:
    # A strongly typed int32 operand traces only under standard promotion.
    return wealth + saving + jnp.asarray(1, dtype=jnp.int32)


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _next_wealth(*, wealth: ContinuousState, saving: ContinuousAction) -> FloatND:
    return wealth + saving


def _next_regime() -> ScalarInt:
    return _RegimeId.done


def _only_initial_age(age: float) -> bool:
    return age == 0


def _model() -> Model:
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
        execution_config=ExecutionConfig(device_memory_bytes=2**32),
    )


def _initial_wealths(*, population: int, reverse: bool) -> list[float]:
    wealths = [float(1 + subject % 3) for subject in range(population)]
    return wealths[::-1] if reverse else wealths


def _simulate_panels(
    *, workers: int, population: int, seed: int, reverse: bool, repeats: int
) -> list[pd.DataFrame]:
    """Solve serially, then simulate `repeats` times on one fresh model."""
    model = _model()
    wealths = _initial_wealths(population=population, reverse=reverse)
    initial = {
        "wealth": jnp.asarray(wealths),
        "age": jnp.zeros(population),
        "regime_id": jnp.full(population, _RegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params=_PARAMS, log_level="off", max_compilation_workers=1)
    return [
        model.simulate(
            params=_PARAMS,
            initial_conditions=initial,
            solution=solution,
            log_level="off",
            max_compilation_workers=workers,
            seed=seed,
        ).to_dataframe(use_labels=False)
        for _ in range(repeats)
    ]


def _simulate_under_scoped_standard_promotion(
    *, workers: int, population: int, seed: int, reverse: bool, repeats: int
) -> list[pd.DataFrame]:
    """Simulate with strict promotion process-wide and standard promotion scoped."""
    previous = jax.config.jax_numpy_dtype_promotion
    try:
        jax.config.update("jax_numpy_dtype_promotion", "strict")
        with jax.numpy_dtype_promotion("standard"):
            return _simulate_panels(
                workers=workers,
                population=population,
                seed=seed,
                reverse=reverse,
                repeats=repeats,
            )
    finally:
        jax.config.update("jax_numpy_dtype_promotion", previous)


def _reference_rows(wealths: Sequence[float]) -> list[dict[str, object]]:
    """Return the panel rows of the two-period economy in exact arithmetic.

    With discount factor zero, saving in {1, 2}, utility `w + a + 1`, next wealth
    `w + a` and terminal utility `w`, saving 2 is the unique optimum for every
    initial wealth in [1, 3]; next wealth then stays on the model's grid.
    """
    rows: list[dict[str, object]] = []
    for subject, value in enumerate(wealths):
        wealth = Fraction(value)
        utility, saving = max((wealth + action + 1, action) for action in (1, 2))
        rows.append(
            {
                "subject_id": subject,
                "period": 0,
                "regime_name": "alive",
                "wealth": float(wealth),
                "saving": float(saving),
                "value": float(utility),
            }
        )
        rows.append(
            {
                "subject_id": subject,
                "period": 1,
                "regime_name": "done",
                "wealth": float(wealth + saving),
                "value": float(wealth + saving),
            }
        )
    return rows


_CASES = [
    pytest.param(2, 3, 0, False, id="two_workers"),
    pytest.param(4, 7, 1, True, id="four_workers_reversed"),
]


@pytest.mark.parametrize(("workers", "population", "seed", "reverse"), _CASES)
def test_parallel_planning_under_scoped_promotion_simulates_the_serial_panel(
    *, workers: int, population: int, seed: int, reverse: bool
) -> None:
    """Cold and warm parallel-planned panels equal the serial panel exactly."""
    (serial,) = _simulate_under_scoped_standard_promotion(
        workers=1, population=population, seed=seed, reverse=reverse, repeats=1
    )
    parallel = _simulate_under_scoped_standard_promotion(
        workers=workers, population=population, seed=seed, reverse=reverse, repeats=2
    )
    pd.testing.assert_frame_equal(
        pd.concat(parallel, ignore_index=True),
        pd.concat([serial, serial], ignore_index=True),
        check_exact=True,
    )


@pytest.mark.parametrize(
    ("workers", "population", "seed", "reverse"),
    [pytest.param(1, 1, 0, False, id="serial"), *_CASES],
)
def test_parallel_planning_under_scoped_promotion_matches_the_exact_reference(
    *, workers: int, population: int, seed: int, reverse: bool
) -> None:
    """Every simulated row equals the exact-arithmetic optimum of its subject."""
    (panel,) = _simulate_under_scoped_standard_promotion(
        workers=workers, population=population, seed=seed, reverse=reverse, repeats=1
    )
    actual = panel.sort_values(["subject_id", "period"]).to_dict("records")
    expected = _reference_rows(_initial_wealths(population=population, reverse=reverse))
    mismatches = [
        (row, target)
        for row, target in zip(actual, expected, strict=True)
        if any(row[name] != value for name, value in target.items())
    ]
    assert (len(actual), mismatches) == (len(expected), [])


def _record_context(
    *,
    observed: list[tuple[int, str, object, object, object]],
    token: contextvars.ContextVar[str],
) -> None:
    observed.append(
        (
            threading.get_ident(),
            token.get(),
            jax.config.jax_numpy_dtype_promotion,
            jax.config.jax_default_matmul_precision,
            jax.config.jax_enable_x64,
        )
    )


@pytest.mark.parametrize("setting", ["promotion", "precision", "x64", "combined"])
def test_no_forward_unit_runs_under_a_different_trace_context(setting: str) -> None:
    """Every unit that runs sees the caller's contextvars and scoped JAX settings."""
    token = contextvars.ContextVar("caller_token", default="outside")
    reset = token.set("caller")
    observed: list[tuple[int, str, object, object, object]] = []
    old_promotion = jax.config.jax_numpy_dtype_promotion
    old_precision = jax.config.jax_default_matmul_precision
    old_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_numpy_dtype_promotion", "standard")
        jax.config.update("jax_default_matmul_precision", "default")
        with ExitStack() as stack:
            if setting in ("promotion", "combined"):
                stack.enter_context(jax.numpy_dtype_promotion("strict"))
            if setting in ("precision", "combined"):
                stack.enter_context(jax.default_matmul_precision("highest"))
            if setting in ("x64", "combined"):
                stack.enter_context(jax.enable_x64(not old_x64))
            expected = (
                jax.config.jax_numpy_dtype_promotion,
                jax.config.jax_default_matmul_precision,
                jax.config.jax_enable_x64,
            )
            unit = partial(_record_context, observed=observed, token=token)
            _compile_forward_units_in_parallel(n_workers=2, units=(unit, unit))
    finally:
        jax.config.update("jax_numpy_dtype_promotion", old_promotion)
        jax.config.update("jax_default_matmul_precision", old_precision)
        token.reset(reset)
    assert [row[1:] for row in observed if row[1:] != ("caller", *expected)] == []


@pytest.mark.parametrize(("workers", "count"), [(1, 2), (2, 0), (2, 1)])
def test_serial_and_single_unit_planning_runs_no_unit_on_the_pool(
    *, workers: int, count: int
) -> None:
    """One worker, or at most one unit, leaves every unit to the ordered walk."""
    observed: list[int] = []
    _compile_forward_units_in_parallel(
        n_workers=workers,
        units=tuple(partial(observed.append, 1) for _ in range(count)),
    )
    assert observed == []


def _run_on_fresh_thread(func: Callable[[], None]) -> None:
    """Run `func` on a new thread, whose JAX settings are the process-wide ones."""
    failures: list[BaseException] = []

    def target() -> None:
        try:
            func()
        except BaseException as error:  # noqa: BLE001 - re-raised on the caller
            failures.append(error)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=_THREAD_TIMEOUT_SECONDS)
    assert not thread.is_alive(), "the pool did not finish its units"
    if failures:
        raise failures[0]


def test_matching_trace_context_runs_every_unit_off_the_calling_thread() -> None:
    """Under process-wide settings, both units run on workers with caller context."""
    observed: list[tuple[int, str, object, object, object]] = []
    owners: list[int] = []

    def call() -> None:
        owners.append(threading.get_ident())
        token = contextvars.ContextVar("caller_token", default="missing")
        token.set("visible")
        unit = partial(_record_context, observed=observed, token=token)
        _compile_forward_units_in_parallel(n_workers=2, units=(unit, unit))

    _run_on_fresh_thread(call)
    assert sorted((row[0] != owners[0], row[1]) for row in observed) == [
        (True, "visible"),
        (True, "visible"),
    ]


def _raise_unit_error() -> None:
    raise ValueError("unit failed under the caller's settings")


def test_matching_trace_context_propagates_unit_errors() -> None:
    """A unit that fails on a worker raises its own error on the caller."""

    def call() -> None:
        _compile_forward_units_in_parallel(
            n_workers=2, units=(_raise_unit_error, _raise_unit_error)
        )

    with pytest.raises(ValueError, match="failed under the caller's settings"):
        _run_on_fresh_thread(call)
