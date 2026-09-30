"""Chunk planning traces under the caller's scoped JAX trace settings.

JAX keeps settings such as dtype promotion, matmul precision and x64 in
thread-local state. Chunk planning lowers every forward program on the calling
thread and hands only compilation to the worker pool, so a simulation that is
valid under the caller's scoped settings simulates the same panel for every
worker count, and every program is traced under the caller's settings.
"""

import contextvars
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from fractions import Fraction

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from lcm import AgeGrid, Choose, LinSpacedGrid, Model, Regime, categorical
from lcm.execution import ExecutionConfig
from lcm.solver_api import SolutionResult
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_PARAMS = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}

_CALLER_TOKEN = contextvars.ContextVar("caller_token", default="outside")


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


def _model() -> Model:
    return Model(
        regimes={
            "alive": Regime(
                regime_transitions=Choose(func=_next_regime, targets=("done",)),
                functions={"utility": _utility},
                actions={"saving": LinSpacedGrid(start=1, stop=2, n_points=2)},
            ),
            "done": Regime(
                regime_transitions=None, functions={"utility": _terminal_utility}
            ),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=5, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=2**32),
        initial_regimes={0: "alive"},
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


@contextmanager
def _scoped_settings(*, setting: str) -> Iterator[tuple[object, object, object]]:
    """Scope JAX trace settings that differ from the process-wide ones.

    Yields the effective promotion, matmul precision and x64 inside the scope.
    Process-wide promotion is strict whenever standard promotion is scoped, so the
    model's utility traces only under the scoped setting.
    """
    old_promotion = jax.config.jax_numpy_dtype_promotion
    old_precision = jax.config.jax_default_matmul_precision
    old_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_default_matmul_precision", "default")
        with ExitStack() as stack:
            if setting in ("promotion", "combined"):
                jax.config.update("jax_numpy_dtype_promotion", "strict")
                stack.enter_context(jax.numpy_dtype_promotion("standard"))
            if setting in ("precision", "combined"):
                stack.enter_context(jax.default_matmul_precision("highest"))
            if setting in ("x64", "combined"):
                stack.enter_context(jax.enable_x64(not old_x64))
            yield (
                jax.config.jax_numpy_dtype_promotion,
                jax.config.jax_default_matmul_precision,
                jax.config.jax_enable_x64,
            )
    finally:
        jax.config.update("jax_numpy_dtype_promotion", old_promotion)
        jax.config.update("jax_default_matmul_precision", old_precision)


def _simulate_recording_traces(
    *, monkeypatch: pytest.MonkeyPatch, workers: int
) -> list[tuple[int, str, object, object, object]]:
    """Solve serially, then simulate with `workers` compile threads.

    Returns, for every program lowered by the simulation, the lowering thread,
    the caller's context token and the effective JAX trace settings there. JAX
    traces a program on the thread that lowers it.
    """
    model = _model()
    solution = model.solve(params=_PARAMS, log_level="off", max_compilation_workers=1)
    traces: list[tuple[int, str, object, object, object]] = []
    lower_body = jax.stages.Traced.lower

    def record_and_lower(
        self: jax.stages.Traced, *args: object, **kwargs: object
    ) -> object:
        traces.append(
            (
                threading.get_ident(),
                _CALLER_TOKEN.get(),
                jax.config.jax_numpy_dtype_promotion,
                jax.config.jax_default_matmul_precision,
                jax.config.jax_enable_x64,
            )
        )
        return lower_body(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    with monkeypatch.context() as patch:
        patch.setattr(jax.stages.Traced, "lower", record_and_lower)
        model.simulate(
            params=_PARAMS,
            initial_conditions={
                "wealth": jnp.asarray([1.0, 2.0, 3.0]),
                "age": jnp.zeros(3),
                "regime_id": jnp.full(3, _RegimeId.alive, dtype=jnp.int32),
            },
            solution=solution,
            log_level="off",
            max_compilation_workers=workers,
            seed=0,
        )
    return traces


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize(
    "setting", ["process_wide", "promotion", "precision", "x64", "combined"]
)
def test_every_simulated_program_traces_on_the_caller_under_its_settings(
    *, monkeypatch: pytest.MonkeyPatch, setting: str, workers: int
) -> None:
    """Each simulated program is traced on the calling thread, under the caller's
    contextvars and the caller's scoped JAX trace settings."""
    reset = _CALLER_TOKEN.set("caller")
    try:
        with _scoped_settings(setting=setting) as expected:
            traces = _simulate_recording_traces(
                monkeypatch=monkeypatch, workers=workers
            )
    finally:
        _CALLER_TOKEN.reset(reset)
    caller = (threading.get_ident(), "caller", *expected)
    assert (len(traces) > 0, [row for row in traces if row != caller]) == (True, [])


class _CompileFailedError(Exception):
    """A compile on the worker pool failed."""


def _raise_on_pool_threads(caller: int) -> object:
    """Return a `Lowered.compile` that fails on every thread but `caller`."""
    compile_body = jax.stages.Lowered.compile

    def compile_or_fail(
        self: jax.stages.Lowered, *args: object, **kwargs: object
    ) -> object:
        if threading.get_ident() != caller:
            raise _CompileFailedError("the pool refused to compile")
        return compile_body(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    return compile_or_fail


_PROGRAM = r"(alive|done) (decision|transition|route|action decoder) \(period \d+\)"


def _program_notes(error: BaseException) -> list[str]:
    """Return the notes naming a program, without JAX's traceback notes."""
    return [
        note for note in getattr(error, "__notes__", []) if note.startswith("while ")
    ]


def _without_widths(note: str) -> str:
    """Drop the axis widths from a note naming a program."""
    return re.sub(r", widths=\{[^}]*\}", "", note)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_lowering_error_is_raised_on_the_caller_naming_the_program(
    workers: int,
) -> None:
    """A forward program that fails to trace raises its own error, noted with the
    program being lowered.

    The alive utility adds a strongly typed int32 operand, so it traces under
    standard promotion only: the solve succeeds under it and simulation, under
    strict promotion, fails to trace the alive decision.
    """
    model = _model()
    solution = model.solve(params=_PARAMS, log_level="off", max_compilation_workers=1)
    previous = jax.config.jax_numpy_dtype_promotion
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    try:
        with pytest.raises(jax.dtypes.TypePromotionError) as raised:
            _simulate_two_subjects(model=model, solution=solution, workers=workers)
    finally:
        jax.config.update("jax_numpy_dtype_promotion", previous)
    notes = _program_notes(raised.value)
    assert list(map(_without_widths, notes)) == [
        "while lowering alive decision (period 0)"
    ]


def _simulate_two_subjects(
    *, model: Model, solution: SolutionResult, workers: int
) -> None:
    """Simulate two alive subjects with `workers` compile threads."""
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([1.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, _RegimeId.alive, dtype=jnp.int32),
        },
        solution=solution,
        log_level="off",
        max_compilation_workers=workers,
        seed=0,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_a_compile_error_is_raised_on_the_caller_naming_the_program(
    *, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    """A forward program that fails to compile on the pool raises its own error,
    noted with the program being compiled."""
    model = _model()
    solution = model.solve(params=_PARAMS, log_level="off", max_compilation_workers=1)
    monkeypatch.setattr(
        jax.stages.Lowered, "compile", _raise_on_pool_threads(threading.get_ident())
    )
    with pytest.raises(_CompileFailedError) as raised:
        _simulate_two_subjects(model=model, solution=solution, workers=workers)
    notes = _program_notes(raised.value)
    assert [
        re.fullmatch(rf"while compiling {_PROGRAM}", _without_widths(note)) is not None
        for note in notes
    ] == [True]
