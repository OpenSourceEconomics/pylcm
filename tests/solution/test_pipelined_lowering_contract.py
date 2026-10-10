"""Contract of one compilation wave: exact outputs, keyed identity, error paths.

The programs are integer-valued cumulative sums, so every intermediate is exactly
representable in fp32 and fp64 and the literal Python oracle below is bit-exact
regardless of reduction grouping. Instrumentation only controls thread
interleavings.
"""

import logging
import threading
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypedDict, Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
)
from _lcm.execution.output_layout import (
    VALUE,
    ResolvedOutputLayout,
    resolve_output_layout,
)
from _lcm.solution import backward_induction
from _lcm.time import TimeAxis
from _lcm.typing import HostArray, PytreeValue, ShapeDtypePytree
from lcm import AgeGrid
from lcm.typing import ReferenceName
from tests.solution.test_pipelined_lowering import (
    _CompileKwargs,
    _RolesKwargs,
)


class _WaveKwargs(TypedDict):
    new_lowerings: Mapping[Hashable, backward_induction._CoreCandidate]
    resolved_programs: Mapping[backward_induction._CoreCandidate, ResolvedCoreProgram]
    all_layouts: Mapping[backward_induction._CoreTriple, ResolvedOutputLayout]
    internal_templates: Mapping[
        backward_induction._CoreCandidate, Mapping[ReferenceName, ShapeDtypePytree]
    ]
    donations: Mapping[
        backward_induction._CoreCandidate,
        tuple[backward_induction.ResolvedDonation, ...],
    ]
    ages: TimeAxis
    n_triples_per_lowering: Mapping[Hashable, int]
    log_kernel_memory: bool
    n_workers: int
    logger: logging.Logger
    compiled: dict[Hashable, jax.stages.Compiled]
    labels: dict[Hashable, str]


type _ArrayBits = tuple[tuple[int, ...], np.dtype, bytes]


class _ReversedObservations(TypedDict):
    finished: list[Hashable]
    identity: bool
    threads: tuple[list[str], list[Hashable], list[bool]]
    values: list[_ArrayBits]
    expected_values: list[_ArrayBits]


class _MemoryKwargs(TypedDict):
    compiled: jax.stages.Compiled
    label: str
    logger: logging.Logger


_KEYS = ("first", "second", "third")
_WAIT_SECONDS = 10.0  # Bounded synchronisation timeout, not a performance bound.


def _program(*, wealth: jax.Array, scale: int) -> jax.Array:
    """Scaled cumulative sum; integer inputs keep it exact in any float dtype."""
    return jnp.cumsum(wealth * scale)


def _wave_kwargs(
    *, n: int, n_workers: int, keys: tuple[str, ...] = _KEYS
) -> tuple[_WaveKwargs, jax.Array]:
    """Arguments for one wave of programs, one per key, over `n` integer states."""
    wealth = jnp.asarray(np.arange(n) - n // 2, dtype=jnp.asarray(0.0).dtype)
    candidates: dict[Hashable, backward_induction._CoreCandidate] = {
        key: ((key, 0, key), ()) for key in keys
    }
    resolved = {
        candidate: ResolvedCoreProgram(
            name=str(key),
            function=_program,
            arguments=MappingProxyType({"wealth": wealth}),
            static_kwargs=MappingProxyType({"scale": i + 1}),
            requirements=CoreExecutionRequirements(),
            output_roles=VALUE,
            disposition=CoreExecutionDisposition.PLANNED,
            donation_candidates=(),
            tile_widths=MappingProxyType({}),
            specialization_key=(),
            input_transfer_plan=(),
        )
        for i, (key, candidate) in enumerate(candidates.items())
    }
    kwargs: _WaveKwargs = {
        "new_lowerings": candidates,
        "resolved_programs": resolved,
        "all_layouts": {
            triple: resolve_output_layout(
                core_key=triple[2],
                value_template=wealth,
                state_order=("wealth",),
                output_roles=VALUE,
            )
            for triple, _ in candidates.values()
        },
        "internal_templates": {candidate: {} for candidate in candidates.values()},
        "donations": dict.fromkeys(candidates.values(), ()),
        "ages": AgeGrid(start=0, inclusive_stop=1, step="Y"),
        "n_triples_per_lowering": dict.fromkeys(keys, 1),
        "log_kernel_memory": False,
        "n_workers": n_workers,
        "logger": logging.getLogger("pipelined-lowering-contract"),
        "compiled": {},
        "labels": {},
    }
    return kwargs, wealth


def _literal_cumsum(*, wealth: jax.Array, scale: int) -> HostArray:
    """Scaled cumulative sum by a plain Python loop over integers."""
    source = np.asarray(wealth)
    total = 0
    result = []
    for value in source.tolist():
        total += int(value) * scale
        result.append(total)
    return np.asarray(result, dtype=source.dtype)


def _bits(array: PytreeValue) -> _ArrayBits:
    """Shape, dtype and raw bytes of an array on the host."""
    host = np.asarray(jax.device_get(array))
    return host.shape, host.dtype, host.tobytes(order="C")


_SHAPES_AND_WORKERS = pytest.mark.parametrize(
    ("n", "n_workers"), [(1, 1), (1, 2), (17, 1), (17, 2)]
)


@_SHAPES_AND_WORKERS
@pytest.mark.parametrize("scale", [1, 2, 3])
def test_lower_and_compile_wave_executables_match_literal_oracle(
    *, n: int, n_workers: int, scale: int
) -> None:
    """Each executable reproduces the literal cumulative sum bit for bit."""
    kwargs, wealth = _wave_kwargs(n=n, n_workers=n_workers)
    backward_induction._lower_and_compile_wave(**kwargs)
    key = _KEYS[scale - 1]
    assert _bits(kwargs["compiled"][key](wealth=wealth)) == _bits(
        _literal_cumsum(wealth=wealth, scale=scale)
    )


@_SHAPES_AND_WORKERS
@pytest.mark.parametrize("scale", [1, 2, 3])
def test_lower_and_compile_wave_executables_match_direct_execution(
    *, n: int, n_workers: int, scale: int
) -> None:
    """Each executable's output equals a direct `jit.lower.compile` of the program."""
    kwargs, wealth = _wave_kwargs(n=n, n_workers=n_workers)
    backward_induction._lower_and_compile_wave(**kwargs)
    key = _KEYS[scale - 1]
    direct = (
        jax.jit(
            _program,
            static_argnames=("scale",),
            out_shardings=kwargs["all_layouts"][(key, 0, key)].out_shardings,
        )
        .lower(wealth=wealth, scale=scale)
        .compile()
    )
    assert _bits(kwargs["compiled"][key](wealth=wealth)) == _bits(direct(wealth=wealth))


@_SHAPES_AND_WORKERS
@pytest.mark.parametrize("output", ["compiled", "labels"])
def test_lower_and_compile_wave_publishes_every_key(
    *, n: int, n_workers: int, output: Literal["compiled", "labels"]
) -> None:
    """Executables and labels land under exactly the wave's lowering keys."""
    kwargs, _ = _wave_kwargs(n=n, n_workers=n_workers)
    backward_induction._lower_and_compile_wave(**kwargs)
    assert set(kwargs[output]) == set(_KEYS)


def _run_reversed_finishing(monkeypatch: pytest.MonkeyPatch) -> _ReversedObservations:
    """Run a two-worker wave in which the first compile finishes last."""
    kwargs, wealth = _wave_kwargs(n=17, n_workers=2)
    original_compile = backward_induction._compile_and_log
    original_roles = backward_induction._assert_lowered_output_roles
    caller = threading.get_ident()
    third_compiled = threading.Event()
    lock = threading.Lock()
    finished: list[Hashable] = []
    returned: dict[Hashable, jax.stages.Compiled] = {}
    lowered_off_caller: list[str] = []
    compiled_on_caller: list[Hashable] = []
    first_waited: list[bool] = []

    def observed_roles(**arguments: Unpack[_RolesKwargs]) -> None:
        if threading.get_ident() != caller:
            lowered_off_caller.append(arguments["label"])
        original_roles(**arguments)

    def reverse_first(
        **arguments: Unpack[_CompileKwargs],
    ) -> tuple[Hashable, jax.stages.Compiled]:
        key = arguments["lowering_key"]
        if threading.get_ident() == caller:
            compiled_on_caller.append(key)
        if key == "first":
            first_waited.append(third_compiled.wait(_WAIT_SECONDS))
        pair = original_compile(**arguments)
        with lock:
            finished.append(key)
            returned[key] = pair[1]
        if key == "third":
            third_compiled.set()
        return pair

    monkeypatch.setattr(backward_induction, "_compile_and_log", reverse_first)
    monkeypatch.setattr(
        backward_induction, "_assert_lowered_output_roles", observed_roles
    )
    backward_induction._lower_and_compile_wave(**kwargs)
    return {
        "finished": finished,
        "identity": all(kwargs["compiled"][key] is returned[key] for key in _KEYS),
        "threads": (lowered_off_caller, compiled_on_caller, first_waited),
        "values": [_bits(kwargs["compiled"][key](wealth=wealth)) for key in _KEYS],
        "expected_values": [
            _bits(_literal_cumsum(wealth=wealth, scale=i + 1))
            for i in range(len(_KEYS))
        ],
    }


def _reversed_order(obs: _ReversedObservations) -> bool:
    return obs["finished"] == ["second", "third", "first"]


def _reversed_identity(obs: _ReversedObservations) -> bool:
    return obs["identity"]


def _reversed_threads(obs: _ReversedObservations) -> bool:
    return obs["threads"] == ([], [], [True])


def _reversed_values(obs: _ReversedObservations) -> bool:
    return obs["values"] == obs["expected_values"]


@pytest.mark.parametrize(
    "holds",
    [_reversed_order, _reversed_identity, _reversed_threads, _reversed_values],
    ids=["finishing-order", "object-identity", "thread-placement", "values"],
)
def test_lower_and_compile_wave_keys_survive_reversed_worker_finishing(
    *, monkeypatch: pytest.MonkeyPatch, holds: Callable[[_ReversedObservations], bool]
) -> None:
    """Workers finishing in reverse order still publish each executable by key.

    - the first compile is held until the third has finished;
    - each published executable is the very object its worker returned;
    - lowering stays on the calling thread and compiles run on pool threads;
    - every executable reproduces its literal oracle.
    """
    assert holds(_run_reversed_finishing(monkeypatch))


class _RecordingExecutor(ThreadPoolExecutor):
    """Thread pool that remembers every executor and future it creates."""

    instances: ClassVar[list[_RecordingExecutor]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.futures: list[Future] = []
        _RecordingExecutor.instances.append(self)

    def submit(self, *args: Any, **kwargs: Any) -> Future:
        future = super().submit(*args, **kwargs)
        self.futures.append(future)
        return future


def _run_lowering_error_while_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, bool]:
    """Fail the third of four lowerings while the first compile is still running.

    One compile worker: the first compile blocks until released, so the second
    compile is queued but not started when the third lowering fails.
    """
    keys = ("first", "second", "third", "fourth")
    kwargs, _ = _wave_kwargs(n=1, n_workers=1, keys=keys)
    original_compile = backward_induction._compile_and_log
    original_roles = backward_induction._assert_lowered_output_roles
    release_first = threading.Event()
    first_finished = threading.Event()
    started: list[Hashable] = []
    lowered: list[str] = []
    lower_error = RuntimeError("distinct third-lowering error")

    def held_compile(
        **arguments: Unpack[_CompileKwargs],
    ) -> tuple[Hashable, jax.stages.Compiled]:
        started.append(arguments["lowering_key"])
        if arguments["lowering_key"] == "first":
            release_first.wait(_WAIT_SECONDS)
            first_finished.set()
        return original_compile(**arguments)

    def fail_third(**arguments: Unpack[_RolesKwargs]) -> None:
        lowered.append(arguments["label"])
        original_roles(**arguments)
        if len(lowered) == 3:
            raise lower_error

    _RecordingExecutor.instances = []
    monkeypatch.setattr(backward_induction, "ThreadPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(backward_induction, "_compile_and_log", held_compile)
    monkeypatch.setattr(backward_induction, "_assert_lowered_output_roles", fail_third)
    raised: BaseException | None = None
    try:
        backward_induction._lower_and_compile_wave(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    returned_while_first_held = not first_finished.is_set()
    release_first.set()
    (pool,) = _RecordingExecutor.instances
    for thread in tuple(pool._threads):
        thread.join(_WAIT_SECONDS)
    return {
        "raised_is_lowering_error": raised is lower_error,
        "raised_while_first_compile_runs": returned_while_first_held,
        "queued_compile_cancelled": pool.futures[1].cancelled(),
        "queued_compile_never_started": started == ["first"],
        "no_further_lowering": len(lowered) == 3,
        "note_names_failing_program": any(
            kwargs["labels"]["third"] in note
            for note in getattr(raised, "__notes__", [])
        ),
        "nothing_published": kwargs["compiled"] == {},
        "no_threads_left": not any(t.is_alive() for t in pool._threads),
        "no_futures_left": all(future.done() for future in pool.futures),
    }


@pytest.mark.parametrize(
    "observation",
    [
        "raised_is_lowering_error",
        "raised_while_first_compile_runs",
        "queued_compile_cancelled",
        "queued_compile_never_started",
        "no_further_lowering",
        "note_names_failing_program",
        "nothing_published",
        "no_threads_left",
        "no_futures_left",
    ],
)
def test_lower_and_compile_wave_lowering_error_stops_the_wave_at_once(
    *, monkeypatch: pytest.MonkeyPatch, observation: str
) -> None:
    """A lowering error is raised at once, without waiting for queued compiles.

    - the raised exception is the lowering error itself, raised while an earlier
      compile is still running;
    - compiles queued but not started are cancelled and no later program is lowered;
    - the exception carries a note naming the program whose lowering failed;
    - no executable is published, even from compiles that were already running;
    - once the running compile ends, no pool thread or unfinished future remains.
    """
    assert _run_lowering_error_while_compiling(monkeypatch)[observation]


def _run_failing_second_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, bool]:
    """Fail the second of five compiles while the first compile is still running.

    Two compile workers. The first and third compiles are held until released.
    Submitting the fourth compile releases the second, which fails, and returns
    only once that failure is recorded. The second's worker then takes the third
    compile, so the fourth is queued but not started when the wave next checks
    for compile errors, before the fifth lowering.
    """
    keys = ("first", "second", "third", "fourth", "fifth")
    kwargs, _ = _wave_kwargs(n=1, n_workers=2, keys=keys)
    original_compile = backward_induction._compile_and_log
    original_roles = backward_induction._assert_lowered_output_roles
    compile_error = ValueError("distinct second-compilation error")
    release_held = threading.Event()
    release_second = threading.Event()
    first_finished = threading.Event()
    started: list[Hashable] = []
    lowered: list[str] = []
    lock = threading.Lock()

    def held_or_failing_compile(
        **arguments: Unpack[_CompileKwargs],
    ) -> tuple[Hashable, jax.stages.Compiled]:
        key = arguments["lowering_key"]
        with lock:
            started.append(key)
        if key == "second":
            release_second.wait(_WAIT_SECONDS)
            raise compile_error
        if key in ("first", "third"):
            release_held.wait(_WAIT_SECONDS)
        result = original_compile(**arguments)
        if key == "first":
            first_finished.set()
        return result

    def record_lowering(**arguments: Unpack[_RolesKwargs]) -> None:
        lowered.append(arguments["label"])
        original_roles(**arguments)

    class FailSecondOnFourthSubmit(_RecordingExecutor):
        def submit(self, *args: Any, **kwargs: Any) -> Future:
            future = super().submit(*args, **kwargs)
            if len(self.futures) == 4:
                release_second.set()
                wait(self.futures[1:2], timeout=_WAIT_SECONDS)
            return future

    _RecordingExecutor.instances = []
    monkeypatch.setattr(
        backward_induction, "ThreadPoolExecutor", FailSecondOnFourthSubmit
    )
    monkeypatch.setattr(backward_induction, "_compile_and_log", held_or_failing_compile)
    monkeypatch.setattr(
        backward_induction, "_assert_lowered_output_roles", record_lowering
    )
    raised: BaseException | None = None
    try:
        backward_induction._lower_and_compile_wave(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    returned_while_first_held = not first_finished.is_set()
    release_held.set()
    release_second.set()
    (pool,) = _RecordingExecutor.instances
    for thread in tuple(pool._threads):
        thread.join(_WAIT_SECONDS)
    return {
        "raised_is_compile_error": raised is compile_error,
        "raised_while_first_compile_runs": returned_while_first_held,
        "queued_compile_cancelled": pool.futures[3].cancelled(),
        "queued_compile_never_started": "fourth" not in started,
        "no_further_lowering": len(lowered) == 4,
        "note_names_failing_program": any(
            note == f"while compiling {kwargs['labels']['second']}"
            for note in getattr(raised, "__notes__", [])
        ),
        "nothing_published": kwargs["compiled"] == {},
    }


@pytest.mark.parametrize(
    "observation",
    [
        "raised_is_compile_error",
        "raised_while_first_compile_runs",
        "queued_compile_cancelled",
        "queued_compile_never_started",
        "no_further_lowering",
        "note_names_failing_program",
        "nothing_published",
    ],
)
def test_lower_and_compile_wave_worker_exception_propagates(
    *, monkeypatch: pytest.MonkeyPatch, observation: str
) -> None:
    """A compile worker's exception stops the wave as soon as the caller sees it.

    - the raised exception is the compile error itself, raised while an earlier
      compile is still running;
    - compiles queued but not started are cancelled and no later program is lowered;
    - the exception carries a note naming the program whose compile failed;
    - no executable is published.
    """
    assert _run_failing_second_compile(monkeypatch)[observation]


class _CompileWorkerError(Exception):
    """Stand-in for an error raised inside a compile worker."""


def _raise_from_compile(monkeypatch: pytest.MonkeyPatch) -> _WaveKwargs:
    """Make `Lowered.compile` raise; return the wave's arguments."""

    def failing_compile(_: jax.stages.Lowered) -> jax.stages.Compiled:
        raise _CompileWorkerError("compile failed")

    monkeypatch.setattr(jax.stages.Lowered, "compile", failing_compile)
    kwargs, _ = _wave_kwargs(n=3, n_workers=1)
    return kwargs


def _raise_from_memory_diagnostic(monkeypatch: pytest.MonkeyPatch) -> _WaveKwargs:
    """Make the kernel-memory diagnostic raise; return the wave's arguments."""

    def failing_diagnostic(**_: Unpack[_MemoryKwargs]) -> None:
        raise _CompileWorkerError("memory analysis failed")

    monkeypatch.setattr(backward_induction, "_log_kernel_memory", failing_diagnostic)
    kwargs, _ = _wave_kwargs(n=3, n_workers=1)
    kwargs["log_kernel_memory"] = True
    return kwargs


_WORKER_FAILURES = pytest.mark.parametrize(
    "arrange",
    [_raise_from_compile, _raise_from_memory_diagnostic],
    ids=["compile", "memory-diagnostic"],
)


@_WORKER_FAILURES
def test_lower_and_compile_wave_worker_exception_keeps_its_type(
    *,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[pytest.MonkeyPatch], _WaveKwargs],
) -> None:
    """A compile worker's exception surfaces with its original type."""
    kwargs = arrange(monkeypatch)
    raised: BaseException | None = None
    try:
        backward_induction._lower_and_compile_wave(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    assert type(raised) is _CompileWorkerError


@_WORKER_FAILURES
def test_lower_and_compile_wave_worker_exception_names_its_program(
    *,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[pytest.MonkeyPatch], _WaveKwargs],
) -> None:
    """A compile worker's exception carries a note naming the failing program."""
    kwargs = arrange(monkeypatch)
    with pytest.raises(_CompileWorkerError) as caught:
        backward_induction._lower_and_compile_wave(**kwargs)
    notes = getattr(caught.value, "__notes__", [])
    assert any(label in note for note in notes for label in kwargs["labels"].values())
