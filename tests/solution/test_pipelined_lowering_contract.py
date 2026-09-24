"""Contract of one compilation wave: exact outputs, keyed identity, error paths.

The programs are integer-valued cumulative sums, so every intermediate is exactly
representable in fp32 and fp64 and the literal Python oracle below is bit-exact
regardless of reduction grouping. Instrumentation only controls thread
interleavings.
"""

import logging
import threading
from collections.abc import Callable, Hashable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
)
from _lcm.execution.output_layout import VALUE, resolve_output_layout
from _lcm.solution import backward_induction
from lcm import AgeGrid

_KEYS = ("first", "second", "third")
_WAIT_SECONDS = 10.0  # Bounded synchronisation timeout, not a performance bound.


def _program(*, wealth: jax.Array, scale: int) -> jax.Array:
    """Scaled cumulative sum; integer inputs keep it exact in any float dtype."""
    return jnp.cumsum(wealth * scale)


def _wave_kwargs(*, n: int, n_workers: int) -> tuple[dict[str, Any], jax.Array]:
    """Arguments for one wave of three programs over `n` integer-valued states."""
    wealth = jnp.asarray(np.arange(n) - n // 2, dtype=jnp.asarray(0.0).dtype)
    candidates: dict[Hashable, backward_induction._CoreCandidate] = {
        key: ((key, 0, key), ()) for key in _KEYS
    }
    resolved = {
        candidate: ResolvedCoreProgram(
            name=str(key),
            function=_program,
            arguments={"wealth": wealth},
            static_kwargs={"scale": i + 1},
            requirements=CoreExecutionRequirements(),
            output_roles=VALUE,
            disposition=CoreExecutionDisposition.PLANNED,
            donation_candidates=(),
            tile_widths={},
            specialization_key=(),
            input_transfer_plan=(),
        )
        for i, (key, candidate) in enumerate(candidates.items())
    }
    kwargs = {
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
        "ages": AgeGrid(start=0, stop=1, step="Y"),
        "n_triples_per_lowering": dict.fromkeys(_KEYS, 1),
        "log_kernel_memory": False,
        "n_workers": n_workers,
        "logger": logging.getLogger("pipelined-lowering-contract"),
        "compiled": {},
        "labels": {},
    }
    return kwargs, wealth


def _literal_cumsum(*, wealth: jax.Array, scale: int) -> np.ndarray:
    """Scaled cumulative sum by a plain Python loop over integers."""
    source = np.asarray(wealth)
    total = 0
    result = []
    for value in source.tolist():
        total += int(value) * scale
        result.append(total)
    return np.asarray(result, dtype=source.dtype)


def _bits(array: Any) -> tuple[tuple[int, ...], np.dtype, bytes]:
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
    *, n: int, n_workers: int, output: str
) -> None:
    """Executables and labels land under exactly the wave's lowering keys."""
    kwargs, _ = _wave_kwargs(n=n, n_workers=n_workers)
    backward_induction._lower_and_compile_wave(**kwargs)
    assert set(kwargs[output]) == set(_KEYS)


def _run_reversed_finishing(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run a two-worker wave in which the first compile finishes last."""
    kwargs, wealth = _wave_kwargs(n=17, n_workers=2)
    original_compile = backward_induction._compile_and_log
    original_roles = backward_induction._assert_lowered_output_roles
    caller = threading.get_ident()
    third_compiled = threading.Event()
    lock = threading.Lock()
    finished: list[str] = []
    returned: dict[str, jax.stages.Compiled] = {}
    lowered_off_caller: list[str] = []
    compiled_on_caller: list[str] = []
    first_waited: list[bool] = []

    def observed_roles(**arguments: Any) -> None:
        if threading.get_ident() != caller:
            lowered_off_caller.append(arguments["label"])
        original_roles(**arguments)

    def reverse_first(**arguments: Any) -> Any:
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


def _reversed_order(obs: dict[str, Any]) -> bool:
    return obs["finished"] == ["second", "third", "first"]


def _reversed_identity(obs: dict[str, Any]) -> bool:
    return obs["identity"]


def _reversed_threads(obs: dict[str, Any]) -> bool:
    return obs["threads"] == ([], [], [True])


def _reversed_values(obs: dict[str, Any]) -> bool:
    return obs["values"] == obs["expected_values"]


@pytest.mark.parametrize(
    "holds",
    [_reversed_order, _reversed_identity, _reversed_threads, _reversed_values],
    ids=["finishing-order", "object-identity", "thread-placement", "values"],
)
def test_lower_and_compile_wave_keys_survive_reversed_worker_finishing(
    *, monkeypatch: pytest.MonkeyPatch, holds: Callable[[dict[str, Any]], bool]
) -> None:
    """Workers finishing in reverse order still publish each executable by key.

    - the first compile is held until the third has finished;
    - each published executable is the very object its worker returned;
    - lowering stays on the calling thread and compiles run on pool threads;
    - every executable reproduces its literal oracle.
    """
    assert holds(_run_reversed_finishing(monkeypatch))


def _run_lowering_error_during_compile_error(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Fail the third lowering while the first compile is failing concurrently."""
    kwargs, _ = _wave_kwargs(n=1, n_workers=1)
    original_compile = backward_induction._compile_and_log
    original_roles = backward_induction._assert_lowered_output_roles
    third_lowered = threading.Event()
    first_finished = threading.Event()
    second_finished = threading.Event()
    lower_error = RuntimeError("distinct third-lowering error")
    compile_error = ValueError("distinct first-compilation error")
    calls = 0

    def observed_compile(**arguments: Any) -> Any:
        if arguments["lowering_key"] == "first":
            try:
                third_lowered.wait(_WAIT_SECONDS)
                raise compile_error
            finally:
                first_finished.set()
        result = original_compile(**arguments)
        second_finished.set()
        return result

    def fail_third(**arguments: Any) -> None:
        nonlocal calls
        calls += 1
        original_roles(**arguments)
        if calls == 3:
            third_lowered.set()
            raise lower_error

    monkeypatch.setattr(backward_induction, "_compile_and_log", observed_compile)
    monkeypatch.setattr(backward_induction, "_assert_lowered_output_roles", fail_third)
    raised: BaseException | None = None
    try:
        backward_induction._lower_and_compile_wave(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    return {
        "raised_is_lowering_error": raised is lower_error,
        "first_finished": first_finished.is_set(),
        "second_finished": second_finished.is_set(),
        "nothing_published": kwargs["compiled"] == {},
    }


@pytest.mark.parametrize(
    "observation",
    [
        "raised_is_lowering_error",
        "first_finished",
        "second_finished",
        "nothing_published",
    ],
)
def test_lower_and_compile_wave_lowering_error_wins_and_submitted_queue_drains(
    *, monkeypatch: pytest.MonkeyPatch, observation: str
) -> None:
    """A lowering error wins over a concurrent compile error.

    - the raised exception is the lowering error itself;
    - the failing compile and the already-submitted compile both run to the end;
    - no executable is published.
    """
    assert _run_lowering_error_during_compile_error(monkeypatch)[observation]


def _run_failing_second_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Raise a distinct error from the second program's compile worker."""
    kwargs, _ = _wave_kwargs(n=1, n_workers=2)
    original_compile = backward_induction._compile_and_log
    compile_error = ValueError("distinct second-compilation error")
    completed: set[str] = set()
    lock = threading.Lock()

    def fail_second(**arguments: Any) -> Any:
        key = arguments["lowering_key"]
        if key == "second":
            raise compile_error
        result = original_compile(**arguments)
        with lock:
            completed.add(key)
        return result

    monkeypatch.setattr(backward_induction, "_compile_and_log", fail_second)
    raised: BaseException | None = None
    try:
        backward_induction._lower_and_compile_wave(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    return {
        "raised_is_compile_error": raised is compile_error,
        "others_completed": completed == {"first", "third"},
    }


@pytest.mark.parametrize("observation", ["raised_is_compile_error", "others_completed"])
def test_lower_and_compile_wave_worker_exception_propagates(
    *, monkeypatch: pytest.MonkeyPatch, observation: str
) -> None:
    """A compile worker's exception propagates as the same object.

    The wave still waits for every other submitted compile to finish.
    """
    assert _run_failing_second_compile(monkeypatch)[observation]


class _CompileWorkerError(Exception):
    """Stand-in for an error raised inside a compile worker."""


def _raise_from_compile(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make `Lowered.compile` raise; return the wave's arguments."""

    def failing_compile(_: jax.stages.Lowered) -> jax.stages.Compiled:
        raise _CompileWorkerError("compile failed")

    monkeypatch.setattr(jax.stages.Lowered, "compile", failing_compile)
    kwargs, _ = _wave_kwargs(n=3, n_workers=1)
    return kwargs


def _raise_from_memory_diagnostic(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make the kernel-memory diagnostic raise; return the wave's arguments."""

    def failing_diagnostic(**_: Any) -> None:
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
    arrange: Callable[[pytest.MonkeyPatch], dict[str, Any]],
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
    arrange: Callable[[pytest.MonkeyPatch], dict[str, Any]],
) -> None:
    """A compile worker's exception carries a note naming the failing program."""
    kwargs = arrange(monkeypatch)
    with pytest.raises(_CompileWorkerError) as caught:
        backward_induction._lower_and_compile_wave(**kwargs)
    notes = getattr(caught.value, "__notes__", [])
    assert any(label in note for note in notes for label in kwargs["labels"].values())
