"""Count the JAX trace, lowering, and compile requests issued inside a block.

The counts come from JAX's own instrumentation: `jax.monitoring` invokes every
registered duration listener once per top-level jaxpr trace, per jaxpr-to-MLIR
lowering, and per backend compilation request. Registration is process-global,
so a block must contain only the work being measured.

A *request* is what is counted, not the work it causes. The compile event wraps
`compile_or_get_cached`, so a hit in JAX's persistent compilation cache raises
the count exactly as a cold compilation does. Zero compile requests therefore
means the executable was already resident in this process; a non-zero count
does not by itself mean XLA ran.
"""

import contextlib
import dataclasses
from collections.abc import Iterator

import jax._src.monitoring
import jax.monitoring

# Monitoring event JAX records once per top-level jaxpr trace.
TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"

# Monitoring event JAX records once per jaxpr-to-MLIR module conversion.
LOWERING_EVENT = "/jax/core/compile/jaxpr_to_mlir_module_duration"

# Monitoring event JAX records once per backend compilation request.
COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


@dataclasses.dataclass
class CompileRequestCounts:
    """Requests observed inside one `count_compile_requests` block."""

    trace_requests: int = 0
    """Top-level jaxpr traces JAX was asked for."""

    lowering_requests: int = 0
    """Jaxpr-to-MLIR module conversions JAX was asked for."""

    compile_requests: int = 0
    """Backend compilations JAX was asked for, persistent-cache hits included."""


@contextlib.contextmanager
def count_compile_requests() -> Iterator[CompileRequestCounts]:
    """Count the trace, lowering, and compile requests issued inside the block.

    Registers one `jax.monitoring` duration listener for the duration of the
    block and unregisters it afterwards. The counts are those of the running
    process, so the block must contain only the work being measured.
    """
    counts = CompileRequestCounts()
    listener = _EventCounter(counts=counts)
    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield counts
    finally:
        listener.disarmed = True
        _unregister(listener=listener)


_FIELD_BY_EVENT = {
    TRACE_EVENT: "trace_requests",
    LOWERING_EVENT: "lowering_requests",
    COMPILE_EVENT: "compile_requests",
}


@dataclasses.dataclass(eq=False)
class _EventCounter:
    """Add one to `counts` for every duration event of a counted kind.

    Identity comparison (`eq=False`) keeps `unregister_event_duration_listener`
    from removing a listener belonging to a different block. A disarmed
    listener ignores every event, so one that cannot be taken out of JAX's list
    still stops contributing to any later block's counts.
    """

    counts: CompileRequestCounts
    """Counts this listener adds to."""

    disarmed: bool = False
    """Whether this listener has stopped counting."""

    # keyword-only-exempt: library-callback=jax.monitoring.record_event_duration_secs
    def __call__(self, event: str, duration_secs: float, **kwargs: str | int) -> None:
        """Add one to the field the event maps to, ignoring every other event."""
        if self.disarmed:
            return
        field = _FIELD_BY_EVENT.get(event)
        if field is not None:
            setattr(self.counts, field, getattr(self.counts, field) + 1)


def _unregister(*, listener: _EventCounter) -> bool:
    """Take one listener back out of JAX's duration-listener list.

    Unregistering asserts membership, and `clear_event_listeners()` rebinds the
    list, so a listener something else in the process already dropped would
    raise where the caller unregisters it. Under `-O` the assert is stripped
    and `list.remove` raises `ValueError` in its place. Either way the caller
    unregisters from a `finally`, where an exception would replace whatever the
    measured block itself raised.

    So a failure is reported as a return value rather than raised, and only
    after the list is consulted: a listener that is genuinely gone is a clean
    teardown, while one still in the list would keep counting into a later
    block and is the case worth knowing about.

    Returns:
        Whether the listener is out of the list when this returns.

    """
    try:
        jax.monitoring.unregister_event_duration_listener(listener)
    except AssertionError, ValueError:
        return listener not in jax._src.monitoring.get_event_duration_listeners()  # noqa: SLF001
    return True
