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
        # Unregistering asserts membership, and `clear_event_listeners()`
        # rebinds the listener list, so a listener dropped by anything else in
        # the process would raise out of this `finally` and mask whatever the
        # block itself raised.
        with contextlib.suppress(AssertionError, ValueError):
            jax.monitoring.unregister_event_duration_listener(listener)


_FIELD_BY_EVENT = {
    TRACE_EVENT: "trace_requests",
    LOWERING_EVENT: "lowering_requests",
    COMPILE_EVENT: "compile_requests",
}


@dataclasses.dataclass(frozen=True, eq=False)
class _EventCounter:
    """Add one to `counts` for every duration event of a counted kind.

    Identity comparison (`eq=False`) keeps `unregister_event_duration_listener`
    from removing a listener belonging to a different block.
    """

    counts: CompileRequestCounts
    """Counts this listener adds to."""

    # keyword-only-exempt: library-callback=jax.monitoring.record_event_duration_secs
    def __call__(self, event: str, duration_secs: float, **kwargs: str | int) -> None:
        """Add one to the field the event maps to, ignoring every other event."""
        field = _FIELD_BY_EVENT.get(event)
        if field is not None:
            setattr(self.counts, field, getattr(self.counts, field) + 1)
