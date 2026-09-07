"""Count JAX traces, lowerings, and compiles inside a block.

The counts come from JAX's own instrumentation: `jax.monitoring` invokes every
registered duration listener once per top-level jaxpr trace, per jaxpr-to-MLIR
lowering, and per backend compilation. Registration is process-global, so a
block must contain only the work being measured.
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
class DispatchCounts:
    """Counts observed inside one `count_dispatches` block."""

    traces: int = 0
    """Top-level jaxpr traces JAX performed."""

    lowerings: int = 0
    """Jaxpr-to-MLIR module conversions JAX performed."""

    compiles: int = 0
    """Backend compilation requests JAX issued."""


@contextlib.contextmanager
def count_dispatches() -> Iterator[DispatchCounts]:
    """Count JAX traces, lowerings, and compiles performed inside the block.

    Registers one `jax.monitoring` duration listener for the duration of the
    block and unregisters it afterwards. The counts are those of the running
    process, so the block must contain only the work being measured.
    """
    counts = DispatchCounts()
    listener = _EventCounter(counts=counts)
    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield counts
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)


_FIELD_BY_EVENT = {
    TRACE_EVENT: "traces",
    LOWERING_EVENT: "lowerings",
    COMPILE_EVENT: "compiles",
}


@dataclasses.dataclass(frozen=True, eq=False)
class _EventCounter:
    """Add one to `counts` for every duration event of a counted kind.

    Identity comparison (`eq=False`) keeps `unregister_event_duration_listener`
    from removing a listener belonging to a different block.
    """

    counts: DispatchCounts
    """Counts this listener adds to."""

    # keyword-only-exempt: library-callback=jax.monitoring.record_event_duration_secs
    def __call__(self, event: str, duration_secs: float, **kwargs: str | int) -> None:
        """Add one to the field the event maps to, ignoring every other event."""
        field = _FIELD_BY_EVENT.get(event)
        if field is not None:
            setattr(self.counts, field, getattr(self.counts, field) + 1)
