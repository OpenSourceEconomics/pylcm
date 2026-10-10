"""Normalize optional compiler memory reports at the execution seam."""

import dataclasses
import functools
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from _lcm.typing import PytreeValue


@runtime_checkable
class CompilerMemoryReport(Protocol):
    """The byte counters pylcm reads off `Compiled.memory_analysis()`.

    JAX returns one attribute record with every counter as an integer on CPU and GPU,
    for single-device and sharded executables alike.
    """

    peak_memory_in_bytes: int
    argument_size_in_bytes: int
    output_size_in_bytes: int
    alias_size_in_bytes: int
    temp_size_in_bytes: int
    generated_code_size_in_bytes: int
    host_argument_size_in_bytes: int
    host_output_size_in_bytes: int
    host_alias_size_in_bytes: int
    host_temp_size_in_bytes: int
    host_generated_code_size_in_bytes: int


@runtime_checkable
class MemoryAnalyzable(Protocol):
    """Compiler result exposing JAX-style memory analysis."""

    def memory_analysis(self) -> CompilerMemoryReport | None:
        """Return compiler workspace statistics, or `None` when unsupported."""
        ...


@dataclasses.dataclass(frozen=True)
class CompilerMemoryBytes:
    """Backend-independent byte counts from JAX compiler memory analysis."""

    generated_code_size_in_bytes: int | None
    argument_size_in_bytes: int | None
    output_size_in_bytes: int | None
    alias_size_in_bytes: int | None
    temp_size_in_bytes: int | None
    peak_memory_in_bytes: int | None
    host_generated_code_size_in_bytes: int | None
    host_argument_size_in_bytes: int | None
    host_output_size_in_bytes: int | None
    host_alias_size_in_bytes: int | None
    host_temp_size_in_bytes: int | None


def compiler_memory_bytes(
    *, compiled: MemoryAnalyzable | Callable[..., PytreeValue]
) -> CompilerMemoryBytes | None:
    """Normalize a backend memory-analysis object to stable integer byte fields.

    Memory reporting is an optional backend capability, so each of these yields no
    report rather than changing compilation or replay behavior:

    - a core without `memory_analysis`;
    - an analysis that raises or returns `None`;
    - a report that is not a complete `CompilerMemoryReport`.

    A counter the backend leaves unset yields `None`.
    """
    if not isinstance(compiled, MemoryAnalyzable):
        return None
    try:
        stats = compiled.memory_analysis()
    except Exception:  # noqa: BLE001 - analysis is optional across JAX backends
        return None
    if not isinstance(stats, CompilerMemoryReport):
        return None

    optional_bytes = functools.partial(_optional_bytes, stats=stats)
    return CompilerMemoryBytes(
        generated_code_size_in_bytes=optional_bytes(
            name="generated_code_size_in_bytes"
        ),
        argument_size_in_bytes=optional_bytes(name="argument_size_in_bytes"),
        output_size_in_bytes=optional_bytes(name="output_size_in_bytes"),
        alias_size_in_bytes=optional_bytes(name="alias_size_in_bytes"),
        temp_size_in_bytes=optional_bytes(name="temp_size_in_bytes"),
        peak_memory_in_bytes=optional_bytes(name="peak_memory_in_bytes"),
        host_generated_code_size_in_bytes=optional_bytes(
            name="host_generated_code_size_in_bytes"
        ),
        host_argument_size_in_bytes=optional_bytes(name="host_argument_size_in_bytes"),
        host_output_size_in_bytes=optional_bytes(name="host_output_size_in_bytes"),
        host_alias_size_in_bytes=optional_bytes(name="host_alias_size_in_bytes"),
        host_temp_size_in_bytes=optional_bytes(name="host_temp_size_in_bytes"),
    )


def _optional_bytes(*, stats: CompilerMemoryReport, name: str) -> int | None:
    """Read one byte count off a backend memory report, `None` when unset."""
    value = getattr(stats, name)
    return None if value is None else int(value)
