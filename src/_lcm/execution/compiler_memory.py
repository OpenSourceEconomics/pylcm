"""Normalize optional compiler memory reports at the execution seam."""

import dataclasses
from typing import Any


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


def compiler_memory_bytes(*, compiled: Any) -> CompilerMemoryBytes | None:  # noqa: ANN401
    """Normalize a backend memory-analysis object to stable integer byte fields.

    Memory reporting is an optional backend capability. Unsupported executables,
    missing reports, and missing individual fields therefore return ``None`` at
    the corresponding level rather than changing compilation or replay behavior.
    """
    try:
        stats = compiled.memory_analysis()
    except Exception:  # noqa: BLE001 - analysis is optional across JAX backends
        return None
    if stats is None:
        return None

    def optional_bytes(name: str) -> int | None:
        value = getattr(stats, name, None)
        return None if value is None else int(value)

    return CompilerMemoryBytes(
        generated_code_size_in_bytes=optional_bytes("generated_code_size_in_bytes"),
        argument_size_in_bytes=optional_bytes("argument_size_in_bytes"),
        output_size_in_bytes=optional_bytes("output_size_in_bytes"),
        alias_size_in_bytes=optional_bytes("alias_size_in_bytes"),
        temp_size_in_bytes=optional_bytes("temp_size_in_bytes"),
        peak_memory_in_bytes=optional_bytes("peak_memory_in_bytes"),
        host_generated_code_size_in_bytes=optional_bytes(
            "host_generated_code_size_in_bytes"
        ),
        host_argument_size_in_bytes=optional_bytes("host_argument_size_in_bytes"),
        host_output_size_in_bytes=optional_bytes("host_output_size_in_bytes"),
        host_alias_size_in_bytes=optional_bytes("host_alias_size_in_bytes"),
        host_temp_size_in_bytes=optional_bytes("host_temp_size_in_bytes"),
    )
