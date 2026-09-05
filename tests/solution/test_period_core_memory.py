"""A captured period's cores can be compiled for their memory without running them.

The analyzer lowers the production cores against the capture's logical shapes and
reads each executable's compiler memory analysis. Nothing is executed, so a core
whose estimated allocation exceeds the device can still be measured.
"""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import jax
import jax.numpy as jnp

from _lcm.execution.compiler_memory import (
    CompilerMemoryBytes,
    compiler_memory_bytes,
)
from _lcm.solution import period_replay
from _lcm.solution.period_capture import _PAYLOAD_NAME


class _CompileOnlyExecutable:
    def __init__(self, *, offset: int) -> None:
        self.executed = False
        self.stats = SimpleNamespace(
            generated_code_size_in_bytes=offset + 1,
            argument_size_in_bytes=offset + 2,
            output_size_in_bytes=offset + 3,
            alias_size_in_bytes=offset + 4,
            temp_size_in_bytes=offset + 5,
            peak_memory_in_bytes=offset + 6,
            host_generated_code_size_in_bytes=offset + 7,
            host_argument_size_in_bytes=offset + 8,
            host_output_size_in_bytes=offset + 9,
            host_alias_size_in_bytes=offset + 10,
            host_temp_size_in_bytes=None,
        )

    def memory_analysis(self) -> SimpleNamespace:
        return self.stats

    def __call__(self, **_kwargs: object) -> None:
        self.executed = True
        raise AssertionError("the memory analyzer executed a compiled core")


def _reports(
    reports: Mapping[str, CompilerMemoryBytes | None],
) -> dict[str, dict[str, int | None]]:
    """Read every per-core report as a plain dict, failing on a missing one."""
    read: dict[str, dict[str, int | None]] = {}
    for name, report in reports.items():
        assert report is not None, name
        read[name] = asdict(report)
    return read


def test_core_memory_analyzer_compiles_but_never_executes(
    *, monkeypatch, tmp_path: Path
) -> None:
    """Production cores are reported per name; none is called."""
    executables = {
        "main": _CompileOnlyExecutable(offset=100),
        "replay": _CompileOnlyExecutable(offset=200),
    }
    production = {
        name: SimpleNamespace(compiled=executable)
        for name, executable in executables.items()
    }
    calls = []

    def fake_production_compile(*, regime, period, **_unused):
        calls.append(("production", regime, period))
        return production

    monkeypatch.setattr(
        period_replay, "_compile_cores_for_one_period", fake_production_compile
    )
    payload = {
        "regime": object(),
        "period": 1,
        "core_tile_widths": {},
        "kernel_kwargs": {
            "regime_name": "parent",
            "ages": SimpleNamespace(values=jnp.array([40.0, 41.0])),
        },
    }
    with (tmp_path / _PAYLOAD_NAME).open("wb") as stream:
        cloudpickle.dump(payload, stream)

    analysis = period_replay.analyze_period_core_memory(directory=tmp_path)

    assert [call[0] for call in calls] == ["production"]
    assert _reports(analysis.core_memory_bytes) == {
        name: vars(executable.stats) for name, executable in executables.items()
    }
    assert (analysis.regime_name, analysis.period, analysis.age) == ("parent", 1, 41.0)
    assert analysis.preserves_production_sharding is False
    assert not any(core.compiled.executed for core in production.values())


def test_compiler_memory_bytes_normalizes_real_and_unsupported_backends() -> None:
    """Backend-specific stats become integer-or-None fields, or no report."""
    compiled = jax.jit(lambda values: values + 1).lower(jnp.ones(2)).compile()

    report = compiler_memory_bytes(compiled=compiled)

    raw = compiled.memory_analysis()
    if raw is None:
        assert report is None
    else:
        assert report is not None
        assert all(
            value is None or isinstance(value, int) for value in asdict(report).values()
        )

    class UnsupportedExecutable:
        def __init__(self, *, raises: bool) -> None:
            self.raises = raises

        def memory_analysis(self) -> None:
            if self.raises:
                raise RuntimeError("backend has no memory analysis")

    assert compiler_memory_bytes(compiled=UnsupportedExecutable(raises=False)) is None
    assert compiler_memory_bytes(compiled=UnsupportedExecutable(raises=True)) is None
