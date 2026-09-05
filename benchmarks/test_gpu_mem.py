"""Tests for the GPU peak-memory measurement harness."""

# ruff: noqa: SLF001

import json
import secrets
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.asv import _gpu_mem, bench_mahler_yum
from benchmarks.asv._gpu_mem import _PROJECT_ROOT, _subprocess_env

# Blocks the `resource` module before the harness is imported, so the fresh
# interpreter sees exactly what a host without that module (Windows) sees.
_WITHOUT_RESOURCE_PREAMBLE = "import sys; sys.modules['resource'] = None\n"


def _run_without_resource_module(*, code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter that cannot import `resource`."""
    return subprocess.run(
        [sys.executable, "-c", _WITHOUT_RESOURCE_PREAMBLE + code],
        capture_output=True,
        text=True,
        check=False,
        cwd=_PROJECT_ROOT,
    )


def test_subprocess_env_disables_autotuning():
    """GPU-mem subprocess disables XLA autotuning for a deterministic compile."""
    env = _subprocess_env({"PATH": "/usr/bin"})
    assert "--xla_gpu_autotune_level=0" in env["XLA_FLAGS"]


def test_subprocess_env_appends_to_existing_xla_flags():
    """Existing `XLA_FLAGS` are kept; the autotune flag is appended, not clobbered."""
    env = _subprocess_env({"XLA_FLAGS": "--xla_gpu_foo=1"})
    assert "--xla_gpu_foo=1" in env["XLA_FLAGS"]
    assert "--xla_gpu_autotune_level=0" in env["XLA_FLAGS"]


def test_subprocess_env_drops_mem_fraction():
    """The subprocess gets full GPU memory: the MEM_FRACTION cap is removed."""
    env = _subprocess_env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3"})
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env


def test_gpu_mem_imports_on_a_host_without_resource_module():
    """The harness imports on hosts whose Python has no `resource` module."""
    result = _run_without_resource_module(code="import benchmarks.asv._gpu_mem\n")
    assert result.returncode == 0, result.stderr


def test_cpu_peak_bytes_is_nan_on_a_host_without_resource_module():
    """Without `resource`, the host peak is reported as NaN — unknown, not zero."""
    result = _run_without_resource_module(
        code=(
            "from benchmarks.asv._gpu_mem import _get_cpu_peak_bytes\n"
            "print(repr(_get_cpu_peak_bytes()))\n"
        )
    )
    assert result.stdout.strip() == "nan", result.stderr


@pytest.mark.parametrize("raw_peak", [True, 1.5, "123", -1])
def test_gpu_peak_bytes_rejects_non_exact_backend_values(
    *,
    raw_peak: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend statistics must already be an exact non-negative byte count."""

    class _Device:
        def memory_stats(self) -> dict[str, object]:
            return {"peak_bytes_in_use": raw_peak}

    monkeypatch.setattr("jax.effects_barrier", lambda: None)

    with pytest.raises(RuntimeError, match="non-negative exact int"):
        _gpu_mem._get_gpu_peak_bytes(device=_Device())


def test_gpu_memory_profile_runs_three_fresh_phases_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each exact phase contributes its own untouched child-process peak."""
    peaks = {
        _gpu_mem.AUTOMATIC_SOLVE_SIMULATE: 101,
        _gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE: 202,
        _gpu_mem.LOAD_SUPPLIED_SOLUTION_SIMULATE: 303,
    }
    calls: list[str] = []
    phase_roots: list[Path] = []
    tokens: list[str] = []

    # keyword-only-exempt: library-callback=subprocess.run
    def _run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text
        assert not check
        assert cwd == _PROJECT_ROOT
        phase = command[command.index("--profile-phase") + 1]
        archive_path = Path(command[command.index("--archive") + 1])
        token = command[command.index("--invocation-token") + 1]
        calls.append(phase)
        tokens.append(token)
        phase_roots.append(Path(env["TMPDIR"]).parent)

        if phase == _gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE:
            archive_path.write_bytes(b"persisted solution")
        elif phase == _gpu_mem.LOAD_SUPPLIED_SOLUTION_SIMULATE:
            assert archive_path.read_bytes() == b"persisted solution"

        archive = (
            None
            if phase == _gpu_mem.AUTOMATIC_SOLVE_SIMULATE
            else _gpu_mem._file_provenance(archive_path)
        )
        record = {
            "protocol_version": 1,
            "phase": phase,
            "bench_module": "benchmarks.fake",
            "bench_class": "FakeBenchmark",
            "invocation_token": token,
            "pid": 1000 + len(calls),
            "project_root": str(_PROJECT_ROOT),
            "executable": str(Path(sys.executable).resolve()),
            "environment": {key: env[key] for key in _gpu_mem._PROFILE_ENV_KEYS},
            "device": {"id": 0, "kind": "Fake GPU", "platform": "gpu"},
            "archive": archive,
            "peak_bytes_in_use": peaks[phase],
        }
        stdout = f"diagnostic\n{_gpu_mem._PROFILE_MARKER} {json.dumps(record)}\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(_gpu_mem.subprocess, "run", _run)

    result = _gpu_mem.measure_gpu_memory_profile(
        bench_module="benchmarks.fake",
        bench_class="FakeBenchmark",
    )

    assert calls == list(_gpu_mem.GPU_MEMORY_PHASES)
    assert result == peaks
    assert len(set(tokens)) == 3
    assert len(set(phase_roots)) == 3
    assert [path.name for path in phase_roots] == list(_gpu_mem.GPU_MEMORY_PHASES)


def test_gpu_memory_profile_rejects_reused_child_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate phase invocations must use distinct operating-system children."""
    child_pids = iter((123, 123))

    # keyword-only-exempt: library-callback=subprocess.run
    def _run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text
        assert not check
        assert cwd == _PROJECT_ROOT
        assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        phase = command[command.index("--profile-phase") + 1]
        if phase == _gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE:
            archive_path = Path(command[command.index("--archive") + 1])
            archive_path.write_bytes(b"persisted solution")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def _parse(**_: object) -> tuple[int, int]:
        return 1, next(child_pids)

    monkeypatch.setattr(_gpu_mem.subprocess, "run", _run)
    monkeypatch.setattr(_gpu_mem, "_parse_profile_child_result", _parse)

    with pytest.raises(RuntimeError, match="reused child pid 123"):
        _gpu_mem.measure_gpu_memory_profile(
            bench_module="benchmarks.fake",
            bench_class="FakeBenchmark",
        )


def _profile_test_env(*, root: Path) -> dict[str, str]:
    env = _subprocess_env({})
    env.update(
        {
            "TMPDIR": str(root / "tmp"),
            "XDG_CACHE_HOME": str(root / "xdg"),
            "JAX_COMPILATION_CACHE_DIR": str(root / "jax"),
        }
    )
    return env


def _valid_profile_child_record(
    *,
    env: dict[str, str],
    invocation_token: str,
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "phase": _gpu_mem.AUTOMATIC_SOLVE_SIMULATE,
        "bench_module": "benchmarks.fake",
        "bench_class": "FakeBenchmark",
        "invocation_token": invocation_token,
        "pid": 123,
        "project_root": str(_PROJECT_ROOT),
        "executable": str(Path(sys.executable).resolve()),
        "environment": {key: env[key] for key in _gpu_mem._PROFILE_ENV_KEYS},
        "device": {"id": 0, "kind": "Fake GPU", "platform": "gpu"},
        "archive": None,
        "peak_bytes_in_use": 1234,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("phase", "wrong_phase"),
        ("pid", True),
        ("peak_bytes_in_use", True),
        ("device", {"id": 0, "kind": "CPU", "platform": "cpu"}),
    ],
)
def test_gpu_memory_profile_rejects_malformed_child_provenance(
    *,
    field: str,
    bad_value: object,
    tmp_path: Path,
) -> None:
    """A successful exit cannot admit a mislabeled or non-GPU measurement."""
    env = _profile_test_env(root=tmp_path)
    invocation_token = secrets.token_hex(16)
    record = _valid_profile_child_record(
        env=env,
        invocation_token=invocation_token,
    )
    record[field] = bad_value

    with pytest.raises(RuntimeError):
        _gpu_mem._validate_profile_child_result(
            record=record,
            phase=_gpu_mem.AUTOMATIC_SOLVE_SIMULATE,
            bench_module="benchmarks.fake",
            bench_class="FakeBenchmark",
            invocation_token=invocation_token,
            env=env,
            archive=None,
        )


def test_gpu_memory_profile_requires_exactly_one_child_marker(
    tmp_path: Path,
) -> None:
    """Duplicate success records are ambiguous and fail closed."""
    env = _profile_test_env(root=tmp_path)
    invocation_token = secrets.token_hex(16)
    payload = json.dumps(
        _valid_profile_child_record(
            env=env,
            invocation_token=invocation_token,
        )
    )
    result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            f"{_gpu_mem._PROFILE_MARKER} {payload}\n"
            f"{_gpu_mem._PROFILE_MARKER} {payload}\n"
        ),
        stderr="",
    )

    with pytest.raises(RuntimeError, match="2 result markers"):
        _gpu_mem._parse_profile_child_result(
            result=result,
            phase=_gpu_mem.AUTOMATIC_SOLVE_SIMULATE,
            bench_module="benchmarks.fake",
            bench_class="FakeBenchmark",
            invocation_token=invocation_token,
            env=env,
            archive=None,
        )


def test_mahler_yum_asv_surface_has_exact_phase_trackers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mahler-Yum exposes one independent series for each profile phase."""
    measured = dict(zip(_gpu_mem.GPU_MEMORY_PHASES, (101, 202, 303), strict=True))

    def _measure(*, bench_module: str, bench_class: str) -> dict[str, int]:
        assert bench_module == "benchmarks.asv.bench_mahler_yum"
        assert bench_class == "MahlerYum"
        return measured

    monkeypatch.setattr(_gpu_mem, "measure_gpu_memory_profile", _measure)

    instance = bench_mahler_yum.MahlerYumGpuPeakMem()
    cache = instance.setup_cache()
    instance.setup(cache)

    assert instance.track_peak_gpu_mem_automatic_solve_simulate(cache) == 101
    assert instance.track_peak_gpu_mem_solve_save_all_persistable(cache) == 202
    assert instance.track_peak_gpu_mem_load_supplied_solution_simulate(cache) == 303
    metric_names = {
        name
        for name in dir(type(instance))
        if name.startswith(("time_", "peakmem_", "track_"))
    }
    assert metric_names == {
        "track_peak_gpu_mem_automatic_solve_simulate",
        "track_peak_gpu_mem_solve_save_all_persistable",
        "track_peak_gpu_mem_load_supplied_solution_simulate",
    }


class _FakeMahlerSolution:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save(self, *, path: Path) -> None:
        self.saved_paths.append(path)


class _FakeMahlerModel:
    def __init__(self, solution: _FakeMahlerSolution) -> None:
        self.solution = solution
        self.solve_calls: list[dict[str, object]] = []
        self.simulate_calls: list[dict[str, object]] = []

    def solve(self, **kwargs: object) -> _FakeMahlerSolution:
        self.solve_calls.append(kwargs)
        return self.solution

    def simulate(self, **kwargs: object) -> object:
        self.simulate_calls.append(kwargs)
        return object()


def test_mahler_yum_gpu_memory_phases_dispatch_exact_workloads(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mahler-Yum dispatches automatic, persistence, and supplied-solution work."""
    saved = _FakeMahlerSolution()
    loaded = _FakeMahlerSolution()
    model = _FakeMahlerModel(saved)
    benchmark = bench_mahler_yum.MahlerYum()
    benchmark.model = model
    benchmark.model_params = {"p": 1}
    benchmark.initial_conditions = {"state": object()}
    archive = tmp_path / "solution.lcm"
    load_calls: list[Path] = []

    def _load_solution(*, path: Path) -> _FakeMahlerSolution:
        load_calls.append(path)
        return loaded

    monkeypatch.setattr("lcm.persistence.load_solution", _load_solution)

    benchmark.execute_gpu_memory_phase(
        phase=_gpu_mem.AUTOMATIC_SOLVE_SIMULATE,
        archive_path=archive,
    )
    assert model.simulate_calls[-1] == {
        "params": benchmark.model_params,
        "initial_conditions": benchmark.initial_conditions,
        "log_level": "off",
    }

    benchmark.execute_gpu_memory_phase(
        phase=_gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE,
        archive_path=archive,
    )
    from lcm.solver_api import ResultRetention

    assert model.solve_calls[-1] == {
        "params": benchmark.model_params,
        "log_level": "off",
        "retention": ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    }
    assert saved.saved_paths == [archive]

    benchmark.execute_gpu_memory_phase(
        phase=_gpu_mem.LOAD_SUPPLIED_SOLUTION_SIMULATE,
        archive_path=archive,
    )
    assert load_calls == [archive]
    assert model.simulate_calls[-1] == {
        "params": benchmark.model_params,
        "initial_conditions": benchmark.initial_conditions,
        "solution": loaded,
        "log_level": "off",
    }
