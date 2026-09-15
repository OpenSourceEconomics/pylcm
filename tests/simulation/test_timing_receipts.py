"""Timing receipts retain the observations needed to assess an acceptance failure."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import jax.monitoring
import pytest

from benchmarks.asv._compile_counters import (
    COMPILE_EVENT,
    LOWERING_EVENT,
    TRACE_EVENT,
)
from tests.ci.simulation_timings import TimingMeasurement
from tests.simulation import test_compile_requests as timing_tests


def test_timed_batch_retains_order_and_excludes_warmup_compilations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipts contain every timed sample and only the timed calls' compile events."""
    calls: list[str] = []
    clock = iter(
        (0, 100, 100, 300, 300, 301, 301, 305, 305, 308, 308, 310, 310, 312, 312, 318)
    )

    def solve(**_kwargs: object) -> None:
        pass

    def simulate(*, log_level: str, **_kwargs: object) -> SimpleNamespace:
        calls.append(log_level)
        for event, repeats in (
            (TRACE_EVENT, 1),
            (LOWERING_EVENT, 2),
            (COMPILE_EVENT, 3),
        ):
            for _ in range(repeats):
                jax.monitoring.record_event_duration_secs(event, 0.001)
        return SimpleNamespace(raw_results=None)

    def witness() -> tuple[SimpleNamespace, dict, dict]:
        return SimpleNamespace(solve=solve, simulate=simulate), {}, {}

    monkeypatch.setattr(timing_tests, "WITNESSES", {"receipt": witness})
    monkeypatch.setattr(
        timing_tests, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )

    measurement = timing_tests._median_host_times(
        witness="receipt", log_level="progress", repeats=3, stub_preflight=False
    )

    assert calls == ["off", "progress"] * 4
    assert measurement.samples == (
        ("off", 1.0),
        ("progress", 4.0),
        ("off", 3.0),
        ("progress", 2.0),
        ("off", 2.0),
        ("progress", 6.0),
    )
    assert measurement.off_seconds == pytest.approx(2.0)
    assert measurement.progress_seconds == pytest.approx(4.0)
    assert (
        measurement.trace_requests,
        measurement.lowering_requests,
        measurement.compile_requests,
    ) == (6, 12, 18)


def test_receipt_preserves_observations_and_worker_identity(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker publishes raw observations and medians in a portable JSON file."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    measurement = TimingMeasurement(
        samples=(("off", 1.0), ("progress", 3.0)),
        trace_requests=2,
        lowering_requests=3,
        compile_requests=4,
    )

    receipt = measurement.write_receipt(
        directory=tmp_path,
        nodeid="tests/test_timing.py::test_wall_time[dissolution]",
        witness="dissolution",
        stub_preflight=True,
    )

    paths = tuple(tmp_path.rglob("*.json"))
    assert len(paths) == 1
    assert paths[0].read_text(encoding="utf-8").strip() == receipt
    observed = json.loads(receipt)
    assert observed["nodeid"] == "tests/test_timing.py::test_wall_time[dissolution]"
    assert observed["witness"] == "dissolution"
    assert observed["stub_preflight"] is True
    assert observed["worker_id"] == "gw3"
    assert observed["pid"] == os.getpid()
    assert observed["precision"] == (64 if jax.config.jax_enable_x64 else 32)
    assert observed["backend"] == jax.default_backend()
    assert observed["samples"] == [["off", 1.0], ["progress", 3.0]]
    assert observed["medians_seconds"] == {"off": 1.0, "progress": 3.0}
    assert observed["progress_over_off"] == pytest.approx(3.0)
    assert observed["compile_requests"] == {"trace": 2, "lowering": 3, "compile": 4}


def test_xdist_workers_preserve_success_receipts_and_full_failure_details(
    tmp_path: Path,
) -> None:
    """Two workers retain every receipt and carry complete failure JSON into xunit2."""
    receipts = tmp_path / "receipts"
    child = tmp_path / "test_worker_receipts.py"
    child.write_text(
        f"""
from pathlib import Path

import pytest

from tests.ci.simulation_timings import TimingMeasurement


@pytest.mark.parametrize("progress", [1.25, 2.0], ids=["passes", "fails"])
def test_receipt(*, request: pytest.FixtureRequest, progress: float) -> None:
    measurement = TimingMeasurement(
        samples=(("off", 1.0), ("progress", progress)) * 3,
        trace_requests=0, lowering_requests=0, compile_requests=0,
    )
    receipt = measurement.write_receipt(
        directory=Path({str(receipts)!r}), nodeid=request.node.nodeid,
        witness="dissolution", stub_preflight=False,
    )
    assert progress <= 1.5, "TIMING_RECEIPT=" + receipt
""",
        encoding="utf-8",
    )
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\njunit_family = xunit2\n", encoding="utf-8")
    junit = tmp_path / "junit.xml"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(Path(__file__).parents[2]), env.get("PYTHONPATH", ""))
    )
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            str(child),
            "-v",
            "-c",
            str(config),
            "--confcutdir",
            str(tmp_path),
            "-n",
            "2",
            "--dist",
            "each",
            f"--junitxml={junit}",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (tmp_path / "worker-output.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    assert result.returncode == 1, result.stdout + result.stderr
    cases = ET.parse(junit).findall(".//testcase")  # noqa: S314
    failures = [failure for case in cases for failure in case.findall("failure")]
    assert len(cases) == 4
    assert len(failures) == 2
    assert all(
        case.find("error") is None and case.find("skipped") is None for case in cases
    )
    files = tuple(receipts.glob("*.json"))
    assert len(files) == 4
    raw = [path.read_text(encoding="utf-8").strip() for path in files]
    records = [json.loads(value) for value in raw]
    assert {record["worker_id"] for record in records} == {"gw0", "gw1"}
    assert len({record["pid"] for record in records}) == 2
    assert all(
        record["precision"] == 64 and record["backend"] == "cpu" for record in records
    )
    for value, record in zip(raw, records, strict=True):
        assert len(record["samples"]) == 6
        assert record["compile_requests"] == {"trace": 0, "lowering": 0, "compile": 0}
        if record["progress_over_off"] > 1.5:
            assert any(
                "TIMING_RECEIPT=" + value in (failure.text or "")
                for failure in failures
            )
