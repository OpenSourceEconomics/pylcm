"""Contracts for the benchmark workflows and the CPU benchmark-harness lane."""

from pathlib import Path

import yaml

from tests.ci.cpu_suite_invocations import (
    benchmark_harness_invocation_argvs,
    cpu_suite_invocation_argvs,
)

_BENCHMARK_JOB = "tests-benchmarks"
_BENCHMARK_SUBPROJECTS = (
    "benchmarks/continuous_scaling",
    "benchmarks/discrete_control",
)


def _cpu_workflow() -> dict:
    return yaml.safe_load(Path(".github/workflows/cpu.yml").read_text(encoding="utf-8"))


def _benchmark_harness_invocations() -> list[tuple[str, list[str]]]:
    """Return `(job, argv)` for every benchmark-harness invocation in `cpu.yml`."""
    return [
        (job_name, argv)
        for job_name, job in _cpu_workflow()["jobs"].items()
        for step in job.get("steps", ())
        for argv in benchmark_harness_invocation_argvs(str(step.get("run", "")))
    ]


def _the_benchmark_harness_argv() -> list[str]:
    ((_, argv),) = _benchmark_harness_invocations()
    return argv


def test_benchmark_workflow_materializes_main_without_network_credentials() -> None:
    """ASV gets a local main ref after checkout credentials are removed."""
    workflow = yaml.safe_load(
        Path(".github/workflows/benchmark-pr.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["run-benchmarks"]["steps"]
    checkout = next(
        step
        for step in steps
        if step.get("uses", "").partition("@")[0] == "actions/checkout"
    )
    ensure_main = next(
        step for step in steps if step.get("name") == "Ensure main ref exists"
    )

    assert checkout["with"]["persist-credentials"] is False
    assert ensure_main["run"] == "git branch --force main origin/main"


def test_cpu_workflow_runs_the_benchmark_harness_tests_once_in_their_own_job() -> None:
    """Exactly one `cpu.yml` invocation collects `benchmarks/`, in its own job."""
    assert [job for job, _ in _benchmark_harness_invocations()] == [_BENCHMARK_JOB]


def test_benchmark_harness_lane_collects_the_benchmarks_directory() -> None:
    """The lane names `benchmarks` itself, since `testpaths` names only `tests`."""
    assert "benchmarks" in _the_benchmark_harness_argv()


def test_benchmark_harness_lane_ignores_the_standalone_subprojects() -> None:
    """The scaling and discrete-control subprojects keep their own harnesses."""
    argv = _the_benchmark_harness_argv()
    assert [
        argument.partition("=")[2]
        for argument in argv
        if argument.startswith("--ignore=")
    ] == list(_BENCHMARK_SUBPROJECTS)


def test_benchmark_harness_lane_streams_one_line_per_test() -> None:
    """`-v` leaves a per-test record even when the run is killed part-way."""
    assert "-v" in _the_benchmark_harness_argv()


def test_benchmark_harness_lane_writes_a_junit_report() -> None:
    """A finished run leaves a JUnit report whose counts cannot be short."""
    assert any(arg.startswith("--junitxml=") for arg in _the_benchmark_harness_argv())


def test_benchmark_harness_lane_is_not_a_cpu_suite_invocation() -> None:
    """The suite-policy contracts do not apply where `tests/conftest.py` never loads."""
    run = "pixi run -e tests-cpu pytest benchmarks --ignore=benchmarks/x -v"
    assert cpu_suite_invocation_argvs(run) == []


def test_cpu_gate_waits_for_the_benchmark_harness_lane() -> None:
    """The required `cpu` check fails when the benchmark-harness lane fails."""
    assert _BENCHMARK_JOB in _cpu_workflow()["jobs"]["cpu"]["needs"]
