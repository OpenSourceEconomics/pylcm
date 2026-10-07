"""Contracts for the benchmark workflows and the CPU benchmark-harness lane."""

from pathlib import Path

import yaml

from tests.ci import ci_workloads
from tests.ci.cpu_suite_invocations import (
    benchmark_harness_invocation_argvs,
    cpu_suite_invocation_argvs,
)

_BENCHMARK_JOB = "tests-benchmarks"
_BENCHMARK_COVERAGE = "coverage-benchmarks"
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


def test_benchmark_harness_lane_measures_coverage_of_the_whole_checkout() -> None:
    """The lane measures coverage so the harness code it exercises is counted."""
    assert "--cov=./" in _the_benchmark_harness_argv()


def test_benchmark_harness_lane_writes_its_own_coverage_report() -> None:
    """The lane's coverage lands in a report named after the lane."""
    assert (
        f"--cov-report=xml:reports/{_BENCHMARK_COVERAGE}.xml"
        in _the_benchmark_harness_argv()
    )


def test_benchmark_harness_lane_uploads_its_coverage_artifact() -> None:
    """The lane uploads its report as the artifact the combine stage downloads."""
    steps = _cpu_workflow()["jobs"][_BENCHMARK_JOB]["steps"]
    assert [
        step["with"]["path"]
        for step in steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
        and step["with"]["name"] == _BENCHMARK_COVERAGE
    ] == [f"reports/{_BENCHMARK_COVERAGE}.xml"]


def test_benchmark_harness_coverage_is_a_recorded_contributor() -> None:
    """The combine stage refuses to publish without the benchmark coverage."""
    assert _BENCHMARK_COVERAGE in ci_workloads.coverage_contributors()


def test_coverage_combine_stage_waits_for_the_benchmark_harness_lane() -> None:
    """The single Codecov upload starts only after the lane has reported."""
    assert _BENCHMARK_JOB in _cpu_workflow()["jobs"]["coverage"]["needs"]


def test_codecov_ignores_the_benchmark_test_modules() -> None:
    """Benchmark test modules are not patch lines; the harness under them is."""
    codecov = yaml.safe_load(Path("codecov.yml").read_text(encoding="utf-8"))
    assert "benchmarks/test_*.py" in codecov["ignore"]
