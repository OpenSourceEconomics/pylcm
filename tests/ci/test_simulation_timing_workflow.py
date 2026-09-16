"""CPU timing witnesses run once after each platform's parallel battery."""

from pathlib import Path

import pytest
import yaml

from tests.ci.cpu_suite_invocations import cpu_suite_invocation_argvs

_REPO_ROOT = Path(__file__).parents[2]
_TIMING_NODES = (
    (
        "tests/simulation/test_compile_requests.py::"
        "test_simulation_loop_host_time_at_progress_is_within_the_bar_of_off"
    ),
    (
        "tests/simulation/test_compile_requests.py::"
        "test_simulate_host_time_at_progress_is_within_the_bar_of_off"
    ),
    (
        "tests/simulation/test_preflight_contract.py::"
        "test_unstubbed_warm_full_call_progress_meets_existing_time_bar"
    ),
)
# The timing witnesses now have a job of their own, reached once per route by
# its matrix, rather than a link at the end of each general leg's chain. "Once
# after the parallel work" became "never next to any parallel work".
_TIMING_JOB = "tests-timing"
_TIMING_STEP = "Run the serial timing witnesses"
_GENERAL_STEPS = (
    ("tests", "Run one general shard"),
    ("tests-fp32", "Run one general shard"),
)


def _workflow() -> dict:
    return yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )


def _step(*, job: str, step_name: str) -> dict:
    return next(
        entry
        for entry in _workflow()["jobs"][job]["steps"]
        if entry.get("name") == step_name
    )


@pytest.mark.parametrize(("job", "step_name"), _GENERAL_STEPS)
def test_general_shards_deselect_every_timing_witness(
    *, job: str, step_name: str
) -> None:
    """No general shard may run a timing case next to four xdist workers.

    A bar measured against three other workers' load is a bar measured against
    noise. Every shard of every general lane deselects all three, so no shard
    of any split can pick one up.
    """
    (argv,) = cpu_suite_invocation_argvs(_step(job=job, step_name=step_name)["run"])
    for node in _TIMING_NODES:
        assert argv.count(f"--deselect={node}") == 1, node


def test_the_timing_job_runs_all_three_witnesses_once_serially() -> None:
    """One serial, uncontended invocation carries exactly the three witnesses."""
    step = _step(job=_TIMING_JOB, step_name=_TIMING_STEP)
    run = step["run"]
    argvs = cpu_suite_invocation_argvs(run)
    assert len(argvs) == 1
    serial = argvs[0]
    assert tuple(argument for argument in serial if argument.startswith("tests/")) == (
        _TIMING_NODES
    )
    assert serial[serial.index("-n") + 1] == "0"
    assert "-v" in serial
    assert "--precision=$PYLCM_CI_PRECISION" in serial
    assert "--policy-child" in serial
    assert "--ci-policy=full" in serial
    assert "--hardware-profile=cpu" in serial
    assert not any(
        argument.startswith(("--deselect", "--ignore")) for argument in serial
    )
    assert "-k" not in serial
    assert "-m" not in serial
    reports = [
        arg.removeprefix("--junitxml=")
        for arg in serial
        if arg.startswith("--junitxml=")
    ]
    assert len(reports) == 1
    assert reports[0] == "$pylcm_junit_report"
    normalized = " ".join(run.replace("\\\n", " ").split())
    assert (
        "pixi run -e tests-cpu python -m "
        'tests.ci.check_simulation_timing_report "$pylcm_junit_report"'
    ) in normalized


def test_the_timing_job_covers_every_route_the_general_lanes_do() -> None:
    """Each (OS, precision) route that runs general work also runs the bars.

    Dropping a route here would silently stop measuring that platform's timing
    while its general lane kept passing. macOS is the one deliberate exception:
    the hosted macOS runners never met the measurement's steadiness
    precondition (control-leg relative IQR 0.17 to 0.54 over eight fresh
    batches against a 0.15 ceiling), and a declined row is a failure, so the
    lane could only ever be red. The six timing obligations are measured on
    the three remaining routes; a macOS route may return once a steady runner
    exists, and this test must then be widened, not the bar.
    """
    include = _workflow()["jobs"][_TIMING_JOB]["strategy"]["matrix"]["include"]
    routes = {(entry["os"], entry["precision"]) for entry in include}
    assert routes == {
        ("ubuntu-latest", 64),
        ("windows-latest", 64),
        ("ubuntu-latest", 32),
    }
    assert ("macos-latest", 64) not in routes


def test_the_timing_job_runs_no_other_payload() -> None:
    """The timing job carries exactly one CPU-suite invocation, so it is uncontended."""
    job = _workflow()["jobs"][_TIMING_JOB]
    invocations = [
        argv
        for step in job["steps"]
        for argv in cpu_suite_invocation_argvs(str(step.get("run", "")))
    ]
    assert len(invocations) == 1


def test_slow_solution_shards_keep_their_slow_selector() -> None:
    """The slow shards cannot duplicate the unmarked timing witnesses."""
    step = _step(job="tests-slow-solution", step_name="Run one slow solution shard")
    (argv,) = cpu_suite_invocation_argvs(step["run"])
    assert argv[argv.index("-m") + 1] == "slow"
    assert not any(node in argv for node in _TIMING_NODES)
