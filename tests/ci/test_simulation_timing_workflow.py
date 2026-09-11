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
)
_ROUTES = (
    ("tests", "Run pytest", 64),
    ("tests", "Run pytest and collect coverage", 64),
    ("tests-fp32", "Run pytest at fp32", 32),
)


@pytest.mark.parametrize(("job", "step_name", "precision"), _ROUTES)
def test_each_cpu_route_runs_timing_cases_once_after_parallel_work(
    *, job: str, step_name: str, precision: int
) -> None:
    """Route both timing families to one serial process with complete reports."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    step = next(
        entry
        for entry in workflow["jobs"][job]["steps"]
        if entry.get("name") == step_name
    )
    run = step["run"]
    argvs = cpu_suite_invocation_argvs(run)
    parallel = argvs[0]
    assert all(parallel.count(f"--deselect={node}") == 1 for node in _TIMING_NODES)
    matches = [argv for argv in argvs if any(node in argv for node in _TIMING_NODES)]
    assert len(matches) == 1
    serial = matches[0]
    assert serial == argvs[-1]
    assert tuple(argument for argument in serial if argument.startswith("tests/")) == (
        _TIMING_NODES
    )
    assert serial[serial.index("-n") + 1] == "0"
    assert "-v" in serial
    assert f"--precision={precision}" in serial
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
    assert reports[0].startswith("reports/junit-cpu-")
    assert reports[0].endswith("simulation-timing.xml")
    normalized = run.replace("\\\n", " ")
    assert (
        "&& pixi run -e tests-cpu python -m "
        f"tests.ci.check_simulation_timing_report {reports[0]}"
        in " ".join(normalized.split())
    )
    if step_name == "Run pytest and collect coverage":
        assert "--cov=./" in serial
        assert "--cov-append" in serial
        assert "--cov-report=xml" in serial
        assert all("--cov-report=xml" not in argv for argv in argvs[:-1])
    else:
        assert not any(arg.startswith("--cov") for arg in serial)


def test_slow_solution_shards_keep_their_slow_selector() -> None:
    """The slow shards cannot duplicate the unmarked timing witnesses."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    step = next(
        entry
        for entry in workflow["jobs"]["tests-slow-solution"]["steps"]
        if entry.get("name") == "Run one slow solution shard"
    )
    (argv,) = cpu_suite_invocation_argvs(step["run"])
    assert argv[argv.index("-m") + 1] == "slow"
    assert not any(node in argv for node in _TIMING_NODES)
