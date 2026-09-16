"""Contracts between the CPU workflow and supported platform capabilities."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from tests.ci import ci_workloads
from tests.ci.cpu_suite_invocations import (
    EIGHT_DEVICE_TEST_FILES,
    FOUR_DEVICE_TEST_FILES,
    carries_policy_activation_flags,
    cpu_suite_invocation_argvs,
    four_device_pinning_test_files,
    ignore_implicit_eight_device_collection,
)

_REPO_ROOT = Path(__file__).parents[2]


def _workflow() -> dict:
    return yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )


def _all_cpu_suite_invocations() -> list[tuple[str, str, list[str]]]:
    """Return `(job, step name, argv)` for every CPU-suite pytest invocation.

    Scanning the whole workflow, rather than two named steps, is what lets the
    topology lanes live in their own jobs: a lane moved from the general chain
    into `tests-topology-*` is still found here, while a lane that vanished
    altogether is not, and the singleton assertions below fail.
    """
    return [
        (job_name, step.get("name", ""), argv)
        for job_name, job in _workflow()["jobs"].items()
        for step in job.get("steps", ())
        if isinstance(step.get("run"), str)
        for argv in cpu_suite_invocation_argvs(step["run"])
    ]


# Each four-CPU-device file now has exactly one invocation in the whole
# workflow: the topology job runs each file in its own step and reaches both
# precisions through its matrix, instead of the two precision legs each
# carrying their own copy of the chain.
_FOUR_DEVICE_INVOCATION_CASES = tuple(FOUR_DEVICE_TEST_FILES)


def test_windows_cpu_suite_has_no_missing_kernel_skip_policy():
    """Windows builds the native kernel, so its CPU suite needs no skip workaround."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    matrix_entries = workflow["jobs"]["tests"]["strategy"]["matrix"]["include"]
    windows = next(entry for entry in matrix_entries if entry["os"] == "windows-latest")

    arguments = shlex.split(windows.get("pytest_extra", ""))
    obsolete_options = {
        "--exact-kernel-skip-inventory",
        "--expected-exact-kernel-skip-inventory",
        "--max-total-skips",
    }

    assert not obsolete_options.intersection(
        argument.partition("=")[0] for argument in arguments
    )


def test_every_four_device_test_file_is_registered():
    """Every test file pinning several CPU devices is in `FOUR_DEVICE_TEST_FILES`.

    Registration is what gives a file its own-process invocation in both
    precision legs; the tests below check those invocations, but only for files
    the registry names. A file that pins the topology and is left out of the
    registry runs inside the shared `tests` invocation instead, where the
    backend is already initialised, so its pin raises and every test in it
    skips without failing anything.
    """
    assert set(four_device_pinning_test_files(tests_root=_REPO_ROOT / "tests")) == set(
        FOUR_DEVICE_TEST_FILES
    )


def test_no_lane_uploads_coverage_to_codecov_before_the_combine_stage():
    """Only the dedicated `coverage` job talks to Codecov.

    The suite used to accumulate coverage with `--cov-append` along one `&&`
    chain and publish from the chain's last link. Independent jobs cannot
    append to each other's data file, so that contract is replaced by an
    explicit one: every lane uploads its own complete XML as an artifact and a
    single combine stage checks the full contributor set before publishing. A
    lane uploading on its own would republish a partial picture, which reads as
    a coverage drop rather than as a missing lane.
    """
    uploaders = [
        job_name
        for job_name, job in _workflow()["jobs"].items()
        for step in job.get("steps", ())
        if str(step.get("uses", "")).startswith("codecov/codecov-action")
    ]

    assert uploaders == ["coverage"], (
        f"jobs uploading to Codecov: {uploaders}; expected only the combine stage"
    )


def test_every_recorded_coverage_contributor_uploads_its_artifact():
    """Each lane the manifest records as contributing coverage uploads one.

    This is the replacement completeness contract. The combine stage refuses to
    publish unless every recorded contributor is present
    (`tests/ci/check_coverage_manifest.py`); this test checks the other half --
    that the workflow actually contains a step producing each of them, so the
    manifest cannot name a lane that no job would ever deliver.
    """
    expected = set(ci_workloads.coverage_contributors())
    uploaded = {
        (step.get("with", {}) or {}).get("name", "")
        for job in _workflow()["jobs"].values()
        for step in job.get("steps", ())
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    }
    # Matrix lanes publish a templated artifact name; compare on the literal
    # prefix before the first `${{`.
    literal = {name.split("${{")[0] for name in uploaded}
    missing = sorted(
        name
        for name in expected
        if name not in uploaded
        and not any(name.startswith(prefix) for prefix in literal if prefix)
    )

    assert not missing, f"recorded coverage contributors with no upload step: {missing}"


def test_the_combine_stage_checks_the_manifest_before_uploading():
    """The `coverage` job runs the contributor check ahead of the Codecov step."""
    steps = _workflow()["jobs"]["coverage"]["steps"]
    check = next(
        index
        for index, step in enumerate(steps)
        if "check_coverage_manifest" in str(step.get("run", ""))
    )
    upload = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("codecov/codecov-action")
    )

    assert check < upload


def _step_run_block(*, job: str, step_name: str) -> str:
    """Return the `run:` script of one named step in one workflow job."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"][job]["steps"]
    step = next(entry for entry in steps if entry.get("name") == step_name)
    return step["run"]


def _pytest_invocation_argvs(*, job: str, step_name: str) -> list[list[str]]:
    """Return the argv of every CPU-suite pytest invocation in one step."""
    return cpu_suite_invocation_argvs(_step_run_block(job=job, step_name=step_name))


def _invocations_naming(*, four_device_file: str) -> list[tuple[str, str, list[str]]]:
    """Return every CPU-suite invocation in the workflow naming `four_device_file`."""
    return [
        case for case in _all_cpu_suite_invocations() if four_device_file in case[2]
    ]


def _invocations_naming_any(*, test_file: str) -> list[tuple[str, str, list[str]]]:
    """Return every CPU-suite invocation in the workflow naming `test_file`."""
    return [case for case in _all_cpu_suite_invocations() if test_file in case[2]]


def _sole_invocation(*, four_device_file: str) -> list[str]:
    """Return the one invocation argv naming `four_device_file`.

    `test_four_device_file_appears_in_exactly_one_invocation` is the test that
    names and asserts this singleton precondition; every other property test
    below reuses this helper to reach the one invocation it inspects.
    """
    return _invocations_naming(four_device_file=four_device_file)[0][2]


@pytest.mark.parametrize("four_device_file", _FOUR_DEVICE_INVOCATION_CASES)
def test_four_device_file_appears_in_exactly_one_invocation(*, four_device_file: str):
    """Each four-CPU-device test file is named by exactly one pytest invocation.

    Such a file pins a four-CPU-device topology at import, a pin that depends on
    running alone in its process; naming the file from zero or from more than
    one invocation means it either never runs or no longer runs alone.
    """
    matches = _invocations_naming(four_device_file=four_device_file)
    assert len(matches) == 1, (
        f"expected exactly one pytest invocation naming {four_device_file}, "
        f"found {len(matches)} in {[(job, step) for job, step, _ in matches]}"
    )


@pytest.mark.parametrize("four_device_file", _FOUR_DEVICE_INVOCATION_CASES)
def test_four_device_file_runs_without_other_test_paths(*, four_device_file: str):
    """A four-CPU-device test file's invocation names no other test path.

    Sharing the invocation with `tests` or another `tests/...` target would
    fold the file back into a multi-file process, defeating the import-time
    device-count pin that assumes it runs alone.
    """
    argv = _sole_invocation(four_device_file=four_device_file)
    other_targets = [
        argument
        for argument in argv
        if (argument == "tests" or argument.startswith("tests/"))
        and argument != four_device_file
    ]
    assert not other_targets, (
        f"{four_device_file} shares its invocation with {other_targets}"
    )


@pytest.mark.parametrize("four_device_file", _FOUR_DEVICE_INVOCATION_CASES)
def test_four_device_file_invocation_passes_the_worker_count_flag(
    *, four_device_file: str
):
    """A four-CPU-device test file's invocation states its worker count explicitly.

    An implicit worker count would leave the invocation's process-isolation
    guarantee undeclared; the sibling test then checks the count itself.
    """
    argv = _sole_invocation(four_device_file=four_device_file)
    assert "-n" in argv, f"{four_device_file}'s invocation is missing -n"


@pytest.mark.parametrize("four_device_file", _FOUR_DEVICE_INVOCATION_CASES)
def test_four_device_file_runs_at_worker_count_zero(*, four_device_file: str):
    """A four-CPU-device test file's invocation runs at `-n 0`, its own process.

    Any other worker count would distribute the file's tests across xdist
    workers that fork before the file's own import-time device-count pin runs,
    so the pin would apply to at most one worker and the rest would skip.
    """
    argv = _sole_invocation(four_device_file=four_device_file)
    worker_count = argv[argv.index("-n") + 1]
    assert worker_count == "0", (
        f"{four_device_file} runs at -n {worker_count!r} instead of its own "
        "process (-n 0)"
    )


@pytest.mark.parametrize("four_device_file", _FOUR_DEVICE_INVOCATION_CASES)
def test_four_device_file_invocation_omits_policy_activation_flags(
    *, four_device_file: str
):
    """A four-CPU-device test file's invocation never activates the CI policy launcher.

    `--ci-policy` or `--full-suite` drives `pytest_policy.configure()`, which
    resolves an explicit `--hardware-profile` by querying `jax.default_backend()`
    during `pytest_configure` — before pytest imports any test module. That
    query initialises the JAX backend ahead of this file's own import-time
    four-CPU-device pin, so the pin sees an already-initialised backend, never
    applies, and every test in the file silently skips.
    """
    argv = _sole_invocation(four_device_file=four_device_file)
    assert not carries_policy_activation_flags(argv), (
        f"{four_device_file}'s invocation carries a CI policy activation flag, "
        "which silently skips every test in the file"
    )


@pytest.mark.parametrize("test_file", EIGHT_DEVICE_TEST_FILES)
def test_eight_device_witness_has_one_fresh_full_policy_invocation(
    *, test_file: str
) -> None:
    """Environment topology precedes policy initialization; no native skip passes.

    One invocation for the whole workflow now, not one per precision leg: the
    topology job reaches both precisions through its matrix and passes the
    precision in `$PYLCM_CI_PRECISION`, so the argv is precision-agnostic and
    the JUnit name interpolates it.
    """
    matches = _invocations_naming_any(test_file=test_file)
    assert len(matches) == 1, (
        f"expected exactly one invocation naming {test_file}, found "
        f"{[(job, step) for job, step, _ in matches]}"
    )
    job_name, step_name, argv = matches[0]
    run = _step_run_block(job=job_name, step_name=step_name)
    assert [arg for arg in argv if arg == "tests" or arg.startswith("tests/")] == [
        test_file
    ]
    assert argv[argv.index("-n") + 1] == "0"
    assert argv.count("JAX_NUM_CPU_DEVICES=8") == 1
    assert argv.count("JAX_PLATFORMS=cpu") == 1
    assert (
        "XLA_FLAGS=${XLA_FLAGS:+$XLA_FLAGS }--xla_force_host_platform_device_count=8"
        in argv
    )
    assert argv.index("JAX_NUM_CPU_DEVICES=8") < argv.index("pixi")
    for flag in (
        "--policy-child",
        "--ci-policy=full",
        "--hardware-profile=cpu",
        "--precision=$PYLCM_CI_PRECISION",
        "-v",
    ):
        assert argv.count(flag) == 1, flag
    assert "-k" not in argv
    assert "-m" not in argv
    assert not any(arg.startswith(("--ignore", "--deselect")) for arg in argv)
    reports = [
        arg.removeprefix("--junitxml=") for arg in argv if arg.startswith("--junitxml=")
    ]
    assert len(reports) == 1
    # The report path is bound once to a shell variable and reused, so the
    # invocation and its population guard cannot drift onto different files.
    assert reports[0] == "$pylcm_junit"
    normalized = " ".join(run.replace("\\\n", " ").split())
    assigned = [
        line.strip().removeprefix("pylcm_junit=")
        for line in run.splitlines()
        if line.strip().startswith("pylcm_junit=")
    ]
    assert len(assigned) == 1, assigned
    assert assigned[0].startswith("reports/junit-cpu-fp$PYLCM_CI_PRECISION-")
    assert assigned[0].endswith(".xml")
    assert 'check_population "$pylcm_junit" no-skips' in normalized
    assert "EVERYTHING SKIPPED" in run
    assert '"${2:-}" = "no-skips"' in run
    assert '"$skipped" -ne 0' in run


def _assert_unchained(*, step: dict) -> None:
    """Fail unless this step's single pytest invocation stands on its own.

    `&&` alone is not the test: the shared `check_population` helper legitimately
    uses it inside its own body. What must not appear is a *pytest* invocation
    joined to another command, which is what would restore the chain this batch
    removed -- and with it the behaviour where the first failing lane prevents
    every later lane from running at all.
    """
    normalized = " ".join(str(step["run"]).replace("\\\n", " ").split())
    for chained in ("&& pixi run", "&& JAX_NUM_CPU_DEVICES", "&& XLA_FLAGS"):
        assert chained not in normalized, (
            f"{step.get('name')} chains its pytest invocation ({chained!r})"
        )


def test_every_eight_device_lane_lives_in_its_own_unchained_step() -> None:
    """No two eight-device witnesses share a step, and none is `&&`-chained.

    Equal device count is not evidence about cache, environment or serial-group
    compatibility, so merging two of these files into one process is not
    authorized. One file per step also means one exit status and one report per
    witness instead of a chain that stops at its first failure.
    """
    job = _workflow()["jobs"]["tests-topology-eight-device"]
    payload_steps = [
        step
        for step in job["steps"]
        if cpu_suite_invocation_argvs(str(step.get("run", "")))
    ]
    assert len(payload_steps) == len(EIGHT_DEVICE_TEST_FILES)
    for step in payload_steps:
        argvs = cpu_suite_invocation_argvs(step["run"])
        assert len(argvs) == 1, f"{step.get('name')} runs {len(argvs)} invocations"
        _assert_unchained(step=step)


def test_every_topology_lane_writes_its_own_junit_report() -> None:
    """No two topology lanes share a JUnit filename.

    Each lane is a separate process with its own population guard, so two lanes
    writing the same path would leave the second overwriting the first: the
    guard would pass twice while only one lane's results survived to be read
    back. The names are historical short forms rather than the file's own slug,
    so uniqueness -- not a name match -- is what this checks.
    """
    for job_name in ("tests-topology-four-device", "tests-topology-eight-device"):
        assigned = [
            line.strip().removeprefix("pylcm_junit=")
            for step in _workflow()["jobs"][job_name]["steps"]
            for line in str(step.get("run", "")).splitlines()
            if line.strip().startswith("pylcm_junit=")
        ]
        assert assigned, job_name
        assert len(assigned) == len(set(assigned)), (
            f"{job_name} reuses a JUnit path: {sorted(assigned)}"
        )


def test_every_four_device_lane_lives_in_its_own_unchained_step() -> None:
    """Same isolation contract for the import-time four-device pins."""
    job = _workflow()["jobs"]["tests-topology-four-device"]
    payload_steps = [
        step
        for step in job["steps"]
        if cpu_suite_invocation_argvs(str(step.get("run", "")))
    ]
    assert len(payload_steps) == len(FOUR_DEVICE_TEST_FILES)
    for step in payload_steps:
        argvs = cpu_suite_invocation_argvs(step["run"])
        assert len(argvs) == 1, f"{step.get('name')} runs {len(argvs)} invocations"
        _assert_unchained(step=step)


@pytest.mark.parametrize("test_file", EIGHT_DEVICE_TEST_FILES)
def test_eight_device_registry_excludes_only_implicit_collection(
    test_file: str,
) -> None:
    path = _REPO_ROOT / test_file
    for args in (("tests",), ("tests/simulation",), ("tests/test_model.py",)):
        assert ignore_implicit_eight_device_collection(
            collection_path=path, root=_REPO_ROOT, invocation_args=args
        )
    for argument in (test_file, str(path), f"{test_file}::test_witness"):
        assert not ignore_implicit_eight_device_collection(
            collection_path=path, root=_REPO_ROOT, invocation_args=(argument,)
        )
    assert not ignore_implicit_eight_device_collection(
        collection_path=_REPO_ROOT / "tests/test_model.py",
        root=_REPO_ROOT,
        invocation_args=("tests",),
    )


def _general_shard_counts() -> dict[str, int]:
    """Return the shard count `cpu.yml` runs for each general leg."""
    jobs = _workflow()["jobs"]
    counts = {
        entry["leg"]: int(entry["shards"])
        for entry in jobs["tests"]["strategy"]["matrix"]["include"]
    }
    fp32 = jobs["tests-fp32"]
    counts[fp32["env"]["PYLCM_CI_GENERAL_LEG"]] = int(
        fp32["env"]["PYLCM_CI_GENERAL_SHARDS"]
    )
    return counts


def test_the_workflow_runs_exactly_the_shard_counts_the_manifest_records():
    """`cpu.yml`'s matrices and the manifest's `shard_layout` agree.

    The two halves are written apart: the manifest decides which files a shard
    of a leg selects, and the workflow decides how many shards of that leg get
    a runner. A leg whose workflow count is lower than its manifest count
    silently drops every file the missing shards own -- each remaining job
    still passes, and only the count reveals the gap -- while a higher one
    schedules a job whose shard index the sharder rejects.
    """
    layout = ci_workloads.shard_layout()
    assert _general_shard_counts() == {
        leg: cfg["shards"] for leg, cfg in layout["general"].items()
    }

    solution = {
        (int(entry["precision"]), int(entry["shard"])): int(entry["shards"])
        for entry in _workflow()["jobs"]["tests-slow-solution"]["strategy"]["matrix"][
            "include"
        ]
    }
    for leg, cfg in layout["solution"].items():
        precision = 64 if "64" in leg else 32
        assert {shard for prec, shard in solution if prec == precision} == set(
            range(1, cfg["shards"] + 1)
        ), leg
        assert {
            count for (prec, _), count in solution.items() if prec == precision
        } == {cfg["shards"]}, leg


def test_every_general_leg_matrix_entry_declares_its_own_shard_count():
    """No general matrix entry inherits a shard count from another leg.

    The legs no longer share one count -- Windows takes more shards than macOS
    for the same universe because its payload and its setup are both larger --
    so `shards` has to travel with the entry that uses it, not with the job.
    """
    for entry in _workflow()["jobs"]["tests"]["strategy"]["matrix"]["include"]:
        assert {"os", "leg", "shard", "shards"} <= set(entry), entry
        assert 1 <= int(entry["shard"]) <= int(entry["shards"])
