"""Workflow commands activate every structured execution-policy selector.

Every CPU-suite invocation activates the full machine policy except the two
isolated four-CPU-device invocations (`tests/test_distributed.py`,
`tests/execution/test_transfer_catalogue.py`), which carry
`--policy-child --hardware-profile=cpu` without policy activation instead.
That an isolated invocation carries no `--ci-policy`/`--full-suite` is asserted
in `tests/ci/test_cpu_workflow_contract.py`
(`test_four_device_file_invocation_omits_policy_activation_flags`), which this
module does not repeat; both files identify the exemption through the same
`is_isolated_four_device_invocation` predicate, so they agree by construction.
"""

from pathlib import Path

import pytest
import yaml

from tests.ci.cpu_suite_invocations import (
    cpu_suite_invocation_argvs,
    is_isolated_four_device_invocation,
)

_WORKFLOWS = Path(".github/workflows")


def _steps(path: Path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", ()):
            yield job_name, step.get("name", ""), step


def _cpu_suite_invocation_cases() -> list[tuple[str, str, list[str]]]:
    """Return `(job, step name, argv)` for every CPU-suite pytest invocation."""
    return [
        (job_name, step_name, argv)
        for job_name, step_name, step in _steps(_WORKFLOWS / "cpu.yml")
        for argv in cpu_suite_invocation_argvs(str(step.get("run", "")))
    ]


def _case_id(case: tuple[str, str, list[str]]) -> str:
    """A readable pytest id naming the job, step, and target of one invocation."""
    job_name, step_name, argv = case
    targets = [
        argument
        for argument in argv
        if argument == "tests" or argument.startswith("tests/")
    ]
    return f"{job_name}/{step_name}/{' '.join(targets) or argv[-1]}"


_CPU_SUITE_INVOCATIONS = _cpu_suite_invocation_cases()
_FULL_POLICY_CASES = [
    case
    for case in _CPU_SUITE_INVOCATIONS
    if not is_isolated_four_device_invocation(case[2])
]
_ISOLATED_CASES = [
    case
    for case in _CPU_SUITE_INVOCATIONS
    if is_isolated_four_device_invocation(case[2])
]


def test_full_policy_cpu_suite_invocations_exist() -> None:
    """At least one CPU-suite invocation activates the full machine policy.

    Guards the two property tests below against vacuously passing over an
    empty parametrize list — if every invocation were wrongly classified as
    the isolated four-device exemption, those tests would silently check
    nothing.
    """
    assert len(_FULL_POLICY_CASES) > 0


def test_isolated_four_device_invocations_exist() -> None:
    """At least one CPU-suite invocation is the isolated four-device exemption.

    Guards the exemption's own property tests against vacuously passing over
    an empty parametrize list.
    """
    assert len(_ISOLATED_CASES) > 0


@pytest.mark.parametrize("case", _FULL_POLICY_CASES, ids=_case_id)
def test_full_policy_invocation_carries_policy_child(
    case: tuple[str, str, list[str]],
) -> None:
    """Every non-exempt CPU-suite invocation declares itself a policy child.

    `--policy-child` is what lets `pytest_policy.configure()` accept
    `--ci-policy=full`, which is otherwise private to the policy launcher.
    """
    _, _, argv = case
    assert argv.count("--policy-child") == 1


@pytest.mark.parametrize("case", _FULL_POLICY_CASES, ids=_case_id)
def test_full_policy_invocation_activates_the_full_policy(
    case: tuple[str, str, list[str]],
) -> None:
    """Every non-exempt CPU-suite invocation runs at the full execution-policy tier.

    A CI leg that ran a bounded tier instead would silently drop whichever
    tests that tier deselects, without any test failure naming what was lost.
    """
    _, _, argv = case
    assert argv.count("--ci-policy=full") == 1


@pytest.mark.parametrize("case", _FULL_POLICY_CASES, ids=_case_id)
def test_full_policy_invocation_declares_the_cpu_hardware_profile(
    case: tuple[str, str, list[str]],
) -> None:
    """Every non-exempt CPU-suite invocation declares the CPU hardware profile.

    An unset or wrong profile would let `pytest_policy` classify tests against
    the wrong backend's capability declarations.
    """
    _, _, argv = case
    assert argv.count("--hardware-profile=cpu") == 1


@pytest.mark.parametrize("case", _ISOLATED_CASES, ids=_case_id)
def test_isolated_four_device_invocation_carries_policy_child(
    case: tuple[str, str, list[str]],
) -> None:
    """An isolated four-device invocation still declares itself a policy child.

    `--policy-child` alone (without `--ci-policy`/`--full-suite`) is a no-op in
    `pytest_policy.configure()`, so declaring it costs nothing while keeping
    the invocation visibly a recognised CI child rather than an ad hoc one.
    """
    _, _, argv = case
    assert argv.count("--policy-child") == 1


@pytest.mark.parametrize("case", _ISOLATED_CASES, ids=_case_id)
def test_isolated_four_device_invocation_declares_the_cpu_hardware_profile(
    case: tuple[str, str, list[str]],
) -> None:
    """An isolated four-device invocation still declares the CPU hardware profile.

    Declaring `--hardware-profile=cpu` here is likewise inert without a policy
    to drive, but keeps the invocation's stated platform truthful either way.
    """
    _, _, argv = case
    assert argv.count("--hardware-profile=cpu") == 1


@pytest.mark.parametrize(
    ("workflow", "suite_name", "suite_command", "diagnostic_names"),
    [
        (
            "gpu32.yml",
            "Run the bounded GPU policy",
            (
                "pixi run -e tests-cuda12 test -- "
                "--ci-policy=pr --hardware-profile=gpu-small --precision=auto "
                "--environment=tests-cuda12 --report-dir=reports/policy "
                "--log-cli-level=INFO tests"
            ),
            ("Check restored lookup in both GPU precisions",),
        ),
        (
            "gpu64.yml",
            "Run the bounded fp64 GPU policy",
            (
                "pixi run -e tests-cuda12 test -- "
                "--ci-policy=pr --hardware-profile=gpu-small --precision=64 "
                "--environment=tests-cuda12 --report-dir=reports/policy-fp64 tests"
            ),
            (),
        ),
    ],
)
def test_gpu_suite_invocations_use_the_bounded_policy_launcher(
    *,
    workflow: str,
    suite_name: str,
    suite_command: str,
    diagnostic_names: tuple[str, ...],
) -> None:
    """Keep one declared suite and only the explicitly named scoped diagnostic."""
    policy_steps = [
        (name, step["run"])
        for _, name, step in _steps(_WORKFLOWS / workflow)
        if " test --" in str(step.get("run", ""))
    ]
    assert tuple(name for name, _ in policy_steps) == (suite_name, *diagnostic_names)
    assert policy_steps[0][1] == suite_command


def test_gpu_lookup_diagnostic_has_two_bounded_literal_precision_legs() -> None:
    """Retain both leg statuses without broadening the exact lookup selection."""
    (step,) = [
        step
        for _, name, step in _steps(_WORKFLOWS / "gpu32.yml")
        if name == "Check restored lookup in both GPU precisions"
    ]
    assert step == {
        "name": "Check restored lookup in both GPU precisions",
        "if": "${{ !cancelled() && steps.native_payload.outcome == 'success' }}",
        "shell": "bash",
        "run": r"""mkdir -p reports/lookup-fp64 reports/lookup-fp32
node='tests/simulation/test_policy_lookup.py::'
node+='test_restored_budgeted_lookup_has_analytic_last_decision_policy'
first_rc=0
set +e
for precision in 64 32; do
  report_dir="reports/lookup-fp${precision}"
  timeout --signal=TERM --kill-after=30s 600s \
    pixi run --locked -e tests-cuda12 test -- \
    --ci-policy=pr --hardware-profile=gpu-small \
    --precision="$precision" --environment=tests-cuda12 \
    --report-dir="$report_dir" --log-cli-level=INFO \
    "$node" \
    > "$report_dir/runner.log" 2>&1
  leg_rc=$?
  printf '%s\n' "$leg_rc" > "$report_dir/runner.rc"
  cat "$report_dir/runner.log"
  if test "$first_rc" -eq 0 && test "$leg_rc" -ne 0; then
    first_rc=$leg_rc
  fi
done
exit "$first_rc"
""",
    }
