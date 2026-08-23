"""Workflow commands activate every structured execution-policy selector."""

from pathlib import Path

import yaml

_WORKFLOWS = Path(".github/workflows")


def _steps(path: Path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        yield from job.get("steps", ())


def test_every_cpu_suite_invocation_activates_full_machine_policy() -> None:
    """CPU suite children enforce capability, tier, and isolation declarations."""
    found = 0
    for step in _steps(_WORKFLOWS / "cpu.yml"):
        command = str(step.get("run", ""))
        count = command.count("pixi run -e tests-cpu pytest")
        count += command.count("pixi run -e tests-cpu tests")
        if count == 0:
            continue
        found += count
        assert command.count("--policy-child") == count
        assert command.count("--ci-policy=full") == count
        assert command.count("--hardware-profile=cpu") == count
    assert found > 0


def test_gpu_suite_invocations_use_the_bounded_policy_launcher() -> None:
    """Both GPU workflows activate their declared bounded policy."""
    for name in ("gpu32.yml", "gpu64.yml"):
        commands = [str(step.get("run", "")) for step in _steps(_WORKFLOWS / name)]
        policy_commands = [command for command in commands if " test --" in command]
        assert len(policy_commands) == 1
        assert "--ci-policy=pr" in policy_commands[0]
        assert "--hardware-profile=gpu-small" in policy_commands[0]
