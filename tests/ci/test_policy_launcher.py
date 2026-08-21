"""Contract for the public policy launcher and its concrete pytest children."""

from tests.ci.policy_launcher import (
    child_command,
    default_environment,
    isolation_legs,
    precision_legs,
)


def test_auto_precision_expands_to_separate_processes():
    """The launcher never switches JAX precision inside one pytest process."""
    assert precision_legs("auto") == ("64", "32")


def test_process_isolation_expands_to_separate_children():
    """Shared and fresh contracts never execute in the same pytest process."""
    assert isolation_legs() == ("shared", "fresh")


def test_default_launcher_uses_cpu_but_reuses_an_explicit_cuda_environment():
    """The unqualified public task is unambiguous and GPU tasks stay on GPU."""
    assert default_environment(None) == "tests-cpu"
    assert default_environment("default") == "tests-cpu"
    assert default_environment("tests-cuda12") == "tests-cuda12"


def test_full_suite_child_keeps_the_current_machine_boundary(tmp_path):
    """A full child receives one precision, one profile, and a private report."""
    command = child_command(
        environment="tests-cuda12",
        policy="full",
        profile="gpu-small",
        precision="32",
        isolation="fresh",
        report_dir=tmp_path,
        pytest_args=("tests",),
    )

    assert command[:4] == ("pixi", "run", "-e", "tests-cuda12")
    assert "--policy-child" in command
    assert "--ci-policy=full" in command
    assert "--hardware-profile=gpu-small" in command
    assert "--precision=32" in command
    assert "--isolation-mode=fresh" in command
    assert (
        f"--selection-report={tmp_path / 'selection-gpu-small-fp32-fresh.json'}"
        in command
    )
    assert f"--junitxml={tmp_path / 'junit-gpu-small-fp32-fresh.xml'}" in command
