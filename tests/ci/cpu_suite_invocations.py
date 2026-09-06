"""Parse `cpu.yml`'s pytest invocations and identify the four-device exemption.

`tests/test_distributed.py` and `tests/execution/test_transfer_catalogue.py` pin a
four-CPU-device topology at import; the pin only takes effect in a process that has
not already touched a JAX backend. CI's execution-policy launcher
(`--ci-policy`/`--full-suite`) resolves an explicit `--hardware-profile` by querying
`jax.default_backend()` during `pytest_configure` -- before pytest imports any test
module -- so activating it ahead of either file's own invocation would initialise
the backend first and silently skip every test in the file. Both files' own-process
invocations therefore carry `--policy-child --hardware-profile=cpu` without policy
activation.

This module is the single place that parses a workflow step's pytest invocations and
identifies which one is such an exemption, so every contract test that needs the
exemption agrees with the others by construction instead of by a separately
maintained list.
"""

import re
import shlex

FOUR_DEVICE_TEST_FILES = (
    "tests/test_distributed.py",
    "tests/execution/test_transfer_catalogue.py",
)

# Other `pixi run -e tests-cpu ...` commands in `cpu.yml` (the exact-kernel capability
# matrix, the native-payload probe, cache pruning) invoke a different entry point.
_CPU_SUITE_MARKERS = ("pixi run -e tests-cpu pytest", "pixi run -e tests-cpu tests")


def cpu_suite_invocation_argvs(run_text: str) -> list[list[str]]:
    """Return the argv of every CPU-suite pytest invocation in one `run:` block.

    A `run:` block chains commands with backslash line continuations and `&&`.
    Joining continuations collapses each chained command onto one line, so
    splitting on `&&` recovers the individual commands the block actually runs.
    """
    normalized = run_text.replace("\\\n", " ")
    commands = [
        segment.strip()
        for line in normalized.splitlines()
        for segment in re.split(r"\s&&\s", line)
        if segment.strip()
    ]
    return [
        shlex.split(command)
        for command in commands
        if any(marker in command for marker in _CPU_SUITE_MARKERS)
    ]


def is_isolated_four_device_invocation(argv: list[str]) -> bool:
    """Return whether `argv` is one four-CPU-device file's own-process invocation.

    Identified structurally rather than by a file-name whitelist: names exactly
    one of the two four-CPU-device test files, no other test path alongside it,
    and runs at `-n 0`.
    """
    four_device_targets = [
        argument for argument in argv if argument in FOUR_DEVICE_TEST_FILES
    ]
    if len(four_device_targets) != 1:
        return False
    other_targets = [
        argument
        for argument in argv
        if (argument == "tests" or argument.startswith("tests/"))
        and argument not in four_device_targets
    ]
    if other_targets:
        return False
    if "-n" not in argv:
        return False
    return argv[argv.index("-n") + 1] == "0"


def carries_policy_activation_flags(argv: list[str]) -> bool:
    """Return whether `argv` passes `--ci-policy` or `--full-suite`.

    Either flag drives `pytest_policy.configure()`, which resolves an explicit
    `--hardware-profile` by querying `jax.default_backend()` during
    `pytest_configure` -- before pytest imports any test module. An isolated
    four-CPU-device invocation carrying either flag would have its backend
    initialised ahead of its own import-time topology pin, so the pin would
    never apply and every test in the file would silently skip.
    """
    return any(
        argument == "--full-suite" or argument.partition("=")[0] == "--ci-policy"
        for argument in argv
    )
