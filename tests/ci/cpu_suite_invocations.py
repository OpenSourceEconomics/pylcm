"""Parse `cpu.yml`'s pytest invocations and identify the four-device exemption.

The files listed in `FOUR_DEVICE_TEST_FILES` pin a four-CPU-device topology at
import; the pin only takes effect in a process that has not already touched a JAX
backend. CI's execution-policy launcher
(`--ci-policy`/`--full-suite`) resolves an explicit `--hardware-profile` by querying
`jax.default_backend()` during `pytest_configure` — before pytest imports any test
module — so activating it ahead of any such file's own invocation would initialise
the backend first and silently skip every test in the file. Each such file's
own-process invocation therefore carries `--policy-child --hardware-profile=cpu`
without policy activation.

This module is the single place that parses a workflow step's pytest invocations and
identifies which one is such an exemption, so every contract test that needs the
exemption agrees with the others by construction instead of by a separately
maintained list. It also discovers which test files pin such a topology, so the
registry below is checked against the suite rather than trusted.
"""

import ast
import re
import shlex
from pathlib import Path

FOUR_DEVICE_TEST_FILES = (
    "tests/test_distributed.py",
    "tests/execution/test_transfer_catalogue.py",
    "tests/test_distributed_placement.py",
    "tests/test_distributed_lifetime.py",
    "tests/test_distributed_template_placement.py",
    "tests/test_distributed_simulation_values.py",
    "tests/test_distributed_simulation_value_reads.py",
    "tests/test_distributed_solve_readiness.py",
    "tests/test_distributed_eager_weak_inputs.py",
    "tests/simulation/test_operand_placement.py",
    "tests/simulation/test_host_operations.py",
    "tests/test_distributed_taste_stream.py",
    "tests/test_distributed_entry_allocations.py",
)

#: The configuration option a multi-device test file pins when it is imported.
_DEVICE_COUNT_OPTION = "jax_num_cpu_devices"

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
    one of the four-CPU-device test files, no other test path alongside it,
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
    `pytest_configure` — before pytest imports any test module. An isolated
    four-CPU-device invocation carrying either flag would have its backend
    initialised ahead of its own import-time topology pin, so the pin would
    never apply and every test in the file would silently skip.
    """
    return any(
        argument == "--full-suite" or argument.partition("=")[0] == "--ci-policy"
        for argument in argv
    )


def four_device_pinning_test_files(*, tests_root: Path) -> tuple[str, ...]:
    """Return every test file that pins a multi-CPU-device topology at import.

    Read from each module's own statements rather than from its text: the pin is
    a module-level `jax.config.update("jax_num_cpu_devices", n)` with `n` above
    one, wherever the module places it — the registered files put it inside a
    module-level `try`. A file that spawns a child process pinning a topology
    writes that call inside a string literal, and a file pinning one device asks
    for no isolation; neither is a four-device file, and neither is matched.

    Args:
        tests_root: The test suite directory to scan.

    Returns:
        Tuple of repository-relative POSIX paths, ascending.

    """
    repo_root = tests_root.parent
    found = [
        path.relative_to(repo_root).as_posix()
        for path in sorted(tests_root.rglob("test_*.py"))
        if _pins_several_devices(module=ast.parse(path.read_text(encoding="utf-8")))
    ]
    return tuple(found)


def _pins_several_devices(*, module: ast.Module) -> bool:
    """Return whether a parsed module pins more than one CPU device at import."""
    return any(
        _is_multi_device_pin(statement=statement)
        for statement in _import_time_statements(body=module.body)
    )


def _import_time_statements(*, body: list[ast.stmt]) -> list[ast.stmt]:
    """Return the statements a module runs when imported, blocks included.

    `try`, `if` and `with` bodies run at import; a function or class body does
    not, so neither is descended into.
    """
    statements: list[ast.stmt] = []
    for statement in body:
        statements.append(statement)
        if isinstance(statement, ast.Try):
            statements.extend(
                _import_time_statements(
                    body=[
                        *statement.body,
                        *[
                            node
                            for handler in statement.handlers
                            for node in handler.body
                        ],
                        *statement.orelse,
                        *statement.finalbody,
                    ]
                )
            )
        elif isinstance(statement, ast.If | ast.With):
            statements.extend(
                _import_time_statements(body=[*statement.body, *_orelse(statement)])
            )
    return statements


def _orelse(statement: ast.stmt) -> list[ast.stmt]:
    """Return an `if`'s else-branch, or nothing for a statement without one."""
    return statement.orelse if isinstance(statement, ast.If) else []


def _is_multi_device_pin(*, statement: ast.stmt) -> bool:
    """Return whether one statement pins more than one CPU device."""
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    if _dotted_name(node=call.func) != "jax.config.update" or len(call.args) != 2:
        return False
    option, count = call.args
    return (
        isinstance(option, ast.Constant)
        and option.value == _DEVICE_COUNT_OPTION
        and isinstance(count, ast.Constant)
        and isinstance(count.value, int)
        and count.value > 1
    )


def _dotted_name(*, node: ast.expr) -> str:
    """Return an attribute chain as a dotted name, or the empty string."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))
