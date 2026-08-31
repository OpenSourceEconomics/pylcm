from pathlib import Path

import pytest

from tests.ci.keyword_only_convention import (
    find_keyword_only_violations,
    main,
)


def test_reports_free_functions_with_two_positional_parameters(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "def one_parameter(value):\n"
        "    return value\n"
        "\n"
        "def noncompliant(left, right):\n"
        "    return left + right\n"
        "\n"
        "def compliant(*, left, right):\n"
        "    return left + right\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [
        (
            violation.path,
            violation.line,
            violation.qualified_name,
            violation.code,
            violation.positional_parameters,
        )
        for violation in violations
    ] == [(source, 4, "noncompliant", "KWO001", ("left", "right"))]


def test_ignores_implicit_method_receivers(tmp_path: Path) -> None:
    source = tmp_path / "methods.py"
    source.write_text(
        "class Calculator:\n"
        "    def one_parameter(self, value):\n"
        "        return value\n"
        "\n"
        "    def add(self, left, right):\n"
        "        return left + right\n"
        "\n"
        "    @classmethod\n"
        "    def from_pair(cls, left, right):\n"
        "        return cls(left + right)\n"
        "\n"
        "    def compliant(self, *, left, right):\n"
        "        return left + right\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [
        (violation.qualified_name, violation.positional_parameters)
        for violation in violations
    ] == [
        ("Calculator.add", ("left", "right")),
        ("Calculator.from_pair", ("left", "right")),
    ]


def test_covers_mixed_async_and_nested_definitions(tmp_path: Path) -> None:
    source = tmp_path / "nested.py"
    source.write_text(
        "async def outer(value, /, *, option):\n"
        "    def nested(item, *, setting):\n"
        "        return item, setting\n"
        "    return nested(value, setting=option)\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [
        (violation.qualified_name, violation.positional_parameters)
        for violation in violations
    ] == [
        ("outer", ("value",)),
        ("outer.nested", ("item",)),
    ]


def test_audits_library_callback_exemptions(tmp_path: Path) -> None:
    source = tmp_path / "callbacks.py"
    source.write_text(
        "# keyword-only-exempt: library-callback=jax.lax.scan\n"
        "def scan_body(carry, item):\n"
        "    return carry, item\n"
        "\n"
        "# keyword-only-exempt: library-callback=\n"
        "def unexplained(left, right):\n"
        "    return left + right\n"
        "\n"
        "# keyword-only-exempt: library-callback=jax.custom_jvp\n"
        "def stale(*, primals, tangents):\n"
        "    return primals, tangents\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [
        (violation.line, violation.qualified_name, violation.code)
        for violation in violations
    ] == [
        (5, "unexplained", "KWO002"),
        (9, "stale", "KWO003"),
    ]


def test_limits_arithmetic_exemption_to_declared_operators(tmp_path: Path) -> None:
    source = tmp_path / "src" / "_lcm" / "egm" / "upper_envelope" / "double_double.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "'''Keyword-only exemption: arithmetic-only module.'''\n"
        "def two_sum(left, right):\n"
        "    return left + right\n"
    )

    assert find_keyword_only_violations(paths=[source]) == ()

    source.write_text(
        source.read_text()
        + "\ndef unrelated_helper(left, right):\n"
        + "    return left + right\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [(violation.qualified_name, violation.code) for violation in violations] == [
        ("unrelated_helper", "KWO004")
    ]

    source.write_text(
        "'''Keyword-only exemption: arithmetic-only module.'''\n"
        "def two_sum(*, left, right):\n"
        "    return left + right\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [(violation.qualified_name, violation.code) for violation in violations] == [
        ("<module>", "KWO003")
    ]


def test_accepts_repository_arithmetic_module() -> None:
    repository_root = Path(__file__).parents[2]
    source = (
        repository_root / "src" / "_lcm" / "egm" / "upper_envelope" / "double_double.py"
    )

    assert find_keyword_only_violations(paths=[source]) == ()


def test_rejects_orphaned_callback_exemption(tmp_path: Path) -> None:
    source = tmp_path / "orphan.py"
    source.write_text(
        "# keyword-only-exempt: library-callback=jax.lax.scan\nCONSTANT = 1\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [
        (violation.line, violation.qualified_name, violation.code)
        for violation in violations
    ] == [(1, "<module>", "KWO003")]


def test_variadic_forwarders_do_not_trigger_the_named_argument_rule(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variadic.py"
    source.write_text(
        "def forward(value, *args, **kwargs):\n"
        "    return value, args, kwargs\n"
        "\n"
        "def still_noncompliant(left, right, *rest):\n"
        "    return left, right, rest\n"
    )

    violations = find_keyword_only_violations(paths=[source])

    assert [
        (violation.qualified_name, violation.positional_parameters)
        for violation in violations
    ] == [("still_noncompliant", ("left", "right"))]


def test_cli_reports_violations_for_pre_commit(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "cli.py"
    source.write_text("def noncompliant(left, right):\n    return left + right\n")

    exit_code = main(paths=[source])

    assert exit_code == 1
    assert capsys.readouterr().out == (
        f"{source}:1: KWO001 noncompliant: "
        "make positional parameters keyword-only: left, right\n"
    )
