"""Tests for the dead-code check in `tests/ci/dead_code.py`."""

from pathlib import Path

import pytest

from tests.ci.dead_code import AllowEntry, check, find_orphans, load_allowlist


def _write(*, root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def test_reports_internal_test_only_private_public_and_test_helpers(
    tmp_path: Path,
) -> None:
    """Scope: internal code, test-only internal code, `lcm` privates, test helpers."""
    _write(
        root=tmp_path,
        files={
            "src/_lcm/engine.py": (
                "def used():\n    return 1\n\n"
                "def dead():\n    return 2\n\n"
                "def only_tested():\n    return 3\n"
            ),
            "src/lcm/api.py": (
                "from _lcm.engine import used\n\n"
                "def public():\n    return used()\n\n"
                "def _private():\n    return 4\n"
            ),
            "tests/test_engine.py": (
                "from _lcm.engine import only_tested\n\n"
                "def _helper():\n    return 5\n\n"
                "def test_it():\n    assert only_tested() == 3\n"
            ),
        },
    )

    found = {(o.path, o.name, o.test_only) for o in find_orphans(root=tmp_path)}

    assert found == {
        ("src/_lcm/engine.py", "dead", False),
        ("src/_lcm/engine.py", "only_tested", True),
        ("src/lcm/api.py", "_private", False),
        ("tests/test_engine.py", "_helper", False),
    }


def test_allowlist_excuses_by_glob_and_reports_stale_entries(tmp_path: Path) -> None:
    """An entry excuses matching orphans; one that matches nothing is a violation."""
    _write(
        root=tmp_path,
        files={"src/_lcm/solver.py": "def validate_model():\n    return 1\n"},
    )
    allowlist = [
        AllowEntry(path="src/_lcm/*.py", name="validate_*", reason="protocol"),
        AllowEntry(path="src/_lcm/gone.py", name="vanished", reason="stale"),
    ]

    messages = check(root=tmp_path, allowlist=allowlist)

    assert messages == [
        (
            "dead-code allowlist entry src/_lcm/gone.py::vanished matches nothing; "
            "remove it."
        )
    ]


def test_refuses_an_allowlist_entry_without_a_reason(tmp_path: Path) -> None:
    allowlist = tmp_path / "allow.toml"
    allowlist.write_text(
        '[[allow]]\npath = "src/_lcm/x.py"\nname = "f"\nreason = " "\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no reason"):
        load_allowlist(path=allowlist)


def test_refuses_a_file_vulture_cannot_read(tmp_path: Path) -> None:
    """An unreadable file would hide every use inside it."""
    _write(root=tmp_path, files={"src/_lcm/broken.py": "def f(:\n"})

    with pytest.raises(RuntimeError, match="could not read every file"):
        find_orphans(root=tmp_path)
