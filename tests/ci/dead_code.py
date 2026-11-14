"""Refuse internal code that nothing outside the test suite calls.

Two vulture passes over the repository:

- Without `tests/`: a function, method, property or class in `src/_lcm` or
  `src/_lcm_regime_finder`, or a `_`-private one in `src/lcm` or `src/lcm_examples`,
  that no production, benchmark, tool or documentation code uses is orphaned. Code
  whose only callers are tests counts as orphaned too: either it moves into the test
  suite or it goes.
- With `tests/`: an unused function, method, property or class in `tests/` is
  orphaned.

Public names in `src/lcm` are the API and are not reported. Variables and attributes
are not reported (dataclass and record fields are read by name at runtime), nor are
imports (ruff's F401 covers them).

A use that vulture cannot see, such as a protocol method called through an instance
it cannot type, a JAX rule registered by decorator or a pytest hook, is declared in
`dead-code-allowlist.toml` with its reason. An entry that no longer matches anything
is itself a violation, so the allowlist cannot outlive the code it excuses.
"""

import fnmatch
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import vulture

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).with_name("dead-code-allowlist.toml")
CONSUMERS = ("src", "benchmarks", "tools", "docs", "acceptance", "hatch_build.py")
TESTS = "tests"
KINDS = frozenset({"function", "method", "property", "class"})
INTERNAL_PACKAGES = ("src/_lcm/", "src/_lcm_regime_finder/")
PUBLIC_PACKAGES = ("src/lcm/", "src/lcm_examples/")


@dataclass(frozen=True, kw_only=True)
class Orphan:
    """One unused definition, located by repository-relative path."""

    path: str
    line: int
    kind: str
    name: str
    test_only: bool


@dataclass(frozen=True, kw_only=True)
class AllowEntry:
    """An allowlisted use vulture cannot see; `path` and `name` are globs."""

    path: str
    name: str
    reason: str

    def matches(self, orphan: Orphan) -> bool:
        """Return whether this entry excuses `orphan`."""
        return fnmatch.fnmatchcase(orphan.name, self.name) and fnmatch.fnmatch(
            orphan.path, self.path
        )


def _unused(*, root: Path, paths: tuple[str, ...]) -> list[tuple[str, int, str, str]]:
    """Return `(path, line, kind, name)` for vulture's unused definitions."""
    # pytest resolves fixtures by parameter name or `name=` alias, or applies them
    # with `autouse=True`; vulture sees none of these.
    scanner = vulture.Vulture(ignore_decorators=["@pytest.fixture"])
    scanner.scavenge([str(root / p) for p in paths if (root / p).exists()])
    if scanner.exit_code != vulture.core.ExitCode.NoDeadCode:
        # A file vulture cannot parse is skipped, and every use inside it is lost.
        msg = "vulture could not read every file; see its message above."
        raise RuntimeError(msg)
    return [
        (
            Path(item.filename).resolve().relative_to(root).as_posix(),
            item.first_lineno,
            item.typ,
            item.name,
        )
        for item in scanner.get_unused_code(min_confidence=0)
        # A dunder is called by the data model, never by name.
        if item.typ in KINDS
        and not (item.name.startswith("__") and item.name.endswith("__"))
    ]


def _reported_without_tests(*, path: str, name: str) -> bool:
    """Return whether a definition unused outside the tests is in scope."""
    if path.startswith(INTERNAL_PACKAGES):
        return True
    return path.startswith(PUBLIC_PACKAGES) and name.startswith("_")


def find_orphans(*, root: Path) -> list[Orphan]:
    """Return every orphaned definition under `root`, test-only ones marked."""
    without_tests = _unused(root=root, paths=CONSUMERS)
    with_tests = set(_unused(root=root, paths=(*CONSUMERS, TESTS)))
    orphans = [
        Orphan(
            path=path,
            line=line,
            kind=kind,
            name=name,
            test_only=(path, line, kind, name) not in with_tests,
        )
        for path, line, kind, name in without_tests
        if _reported_without_tests(path=path, name=name)
    ]
    orphans.extend(
        Orphan(path=path, line=line, kind=kind, name=name, test_only=False)
        for path, line, kind, name in with_tests
        if path.startswith(f"{TESTS}/")
    )
    return sorted(orphans, key=lambda o: (o.path, o.line))


def load_allowlist(*, path: Path) -> list[AllowEntry]:
    """Read the allowlist; every entry must name a path, a name and a reason."""
    entries = tomllib.loads(path.read_text(encoding="utf-8")).get("allow", [])
    allow = []
    for entry in entries:
        if not entry.get("reason", "").strip():
            msg = f"dead-code allowlist entry {entry} has no reason."
            raise ValueError(msg)
        allow.append(
            AllowEntry(path=entry["path"], name=entry["name"], reason=entry["reason"])
        )
    return allow


def check(*, root: Path, allowlist: list[AllowEntry]) -> list[str]:
    """Return one message per unexcused orphan and per stale allowlist entry."""
    orphans = find_orphans(root=root)
    messages = [
        f"{o.path}:{o.line}: unused {o.kind} {o.name!r}"
        + (" (used only by tests)" if o.test_only else "")
        for o in orphans
        if not any(entry.matches(o) for entry in allowlist)
    ]
    messages.extend(
        f"dead-code allowlist entry {entry.path}::{entry.name} matches nothing; "
        "remove it."
        for entry in allowlist
        if not any(entry.matches(o) for o in orphans)
    )
    return messages


def main() -> int:
    """Print every violation and return a non-zero status if there is one."""
    messages = check(root=ROOT, allowlist=load_allowlist(path=ALLOWLIST))
    for message in messages:
        sys.stdout.write(f"{message}\n")
    if messages:
        sys.stdout.write(
            f"\n{len(messages)} dead-code violation(s). Delete the code, move code "
            "used only by tests into the test suite, or declare a use vulture "
            f"cannot see in {ALLOWLIST.relative_to(ROOT)} with its reason.\n"
        )
    return 1 if messages else 0


if __name__ == "__main__":
    sys.exit(main())
