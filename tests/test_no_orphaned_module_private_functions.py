"""No module-private function outlives the call site it was written for.

An unused private function beside a twin can indicate a dropped consumer. Explicit
imports from sibling modules keep a shared helper live even when its own module
does not call it. Resolve the source module of each import so that importing one
copy cannot hide an unused copy elsewhere. Dynamic consumers that this AST check
does not model must be named in `ALLOWED` with a reason.
"""

import ast
import collections
import pathlib
from importlib.util import resolve_name

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# `_bind_forward_refs` is intentionally re-implemented as a small local helper in
# several modules rather than shared; each copy is used via a path this AST-level
# check does not model. Add to this list only with a reason.
ALLOWED = {"_bind_forward_refs"}


def _module_private_defs_and_names(
    path: pathlib.Path,
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """Return private definitions, referenced names, and explicit source imports."""
    try:
        # Python source is UTF-8 by definition; `read_text()` would decode with the
        # locale default instead (cp1252 on Windows), which has no 0x81 and so dies on
        # the superscript minus in `g⁻¹`-style math notation the docstrings use.
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a syntax error is another test's problem
        return set(), set(), set()
    defs = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("_")
        and not node.name.startswith("__")
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported: set[tuple[str, str]] = set()
    package = ".".join(path.relative_to(SRC).parent.parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            imported.update((module, alias.name) for alias in node.names)
    return defs, used, imported


def _module_name(path: pathlib.Path) -> str:
    """Return the import name of a source file or package initializer."""
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_no_module_private_function_is_dead_beside_a_twin() -> None:
    """Refuse an unused private copy after accounting for explicit imports."""
    defining_modules: dict[str, list[str]] = collections.defaultdict(list)
    dead: list[tuple[str, str, str]] = []
    imported: set[tuple[str, str]] = set()

    for path in sorted(SRC.rglob("*.py")):
        defs, used, imports = _module_private_defs_and_names(path)
        imported.update(imports)
        rel = str(path.relative_to(SRC))
        for name in defs:
            defining_modules[name].append(rel)
        dead.extend((rel, _module_name(path), name) for name in defs - used)

    orphans = sorted(
        (rel, name)
        for rel, module, name in dead
        if name not in ALLOWED
        and len(defining_modules[name]) > 1
        and (module, name) not in imported
    )

    assert not orphans, (
        "module-private function(s) have neither local use nor an explicit import, "
        "while a same-named function exists in another module:\n"
        + "\n".join(f"  {rel}: {name}" for rel, name in orphans)
    )


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_allowlist_entries_still_exist(name: str):
    """An allowlist entry that no longer exists is stale and must be removed.

    Without this, the allowlist silently grows into a place where a real defect can
    hide -- an exemption has to keep naming something real.
    """
    found = any(
        name in _module_private_defs_and_names(path)[0] for path in SRC.rglob("*.py")
    )
    assert found, f"allowlist entry {name!r} matches nothing; remove it"


@pytest.mark.parametrize(
    ("owner_file", "consumer_file", "import_line", "imported_name"),
    [
        ("pkg/kernel.py", "pkg/use.py", "from pkg.kernel import _shared", "_shared"),
        (
            "pkg/kernel.py",
            "pkg/use.py",
            "from pkg.kernel import _shared as shared",
            "shared",
        ),
        (
            "pkg/kernel.py",
            "pkg/use.py",
            "from .kernel import _shared as shared",
            "shared",
        ),
        ("pkg/kernel.py", "pkg/sub/use.py", "from ..kernel import _shared", "_shared"),
        ("pkg/__init__.py", "pkg/use.py", "from . import _shared", "_shared"),
    ],
)
def test_an_imported_private_function_has_a_consumer(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    owner_file: str,
    consumer_file: str,
    import_line: str,
    imported_name: str,
) -> None:
    """A sibling's import keeps its exact source function alive despite a twin."""
    owner = tmp_path / owner_file
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text("def _shared():\n    return 1\n", encoding="utf-8")
    consumer = tmp_path / consumer_file
    consumer.parent.mkdir(parents=True, exist_ok=True)
    consumer.write_text(f"{import_line}\n{imported_name}()\n", encoding="utf-8")
    (tmp_path / "twin.py").write_text(
        "def _shared():\n    return 2\n_shared()\n", encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "SRC", tmp_path)

    test_no_module_private_function_is_dead_beside_a_twin()


def test_importing_a_twin_does_not_hide_the_unused_copy(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """An import only protects the function in the module it names."""
    (tmp_path / "unused.py").write_text(
        "def _shared():\n    return 1\n", encoding="utf-8"
    )
    (tmp_path / "twin.py").write_text(
        "def _shared():\n    return 2\n_shared()\n", encoding="utf-8"
    )
    (tmp_path / "consumer.py").write_text(
        "from twin import _shared\n_shared()\n", encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "SRC", tmp_path)

    with pytest.raises(AssertionError, match=r"unused\.py: _shared"):
        test_no_module_private_function_is_dead_beside_a_twin()
