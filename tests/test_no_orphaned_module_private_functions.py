"""No module-private function outlives the call site it was written for.

When a module keeps a private function that nothing in that module calls, while a
function of the same name lives in another module, one of two copies has lost its
consumer: the work that copy performs has silently dropped out of the program. The
surviving definition keeps imports, type checking, and every behavioural test green,
so no other check in the suite can see it -- a behavioural guard only fires for
models that exercise the specific computation that went missing, and the point of
this module is to catch the *shape* of the mistake instead, wherever it happens and
whatever the function is called.

The signature is deliberately narrow. "Module-private function with no in-module
use" on its own is noisy, because most such functions are imported by sibling
modules and are perfectly alive. Requiring a same-named twin in another module is
what turns the observation into evidence, and it leaves few enough legitimate cases
repo-wide that each one can be named in `ALLOWED` below.
"""

import ast
import collections
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# `_bind_forward_refs` is intentionally re-implemented as a small local helper in
# several modules rather than shared; each copy is used via a path this AST-level
# check does not model. Add to this list only with a reason.
ALLOWED = {"_bind_forward_refs"}


def _module_private_defs_and_names(
    path: pathlib.Path,
) -> tuple[set[str], set[str]]:
    """Return (module-level private function names, all names referenced) for `path`."""
    try:
        # Python source is UTF-8 by definition; `read_text()` would decode with the
        # locale default instead (cp1252 on Windows), which has no 0x81 and so dies on
        # the superscript minus in `g⁻¹`-style math notation the docstrings use.
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a syntax error is another test's problem
        return set(), set()
    defs = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("_")
        and not node.name.startswith("__")
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    return defs, used


def test_no_module_private_function_is_dead_beside_a_twin():
    """No module-private function is dead in its own module while duplicated elsewhere.

    That combination means one of two copies lost its consumer, so whatever that
    copy computed is not computed anywhere.
    """
    defining_modules: dict[str, list[str]] = collections.defaultdict(list)
    dead: list[tuple[str, str]] = []

    for path in sorted(SRC.rglob("*.py")):
        defs, used = _module_private_defs_and_names(path)
        rel = str(path.relative_to(SRC))
        for name in defs:
            defining_modules[name].append(rel)
        dead.extend((rel, name) for name in defs - used)

    orphans = sorted(
        (rel, name)
        for rel, name in dead
        if name not in ALLOWED and len(defining_modules[name]) > 1
    )

    assert not orphans, (
        "module-private function(s) defined but never used in their own module, "
        "while a same-named function exists in another module -- a dropped consumer "
        "left behind by a merge:\n"
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
