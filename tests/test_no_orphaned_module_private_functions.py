"""Guard against a merge dropping a consumer but leaving its definition behind.

Cascade merge 80f5e79 dropped the solve hot loop's call to `_states_for_period`
while keeping a second copy of the function in `diagnostics.py`. The result was a
`backward_induction.py` that still *defined* `_states_for_period` and never called
it -- so the age-specialized per-period grids were built, stored, and silently
ignored, and every period solved on the base axis.

Nothing caught it. The behavioural guard for that specific defect lives in
`tests/solution/test_age_specialized_grid_solve.py`, but it can only fire for models
that use an `AgeSpecializedGrid`. This test catches the *shape* of the mistake
instead, so the next dropped consumer is caught wherever it happens and whatever it
is called -- in particular during a cascade merge, which is when it happened.

The signature is deliberately narrow, because "module-private function with no
in-module use" alone is noisy (~50 hits: most are imported by sibling modules). The
precise tell is a module-private function that is dead *within its own module* while
a same-named function exists in another module -- i.e. one of two copies stopped
being used. That is 3 benign hits repo-wide, all allowlisted below.
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
        tree = ast.parse(path.read_text())
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

    That combination means one of two copies lost its consumer -- the exact footprint
    of the `_states_for_period` drop in cascade 80f5e79.
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
