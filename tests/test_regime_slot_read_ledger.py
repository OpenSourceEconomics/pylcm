"""Every read of a regime's three declaring slots is accounted for, by name.

`functions`, `constraints` and `transition` hold what a model author wrote,
declarations included; `decomposed_functions`, `decomposed_constraints` and
`decomposed_transition` hold what the engine runs. Which of the two a given
line wants is a per-line judgement, and getting it wrong is usually silent — a
declaration filtered out by `callable()`, a name looked up with `in` and not
found. So the judgement is written down once per site in
`regime_slot_read_ledger.csv`, and this module checks the ledger against the
source it describes.

The ledger is a partition: every read in `src/` appears in it exactly once, and
no ledger row describes a read that is no longer there. Adding, moving or
switching a read fails here until the ledger says why.
"""

import ast
import csv
import pathlib

import pytest

_SRC = pathlib.Path(__file__).parents[1] / "src"
_LEDGER = pathlib.Path(__file__).parent / "regime_slot_read_ledger.csv"

_SLOT_OF_ACCESSOR = {
    "functions": "functions",
    "constraints": "constraints",
    "transition": "transition",
    "decomposed_functions": "functions",
    "decomposed_constraints": "constraints",
    "decomposed_transition": "transition",
}

# Receivers that name a regime. `self` counts only inside the class itself.
_RECEIVERS = frozenset({"user_regime", "regime", "target_regime", "source", "self"})

_CLASSIFICATIONS = frozenset({"decomposed", "raw", "indifferent", "not_a_user_regime"})

_KEY_FIELDS = ("path", "function", "slot", "accessor", "receiver", "reads")


def _receiver_name(node: ast.expr) -> str | None:
    """The leftmost plain name a receiver expression is rooted at."""
    while True:
        match node:
            case ast.Name(id=name):
                return name
            case ast.Subscript(value=inner) | ast.Attribute(value=inner):
                node = inner
            case ast.Call(func=inner):
                node = inner
            case _:
                return None


def _attributes_with_enclosing_function(tree: ast.AST):
    """Yield each attribute access paired with the function enclosing it."""
    stack: list[str] = []

    def walk(node: ast.AST):
        entered = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        if entered:
            stack.append(node.name)
        if isinstance(node, ast.Attribute):
            yield node, (stack[-1] if stack else "<module>")
        for child in ast.iter_child_nodes(node):
            yield from walk(child)
        if entered:
            stack.pop()

    yield from walk(tree)


def _scan_slot_reads() -> dict[tuple[str, ...], int]:
    """Count every declaring-slot read under `src/`, keyed by site.

    Returns:
        Mapping of (path, function, slot, accessor, receiver) to how many reads
        that site holds. Line numbers are deliberately not part of the key, so
        the ledger does not churn on every unrelated edit above a site.
    """
    counts: dict[tuple[str, ...], int] = {}
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(_SRC.parent).as_posix()
        for node, enclosing in _attributes_with_enclosing_function(tree):
            if node.attr not in _SLOT_OF_ACCESSOR:
                continue
            receiver = _receiver_name(node.value)
            if receiver not in _RECEIVERS:
                continue
            if receiver == "self" and relative != "src/lcm/regime.py":
                continue
            key = (
                relative,
                enclosing,
                _SLOT_OF_ACCESSOR[node.attr],
                node.attr,
                receiver,
            )
            counts[key] = counts.get(key, 0) + 1
    return counts


def _row_id(row: dict[str, str]) -> str:
    """Name a parametrized case after the site it describes."""
    return f"{row['path']}::{row['function']}::{row['accessor']}"


def _ledger_rows() -> list[dict[str, str]]:
    """The ledger as written."""
    with _LEDGER.open() as fh:
        return list(csv.DictReader(fh))


def _scanned_keys() -> set[tuple[str, ...]]:
    """The scan, in the ledger's own key shape."""
    return {(*key, str(count)) for key, count in _scan_slot_reads().items()}


def _ledger_keys() -> set[tuple[str, ...]]:
    """The ledger, in the scan's key shape."""
    return {tuple(row[field] for field in _KEY_FIELDS) for row in _ledger_rows()}


def test_the_ledger_describes_no_read_that_is_gone():
    """A removed or moved read leaves a ledger row with nothing to describe."""
    assert sorted(_ledger_keys() - _scanned_keys()) == []


def test_every_read_in_the_source_is_in_the_ledger():
    """A new or switched read has to say which view it wants, and why."""
    assert sorted(_scanned_keys() - _ledger_keys()) == []


def test_the_ledger_names_each_read_once():
    """One row per site, so the class counts are a partition of the reads."""
    keys = [tuple(row[field] for field in _KEY_FIELDS) for row in _ledger_rows()]

    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("row", _ledger_rows(), ids=_row_id)
def test_each_row_carries_a_classification_the_accessor_agrees_with(row):
    """A row's verdict and the line it describes cannot say different things.

    `decomposed` means the line asks what the engine runs, so it must read a
    decomposed view; `raw` means it asks what the author declared, so it must
    read the declaring slot. The other two verdicts are about lines the
    distinction does not reach, and either accessor is honest there.
    """
    agrees = {
        "decomposed": row["accessor"].startswith("decomposed_"),
        "raw": row["accessor"] == row["slot"],
    }.get(row["classification"], True)

    assert agrees, (
        f"{row['path']}::{row['function']} is classified "
        f"{row['classification']!r} but reads {row['accessor']!r}"
    )


@pytest.mark.parametrize("row", _ledger_rows(), ids=_row_id)
def test_each_row_reaches_a_verdict(row):
    """Every site is decided, so the class counts partition the reads."""
    assert row["classification"] in _CLASSIFICATIONS


@pytest.mark.parametrize("row", _ledger_rows(), ids=_row_id)
def test_each_row_gives_a_reason(row):
    """A verdict without a reason is a note to nobody."""
    assert len(row["rationale"]) > 20


@pytest.mark.parametrize("row", _ledger_rows(), ids=_row_id)
def test_each_row_says_whether_a_miss_would_be_silent(row):
    """Which sites need a positive test is the ledger's most useful column."""
    assert row["fails_silently"] in {"yes", "no"}
