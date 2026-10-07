#!/usr/bin/env python3
"""Re-anchor the certificate's AST corridor pins to a reviewed source edit.

`check_seals.py --fix` repairs the two *byte* seals — the generated inventory and
`_SOURCE_SEALS`. It cannot repair the semantic pins in `direct_flow.py`: the
per-callable AST digests and the per-module transport surfaces. Those are what
exit code 2 reports, and this tool is the named next step for them.

Every pin stands once, in the `_CORRIDOR_PINS` store of `direct_flow.py`, keyed
by the certified source it describes; the certificate families select their
subsets from that store by name. The contract is narrow on purpose: only pins
owned by the sources named on the command line are rewritten, each at its own
entry in the store.

- a pin owned by a source that was **not** named must still match the tree; if
  one drifted, an unintended edit reached a certified source and the run is
  refused without writing anything;
- a name stored twice with *different* digests is refused rather than guessed
  at;
- field tuples, enum bodies, binding counts and `expected_imports` entries are
  reviewable prose, so they are never rewritten. A remaining verifier error is
  reported by name for a hand edit.

Digests are recomputed with `direct_flow.py`'s own helpers, never with a local
reimplementation, so a change to how the certificate hashes a callable cannot
silently disagree with how this tool re-pins it.

A pin is identified by its source and name; the line it stands on is read
afresh on every run, because line numbers in `direct_flow.py` move whenever a
seal is added.

Exit codes:

- `0` ⇒ nothing to do, or (without `--check`) the named sources' pins were
  rewritten
- `1` ⇒ drift outside the named sources, an ambiguous pin, or — under `--check`
  — drift the named sources own
"""

import argparse
import ast
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

DIRECT_FLOW_PATH = "tests/candidate_certificate/direct_flow.py"
_DIGEST_LENGTH = 64
_STORE = "_CORRIDOR_PINS"


@dataclass(frozen=True)
class CorridorPin:
    """One recomputable anchor in the pin store of `direct_flow.py`."""

    source: str
    """Repository-relative path of the certified source this pin describes."""

    kind: str
    """Either `"module surface"` or `"callable"`."""

    name: str
    """Qualified callable name, or the source path for a module surface."""

    pinned: str
    """The digest literal currently written in `direct_flow.py`."""

    line: int
    """1-based line of the digest literal in `direct_flow.py`, read on this run."""

    column: int
    """0-based offset of the literal's opening quote on that line."""


@dataclass(frozen=True)
class RepinOutcome:
    """What one run found, and what it would write."""

    drifted: tuple[tuple[CorridorPin, str], ...]
    """In-scope pins whose recomputed digest differs, with the new digest."""

    foreign: tuple[tuple[CorridorPin, str], ...]
    """Out-of-scope pins that drifted; any entry refuses the run."""

    ambiguous: tuple[str, ...]
    """Messages naming pins this tool refuses to touch."""


def collect_pins(*, repo_root: Path) -> tuple[CorridorPin, ...]:
    """Return every corridor pin in the store, attributed to its source.

    The store is `{source: (surface or None, {qualname: digest})}`; the source key
    is a module-level `*_SOURCE` constant, resolved against the loaded module
    rather than guessed.
    """
    tree = ast.parse((repo_root / DIRECT_FLOW_PATH).read_text(encoding="utf-8"))
    module = _load_direct_flow(root=repo_root)
    store = _store(tree)
    pins: list[CorridorPin] = []
    for key_node, value_node in zip(store.keys, store.values, strict=True):
        source = _resolve_source_key(node=key_node, module=module)
        if source is None or not isinstance(value_node, ast.Tuple):
            continue
        if len(value_node.elts) != 2:
            continue
        surface_node, inner_node = value_node.elts
        surface = _digest_literal(surface_node)
        if surface is not None:
            pins.append(
                CorridorPin(
                    source=source,
                    kind="module surface",
                    name=source,
                    pinned=surface,
                    line=surface_node.lineno,
                    column=surface_node.col_offset,
                )
            )
        if isinstance(inner_node, ast.Dict):
            pins.extend(_callable_pins(source=source, contracts=inner_node))
    return tuple(pins)


def evaluate(*, repo_root: Path, changed_sources: frozenset[str]) -> RepinOutcome:
    """Recompute every pin and split the drift into in-scope and out-of-scope."""
    module = _load_direct_flow(root=repo_root)
    trees: dict[str, ast.Module] = {}
    drifted: list[tuple[CorridorPin, str]] = []
    foreign: list[tuple[CorridorPin, str]] = []
    ambiguous: list[str] = []

    for pin, conflicting in _distinct_pins(collect_pins(repo_root=repo_root)):
        if conflicting:
            ambiguous.append(
                f"{pin.source}::{pin.name}: pinned to {len(conflicting)} different "
                "digests; one name must denote one fact, so re-pin it by hand"
            )
            continue
        try:
            tree = _source_tree(repo_root=repo_root, source=pin.source, cache=trees)
            recomputed = _recompute(pin=pin, tree=tree, module=module)
        except (OSError, SyntaxError, ValueError) as error:
            ambiguous.append(f"{pin.source}::{pin.name}: cannot recompute: {error}")
            continue
        if recomputed == pin.pinned:
            continue
        if pin.source in changed_sources:
            drifted.append((pin, recomputed))
        else:
            foreign.append((pin, recomputed))

    return RepinOutcome(
        drifted=tuple(drifted), foreign=tuple(foreign), ambiguous=tuple(ambiguous)
    )


def _distinct_pins(
    pins: Sequence[CorridorPin],
) -> list[tuple[CorridorPin, tuple[str, ...]]]:
    """Flag a name stored twice with two different digests.

    A dict literal can repeat a key. Two *different* digests under one name
    denote two facts, and this tool refuses them rather than picking one.
    """
    grouped: dict[tuple[str, str, str], list[CorridorPin]] = {}
    for pin in pins:
        grouped.setdefault((pin.source, pin.kind, pin.name), []).append(pin)
    collapsed: list[tuple[CorridorPin, tuple[str, ...]]] = []
    for group in grouped.values():
        values = sorted({pin.pinned for pin in group})
        collapsed.append((group[0], () if len(values) == 1 else tuple(values)))
    return collapsed


def rewrite(*, repo_root: Path, outcome: RepinOutcome) -> int:
    """Replace each drifted digest at its own store entry; return the pins rewritten.

    Two sources can pin callables with identical bodies, so one digest may stand
    under several names. Rewriting by position touches only the drifted entry.
    """
    path = repo_root / DIRECT_FLOW_PATH
    lines = path.read_text(encoding="utf-8").split("\n")
    for pin, recomputed in outcome.drifted:
        line = lines[pin.line - 1]
        start = pin.column + 1
        if line[start : start + _DIGEST_LENGTH] != pin.pinned:
            raise ValueError(
                f"{DIRECT_FLOW_PATH}:{pin.line}: expected {pin.pinned} for "
                f"{pin.source}::{pin.name}"
            )
        lines[pin.line - 1] = line[:start] + recomputed + line[start + _DIGEST_LENGTH :]
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(outcome.drifted)


def format_table(outcome: RepinOutcome) -> str:
    """Render the before/after table for the pins this run rewrites."""
    if not outcome.drifted:
        return "no corridor pin owned by the named sources drifted"
    header = f"{'source':<52} {'kind':<15} {'name':<45} before -> after"
    rows = [header]
    for pin, recomputed in sorted(
        outcome.drifted, key=lambda item: (item[0].source, item[0].name)
    ):
        rows.append(
            f"{pin.source:<52} {pin.kind:<15} {pin.name:<45} "
            f"{pin.pinned[:12]}... -> {recomputed[:12]}..."
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Re-pin the named sources' corridors, or report drift under `--check`."""
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    changed_sources = frozenset(args.changed_source)

    unknown = sorted(s for s in changed_sources if not (repo_root / s).is_file())
    if unknown:
        print(f"no such source under {repo_root}: {unknown}")
        return 1

    outcome = evaluate(repo_root=repo_root, changed_sources=changed_sources)

    for message in outcome.ambiguous:
        print(f"ambiguous: {message}")
    for pin, recomputed in outcome.foreign:
        print(
            f"unnamed source drifted: {pin.source}::{pin.name} "
            f"({pin.pinned[:12]}... -> {recomputed[:12]}...)"
        )
    if outcome.ambiguous or outcome.foreign:
        print(
            "\nRefusing to write. Name every source you intended to change, or "
            "revert the unintended edit. Never widen a pin to make a changed "
            "route green."
        )
        return 1

    print(format_table(outcome))
    if args.check:
        return 1 if outcome.drifted else 0
    if outcome.drifted:
        rewritten = rewrite(repo_root=repo_root, outcome=outcome)
        print(f"\nrewrote {rewritten} corridor pin(s) in {DIRECT_FLOW_PATH}")
    _report_hand_edits(repo_root=repo_root)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: derived from this script's location)",
    )
    parser.add_argument(
        "--changed-source",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "repository-relative path of a source whose edit is intended; "
            "repeat once per source. A pin owned by any other source must "
            "still match the tree."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit non-zero without writing",
    )
    return parser


def _load_direct_flow(*, root: Path) -> ModuleType:
    """Load the verifier from the tree under repair, not from this script's."""
    path = root / DIRECT_FLOW_PATH
    spec = importlib.util.spec_from_file_location("_certificate_repin_flow", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_tree(
    *, repo_root: Path, source: str, cache: dict[str, ast.Module]
) -> ast.Module:
    if source not in cache:
        text = (repo_root / source).read_text(encoding="utf-8")
        cache[source] = ast.parse(text, filename=source)
    return cache[source]


def _recompute(*, pin: CorridorPin, tree: ast.Module, module: ModuleType) -> str:
    """Recompute one pin with the certificate's own hashing helpers."""
    if pin.kind == "module surface":
        return str(module._transport_module_surface(tree))
    if "." in pin.name:
        class_name, method_name = pin.name.split(".", maxsplit=1)
        _, node = module._method_definition(
            tree=tree, class_name=class_name, method_name=method_name
        )
    else:
        node = module._definition(tree=tree, name=pin.name)
    return str(module._callable_ast_sha256(node))


def _digest_literal(node: ast.expr | None) -> str | None:
    """Return the SHA-256 hex string this node spells, or None."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    value = node.value
    if len(value) != _DIGEST_LENGTH:
        return None
    return value if all(c in "0123456789abcdef" for c in value) else None


def _store(tree: ast.Module) -> ast.Dict:
    """Return the literal dict assigned to the pin store."""
    for statement in tree.body:
        if isinstance(statement, ast.AnnAssign):
            target = statement.target
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == _STORE
            and isinstance(statement.value, ast.Dict)
        ):
            return statement.value
    raise ValueError(f"{DIRECT_FLOW_PATH} has no literal {_STORE} store")


def _resolve_source_key(*, node: ast.expr | None, module: ModuleType) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value.endswith(".py") else None
    if isinstance(node, ast.Name):
        value = getattr(module, node.id, None)
        return value if isinstance(value, str) and value.endswith(".py") else None
    return None


def _callable_pins(*, source: str, contracts: ast.Dict) -> list[CorridorPin]:
    pins: list[CorridorPin] = []
    for key_node, value_node in zip(contracts.keys, contracts.values, strict=True):
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        digest = _digest_literal(value_node)
        if digest is None:
            continue
        pins.append(
            CorridorPin(
                source=source,
                kind="callable",
                name=key_node.value,
                pinned=digest,
                line=value_node.lineno,
                column=value_node.col_offset,
            )
        )
    return pins


def _report_hand_edits(*, repo_root: Path) -> None:
    """Name the verifier errors no recomputation can repair."""
    module = _load_direct_flow(root=repo_root)
    result = module.verify_direct_candidate_flow(repo_root=repo_root)
    remaining = [
        error for error in result["errors"] if "source seal mismatch" not in error
    ]
    if not remaining:
        return
    print(
        "\nThe verifier still reports errors no recomputation can repair. Each "
        "one is a field tuple, enum body, binding count or import list that a "
        "reviewer edits by hand:"
    )
    for error in remaining:
        print(f"  {error}")


if __name__ == "__main__":
    sys.exit(main())
