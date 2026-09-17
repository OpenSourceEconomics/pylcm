#!/usr/bin/env python3
"""Re-anchor the certificate's AST corridor pins to a reviewed source edit.

`check_seals.py --fix` repairs the two *byte* seals — the generated inventory and
`_SOURCE_SEALS`. It cannot repair the semantic pins in `direct_flow.py`: the
per-callable AST digests and the per-module transport surfaces. Those are what
exit code 2 reports, and this tool is the named next step for them.

The contract is narrow on purpose. Every pin is attributed to the certified
source it describes, and only pins owned by the sources named on the command
line are rewritten:

- a pin owned by a source that was **not** named must still match the tree; if
  one drifted, an unintended edit reached a certified source and the run is
  refused without writing anything;
- a name pinned to two *different* digests, or a pin this tool cannot attribute
  to exactly one source, is refused rather than guessed at. One source is
  referenced by several contract dicts, so a repeated pin of the same digest is
  one fact and is rewritten everywhere it stands;
- field tuples, enum bodies, binding counts and `expected_imports` entries are
  reviewable prose, so they are never rewritten. A remaining verifier error is
  reported by name for a hand edit.

Digests are recomputed with `direct_flow.py`'s own helpers, never with a local
reimplementation, so a change to how the certificate hashes a callable cannot
silently disagree with how this tool re-pins it.

Anchors are named by search string, not by line number: line numbers in
`direct_flow.py` move whenever a seal is added.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TypeGuard

DIRECT_FLOW_PATH = "tests/candidate_certificate/direct_flow.py"
_DIGEST_LENGTH = 64
_VERIFY_FUNCTION = "verify_direct_candidate_flow"


@dataclass(frozen=True)
class CorridorPin:
    """One recomputable anchor in `direct_flow.py`.

    A pin is identified by the digest literal standing in the file, never by a
    line number, because adding a seal moves every line below it.
    """

    source: str
    """Repository-relative path of the certified source this pin describes."""

    kind: str
    """Either `"module surface"` or `"callable"`."""

    name: str
    """Qualified callable name, or the source path for a module surface."""

    pinned: str
    """The digest literal currently written in `direct_flow.py`."""


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
    """Return every recomputable corridor pin, attributed to its source."""
    direct_flow_path = repo_root / DIRECT_FLOW_PATH
    tree = ast.parse(direct_flow_path.read_text(encoding="utf-8"))
    module = _load_direct_flow(root=repo_root)
    sources_by_function = _sources_by_function(tree=tree)
    pins = [*_nested_contract_pins(tree=tree, module=module)]
    pins.extend(_bare_contract_pins(tree=tree, sources_by_function=sources_by_function))
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
    """Collapse repeated pins of one fact; flag names pinned to two facts.

    One certified source is referenced by several contract dicts, so the same
    module surface or callable digest is written more than once. Those repeats
    denote one fact and are rewritten together. Two *different* digests under
    one name do not, and this tool refuses them rather than picking one.
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
    """Replace each drifted digest literal in place; return the pins rewritten."""
    path = repo_root / DIRECT_FLOW_PATH
    text = path.read_text(encoding="utf-8")
    for pin, recomputed in outcome.drifted:
        text = text.replace(pin.pinned, recomputed)
    path.write_text(text, encoding="utf-8")
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


def _nested_contract_pins(*, tree: ast.Module, module: ModuleType) -> list[CorridorPin]:
    """Return pins from `{source: (surface, {qualname: digest})}` literals.

    The source key is either a path string or a module-level `*_SOURCE`
    constant, which is resolved against the loaded module rather than guessed.
    """
    pins: list[CorridorPin] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            source = _resolve_source_key(node=key_node, module=module)
            if source is None or not isinstance(value_node, ast.Tuple):
                continue
            if len(value_node.elts) != 2:
                continue
            surface_node, inner_node = value_node.elts
            if not isinstance(inner_node, ast.Dict):
                continue
            surface = _digest_literal(surface_node)
            if surface is None:
                continue
            pins.append(
                CorridorPin(
                    source=source,
                    kind="module surface",
                    name=source,
                    pinned=surface,
                )
            )
            pins.extend(_callable_pins(source=source, contracts=inner_node))
    return pins


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
            )
        )
    return pins


def _bare_contract_pins(
    *, tree: ast.Module, sources_by_function: Mapping[str, frozenset[str]]
) -> list[CorridorPin]:
    """Return pins from `contracts={...}` passed straight to the digest checker.

    Such a dict names no source of its own; the source is the one the enclosing
    verifier function is applied to in `verify_direct_candidate_flow`. A
    function applied to several sources cannot attribute its bare pins, so its
    pins are dropped here and the enclosing source is repaired by hand.
    """
    pins: list[CorridorPin] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        owned = sources_by_function.get(function.name, frozenset())
        if len(owned) != 1:
            continue
        source = next(iter(owned))
        for node in ast.walk(function):
            if not _is_exact_callable_call(node):
                continue
            for keyword in node.keywords:
                if keyword.arg == "contracts" and isinstance(keyword.value, ast.Dict):
                    pins.extend(_callable_pins(source=source, contracts=keyword.value))
    return pins


def _is_exact_callable_call(node: ast.AST) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_exact_callable_errors"
    )


def _sources_by_function(*, tree: ast.Module) -> dict[str, frozenset[str]]:
    """Map each verifier helper to the certified sources it is applied to.

    Read off `verify_direct_candidate_flow`, which parses each certified source
    once into `parsed` and then hands the resulting tree to its helpers. A
    helper reached with two different trees owns two sources, and this tool
    then declines to attribute its bare pins.
    """
    verify = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == _VERIFY_FUNCTION
        ),
        None,
    )
    if verify is None:
        return {}
    constants = _module_string_constants(tree=tree)
    bindings: dict[str, frozenset[str]] = {}
    _bind_trees(statements=verify.body, constants=constants, bindings=bindings)

    owners: dict[str, set[str]] = {}
    for node in ast.walk(verify):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        arguments = [*node.args, *(kw.value for kw in node.keywords)]
        for argument in arguments:
            if isinstance(argument, ast.Name) and argument.id in bindings:
                owners.setdefault(node.func.id, set()).update(bindings[argument.id])
    return {name: frozenset(sources) for name, sources in owners.items()}


def _module_string_constants(*, tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        value = statement.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            constants[target.id] = value.value
    return constants


def _bind_trees(
    *,
    statements: Sequence[ast.stmt],
    constants: Mapping[str, str],
    bindings: dict[str, frozenset[str]],
    loop_sources: frozenset[str] | None = None,
) -> None:
    """Record which parsed source each local tree variable holds."""
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            sources = _parsed_get_sources(
                node=statement.value, constants=constants, loop_sources=loop_sources
            )
            if isinstance(target, ast.Name) and sources:
                bindings[target.id] = bindings.get(target.id, frozenset()) | sources
        if isinstance(statement, ast.For):
            iterated = _iterated_sources(node=statement.iter, constants=constants)
            _bind_trees(
                statements=statement.body,
                constants=constants,
                bindings=bindings,
                loop_sources=iterated,
            )
        elif isinstance(statement, ast.If):
            for branch in (statement.body, statement.orelse):
                _bind_trees(
                    statements=branch,
                    constants=constants,
                    bindings=bindings,
                    loop_sources=loop_sources,
                )


def _parsed_get_sources(
    *,
    node: ast.expr,
    constants: Mapping[str, str],
    loop_sources: frozenset[str] | None,
) -> frozenset[str]:
    """Return the sources a `parsed.get(...)` expression can yield."""
    if not _is_parsed_get(node):
        return frozenset()
    argument = node.args[0]
    if isinstance(argument, ast.Name):
        if argument.id in constants:
            return frozenset({constants[argument.id]})
        return loop_sources or frozenset()
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return frozenset({argument.value})
    return frozenset()


def _is_parsed_get(node: ast.expr) -> TypeGuard[ast.Call]:
    """Report whether `node` is a `parsed.get(<key>)` call with a key."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _name_is(node=node.func.value, expected="parsed")
        and bool(node.args)
    )


def _name_is(*, node: ast.expr, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _iterated_sources(
    *, node: ast.expr, constants: Mapping[str, str]
) -> frozenset[str]:
    if not isinstance(node, ast.Tuple | ast.List | ast.Set):
        return frozenset()
    sources: set[str] = set()
    for element in node.elts:
        if isinstance(element, ast.Name) and element.id in constants:
            sources.add(constants[element.id])
        elif isinstance(element, ast.Constant) and isinstance(element.value, str):
            sources.add(element.value)
    return frozenset(sources)


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
