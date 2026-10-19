#!/usr/bin/env python3
"""Check that both candidate-certificate seal files match the tree, or repair them.

The certificate pins every certified source by digest twice: the generated
inventory `sources.json` and the hand-maintained `_SOURCE_SEALS` map in
`direct_flow.py`. Editing a certified source leaves both stale, and the certificate
tests then fail on every CI platform. This check takes well under a second, so it
runs as a pre-commit hook on any change under `src/` or to the certificate files.

Exit codes:

- `0` ⇒ both files agree with the tree
- `1` ⇒ a seal drifted; each offending source is printed with its expected and
  actual digest, and `--fix` regenerates the inventory (keeping each profile's
  hand-set exclusions) and rewrites the seal map
- `2` ⇒ the direct-flow verifier reports an error that is not a seal. A reseal
  cannot repair it; `repin_corridors.py` re-anchors the AST corridor pins, which
  stand once in the `_CORRIDOR_PINS` store, for the sources whose edit was
  intended, and the remaining anchors — field
  tuples, enum bodies, binding counts — are reviewed and edited by hand
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import cast

try:
    from generate_sources import (
        INVENTORY_PATH,
        build_inventory,
        canonical_json,
        sha256_file,
    )
except ModuleNotFoundError:  # Imported as tests.candidate_certificate.check_seals.
    from tests.candidate_certificate.generate_sources import (
        INVENTORY_PATH,
        build_inventory,
        canonical_json,
        sha256_file,
    )

DIRECT_FLOW_PATH = "tests/candidate_certificate/direct_flow.py"
_SEAL_MISMATCH = re.compile(r"^(?P<path>[^:]+): source seal mismatch: ")


def _load_direct_flow(*, root: Path) -> ModuleType:
    """Load the seal map and verifier from the tree under check, not this script's.

    The map is read fresh on every call, so a check after `--fix` sees the rewritten
    digests, and a tree other than this checkout is verified against its own map.
    """
    path = (root / DIRECT_FLOW_PATH).resolve()
    name = "_certificate_direct_flow_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    completed = False
    try:
        spec.loader.exec_module(module)
        if Path(cast("str", module.__file__)).resolve() != path:
            raise ImportError(f"loaded verifier origin differs from {path}")
        completed = True
        return module
    finally:
        if not completed:
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = cast("ModuleType", previous)


def check(*, repo_root: Path) -> tuple[list[str], list[str]]:
    """Return the seal mismatches and every other verifier error, as messages."""
    root = repo_root.resolve()
    direct_flow = _load_direct_flow(root=root)
    seal_errors: list[str] = []
    other_errors: list[str] = []

    inventory_path = root / INVENTORY_PATH
    generated = build_inventory(root)
    try:
        committed = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        committed = None
        seal_errors.append(f"{INVENTORY_PATH}: unreadable: {error}")
    if committed is not None and committed != generated:
        committed_digests = {
            item["path"]: item["sha256"] for item in committed.get("sources", [])
        }
        for item in generated["sources"]:
            expected = committed_digests.get(item["path"])
            if expected != item["sha256"]:
                seal_errors.append(
                    f"{item['path']}: inventory seal mismatch: expected {expected}, "
                    f"got {item['sha256']}"
                )
        if not seal_errors:
            seal_errors.append(
                f"{INVENTORY_PATH}: differs from the generated inventory"
            )

    result = direct_flow.verify_direct_candidate_flow(repo_root=root)
    for error in result["errors"]:
        (seal_errors if _SEAL_MISMATCH.match(error) else other_errors).append(error)
    return seal_errors, other_errors


def _source_seal_key(
    *, key: ast.expr | None, direct_flow: ModuleType
) -> tuple[str, str]:
    """Resolve the store's declared source names and literal repository paths."""
    if isinstance(key, ast.Name):
        source = getattr(direct_flow, key.id)
        if not isinstance(source, str):
            raise ValueError("A source seal name must identify one repository path")
        return source, key.id
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value, json.dumps(key.value)
    raise ValueError("Source seals require declared names or literal paths")


def _with_current_source_seals(
    *, original: str, direct_flow: ModuleType, root: Path
) -> str:
    """Refresh every stored byte seal and add each explicitly certified source."""
    tree = ast.parse(original)
    stores = [
        statement.value
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_SOURCE_SEALS"
    ]
    store = stores[0] if len(stores) == 1 else None
    if not isinstance(store, ast.Dict):
        raise ValueError("The source seal store must be one explicit dictionary")
    entries = [_source_seal_key(key=key, direct_flow=direct_flow) for key in store.keys]
    present = {source for source, _key in entries}
    entries.extend(
        (source, json.dumps(source))
        for source in direct_flow._CERTIFIED_CORRIDOR_SOURCES
        if source not in present
    )
    replacement = (
        "{\n"
        + "".join(
            f'    {key}: "{sha256_file(root / source)}",\n' for source, key in entries
        )
        + "}"
    )
    lines = original.splitlines(keepends=True)
    start = sum(map(len, lines[: store.lineno - 1])) + store.col_offset
    if store.end_lineno is None or store.end_col_offset is None:
        raise ValueError("The source seal store has no complete AST span")
    end = sum(map(len, lines[: store.end_lineno - 1])) + store.end_col_offset
    return original[:start] + replacement + original[end:]


def fix(*, repo_root: Path) -> list[str]:
    """Regenerate the inventory and rewrite the seal map; return the rewritten files."""
    root = repo_root.resolve()
    rewritten: list[str] = []

    inventory_path = root / INVENTORY_PATH
    inventory = canonical_json(build_inventory(root))
    if inventory_path.read_text(encoding="utf-8") != inventory:
        inventory_path.write_text(inventory, encoding="utf-8")
        rewritten.append(INVENTORY_PATH)

    direct_flow = _load_direct_flow(root=root)
    seal_map_path = root / DIRECT_FLOW_PATH
    original = seal_map_path.read_text(encoding="utf-8")
    updated = _with_current_source_seals(
        original=original, direct_flow=direct_flow, root=root
    )
    if updated != original:
        seal_map_path.write_text(updated, encoding="utf-8")
        rewritten.append(DIRECT_FLOW_PATH)
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: derived from this script's location)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="regenerate the inventory and rewrite the seal map, then re-check",
    )
    args = parser.parse_args()

    if args.fix:
        for path in fix(repo_root=args.repo_root):
            print(f"rewrote {path}")

    seal_errors, other_errors = check(repo_root=args.repo_root)
    if not seal_errors and not other_errors:
        print("candidate certificate seals match the tree")
        return 0
    for error in seal_errors + other_errors:
        print(error)
    if seal_errors:
        print(
            "\nA certified source changed. Reseal both files with\n"
            "    pixi run python tests/candidate_certificate/check_seals.py --fix\n"
            "and stage the rewritten inventory and seal map."
        )
        return 1
    print(
        "\nThe direct-flow verifier reports an error a reseal cannot repair: an "
        "AST corridor pin, not a byte seal.\n"
        "If the edit to each certified source below was intended, re-anchor "
        "those corridors with\n"
        "    pixi run python tests/candidate_certificate/repin_corridors.py \\\n"
        "        --changed-source <path> [--changed-source <path> ...]\n"
        "which refuses to write if a pin outside the named sources drifted. "
        "Field tuples, enum bodies\n"
        "and binding counts it leaves alone: those are reviewed and edited by "
        "hand.\n"
        "See docs/development/certification.md. Never widen an anchor to make "
        "a changed route green."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
