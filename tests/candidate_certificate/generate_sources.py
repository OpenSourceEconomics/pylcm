#!/usr/bin/env python3
"""Generate the candidate certificate's exact source inventory from its AST."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

CERTIFICATE_PATH = "tests/test_grid_search_candidate_certificate.py"
INVENTORY_PATH = "tests/candidate_certificate/sources.json"
REQUIRED_PROFILES = ("fast", "certified")


def sha256_file(path: Path) -> str:
    """Hash UTF-8 text with checkout-independent LF newlines."""
    canonical = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def derive_source_paths(certificate: Path) -> tuple[str, ...]:
    """Derive unique repo-relative sources from literal ``_parse`` obligations."""
    tree = ast.parse(certificate.read_text(encoding="utf-8"), filename=str(certificate))
    paths: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "_parse":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        value = node.args[0].value
        if isinstance(value, str):
            paths.append(Path(value).as_posix())
    unique = tuple(sorted(set(paths)))
    if not unique:
        raise ValueError("certificate contains no literal _parse source obligations")
    return unique


def inventory_digest(sources: list[dict[str, str]]) -> str:
    """Hash the canonical ordered source-record representation."""
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_inventory(repo_root: Path) -> dict[str, Any]:
    """Build the canonical inventory from certificate obligations and source bytes."""
    root = repo_root.resolve()
    certificate = root / CERTIFICATE_PATH
    sources = [
        {"path": relative, "sha256": sha256_file(root / relative)}
        for relative in derive_source_paths(certificate)
    ]
    digest = inventory_digest(sources)
    return {
        "schema_version": "1",
        "certificate": CERTIFICATE_PATH,
        "generation_rule": (
            "unique sorted literal _parse(<repo-relative path>) call arguments "
            "in the certificate AST"
        ),
        "sources": sources,
        "source_inventory_sha256": digest,
        "derived_policy": {
            "profiles": {
                profile: {
                    "candidate_sources": sources,
                    "source_inventory_sha256": digest,
                    "source_count": len(sources),
                }
                for profile in REQUIRED_PROFILES
            }
        },
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    generated = build_inventory(root)
    inventory_path = root / INVENTORY_PATH

    if args.write:
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(canonical_json(generated), encoding="utf-8")

    if args.check:
        try:
            committed = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            result = {
                "schema_version": "1",
                "result": "fail",
                "inventory": INVENTORY_PATH,
                "error": str(error),
            }
            print(canonical_json(result), end="")
            return 1
        matches = committed == generated
        result = {
            "schema_version": "1",
            "result": "pass" if matches else "fail",
            "inventory": INVENTORY_PATH,
            "certificate": CERTIFICATE_PATH,
            "derived_source_paths": [item["path"] for item in generated["sources"]],
            "source_inventory_sha256": generated["source_inventory_sha256"],
            "matches_generated_inventory": matches,
        }
        print(canonical_json(result), end="")
        return 0 if matches else 1

    rendered = canonical_json(generated)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
