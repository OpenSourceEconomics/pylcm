#!/usr/bin/env python3
"""Unified standard-library verifier for the candidate-set certificate.

The verifier joins one AST-derived source inventory to its committed JSON, the
source bytes, an optional explicit profile contract, an optional derived policy,
and the JAX-backed certificate matrix. It also owns the independent scalar
masked-argmax oracle and the mutation controls used by the bounded repair.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from generate_sources import (
        CERTIFICATE_PATH,
        INVENTORY_PATH,
        REQUIRED_PROFILES,
        build_inventory,
        derive_source_paths,
        inventory_digest,
        sha256_file,
    )
except ModuleNotFoundError:  # Imported as tests.candidate_certificate.verify.
    from tests.candidate_certificate.generate_sources import (
        CERTIFICATE_PATH,
        INVENTORY_PATH,
        REQUIRED_PROFILES,
        build_inventory,
        derive_source_paths,
        inventory_digest,
        sha256_file,
    )

SourceRecord = dict[str, str]
# Anchor plus source set as declared by one contract or policy profile.
ProfileDeclaration = dict[str, Any]
Mask = tuple[bool, ...]
_Q_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def canonical_json(payload: dict[str, Any]) -> str:
    """Render deterministic UTF-8 JSON text."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def nonempty_feasibility_masks(n_candidates: int) -> tuple[Mask, ...]:
    """Enumerate every nonempty Boolean mask in increasing bit order."""
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    return tuple(
        tuple(bool(bits & (1 << index)) for index in range(n_candidates))
        for bits in range(1, 1 << n_candidates)
    )


def reference_masked_argmax(
    values: Sequence[float], mask: Sequence[bool]
) -> tuple[int, float]:
    """Return the unique feasible maximizer using an explicit scalar loop.

    Empty support and ties are refused. The certificate asserts a unique
    maximizer and must not silently introduce a tie convention.
    """
    if len(values) != len(mask) or not values:
        raise ValueError("values and mask must be nonempty and have equal length")
    best_index: int | None = None
    # Only ever read once `best_index` is set, so the initial value is never compared.
    best_value = 0.0
    tied = False
    for index, (raw_value, feasible) in enumerate(zip(values, mask, strict=True)):
        if not feasible:
            continue
        value = float(raw_value)
        if best_index is None or value > best_value:
            best_index = index
            best_value = value
            tied = False
        elif value == best_value:
            tied = True
    if best_index is None:
        raise ValueError("all-infeasible masks are outside the certificate domain")
    if tied:
        raise ValueError("reference maximizer is not unique")
    return best_index, best_value


def rank_vectors(n_candidates: int) -> tuple[tuple[float, ...], ...]:
    """Enumerate strict orderings, one per candidate, each won by that candidate.

    Ordering `j` gives candidate `j` the largest value and assigns the remaining
    values by increasing distance from it, so every ordering is a strict total order
    with no ties and every candidate is the strict maximum of exactly one ordering.
    A single fixed ordering can only ever place one candidate in the winning role,
    which leaves an omission of any other candidate invisible under full support.
    """
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    return tuple(
        tuple(
            float(n_candidates - ((index - winner) % n_candidates))
            for index in range(n_candidates)
        )
        for winner in range(n_candidates)
    )


def run_rank_neighborhood_controls() -> dict[str, dict[str, Any]]:
    """Show that a fixed ordering cannot see an omission of a non-winning candidate.

    The control omits flat cell 0 whenever every candidate is feasible. Under the
    ascending ordering candidate 0 never wins a multi-feasible mask, so the omission
    is invisible; under the ordering that candidate 0 wins, it changes the published
    winner and value.
    """
    n = len(_Q_VALUES)
    orderings = rank_vectors(n)
    all_feasible: Mask = (True,) * n

    def omit_first(mask: Mask) -> Mask:
        return (False, *mask[1:]) if all(mask) else mask

    def outcome(values: Sequence[float]) -> dict[str, Any]:
        reference_index, reference_value = reference_masked_argmax(values, all_feasible)
        mutated_index, mutated_value = _winner_after(values, all_feasible, omit_first)
        return {
            "values": list(values),
            "reference_index": reference_index,
            "reference_value": reference_value,
            "mutated_index": mutated_index,
            "mutated_value": mutated_value,
            "value_gap": reference_value - mutated_value,
            "detected": (reference_index, reference_value)
            != (mutated_index, mutated_value),
        }

    ascending = outcome(_Q_VALUES)
    candidate_zero_wins = outcome(orderings[0])
    return {
        "fixed_ascending_ordering": ascending,
        "ordering_won_by_candidate_zero": candidate_zero_wins,
        "every_candidate_wins_exactly_one_ordering": {
            "winners": [
                reference_masked_argmax(values, all_feasible)[0] for values in orderings
            ],
            "covers_every_candidate": sorted(
                reference_masked_argmax(values, all_feasible)[0] for values in orderings
            )
            == list(range(n)),
        },
    }


def _winner_after(
    values: Sequence[float], mask: Mask, transform: Callable[[Mask], Mask]
) -> tuple[int, float]:
    return reference_masked_argmax(values, transform(mask))


def _pinned_witness() -> dict[str, Any]:
    before = (True,) * len(_Q_VALUES)
    after = (*before[:-1], False)
    solve_index, solve_value = reference_masked_argmax(_Q_VALUES, before)
    simulate_index, simulate_value = reference_masked_argmax(_Q_VALUES, after)
    return {
        "q_values_flat": list(_Q_VALUES),
        "all_feasible_before_mutation": list(before),
        "all_feasible_after_mutation": list(after),
        "solve_winner_flat_index": solve_index,
        "mutated_simulate_winner_flat_index": simulate_index,
        "value_gap": solve_value - simulate_value,
        "policy_disagreement_mass_for_affected_row": float(
            solve_index != simulate_index
        ),
    }


def run_mask_mutation_controls() -> dict[str, dict[str, Any]]:
    """Exercise MT6 and a distinct generated intermediate-mask omission."""
    masks = nonempty_feasibility_masks(len(_Q_VALUES))
    all_feasible = next(mask for mask in masks if all(mask))
    intermediate = next(
        mask for mask in masks if sum(mask) == 3 and mask[-1] and not all(mask)
    )

    def mt6(mask: Mask) -> Mask:
        return (*mask[:-1], False) if all(mask) else mask

    def generated(mask: Mask) -> Mask:
        return (
            (*mask[:-1], False)
            if sum(mask) == 3 and mask[-1] and not all(mask)
            else mask
        )

    def outcome(mask: Mask, transform: Callable[[Mask], Mask]) -> dict[str, Any]:
        reference_index, reference_value = reference_masked_argmax(_Q_VALUES, mask)
        mutated_index, mutated_value = _winner_after(_Q_VALUES, mask, transform)
        return {
            "mask": list(mask),
            "reference_index": reference_index,
            "reference_value": reference_value,
            "mutated_index": mutated_index,
            "mutated_value": mutated_value,
            "value_gap": reference_value - mutated_value,
            "detected": (reference_index, reference_value)
            != (mutated_index, mutated_value),
        }

    pinned = outcome(all_feasible, mt6)
    pinned["witness"] = _pinned_witness()
    generated_result = outcome(intermediate, generated)
    generated_result["neither_one_hot_nor_all_feasible"] = (
        1 < sum(intermediate) < len(intermediate)
    )
    generated_result["detected_mask_count"] = sum(
        _winner_after(_Q_VALUES, mask, generated)
        != reference_masked_argmax(_Q_VALUES, mask)
        for mask in masks
        if sum(mask) == 3 and mask[-1] and not all(mask)
    )
    return {
        "all_feasible_last_cell": pinned,
        "intermediate_three_cell": generated_result,
    }


def _load_json(path: Path) -> Any:
    """Read JSON while rejecting duplicate object keys."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def _records(raw: Any, *, label: str) -> list[SourceRecord]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    records: list[SourceRecord] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] is not an object")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"{label}[{index}] must contain string path and sha256")
        records.append(
            {
                "path": Path(path).as_posix(),
                "sha256": digest.removeprefix("sha256:"),
            }
        )
    return records


def _records_from_inventory(payload: dict[str, Any]) -> list[SourceRecord]:
    return _records(payload.get("sources"), label="sources.json sources")


def _counter(records: Iterable[SourceRecord]) -> Counter[tuple[str, str]]:
    return Counter((item["path"], item["sha256"]) for item in records)


def _compare_records(
    label: str,
    expected: list[SourceRecord],
    observed: list[SourceRecord],
) -> tuple[list[str], set[str]]:
    """Compare exact multisets, retaining duplicates and naming every path."""
    errors: list[str] = []
    offending: set[str] = set()
    expected_paths = Counter(item["path"] for item in expected)
    observed_paths = Counter(item["path"] for item in observed)
    for path in sorted(set(expected_paths) | set(observed_paths)):
        if expected_paths[path] != observed_paths[path]:
            errors.append(
                f"{label}: path multiplicity mismatch for {path}: "
                f"expected {expected_paths[path]}, observed {observed_paths[path]}"
            )
            offending.add(path)
    expected_records = _counter(expected)
    observed_records = _counter(observed)
    for path, digest in sorted(set(expected_records) | set(observed_records)):
        if expected_records[(path, digest)] != observed_records[(path, digest)]:
            errors.append(
                f"{label}: digest record mismatch for {path} sha256:{digest}: "
                f"expected {expected_records[(path, digest)]}, "
                f"observed {observed_records[(path, digest)]}"
            )
            offending.add(path)
    return errors, offending


def _contract_profiles(path: Path) -> dict[str, ProfileDeclaration]:
    """Parse anchor and source declarations from the narrow profile-contract grammar.

    The primary `source` / `source_digest` pair is the inventory **anchor**: the
    generated inventory file and the digest of its bytes. `additional_sources` carries
    the certified source set itself. Anchoring on the inventory file is what lets the
    compiled policy — which can express one path and one digest — bind the whole set,
    because the inventory's bytes change whenever any certified source's digest does.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    sources: dict[str, list[SourceRecord]] = {}
    anchors: dict[str, SourceRecord | None] = {}
    current_profile: str | None = None
    candidate_indent: int | None = None
    primary_path: str | None = None
    primary_digest: str | None = None
    additional_indent: int | None = None

    def finish_primary() -> None:
        nonlocal primary_path, primary_digest
        if current_profile is None:
            primary_path = None
            primary_digest = None
            return
        if primary_path is not None or primary_digest is not None:
            anchors[current_profile] = {
                "path": Path(primary_path or "<missing-anchor-path>").as_posix(),
                "sha256": (primary_digest or "<missing-anchor-digest>").removeprefix(
                    "sha256:"
                ),
            }
        primary_path = None
        primary_digest = None

    for line in [*lines, ""]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 2 and stripped.endswith(":"):
            finish_primary()
            current_profile = stripped[:-1]
            sources.setdefault(current_profile, [])
            anchors.setdefault(current_profile, None)
            candidate_indent = None
            additional_indent = None
            continue
        if current_profile is None:
            continue
        if stripped == "candidate_set_error:" and indent > 2:
            finish_primary()
            candidate_indent = indent
            additional_indent = None
            continue
        if candidate_indent is None:
            continue
        if stripped and indent <= candidate_indent:
            finish_primary()
            candidate_indent = None
            additional_indent = None
            continue
        field_indent = candidate_indent + 2
        if additional_indent is not None and stripped and indent < additional_indent:
            additional_indent = None
        if indent == field_indent and stripped.startswith("source:"):
            primary_path = stripped.split(":", 1)[1].strip().strip('"')
        elif indent == field_indent and stripped.startswith("source_digest:"):
            primary_digest = stripped.split(":", 1)[1].strip().strip('"')
        elif indent == field_indent and stripped == "additional_sources:":
            finish_primary()
            additional_indent = field_indent + 2
        elif (
            additional_indent is not None
            and indent == additional_indent
            and ":" in stripped
        ):
            raw_path, raw_digest = stripped.split(":", 1)
            sources.setdefault(current_profile, []).append(
                {
                    "path": raw_path.strip().strip('"'),
                    "sha256": raw_digest.strip().strip('"').removeprefix("sha256:"),
                }
            )
    finish_primary()
    return {
        profile: {"anchor": anchors.get(profile), "sources": records}
        for profile, records in sources.items()
        if profile in REQUIRED_PROFILES or records or anchors.get(profile)
    }


def _policy_profiles(path: Path) -> dict[str, ProfileDeclaration]:
    """Read a derived policy in either the rich or the compiled projection.

    A bundler compiles the contract down to one `candidate_source` path and one
    `candidate_source_digest` per profile, which is why the contract anchors on the
    inventory file: the compiled projection then still names the whole set by proxy.
    The richer `candidate_sources` list is read too, so the same reader serves the
    inventory's own derived policy.
    """
    payload = _load_json(path)
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, dict):
        raise ValueError("derived policy must contain a profiles object")
    result: dict[str, ProfileDeclaration] = {}
    for profile, entry in profiles.items():
        if not isinstance(entry, dict):
            raise ValueError(f"derived policy profile {profile!r} is not an object")
        records: list[SourceRecord] = []
        if entry.get("candidate_sources") is not None:
            records = _records(
                entry.get("candidate_sources"),
                label=f"derived policy {profile} candidate_sources",
            )
        anchor: SourceRecord | None = None
        anchor_path = entry.get("candidate_source")
        anchor_digest = entry.get("candidate_source_digest")
        if anchor_path is not None or anchor_digest is not None:
            if not isinstance(anchor_path, str) or not isinstance(anchor_digest, str):
                raise ValueError(
                    f"derived policy {profile} candidate_source and "
                    "candidate_source_digest must both be strings"
                )
            anchor = {
                "path": Path(anchor_path).as_posix(),
                "sha256": anchor_digest.removeprefix("sha256:"),
            }
        result[profile] = {"anchor": anchor, "sources": records}
    return result


def _anchor_errors(
    label: str, expected: SourceRecord, observed: SourceRecord | None
) -> tuple[list[str], set[str]]:
    """Require an inventory anchor to be present and exactly equal."""
    if observed is None:
        return ([f"{label}: no inventory anchor declared"], {INVENTORY_PATH})
    if observed != expected:
        message = (
            f"{label}: inventory anchor mismatch: expected "
            f"{expected['path']} sha256:{expected['sha256']}, observed "
            f"{observed['path']} sha256:{observed['sha256']}"
        )
        return (
            [message],
            {INVENTORY_PATH, observed["path"]},
        )
    return ([], set())


def _certificate_matrix_errors(certificate: Path) -> tuple[list[str], set[str]]:
    tree = ast.parse(certificate.read_text(encoding="utf-8"), filename=str(certificate))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    def direct_calls(node: ast.AST) -> set[str]:
        found: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                found.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                found.add(child.func.attr)
        return found

    def transitive_calls(name: str, seen: set[str] | None = None) -> set[str]:
        seen = set() if seen is None else seen
        if name in seen or name not in functions:
            return set()
        seen.add(name)
        result = direct_calls(functions[name])
        for called in tuple(result):
            result |= transitive_calls(called, seen)
        return result

    suffix = "_matches_reference_over_every_nonempty_feasibility_mask"
    required = {
        f"test_singleton_solve{suffix}": "_solve_mask_case",
        f"test_singleton_simulate{suffix}": "_simulate_mask_case",
        f"test_collective_solve{suffix}": "_solve_mask_case",
        f"test_collective_simulate{suffix}": "_simulate_mask_case",
    }
    errors: list[str] = []
    offending: set[str] = set()

    generated_matrices = {
        "_NONEMPTY_FEASIBILITY_MASKS": "nonempty_feasibility_masks(len(_CANDIDATES))",
        "_RANK_VECTORS": "rank_vectors(len(_CANDIDATES))",
    }
    for constant, expected_assignment in sorted(generated_matrices.items()):
        assignment: ast.expr | None = None
        for top_level in tree.body:
            if (
                isinstance(top_level, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == constant
                    for target in top_level.targets
                )
            ) or (
                isinstance(top_level, ast.AnnAssign)
                and isinstance(top_level.target, ast.Name)
                and top_level.target.id == constant
            ):
                assignment = top_level.value
        if assignment is None or ast.unparse(assignment) != expected_assignment:
            errors.append(
                f"certificate matrix: {constant} must be assigned "
                f"directly from {expected_assignment}"
            )
            offending.add(CERTIFICATE_PATH)
    for name, route_marker in sorted(required.items()):
        node = functions.get(name)
        if node is None:
            errors.append(f"certificate matrix: missing {name}")
            offending.add(CERTIFICATE_PATH)
            continue
        for constant in sorted(generated_matrices):
            direct_loops = [
                loop
                for loop in ast.walk(node)
                if isinstance(loop, ast.For)
                and isinstance(loop.iter, ast.Name)
                and loop.iter.id == constant
            ]
            if len(direct_loops) != 1:
                errors.append(
                    f"certificate matrix: {name} must iterate directly and exactly "
                    f"once over {constant}"
                )
                offending.add(CERTIFICATE_PATH)
        calls = transitive_calls(name)
        for marker in ("reference_masked_argmax", route_marker):
            if marker not in calls:
                errors.append(f"certificate matrix: {name} does not reach {marker}")
                offending.add(CERTIFICATE_PATH)
    source = ast.unparse(tree)
    for marker in (
        "nonempty_feasibility_masks(len(_CANDIDATES))",
        "rank_vectors(len(_CANDIDATES))",
        "sources.json",
        "derive_source_paths(Path(__file__))",
    ):
        if marker not in source:
            errors.append(f"certificate architecture: missing {marker}")
            offending.add(CERTIFICATE_PATH)
    for node in tree.body:
        value: ast.expr | None = None
        target: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and target.id == "CERTIFIED_SOURCES"
            and isinstance(value, ast.Tuple)
        ):
            errors.append(
                "certificate architecture: CERTIFIED_SOURCES is still a literal tuple"
            )
            offending.add(CERTIFICATE_PATH)
    return errors, offending


def verify_repository(
    *,
    repo_root: Path,
    inventory_path: Path | None = None,
    contract: Path | None = None,
    policy: Path | None = None,
) -> dict[str, Any]:
    """Verify every supplied representation against one generated inventory."""
    root = repo_root.resolve()
    inventory = (
        inventory_path.resolve()
        if inventory_path is not None
        else root / INVENTORY_PATH
    )
    errors: list[str] = []
    offending: set[str] = set()
    details: dict[str, Any] = {}
    try:
        committed_payload = _load_json(inventory)
        if not isinstance(committed_payload, dict):
            raise ValueError("sources.json root is not an object")
        committed = _records_from_inventory(committed_payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {
            "ok": False,
            "result": "fail",
            "errors": [f"inventory: {error}"],
            "offending_paths": [INVENTORY_PATH],
            "details": {},
        }

    generated_payload = build_inventory(root)
    generated = _records_from_inventory(generated_payload)
    new_errors, new_paths = _compare_records(
        "certificate AST vs committed sources.json", generated, committed
    )
    errors += new_errors
    offending |= new_paths

    duplicate_paths = sorted(
        path
        for path, count in Counter(item["path"] for item in committed).items()
        if count != 1
    )
    for path in duplicate_paths:
        errors.append(f"sources.json: duplicate source path {path}")
        offending.add(path)

    actual_inventory_digest = inventory_digest(committed)
    if committed_payload.get("source_inventory_sha256") != actual_inventory_digest:
        errors.append(
            "sources.json: source_inventory_sha256 mismatch: expected "
            f"{actual_inventory_digest}, observed "
            f"{committed_payload.get('source_inventory_sha256')}"
        )
        offending.add(INVENTORY_PATH)

    internal_policy = committed_payload.get("derived_policy")
    internal_profiles = (
        internal_policy.get("profiles") if isinstance(internal_policy, dict) else None
    )
    internal_policy_records: dict[str, list[SourceRecord]] = {}
    if not isinstance(internal_profiles, dict):
        errors.append("derived policy: missing profiles object in sources.json")
        offending.add(INVENTORY_PATH)
    else:
        for profile in REQUIRED_PROFILES:
            entry = internal_profiles.get(profile)
            if not isinstance(entry, dict):
                errors.append(f"derived policy: missing profile {profile}")
                offending.add(INVENTORY_PATH)
                continue
            try:
                observed = _records(
                    entry.get("candidate_sources"),
                    label=f"derived policy {profile} candidate_sources",
                )
            except ValueError as error:
                errors.append(str(error))
                offending.add(INVENTORY_PATH)
                continue
            internal_policy_records[profile] = observed
            new_errors, new_paths = _compare_records(
                f"sources.json derived policy {profile}", committed, observed
            )
            errors += new_errors
            offending |= new_paths
            if entry.get("source_inventory_sha256") != actual_inventory_digest:
                errors.append(
                    f"derived policy: {profile} inventory digest differs from "
                    "sources.json"
                )
                offending.add(INVENTORY_PATH)
            if entry.get("source_count") != len(committed):
                errors.append(
                    f"derived policy: {profile} source count differs from sources.json"
                )
                offending.add(INVENTORY_PATH)

    disk_records: list[SourceRecord] = []
    for item in committed:
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"source bytes: missing {item['path']}")
            offending.add(item["path"])
            continue
        disk_records.append({"path": item["path"], "sha256": sha256_file(path)})
    new_errors, new_paths = _compare_records(
        "sources.json vs source bytes", committed, disk_records
    )
    errors += new_errors
    offending |= new_paths

    expected_anchor: SourceRecord = {
        "path": INVENTORY_PATH,
        "sha256": sha256_file(inventory),
    }

    contract_profiles: dict[str, ProfileDeclaration] = {}
    if contract is not None:
        try:
            contract_profiles = _contract_profiles(contract.resolve())
        except (OSError, ValueError) as error:
            errors.append(f"profile contract: {error}")
            offending.add(str(contract))
        for profile in REQUIRED_PROFILES:
            declaration = contract_profiles.get(profile, {})
            new_errors, new_paths = _compare_records(
                f"profile contract {profile}", committed, declaration.get("sources", [])
            )
            errors += new_errors
            offending |= new_paths
            new_errors, new_paths = _anchor_errors(
                f"profile contract {profile}",
                expected_anchor,
                declaration.get("anchor"),
            )
            errors += new_errors
            offending |= new_paths

    policy_profiles: dict[str, ProfileDeclaration] = {}
    if policy is not None:
        try:
            policy_profiles = _policy_profiles(policy.resolve())
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"derived policy: {error}")
            offending.add(str(policy))
        for profile in REQUIRED_PROFILES:
            declaration = policy_profiles.get(profile)
            if declaration is None:
                errors.append(f"explicit derived policy: missing profile {profile}")
                offending.add(str(policy))
                continue
            new_errors, new_paths = _anchor_errors(
                f"explicit derived policy {profile}",
                expected_anchor,
                declaration.get("anchor"),
            )
            errors += new_errors
            offending |= new_paths
            declared_sources = declaration.get("sources", [])
            if declared_sources:
                new_errors, new_paths = _compare_records(
                    f"explicit derived policy {profile}", committed, declared_sources
                )
                errors += new_errors
                offending |= new_paths

    matrix_errors, matrix_paths = _certificate_matrix_errors(root / CERTIFICATE_PATH)
    errors += matrix_errors
    offending |= matrix_paths
    details.update(
        {
            "ast_source_paths": list(derive_source_paths(root / CERTIFICATE_PATH)),
            "committed_sources": committed,
            "source_inventory_sha256": actual_inventory_digest,
            "inventory_anchor": expected_anchor,
            "source_bytes": disk_records,
            "internal_derived_policy_profiles": internal_policy_records,
            "contract_profiles": contract_profiles,
            "explicit_policy_profiles": policy_profiles,
        }
    )
    return {
        "ok": not errors,
        "result": "pass" if not errors else "fail",
        "errors": errors,
        "offending_paths": sorted(offending),
        "details": details,
    }


def run_source_set_mutation_controls(*, repo_root: Path) -> dict[str, dict[str, Any]]:
    """Check the exact set-comparator against the required perturbation family."""
    clean = _records_from_inventory(build_inventory(repo_root.resolve()))
    if len(clean) != 2:
        raise ValueError(
            "source-set controls require exactly two baseline sources, "
            f"found {len(clean)}"
        )
    primary, secondary = clean
    fake_path = "src/_lcm/candidate_certificate/undeclared.py"
    mutants: dict[str, tuple[list[SourceRecord], str]] = {
        "delete": ([primary], secondary["path"]),
        "add": (
            [*clean, {"path": fake_path, "sha256": "0" * 64}],
            fake_path,
        ),
        "duplicate": ([*clean, primary.copy()], primary["path"]),
        "rename": (
            [
                primary,
                {
                    "path": secondary["path"] + ".renamed",
                    "sha256": secondary["sha256"],
                },
            ],
            secondary["path"] + ".renamed",
        ),
        "byte_change": (
            [primary, {"path": secondary["path"], "sha256": "0" * 64}],
            secondary["path"],
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (records, expected_path) in mutants.items():
        errors, paths = _compare_records(f"self-test {name}", clean, records)
        result[name] = {
            "rejected": bool(errors) and expected_path in paths,
            "offending_paths": sorted(paths),
            "errors": errors,
        }
    return result


def certificate_neighborhood_coverage(*, repo_root: Path) -> dict[str, Any]:
    errors, paths = _certificate_matrix_errors(repo_root.resolve() / CERTIFICATE_PATH)
    return {
        "ok": not errors,
        "errors": errors,
        "offending_paths": sorted(paths),
        "nonempty_mask_count": len(nonempty_feasibility_masks(len(_Q_VALUES))),
    }


def _self_tests(repo_root: Path) -> tuple[dict[str, Any], bool]:
    source_controls = run_source_set_mutation_controls(repo_root=repo_root)
    mask_controls = run_mask_mutation_controls()
    coverage = certificate_neighborhood_coverage(repo_root=repo_root)
    source_green = all(item["rejected"] for item in source_controls.values())
    mask_green = all(item["detected"] for item in mask_controls.values())
    mask_green &= bool(
        mask_controls["intermediate_three_cell"]["neither_one_hot_nor_all_feasible"]
    )
    mask_green &= (
        mask_controls["all_feasible_last_cell"]["witness"] == _pinned_witness()
    )
    all_feasible = dict(mask_controls["all_feasible_last_cell"])
    all_feasible.pop("witness", None)
    intermediate = mask_controls["intermediate_three_cell"]
    one_hot_unchanged = all(
        reference_masked_argmax(_Q_VALUES, mask)
        == _winner_after(
            _Q_VALUES,
            mask,
            lambda candidate_mask: (
                (*candidate_mask[:-1], False) if all(candidate_mask) else candidate_mask
            ),
        )
        for mask in nonempty_feasibility_masks(len(_Q_VALUES))
        if sum(mask) == 1
    )
    mask_green &= one_hot_unchanged
    legacy_masks = {
        "all_feasible": all_feasible,
        "intermediate": intermediate,
        "one_hot_masks_unchanged_by_mt6": one_hot_unchanged,
        "generated_intermediate_masks_detected": intermediate["detected_mask_count"],
        "nonempty_mask_count": len(nonempty_feasibility_masks(len(_Q_VALUES))),
    }
    payload = {
        "source_set_perturbations": source_controls,
        "mask_neighborhood_perturbations": legacy_masks,
        "mt6_all_feasible": {
            "detected": mask_controls["all_feasible_last_cell"]["detected"],
            "witness": mask_controls["all_feasible_last_cell"]["witness"],
        },
        "generated_intermediate": intermediate,
        "certificate_neighborhood_coverage": coverage,
        "all_controls_sensitive": source_green and mask_green and coverage["ok"],
    }
    return payload, bool(payload["all_controls_sensitive"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    verification = verify_repository(
        repo_root=root,
        inventory_path=args.inventory,
        contract=args.contract,
        policy=args.policy,
    )
    errors = list(verification["errors"])
    self_tests: dict[str, Any] | None = None
    if args.self_test:
        self_tests, controls_green = _self_tests(root)
        if not controls_green:
            errors.append("self-test: one or more mutation controls did not fire")

    payload = {
        "schema_version": "1",
        "result": "pass" if not errors else "fail",
        "repo_root": str(root),
        "inventory": str(
            args.inventory.resolve() if args.inventory else root / INVENTORY_PATH
        ),
        "contract": str(args.contract.resolve()) if args.contract else None,
        "policy": str(args.policy.resolve()) if args.policy else None,
        "errors": errors,
        "offending_paths": verification["offending_paths"],
        "details": verification["details"],
        "reference": {
            "q_values_flat": list(_Q_VALUES),
            "nonempty_mask_count": len(nonempty_feasibility_masks(len(_Q_VALUES))),
            "all_infeasible_out_of_domain": True,
            "ties_refused": True,
        },
        "self_test": self_tests,
        "self_tests": self_tests,
    }
    print(canonical_json(payload), end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
