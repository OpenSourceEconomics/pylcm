#!/usr/bin/env python3
"""Port MT6 to a repository-root feasibility-neighborhood control.

Exit 1 means the multi-feasible omission remains admitted. Exit 0 means the
unified verifier detects both the pinned all-feasible omission and a generated
intermediate-mask omission. Exit 2 means an instrument error.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CERTIFICATE = Path("tests/test_grid_search_candidate_certificate.py")
VERIFIER = Path("tests/candidate_certificate/verify.py")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _defs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _calls(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            found.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            found.add(child.func.attr)
    return found


def _transitive(
    name: str,
    defs: dict[str, ast.FunctionDef],
    seen: set[str] | None = None,
) -> set[str]:
    seen = set() if seen is None else seen
    if name in seen or name not in defs:
        return set()
    seen.add(name)
    direct = _calls(defs[name])
    out = set(direct)
    for called in direct:
        out |= _transitive(called, defs, seen)
    return out


def _fixture_constraint(tree: ast.Module, name: str) -> str | None:
    node = _defs(tree).get(name)
    if node is None:
        return None
    for call in ast.walk(node):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_build_model"
        ):
            for keyword in call.keywords:
                if keyword.arg == "constraints":
                    return ast.unparse(keyword.value)
    return None


def _finite_witness() -> dict[str, Any]:
    q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    before = [True] * 6
    after = [True, True, True, True, True, False]
    solve = max((i for i, ok in enumerate(before) if ok), key=q.__getitem__)
    simulate = max((i for i, ok in enumerate(after) if ok), key=q.__getitem__)
    return {
        "q_values_flat": q,
        "all_feasible_before_mutation": before,
        "all_feasible_after_mutation": after,
        "solve_winner_flat_index": solve,
        "mutated_simulate_winner_flat_index": simulate,
        "value_gap": q[solve] - q[simulate],
        "policy_disagreement_mass_for_affected_row": float(solve != simulate),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    certificate = repo / CERTIFICATE
    verifier = repo / VERIFIER
    try:
        witness = _finite_witness()
        if not verifier.exists():
            tree = ast.parse(certificate.read_text(encoding="utf-8"))
            defs = _defs(tree)
            sweep_names = {
                "test_simulation_routes_to_every_declared_candidate",
                "test_simulation_routes_a_household_to_every_declared_candidate",
            }
            sweeps = sorted(
                name
                for name in sweep_names
                if name in defs and "simulate" in _transitive(name, defs)
            )
            fixtures: dict[str, str | None] = {}
            for name in sweeps:
                for argument in defs[name].args.args:
                    if argument.arg.endswith("model"):
                        constraint = _fixture_constraint(tree, argument.arg)
                        if constraint is not None:
                            fixtures[argument.arg] = constraint
            only_one_hot = bool(fixtures) and all(
                "_only_target" in value
                for value in fixtures.values()
                if value is not None
            )
            unique_max_simulate = any(
                "unique_maximizer_model"
                in [argument.arg for argument in defs[name].args.args]
                for name in sweeps
            )
            gap = (
                len(sweeps) == 2
                and only_one_hot
                and not unique_max_simulate
                and witness["solve_winner_flat_index"] == 5
                and witness["mutated_simulate_winner_flat_index"] == 4
                and witness["value_gap"] == 1.0
            )
            _emit(
                {
                    "schema_version": "1",
                    "result": "gap_reproduced" if gap else "gap_not_reproduced",
                    "mutation": (
                        "suppress the final simulate candidate only under a "
                        "multi-feasible mask"
                    ),
                    "parametrized_simulate_sweeps": sweeps,
                    "simulate_sweep_fixtures": fixtures,
                    "simulate_sweeps_use_unique_maximizer_model": unique_max_simulate,
                    "all_parametrized_simulate_sweeps_use_only_target_constraint": (
                        only_one_hot
                    ),
                    "finite_witness": witness,
                }
            )
            return 1 if gap else 0

        process = subprocess.run(
            [sys.executable, str(verifier), "--repo-root", str(repo), "--self-test"],
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError:
            payload = {"raw_stdout": process.stdout, "raw_stderr": process.stderr}
        raw_self_tests = payload.get("self_test")
        self_tests: dict[str, Any] = (
            raw_self_tests if isinstance(raw_self_tests, dict) else {}
        )
        masks = self_tests.get("mask_neighborhood_perturbations", {})
        mt6 = masks.get("all_feasible", {})
        generated = masks.get("intermediate", {})
        closed = (
            process.returncode == 0
            and mt6.get("detected") is True
            and generated.get("detected") is True
            and mt6.get("reference_index") == witness["solve_winner_flat_index"]
            and mt6.get("mutated_index")
            == witness["mutated_simulate_winner_flat_index"]
            and mt6.get("value_gap") == witness["value_gap"]
            and masks.get("nonempty_mask_count") == 63
            and int(masks.get("generated_intermediate_masks_detected", 0)) > 0
            and self_tests.get("all_controls_sensitive") is True
        )
        _emit(
            {
                "schema_version": "1",
                "result": "gap_not_reproduced" if closed else "gap_reproduced",
                "verifier_exit": process.returncode,
                "mt6_all_feasible": mt6,
                "generated_intermediate": generated,
                "finite_witness": witness,
                "verifier_payload": payload,
            }
        )
        return 0 if closed else 1
    except Exception as error:
        _emit(
            {
                "schema_version": "1",
                "result": "instrument_error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
