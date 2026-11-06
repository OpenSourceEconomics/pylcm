"""Full interval search without retaining a layout for every candidate.

Each same-lane dependency is a self-read or a next-date edge. Consequently an
interval's outgoing incidences are its cross-lane incidences plus its final
next-date incidence. Bit sets give its allocated union axes. The original full
segment evaluator still materializes and checks every selected segment.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import time
from typing import Any

from . import core
from .expressions import AdmissionError


class SegmentIndex:
    def __init__(self, problem: dict, model: dict, lane: list[str]):
        self.problem, self.model, self.lane = problem, model, lane
        self.atoms = [model["by_id"][key] for key in lane]
        self.state_names = sorted(problem["states"])
        self.action_names = sorted(problem["actions"])
        state_bits = {name: 1 << i for i, name in enumerate(self.state_names)}
        action_bits = {name: 1 << i for i, name in enumerate(self.action_names)}
        self.state_masks = [sum(state_bits[x] for x in a["live_states"]) for a in self.atoms]
        self.action_masks = [sum(action_bits[x] for x in a["free_actions"]) for a in self.atoms]
        self.global_states = sum(state_bits[x] for x in problem.get("implementation", {}).get("global_state_axes", self.state_names))
        self.global_actions = sum(action_bits[x] for x in problem.get("implementation", {}).get("global_action_axes", self.action_names))
        positions = {key: i for i, key in enumerate(lane)}
        outgoing = defaultdict(set)
        for edge in model["edges"]:
            if edge["source"] in positions:
                outgoing[positions[edge["source"]]].add(edge["target"])
        self.cross = [0]
        self.forward = []
        for i in range(len(lane)):
            foreign = sum(target not in positions for target in outgoing[i])
            same = [positions[target] for target in outgoing[i] if target in positions]
            if any(target not in {i, i + 1} for target in same):
                raise AdmissionError("Compact segment contract requires same-lane self/next-date dependencies")
            self.cross.append(self.cross[-1] + foreign)
            self.forward.append(int(i + 1 in same))
        self.ends = [len(lane)] * len(lane)
        role = lambda a: (core.canonical(a["role"]), a["routing_owner"], a["terminal"], a["allocation"])
        for i in range(len(lane) - 2, -1, -1):
            self.ends[i] = (self.ends[i + 1] if role(self.atoms[i]) == role(self.atoms[i + 1])
                            and (problem.get("implementation", {}).get("allow_sparse_admission", False)
                                 or self.atoms[i + 1]["age_index"] == self.atoms[i]["age_index"] + 1) else i + 1)
        self.single_allowed = [core.segment(problem, model, lane, i, i + 1)[0] is not None for i in range(len(lane))]

    @lru_cache(maxsize=4096)
    def state_size(self, mask: int) -> int:
        return math.prod(core.size(self.problem["states"][name]) for i, name in enumerate(self.state_names)
                         if mask & (1 << i) and core.solve_axis(self.problem["states"][name]))

    @lru_cache(maxsize=4096)
    def action_size(self, mask: int) -> int:
        return math.prod(core.size(self.problem["actions"][name]) for i, name in enumerate(self.action_names) if mask & (1 << i))

    def from_start(self, start: int):
        sm = am = 0
        implementation = self.problem.get("implementation", {})
        unit = implementation.get("boundary_unit", 1)
        if type(unit) is not int or unit < 0:
            raise AdmissionError("boundary_unit must be a nonnegative integer")
        for stop in range(start + 1, len(self.lane) + 1):
            sm |= self.state_masks[stop - 1]
            am |= self.action_masks[stop - 1]
            if stop > self.ends[start] or (stop == start + 1 and not self.single_allowed[start]):
                yield stop, None
                continue
            ns = self.state_size(self.global_states if self.atoms[start]["allocation"] == "global_union" else sm)
            na = self.action_size(self.global_actions if self.atoms[start]["allocation"] == "global_union" else am)
            if (ns > implementation.get("max_block_state_entries", math.inf)
                    or na > implementation.get("max_block_action_entries", math.inf)
                    or ns * na > implementation.get("max_block_entries", math.inf)):
                yield stop, None
                continue
            crossing = self.cross[stop] - self.cross[start] + self.forward[stop - 1]
            enumeration = (stop - start) * ns * na
            boundary_rows = crossing * ns * na
            yield stop, {"enumeration": enumeration, "boundary": unit * boundary_rows,
                         "boundary_rows": boundary_rows, "boundary_edges": crossing,
                         "work": enumeration + unit * boundary_rows, "blocks": 1, "storage": (stop - start) * ns}


def _hash_record(hasher: Any, key: str, start: int, stop: int, cost: dict) -> None:
    hasher.update(json.dumps([key, start, stop, cost], sort_keys=True, separators=(",", ":")).encode())
    hasher.update(b"\n")


def _materialize(problem: dict, model: dict, key: str, start: int, stop: int, cost: dict | None = None) -> dict:
    result, reason = core.segment(problem, model, model["lanes"][key], start, stop)
    if result is None or (cost is not None and result["cost"] != cost):
        raise AdmissionError("Compact/full segment contract mismatch: " + str(reason))
    return result


def find_compact(problem: dict, budget: dict | None = None) -> dict:
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "INVALID", "problem_sha256": core.digest(problem),
              "layers": {name: {"status": "NOT_RUN", "checks": 0, "notes": []} for name in core.LAYERS}, "diagnostics": []}
    active = "admission"
    try:
        model = core.prepare(problem)
        report["layers"]["admission"] = {"status": "PASS", "checks": len(model["atoms"]),
            "notes": ["Constructed graph support admitted; model construction and numerical solve validation are recorded separately."]}
        active = "semantic"
        original = core.original_partition(problem, model)
        original_ok = core.original_admissibility(problem, model, original)
        report["layers"]["semantic"] = {"status": "PASS", "checks": len(model["atoms"]) + len(model["edges"]),
            "notes": ["Conditional IR projection from conservative original dependency facts; opaque functions/operators remain unchanged. This is not an independent source/backend equivalence proof."]}
        report["layers"]["numerical"] = {"status": "NOT_RUN", "checks": model["numerical_admission"]["rows_checked"],
            "notes": ["Native solves, simulations and opaque numerical rows were not executed."]}
        report["layers"]["source_adapter"] = {"status": "PASS", "checks": len(model["atoms"]),
            "notes": ["Actual constructed pylcm graph and finalized regime declarations consumed; builder syntax is irrelevant. Callback dependency extraction remains a separate trust assumption."]}
        report["layers"]["implementation"] = {"status": "PASS", "checks": len(model["atoms"]),
            "notes": ["Declared graph-lane, actor, terminal, ownership and allocation restrictions retained."]}
        active = "efficiency"
        budget = budget or {}
        maximum = int(budget.get("candidate_segments", problem.get("limits", {}).get("candidate_segments", 5_000_000)))
        seconds = float(budget.get("seconds", problem.get("limits", {}).get("seconds", 1800)))
        if maximum < 0 or seconds < 0:
            raise AdmissionError("Invalid negative search budget")
        count = admitted = 0
        complete = True
        potentials, cuts = {}, {}
        refined = []
        feasible = True
        family = hashlib.sha256()
        for key in sorted(model["lanes"]):
            lane = model["lanes"][key]
            index = SegmentIndex(problem, model, lane)
            n = len(lane)
            best = [None] * (n + 1)
            best[n] = [0, 0]
            choices = {}
            for i in range(n - 1, -1, -1):
                option = None
                for j, cost in index.from_start(i):
                    if count >= maximum or time.perf_counter() - started >= seconds:
                        complete = False
                        break
                    count += 1
                    if cost is None:
                        continue
                    admitted += 1
                    _hash_record(family, key, i, j, cost)
                    if best[j] is not None:
                        candidate = (cost["work"] + best[j][0], 1 + best[j][1], j)
                        if option is None or candidate < option:
                            option = candidate
                            choices[i] = (j, cost)
                if not complete:
                    break
                if option is not None:
                    best[i] = list(option[:2])
            potentials[key] = best
            if not complete:
                break
            if best[0] is None:
                feasible = False
                continue
            i = 0
            path = []
            while i < n:
                j, cost = choices[i]
                refined.append(_materialize(problem, model, key, i, j, cost))
                path.append([i, j])
                i = j
            cuts[key] = path
        chosen = original
        if complete and feasible:
            oc, rc = core.total_cost(original), core.total_cost(refined)
            if not original_ok["admissible"] or (rc["work"], rc["blocks"]) <= (oc["work"], oc["blocks"]):
                chosen = refined
        elif complete and not original_ok["admissible"]:
            raise AdmissionError("No admissible partition exists")
        branch = "refined" if chosen is refined else "original"
        for i, selected in enumerate(chosen):
            selected["id"] = f"R{i:03d}"
        cost = core.total_cost(chosen)
        public = core._public_model(model)
        report.update({"status": "OPTIMAL" if complete else "INCOMPLETE", "analysis": public,
            "original": {"cost": core.total_cost(original), "regimes": original, **original_ok},
            "recommended": {"cost": cost, "regimes": chosen, "external_report_tags": problem.get("external_report_tags", []),
                "incumbent_admissible": branch == "refined" or original_ok["admissible"],
                "transitions": core.reported_transitions(model, chosen)},
            "search": {"complete": complete, "segments_evaluated": count, "admitted_segments": admitted,
                "optimality_scope": "Original grouping or every admissible contiguous interval layout in the declared fixed graph lanes and semantic interfaces.",
                "objective": ["declared enumeration plus source-row boundary work", "block count", "earliest cuts; refined branch on exact score tie"],
                "lower_bound": [cost["work"], cost["blocks"]] if complete else [0, 0], "gap": 0 if complete else None,
                "incomplete_reason": None if complete else "explicit candidate/time budget exhausted",
                "counterfactual_segments": [], "rejected_segments": [],
                "counterfactual_storage": "All intervals regenerated by the verifier; streaming family digest replaces stored candidate layouts."},
            "certificate": {"version": 2, "problem_sha256": core.digest(problem), "analysis_sha256": core.digest(public),
                "complete": complete, "branch": branch, "potentials": potentials if complete else {},
                "chosen_segments": cuts if branch == "refined" else {}, "score": [cost["work"], cost["blocks"]],
                "family_sha256": family.hexdigest(), "family_encoding": "newline-separated canonical JSON [lane,start,stop,cost], reverse start order"}})
        report["diagnostics"] = core._diagnostics(problem, model, chosen)
        report["layers"]["efficiency"] = {"status": "PASS" if complete else "UNRESOLVED", "checks": count,
            "notes": ["Full interval family searched; scores stored compactly. Work is a captured-grid proxy, not measured hardware time or production memory."]}
    except (AdmissionError, ValueError, TypeError, KeyError) as exc:
        report["layers"][active] = {"status": "FAIL", "checks": 1, "notes": [str(exc)]}
        report["error"] = str(exc)
    report["measured_search_seconds"] = time.perf_counter() - started
    return report


def verify_compact(problem: dict, report: dict) -> dict:
    """Check all inequalities and an attaining path without running optimization."""
    errors = []
    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)
    try:
        cert = report["certificate"]
        require(cert["problem_sha256"] == core.digest(problem) == report["problem_sha256"], "Problem identity mismatch")
        if errors:
            return {"status": "FAIL", "errors": errors, "optimum_verified": False}
        model = core.prepare(problem)
        public = core._public_model(model)
        require(cert["analysis_sha256"] == core.digest(public) and report["analysis"] == public, "Derived interfaces altered")
        original = core.original_partition(problem, model)
        original_ok = core.original_admissibility(problem, model, original)
        oc = core.total_cost(original)
        require(report["original"]["cost"] == oc and report["original"]["admissible"] == original_ok["admissible"], "Original cost/admission altered")
        if not cert["complete"]:
            for i, selected in enumerate(original):
                selected["id"] = f"R{i:03d}"
            require(report["status"] == "INCOMPLETE" and not report["search"]["complete"], "Incomplete search mislabeled")
            require(cert["branch"] == "original" and report["recommended"]["regimes"] == original and report["recommended"]["cost"] == oc, "Incomplete incumbent altered")
            return {"status": "FAIL" if errors else "INCOMPLETE", "errors": errors, "optimum_verified": False,
                    "incumbent_verified": not errors and original_ok["admissible"]}
        require(report["status"] == "OPTIMAL" and report["search"]["complete"], "Complete status mismatch")
        refined = []
        refined_score = [0, 0]
        all_reachable = True
        family = hashlib.sha256()
        count = inequalities = 0
        for key in sorted(model["lanes"]):
            lane = model["lanes"][key]
            index = SegmentIndex(problem, model, lane)
            n = len(lane)
            pot = cert["potentials"].get(key)
            require(isinstance(pot, list) and len(pot) == n + 1, "Potential length mismatch")
            if not isinstance(pot, list) or len(pot) != n + 1:
                continue
            require(pot[-1] == [0, 0], "Invalid sink potential")
            require(all(v is None or isinstance(v, list) and len(v) == 2 and all(type(x) is int and x >= 0 for x in v) for v in pot), "Invalid potential entry")
            if errors:
                return {"status": "FAIL", "errors": errors, "optimum_verified": False}
            reachable = {n}
            attaining = {}
            for i in range(n - 1, -1, -1):
                has_suffix = False
                for j, cost in index.from_start(i):
                    count += 1
                    if cost is None:
                        continue
                    _hash_record(family, key, i, j, cost)
                    if j in reachable:
                        has_suffix = True
                    if pot[j] is not None:
                        require(pot[i] is not None, "Reachable suffix falsely labeled unreachable")
                        if pot[i] is not None:
                            rhs = [cost["work"] + pot[j][0], 1 + pot[j][1]]
                            require(tuple(pot[i]) <= tuple(rhs), "Lower-bound inequality violated")
                            inequalities += 1
                            if pot[i] == rhs and i not in attaining:
                                attaining[i] = (j, cost)
                if has_suffix:
                    reachable.add(i)
                require((pot[i] is not None) == (i in reachable), "False feasibility in potential table")
            if 0 not in reachable:
                all_reachable = False
                continue
            refined_score[0] += pot[0][0]
            refined_score[1] += pot[0][1]
            cursor = 0
            path = []
            while cursor < n:
                require(cursor in attaining, "Potential lower bound lacks an attaining path")
                if cursor not in attaining:
                    break
                j, cost = attaining[cursor]
                refined.append(_materialize(problem, model, key, cursor, j, cost))
                path.append([cursor, j])
                cursor = j
            if cert["branch"] == "refined":
                require(cert["chosen_segments"].get(key) == path, "Attaining path/earliest tie rule mismatch")
        require(family.hexdigest() == cert["family_sha256"], "Admissible family digest mismatch")
        branch = "refined" if all_reachable and (not original_ok["admissible"] or tuple(refined_score) <= (oc["work"], oc["blocks"])) else "original"
        require(cert["branch"] == branch, "Incorrect original/refined branch")
        selected = refined if branch == "refined" else original
        for i, value in enumerate(selected):
            value["id"] = f"R{i:03d}"
        cost = core.total_cost(selected)
        require(report["recommended"]["regimes"] == selected and report["recommended"]["cost"] == cost, "Selected layout or cost altered")
        require(cert["score"] == [cost["work"], cost["blocks"]] == report["search"]["lower_bound"] and report["search"]["gap"] == 0, "Score/lower bound/gap mismatch")
        require(report["recommended"]["transitions"] == core.reported_transitions(model, selected), "Boundary projections altered")
        require(report["recommended"].get("external_report_tags", []) == problem.get("external_report_tags", []), "External associations altered")
        return {"status": "FAIL" if errors else "PASS", "errors": errors, "optimum_verified": not errors,
                "segments_reconstructed": count, "bellman_inequalities_checked": inequalities,
                "score": [cost["work"], cost["blocks"]],
                "scope": "Exact score optimum over the complete declared fixed-interface interval/graph-lane family; conditional source bridge, no native numerical or hardware-speed certificate."}
    except Exception as exc:
        return {"status": "FAIL", "errors": errors + [type(exc).__name__ + ": " + str(exc)], "optimum_verified": False}
