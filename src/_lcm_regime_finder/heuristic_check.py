"""Independent checks for a bounded graph-layout search result.

This module checks *candidate membership*, integer costs and a recommendation.
It never calls a producer search, a complete-family recurrence, a signature
assignment enumerator, or a cost-layer frontier solver.  A PASS here has no
source-semantic or global-optimality meaning.  The production entry point must
also replay the original source with ``prepare_decision_context`` and obtain an
``adopt`` decision for the particular proposed representation.

The retained archive is checked as a finite set.  The producer's claim that it
kept every nondominated point it evaluated follows from its online pruning
invariant, not from this checker reconstructing the history of its search.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json

from .frontier_check import (
    IndependentFamily,
    METRICS,
    _Bounded,
    _antichain,
    _independent_endpoint_preference,
)


SCHEMA = "regime-heuristic-frontier/v1"
HEURISTIC_ENDPOINT_POLICY = (
    "Heuristic search may omit admissible layouts. By default the recommendation "
    "defers extra final-decision-date regimes used only to omit newly inactive "
    "coordinates; allow_final_only_split explicitly overrides this preference. "
    "Retained cosmetic candidates remain visible and receive the same local checks."
)
RECOMMENDATION_RULE = (
    "fixed-reference positive weighted scalar cost; strict improvement meeting "
    "the relative threshold; deterministic cost/layout/policy ties; retain the "
    "original otherwise"
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _vector(cost):
    return tuple(cost[name] for name in METRICS)


def _layout_key(plan):
    return _json([(block["cells"], block["option_by_cell"])
                  for block in plan["blocks"]])


def _original_cost(family):
    """Price the *unchanged* original axes, once per original regime.

    Original interfaces need not be offered by dependency reduction.  We thus
    read the original cells directly.  A conflicting raw group has no cost in
    this allocation model; keeping the original remains meaningful even then.
    This is not permission to invent a merged grid or to drop a context.
    """
    groups = defaultdict(list)
    for cell in family.cells.values():
        groups[cell["regime"]].append(cell)
    total = dict.fromkeys(METRICS, 0)
    for cells in groups.values():
        family.bound()
        if len({family.role(cell) for cell in cells}) != 1:
            return None
        if family.allocation["kind"] == "global_union":
            states, actions = family.global_axes
        else:
            states, actions = (family.union(cell[kind] for cell in cells)
                               for kind in ("states", "actions"))
        if states is None or actions is None:
            return None
        ns, na = family.sizes(states), family.sizes(actions)
        members = {cell["id"] for cell in cells}
        crossings = sum(source in members and target not in members
                        for source, target in family.pairs)
        total["work"] += (len(members) + crossings * family.boundary_unit) * ns * na
        total["storage"] += len(members) * ns
        total["clarity"] += family.weights["block"]
    return total


def _objective(result, original_cost):
    objective = result["objective"]
    weights = {name: Fraction(str(objective["weights"][name])) for name in METRICS}
    scales = objective["reference_scales"]
    threshold = Fraction(str(objective.get("minimum_improvement", "0")))
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("objective weights must be strictly positive")
    if (set(scales) != set(METRICS)
            or any(type(scales[name]) is not int or scales[name] <= 0 for name in METRICS)):
        raise ValueError("reference scales must be positive integer costs")
    if not 0 <= threshold < 1:
        raise ValueError("minimum_improvement must lie in [0, 1)")
    reference_kind = objective["reference_kind"]
    if reference_kind == "original":
        if original_cost is None or scales != {name: max(1, original_cost[name]) for name in METRICS}:
            raise ValueError("reference scales do not match unchanged original costs")
    elif reference_kind == "unit_fallback":
        if original_cost is not None or scales != dict.fromkeys(METRICS, 1):
            raise ValueError("invalid unit fallback reference")
    elif reference_kind != "explicit":
        raise ValueError("unknown fixed reference kind")
    return weights, scales, threshold


def _choose(family, result, allowed_ids):
    original_cost = _original_cost(family)
    weights, scales, threshold = _objective(result, original_cost)
    points = result["candidates"]
    override = result["recommendation"]["endpoint_preference"]["allow_final_only_split"]
    if type(override) is not bool:
        raise ValueError("allow_final_only_split must be a boolean")
    deferred = sorted(point["id"] for point in points
                      if _independent_endpoint_preference(family, point))
    allowed = None if allowed_ids is None else set(allowed_ids)
    eligible = [point for point in points
                if (allowed is None or point["id"] in allowed)
                and (override or point["id"] not in deferred)]

    def score(cost):
        return sum(weights[name] * Fraction(cost[name], scales[name]) for name in METRICS)

    original_score = score(original_cost) if original_cost is not None else None
    winner = min(eligible, key=lambda point: (
        score(point["cost"]), _vector(point["cost"]), _layout_key(point),
        point["projection_policy"], point["id"])) if eligible else None
    worthwhile = (winner is not None and original_score is not None
                  and score(winner["cost"]) < original_score
                  and score(winner["cost"]) <= original_score * (1 - threshold))
    return {
        "graph_id": winner["id"] if worthwhile else "retain-original",
        "operation": "representation_specification" if worthwhile else "retain_original",
        "score": str(score(winner["cost"])) if worthwhile else (
            str(original_score) if original_score is not None else None),
        "original_score": str(original_score) if original_score is not None else None,
        "weights": {name: str(weights[name]) for name in METRICS},
        "reference_scales": dict(scales),
        "minimum_improvement": str(threshold),
        "endpoint_preference": {
            "allow_final_only_split": override,
            "deferred_graph_ids": deferred,
            "reason": HEURISTIC_ENDPOINT_POLICY,
        },
        "rule": RECOMMENDATION_RULE,
        "authority": "cost_only",
    }


def choose_candidate(analysis, result, *, allowed_ids=None):
    """Independently rank an already checked retained candidate set.

    ``allowed_ids`` is a selection restriction, not a certificate.  Only the
    caller that *just replayed* source and candidate decisions may attach
    source-checked authority to the returned record.  An empty set explicitly
    retains the original.  No model-size or elapsed-time ceiling is introduced.
    """
    family = IndependentFamily(analysis, seconds=float("inf"))
    return _choose(family, result, allowed_ids)


def _check_block_details(family, point):
    """Bind display/cost-detail fields in addition to admitted_plan's contract."""
    errors = []
    policy = next(record["option_by_cell"] for record in family.analysis["projection_policies"]
                  if record["id"] == point["projection_policy"])
    for block in point["blocks"]:
        ids = block["cells"]
        expected = family.block(ids, policy)
        if expected is None:
            continue  # The ordinary admission check reports this contradiction.
        cells = [family.cells[cell] for cell in ids]
        if block.get("original_regimes") != sorted({cell["regime"] for cell in cells}):
            errors.append("incorrect displayed original regimes")
        if block.get("allocation_kind") != family.allocation["kind"]:
            errors.append("incorrect displayed allocation kind")
        if block.get("peak_live_storage") is not None:
            errors.append("structural storage was promoted to measured peak storage")
        ns, na = family.sizes(expected["state_axes"]), family.sizes(expected["action_axes"])
        members = set(ids)
        crossings = sum(source in members and target not in members for source, target in family.pairs)
        characteristics = { _json(projection) for cell in ids
                            for projection in family.options[cell][policy[cell]]["state_projection"].values()
                            if projection["kind"] == "characteristic" }
        detail = {
            "enumeration": len(ids) * ns * na,
            "boundary": crossings * ns * na * family.boundary_unit,
            "boundary_incidences": crossings,
            "state_rows_per_atom": ns,
            "action_rows_per_atom": na,
            "dispatch_clauses": len({cell["regime"] for cell in cells}) - 1,
            "characteristic_count": len(characteristics),
        }
        if block.get("cost_detail") != detail:
            errors.append("incorrect displayed block cost details")
    return errors


def check_heuristic_frontier(analysis, result, *, limits=None):
    """Verify supplied candidates without reconstructing the search family.

    Optional ``limits['seconds']`` bounds checking cooperatively.  Exhaustion
    returns UNRESOLVED and cannot authorize a new graph.  Other exact-search
    limits have no role here, because no grouping or interval enumeration runs.
    """
    errors, checked = [], 0
    limits = limits or {}
    try:
        seconds = limits.get("seconds", float("inf"))
        if seconds is None:
            seconds = float("inf")
        family = IndependentFamily(analysis, seconds=seconds)
        family.bound()
        if len(family.cells) != len(analysis["declarations"]["cells"]):
            errors.append("duplicate original admission cells")
        if result.get("schema") != SCHEMA or result.get("status") != "HEURISTIC":
            errors.append("heuristic schema/status changed")
        if result.get("analysis_sha256") != family.analysis_digest:
            errors.append("analysis identity digest changed")
        if result.get("allocation_contract") != family.allocation:
            errors.append("allocation contract changed")
        if result.get("clarity_weights") != family.weights:
            errors.append("clarity costs changed")
        if result.get("endpoint_clarity_policy") != HEURISTIC_ENDPOINT_POLICY:
            errors.append("fixed final-period preference changed")
        search = result["search"]
        if search.get("complete") is not False or search.get("optimality_gap") != "unknown":
            errors.append("heuristic search claimed completeness or an unproved gap")
        if result.get("certificate", {}).get("complete", False) is not False:
            errors.append("heuristic result borrowed an exact complete certificate")
        if result.get("frontier_complete_verified", False) is not False:
            errors.append("heuristic result claimed a complete frontier")
        original = result["original"]
        if original.get("operation") != "retain_original" or original.get("cost") != _original_cost(family):
            errors.append("unchanged original baseline changed")
        points = result["candidates"]
        ids = [point["id"] for point in points]
        if len(ids) != len(set(ids)) or "retain-original" in ids:
            errors.append("duplicate or reserved candidate graph ID")
        for point in points:
            family.bound()
            plan_errors = family.admitted_plan(point)
            if not plan_errors:
                plan_errors.extend(_check_block_details(family, point))
            errors.extend(point["id"] + ": " + issue for issue in plan_errors)
            checked += 1
        by_id = {point["id"]: point for point in points}
        frontier = result["frontier"]
        if any(by_id.get(point["id"]) != point for point in frontier):
            errors.append("frontier point is not the retained candidate with this identity")
        vectors = [_vector(point["cost"]) for point in frontier]
        if len(vectors) != len(set(vectors)) or sorted(vectors) != _antichain(vectors):
            errors.append("reported frontier is not a unique nondominated cost set")
        if sorted(vectors) != _antichain(_vector(point["cost"]) for point in points):
            errors.append("reported frontier omits a nondominated retained cost")
        representatives = {}
        for point in points:
            value = _vector(point["cost"])
            rank = (_layout_key(point), point["projection_policy"], point["id"])
            if value not in representatives or rank < representatives[value][0]:
                representatives[value] = (rank, point["id"])
        if any(representatives[_vector(point["cost"])][1] != point["id"] for point in frontier):
            errors.append("frontier equal-cost layout tie differs from retained candidate ordering")
        evaluated = search.get("evaluated_candidates")
        if type(evaluated) is not int or evaluated < len(points):
            errors.append("invalid candidate accounting")
        # Candidate enumeration history is intentionally not replayed.  The
        # returned set's exact antichain is distinct from a family cover proof.
        expected = _choose(family, result, None)
        if result["recommendation"] != expected:
            errors.append("cost recommendation differs from independent fixed-reference ranking")
        family.bound()
        return {
            "status": "FAIL" if errors else "PASS", "errors": errors,
            "unresolved": [], "checking_complete": True,
            "frontier_complete_verified": False,
            "checked_complete_plans": checked,
            "retained_antichain_verified": not errors,
            "examined_history_replayed": False,
            "original_baseline_verified": not errors,
            "source_semantics": "NOT_CHECKED_HERE",
            "dense_cartesian_rows_evaluated": 0,
            "scope": "independent candidate admission, exact integer costs, retained antichain and fixed-reference choice; original-source semantics and global optimality are separate",
        }
    except _Bounded as exc:
        return {"status": "UNRESOLVED", "errors": errors,
                "unresolved": [str(exc)], "checking_complete": False,
                "frontier_complete_verified": False,
                "checked_complete_plans": checked,
                "source_semantics": "NOT_CHECKED_HERE"}
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, StopIteration) as exc:
        return {"status": "FAIL", "errors": errors + ["malformed heuristic result: " + str(exc)],
                "unresolved": [], "checking_complete": False,
                "frontier_complete_verified": False,
                "checked_complete_plans": checked,
                "source_semantics": "NOT_CHECKED_HERE"}
