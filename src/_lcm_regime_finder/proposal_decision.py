"""Source-bound, independently replayed decisions for individual proposals.

These records authorize a representation *specification* relative to the supplied
source contract. They do not certify emitted Python, paper economics or measured
speed. No caller-supplied PASS flag is accepted. Source semantics are replayed once
per batch; each candidate's admission and structural costs are checked separately.
The complete-family optimizer is deliberately not rerun here.
"""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy

from .frontier_check import IndependentFamily, _Bounded
from .semantic_check import check_semantics


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class _DecisionContext:
    """In-memory independent replay result, never deserialized from a status row."""

    def __init__(self, original, analysis, semantic):
        self._original = original
        self._analysis = analysis
        self._semantic = deepcopy(semantic)
        self.source_digest = semantic.get("source_sha256") or _digest(original)
        self.declaration_digest = semantic.get("normalized_sha256") or _digest(analysis["declarations"])
        self.analysis_digest = self._analysis_digest(analysis)

    def _analysis_digest(self, analysis):
        return _digest({key: value for key, value in analysis.items() if key != "declarations"}
                       | {"declarations_sha256": self.declaration_digest})

    @property
    def semantic(self):
        # The enclosing producer may serialize or annotate its copy; this does
        # not turn a serialized PASS flag back into independently checked data.
        return deepcopy(self._semantic)

    def validate(self, original, analysis):
        if (original is not self._original or analysis is not self._analysis
                or _digest(original) != self.source_digest
                or _digest(analysis["declarations"]) != self.declaration_digest
                or self._analysis_digest(analysis) != self.analysis_digest):
            raise ValueError("decision context does not match unchanged source and analysis")


def prepare_decision_context(original, analysis, *, check_coverage=True):
    """Run the independent semantic checker once for analysis and decisions.

    Pass the returned context directly in memory. Its semantic property supplies
    the ordinary global result; later decisions filter scoped diagnostics. A
    different or mutated source/analysis requires a fresh independent replay.
    """
    semantic = check_semantics(original, analysis["declarations"], analysis,
                               check_coverage=check_coverage)
    return _DecisionContext(original, analysis, semantic)


def _selected(policies, index, plan):
    policy = policies[plan["projection_policy"]]
    return {cid: index[cid][oid] for cid, oid in policy.items()}


def _unchanged_interface(cell, option, callbacks):
    """Sufficient syntactic condition for opaque-primitive congruence.

    Equality of every argument, binding and original callable is sufficient for
    any pure deterministic interpretation. We require the complete local axes
    and every original provider; no unavailable argument is guessed or dropped.
    Bound arguments have already passed the independent source check. A failed
    source/binding check is never discharged by this predicate.
    """
    if option.get("states") != cell.get("states") or option.get("actions") != cell.get("actions"):
        return False
    if option.get("state_projection") != {s: {"kind": "identity", "target": s} for s in cell.get("states", {})}:
        return False
    if option.get("action_reconstruction") != {a: {"kind": "identity", "source": a} for a in cell.get("actions", {})}:
        return False
    for phase in ("solve", "simulation"):
        providers = cell.get("phase_functions", {}).get(phase, cell.get("functions", {}))
        records = option.get("function_interfaces", {}).get(phase, {})
        if set(records) != set(providers):
            return False
        for name, callback_id in providers.items():
            record, callback = records[name], callbacks[callback_id]
            arguments = callback.get("arguments", [])
            if not callback.get("identity"):
                return False
            if record.get("original_callback") != callback_id or record.get("source_identity") != callback.get("identity"):
                return False
            if record.get("original_arguments") != arguments or record.get("unused_arguments"):
                return False
            if record.get("implementation", {}).get("kind") != "source_bound_partial_application":
                return False
            offered = list(record.get("arguments", [])) + list(record.get("bound_arguments", {}))
            if len(offered) != len(set(offered)) or sorted(offered) != sorted(arguments):
                return False
    return True


def _callback_cells(declarations):
    uses = {key: set() for key in declarations["callbacks"]}
    for cell in declarations["cells"]:
        values = list(cell.get("functions", {}).values())
        for providers in cell.get("phase_functions", {}).values():
            values.extend(providers.values())
        values.extend(c.get("callback") for c in cell.get("consumers", []))
        for value in values:
            if isinstance(value, str) and value in uses:
                uses[value].add(cell["id"])
    for relation in declarations["relations"]:
        values = list(relation.get("callbacks", []))
        for update in relation.get("state_updates", {}).values():
            values.append(update.get("callback") if isinstance(update, dict) else update)
        for value in values:
            if isinstance(value, str) and value in uses:
                uses[value].update([relation["source"], *relation.get("targets", [])])
    # A declaration wrapper owns the calls of its captured members. An empty
    # direct-use set is not evidence that a nested callback is unused.
    pending = list(uses)
    while pending:
        parent = pending.pop()
        for member in declarations["callbacks"][parent].get("members", []):
            if member in uses and not uses[parent] <= uses[member]:
                uses[member].update(uses[parent])
                pending.append(member)
    return uses


def _interpretation_obligation_callbacks(original, declarations, obligations):
    """Bind only interpretation gaps to exact original frozen callables.

    A parser's inability to lower a callable does not prevent congruence of an
    unchanged application. This deliberately does not establish its meaning or
    purity, replace the source checker, or discharge capture/variadic/query
    premises. Supported Model callbacks retain their existing pure-function
    contract. Resolve the input pointer independently before using locality.
    """
    paths = {item.get("source_path") for item in obligations
             if item.get("kind") == "callback_interpretation"
             and isinstance(item.get("source_path"), str)}
    matches = {}
    for callback_id, callback in declarations["callbacks"].items():
        path = callback.get("source_path")
        if path not in paths or not path.startswith("/"):
            continue
        raw = original
        try:
            for token in path.split("/")[1:]:
                token = token.replace("~1", "/").replace("~0", "~")
                raw = raw[int(token)] if isinstance(raw, list) else raw[token]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if not isinstance(raw, dict) or raw.get("kind") != "callable":
            continue
        if (raw.get("capture_contract") not in {"inspection-only-frozen-bindings-1", "inspection-only-frozen-bindings-2"}
                or raw.get("capture_status", "COMPLETE") != "COMPLETE"
                or raw.get("variadic") or raw.get("obligations")
                or not raw.get("identity_sha256") or not raw.get("binding_sha256")
                or callback.get("original") != raw
                or callback.get("identity") != raw["identity_sha256"]
                or callback.get("arguments") != raw.get("arguments")
                or callback.get("source") != raw.get("source")
                or callback.get("bindings") != raw.get("bindings", {})
                or callback.get("status") != "opaque"
                or callback.get("expression") is not None):
            continue
        provenance = callback.get("provenance", {})
        if provenance.get("source_path") != path or provenance.get("sha256") != _digest(raw):
            continue
        matches.setdefault(path, []).append(callback_id)
    # Ambiguous identity is not resolved by choosing whichever use looks safe.
    return {path: ids[0] for path, ids in matches.items() if len(ids) == 1}


def _relevant(diagnostic, selected, policy_id):
    scope = diagnostic.get("scope", "source")
    if scope == "family":
        return False
    if scope == "option":
        option = selected.get(diagnostic.get("cell"), {})
        return diagnostic.get("option") in (None, option.get("id"))
    if scope == "policy":
        return diagnostic.get("policy") in (None, policy_id)
    return True


def _changes(cells, selected, plan):
    changes = []
    for cid, option in selected.items():
        cell = cells[cid]
        projections = option.get("state_projection", {})
        actions = option.get("action_reconstruction", {})
        if (option.get("states") != cell["states"] or option.get("actions") != cell["actions"]
                or any(p.get("kind") != "identity" for p in projections.values())
                or any(a.get("kind") != "identity" for a in actions.values())):
            changes.append({"cell": cid, "original_regime": cell["regime"],
                            "period": cell["period"], "age": cell.get("age"),
                            "original_states": cell["states"], "states": option.get("states"),
                            "original_actions": cell["actions"], "actions": option.get("actions"),
                            "state_projection": projections, "action_reconstruction": actions})
    original_groups = {}
    for cid, cell in cells.items():
        original_groups.setdefault(cell["regime"], set()).add(cid)
    changed_cells = {row["cell"] for row in changes}
    bands = []
    # Exact admissions are authoritative. We never interpolate missing calendar ages.
    for block in plan.get("blocks", []):
        members = set(block["cells"])
        regimes = {cells[cid]["regime"] for cid in members if cid in cells}
        kinds = []
        if members & changed_cells:
            kinds.append("interface")
        if len(regimes) > 1:
            kinds.append("merge")
        if any(members & original_groups[regime] != original_groups[regime] for regime in regimes):
            kinds.append("split")
        if kinds:
            bands.append({"cells": block["cells"], "periods": block.get("periods", []),
                          "ages": block.get("ages", []), "original_regimes": sorted(regimes),
                          "change_kinds": kinds})
    return changes, bands


def _preserved_context(original, cells, plan, obligation):
    """Carry an opaque original decision context by identity, never equivalence.

    The result is only for representation specifications: the original actor,
    observation, commitment and discount context must be carried unchanged.
    State/action factorization, typed incidence and all other semantic premises
    remain independently checked; this rule does not discharge any of them.
    """
    path = obligation.get("source_path")
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    raw = original
    try:
        for token in path.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            raw = raw[int(token)] if isinstance(raw, list) else raw[token]
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    original_digest = _digest(raw)
    affected = {cid for cid, cell in cells.items() if cell.get("source_path") == path}
    if not affected:
        return None
    protected = ("regime", "role", "ownership", "stage", "terminal",
                 "declaration_type", "structural_roles", "solver_identity")
    records = []
    counts = dict.fromkeys(affected, 0)
    for cid in affected:
        cell = cells[cid]
        provenance = cell.get("provenance", {})
        if (cell.get("role") is not None or provenance.get("source_path") != path
                or provenance.get("sha256") != original_digest):
            return None
        records.append({key: deepcopy(cell.get(key)) for key in
                        ("id", "regime", "period", "age", "source_path", *protected[1:])})
    for block in plan.get("blocks", []):
        members = block.get("cells", [])
        if not affected.intersection(members):
            continue
        if (not members or any(cid not in affected for cid in members)
                or len({_digest([cells[cid].get(key) for key in protected]) for cid in members}) != 1):
            return None
        for cid in members:
            counts[cid] += 1
    if not all(count == 1 for count in counts.values()):
        return None
    return {"source_path": path, "original_declaration_sha256": original_digest,
            "cells": sorted(records, key=lambda row: row["id"]),
            "rule": "same original regime and dated context; carry opaque actor/information/commitment/discount authority unchanged",
            "not_proved": "no cross-context economic equivalence, emitted code, or extra observation loss is authorized"}


def build_decisions(original, analysis, scenarios, *, seconds=900, _context=None):
    """Return retain plus per-plan decisions; scenario indexes are zero based.

    The budget is cooperative between checks. The caller must run the entire
    operation inside the existing process-tree resource boundary. A saved
    incomplete frontier can contain an adoptable local proposal without a global
    optimality claim. Source/option/policy diagnostics are independently scoped;
    unscoped legacy diagnostics remain conservatively global.
    """
    started = time.perf_counter()
    deadline = started + max(0, seconds)
    declarations = analysis["declarations"]
    if _context is not None:
        if not isinstance(_context, _DecisionContext):
            raise TypeError("decision context must be a live independent replay context")
        _context.validate(original, analysis)
        source_digest, declaration_digest = _context.source_digest, _context.declaration_digest
    else:
        source_digest, declaration_digest = _digest(original), _digest(declarations)
    cells = {c["id"]: c for c in declarations["cells"]}
    callbacks = declarations["callbacks"]
    callback_uses = _callback_cells(declarations)
    interpretation_callbacks = _interpretation_obligation_callbacks(
        original, declarations, analysis.get("obligations", []))
    policies = {p["id"]: p["option_by_cell"] for p in analysis["projection_policies"]}
    option_index = {cid: {o["id"]: o for o in row["options"]}
                    for cid, row in analysis["interfaces"].items()}
    retain_candidate = {"operation": "retain_original", "source_snapshot_sha256": source_digest}
    decisions = [{"schema": "regime-proposal-decision/v1", "graph_id": "retain-original",
        "scenario_index": None, "allocation_contract_sha256": None,
        "source_digest_kind": "canonical-json-sha256",
        "source_snapshot_sha256": source_digest, "declarations_sha256": declaration_digest,
        "candidate_sha256": _digest(retain_candidate), "operation": "retain_original",
        "disposition": "adopt", "reasons": ["Keep the exact supplied original artifact; no transformation is authorized."],
        "changed_interfaces": [], "changed_age_bands": [],
        "semantic_checks": {"status": "PASS", "rule": "reflexivity of unchanged original captured object"},
        "residual_obligations": [], "discharged_unchanged_obligations": [],
        "family_coverage": {"candidate_admission": "identity", "frontier_complete_verified": False},
        "structural_cost_tradeoff": {"cost": None, "authority": "not_claimed"},
        "adoption_scope": "retain existing artifact only; original economics and numerical validity remain separate"}]
    if _context is not None:
        semantic = _context.semantic
    elif time.perf_counter() < deadline:
        # Coverage is checked separately. Do not enumerate the family again to
        # decide whether one complete candidate preserves its inputs and laws.
        semantic = check_semantics(original, declarations, analysis, check_coverage=False)
    else:
        semantic = {"status": "NOT_RUN", "errors": [], "unresolved": [{"kind": "proposal checking budget exhausted", "scope": "source"}]}
    for scenario_index, scenario in enumerate(scenarios):
        allocation = scenario.get("allocation_contract", {})
        scoped = dict(analysis, allocation_scenario=allocation)
        family = None
        # A heuristic keeps a preference-eligible incumbent even if a deferred
        # cosmetic candidate dominates it.  Both are genuine complete plans and
        # need the same independent source/admission checks.  Exact schemas keep
        # their existing frontier and certificate meanings.
        plans = (scenario.get("candidates", [])
                 if scenario.get("schema") == "regime-heuristic-frontier/v1"
                 else scenario.get("frontier", []))
        for plan in plans:
            plan_errors, residual, discharged = [], [], []
            selected = {}
            expired = time.perf_counter() >= deadline
            try:
                selected = _selected(policies, option_index, plan)
                if set(selected) != set(cells):
                    raise ValueError("candidate interface coverage")
                if not expired:
                    if family is None:
                        family = IndependentFamily(scoped, seconds=max(0, deadline - time.perf_counter()))
                    family.bound()
                    plan_errors = family.admitted_plan(plan)
                    family.bound()
            except _Bounded:
                expired = True
            except (KeyError, TypeError, ValueError) as exc:
                plan_errors.append("candidate admission: " + str(exc))
            unchanged = {cid for cid, option in selected.items()
                         if cid in cells and _unchanged_interface(cells[cid], option, callbacks)}
            errors = [d for d in semantic.get("errors", []) if _relevant(d, selected, plan.get("projection_policy"))]
            for item in semantic.get("unresolved", []):
                if not _relevant(item, selected, plan.get("projection_policy")):
                    continue
                callback = item.get("callback")
                # Parametric congruence is a proof of preservation for a wholly
                # unchanged use, not an interpretation of the opaque primitive.
                safe_opaque = item.get("kind") == "opaque consumer factorization" and item.get("cell") in unchanged
                safe_callback = (item.get("scope") == "callback" and item.get("kind") == "callback proof unavailable"
                                 and callback in callback_uses and callback_uses[callback] <= unchanged)
                if safe_opaque or safe_callback:
                    discharged.append({"obligation": item, "rule": "unchanged bound primitive and complete argument tuple"})
                else:
                    residual.append(item)
            # Normalization obligations must remain visible even when the
            # symbolic checker happens not to need them for its own grammar.
            for item in analysis.get("obligations", []):
                if item.get("kind") == "allocation_contract":
                    continue
                if item.get("kind") == "role_timing_contract":
                    context = _preserved_context(original, cells, plan, item)
                    if context is not None:
                        discharged.append({"obligation": item,
                            "rule": "original decision context carried by identity, not cross-role equivalence",
                            "context_preservation": context})
                        continue
                callback = interpretation_callbacks.get(item.get("source_path"))
                uses = callback_uses.get(callback, set())
                if (item.get("kind") == "callback_interpretation" and uses
                        and uses <= unchanged):
                    discharged.append({"obligation": item, "callback": callback,
                        "rule": "source-bound unchanged application; interpretation remains unknown",
                        "unchanged_cells": sorted(uses)})
                    continue
                if item not in residual:
                    residual.append(item)
            if expired or semantic.get("status") == "NOT_RUN":
                residual.append({"kind": "proposal checking budget exhausted", "scope": "proposal"})
            semantic_residual = list(residual)
            authoritative = (allocation == declarations.get("allocation")
                             and allocation.get("status") in ("known", "declared")
                             and allocation.get("kind") in ("per_block", "global_union"))
            if not authoritative:
                residual.append({"kind": "allocation_contract", "detail": "costs are conditional representation scenarios"})
            changes, bands = _changes(cells, selected, plan)
            disposition = "reject" if plan_errors or errors else "investigate" if residual else "adopt"
            candidate = {"plan": plan, "selected_interfaces": selected,
                         "source_snapshot_sha256": source_digest, "declarations_sha256": declaration_digest}
            decisions.append({"schema": "regime-proposal-decision/v1", "graph_id": plan.get("id"),
                "scenario_index": scenario_index, "allocation_contract_sha256": _digest(allocation),
                "source_digest_kind": "canonical-json-sha256",
                "source_snapshot_sha256": source_digest, "declarations_sha256": declaration_digest,
                "candidate_sha256": _digest(candidate), "operation": "representation_specification",
                "disposition": disposition,
                "reasons": (["Independent candidate admission or semantic checks reject this proposal."] if disposition == "reject" else
                            ["Named local premises or checking budget remain unresolved."] if disposition == "investigate" else
                            ["Independent local checks pass under the supplied source and allocation contracts."]),
                "changed_interfaces": changes, "changed_age_bands": bands,
                "semantic_checks": {"status": "FAIL" if errors or plan_errors else "UNRESOLVED" if semantic_residual else "PASS",
                                    "errors": errors, "candidate_admission_errors": plan_errors,
                                    "unresolved": semantic_residual,
                                    "source_sha256": semantic.get("source_sha256"),
                                    "normalized_sha256": semantic.get("normalized_sha256")},
                "residual_obligations": residual, "discharged_unchanged_obligations": discharged,
                "family_coverage": {"candidate_admission": "NOT_RUN" if expired else "FAIL" if plan_errors else "PASS",
                                    "producer_status": scenario.get("status"),
                                    "projection_discovery_claim": analysis.get("discovery_complete") is True,
                                    "frontier_complete_verified": False,
                                    "scope": "local membership checked; complete-family claims require the separate frontier certificate"},
                "structural_cost_tradeoff": {"cost": plan.get("cost"),
                                             "authority": "declared" if authoritative else "conditional",
                                             "allocation_contract": allocation,
                                             "wall_time_saving": None, "peak_device_memory_saving": None},
                "adoption_scope": "representation specification; emitted native code, measured performance and paper validity are not certified"})
    return decisions


def verify_decisions(original, analysis, scenarios, records, *, seconds=900, _context=None,
                     include_replayed=False):
    """Replay stable anchors and positive claims, allowing stronger later checks.

    An honestly unverified investigate record may become adopt or reject on a
    better-resourced replay. That is strengthened evidence, not corruption of
    the saved certificate. A saved positive safety/admission claim still needs
    independent support; known contradictions take precedence over exhaustion.

    ``include_replayed`` exposes the freshly rebuilt local records to an
    in-memory production verifier so it can rank checked candidates without a
    second source/candidate replay.  Saved producer decisions are never used as
    the authority for this selection.
    """
    expected = build_decisions(original, analysis, scenarios, seconds=seconds, _context=_context)
    errors = []
    key = lambda record: (record.get("scenario_index"), record.get("graph_id"))
    actual = {key(record): record for record in records}
    if len(actual) != len(records):
        errors.append("duplicate proposal decision keys")
    if set(actual) != {key(record) for record in expected}:
        errors.append("proposal decision coverage")
    exhausted = any(any(o.get("kind") == "proposal checking budget exhausted" for o in row.get("residual_obligations", [])) for row in expected)
    strengthened = []
    unavailable_claims = []
    stable_fields = ("schema", "graph_id", "scenario_index", "source_digest_kind",
                     "source_snapshot_sha256", "declarations_sha256", "candidate_sha256",
                     "allocation_contract_sha256", "operation", "changed_interfaces",
                     "changed_age_bands", "structural_cost_tradeoff", "adoption_scope")
    for record in expected:
        saved = actual.get(key(record))
        if saved is None:
            continue
        label = str(key(record))
        for field in stable_fields:
            if saved.get(field) != record.get(field):
                errors.append("decision anchor differs: " + field + " at " + label)
        expected_contexts = {item['obligation']['source_path']: item['context_preservation']
                             for item in record.get('discharged_unchanged_obligations', [])
                             if 'context_preservation' in item}
        saved_contexts = {item.get('obligation', {}).get('source_path'): item['context_preservation']
                          for item in saved.get('discharged_unchanged_obligations', [])
                          if 'context_preservation' in item}
        if (any(expected_contexts.get(path) != context for path, context in saved_contexts.items())
                or (saved.get('disposition') == 'adopt' and saved_contexts != expected_contexts)):
            errors.append('original context preservation claim changed at ' + label)
        old_family = saved.get("family_coverage", {})
        new_family = record.get("family_coverage", {})
        for field in ("producer_status", "projection_discovery_claim", "frontier_complete_verified"):
            if old_family.get(field) != new_family.get(field):
                errors.append("decision family claim differs: " + field + " at " + label)
        old_semantic = saved.get("semantic_checks", {})
        new_semantic = record.get("semantic_checks", {})
        for field, anchor in (("source_sha256", record["source_snapshot_sha256"]),
                              ("normalized_sha256", record["declarations_sha256"])):
            if old_semantic.get(field) is not None and old_semantic[field] != anchor:
                errors.append("decision semantic anchor differs: " + field + " at " + label)
        disposition = saved.get("disposition")
        if disposition not in ("adopt", "investigate", "reject"):
            errors.append("unknown saved disposition at " + label)
        old_errors = old_semantic.get("errors", []) + old_semantic.get("candidate_admission_errors", [])
        if disposition == "adopt":
            if (old_semantic.get("status") != "PASS" or old_errors
                    or saved.get("residual_obligations")
                    or old_family.get("candidate_admission") not in ("PASS", "identity")):
                errors.append("saved adopt lacks its own required checked premises at " + label)
            if record.get("disposition") == "reject":
                errors.append("saved adopt is rejected by independent replay at " + label)
            elif record.get("disposition") != "adopt":
                unavailable_claims.append("saved adopt cannot currently be re-established at " + label)
        elif disposition == "reject":
            if old_semantic.get("status") != "FAIL" or not old_errors:
                errors.append("saved reject lacks a checked contradiction at " + label)
            if record.get("disposition") == "adopt":
                errors.append("saved rejection is not reproduced at " + label)
            elif record.get("disposition") != "reject":
                unavailable_claims.append("saved rejection cannot currently be re-established at " + label)
        elif disposition == "investigate":
            if old_errors or old_semantic.get("status") == "FAIL" or not saved.get("residual_obligations"):
                errors.append("inconsistent saved investigation record at " + label)
            if record.get("disposition") != "investigate":
                strengthened.append({"scenario_index": record["scenario_index"], "graph_id": record["graph_id"],
                                     "saved_disposition": disposition, "replayed_disposition": record["disposition"]})
        for kind, old, new in (("semantics", old_semantic.get("status"), new_semantic.get("status")),
                               ("candidate admission", old_family.get("candidate_admission"), new_family.get("candidate_admission"))):
            if old == "PASS" and new == "FAIL":
                errors.append("saved positive " + kind + " claim fails replay at " + label)
            elif old == "PASS" and new not in ("PASS", "identity"):
                unavailable_claims.append("saved positive " + kind + " claim unavailable at " + label)
    receipt = {"status": "FAIL" if errors else "UNRESOLVED" if exhausted or unavailable_claims else "PASS",
            "errors": errors, "source_snapshot_sha256": _digest(original),
            "unresolved": unavailable_claims, "strengthened": strengthened,
            "checked_decisions": len(expected), "dense_cartesian_rows_evaluated": 0,
            "scope": "stable anchors and independently supported positive claims; conservative investigation may be strengthened; full optimizer completeness checked separately"}
    if include_replayed:
        receipt["replayed_decisions"] = expected
    return receipt
