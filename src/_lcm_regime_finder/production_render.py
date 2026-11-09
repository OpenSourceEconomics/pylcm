"""Portable reports for metadata-only regime discovery.

No producer, scorer, callback, solver, browser, plotting package or image package
is imported. Graph PNGs use a small deterministic standard-library rasterizer.
The saved report.json is the unchanged result; report-data.json adds exact graph
and selected-interface records for reviewing every frontier alternative.
"""
from __future__ import annotations
from .records import RenderPoint, RenderScenario

import binascii
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import re
import struct
import unicodedata
import zlib


METRICS = ("work", "storage", "clarity")
PAGE_NODES = 12
MAX_PNG_PAGES = 100
COLORS = {"transition": "#147d86", "joint": "#7c3aed", "imputation": "#087f9b",
          "value_read": "#2563eb", "gate": "#a16207", "exit": "#c2410c",
          "fallback": "#64748b", "reporting": "#be185d"}
KIND_NAMES = {"transition": "Transition", "joint": "Joint lottery", "imputation": "Imputation",
              "value_read": "Value read", "gate": "Gate", "exit": "Exit",
              "fallback": "Fallback", "reporting": "Reporting association"}
STATUS_NAMES = {"HEURISTIC": "Heuristic effort; unknown optimality gap",
                "RETAINED_CANDIDATES_CHECKED": "Retained candidate admission and costs checked",
                "COMPLETE": "Complete within the stated family", "INCOMPLETE": "Incomplete",
                "UNRESOLVED": "Unresolved obligations", "INVALID": "Verification failed",
                "FAIL": "Failed", "PASS": "Passed", "VERIFIED": "Independently checked",
                "DECLARED": "Declared contract", "NOT_RUN": "Not run", "UNKNOWN": "Not supplied"}
LAYER_NAMES = {"source_semantics": "Source and semantic check", "discovery": "Candidate-family coverage",
               "allocation": "Backend allocation contract", "frontier": "Independent frontier check",
               "native_numerical_execution": "Native solve and simulation"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _text(value):
    if value is None:
        return "Not supplied"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _h(value):
    return html.escape(_text(value), quote=True)


def _md(value):
    value = re.sub(r"\s+", " ", _text(value)).strip()
    value = html.escape(value, quote=False)
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", value)


def _number(value):
    return f"{value:,}" if type(value) is int else _text(value)


def _endpoint_note(recommendation: dict) -> str:
    preference = recommendation.get('endpoint_preference')
    if not preference:
        return ''
    return (preference['reason'] + ' Explicit override: ' + str(preference['allow_final_only_split'])
            + '. Endpoint-only alternatives: ' + (', '.join(preference['deferred_graph_ids']) or 'none')
            + '. They remain visible as candidates. Use --allow-final-only-split or allow_final_only_split=True to permit their recommendation.'
            + (' No frontier graph satisfies the default preference; retain the original representation.'
               if recommendation.get('graph_id') is None else ''))


def _recommendation_note(scenario):
    if scenario.get('schema') == 'regime-heuristic-frontier/v1':
        return ('The selected graph must pass its independent local checks and strictly improve '
                'the unchanged original cost by the configured relative threshold. Weights and '
                'reference scales are fixed within a run. Changing weights and rerunning can '
                'change both the examined candidates and the retained antichain; the optimality gap remains unknown.')
    return ('The rule minimizes the sum of weight × cost / reference scale among preference-eligible '
            'frontier points, using positive weights and exact rational arithmetic. Default equal '
            'weights balance the normalized costs; they do not change the complete Pareto frontier.')


def _status(value):
    if isinstance(value, dict):
        value = value.get("status", "UNKNOWN")
    key = str(value or "UNKNOWN").upper()
    return STATUS_NAMES.get(key, key.replace("_", " ").capitalize())


def _badge(value):
    key = str(value.get("status") if isinstance(value, dict) else value or "UNKNOWN").upper()
    color = "good" if key in {"COMPLETE", "PASS", "VERIFIED"} else "bad" if key in {"INVALID", "FAIL", "FAILED"} else "neutral"
    return f'<span class="badge {color}">{_h(_status(value))}</span>'


def _declarations(result):
    analysis = result.get("analysis") or {}
    return analysis, analysis.get("declarations") or {}


def _model_name(result, declarations):
    identity = result.get("source_identity") or declarations.get("source_identity") or {}
    if not isinstance(identity, dict):
        identity = {"name": identity}
    return (result.get("name") or declarations.get("name") or identity.get("project")
            or identity.get("name") or identity.get("model") or "Original model")


def _dates(values, limit=None):
    values = list(dict.fromkeys(_text(x) for x in values))
    if not values:
        return "Not supplied"
    if limit and len(values) > limit:
        return ", ".join(values[:3]) + " … " + ", ".join(values[-2:]) + f" ({len(values)} declared dates)"
    return ", ".join(values)


def _axes(axes, domains):
    parts = []
    for axis, domain_id in sorted((axes or {}).items()):
        domain = domains.get(domain_id, {})
        size = domain.get("size")
        parts.append(f"{axis} ({_number(size)} values)" if type(size) is int else f"{axis} (size not supplied)")
    return "; ".join(parts) or "None"


def _selected_options(analysis, plan):
    options = {cid: {o.get("id"): o for o in record.get("options", [])}
               for cid, record in analysis.get("interfaces", {}).items()}
    selected = {}
    missing = []
    for block in plan.get("blocks", []):
        for cid, option_id in block.get("option_by_cell", {}).items():
            if option_id in options.get(cid, {}):
                selected[cid] = options[cid][option_id]
            else:
                missing.append({"cell": cid, "option_id": option_id})
    return selected, missing


def _consequences(analysis, declarations, plan):
    selected, missing = _selected_options(analysis, plan)
    cells = {c.get("id"): c for c in declarations.get("cells", [])}
    drops, characteristics, constants = defaultdict(list), defaultdict(list), defaultdict(list)
    for cid, option in selected.items():
        date = cells.get(cid, {}).get("age", cells.get(cid, {}).get("period"))
        for axis, projection in option.get("state_projection", {}).items():
            if projection.get("kind") == "drop":
                drops[axis].append({"cell": cid, "date": date})
            elif projection.get("kind") == "characteristic":
                key = (axis, projection.get("target", "characteristic"), projection.get("callback"))
                characteristics[key].append({"cell": cid, "date": date})
        for axis, reconstruction in option.get("action_reconstruction", {}).items():
            if reconstruction.get("kind") == "constant":
                constants[(axis, _text(reconstruction.get("value")))].append({"cell": cid, "date": date})
    original_blocks = defaultdict(set)
    merged = []
    for i, block in enumerate(plan.get("blocks", []), 1):
        labels = block.get("original_regimes", [])
        for label in labels:
            original_blocks[label].add(i)
        if len(labels) > 1:
            merged.append({"block": i, "original_regimes": labels})
    sentences = []
    for axis, items in sorted(drops.items()):
        sentences.append(f"Omit {axis} from the decision state in {len(items)} declared cells (dates {_dates([x['date'] for x in items], 10)}).")
    for (axis, target, _), items in sorted(characteristics.items(), key=lambda x: str(x[0])):
        sentences.append(f"Represent {axis} through {target} in {len(items)} declared cells (dates {_dates([x['date'] for x in items], 10)}).")
    for (axis, value), items in sorted(constants.items()):
        sentences.append(f"Reconstruct action {axis} as {value} in {len(items)} declared cells; retain the complete feasible-action correspondence.")
    if merged:
        sentences.append(f"Combine compatible original regimes in {len(merged)} blocks while preserving original-regime dispatch.")
    split = {name: sorted(indices) for name, indices in original_blocks.items() if len(indices) > 1}
    if split:
        sentences.append("Split the declared dates of " + ", ".join(sorted(split)) + " across multiple blocks.")
    if not sentences and selected:
        sentences.append("The selected interfaces retain the recorded state and action coordinates; differences are in the dated block layout.")
    if missing:
        sentences.append(f"The result omits {len(missing)} selected interface records; those consequences cannot be reconstructed from this report.")
    if not selected:
        sentences.append("No selected semantic interface was supplied; model consequences remain unresolved.")
    return {"sentences": sentences, "selected_interfaces": selected, "missing_interfaces": missing,
            "dropped_states": [{"state": name, "cells": rows} for name, rows in sorted(drops.items())],
            "categorical_maps": [{"state": key[0], "target": key[1], "callback": key[2], "cells": rows}
                                 for key, rows in sorted(characteristics.items(), key=lambda x: str(x[0]))],
            "forced_actions": [{"action": key[0], "value_display": key[1], "cells": rows}
                               for key, rows in sorted(constants.items())],
            "merged_blocks": merged, "split_original_regimes": split}


def _graph(declarations, plan=None, title="Original declaration graph"):
    cells = {c.get("id"): c for c in declarations.get("cells", [])}
    nodes, owner, ordering = [], {}, {}
    if plan is None:
        ordered = sorted(cells.values(), key=lambda c: (_text(c.get("period")), _text(c.get("age")), _text(c.get("id"))))
        # Numeric source periods are ordered numerically, without changing them.
        if all(type(c.get("period")) in (int, float) for c in ordered):
            ordered.sort(key=lambda c: (c["period"], _text(c.get("id"))))
        for i, cell in enumerate(ordered):
            key = f"N{i}"
            owner[cell["id"]] = key;ordering[key] = i
            nodes.append({"id": key, "kind": "cell", "label": _text(cell.get("regime", cell["id"])),
                          "dates": [cell.get("age", cell.get("period"))], "cells": [cell["id"]],
                          "states": list(cell.get("states", {})), "actions": list(cell.get("actions", {})),
                          "stage": cell.get("stage"), "terminal": bool(cell.get("terminal")), "order": i * 10})
    else:
        for i, block in enumerate(plan.get("blocks", [])):
            key = f"N{i}";ordering[key] = i
            for cid in block.get("cells", []):
                owner[cid] = key
            nodes.append({"id": key, "kind": "block", "label": " / ".join(block.get("original_regimes", [])) or f"Block {i+1}",
                          "dates": block.get("ages", []), "cells": block.get("cells", []),
                          "states": list(block.get("state_axes", {})), "actions": list(block.get("action_axes", {})),
                          "terminal": bool(block.get("cells")) and all(cells.get(cid, {}).get("terminal") for cid in block.get("cells", [])),
                          "order": i * 10})
    edges, internal = [], []
    for index, relation in enumerate(declarations.get("relations", [])):
        source = owner.get(relation.get("source"))
        destinations = list(dict.fromkeys(owner[c] for c in relation.get("targets", []) if c in owner))
        if source is None:
            continue
        base = {"relation_id": relation.get("id"), "kind": relation.get("kind", "unknown"),
                "phase": relation.get("phase", "not supplied"), "support": relation.get("support", "not supplied"),
                "source_cells": [relation.get("source")], "target_cells": relation.get("targets", [])}
        if not destinations:
            internal.append(dict(base, source=source, reason="No declared target cell in this view"));continue
        # Reify a full joint/multi-target relation, preserving its joint identity.
        if relation.get("kind") == "joint" or len(relation.get("targets", [])) > 1:
            joint_id = f"J{index}"
            nodes.append({"id": joint_id, "kind": "joint", "label": KIND_NAMES.get(relation.get("kind"), "Multi-target relation"),
                          "relation_id": relation.get("id"), "dates": [], "cells": [], "states": relation.get("joint_outputs", []),
                          "actions": [], "order": ordering.get(source, 0) * 10 + 5 + index / (len(declarations.get("relations", [])) + 1)})
            edges.append(dict(base, source=source, target=joint_id))
            edges.extend(dict(base, source=joint_id, target=destination) for destination in destinations)
        else:
            for destination in destinations:
                if source == destination:
                    internal.append(dict(base, source=source, target=destination, reason="Relation remains inside this block"))
                    edges.append(dict(base, source=source, target=destination, internal=True))
                else:
                    edges.append(dict(base, source=source, target=destination))
    nodes.sort(key=lambda n: (n["order"], n["id"]))
    return {"title": title, "nodes": nodes, "edges": edges, "internal_relations": internal,
            "relation_count": len(declarations.get("relations", [])),
            "scope": "Declared typed relations; no independence inferred from coordinates or layout."}


def _mermaid_label(value):
    return re.sub(r"\s+", " ", _text(value)).replace('"', "'").replace("`", "'").replace("<", "(").replace(">", ")").replace("[", "(").replace("]", ")")


def graph_mermaid(graph):
    lines = ["flowchart TD"]
    if not graph["nodes"]:
        lines.append("  %% No declared cells were supplied; graph unavailable.")
    for node in graph["nodes"]:
        if node["kind"] == "joint":
            label = node["label"] + " | " + _text(node.get("relation_id"))
            lines.append(f'  {node["id"]}{{"{_mermaid_label(label)}"}}')
        else:
            label = node["label"] + " | dates " + _dates(node["dates"], 8)
            label += " | states: " + (", ".join(node["states"]) or "none")
            label += " | actions: " + (", ".join(node["actions"]) or "none")
            lines.append(f'  {node["id"]}["{_mermaid_label(label)}"]')
    for edge in graph["edges"]:
        kind = KIND_NAMES.get(edge["kind"], _text(edge["kind"]))
        label = f'{kind}; {edge["phase"]}; {_text(edge["relation_id"])}'
        arrow = "-.->" if edge.get("support") == "zero" else "-->"
        lines.append(f'  {edge["source"]} {arrow}|"{_mermaid_label(label)}"| {edge["target"]}')
    for item in graph["internal_relations"]:
        lines.append("  %% Internal or untargeted relation: " + _mermaid_label(item))
    return "\n".join(lines) + "\n"


def _view(result):
    analysis, declarations = _declarations(result)
    domains = dict(declarations.get("domains", {})) | analysis.get("derived_domains", {})
    original = _graph(declarations)
    decisions = {(row.get("scenario_index"), row.get("graph_id")): row
                 for row in result.get("proposal_decisions", [])}
    selections = {row.get('scenario_index'): row for row in result.get('recommendations', [])}
    scenarios: list[RenderScenario] = []
    for si, scenario in enumerate(result.get("scenarios", []), 1):
        contract = scenario.get("allocation_contract") or {}
        conditional = contract.get("status") not in ("known", "declared")
        kind = contract.get("kind", "not supplied")
        title = {"per_block": "Separate allocation for each block", "global_union": "Shared global axis allocation"}.get(kind, "Allocation contract not supplied")
        if conditional:
            title += " — conditional scenario"
        points: list[RenderPoint] = []
        if result.get('search_mode') == 'heuristic':
            # Presentation uses the source-checked selection; the original
            # cost-only preference remains untouched in canonical report.json.
            scenario = dict(scenario, cost_recommendation=scenario.get('recommendation'),
                            recommendation=selections.get(si - 1, {}))
        recommendation = scenario.get("recommendation") or {}
        for pi, point in enumerate(scenario.get('candidates', scenario.get("frontier", [])), 1):
            prefix = f"scenario-{si:03d}-option-{pi:04d}"
            consequences = _consequences(analysis, declarations, point)
            graph = _graph(declarations, point, f"Scenario {si}, option {pi}")
            points.append({"ordinal": pi, "prefix": prefix, "point": point, "graph": graph,
                           "cost_preferred": point.get("id") == recommendation.get("graph_id"),
                           "decision": decisions.get((si - 1, point.get("id")), {
                               "disposition": "investigate",
                               "reasons": ["No independent per-proposal decision was supplied."]}),
                           "consequences": consequences})
        scenarios.append({"ordinal": si, "title": title, "conditional": conditional,
                          "scenario": scenario, "points": points})
    preview_order = [(p["prefix"], p["graph"]) for s in scenarios for p in s["points"] if p["cost_preferred"]]
    preview_order += [("original", original)]
    preview_order += [(p["prefix"], p["graph"]) for s in scenarios for p in s["points"] if not p["cost_preferred"]]
    previews, remaining = {}, MAX_PNG_PAGES
    for prefix, graph in preview_order:
        count = min(remaining, _pages(graph))
        previews[prefix] = count;remaining -= count
    return {"result": result, "analysis": analysis, "declarations": declarations, "domains": domains,
            "name": _model_name(result, declarations), "original_graph": original, "scenarios": scenarios,
            "preview_counts": previews}


def _obligations(view):
    result, analysis = view["result"], view["analysis"]
    rows = []
    for scope, items in (("Analysis", analysis.get("obligations", [])),
                         ("Source check", (result.get("semantic_check") or {}).get("unresolved", [])),
                         ("Source check failure", (result.get("semantic_check") or {}).get("errors", []))):
        rows.extend({"scope": scope, "detail": value} for value in items)
    for scenario in view["scenarios"]:
        check = scenario["scenario"].get("independent_check") or {}
        for key in ("errors", "unresolved"):
            rows.extend({"scope": f'Scenario {scenario["ordinal"]} {key}', "detail": value}
                        for value in check.get(key, []))
        for key in ("error", "exhaustion_reason"):
            if scenario["scenario"].get(key):
                rows.append({"scope": f'Scenario {scenario["ordinal"]}', "detail": scenario["scenario"][key]})
    if result.get("error"):
        rows.append({"scope": "Run", "detail": result["error"]})
    return rows


def _headline(view):
    if not view["analysis"]:
        return "No semantic interface was supplied. Model consequences and a transformation recommendation remain unresolved."
    status = (view["result"].get("semantic_check") or {}).get("status")
    count = len(_obligations(view))
    if status == "FAIL":
        return "The independent source/semantic check failed. The displayed alternatives are candidates for review, not established source-equivalent transformations."
    if view['result'].get('search_mode') == 'heuristic':
        return ('Heuristic search returned candidate layouts with an unknown optimality gap. '
                'Recorded local checks determine which specifications can be recommended; unresolved premises retain the original. '
                'The retained antichain is not a certificate of global completeness.')
    if status != "PASS":
        return f"Source or semantic obligations remain unresolved ({count} recorded items). Candidate representations and conditional cost scenarios are shown with their individual checks."
    n = sum(len(s["points"]) for s in view["scenarios"])
    if view["result"].get("status") != "COMPLETE":
        return f"The source/semantic check passed. The result contains {n} examined alternatives; unresolved allocation, search coverage or other obligations qualify the recommendation."
    return f"The independent source/semantic check passed. The complete declared-family search returned {n} frontier alternatives across {len(view['scenarios'])} allocation scenarios."


def _pages(graph):
    return max(1, math.ceil(len(graph["nodes"]) / PAGE_NODES)) if graph["nodes"] else 0


def _image_names(prefix, graph, previews=None):
    count = _pages(graph) if previews is None else previews.get(prefix, 0)
    return [f"graphs/{prefix}-page-{i+1:03d}.png" for i in range(count)]


def _preview_note(prefix, graph, previews):
    shown, total = previews.get(prefix, 0), _pages(graph)
    if shown < total:
        return f"PNG preview limit: {shown} of {total} pages shown. The full graph remains in Mermaid and report-data.json. This presentation limit does not change the frontier or search status."
    return ""


def render_markdown(result, *, _prepared_view=None):
    """Return a separate, portable Markdown report for one production result."""
    view = _prepared_view if _prepared_view is not None else _view(result)
    analysis = view["analysis"];decl = view["declarations"]
    lines = [f'# Regime representation report: {_md(view["name"])}', "",
             f'**Result: {_md(_status(result.get("status")))}.**', "",
             _md(_headline(view)), "",
             "This report describes metadata and source analysis. Native solves and simulations have their own status below.", "",
             "## Evidence and input scope", "", "| Check | Recorded status |", "|---|---|"]
    layers = result.get("layers") or {}
    for key, title in LAYER_NAMES.items():
        lines.append(f'| {title} | {_md(_status(layers.get(key)))} |')
    lines += ["", f'Input configuration: {_md(decl.get("production_metadata", "Not supplied"))}.', "",
              f'Declared cells: {len(decl.get("cells", []))}; original regimes: {len({c.get("regime") for c in decl.get("cells", [])})}; typed relations: {len(decl.get("relations", []))}.', "",
              "## Cost definitions", "",
              "Work counts declared state-action evaluations plus the declared boundary incidence cost. Storage is the sum of allocated value entries. Clarity is the declared weighted count of blocks, characteristics and dispatch clauses. These are integer representation costs; hardware runtime and peak live/device memory require separate measurements.", ""]
    lines += ["## Proposal decisions", "",
              "Use the independently checked disposition for each proposal before changing a model. A cost preference alone does not authorize a transformation. Retaining the unchanged original is a separate decision with no optimization or speed claim.", "",
              "[Exact source-bound decision records](proposal-decisions.json).", ""]
    for record in result.get("proposal_decisions", []):
        if record.get("scenario_index") is None:
            lines += [f'Original representation: **{_md(record.get("disposition", "investigate"))}**.', "",
                      _md(record.get("reasons", [])), ""]
    if not view["scenarios"]:
        lines += ["## No ranked alternatives", "", "No frontier scenario was supplied. The result cannot establish a recommended transformation or complete discovery.", ""]
    for s in view["scenarios"]:
        scenario = s["scenario"];rec = scenario.get("recommendation") or {};cert = scenario.get("certificate") or scenario.get("search") or {}
        lines += [f'## Scenario {s["ordinal"]}: {_md(s["title"])}', "",
                  f'Search: **{_md(_status(scenario.get("status")))}**. Independent check: **{_md(_status((scenario.get("independent_check") or {}).get("status")))}**.', "",
                  f'Frontier scope: {_md(cert.get("frontier_scope", "Not supplied"))}.', "",
                  f'Admissible family: {_md(cert.get("family", "Not supplied"))}.', "",
                  f'Clarity weights: {_md(scenario.get("clarity_weights", "Not supplied"))}. Boundary unit: {_md((scenario.get("allocation_contract") or {}).get("boundary_unit"))}.', "",
                  "| Option | Cost preference | Work | Value entries | Clarity | Blocks |", "|---|---|---:|---:|---:|---:|"]
        for p in s["points"]:
            cost = p["point"].get("cost", {})
            lines.append(f'| {p["ordinal"]} | {"Yes" if p["cost_preferred"] else ""} | ' +
                         " | ".join(_md(_number(cost.get(k))) for k in METRICS) + f' | {len(p["point"].get("blocks", []))} |')
        if not s["points"]:
            lines += ["", "No admissible complete candidate plan was supplied."]
        if rec:
            lines += ["", f'Cost preference: graph {_md(rec.get("graph_id"))}, exact normalized score **{_md(rec.get("score"))}**.', "",
                      f'Weights: {_md(rec.get("weights"))}. Reference scales: {_md(rec.get("reference_scales"))}.', "",
                      _recommendation_note(scenario), ""]
            if _endpoint_note(rec):
                lines += [_md(_endpoint_note(rec)), '']
        if scenario.get("status") == "HEURISTIC":
            lines += ["**Heuristic scope.** The returned layouts have individual check records. Search can omit better layouts; the optimality gap is unknown. The independent checker verifies returned candidates, not an exhaustive search history.", "",
                      f'Stopping reason: {_md(cert.get("stopping_reason"))}. Examined candidate count (producer accounting): {_md(cert.get("evaluated_candidates"))}.', ""]
        elif scenario.get("status") != "COMPLETE":
            lines += ["**Search is incomplete.** These are complete candidate layouts examined so far; unexamined candidates may dominate them.", "",
                      f'Exhaustion: {_md(scenario.get("exhaustion_reason", "Not supplied"))}. Recorded lower bound: {_md(cert.get("lower_bound"))}.', ""]
        for p in s["points"]:
            lines += [f'### Option {p["ordinal"]}' + (" — cost preference" if p["cost_preferred"] else ""), ""]
            lines += [f'Disposition: **{_md(p["decision"].get("disposition", "investigate"))}**.', "",
                      _md(p["decision"].get("reasons", [])), ""]
            lines += ["- " + _md(sentence) for sentence in p["consequences"]["sentences"]]
            lines += ["", "| Block | Original regimes | Declared dates | Allocated states | Allocated actions |", "|---|---|---|---|---|"]
            for i, block in enumerate(p["point"].get("blocks", []), 1):
                lines.append("| " + " | ".join([str(i), _md(", ".join(block.get("original_regimes", []))),
                    _md(_dates(block.get("ages", []))), _md(_axes(block.get("state_axes"), view["domains"])),
                    _md(_axes(block.get("action_axes"), view["domains"]))]) + " |")
            lines += ["", f'[Full typed Mermaid graph](graphs/{p["prefix"]}.mmd). Exact cell admissions, projections, action reconstructions and costs are in report-data.json.', ""]
            note = _preview_note(p["prefix"], p["graph"], view["preview_counts"])
            if note:
                lines += [note, ""]
            for filename in _image_names(p["prefix"], p["graph"], view["preview_counts"]):
                lines += [f'![Scenario {s["ordinal"]}, option {p["ordinal"]} graph page]({filename})', ""]
    lines += ["## Original typed graph", "", "[Full Mermaid graph](graphs/original.mmd). Joint and multi-target relations retain a separate relation node; phase labels remain explicit.", ""]
    note = _preview_note("original", view["original_graph"], view["preview_counts"])
    if note:
        lines += [note, ""]
    for filename in _image_names("original", view["original_graph"], view["preview_counts"]):
        lines += [f'![Original declaration graph page]({filename})', ""]
    if not view["original_graph"]["nodes"]:
        lines += ["No original graph metadata was supplied.", ""]
    lines += ["## Obligations and limits", ""]
    obligations = _obligations(view)
    lines += [f'- **{_md(row["scope"])}:** {_md(row["detail"])}' for row in obligations] or ["No additional obligation record was supplied. This does not substitute for the individual check statuses."]
    lines += ["", "Semantic decision interfaces and dense block layouts are distinct. Padding axes are allocated and charged; they do not restore discarded information. Exact state and action records remain in report-data.json. Unknown backend allocation yields conditional scenarios, not a production speed or memory claim.", "",
              "## Provenance and execution", "", f'Source identity: {_md(result.get("source_identity", decl.get("source_identity")))}.', "",
              f'Recorded metadata execution: {_md(analysis.get("production_execution", "Not supplied"))}.', "",
              (f'Resource record: {_md(result.get("resources", result.get("measurement")))}.'
               if result.get("resources", result.get("measurement")) is not None else
               '[Whole-command resource measurement](resource.json): supplied by the parent run wrapper after analysis, independent checking and initial report generation; no inline measurement was supplied.'), "",
              "report.json preserves the complete supplied result. report-data.json contains exact graph adjacency and every selected interface. The report manifest records generated-file hashes. Graph PNGs are bounded pages with abbreviated labels; the full graphs and exact dates remain available in Mermaid/JSON.", ""]
    return "\n".join(lines)


CSS = """
:root{--ink:#172c42;--muted:#526577;--line:#dce5eb;--paper:#f4f7f9;--accent:#0b6f78}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1260px;margin:auto;padding:32px}h1{font-size:34px;line-height:1.15;margin:10px 0}h2{font-size:25px;margin:32px 0 12px}h3{font-size:20px}h4{font-size:16px}
.eyebrow{letter-spacing:.14em;text-transform:uppercase;font-size:12px;color:var(--accent);font-weight:750}.lead{max-width:920px;color:var(--muted)}
nav{display:flex;gap:18px;flex-wrap:wrap;margin:22px 0}a{color:#086b85;text-underline-offset:3px}.card,section.scenario{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0}
.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.metric{border-left:4px solid var(--accent);padding:14px 18px;background:#eef7f7}.metric strong{display:block;font-size:22px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.metric span{font-size:13px;color:var(--muted)}
.badge{display:inline-block;padding:4px 10px;border-radius:20px;background:#e8eef4;color:#375168;font-size:12px;font-weight:700}.good{background:#e4f3eb;color:#175b37}.bad{background:#fce8e6;color:#9a2d26}.neutral{background:#f6eedb;color:#785518}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}th{text-align:left;background:#eef3f6;color:#355064}th,td{padding:10px;border-bottom:1px solid var(--line);vertical-align:top}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.recommended{background:#eef8f6}.note{color:var(--muted);font-size:13px}.callout{padding:14px 18px;border-left:4px solid #b98725;background:#fcf7e9}
details{border-top:1px solid var(--line);padding:13px 0}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:14px;border-radius:7px;font:12px/1.5 ui-monospace,monospace;max-height:420px;overflow:auto}.graphs{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}figure{margin:10px 0}figure img{max-width:100%;height:auto;border:1px solid var(--line);background:white;border-radius:8px}figcaption{font-size:12px;color:var(--muted);margin:6px 0}.chart{max-width:700px}.scope{font-size:14px;color:var(--muted)}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}ul{padding-left:22px}footer{border-top:1px solid var(--line);padding:22px 0;color:var(--muted);font-size:12px}@media(max-width:760px){main{padding:18px}.metrics,.two{grid-template-columns:1fr}.graphs{grid-template-columns:1fr}h1{font-size:28px}}@media print{body{background:white}main{padding:0}.card,section.scenario{break-inside:avoid}details{display:block}nav{display:none}}
"""


def _table(headers, rows, numeric=()):
    head = "".join(f"<th>{_h(x)}</th>" for x in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{_h(x)}</td>' for i, x in enumerate(row)) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def _graph_html(prefix, graph, previews):
    if not graph["nodes"]:
        return '<p class="note">No declared graph metadata was supplied.</p>'
    count = _pages(graph)
    result = [f'<p><a href="graphs/{prefix}.mmd">Full typed Mermaid graph</a> · {len(graph["nodes"])} graph nodes · {len(graph["edges"])} displayed directed links · {len(graph["internal_relations"])} internal or untargeted relations retained in JSON.</p>', '<div class="graphs">']
    note = _preview_note(prefix, graph, previews)
    if note:
        result.insert(1, f'<p class="callout">{_h(note)}</p>')
    for i, filename in enumerate(_image_names(prefix, graph, previews), 1):
        result.append(f'<figure><a href="{filename}"><img loading="lazy" src="{filename}" alt="{_h(graph["title"])}; graph page {i} of {count}"></a><figcaption>Page {i} of {count}. Cross-page relations are marked; all adjacency and exact labels remain in Mermaid and report-data.json.</figcaption></figure>')
    return "".join(result) + "</div>"


def _scatter(points):
    valid = [p for p in points if all(type(p["point"].get("cost", {}).get(k)) is int and p["point"]["cost"][k] >= 0 for k in METRICS)]
    if len(valid) < 2:
        return ""
    max_w = max(1, max(p["point"]["cost"]["work"] for p in valid));max_s = max(1, max(p["point"]["cost"]["storage"] for p in valid))
    dots = []
    for p in valid:
        cost = p["point"]["cost"]
        x = 78 + 500 * float(Fraction(cost["work"], max_w));y = 260 - 210 * float(Fraction(cost["storage"], max_s))
        color = "#0b6f78" if p["cost_preferred"] else "#547ca4"
        title = f'Option {p["ordinal"]}: work {_number(cost["work"])}, value entries {_number(cost["storage"])}, clarity {_number(cost["clarity"])}'
        dots.append(f'<g><title>{_h(title)}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"/><text x="{x+9:.2f}" y="{y-8:.2f}" font-size="12" fill="#172c42">{p["ordinal"]}</text></g>')
    return ('<figure class="chart"><svg viewBox="0 0 660 320" role="img" aria-label="Candidate work versus allocated value entries; lower is better on both axes">'
            '<rect width="660" height="320" fill="#f7fafc"/><path d="M78 32V260H612" fill="none" stroke="#71869a"/>'
            '<text x="330" y="304" text-anchor="middle" font-size="13" fill="#355064">Declared work (0 to ' + _h(_number(max_w)) + ')</text>'
            '<text x="14" y="22" font-size="13" fill="#355064">Value entries: 0 to ' + _h(_number(max_s)) + '</text>'
            + "".join(dots) + '</svg><figcaption>Work and value-entry tradeoffs. Labels are option numbers; the cost-preferred point is teal. The table includes the third objective, clarity, in exact units.</figcaption></figure>')


def render_html(result, *, _prepared_view=None):
    """Return self-contained HTML referencing only the report's local graph files."""
    view = _prepared_view if _prepared_view is not None else _view(result)
    analysis = view["analysis"];decl = view["declarations"]
    title = f'Regime representation report — {view["name"]}'
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             f'<title>{_h(title)}</title><style>{CSS}</style></head><body><main>',
             '<header><div class="eyebrow">Source and metadata analysis</div>', f'<h1>{_h(view["name"])}</h1>',
             f'<p>{_badge(result.get("status"))}</p><p class="lead">{_h(_headline(view))}</p><p class="scope">Compare declared work, allocated value entries and structural complexity. Source validity, family coverage and native numerical execution remain separate checks.</p></header>',
             '<nav><a href="#overview">Input and checks</a><a href="#alternatives">Candidate alternatives</a><a href="#original">Original graph</a><a href="#obligations">Obligations</a><a href="#provenance">Provenance</a><a href="report.md">Markdown</a><a href="report.json">Complete JSON</a></nav>',
             '<section class="card" id="overview"><h2>Input and checks</h2><div class="two"><div>',
             f'<p><strong>{len(decl.get("cells", []))}</strong> declared cells · <strong>{len({c.get("regime") for c in decl.get("cells", [])})}</strong> original regimes · <strong>{len(decl.get("relations", []))}</strong> typed relations</p>',
             f'<p class="scope">{_h(decl.get("production_metadata", "Original configuration metadata was not supplied."))}</p></div><div>']
    layers = result.get("layers") or {}
    for key, label in LAYER_NAMES.items():
        parts.append(f'<p>{label}: {_badge(layers.get(key))}</p>')
    parts += ['</div></div><details><summary>Original domains and declared sizes</summary>',
              _table(["Domain", "Kind", "Declared size", "Decoded values supplied", "Source"],
                     [[key, row.get("kind"), _number(row.get("size")), len(row.get("values", [])), row.get("source_path")]
                      for key, row in sorted(decl.get("domains", {}).items())]), '</details></section>',
              '<section class="card"><h2>What the three costs mean</h2><div class="metrics">',
              '<div class="metric"><strong>Work</strong><span>Declared state-action evaluations plus source-row boundary incidence cost.</span></div>',
              '<div class="metric"><strong>Value entries</strong><span>Sum of allocated value-table entries; peak live or device memory is a separate quantity.</span></div>',
              '<div class="metric"><strong>Clarity</strong><span>Explicit weighted block, characteristic and dispatch complexity.</span></div></div>',
              '<p class="note">These are exact integer representation costs under the stated allocation contract. They do not measure hardware runtime. A block may allocate padding axes that are absent from an individual cell’s semantic state.</p></section>',
              '<section class="card"><h2>Proposal decisions</h2><p>Use the independent disposition for each proposal before changing a model. A cost preference alone does not authorize a transformation. Retaining the unchanged original carries no optimization or speed claim.</p><p><a href="proposal-decisions.json">Exact source-bound decision records</a></p></section>',
              '<div id="alternatives"><h2>Candidate alternatives</h2>']
    if not view["scenarios"]:
        parts.append('<p class="callout">No frontier scenario was supplied. No transformation recommendation or complete-discovery claim can be inferred from this report.</p>')
    for s in view["scenarios"]:
        scenario = s["scenario"];rec = scenario.get("recommendation") or {};cert = scenario.get("certificate") or scenario.get("search") or {}
        check = scenario.get("independent_check") or {}
        parts += [f'<section class="scenario"><div class="eyebrow">Scenario {s["ordinal"]}</div><h3>{_h(s["title"])}</h3>',
                  f'<p>Search {_badge(scenario.get("status"))} · independent check {_badge(check.get("status"))}</p>',
                  f'<p class="scope">{_h(cert.get("frontier_scope", "Frontier scope was not supplied."))}</p>']
        if s["conditional"]:
            parts.append('<p class="callout">This is a representation scenario. The actual backend allocation contract is unresolved, so these costs do not establish a production ranking.</p>')
        if scenario.get("status") == "HEURISTIC":
            parts.append('<p class="callout">Heuristic scope: use the recorded local admission and cost-check statuses. Better unexamined layouts may exist; the optimality gap is unknown. No global-family certificate is claimed.</p>')
        elif scenario.get("status") != "COMPLETE":
            parts.append(f'<p class="callout">Search is incomplete. Unexamined candidates may dominate these examined complete plans. Reason: {_h(scenario.get("exhaustion_reason", scenario.get("error", "Not supplied")))}. Recorded lower bound: {_h(cert.get("lower_bound"))}.</p>')
        if rec:
            chosen = next((p for p in s["points"] if p["cost_preferred"]), None)
            label = f'Option {chosen["ordinal"]}' if chosen else _text(rec.get("graph_id"))
            parts += [f'<h4>Cost preference: {_h(label)}</h4><p>Exact normalized score <strong>{_h(rec.get("score"))}</strong>.</p>',
                      _table(["Metric", "Positive weight", "Reference scale"], [[key, rec.get("weights", {}).get(key), _number(rec.get("reference_scales", {}).get(key))] for key in METRICS]),
                      '<p class="note">' + _h(_recommendation_note(scenario)) + '</p>']
            if _endpoint_note(rec):
                parts.append('<p class="callout">' + _h(_endpoint_note(rec)) + '</p>')
        parts.append(_scatter(s["points"]))
        parts.append(_table(["Option", "Cost preference", "Work", "Value entries", "Clarity", "Blocks"],
            [[p["ordinal"], "Yes" if p["cost_preferred"] else "", *[_number(p["point"].get("cost", {}).get(k)) for k in METRICS], len(p["point"].get("blocks", []))] for p in s["points"]], numeric=(2, 3, 4, 5)))
        if not s["points"]:
            parts.append('<p>No admissible complete candidate plan was supplied.</p>')
        parts += ['<details><summary>Search family, cost units and certificate</summary>',
                  f'<p>{_h(cert.get("family", "Not supplied"))}</p><p>Clarity weights: {_h(scenario.get("clarity_weights"))}. Boundary unit: {_h((scenario.get("allocation_contract") or {}).get("boundary_unit"))}.</p>',
                  f'<p>Search counts: {_h(cert.get("search_counts"))}. Family identity: {_h(cert.get("family_sha256"))}.</p></details>']
        for p in s["points"]:
            parts.append(f'<details{" open" if p["cost_preferred"] else ""}><summary>Option {p["ordinal"]}' + (" · cost preference" if p["cost_preferred"] else "") + f' · {_h(p["point"].get("id"))}</summary>')
            parts.append(f'<p><strong>Disposition: {_h(p["decision"].get("disposition", "investigate"))}</strong></p>')
            parts.append('<p class="scope">' + _h(p["decision"].get("reasons", [])) + '</p>')
            if (p["decision"].get("semantic_checks") or {}).get("status") != "PASS":
                parts.append('<p class="callout">The local source/semantic check has not passed. These are candidate changes, not an established source-equivalent transformation.</p>')
            parts.append('<h4>Consequences for this model</h4><ul>' + "".join(f'<li>{_h(x)}</li>' for x in p["consequences"]["sentences"]) + '</ul>')
            parts.append(_table(["Block", "Original regimes", "Declared dates", "Allocated states", "Allocated actions", "Work", "Value entries", "Clarity"],
                [[i, ", ".join(b.get("original_regimes", [])), _dates(b.get("ages", [])), _axes(b.get("state_axes"), view["domains"]), _axes(b.get("action_axes"), view["domains"]),
                  *[_number(b.get("cost", {}).get(k)) for k in METRICS]] for i, b in enumerate(p["point"].get("blocks", []), 1)], numeric=(5, 6, 7)))
            parts += [_graph_html(p["prefix"], p["graph"], view["preview_counts"]), '<details><summary>Exact cell projections, action reconstruction and source witnesses</summary>',
                      f'<pre>{_h(_json(p["consequences"]["selected_interfaces"]))}</pre></details>',
                      '<p class="note">Exact block cells, dates, allocation axes, projection records, source relations and all costs are retained in report-data.json.</p></details>']
        parts.append('</section>')
    parts += ['</div><section class="card" id="original"><h2>Original typed graph</h2>',
              '<p class="scope">Joint and multi-target relations retain explicit relation nodes. Phase, reporting, fallback and value-read links are labeled. Declared zero-support links are shown distinctly in Mermaid; they are not silently turned into positive transitions.</p>',
              _graph_html("original", view["original_graph"], view["preview_counts"]), '</section>',
              '<section class="card" id="obligations"><h2>Obligations and limits</h2>']
    obligations = _obligations(view)
    if obligations:
        parts.append(_table(["Scope", "Recorded obligation or failure"], [[x["scope"], _text(x["detail"])] for x in obligations]))
    else:
        parts.append('<p>No additional obligation record was supplied. Consult the individual source, coverage, allocation and numerical statuses above.</p>')
    parts += ['<p class="note">A complete optimizer certificate does not by itself establish source equivalence. Opaque callable interpretation, missing source facts, actual allocation, unsupported numerical operators and native solve/simulation validation retain their own scope.</p></section>',
              '<section class="card" id="provenance"><h2>Provenance and execution</h2>',
              f'<h4>Source identity</h4><pre>{_h(_json(result.get("source_identity", decl.get("source_identity"))))}</pre>',
              f'<h4>Recorded metadata execution</h4><pre>{_h(_json(analysis.get("production_execution", "Not supplied")))}</pre>',
              ('<h4>Resource record</h4><pre>' + _h(_json(result.get("resources", result.get("measurement")))) + '</pre>'
               if result.get("resources", result.get("measurement")) is not None else
               '<h4>Resource record</h4><p><a href="resource.json">Whole-command resource measurement</a> is supplied by the parent run wrapper after analysis, independent checking and initial report generation. No inline measurement was supplied.</p>'),
              '<p><a href="report.json">Complete unchanged result</a> · <a href="report-data.json">Exact graph and selected-interface records</a> · <a href="report-manifest.json">Generated file hashes</a> · <a href="report.md">Markdown report</a></p></section>',
              '<footer>PNG diagrams use deterministic, bounded standard-library rasterization. Labels are abbreviated only in the image; full graph topology, cells and exact source metadata remain in the accompanying files. Rendering performs no economic-model evaluation.</footer></main></body></html>']
    return "".join(parts)


# 5-by-7 bitmap font: each row is a five-bit mask. The PNG implementation has no
# optional font/image dependency and emits real RGB images on minimal installs.
_FONT_ROWS = {
    "A":"01110 10001 10001 11111 10001 10001 10001","B":"11110 10001 10001 11110 10001 10001 11110",
    "C":"01111 10000 10000 10000 10000 10000 01111","D":"11110 10001 10001 10001 10001 10001 11110",
    "E":"11111 10000 10000 11110 10000 10000 11111","F":"11111 10000 10000 11110 10000 10000 10000",
    "G":"01111 10000 10000 10111 10001 10001 01111","H":"10001 10001 10001 11111 10001 10001 10001",
    "I":"11111 00100 00100 00100 00100 00100 11111","J":"00111 00010 00010 00010 10010 10010 01100",
    "K":"10001 10010 10100 11000 10100 10010 10001","L":"10000 10000 10000 10000 10000 10000 11111",
    "M":"10001 11011 10101 10101 10001 10001 10001","N":"10001 11001 10101 10011 10001 10001 10001",
    "O":"01110 10001 10001 10001 10001 10001 01110","P":"11110 10001 10001 11110 10000 10000 10000",
    "Q":"01110 10001 10001 10001 10101 10010 01101","R":"11110 10001 10001 11110 10100 10010 10001",
    "S":"01111 10000 10000 01110 00001 00001 11110","T":"11111 00100 00100 00100 00100 00100 00100",
    "U":"10001 10001 10001 10001 10001 10001 01110","V":"10001 10001 10001 10001 10001 01010 00100",
    "W":"10001 10001 10001 10101 10101 10101 01010","X":"10001 10001 01010 00100 01010 10001 10001",
    "Y":"10001 10001 01010 00100 00100 00100 00100","Z":"11111 00001 00010 00100 01000 10000 11111",
    "0":"01110 10001 10011 10101 11001 10001 01110","1":"00100 01100 00100 00100 00100 00100 01110",
    "2":"01110 10001 00001 00010 00100 01000 11111","3":"11110 00001 00001 01110 00001 00001 11110",
    "4":"00010 00110 01010 10010 11111 00010 00010","5":"11111 10000 10000 11110 00001 00001 11110",
    "6":"01110 10000 10000 11110 10001 10001 01110","7":"11111 00001 00010 00100 01000 01000 01000",
    "8":"01110 10001 10001 01110 10001 10001 01110","9":"01110 10001 10001 01111 00001 00001 01110",
    " ":"00000 00000 00000 00000 00000 00000 00000","-":"00000 00000 00000 11111 00000 00000 00000",
    "_":"00000 00000 00000 00000 00000 00000 11111",".":"00000 00000 00000 00000 00000 00110 00110",
    ",":"00000 00000 00000 00000 00110 00110 00100",":":"00000 00110 00110 00000 00110 00110 00000",
    "/":"00001 00010 00010 00100 01000 01000 10000","|":"00100 00100 00100 00100 00100 00100 00100",
    "(":"00010 00100 01000 01000 01000 00100 00010",")":"01000 00100 00010 00010 00010 00100 01000",
    "+":"00000 00100 00100 11111 00100 00100 00000","=":"00000 11111 00000 11111 00000 00000 00000",
    ">":"10000 01000 00100 00010 00100 01000 10000","?":"01110 10001 00001 00010 00100 00000 00100",
    ";":"00000 00110 00110 00000 00110 00110 00100","%":"11001 11010 00100 01000 10110 00110 00000"}
FONT = {char: [int(row, 2) for row in pattern.split()] for char, pattern in _FONT_ROWS.items()}


def _rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


class _Raster:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.data = bytearray(bytes((247, 250, 252)) * width * height)

    def rectangle(self, x0, y0, x1, y1, color):
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(self.width, int(x1)), min(self.height, int(y1))
        if x1 <= x0 or y1 <= y0:
            return
        fill = bytes(color) * (x1-x0)
        for y in range(y0, y1):
            begin = (y*self.width+x0)*3
            self.data[begin:begin+len(fill)] = fill

    def line(self, x0, y0, x1, y1, color, width=2, dashed=False):
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        dx, dy = abs(x1-x0), -abs(y1-y0);sx = 1 if x0<x1 else -1;sy = 1 if y0<y1 else -1;err = dx+dy
        step = 0
        while True:
            if not dashed or step % 12 < 6:
                self.rectangle(x0, y0, x0+width, y0+width, color)
            if x0 == x1 and y0 == y1:
                return
            twice = 2*err
            if twice >= dy:
                err += dy;x0 += sx
            if twice <= dx:
                err += dx;y0 += sy
            step += 1

    def arrow(self, x0, y0, x1, y1, color, dashed=False):
        self.line(x0, y0, x1, y1, color, dashed=dashed)
        angle = math.atan2(y1-y0, x1-x0)
        for delta in (-.45, .45):
            self.line(x1, y1, x1-11*math.cos(angle+delta), y1-11*math.sin(angle+delta), color)

    def text(self, x, y, value, color=(23, 44, 66), scale=2, maximum=None):
        value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "replace").decode().upper()
        if maximum is not None and len(value) > maximum:
            value = value[:max(0, maximum-3)] + "..."
        for char in value:
            for row, mask in enumerate(FONT.get(char, FONT["?"])):
                for col in range(5):
                    if mask & (1 << (4-col)):
                        self.rectangle(x+col*scale, y+row*scale, x+(col+1)*scale, y+(row+1)*scale, color)
            x += 6*scale

    def save(self, path):
        def chunk(kind, payload):
            return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", binascii.crc32(kind+payload)&0xffffffff)
        scan = b"".join(b"\x00" + self.data[y*self.width*3:(y+1)*self.width*3] for y in range(self.height))
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(scan, 6)) + chunk(b"IEND", b"")
        Path(path).write_bytes(png)


def _write_graph_pages(graph, prefix, outdir, max_pages):
    nodes = graph["nodes"];all_pages = {node["id"]: i//PAGE_NODES+1 for i, node in enumerate(nodes)}
    files = []
    for page in range(min(_pages(graph), max_pages)):
        current = nodes[page*PAGE_NODES:(page+1)*PAGE_NODES]
        rows = max(1, math.ceil(len(current)/3));height = 160+rows*146+130
        canvas = _Raster(1000, height)
        canvas.rectangle(0, 0, 1000, 85, (23, 44, 66))
        canvas.text(24, 18, graph["title"], (255, 255, 255), maximum=77)
        canvas.text(24, 48, f"PAGE {page+1} OF {_pages(graph)} | TYPED DECLARED RELATIONS", (183, 216, 225), maximum=77)
        positions = {}
        for index, node in enumerate(current):
            x, y = 24+(index%3)*328, 112+(index//3)*146
            positions[node["id"]] = (x, y)
        incoming, outgoing = Counter(), Counter()
        for edge in graph["edges"]:
            s, t = edge["source"], edge["target"]
            if s in positions and t in positions:
                x, y = positions[s];xx, yy = positions[t]
                color = _rgb(COLORS.get(edge["kind"], "#334155"))
                if s == t:
                    canvas.line(x+250, y, x+250, y-12, color, dashed=edge.get("support") == "zero")
                    canvas.line(x+250, y-12, x+20, y-12, color, dashed=edge.get("support") == "zero")
                    canvas.arrow(x+20, y-12, x+20, y, color, dashed=edge.get("support") == "zero")
                    continue
                if yy == y:
                    start, end = ((x+296, y+53), (xx, yy+53)) if xx>x else ((x, y+53), (xx+296, yy+53))
                else:
                    start, end = ((x+148, y+106), (xx+148, yy)) if yy>y else ((x+148, y), (xx+148, yy+106))
                canvas.arrow(*start, *end, color, dashed=edge.get("support") == "zero")
            elif s in positions:
                outgoing[s] += 1
            elif t in positions:
                incoming[t] += 1
        for node in current:
            x, y = positions[node["id"]]
            border = _rgb("#7c3aed" if node["kind"] == "joint" else "#147d86")
            canvas.rectangle(x, y, x+296, y+106, border)
            canvas.rectangle(x+2, y+2, x+294, y+104, (255, 255, 255))
            canvas.rectangle(x+2, y+2, x+7, y+104, border)
            heading = node["id"] + " " + node["label"]
            canvas.text(x+14, y+11, heading, maximum=22)
            canvas.text(x+14, y+33, "DATES: " + _dates(node["dates"], 4) if node["kind"] != "joint" else "FULL JOINT / MULTI-TARGET", (70, 90, 110), maximum=22)
            canvas.text(x+14, y+55, "STATE: " + (", ".join(node["states"]) or "NONE"), (70, 90, 110), maximum=22)
            canvas.text(x+14, y+77, "ACTION: " + (", ".join(node["actions"]) or "NONE"), (70, 90, 110), maximum=22)
            if incoming[node["id"]] or outgoing[node["id"]]:
                canvas.text(x+8, y+116, f'OTHER PAGES: {incoming[node["id"]]} IN / {outgoing[node["id"]]} OUT', (98, 74, 25), scale=1, maximum=46)
        baseline = 126+rows*146
        kinds = sorted({edge["kind"] for edge in graph["edges"]})
        for i, kind in enumerate(kinds[:8]):
            x, y = 24+(i%4)*244, baseline+(i//4)*28
            canvas.rectangle(x, y+2, x+15, y+15, _rgb(COLORS.get(kind, "#334155")))
            canvas.text(x+23, y+2, KIND_NAMES.get(kind, kind), scale=1, maximum=31)
        if any(edge.get("support") == "zero" for edge in graph["edges"]):
            canvas.text(24, height-53, "DASHED LINKS: DECLARED EXACT ZERO SUPPORT", (82, 101, 119), scale=1, maximum=155)
        canvas.text(24, height-33, "FULL PHASE LABELS, EXACT DATES AND CROSS-PAGE EDGES: MERMAID / REPORT-DATA.JSON", (82, 101, 119), scale=1, maximum=155)
        filename = f"graphs/{prefix}-page-{page+1:03d}.png"
        canvas.save(Path(outdir)/filename);files.append(filename)
    return files


def write_production_report(result, outdir):
    """Write HTML, Markdown, exact JSON, full Mermaid and paginated graph PNGs.

    Returns a JSON-serializable manifest. Filenames never depend on user labels;
    all user text is escaped in HTML/Markdown and sanitized inside Mermaid labels.
    Existing files with the same generated names are replaced deterministically.
    """
    outdir = Path(outdir);outdir.mkdir(parents=True, exist_ok=True);(outdir/"graphs").mkdir(exist_ok=True)
    from .structural_store import relocate_captures
    result = relocate_captures(result, outdir)
    view = _view(result)
    files = []
    def save(name, content):
        (outdir/name).write_text(content, encoding="utf-8");files.append(name)
    save("report.json", _json(result))
    save("report.md", render_markdown(result, _prepared_view=view))
    save("report.html", render_html(result, _prepared_view=view))
    save("proposal-decisions.json", _json(result.get("proposal_decisions", [])))
    graphs = [("original", view["original_graph"])]
    data = {"schema": "regime-render-data/v1", "result_status": result.get("status"),
            "source_identity": result.get("source_identity"), "original_cells": view["declarations"].get("cells", []),
            "original_relations": view["declarations"].get("relations", []),
            "original_graph": view["original_graph"], "png_preview_counts": view["preview_counts"],
            "presentation_only_png_limit": MAX_PNG_PAGES, "scenarios": []}
    for s in view["scenarios"]:
        record = {"ordinal": s["ordinal"], "allocation_contract": s["scenario"].get("allocation_contract"),
                  "status": s["scenario"].get("status"), "recommendation": s["scenario"].get("recommendation"), "alternatives": []}
        for p in s["points"]:
            record["alternatives"].append({"ordinal": p["ordinal"], "cost_preferred": p["cost_preferred"],
                "plan": p["point"], "decision": p["decision"],
                "consequences": p["consequences"], "graph": p["graph"]})
            graphs.append((p["prefix"], p["graph"]))
        data["scenarios"].append(record)
    save("report-data.json", _json(data))
    for prefix, graph in graphs:
        save(f"graphs/{prefix}.mmd", graph_mermaid(graph))
        files.extend(_write_graph_pages(graph, prefix, outdir, view["preview_counts"].get(prefix, 0)))
    manifest = {"schema": "regime-report-manifest/v1", "status": "WRITTEN",
                "renderer": "stdlib deterministic HTML/Markdown/Mermaid/RGB PNG",
                "input_status": result.get("status"), "models": 1,
                "scenario_count": len(view["scenarios"]), "frontier_alternative_count": sum(len(s["points"]) for s in view["scenarios"]),
                "graph_count": len(graphs), "png_page_count": sum(view["preview_counts"].values()),
                "png_preview_limit": MAX_PNG_PAGES,
                "omitted_preview_page_count": sum(_pages(graph) for _, graph in graphs) - sum(view["preview_counts"].values()),
                "preview_priority": "cost-preferred alternatives, original graph, other alternatives",
                "max_nodes_per_png_page": PAGE_NODES, "max_png_width": 1000, "max_png_height": 874,
                "numerical_callbacks_executed": 0, "rendered_from_recorded_result_only": True,
                "files": [{"path": name, "bytes": (outdir/name).stat().st_size,
                           "sha256": hashlib.sha256((outdir/name).read_bytes()).hexdigest()} for name in files]}
    (outdir/"report-manifest.json").write_text(_json(manifest), encoding="utf-8")
    return manifest
