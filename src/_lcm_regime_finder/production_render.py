"""Portable reports of a model's layout curve.

No producer, scorer, callback, solver, browser, plotting package or image package
is imported. Graph PNGs use a small deterministic standard-library rasterizer.
The saved report.json is the unchanged result; report-data.json adds the graph of
every curve layout.
"""
from __future__ import annotations

import binascii
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re
import struct
import textwrap
import unicodedata
import zlib


METRICS = ("work", "storage", "clarity")
PAGE_NODES = 12
MAX_PNG_PAGES = 100
COLORS = {"transition": "#147d86", "joint": "#7c3aed", "imputation": "#087f9b",
          "value_read": "#2563eb", "gate": "#a16207", "composition_action": "#c2410c",
          "fallback": "#64748b", "reporting": "#be185d"}
KIND_NAMES = {"transition": "Transition", "joint": "Joint lottery", "imputation": "Imputation",
              "value_read": "Value read", "gate": "Gate", "composition_action": "Composition action",
              "fallback": "Fallback", "reporting": "Reporting association"}
STATUS_NAMES = {"RESTRICTED": "Upper bound: some statuses kept as ordinary coordinates",
                "COMPLETE": "Complete within the stated family", "INCOMPLETE": "Incomplete",
                "UNRESOLVED": "Unresolved obligations", "INVALID": "Verification failed",
                "FAIL": "Failed", "PASS": "Passed", "VERIFIED": "Independently checked",
                "DECLARED": "Declared contract", "NOT_RUN": "Not run", "UNKNOWN": "Not supplied"}
LAYER_NAMES = {"facts": "Dependency facts backend", "source_semantics": "Source and semantic check",
               "layouts": "Layout search", "layout_check": "Independent layout check",
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
    name = (result.get("name") or declarations.get("name") or identity.get("project")
            or identity.get("name") or identity.get("model"))
    return name if name and not re.fullmatch(r'[0-9a-f]{32,64}', str(name)) else "Regime structure analysis"


def _dates(values, limit=None):
    values = list(dict.fromkeys(_text(x) for x in values))
    if not values:
        return "Not supplied"
    if limit and len(values) > limit:
        return ", ".join(values[:3]) + " … " + ", ".join(values[-2:]) + f" ({len(values)} declared dates)"
    return ", ".join(values)


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
                          "dates": [cell.get("age") if cell.get("age") is not None else cell.get("period")], "cells": [cell["id"]],
                          "states": list(cell.get("states", {})), "actions": list(cell.get("actions", {})),
                          "stage": cell.get("stage"), "terminal": bool(cell.get("terminal")), "order": i * 10})
    else:
        for i, block in enumerate(plan.get("blocks", [])):
            key = f"N{i}";ordering[key] = i
            for cid in block.get("cells", []):
                owner[cid] = key
            nodes.append({"id": key, "kind": "block", "label": " / ".join(block.get("original_regimes", [])) or f"Block {i+1}",
                          "dates": [cells[cid].get('age') if cells[cid].get('age') is not None else cells[cid].get('period') for cid in block.get('cells', [])], "cells": block.get("cells", []),
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


def _pages(graph):
    return max(1, math.ceil(len(graph["nodes"]) / PAGE_NODES)) if graph["nodes"] else 0


def _image_names(prefix, graph, previews=None):
    count = _pages(graph) if previews is None else previews.get(prefix, 0)
    return [f"graphs/{prefix}-page-{i+1:03d}.png" for i in range(count)]


def _preview_note(prefix, graph, previews):
    shown, total = previews.get(prefix, 0), _pages(graph)
    if shown < total:
        return f"PNG preview limit: {shown} of {total} pages shown. The full graph remains in Mermaid and report-data.json. This presentation limit does not change the curve."
    return ""


CSS = """
.graphs{overflow-x:auto}.graphs svg{display:block;width:1060px;max-width:none;height:auto}.graphs figure{overflow-x:auto;max-width:100%}
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
    for page, filename in enumerate(_image_names(prefix, graph, previews)):
        nodes = graph['nodes'][page*PAGE_NODES:(page+1)*PAGE_NODES]
        labels = {node['id']: [line for label in (
            node['label'], 'Dates: '+_dates(node['dates']),
            'States: '+(', '.join(node['states']) or 'none'),
            'Actions: '+(', '.join(node['actions']) or 'none'))
            for line in textwrap.wrap(label, width=34)] for node in nodes}
        box_height = 30 + max(len(lines) for lines in labels.values())*18
        positions = {node['id']: (10+(index%3)*350, 12+(index//3)*(box_height+45)) for index,node in enumerate(nodes)}
        height = math.ceil(len(nodes)/3)*(box_height+45)+20
        marker = prefix+'-arrow-'+str(page)
        svg = [f'<svg viewBox="0 0 1060 {height}" role="img" aria-label="{_h(graph["title"])}"><defs><marker id="{marker}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#526577"/></marker></defs>']
        for edge in graph['edges']:
            if edge['source'] in positions and edge['target'] in positions:
                x,y=positions[edge['source']];xx,yy=positions[edge['target']]
                svg.append(f'<path d="M{x+165},{y+box_height} L{xx+165},{yy}" stroke="#526577" fill="none" marker-end="url(#{marker})"><title>{_h(KIND_NAMES.get(edge["kind"],edge["kind"]))} · {_h(edge.get("phase"))}</title></path>')
        for node in nodes:
            x,y=positions[node['id']]
            svg.append(f'<g><title>{_h(node["label"])}</title><rect x="{x}" y="{y}" width="330" height="{box_height}" rx="6" fill="white" stroke="#147d86"/>')
            for index,line in enumerate(labels[node['id']]):
                svg.append(f'<text x="{x+12}" y="{y+22+index*18}" font-size="14" fill="#172c42">{_h(line)}</text>')
            svg.append('</g>')
        svg.append('</svg>')
        result.append('<figure>'+''.join(svg)+f'<figcaption>Page {page+1} of {count}. Full labels are wrapped. <a href="{filename}">Download PNG preview</a>; exact cross-page relations remain in Mermaid and JSON.</figcaption></figure>')
    return "".join(result) + "</div>"


def _plan(point, declarations):
    """A curve layout as blocks of original cells, for `_graph`."""
    by_place = {(c["period"], c["regime"]): c["id"] for c in declarations.get("cells", [])}
    blocks = []
    for group in point["layout"]:
        for block in group["blocks"]:
            fibres = block["fibres"]
            blocks.append({"cells": list(dict.fromkeys(by_place[f["period"], f["regime"]] for f in fibres)),
                           "original_regimes": sorted({f["regime"] for f in fibres}),
                           "state_axes": block["states"], "action_axes": block["actions"]})
    return {"blocks": blocks}


def _view(result):
    analysis, declarations = _declarations(result)
    layouts = result.get("layouts") or {}
    original = _graph(declarations)
    points = []
    for point in layouts.get("curve", []):
        prefix = f"layout-k{point['max_blocks']:03d}"
        points.append({"prefix": prefix, "point": point,
                       "graph": _graph(declarations, _plan(point, declarations),
                                       f"Layout with at most {point['max_blocks']} regimes")})
    chosen = result.get("recommendation") or {}
    index = chosen.get("curve_index") if chosen.get("operation") == "layout" else None
    order = ([points[index]] if index is not None else []) + [("original", original)]
    order = [(p["prefix"], p["graph"]) if isinstance(p, dict) else p for p in order]
    order += [(p["prefix"], p["graph"]) for i, p in enumerate(points) if i != index]
    previews, remaining = {}, MAX_PNG_PAGES
    for prefix, graph in order:
        previews[prefix] = min(_pages(graph), remaining); remaining -= previews[prefix]
    return {"result": result, "analysis": analysis, "declarations": declarations, "layouts": layouts,
            "name": _model_name(result, declarations), "original_graph": original, "points": points,
            "recommended": index, "preview_counts": previews}


def _curve_rows(view):
    return [("recommended" if i == view["recommended"] else "", p["point"]["max_blocks"], p["point"]["blocks"],
             _number(p["point"]["evaluations"]), _number(p["point"]["storage"]))
            for i, p in enumerate(view["points"])]


def _block_rows(point):
    return [(" / ".join(sorted({f["regime"] for f in b["fibres"]})), _dates(b["periods"], 6),
             ", ".join(b["states"]) or "none", ", ".join(b["actions"]) or "none",
             _number(b["evaluations"]), _number(b["storage"]))
            for g in point["layout"] for b in g["blocks"]]


def _headline(view):
    result, layouts = view["result"], view["layouts"]
    chosen = result.get("recommendation") or {}
    if not layouts:
        return "No layout curve was produced: " + _text(result.get("error") or chosen.get("reason")) + "."
    text = f'The curve has {len(view["points"])} points. '
    if layouts.get("status") == "RESTRICTED":
        text += "Statuses " + ", ".join(layouts.get("restricted_statuses", [])) + " were kept as ordinary coordinates, so every count is an upper bound. "
    if chosen.get("operation") == "layout":
        return text + f'The cost model recommends the layout with at most {chosen["max_blocks"]} regimes.'
    return text + "The cost model retains the original model: " + _text(chosen.get("reason")) + "."


def render_markdown(result, *, _prepared_view=None):
    view = _prepared_view or _view(result)
    original = result.get("original_cost") or {}
    lines = ["# " + _md(view["name"]), "", _md(_headline(view)), "",
             "## Checks", ""]
    lines += [f"- {_md(LAYER_NAMES.get(k, k))}: {_md(_status(v))}" for k, v in (result.get("layers") or {}).items()]
    lines += ["", "## Original model", "",
              f'Evaluations {_md(_number(original.get("work")))}, stored values {_md(_number(original.get("storage")))}, '
              f'regimes {_md(_number(original.get("clarity")))}.', "",
              "## Layout curve", "", "| | at most k | regimes | evaluations | stored values |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(_md(x) for x in row) + " |" for row in _curve_rows(view)]
    for i, p in enumerate(view["points"]):
        lines += ["", f'## At most {p["point"]["max_blocks"]} regimes' + (" (recommended)" if i == view["recommended"] else ""), "",
                  "| regimes | periods | states | actions | evaluations | stored values |", "|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join(_md(x) for x in row) + " |" for row in _block_rows(p["point"])]
        lines += ["", f'Graph: [graphs/{p["prefix"]}.mmd](graphs/{p["prefix"]}.mmd).']
    lines += ["", "No recommendation is applied automatically; native solve and simulation were not run.", ""]
    return "\n".join(lines)


def render_html(result, *, _prepared_view=None):
    view = _prepared_view or _view(result)
    original = result.get("original_cost") or {}
    layers = "".join(f"<li>{_h(LAYER_NAMES.get(k, k))}: {_badge(v)}</li>" for k, v in (result.get("layers") or {}).items())
    parts = [f'<main><p class="eyebrow">Regime layout analysis</p><h1>{_h(view["name"])}</h1>',
             f'<p class="lead">{_h(_headline(view))}</p><p>Status: {_badge(result.get("status"))}</p><ul>{layers}</ul>',
             '<section class="card"><h2>Original model</h2><div class="metrics">' +
             "".join(f'<div class="metric"><strong>{_h(_number(original.get(k)))}</strong><span>{label}</span></div>'
                     for k, label in (("work", "evaluations"), ("storage", "stored values"), ("clarity", "regimes"))) +
             "</div>" + _graph_html("original", view["original_graph"], view["preview_counts"]) + "</section>",
             '<section class="card"><h2>Layout curve</h2>' +
             _table(("", "at most k", "regimes", "evaluations", "stored values"), _curve_rows(view), numeric=(1, 2, 3, 4)) + "</section>"]
    for i, p in enumerate(view["points"]):
        title = f'At most {p["point"]["max_blocks"]} regimes' + (" (recommended)" if i == view["recommended"] else "")
        parts.append(f'<section class="card"><h2>{_h(title)}</h2>' +
                     _table(("regimes", "periods", "states", "actions", "evaluations", "stored values"),
                            _block_rows(p["point"]), numeric=(4, 5)) +
                     _graph_html(p["prefix"], p["graph"], view["preview_counts"]) + "</section>")
    parts.append("<footer>No recommendation is applied automatically; native solve and simulation were not run.</footer></main>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_h(view["name"])}</title><style>{CSS}</style></head><body>' + "".join(parts) + "</body></html>")


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
    graphs = [("original", view["original_graph"])] + [(p["prefix"], p["graph"]) for p in view["points"]]
    data = {"schema": "regime-render-data/v2", "result_status": result.get("status"),
            "source_identity": result.get("source_identity"), "original_cells": view["declarations"].get("cells", []),
            "original_relations": view["declarations"].get("relations", []),
            "original_graph": view["original_graph"], "png_preview_counts": view["preview_counts"],
            "presentation_only_png_limit": MAX_PNG_PAGES,
            "layouts": [{"max_blocks": p["point"]["max_blocks"], "graph": p["graph"]} for p in view["points"]]}
    save("report-data.json", _json(data))
    for prefix, graph in graphs:
        save(f"graphs/{prefix}.mmd", graph_mermaid(graph))
        files.extend(_write_graph_pages(graph, prefix, outdir, view["preview_counts"].get(prefix, 0)))
    manifest = {"schema": "regime-report-manifest/v1", "status": "WRITTEN",
                "renderer": "stdlib deterministic HTML/inline SVG/Markdown/Mermaid/RGB PNG",
                "input_status": result.get("status"), "models": 1,
                "curve_points": len(view["points"]),
                "graph_count": len(graphs), "png_page_count": sum(view["preview_counts"].values()),
                "png_preview_limit": MAX_PNG_PAGES,
                "omitted_preview_page_count": sum(_pages(graph) for _, graph in graphs) - sum(view["preview_counts"].values()),
                "preview_priority": "recommended layout, original graph, other layouts",
                "max_nodes_per_png_page": PAGE_NODES, "max_png_width": 1000, "max_png_height": 874,
                "numerical_callbacks_executed": 0, "rendered_from_recorded_result_only": True,
                "files": [{"path": name, "bytes": (outdir/name).stat().st_size,
                           "sha256": hashlib.sha256((outdir/name).read_bytes()).hexdigest()} for name in files]}
    (outdir/"report-manifest.json").write_text(_json(manifest), encoding="utf-8")
    return manifest
