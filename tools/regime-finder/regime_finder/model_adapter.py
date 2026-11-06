"""Adapt a constructed pylcm graph, independently of its Python builder.

The graph and finalized declarations are authoritative for timing, grids and
phase support. Callback bodies are inspected only for conservative dependency
and exact Boolean simplification. They are never numerically sampled.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .source_adapter import CallbackAnalysis, Function, SourceAdapterError
from .callable_identity import primitive_operation


class ModelAdapterError(ValueError):
    """The constructed graph lacks a required observable declaration."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, list):
        return [plain(x) for x in value]
    if not isinstance(value, dict):
        return value
    if value.get("kind") == "array":
        return plain(value.get("values", value)) if "values" in value else value
    if "fraction" in value:
        return Fraction(value["fraction"])
    if value.get("kind") == "categorical-class":
        return {k: plain(v) for k, v in value["members"].items()}
    if value.get("kind") == "dataclass":
        return {k: plain(v) for k, v in value.get("fields", {}).items()}
    return value


def fields(value: Any) -> dict[str, Any]:
    return value.get("fields", {}) if isinstance(value, dict) else {}


class _RuntimeScope:
    def __init__(self, bindings: dict[str, Any]):
        self.bindings = {k: plain(v) for k, v in bindings.items()}

    def lookup(self, name: str) -> Any:
        if name not in self.bindings:
            raise SourceAdapterError("Unbound runtime scalar " + name)
        value = self.bindings[name]
        if isinstance(value, dict) and "kind" in value:
            raise SourceAdapterError("Opaque runtime binding " + name)
        return value

    def evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return self.lookup(node.id)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Attribute):
            parent = self.evaluate(node.value)
            if isinstance(parent, dict) and node.attr in parent:
                return parent[node.attr]
        raise SourceAdapterError("Unresolved runtime constant")


class _RuntimeModule:
    def __init__(self, metadata: dict[str, Any], callable_records=None):
        self.metadata = metadata
        self.callable_records = callable_records or metadata.get("callable_records", {})
        self.digest = digest({key: value for key, value in metadata.items() if key not in {"callable_records", "capture_report"}})
        self.path = Path(str(metadata.get("source_file", "generated-callback")) + "#" + self.digest)
        self.functions: dict[str, Function] = {}
        bindings = {**metadata.get("bindings", {}), **metadata.get("partial_keywords", {})}
        self.scope = _RuntimeScope(bindings)

    def anchor(self, node: ast.AST) -> dict[str, Any]:
        return {"path": self.metadata.get("source_file"), "sha256": self.metadata.get("source_sha256"),
                "line": self.metadata.get("source_line"), "callable_fingerprint": self.digest}


class _GraphAnalysis(CallbackAnalysis):
    def predicate(self, entry: Any) -> Any:
        if isinstance(entry, Function) and entry.module.metadata.get("analysis_only_stub"):
            return self.opaque(entry)
        return super().predicate(entry)

    def value(self, entry: Any) -> Any:
        """Preserve conditional argument effects without evaluating numerical code."""
        if isinstance(entry, dict):
            return {"opaque": self.opaque(entry)["opaque"], "reads": [],
                    "arguments": [self.value(value) for _, value in sorted(entry.items())]}
        if not isinstance(entry, Function):
            return self.opaque(entry)
        if entry.module.metadata.get("analysis_only_stub"):
            return self.opaque(entry)
        local = {}
        assigned = {arg.arg for arg in entry.node.args.posonlyargs + entry.node.args.args + entry.node.args.kwonlyargs}
        for stmt in entry.node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
                if any(t.id in assigned for t in stmt.targets):
                    return self.opaque(entry)
                for target in stmt.targets:
                    local[target.id] = stmt.value
                    assigned.add(target.id)
            elif isinstance(stmt, ast.Return):
                return self._value_node(entry, stmt.value, local)
            elif isinstance(stmt, ast.Delete):
                continue
            else:
                return self.opaque(entry)
        return self.opaque(entry)

    def _value_node(self, fn: Function, node: ast.AST, local: dict, seen: tuple = ()) -> Any:
        if isinstance(node, ast.Name) and node.id in local and node.id not in seen:
            return self._value_node(fn, local[node.id], local, seen + (node.id,))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name):
            try:
                return self._scalar(fn, node, local, seen)
            except SourceAdapterError:
                return self._predicate_node(fn, node, local, seen)
        if isinstance(node, ast.IfExp):
            return {"if": self._predicate_node(fn, node.test, local, seen),
                    "then": self._value_node(fn, node.body, local, seen),
                    "else": self._value_node(fn, node.orelse, local, seen)}
        if isinstance(node, ast.Call) and primitive_operation(fn.module.metadata, node.func, fn.module.callable_records) == "select" and len(node.args) == 3 and not node.keywords:
            return {"if": self._predicate_node(fn, node.args[0], local, seen),
                    "then": self._value_node(fn, node.args[1], local, seen),
                    "else": self._value_node(fn, node.args[2], local, seen)}
        children = None
        if isinstance(node, ast.BinOp):
            children = [node.left, node.right]
        elif isinstance(node, ast.UnaryOp):
            children = [node.operand]
        elif isinstance(node, ast.Call):
            children = list(node.args) + [kw.value for kw in node.keywords]
            if isinstance(node.func, ast.Attribute):
                children.append(node.func.value)
        elif isinstance(node, ast.Subscript):
            children = [node.value, node.slice]
        if children is not None:
            return {"opaque": digest({"body": ast.dump(node, include_attributes=False), "binding": fn.module.digest}),
                    "reads": [], "arguments": [self._value_node(fn, child, local, seen) for child in children],
                    "source_anchor": fn.module.anchor(node)}
        return self._predicate_node(fn, node, local, seen)

    def leaf_reads(self, entry: Any, seen: tuple = ()) -> set[str]:
        result = super().leaf_reads(entry, seen)
        if isinstance(entry, Function) and getattr(self, "period_only", False) and "age" in self.function_reads(entry):
            result.discard("age")
            if "age" in self.bindings:
                if "age" in seen:
                    raise SourceAdapterError("Cyclic economic-age provider")
                result |= self.leaf_reads(self.bindings["age"], seen + ("age",))
            else:
                result.add("source__age")
                self.external_parameters.setdefault("source__age", []).append({"function": entry.node.name, "source_anchor": entry.module.anchor(entry.node)})
        return result

    def _scalar(self, fn: Function, node: ast.AST, local: dict, seen: tuple = ()) -> str:
        if isinstance(node, ast.Name) and node.id == "period":
            return self.period_expression
        if isinstance(node, ast.Name) and node.id == "age" and getattr(self, "period_only", False):
            if "age" not in self.bindings or getattr(self, "resolving_age", False):
                raise SourceAdapterError("Economic age is not the period counter")
            self.resolving_age = True
            try:
                value = self.predicate(self.bindings["age"])
            finally:
                self.resolving_age = False
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float, bool)):
                return repr(value)
            raise SourceAdapterError("Economic-age provider is opaque")
        return super()._scalar(fn, node, local, seen)

    def opaque(self, entry: Any) -> dict[str, Any]:
        if isinstance(entry, dict):
            return {"opaque": digest({key: self.opaque(value)["opaque"] for key, value in sorted(entry.items())}),
                    "reads": sorted(self.leaf_reads(entry))}
        result = super().opaque(entry)
        if isinstance(entry, Function):
            result["opaque"] = digest({"source": result["opaque"], "runtime_binding": entry.module.digest})
        return result

    def _predicate_node(self, fn: Function, node: ast.AST, local: dict, seen: tuple = ()) -> Any:
        if isinstance(node, ast.IfExp):
            return {"if": self._predicate_node(fn, node.test, local, seen),
                    "then": self._predicate_node(fn, node.body, local, seen),
                    "else": self._predicate_node(fn, node.orelse, local, seen)}
        if isinstance(node, ast.Call):
            name = primitive_operation(fn.module.metadata, node.func, fn.module.callable_records)
            if name == "select" and len(node.args) == 3 and not node.keywords:
                return {"if": self._predicate_node(fn, node.args[0], local, seen),
                        "then": self._predicate_node(fn, node.args[1], local, seen),
                        "else": self._predicate_node(fn, node.args[2], local, seen)}
            if name in {"logical_and", "logical_or"} and len(node.args) == 2:
                return {"and" if name == "logical_and" else "or":
                        [self._predicate_node(fn, arg, local, seen) for arg in node.args]}
        return super()._predicate_node(fn, node, local, seen)


def _callback(value: Any, coordinates: set[str], notes: list[str], callable_records=None) -> Any:
    if isinstance(value, dict) and value.get("kind") == "callable":
        metadata = value
        if metadata.get("capture_status", "COMPLETE") != "COMPLETE":
            notes.append("Changed/incomplete capture: retain all original coordinates")
            metadata = {**metadata, "arguments": sorted(coordinates), "variadic": True, "source": ""}
        if "/tests/" in (metadata.get("source_file") or ""):
            notes.append("Test-defined callback: retain all original coordinates; scientific source bridge unresolved")
            metadata = {**metadata, "arguments": sorted(coordinates), "variadic": True, "source": ""}
        module = _RuntimeModule(metadata, callable_records)
        source = metadata.get("source", "")
        try:
            tree = ast.parse(source)
            fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef))
            actual = metadata.get("arguments", [])
            original = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
            if set(actual) != set(original) or metadata.get("variadic"):
                raise ValueError("Evaluated signature differs from wrapped source")
        except (SyntaxError, StopIteration, ValueError):
            metadata = {**metadata, "analysis_only_stub": True}
            module = _RuntimeModule(metadata, callable_records)
            args = metadata.get("arguments", [])
            if metadata.get("variadic"):
                args = sorted(coordinates)
                notes.append("Unavailable variadic callback: all original coordinates retained")
            fn = ast.parse("def callback(" + ",".join(args) + "):\n    return (" + ",".join(args) + ",)\n" if args
                           else "def callback():\n    return 0\n").body[0]
        fn.lineno = 1
        fn.end_lineno = getattr(fn, "end_lineno", 1)
        return Function(module, fn)
    if isinstance(value, dict) and value.get("kind") in {"dataclass", "object"}:
        attrs = fields(value)
        short = value.get("type", "").split(".")[-1]
        if short.endswith("SpecializedFunction"):
            resolved = value.get("resolved_by_period", [])
            if resolved:
                return {str(v["period"]): _callback(v["value"], coordinates, notes, callable_records) for v in resolved}
            notes.append("Unresolved dated function: all original coordinates retained")
            return _callback({"kind": "callable", "variadic": True, "arguments": sorted(coordinates)}, coordinates, notes, callable_records)
        if short == "Phased":
            return {phase: _callback(attrs[phase], coordinates, notes, callable_records) for phase in ("solve", "simulate")}
        if short in {"StochasticTransition", "DeterministicTransition"}:
            for name in ("func", "function"):
                if name in attrs:
                    return _callback(attrs[name], coordinates, notes, callable_records)
        if short == "_Constant":
            return _callback({"kind": "callable", "source": "def fixed():\n    return 0\n",
                              "arguments": [], "constant_payload": value, "analysis_only_stub": True}, coordinates, notes, callable_records)
        if short == "_PeriodDispatch" and attrs.get("cases"):
            return {str(i): _callback(v, coordinates, notes, callable_records) for i, v in enumerate(attrs["cases"])}
        # Period-dispatched laws and metadata carriers conservatively retain
        # the union of their original callback inputs across every date.
        nested = [v for v in _walk(value) if isinstance(v, dict) and v.get("kind") == "callable"]
        if nested:
            return {str(i): _callback(v, coordinates, notes, callable_records) for i, v in enumerate(nested)}
    if isinstance(value, dict) and "kind" not in value:
        return {str(k): _callback(v, coordinates, notes, callable_records) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _callback({"kind": "callable", "source": "def constant():\n    return 0\n", "arguments": [], "analysis_only_stub": True, "original_payload": value}, coordinates, notes, callable_records)
    notes.append("Opaque declaration requires all original state/action inputs")
    return _callback({"kind": "callable", "variadic": True, "arguments": sorted(coordinates)}, coordinates, notes, callable_records)


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _grid(value: dict[str, Any]) -> dict[str, Any]:
    attrs = fields(value)
    short = value.get("type", "").split(".")[-1]
    if short == "Phased":
        solve, simulate = attrs.get("solve"), attrs.get("simulate")
        imputed = isinstance(solve, dict) and solve.get("kind") == "callable"
        result = _grid(simulate if imputed else solve)
        result.update(grid_id=digest(value), solve_axis=not imputed, phase_declaration=value)
        return result
    if short == "ConditionalAction":
        result = _grid(attrs["grid"])
        result.update(grid_id=digest(value), conditional_action=attrs)
        return result
    if short in {"AgeSpecializedGrid", "PeriodSpecializedGrid"}:
        resolved = value.get("resolved_representative")
        if resolved:
            count = value.get("resolved_cardinality")
            result = ({"size": count, "kind": short} if isinstance(count, int) and count > 0 else _grid(resolved))
            result.update(grid_id=digest(value), specialized_declaration=value)
            return result
    result: dict[str, Any] = {"grid_id": digest(value), "kind": short, "declaration": value}
    if short == "DiscreteGrid":
        codes = plain(attrs.get("codes"))
        vals = plain(attrs.get("values"))
        if vals is None:
            vals = codes
        if not isinstance(vals, list) or not vals:
            raise ModelAdapterError("Discrete grid does not expose its codes/values")
        result.update(values=vals, codes=codes, category_labels=attrs.get("categories"))
        return result
    count = plain(attrs.get("n_points"))
    if count is None and isinstance(attrs.get("points_per_segment"), list):
        counts = plain(attrs["points_per_segment"])
        if all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in counts):
            count = sum(counts)
    points = plain(attrs.get("points"))
    if count is None and isinstance(points, list):
        count = len(points)
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        result["size"] = count
        return result
    raise ModelAdapterError("Grid cardinality is not available: " + short)


def _rewrite(expr: Any, replacements: dict[str, str], scalar_replacements: dict[str, str] | None = None) -> Any:
    scalar_replacements = scalar_replacements or replacements
    if isinstance(expr, list):
        return [_rewrite(v, replacements, scalar_replacements) for v in expr]
    if isinstance(expr, dict):
        if "opaque" in expr:
            return {**expr, "reads": sorted({replacements.get(x, x) for x in expr["reads"]}),
                    **({"arguments": [_rewrite(v, replacements, scalar_replacements) for v in expr["arguments"]]} if "arguments" in expr else {})}
        return {k: _rewrite(v, replacements, scalar_replacements) for k, v in expr.items()}
    if not isinstance(expr, str):
        return expr
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return ast.copy_location(ast.parse(scalar_replacements.get(node.id, node.id), mode="eval").body, node)
    return ast.unparse(Rename().visit(ast.parse(expr, mode="eval")).body)


def _lanes(nodes: dict[str, list[int]]) -> dict[str, int]:
    """Sequential nonoverlapping modes can share a lane; parallel modes cannot."""
    occupied: list[set[int]] = []
    result = {}
    for name in sorted(nodes, key=lambda n: (min(nodes[n]), max(nodes[n]), n)):
        ages = set(nodes[name])
        lane = next((i for i, admitted in enumerate(occupied) if not ages & admitted), len(occupied))
        if lane == len(occupied):
            occupied.append(set())
        occupied[lane].update(ages)
        result[name] = lane
    return result


def _semantic_shape(value: Any) -> Any:
    """A label/path-independent structural key for repeated graph stages."""
    if isinstance(value, list):
        return [_semantic_shape(v) for v in value]
    if not isinstance(value, dict):
        return value
    if value.get("kind") == "callable":
        try:
            tree = ast.parse(value.get("source", ""))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    node.name = "callback"
                    node.decorator_list = []
                    node.body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
            source = ast.dump(tree, include_attributes=False)
        except SyntaxError:
            source = value.get("source", "")
        return {"arguments": value.get("arguments"), "body": source,
                "bindings": _semantic_shape(value.get("bindings", {})),
                "partial_keywords": _semantic_shape(value.get("partial_keywords", {}))}
    omitted = {"source_file", "source_sha256", "source_line", "module", "qualname", "description"}
    return {k: _semantic_shape(v) for k, v in value.items() if k not in omitted}


def _stage_channels(snapshot: dict, regimes: dict, nodes: dict) -> tuple[dict[str, str], int]:
    graph = fields(snapshot.get("graph"))
    if graph.get("coordinate_kind", "age") != "period":
        return {name: "main" for name in regimes}, 1
    phase = fields(fields(snapshot.get("reachability")).get("solution"))
    outgoing = phase.get("candidate_targets_by_source", {})
    if not outgoing:
        outgoing = {name: [target for row in phase.get("targets_by_period", []) for target in row.get(name, [])] for name in regimes}
    terminal = {name for name in regimes if not outgoing.get(name)}
    recurrences = {name: math.gcd(*(b - a for a, b in zip(admitted, admitted[1:])))
                   for name, admitted in nodes.items() if name not in terminal and len(admitted) >= 3}
    nonunit = [step for step in recurrences.values() if step > 1]
    cadence = math.gcd(*nonunit) if nonunit else 1
    if cadence <= 1:
        return {name: "main" for name in regimes}, 1
    signatures = defaultdict(set)
    patterns = {}
    for name, declaration in regimes.items():
        step = recurrences.get(name, cadence)
        residues = {tick % step for tick in nodes[name]}
        pattern = ("continuous" if step == 1 or len(residues) > 1 else f"stage_{step}_{next(iter(residues))}")
        patterns[name] = pattern
        if name not in terminal:
            payload = {key: declaration.get(key) for key in ("states", "actions", "functions", "constraints", "koopmans_aggregator", "certainty_equivalent", "stakeholders", "__routing_owner")}
            signatures[pattern].add(digest(_semantic_shape(payload)))
    keys = {pattern: digest(sorted(items))[:12] for pattern, items in signatures.items()}
    channels = {}
    for name in regimes:
        channels[name] = ("absorbing" if name in terminal else
                          "stage_" + keys[patterns[name]])
    return channels, cadence


def _value_route_obligations(snapshot: dict, regimes: dict) -> tuple[dict[str, str], set[str]]:
    owners = defaultdict(set)
    protected = set()
    def references(value: Any) -> set[str]:
        return {fields(item)["regime"] for item in _walk(value) if isinstance(item, dict)
                and isinstance(item.get("type"), str)
                and item["type"].split(".")[-1] == "ProjectedRegimeValue"
                and isinstance(fields(item).get("regime"), str)}
    for source, declaration in regimes.items():
        targets = references(declaration)
        if targets:
            protected.update({source, *targets})
    for source, law in fields(snapshot.get("graph")).get("laws", {}).items():
        for target, gate in fields(law).get("gated_edges", {}).items():
            protected.update({source, target})
            protected.update(references(gate))
            for role, route in fields(gate).get("legs", {}).items():
                for fallback in references(fields(route).get("fallback")):
                    owners[fallback].add(role)
    return {name: json.dumps(sorted(roles)) for name, roles in owners.items()}, protected


def adapt_snapshot(snapshot: dict[str, Any], *, name: str = "Constructed pylcm model") -> dict[str, Any]:
    if snapshot.get("schema") != "constructed-pylcm-model-graph-1":
        raise ModelAdapterError("Expected a constructed pylcm graph snapshot")
    graph = fields(snapshot.get("graph"))
    reach = fields(snapshot.get("reachability"))
    raw_nodes = graph.get("reachability", snapshot.get("reachability"))
    reach = fields(raw_nodes) or reach
    pairs = reach.get("nodes", [])
    if not pairs:
        raise ModelAdapterError("Constructed model exposes no valued graph nodes")
    rational = [(Fraction(str(plain(age))), regime) for age, regime in pairs]
    divisor = math.lcm(*(a.denominator for a, _ in rational))
    nodes: dict[str, list[int]] = defaultdict(list)
    for age, regime in rational:
        nodes[regime].append(int(age * divisor))
    nodes = {name: sorted(set(ages)) for name, ages in nodes.items()}
    def query_only_reference(value):
        if isinstance(value, dict):
            tag = value.get('type')
            if isinstance(tag, str) and tag.split('.')[-1] in {'ProjectedRegimeValue', 'ArrivalRead', 'ExitTarget', 'Counterparty'}:
                return fields(value).get('regime') not in nodes
            return any(query_only_reference(child) for child in value.values())
        return isinstance(value, list) and any(query_only_reference(child) for child in value)
    if any(query_only_reference(fields(raw).get(key, {}))
           for raw in snapshot['user_regimes'].values()
           for key in ('same_period_refs', 'value_constraints', 'exits', 'constraints')):
        raise ModelAdapterError('Legacy scalar adapter cannot represent policy-query owners; use find_regimes(native_snapshot) production normalization')
    ages = sorted({a for aa in nodes.values() for a in aa})
    lanes = _lanes(nodes)
    count_lanes = max(lanes.values()) + 1
    phase_graphs = {phase: fields(reach.get(phase)) for phase in ("solution", "simulation")}
    regimes = {n: fields(r) for n, r in snapshot["user_regimes"].items() if n in nodes}
    routing_owners, protected_value_routes = _value_route_obligations(snapshot, regimes)
    regimes = {n: {**declaration, "__routing_owner": routing_owners.get(n)} for n, declaration in regimes.items()}
    channels, cadence = _stage_channels(snapshot, regimes, nodes)
    domains = {"states": {}, "actions": {}}
    aliases: dict[str, dict[str, str]] = {}
    all_names = {kind: {coord for r in regimes.values() for coord in r.get(kind, {})} for kind in domains}
    identities = defaultdict(set)
    prepared = {}
    for regime, declaration in regimes.items():
        prepared[regime] = {}
        for kind in domains:
            prepared[regime][kind] = {coord: _grid(grid) for coord, grid in declaration.get(kind, {}).items()}
            for coord, domain in prepared[regime][kind].items():
                identities[(kind, coord)].add(domain["grid_id"])
    display_names = {}
    for regime, declaration in prepared.items():
        aliases[regime] = {}
        for kind in domains:
            for coord, domain in declaration[kind].items():
                unique = len(identities[(kind, coord)]) == 1 and coord not in all_names["actions" if kind == "states" else "states"]
                alias = coord if unique else coord + "__" + kind[0] + domain["grid_id"][:10]
                aliases[regime][coord] = alias
                display_names[alias] = coord
                domains[kind][alias] = domain
    programs = []
    external = set()
    notes = []
    laws = graph.get("laws", {})
    all_time_values = plain(fields(snapshot.get("ages")).get("exact_values"))
    if all_time_values is None:
        all_time_values = list(range(snapshot["n_periods"]))
    all_ticks = [int(Fraction(str(plain(v))) * divisor) for v in all_time_values]
    period_expression = "0"
    if len(all_ticks) > 1:
        delta = all_ticks[1] - all_ticks[0]
        if delta > 0 and all(b - a == delta for a, b in zip(all_ticks, all_ticks[1:])):
            period_expression = f"((age - {all_ticks[0]}) / {delta})"
        else:
            period_expression = repr(len(all_ticks) - 1)
            for index in reversed(range(len(all_ticks) - 1)):
                period_expression = f"({index} if age == {all_ticks[index]} else {period_expression})"
    for regime, declaration in regimes.items():
        state_names = set(declaration.get("states", {}))
        action_names = set(declaration.get("actions", {}))
        coordinates = state_names | action_names
        coordinates |= {n + "_code" for n in state_names | action_names}
        local_notes = []
        bindings = {key: _callback(value, coordinates, local_notes, snapshot.get("callable_records")) for key, value in declaration.get("functions", {}).items()}
        transitions = declaration.get("state_transitions", {})
        for state, value in transitions.items():
            bindings.setdefault("next_" + state, _callback(value, coordinates, local_notes, snapshot.get("callable_records")))
        analysis = _GraphAnalysis(None, bindings, coordinates, set(), divisor)
        analysis.period_expression = period_expression
        analysis.period_only = graph.get("coordinate_kind", "age") == "period"
        replacements = dict(aliases[regime])
        for original, alias in aliases[regime].items():
            replacements[original + "_code"] = alias
        scalar_replacements = dict(replacements)
        for kind in domains:
            for original, domain in prepared[regime][kind].items():
                vals, codes = domain.get("values"), domain.get("codes")
                if vals is not None and codes is not None and vals != codes:
                    alias = aliases[regime][original]
                    encoded = repr(codes[-1])
                    for val, code in reversed(list(zip(vals[:-1], codes[:-1]))):
                        encoded = f"({code} if {alias} == {val!r} else {encoded})"
                    scalar_replacements[original + "_code"] = encoded
        def expression(value: Any, *, predicate: bool = False) -> Any:
            entry = _callback(value, coordinates, local_notes, snapshot.get("callable_records"))
            expr = analysis.predicate(entry) if predicate else analysis.value(entry)
            return expr
        reward = expression(declaration.get("functions", {}).get("utility"))
        discount_source = declaration.get("functions", {}).get("discount_factor")
        discount = {"opaque": "unchanged-original-discount", "reads": [],
                    "arguments": [expression(discount_source)] if discount_source is not None else []}
        feasibility = [expression(c, predicate=True) for c in declaration.get("constraints", {}).values()]
        outputs = {key: expression(declaration["functions"][key]) for key in declaration.get("output_only_functions", [])
                   if key in declaration.get("functions", {})}
        for operator in ("koopmans_aggregator", "certainty_equivalent"):
            value = declaration.get(operator)
            if value is not None and value.get("type", "").split(".")[-1] not in {"LinearAggregator", "LinearExpectation"}:
                outputs[operator + "_dependencies"] = expression(value)
        if snapshot.get("shocks"):
            outputs["shock_distribution_dependencies"] = expression(snapshot["shocks"])
        special = {key: declaration.get(key) for key in ("stakeholders", "same_period_refs", "value_constraints", "joint_transitions", "exits")
                   if declaration.get(key)}
        if special or regime in protected_value_routes:
            # Until each value/read/joint route is lowered, retaining all local
            # state axes is a safe advisory result, not semantic certification.
            outputs["unlowered_structural_consumers"] = {"opaque": digest({"special": special, "value_route": regime in protected_value_routes}), "reads": sorted(coordinates)}
            local_notes.append("Value/joint/exit consumers retained conservatively; semantic bridge unresolved")
        rows = []
        for phase, graph_phase in phase_graphs.items():
            targets_by_period = graph_phase.get("targets_by_period", [])
            if len(targets_by_period) not in {snapshot["n_periods"], snapshot["n_periods"] - 1}:
                raise ModelAdapterError("Effective phase graph has no full period-indexed target list")
            time_values = plain(fields(snapshot.get("ages")).get("exact_values"))
            if time_values is None:
                time_values = list(range(snapshot["n_periods"]))
            if len(time_values) != snapshot["n_periods"]:
                raise ModelAdapterError("Time coordinate length does not match the graph")
            target_ages: dict[str, list[int]] = defaultdict(list)
            for period, targets in enumerate(targets_by_period):
                for target in targets.get(regime, []):
                    target_ages[target].append(int(Fraction(str(plain(time_values[period]))) * divisor))
            law = fields(laws.get(regime)).get("transition")
            probability = {"opaque": "original-routing-law:" + digest(law), "reads": [],
                           "arguments": [expression(law)]}
            for target, source_ages in target_ages.items():
                if target not in regimes:
                    raise ModelAdapterError("Effective target lacks its finalized regime: " + target)
                updates = {}
                for state in regimes[target].get("states", {}):
                    law_entry = transitions.get(state)
                    if isinstance(law_entry, dict) and "kind" not in law_entry:
                        law_entry = law_entry.get(target)
                    target_alias = aliases[target][state]
                    if law_entry is not None:
                        updates[target_alias] = {"opaque": "original-state-law:" + digest(law_entry), "reads": [],
                                                 "arguments": [expression(law_entry)]}
                    else:
                        source_grid = declaration.get("states", {}).get(state)
                        kind = (source_grid or {}).get("type", "").split(".")[-1]
                        if kind.endswith("Process"):
                            has_callbacks = any(isinstance(v, dict) and v.get("kind") == "callable" for v in _walk(source_grid))
                            reads = (sorted(coordinates) if has_callbacks or fields(source_grid).get("state_conditioned") else
                                     ([] if "IID" in kind else [state]))
                            updates[target_alias] = {"opaque": "intrinsic-process:" + digest(source_grid), "reads": reads}
                        else:
                            updates[target_alias] = {"opaque": "unlowered-handoff:" + target_alias,
                                                     "reads": sorted(coordinates)}
                            local_notes.append("Handoff without a directly exposed state law: " + regime + "->" + target + ":" + state)
                row = {"target": target, "phase": "solve" if phase == "solution" else "simulation",
                       "when": "age in " + repr(sorted(set(source_ages))),
                       "probability": probability, "updates": updates}
                if count_lanes > 1:
                    row["status_updates"] = {"graph_lane": str(lanes[target])}
                rows.append(row)
        for parameter in analysis.external_parameters:
            alias = "external__" + parameter
            replacements[parameter] = alias
            scalar_replacements[parameter] = alias
            external.add(alias)
        role = {"actor": "individual" if not declaration.get("stakeholders") else "collective:" + repr(declaration["stakeholders"]),
                "observations": ["original graph information"], "commitments": ["original action timing"],
                "discount_boundary": "one original graph period"}
        program = {"id": regime, "when": "age in " + repr(nodes[regime]),
                   "channel": channels[regime],
                   "declared_states": [aliases[regime][n] for n in sorted(state_names)],
                   "declared_actions": [aliases[regime][n] for n in sorted(action_names)],
                   "role": role, "reward": reward, "discount": discount,
                   "routing_owner": routing_owners.get(regime),
                   "feasible": feasibility or True, "outputs": outputs,
                   "terminal": not rows, "terminal_payoff": reward if not rows else "0",
                   "transitions": rows, "allocation": "per_block"}
        if count_lanes > 1:
            program["when"] += " and graph_lane == " + str(lanes[regime])
        if declaration.get("taste_shocks"):
            program["taste_groups"] = {"kind": "unchanged-original", "source": declaration["taste_shocks"]}
        for key in ("reward", "discount", "feasible", "outputs", "terminal_payoff"):
            program[key] = _rewrite(program[key], replacements, scalar_replacements)
        for row in rows:
            row["probability"] = _rewrite(row["probability"], replacements, scalar_replacements)
            row["updates"] = {key: _rewrite(value, replacements, scalar_replacements) for key, value in row["updates"].items()}
        programs.append(program)
        if local_notes:
            notes.append({"regime": regime, "notes": sorted(set(local_notes))})
    starts = graph.get("initial_nodes") or pairs
    result = {"schema_version": 1, "name": name, "ages": ages, "age_scale": "1/" + str(divisor),
              "age_labels": {str(age): str(Fraction(age, divisor)) for age in ages},
              "states": domains["states"], "actions": domains["actions"], "parameters": {},
              "external_parameters": sorted(external), "programs": programs,
              "starts": [{"age": int(Fraction(str(plain(a))) * divisor), "program": regime,
                          **({"status": {"graph_lane": lanes[regime]}} if count_lanes > 1 else {})}
                         for a, regime in starts if regime in regimes],
              "limits": {"status_lanes": max(128, count_lanes)},
              "implementation": {"compact_search": True, "allow_sparse_admission": True},
              "source_adapter": {"kind": "constructed_pylcm_graph", "status": "SUPPORTED_GRAPH",
                 "snapshot_sha256": digest(snapshot), "pylcm_version": snapshot.get("pylcm_version"),
                 "coordinate_names": display_names, "lanes": lanes,
                 "time_coordinate_kind": graph.get("coordinate_kind", "age"),
                 "recurring_graph_cadence": cadence, "stage_channels": channels,
                 "value_route_protected_regimes": sorted(protected_value_routes), "fallback_ownership": routing_owners,
                 "callback_notes": notes, "native_model_construction": snapshot.get("construction_capture_phase", "complete Model constructor"), "native_solve_simulate": "NOT_RUN",
                 "search_scope": "All intervals of admitted nodes within fixed graph lanes and recurring stage channels. Sparse date masks stay explicit; overlapping-mode quotients are outside this family.",
                 "semantic_scope": "Original numerical functions/operators remain opaque and unchanged. The optimum certificate is conditional on conservative dependency extraction; no numerical source/backend transport proof is claimed.",
                 "cost_scope": "Declared captured-grid row and boundary work; no production-grid or wall-time optimum is claimed."}}
    if count_lanes > 1:
        result["status_axes"] = {"graph_lane": list(range(count_lanes))}
    return result


def adapt_model(model: Any, *, name: str = "Constructed pylcm model") -> dict[str, Any]:
    """One public interface for any builder: pass its constructed Model."""
    from .model_snapshot import read_model

    return adapt_snapshot(read_model(model), name=name)
