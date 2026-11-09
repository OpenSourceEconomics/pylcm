"""Static source/API routing analysis without importing the submitted packages.

This is a bounded declaration interpreter, not a Python or numerical runtime.
It reads a selected original routing factory and the actual pinned API guard.
It discovers pairs of equal literal terminal payloads, then interprets the guard
on each proposed fallback-name coalescing. The report covers that factory only.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re

from .source_adapter import (Declaration, Function, Module, Scope,
                             SourceAdapterError, _body_without_docstring, _name)


@dataclass(frozen=True)
class Constructed:
    kind: str
    fields: dict
    arguments: list
    anchor: dict


def _materialize(value):
    """Freeze loop-dependent declaration fields at their declaration site."""
    if isinstance(value, Declaration):
        fields = {}
        for kw in value.node.keywords:
            if kw.arg is None:
                fields.update(_materialize(value.scope.evaluate(kw.value)))
            else:
                fields[kw.arg] = _materialize(value.scope.evaluate(kw.value))
        return Constructed(value.kind, fields,
                           [_materialize(value.scope.evaluate(x)) for x in value.node.args],
                           value.scope.module.anchor(value.node))
    if isinstance(value, dict):
        return {k: _materialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize(x) for x in value]
    if isinstance(value, tuple):
        return tuple(_materialize(x) for x in value)
    return value


class _Returned(Exception):
    def __init__(self, value): self.value = value


class _ApiRejected(Exception):
    def __init__(self, exception, message, anchor):
        self.exception, self.message, self.anchor = exception, message, anchor


class DeclarationInterpreter:
    """Finite assignments/loops/branches with an explicit statement whitelist."""
    def __init__(self, scope, statement_budget=10000):
        self.scope = scope
        self.remaining = statement_budget

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.scope.values[target.id] = value
        elif isinstance(target, ast.Subscript):
            container = self.scope.evaluate(target.value)
            if not isinstance(container, dict):
                raise SourceAdapterError("Only local declaration-dictionary writes are supported")
            container[self.scope.evaluate(target.slice)] = value
        elif isinstance(target, (ast.List, ast.Tuple)):
            if len(target.elts) != len(value):
                raise SourceAdapterError("Declaration assignment arity mismatch")
            for sub, item in zip(target.elts, value): self.assign(sub, item)
        else:
            raise SourceAdapterError("Unsupported declaration assignment target")

    def statements(self, body):
        for stmt in body:
            self.remaining -= 1
            if self.remaining < 0:
                raise SourceAdapterError("Structural declaration statement budget exhausted")
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if stmt.value is None: continue
                value = _materialize(self.scope.evaluate(stmt.value))
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets: self.assign(target, value)
            elif isinstance(stmt, ast.For):
                values = self.scope.evaluate(stmt.iter)
                if not isinstance(values, (list, tuple, range)) or len(values) > self.remaining:
                    raise SourceAdapterError("Only bounded structural loops are supported")
                for value in values:
                    self.assign(stmt.target, value)
                    self.statements(stmt.body)
                self.statements(stmt.orelse)
            elif isinstance(stmt, ast.If):
                self.statements(stmt.body if self.scope.evaluate(stmt.test) else stmt.orelse)
            elif isinstance(stmt, ast.Return):
                raise _Returned(_materialize(self.scope.evaluate(stmt.value)))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                    and isinstance(stmt.value.value, str):
                continue
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                    and isinstance(stmt.value.func, ast.Attribute):
                call = stmt.value
                container = self.scope.evaluate(call.func.value)
                arguments = [self.scope.evaluate(a) for a in call.args]
                if call.keywords or len(arguments) != 1:
                    raise SourceAdapterError("Unsupported structural collection mutation")
                if call.func.attr == "append" and isinstance(container, list):
                    container.append(arguments[0])
                elif call.func.attr == "add" and isinstance(container, set):
                    container.add(arguments[0])
                else:
                    raise SourceAdapterError("Only local list.append/set.add mutations are supported")
            elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                name = _name(stmt.exc.func)
                if name != "ModelInitializationError" or len(stmt.exc.args) != 1 or stmt.exc.keywords:
                    raise SourceAdapterError("Unsupported source API exception")
                raise _ApiRejected(name, str(self.scope.evaluate(stmt.exc.args[0])),
                                   self.scope.module.anchor(stmt))
            elif isinstance(stmt, ast.Pass):
                continue
            else:
                raise SourceAdapterError(f"Unsupported declaration statement: {type(stmt).__name__}")

    def run(self, body):
        try:
            self.statements(body)
        except _Returned as result:
            return result.value
        return None


def _read_api_function(path, name):
    """Parse the selected complete guard without requiring its Python 3.14 module.

    The supplied pinned module contains newer syntax elsewhere. Extraction is by
    the requested top-level function name, with full-file and body hashes and
    original line numbers. No alternate guard or copied rule is substituted.
    """
    path = Path(path).resolve(); text = path.read_text(); lines = text.splitlines(True)
    matches = [i for i, line in enumerate(lines) if re.match(rf"^def {re.escape(name)}\s*\(", line)]
    if len(matches) != 1: raise SourceAdapterError("A unique top-level pinned API guard is required")
    start = matches[0]
    stop = next((i for i in range(start+1, len(lines))
                 if re.match(r"^(?:def |async def |class |@)", lines[i])), len(lines))
    tree = ast.parse("".join(lines[start:stop]))
    ast.increment_lineno(tree, start)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None: raise SourceAdapterError("Selected API guard did not parse")
    module = Module.__new__(Module)
    module.path, module.text, module.tree = path, text, tree
    module.digest = hashlib.sha256(text.encode()).hexdigest()
    module.symbols, module.classes, module.imports = {}, {}, {}
    module.cache = {str(path): module}
    module.functions = {name: Function(module, fn)}
    module.scope = Scope(module)
    return module, fn


def _literal_return(fn):
    if not isinstance(fn, Function): return None
    if fn.node.args.args or fn.node.args.posonlyargs or fn.node.args.kwonlyargs: return None
    body = _body_without_docstring(fn.node)
    if len(body) != 1 or not isinstance(body[0], ast.Return): return None
    value = body[0].value
    if isinstance(value, ast.Call) and _name(value.func) in {"jnp.asarray", "np.asarray"}:
        if len(value.args) == 1 and not value.keywords: value = value.args[0]
        elif not value.args and len(value.keywords) == 1 and value.keywords[0].arg == "a":
            value = value.keywords[0].value
        else: return None
    if isinstance(value, ast.Constant) and isinstance(value.value, (bool, int, float)):
        return {"literal": value.value, "source_anchor": fn.module.anchor(fn.node)}
    return None


def _terminal_payload(regime):
    if not isinstance(regime, Constructed) or regime.kind != "Regime": return None
    fields = regime.fields
    if any(fields.get(k) for k in ("states", "actions", "constraints", "state_transitions")): return None
    if set(fields.get("functions", {})) != {"utility"}: return None
    payoff = _literal_return(fields["functions"]["utility"])
    if payoff is None: return None
    return {"state_axes": [], "action_axes": [], "literal_payoff": payoff["literal"],
            "source_anchor": regime.anchor, "payoff_source_anchor": payoff["source_anchor"]}


def _run_guard(module, node, routes):
    scope = Scope(module, values={"prefix": "source-routing analysis: ", "edge": {"legs": routes}})
    try:
        DeclarationInterpreter(scope).run(_body_without_docstring(node))
    except _ApiRejected as rejected:
        return {"status": "REJECTED", "exception": rejected.exception,
                "message": rejected.message, "source_anchor": rejected.anchor}
    return {"status": "ACCEPTED"}


def analyze_routing_source(source_path, api_path, config=None):
    """Discover local equal-payload pairs and test actual API routing admission.

    Configuration selects original source entry points and ordinary factory
    inputs. It supplies no candidate pairs, ownership masks, or expected graph.
    """
    config = dict(config or {})
    forbidden = {"pairs", "merges", "routing_owners", "expected_graph", "cuts", "boundaries"} & set(config)
    if forbidden: raise SourceAdapterError(f"Answer-bearing routing inputs are not accepted: {sorted(forbidden)}")
    module = Module(source_path, config.get("symbols"))
    constructor = config.get("constructor", "build_model")
    parent = module.functions.get(constructor)
    if parent is None: raise SourceAdapterError("Routing constructor is absent")
    factory_name = config.get("factory")
    factories = [n for n in parent.node.body if isinstance(n, ast.FunctionDef) and n.name == factory_name]
    if len(factories) != 1: raise SourceAdapterError("A unique direct nested routing factory is required")
    factory = factories[0]
    parameters = dict(config.get("factory_parameters", {}))
    required = {x.arg for x in factory.args.args + factory.args.posonlyargs + factory.args.kwonlyargs}
    if set(parameters) != required: raise SourceAdapterError("Supply every original routing factory parameter exactly once")
    scope = Scope(module, parent.node.body, values=parameters)
    gate = DeclarationInterpreter(scope).run(_body_without_docstring(factory))
    if not isinstance(gate, Constructed) or gate.kind != "Gate":
        raise SourceAdapterError("Selected source factory does not return a Gate")
    original_routes = gate.fields.get("routes")
    if not isinstance(original_routes, dict): raise SourceAdapterError("Gate routes must be a finite source dictionary")
    source_regimes = scope.lookup("regimes")
    routes, records = {}, []
    for leg, route in original_routes.items():
        if not isinstance(route, Constructed) or route.kind != "StakeholderRoute":
            raise SourceAdapterError("Only explicit StakeholderRoute declarations are supported")
        fallback = route.fields.get("fallback")
        if not isinstance(fallback, Constructed) or fallback.kind != "ProjectedRegimeValue":
            raise SourceAdapterError("Phase-separated fallback declarations need a routing adapter extension")
        name = fallback.fields.get("regime")
        routes[leg] = {"solve_fallback": {"regime": name}, "simulate_fallback": {"regime": name}}
        payload = _terminal_payload(source_regimes.get(name))
        records.append({"leg": leg, "target_stakeholder": route.fields.get("target_stakeholder"),
                        "fallback_regime": name,
                        "projection_coordinates": sorted(fallback.fields.get("projection", {})),
                        "routing_owner": f"{factory_name}:{json.dumps(parameters,sort_keys=True)}:{leg}",
                        "local_terminal_payload": payload,
                        "source_anchor": route.anchor})
    api, guard = _read_api_function(api_path, config.get("api_guard", "_fail_if_duplicate_fallback_regimes"))
    baseline = _run_guard(api, guard, routes)
    if baseline["status"] != "ACCEPTED":
        raise SourceAdapterError("The original selected routing factory already fails its pinned API guard")
    candidates = []
    for left, right in itertools.combinations(records, 2):
        lp, rp = left["local_terminal_payload"], right["local_terminal_payload"]
        if lp is None or rp is None: continue
        signature = lambda p: (p["state_axes"],p["action_axes"],p["literal_payoff"])
        if signature(lp) != signature(rp): continue
        altered = copy.deepcopy(routes)
        for route in altered.values():
            for phase in ("solve_fallback", "simulate_fallback"):
                if route[phase]["regime"] == right["fallback_regime"]:
                    route[phase]["regime"] = left["fallback_regime"]
        admission = _run_guard(api, guard, altered)
        candidates.append({"fallback_regimes": [left["fallback_regime"], right["fallback_regime"]],
                           "legs": [left["leg"], right["leg"]],
                           "local_payload_check": "EQUAL_LITERAL_TERMINAL_PAYLOAD",
                           "implementation_admission": admission,
                           "decision": "RETAIN" if admission["status"] == "REJECTED" else "UNRESOLVED",
                           "reason": "Pinned per-leg fallback admission rejects coalescing." if admission["status"] == "REJECTED"
                                     else "Local equality and this guard alone do not prove a whole-model merge."})
    return {"schema_version": 1, "analysis": "source_factory_routing_admission",
            "status": "COMPLETE_FOR_SELECTED_FACTORY",
            "source_adapter": {"kind": "pylcm_routing_ast", "status": "SUPPORTED_STATIC",
                               "execution_status": "SOURCE_DECLARATIONS_AND_PINNED_API_GUARD_INTERPRETED",
                               "native_model_execution": "NOT_RUN", "no_oracle_inputs": True},
            "source": {"path": str(module.path), "sha256": module.digest,
                       "factory": factory_name, "factory_parameters": parameters,
                       "constructor_inputs": config.get("symbols", {}),
                       "source_anchor": module.anchor(factory)},
            "pinned_api": {"path": str(api.path), "sha256": api.digest,
                           "guard": guard.name, "source_anchor": api.anchor(guard),
                           "guard_ast_sha256": hashlib.sha256(ast.dump(guard,include_attributes=False).encode()).hexdigest()},
            "predicate_literal": _literal_return(gate.fields.get("predicate")),
            "original_api_admission": baseline, "routes": records, "candidate_checks": candidates,
            "counts": {"routes": len(records), "literal_terminal_routes": sum(r["local_terminal_payload"] is not None for r in records),
                       "equal_payload_pairs": len(candidates),
                       "retained_pairs": sum(c["decision"] == "RETAIN" for c in candidates)},
            "scope": "The selected original factory, its literal terminal payloads, and the actual selected API guard. No whole-model Foerster equivalence, global optimum, role coarsening, or simulation execution is claimed."}


def analyze_routing_manifest(manifest_path):
    path = Path(manifest_path).resolve(); config = json.loads(path.read_text())
    if config.pop("adapter", None) != "pylcm_routing_ast": raise SourceAdapterError("Unknown routing adapter")
    source = path.parent / config.pop("source")
    api = path.parent / config.pop("api_source")
    return analyze_routing_source(source, api, config)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(analyze_routing_manifest(args.manifest), indent=2)+"\n")


if __name__ == "__main__": main()
