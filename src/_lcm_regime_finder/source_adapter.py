"""Conservative, executable AST adapter for a declared pylcm source subset.

This module never imports or executes the submitted model, JAX, or pylcm. It reads
constructor declarations and callback bodies, emits exact Boolean predicates when
supported, and keeps all dependencies of other pure callbacks as opaque nodes.
Only original model source and ordinary parameter/grid inputs are accepted.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
import hashlib
import itertools
import json
import math
import operator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class SourceAdapterError(ValueError):
    """The requested source construct is outside this adapter's declared subset."""


@dataclass(frozen=True)
class Symbol:
    path: str


@dataclass(frozen=True)
class Category:
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class Function[ModuleType]:
    module: ModuleType
    node: ast.FunctionDef


@dataclass(frozen=True)
class Declaration:
    kind: str
    node: ast.Call
    scope: "Scope"

    def keyword(self, name, default=None):
        for item in self.node.keywords:
            if item.arg == name:
                return self.scope.evaluate(item.value)
        return default


_CONSTRUCTORS = {
    "Regime", "Model", "AgeGrid", "AgeRange", "PeriodRange", "Periods",
    "DiscreteGrid", "LinSpacedGrid", "IrregSpacedGrid", "NormalIIDProcess",
    "TauchenAR1Process", "Phased", "Transition", "StochasticTransition",
    "ExtremeValueTasteShocks", "InitialNodes", "ByAge", "ByPeriod",
    "LinearAggregator", "fixed_transition",
    "ProjectedRegimeValue", "StakeholderRoute", "Gate",
}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow,
           ast.BitOr: operator.or_, ast.BitAnd: operator.and_}
_COMPARISON: Mapping[type[ast.cmpop], Callable[..., bool]] = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
               ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
               ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}  # codespell:ignore

_SUPPORTED_KEYWORDS = {
    "Model": {"ages", "regimes", "edges", "initial_nodes", "fixed_params", "regime_id_class", "enable_jit", "description"},
    "Regime": {"states", "state_transitions", "actions", "functions", "constraints", "taste_shocks", "description"},
    "AgeGrid": {"start", "inclusive_stop", "step"},
    "AgeRange": {"start", "exclusive_stop"},
    "Periods": {"values"},
    "Transition": {"targets", "law"},
    "DiscreteGrid": {"category_class", "values"},
    "LinSpacedGrid": {"start", "stop", "n_points"},
    "IrregSpacedGrid": {"points", "n_points"},
    "NormalIIDProcess": {"mu", "sigma", "n_points", "gauss_hermite"},
    "TauchenAR1Process": {"n_points", "mean", "rho", "sigma", "width"},
    "Phased": {"solve", "simulate"},
    "StochasticTransition": {"func"},
    "ExtremeValueTasteShocks": set(),
    "fixed_transition": set(),
}


def _validate_declaration(decl):
    if not isinstance(decl, Declaration):
        raise SourceAdapterError("A supported source declaration is required")
    allowed = _SUPPORTED_KEYWORDS.get(decl.kind)
    if allowed is None:
        raise SourceAdapterError(f"Unsupported source declaration {decl.kind}")
    actual = {kw.arg for kw in decl.node.keywords}
    if None in actual or actual-allowed:
        raise SourceAdapterError(f"Unsupported {decl.kind} keywords/consumer roots: {sorted(str(x) for x in actual-allowed)}")
    positional = 1 if decl.kind in {"DiscreteGrid", "fixed_transition"} else 0
    if len(decl.node.args) > positional:
        raise SourceAdapterError(f"Unsupported positional {decl.kind} declaration")
    if decl.kind == "fixed_transition" and len(decl.node.args) != 1:
        raise SourceAdapterError("fixed_transition requires one original coordinate name")


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _arguments(node):
    return [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]


def _body_without_docstring(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body.pop(0)
    return body


def _validate_constructor_body(fn):
    """Reject declaration mutations that a lazy binding environment cannot model.

    Source parameter validators are recorded assumptions, not executed here. A
    validation-only branch may bind a diagnostic message and then raise; it may
    not conditionally change a grid, consumer, or regime on a successful path.
    """
    assigned = set(_arguments(fn))
    def validation_branch(body):
        return bool(body) and isinstance(body[-1], ast.Raise) and all(
            isinstance(s, (ast.Assign, ast.AnnAssign)) and
            all(isinstance(t, ast.Name) for t in (s.targets if isinstance(s, ast.Assign) else [s.target]))
            for s in body[:-1])
    for stmt in _body_without_docstring(fn):
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if any(not isinstance(t, ast.Name) for t in targets):
                raise SourceAdapterError("Constructor mutation requires a sequential declaration adapter")
            names = [t.id for t in targets]
            if any(name in assigned for name in names):
                raise SourceAdapterError("Constructor rebinding requires a sequential declaration adapter")
            assigned.update(names)
        elif isinstance(stmt, (ast.Return, ast.Assert)):
            continue
        elif isinstance(stmt, ast.If) and validation_branch(stmt.body) \
                and (not stmt.orelse or validation_branch(stmt.orelse)):
            continue
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "validate" \
                and not stmt.value.args and not stmt.value.keywords:
            continue
        else:
            raise SourceAdapterError(f"Unsupported constructor control flow/consumer mutation: {type(stmt).__name__}")


class Module:
    def __init__(self, path, symbols=None, cache=None, reads=None):
        self.path = Path(path).resolve()
        self.reads = reads
        self.text = reads.read_text(path) if reads is not None else self.path.read_text()
        self.tree = ast.parse(self.text, filename=str(self.path))
        self.digest = hashlib.sha256(self.text.encode()).hexdigest()
        self.symbols = dict(symbols or {})
        self.cache = cache if cache is not None else {}
        self.cache[str(self.path)] = self
        self.functions = {n.name: Function(self, n) for n in self.tree.body
                          if isinstance(n, ast.FunctionDef)}
        self.classes = {n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)}
        self.imports = {}
        for n in self.tree.body:
            if isinstance(n, ast.ImportFrom) and n.module:
                for alias in n.names:
                    self.imports[alias.asname or alias.name] = (n.module, alias.name)
        self.scope = Scope(self, self.tree.body)

    def anchor(self, node):
        return {"path": str(self.path), "sha256": self.digest,
                "line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno)}

    def import_path(self, name):
        module, _ = self.imports[name]
        return self.path.parent / (module.rsplit(".", 1)[-1] + ".py")

    def imported(self, name):
        module, original = self.imports[name]
        # Only sibling source modules are followed, never installed packages.
        path = self.import_path(name)
        if path.is_file():
            child = self.cache.get(str(path.resolve())) or Module(path, self.symbols, self.cache, reads=self.reads)
            return child.scope.lookup(original)
        raise SourceAdapterError(f"Unresolved imported constant {name}: {module}")


class Scope:
    def __init__(self, module, body=(), values=None):
        self.module = module
        self.values = dict(values or {})
        self.bindings = {}
        self.resolving = set()
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.bindings[target.id] = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
                self.bindings[stmt.target.id] = stmt.value

    def lookup(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.module.symbols:
            return self.module.symbols[name]
        if name in self.bindings:
            if name in self.resolving:
                raise SourceAdapterError(f"Cyclic source binding {name}")
            self.resolving.add(name)
            try:
                value = self.evaluate(self.bindings[name])
                self.values[name] = value
                return value
            finally:
                self.resolving.remove(name)
        if name in self.module.functions:
            return self.module.functions[name]
        if name in self.module.classes:
            node = self.module.classes[name]
            if any((_name(d.func) if isinstance(d, ast.Call) else _name(d)) == "categorical"
                   for d in node.decorator_list):
                members = tuple(n.target.id for n in node.body
                                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name))
                return Category(name, members)
        if self is not self.module.scope and name in self.module.scope.bindings:
            return self.module.scope.lookup(name)
        if name in self.module.imports:
            try:
                return self.module.imported(name)
            except SourceAdapterError:
                pass
        if any(k.startswith(name + ".") for k in self.module.symbols) or name in {
                "ScalarInt", "ScalarFloat", "float", "int"}:
            return Symbol(name)
        raise SourceAdapterError(f"A constructor input is required: {name}")

    def evaluate(self, node):
        dotted = _name(node)
        if dotted in self.module.symbols:
            return self.module.symbols[dotted]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.lookup(node.id)
        if isinstance(node, ast.Attribute):
            value = self.evaluate(node.value)
            if isinstance(value, Category):
                return value.members.index(node.attr)
            if isinstance(value, Symbol):
                path = value.path + "." + node.attr
                return self.module.symbols.get(path, Symbol(path))
            if isinstance(value, dict) and node.attr in value:
                return value[node.attr]
            raise SourceAdapterError(f"Unsupported constructor attribute {ast.unparse(node)}")
        if isinstance(node, (ast.List, ast.Tuple)):
            values = []
            for child in node.elts:
                if isinstance(child, ast.Starred):
                    values.extend(self.evaluate(child.value))
                else:
                    values.append(self.evaluate(child))
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    result.update(self.evaluate(value))
                else:
                    result[self.evaluate(key)] = self.evaluate(value)
            return result
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](self.evaluate(node.left), self.evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand)
            if isinstance(node.op, ast.USub): return -value
            if isinstance(node.op, ast.UAdd): return +value
            if isinstance(node.op, ast.Not): return not value
            if isinstance(node.op, ast.Invert): return not value if isinstance(value, bool) else ~value
        if isinstance(node, ast.Subscript):
            return self.evaluate(node.value)[self.evaluate(node.slice)]
        if isinstance(node, ast.Slice):
            return slice(*(self.evaluate(x) if x else None for x in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.IfExp):
            return self.evaluate(node.body if self.evaluate(node.test) else node.orelse)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self.evaluate(x) for x in node.values)
            if isinstance(node.op, ast.Or):
                return any(self.evaluate(x) for x in node.values)
        if isinstance(node, ast.Compare):
            vals = [self.evaluate(node.left)] + [self.evaluate(x) for x in node.comparators]
            return all(_COMPARISON[type(op)](a, b) for op, a, b in zip(node.ops, vals, vals[1:]))
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp)):
            rows = [dict(self.values)]
            for generator in node.generators:
                next_rows = []
                for values in rows:
                    scope = Scope(self.module, values=values)
                    scope.bindings = self.bindings
                    for item in scope.evaluate(generator.iter):
                        bound = dict(values)
                        self._bind(generator.target, item, bound)
                        inner = Scope(self.module, values=bound)
                        inner.bindings = self.bindings
                        if all(inner.evaluate(x) for x in generator.ifs):
                            next_rows.append(bound)
                rows = next_rows
            if isinstance(node, ast.DictComp):
                result = {}
                for values in rows:
                    inner = Scope(self.module, values=values); inner.bindings = self.bindings
                    result[inner.evaluate(node.key)] = inner.evaluate(node.value)
                return result
            result = []
            for values in rows:
                inner = Scope(self.module, values=values); inner.bindings = self.bindings
                result.append(inner.evaluate(node.elt))
            return result
        if isinstance(node, ast.JoinedStr):
            return "".join(str(x.value) if isinstance(x, ast.Constant) else
                           format(self.evaluate(x.value), self.evaluate(x.format_spec) if x.format_spec else "")
                           for x in node.values)
        if isinstance(node, ast.Call):
            name = _name(node.func)
            short = name.rsplit(".", 1)[-1] if name else None
            if short in _CONSTRUCTORS:
                return Declaration(short, node, self)
            if isinstance(node.func, ast.Call) and _name(node.func.func) == "categorical":
                return self.evaluate(node.args[0])
            if name == "make_dataclass":
                fields = self.evaluate(node.args[1])
                return Category(self.evaluate(node.args[0]), tuple(x[0] for x in fields))
            if name == "characteristics":
                kw = {x.arg: self.evaluate(x.value) for x in node.keywords}
                cases = kw["by_category"]; cat = kw["category_class"]
                keys = {k for row in cases.values() for k in row}
                return {k: [cases[c][k] for c in cat.members] for k in keys}
            args = [self.evaluate(x) for x in node.args]
            kw = {x.arg: self.evaluate(x.value) for x in node.keywords}
            if name in {"len", "tuple", "list", "set", "sorted", "int", "float", "min", "max", "abs", "range", "Fraction"}:
                return {"len": len, "tuple": tuple, "list": list, "set": set, "sorted": sorted,
                        "int": int, "float": float, "min": min, "max": max, "abs": abs,
                        "range": range, "Fraction": Fraction}[name](*args, **kw)
            if name in {"np.asarray", "jnp.asarray", "np.array", "jnp.array"}:
                return args[0]
            if name in {"np.arange", "jnp.arange"}:
                start, stop, step = (0, args[0], 1) if len(args) == 1 else \
                    (args[0], args[1], args[2] if len(args) > 2 else 1)
                return [start + k * step for k in range(math.ceil((stop-start)/step))]
            if name == "np.linspace":
                return [args[0] + (args[1]-args[0])*k/(args[2]-1) for k in range(args[2])]
            if short == "max" and isinstance(node.func, ast.Attribute) and not args:
                return max(self.evaluate(node.func.value))
            if short in {"values", "keys", "items"} and isinstance(node.func, ast.Attribute) and not args and not kw:
                value = self.evaluate(node.func.value)
                if isinstance(value, dict):
                    return list({"values":dict.values, "keys":dict.keys, "items":dict.items}[short](value))
            fn = self.module.functions.get(name)
            if fn:
                return self.call_helper(fn, args, kw)
        raise SourceAdapterError(f"Unsupported declaration expression at {self.module.path}:{getattr(node, 'lineno', '?')}: {ast.unparse(node)}")

    @staticmethod
    def _bind(target, value, env):
        if isinstance(target, ast.Name): env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for sub, item in zip(target.elts, value): Scope._bind(sub, item, env)
        else: raise SourceAdapterError("Unsupported comprehension target")

    def call_helper(self, fn, args, kwargs):
        _validate_constructor_body(fn.node)
        params = _arguments(fn.node)
        values = dict(zip(params, args)); values.update(kwargs)
        local = Scope(fn.module, fn.node.body, values)
        returns = [x for x in fn.node.body if isinstance(x, ast.Return)]
        if len(returns) != 1:
            raise SourceAdapterError(f"Constructor helper {fn.node.name} requires a unique unconditional return")
        return local.evaluate(returns[0].value)


def _grid(decl):
    if not isinstance(decl, Declaration):
        raise SourceAdapterError("A state/action grid must be a supported declaration")
    _validate_declaration(decl)
    if decl.kind == "Phased":
        solve, sim = decl.keyword("solve"), decl.keyword("simulate")
        if isinstance(solve, Function) and _arguments(solve.node):
            raise SourceAdapterError("State/action-dependent Phased.solve imputation requires a dependency-aware phase adapter; only zero-argument imputation is supported")
        chosen = sim if isinstance(solve, Function) else solve
        result = _grid(chosen)
        if not isinstance(solve, Function):
            simulation = _grid(sim)
            if {k:v for k,v in result.items() if k != "source_anchor"} != {k:v for k,v in simulation.items() if k != "source_anchor"}:
                raise SourceAdapterError("Distinct solve/simulation grids require a phase-aware domain adapter")
        else:
            result["solve_imputation_source"] = solve.module.anchor(solve.node)
        result["solve_axis"] = not isinstance(solve, Function)
        result["simulation_axis"] = True
        return result
    identity = {"declaration": ast.dump(decl.node, include_attributes=False),
                "module_sha256": decl.scope.module.digest,
                "constructor_inputs": decl.scope.module.symbols}
    result = {"grid_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              "kind": decl.kind, "source_anchor": decl.scope.module.anchor(decl.node)}
    if decl.kind == "DiscreteGrid":
        cat = decl.keyword("category_class")
        if cat is None and decl.node.args:
            cat = decl.scope.evaluate(decl.node.args[0])
        if isinstance(cat, Category):
            result.update(values=list(range(len(cat.members))), decoded_values=list(cat.members))
        else:
            values = decl.keyword("values")
            if values is None:
                raise SourceAdapterError("DiscreteGrid requires categories or explicit values")
            result.update(values=list(values))
    else:
        count = decl.keyword("n_points")
        points = decl.keyword("points")
        if count is None and points is not None:
            count = len(points)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise SourceAdapterError(f"An explicit positive grid size is required for {decl.kind}")
        result["size"] = count
        # Preserve the original numerical grid by identity. Continuous values
        # are never expanded into an artificial categorical model.
    return result


class CallbackAnalysis:
    def __init__(self, module, bindings, state_names, action_names, age_divisor):
        self.module, self.bindings = module, bindings
        self.coordinates = set(state_names) | set(action_names)
        self.age_divisor = age_divisor
        self.dependency_cache = {}
        self.external_parameters = {}

    def function_reads(self, fn, stack=()):
        key = (str(fn.module.path), fn.node.name)
        if key in self.dependency_cache: return self.dependency_cache[key]
        if key in stack: raise SourceAdapterError(f"Recursive callback {fn.node.name}")
        params = set(_arguments(fn.node)); local = {p:{p} for p in params}; result = set()
        def reads(node):
            if isinstance(node, ast.Name):
                return set(local.get(node.id, set()))
            if isinstance(node, ast.Call):
                name = _name(node.func)
                callee = fn.module.functions.get(name)
                if callee:
                    used = self.function_reads(callee, stack + (key,))
                    args = dict(zip(_arguments(callee.node), node.args))
                    args.update({kw.arg: kw.value for kw in node.keywords if kw.arg})
                    return set().union(*(reads(args[x]) for x in used if x in args))
            return set().union(*(reads(c) for c in ast.iter_child_nodes(node)))
        # The supported callback subset is straight-line pure expressions. For
        # control flow or mutation, retain every syntactically loaded argument.
        body = _body_without_docstring(fn.node)
        unsupported = any(isinstance(n, (ast.For, ast.While, ast.If, ast.Try, ast.With,
                                         ast.AugAssign, ast.Yield, ast.Await)) for stmt in body for n in ast.walk(stmt))
        unsupported |= any(isinstance(stmt,ast.Assign) and any(not isinstance(t,ast.Name) for t in stmt.targets)
                           for stmt in body)
        if unsupported:
            result = {n.id for stmt in body for n in ast.walk(stmt)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in params}
        else:
            for stmt in body:
                if isinstance(stmt, ast.Assign):
                    dependency = reads(stmt.value)
                    for target in stmt.targets:
                        if isinstance(target, ast.Name): local[target.id] = set(dependency)
                        else: result |= reads(stmt)
                elif isinstance(stmt, ast.Return): result |= reads(stmt.value)
                elif isinstance(stmt, ast.Expr): result |= reads(stmt.value)
                elif isinstance(stmt, ast.Delete):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name): local.pop(target.id,None)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.value is not None: local[stmt.target.id] = reads(stmt.value)
                else: result |= reads(stmt)
        self.dependency_cache[key] = result
        return result

    def leaf_reads(self, entry, seen=()):
        if isinstance(entry, Function):
            names = self.function_reads(entry)
            result = set()
            for name in names:
                if name in self.coordinates or name in {"age", "period"}:
                    result.add("age" if name == "period" else name)
                elif name in self.bindings:
                    if name in seen:
                        raise SourceAdapterError(f"Cyclic callback dependency {name}")
                    result |= self.leaf_reads(self.bindings[name], seen + (name,))
                else:
                    # The original callback signature, rather than an unknown
                    # DSL identifier, is the evidence that this leaf is an
                    # externally supplied fixed source parameter.
                    result.add(name)
                    witness = {"function": entry.node.name,
                               "source_anchor": entry.module.anchor(entry.node)}
                    existing = self.external_parameters.setdefault(name, [])
                    if witness not in existing:
                        existing.append(witness)
            return result
        if isinstance(entry, Declaration):
            _validate_declaration(entry)
            if entry.kind == "fixed_transition":
                return {entry.scope.evaluate(entry.node.args[0])}
            if entry.kind == "StochasticTransition": return self.leaf_reads(entry.keyword("func"), seen)
            if entry.kind == "Phased":
                return self.leaf_reads(entry.keyword("solve"), seen) | self.leaf_reads(entry.keyword("simulate"), seen)
            raise SourceAdapterError(f"Unsupported callback declaration {entry.kind}")
        if isinstance(entry, dict):
            return set().union(*(self.leaf_reads(v, seen) for v in entry.values()))
        raise SourceAdapterError(f"Unsupported source callback {type(entry).__name__}")

    def opaque(self, entry):
        if isinstance(entry, Function):
            canonical = ast.dump(ast.Module(body=_body_without_docstring(entry.node), type_ignores=[]), include_attributes=False)
            anchor = entry.module.anchor(entry.node)
        elif isinstance(entry, Declaration):
            canonical = ast.dump(entry.node, include_attributes=False)
            anchor = entry.scope.module.anchor(entry.node)
        else:
            canonical = repr(entry); anchor = None
        result = {"opaque": hashlib.sha256(canonical.encode()).hexdigest(),
                  "reads": sorted(self.leaf_reads(entry))}
        if anchor: result["source_anchor"] = anchor
        return result

    def predicate(self, fn):
        if not isinstance(fn, Function): return self.opaque(fn)
        local = {}
        assigned = set(_arguments(fn.node))
        for stmt in _body_without_docstring(fn.node):
            if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
                if any(t.id in assigned for t in stmt.targets): return self.opaque(fn)
                for target in stmt.targets: local[target.id] = stmt.value
                assigned.update(t.id for t in stmt.targets)
            elif isinstance(stmt, ast.Return): return self._predicate_node(fn, stmt.value, local)
            elif isinstance(stmt, ast.Delete): continue
            else: return self.opaque(fn)
        return self.opaque(fn)

    def _predicate_node(self, fn, node, local, seen=()):
        if isinstance(node, ast.Name) and node.id in local and node.id not in seen:
            return self._predicate_node(fn, local[node.id], local, seen + (node.id,))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            return {"and" if isinstance(node.op, ast.BitAnd) else "or": [
                self._predicate_node(fn, node.left, local, seen), self._predicate_node(fn, node.right, local, seen)]}
        if isinstance(node, ast.BoolOp):
            return {"and" if isinstance(node.op, ast.And) else "or":
                    [self._predicate_node(fn, x, local, seen) for x in node.values]}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            return {"not": self._predicate_node(fn, node.operand, local, seen)}
        try:
            return self._scalar(fn, node, local, seen)
        except SourceAdapterError:
            # A subexpression keeps exactly its live argument ancestry. Its
            # containing AND/OR can still simplify on an exact age restriction.
            temporary = ast.FunctionDef(name=fn.node.name, args=fn.node.args,
                                        body=[ast.Return(value=node)], decorator_list=[])
            temporary.lineno = fn.node.lineno; temporary.end_lineno = fn.node.end_lineno
            # Expand the local bindings for the opaque dependency calculation.
            temporary.body = [ast.Assign(targets=[ast.Name(id=k, ctx=ast.Store())], value=v)
                              for k, v in local.items()] + temporary.body
            key = (str(fn.module.path), temporary.name)
            old = self.dependency_cache.pop(key, None)
            result = self.opaque(Function(fn.module, temporary))
            self.dependency_cache.pop(key, None)
            if old is not None: self.dependency_cache[key] = old
            return result

    def _scalar(self, fn, node, local, seen=()):
        if isinstance(node, ast.Name):
            if node.id in local and node.id not in seen:
                return self._scalar(fn, local[node.id], local, seen + (node.id,))
            if node.id in self.coordinates: return node.id
            if node.id in {"age", "period"}:
                return "age" if self.age_divisor == 1 else f"(age / {self.age_divisor})"
            value = fn.module.scope.lookup(node.id)
            if isinstance(value, (int, float, bool, str, list, tuple, dict)):
                return repr(value)
            raise SourceAdapterError(f"Nonliteral scalar {node.id}")
        if isinstance(node, ast.Constant): return repr(node.value)
        if isinstance(node, ast.Attribute):
            value = fn.module.scope.evaluate(node)
            if isinstance(value, (int, float, bool)): return repr(value)
            raise SourceAdapterError("Nonliteral attribute")
        if isinstance(node, ast.Compare):
            operators: Mapping[type[ast.cmpop], str]={ast.Lt:"<", ast.LtE:"<=", ast.Gt:">", ast.GtE:">=",
                         ast.Eq:"==", ast.NotEq:"!=", ast.In:"in", ast.NotIn:"not in"}  # codespell:ignore
            return "(" + self._scalar(fn, node.left, local, seen) + " " + " ".join(
                operators[type(op)] + " " + self._scalar(fn, rhs, local, seen)
                for op, rhs in zip(node.ops, node.comparators)) + ")"
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY and not isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            operators = {ast.Add:"+", ast.Sub:"-", ast.Mult:"*", ast.Div:"/", ast.FloorDiv:"//", ast.Mod:"%", ast.Pow:"**"}
            return f"({self._scalar(fn,node.left,local,seen)} {operators[type(node.op)]} {self._scalar(fn,node.right,local,seen)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return f"({'-' if isinstance(node.op,ast.USub) else '+'}{self._scalar(fn,node.operand,local,seen)})"
        raise SourceAdapterError("Non-scalar callback expression")


def _age_grid(decl):
    if not isinstance(decl, Declaration) or decl.kind != "AgeGrid":
        raise SourceAdapterError("This adapter requires an explicit AgeGrid constructor")
    _validate_declaration(decl)
    start, stop = Fraction(str(decl.keyword("start"))), Fraction(str(decl.keyword("inclusive_stop")))
    step = {"Y": Fraction(1), "6M": Fraction(1, 2), "M": Fraction(1, 12)}.get(decl.keyword("step"))
    if step is None: raise SourceAdapterError("Supported AgeGrid steps are Y, 6M, and M")
    denominator = math.lcm(start.denominator, stop.denominator, step.denominator)
    ages = list(range(int(start*denominator), int(stop*denominator)+1, int(step*denominator)))
    return ages, denominator


def _in_range(condition, source_age, minimum):
    if isinstance(condition, Declaration):
        _validate_declaration(condition)
        if condition.kind == "AgeRange":
            start = condition.keyword("start", minimum)
            stop = condition.keyword("exclusive_stop", math.inf)
            return source_age >= start and source_age < stop
        if condition.kind == "Periods": return source_age in condition.keyword("values")
    return source_age == condition


def _edge_targets(entry):
    if isinstance(entry, dict): return entry, None
    if isinstance(entry, Declaration) and entry.kind == "Transition":
        _validate_declaration(entry)
        targets = entry.keyword("targets")
        if not isinstance(targets, dict):
            raise SourceAdapterError("Transition without explicit target age ranges is not yet supported")
        return targets, entry.keyword("law")
    raise SourceAdapterError("Phased/ByPeriod routing requires a separate adapter extension")


def _identity_law(entry, state):
    if isinstance(entry, Function):
        body = _body_without_docstring(entry.node)
        return len(body) == 1 and isinstance(body[0], ast.Return) and \
            isinstance(body[0].value, ast.Name) and body[0].value.id == state
    if isinstance(entry, Declaration):
        if entry.kind == "fixed_transition":
            return entry.scope.evaluate(entry.node.args[0]) == state
        if entry.kind == "Phased":
            return _identity_law(entry.keyword("solve"),state) and _identity_law(entry.keyword("simulate"),state)
    if isinstance(entry, dict): return bool(entry) and all(_identity_law(v,state) for v in entry.values())
    return False


def adapt_source(source_path, config=None, *, reads=None):
    """Derive one conservative finder input from original pylcm source.

    ``config`` supplies constructor name and ordinary numeric ``symbols`` (for
    example ``cal.entry_age`` or ``n_assets``). It cannot supply cuts, reduced
    domains, state maps, callback dependency masks, or a proposed graph.
    """
    config = dict(config or {})
    forbidden = {"cuts", "boundaries", "reduced", "state_map", "action_embedding",
                 "necessary_states", "necessary_actions", "expected_graph"} & set(config)
    if forbidden: raise SourceAdapterError(f"Answer-bearing adapter inputs are not accepted: {sorted(forbidden)}")
    accepted = {"adapter", "source", "name", "constructor", "symbols", "discount_factor", "external_report_tags"}
    if set(config)-accepted:
        raise SourceAdapterError(f"Unsupported source adapter configuration: {sorted(set(config)-accepted)}")
    if "discount_factor" not in config:
        raise SourceAdapterError("Declare the unchanged discount_factor in the source manifest")
    module = Module(source_path, config.get("symbols"), reads=reads)
    constructor = config.get("constructor", "create_model")
    fn = module.functions.get(constructor)
    if fn is None: raise SourceAdapterError(f"Constructor {constructor!r} is absent")
    _validate_constructor_body(fn.node)
    scope = Scope(module, fn.node.body)
    models = [x for stmt in fn.node.body for x in ast.walk(stmt)
              if isinstance(x, ast.Call) and _name(x.func) == "Model"]
    if len(models) != 1: raise SourceAdapterError("A unique direct Model constructor is required")
    model = Declaration("Model", models[0], scope)
    _validate_declaration(model)
    age_list, divisor = _age_grid(model.keyword("ages"))
    regimes = model.keyword("regimes")
    if not isinstance(regimes, dict) or not all(isinstance(v, Declaration) and v.kind == "Regime" for v in regimes.values()):
        raise SourceAdapterError("Regime declarations must be resolvable constructor mappings")
    edges = model.keyword("edges", {})
    if not isinstance(edges, dict): raise SourceAdapterError("Phase-specific routing is not in the current constructor subset")
    starts = model.keyword("initial_nodes")
    if not isinstance(starts, (list, tuple)):
        raise SourceAdapterError("Explicit (age, regime) initial nodes are required")
    source_nodes = {(int(Fraction(str(age))*divisor), name) for age, name in starts}
    if not source_nodes: raise SourceAdapterError("The source model declares no admitted initial nodes")
    positions = {age:i for i,age in enumerate(age_list)}
    edge_rows = {}; laws = {}
    for name in regimes:
        if name in edges:
            edge_rows[name], laws[name] = _edge_targets(edges[name])
        else: edge_rows[name], laws[name] = {}, None
    todo = list(source_nodes)
    while todo:
        tick, name = todo.pop()
        if tick not in positions or name not in regimes: raise SourceAdapterError("Invalid source initial/transition node")
        i = positions[tick]
        if i+1 == len(age_list): continue
        for target, condition in edge_rows[name].items():
            if _in_range(condition, tick/divisor, age_list[0]/divisor):
                nxt = (age_list[i+1], target)
                if nxt not in source_nodes: source_nodes.add(nxt); todo.append(nxt)
    overlapping = any(sum((tick, name) in source_nodes for name in regimes)>1 for tick in age_list)
    modes = {name:i for i,name in enumerate(regimes)} if overlapping else {}
    states, actions, programs = {}, {}, []
    external_parameters = {}
    for name, declaration in regimes.items():
        _validate_declaration(declaration)
        state_entries = declaration.keyword("states", {})
        action_entries = declaration.keyword("actions", {})
        functions = declaration.keyword("functions", {})
        constraints = declaration.keyword("constraints", {})
        transitions = declaration.keyword("state_transitions", {})
        for target, entries in ((states, state_entries), (actions, action_entries)):
            for coordinate, grid in entries.items():
                info = _grid(grid)
                if coordinate in target and target[coordinate] != info:
                    # Source anchors do not change the declared grid identity.
                    old = {k:v for k,v in target[coordinate].items() if k != "source_anchor"}
                    new = {k:v for k,v in info.items() if k != "source_anchor"}
                    if old != new: raise SourceAdapterError(f"Different original grids for {coordinate} require per-program domains")
                target[coordinate] = info
        analysis = CallbackAnalysis(module, functions, state_entries, action_entries, divisor)
        ticks = sorted(t for t, n in source_nodes if n == name)
        if not ticks: continue
        when = f"age in {ticks!r}"
        if overlapping: when += f" and source_mode == {modes[name]}"
        terminal = not any(_in_range(c,t/divisor,age_list[0]/divisor)
                           for c in edge_rows[name].values() for t in ticks)
        reward = analysis.opaque(functions.get("utility"))
        rows = []
        for target, condition in edge_rows[name].items():
            valid = [t for t in ticks if _in_range(condition,t/divisor,age_list[0]/divisor)]
            if not valid: continue
            updates = {}
            for coordinate in regimes[target].keyword("states", {}):
                entry = transitions.get(coordinate)
                if isinstance(entry, dict): entry = entry.get(target)
                if entry is None:
                    grid = state_entries.get(coordinate)
                    if isinstance(grid, Declaration) and grid.kind in {"NormalIIDProcess", "TauchenAR1Process", "Phased"}:
                        updates[coordinate] = {"opaque": "intrinsic-process:"+states[coordinate]["grid_id"],
                                               "reads": [coordinate] if grid.kind != "NormalIIDProcess" else [],
                                               "source_anchor": module.anchor(grid.node)}
                    elif coordinate in state_entries:
                        raise SourceAdapterError(f"Missing source law for {name}->{target}:{coordinate}")
                    else:
                        raise SourceAdapterError(f"New target state requires an explicit source law: {name}->{target}:{coordinate}")
                else: updates[coordinate] = analysis.opaque(entry)
            if laws[name] is None:
                probability = f"1 if age in {valid!r} else 0"
            else:
                probability = {"opaque": "routing:"+analysis.opaque(laws[name])["opaque"]+":"+str(modes.get(target,target)),
                               "reads": sorted(analysis.leaf_reads(laws[name])),
                               "source_anchor": module.anchor(laws[name].node)}
            row = {"target": target, "probability": probability, "updates": updates,
                   "phase": "both", "when": f"age in {valid!r}"}
            if overlapping: row["status_updates"] = {"source_mode": str(modes[target])}
            rows.append(row)
        program = {"id": name, "when": when,
                   "declared_states": list(state_entries), "declared_actions": list(action_entries),
                   "role": {"actor": "individual", "observations": ["beginning-of-stage information"],
                            "commitments": ["current action before next-state draws"], "discount_boundary": "one source period"},
                   "reward": reward, "discount": "discount_factor",
                   "feasible": [analysis.predicate(v) for v in constraints.values()] or "True",
                   "terminal": terminal, "terminal_payoff": reward if terminal else "0",
                   "transitions": rows, "outputs": {}, "allocation": "per_block",
                   "source_anchor": module.anchor(declaration.node)}
        if declaration.keyword("taste_shocks") is not None:
            taste = declaration.keyword("taste_shocks")
            _validate_declaration(taste)
            if taste.kind != "ExtremeValueTasteShocks":
                raise SourceAdapterError("Only the original default ExtremeValueTasteShocks declaration is supported")
            program["taste_groups"] = {"kind": "source-declared", "discrete_axes":
                                      [k for k in action_entries if actions[k]["kind"] == "DiscreteGrid"],
                                      "source_anchor": module.anchor(declaration.node)}
        for parameter, witnesses in analysis.external_parameters.items():
            existing = external_parameters.setdefault(parameter, [])
            existing.extend(w for w in witnesses if w not in existing)
        programs.append(program)
    # External tags are obligations, not instructions to remove a Bellman axis.
    tags = list(config.get("external_report_tags", []))
    for tag in tags:
        if tag not in states: raise SourceAdapterError(f"Unknown external report tag {tag}")
        for name, declaration in regimes.items():
            if tag in declaration.keyword("states", {}) and edge_rows[name]:
                if not _identity_law(declaration.keyword("state_transitions", {}).get(tag), tag):
                    raise SourceAdapterError(f"External tag {tag} is not source-proved immutable in {name}")
    result = {"schema_version": 1, "name": config.get("name", module.path.stem),
              "ages": age_list, "age_scale": f"1/{divisor}",
              "age_labels": {str(t):str(Fraction(t,divisor)) for t in age_list},
              "states": states, "actions": actions,
              "parameters": {"discount_factor": config["discount_factor"]},
              "external_parameters": sorted(external_parameters),
              "programs": programs, "external_report_tags": tags,
              "starts": [{"age":t,"program":n,**({"status":{"source_mode":modes[n]}} if overlapping else {})}
                         for t,n in sorted({(int(Fraction(str(a))*divisor),n) for a,n in starts})],
              "source_adapter": {"kind":"pylcm_ast", "status":"SUPPORTED_STATIC",
                "execution_status":"STRUCTURAL_SOURCE_TRANSLATION_EXECUTED",
                "native_model_execution":"NOT_RUN", "constructor":constructor,
                "input_source":str(module.path),"input_sha256":module.digest,
                "constructor_inputs":config.get("symbols",{}),
                "external_parameter_declarations":external_parameters,
                "other_callback_parameters":"Uninterpreted and held identical across every candidate; no parameter values or equations are changed.",
                "source_files":[{"path":p,"sha256":m.digest} for p,m in sorted(module.cache.items())],
                "sources":[{"path":p,"sha256":m.digest} for p,m in sorted(module.cache.items())],
                "supported_claim":"Conservative exact structural dependencies and supported Boolean feasibility; all original continuous grids/operators remain unchanged.",
                "limitations":["No native JAX/pylcm execution or source numerical-value validation.",
                   "Opaque callback purity and source parameter validity remain explicit source bridge assumptions.",
                   "No support pruning, compound-state quotient discovery, arbitrary continuous-source equivalence, or nonlinear/taste-law proof.",
                   "Overlapping original source modes form a fixed observed status family; cross-mode coalescing is outside this adapter's class."],
                "no_oracle_inputs":True}}
    result["source_adapter"]["unsupported"] = list(result["source_adapter"]["limitations"])
    if overlapping: result["status_axes"] = {"source_mode":list(modes.values())}
    result["source_adapter"]["mode_labels"] = {str(v):k for k,v in modes.items()}
    return result


def manifest_source(manifest_path, *, reads):
    config = dict(reads.document(manifest_path))
    if config.get("adapter") != "pylcm_ast": raise SourceAdapterError("Unknown source adapter")
    source = reads.child(manifest_path, 'source', config.pop("source"), base='resolved_parent')
    config.pop("adapter", None)
    return source, config


def register_source_inputs(source, reads, visited=None):
    """Register sibling imports by the same Module resolution used at runtime."""
    visited = set() if visited is None else visited
    resolved = reads.bind(source)
    if resolved in visited: return
    visited.add(resolved)
    module = Module(source, reads=reads)
    for name in module.imports:
        path = module.import_path(name)
        if path.is_file():
            reads.child(source, 'import.'+name, str(path), base='resolved_parent')
            register_source_inputs(path, reads, visited)


def adapt_manifest(manifest_path, *, reads=None):
    if reads is None:
        from .input_reads import discover_inputs
        reads = discover_inputs([manifest_path])
    source, config = manifest_source(manifest_path, reads=reads)
    return adapt_source(source, config, reads=reads)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = adapt_manifest(args.manifest)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
