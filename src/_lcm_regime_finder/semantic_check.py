"""Independent declaration/projection certificate checker.

No producer translator, preparation function, segment builder, or scorer is
imported. Callback proofs are replayed directly from captured original Python
AST and call-target identities. This is a conservative, explicit pure scalar
language, not a claim to decide arbitrary Python equivalence. Only a single
categorical characteristic domain may be evaluated as construction metadata;
state/action products and Bellman rows are never enumerated here.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from .records import SemanticCheck, Diagnostic
import hashlib
import json
import math
import operator
from itertools import product
from typing import Any, Mapping


_UNKNOWN = object()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _decode(value: Any) -> Any:
    if isinstance(value, list): return [_decode(item) for item in value]
    if not isinstance(value, dict): return value
    if set(value) == {"fraction"}: return Fraction(value["fraction"])
    if value.get("kind") == "array" and "values" in value: return _decode(value["values"])
    if value.get("kind") == "categorical-class": return {key: _decode(item) for key, item in value.get("members", {}).items()}
    return {key: _decode(item) for key, item in value.items()}


def _value_key(value: Any) -> Any:
    """Exact scalar image identity, with no categorical-code reinterpretation.

    The admitted scalar expression language uses rational numeric literals, so
    0 and 0.0 describe the same numeric image. Booleans and strings remain
    distinct even though Python happens to equate True with 1.
    """
    value = _decode(value)
    if isinstance(value, bool): return ("boolean", value)
    if isinstance(value, (int, float, Fraction)):
        return ("number", str(Fraction(str(value))))
    if isinstance(value, list): return ("list", tuple(_value_key(item) for item in value))
    if isinstance(value, tuple): return ("tuple", tuple(_value_key(item) for item in value))
    if isinstance(value, dict): return ("mapping", tuple(sorted((str(key), _value_key(item)) for key, item in value.items())))
    return (type(value).__name__, value)


@dataclass(frozen=True)
class _Abstract:
    reads: frozenset[str] = frozenset()
    value: Any = _UNKNOWN
    proved: bool = True


def _constant(value: Any) -> _Abstract:
    return _Abstract(value=value)


def _join(*values: _Abstract, proved: bool = True) -> _Abstract:
    return _Abstract(frozenset().union(*(value.reads for value in values)),
                     proved=proved and all(value.proved for value in values))


def _contains_unresolved_specialization(value):
    """Inspect original declaration slots; never trust a producer marker.

    A selected concrete callable no longer contains the specialization wrapper.
    Callable bodies/bindings are not declaration slots and are not traversed.
    """
    if isinstance(value, list):
        return any(_contains_unresolved_specialization(item) for item in value)
    if not isinstance(value, Mapping):
        return False
    if value.get('kind') in {'callable', 'primitive', 'callable-reference'}:
        return False
    tag = value.get('type', '')
    if isinstance(tag, str) and tag.rsplit('.', 1)[-1] in {'AgeSpecializedFunction', 'PeriodSpecializedFunction'}:
        return True
    children = value.get('fields', {}) if value.get('kind') in {'dataclass', 'object'} else value
    return any(_contains_unresolved_specialization(item) for item in children.values())


def _original_graph_omission(value):
    if isinstance(value,list): return any(_original_graph_omission(item) for item in value)
    if not isinstance(value,Mapping): return False
    if value.get('kind') in {'callable','primitive','callable-reference'}:
        return any(item=='Original graph checkpoint omits optional callback expansion'
                   for item in value.get('obligations',[]))
    slots=value.get('fields',{}) if value.get('kind') in {'object','dataclass'} else value
    return (any(_original_graph_omission(item) for item in slots.values())
            or _original_graph_omission(value.get('callable')))


def _original_inputs_unresolved(value, captures):
    """Re-derive finite effective input authority from the original payload.

    Producer flags, statuses and obligations are not evidence. Captured helper
    bodies are not outer declaration slots. A finite advertised signature may
    legitimately wrap a variadic implementation and keeps all advertised inputs.
    """
    if isinstance(value, list):
        return any(_original_inputs_unresolved(item, captures) for item in value)
    if not isinstance(value, Mapping):
        return False
    # Direct regime-declarations/v1 callbacks may omit a kind tag. Their
    # bindings/call targets are still bodies, not nested declaration slots.
    callable_envelope = (value.get('kind') in {'callable', 'primitive'}
                         or ('kind' not in value and 'arguments' in value
                             and ('source' in value or 'bindings' in value)))
    if callable_envelope:
        names = value.get('arguments')
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            return True
        if len(set(names)) != len(names):
            return True
        if 'parameter_kinds' not in value or value['parameter_kinds'] is None:
            # The older capture contract records finite argument names and
            # explicitly marks signatures it could not enumerate.
            return bool(value.get('variadic', False))
        kinds = value['parameter_kinds']
        if not isinstance(kinds, Mapping) or set(kinds) != set(names):
            return True
        return not all(kind in ('POSITIONAL_ONLY', 'POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY')
                       for kind in kinds.values())
    if value.get('callable_records') and not captures.records:
        captures = _CallableRecords(value['callable_records'])
    if value.get('kind') == 'callable-reference':
        resolved = captures.resolve(value)
        if not isinstance(resolved, Mapping) or resolved.get('kind') not in {'callable', 'primitive'}:
            return True
        return _original_inputs_unresolved(resolved, captures)
    if value.get('kind') in {'object', 'dataclass'} and 'callable' in value:
        primary = value['callable']
        if not isinstance(primary, Mapping) or primary.get('kind') not in ('callable', 'primitive', 'callable-reference'):
            return True
        return _original_inputs_unresolved(primary, captures)
    slots = value.get('fields', {}) if value.get('kind') in {'object', 'dataclass'} else value
    return (any(_original_inputs_unresolved(item, captures) for item in slots.values())
            or _original_inputs_unresolved(value.get('callable'), captures))


class _CallableRecords:
    """Independent verification of the source-bound compact capture format.

    This checker does not import the capture writer or its hash helpers. A
    record is checked once, and references are then constant-size lookups.
    No body or binding graph is recursively expanded.
    """

    def __init__(self, records=None):
        self.records = records or {}
        self.errors = []
        self.binding_values = {}
        self.helper_keys = {}
        self.functions = {}
        if not isinstance(self.records, dict):
            self.errors.append("callable record table is not a mapping")
            self.records = {}
            return
        omitted = {"record_sha256", "identity_sha256", "capture_graph_sha256",
                   "callable_records", "capture_report"}

        def payload(value, *, root=False):
            if isinstance(value, dict):
                if value.get("kind") == "callable-reference":
                    return {"kind": "callable-reference", "capture_ref": value.get("capture_ref")}
                return {key: payload(item) for key, item in value.items() if not root or key not in omitted}
            if isinstance(value, list):
                return [payload(item) for item in value]
            return value

        self.payload = payload

        hashes = {}
        for ref, record in self.records.items():
            if not isinstance(record, dict) or record.get("capture_ref") != ref:
                self.errors.append("callable record key/identity mismatch: " + str(ref))
                continue
            hashes[ref] = _digest(payload(record, root=True))
            if hashes[ref] != record.get("record_sha256"):
                self.errors.append("callable record payload digest changed: " + ref)
        graph = _digest(hashes)
        for ref, record in self.records.items():
            if not isinstance(record, dict) or ref not in hashes:
                continue
            identity = _digest({"capture_ref": ref, "record_sha256": hashes[ref],
                                "capture_graph_sha256": graph})
            if record.get("capture_graph_sha256") != graph or record.get("identity_sha256") != identity:
                self.errors.append("callable graph identity changed: " + ref)
            pending = list(record.values())
            while pending:
                value = pending.pop()
                if isinstance(value, list):
                    pending.extend(value)
                elif isinstance(value, dict):
                    if value.get("kind") == "callable-reference":
                        target = self.records.get(value.get("capture_ref"), {})
                        if (not target or value.get("identity_sha256") != target.get("identity_sha256")
                                or value.get("capture_graph_sha256") != graph):
                            self.errors.append("callable reference target changed: " + str(value.get("capture_ref")))
                    else:
                        pending.extend(value.values())

    def resolve(self, value):
        if not isinstance(value, Mapping):
            return value
        if value.get("kind") != "callable-reference" and "capture_ref" not in value:
            return value
        target = self.records.get(value.get("capture_ref"))
        if (self.errors or not target or value.get("identity_sha256") != target.get("identity_sha256")
                or value.get("capture_graph_sha256") != target.get("capture_graph_sha256")):
            return None
        if value.get("kind") != "callable-reference" and _digest(self.payload(value, root=True)) != target.get("record_sha256"):
            return None
        return target

    def function(self, source):
        # Cache syntax only; dates, bindings and proof results remain local.
        if source not in self.functions:
            try:
                self.functions[source] = next(node for node in ast.parse(source).body
                                              if isinstance(node, ast.FunctionDef))
            except (SyntaxError, StopIteration):
                self.functions[source] = None
        return self.functions[source]

    def helper_key(self, value):
        # Retain the actual object so id reuse cannot alias another payload.
        # This cache is local to the independently checked frozen transaction.
        key = id(value)
        if key not in self.helper_keys:
            self.helper_keys[key] = (value, 'helper:' + _digest(value))
        return self.helper_keys[key][1]

    def binding_value(self, value):
        """Independently decode typed immutable bindings under a scalar budget.

        Shared children remain shared. The memo also stores their saturated
        logical size, so an exponentially large repeated data DAG cannot evade
        the finite scalar-table language merely by having few capture records.
        """
        def visit(item, active, depth):
            if depth > 64:
                raise ValueError('captured binding exceeds scalar depth budget')
            if isinstance(item, dict) and item.get('kind') == 'callable-reference':
                target = self.resolve(item)
                ref = item.get('capture_ref')
                if (not isinstance(target, dict) or target.get('kind') != 'binding-record'
                        or target.get('capture_status', 'COMPLETE') != 'COMPLETE'):
                    raise ValueError('capture reference has no complete immutable-data target')
                if ref in active:
                    raise ValueError('cyclic captured binding lacks scalar semantics')
                if ref not in self.binding_values:
                    decoded, size = visit(target.get('value'), active | {ref}, depth + 1)
                    binding_type = target.get('binding_type')
                    if binding_type == 'builtins.tuple': decoded = tuple(decoded)
                    elif binding_type == 'builtins.set': decoded = set(decoded)
                    elif binding_type == 'builtins.frozenset': decoded = frozenset(decoded)
                    self.binding_values[ref] = (decoded, size)
                return self.binding_values[ref]
            if isinstance(item, dict):
                if set(item) == {'fraction'}: return Fraction(item['fraction']), 1
                if item.get('kind') == 'array' and 'values' in item:
                    return visit(item['values'], active, depth + 1)
                if item.get('kind') == 'categorical-class':
                    return visit(item.get('members', {}), active, depth + 1)
                if item.get('kind') in {'callable', 'primitive', 'module', 'unresolved'}:
                    raise ValueError('callable or opaque binding is not immutable scalar data')
                result, size = {}, 1
                for key, child in item.items():
                    decoded, child_size = visit(child, active, depth + 1)
                    size += child_size
                    if size > 8192: raise ValueError('captured binding exceeds scalar item budget')
                    result[key] = decoded
                return result, size
            if isinstance(item, (list, tuple)):
                result, size = [], 1
                for child in item:
                    decoded, child_size = visit(child, active, depth + 1)
                    size += child_size
                    if size > 8192: raise ValueError('captured binding exceeds scalar item budget')
                    result.append(decoded)
                return (tuple(result) if isinstance(item, tuple) else result), size
            if isinstance(item, str) and len(item.encode('utf-8')) > 65536:
                raise ValueError('captured binding exceeds scalar string budget')
            return item, 1
        return visit(value, set(), 0)[0]


class SourceProof:
    """Abstract interpretation of ORIGINAL callbacks, with bounded recursion.

    ``static`` contains dates, fixed parameters or a single characteristic input.
    ``providers`` is the original named function DAG. ``stop`` maps a certified
    characteristic's original provider name to its retained projected axis.
    Unknown calls remain conservative and are never interpreted by spelling.
    """

    def __init__(self, callbacks: Mapping[str, dict], providers: Mapping[str, str] | None = None,
                 static: Mapping[str, Any] | None = None, stop: Mapping[str, str] | None = None,
                 callable_records=None):
        self.callbacks = callbacks
        self.providers = providers or {}
        self.static = static or {}
        self.stop = stop or {}
        self.callable_records = (callable_records if isinstance(callable_records, _CallableRecords)
                                 else _CallableRecords(callable_records))
        self.notes: set[str] = set()
        self.steps = 0
        self.active: list[str] = []
        self.resolving_inputs: set[str] = set()

    def _binding(self, value):
        try:
            return _constant(self.callable_records.binding_value(value))
        except (ValueError, TypeError, KeyError, RecursionError):
            self.notes.add('original binding has no bounded immutable scalar interpretation')
            return _Abstract(proved=False)

    def _name(self, name: str) -> _Abstract:
        if name in self.static: return _constant(_decode(self.static[name]))
        if name in self.stop:
            target = self.stop[name]
            return _constant(self.static[target]) if target in self.static else _Abstract(frozenset({target}))
        if name in self.providers:
            target = self.providers[name]
            if isinstance(target, dict) and "derived_code_of" in target:
                value = self._name(target["derived_code_of"])
                if value.value is _UNKNOWN: return value
                indexes = [index for index, item in enumerate(target.get("values", []))
                           if _value_key(item) == _value_key(value.value)]
                if len(indexes) == 1 and len(target.get("codes", [])) == len(target.get("values", [])):
                    return _constant(target["codes"][indexes[0]])
                self.notes.add("derived category code has no unique original decoded value")
                return _Abstract(value.reads, proved=False)
            if target in self.active or target in self.resolving_inputs:
                self.notes.add("cyclic original function DAG")
                return _Abstract(frozenset({name}), proved=False)
            return self.callback(target)
        return _Abstract(frozenset({name}))

    def callback(self, callback: str | Mapping[str, Any], supplied: Mapping[str, _Abstract] | None = None) -> _Abstract:
        if self.steps > 100_000 or len(self.active) > 32:
            self.notes.add("source proof budget exhausted")
            return _join(_Abstract(frozenset(self.providers)),
                         *(supplied or {}).values(), proved=False)
        if isinstance(callback, str):
            for name, target in self.providers.items():
                if target == callback and name in self.stop:
                    return self._name(name)
            if callback not in self.callbacks:
                self.notes.add("missing original callback " + callback)
                return _Abstract(frozenset({callback}), proved=False)
            record = self.callbacks[callback]
            ident = callback
        else:
            record, ident = callback, self.callable_records.helper_key(callback)
        meta = record.get("original", record)
        if not isinstance(meta, Mapping):
            return _constant(_decode(meta))
        if meta.get("callable_records") and not self.callable_records.records:
            self.callable_records = _CallableRecords(meta["callable_records"])
        resolved = self.callable_records.resolve(meta)
        invalid_capture = resolved is None
        if resolved is None:
            self.notes.add("shared original callback reference is unavailable or changed")
        else:
            meta = resolved
        formals = list(meta.get("raw_arguments", meta.get("arguments", record.get("arguments", []))))
        provided = supplied or {}
        env = {key: self._binding(value) for key, value in meta.get("bindings", {}).items()}
        effective = list(meta.get("arguments", formals))
        exposed = set(effective)
        input_names = list(dict.fromkeys([*formals, *effective]))
        # Keyword partials supply defaults, not immutable application values.
        # Exposed formals resolve from the model DAG at the outer boundary and
        # from explicit arguments/effective defaults at a helper call site.
        env.update({key: self._binding(value) for key, value in meta.get("partial_bindings", meta.get("partial_keywords", {})).items()
                    if key not in exposed})
        self.resolving_inputs.add(ident)
        try:
            for name in input_names:
                if name in provided: env[name] = provided[name]
                elif name not in env:
                    if supplied is not None and name in meta.get("defaults", {}):
                        env[name] = self._binding(meta["defaults"][name])
                    else:
                        env[name] = self._name(name)
        finally:
            self.resolving_inputs.discard(ident)
        # The callable's effective signature can differ from the inspected
        # wrapper's raw self/*args/**kwargs signature (e.g. a library ufunc).
        # Every actual argument has already been evaluated in the caller's
        # environment. An unavailable body must retain those dependencies;
        # rebuilding a fallback from raw implementation names loses them.
        unresolved_specialization = _contains_unresolved_specialization(meta)
        unresolved_inputs = _original_inputs_unresolved(meta, self.callable_records)
        structural = (record.get('dependency_contract') == 'pylcm-declared-inputs/v1'
                      and not unresolved_specialization and not unresolved_inputs)
        fallback_names = effective if structural else input_names
        fallback = _join(*(env[name] for name in fallback_names), *provided.values(), proved=structural)
        if invalid_capture or self.callable_records.errors or meta.get("capture_status", "COMPLETE") != "COMPLETE":
            self.notes.add("original compact capture is incomplete or its identity changed")
            return _join(fallback, proved=False)
        if unresolved_specialization:
            self.notes.add('specialized economic callable/signature is not captured')
            return _join(fallback, proved=False)
        if unresolved_inputs:
            self.notes.add('finite effective original input signature is unavailable')
            return _join(fallback, proved=False)
        if structural:
            from .graph_scope import identity_state
            state = identity_state(meta)
            if state is not None: return env[state]
            if record.get('status') == 'structural' and record.get('expression') is None:
                # Opaque Model bodies require only their complete effective inputs.
                return fallback
        if meta.get("variadic"):
            self.notes.add("variadic original callback " + ident)
            return fallback
        fn = self.callable_records.function(meta.get("source", ""))
        if fn is None:
            self.notes.add("opaque original callback " + ident)
            return fallback
        parsed = [arg.arg for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
        if fn.decorator_list and meta.get("capture_contract") not in {"inspection-only-frozen-bindings-1", "inspection-only-frozen-bindings-2"}:
            self.notes.add("legacy decorated body is not the actual callable " + ident)
            return fallback
        if fn.args.vararg or fn.args.kwarg or set(parsed) != set(formals):
            self.notes.add("captured signature/source mismatch " + ident)
            return fallback
        self.active.append(ident)
        try:
            answer = self._body(fn.body, env, meta)
            return fallback if answer is None or (structural and not answer.proved) else answer
        except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError, IndexError):
            self.notes.add("unsupported original scalar operation " + ident)
            return fallback
        finally:
            self.active.pop()

    def _body(self, statements: list, env: dict, meta: Mapping) -> _Abstract | None:
        for stmt in statements:
            self.steps += 1
            if self.steps > 100_000: return None
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str): continue
            if isinstance(stmt, ast.Return): return self._expr(stmt.value, env, meta)
            if isinstance(stmt, ast.Assign) and all(isinstance(target, ast.Name) for target in stmt.targets):
                value = self._expr(stmt.value, env, meta)
                for target in stmt.targets: env[target.id] = value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                env[stmt.target.id] = self._expr(stmt.value, env, meta)
            elif isinstance(stmt, ast.Delete) and all(isinstance(target, ast.Name) for target in stmt.targets):
                for target in stmt.targets: env.pop(target.id, None)
            elif isinstance(stmt, ast.If):
                test = self._expr(stmt.test, env, meta)
                if test.value is not _UNKNOWN:
                    result = self._body(stmt.body if test.value else stmt.orelse, env, meta)
                    if result is not None: return result
                else:
                    left = self._body(stmt.body, dict(env), meta)
                    right = self._body(stmt.orelse, dict(env), meta)
                    if left is None or right is None: return None
                    if left.value is not _UNKNOWN and _value_key(left.value) == _value_key(right.value): return left
                    return _join(test, left, right)
            else:
                return None
        return None

    def _expr(self, node: ast.AST, env: Mapping[str, _Abstract], meta: Mapping) -> _Abstract:
        self.steps += 1
        if self.steps > 100_000: raise ValueError("source budget")
        e = lambda child: self._expr(child, env, meta)
        if isinstance(node, ast.Constant): return _constant(node.value)
        if isinstance(node, ast.Name): return env[node.id] if node.id in env else self._name(node.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [e(child) for child in node.elts]
            if all(value.value is not _UNKNOWN for value in values):
                result = [value.value for value in values]
                return _constant(tuple(result) if isinstance(node, ast.Tuple) else result)
            return _join(*values)
        if isinstance(node, ast.Dict):
            pairs = [(e(key), e(value)) for key, value in zip(node.keys, node.values)]
            if all(key.value is not _UNKNOWN and value.value is not _UNKNOWN for key, value in pairs): return _constant({key.value: value.value for key, value in pairs})
            return _join(*(item for pair in pairs for item in pair))
        if isinstance(node, ast.Attribute):
            value = e(node.value)
            if value.value is not _UNKNOWN and isinstance(value.value, dict) and node.attr in value.value:
                return _constant(value.value[node.attr])
            return _Abstract(value.reads, proved=False)
        if isinstance(node, ast.Subscript):
            value, index = e(node.value), e(node.slice)
            if value.value is not _UNKNOWN and index.value is not _UNKNOWN: return _constant(value.value[index.value])
            return _join(value, index)
        if isinstance(node, ast.IfExp):
            condition = e(node.test)
            if condition.value is not _UNKNOWN: return e(node.body if condition.value else node.orelse)
            left, right = e(node.body), e(node.orelse)
            if left.value is not _UNKNOWN and _value_key(left.value) == _value_key(right.value): return left
            return _join(condition, left, right)
        if isinstance(node, ast.UnaryOp):
            value = e(node.operand)
            operation = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}.get(type(node.op))
            return _constant(operation(value.value)) if operation and value.value is not _UNKNOWN else value
        if isinstance(node, ast.BinOp):
            left, right = e(node.left), e(node.right)
            operation = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                         ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                         ast.BitAnd: operator.and_, ast.BitOr: operator.or_}.get(type(node.op))
            if operation and left.value is not _UNKNOWN and right.value is not _UNKNOWN: return _constant(operation(left.value, right.value))
            return _join(left, right)
        if isinstance(node, ast.BoolOp):
            values = [e(child) for child in node.values]
            if isinstance(node.op, ast.And) and any(value.value is not _UNKNOWN and not value.value for value in values): return _constant(False)
            if isinstance(node.op, ast.Or) and any(value.value is not _UNKNOWN and value.value for value in values): return _constant(True)
            if all(value.value is not _UNKNOWN for value in values): return _constant(all(value.value for value in values) if isinstance(node.op, ast.And) else any(value.value for value in values))
            return _join(*values)
        if isinstance(node, ast.Compare):
            values = [e(node.left)] + [e(child) for child in node.comparators]
            operations: Mapping[type[ast.cmpop], Callable[..., bool]] = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                          ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}  # codespell:ignore
            if all(value.value is not _UNKNOWN for value in values) and all(type(op) in operations for op in node.ops):
                return _constant(all(operations[type(op)](a.value, b.value) for op, a, b in zip(node.ops, values, values[1:])))
            return _join(*values)
        if isinstance(node, ast.Call):
            target = self.callable_records.resolve(meta.get("call_targets", {}).get(ast.unparse(node.func)))
            args = [e(arg) for arg in node.args]
            kwargs = {kw.arg: e(kw.value) for kw in node.keywords if kw.arg is not None}
            if isinstance(target, dict) and target.get("kind") == "callable":
                names = target.get("arguments", target.get("raw_arguments", []))
                kinds = target.get("parameter_kinds", {})
                positional = [name for name in names if not kinds or kinds.get(name) in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}]
                supplied_names = positional[:len(args)]
                partial_has_kinds = not ("partial_args" in target or "partial_keywords" in target) or bool(kinds)
                valid = (partial_has_kinds and len(args) <= len(positional)
                         and not any(isinstance(arg, ast.Starred) for arg in node.args)
                         and all(kw.arg is not None for kw in node.keywords)
                         and len(kwargs) == len(node.keywords)
                         and not (set(kwargs) & set(supplied_names))
                         and all(name in names and kinds.get(name) != "POSITIONAL_ONLY" for name in kwargs))
                supplied_values = {**dict(zip(supplied_names, args)), **kwargs}
                if valid and all(name in supplied_values or name in target.get("defaults", {}) for name in names):
                    return self.callback(target, supplied_values)
            if isinstance(target, dict) and target.get("kind") == "primitive" and target.get("resolution") == "actual_object_identity":
                canonical, operation = target.get("canonical"), target.get("operation")
                allowed = {"numpy.where": "select", "jax.numpy.where": "select",
                           "numpy.logical_and": "logical_and", "jax.numpy.logical_and": "logical_and",
                           "numpy.logical_or": "logical_or", "jax.numpy.logical_or": "logical_or",
                           "numpy.asarray": "asarray", "jax.numpy.asarray": "asarray"}
                if allowed.get(canonical) == operation and not kwargs:
                    if operation == "select" and len(args) == 3:
                        if args[0].value is not _UNKNOWN: return args[1] if args[0].value else args[2]
                        if args[1].value is not _UNKNOWN and _value_key(args[1].value) == _value_key(args[2].value): return args[1]
                        return _join(*args)
                    if operation == "asarray" and len(args) == 1: return args[0]
                    if operation in {"logical_and", "logical_or"} and len(args) == 2:
                        if all(arg.value is not _UNKNOWN for arg in args): return _constant(bool(args[0].value and args[1].value) if operation == "logical_and" else bool(args[0].value or args[1].value))
                        return _join(*args)
            self.notes.add("opaque call target " + ast.unparse(node.func))
            # Unknown helper may have captured dependencies. Its signature and
            # every original formal remain a conservative proof obligation.
            return _join(*args, *kwargs.values(), *env.values(), proved=False)
        return _join(*env.values(), proved=False)


def _pointer(value: Any, path: str) -> Any:
    if path == "": return value
    for part in path.lstrip("/").split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _identity_update(update: Any, state: str) -> bool:
    return update == state or (isinstance(update, dict) and
        ((update.get("kind") == "identity" and update.get("state", update.get("source")) == state)
         or update.get("unchanged") == state))


def _dated(cell: dict) -> dict:
    return {key: cell[key] for key in ("age", "period") if cell.get(key) is not None}


def _static(cell: dict, domains: dict) -> dict:
    result = _dated(cell)
    for axis, did in cell.get("states", {}).items():
        domain = domains.get(did, {})
        if domain.get("size") == 1 and len(domain.get("values", [])) == 1:
            result[axis] = domain["values"][0]
    return result


def _codes(cell: dict, domains: dict, static: dict) -> dict:
    result = dict(static)
    for axis, domain_id in {**cell.get("states", {}), **cell.get("actions", {})}.items():
        if axis not in static: continue
        domain = domains.get(domain_id, {})
        values, codes = domain.get("values", []), domain.get("codes", [])
        indices = [index for index, value in enumerate(values) if _digest(value) == _digest(static[axis])]
        if len(indices) == 1 and len(codes) == len(values):
            argument = cell.get("category_codes", {}).get(axis, {}).get("argument", axis + "_code")
            result[argument] = codes[indices[0]]
    return result


def _providers(cell: dict, domains: dict, phase: str = "solve") -> dict:
    result = dict(cell.get("phase_functions", {}).get(phase, cell.get("functions", {})))
    for alias, output in cell.get("derived_code_providers", {}).items():
        domain = domains.get(cell.get("derived_categoricals", {}).get(output), {})
        result[alias] = {"derived_code_of": output, "values": domain.get("values", []), "codes": domain.get("codes", [])}
    return result


def _identity_callback(callback: dict, state: str, captures=None) -> bool:
    """Syntactic alias proof, not a one-point numerical identity test."""
    original = callback.get("original", callback)
    if not isinstance(original, dict): return False
    bound=original.get('bound_self',{})
    if (bound.get('type')=='_lcm.identity_transition._IdentityTransition'
            and bound.get('fields',{}).get('_state_name')==state
            and original.get('arguments')==[state]):return True
    source = original.get("source", "")
    fn = (captures or _CallableRecords()).function(source)
    if fn is None:
        return False
    aliases = {argument.arg: argument.arg for argument in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str): continue
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Name) and all(isinstance(target, ast.Name) for target in stmt.targets):
            value = aliases.get(stmt.value.id)
            for target in stmt.targets: aliases[target.id] = value
        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return aliases.get(stmt.value.id) == state
        else:
            return False
    return False


def _closed_native_constant(original: Mapping[str, Any], captures=None) -> bool:
    """Prove only the absence of cell inputs for a captured routing constant.

    The numerical array constructor stays opaque and unchanged. This bridge
    checks the actual bound receiver and source shape; a type spelling or an
    advertised empty signature alone is insufficient. It does not authenticate
    the capture process or prove arbitrary Python source equivalence.
    """
    if original.get("type") not in {
        "_lcm.regime_building.schedules._RegimeCodeConstant",
        "_lcm.regime_building.schedules._ProbabilityConstant",
    }:
        return False
    declaration = original.get("fields", {})
    meta = original.get("callable", {})
    if set(declaration) != {"value"} or type(declaration["value"]) not in (bool, int, float, str):
        return False
    receiver = meta.get("bound_self", {})
    if receiver.get("type") != original["type"] or "value" not in receiver.get("fields", {}):
        return False
    if _value_key(receiver["fields"]["value"]) != _value_key(declaration["value"]):
        return False
    if meta.get("capture_contract") not in {"inspection-only-frozen-bindings-1", "inspection-only-frozen-bindings-2"} or meta.get("arguments") != []:
        return False
    if meta.get("raw_arguments") != ["self"] or meta.get("variadic") is not False:
        return False
    module = meta.get("bindings", {}).get("jnp", {})
    if module.get("kind") != "module" or module.get("name") != "jax.numpy":
        return False
    fn = (captures or _CallableRecords()).function(meta.get("source", ""))
    if fn is None:
        return False
    formals = [arg.arg for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    body = [node for node in fn.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
    if formals != ["self"] or fn.args.vararg or fn.args.kwarg or len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    call = body[0].value
    return (isinstance(call, ast.Call) and not call.keywords and len(call.args) == 1
            and isinstance(call.func, ast.Attribute) and call.func.attr == "asarray"
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "jnp"
            and isinstance(call.args[0], ast.Attribute) and call.args[0].attr == "value"
            and isinstance(call.args[0].value, ast.Name) and call.args[0].value.id == "self")


def _feasibility_channels(consumer: Mapping[str, Any], decision_phase: str) -> tuple[str, ...]:
    """Independent applicability interpretation; absent phases have no constraint.

    Native current truth constraints explicitly apply in both decision modes,
    even though their provider is evaluated in the simulation channel.
    """
    supplied = consumer.get("phase", "both")
    channels = ("solve", "simulation") if supplied == "both" else (supplied,)
    applicability = consumer.get("decision_phases")
    if applicability is None:
        return (decision_phase,) if decision_phase in channels else ()
    return channels if decision_phase in applicability else ()


def check_semantics(original: dict, normalized: dict, analysis: dict, *, check_coverage: bool = True) -> SemanticCheck:
    """Reject malformed certificates as data; never promote unchecked input."""
    try:
        return _check_semantics(original, normalized, analysis, check_coverage=check_coverage)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        return {"status": "FAIL", "errors": [{"kind": "malformed semantic certificate", "detail": str(exc)}],
                "unresolved": [], "checks": 0, "dense_cartesian_rows_evaluated": 0,
                "projection_family_complete_verified": False}


def _check_semantics(original: dict, normalized: dict, analysis: dict, *, check_coverage: bool = True) -> SemanticCheck:
    """Verify source retention, dependencies, projections and future closure.

    PASS is relative to the supplied original declaration/capture contract.
    Missing original source or actual-object bindings yield explicit unresolved
    obligations; they are never promoted to a complete-discovery statement.
    """
    errors: list[Diagnostic] = []
    unresolved: list[Diagnostic] = []
    checks = 0
    diagnostic_scope = {"scope": "source"}
    def fail(kind: str, **data): errors.append({"kind": kind, **diagnostic_scope, **data})
    def unknown(kind: str, **data):
        item: Diagnostic = {"kind": kind, **diagnostic_scope, **data}
        if item not in unresolved: unresolved.append(item)
    if normalized.get("schema") != "regime-declarations/v1":
        return {"status": "FAIL", "errors": [{"kind": "declaration schema"}], "unresolved": [], "checks": 0}
    if normalized.get("original", original) != original: fail("original capture changed")
    if analysis.get("declarations") != normalized: fail("analysis declaration identity")
    if original.get("schema") == "regime-declarations/v1":
        for key in ("domains", "callbacks", "cells", "relations", "characteristics", "allocation", "source_identity"):
            if normalized.get(key) != original.get(key): fail("original declarative field changed", field=key)
    elif original.get("schema") == "constructed-pylcm-model-graph-1" or "graph_snapshot" in original or "regimes" in original:
        _check_capture(original, normalized, fail, unknown)
    else:
        unknown("original source contract is not supported", schema=original.get("schema"))
    captures = _CallableRecords(normalized.get("callable_records", {}))
    for error in captures.errors:
        fail("compact callable capture identity", detail=error)
    callbacks = normalized.get("callbacks", {})
    domains = {**normalized.get("domains", {}), **analysis.get("derived_domains", {})}
    cells = {cell["id"]: cell for cell in normalized.get("cells", [])}
    if len(cells) != len(normalized.get("cells", [])): fail("duplicate original cell IDs")
    if set(analysis.get("interfaces", {})) != set(cells): fail("interface cell coverage")
    for key, callback in callbacks.items():
        diagnostic_scope = {"scope": "callback", "callback": key}
        if _original_graph_omission(callback.get('original',callback)):
            unknown('original graph callback signature was not captured',callback=key)
        if _original_inputs_unresolved(callback.get('original', callback), captures):
            unknown('finite effective callback input signature was not captured', callback=key)
        proof = SourceProof({key: callback}, callable_records=captures)
        dependency = proof.callback(key)
        raw_callback = callback.get("original", callback)
        if (isinstance(raw_callback, dict) and "capture_ref" in raw_callback
                and proof.callable_records.resolve(raw_callback) is None):
            fail("original compact callback record copy changed", callback=key)
        declared = set(callback.get("reads", callback.get("arguments", [])))
        formals = set(callback.get("arguments", []))
        if callback.get('dependency_contract') == 'pylcm-declared-inputs/v1':
            if normalized.get('analysis_scope') != 'pylcm-structural-graph/v1':
                fail('declared-input contract outside structural Model scope', callback=key)
            if isinstance(raw_callback, dict) and 'arguments' in raw_callback and callback.get('arguments') != raw_callback['arguments']:
                fail('Model callback declared signature changed', callback=key)
            if isinstance(raw_callback,dict) and raw_callback.get('type') in {'lcm.decoupled.Decoupled', 'lcm.shocks.ShockGroup'}:
                def declared_inputs(value):
                    if isinstance(value,dict):
                        if value.get('kind') in {'callable','primitive','callable-reference'}:
                            resolved=captures.resolve(value)
                            return set(resolved.get('arguments',[])) if resolved else set()
                        if value.get('type')=='lcm.phased.Phased':
                            if callback.get('selected_phase') is None:
                                return set().union(*(declared_inputs(v) for v in value.get('fields',{}).values()))
                            side='solve' if callback.get('selected_phase')=='solve' else 'simulate'
                            return declared_inputs(value.get('fields',{}).get(side))
                        return set().union(*(declared_inputs(v) for v in value.values()))
                    if isinstance(value,list):return set().union(*(declared_inputs(v) for v in value))
                    return set()
                if raw_callback.get('type') == 'lcm.decoupled.Decoupled' and set(raw_callback.get('fields',{}))!={'objective','value'}:
                    unknown('declared aggregator fields not captured',callback=key)
                expected=declared_inputs(raw_callback)
                if set(callback.get('arguments',[]))!=expected or set(callback.get('reads',[]))!=expected:
                    kind = 'shock-law' if raw_callback.get('type') == 'lcm.shocks.ShockGroup' else 'aggregator'
                    fail('declared '+kind+' inputs changed',callback=key)
            if isinstance(raw_callback, dict) and '/states/' in callback.get('source_path',''):
                typ=raw_callback.get('type')
                iid={'lcm.processes.'+n for n in ('NormalIIDProcess','UniformIIDProcess','LogNormalIIDProcess','NormalMixtureIIDProcess')}
                iid.add('_lcm.processes.joint_normal_lowering._JointNormalCoordinate')
                ar1={'lcm.processes.'+n for n in ('TauchenAR1Process','RouwenhorstAR1Process','TauchenNormalMixtureAR1Process')}|{'_lcm.processes.ar1_on_nodes._TauchenOnNodesProcess'}
                if typ in iid|ar1 and raw_callback.get('fields',{}).get('state_conditioned') is None:
                    axis=callback['source_path'].split('/states/',1)[1].split('/',1)[0].replace('~1','/').replace('~0','~')
                    expected=[] if typ in iid else [axis]
                    if callback.get('arguments')!=expected or callback.get('reads')!=expected:
                        fail('intrinsic process dependency changed',callback=key)
        if dependency.reads & formals - declared: fail("omitted original callback dependency", callback=key, missing=sorted(dependency.reads & formals - declared))
        if callback.get("status") == "proved" and not dependency.proved: unknown("callback proof unavailable", callback=key, reasons=sorted(proof.notes))
        checks += 1
    checked_options: dict[str, list[dict]] = {}
    for cell_id, cell in cells.items():
        diagnostic_scope = {"scope": "source", "cell": cell_id}
        options = analysis.get("interfaces", {}).get(cell_id, {}).get("options", [])
        checked_options[cell_id] = options
        if not options: fail("missing cell representation", cell=cell_id)
        if len({option.get("id") for option in options}) != len(options): fail("duplicate option IDs", cell=cell_id)
        for option in options:
            diagnostic_scope = {"scope": "option", "cell": cell_id, "option": option.get("id")}
            state_projection = option.get("state_projection", {})
            reconstruction = option.get("action_reconstruction", {})
            if set(state_projection) != set(cell.get("states", {})): fail("state projection coverage", cell=cell_id, option=option.get("id"))
            if set(reconstruction) != set(cell.get("actions", {})): fail("action reconstruction coverage", cell=cell_id, option=option.get("id"))
            retained: set[str] = set()
            stop: dict[str, str] = {}
            static = _static(cell, domains)
            for state, projection in state_projection.items():
                if state not in cell.get("states", {}): continue
                kind = projection.get("kind")
                target = projection.get("target", state)
                if kind == "identity":
                    if option.get("states", {}).get(target) != cell["states"][state]: fail("identity domain changed", cell=cell_id, state=state)
                    retained.add(state)
                elif kind == "drop":
                    if state in option.get("states", {}): fail("dropped state retained as placeholder", cell=cell_id, state=state)
                elif kind == "characteristic":
                    callback_id = projection.get("callback")
                    characteristic = next((record for record in normalized.get("characteristics", {}).values()
                                           if record.get("callback") == callback_id and record.get("state") == state
                                           and record.get("domain") == cell["states"][state]), None)
                    if characteristic is None:
                        fail("projection is not an existing characteristic", cell=cell_id, state=state)
                        continue
                    original_domain = domains.get(cell["states"][state], {})
                    values = original_domain.get("values")
                    if not isinstance(values, list) or not values or len(values) > 100_000:
                        unknown("characteristic construction domain unavailable", cell=cell_id, state=state)
                        continue
                    projected = []
                    for value in values:
                        proof = SourceProof(callbacks, _providers(cell, domains), _codes(cell, domains, {**static, state: value}), callable_records=captures)
                        answer = proof.callback(callback_id)
                        if not answer.proved or answer.value is _UNKNOWN or answer.reads:
                            unknown("characteristic is outside pure construction language", callback=callback_id)
                            break
                        projected.append(answer.value)
                    if len(projected) == len(values):
                        if [_value_key(value) for value in projection.get("values", [])] != [_value_key(value) for value in projected]: fail("characteristic map differs from original function", cell=cell_id, state=state)
                        derived_id = projection.get("domain", option.get("states", {}).get(target))
                        derived = domains.get(derived_id, {})
                        images = set(_value_key(value) for value in projected)
                        if derived.get("size") != len(images): fail("characteristic image cardinality", cell=cell_id, state=state)
                        if "values" in derived and {_value_key(value) for value in derived["values"]} != images: fail("characteristic image values", cell=cell_id, state=state)
                    if domains.get(projection.get("domain"), {}).get("size") == 1:
                        constant_values = domains[projection["domain"]].get("values", [])
                        if len(constant_values) != 1 or option.get("constants", {}).get(target, _UNKNOWN) != constant_values[0]:
                            fail("constant characteristic reconstruction missing", cell=cell_id, state=state)
                        else:
                            static[target] = constant_values[0]
                        if target in option.get("states", {}): fail("constant characteristic retained as placeholder", cell=cell_id, state=state)
                    elif option.get("states", {}).get(target) != projection.get("domain", option.get("states", {}).get(target)): fail("characteristic target domain mismatch", cell=cell_id, state=state)
                    for name, provider in cell.get("functions", {}).items():
                        if provider == callback_id: stop[name] = target
                    retained.add(target)
                else:
                    fail("unknown state projection", cell=cell_id, state=state)
            for action, embedding in reconstruction.items():
                if action not in cell.get("actions", {}): continue
                if embedding.get("kind") == "identity":
                    if option.get("actions", {}).get(action) != cell["actions"][action]: fail("action identity domain changed", cell=cell_id, action=action)
                    retained.add(action)
                elif embedding.get("kind") == "constant":
                    domain = domains.get(cell["actions"][action], {})
                    if embedding.get("value") not in domain.get("values", []): fail("forced action is not a decoded original value", cell=cell_id, action=action)
                    if action in option.get("actions", {}): fail("forced action placeholder remains", cell=cell_id, action=action)
                    static[action] = embedding.get("value")
                    _check_forced(action, embedding.get("value"), cell, callbacks, domains, fail, unknown, captures)
                else: fail("unknown action embedding", cell=cell_id, action=action)
            for constant, value in option.get("constants", {}).items():
                if constant not in static or _digest(static[constant]) != _digest(value):
                    fail("unproved reconstruction constant", cell=cell_id, name=constant)
            _check_function_interfaces(cell, option, callbacks, domains, static, fail, unknown, captures)
            original_axes = set(cell.get("states", {})) | set(cell.get("actions", {}))
            for consumer in cell.get("consumers", []):
                for phase in (("solve", "simulation") if consumer.get("phase", "both") == "both" else (consumer["phase"],)):
                    providers = _providers(cell, domains, phase)
                    phase_stop = dict(stop)
                    for name, provider in providers.items():
                        if isinstance(provider, dict): continue
                        for projection in state_projection.values():
                            if projection.get("kind") == "characteristic" and callbacks.get(provider, {}).get("identity") == callbacks.get(projection.get("callback"), {}).get("identity"):
                                phase_stop[name] = projection["target"]
                    proof = SourceProof(callbacks, providers, _codes(cell, domains, static), phase_stop, captures)
                    dependency = proof.callback(consumer.get("callback", ""))
                    required = {name[:-5] if name.endswith("_code") and name[:-5] in original_axes else name for name in dependency.reads}
                    missing = (required & original_axes) - retained - set(static)
                    if missing: fail("consumer does not factor through representation", cell=cell_id, option=option.get("id"), consumer=consumer.get("kind"), callback=consumer.get("callback"), phase=phase, states=sorted(missing))
                    if not dependency.proved: unknown("opaque consumer factorization", cell=cell_id, callback=consumer.get("callback"), phase=phase, reasons=sorted(proof.notes))
                    checks += 1
            option_axes = set(option.get("states", {})) | set(option.get("actions", {}))
            justified = retained | {state for state, projection in state_projection.items() if projection.get("kind") == "identity"}
            if option_axes - justified: fail("representation introduces an unproved axis", cell=cell_id, axes=sorted(option_axes - justified))
    # Offered options are combined only by explicit GLOBAL projection policies.
    # Independently choosing a coarse source and a detailed future interface
    # would be unsound; every policy incidence is replayed below.
    policies = analysis.get("projection_policies", [])
    diagnostic_scope = {"scope": "source"}
    if not policies:
        if analysis.get("discovery_complete"): fail("no coherent projection policies")
        else: unknown("incomplete discovery produced no coherent projection incumbent")
    if len({policy.get("id") for policy in policies}) != len(policies): fail("duplicate projection policy IDs")
    option_index = {cell: {option.get("id"): option for option in options} for cell, options in checked_options.items()}
    for policy in policies:
        diagnostic_scope = {"scope": "policy", "policy": policy.get("id")}
        choices = policy.get("option_by_cell", {})
        if set(choices) != set(cells): fail("projection policy cell coverage", policy=policy.get("id"))
        for cell, option in choices.items():
            if option not in option_index.get(cell, {}): fail("projection policy refers to missing option", policy=policy.get("id"), cell=cell)
    relation_dependencies = {}
    for relation in normalized.get("relations", []):
        diagnostic_scope = {"scope": "source", "relation": relation.get("id")}
        source = relation.get("source")
        if source not in cells:
            fail("relation source is not admitted", relation=relation.get("id")); continue
        for target in relation.get("targets", []):
            if target not in cells: fail("relation target is not admitted", relation=relation.get("id"), target=target)
        for policy in policies:
            diagnostic_scope = {"scope": "policy", "policy": policy.get("id"), "relation": relation.get("id")}
            choices = policy.get("option_by_cell", {})
            source_option = option_index.get(source, {}).get(choices.get(source))
            if source_option is None: continue
            source_cell = cells[source]
            stops = {name: projection.get("target", state)
                     for state, projection in source_option.get("state_projection", {}).items() if projection.get("kind") == "characteristic"
                     for name, provider in source_cell.get("functions", {}).items() if provider == projection.get("callback")}
            static = _static(source_cell, domains)
            static.update(source_option.get("constants", {}))
            available = set(source_option.get("states", {})) | set(source_option.get("actions", {}))
            for action, embedding in source_option.get("action_reconstruction", {}).items():
                if embedding.get("kind") == "constant": static[action] = embedding.get("value")
            phases = ("solve", "simulation") if relation.get("phase", "both") == "both" else (relation["phase"],)
            def relation_dependency(callback, phase):
                # Frozen local objects, not producer IDs or producer verdicts.
                key = (source_cell['id'], id(source_option), callback, phase)
                if key in relation_dependencies:
                    return relation_dependencies[key]
                providers = _providers(source_cell, domains, phase)
                phase_stops = dict(stops)
                for name, provider in providers.items():
                    if isinstance(provider, dict): continue
                    for projection in source_option.get("state_projection", {}).values():
                        if projection.get("kind") == "characteristic" and (provider == projection.get("callback") or callbacks.get(provider, {}).get("identity") == callbacks.get(projection.get("callback"), {}).get("identity")):
                            phase_stops[name] = projection["target"]
                answer = SourceProof(callbacks, providers, _codes(source_cell, domains, static), phase_stops, captures).callback(callback)
                axes = set(source_cell.get("states", {})) | set(source_cell.get("actions", {}))
                required = {name[:-5] if name.endswith("_code") and name[:-5] in axes else name for name in answer.reads}
                missing = required & axes - available - set(static)
                relation_dependencies[key] = missing
                return missing
            for callback in relation.get("callbacks", []):
                for phase in phases:
                    missing = relation_dependency(callback, phase)
                    if missing: fail("phase/joint law does not factor", relation=relation.get("id"), phase=phase, states=sorted(missing))
            for target in relation.get("targets", []):
                target_option = option_index.get(target, {}).get(choices.get(target))
                for target_option in ([] if target_option is None else [target_option]):
                    for state, target_projection in target_option.get("state_projection", {}).items():
                        if "produces" in relation and state not in relation["produces"]: continue
                        if target_projection.get("kind") == "drop": continue
                        update = relation.get("state_updates", {}).get(state)
                        source_projection = source_option.get("state_projection", {}).get(state, {})
                        if _identity_update(update, state) or (isinstance(update, str) and update in callbacks and _identity_callback(callbacks[update], state, captures)):
                            compatible = source_projection.get("kind") == "identity" or (
                                source_projection.get("kind") == target_projection.get("kind") == "characteristic"
                                and source_projection.get("callback") == target_projection.get("callback")
                                and source_projection.get("values") == target_projection.get("values"))
                            if not compatible: fail("future projected identity law loses history", relation=relation.get("id"), state=state)
                        elif isinstance(update, str) and update in callbacks:
                            for phase in phases:
                                missing = relation_dependency(update, phase)
                                if missing: fail("future update depends on removed detail", relation=relation.get("id"), state=state, phase=phase, missing=sorted(missing))
                        elif relation.get("kind") in {"value_read", "gate", "exit", "fallback", "reporting"}:
                            if state in cells.get(target, {}).get("states", {}) and state not in available:
                                fail("same-period/outside consumer loses state", relation=relation.get("id"), state=state)
                        else:
                            unknown("unproved full projected state law", relation=relation.get("id"), state=state)
                        checks += 1
    diagnostic_scope = {"scope": "family"}
    if check_coverage:
        policy_coverage = _policy_coverage(normalized, analysis, fail, unknown, captures)
    else:
        policy_coverage = False
        unknown("projection-family completeness was not checked in this local safety pass")
    return {"status": "FAIL" if errors else ("UNRESOLVED" if unresolved else "PASS"),
            "errors": errors, "unresolved": unresolved, "checks": checks,
            "source_sha256": _digest(original), "normalized_sha256": _digest(normalized),
            "guarantee": ("original declarations, bound callback factorization, full typed "
                          "relation preservation and " + (
                              "every coherent projection policy in the independently checked finite family"
                              if policy_coverage else
                              "supplied coherent projection policies only; full-family coverage is not established")),
            "dense_cartesian_rows_evaluated": 0,
            "projection_family_complete_verified": policy_coverage,
            "trusted_boundary": ["supplied original declaration/capture", "captured actual-object primitive identity", "pure deterministic scalar callback contract", "original joint law retained as one relation"]}


def _policy_coverage(declaration: dict, analysis: dict, fail, unknown, captures=None) -> bool:
    """Independent closure/CSP replay of categorical policy completeness.

    This derives live dependencies and eligible maps from original callback AST,
    not the producer's candidate list. A Boolean constraint search propagates
    implications in both directions, so a 100-date chain needs 101 policies,
    rather than materializing 2**100 assignments. Different existing maps of the
    same original axis are alternatives in this declared finite family.
    """
    captures = captures or _CallableRecords(declaration.get("callable_records", {}))
    cells = {cell["id"]: cell for cell in declaration["cells"]}
    callbacks, domains = declaration["callbacks"], declaration["domains"]
    relations = [relation for relation in declaration["relations"] if relation.get("support") != "zero"]
    def phases(record): return ("solve", "simulation") if record.get("phase", "both") == "both" else (record["phase"],)
    static_by_cell = {cell_id: _static(cell, domains) for cell_id, cell in cells.items()}
    expected_forced = {}
    for cell_id, cell in cells.items():
        expected_forced[cell_id] = {}
        constraints = [consumer for consumer in cell.get("consumers", []) if consumer.get("kind") in {"constraint", "feasible", "feasibility"}]
        decision_phases = cell.get("decision_phases", ("solve", "simulation"))
        for axis, domain_id in cell["actions"].items():
            values = domains[domain_id].get("values")
            if not isinstance(values, list) or not values or len(values) > 100_000: continue
            possible = []
            for value in values:
                static = _codes(cell, domains, {**static_by_cell[cell_id], axis: value})
                excluded_by_phase = []
                for decision_phase in decision_phases:
                    ruled_out = False
                    for constraint in constraints:
                        for phase in _feasibility_channels(constraint, decision_phase):
                            providers = _providers(cell, domains, phase)
                            answer = SourceProof(callbacks, providers, static, callable_records=captures).callback(constraint["callback"])
                            ruled_out |= answer.proved and answer.value is not _UNKNOWN and not answer.value
                    excluded_by_phase.append(ruled_out)
                if not all(excluded_by_phase): possible.append(value)
            if len(possible) == 1:
                expected_forced[cell_id][axis] = possible[0]
        static_by_cell[cell_id].update(expected_forced[cell_id])
        if analysis.get("discovery_complete"):
            for axis, value in expected_forced[cell_id].items():
                for option in analysis.get("interfaces", {}).get(cell_id, {}).get("options", []):
                    reconstruction = option.get("action_reconstruction", {}).get(axis, {})
                    if reconstruction.get("kind") != "constant" or _digest(reconstruction.get("value")) != _digest(value):
                        fail("complete discovery omitted a provably forced original choice", cell=cell_id, action=axis, value=value)
    def original_dependency(callback, cell, phase, stop_id=None):
        providers = _providers(cell, domains, phase)
        stops = {name: "@retained_characteristic" for name, provider in providers.items()
                 if not isinstance(provider, dict) and stop_id is not None and (provider == stop_id or callbacks.get(provider, {}).get("identity") == callbacks.get(stop_id, {}).get("identity"))}
        proof = SourceProof(callbacks, providers, _codes(cell, domains, static_by_cell[cell["id"]]), stops, captures)
        value = proof.callback(callback)
        axes = set(cell["states"]) | set(cell["actions"])
        original_callback = callbacks.get(callback, {}).get('original', {})
        if not isinstance(original_callback, Mapping):
            return {name[:-5] if name.endswith('_code') and name[:-5] in axes else name for name in value.reads}
        if original_callback.get('type') == '_lcm.certainty_equivalent.LinearExpectation':
            # This pinned built-in reduces supplied continuation values. State
            # dependence enters through the separately checked transition law,
            # not through a hidden current-state argument of the reducer.
            return set()
        if _closed_native_constant(original_callback, captures):
            # The route's ORIGINAL bound value is immutable and the callback
            # receives no current coordinates. In particular, a last-decision
            # constant route into a state-free terminal does not keep an
            # otherwise irrelevant history/rate coordinate alive.
            return set()
        if not value.proved and (original_callback.get('kind') not in {'callable', 'primitive'}
                                or original_callback.get('variadic')
                                or _original_inputs_unresolved(original_callback, captures)):
            # An opaque native callback is not evidence of independence. The
            # admitted conservative family retains its complete local domain;
            # its unresolved source obligation remains in the check result.
            return axes
        return {name[:-5] if name.endswith("_code") and name[:-5] in axes else name for name in value.reads}
    dependency_cache = {}
    current_cache = {}
    relations_by_source = defaultdict(list)
    for relation in relations: relations_by_source[relation['source']].append(relation)
    def dependency(callback, cell, phase, stop_id=None):
        # Static bindings are finalized above, and the supplied source DAG is
        # frozen for this check. Cache only this checker's own source result.
        key = (callback, cell['id'], phase, stop_id)
        if key not in dependency_cache:
            dependency_cache[key] = frozenset(original_dependency(callback, cell, phase, stop_id))
        return dependency_cache[key]
    def current(cell, stop_id=None):
        key = (cell['id'], stop_id)
        if key in current_cache: return set(current_cache[key])
        names = set()
        for consumer in cell.get("consumers", []):
            for phase in phases(consumer): names.update(dependency(consumer["callback"], cell, phase, stop_id))
        for relation in relations_by_source[cell['id']]:
            for callback in relation.get("callbacks", []):
                for phase in phases(relation): names.update(dependency(callback, cell, phase, stop_id))
        current_cache[key] = frozenset(names)
        return names
    def is_identity(update, axis):
        return _identity_update(update, axis) or (isinstance(update, str) and update in callbacks and _identity_callback(callbacks[update], axis, captures))
    def update_reads(relation, axis, stop_id=None):
        cell = cells[relation["source"]]
        update = relation.get("state_updates", {}).get(axis)
        if isinstance(update, str) and update in cell["states"]: return {update}
        if isinstance(update, dict) and update.get("kind") == "identity": return {update.get("source", update.get("state", axis))}
        if isinstance(update, dict) and update.get("kind") == "constant": return set()
        if isinstance(update, dict): update = update.get("callback")
        if isinstance(update, str) and update in callbacks:
            return set().union(*(dependency(update, cell, phase, stop_id) for phase in phases(relation)))
        return set(cell["states"])
    live = {cell_id: (current(cell) & set(cell["states"])) - set(static_by_cell[cell_id]) for cell_id, cell in cells.items()}
    changed = True
    while changed:
        changed = False
        for relation in relations:
            source = relation["source"]
            for target in relation.get("targets", []):
                for axis in tuple(live[target]):
                    if "produces" in relation and axis not in relation["produces"]: continue
                    needed = (update_reads(relation, axis) & set(cells[source]["states"])) - set(static_by_cell[source])
                    if needed - live[source]: live[source].update(needed); changed = True
    for cell_id, expected in live.items():
        claimed = set(analysis.get("dependency_closure", {}).get(cell_id, []))
        if not expected <= claimed: fail("original future dependency closure omitted", cell=cell_id, missing=sorted(expected - claimed))
    choices_by_axis = defaultdict(list)
    proof_limited = False
    for characteristic_id, characteristic in sorted(declaration.get("characteristics", {}).items()):
        axis, did, callback = characteristic["state"], characteristic["domain"], characteristic["callback"]
        values = domains.get(did, {}).get("values")
        if not isinstance(values, list) or not values or len(values) > 100_000: continue
        image = []
        for index, value in enumerate(values):
            static = {axis: value}
            codes = domains[did].get("codes", [])
            if len(codes) == len(values):
                static[characteristic.get("code_argument", axis + "_code")] = codes[index]
            answer = SourceProof(callbacks, static=static, callable_records=captures).callback(callback)
            if not answer.proved or answer.value is _UNKNOWN or answer.reads: break
            image.append(answer.value)
        if len(image) != len(values) or len({_value_key(value) for value in image}) >= len(values): continue
        eligible = {cell_id for cell_id, cell in cells.items() if cell["states"].get(axis) == did and axis in live[cell_id] and axis not in current(cell, callback)}
        changed = True
        while changed:
            changed = False
            for relation in relations:
                source = relation["source"]
                if source not in eligible: continue
                for target in relation.get("targets", []):
                    for target_axis in live[target]:
                        if "produces" in relation and target_axis not in relation["produces"]: continue
                        update = relation.get("state_updates", {}).get(target_axis)
                        safe = (target in eligible) if target_axis == axis and is_identity(update, axis) else axis not in update_reads(relation, target_axis, callback)
                        if not safe: eligible.remove(source); changed = True; break
                    if source not in eligible: break
        if not eligible: continue
        if len(eligible) > 750:
            unknown("independent projection-family recursion budget exhausted", eligible_cells=len(eligible)); return False
        successors = {cell: set() for cell in eligible}
        for relation in relations:
            source = relation["source"]
            if source in eligible and ("produces" not in relation or axis in relation["produces"]) and is_identity(relation.get("state_updates", {}).get(axis), axis):
                successors[source].update(target for target in relation.get("targets", []) if target in eligible and axis in live[target])
        predecessors = {cell: {source for source, targets in successors.items() if cell in targets} for cell in eligible}
        regions = []
        def assign(selected, rejected):
            nonlocal proof_limited
            if len(regions) > 20_000:
                proof_limited = True; return
            queue = list(selected)
            selected = set(selected); rejected = set(rejected)
            while queue:
                for target in successors[queue.pop()]:
                    if target not in selected: selected.add(target); queue.append(target)
            queue = list(rejected)
            while queue:
                for source in predecessors[queue.pop()]:
                    if source not in rejected: rejected.add(source); queue.append(source)
            if selected & rejected: return
            unassigned = eligible - selected - rejected
            if not unassigned:
                if selected: regions.append(frozenset(selected))
                return
            cell = min(unassigned)
            assign(selected, rejected | {cell})
            assign(selected | {cell}, rejected)
        assign(set(), set())
        choices_by_axis[axis].extend((callback, region) for region in regions)
    if proof_limited:
        unknown("independent projection-family coverage budget exhausted"); return False
    def policy_key(choices):
        result = []
        for cell_id in sorted(cells):
            states = []
            for state in sorted(cells[cell_id]["states"]):
                selected = next((callback for axis, (callback, region) in choices.items() if axis == state and cell_id in region), None)
                states.append((state, "characteristic", selected) if selected else (state, "identity" if state in live[cell_id] else "drop"))
            result.append((cell_id, states))
        return _digest(result)
    expected = set()
    axes = sorted(choices_by_axis)
    for combination in product(*([[None] + choices_by_axis[axis] for axis in axes])):
        selected = {axis: choice for axis, choice in zip(axes, combination) if choice is not None}
        expected.add(policy_key(selected))
        if len(expected) > 20_000:
            unknown("independent projection-family coverage budget exhausted"); return False
    actual = set()
    options = {cell: {option["id"]: option for option in record["options"]} for cell, record in analysis["interfaces"].items()}
    for policy in analysis.get("projection_policies", []):
        key = []
        for cell_id in sorted(cells):
            option = options[cell_id].get(policy.get("option_by_cell", {}).get(cell_id), {})
            states = []
            for state in sorted(cells[cell_id]["states"]):
                projection = option.get("state_projection", {}).get(state, {})
                kind = projection.get("kind")
                states.append((state, kind, projection.get("callback")) if kind == "characteristic" else (state, kind))
            key.append((cell_id, states))
        actual.add(_digest(key))
    if actual - expected:
        if analysis.get("discovery_complete"):
            fail("projection policy is outside independently derived family", extra=len(actual - expected), dependency_disagreement=[{"cell": cid, "independent": sorted(live[cid]), "producer": analysis.get("dependency_closure", {}).get(cid, [])} for cid in cells if set(live[cid]) != set(analysis.get("dependency_closure", {}).get(cid, []))])
        else:
            unknown("incomplete discovery retains conservative representation policies", count=len(actual - expected))
    if analysis.get("discovery_complete") is True:
        if expected != actual:
            fail("complete claim omits coherent categorical projection policies", missing=len(expected - actual)); return False
        return True
    unknown("categorical projection discovery is explicitly incomplete", proved_policies=len(actual), full_family_policies=len(expected))
    return False


def _check_forced(action: str, selected: Any, cell: dict, callbacks: dict, domains: dict, fail, unknown, captures=None) -> None:
    constraints = [consumer for consumer in cell.get("consumers", []) if consumer.get("kind") in {"feasible", "constraint", "feasibility"}]
    domain = domains.get(cell.get("actions", {}).get(action), {})
    values = domain.get("values")
    if not isinstance(values, list) or len(values) > 100_000:
        unknown("forced-action categorical domain unavailable", cell=cell["id"], action=action); return
    if not constraints:
        if len(values) != 1: fail("non-singleton action removed without feasibility proof", cell=cell["id"], action=action)
        return
    decision_phases = cell.get("decision_phases", ("solve", "simulation"))
    for index, value in enumerate(values):
        static = _codes(cell, domains, {**_static(cell, domains), action: value})
        exclusions = []
        for decision_phase in decision_phases:
            results = [SourceProof(callbacks, _providers(cell, domains, phase), static, callable_records=captures).callback(constraint["callback"])
                       for constraint in constraints for phase in _feasibility_channels(constraint, decision_phase)]
            exclusions.append(any(result.proved and result.value is not _UNKNOWN and not result.value for result in results))
        if value != selected and not all(exclusions):
            fail("alternative action was not proved infeasible", cell=cell["id"], action=action, value=value)
        if value == selected and any(exclusions):
            fail("selected forced action is infeasible", cell=cell["id"], action=action, value=value)


def _check_function_interfaces(cell, option, callbacks, domains, static, fail, unknown, captures=None):
    interfaces = option.get("function_interfaces")
    if interfaces is None:
        unknown("reduced function interface plan absent", cell=cell["id"]); return
    projection = option.get("state_projection", {})
    for phase in ("solve", "simulation"):
        providers = cell.get("phase_functions", {}).get(phase, cell.get("functions", {}))
        records = interfaces.get(phase, {})
        if set(records) != set(providers): fail("reduced function-provider coverage", cell=cell["id"], phase=phase)
        # Reachability is computed on direct ORIGINAL formal dependencies; this
        # retains names of DAG providers instead of flattening to state leaves.
        roots = [consumer["callback"] for consumer in cell.get("consumers", []) if consumer.get("phase", "both") in (phase, "both")]
        reachable = set()
        pending = list(roots)
        direct_static = _codes(cell, domains, static)
        while pending:
            callback = pending.pop()
            if callback in reachable or callback not in callbacks: continue
            reachable.add(callback)
            if any(record.get("original_callback") == callback and record.get("implementation", {}).get("kind") in {"read_reduced_coordinate", "bound_constant"} for record in records.values()): continue
            answer = SourceProof(callbacks, static=direct_static, callable_records=captures).callback(callback)
            pending.extend(providers[name] for name in answer.reads if name in providers)
        for name, record in records.items():
            callback_id = providers.get(name)
            callback = callbacks.get(callback_id, {})
            if record.get("original_callback") != callback_id or record.get("source_identity") != callback.get("identity"):
                fail("reduced function source identity", cell=cell["id"], phase=phase, function=name)
            formals = callback.get("arguments", [])
            if record.get("original_arguments") != formals: fail("reduced function original signature", cell=cell["id"], function=name)
            kind = record.get("implementation", {}).get("kind")
            if kind == "omit_unconsumed_provider":
                if callback_id in reachable: fail("consumed original provider was omitted", cell=cell["id"], phase=phase, function=name)
                if record.get("arguments") != []: fail("omitted provider retains an interface", cell=cell["id"], function=name)
                continue
            replacement = next((item for item in projection.values() if item.get("kind") == "characteristic" and
                                (item.get("callback") == callback_id or callbacks.get(item.get("callback"), {}).get("identity") == callback.get("identity"))), None)
            if kind in {"read_reduced_coordinate", "bound_constant"}:
                if replacement is None:
                    fail("function replaced without an original characteristic witness", cell=cell["id"], function=name); continue
                target = replacement["target"]
                if kind == "read_reduced_coordinate":
                    if record["implementation"].get("coordinate") != target or record.get("arguments") != [target] or target not in option.get("states", {}): fail("characteristic function interface changed", cell=cell["id"], function=name)
                elif record.get("arguments") != [] or target not in static or _digest(record["implementation"].get("value")) != _digest(static[target]):
                    fail("characteristic function constant changed", cell=cell["id"], function=name)
                continue
            if kind != "source_bound_partial_application":
                fail("unknown reduced function implementation", cell=cell["id"], function=name); continue
            arguments = record.get("arguments", [])
            bound = record.get("bound_arguments", {})
            unused = record.get("unused_arguments", [])
            categories = list(arguments) + list(bound) + list(unused)
            if sorted(categories) != sorted(formals) or len(categories) != len(set(categories)):
                fail("reduced formal arguments do not partition original signature", cell=cell["id"], function=name)
            proved_static = dict(direct_static)
            for argument, value in bound.items():
                expected = proved_static.get(argument, _UNKNOWN)
                if expected is _UNKNOWN and argument in providers:
                    answer = SourceProof(callbacks, providers, proved_static, callable_records=captures).callback(providers[argument])
                    if answer.proved and not answer.reads: expected = answer.value
                if expected is _UNKNOWN and argument in callback.get("bindings", {}):
                    try:
                        expected = (captures.binding_value(callback['bindings'][argument]) if captures is not None
                                    else _decode(callback['bindings'][argument]))
                    except (ValueError, TypeError, KeyError, RecursionError):
                        expected = _UNKNOWN
                # The admitted scalar language distinguishes Boolean/string
                # values but identifies exact numeric images (e.g. 1 and 1.0).
                # This is source-derived equality, never a tolerance or a
                # guessed wrapper binding.
                if expected is _UNKNOWN or _value_key(expected) != _value_key(value): fail("unproved bound original argument", cell=cell["id"], function=name, argument=argument)
                else: proved_static[argument] = expected
            answer = SourceProof(callbacks, static=proved_static, callable_records=captures).callback(callback_id)
            if set(unused) & answer.reads: fail("live original argument removed from function interface", cell=cell["id"], function=name, arguments=sorted(set(unused) & answer.reads))


def _original_schedule(source, dates):
    """Interpret the finite public schedule vocabulary independently.

    Only date membership, regime labels and typed references are visited. No
    transition probability, payoff, action row, or model callback is evaluated.
    """
    def fields(value): return value.get("fields", {}) if isinstance(value, dict) else {}
    def short(value):
        tag = value.get("type") if isinstance(value, dict) else None
        return tag.split(".")[-1] if isinstance(tag, str) else ""
    def items(value):
        if isinstance(value, dict) and value.get("kind") == "mapping": return value.get("items", [])
        return list(value.items()) if isinstance(value, dict) else []
    def rational(value): return Fraction(str(_decode(value)))
    exact_dates = list(map(rational, dates))
    clock = "age" if source.get("ages") is not None else "period"
    selector_tests = {}
    def selects(selector, period):
        key = id(selector)
        if key not in selector_tests:
            kind, data = short(selector), fields(selector)
            if kind in {"AgeRange", "PeriodRange"}:
                lower = None if data.get("start") is None else rational(data["start"])
                upper = None if data.get("exclusive_stop") is None else rational(data["exclusive_stop"])
                test = lambda coordinate: (lower is None or coordinate >= lower) and (upper is None or coordinate < upper)
            elif kind == "Periods":
                values = frozenset(_decode(data.get("values", [])))
                test = lambda coordinate: coordinate in values
            elif kind == "range":
                original_range = range(data.get("start", 0), data["stop"], data.get("step", 1))
                test = lambda coordinate: coordinate.denominator == 1 and int(coordinate) in original_range
            else:
                values = frozenset(rational(value) for value in (selector if isinstance(selector, list) else [selector]))
                test = lambda coordinate: coordinate in values
            selector_tests[key] = (selector, kind in {"PeriodRange", "Periods"}, test)
        _, period_clock, test = selector_tests[key]
        return test(Fraction(period) if period_clock else exact_dates[period])
    def phase(value, side):
        while short(value) == "Phased": value = fields(value).get("solve" if side == "solve" else "simulate")
        return value
    def law_at(value, side, period):
        value = phase(value, side)
        if short(value) not in {"ByAge", "ByPeriod"}: return value
        data = fields(value)
        until = data.get("until", data.get("_until"))
        if until:
            u = fields(until) or until
            boundary = rational(u.get("stop_age_exclusive", u.get("stop_period_exclusive")))
            coordinates = [Fraction(index) for index in range(len(dates))] if short(value) == "ByPeriod" else exact_dates
            if boundary not in coordinates: raise ValueError("Until boundary is not an original date")
            stop = coordinates.index(boundary)
            start_value = u.get("start_age_inclusive", u.get("start_period_inclusive"))
            start = 0 if start_value is None else coordinates.index(rational(start_value))
            if stop == 0 or start >= stop: raise ValueError("Invalid original Until interval")
            if not start <= period < stop: return None
            return phase(u.get("then") if period == stop - 1 else u.get("law"), side)
        cases = data.get("cases", data.get("_cases", []))
        cases = items(cases) if isinstance(cases, dict) else cases
        matching = [law for selector, law in cases if selects(selector, period)]
        if len(matching) > 1: raise ValueError("Overlapping original date cases")
        return phase(matching[0] if matching else data.get("default", data.get("_default")), side)
    starts = source.get("initial_nodes", [])
    admitted_starts = set()
    if isinstance(starts, list):
        for date, regime in starts: admitted_starts.add((exact_dates.index(rational(date)), regime))
    else:
        data = fields(starts) or starts
        if "by_age" in data or "by_period" in data:
            supplied = data.get("by_age") if data.get("by_age") is not None else data.get("by_period")
        else: supplied = data
        for selector, regimes in items(supplied):
            names = [regimes] if isinstance(regimes, str) else regimes
            for period in range(len(dates)):
                if selects(selector, period): admitted_starts.update((period, name) for name in names)
    regimes = source.get("regimes", source.get("user_regimes", {}))
    admitted = set(admitted_starts)
    pending = list(admitted)
    edges = set()
    gates = []
    original_laws = {}
    while pending:
        period, regime = pending.pop()
        if regime not in regimes: raise ValueError("Original schedule names missing regime")
        additions = set()
        for side in ("solve", "simulation"):
            # Independent physical reachability follows only scheduled moves
            # and explicit settlement routes. Current value/policy reads are
            # replayed by the separate demand registry in _check_capture.
            if period == len(dates) - 1: continue
            phase_edges = phase(source.get("edges", {}), side)
            entry = phase_edges.get(regime) if isinstance(phase_edges, dict) else None
            if entry is None: continue
            if short(entry) == "Transition":
                data = fields(entry)
                original_laws[(period, regime, side)] = law_at(data.get("law"), side, period)
                if data.get("targets") is not None:
                    targets = [target for target, selector in items(data["targets"]) if selects(selector, period)]
                else:
                    law = law_at(data.get("law"), side, period)
                    if law is None: targets = []
                    elif isinstance(law, str): targets = [law]
                    elif isinstance(law, dict) and "kind" not in law: targets = list(law)
                    else: raise ValueError("Original numerical routing law lacks declared targets")
                gate_map = _independent_gate_map(data, "gates", "arrivals", "participation")
            elif isinstance(entry, dict) and "kind" not in entry:
                targets = [target for target, selector in entry.items() if selects(selector, period)]
                gate_map = {}
            else: raise ValueError("Unsupported original graph schedule declaration")
            for target in targets:
                edges.add((period, regime, target, side))
                additions.add((period + 1, target))
                if target in gate_map:
                    gate = phase(gate_map[target], side)
                    gates.append((period, regime, target, side, gate))
                    gate_data = fields(gate)
                    routes = gate_data.get("legs", gate_data.get("routes", {}))
                    for route in routes.values():
                        fallback = phase(fields(phase(route, side)).get("fallback"), side)
                        if fallback is not None:
                            if short(fallback) != "ProjectedRegimeValue":
                                raise ValueError("Original fallback has no supported physical-settlement contract")
                            additions.add((period + 1, fields(fallback)["regime"]))
        for addition in additions - admitted:
            admitted.add(addition); pending.append(addition)
    return admitted, edges, admitted_starts, gates, original_laws


def _independent_gate_map(payload, *keys):
    pairs = [(target, value) for key in keys for target, value in payload.get(key, {}).items()]
    if len({target for target, _ in pairs}) != len(pairs):
        raise ValueError("Original target has conflicting participation/gate declarations")
    return dict(pairs)


def _check_current_type_sources(original, fail):
    from .current_api_contract import matches_authority
    pending = [original]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            tag = value.get("type", "")
            if value.get("kind") not in ("dataclass", "object") or not isinstance(tag, str):
                tag = ""
            label = tag.split(".")[-1]
            required = label in ("ParticipationRead", "ParticipationCheck") or (label == "Transition" and bool(value.get("fields", {}).get("participation")))
            if required:
                if not matches_authority(name=tag, observed=value.get("canonical_type_source")):
                    fail("original participation class authority changed", declaration_type=tag)
            pending.extend(value.values())


def _check_capture(original: dict, normalized: dict, fail, unknown) -> None:
    _check_current_type_sources(original, fail)
    """Independent source-pointer and graph-admission replay for native captures.

    The declaration ingestor supplies source pointers, not a replacement oracle.
    Every pointer is dereferenced against the untouched original object here.
    More detailed unsupported declaration semantics remain explicit obligations.
    """
    def fields(value): return value.get("fields", {}) if isinstance(value, dict) and value.get("kind") in {"object", "dataclass"} else (value if isinstance(value, dict) else {})
    def short(value):
        tag = value.get("type") if isinstance(value, dict) else None
        return tag.split(".")[-1] if isinstance(tag, str) else ""
    payload_digests = {}
    def payload_digest(value):
        # Independent transaction-local cache; no producer hashes or cached
        # conclusions enter the original-source equality checks.
        key = id(value)
        if key not in payload_digests:
            payload_digests[key] = (value, _digest(value))
        return payload_digests[key][1]
    def root_keys(rows):
        return sorted(_digest((kind, phase, payload_digest(value))) for kind, phase, value in rows)
    def selected(value, phase, period):
        if short(value) == "Phased": return selected(fields(value).get("solve" if phase == "solve" else "simulate"), phase, period)
        if short(value) in {"AgeSpecializedFunction", "PeriodSpecializedFunction"}:
            matches = [item["value"] for item in value.get("resolved_by_period", []) if item.get("period") == period]
            return matches[0] if len(matches) == 1 else value
        if short(value) == "_PeriodDispatch":
            data = fields(value)
            cases, indexes = data.get("cases", []), data.get("case_by_period", [])
            if period < len(indexes) and type(indexes[period]) is int and 0 <= indexes[period] < len(cases): return selected(cases[indexes[period]], phase, period)
        return value
    source = original.get("graph_snapshot", original)
    if normalized.get("callable_records", {}) != source.get("callable_records", {}):
        fail("original shared callable registry changed")
    graph = fields(source.get("graph"))
    reach = fields(graph.get("reachability", source.get("reachability")))
    time = _decode(fields(source.get("ages")).get("exact_values"))
    if not isinstance(time, list): time = list(range(source.get("n_periods", 0)))
    def number(value):
        value = _decode(value)
        return str(Fraction(str(value)))
    positions = {number(value): index for index, value in enumerate(time)}
    pairs = list(reach.get("nodes", graph.get("nodes", source.get("admissions", []))))
    for pair in reach.get("visited_nodes", []):
        if pair not in pairs: pairs.append(pair)
    original_schedule = None
    if not pairs and "edges" in source:
        try: original_schedule = _original_schedule(source, time)
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc: unknown("independent original schedule obligation", detail=str(exc))
    expected = original_schedule[0] if original_schedule else {(positions.get(number(date)), regime) for date, regime in pairs}
    actual = {(cell.get("period"), cell.get("regime")) for cell in normalized.get("cells", [])}
    if expected != actual: fail("original sparse date/regime admission changed")
    cells = {cell["id"]: cell for cell in normalized.get("cells", [])}
    by_pair = {(cell["period"], cell["regime"]): cell for cell in cells.values()}
    callbacks = normalized.get("callbacks", {})
    domains = normalized.get("domains", {})
    coordinate_kind = graph.get("coordinate_kind", "age" if source.get("ages") is not None else "period")
    contract = source.get("semantic_contract", {})
    if normalized.get("original_contract", contract) != contract:
        # A separately supplied contract must itself be an independently
        # supplied input to establish its identity. Never infer it from roles.
        unknown("external production contract is a separate trusted input")
        contract = normalized.get("original_contract", {})
    for domain_id, domain in domains.items():
        try: raw = _pointer(original, domain.get("source_path", ""))
        except (KeyError, IndexError, ValueError, TypeError): raw = None
        if raw is None:
            unknown("domain source pointer unavailable", domain=domain_id); continue
        if domain.get("original") != raw: fail("original domain payload changed", domain=domain_id)
        raw_fields = fields(raw)
        raw_values, raw_codes = _decode(raw_fields.get("values")), _decode(raw_fields.get("codes"))
        from .structural_context import reference_info
        for label, value in (('values', raw_values), ('codes', raw_codes)):
            ref = value.get('structural_ref', value) if isinstance(value, dict) else value
            info = reference_info(ref)
            if info is not None and domain.get('kind') == 'categorical':
                if domain.get('structural_' + label) != ref:
                    fail('original structural category payload changed', domain=domain_id)
                count = info.get('length')
                if info['type'] == 'array':
                    if len(info['shape']) != 1: fail('categorical structural array is not one-dimensional', domain=domain_id)
                    else: count = info['shape'][0]
                if type(count) is not int or domain.get('size') != count:
                    fail('structural category cardinality changed', domain=domain_id)
        if isinstance(raw_values, list) and domain.get("values") != raw_values: fail("decoded original grid values changed", domain=domain_id)
        if isinstance(raw_codes, list) and domain.get("codes") != raw_codes: fail("original category codes changed", domain=domain_id)
        cardinality = len(raw_values) if isinstance(raw_values, list) else _decode(raw_fields.get("n_points"))
        if cardinality is None and isinstance(_decode(raw_fields.get("points")), list): cardinality = len(_decode(raw_fields["points"]))
        if type(cardinality) is int and domain.get("size") != cardinality: fail("production grid cardinality changed", domain=domain_id)
        if 'SpecializedGrid' in short(raw):
            cardinality=raw.get('resolved_cardinality')
            if type(cardinality) is int and cardinality>0:
                if domain.get('size')!=cardinality:fail('specialized native grid cardinality changed',domain=domain_id)
                if raw.get('resolved_cardinality_source'):
                    specialization=domain.get('specialization',{})
                    if (specialization.get('period') is not None
                            or specialization.get('identity_scope')!='original-shape-invariant-program'
                            or specialization.get('factory')!=raw_fields.get('build')):
                        fail('specialized grid program identity changed',domain=domain_id)
        if short(raw) == "Phased":
            imputed = isinstance(raw_fields.get("solve"), dict) and raw_fields["solve"].get("kind") == "callable"
            if domain.get("solve_axis") != (not imputed): fail("original carried/solve domain distinction changed", domain=domain_id)
            for phase, original_field in (("simulation", "simulate"), ("solve", "solve")):
                phase_id = domain.get("phase_domains", {}).get(phase)
                if phase == "solve" and imputed:
                    if phase_id is not None or callbacks.get(domain.get("solve_imputation"), {}).get("original") != raw_fields["solve"]:
                        fail("original carried-state imputation changed", domain=domain_id)
                elif domains.get(phase_id, {}).get("original") != raw_fields.get(original_field): fail("original phase grid changed", domain=domain_id, phase=phase)
    for callback_id, callback in normalized.get("callbacks", {}).items():
        path = callback.get("source_path")
        try: callback_source = _pointer(original, path) if isinstance(path, str) else _UNKNOWN
        except (KeyError, IndexError, ValueError, TypeError): callback_source = _UNKNOWN
        if callback_source is _UNKNOWN:
            unknown("callback source pointer unavailable", callback=callback_id, path=path); continue
        captured = callback.get("original", {})
        if captured != callback_source: fail("captured callback differs from original source pointer", callback=callback_id)
        if not isinstance(callback_source, dict): continue
        for key in ("source", "arguments", "bindings", "call_targets"):
            if key in callback_source and key in callback and callback[key] != callback_source[key]: fail("normalized callback source field changed", callback=callback_id, field=key)
    original_regimes = {key: dict(fields(value)) for key, value in source.get("user_regimes", source.get("regimes", {})).items()}
    for declaration in original_regimes.values():
        for key in ("states", "actions", "functions", "constraints", "state_transitions", "derived_categoricals"):
            declaration[key] = {name: value for name, value in {**source.get(key, {}), **declaration.get(key, {})}.items() if value is not None}
        for key in ("koopmans_aggregator", "certainty_equivalent", "pareto_objective"):
            if declaration.get(key) is None and source.get(key) is not None: declaration[key] = source[key]
        declaration["output_only_functions"] = list(dict.fromkeys([*source.get("output_only_functions", []), *declaration.get("output_only_functions", [])]))
    starts = graph.get("initial_nodes", source.get("initial_nodes", []))
    start_pairs = original_schedule[2] if original_schedule else ({(positions.get(number(date)), regime) for date, regime in starts} if isinstance(starts, list) else None)
    for cell in cells.values():
        declaration = original_regimes.get(cell.get("regime"), {})
        period, regime = cell.get("period"), cell.get("regime")
        if type(period) is not int or not 0 <= period < len(time):
            fail("original period coordinate changed", cell=cell.get("id")); continue
        if coordinate_kind == "age":
            if cell.get("age") is None or number(cell["age"]) != number(time[period]): fail("biological age changed", cell=cell["id"])
        elif cell.get("age") is not None:
            fail("period counter was relabeled biological age", cell=cell["id"])
        if cell.get("role") != contract.get("roles", {}).get(regime, declaration.get("semantic_role")): fail("original role/information/discount contract changed", cell=cell["id"])
        if normalized.get("analysis_scope") == "pylcm-structural-graph/v1" and cell.get("decision_objective") != declaration.get("pareto_objective"):
            fail("original decision objective changed", cell=cell["id"])
        if cell.get("stage") != contract.get("stages", {}).get(regime, declaration.get("stage")): fail("original decision stage changed", cell=cell["id"])
        raw_regime = source.get("user_regimes", source.get("regimes", {})).get(regime, {})
        if cell.get("declaration_type") != raw_regime.get("type", "original-declaration-mapping"): fail("original declaration type changed", cell=cell["id"])
        structural_roles = {key: value for key, value in declaration.items() if key in {"liquid", "outer_continuous"} and value is not None}
        if cell.get("structural_roles", {}) != structural_roles: fail("original solver coordinate roles changed", cell=cell["id"])
        solver = payload_digest(declaration["solver"]) if declaration.get("solver") is not None else None
        if cell.get("solver_identity") != solver: fail("original solver identity changed", cell=cell["id"])
        if cell.get("report_tags", []) != contract.get("report_tags", {}).get(regime, []): fail("reporting tag contract changed", cell=cell["id"])
        if start_pairs is not None and cell.get("admissible_start") != ((period, regime) in start_pairs): fail("admissible original start changed", cell=cell["id"])
        if set(cell.get("states", {})) != set(declaration.get("states", {})): fail("original state axes omitted", cell=cell.get("id"))
        if set(cell.get("actions", {})) != set(declaration.get("actions", {})): fail("original action axes omitted", cell=cell.get("id"))
        for kind in ("states", "actions"):
            for axis, domain_id in cell.get(kind, {}).items():
                if domains.get(domain_id, {}).get("original") != declaration.get(kind, {}).get(axis): fail("cell domain is not its original declaration", cell=cell["id"], axis=axis)
        if set(cell.get("derived_categoricals", {})) != set(declaration.get("derived_categoricals", {})): fail("original derived category domain omitted", cell=cell["id"])
        for output, domain_id in cell.get("derived_categoricals", {}).items():
            if domains.get(domain_id, {}).get("original") != declaration.get("derived_categoricals", {}).get(output): fail("original derived category domain changed", cell=cell["id"], output=output)
        if cell.get("derived_code_providers", {}) != {output + "_code": output for output in declaration.get("derived_categoricals", {})}: fail("original derived category code provider changed", cell=cell["id"])
        expected_roots = []
        expected_shocks = []
        for phase in ("solve", "simulation"):
            registrations = selected(declaration.get('shocks', []), phase, period)
            if not isinstance(registrations, list) or any(not isinstance(name, str) for name in registrations):
                fail('original shock registrations unavailable', cell=cell['id'], phase=phase)
                registrations = []
            for name in registrations:
                original_group = source.get('shocks', {}).get(name)
                if short(selected(original_group, phase, period)) != 'ShockGroup':
                    fail('original registered shock group unavailable', cell=cell['id'], phase=phase, group=name)
                expected_roots.append(('shock_law', phase, selected(original_group, phase, period)))
                expected_shocks.append((name, phase, payload_digest(selected(original_group, phase, period))))
            providers = cell.get("phase_functions", {}).get(phase, cell.get("functions", {}))
            for name, value in declaration.get("functions", {}).items():
                record = callbacks.get(providers.get(name), {})
                if record.get("original") != selected(value, phase, period): fail("original derived-function provider changed", cell=cell["id"], phase=phase, function=name)
            for axis, value in declaration.get("state_transitions", {}).items():
                if isinstance(value,dict) and not value:continue
                record = callbacks.get(providers.get("next_" + axis), {})
                if record.get("original") != selected(value, phase, period): fail("original next-state provider changed", cell=cell["id"], phase=phase, state=axis)
            if phase == "solve":
                if declaration.get("certainty_equivalent") is not None:
                    expected_roots.append(("certainty_equivalent", "solve", selected(declaration["certainty_equivalent"], "solve", period)))
                continue
            for kind, mapping in [("reward", {"utility": declaration.get("functions", {}).get("utility")}),
                                  ("feasible", declaration.get("constraints", {})),
                                  ("value_constraint", declaration.get("value_constraints", {}))]:
                for value in mapping.values():
                    if value is None: continue
                    value = selected(value, phase, period)
                    actual_kind = kind
                    if short(value) == "ValueDependentConstraint":
                        actual_kind, value = "value_constraint", fields(value).get("predicate")
                    expected_roots.append((actual_kind, phase, value))
            for name in declaration.get("output_only_functions", []):
                if name in declaration.get("functions", {}): expected_roots.append(("reporting", phase, selected(declaration["functions"][name], phase, period)))
            for name in ("koopmans_aggregator", "pareto_objective"):
                if declaration.get(name) is not None: expected_roots.append((name, phase, selected(declaration[name], phase, period)))
            if "discount_factor" in declaration.get("functions", {}): expected_roots.append(("discount", phase, selected(declaration["functions"]["discount_factor"], phase, period)))
        for value in structural_roles.values(): expected_roots.append(("structural_margin", "solve", value))
        supported_slots = {"states", "actions", "functions", "constraints", "state_transitions", "joint_transitions", "derived_categoricals",
                           "output_only_functions", "koopmans_aggregator", "certainty_equivalent", "pareto_objective", "value_constraints",
                           "same_period_refs", "exits", "description", "solver", "taste_shocks", "shocks", "stakeholders", "liquid", "outer_continuous", "semantic_role", "stage"}
        for slot, value in declaration.items():
            if slot not in supported_slots and value is not None and value != {} and value != []:
                expected_roots.append(("unresolved_declaration_slot", "both", value))
        actual_roots = [(consumer.get("kind"), consumer.get("phase"), callbacks.get(consumer.get("callback"), {}).get("original")) for consumer in cell.get("consumers", [])]
        actual_shocks = [(consumer.get('group'), consumer.get('phase'), payload_digest(callbacks.get(consumer.get('callback'), {}).get('original')))
                         for consumer in cell.get('consumers', []) if consumer.get('kind') == 'shock_law']
        if sorted(expected_shocks, key=repr) != sorted(actual_shocks, key=repr):
            fail('original shock registration wiring changed', cell=cell['id'])
        if root_keys(expected_roots) != root_keys(actual_roots): fail("original root consumer coverage changed", cell=cell["id"])
        for consumer in cell.get("consumers", []):
            if consumer.get("kind") == "feasible" and consumer.get("decision_phases") != ["solve", "simulation"]:
                fail("original truth-feasibility applicability changed", cell=cell["id"], callback=consumer.get("callback"))
    expected_edges = set()
    if original_schedule:
        expected_edges = {(by_pair[(period, source_regime)]["id"], by_pair[(period + 1, target_regime)]["id"], phase)
                          for period, source_regime, target_regime, phase in original_schedule[1] if (period, source_regime) in by_pair and (period + 1, target_regime) in by_pair}
    for phase, source_phase in [("solve", "solution"), ("simulation", "simulation")]:
        rows = fields(reach.get(source_phase)).get("targets_by_period", source.get("phase_targets", {}).get(phase, []))
        for period, mapping in enumerate(rows):
            for regime, targets in mapping.items():
                for target in targets:
                    if (period, regime) in by_pair and (period + 1, target) in by_pair:
                        expected_edges.add((by_pair[(period, regime)]["id"], by_pair[(period + 1, target)]["id"], phase))
    actual_edges = {(relation.get("source"), target, phase)
                    for relation in normalized.get("relations", []) if relation.get("kind") == "transition"
                    for target in relation.get("targets", [])
                    for phase in (("solve", "simulation") if relation.get("phase") == "both" else (relation.get("phase"),))}
    if actual_edges != expected_edges: fail("complete phase-specific support graph changed")
    departing_cells = {source_id for source_id, _, _ in expected_edges}
    for cell in cells.values():
        if cell.get("terminal") != (cell['id'] not in departing_cells): fail("terminal declaration changed", cell=cell["id"])
    relations = normalized.get("relations", [])
    def callback_payloads(identifiers): return [callbacks.get(identifier, {}).get("original") for identifier in identifiers]
    def same_payloads(left, right): return sorted(map(payload_digest, left)) == sorted(map(payload_digest, right))
    # Reified payload equality is insufficient if its consumer wiring is lost.
    # Bind every original output and law input to the matching typed incidence.
    for cell in cells.values():
        for axis, did in cell.get("states", {}).items():
            domain = domains[did]
            if not domain.get("solve_imputation"): continue
            matches = [relation for relation in relations if relation.get("kind") == "imputation" and relation.get("source") == cell["id"] and relation.get("targets") == [cell["id"]] and relation.get("phase") == "solve" and axis in relation.get("state_updates", {})]
            if len(matches) != 1 or matches[0].get("produces") != [axis] or matches[0]["state_updates"].get(axis) != domain["solve_imputation"]:
                fail("original carried-state imputation incidence changed", cell=cell["id"], state=axis)
    expected_owners = defaultdict(list)
    for source_id, target_id, phase in sorted(expected_edges):
        cell, target = cells[source_id], cells[target_id]
        declaration = original_regimes[cell["regime"]]
        transitions = [relation for relation in relations if relation.get("kind") == "transition" and relation.get("source") == source_id and relation.get("targets") == [target_id] and relation.get("phase") == phase]
        if len(transitions) != 1:
            fail("original phase transition incidence duplicated/omitted", source=source_id, target=target_id, phase=phase)
            continue
        transition = transitions[0]
        law = (original_schedule[4].get((cell["period"], cell["regime"], phase)) if original_schedule else
               selected(fields(graph.get("laws", {}).get(cell["regime"], {})).get("transition"), phase, cell["period"]))
        if not same_payloads(callback_payloads(transition.get("callbacks", [])), [] if law is None else [law]):
            fail("original transition law consumer changed", relation=transition.get("id"))
        produced_by_joint = set()
        for joint_name, raw in declaration.get("joint_transitions", {}).get(target["regime"], {}).items():
            payload = selected(raw, phase, cell["period"])
            matches = [relation for relation in relations if relation.get("kind") == "joint" and relation.get("source") == source_id and target_id in relation.get("targets", []) and relation.get("phase") == phase and relation.get("joint_name") == joint_name]
            if len(matches) != 1:
                fail("full original joint relation omitted/duplicated", source=source_id, target=target_id, phase=phase, joint=joint_name); continue
            relation = matches[0]
            if relation.get("original") != payload: fail("original full joint law/support changed", relation=relation.get("id"))
            outputs = fields(payload).get("outputs", {})
            produced_by_joint.update(outputs)
            if set(relation.get("joint_outputs", {})) != set(outputs): fail("joint decoded output coverage changed", relation=relation.get("id"))
            if relation.get("produces") != sorted(outputs) or relation.get("state_updates") != relation.get("joint_outputs"):
                fail("joint output incidence changed", relation=relation.get("id"))
            joint_fields = fields(payload)
            expected_callbacks = [joint_fields[key] for key in ("probabilities", "support") if isinstance(joint_fields.get(key), dict) and joint_fields[key].get("kind") == "callable"]
            if not same_payloads(callback_payloads(relation.get("callbacks", [])), expected_callbacks): fail("full joint law callback omitted/changed", relation=relation.get("id"))
            if relation.get("support") != joint_fields.get("support") or relation.get("support_size") != joint_fields.get("support_size"): fail("full joint support changed", relation=relation.get("id"))
            for axis, raw_callback in outputs.items():
                if callbacks.get(relation.get("joint_outputs", {}).get(axis), {}).get("original") != selected(raw_callback, phase, cell["period"]): fail("joint decoded output callback changed", relation=relation.get("id"), axis=axis)
        target_axes = {axis for axis, did in target["states"].items() if phase != "solve" or domains[did].get("solve_axis", True)} - produced_by_joint
        if set(transition.get("produces", [])) != target_axes: fail("original state-producer coverage changed", relation=transition.get("id"))
        for axis in target_axes:
            raw = declaration.get("state_transitions", {}).get(axis)
            if isinstance(raw, dict) and "kind" not in raw: raw = raw.get(target["regime"])
            if raw is not None: raw = selected(raw, phase, cell["period"])
            else:
                candidate = selected(declaration.get("states", {}).get(axis), phase, cell["period"])
                latent='_lcm.processes.joint_normal_lowering._JointNormalCoordinate'
                if short(candidate).endswith("Process") or (isinstance(candidate,dict) and candidate.get('type')==latent): raw = candidate
                elif candidate is None:
                    arriving=selected(original_regimes[target['regime']].get('states',{}).get(axis),phase,target['period'])
                    iid={'lcm.processes.'+n for n in ('NormalIIDProcess','UniformIIDProcess','LogNormalIIDProcess','NormalMixtureIIDProcess')}
                    iid.add(latent)
                    if (isinstance(arriving,dict) and arriving.get('type') in iid
                            and fields(arriving).get('state_conditioned') is None):raw=arriving
            if raw is None:
                unknown("original state producer is not exposed", source=source_id, target=target_id, state=axis, phase=phase)
            elif callbacks.get(transition.get("state_updates", {}).get(axis), {}).get("original") != raw:
                fail("original state update consumer changed", relation=transition.get("id"), state=axis)
            elif isinstance(raw,dict) and raw.get('type') in {'lcm.processes.'+n for n in ('TauchenAR1Process','RouwenhorstAR1Process','TauchenNormalMixtureAR1Process')}|{'_lcm.processes.ar1_on_nodes._TauchenOnNodesProcess'}:
                callback=callbacks[transition['state_updates'][axis]]
                if callback.get('arguments')!=[axis]:
                    fail('original process dependency owner changed',relation=transition.get('id'),state=axis)
    # Independently bind every owned fallback to the leg of the ORIGINAL gate.
    laws = graph.get("laws", {})
    def refs(value, phase=None):
        if short(value) == "Phased" and phase is not None:
            yield from refs(fields(value).get("solve" if phase == "solve" else "simulate"), phase)
        elif short(value) in {"ProjectedRegimeValue", "ArrivalRead", "ParticipationRead", "ExitTarget", "Counterparty"}:
            yield value
        elif isinstance(value, dict):
            for child in value.values(): yield from refs(child, phase)
        elif isinstance(value, list):
            for child in value: yield from refs(child, phase)
    # Replay policy demand from the original declarations, independently of
    # the producer's admitted decision-cell table. Query ownership is not a
    # physical visit and does not establish a stored value-bank allocation.
    query_expected = {}
    declaration_digests = {}
    base = "/graph_snapshot" if "graph_snapshot" in original else ""
    regime_key = "user_regimes" if "user_regimes" in source else "regimes"
    def ptr(parent, key):
        return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")
    def reference_locations(value, path, phase):
        if short(value) == "Phased":
            side = "solve" if phase == "solve" else "simulate"
            yield from reference_locations(fields(value).get(side), ptr(ptr(path, "fields"), side), phase)
        elif short(value) in {"ProjectedRegimeValue", "ArrivalRead", "ParticipationRead", "ExitTarget", "Counterparty"}:
            yield value, path
        elif isinstance(value, dict):
            for key, child in value.items(): yield from reference_locations(child, ptr(path, key), phase)
        elif isinstance(value, list):
            for index, child in enumerate(value): yield from reference_locations(child, ptr(path, index), phase)
    def original_references(name, phase):
        raw = source.get(regime_key, {}).get(name, {})
        declaration = original_regimes[name]
        root = ptr(ptr(base, regime_key), name)
        root_fields = ptr(root, "fields") if raw.get("kind") in {"object", "dataclass"} else root
        keys = ["same_period_refs", "value_constraints", "exits"]
        if not declaration.get("same_period_refs"): keys.append("constraints")
        for key in keys:
            location = ptr(root_fields, key) if key in fields(raw) else ptr(base, key)
            yield from reference_locations(declaration.get(key, {}), location, phase)
    def demand(name, period, phase):
        if name not in original_regimes or not 0 <= period < len(time): return None
        if (period, name) in by_pair: return None
        qid = "Q" + _digest([period, name, phase])[:20]
        if qid in query_expected: return qid
        raw = source.get(regime_key, {}).get(name, {})
        declaration = original_regimes[name]
        if name not in declaration_digests: declaration_digests[name] = _digest(raw)
        expected = {"id": qid, "regime": name, "period": period, "phase": phase,
                    "coordinate": number(time[period]), "coordinate_kind": coordinate_kind,
                    "source_path": ptr(ptr(base, regime_key), name), "declaration_sha256": declaration_digests[name],
                    "states": sorted(declaration.get("states", {})), "actions": sorted(declaration.get("actions", {})),
                    "role": contract.get("roles", {}).get(name, declaration.get("semantic_role")),
                    "stage": contract.get("stages", {}).get(name, declaration.get("stage")),
                    "admission": "policy_query_only", "storage": "not_asserted", "dependencies": []}
        query_expected[qid] = expected
        for reference, location in original_references(name, phase):
            other = fields(reference).get("regime")
            physical = by_pair.get((period, other))
            if physical: destination = {"cell": physical["id"]}
            else:
                nested = demand(other, period, phase)
                destination = {"query_owner": nested} if nested is not None else {"unavailable_owner": other}
            expected["dependencies"].append({**destination, "source_path": location,
                                               "reference_sha256": _digest(reference)})
        return qid
    def check_reference(raw, origin, phase, kind, owner=None, incoming=None):
        data = fields(raw)
        target = by_pair.get((cells[origin]["period"], data.get("regime")))
        if target is None:
            query = demand(data.get("regime"), cells[origin]["period"], phase)
            if query is None:
                unknown("original same-period reference target unavailable", source=origin, regime=data.get("regime")); return
            matches = [relation for relation in relations if relation.get("kind") == "policy_query"
                       and relation.get("source") == origin and relation.get("query_target") == query
                       and relation.get("phase") == phase and relation.get("reference_kind") == kind
                       and relation.get("reference_sha256") == _digest(raw)
                       and relation.get("ownership") == owner and relation.get("incoming_source") == incoming]
            if not matches:
                fail("original policy query ownership/incidence omitted or changed", source=origin, phase=phase); return
            for relation in matches:
                if relation.get("targets") != [] or relation.get("produces") != [] or relation.get("state_updates") != {}:
                    fail("policy query was given a physical transition or state producer", relation=relation.get("id"))
                if not same_payloads(callback_payloads(relation.get("callbacks", [])), [raw]):
                    fail("original policy query source payload changed", relation=relation.get("id"))
                allowed_statuses={'unresolved','structural'} if normalized.get('analysis_scope')=='pylcm-structural-graph/v1' else {'unresolved'}
                if any(callbacks.get(cid, {}).get("status") not in allowed_statuses for cid in relation.get("callbacks", [])):
                    fail("unproved policy query factorization was asserted", relation=relation.get("id"))
                read_type = data.get("kind", "Participating" if short(raw) == "ExitTarget" else "Raw" if short(raw) == "Counterparty" else "value")
                if relation.get("read_type") != read_type or relation.get("read_field") != data.get("field", "value") or relation.get("stakeholder") != data.get("stakeholder", data.get("role")) or relation.get("observes") != data.get("observes", []):
                    fail("original policy query information/read contract changed", relation=relation.get("id"))
                try: pointed = _pointer(original, relation.get("source_path", ""))
                except (KeyError, IndexError, ValueError, TypeError): pointed = None
                if pointed != raw: fail("original policy query source pointer changed", relation=relation.get("id"))
            if normalized.get("analysis_scope") != "pylcm-structural-graph/v1":
                unknown("original policy query factorization unproved", source=origin, regime=data.get("regime"), phase=phase)
            return
        matches = [relation for relation in relations if relation.get("kind") == kind and relation.get("source") == origin and relation.get("targets") == [target["id"]] and relation.get("phase") == phase and relation.get("original") == raw and relation.get("ownership") == owner and relation.get("incoming_source") == incoming]
        if not matches:
            fail("original same-period reference incidence omitted/changed", source=origin, target=target["id"], phase=phase, reference_kind=kind); return
        projection = selected(data.get("projection", {}), phase, cells[origin]["period"]) or {}
        for relation in matches:
            if set(relation.get("state_updates", {})) != set(projection) or relation.get("produces") != sorted(projection): fail("original reference projection output coverage changed", relation=relation.get("id"))
            for axis, callback in projection.items():
                if callbacks.get(relation.get("state_updates", {}).get(axis), {}).get("original") != callback: fail("original reference projection callback changed", relation=relation.get("id"), state=axis)
            expected_callbacks = [data["utility_adjustment"]] if data.get("utility_adjustment") is not None else []
            if not same_payloads(callback_payloads(relation.get("callbacks", [])), expected_callbacks): fail("original reference adjustment callback changed", relation=relation.get("id"))
            read_type = data.get("kind", "Participating" if short(raw) == "ExitTarget" else "Raw" if short(raw) == "Counterparty" else "value")
            if relation.get("read_type") != read_type or relation.get("read_field") != data.get("field", "value") or relation.get("stakeholder") != data.get("stakeholder", data.get("role")) or relation.get("observes") != data.get("observes", []): fail("original reference information/read contract changed", relation=relation.get("id"))
    for cell in cells.values():
        declaration = original_regimes[cell["regime"]]
        keys = ["same_period_refs", "value_constraints", "exits"]
        if not declaration.get("same_period_refs"): keys.append("constraints")
        for key in keys:
            for phase in ("solve", "simulation"):
                for raw in refs(declaration.get(key, {}), phase):
                    check_reference(raw, cell["id"], phase, "exit" if key == "exits" else "value_read")
    for cell in cells.values():
        native_law = fields(laws.get(cell["regime"]))
        declared_gates = _independent_gate_map(native_law, "gated_edges", "arrivals", "participation")
        gate_records = [(target_regime, side, selected(gate, side, cell["period"] + 1))
                        for target_regime, gate in declared_gates.items() for side in ("solve", "simulation")]
        if original_schedule:
            gate_records += [(target, side, gate) for period, regime, target, side, gate in original_schedule[3] if period == cell["period"] and regime == cell["regime"]]
        for target_regime, gate_phase, gate in gate_records:
            landing = by_pair.get((cell["period"] + 1, target_regime))
            # Independently require the actual phase/date incidence. Static
            # unused gate declarations impose no current fallback ownership.
            if landing is None or (cell["id"], landing["id"], gate_phase) not in expected_edges:
                continue
            if landing:
                matches = [relation for relation in relations if relation.get("kind") in {"gate", "arrival"} and relation.get("source") == landing["id"] and relation.get("incoming_source") == cell["id"] and relation.get("phase") == gate_phase and relation.get("original") == gate]
                if not matches: fail("gate is not evaluated at original landing grid/date", source=cell["id"], target=landing["id"])
                for relation in matches:
                    predicate = fields(gate).get("predicate", fields(gate).get("gate"))
                    expected_callbacks = [predicate] if predicate is not None else []
                    if not same_payloads(callback_payloads(relation.get("callbacks", [])), expected_callbacks): fail("original landing predicate consumer changed", relation=relation.get("id"))
                    if relation.get("produces") != [] or relation.get("observes") != fields(gate).get("observes", []): fail("original landing information contract changed", relation=relation.get("id"))
                for phase in (("solve", "simulation") if gate_phase == "both" else (gate_phase,)):
                    for raw_reference in refs(fields(gate).get("references", fields(gate).get("reads", fields(gate).get("gate_refs", {}))), phase):
                        check_reference(raw_reference, landing["id"], phase, "value_read", {"source_regime": cell["regime"], "target_regime": target_regime}, cell["id"])
            for leg, route in fields(gate).get("legs", fields(gate).get("routes", {})).items():
                owner = {"source_regime": cell["regime"], "target_regime": target_regime, "leg": leg}
                fallback = fields(route).get("fallback")
                if gate_phase != "both": fallback = selected(fallback, gate_phase, cell["period"])
                for raw_reference in refs(fallback):
                    target = by_pair.get((cell["period"] + 1, fields(raw_reference).get("regime")))
                    landing = by_pair.get((cell["period"] + 1, target_regime))
                    if target is None:
                        if landing is not None:
                            for phase in (("solve", "simulation") if gate_phase == "both" else (gate_phase,)):
                                check_reference(raw_reference, landing["id"], phase, "fallback", owner, cell["id"])
                        continue
                    matches = [relation for relation in relations if relation.get("kind") == "fallback" and landing is not None and relation.get("source") == landing["id"] and relation.get("incoming_source") == cell["id"] and target["id"] in relation.get("targets", []) and relation.get("original") == raw_reference and relation.get("ownership") == owner]
                    if not matches: fail("owned fallback route omitted or ownership changed", source=cell["id"], owner=owner)
                    if landing is not None:
                        for phase in (("solve", "simulation") if gate_phase == "both" else (gate_phase,)):
                            check_reference(raw_reference, landing["id"], phase, "fallback", owner, cell["id"])
                    if owner not in expected_owners[target["id"]]: expected_owners[target["id"]].append(owner)
    for cell in cells.values():
        expected_owner = expected_owners.get(cell["id"], [])
        if sorted(map(_digest, cell.get("ownership") or [])) != sorted(map(_digest, expected_owner)):
            fail("original fallback ownership changed", cell=cell["id"])
    if normalized.get("query_owners", {}) != query_expected:
        fail("original demanded policy-owner registry changed")
    if normalized.get('analysis_scope') == 'pylcm-structural-graph/v1':
        if source.get('schema') != 'constructed-pylcm-model-graph-1':
            fail('structural graph scope requires a constructed Model')
        allocation = normalized.get('allocation', {})
        if (allocation.get('kind') != 'per_block' or allocation.get('status') != 'declared'
                or allocation.get('identity') != 'pylcm-regime-domains/v1'
                or allocation.get('boundary_unit') != 1):
            fail('pylcm regime-specific structural allocation changed')
        return
    original_allocation = contract.get("allocation", source.get("allocation_contract", {"status": "unknown"}))
    if isinstance(original_allocation, dict) and original_allocation.get("kind", original_allocation.get("mode")) in {"per_block", "global_union"}:
        for key, value in original_allocation.items():
            if normalized.get("allocation", {}).get(key) != value: fail("source allocation contract changed", field=key)
