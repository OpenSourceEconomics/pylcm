"""Independent, bounded numerical oracles for declaration regressions.

This module imports no _lcm_regime_finder producer, translator, scorer, or evaluator.
It is deliberately not a production analysis path. The source interpreter accepts
an inspectable scalar Python subset and resolves every call through an explicit
captured call target. It never imports JAX or supplies a counterfeit JAX module.
The primitive implementations below are part of this oracle's stated boundary.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Hashable, Iterable, Mapping
import hashlib
import json
from fractions import Fraction
import math
from typing import Any, TypedDict


class OracleUnsupported(ValueError):
    """The original callback is outside this tiny numerical oracle's language."""


def exact(value: int | float | str | Fraction) -> Fraction:
    if isinstance(value, bool):
        raise OracleUnsupported("A Boolean is not a numerical probability")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, str)):
        return Fraction(value)
    if isinstance(value, float) and math.isfinite(value):
        return Fraction(str(value))
    raise OracleUnsupported("Expected a finite scalar number")


def unpack(value: Any) -> Any:  # noqa: PAN002 - Original Python callback values and captures have intentionally arbitrary types.
    """Decode original capture values; codes are never substituted for values."""
    if isinstance(value, list):
        return [unpack(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("kind") == "array":
        if "values" not in value:
            raise OracleUnsupported("Original array payload is unavailable")
        return unpack(value["values"])
    if value.get("kind") == "categorical-class":
        return {key: unpack(item) for key, item in value["members"].items()}
    if set(value) == {"fraction"}:
        return Fraction(value["fraction"])
    return {key: unpack(item) for key, item in value.items()}


class ScalarSourceOracle:
    """Read original callback AST directly, without translated analysis IR.

    Supported: literals, scalar arithmetic/comparison/Boolean operations,
    indexing, local assignment/reassignment, delete, if and return. A call must
    have a full ``call_targets[ast.unparse(callee)]`` record. Custom callable
    targets recurse into their original AST. A primitive is admitted only with
    ``resolution='actual_object_identity'`` and an explicitly allowed canonical
    identity. These records are trusted capture evidence, not fabricated here.
    A separate capture checker must establish their relation to actual objects.
    """

    _PRIMITIVES = {
        "numpy.where": "select", "jax.numpy.where": "select",
        "numpy.logical_and": "logical_and", "jax.numpy.logical_and": "logical_and",
        "numpy.logical_or": "logical_or", "jax.numpy.logical_or": "logical_or",
        "numpy.asarray": "asarray", "jax.numpy.asarray": "asarray",
        "builtins.abs": "abs", "builtins.min": "min", "builtins.max": "max",
    }

    def __init__(self, *, max_steps: int = 100_000, max_depth: int = 24):
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.steps = 0
        self.used_primitives: set[str] = set()
        self.callable_records: Mapping[str, Any] = {}  # noqa: PAN002 - Original Python callback values and captures have intentionally arbitrary types.
        self.binding_values = {}

    def call(self, metadata: Mapping[str, Any], values: Mapping[str, Any]) -> Any:  # noqa: PAN002 - Original Python callback values and captures have intentionally arbitrary types.
        self.steps = 0
        self.callable_records = metadata.get("callable_records", {})
        self.binding_values = {}
        return self._call(metadata, (), dict(values), 0)

    def _target(self, target):
        """Independent finite-reference validation; no producer helper imported."""
        if not isinstance(target, dict) or target.get("kind") != "callable-reference":
            return target
        record = self.callable_records.get(target.get("capture_ref"))
        if not isinstance(record, dict):
            raise OracleUnsupported("Missing original shared callable record")
        def strip(value):
            if isinstance(value, dict):
                if value.get("kind") == "callable-reference":
                    return {"kind": "callable-reference", "capture_ref": value.get("capture_ref")}
                return {key: strip(item) for key, item in value.items()}
            return [strip(item) for item in value] if isinstance(value, list) else value
        omitted = {"identity_sha256", "record_sha256", "capture_graph_sha256", "callable_records", "capture_report"}
        payload = {key: strip(value) for key, value in record.items() if key not in omitted}
        local = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        if (record.get("identity_sha256") != target.get("identity_sha256") or
                record.get("capture_graph_sha256") != target.get("capture_graph_sha256") or
                record.get("record_sha256") != local):
            raise OracleUnsupported("Original shared callable identity changed")
        return record

    def _binding(self, value, active=frozenset()):
        if isinstance(value, dict) and value.get('kind') == 'callable-reference':
            record = self._target(value)
            if record.get('kind') != 'binding-record':
                return value
            ref = value.get('capture_ref')
            if ref in active or record.get('capture_status', 'COMPLETE') != 'COMPLETE':
                raise OracleUnsupported('Cyclic, changed, or incomplete original binding')
            if ref not in self.binding_values:
                decoded = self._binding(record['value'], active | {ref})
                if record.get('binding_type') == 'builtins.tuple':
                    decoded = tuple(decoded)
                elif record.get('binding_type') in {'builtins.set', 'builtins.frozenset'}:
                    decoded = frozenset(decoded) if record['binding_type'] == 'builtins.frozenset' else set(decoded)
                self.binding_values[ref] = decoded
            return self.binding_values[ref]
        if isinstance(value, list):
            return [self._binding(item, active) for item in value]
        if isinstance(value, dict):
            return {key: self._binding(item, active) for key, item in value.items()}
        return value

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise OracleUnsupported("Tiny numerical oracle step budget exhausted")

    def _call(self, meta: Mapping[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any], depth: int) -> Any:  # noqa: PAN002 - Original Python callback values and captures have intentionally arbitrary types.
        if meta.get("capture_status", "COMPLETE") != "COMPLETE":
            raise OracleUnsupported("Original capture is incomplete or changed")
        if depth > self.max_depth:
            raise OracleUnsupported("Tiny numerical oracle helper depth exhausted")
        try:
            tree = ast.parse(meta["source"])
            fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        except (KeyError, SyntaxError, StopIteration) as exc:
            raise OracleUnsupported("Original callable source is unavailable") from exc
        formals = [item.arg for item in fn.args.posonlyargs + fn.args.args]
        keyword_only = [item.arg for item in fn.args.kwonlyargs]
        if fn.args.vararg or fn.args.kwarg or len(args) > len(formals):
            raise OracleUnsupported("Variadic or incompatible original signature")
        partial = unpack(self._binding(meta.get("partial_bindings", meta.get("partial_keywords", {}))))
        env = unpack(self._binding(meta.get("bindings", {})))
        supplied = {**partial, **dict(zip(formals, args)), **kwargs}
        if set(supplied) - set(formals + keyword_only):
            raise OracleUnsupported("Unknown original formal argument")
        defaults = unpack(self._binding(meta.get("defaults", {})))
        for name in formals + keyword_only:
            if name in supplied:
                env[name] = supplied[name]
            elif name in defaults:
                env[name] = defaults[name]
            else:
                raise OracleUnsupported("Missing original formal argument: " + name)
        returned, value = self._statements(fn.body, env, meta, depth)
        if not returned:
            raise OracleUnsupported("Original callback has no scalar return")
        return value

    def _statements(self, statements: list[ast.stmt], env: dict[str, Any], meta: Mapping[str, Any], depth: int) -> tuple[bool, Any]:  # noqa: PAN002 - Original callback local values and captured records are arbitrary Python values.
        for stmt in statements:
            self._tick()
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            if isinstance(stmt, ast.Return):
                return True, self._expr(stmt.value, env, meta, depth) if stmt.value is not None else None
            if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
                value = self._expr(stmt.value, env, meta, depth)
                for target in stmt.targets:
                    assert isinstance(target, ast.Name)
                    env[target.id] = value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                env[stmt.target.id] = self._expr(stmt.value, env, meta, depth)
            elif isinstance(stmt, ast.Delete) and all(isinstance(t, ast.Name) for t in stmt.targets):
                for target in stmt.targets:
                    assert isinstance(target, ast.Name)
                    env.pop(target.id, None)
            elif isinstance(stmt, ast.If):
                branch = stmt.body if self._expr(stmt.test, env, meta, depth) else stmt.orelse
                returned, value = self._statements(branch, env, meta, depth)
                if returned:
                    return returned, value
            else:
                raise OracleUnsupported("Unsupported original statement: " + type(stmt).__name__)
        return False, None

    def _expr(self, node: ast.AST, env: dict[str, Any], meta: Mapping[str, Any], depth: int) -> Any:  # noqa: PAN002 - Original Python callback values and captures have intentionally arbitrary types.
        self._tick()
        e = lambda child: self._expr(child, env, meta, depth)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return exact(node.value)
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise OracleUnsupported("Unbound original name: " + node.id)
            return env[node.id]
        if isinstance(node, ast.Attribute):
            value = e(node.value)
            if isinstance(value, dict) and node.attr in value:
                return value[node.attr]
            raise OracleUnsupported("Unresolved original attribute")
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [e(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            return {e(key): e(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            value, index = e(node.value), e(node.slice)
            if isinstance(index, Fraction) and index.denominator == 1:
                index = int(index)
            try:
                return value[index]
            except (IndexError, TypeError, KeyError) as exc:
                raise OracleUnsupported("Invalid original table lookup") from exc
        if isinstance(node, ast.IfExp):
            return e(node.body if e(node.test) else node.orelse)
        if isinstance(node, ast.UnaryOp):
            x = e(node.operand)
            if isinstance(node.op, ast.Not): return not x
            if isinstance(node.op, ast.USub): return -x
            if isinstance(node.op, ast.UAdd): return +x
            if isinstance(node.op, ast.Invert): return not x if isinstance(x, bool) else ~x
        if isinstance(node, ast.BoolOp):
            values = [e(item) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.BinOp):
            left, right = e(node.left), e(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return exact(left) / exact(right)
            if isinstance(node.op, ast.FloorDiv): return left // right
            if isinstance(node.op, ast.Mod): return left % right
            if isinstance(node.op, ast.Pow) and isinstance(right, int) and abs(right) <= 32: return left ** right
            if isinstance(node.op, ast.BitAnd): return left & right
            if isinstance(node.op, ast.BitOr): return left | right
        if isinstance(node, ast.Compare):
            left = e(node.left)
            for op, child in zip(node.ops, node.comparators):
                right = e(child)
                comparisons: Mapping[type[ast.cmpop], Callable[[], bool]] = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                               ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                               ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                               ast.In: lambda: left in right, ast.NotIn: lambda: left not in right}  # codespell:ignore
                if type(op) not in comparisons:
                    raise OracleUnsupported("Unsupported original comparison")
                if not comparisons[type(op)](): return False
                left = right
            return True
        if isinstance(node, ast.Call):
            spelling = ast.unparse(node.func)
            target = self._target(meta.get("call_targets", {}).get(spelling))
            if not isinstance(target, dict):
                raise OracleUnsupported("No captured call target for " + spelling)
            args = tuple(e(arg) for arg in node.args)
            if any(kw.arg is None for kw in node.keywords):
                raise OracleUnsupported("Expanded original call arguments are unsupported")
            kwargs = {kw.arg: e(kw.value) for kw in node.keywords if kw.arg is not None}
            if target.get("kind") == "callable":
                return self._call(target, args, kwargs, depth + 1)
            canonical = target.get("canonical")
            operation = self._PRIMITIVES.get(canonical)
            if target.get("kind") != "primitive" or target.get("resolution") != "actual_object_identity" or operation != target.get("operation"):
                raise OracleUnsupported("Original call has no admitted primitive identity: " + spelling)
            self.used_primitives.add(canonical)
            if operation == "select" and len(args) == 3 and not kwargs: return args[1] if args[0] else args[2]
            if operation == "logical_and" and len(args) == 2 and not kwargs: return bool(args[0] and args[1])
            if operation == "logical_or" and len(args) == 2 and not kwargs: return bool(args[0] or args[1])
            if operation == "asarray" and len(args) == 1 and not kwargs: return args[0]
            if operation == "abs" and len(args) == 1 and not kwargs: return abs(args[0])
            if operation == "min" and not kwargs: return min(args)
            if operation == "max" and not kwargs: return max(args)
            raise OracleUnsupported("Unsupported original primitive signature")
        raise OracleUnsupported("Unsupported original expression: " + type(node).__name__)


def joint_pushforward[Target, Projected: Hashable](rows: Iterable[tuple[Target, int | float | str | Fraction]], projection: Callable[[Target], Projected]) -> dict[Projected, Fraction]:
    """Aggregate COMPLETE target tuples, never independently factor marginals."""
    result: dict[Projected, Fraction] = {}
    for target, probability in rows:
        mass = exact(probability)
        if mass < 0:
            raise OracleUnsupported("Negative original joint mass")
        if mass == 0:
            continue
        projected = projection(target)
        result[projected] = result.get(projected, Fraction(0)) + mass
    return result


class TransportResult(TypedDict):
    status: str
    errors: list[str]
    max_absolute_residual: str
    original_support: int
    reduced_support: int


def independent_joint_transport[Target, Projected: Hashable](original: Iterable[tuple[Target, int | float | str | Fraction]], reduced: Iterable[tuple[Projected, int | float | str | Fraction]], projection: Callable[[Target], Projected],
                                *, mass_absolute: str = "1e-12", residual_absolute: str = "1e-12") -> TransportResult:
    """Tiny regression law comparison with normalization tolerance and exact support.

    Positive support is compared before numerical residuals. A 1e-50 atom cannot
    disappear merely because a normalization or residual budget is much larger.
    """
    before = joint_pushforward(original, projection)
    after = joint_pushforward(reduced, lambda target: target)
    errors = []
    mass_budget, residual_budget = exact(mass_absolute), exact(residual_absolute)
    for label, law in [("original", before), ("reduced", after)]:
        if abs(sum(law.values(), Fraction(0)) - 1) > mass_budget:
            errors.append(label + " normalization")
    if set(before) != set(after): errors.append("positive joint support")
    residual = max((abs(before.get(key, 0) - after.get(key, 0)) for key in set(before) | set(after)), default=Fraction(0))
    if residual > residual_budget: errors.append("joint probability transport")
    return {"status": "FAIL" if errors else "PASS", "errors": errors,
            "max_absolute_residual": str(residual), "original_support": len(before), "reduced_support": len(after)}




def nondominated[Value](items: Iterable[Value], metrics=lambda item: item) -> list[Value]:
    """Literal all-pairs Pareto oracle, independent of a search recurrence."""
    items = list(items)
    result = []
    for index, item in enumerate(items):
        point = tuple(metrics(item))
        if not any(other_index != index and
                   all(a <= b for a, b in zip(tuple(metrics(other)), point)) and
                   any(a < b for a, b in zip(tuple(metrics(other)), point))
                   for other_index, other in enumerate(items)):
            result.append(item)
    return result
