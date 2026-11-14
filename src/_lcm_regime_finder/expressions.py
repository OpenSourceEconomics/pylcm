"""Small, non-executing expression language and conservative partial evaluation.

No eval, imports, attribute access, or user Python execution is used. An opaque
node denotes a pure total primitive with explicitly declared input variables;
source adapters record where those original dependency facts came from.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
import operator


class AdmissionError(ValueError):
    pass


@dataclass(frozen=True)
class Value:
    known: bool
    value: object = None
    reads: frozenset[str] = frozenset()


def literal(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdmissionError('Nonfinite numeric literal')
        return Fraction(str(value))
    if isinstance(value, list):
        return [literal(x) for x in value]
    if isinstance(value, tuple):
        return tuple(literal(x) for x in value)
    if isinstance(value, dict):
        return {k: literal(v) for k, v in value.items()}
    return value


def unknown(*values):
    return Value(False, reads=frozenset().union(*(v.reads for v in values)))


def conjunction(values):
    values = list(values)
    if any(v.known and not v.value for v in values):
        return Value(True, False)
    if all(v.known for v in values):
        return Value(True, True)
    return unknown(*values)


def disjunction(values):
    values = list(values)
    if any(v.known and v.value for v in values):
        return Value(True, True)
    if all(v.known for v in values):
        return Value(True, False)
    return unknown(*values)


BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: lambda x, y: Fraction(x) / Fraction(y), ast.FloorDiv: operator.floordiv,
       ast.Mod: operator.mod, ast.Pow: operator.pow}
CMP: Mapping[type[ast.cmpop], Callable[..., bool]] = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
       ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y}  # codespell:ignore
FUN = {'min': min, 'max': max, 'abs': abs, 'len': len,
       'int': int, 'bool': bool, 'sum': sum}


@lru_cache(maxsize=8192)
def _parse_expression(expr):
    return ast.parse(expr, mode='eval').body


@lru_cache(maxsize=8192)
def _constant_items(node):
    return tuple(literal(child.value) for child in node.elts)


@lru_cache(maxsize=8192)
def _constant_members(node):
    return frozenset(_constant_items(node))


def partial(expr, env=None):
    """Return an exact known value or a conservative free-variable set."""
    env = env or {}
    if isinstance(expr, (int, float, bool)) or expr is None:
        return Value(True, literal(expr))
    if isinstance(expr, list):
        return conjunction(partial(e, env) for e in expr)
    if isinstance(expr, dict):
        if 'opaque' in expr:
            reads = expr.get('reads')
            if not isinstance(reads, list) or not all(isinstance(x, str) for x in reads):
                raise AdmissionError('opaque primitive requires original reads list')
            if not isinstance(expr.get('arguments', []), list):
                raise AdmissionError('opaque arguments must be an expression list')
            arguments = [partial(value, env) for value in expr.get('arguments', [])]
            return Value(False, reads=frozenset(x for x in reads if x not in env).union(*(value.reads for value in arguments)))
        if 'and' in expr:
            return conjunction(partial(e, env) for e in expr['and'])
        if 'or' in expr:
            return disjunction(partial(e, env) for e in expr['or'])
        if 'not' in expr:
            v = partial(expr['not'], env)
            return Value(True, not v.value) if v.known else v
        if 'if' in expr:
            cond = partial(expr['if'], env)
            if cond.known:
                return partial(expr['then'] if cond.value else expr['else'], env)
            return unknown(cond, partial(expr['then'], env), partial(expr['else'], env))
        raise AdmissionError('Unsupported expression object')
    if not isinstance(expr, str):
        raise AdmissionError('Expression must be scalar text, literal, conjunction, or opaque primitive')
    try:
        tree = _parse_expression(expr)
    except SyntaxError as exc:
        raise AdmissionError(f'Invalid expression: {expr!r}') from exc

    def visit(node):
        if isinstance(node, ast.Constant):
            return Value(True, literal(node.value))
        if isinstance(node, ast.Name):
            return Value(True, literal(env[node.id])) if node.id in env else Value(False, reads=frozenset([node.id]))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if all(isinstance(child, ast.Constant) for child in node.elts):
                vals = _constant_items(node)
                return Value(True, vals if isinstance(node, ast.Tuple) else list(vals))
            items = [visit(x) for x in node.elts]
            if not all(x.known for x in items):
                return unknown(*items)
            vals = [x.value for x in items]
            return Value(True, tuple(vals) if isinstance(node, ast.Tuple) else vals)
        if isinstance(node, ast.Dict):
            keys, vals = [visit(x) for x in node.keys], [visit(x) for x in node.values]
            if not all(x.known for x in keys + vals):
                return unknown(*(keys + vals))
            return Value(True, {k.value: v.value for k, v in zip(keys, vals)})
        if isinstance(node, ast.Subscript):
            a, b = visit(node.value), visit(node.slice)
            if not a.known or not b.known:
                return unknown(a, b)
            try:
                index = int(b.value) if isinstance(b.value, Fraction) and b.value.denominator == 1 else b.value
                return Value(True, a.value[index])
            except (KeyError, IndexError, TypeError) as exc:
                raise AdmissionError(f'Invalid table index in {expr!r}') from exc
        if isinstance(node, ast.UnaryOp):
            v = visit(node.operand)
            if not v.known:
                return v
            if isinstance(node.op, ast.Not):
                return Value(True, not v.value)
            if isinstance(node.op, ast.USub):
                return Value(True, -v.value)
            if isinstance(node.op, ast.UAdd):
                return v
            raise AdmissionError('Unsupported unary operator')
        if isinstance(node, ast.BoolOp):
            return (conjunction if isinstance(node.op, ast.And) else disjunction)(visit(x) for x in node.values)
        if isinstance(node, ast.IfExp):
            c = visit(node.test)
            if c.known:
                return visit(node.body if c.value else node.orelse)
            return unknown(c, visit(node.body), visit(node.orelse))
        if isinstance(node, ast.BinOp) and type(node.op) in BIN:
            a, b = visit(node.left), visit(node.right)
            if not a.known or not b.known:
                return unknown(a, b)
            try:
                return Value(True, literal(BIN[type(node.op)](a.value, b.value)))
            except (ValueError, ZeroDivisionError, TypeError, OverflowError) as exc:
                raise AdmissionError(f'Undefined arithmetic in {expr!r}') from exc
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):  # codespell:ignore
                rhs = node.comparators[0]
                if isinstance(rhs, (ast.List, ast.Tuple, ast.Set)) and all(isinstance(child, ast.Constant) for child in rhs.elts):
                    left = visit(node.left)
                    if not left.known:
                        return left
                    answer = left.value in _constant_members(rhs)
                    return Value(True, answer if isinstance(node.ops[0], ast.In) else not answer)
            items = [visit(node.left)] + [visit(x) for x in node.comparators]
            if not all(x.known for x in items):
                return unknown(*items)
            if not all(type(op) in CMP for op in node.ops):
                raise AdmissionError('Unsupported comparison')
            return Value(True, all(CMP[type(op)](a.value,b.value) for op,a,b in zip(node.ops,items,items[1:])))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUN and not node.keywords:
            args = [visit(x) for x in node.args]
            if not all(x.known for x in args):
                return unknown(*args)
            try:
                return Value(True, literal(FUN[node.func.id](*(x.value for x in args))))
            except (ValueError, TypeError, ZeroDivisionError) as exc:
                raise AdmissionError(f'Invalid primitive call in {expr!r}') from exc
        raise AdmissionError(f'Unsupported syntax {type(node).__name__} in scalar expression')
    return visit(tree)


def evaluate(expr, env):
    result = partial(expr, env)
    if not result.known:
        raise AdmissionError('Expression is not numerically executable: ' + ','.join(sorted(result.reads)))
    return result.value


def jsonable(value):
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return value
