#!/usr/bin/env python3
"""Prove route-local candidate-array flow from ``Q_and_F`` to full reducers.

The executable certificate is intentionally finite.  Its universal half is this
standard-library AST proof: on each ordinary GridSearch route, the exact arrays
bound by ``Q_arr, F_arr = Q_and_F(...)`` must reach the full reducer without an
intervening candidate-changing expression.  The proof is strict by design.  A
new statement in the certified corridor is not assumed harmless; it has to be
made part of the explicit, independently checked representation allowlist.

The four corridors are:

* singleton solve -> ``Q_arr.max(where=F_arr, ...)``;
* singleton simulate -> ``argmax_and_max(Q_arr, where=F_arr, ...)``;
* collective solve -> ``collective_readout(..., feasibility=F_arr, ...)``;
* collective simulate -> ``collective_argmax_and_readout(...,
  feasibility=F_arr, ...)``.

The only allowed representation change is the collective split of the trailing
stakeholder axis, exactly ``Q_arr[..., index]`` for every enumerated stakeholder.
It cannot select, reorder, or mask an action axis. The common feasibility array
is passed by identity. The shared axis-move, flatten, scalarization, full argmax,
collective delegation, and value-gather bodies are pinned as exact AST shapes,
so moving a filter into a helper does not evade the route proof.
"""

# Exact production and mutation snippets intentionally preserve long source lines.
# ruff: noqa: E501

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path
from typing import Any

MAX_Q_SOURCE = "src/_lcm/regime_building/max_Q_over_a.py"
ARGMAX_SOURCE = "src/_lcm/regime_building/argmax.py"
COLLECTIVE_SOURCE = "src/_lcm/regime_building/collective.py"

_CERTIFIED_CORRIDOR_SOURCES = (MAX_Q_SOURCE, ARGMAX_SOURCE, COLLECTIVE_SOURCE)


def canonical_json(payload: dict[str, Any]) -> str:
    """Render deterministic JSON for command-line controls."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _name(node: ast.AST | None, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _unparse(node: ast.AST | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def _target_names(target: ast.expr) -> tuple[str, ...] | None:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple) and all(
        isinstance(item, ast.Name) for item in target.elts
    ):
        return tuple(item.id for item in target.elts if isinstance(item, ast.Name))
    return None


def _assigned_names(statement: ast.stmt) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets.extend(statement.targets)
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        targets.append(statement.target)
    elif isinstance(statement, ast.Delete):
        targets.extend(statement.targets)
    found: set[str] = set()
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                found.add(child.id)
    return found


def _definition(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one top-level function {name!r}, found {len(matches)}"
        )
    return matches[0]


def _ordinary_nested(
    tree: ast.Module, *, outer_name: str, nested_name: str
) -> ast.FunctionDef:
    """Return the non-taste-shock nested reducer, identified by its route branch."""
    outer = _definition(tree, outer_name)
    matches = [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.FunctionDef)
        and node is not outer
        and node.name == nested_name
        and any(
            isinstance(statement, ast.If)
            and ast.unparse(statement.test) == "stakeholders is not None"
            for statement in node.body
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one ordinary nested {nested_name!r} in {outer_name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _body_without_docstring(node: ast.FunctionDef) -> list[ast.stmt]:
    """Return executable statements, excluding the function docstring."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _ast_key(node: ast.AST) -> str:
    """Return a location-independent structural representation."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _expected_statements(source: str) -> list[ast.stmt]:
    """Parse an allowlisted statement sequence under the running Python AST."""
    return ast.parse(source).body


def _body_matches(node: ast.FunctionDef, expected_source: str) -> bool:
    observed = [_ast_key(item) for item in _body_without_docstring(node)]
    expected = [_ast_key(item) for item in _expected_statements(expected_source)]
    return observed == expected


def _positional_signature(
    node: ast.FunctionDef,
    *,
    names: tuple[str, ...],
    defaults: tuple[str, ...] = (),
) -> bool:
    args = node.args
    return (
        not args.posonlyargs
        and tuple(item.arg for item in args.args) == names
        and args.vararg is None
        and not args.kwonlyargs
        and not args.kw_defaults
        and args.kwarg is None
        and tuple(ast.unparse(item) for item in args.defaults) == defaults
    )


def _keyword_only_signature(node: ast.FunctionDef, *, names: tuple[str, ...]) -> bool:
    args = node.args
    return (
        not args.posonlyargs
        and not args.args
        and args.vararg is None
        and tuple(item.arg for item in args.kwonlyargs) == names
        and all(item is None for item in args.kw_defaults)
        and args.kwarg is None
        and not args.defaults
    )


def _q_and_f_origin(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    if _target_names(statement.targets[0]) != ("Q_arr", "F_arr"):
        return False
    if not (
        isinstance(statement.value, ast.Call)
        and _call_name(statement.value) == "Q_and_F"
    ):
        return False
    if statement.value.args:
        return False
    if _unparse(_keyword(statement.value, "next_regime_to_V_arr")) != (
        "next_regime_to_V_arr"
    ):
        return False
    splats = [item.value for item in statement.value.keywords if item.arg is None]
    return len(splats) == 1 and _name(splats[0], "states_actions_params")


def _negative_infinity(node: ast.expr | None) -> bool:
    return node is not None and ast.unparse(node) == "-jnp.inf"


def _exact_singleton_solve_return(statement: ast.stmt) -> bool:
    if not (
        isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call)
    ):
        return False
    call = statement.value
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "max"
        and _name(call.func.value, "Q_arr")
        and not call.args
    ):
        return False
    return (
        _name(_keyword(call, "where"), "F_arr")
        and _negative_infinity(_keyword(call, "initial"))
        and _keyword(call, "axis") is None
        and {item.arg for item in call.keywords} == {"where", "initial"}
    )


def _exact_singleton_simulate_return(statement: ast.stmt) -> bool:
    if not (
        isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call)
    ):
        return False
    call = statement.value
    return (
        _call_name(call) == "argmax_and_max"
        and len(call.args) == 1
        and _name(call.args[0], "Q_arr")
        and _name(_keyword(call, "where"), "F_arr")
        and _negative_infinity(_keyword(call, "initial"))
        and _keyword(call, "axis") is None
        and {item.arg for item in call.keywords} == {"where", "initial"}
    )


def _exact_action_axes(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == ("action_axes",)
        and ast.unparse(statement.value) == "tuple(range(F_arr.ndim))"
    )


def _exact_stakeholder_split(statement: ast.stmt) -> bool:
    """Allow only a split of the trailing stakeholder axis, never an action axis."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == ("stakeholder_Q",)
        and isinstance(statement.value, ast.DictComp)
    ):
        return False
    comp = statement.value
    if not (_name(comp.key, "name") and isinstance(comp.value, ast.Subscript)):
        return False
    if not _name(comp.value.value, "Q_arr"):
        return False
    slice_node = comp.value.slice
    if not (
        isinstance(slice_node, ast.Tuple)
        and len(slice_node.elts) == 2
        and isinstance(slice_node.elts[0], ast.Constant)
        and slice_node.elts[0].value is Ellipsis
        and _name(slice_node.elts[1], "index")
    ):
        return False
    if len(comp.generators) != 1:
        return False
    generator = comp.generators[0]
    return (
        _target_names(generator.target) == ("index", "name")
        and isinstance(generator.iter, ast.Call)
        and _call_name(generator.iter) == "enumerate"
        and len(generator.iter.args) == 1
        and _name(generator.iter.args[0], "stakeholders")
        and not generator.iter.keywords
        and not generator.ifs
        and generator.is_async == 0
    )


def _exact_weights_call(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node) == "_evaluate_pareto_weights"
        and not node.args
        and _name(_keyword(node, "pareto_weights"), "pareto_weights")
        and _name(_keyword(node, "states_actions_params"), "states_actions_params")
        and {item.arg for item in node.keywords}
        == {"pareto_weights", "states_actions_params"}
    )


def _exact_collective_reducer_assignment(
    statement: ast.stmt, *, simulate: bool
) -> bool:
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.value, ast.Call)
    ):
        return False
    expected_targets = (
        ("argmax_flat", "values", "_dissolution")
        if simulate
        else ("values", "dissolution")
    )
    expected_call = (
        "collective_argmax_and_readout" if simulate else "collective_readout"
    )
    call = statement.value
    return (
        _target_names(statement.targets[0]) == expected_targets
        and _call_name(call) == expected_call
        and not call.args
        and _name(_keyword(call, "stakeholder_Q"), "stakeholder_Q")
        and _name(_keyword(call, "feasibility"), "F_arr")
        and _exact_weights_call(_keyword(call, "weights"))
        and _name(_keyword(call, "action_axes"), "action_axes")
        and {item.arg for item in call.keywords}
        == {"stakeholder_Q", "feasibility", "weights", "action_axes"}
    )


def _exact_values_stack(node: ast.expr | None) -> bool:
    if not isinstance(node, ast.Call) or ast.unparse(node.func) != "jnp.stack":
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], ast.ListComp):
        return False
    comp = node.args[0]
    if not (
        isinstance(comp.elt, ast.Subscript)
        and _name(comp.elt.value, "values")
        and _name(comp.elt.slice, "name")
        and len(comp.generators) == 1
    ):
        return False
    generator = comp.generators[0]
    return (
        _name(generator.target, "name")
        and _name(generator.iter, "stakeholders")
        and not generator.ifs
        and generator.is_async == 0
        and _unparse(_keyword(node, "axis")) == "-1"
        and {item.arg for item in node.keywords} == {"axis"}
    )


def _exact_collective_return(statement: ast.stmt, *, simulate: bool) -> bool:
    if not (
        isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple)
    ):
        return False
    if simulate:
        return len(statement.value.elts) == 2 and all(
            _name(node, expected)
            for node, expected in zip(
                statement.value.elts, ("argmax_flat", "V_stacked"), strict=True
            )
        )
    return (
        len(statement.value.elts) == 2
        and _exact_values_stack(statement.value.elts[0])
        and _name(statement.value.elts[1], "dissolution")
    )


def _exact_collective_body(node: ast.If, *, simulate: bool) -> bool:  # noqa: PLR0911
    if ast.unparse(node.test) != "stakeholders is not None" or node.orelse:
        return False
    expected_length = 5 if simulate else 4
    if len(node.body) != expected_length:
        return False
    if not _exact_action_axes(node.body[0]):
        return False
    if not _exact_stakeholder_split(node.body[1]):
        return False
    if not _exact_collective_reducer_assignment(node.body[2], simulate=simulate):
        return False
    if simulate:
        stack = node.body[3]
        if not (
            isinstance(stack, ast.Assign)
            and len(stack.targets) == 1
            and _target_names(stack.targets[0]) == ("V_stacked",)
            and _exact_values_stack(stack.value)
        ):
            return False
        return _exact_collective_return(node.body[4], simulate=True)
    return _exact_collective_return(node.body[3], simulate=False)


def _productmap_binding_errors(
    tree: ast.Module, *, outer_name: str, nested_name: str
) -> list[str]:
    """Require one unwrapped, unbatched action product and no later rebinding."""
    try:
        outer = _definition(tree, outer_name)
    except ValueError as error:
        return [f"{outer_name}: {error}"]
    assignments = [
        statement
        for statement in ast.walk(outer)
        if isinstance(statement, ast.Assign)
        and any(_target_names(target) == ("Q_and_F",) for target in statement.targets)
    ]
    errors: list[str] = []
    if len(assignments) != 1:
        errors.append(
            f"{outer_name}: Q_and_F must be bound exactly once to productmap; "
            f"found {len(assignments)} assignments"
        )
    else:
        value = assignments[0].value
        if not (
            isinstance(value, ast.Call)
            and _call_name(value) == "productmap"
            and not value.args
            and _name(_keyword(value, "func"), "Q_and_F")
            and _name(_keyword(value, "variables"), "action_names")
            and _unparse(_keyword(value, "batch_sizes"))
            == "dict.fromkeys(action_names, 0)"
            and {item.arg for item in value.keywords}
            == {"func", "variables", "batch_sizes"}
        ):
            errors.append(
                f"{outer_name}: action productmap is wrapped, filtered, batched, "
                "or does not consume the original Q_and_F"
            )
    rebindings = [
        statement
        for statement in ast.walk(outer)
        if isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
        and nested_name in _assigned_names(statement)
    ]
    if rebindings:
        errors.append(
            f"{outer_name}: returned reducer {nested_name} is rebound after definition"
        )
    return errors


def _corridor_errors(
    tree: ast.Module,
    *,
    outer_name: str,
    nested_name: str,
    simulate: bool,
) -> list[str]:
    label = "simulate" if simulate else "solve"
    errors: list[str] = []
    try:
        nested = _ordinary_nested(tree, outer_name=outer_name, nested_name=nested_name)
    except ValueError as error:
        return [f"{label}: {error}"]
    if not (
        len(nested.decorator_list) == 1
        and isinstance(nested.decorator_list[0], ast.Call)
        and _call_name(nested.decorator_list[0]) == "with_signature"
    ):
        errors.append(f"{label}: ordinary reducer has a non-allowlisted decorator")
    if len(nested.body) != 3:
        errors.append(
            f"{label}: ordinary reducer corridor must contain exactly origin, "
            f"collective branch, singleton return; found {len(nested.body)} statements"
        )
        return errors
    origin, collective, singleton = nested.body
    if not _q_and_f_origin(origin):
        errors.append(
            f"{label}: first corridor statement is not exact Q_arr/F_arr origin"
        )
    if not isinstance(collective, ast.If) or not _exact_collective_body(
        collective, simulate=simulate
    ):
        errors.append(
            f"{label}: collective corridor contains a non-allowlisted transformation"
        )
    singleton_ok = (
        _exact_singleton_simulate_return(singleton)
        if simulate
        else _exact_singleton_solve_return(singleton)
    )
    if not singleton_ok:
        errors.append(
            f"{label}: singleton corridor does not pass raw Q_arr/F_arr directly "
            "to the full reducer"
        )
    for statement in ast.walk(nested):
        if isinstance(statement, ast.stmt) and statement is not origin:
            assigned = _assigned_names(statement)
            if {"Q_arr", "F_arr"} & assigned:
                errors.append(
                    f"{label}: candidate arrays are reassigned or deleted at line "
                    f"{getattr(statement, 'lineno', '?')}"
                )
    return errors


def _argmax_reducer_errors(tree: ast.Module) -> list[str]:
    """Certify the shared argmax and its representation helpers exactly."""
    errors: list[str] = []
    try:
        node = _definition(tree, "argmax_and_max")
        move = _definition(tree, "_move_axes_to_back")
        flatten = _definition(tree, "_flatten_last_n_axes")
    except ValueError as error:
        return [f"argmax reducer: {error}"]

    if node.decorator_list or move.decorator_list or flatten.decorator_list:
        errors.append("argmax reducer: decorators are not allowlisted")
    if not _positional_signature(
        node,
        names=("a", "axis", "initial", "where"),
        defaults=("None", "None", "None"),
    ):
        errors.append("argmax reducer: signature/defaults changed")
    if not _positional_signature(move, names=("a", "axes")):
        errors.append("argmax reducer: axis-move helper signature changed")
    if not _positional_signature(flatten, names=("a", "n")):
        errors.append("argmax reducer: flatten helper signature changed")

    expected_argmax = r"""if axis is None:
    axis = tuple(range(a.ndim))
elif isinstance(axis, int):
    axis = (axis,)
if a.ndim == 0 or len(axis) == 0:
    return jnp.array(0, dtype=jnp.int32), a
if a.ndim != 0:
    a = _move_axes_to_back(a, axes=axis)
    a = _flatten_last_n_axes(a, n=len(axis))
if where is not None and where.ndim != 0:
    where = _move_axes_to_back(where, axes=axis)
    where = _flatten_last_n_axes(where, n=len(axis))
_max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)
max_value_mask = a == _max
if where is not None:
    max_value_mask = jnp.logical_and(max_value_mask, where)
_argmax = jnp.argmax(max_value_mask, axis=-1).astype(jnp.int32)
return _argmax, _max.reshape(_argmax.shape)
"""
    expected_move = r"""front_axes = sorted(set(range(a.ndim)) - set(axes))
return a.transpose((*front_axes, *axes))
"""
    expected_flatten = r"""return a.reshape(*a.shape[:-n], -1)
"""
    if not _body_matches(node, expected_argmax):
        errors.append(
            "argmax reducer: executable body differs from the full paired "
            "value/feasibility reduction"
        )
    if not _body_matches(move, expected_move):
        errors.append(
            "argmax reducer: action-axis move is not the exact order-preserving "
            "representation change"
        )
    if not _body_matches(flatten, expected_flatten):
        errors.append(
            "argmax reducer: action-axis flatten is not the exact shape-only "
            "representation change"
        )

    if any(
        isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
        and {
            "argmax_and_max",
            "_move_axes_to_back",
            "_flatten_last_n_axes",
        }
        & _assigned_names(statement)
        for statement in tree.body
    ):
        errors.append("argmax reducer: module-level rebinding is not allowlisted")
    return errors


def _collective_reducer_errors(tree: ast.Module) -> list[str]:
    """Certify collective scalarization, argmax, delegation, and gather exactly."""
    errors: list[str] = []
    try:
        argmax_node = _definition(tree, "collective_argmax_and_readout")
        readout_node = _definition(tree, "collective_readout")
        weighted = _definition(tree, "_weighted_sum")
        gather = _definition(tree, "_gather_along_actions")
    except ValueError as error:
        return [f"collective reducer: {error}"]

    if any(
        node.decorator_list for node in (argmax_node, readout_node, weighted, gather)
    ):
        errors.append("collective reducer: decorators are not allowlisted")
    signatures = (
        (
            argmax_node,
            ("stakeholder_Q", "feasibility", "weights", "action_axes"),
            "argmax/readout",
        ),
        (
            readout_node,
            ("stakeholder_Q", "feasibility", "weights", "action_axes"),
            "readout delegation",
        ),
        (weighted, ("stakeholder_Q", "weights"), "weighted scalarization"),
        (gather, ("q", "argmax_flat", "action_axes"), "value gather"),
    )
    for node, names, label in signatures:
        if not _keyword_only_signature(node, names=names):
            errors.append(f"collective reducer: {label} signature changed")

    expected_argmax = r"""if not stakeholder_Q:
    msg = "collective_argmax_and_readout requires at least one stakeholder."
    raise ValueError(msg)
if set(stakeholder_Q) != set(weights):
    msg = (
        "stakeholder_Q and weights must have identical keys; got "
        f"{sorted(stakeholder_Q)} vs {sorted(weights)}."
    )
    raise ValueError(msg)
objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)
argmax_flat, _ = argmax_and_max(
    objective, axis=action_axes, initial=-jnp.inf, where=feasibility
)
dissolution = (
    ~jnp.any(feasibility, axis=action_axes) if action_axes else ~feasibility
)
values = {
    name: jnp.where(
        dissolution,
        -jnp.inf,
        _gather_along_actions(
            q=q, argmax_flat=argmax_flat, action_axes=action_axes
        ),
    )
    for name, q in stakeholder_Q.items()
}
return argmax_flat, values, dissolution
"""
    expected_readout = r"""_argmax_flat, values, dissolution = collective_argmax_and_readout(
    stakeholder_Q=stakeholder_Q,
    feasibility=feasibility,
    weights=weights,
    action_axes=action_axes,
)
return values, dissolution
"""
    expected_weighted = r"""names = list(stakeholder_Q)
def _term(name: str) -> FloatND:
    return zero_safe_weighted_term(
        weight=jnp.asarray(weights[name]),
        value=stakeholder_Q[name],
        subnormal_is_accounted_for=False,
    )
terms = [_term(name) for name in names]
if len(terms) <= _LARGEST_ORDER_FREE_HOUSEHOLD:
    objective = terms[0]
    for term in terms[1:]:
        objective = objective + term
    return objective
return sum_in_value_order(jnp.stack(terms, axis=0), axis=0)
"""
    expected_gather = r"""if not action_axes:
    return q
q_moved = _move_axes_to_back(q, axes=action_axes)
q_flat = _flatten_last_n_axes(q_moved, n=len(action_axes))
gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)
return gathered[..., 0]
"""
    expected = (
        (argmax_node, expected_argmax, "household argmax/readout"),
        (readout_node, expected_readout, "solve-side delegation"),
        (weighted, expected_weighted, "pointwise weighted scalarization"),
        (gather, expected_gather, "shared-index stakeholder gather"),
    )
    for node, body, label in expected:
        if not _body_matches(node, body):
            errors.append(
                f"collective reducer: {label} differs from the exact allowlisted flow"
            )

    if any(
        isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
        and {
            "collective_argmax_and_readout",
            "collective_readout",
            "_weighted_sum",
            "_gather_along_actions",
        }
        & _assigned_names(statement)
        for statement in tree.body
    ):
        errors.append("collective reducer: module-level rebinding is not allowlisted")
    return errors


def verify_direct_candidate_flow(*, repo_root: Path) -> dict[str, Any]:
    """Verify the complete direct-flow architecture against one repository tree."""
    root = repo_root.resolve()
    errors: list[str] = []
    offending: set[str] = set()
    parsed: dict[str, ast.Module] = {}
    for relative in _CERTIFIED_CORRIDOR_SOURCES:
        path = root / relative
        try:
            parsed[relative] = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
        except (OSError, SyntaxError) as error:
            errors.append(f"{relative}: {error}")
            offending.add(relative)
    max_tree = parsed.get(MAX_Q_SOURCE)
    if max_tree is not None:
        binding_errors = _productmap_binding_errors(
            max_tree, outer_name="get_max_Q_over_a", nested_name="max_Q_over_a"
        )
        binding_errors += _productmap_binding_errors(
            max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
        )
        solve_errors = _corridor_errors(
            max_tree,
            outer_name="get_max_Q_over_a",
            nested_name="max_Q_over_a",
            simulate=False,
        )
        simulate_errors = _corridor_errors(
            max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
            simulate=True,
        )
        errors.extend(binding_errors)
        errors.extend(solve_errors)
        errors.extend(simulate_errors)
        if binding_errors or solve_errors or simulate_errors:
            offending.add(MAX_Q_SOURCE)
    argmax_tree = parsed.get(ARGMAX_SOURCE)
    if argmax_tree is not None:
        new_errors = _argmax_reducer_errors(argmax_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(ARGMAX_SOURCE)
    collective_tree = parsed.get(COLLECTIVE_SOURCE)
    if collective_tree is not None:
        new_errors = _collective_reducer_errors(collective_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(COLLECTIVE_SOURCE)
    return {
        "ok": not errors,
        "result": "pass" if not errors else "fail",
        "errors": errors,
        "offending_paths": sorted(offending),
        "routes": {
            "singleton_solve": "Q_and_F -> Q_arr.max(where=F_arr)",
            "singleton_simulate": "Q_and_F -> argmax_and_max(Q_arr, where=F_arr)",
            "collective_solve": (
                "Q_and_F -> trailing stakeholder split -> "
                "collective_readout(feasibility=F_arr)"
            ),
            "collective_simulate": (
                "Q_and_F -> trailing stakeholder split -> "
                "collective_argmax_and_readout(feasibility=F_arr)"
            ),
        },
        "certified_corridor_sources": list(_CERTIFIED_CORRIDOR_SOURCES),
    }


def _insert_before_nth(
    text: str, marker: str, insertion: str, *, occurrence: int
) -> str:
    start = -1
    for _ in range(occurrence):
        start = text.find(marker, start + 1)
        if start < 0:
            raise ValueError(
                f"marker not found for occurrence {occurrence}: {marker!r}"
            )
    return text[:start] + insertion + text[start:]


def direct_flow_mutations(source: str) -> dict[str, str]:
    """Generate the required route/value/support/shape/index perturbation family."""
    mutations: dict[str, str] = {}
    solve_singleton = "            return Q_arr.max(where=F_arr, initial=-jnp.inf)"
    simulate_singleton = (
        "            return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)"
    )
    collective_marker = "                action_axes = tuple(range(F_arr.ndim))"

    mutations["singleton_solve:q_order"] = _insert_before_nth(
        source,
        solve_singleton,
        "            Q_flat = Q_arr.reshape(-1)\n"
        "            order_filter = Q_flat[0] > Q_flat[1]\n"
        "            F_arr = jnp.where(\n"
        "                order_filter,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["singleton_simulate:mt9_rank_permutation"] = _insert_before_nth(
        source,
        simulate_singleton,
        "            Q_flat = Q_arr.reshape(-1)\n"
        "            mt9_order = (\n"
        "                (Q_flat[0] > Q_flat[2])\n"
        "                & (Q_flat[2] > Q_flat[1])\n"
        "                & (Q_flat[1] > Q_flat[3])\n"
        "                & (Q_flat[3] > Q_flat[4])\n"
        "                & (Q_flat[4] > Q_flat[5])\n"
        "                & jnp.all(F_arr)\n"
        "            )\n"
        "            F_arr = jnp.where(\n"
        "                mt9_order,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["singleton_simulate:q_gap"] = _insert_before_nth(
        source,
        simulate_singleton,
        "            gap_filter = (\n"
        "                Q_arr.reshape(-1)[0] - Q_arr.reshape(-1)[1] > 0.5\n"
        "            )\n"
        "            F_arr = jnp.where(\n"
        "                gap_filter,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["collective_solve:support_size"] = _insert_before_nth(
        source,
        collective_marker,
        "                support_filter = jnp.sum(F_arr) > 1\n"
        "                F_arr = jnp.where(\n"
        "                    support_filter,\n"
        "                    F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                    F_arr,\n"
        "                )\n",
        occurrence=1,
    )
    mutations["collective_simulate:shape_axis"] = _insert_before_nth(
        source,
        collective_marker,
        "                shape_filter = (F_arr.ndim == 2) & (F_arr.shape[-1] > 1)\n"
        "                F_arr = jnp.where(\n"
        "                    shape_filter,\n"
        "                    F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                    F_arr,\n"
        "                )\n",
        occurrence=2,
    )

    mutations["singleton_solve:inline_where_transform"] = source.replace(
        "return Q_arr.max(where=F_arr, initial=-jnp.inf)",
        "return Q_arr.max(\n"
        "                where=F_arr.reshape(-1).at[0].set(False)\n"
        "                .reshape(F_arr.shape),\n"
        "                initial=-jnp.inf,\n"
        "            )",
        1,
    )
    mutations["singleton_simulate:inline_q_transform"] = source.replace(
        "return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)",
        "return argmax_and_max(\n"
        "                Q_arr.reshape(-1)[::-1], where=F_arr, initial=-jnp.inf\n"
        "            )",
        1,
    )
    mutations["collective_solve:inline_feasibility_transform"] = source.replace(
        "feasibility=F_arr,",
        "feasibility=F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),",
        1,
    )
    mutations["collective_simulate:action_axis_slice"] = source.replace(
        "name: Q_arr[..., index] for index, name in enumerate(stakeholders)",
        "name: Q_arr[1:, ..., index] for index, name in enumerate(stakeholders)",
        2,
    )
    mutations["solve:wrapped_productmap_input"] = source.replace(
        "func=Q_and_F,\n        variables=action_names,",
        "func=lambda **kwargs: Q_and_F(**kwargs),\n        variables=action_names,",
        1,
    )
    # Replace the second productmap binding independently for simulate.
    first = source.find("func=Q_and_F,\n        variables=action_names,")
    second = source.find("func=Q_and_F,\n        variables=action_names,", first + 1)
    if second < 0:
        raise ValueError("second Q_and_F productmap binding not found")
    mutations["simulate:wrapped_productmap_input"] = source[:second] + source[
        second:
    ].replace(
        "func=Q_and_F,\n        variables=action_names,",
        "func=lambda **kwargs: Q_and_F(**kwargs),\n        variables=action_names,",
        1,
    )

    route_specs = {
        "singleton_solve": (solve_singleton, 1, "            "),
        "singleton_simulate": (simulate_singleton, 1, "            "),
        "collective_solve": (collective_marker, 1, "                "),
        "collective_simulate": (collective_marker, 2, "                "),
    }
    for route, (marker, occurrence, indent) in route_specs.items():
        for index in range(6):
            insertion = (
                f"{indent}F_arr = F_arr.reshape(-1).at[{index}]"
                ".set(False).reshape(F_arr.shape)\n"
            )
            mutations[f"{route}:candidate_index_{index}"] = _insert_before_nth(
                source, marker, insertion, occurrence=occurrence
            )
    return mutations


def direct_flow_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Return semantic mutations across every certified corridor source."""
    root = repo_root.resolve()
    max_source = (root / MAX_Q_SOURCE).read_text(encoding="utf-8")
    argmax_source = (root / ARGMAX_SOURCE).read_text(encoding="utf-8")
    collective_source = (root / COLLECTIVE_SOURCE).read_text(encoding="utf-8")
    specs: dict[str, dict[str, str]] = {
        name: {"path": MAX_Q_SOURCE, "source": mutated}
        for name, mutated in direct_flow_mutations(max_source).items()
    }

    def replace_once(source: str, old: str, new: str, *, label: str) -> str:
        if source.count(old) != 1:
            raise ValueError(
                f"{label}: expected one mutation marker, found {source.count(old)}"
            )
        return source.replace(old, new, 1)

    argmax_cases = {
        "shared_argmax:q_order_early_return": replace_once(
            argmax_source,
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            "    if a.reshape(-1)[0] > a.reshape(-1)[1]:\n"
            "        return jnp.array(1, dtype=jnp.int32), a.reshape(-1)[1]\n"
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            label="argmax q-order",
        ),
        "shared_argmax:support_filter": replace_once(
            argmax_source,
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            "    where = jnp.where(\n"
            "        jnp.sum(where) > 1,\n"
            "        where.reshape(-1).at[0].set(False).reshape(where.shape),\n"
            "        where,\n"
            "    )\n"
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            label="argmax support",
        ),
        "shared_argmax:axis_prefix": replace_once(
            argmax_source,
            "        axis = tuple(range(a.ndim))",
            "        axis = tuple(range(a.ndim - 1))",
            label="argmax axis prefix",
        ),
        "shared_argmax:axis_reorder": replace_once(
            argmax_source,
            "    return a.transpose((*front_axes, *axes))",
            "    return a.transpose((*front_axes, *reversed(axes)))",
            label="argmax axis reorder",
        ),
        "shared_argmax:flatten_drop_last": replace_once(
            argmax_source,
            "    return a.reshape(*a.shape[:-n], -1)",
            "    return a[..., :-1].reshape(*a.shape[:-n], -1)",
            label="argmax flatten drop",
        ),
    }
    specs.update(
        {
            name: {"path": ARGMAX_SOURCE, "source": mutated}
            for name, mutated in argmax_cases.items()
        }
    )

    collective_cases = {
        "shared_collective:q_gap_filter": replace_once(
            collective_source,
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)\n"
            "    objective = jnp.where(\n"
            "        objective.reshape(-1)[0] - objective.reshape(-1)[1] > 0.5,\n"
            "        objective.reshape(-1).at[0].set(-jnp.inf).reshape(\n"
            "            objective.shape\n"
            "        ),\n"
            "        objective,\n"
            "    )",
            label="collective q-gap",
        ),
        "shared_collective:feasibility_inline_filter": replace_once(
            collective_source,
            "        objective, axis=action_axes, initial=-jnp.inf, where=feasibility",
            "        objective, axis=action_axes, initial=-jnp.inf,\n"
            "        where=feasibility.reshape(-1).at[0].set(False).reshape(\n"
            "            feasibility.shape\n"
            "        )",
            label="collective feasibility inline",
        ),
        "shared_collective:action_axis_prefix": replace_once(
            collective_source,
            "        objective, axis=action_axes, initial=-jnp.inf, where=feasibility",
            "        objective, axis=action_axes[:-1], initial=-jnp.inf, where=feasibility",
            label="collective action axis",
        ),
        "shared_collective:gather_next_candidate": replace_once(
            collective_source,
            "    gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)",
            "    gathered = jnp.take_along_axis(q_flat, (argmax_flat + 1)[..., None], axis=-1)",
            label="collective gather",
        ),
        "shared_collective:early_candidate_return": replace_once(
            collective_source,
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            "    if jnp.all(feasibility):\n"
            "        return (\n"
            "            jnp.array(1, dtype=jnp.int32),\n"
            "            {name: q.reshape(-1)[1] for name, q in stakeholder_Q.items()},\n"
            "            jnp.array(False),\n"
            "        )\n"
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            label="collective early return",
        ),
    }
    specs.update(
        {
            name: {"path": COLLECTIVE_SOURCE, "source": mutated}
            for name, mutated in collective_cases.items()
        }
    )
    return specs


def run_direct_flow_mutation_controls(*, repo_root: Path) -> dict[str, Any]:
    """Show every semantic mutation is rejected by the route-local AST proof.

    The repository control wraps these exact mutations in full source inventory,
    contract, compiled-policy, and manifest re-anchoring. This in-process half
    isolates the semantic proof and changes only temporary copies.
    """
    root = repo_root.resolve()
    clean = verify_direct_candidate_flow(repo_root=root)
    originals = {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in _CERTIFIED_CORRIDOR_SOURCES
    }
    cases: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as raw:
        temp_root = Path(raw) / "repo"
        for relative, source in originals.items():
            target = temp_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        for name, spec in direct_flow_mutation_specs(repo_root=root).items():
            relative = spec["path"]
            target = temp_root / relative
            target.write_text(spec["source"], encoding="utf-8")
            result = verify_direct_candidate_flow(repo_root=temp_root)
            cases[name] = {
                "path": relative,
                "rejected": not result["ok"],
                "errors": result["errors"],
                "offending_paths": result["offending_paths"],
            }
            target.write_text(originals[relative], encoding="utf-8")
    admitted = sorted(name for name, result in cases.items() if not result["rejected"])
    return {
        "clean": clean,
        "mutations": cases,
        "mutation_count": len(cases),
        "admitted_mutations": admitted,
        "all_rejected": clean["ok"] and not admitted,
    }
