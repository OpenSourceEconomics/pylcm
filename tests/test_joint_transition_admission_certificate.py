"""Semantic controls for joint transition producer admission."""

# Exact production and mutation snippets intentionally preserve long source lines.
# ruff: noqa: E501

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_TRANSITION_CHECKS = "src/_lcm/transition_checks.py"


def _definition(*, tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _calls(*, node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (
            (isinstance(child.func, ast.Name) and child.func.id == name)
            or (isinstance(child.func, ast.Attribute) and child.func.attr == name)
        )
    ]


def _keyword(*, call: ast.Call, name: str) -> str | None:
    values = [ast.unparse(item.value) for item in call.keywords if item.arg == name]
    assert len(values) <= 1
    return values[0] if values else None


def _mutate_definition(*, source: str, name: str, old: str, new: str) -> str:
    """Replace one exact expression inside one named function."""
    tree = ast.parse(source)
    node = _definition(tree=tree, name=name)
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    definition_source = "".join(lines[start:end])
    assert definition_source.count(old) == 1
    lines[start:end] = [definition_source.replace(old, new)]
    return "".join(lines)


def _joint_transition_admission_errors(  # noqa: C901, PLR0912, PLR0915
    *, source: str
) -> list[str]:
    tree = ast.parse(source)
    errors: list[str] = []

    sequence = _definition(tree=tree, name="_validate_transition_sequence")
    calls = _calls(node=sequence, name="validate_joint_transitions_all_periods")
    if len(calls) != 1 or _keyword(call=calls[0], name="memory") != "memory":
        errors.append("transition sequence must forward memory to joint validation")

    sweep = _definition(tree=tree, name="validate_joint_transitions_all_periods")
    sweep_source = ast.unparse(sweep)
    if (
        "current_memory = summary.memory if summary is not None and "
        "summary.memory is not None else memory" not in sweep_source
    ):
        errors.append("summary memory must remain authoritative for joint laws")
    weight_calls = _calls(node=sweep, name="_evaluate_joint_weights")
    if (
        len(weight_calls) != 1
        or _keyword(call=weight_calls[0], name="memory") != "current_memory"
    ):
        errors.append("joint weights must receive current memory")
    law_calls = _calls(node=sweep, name="_validate_joint_laws")
    if (
        len(law_calls) != 1
        or _keyword(call=law_calls[0], name="memory") != "current_memory"
    ):
        errors.append("joint owners must receive current memory")

    owner = _definition(tree=tree, name="_own_transition_outputs")
    ownership_calls = _calls(node=owner, name="_set_transition_outputs")
    observed_owner_updates = {
        (_keyword(call=call, name="memory"), _keyword(call=call, name="outputs"))
        for call in ownership_calls
    }
    if ("memory", "outputs") not in observed_owner_updates:
        errors.append("temporary joint outputs must become owners")
    if ("memory", "restore") not in observed_owner_updates:
        errors.append("temporary joint outputs must restore their prior owner")

    setter = _definition(tree=tree, name="_set_transition_outputs")
    setter_source = ast.unparse(setter)
    normalizer = _definition(tree=tree, name="_transition_owner_tree")
    normalizer_source = ast.unparse(normalizer)
    if (
        "memory.set_derived(_transition_owner_tree(outputs))" not in setter_source
        or "if isinstance(tree, Mapping):" not in normalizer_source
        or "return {key: _transition_owner_tree(value) for key, value in tree.items()}"
        not in normalizer_source
    ):
        errors.append("joint mapping owners must expose every array leaf")

    laws = _definition(tree=tree, name="_validate_joint_laws")
    laws_source = ast.unparse(laws)
    ownership = _calls(node=laws, name="_own_transition_outputs")
    observed_owners = {
        (_keyword(call=call, name="outputs"), _keyword(call=call, name="restore"))
        for call in ownership
    }
    if ("weights", None) not in observed_owners:
        errors.append("weight mapping must remain owned through all joint checks")
    if (
        "owned = weights if support is None else (weights, support)" not in laws_source
        or ("owned", "weights") not in observed_owners
    ):
        errors.append("support pytree must be co-owned with its weight mapping")
    support_scopes = [
        child
        for child in ast.walk(laws)
        if isinstance(child, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _keyword(call=item.context_expr, name="outputs") == "owned"
            for item in child.items
        )
    ]
    if len(support_scopes) != 1 or not _calls(
        node=support_scopes[0], name="_validate_joint_probabilities"
    ):
        errors.append("support ownership must span probability admission")
    if "del owned, support" not in laws_source:
        errors.append("completed support locals must release after their last use")
    for name, message in (
        ("_evaluate_joint_support", "support producer must receive current memory"),
        ("_validate_joint_support", "support checks must receive current memory"),
        (
            "_validate_joint_probabilities",
            "weight checks must receive current memory",
        ),
    ):
        calls = _calls(node=laws, name=name)
        if len(calls) != 1 or _keyword(call=calls[0], name="memory") != "memory":
            errors.append(message)
    if laws_source.index("_validate_joint_support(") > laws_source.index(
        "_check_joint_support_schema("
    ):
        errors.append("support validity must precede static schema publication")

    support = _definition(tree=tree, name="_evaluate_joint_support")
    admitted = _calls(node=support, name="_evaluate_admitted_transition_producer")
    if (
        len(admitted) != 1
        or _keyword(call=admitted[0], name="function") != "func"
        or _keyword(call=admitted[0], name="arguments") != "MappingProxyType(kwargs)"
        or _keyword(call=admitted[0], name="memory") != "memory"
    ):
        errors.append("complete support provider must use admitted producer")

    support_check = _definition(tree=tree, name="_validate_joint_support")
    if any(
        _keyword(call=call, name="tree") == "support"
        for call in _calls(node=support_check, name="hold")
    ):
        errors.append("completed support must not survive its synchronous reduction")
    admitted = _calls(node=support_check, name="run_simulation_operation")
    if (
        len(admitted) != 1
        or _keyword(call=admitted[0], name="memory") != "memory"
        or _keyword(call=admitted[0], name="function") != "_support_finiteness_flags"
    ):
        errors.append("serial support finiteness must use admitted reduction")

    probability_check = _definition(tree=tree, name="_validate_joint_probabilities")
    if any(
        _keyword(call=call, name="tree") == "probs"
        for call in _calls(node=probability_check, name="hold")
    ):
        errors.append("completed weights must not survive their synchronous reduction")
    admitted = _calls(node=probability_check, name="run_simulation_operation")
    if (
        len(admitted) != 1
        or _keyword(call=admitted[0], name="memory") != "memory"
        or _keyword(call=admitted[0], name="function") != "_joint_probability_flags"
    ):
        errors.append("serial joint weights must use admitted reduction")

    if "del evaluated, weights" not in sweep_source:
        errors.append("completed weight locals must release after their last use")

    weights = _definition(tree=tree, name="_evaluate_joint_weights")
    weights_source = ast.unparse(weights)
    if (
        "function = partial(_joint_weight_law, grid_names=tuple(grid_args), func=func)"
        not in weights_source
    ):
        errors.append("joint weight evaluator must bind its complete producer")
    admitted = _calls(node=weights, name="_evaluate_admitted_transition_producer")
    if len(admitted) != 1 or _keyword(call=admitted[0], name="memory") != "memory":
        errors.append("joint weight mapping must use admitted producer")
    if (
        "n_cells = prod((array.size for array in grid_args.values())) if grid_args "
        "else None" not in weights_source
    ):
        errors.append("joint source-cell count must derive from declared grid shapes")

    weight_law = _definition(tree=tree, name="_joint_weight_law")
    weight_law_source = ast.unparse(weight_law)
    if (
        "jnp.meshgrid(" not in weight_law_source
        or "indexing='ij'" not in weight_law_source
        or "jax.vmap(" not in weight_law_source
    ):
        errors.append("joint Cartesian weight production must preserve ij order")
    return errors


def test_joint_transition_admission_contract_is_complete() -> None:
    """The checked-in joint producer route obeys its reviewed admission contract."""
    errors = _joint_transition_admission_errors(
        source=(_ROOT / _TRANSITION_CHECKS).read_text()
    )
    assert not errors, errors


@pytest.mark.parametrize(
    ("definition", "old", "new", "expected"),
    [
        (
            "_validate_transition_sequence",
            "    validate_joint_transitions_all_periods(\n        regimes=regimes,\n        flat_params=flat_params,\n        ages=ages,\n        logger=logger,\n        summary=summary,\n        process_grid_resolver=process_grid_resolver,\n        memory=memory,\n    )",
            "    validate_joint_transitions_all_periods(\n        regimes=regimes,\n        flat_params=flat_params,\n        ages=ages,\n        logger=logger,\n        summary=summary,\n        process_grid_resolver=process_grid_resolver,\n        memory=None,\n    )",
            "transition sequence must forward memory to joint validation",
        ),
        (
            "validate_joint_transitions_all_periods",
            "summary.memory if summary is not None and summary.memory is not None else memory",
            "summary.memory if summary is not None and False else memory",
            "summary memory must remain authoritative for joint laws",
        ),
        (
            "validate_joint_transitions_all_periods",
            "                        memory=current_memory,\n                    )\n                    if evaluated is None:",
            "                        memory=None,\n                    )\n                    if evaluated is None:",
            "joint weights must receive current memory",
        ),
        (
            "validate_joint_transitions_all_periods",
            "                            support_schemas=support_schemas,\n                            memory=current_memory,\n                        )",
            "                            support_schemas=support_schemas,\n                            memory=None,\n                        )",
            "joint owners must receive current memory",
        ),
        (
            "_own_transition_outputs",
            "    _set_transition_outputs(memory=memory, outputs=outputs)",
            "    _set_transition_outputs(memory=memory, outputs=())",
            "temporary joint outputs must become owners",
        ),
        (
            "_own_transition_outputs",
            "        _set_transition_outputs(memory=memory, outputs=restore)",
            "        _set_transition_outputs(memory=memory, outputs=outputs)",
            "temporary joint outputs must restore their prior owner",
        ),
        (
            "_transition_owner_tree",
            "    if isinstance(tree, Mapping):",
            "    if False:",
            "joint mapping owners must expose every array leaf",
        ),
        (
            "_validate_joint_laws",
            "with _own_transition_outputs(memory=memory, outputs=weights):",
            "with _own_transition_outputs(memory=memory, outputs=()):",
            "weight mapping must remain owned through all joint checks",
        ),
        (
            "_validate_joint_laws",
            "            owned = weights if support is None else (weights, support)",
            "            owned = weights",
            "support pytree must be co-owned with its weight mapping",
        ),
        (
            "_validate_joint_laws",
            "            with _own_transition_outputs(memory=memory, outputs=owned, restore=weights):",
            "            with _own_transition_outputs(memory=memory, outputs=owned, restore=()):",
            "support pytree must be co-owned with its weight mapping",
        ),
        (
            "_validate_joint_laws",
            "            with _own_transition_outputs(memory=memory, outputs=owned, restore=weights):",
            "            with _own_transition_outputs(memory=memory, outputs=weights, restore=weights):",
            "support ownership must span probability admission",
        ),
        (
            "_validate_joint_laws",
            "            del owned, support",
            "            del owned",
            "completed support locals must release after their last use",
        ),
        (
            "validate_joint_transitions_all_periods",
            "                        del evaluated, weights",
            "                        del evaluated",
            "completed weight locals must release after their last use",
        ),
        (
            "_evaluate_joint_support",
            "        memory=memory,\n    )",
            "        memory=None,\n    )",
            "complete support provider must use admitted producer",
        ),
        (
            "_validate_joint_support",
            "    if summary is not None:\n        summary.append(",
            "    if summary is not None:\n        memory.hold(tree=support)\n        summary.append(",
            "completed support must not survive its synchronous reduction",
        ),
        (
            "_validate_joint_support",
            "                memory=memory,\n                function=_support_finiteness_flags,",
            "                memory=None,\n                function=_support_finiteness_flags,",
            "serial support finiteness must use admitted reduction",
        ),
        (
            "_validate_joint_probabilities",
            "    if summary is not None:\n        summary.append(",
            "    if summary is not None:\n        memory.hold(tree=probs)\n        summary.append(",
            "completed weights must not survive their synchronous reduction",
        ),
        (
            "_validate_joint_probabilities",
            "            memory=memory,\n            function=_joint_probability_flags,",
            "            memory=None,\n            function=_joint_probability_flags,",
            "serial joint weights must use admitted reduction",
        ),
        (
            "_evaluate_joint_weights",
            "            memory=memory,\n        )",
            "            memory=None,\n        )",
            "joint weight mapping must use admitted producer",
        ),
        (
            "_evaluate_joint_weights",
            "n_cells = prod(array.size for array in grid_args.values()) if grid_args else None",
            "n_cells = 0 if grid_args else None",
            "joint source-cell count must derive from declared grid shapes",
        ),
        (
            "_joint_weight_law",
            'mesh = jnp.meshgrid(*(grid_args[name] for name in grid_names), indexing="ij")',
            'mesh = jnp.meshgrid(*(grid_args[name] for name in grid_names), indexing="xy")',
            "joint Cartesian weight production must preserve ij order",
        ),
    ],
)
def test_joint_transition_admission_mutation_is_rejected(
    *, definition: str, old: str, new: str, expected: str
) -> None:
    """Every independently named joint admission weakening is rejected."""
    source = (_ROOT / _TRANSITION_CHECKS).read_text()
    mutated = _mutate_definition(source=source, name=definition, old=old, new=new)
    errors = _joint_transition_admission_errors(source=mutated)
    assert expected in errors
