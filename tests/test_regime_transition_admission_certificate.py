"""Semantic controls for regime-transition producer admission."""

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


def _regime_transition_admission_errors(  # noqa: C901, PLR0912
    *, source: str
) -> list[str]:
    tree = ast.parse(source)
    errors: list[str] = []

    sequence = _definition(tree=tree, name="_validate_transition_sequence")
    calls = _calls(node=sequence, name="validate_regime_transitions_all_periods")
    if len(calls) != 1 or _keyword(call=calls[0], name="memory") != "memory":
        errors.append("transition sequence must forward memory to regime validation")

    all_periods = _definition(tree=tree, name="validate_regime_transitions_all_periods")
    calls = _calls(node=all_periods, name="_validate_regime_transition_single")
    if len(calls) != 1 or _keyword(call=calls[0], name="memory") != "memory":
        errors.append("period sweep must forward memory to each regime law")

    single = _definition(tree=tree, name="_validate_regime_transition_single")
    single_source = ast.unparse(single)
    required_single = {
        "summary memory must remain authoritative for regime laws": (
            "current_memory = summary.memory if summary is not None and "
            "summary.memory is not None else memory"
        ),
        "completed regime law must enter its owner": (
            "_check_and_release_regime_probability("
        ),
    }
    for message, expression in required_single.items():
        if expression not in single_source:
            errors.append(message)
    law_calls = _calls(node=single, name="_evaluate_regime_probability_law")
    if (
        len(law_calls) != 1
        or _keyword(call=law_calls[0], name="memory") != "current_memory"
    ):
        errors.append("regime law must receive current memory")
    owner_calls = _calls(node=single, name="_check_and_release_regime_probability")
    if (
        len(owner_calls) != 1
        or _keyword(call=owner_calls[0], name="memory") != "current_memory"
    ):
        errors.append("completed regime law must receive current memory owner")

    evaluator = _definition(tree=tree, name="_evaluate_regime_probability_law")
    evaluator_source = ast.unparse(evaluator)
    for message, expression in {
        "regime evaluator must bind the complete producer": (
            "function = partial(_regime_probability_law, "
            "grid_names=tuple(grid_args), func=func)"
        ),
        "budgeted regime evaluator must use admitted producer": (
            "_evaluate_admitted_transition_producer(function=function, "
            "arguments=arguments, memory=memory)"
        ),
        "regime evaluator must preserve every output mapping": (
            "return (MappingProxyType(probabilities), MappingProxyType(point))"
        ),
    }.items():
        if expression not in evaluator_source:
            errors.append(message)

    law = _definition(tree=tree, name="_regime_probability_law")
    law_source = ast.unparse(law)
    if (
        "jnp.meshgrid(" not in law_source
        or "indexing='ij'" not in law_source
        or "jax.vmap(" not in law_source
        or "dict(zip(grid_names, flat_arrays, strict=True))" not in law_source
    ):
        errors.append(
            "Cartesian regime production and diagnostic order must remain fused"
        )

    producer = _definition(tree=tree, name="_evaluate_admitted_transition_producer")
    producer_source = ast.unparse(producer)
    required_producer = {
        "transition producer operands must be placed": "place_simulation_arguments(",
        "transition producer buffers must be measured": (
            "measure_buffer_footprint(tree=placed)"
        ),
        "transition producer residency must be unioned": "union_buffer_footprints(",
        "transition producer external residency must be measured": (
            "resident_bytes_by_device("
        ),
        "transition output must use first selected subject device": (
            "output_sharding = simulation_value_sharding("
            "stored_sharding=jax.sharding.SingleDeviceSharding("
            "memory.subject_devices[0]), devices=(memory.subject_devices[0],))"
        ),
        "transition compiler reservation must govern admission": "plan_workspace(",
        "largest transition residency must be charged": (
            "resident_bytes=max(external.values())"
        ),
        "only admitted transition producer may dispatch": (
            "jax.block_until_ready(plan.compiled(**placed))"
        ),
    }
    for message, expression in required_producer.items():
        if expression not in producer_source:
            errors.append(message)

    owner = _definition(tree=tree, name="_check_and_release_regime_probability")
    owner_source = ast.unparse(owner)
    for message, expression in {
        "regime outputs must become temporary owners": (
            "memory.set_derived((regime_transition_probs, state_action_values))"
        ),
        "regime validation must receive current memory": (
            "_validate_regime_transition_probs("
        ),
        "temporary regime outputs must be released": (
            "finally:\n        if memory is not None:\n            memory.set_derived(())"
        ),
    }.items():
        if expression not in owner_source:
            errors.append(message)
    validation_calls = _calls(node=owner, name="_validate_regime_transition_probs")
    if (
        len(validation_calls) != 1
        or _keyword(call=validation_calls[0], name="memory") != "memory"
    ):
        errors.append("regime validation must receive current memory")

    validator = _definition(tree=tree, name="_validate_regime_transition_probs")
    admitted = _calls(node=validator, name="run_simulation_operation")
    if (
        len(admitted) != 1
        or _keyword(call=admitted[0], name="memory") != "memory"
        or _keyword(call=admitted[0], name="function") != "_regime_probability_flags"
    ):
        errors.append("serial regime diagnostics must use admitted flags")
    return errors


def test_regime_transition_admission_contract_is_complete() -> None:
    """The checked-in regime-law route obeys its reviewed admission contract."""
    errors = _regime_transition_admission_errors(
        source=(_ROOT / _TRANSITION_CHECKS).read_text()
    )
    assert not errors, errors


@pytest.mark.parametrize(
    ("definition", "old", "new", "expected"),
    [
        (
            "_validate_transition_sequence",
            "        process_grid_resolver=process_grid_resolver,\n        memory=memory,\n    )\n    validate_state_transitions_all_periods(",
            "        process_grid_resolver=process_grid_resolver,\n        memory=None,\n    )\n    validate_state_transitions_all_periods(",
            "transition sequence must forward memory to regime validation",
        ),
        (
            "validate_regime_transitions_all_periods",
            "                    process_grid_resolver=process_grid_resolver,\n                    memory=memory,",
            "                    process_grid_resolver=process_grid_resolver,\n                    memory=None,",
            "period sweep must forward memory to each regime law",
        ),
        (
            "_validate_regime_transition_single",
            "summary.memory if summary is not None and summary.memory is not None else memory",
            "summary.memory if summary is not None and False else memory",
            "summary memory must remain authoritative for regime laws",
        ),
        (
            "_validate_regime_transition_single",
            "        memory=current_memory,\n    )\n    _check_and_release_regime_probability(",
            "        memory=None,\n    )\n    _check_and_release_regime_probability(",
            "regime law must receive current memory",
        ),
        (
            "_check_and_release_regime_probability",
            "        memory.set_derived((regime_transition_probs, state_action_values))",
            "        memory.set_derived(())",
            "regime outputs must become temporary owners",
        ),
        (
            "_check_and_release_regime_probability",
            "    finally:\n        if memory is not None:\n            memory.set_derived(())",
            "    finally:\n        if memory is not None:\n            memory.set_derived(regime_transition_probs)",
            "temporary regime outputs must be released",
        ),
        (
            "_evaluate_admitted_transition_producer",
            "        resident_bytes=max(external.values()),\n    )\n    return jax.block_until_ready(plan.compiled(**placed))",
            "        resident_bytes=0,\n    )\n    return jax.block_until_ready(plan.compiled(**placed))",
            "largest transition residency must be charged",
        ),
        (
            "_evaluate_admitted_transition_producer",
            "        devices=(memory.subject_devices[0],),\n    )\n    compiler = _TransitionLawCompiler(\n        function=function,",
            "        devices=(jax.devices()[0],),\n    )\n    compiler = _TransitionLawCompiler(\n        function=function,",
            "transition output must use first selected subject device",
        ),
        (
            "_evaluate_admitted_transition_producer",
            "return jax.block_until_ready(plan.compiled(**placed))",
            "return plan.compiled(**placed)",
            "only admitted transition producer may dispatch",
        ),
        (
            "_validate_regime_transition_probs",
            "            memory=memory,\n            function=_regime_probability_flags,",
            "            memory=None,\n            function=_regime_probability_flags,",
            "serial regime diagnostics must use admitted flags",
        ),
        (
            "_regime_probability_law",
            'mesh = jnp.meshgrid(*(grid_args[name] for name in grid_names), indexing="ij")',
            'mesh = jnp.meshgrid(*(grid_args[name] for name in grid_names), indexing="xy")',
            "Cartesian regime production and diagnostic order must remain fused",
        ),
    ],
)
def test_regime_transition_admission_mutation_is_rejected(
    *, definition: str, old: str, new: str, expected: str
) -> None:
    """Every independently named regime admission weakening is rejected."""
    source = (_ROOT / _TRANSITION_CHECKS).read_text()
    mutated = _mutate_definition(source=source, name=definition, old=old, new=new)
    errors = _regime_transition_admission_errors(source=mutated)
    assert expected in errors
