"""Semantic controls for stochastic-state transition entry admission."""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_INITIAL_CONDITIONS = "src/_lcm/simulation/initial_conditions.py"
_TRANSITION_CHECKS = "src/_lcm/transition_checks.py"


def _definition(*, tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _method(*, tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1, class_name
    matches = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"{class_name}.{name}"
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


def _state_transition_admission_errors(  # noqa: C901, PLR0912
    *, initial_conditions: str, transition_checks: str
) -> list[str]:
    initial_tree = ast.parse(initial_conditions)
    transition_tree = ast.parse(transition_checks)
    errors: list[str] = []

    entry = _definition(tree=initial_tree, name="validate_simulation_inputs")
    serial_calls = [
        call
        for call in _calls(node=entry, name="_validate_transition_sequence")
        if _keyword(call=call, name="summary") == "None"
    ]
    if (
        len(serial_calls) != 1
        or _keyword(call=serial_calls[0], name="simulation_memory") != "memory"
    ):
        errors.append("serial transition replay must retain simulation memory")

    sequence = _definition(tree=transition_tree, name="_validate_transition_sequence")
    sequence_source = ast.unparse(sequence)
    if (
        "memory = summary.memory if summary is not None and summary.memory is not "
        "None else simulation_memory" not in sequence_source
    ):
        errors.append("summary memory must remain authoritative")
    state_calls = _calls(node=sequence, name="validate_state_transitions_all_periods")
    if (
        len(state_calls) != 1
        or _keyword(call=state_calls[0], name="memory") != "memory"
    ):
        errors.append("transition sequence must forward simulation memory")

    single = _definition(tree=transition_tree, name="_validate_state_transition_single")
    single_source = ast.unparse(single)
    required_single = {
        "summary memory must remain authoritative": (
            "current_memory = summary.memory if summary is not None and "
            "summary.memory is not None else memory"
        ),
        "completed state law must enter its owner": (
            "_check_and_release_state_probability(probs=probs, "
            "transition=transition, regime_name=regime_name, age=age, "
            "summary=summary, memory=current_memory)"
        ),
    }
    for message, expression in required_single.items():
        if expression not in single_source:
            errors.append(message)
    law_calls = _calls(node=single, name="_evaluate_state_probability_law")
    if (
        len(law_calls) != 1
        or _keyword(call=law_calls[0], name="memory") != "current_memory"
    ):
        errors.append("state law must receive current memory")

    ownership = _definition(
        tree=transition_tree, name="_check_and_release_state_probability"
    )
    ownership_source = ast.unparse(ownership)
    required_ownership = {
        "state probabilities must become a temporary owner": (
            "memory.set_derived(probs)"
        ),
        "state probability checks must receive current memory": (
            "_check_state_probs(probs=probs, transition=transition, "
            "regime_name=regime_name, age=age, summary=summary, memory=memory)"
        ),
        "temporary state probabilities must be released": (
            "finally:\n        if memory is not None:\n            "
            "memory.set_derived(())"
        ),
    }
    for message, expression in required_ownership.items():
        if expression not in ownership_source:
            errors.append(message)

    producer = _definition(tree=transition_tree, name="_evaluate_state_probability_law")
    producer_source = ast.unparse(producer)
    required_producer = {
        "producer operands must be placed": "place_simulation_arguments(",
        "placed operand buffers must be measured": (
            "measure_buffer_footprint(tree=placed)"
        ),
        "caller and placed buffers must be unioned": "union_buffer_footprints(",
        "external residency must be measured": "resident_bytes_by_device(",
        "compiler reservation must govern admission": "plan_workspace(",
        "largest device residency must be charged": (
            "resident_bytes=max(external.values())"
        ),
        "only the admitted program may dispatch": (
            "plan.compiled(**placed).block_until_ready()"
        ),
        "state law output must use the first selected subject device": (
            "output_sharding = simulation_value_sharding("
            "stored_sharding=jax.sharding.SingleDeviceSharding("
            "memory.subject_devices[0]), devices=(memory.subject_devices[0],))"
        ),
        "selected output placement must enter compiler admission": (
            "output_sharding=output_sharding"
        ),
    }
    for message, expression in required_producer.items():
        if expression not in producer_source:
            errors.append(message)

    law = _definition(tree=transition_tree, name="_state_probability_law")
    law_source = ast.unparse(law)
    if "jnp.meshgrid(" not in law_source or "jax.vmap(" not in law_source:
        errors.append("Cartesian state-law production must remain inside the producer")

    compile_call = _calls(
        node=_method(
            tree=transition_tree,
            class_name="_TransitionLawCompiler",
            name="__call__",
        ),
        name="jit",
    )
    if (
        len(compile_call) != 1
        or _keyword(call=compile_call[0], name="keep_unused") != "True"
        or _keyword(call=compile_call[0], name="out_shardings")
        != "self.output_sharding"
    ):
        errors.append("transition compiler must retain inputs and place every output")

    check = _definition(tree=transition_tree, name="_check_state_probs")
    admitted_checks = _calls(node=check, name="run_simulation_operation")
    if (
        len(admitted_checks) != 1
        or _keyword(call=admitted_checks[0], name="memory") != "memory"
        or _keyword(call=admitted_checks[0], name="function")
        != "_state_probability_flags"
    ):
        errors.append("serial probability diagnostics must use admitted reduction")
    return errors


def test_state_transition_admission_contract_is_complete() -> None:
    """The checked-in entry route obeys the reviewed admission contract."""
    errors = _state_transition_admission_errors(
        initial_conditions=(_ROOT / _INITIAL_CONDITIONS).read_text(),
        transition_checks=(_ROOT / _TRANSITION_CHECKS).read_text(),
    )
    assert not errors, errors


@pytest.mark.parametrize(
    ("path", "old", "new", "expected"),
    [
        (
            _INITIAL_CONDITIONS,
            (
                "            simulation_memory=memory,\n        )\n\n\n"
                "def _preflight_memory"
            ),
            (
                "            simulation_memory=None,\n        )\n\n\n"
                "def _preflight_memory"
            ),
            "serial transition replay must retain simulation memory",
        ),
        (
            _TRANSITION_CHECKS,
            (
                "        memory=memory,\n    )\n    "
                "validate_joint_transitions_all_periods("
            ),
            (
                "        memory=None,\n    )\n    "
                "validate_joint_transitions_all_periods("
            ),
            "transition sequence must forward simulation memory",
        ),
        (
            _TRANSITION_CHECKS,
            (
                "        memory=current_memory,\n    )\n    "
                "_check_and_release_state_probability("
            ),
            "        memory=None,\n    )\n    _check_and_release_state_probability(",
            "state law must receive current memory",
        ),
        (
            _TRANSITION_CHECKS,
            (
                "summary.memory\n        if summary is not None and "
                "summary.memory is not None\n        else simulation_memory"
            ),
            (
                "summary.memory\n        if summary is not None and False\n"
                "        else simulation_memory"
            ),
            "summary memory must remain authoritative",
        ),
        (
            _TRANSITION_CHECKS,
            "        memory.set_derived(probs)\n    try:",
            "        memory.set_derived(())\n    try:",
            "state probabilities must become a temporary owner",
        ),
        (
            _TRANSITION_CHECKS,
            (
                "    finally:\n        if memory is not None:\n            "
                "memory.set_derived(())"
            ),
            (
                "    finally:\n        if memory is not None:\n            "
                "memory.set_derived(probs)"
            ),
            "temporary state probabilities must be released",
        ),
        (
            _TRANSITION_CHECKS,
            "        resident_bytes=max(external.values()),",
            "        resident_bytes=0,",
            "largest device residency must be charged",
        ),
        (
            _TRANSITION_CHECKS,
            "plan.compiled(**placed).block_until_ready()",
            "plan.compiled(**placed)",
            "only the admitted program may dispatch",
        ),
        (
            _TRANSITION_CHECKS,
            "            keep_unused=True,",
            "            keep_unused=False,",
            "transition compiler must retain inputs and place every output",
        ),
        (
            _TRANSITION_CHECKS,
            "            out_shardings=self.output_sharding,",
            "            out_shardings=None,",
            "transition compiler must retain inputs and place every output",
        ),
        (
            _TRANSITION_CHECKS,
            "        devices=(memory.subject_devices[0],),",
            "        devices=(jax.devices()[0],),",
            "state law output must use the first selected subject device",
        ),
        (
            _TRANSITION_CHECKS,
            (
                "        flags = run_simulation_operation(\n"
                "            memory=memory,\n"
                "            function=_state_probability_flags,"
            ),
            (
                "        flags = run_simulation_operation(\n"
                "            memory=memory,\n"
                "            function=lambda probabilities: probabilities,"
            ),
            "serial probability diagnostics must use admitted reduction",
        ),
    ],
)
def test_state_transition_admission_mutation_is_rejected(
    *, path: str, old: str, new: str, expected: str
) -> None:
    """Each independently named weakening is rejected by the structural verifier."""
    sources = {
        _INITIAL_CONDITIONS: (_ROOT / _INITIAL_CONDITIONS).read_text(),
        _TRANSITION_CHECKS: (_ROOT / _TRANSITION_CHECKS).read_text(),
    }
    sources[path].index(old)
    sources[path] = sources[path].replace(old, new, 1)

    errors = _state_transition_admission_errors(
        initial_conditions=sources[_INITIAL_CONDITIONS],
        transition_checks=sources[_TRANSITION_CHECKS],
    )
    assert expected in errors
