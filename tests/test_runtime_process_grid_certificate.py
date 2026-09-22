"""Semantic controls for staged runtime process support admission."""

import ast
from itertools import pairwise
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_PROCESS_GRIDS = "src/_lcm/simulation/process_grids.py"


def _definition(*, tree: ast.Module, name: str) -> ast.FunctionDef:
    if "." not in name:
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
    else:
        owner, function = name.split(".", maxsplit=1)
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == owner
        ]
        assert len(classes) == 1, owner
        matches = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == function
        ]
    assert len(matches) == 1, name
    return matches[0]


def _mutate_definition(*, source: str, name: str, old: str, new: str) -> str:
    tree = ast.parse(source)
    node = _definition(tree=tree, name=name)
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    definition_source = "".join(lines[start:end])
    assert definition_source.count(old) == 1
    lines[start:end] = [definition_source.replace(old, new)]
    return "".join(lines)


def _runtime_process_admission_errors(  # noqa: C901, PLR0912, PLR0915
    *, source: str
) -> list[str]:
    tree = ast.parse(source)
    errors: list[str] = []

    supports = ast.unparse(
        _definition(tree=tree, name="SimulationProcessGrids.supports")
    )
    if (
        "type(spec) is NormalIIDProcess and spec.gauss_hermite" not in supports
        or "type(spec) in _STAGED_PROCESS_TYPES" not in supports
    ):
        errors.append("every exact composite built-in must use the staged profile")

    call = ast.unparse(_definition(tree=tree, name="SimulationProcessGrids.__call__"))
    for expression, message in (
        ("_process_fixed_identity(spec)", "fixed composite identity must remain exact"),
        (
            "_staged_parameter_is_weak(value)",
            "composite scalar strength must remain in support identity",
        ),
        (
            "self._produce_staged(spec=spec, parameters=complete, required=required)",
            "composite support must use staged admission",
        ),
    ):
        if expression not in call:
            errors.append(message)

    fixed_identity = ast.unparse(_definition(tree=tree, name="_process_fixed_identity"))
    if (
        "type(value)" not in fixed_identity
        or "_parameter_bytes(value)" not in fixed_identity
    ):
        errors.append("fixed composite type and bytes must remain in identity")
    weak_identity = ast.unparse(
        _definition(tree=tree, name="_staged_parameter_is_weak")
    )
    if (
        "type(value) in (bool, int, float)" not in weak_identity
        or "getattr(value, 'weak_type', False)" not in weak_identity
    ):
        errors.append("host and JAX scalar strength must remain in identity")
    complete = ast.unparse(_definition(tree=tree, name="_complete_process_parameters"))
    if (
        "np.asarray(value, dtype=np.int32) if isinstance(value, bool | int) else value"
        not in complete
    ):
        errors.append("fixed composite scalar promotion must match eager semantics")

    staged = ast.unparse(
        _definition(tree=tree, name="SimulationProcessGrids._produce_staged")
    )
    staged_node = _definition(tree=tree, name="SimulationProcessGrids._produce_staged")
    ordered_stages = (
        "parameter_names = tuple(sorted(parameters))",
        "closed = _trace_process_jaxpr(",
        "recipe = _validated_process_recipe(",
        "for stage in recipe.stages:",
    )
    if not all(expression in staged for expression in ordered_stages) or not all(
        staged.index(left) < staged.index(right)
        for left, right in pairwise(ordered_stages)
    ):
        errors.append("the complete graph must preflight before the first stage")
    recipe_calls = [
        node
        for node in ast.walk(staged_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validated_process_recipe"
    ]
    if (
        len(recipe_calls) != 1
        or len(recipe_calls[0].keywords) != 3
        or not any(
            keyword.arg == "closed"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "closed"
            for keyword in recipe_calls[0].keywords
        )
    ):
        errors.append("the validated recipe must receive the complete graph")
    for expression, message in (
        ("process_stage=True", "staged arithmetic must select its admitted executor"),
        (
            "weak_type=stage.weak_type",
            "conversion strength must flow from recipe to admission",
        ),
        (
            "self.temporary_roots.append(output)",
            "every completed stage must remain in the live inventory",
        ),
    ):
        if expression not in staged:
            errors.append(message)
    if not any(
        isinstance(node, ast.Try)
        and any(
            ast.unparse(statement) == "self.temporary_roots.clear()"
            for statement in node.finalbody
        )
        for node in ast.walk(staged_node)
    ):
        errors.append("staged roots must release on success and failure")

    trace = ast.unparse(_definition(tree=tree, name="_trace_process_jaxpr"))
    call = ast.unparse(_definition(tree=tree, name="_process_grid_call"))
    if (
        "dict(zip(parameter_names, values, strict=True))" not in call
        or "partial(_process_grid_call, spec=spec, parameter_names=parameter_names)"
        not in trace
        or "jax.make_jaxpr(bound)(*parameter_values)" not in trace
    ):
        errors.append("traced inputs must use one explicit positional order")

    recipe = ast.unparse(_definition(tree=tree, name="_validated_process_recipe"))
    for expression, message in (
        ("check_jaxpr(closed)", "the complete JAXPR must pass structural validation"),
        ("for equation in closed.eqns:", "every top-level stage must be validated"),
        (
            "_validate_process_equation(equation=equation, n_points=n_points)",
            "every stage's static semantics must be validated",
        ),
        (
            "_aval_shape(closed.outvars[0]) != (n_points,)",
            "the final support shape must equal its declared extent",
        ),
        (
            "initial_values = (*closed.consts, *parameter_values)",
            "host constants must enter the admitted operand recipe",
        ),
        (
            "host_constant=True",
            "captured constants must validate before stage dispatch",
        ),
        (
            "host_constant=False",
            "attached parameters must validate before stage dispatch",
        ),
        ("stages=tuple(stages)", "replay must consume one immutable stage tuple"),
    ):
        if expression not in recipe:
            errors.append(message)

    attached = ast.unparse(
        _definition(tree=tree, name="_validate_attached_process_value")
    )
    attached_schema = (
        "(getattr(observed, 'shape', None), "
        "str(getattr(observed, 'dtype', None)), "
        "bool(getattr(observed, 'weak_type', False))) != _aval_schema(variable)"
    )
    if (
        "isinstance(value, jax.Array)" not in attached
        or "observed = jax.typeof(value)" not in attached
        or attached_schema not in attached
    ):
        errors.append("attached values must match host and abstract input contracts")

    equation = ast.unparse(_definition(tree=tree, name="_validate_process_equation"))
    if (
        "equation.params.get('name') != '_linspace'" not in equation
        or "_validate_linspace_jaxpr(" not in equation
    ):
        errors.append("nested JITs must match the reviewed eager linspace")
    if (
        "type(weak_type) is not bool" not in equation
        or "weak_type != getattr(equation.outvars[0].aval, 'weak_type', None)"
        not in equation
        or "output_dtype != str(np.dtype(canonical_float_dtype()))" not in equation
    ):
        errors.append("conversion recipe must retain exact weak output semantics")

    linspace = ast.unparse(_definition(tree=tree, name="_validate_linspace_jaxpr"))
    if (
        "_root_equation_schema(equation) != _root_equation_schema(reference.eqns[0])"
        not in linspace
    ):
        errors.append("nested linspace must match its complete reference schema")

    graph = ast.unparse(_definition(tree=tree, name="_jaxpr_schema"))
    equation_schema = ast.unparse(_definition(tree=tree, name="_equation_schema"))
    atom_schema = ast.unparse(_definition(tree=tree, name="_graph_atom_schema"))
    if (
        "variables[variable] = len(variables)" not in graph
        or equation_schema.count("_graph_atom_schema(") != 2
        or "variables[atom]" not in atom_schema
    ):
        errors.append("nested graph schema must preserve canonical dataflow")

    produce = ast.unparse(
        _definition(tree=tree, name="SimulationProcessGrids._produce")
    )
    for expression, message in (
        ("if process_stage:", "composite stages need a distinct dispatch selector"),
        (
            "'exponent': exponent",
            "process power semantics must remain in compilation identity",
        ),
        (
            "'dtype': dtype",
            "process conversion dtype must remain in compilation identity",
        ),
        (
            "'weak_type': weak_type",
            "process conversion strength must remain in compilation identity",
        ),
        (
            "output_sharding=required",
            "every process stage must emit on the selected devices",
        ),
        (
            "plan.compiled.executable(**placed).block_until_ready()",
            "each stage must complete before its inputs can release",
        ),
    ):
        if expression not in produce:
            errors.append(message)

    operation = ast.unparse(_definition(tree=tree, name="_compute_process_stage"))
    if (
        "return jnp.exp(value)" not in operation
        or "return jnp.linspace(" not in operation
    ):
        errors.append("staged execution must preserve vector and linspace arithmetic")
    if "return value * 1.0 if weak_type else jnp.asarray(" not in operation:
        errors.append("conversion execution must preserve traced scalar strength")
    return errors


def test_runtime_process_admission_contract_is_complete() -> None:
    """The checked-in staged process route obeys its reviewed contract."""
    errors = _runtime_process_admission_errors(
        source=(_ROOT / _PROCESS_GRIDS).read_text()
    )
    assert not errors, errors


@pytest.mark.parametrize(
    ("definition", "old", "new", "expected"),
    [
        (
            "SimulationProcessGrids.supports",
            "or (type(spec) is NormalIIDProcess and spec.gauss_hermite)",
            "or False",
            "every exact composite built-in must use the staged profile",
        ),
        (
            "SimulationProcessGrids.supports",
            "or type(spec) in _STAGED_PROCESS_TYPES",
            "or False",
            "every exact composite built-in must use the staged profile",
        ),
        (
            "SimulationProcessGrids.__call__",
            "_process_fixed_identity(spec)",
            "()",
            "fixed composite identity must remain exact",
        ),
        (
            "SimulationProcessGrids.__call__",
            "_staged_parameter_is_weak(value)",
            "False",
            "composite scalar strength must remain in support identity",
        ),
        (
            "_process_fixed_identity",
            "type(value)",
            "None",
            "fixed composite type and bytes must remain in identity",
        ),
        (
            "_process_fixed_identity",
            "_parameter_bytes(value)",
            "None",
            "fixed composite type and bytes must remain in identity",
        ),
        (
            "_staged_parameter_is_weak",
            "type(value) in (bool, int, float)",
            "False",
            "host and JAX scalar strength must remain in identity",
        ),
        (
            "_staged_parameter_is_weak",
            'getattr(value, "weak_type", False)',
            "False",
            "host and JAX scalar strength must remain in identity",
        ),
        (
            "_complete_process_parameters",
            "np.asarray(value, dtype=np.int32)",
            "value",
            "fixed composite scalar promotion must match eager semantics",
        ),
        (
            "_complete_process_parameters",
            "else value",
            "else np.asarray(value, dtype=canonical_float_dtype())",
            "fixed composite scalar promotion must match eager semantics",
        ),
        (
            "SimulationProcessGrids.__call__",
            (
                "self._produce_staged(\n                    spec=spec, "
                "parameters=complete, required=required\n                )"
            ),
            "spec.compute_gridpoints(**complete)",
            "composite support must use staged admission",
        ),
        (
            "SimulationProcessGrids._produce_staged",
            "tuple(sorted(parameters))",
            "tuple(parameters)",
            "the complete graph must preflight before the first stage",
        ),
        (
            "SimulationProcessGrids._produce_staged",
            "closed=closed",
            "closed=closed.replace(eqns=closed.eqns[:1])",
            "the validated recipe must receive the complete graph",
        ),
        (
            "SimulationProcessGrids._produce_staged",
            "process_stage=True",
            "process_stage=False",
            "staged arithmetic must select its admitted executor",
        ),
        (
            "SimulationProcessGrids._produce_staged",
            "weak_type=stage.weak_type",
            "weak_type=False",
            "conversion strength must flow from recipe to admission",
        ),
        (
            "SimulationProcessGrids._produce_staged",
            "self.temporary_roots.append(output)",
            "None",
            "every completed stage must remain in the live inventory",
        ),
        (
            "SimulationProcessGrids._produce_staged",
            "finally:\n            self.temporary_roots.clear()",
            "finally:\n            pass",
            "staged roots must release on success and failure",
        ),
        (
            "_process_grid_call",
            "strict=True",
            "strict=False",
            "traced inputs must use one explicit positional order",
        ),
        (
            "_validated_process_recipe",
            "check_jaxpr(closed)",
            "None",
            "the complete JAXPR must pass structural validation",
        ),
        (
            "_validated_process_recipe",
            "for equation in closed.eqns:",
            "for equation in closed.eqns[:1]:",
            "every top-level stage must be validated",
        ),
        (
            "_validated_process_recipe",
            "_validate_process_equation(equation=equation, n_points=n_points)",
            "None",
            "every stage's static semantics must be validated",
        ),
        (
            "_validated_process_recipe",
            "_aval_shape(closed.outvars[0]) != (n_points,)",
            "False",
            "the final support shape must equal its declared extent",
        ),
        (
            "_validated_process_recipe",
            "initial_values = (*closed.consts, *parameter_values)",
            "initial_values = parameter_values",
            "host constants must enter the admitted operand recipe",
        ),
        (
            "_validated_process_recipe",
            "host_constant=True",
            "host_constant=False",
            "captured constants must validate before stage dispatch",
        ),
        (
            "_validated_process_recipe",
            "host_constant=False",
            "host_constant=True",
            "attached parameters must validate before stage dispatch",
        ),
        (
            "_validate_attached_process_value",
            "isinstance(value, jax.Array)",
            "False",
            "attached values must match host and abstract input contracts",
        ),
        (
            "_validate_attached_process_value",
            "observed = jax.typeof(value)",
            "observed = variable.aval",
            "attached values must match host and abstract input contracts",
        ),
        (
            "_validate_attached_process_value",
            'bool(getattr(observed, "weak_type", False)),',
            "_aval_schema(variable)[2],",
            "attached values must match host and abstract input contracts",
        ),
        (
            "_validated_process_recipe",
            "stages=tuple(stages)",
            "stages=()",
            "replay must consume one immutable stage tuple",
        ),
        (
            "_validate_process_equation",
            'equation.params.get("name") != "_linspace"',
            "False",
            "nested JITs must match the reviewed eager linspace",
        ),
        (
            "_validate_process_equation",
            "type(weak_type) is not bool",
            "False",
            "conversion recipe must retain exact weak output semantics",
        ),
        (
            "_validate_linspace_jaxpr",
            "_root_equation_schema(reference.eqns[0])",
            "()",
            "nested linspace must match its complete reference schema",
        ),
        (
            "_jaxpr_schema",
            "variables[variable] = len(variables)",
            "variables[variable] = 0",
            "nested graph schema must preserve canonical dataflow",
        ),
        (
            "_graph_atom_schema",
            "variables[atom]",
            "0",
            "nested graph schema must preserve canonical dataflow",
        ),
        (
            "SimulationProcessGrids._produce",
            "if process_stage:",
            "if False:",
            "composite stages need a distinct dispatch selector",
        ),
        (
            "SimulationProcessGrids._produce",
            '"exponent": exponent',
            '"exponent": 1',
            "process power semantics must remain in compilation identity",
        ),
        (
            "SimulationProcessGrids._produce",
            '"dtype": dtype',
            '"dtype": "float32"',
            "process conversion dtype must remain in compilation identity",
        ),
        (
            "SimulationProcessGrids._produce",
            '"weak_type": weak_type',
            '"weak_type": False',
            "process conversion strength must remain in compilation identity",
        ),
        (
            "SimulationProcessGrids._produce",
            "output_sharding=required",
            "output_sharding=None",
            "every process stage must emit on the selected devices",
        ),
        (
            "SimulationProcessGrids._produce",
            "plan.compiled.executable(**placed).block_until_ready()",
            "plan.compiled.executable(**placed)",
            "each stage must complete before its inputs can release",
        ),
        (
            "_compute_process_stage",
            "return jnp.exp(value)",
            "return value",
            "staged execution must preserve vector and linspace arithmetic",
        ),
        (
            "_compute_process_stage",
            "value * 1.0 if weak_type else",
            "value if weak_type else",
            "conversion execution must preserve traced scalar strength",
        ),
    ],
)
def test_runtime_process_admission_mutation_is_rejected(
    *, definition: str, old: str, new: str, expected: str
) -> None:
    """Every independently named staged process weakening is rejected."""
    source = (_ROOT / _PROCESS_GRIDS).read_text()
    mutated = _mutate_definition(source=source, name=definition, old=old, new=new)
    errors = _runtime_process_admission_errors(source=mutated)
    assert expected in errors
