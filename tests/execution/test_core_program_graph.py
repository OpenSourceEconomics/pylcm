"""The native graph and centralized legacy adapter are the only core authority."""

import ast
import inspect
import textwrap
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    CoreProgramGraphAware,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.output_layout import VALUE
from _lcm.solution import backward_induction, period_replay
from _lcm.solution.backward_induction import _resolve_program_for_execution


def _identity(*, value: object) -> object:
    return value


def _double(*, value: object) -> object:
    return value


def _context() -> CoreBuildContext:
    return CoreBuildContext(
        state_action_space=object(),
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        ages=object(),
    )


class _NativeKernel:
    def __init__(self, graph: Mapping[str, CoreProgram]) -> None:
        self.graph = graph

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return self.graph


class _LegacyKernel:
    def __init__(self, cores: Mapping[str, Callable[..., object]]) -> None:
        self._cores = cores
        self.built: list[str] = []

    def cores(self) -> Mapping[str, Callable[..., object]]:
        return self._cores

    def build_lower_args(
        self, *, core_key: str, **_context: object
    ) -> Mapping[str, object]:
        self.built.append(core_key)
        return MappingProxyType({"value": jnp.asarray(1.0)})


def _native_program(
    *, name: str = "main", reason: str = "test_dense_route"
) -> CoreProgram:
    return CoreProgram(
        name=name,
        function=_identity,
        argument_builder=lambda _context: MappingProxyType({"value": jnp.asarray(1.0)}),
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason=reason,
    )


def test_native_graph_is_snapshotted_and_materialized_through_one_builder() -> None:
    declaration = _native_program()
    source = {"main": declaration}
    kernel = _NativeKernel(source)

    assert isinstance(kernel, CoreProgramGraphAware)
    graph = core_program_graph(kernel=kernel)
    source["other"] = _native_program(name="other")

    assert tuple(graph) == ("main",)
    with pytest.raises(TypeError):
        cast("dict[str, CoreProgram]", graph)["injected"] = declaration

    materialized = materialize_core_program(program=graph["main"], context=_context())
    resolved = resolve_core_program(program=materialized, tile_widths={})

    assert resolved.name == "main"
    assert resolved.function is declaration.function
    assert resolved.disposition is CoreExecutionDisposition.DENSE
    assert resolved.disposition_reason == "test_dense_route"
    assert resolved.output_roles is VALUE
    assert tuple(resolved.arguments) == ("value",)


def test_shared_resolver_preserves_the_entire_native_program_contract() -> None:
    declaration = _native_program()

    resolved_by_mode = {
        mode: _resolve_program_for_execution(
            program=materialize_core_program(
                program=declaration,
                context=_context(),
            ),
            source_value_template=jnp.asarray(0.0),
            source=("source", 0, "main"),
        )
        for mode in ("eager", "aot", "period-replay")
    }

    reference = resolved_by_mode["eager"]
    for resolved in resolved_by_mode.values():
        assert resolved.function is declaration.function
        assert resolved.requirements is declaration.requirements
        assert resolved.output_roles is declaration.output_roles
        assert resolved.disposition is declaration.disposition
        assert resolved.disposition_reason == declaration.disposition_reason
        assert resolved.specialization_key == reference.specialization_key
        assert tuple(resolved.arguments) == tuple(reference.arguments)
        assert jnp.array_equal(
            cast("jax.Array", resolved.arguments["value"]),
            cast("jax.Array", reference.arguments["value"]),
        )


def test_eager_aot_and_replay_entry_paths_cross_the_same_resolution_seam() -> None:
    compile_tree = _function_tree(backward_induction._compile_all_functions)
    collect_tree = _function_tree(
        backward_induction._resolve_output_layouts_and_lowering_keys
    )
    replay_tree = _function_tree(period_replay._compile_cores_for_one_period)

    compile_calls = _direct_call_lines(compile_tree)
    collect_calls = _direct_call_lines(collect_tree)
    replay_calls = _direct_call_lines(replay_tree)

    assert len(compile_calls["core_program_graph"]) == 1
    assert len(compile_calls["_resolve_output_layouts_and_lowering_keys"]) == 1
    assert len(collect_calls["materialize_core_program"]) == 1
    assert len(collect_calls["_resolve_program_for_execution"]) == 1
    assert len(replay_calls["core_program_graph"]) == 1
    assert len(replay_calls["materialize_core_program"]) == 1
    assert len(replay_calls["_resolve_program_for_execution"]) == 1

    eager_branch = next(
        node
        for node in ast.walk(compile_tree)
        if isinstance(node, ast.If) and _is_not_enable_jit(node.test)
    )
    assert (
        compile_calls["core_program_graph"][0]
        < compile_calls["_resolve_output_layouts_and_lowering_keys"][0]
        < eager_branch.lineno
    )


def _function_tree(function: Callable[..., object]) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    function_node = tree.body[0]
    assert isinstance(function_node, ast.FunctionDef)
    return function_node


def _direct_call_lines(tree: ast.FunctionDef) -> dict[str, list[int]]:
    calls: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node.lineno)
    return calls


def _is_not_enable_jit(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "enable_jit"
    )


def test_dense_reason_is_part_of_resolved_specialization_identity() -> None:
    context = _context()
    first = _resolve_program_for_execution(
        program=materialize_core_program(
            program=_native_program(reason="first_dense_reason"), context=context
        ),
        source_value_template=jnp.asarray(0.0),
        source=("source", 0, "main"),
    )
    second = _resolve_program_for_execution(
        program=materialize_core_program(
            program=_native_program(reason="second_dense_reason"), context=context
        ),
        source_value_template=jnp.asarray(0.0),
        source=("source", 0, "main"),
    )

    assert first.specialization_key != second.specialization_key


def test_native_graph_key_and_declared_name_must_match() -> None:
    kernel = _NativeKernel({"wrong": _native_program(name="main")})

    with pytest.raises(ValueError, match="key must equal"):
        core_program_graph(kernel=kernel)


@pytest.mark.parametrize(
    "method_name",
    [
        "cores",
        "core",
        "unwrapped_core",
        "streamed_core",
        "build_lower_args",
        "build_core_program",
        "target_value_accesses",
        "output_roles",
        "core_for_output_layout",
    ],
)
def test_native_graph_rejects_every_parallel_declaration_seam(
    method_name: str,
) -> None:
    kernel = _NativeKernel({"main": _native_program()})
    setattr(kernel, method_name, lambda **_kwargs: None)

    with pytest.raises(TypeError, match="duplicate execution authorities"):
        core_program_graph(kernel=kernel)


@pytest.mark.parametrize(
    ("disposition", "reason", "message"),
    [
        pytest.param(
            CoreExecutionDisposition.DENSE,
            None,
            "must declare a non-empty",
            id="dense-without-reason",
        ),
        pytest.param(
            CoreExecutionDisposition.PLANNED,
            "unexpected",
            "cannot declare a reason",
            id="planned-with-reason",
        ),
    ],
)
def test_native_graph_rejects_incoherent_disposition_reason(
    *,
    disposition: CoreExecutionDisposition,
    reason: str | None,
    message: str,
) -> None:
    program = CoreProgram(
        name="main",
        function=_identity,
        argument_builder=lambda _context: MappingProxyType({"value": jnp.asarray(1.0)}),
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=disposition,
        disposition_reason=reason,
    )

    with pytest.raises(ValueError, match=message):
        core_program_graph(kernel=_NativeKernel({"main": program}))


@pytest.mark.parametrize(
    "cores",
    [
        pytest.param({"main": _identity}, id="single"),
        pytest.param({"first": _identity, "second": _double}, id="multi"),
    ],
)
def test_legacy_adapter_synthesizes_one_explicit_unplanned_graph(
    cores: Mapping[str, Callable[..., object]],
) -> None:
    kernel = _LegacyKernel(cores)
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == tuple(cores)
    assert all(
        program.disposition is CoreExecutionDisposition.LEGACY_UNPLANNED
        for program in graph.values()
    )
    assert all(program.output_roles is None for program in graph.values())
    assert all(
        program.disposition_reason == "legacy_adapter" for program in graph.values()
    )

    for name, program in graph.items():
        materialized = materialize_core_program(program=program, context=_context())
        resolved = resolve_core_program(program=materialized, tile_widths={})
        assert resolved.function is cores[name]
    assert kernel.built == list(cores)


@pytest.mark.parametrize(
    "method_name",
    ["build_core_program", "output_roles", "core_for_output_layout"],
)
def test_legacy_adapter_rejects_a_surviving_duplicate_authority(
    method_name: str,
) -> None:
    kernel = _LegacyKernel({"main": _identity})
    setattr(kernel, method_name, lambda **_kwargs: None)

    with pytest.raises(TypeError, match="duplicate"):
        core_program_graph(kernel=kernel)


def test_materialization_rejects_unknown_donation_candidate() -> None:
    declaration = CoreProgram(
        name="main",
        function=_identity,
        argument_builder=lambda _context: MappingProxyType({"value": jnp.asarray(1.0)}),
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        donation_candidates=("missing",),
        disposition_reason="test_dense_route",
    )

    with pytest.raises(ValueError, match="absent"):
        materialize_core_program(program=declaration, context=_context())
