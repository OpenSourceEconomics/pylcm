"""Checked continuation summaries select the actual declared program before lowering."""

import inspect
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.execution import core_program
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalOutputSpec,
    ProgramScope,
    core_program_graph,
    materialize_core_program,
    select_programs,
)
from _lcm.execution.output_layout import OutputRoleTree, StateAxesLeading
from _lcm.solution import feasibility
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult, QueryInputStatus
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm.solver_api import SIMULATION_POLICY
from lcm.typing import ReferenceName, ScalarFloat


def _plain_core(
    *,
    value: ScalarFloat,
    scale: ScalarFloat | float = 2.0,
    target__offset: ScalarFloat | float = 5.0,
) -> tuple[ScalarFloat]:
    return (value + scale + target__offset,)


def _checked_core(
    *,
    value: ScalarFloat,
    scale: ScalarFloat | float = 2.0,
    target__offset: ScalarFloat | float = 5.0,
) -> BoundedResult[tuple[ScalarFloat]]:
    count_type = getattr(feasibility, "ContinuationReadCounts", None)
    assert callable(count_type), "Checked programs need separate continuation counts."
    return replace(
        BoundedResult(
            output=_plain_core(value=value, scale=scale, target__offset=target__offset),
            bound_status=(),
        ),
        continuation_read_counts=count_type(
            below_first_live=jnp.int32(2), eligible=jnp.int32(3)
        ),
    )


def _arguments(
    context: CoreBuildContext,  # noqa: ARG001
) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
    return {"value": jnp.asarray(7.0)}


def _program(*, name: str = "main", replay: bool = False) -> CoreProgram:
    return CoreProgram(
        name=name,
        function=_plain_core,
        argument_builder=_arguments,
        requirements=CoreExecutionRequirements(),
        output_roles=(StateAxesLeading(state_names=(), shape=()),),
        disposition=CoreExecutionDisposition.PLANNED,
        scope=ProgramScope.REPLAY if replay else ProgramScope.VALUES_ONLY,
        retained_artifact_keys=(SIMULATION_POLICY,) if replay else (),
        retained_artifact_payload_types={SIMULATION_POLICY: EGMSimPolicy}
        if replay
        else {},
        replaces_program="main" if replay else None,
        internal_outputs=(InternalOutputSpec(label="value", path=(0,)),),
    )


def _with_checked(
    *, program: CoreProgram, labels: tuple[str, ...] = ("value",)
) -> CoreProgram:
    assert (
        "collect_continuation_read_counts"
        in inspect.signature(select_programs).parameters
    ), "Program selection must specialize checked continuation work."
    variant_type = getattr(core_program, "_CheckedCoreVariant", None)
    assert callable(variant_type), "A declared core needs one checked output variant."
    roles: OutputRoleTree = jax.tree.map(
        lambda leaf: StateAxesLeading(state_names=(), shape=(), dtype=leaf.dtype),
        _checked_core(value=jnp.asarray(0.0)),
    )
    return replace(
        program,
        checked_variant=variant_type(
            function=_checked_core,
            output_roles=roles,
            internal_outputs=tuple(
                InternalOutputSpec(label=label, path=("output", 0)) for label in labels
            ),
        ),
    )


def _select(
    *, graph: Mapping[str, CoreProgram], retain_replay: bool, enabled: bool
) -> Mapping[str, CoreProgram]:
    signature = inspect.signature(select_programs)
    assert "collect_continuation_read_counts" in signature.parameters, (
        "Program selection must specialize checked continuation work."
    )
    arguments = signature.bind(
        graph=graph,
        retain_replay=retain_replay,
        collect_continuation_read_counts=enabled,
    )
    return select_programs(**arguments.arguments)


@pytest.mark.parametrize("retain_replay", [False, True])
def test_checked_selection_preserves_retention_builder_and_internal_locators(
    *, retain_replay: bool
) -> None:
    """Selection changes the function and container paths before materialization."""
    main = _with_checked(program=_program())
    replay = _with_checked(program=_program(name="replay", replay=True))
    graph = MappingProxyType({"main": main, "replay": replay})
    expected_name = "replay" if retain_replay else "main"
    disabled = _select(graph=graph, retain_replay=retain_replay, enabled=False)
    selected = _select(graph=graph, retain_replay=retain_replay, enabled=True)
    assert tuple(disabled) == tuple(selected) == (expected_name,)
    assert disabled[expected_name] is graph[expected_name]
    program = selected[expected_name]
    assert program.argument_builder is graph[expected_name].argument_builder
    assert program.function is _checked_core
    assert program.requirements is graph[expected_name].requirements
    assert program.internal_outputs == (
        InternalOutputSpec(label="value", path=("output", 0)),
    )
    materialized = materialize_core_program(
        program=program,
        context=CoreBuildContext(
            state_action_space=None,
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=None,
        ),
    )
    assert materialized.function is _checked_core
    assert materialized.output_roles is program.output_roles
    assert materialized.internal_outputs == program.internal_outputs
    result = materialized.function(**materialized.arguments)
    assert isinstance(result, BoundedResult)
    assert_array_equal(result.output[0], 14.0)


@pytest.mark.parametrize("labels", [("foreign",), ("value", "value")])
def test_checked_internal_outputs_keep_unique_producer_labels(
    labels: tuple[str, ...],
) -> None:
    """A checked container cannot erase or duplicate an internal producer label."""
    assert (
        "collect_continuation_read_counts"
        in inspect.signature(select_programs).parameters
    ), "Checked selection must validate its internal-output contract."
    with pytest.raises(ValueError, match=r"[Ll]abel|[Dd]uplicate"):
        core_program_graph(
            kernel=_DCEGMPeriodKernel(
                _core_programs={
                    "main": _with_checked(program=_program(), labels=labels),
                    "replay": _program(name="replay", replay=True),
                },
                regime_name="source",
                stateful_targets=frozenset(),
                transition_target_names=(),
            )
        )


@pytest.mark.parametrize("retain_replay", [False, True])
def test_fixed_parameters_bind_both_declared_functions(*, retain_replay: bool) -> None:
    """Checked and unchecked functions receive source and qualified target params."""
    kernel = _DCEGMPeriodKernel(
        _core_programs={
            "main": _with_checked(program=_program()),
            "replay": _with_checked(program=_program(name="replay", replay=True)),
        },
        regime_name="source",
        stateful_targets=frozenset(),
        transition_target_names=("target",),
    )
    bound = kernel.with_fixed_params(
        fixed_flat_params=MappingProxyType(
            {
                "source": MappingProxyType({"scale": jnp.asarray(3.0)}),
                "target": MappingProxyType({"offset": jnp.asarray(11.0)}),
            }
        )
    )
    graph = core_program_graph(kernel=bound)
    name = "replay" if retain_replay else "main"
    ordinary = _select(graph=graph, retain_replay=retain_replay, enabled=False)[name]
    checked = _select(graph=graph, retain_replay=retain_replay, enabled=True)[name]
    result = checked.function(value=jnp.asarray(7.0))
    assert isinstance(result, BoundedResult)
    ordinary_output = ordinary.function(value=jnp.asarray(7.0))
    assert isinstance(ordinary_output, tuple)
    assert_array_equal(ordinary_output[0], 21.0)
    assert_array_equal(result.output[0], 21.0)


@pytest.mark.parametrize("with_query", [False, True])
def test_absent_continuation_counts_keep_legacy_flattened_children(
    *, with_query: bool
) -> None:
    """Disabled continuation summaries preserve the original children and aux."""
    query = (
        (QueryInputStatus(*(jnp.asarray(a=False) for _ in range(3))),)
        if with_query
        else None
    )
    result = BoundedResult(
        output=(jnp.asarray(7.0),), bound_status=(), query_status=query
    )
    children, aux = feasibility._flatten_bounded_result(result)
    assert aux is None
    assert len(children) == (3 if with_query else 2)
    assert children[0] is result.output
    assert children[1] is result.bound_status
    if with_query:
        assert children[2] is result.query_status


@pytest.mark.parametrize("with_query", [False, True])
def test_continuation_counts_roundtrip_without_becoming_query_status(
    *, with_query: bool
) -> None:
    """The optional integer pair and mandatory query flags retain separate roles."""
    assert "continuation_read_counts" in BoundedResult.__dataclass_fields__, (
        "Bounded transport needs a separate optional continuation count record."
    )
    result = _checked_core(value=jnp.asarray(7.0))
    if with_query:
        result = replace(
            result,
            query_status=(QueryInputStatus(*(jnp.asarray(a=False) for _ in range(3))),),
        )
    leaves, tree = jax.tree.flatten(result)
    restored = jax.tree.unflatten(tree, leaves)
    assert isinstance(restored, BoundedResult)
    assert len(leaves) == (6 if with_query else 3)
    counts = getattr(restored, "continuation_read_counts", None)
    assert counts is not None
    assert_array_equal(counts.below_first_live, 2)
    assert_array_equal(counts.eligible, 3)
    assert counts.below_first_live.shape == counts.eligible.shape == ()
    assert counts.below_first_live.dtype == counts.eligible.dtype == jnp.int32
    assert (restored.query_status is None) is (not with_query)
