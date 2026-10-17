"""Bounded producer locators and period-owned handover keep their separate seams."""

import functools
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ContinuationPayload
from _lcm.engine import Regime, StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalInputRef,
    InternalOutputSpec,
    MaterializedCoreProgram,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
    select_programs,
)
from _lcm.execution.internal_outputs import (
    internal_input_templates,
    resolve_producer,
)
from _lcm.execution.output_layout import (
    VALUE,
    OutputRoleLeaf,
    OutputRoleTree,
    PlannedCore,
    StateAxesLeading,
    resolve_output_layout,
)
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution.backward_induction import _run_period_kernel
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    FeasibilityKernelOutput,
    QueryInputStatus,
    validate_query_input_status,
)
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue, ShapeDtypePytree
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import KernelOutput
from lcm.typing import FloatND, ReferenceName, RegimeName
from tests.execution.test_internal_outputs import _context, _graph, _Kernel
from tests.solution.test_grid_search_streaming_production_path import _build_model

type _Payload = tuple[FloatND, dict[str, FloatND]]


def _bounded_producer(*, x: FloatND) -> BoundedResult[_Payload]:
    return BoundedResult(output=(x + 1.0, {"carry": x * 2.0}), bound_status=())


def test_bounded_internal_paths_supply_abstract_consumer_templates() -> None:
    """The declared ordinary subtree is reached through the known output field."""
    programs = _graph()
    programs["producer"] = replace(
        programs["producer"],
        function=_bounded_producer,
        output_roles=cast(
            "OutputRoleTree",
            jax.tree.map_with_path(_role, _bounded_producer(x=jnp.zeros((3,)))),
        ),
        internal_outputs=(
            InternalOutputSpec(label="value", path=("output", 0)),
            InternalOutputSpec(label="carry", path=("output", 1)),
        ),
    )
    graph = core_program_graph(kernel=_Kernel(programs=programs))
    materialized = {
        name: materialize_core_program(program=program, context=_context())
        for name, program in graph.items()
    }
    producer = resolve_core_program(program=materialized["producer"], tile_widths={})
    record = resolve_producer(program=producer, templates=MappingProxyType({}))
    templates = internal_input_templates(
        program=materialized["consumer"],
        producers={"producer": MappingProxyType({(): record})},
    )
    value_template = templates["upstream_value"]
    assert isinstance(value_template, jax.ShapeDtypeStruct)
    assert value_template.shape == (3,)
    carry_template = templates["upstream_carry"]
    assert isinstance(carry_template, dict)
    carry_leaf = carry_template["carry"]
    assert isinstance(carry_leaf, jax.ShapeDtypeStruct)
    assert carry_leaf.shape == (3,)
    assert isinstance(record.abstract_output, BoundedResult)
    assert value_template.dtype == record.abstract_output.output[0].dtype


def _ordinary(*, x: FloatND, fault: bool) -> BoundedResult[_Payload]:
    return BoundedResult(
        output=(x + 1.0, {"carry": x * 2.0}),
        bound_status=(BoundInputStatus(jnp.asarray(a=False), jnp.asarray(a=False)),),
        query_status=(
            QueryInputStatus(
                jnp.asarray(a=False), jnp.asarray(a=False), jnp.asarray(a=fault)
            ),
        ),
    )


def _checked(*, x: FloatND, fault: bool) -> BoundedResult[_Payload]:
    """Attach distinct structural constants without claiming numerical read events."""
    from _lcm.solution.feasibility import ContinuationReadCounts  # noqa: PLC0415

    return replace(
        _ordinary(x=x, fault=fault),
        continuation_read_counts=ContinuationReadCounts(jnp.int32(2), jnp.int32(3)),
    )


def _consumer(
    *, x: FloatND, upstream_value: FloatND, upstream_carry: dict[str, FloatND]
) -> FloatND:
    return x + upstream_value + upstream_carry["carry"]


def _arguments(
    context: CoreBuildContext,
) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
    assert context.state_action_space is not None
    return {"x": context.state_action_space.states["wealth"]}


# keyword-only-exempt: library-callback=jax.tree.map_with_path
def _role(path: jax.tree_util.KeyPath, leaf: jax.Array) -> OutputRoleLeaf:
    if path == (jax.tree_util.GetAttrKey("output"), jax.tree_util.SequenceKey(0)):
        return VALUE
    return StateAxesLeading(
        state_names=("wealth",) if leaf.shape else (),
        dtype=leaf.dtype,
        shape=leaf.shape,
    )


def _programs(*, fault: bool) -> Mapping[str, CoreProgram]:
    from _lcm.execution.core_program import _CheckedCoreVariant  # noqa: PLC0415

    nodes = jnp.zeros((2,))
    locators = (
        InternalOutputSpec(label="value", path=("output", 0)),
        InternalOutputSpec(label="carry", path=("output", 1)),
    )
    ordinary_roles = cast(
        "OutputRoleTree", jax.tree.map_with_path(_role, _ordinary(x=nodes, fault=fault))
    )
    checked_roles = cast(
        "OutputRoleTree", jax.tree.map_with_path(_role, _checked(x=nodes, fault=fault))
    )
    return MappingProxyType(
        {
            "producer": CoreProgram(
                name="producer",
                function=functools.partial(_ordinary, fault=fault),
                argument_builder=_arguments,
                requirements=CoreExecutionRequirements(),
                output_roles=ordinary_roles,
                disposition=CoreExecutionDisposition.DENSE,
                disposition_reason="deliberately_dense:test",
                internal_outputs=locators,
                checked_variant=_CheckedCoreVariant(
                    function=functools.partial(_checked, fault=fault),
                    output_roles=checked_roles,
                    internal_outputs=locators,
                ),
            ),
            "consumer": CoreProgram(
                name="consumer",
                function=_consumer,
                argument_builder=_arguments,
                requirements=CoreExecutionRequirements(
                    internal_inputs={
                        "upstream_value": InternalInputRef(
                            producer="producer", label="value"
                        ),
                        "upstream_carry": InternalInputRef(
                            producer="producer", label="carry"
                        ),
                    }
                ),
                output_roles=VALUE,
                disposition=CoreExecutionDisposition.DENSE,
                disposition_reason="deliberately_dense:test",
            ),
        }
    )


@dataclass(frozen=True, kw_only=True)
class _Handover:
    """Period-owned decoding hands known ordinary fields to the selected cores."""

    programs: Mapping[str, CoreProgram]
    observed: list[BoundedResult[_Payload]]

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _Handover:
        assert not fixed_flat_params
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],  # noqa: ARG002
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],  # noqa: ARG002
        flat_params: FlatParams,  # noqa: ARG002
        period: int,  # noqa: ARG002
        ages: TimeAxis,  # noqa: ARG002
        logger: logging.Logger,  # noqa: ARG002
    ) -> KernelOutput:
        x = state_action_space.states["wealth"]
        produced = compiled_cores["producer"](x=x)
        assert isinstance(produced, BoundedResult)
        raw = cast("BoundedResult[_Payload]", produced)
        self.observed.append(raw)
        value, carry = raw.output
        result = compiled_cores["consumer"](
            x=x, upstream_value=value, upstream_carry=carry
        )
        assert isinstance(result, jax.Array)
        return FeasibilityKernelOutput(
            value=result,
            feasibility_counts=(jnp.int32(0), jnp.int32(0)),
            bound_status=raw.bound_status,
            query_status=raw.query_status,
        )


def _authority() -> PostDecisionBoundSpec:
    return PostDecisionBoundSpec(
        post_decision="savings",
        liquid_state="wealth",
        action="consumption",
        resources_target="resources",
        kind="literal",
        literal_value=0.0,
        lower_target=None,
        lower_inputs=(),
        dependency_identity=(((), 0.0),),
    )


def _planned_cores(
    *,
    materialized: Mapping[str, MaterializedCoreProgram],
    x: FloatND,
    templates: Mapping[str, ShapeDtypePytree],
) -> dict[str, PlannedCore]:
    compiled_cores = {}
    for name, program in materialized.items():
        layout = resolve_output_layout(
            core_key=name,
            value_template=x,
            state_order=("wealth",),
            output_roles=program.output_roles,
        )
        inputs = (
            MappingProxyType(dict(templates))
            if name == "consumer"
            else MappingProxyType({})
        )
        compiled = (
            jax.jit(program.function, out_shardings=layout.out_shardings)
            .lower(**program.arguments, **inputs)
            .compile()
        )
        compiled_cores[name] = PlannedCore(
            compiled=compiled,
            layout=layout,
            tile_widths=MappingProxyType({}),
            internal_input_templates=inputs,
            name=name,
        )
    return compiled_cores


def _assert_handover(
    *,
    output: KernelOutput,
    observed: list[BoundedResult[_Payload]],
    regime: Regime,
    enabled: bool,
    fault: bool,
) -> None:
    assert len(observed) == 1
    raw = observed[0]
    assert isinstance(output, FeasibilityKernelOutput)
    np.testing.assert_array_equal(output.value, [5.0, 9.0])
    np.testing.assert_array_equal(raw.output[0], [2.0, 3.0])
    np.testing.assert_array_equal(raw.output[1]["carry"], [2.0, 4.0])
    assert output.bound_status is raw.bound_status
    assert output.query_status is raw.query_status
    np.testing.assert_array_equal(output.bound_status[0], [False, False])
    assert output.query_status is not None
    np.testing.assert_array_equal(output.query_status[0], [False, False, fault])
    np.testing.assert_array_equal(output.feasibility_counts, [0, 0])
    if enabled:
        counts = raw.continuation_read_counts
        assert counts is not None
        np.testing.assert_array_equal(counts, [2, 3])
        assert counts.below_first_live.dtype == counts.eligible.dtype == jnp.int32
        assert counts.below_first_live.shape == counts.eligible.shape == ()
    else:
        assert raw.continuation_read_counts is None
    authorities = regime.solution.query_input_authorities
    assert authorities is not None
    if fault:
        with pytest.raises(
            InvalidValueFunctionError, match=r"query_coverage_missing.*done.*period 1"
        ):
            validate_query_input_status(
                statuses=output.query_status, authorities=authorities[0]
            )
    else:
        validate_query_input_status(
            statuses=output.query_status, authorities=authorities[0]
        )


@pytest.mark.parametrize("enabled", [False, True], ids=["unchecked", "checked"])
@pytest.mark.parametrize("fault", [False, True], ids=["valid", "mandatory-fault"])
def test_period_adapter_hands_selected_outputs_to_real_planned_cores(
    *, enabled: bool, fault: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The uniform period funnel keeps numeric handover and mandatory summaries."""
    model = _build_model()
    original = model._regimes["acting"]
    states = original.solution._base_state_action_space
    graph = _programs(fault=fault)
    observed: list[BoundedResult[_Payload]] = []
    adapter = _Handover(programs=graph, observed=observed)
    graph = core_program_graph(kernel=adapter)
    selected = select_programs(
        graph=graph, retain_replay=False, collect_continuation_read_counts=enabled
    )
    producer = selected["producer"]
    assert len(jax.tree.leaves(producer.output_roles)) == (9 if enabled else 7)
    assert producer.argument_builder is graph["producer"].argument_builder
    assert producer.internal_outputs == graph["producer"].internal_outputs
    if enabled:
        variant = graph["producer"].checked_variant
        assert variant is not None
        assert producer.function is variant.function
        assert producer.output_roles is variant.output_roles
    else:
        assert producer is graph["producer"]
    context = CoreBuildContext(
        state_action_space=states,
        next_regime_to_V_arr={},
        next_regime_to_continuation={},
        flat_params={},
        period=0,
        ages=model._time,
    )
    materialized = {
        name: materialize_core_program(program=program, context=context)
        for name, program in selected.items()
    }
    x = states.states["wealth"]
    templates = MappingProxyType(
        {
            "upstream_value": jax.ShapeDtypeStruct(x.shape, x.dtype),
            "upstream_carry": {"carry": jax.ShapeDtypeStruct(x.shape, x.dtype)},
        }
    )
    compiled_cores = _planned_cores(materialized=materialized, x=x, templates=templates)
    dispatches: list[str] = []
    actual_call = PlannedCore.__call__

    def record(self: PlannedCore, **arguments: PytreeValue) -> PytreeValue:
        dispatches.append(self.name)
        return actual_call(self, **arguments)

    monkeypatch.setattr(PlannedCore, "__call__", record)
    authority = (("done", 1, _authority()),)
    regime = replace(
        original,
        solution=replace(
            original.solution,
            period_kernels=MappingProxyType({0: adapter}),
            query_input_authorities={0: authority},
        ),
    )
    output = _run_period_kernel(
        regime=regime,
        regime_name="acting",
        period=0,
        compiled_cores=MappingProxyType(compiled_cores),
        capture_target=None,
        state_action_space=states,
        flat_params=MappingProxyType({}),
        ages=model._time,
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({}),
        logger=logging.getLogger(__name__),
        next_edge_to_V_arr=MappingProxyType({}),
        period_solution={},
        retain_replay=False,
        selected_artifact_keys=frozenset(),
    )
    assert dispatches == ["producer", "consumer"]
    _assert_handover(
        output=output, observed=observed, regime=regime, enabled=enabled, fault=fault
    )
