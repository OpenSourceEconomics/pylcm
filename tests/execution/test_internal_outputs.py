"""A program graph names one program's outputs as another's inputs.

The engine lowers the consumer against the producer's abstract output and checks
at dispatch that the arrays handed over match that declaration.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalInputRef,
    InternalOutputSpec,
    core_program_graph,
    materialize_core_program,
)
from _lcm.execution.internal_outputs import (
    assert_internal_inputs,
    internal_input_templates,
    topological_program_order,
)
from _lcm.execution.output_layout import VALUE


def _producer_function(*, x):
    return x + 1.0, {"carry": x * 2.0}


def _consumer_function(*, x, upstream_value, upstream_carry):
    return x + upstream_value + upstream_carry["carry"]


def _build_x(context):
    del context
    return {"x": jnp.zeros((3,))}


def _graph():
    producer = CoreProgram(
        name="producer",
        function=_producer_function,
        argument_builder=_build_x,
        requirements=CoreExecutionRequirements(),
        output_roles=(VALUE, {"carry": VALUE}),
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="deliberately_dense:test",
        internal_outputs=(
            InternalOutputSpec(label="value", path=(0,)),
            InternalOutputSpec(label="carry", path=(1,)),
        ),
    )
    consumer = CoreProgram(
        name="consumer",
        function=_consumer_function,
        argument_builder=_build_x,
        requirements=CoreExecutionRequirements(
            internal_inputs={
                "upstream_value": InternalInputRef(producer="producer", label="value"),
                "upstream_carry": InternalInputRef(producer="producer", label="carry"),
            }
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="deliberately_dense:test",
    )
    return {"producer": producer, "consumer": consumer}


class _Kernel:
    def __init__(self, *, programs):
        self._programs = programs

    def core_programs(self):
        return self._programs


def _context():
    return CoreBuildContext(
        state_action_space=None,
        next_regime_to_V_arr={},
        next_regime_to_continuation={},
        flat_params={},
        period=0,
        ages=None,
    )


def test_consumers_are_ordered_after_their_producers() -> None:
    graph = core_program_graph(kernel=_Kernel(programs=_graph()))
    assert topological_program_order(graph=graph) == ("producer", "consumer")


def test_templates_take_the_producers_abstract_output_shapes() -> None:
    graph = core_program_graph(kernel=_Kernel(programs=_graph()))
    materialized = {
        name: materialize_core_program(program=program, context=_context())
        for name, program in graph.items()
    }
    templates = cast(
        "Mapping[str, Any]",
        internal_input_templates(
            program=materialized["consumer"], producers=materialized
        ),
    )
    assert templates["upstream_value"].shape == (3,)
    assert templates["upstream_value"].dtype == jnp.zeros((3,)).dtype
    assert templates["upstream_carry"]["carry"].shape == (3,)


def test_a_consumer_lowers_against_the_templates_and_runs_on_real_arrays() -> None:
    graph = core_program_graph(kernel=_Kernel(programs=_graph()))
    materialized = {
        name: materialize_core_program(program=program, context=_context())
        for name, program in graph.items()
    }
    templates = internal_input_templates(
        program=materialized["consumer"], producers=materialized
    )
    lowered = jax.jit(_consumer_function).lower(
        **materialized["consumer"].arguments, **templates
    )
    compiled = lowered.compile()
    value, carry = _producer_function(**materialized["producer"].arguments)
    out = compiled(
        **materialized["consumer"].arguments, upstream_value=value, upstream_carry=carry
    )
    assert out.shape == (3,)


def test_dispatching_a_wrongly_shaped_internal_input_is_refused() -> None:
    templates = {"upstream_value": jax.ShapeDtypeStruct((3,), jnp.float32)}
    with pytest.raises(ValueError, match="upstream_value"):
        assert_internal_inputs(
            arguments={"upstream_value": jnp.zeros((4,), dtype=jnp.float32)},
            templates=templates,
            label="consumer",
        )


@pytest.mark.parametrize(
    ("bad_ref", "match"),
    [
        (InternalInputRef(producer="nobody", label="value"), "nobody"),
        (InternalInputRef(producer="producer", label="missing"), "missing"),
    ],
    ids=["unknown-producer", "unknown-label"],
)
def test_an_internal_input_must_name_a_declared_output(*, bad_ref, match) -> None:
    programs = _graph()
    requirements = CoreExecutionRequirements(
        internal_inputs={"upstream_value": bad_ref}
    )
    programs["consumer"] = dataclasses.replace(
        programs["consumer"], requirements=requirements
    )
    with pytest.raises(ValueError, match=match):
        core_program_graph(kernel=_Kernel(programs=programs))


def test_a_cycle_of_internal_inputs_is_refused() -> None:
    programs = _graph()
    programs["producer"] = dataclasses.replace(
        programs["producer"],
        requirements=CoreExecutionRequirements(
            internal_inputs={"back": InternalInputRef(producer="consumer", label="out")}
        ),
    )
    programs["consumer"] = dataclasses.replace(
        programs["consumer"],
        internal_outputs=(InternalOutputSpec(label="out", path=()),),
    )
    with pytest.raises(ValueError, match="cycle"):
        core_program_graph(kernel=_Kernel(programs=programs))


def test_an_internal_input_may_not_collide_with_a_built_argument() -> None:
    programs = _graph()
    programs["consumer"] = dataclasses.replace(
        programs["consumer"],
        requirements=CoreExecutionRequirements(
            internal_inputs={"x": InternalInputRef(producer="producer", label="value")}
        ),
    )
    graph = core_program_graph(kernel=_Kernel(programs=programs))
    materialized = {
        name: materialize_core_program(program=program, context=_context())
        for name, program in graph.items()
    }
    with pytest.raises(ValueError, match="'x'"):
        internal_input_templates(
            program=materialized["consumer"], producers=materialized
        )
