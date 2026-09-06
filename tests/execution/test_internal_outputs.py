"""A program graph names one program's outputs as another's inputs.

The engine lowers the consumer against the producer's abstract output and checks
at dispatch that the arrays handed over match that declaration.
"""

import dataclasses
from collections.abc import Hashable, Mapping
from types import MappingProxyType
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
    ProgramScope,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
    select_programs,
)
from _lcm.execution.internal_outputs import (
    ResolvedProducer,
    assert_internal_inputs,
    assert_width_invariant_internal_outputs,
    internal_input_templates,
    resolve_producer,
    topological_program_order,
)
from _lcm.execution.output_layout import (
    VALUE,
    PlannedCore,
    resolve_output_layout,
)
from _lcm.typing import FloatND
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import ArtifactKey
from lcm.solvers import StreamableProductAxis


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


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _CountingProducer:
    """A producer body recording how often the engine traces it.

    Comparison stays identity-based, which is what a program's function must
    offer so JAX can key its compilation cache on the raw callable.
    """

    calls: list[int]
    """One entry per call, so a shared producer's trace count is readable."""

    def __call__(self, *, x):
        """Publish the producer's two labelled subtrees and record the call."""
        self.calls.append(1)
        return x + 1.0, {"carry": x * 2.0}


@dataclasses.dataclass(frozen=True, kw_only=True)
class _MaxReduction:
    """Semantics of the streamed candidate axis's reduction."""

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the durable identity of this reduction."""
        return ("tests.internal_outputs.max", 1)


def _records(
    *, materialized: Mapping[str, Any], names: tuple[str, ...]
) -> dict[str, MappingProxyType[Hashable, ResolvedProducer]]:
    """Trace the named producers once each and key them by one width candidate.

    Every producer of these fixtures reads nothing itself and streams nothing, so
    its complete invocation is its argument tree and its one candidate is empty.
    """
    return {
        name: MappingProxyType(_one_record(program=materialized[name]))
        for name in names
    }


def _one_record(*, program: Any) -> dict[Hashable, ResolvedProducer]:
    """Trace one producer that reads nothing and streams nothing."""
    return {
        (): resolve_producer(
            program=resolve_core_program(program=program, tile_widths={}),
            templates=MappingProxyType({}),
        )
    }


def _middle_function(*, upstream):
    """Publish one row from the row its producer published."""
    return upstream * 2.0


def _build_nothing(context):
    """Return no arguments, ignoring the build context."""
    del context
    return {}


def _chain_programs() -> dict[str, CoreProgram]:
    """A root, a middle program reading it, and a consumer reading the middle."""
    return {
        "root": CoreProgram(
            name="root",
            function=_producer_function,
            argument_builder=_build_x,
            requirements=CoreExecutionRequirements(),
            output_roles=(VALUE, {"carry": VALUE}),
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="deliberately_dense:test",
            internal_outputs=(InternalOutputSpec(label="value", path=(0,)),),
        ),
        "middle": CoreProgram(
            name="middle",
            function=_middle_function,
            argument_builder=_build_nothing,
            requirements=CoreExecutionRequirements(
                internal_inputs={
                    "upstream": InternalInputRef(producer="root", label="value")
                }
            ),
            output_roles=VALUE,
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="deliberately_dense:test",
            internal_outputs=(InternalOutputSpec(label="value", path=()),),
        ),
        "leaf": CoreProgram(
            name="leaf",
            function=_middle_function,
            argument_builder=_build_nothing,
            requirements=CoreExecutionRequirements(
                internal_inputs={
                    "upstream": InternalInputRef(producer="middle", label="value")
                }
            ),
            output_roles=VALUE,
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="deliberately_dense:test",
        ),
    }


def _blocking_function(*, x, candidate, width):
    """Publish a reduced row and, separately, one row per candidate block."""
    blocks = -(-candidate.shape[0] // width)
    return x + jnp.max(candidate), jnp.zeros((blocks,), dtype=x.dtype)


def _build_candidates(context):
    """Return the state row and the streamed candidate coordinate."""
    del context
    return {"x": jnp.zeros((3,)), "candidate": jnp.arange(5.0)}


def _streaming_program(*, label: str, path: tuple[int, ...]) -> CoreProgram:
    """A planned producer publishing one label of its two-element raw output."""
    return CoreProgram(
        name="blocking",
        function=_blocking_function,
        argument_builder=_build_candidates,
        requirements=CoreExecutionRequirements(
            streamable_axes=(
                StreamableProductAxis(
                    name="candidate",
                    coordinate_names=("candidate",),
                    coordinate_extents=(5,),
                    canonical_order="c",
                    reduction=_MaxReduction(),
                    width_keyword="width",
                ),
            )
        ),
        output_roles=(VALUE, VALUE),
        disposition=CoreExecutionDisposition.PLANNED,
        internal_outputs=(InternalOutputSpec(label=label, path=path),),
    )


def _streaming_candidates(
    *, label: str, path: tuple[int, ...]
) -> MappingProxyType[Hashable, ResolvedProducer]:
    """Trace one planned producer at two legal widths of its candidate axis."""
    materialized = materialize_core_program(
        program=_streaming_program(label=label, path=path), context=_context()
    )
    records: dict[Hashable, ResolvedProducer] = {
        (("candidate", width),): resolve_producer(
            program=resolve_core_program(
                program=materialized, tile_widths={"candidate": width}
            ),
            templates=MappingProxyType({}),
        )
        for width in (2, 4)
    }
    return MappingProxyType(records)


def test_consumers_are_ordered_after_their_producers() -> None:
    """A graph is ordered so a producer is lowered before the consumer reading it."""
    graph = core_program_graph(kernel=_Kernel(programs=_graph()))
    assert topological_program_order(graph=graph) == ("producer", "consumer")


def test_templates_take_the_producers_abstract_output_shapes() -> None:
    """Each internal-input template carries the shape and dtype it will receive."""
    graph = core_program_graph(kernel=_Kernel(programs=_graph()))
    materialized = {
        name: materialize_core_program(program=program, context=_context())
        for name, program in graph.items()
    }
    templates = cast(
        "Mapping[str, Any]",
        internal_input_templates(
            program=materialized["consumer"],
            producers=_records(materialized=materialized, names=("producer",)),
        ),
    )
    assert templates["upstream_value"].shape == (3,)
    assert templates["upstream_value"].dtype == jnp.zeros((3,)).dtype
    assert templates["upstream_carry"]["carry"].shape == (3,)


def test_a_consumer_lowers_against_the_templates_and_runs_on_real_arrays() -> None:
    """A consumer lowered against the templates accepts the producer's real output."""
    graph = core_program_graph(kernel=_Kernel(programs=_graph()))
    materialized = {
        name: materialize_core_program(program=program, context=_context())
        for name, program in graph.items()
    }
    templates = internal_input_templates(
        program=materialized["consumer"],
        producers=_records(materialized=materialized, names=("producer",)),
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
    """An array whose shape departs from its template is refused at dispatch."""
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
    """Building a graph refuses a reference to an unknown producer or label."""
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
    """A graph whose internal edges close a cycle cannot be ordered and is refused."""
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
    """A name the program builds itself may not also arrive as an internal input."""
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
            program=materialized["consumer"],
            producers=_records(materialized=materialized, names=("producer",)),
        )


def _materialized_graph(programs: Mapping[str, CoreProgram]) -> dict[str, Any]:
    """Materialize every program of a graph against the shared build context."""
    graph = core_program_graph(kernel=_Kernel(programs=programs))
    return {
        name: materialize_core_program(program=program, context=_context())
        for name, program in graph.items()
    }


def test_a_template_for_an_undeclared_label_names_the_program_and_the_label() -> None:
    """Reading a label the producer does not declare is refused by name."""
    materialized = _materialized_graph(_graph())
    materialized["consumer"] = dataclasses.replace(
        materialized["consumer"],
        requirements=CoreExecutionRequirements(
            internal_inputs={
                "upstream_value": InternalInputRef(
                    producer="producer", label="not_declared"
                )
            }
        ),
    )

    with pytest.raises(ValueError, match=r"'consumer'.*'not_declared'.*'producer'"):
        internal_input_templates(
            program=materialized["consumer"],
            producers=_records(materialized=materialized, names=("producer",)),
        )


def test_a_template_for_an_unreachable_path_names_the_path() -> None:
    """A published path the producer's abstract output does not reach is refused."""
    materialized = _materialized_graph(_graph())
    materialized["producer"] = dataclasses.replace(
        materialized["producer"],
        internal_outputs=(InternalOutputSpec(label="value", path=(5,)),),
    )
    materialized["consumer"] = dataclasses.replace(
        materialized["consumer"],
        requirements=CoreExecutionRequirements(
            internal_inputs={
                "upstream_value": InternalInputRef(producer="producer", label="value")
            }
        ),
    )

    with pytest.raises(ValueError, match=r"path \(5,\)"):
        internal_input_templates(
            program=materialized["consumer"],
            producers=_records(materialized=materialized, names=("producer",)),
        )


def test_a_non_array_internal_input_leaf_is_refused_by_name() -> None:
    """A handed-over leaf without a shape and a dtype is refused, naming the leaf."""
    with pytest.raises(ValueError, match="shape and a dtype"):
        assert_internal_inputs(
            arguments={"upstream_value": jnp.zeros((3,), dtype=jnp.float32)},
            templates={"upstream_value": "not an array"},
            label="consumer",
        )


def _scoped_graph():
    """A producer kept only for replay feeding a consumer kept only without it."""
    programs = _graph()
    key = ArtifactKey(type_id="tests.internal_outputs.producer")
    programs["producer"] = dataclasses.replace(
        programs["producer"],
        scope=ProgramScope.REPLAY,
        replaces_program="consumer",
        retained_artifact_keys=(key,),
        retained_artifact_payload_types={key: dict},
    )
    programs["consumer"] = dataclasses.replace(
        programs["consumer"], scope=ProgramScope.VALUES_ONLY
    )
    return programs


def test_selecting_away_a_producer_its_consumer_still_needs_is_refused() -> None:
    """A retention that drops a producer but keeps its consumer names all three."""
    graph = core_program_graph(kernel=_Kernel(programs=_scoped_graph()))

    with pytest.raises(ValueError, match=r"consumer.*upstream_value.*producer"):
        select_programs(graph=graph, retain_replay=False)


def test_a_values_only_producer_without_a_replay_alternative_survives_replay() -> None:
    """Retention keeps an unreplaced values program, so its consumers keep theirs."""
    programs = _graph()
    programs["producer"] = dataclasses.replace(
        programs["producer"], scope=ProgramScope.VALUES_ONLY
    )
    graph = core_program_graph(kernel=_Kernel(programs=programs))

    selected = select_programs(graph=graph, retain_replay=True)

    assert tuple(selected) == ("producer", "consumer")


def test_a_planned_core_names_itself_when_an_internal_input_is_misshapen() -> None:
    """A wrongly shaped handover is refused at dispatch, naming the program."""
    template = jnp.zeros((3,))
    layout = resolve_output_layout(
        core_key="consumer",
        value_template=template,
        state_order=("wealth",),
        output_roles=VALUE,
    )
    core = PlannedCore(
        compiled=_consumer_function,
        layout=layout,
        tile_widths={},
        internal_input_templates={
            "upstream_value": jax.ShapeDtypeStruct((3,), template.dtype)
        },
        name="consumer",
    )

    with pytest.raises(ValueError, match="'consumer'"):
        core(
            x=template,
            upstream_value=jnp.zeros((4,)),
            upstream_carry={"carry": template},
        )


def test_a_producer_is_traced_with_its_own_internal_input_template() -> None:
    """A producer that is itself a consumer reaches its consumer's template."""
    programs = _chain_programs()
    materialized = _materialized_graph(programs)
    root_records = _records(materialized=materialized, names=("root",))
    middle_templates = internal_input_templates(
        program=materialized["middle"], producers=root_records
    )
    middle_record: dict[Hashable, ResolvedProducer] = {
        (): resolve_producer(
            program=resolve_core_program(
                program=materialized["middle"], tile_widths={}
            ),
            templates=middle_templates,
        )
    }

    templates = cast(
        "Mapping[str, Any]",
        internal_input_templates(
            program=materialized["leaf"],
            producers={"middle": MappingProxyType(middle_record)},
        ),
    )

    assert templates["upstream"].shape == (3,)


def test_a_producer_traced_without_its_upstream_template_is_refused() -> None:
    """Dropping a producer's own internal input is a missing keyword, not a guess."""
    materialized = _materialized_graph(_chain_programs())

    with pytest.raises(TypeError, match="upstream"):
        resolve_producer(
            program=resolve_core_program(
                program=materialized["middle"], tile_widths={}
            ),
            templates=MappingProxyType({}),
        )


def test_a_producer_traced_without_its_static_width_is_refused() -> None:
    """Dropping a planned producer's width binding is a missing keyword."""
    candidates = _streaming_candidates(label="value", path=(0,))
    resolved = next(iter(candidates.values()))

    with pytest.raises(TypeError, match="width"):
        resolve_producer(
            program=dataclasses.replace(
                resolve_core_program(
                    program=materialize_core_program(
                        program=_streaming_program(label="value", path=(0,)),
                        context=_context(),
                    ),
                    tile_widths={"candidate": 2},
                ),
                static_kwargs={},
            ),
            templates=resolved.internal_input_templates,
        )


def test_a_planned_producers_static_width_reaches_its_abstract_output() -> None:
    """A width the planner owns sizes the traced body, not a tracer."""
    candidates = _streaming_candidates(label="blocks", path=(1,))

    narrow = next(iter(candidates.values()))

    assert cast("Any", narrow.abstract_output)[1].shape == (3,)


def test_a_width_dependent_published_output_is_refused() -> None:
    """A label whose shape follows the planner's width cannot be lowered against."""
    candidates = _streaming_candidates(label="blocks", path=(1,))

    with pytest.raises(ExecutionPlanningError, match="blocks"):
        assert_width_invariant_internal_outputs(candidates=candidates)


def test_a_width_invariant_published_output_is_admitted() -> None:
    """A reduced label is the same subtree at every legal width."""
    candidates = _streaming_candidates(label="value", path=(0,))

    assert assert_width_invariant_internal_outputs(candidates=candidates) is None


def test_one_producer_feeding_two_consumers_is_traced_once() -> None:
    """A shared producer's abstract output is projected, not retraced."""
    counter = _CountingProducer(calls=[])
    programs = _graph()
    programs["producer"] = dataclasses.replace(programs["producer"], function=counter)
    materialized = _materialized_graph(programs)
    records = _records(materialized=materialized, names=("producer",))
    for _ in range(2):
        internal_input_templates(program=materialized["consumer"], producers=records)

    assert counter.calls == [1]


def test_dispatching_a_wrongly_typed_internal_input_is_refused() -> None:
    """An array whose dtype departs from its template is refused at dispatch."""
    with pytest.raises(ValueError, match="upstream_value"):
        assert_internal_inputs(
            arguments={"upstream_value": jnp.zeros((3,), dtype=jnp.float32)},
            templates={"upstream_value": jax.ShapeDtypeStruct((3,), jnp.int32)},
            label="consumer",
        )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _ScalarPublisher:
    """A producer body publishing one scalar under a per-width typing convention.

    Comparison stays identity-based, which is what a program's function must
    offer so JAX can key its compilation cache on the raw callable.
    """

    weak_widths: frozenset[int]
    """Widths at which the published scalar is weakly typed."""

    def __call__(
        self, *, x: FloatND, candidate: FloatND, width: int
    ) -> tuple[FloatND, FloatND]:
        """Publish the state row and one scalar of this width's convention."""
        del candidate
        return x, (
            jnp.asarray(1.0)
            if width in self.weak_widths
            else jnp.asarray(1.0, dtype=x.dtype)
        )


def _scalar_candidates(
    *, weak_widths: frozenset[int]
) -> MappingProxyType[Hashable, ResolvedProducer]:
    """Trace one scalar-publishing producer at two legal widths of its axis."""
    materialized = materialize_core_program(
        program=dataclasses.replace(
            _streaming_program(label="scalar", path=(1,)),
            function=_ScalarPublisher(weak_widths=weak_widths),
        ),
        context=_context(),
    )
    records: dict[Hashable, ResolvedProducer] = {
        (("candidate", width),): resolve_producer(
            program=resolve_core_program(
                program=materialized, tile_widths={"candidate": width}
            ),
            templates=MappingProxyType({}),
        )
        for width in (2, 4)
    }
    return MappingProxyType(records)


def _width_dependent_scalar_leaves() -> tuple[
    jax.ShapeDtypeStruct, jax.ShapeDtypeStruct
]:
    """Return the scalar published at the narrow and at the wide width."""
    narrow, wide = _scalar_candidates(weak_widths=frozenset({2})).values()
    return (
        cast("Any", narrow.abstract_output)[1],
        cast("Any", wide.abstract_output)[1],
    )


@pytest.mark.parametrize("attribute", ["shape", "dtype"])
def test_a_width_dependent_scalar_publishes_one_shape_and_one_dtype(
    *, attribute: str
) -> None:
    """The two width candidates of the scalar specimen agree on shape and dtype."""
    narrow, wide = _width_dependent_scalar_leaves()

    assert getattr(narrow, attribute) == getattr(wide, attribute)


def test_a_width_dependent_scalar_publishes_two_weak_typings() -> None:
    """The narrow candidate's scalar is weakly typed and the wide one's is not."""
    narrow, wide = _width_dependent_scalar_leaves()

    assert (narrow.weak_type, wide.weak_type) == (True, False)


def test_a_width_dependent_published_weak_typing_is_refused() -> None:
    """A label whose weak typing follows the width cannot be lowered against."""
    candidates = _scalar_candidates(weak_widths=frozenset({2}))

    with pytest.raises(ExecutionPlanningError, match="scalar"):
        assert_width_invariant_internal_outputs(candidates=candidates)


@pytest.mark.parametrize(
    "weak_widths", [frozenset(), frozenset({2, 4})], ids=["strong", "weak"]
)
def test_a_uniformly_typed_published_scalar_is_admitted(
    *, weak_widths: frozenset[int]
) -> None:
    """One weak-typing convention held at every width publishes one subtree."""
    candidates = _scalar_candidates(weak_widths=weak_widths)

    assert assert_width_invariant_internal_outputs(candidates=candidates) is None


@pytest.mark.parametrize("template_weak_type", [False, True])
def test_dispatching_a_differently_weakly_typed_internal_input_is_refused(
    *, template_weak_type: bool
) -> None:
    """A leaf of the template's shape and dtype but other weak typing is refused."""
    weak = jnp.asarray(1.0)
    strong = jnp.asarray(1.0, dtype=weak.dtype)

    with pytest.raises(ValueError, match="upstream_value"):
        assert_internal_inputs(
            arguments={"upstream_value": strong if template_weak_type else weak},
            templates={
                "upstream_value": jax.ShapeDtypeStruct(
                    (), weak.dtype, weak_type=template_weak_type
                )
            },
            label="consumer",
        )


@pytest.mark.parametrize("weak_type", [False, True])
def test_dispatching_an_internal_input_of_the_declared_weak_typing_is_admitted(
    *, weak_type: bool
) -> None:
    """A leaf carrying its template's weak typing is handed over unchanged."""
    weak = jnp.asarray(1.0)
    value = weak if weak_type else jnp.asarray(1.0, dtype=weak.dtype)

    assert (
        assert_internal_inputs(
            arguments={"upstream_value": value},
            templates={
                "upstream_value": jax.ShapeDtypeStruct(
                    (), weak.dtype, weak_type=weak_type
                )
            },
            label="consumer",
        )
        is None
    )
