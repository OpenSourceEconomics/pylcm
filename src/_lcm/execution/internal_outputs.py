"""Typed outputs one program of a graph hands to another at dispatch.

A producer declares the subtrees of its raw output that other programs of the
same graph may consume; a consumer names one of them per argument. The engine
lowers the consumer against the producer's abstract output, so the consumer sees
the exact shapes and dtypes it will receive instead of a stand-in.
"""

import dataclasses
import functools
from collections.abc import Callable, Hashable, Mapping
from types import MappingProxyType

import jax

from _lcm.execution.core_program import (
    CoreProgram,
    InternalOutputSpec,
    MaterializedCoreProgram,
    ResolvedCoreProgram,
    _topological_program_order,
)
from lcm.exceptions import ExecutionPlanningError


# The ordering lives beside the graph validation that needs it, and `core_program`
# cannot import this module, so the public name is a delegate rather than a copy.
def topological_program_order(*, graph: Mapping[str, CoreProgram]) -> tuple[str, ...]:
    """Return graph keys so every producer precedes its consumers.

    Declaration order breaks ties, so a graph without internal edges keeps the
    order its kernel published. A graph whose internal inputs cannot be ordered is
    refused.
    """
    return _topological_program_order(graph=graph)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedProducer:
    """One program's complete invocation and the abstract output it yields.

    A program is lowered from three parts: the dynamic argument tree its builder
    returned once input transfers were applied, the templates of the internal
    inputs it reads itself, and the planner-owned width bindings. A record traces
    those three together, once, so the tree its consumers read is the tree it
    will produce.
    """

    name: str
    """Graph key of the program this record describes."""

    function: Callable[..., object]
    """The exact callable the engine lowers for this program."""

    internal_input_templates: Mapping[str, object]
    """Abstract subtrees the program reads from its own producers."""

    static_kwargs: Mapping[str, int]
    """Planner-owned width bindings, passed to JAX as static keyword arguments."""

    internal_outputs: tuple[InternalOutputSpec, ...]
    """Outputs another program of the same graph may name as an input."""

    abstract_output: object
    """The complete invocation's `jax.eval_shape`, traced once."""

    def __post_init__(self) -> None:
        """Snapshot the caller-owned invocation mappings and declarations."""
        object.__setattr__(
            self,
            "internal_input_templates",
            MappingProxyType(dict(self.internal_input_templates)),
        )
        object.__setattr__(
            self, "static_kwargs", MappingProxyType(dict(self.static_kwargs))
        )
        object.__setattr__(self, "internal_outputs", tuple(self.internal_outputs))


def resolve_producer(
    *, program: ResolvedCoreProgram, templates: Mapping[str, object]
) -> ResolvedProducer:
    """Trace one resolved program's complete invocation and keep its output.

    The widths are bound before tracing rather than handed over as arguments:
    `jax.eval_shape` abstracts every argument it receives, and a width that
    reaches the body as a tracer cannot size a reshape. That is the binding the
    eager dispatch path makes, so the traced body is the lowered body.
    """
    invocation = (
        functools.partial(program.function, **program.static_kwargs)
        if program.static_kwargs
        else program.function
    )
    return ResolvedProducer(
        name=program.name,
        function=program.function,
        internal_input_templates=templates,
        static_kwargs=program.static_kwargs,
        internal_outputs=program.internal_outputs,
        abstract_output=jax.eval_shape(invocation, **program.arguments, **templates),
    )


def internal_input_templates(
    *,
    program: MaterializedCoreProgram,
    producers: Mapping[str, Mapping[Hashable, ResolvedProducer]],
) -> MappingProxyType[str, object]:
    """Return abstract templates for one program's internal inputs.

    Each template is the subtree of the producer's abstract output that the
    producer's `InternalOutputSpec` selects, so its leaves are
    `jax.ShapeDtypeStruct`. A producer is offered as its width candidates, which
    `assert_width_invariant_internal_outputs` has shown publish one subtree per
    label, so the planner's top-ranked candidate names what any of them names.
    The producer is not traced here: its abstract output was traced once, when
    its own invocation became complete. An argument the program builds itself may
    not also be declared as an internal input: the two would silently disagree at
    dispatch.
    """
    templates: dict[str, object] = {}
    for name, ref in program.requirements.internal_inputs.items():
        if name in program.arguments:
            msg = (
                f"Core program {program.name!r} builds an argument {name!r} that its "
                "internal inputs also declare."
            )
            raise ValueError(msg)
        producer = next(iter(producers[ref.producer].values()))
        spec = _declared_output(
            producer=producer, label=ref.label, consumer=program.name
        )
        templates[name] = _select_path(
            tree=producer.abstract_output,
            path=spec.path,
            producer_name=ref.producer,
            label=ref.label,
        )
    return MappingProxyType(templates)


def assert_internal_inputs(
    *, arguments: Mapping[str, object], templates: Mapping[str, object], label: str
) -> None:
    """Fail when a handed-over internal input departs from its declared template."""
    for name, template in templates.items():
        if name not in arguments:
            msg = (
                f"Core program {label!r} was dispatched without internal input "
                f"{name!r}."
            )
            raise ValueError(msg)
        expected = jax.tree.map(_leaf_signature, template)
        actual = jax.tree.map(_leaf_signature, arguments[name])
        if expected != actual:
            msg = (
                f"Core program {label!r} received internal input {name!r} with "
                f"{actual!r}; its producer declares {expected!r}."
            )
            raise ValueError(msg)


def assert_width_invariant_internal_outputs(
    *, candidates: Mapping[Hashable, ResolvedProducer]
) -> None:
    """Fail when one producer publishes different subtrees at different widths.

    A consumer is lowered against one of its producer's subtrees while the
    producer's own width is selected after its candidates are compiled and
    measured. The two decisions are independent only where the published
    subtrees do not depend on the width, which a program that streams its work
    and reduces it satisfies and one that returns its blocks does not. The second
    is refused here rather than lowered against a subtree the selection may
    invalidate.
    """
    if not candidates:
        msg = "A consumed producer must have at least one width candidate."
        raise ExecutionPlanningError(msg)
    records = tuple(candidates.values())
    reference = records[0]
    for spec in reference.internal_outputs:
        expected = _published_signature(record=reference, spec=spec)
        for record in records[1:]:
            actual = _published_signature(record=record, spec=spec)
            if actual == expected:
                continue
            msg = (
                f"Core program {reference.name!r} publishes internal output "
                f"{spec.label!r} as {expected!r} at widths "
                f"{dict(reference.static_kwargs)!r} and as {actual!r} at widths "
                f"{dict(record.static_kwargs)!r}. A published output may not "
                "depend on the width the planner selects."
            )
            raise ExecutionPlanningError(msg)


def consumed_producer_names(*, graph: Mapping[str, CoreProgram]) -> frozenset[str]:
    """Return the programs of one graph whose internal outputs it consumes."""
    return frozenset(
        ref.producer
        for program in graph.values()
        for ref in program.requirements.internal_inputs.values()
    )


def _declared_output(
    *, producer: ResolvedProducer, label: str, consumer: str
) -> InternalOutputSpec:
    """Return the producer's output declaration that one reference names."""
    for spec in producer.internal_outputs:
        if spec.label == label:
            return spec
    msg = (
        f"Core program {consumer!r} reads internal output {label!r} of producer "
        f"{producer.name!r}, which declares "
        f"{tuple(spec.label for spec in producer.internal_outputs)!r}."
    )
    raise ValueError(msg)


def _published_signature(
    *, record: ResolvedProducer, spec: InternalOutputSpec
) -> object:
    """Return the shape-and-dtype tree one label publishes from one record."""
    return jax.tree.map(
        _leaf_signature,
        _select_path(
            tree=record.abstract_output,
            path=spec.path,
            producer_name=record.name,
            label=spec.label,
        ),
    )


def _leaf_signature(leaf: object) -> tuple[tuple[int, ...], str]:
    """Return the shape and dtype spelling that identify one handed-over leaf."""
    shape = getattr(leaf, "shape", None)
    dtype = getattr(leaf, "dtype", None)
    if shape is None or dtype is None:
        msg = (
            "An internal input leaf must be an array carrying a shape and a dtype; "
            f"got {leaf!r}."
        )
        raise ValueError(msg)
    return (tuple(shape), str(dtype))


def _select_path(
    *, tree: object, path: tuple[int | str, ...], producer_name: str, label: str
) -> object:
    """Index one abstract output tree down to the published subtree."""
    node = tree
    for step in path:
        try:
            node = node[step]  # ty: ignore[not-subscriptable]
        except (IndexError, KeyError, TypeError) as error:
            msg = (
                f"Internal output {label!r} of core program {producer_name!r} declares "
                f"path {path!r}, which its abstract output does not reach: step "
                f"{step!r} is not in {node!r}."
            )
            raise ValueError(msg) from error
    return node
