"""Typed outputs one program of a graph hands to another at dispatch.

A producer declares the subtrees of its raw output that other programs of the
same graph may consume; a consumer names one of them per argument. The engine
lowers the consumer against the producer's abstract output, so the consumer sees
the exact shapes and dtypes it will receive instead of a stand-in.
"""

from collections.abc import Mapping
from types import MappingProxyType

import jax

from _lcm.execution.core_program import (
    CoreProgram,
    MaterializedCoreProgram,
    _topological_program_order,
)


def topological_program_order(*, graph: Mapping[str, CoreProgram]) -> tuple[str, ...]:
    """Return graph keys so every producer precedes its consumers.

    Declaration order breaks ties, so a graph without internal edges keeps the
    order its kernel published. A graph whose internal inputs cannot be ordered is
    refused.
    """
    return _topological_program_order(graph=graph)


def internal_input_templates(
    *,
    program: MaterializedCoreProgram,
    producers: Mapping[str, MaterializedCoreProgram],
) -> MappingProxyType[str, object]:
    """Return abstract templates for one program's internal inputs.

    Each template is the subtree of the producer's abstract output that the
    producer's `InternalOutputSpec` selects, so its leaves are
    `jax.ShapeDtypeStruct`. An argument the program builds itself may not also be
    declared as an internal input: the two would silently disagree at dispatch.
    """
    templates: dict[str, object] = {}
    abstract_outputs: dict[str, object] = {}
    for name, ref in program.requirements.internal_inputs.items():
        if name in program.arguments:
            msg = (
                f"Core program {program.name!r} builds an argument {name!r} that its "
                "internal inputs also declare."
            )
            raise ValueError(msg)
        producer = producers[ref.producer]
        if ref.producer not in abstract_outputs:
            abstract_outputs[ref.producer] = jax.eval_shape(
                producer.function, **producer.arguments
            )
        spec = next(s for s in producer.internal_outputs if s.label == ref.label)
        templates[name] = _select_path(
            tree=abstract_outputs[ref.producer], path=spec.path
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


def _leaf_signature(leaf: object) -> tuple[tuple[int, ...], str]:
    """Return the shape and dtype spelling that identify one handed-over leaf."""
    return (tuple(leaf.shape), str(leaf.dtype))  # ty: ignore[unresolved-attribute]


def _select_path(*, tree: object, path: tuple[int | str, ...]) -> object:
    """Index one abstract output tree down to the published subtree."""
    node = tree
    for step in path:
        node = node[step]  # ty: ignore[not-subscriptable]
    return node
