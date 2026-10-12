"""Identify dynamic operands included in a compiled executable's memory report."""

from collections.abc import Mapping

import jax

from _lcm.params.mapping_leaf import MappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf
from _lcm.typing import ArgumentTree, DataclassInstance
from lcm.exceptions import ExecutionPlanningError

# What `Compiled.input_shardings` returns: the positional and keyword input trees
# laid out again inside the same containers, with a sharding for each kept leaf and
# `None` for each eliminated one.
type _InputShardingTree = (
    jax.sharding.Sharding
    | MappingLeaf
    | SequenceLeaf
    | DataclassInstance
    | tuple[_InputShardingTree, ...]
    | list[_InputShardingTree]
    | Mapping[str, _InputShardingTree]
    | None
)


def compiler_input_paths(
    *, compiled: jax.stages.Compiled, arguments: Mapping[str, ArgumentTree]
) -> frozenset[jax.tree_util.KeyPath]:
    """Return keyword-tree paths retained by the exact compiled specialization.

    JAX's public ``input_shardings`` restores the original dynamic input tree,
    using ``None`` for arguments eliminated by compilation. Such arguments can
    still have live owners outside the executable and must remain resident.
    Static bindings are absent from both trees. Captured constants are not
    dynamic arguments and this function grants no exclusion for their owners.
    """
    try:
        shardings = compiled.input_shardings
    except Exception as error:
        raise ExecutionPlanningError(
            "Compiler input shardings are unavailable for residency accounting."
        ) from error
    actual_tree = jax.tree.structure(((), dict(arguments)), is_leaf=_is_none)
    sharding_tree = jax.tree.structure(shardings, is_leaf=_is_none)
    if actual_tree != sharding_tree:
        raise ExecutionPlanningError(
            "Compiler input shardings do not match the dynamic argument tree."
        )
    _, keyword_shardings = shardings
    with_paths, _ = jax.tree_util.tree_flatten_with_path(
        keyword_shardings, is_leaf=_is_none
    )
    if any(
        sharding is not None and not isinstance(sharding, jax.sharding.Sharding)
        for _, sharding in with_paths
    ):
        raise ExecutionPlanningError(
            "Compiler input shardings contain unsupported residency metadata."
        )
    return frozenset(path for path, sharding in with_paths if sharding is not None)


def _is_none(value: ArgumentTree | _InputShardingTree) -> bool:
    """Preserve omitted input slots when comparing the original pytrees.

    JAX hands the predicate every node of the argument tree and of the compiled
    input-sharding tree.
    """
    return value is None
