"""Model-owned store of the immutable structural blueprints warm solves reuse.

A solve resolves every program it dispatches into abstract argument trees,
value-read plans, output layouts and ranked width frontiers. For built-in
GridSearch programs that recipe depends on the solve's inputs only through their
abstract schema: tree structure, shape, dtype, weak typing and placement. A
model keeps a few such recipes, keyed by that schema and by everything else the
recipe reads, so a warm solve with the same schema binds a stored recipe instead
of resolving every program again.

The store holds no concrete array, liveness ledger, donation decision, frontier
cursor or admission. Those are built for every solve from the current inputs.
"""

import dataclasses
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping

import jax

_DEFAULT_MAX_ENTRIES = 4


class UncacheableSchemaError(TypeError):
    """An input carries a leaf whose structure cannot be described abstractly."""


class StructuralBlueprintCache[Blueprint]:
    """A bounded, least-recently-used store of structural blueprints.

    It lives and dies with its model: nothing is shared across models, and a
    pickled model drops it. Hit and miss counts are kept for diagnostics.
    """

    def __init__(self, *, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        """Create an empty store that keeps at most `max_entries` blueprints."""
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer.")
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[Hashable, Blueprint] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self, *, key: Hashable, accept: Callable[[Blueprint], bool] = lambda _: True
    ) -> Blueprint | None:
        """Return the blueprint stored under `key`, or `None` on a miss.

        A stored blueprint `accept` rejects counts as a miss.
        """
        with self._lock:
            blueprint = self._entries.get(key)
            if blueprint is None or not accept(blueprint):
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return blueprint

    def put(self, *, key: Hashable, blueprint: Blueprint) -> None:
        """Store `blueprint` under `key`, evicting the least recently used."""
        with self._lock:
            self._entries[key] = blueprint
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def values(self) -> tuple[Blueprint, ...]:
        """Return the stored blueprints, least recently used first."""
        with self._lock:
            return tuple(self._entries.values())

    def __len__(self) -> int:
        """Return the number of stored blueprints."""
        return len(self._entries)


def abstract_schema(tree: object) -> Hashable:
    """Describe a tree by its structure and per-leaf abstract metadata.

    A leaf contributes its shape, canonical dtype, weak typing, and — for a JAX
    array — whether it is committed and its sharding. Values never enter.

    Raises:
        UncacheableSchemaError: A leaf is not an array-like value, or the tree
            structure is not hashable.

    """
    leaves, treedef = jax.tree.flatten(tree)
    schema = (treedef, tuple(_leaf_schema(leaf) for leaf in leaves))
    try:
        hash(schema)
    except TypeError as error:
        raise UncacheableSchemaError(str(error)) from error
    return schema


def _leaf_schema(leaf: object) -> Hashable:
    """Return one leaf's abstract metadata."""
    if isinstance(leaf, jax.Array):
        return (
            "array",
            leaf.shape,
            leaf.dtype,
            leaf.weak_type,
            leaf.committed,
            leaf.sharding,
        )
    if isinstance(leaf, jax.ShapeDtypeStruct):
        return ("abstract", leaf.shape, leaf.dtype, leaf.weak_type, leaf.sharding)
    try:
        aval = jax.typeof(leaf)
    except TypeError as error:
        raise UncacheableSchemaError(
            f"A {type(leaf).__name__} leaf has no abstract array description."
        ) from error
    return ("host", type(leaf), aval.shape, aval.dtype, aval.weak_type)


def frozen_policy(value: object) -> Hashable:
    """Return a hashable, order-independent rendering of a policy value.

    Mappings become key-sorted item tuples, dataclasses their type and field
    items, sequences tuples and sets frozensets; every other value must already
    be hashable.

    Raises:
        UncacheableSchemaError: A value is neither a container nor hashable.

    """
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        (frozen_policy(key), frozen_policy(item))
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            ),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (field.name, frozen_policy(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, list | tuple):
        return tuple(frozen_policy(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(frozen_policy(item) for item in value)
    try:
        hash(value)
    except TypeError as error:
        raise UncacheableSchemaError(str(error)) from error
    return value
