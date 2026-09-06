"""Physical release of solve-time buffers: registry, eligibility, substitution."""

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.scheduler import (
    BufferRegistry,
    ReleaseRecord,
    buffer_identity,
    release_closed_artifacts,
    replace_leaf_by_identity,
)
from lcm.exceptions import ExecutionPlanningError


def _logger() -> logging.Logger:
    logger = logging.getLogger("lcm.tests.scheduler")
    logger.setLevel(logging.DEBUG)
    return logger


def test_buffer_identity_is_shared_by_two_names_of_one_array() -> None:
    """Identity follows the buffer, not the Python object."""
    array = jnp.arange(4.0)
    alias = jnp.asarray(array)

    assert buffer_identity(array=array) == buffer_identity(array=alias)


def test_buffer_identity_separates_a_copy_from_its_source() -> None:
    """A device_put that forbids aliasing the source buffer is a new buffer.

    A plain `jax.device_put(array, array.sharding)` is a documented no-op fast
    path on this JAX version when the array already lives on the requested
    sharding, so it returns a new Python wrapper over the *same* buffer.
    `may_alias=False` is the JAX-level request for a genuine copy.
    """
    array = jnp.arange(4.0)
    copy = jax.device_put(array, array.sharding, may_alias=False)

    assert buffer_identity(array=array) != buffer_identity(array=copy)


def test_forget_identity_drops_every_key_of_a_buffer_by_its_identity() -> None:
    """A buffer can be forgotten by the identity it had before it was deleted."""
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="value")
    identity = buffer_identity(array=array)

    registry.forget_identity(identity=identity)

    assert registry.artifacts_sharing(array=array) == frozenset()


def test_registry_reports_every_key_registered_on_one_buffer() -> None:
    """Two keys on one buffer are both visible from either key's array."""
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="value")
    registry.register(array=jnp.asarray(array), artifact="leaf")

    assert registry.artifacts_sharing(array=array) == frozenset({"value", "leaf"})


def test_release_deletes_a_buffer_whose_only_key_closed() -> None:
    """A closed, unretained, unaliased artifact is deleted."""
    ledger = PlannedInputLiveness(dispatch_accesses={"node": ("x",)})
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")

    release_closed_artifacts(
        ledger=ledger,
        registry=registry,
        artifacts=("x",),
        arrays_by_artifact=MappingProxyType({"x": array}),
        pending_outputs=(),
        closing_dispatch="node",
        logger=_logger(),
    )

    assert array.is_deleted()


def test_release_returns_one_record_per_released_key() -> None:
    """The record names the artifact and the dispatch that closed it."""
    ledger = PlannedInputLiveness(dispatch_accesses={"node": ("x",)})
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")

    records = release_closed_artifacts(
        ledger=ledger,
        registry=registry,
        artifacts=("x",),
        arrays_by_artifact=MappingProxyType({"x": array}),
        pending_outputs=(),
        closing_dispatch="node",
        logger=_logger(),
    )

    assert records == (ReleaseRecord(artifact="x", closing_dispatch="node"),)


def test_release_of_an_artifact_with_a_remaining_consumer_is_refused() -> None:
    """Dropping a buffer a planned consumer still reads is a planning error."""
    ledger = PlannedInputLiveness(dispatch_accesses={"first": ("x",), "second": ("x",)})
    ledger.commit_successful_dispatch(dispatch="first")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")

    with pytest.raises(ExecutionPlanningError, match="remaining consumer"):
        release_closed_artifacts(
            ledger=ledger,
            registry=registry,
            artifacts=("x",),
            arrays_by_artifact=MappingProxyType({"x": array}),
            pending_outputs=(),
            closing_dispatch="first",
            logger=_logger(),
        )


def test_a_buffer_shared_with_an_open_key_survives_its_closed_key() -> None:
    """A leaf that is also a retained value is kept while the value is retained."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={"node": ("leaf",)}, retained_artifacts=("value",)
    )
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="value")
    registry.register(array=array, artifact="leaf")

    release_closed_artifacts(
        ledger=ledger,
        registry=registry,
        artifacts=("leaf",),
        arrays_by_artifact=MappingProxyType({"leaf": array}),
        pending_outputs=(),
        closing_dispatch="node",
        logger=_logger(),
    )

    assert not array.is_deleted()


def test_release_blocks_on_the_pending_outputs_before_deleting() -> None:
    """Every output the period dispatched is ready before a buffer goes."""
    ledger = PlannedInputLiveness(dispatch_accesses={"node": ("x",)})
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")
    output = jnp.cumsum(jnp.arange(1024.0))

    release_closed_artifacts(
        ledger=ledger,
        registry=registry,
        artifacts=("x",),
        arrays_by_artifact=MappingProxyType({"x": array}),
        pending_outputs=(output,),
        closing_dispatch="node",
        logger=_logger(),
    )

    assert output.is_fully_replicated
    assert not output.is_deleted()


def test_release_logs_the_artifact_key_and_the_closing_dispatch() -> None:
    """The debug record carries both facts as attributes, not only as text."""
    ledger = PlannedInputLiveness(dispatch_accesses={"node": ("x",)})
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # ty: ignore[invalid-assignment]
    logger = _logger()
    logger.addHandler(handler)
    try:
        release_closed_artifacts(
            ledger=ledger,
            registry=registry,
            artifacts=("x",),
            arrays_by_artifact=MappingProxyType({"x": array}),
            pending_outputs=(),
            closing_dispatch="node",
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)

    logged = [
        (r.artifact_key, r.closing_dispatch)  # ty: ignore[unresolved-attribute]
        for r in records
    ]

    assert logged == [("x", "node")]


def test_replace_leaf_by_identity_swaps_exactly_the_named_leaf() -> None:
    """Only the leaf that is the old object is replaced; equal values stay."""
    old = jnp.zeros(2)
    equal_but_other = jnp.zeros(2)
    tree = MappingProxyType({"a": old, "b": equal_but_other})
    new = jnp.ones(2)

    replaced = cast(
        "Mapping[str, jax.Array]", replace_leaf_by_identity(tree=tree, old=old, new=new)
    )

    assert (replaced["a"] is new, replaced["b"] is equal_but_other) == (True, True)


def test_replace_leaf_by_identity_refuses_a_leaf_that_is_not_in_the_tree() -> None:
    """A substitution that would change nothing is a caller error."""
    with pytest.raises(ValueError, match="not a leaf"):
        replace_leaf_by_identity(
            tree=MappingProxyType({"a": jnp.zeros(2)}),
            old=jnp.zeros(2),
            new=jnp.ones(2),
        )
