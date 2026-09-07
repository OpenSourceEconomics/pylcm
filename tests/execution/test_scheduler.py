"""Physical release of solve-time buffers: registry, eligibility, substitution.

A release only ever frees a buffer a dispatch produced. A buffer declared
foreign — one the model holds, or one an output took straight from an input —
is kept, and says so on its own log-record attribute.
"""

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
    DispatchUnit,
    ReleaseRecord,
    ScheduledNode,
    buffer_identity,
    plan_period_waves,
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


def test_a_declared_foreign_buffer_is_reported_as_unproduced() -> None:
    """An array declared through a tree is recognised on its buffer."""
    registry = BufferRegistry()
    array = jnp.arange(4.0)

    registry.declare_not_produced(tree={"grids": {"wealth": array}})

    assert registry.is_not_produced(array=array)


def test_an_undeclared_buffer_is_reported_as_produced() -> None:
    """A buffer no declaration named belongs to the dispatch that made it."""
    registry = BufferRegistry()
    registry.declare_not_produced(tree={"grids": {"wealth": jnp.arange(4.0)}})

    assert not registry.is_not_produced(array=jnp.arange(4.0) + 1.0)


def test_a_declared_foreign_buffer_survives_a_closed_key() -> None:
    """A release leaves a buffer no dispatch produced in place."""
    ledger = PlannedInputLiveness(dispatch_accesses={"node": ("x",)})
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")
    registry.declare_not_produced(tree=(array,))

    release_closed_artifacts(
        ledger=ledger,
        registry=registry,
        artifacts=("x",),
        arrays_by_artifact=MappingProxyType({"x": array}),
        pending_outputs=(),
        closing_dispatch="node",
        logger=_logger(),
    )

    assert not array.is_deleted()


def test_a_kept_buffer_yields_no_release_record() -> None:
    """Keeping a buffer is not a release, so it reports none."""
    ledger = PlannedInputLiveness(dispatch_accesses={"node": ("x",)})
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")
    registry.declare_not_produced(tree=(array,))

    records = release_closed_artifacts(
        ledger=ledger,
        registry=registry,
        artifacts=("x",),
        arrays_by_artifact=MappingProxyType({"x": array}),
        pending_outputs=(),
        closing_dispatch="node",
        logger=_logger(),
    )

    assert records == ()


def test_a_kept_buffer_is_logged_under_its_own_record_attribute() -> None:
    """The debug record of a kept key is distinct from a released one."""
    ledger = PlannedInputLiveness(dispatch_accesses={"node": ("x",)})
    ledger.commit_successful_dispatch(dispatch="node")
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    registry.register(array=array, artifact="x")
    registry.declare_not_produced(tree=(array,))
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
        (r.kept_artifact_key, r.closing_dispatch)  # ty: ignore[unresolved-attribute]
        for r in records
        if hasattr(r, "kept_artifact_key")
    ]

    assert logged == [("x", "node")]


def test_an_output_on_an_input_buffer_is_marked_unproduced() -> None:
    """An output the dispatch handed straight through is not its own product."""
    registry = BufferRegistry()
    array = jnp.arange(4.0)

    registry.declare_passed_through(inputs=(array,), outputs=(jnp.asarray(array),))

    assert registry.is_not_produced(array=array)


def test_a_freshly_computed_output_stays_produced() -> None:
    """An output on its own buffer remains one the dispatch produced."""
    registry = BufferRegistry()
    array = jnp.arange(4.0)
    computed = array + 1.0

    registry.declare_passed_through(inputs=(array,), outputs=(computed,))

    assert not registry.is_not_produced(array=computed)


def test_an_input_no_output_reused_stays_releasable() -> None:
    """Declaring a pass-through leaves the dispatch's other inputs alone."""
    registry = BufferRegistry()
    consumed = jnp.arange(4.0)
    passed = jnp.arange(3.0)

    registry.declare_passed_through(
        inputs=(consumed, passed), outputs=(jnp.asarray(passed),)
    )

    assert not registry.is_not_produced(array=consumed)


def test_a_same_period_value_handed_on_is_marked_unproduced() -> None:
    """A regime republishing another's value of this period did not produce it."""
    registry = BufferRegistry()
    reference_value = jnp.arange(4.0)
    period_solution = {"reference": reference_value}

    registry.declare_passed_through(
        inputs=(MappingProxyType({}), period_solution),
        outputs=(jnp.asarray(reference_value),),
    )

    assert registry.is_not_produced(array=reference_value)


def test_an_unfolded_edge_carried_forward_is_marked_unproduced() -> None:
    """An edge the fold left alone keeps the buffer it already had."""
    registry = BufferRegistry()
    carried = jnp.arange(4.0)
    edges = MappingProxyType({("source", "target"): carried})

    registry.declare_passed_through(inputs=(edges,), outputs=(dict(edges),))

    assert registry.is_not_produced(array=carried)


def test_a_declaration_over_two_channels_marks_the_shared_buffer() -> None:
    """A payload retained on one channel protects the array another published."""
    registry = BufferRegistry()
    shared = jnp.arange(4.0)

    registry.declare_not_produced(
        tree={("regime", "replay-key"): shared, ("regime", "aux-key"): shared}
    )

    assert registry.is_not_produced(array=shared)


def _node(*, regime: str, program: str = "main") -> ScheduledNode:
    return ScheduledNode(period=3, regime=regime, program=program)


def test_independent_units_on_disjoint_devices_share_one_wave() -> None:
    """Two regimes without a same-period read between them dispatch together."""
    waves = plan_period_waves(
        nodes=(_node(regime="a"), _node(regime="b")),
        same_period_dependencies=MappingProxyType({}),
        device_sets=MappingProxyType({"a": frozenset({0}), "b": frozenset({1})}),
    )

    assert waves == (
        (
            DispatchUnit(period=3, regime="a", programs=("main",)),
            DispatchUnit(period=3, regime="b", programs=("main",)),
        ),
    )


def test_a_same_period_read_orders_the_reader_after_its_reference() -> None:
    """A regime reading another regime's same-period value waits for it."""
    waves = plan_period_waves(
        nodes=(_node(regime="reader"), _node(regime="reference")),
        same_period_dependencies=MappingProxyType({"reader": ("reference",)}),
        device_sets=MappingProxyType(
            {"reader": frozenset({0}), "reference": frozenset({1})}
        ),
    )

    assert [tuple(unit.regime for unit in wave) for wave in waves] == [
        ("reference",),
        ("reader",),
    ]


def test_units_sharing_a_device_dispatch_in_separate_waves() -> None:
    """Concurrency needs disjoint submeshes; a shared device serializes."""
    waves = plan_period_waves(
        nodes=(_node(regime="a"), _node(regime="b")),
        same_period_dependencies=MappingProxyType({}),
        device_sets=MappingProxyType({"a": frozenset({0, 1}), "b": frozenset({1})}),
    )

    assert [tuple(unit.regime for unit in wave) for wave in waves] == [
        ("a",),
        ("b",),
    ]


def test_the_programs_of_one_kernel_form_one_unit_in_topological_order() -> None:
    """A kernel's internal edge keeps its programs in one dispatch unit."""
    waves = plan_period_waves(
        nodes=(
            _node(regime="a", program="keeper"),
            _node(regime="a", program="sweep"),
        ),
        same_period_dependencies=MappingProxyType({}),
        device_sets=MappingProxyType({"a": frozenset({0})}),
    )

    assert waves == (
        (DispatchUnit(period=3, regime="a", programs=("keeper", "sweep")),),
    )


def test_declaration_order_breaks_ties_inside_a_wave() -> None:
    """Independent units keep the order their regimes were declared in."""
    waves = plan_period_waves(
        nodes=(_node(regime="second"), _node(regime="first")),
        same_period_dependencies=MappingProxyType({}),
        device_sets=MappingProxyType(
            {"second": frozenset({0}), "first": frozenset({1})}
        ),
    )

    assert tuple(unit.regime for unit in waves[0]) == ("second", "first")


def test_a_dependency_cycle_is_refused() -> None:
    """Two regimes reading each other's same-period value cannot be scheduled."""
    with pytest.raises(ExecutionPlanningError, match="cycle"):
        plan_period_waves(
            nodes=(_node(regime="a"), _node(regime="b")),
            same_period_dependencies=MappingProxyType({"a": ("b",), "b": ("a",)}),
            device_sets=MappingProxyType({"a": frozenset({0}), "b": frozenset({1})}),
        )


def test_a_regime_missing_from_device_sets_is_refused_by_name() -> None:
    """A regime with no declared device set is a planning error, not a KeyError."""
    with pytest.raises(ExecutionPlanningError, match="b"):
        plan_period_waves(
            nodes=(_node(regime="a"), _node(regime="b")),
            same_period_dependencies=MappingProxyType({}),
            device_sets=MappingProxyType({"a": frozenset({0})}),
        )


def test_nodes_of_two_periods_are_refused() -> None:
    """A wave plan covers one period; periods stay strictly backward."""
    with pytest.raises(ValueError, match="one period"):
        plan_period_waves(
            nodes=(
                _node(regime="a"),
                ScheduledNode(period=2, regime="b", program="main"),
            ),
            same_period_dependencies=MappingProxyType({}),
            device_sets=MappingProxyType({"a": frozenset({0}), "b": frozenset({1})}),
        )
