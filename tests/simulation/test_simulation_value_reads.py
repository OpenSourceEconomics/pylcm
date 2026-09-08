"""A period's declared regime readers own only their temporary value copies."""

import gc
import importlib
import weakref
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import ValueRead
from _lcm.execution.scheduler import shares_a_buffer
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    apply_value_transfer,
)
from _lcm.simulation.value_reads import PeriodSimulationReads
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import DISSOLUTION_FLAG, SIMULATION_POLICY


def _read(*, unit: str, core: str = "argmax") -> ValueRead:
    """Name one concrete simulation family reading next period's target value."""
    return ValueRead(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime=unit,
            core_key=core,
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("target",),
        ),
    )


def _stored_value() -> jax.Array:
    """Use a named layout so the one-device destination is a real adapter edge."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("stored",))
    return jax.device_put(jnp.arange(3), jax.NamedSharding(mesh, jax.P()))


def _physical_copy(*, value: object, transfer: ResolvedValueTransfer) -> jax.Array:
    """Make the actual adapter's output physically distinct on one CPU."""
    assert isinstance(value, jax.Array)
    result = apply_value_transfer(value=value, transfer=transfer).copy()
    result.block_until_ready()
    assert not shares_a_buffer(first=result, second=value)
    return result


@pytest.fixture
def owner_factory(monkeypatch: pytest.MonkeyPatch) -> type[PeriodSimulationReads]:
    """Keep the actual adapter metadata checks and force nonvacuous copy ownership."""
    module = importlib.import_module("_lcm.simulation.value_reads")
    monkeypatch.setattr(module, "apply_value_transfer", _physical_copy)
    return module.PeriodSimulationReads


def test_a_single_reader_releases_its_copy_but_preserves_the_original(
    *, owner_factory: type[PeriodSimulationReads]
) -> None:
    """A single complete regime dispatch closes a physically distinct input copy."""
    read = _read(unit="alive")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
    )
    copied = owner.read(unit="alive", read=read, value=stored)
    owner.commit(unit="alive", outputs={"actions": copied + 2})
    owner.finish()

    assert copied.is_deleted()
    np.testing.assert_array_equal(stored, np.array([0, 1, 2]))


def test_occurrences_share_one_copy_until_the_last_regime_commits(
    *, owner_factory: type[PeriodSimulationReads]
) -> None:
    """Several families in one regime count once, and the next regime keeps the copy."""
    first = _read(unit="first")
    another = _read(unit="first", core="transition")
    second = _read(unit="second")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"first": (first, another), "second": (second,)},
        release_enabled=True,
    )
    copied = owner.read(unit="first", read=first, value=stored)
    assert owner.read(unit="first", read=first, value=stored) is copied
    assert owner.read(unit="first", read=another, value=stored) is copied
    owner.commit(unit="first", outputs=copied + 1)
    assert not copied.is_deleted()
    assert owner.read(unit="second", read=second, value=stored) is copied
    owner.commit(unit="second", outputs=copied + 2)
    owner.finish()

    assert copied.is_deleted()
    np.testing.assert_array_equal(stored, np.array([0, 1, 2]))


@pytest.mark.parametrize("unused_first", [True, False])
def test_an_unused_declaration_does_not_leave_a_copy_live(
    *, owner_factory: type[PeriodSimulationReads], unused_first: bool
) -> None:
    """An unused unit can commit before materialization or close a pending reader."""
    unused = _read(unit="unused")
    used = _read(unit="used")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"unused": (unused,), "used": (used,)},
        release_enabled=True,
    )
    if unused_first:
        owner.commit(unit="unused", outputs=())
    copied = owner.read(unit="used", read=used, value=stored)
    owner.commit(unit="used", outputs=copied + 1)
    if not unused_first:
        assert not copied.is_deleted()
        owner.commit(unit="unused", outputs=())
    owner.finish()

    assert copied.is_deleted()


@pytest.mark.parametrize("retained_role", ["actions", "states", "carry", "mask"])
def test_a_copy_returned_in_the_complete_output_tree_stays_readable(
    *, owner_factory: type[PeriodSimulationReads], retained_role: str
) -> None:
    """Any published or carried output alias protects its input's physical shards."""
    read = _read(unit="alive")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
    )
    copied = owner.read(unit="alive", read=read, value=stored)
    output = {retained_role: {"nested": (copied,)}}
    owner.commit(unit="alive", outputs=output)
    owner.finish()

    np.testing.assert_array_equal(output[retained_role]["nested"][0], [0, 1, 2])


def test_eager_mode_never_explicitly_deletes_the_copy(
    *, owner_factory: type[PeriodSimulationReads]
) -> None:
    """A release-disabled owner leaves an acquired array readable after finish."""
    read = _read(unit="alive")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )
    copied = owner.read(unit="alive", read=read, value=stored)
    owner.commit(unit="alive", outputs=copied + 1)
    owner.finish()

    np.testing.assert_array_equal(copied, [0, 1, 2])


def test_finish_keeps_no_concrete_array_references(
    *, owner_factory: type[PeriodSimulationReads]
) -> None:
    """An eager owner keeps only metadata after its complete scope closes."""
    read = _read(unit="alive")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )
    copied = owner.read(unit="alive", read=read, value=stored)
    output = copied + 1
    references = (weakref.ref(stored), weakref.ref(copied), weakref.ref(output))
    owner.commit(unit="alive", outputs={"carry": (output,)})
    owner.finish()
    del copied, stored, output
    gc.collect()

    assert tuple(reference() for reference in references) == (None, None, None)


@pytest.mark.parametrize(
    "change",
    [
        {"core_key": "transition"},
        {"channel": ValueInputChannel.SAME_PERIOD_VALUE},
        {"argument": "another_input"},
        {"path": ("neighbor",)},
    ],
)
def test_undeclared_reader_occurrences_are_rejected(*, change: dict) -> None:
    """Matching the target does not authorize another core, channel or leaf."""
    read = _read(unit="alive")
    undeclared = replace(read, source=replace(read.source, **change))
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )

    with pytest.raises(ExecutionPlanningError, match="undeclared"):
        owner.read(unit="alive", read=undeclared, value=_stored_value())


def test_changing_the_source_array_of_an_artifact_is_rejected(
    *, owner_factory: type[PeriodSimulationReads]
) -> None:
    """A cache hit cannot substitute a different equal-valued source artifact."""
    read = _read(unit="alive")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )
    owner.read(unit="alive", read=read, value=stored)

    with pytest.raises(ExecutionPlanningError, match="source array"):
        owner.read(unit="alive", read=read, value=stored.copy())


@pytest.mark.parametrize("operation", ["read", "commit"])
def test_an_unknown_unit_is_rejected(*, operation: str) -> None:
    """Only the declared active regime roster can acquire or commit."""
    read = _read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )

    attempt = (
        partial(owner.read, unit="unknown", read=read, value=_stored_value())
        if operation == "read"
        else partial(owner.commit, unit="unknown", outputs=())
    )
    with pytest.raises(ExecutionPlanningError, match=r"Unknown.*unknown"):
        attempt()


@pytest.mark.parametrize("operation", ["read", "commit"])
def test_a_committed_unit_cannot_read_or_commit_again(*, operation: str) -> None:
    """A closed unit cannot recopy a deleted entry or decrement its count twice."""
    read = _read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )
    owner.commit(unit="alive", outputs=())

    attempt = (
        partial(owner.read, unit="alive", read=read, value=_stored_value())
        if operation == "read"
        else partial(owner.commit, unit="alive", outputs=())
    )
    with pytest.raises(ExecutionPlanningError, match="committed"):
        attempt()


def test_finish_refuses_missing_commits() -> None:
    """Every active regime must commit even when it acquired no declared input."""
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": ()},
        release_enabled=False,
    )

    with pytest.raises(ExecutionPlanningError, match=r"uncommitted.*alive"):
        owner.finish()


@pytest.mark.parametrize("change", [{"source_period": 1}, {"source_regime": "other"}])
def test_reader_addresses_match_the_declared_period_and_unit(*, change: dict) -> None:
    """A roster cannot assign a different period's or regime's occurrence to a unit."""
    read = _read(unit="alive")
    misaddressed = replace(read, source=replace(read.source, **change))

    with pytest.raises(ExecutionPlanningError, match=r"reader.*period.*unit"):
        PeriodSimulationReads(
            period=0,
            devices=tuple(jax.devices()[:1]),
            reads_by_unit={"alive": (misaddressed,)},
            release_enabled=False,
        )


@pytest.mark.parametrize("shared", [True, False])
def test_single_and_shared_copies_keep_honest_transfer_metadata(
    *, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """One physical acquisition records sharing only when another unit remains."""
    module = importlib.import_module("_lcm.simulation.value_reads")
    transfers: list[ResolvedValueTransfer] = []

    def observe(*, value: object, transfer: ResolvedValueTransfer) -> jax.Array:
        transfers.append(transfer)
        return _physical_copy(value=value, transfer=transfer)

    monkeypatch.setattr(module, "apply_value_transfer", observe)
    first = _read(unit="first")
    second = _read(unit="second")
    stored = _stored_value()
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"first": (first,), "second": (second,)},
        release_enabled=True,
    )
    if not shared:
        owner.commit(unit="second", outputs=())
    copied = owner.read(unit="first", read=first, value=stored)
    assert owner.read(unit="first", read=first, value=stored) is copied
    owner.commit(unit="first", outputs=copied + 1)
    if shared:
        assert owner.read(unit="second", read=second, value=stored) is copied
        owner.commit(unit="second", outputs=copied + 2)
    owner.finish()

    assert [transfer.reused_by_several_consumers for transfer in transfers] == [shared]


def test_every_output_is_ready_before_a_copy_can_be_released(
    *, owner_factory: type[PeriodSimulationReads], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release barrier includes action, state and carry leaves beyond values."""
    read = _read(unit="alive")
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
    )
    copied = owner.read(unit="alive", read=read, value=_stored_value())
    outputs = {
        "value": copied + 1,
        "actions": copied + 2,
        "next_states": {"state": copied + 3},
        "carry": (copied + 4,),
    }
    observed: list[tuple[int, ...]] = []
    original = jax.block_until_ready

    def observe(tree: object) -> object:
        assert not copied.is_deleted()
        observed.append(tuple(id(leaf) for leaf in jax.tree.leaves(tree)))
        return original(tree)

    with monkeypatch.context() as patch:
        patch.setattr(jax, "block_until_ready", observe)
        owner.commit(unit="alive", outputs=outputs)
    owner.finish()

    assert observed[0] == tuple(id(leaf) for leaf in jax.tree.leaves(outputs))
    assert copied.is_deleted()


def test_an_aligned_read_keeps_the_original_array() -> None:
    """A destination layout already satisfied by the source needs no owned copy."""
    read = _read(unit="alive")
    stored = jax.device_put(
        jnp.arange(3), jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
    )
    assert owner.read(unit="alive", read=read, value=stored) is stored
    owner.commit(unit="alive", outputs=stored + 1)
    owner.finish()

    np.testing.assert_array_equal(stored, [0, 1, 2])


def test_incomplete_finish_drops_copied_wrappers_and_closes_the_scope(
    *, owner_factory: type[PeriodSimulationReads]
) -> None:
    """An interrupted scope reports its missing commit without retaining arrays."""
    read = _read(unit="alive")
    stored = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )
    copied = owner.read(unit="alive", read=read, value=stored)
    reference = weakref.ref(copied)
    with pytest.raises(ExecutionPlanningError, match="uncommitted"):
        owner.finish()
    del copied
    gc.collect()

    assert reference() is None
    with pytest.raises(ExecutionPlanningError, match="finished"):
        owner.read(unit="alive", read=read, value=stored)


def test_budget_refusal_happens_before_a_transfer_allocates(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected copy never enters the physical adapter."""
    module = importlib.import_module("_lcm.simulation.value_reads")
    allocated: list[ResolvedValueTransfer] = []

    def observe(*, value: object, transfer: ResolvedValueTransfer) -> jax.Array:
        allocated.append(transfer)
        return _physical_copy(value=value, transfer=transfer)

    def refuse(
        *, transfer: ResolvedValueTransfer, live_values: tuple[jax.Array, ...]
    ) -> None:
        assert transfer.target.regime == "target"
        assert any(value is stored for value in live_values)
        raise ExecutionPlanningError("budget refused")

    monkeypatch.setattr(module, "apply_value_transfer", observe)
    stored = _stored_value()
    read = _read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
        before_transfer=refuse,
    )
    with pytest.raises(ExecutionPlanningError, match="budget refused"):
        owner.read(unit="alive", read=read, value=stored)

    assert allocated == []


def test_the_next_copy_sees_pending_copies_and_retained_outputs(
    *, owner_factory: type[PeriodSimulationReads]
) -> None:
    """The budget view includes earlier units' live copies and complete outputs."""
    first = _read(unit="first")
    second = _read(unit="second")
    neighbor = replace(
        second,
        target=replace(second.target, regime="neighbor"),
        source=replace(second.source, path=("neighbor",)),
    )
    seen: list[tuple[int, ...]] = []

    def observe(
        *, transfer: ResolvedValueTransfer, live_values: tuple[jax.Array, ...]
    ) -> None:
        assert transfer.target.regime in {"target", "neighbor"}
        seen.append(tuple(id(value) for value in live_values))

    stored = _stored_value()
    other = _stored_value()
    owner = owner_factory(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"first": (first,), "second": (second, neighbor)},
        release_enabled=True,
        before_transfer=observe,
    )
    copied = owner.read(unit="first", read=first, value=stored)
    output = copied + 1
    owner.commit(unit="first", outputs={"carry": output})
    assert owner.read(unit="second", read=second, value=stored) is copied
    another = owner.read(unit="second", read=neighbor, value=other)

    assert len(seen) == 2
    assert {id(stored), id(other), id(copied), id(output)} <= set(seen[1])
    owner.commit(unit="second", outputs=another + copied)
    assert all(not value.is_deleted() for value in owner.live_values)
    owner.finish()
    assert owner.live_values == ()


def test_an_aligned_read_never_invokes_the_allocation_callback() -> None:
    """Already aligned originals require neither a copy nor a copy-budget check."""

    def refuse(
        *, transfer: ResolvedValueTransfer, live_values: tuple[jax.Array, ...]
    ) -> None:
        raise AssertionError(
            f"An aligned read allocated {transfer.kind.value} "
            f"with {len(live_values)} live arrays"
        )

    read = _read(unit="alive")
    stored = jax.device_put(
        jnp.arange(3), jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
        before_transfer=refuse,
    )
    assert owner.read(unit="alive", read=read, value=stored) is stored
    owner.commit(unit="alive", outputs=())
    owner.finish()


def test_finish_releases_a_budget_callback_and_its_captured_roots() -> None:
    """The closed owner must not retain an accountant closure over original arrays."""
    marker = _stored_value()

    def observe(
        *,
        transfer: ResolvedValueTransfer,
        live_values: tuple[jax.Array, ...],
        marker: jax.Array,
    ) -> None:
        assert transfer.expected_shape == marker.shape
        jax.block_until_ready((marker, *live_values))

    callback = partial(observe, marker=marker)
    references = (weakref.ref(callback), weakref.ref(marker))
    owner = PeriodSimulationReads(
        period=0,
        devices=tuple(jax.devices()[:1]),
        reads_by_unit={},
        release_enabled=True,
        before_transfer=callback,
    )
    del callback, marker
    owner.finish()
    gc.collect()

    assert tuple(reference() for reference in references) == (None, None)


def _host_replay_read(*, unit: str, core: str = "replay") -> ValueRead:
    """Name one future replay artifact independently of its consuming unit."""
    read = _read(unit=unit, core=core)
    return replace(
        read,
        target=replace(
            read.target,
            kind=ValueArtifactKind.REPLAY_ARTIFACT_LEAF,
            artifact_key=SIMULATION_POLICY,
        ),
        source=replace(read.source, channel=ValueInputChannel.NEXT_REPLAY_ARTIFACT),
    )


@pytest.mark.parametrize("unused_first", [True, False])
def test_host_replay_counts_distinct_units_and_keeps_originals_readable(
    *, unused_first: bool
) -> None:
    """Repeated occurrences count once and unused units close their real count."""
    original = np.array([1, 4, 7], dtype=np.int32)
    first = _host_replay_read(unit="first")
    another = _host_replay_read(unit="first", core="transition")
    second = _host_replay_read(unit="second")
    unused = _host_replay_read(unit="unused")
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={
            "first": (first, another),
            "second": (second,),
            "unused": (unused,),
        },
        release_enabled=True,
    )
    if unused_first:
        owner.commit(unit="unused", outputs=())
    copied = owner.read(unit="first", read=first, value=original)
    assert owner.read(unit="first", read=another, value=original) is copied
    assert owner.read(unit="first", read=first, value=original) is copied
    assert not np.shares_memory(np.asarray(copied), original)
    assert len(owner.live_values) == 1
    assert owner.live_values[0] is copied
    owner.commit(unit="first", outputs=copied + 1)
    assert not copied.is_deleted()
    assert owner.read(unit="second", read=second, value=original) is copied
    owner.commit(unit="second", outputs=copied + 2)
    if not unused_first:
        assert not copied.is_deleted()
        owner.commit(unit="unused", outputs=())
    assert copied.is_deleted()
    owner.finish()
    np.testing.assert_array_equal(original, [1, 4, 7])


@pytest.mark.parametrize(
    ("release_enabled", "published"), [(True, True), (False, False), (False, True)]
)
def test_published_or_eager_host_replay_copies_remain_readable(
    *, release_enabled: bool, published: bool
) -> None:
    """Passing the upload through a result, or eager mode, forbids deletion."""
    original = np.array([True, False, True])
    read = _host_replay_read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={"alive": (read,)},
        release_enabled=release_enabled,
    )
    copied = owner.read(unit="alive", read=read, value=original)
    outputs = {"carry": (copied,)} if published else {"carry": (~copied,)}
    owner.commit(unit="alive", outputs=outputs)
    owner.finish()
    np.testing.assert_array_equal(copied, [True, False, True])
    np.testing.assert_array_equal(original, [True, False, True])


@pytest.mark.parametrize("complete", [True, False])
def test_finish_drops_host_originals_and_uploads_even_on_incomplete_exit(
    *, complete: bool
) -> None:
    """No host source or device wrapper survives solely through a closed owner."""
    original = np.array([1, 4, 7], dtype=np.int32)
    read = _host_replay_read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={"alive": (read,)},
        release_enabled=False,
    )
    copied = owner.read(unit="alive", read=read, value=original)
    copied.block_until_ready()
    references = (weakref.ref(original), weakref.ref(copied))
    if complete:
        owner.commit(unit="alive", outputs=())
        owner.finish()
    else:
        with pytest.raises(ExecutionPlanningError, match="uncommitted"):
            owner.finish()
    assert owner.live_values == ()
    del original, copied
    gc.collect()
    assert tuple(reference() for reference in references) == (None, None)


def test_host_replay_rejects_a_replaced_original_at_the_same_artifact_address() -> None:
    """Equal contents do not permit a different source to reuse an addressed upload."""
    original = np.array([1, 4, 7], dtype=np.int32)
    read = _host_replay_read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
    )
    copied = owner.read(unit="alive", read=read, value=original)
    with pytest.raises(ExecutionPlanningError, match="source array changed"):
        owner.read(unit="alive", read=read, value=original.copy())
    owner.commit(unit="alive", outputs=copied + 1)
    owner.finish()


def test_a_device_transfer_budget_callback_cannot_admit_a_host_upload(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse host provenance before allocation without charging a fake transfer."""

    def no_device_transfer(
        *, transfer: ResolvedValueTransfer, live_values: tuple[jax.Array, ...]
    ) -> None:
        raise AssertionError(f"Invented device transfer: {transfer!r}, {live_values!r}")

    def no_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            f"Host upload happened before refusal: {args!r}, {kwargs!r}"
        )

    original = np.array([1, 4, 7], dtype=np.int32)
    read = _host_replay_read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
        before_transfer=no_device_transfer,
    )
    monkeypatch.setattr(jax, "device_put", no_allocation)
    with pytest.raises(ExecutionPlanningError, match=r"Host replay placement.*budget"):
        owner.read(unit="alive", read=read, value=original)
    owner.commit(unit="alive", outputs=())
    owner.finish()
    np.testing.assert_array_equal(original, [1, 4, 7])


def test_host_replay_preserves_distinct_artifact_keys_for_one_numpy_original() -> None:
    """Publishing one key must not keep another key's temporary copy alive."""
    original = np.array([True, False, True])
    policy = _host_replay_read(unit="alive")
    flags = replace(
        policy, target=replace(policy.target, artifact_key=DISSOLUTION_FLAG)
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={"alive": (policy, flags)},
        release_enabled=True,
    )
    copied_policy = owner.read(unit="alive", read=policy, value=original)
    copied_flags = owner.read(unit="alive", read=flags, value=original)
    assert copied_policy is not copied_flags
    assert not shares_a_buffer(first=copied_policy, second=copied_flags)
    owner.commit(unit="alive", outputs={"policy": copied_policy})
    owner.finish()
    assert copied_flags.is_deleted()
    np.testing.assert_array_equal(copied_policy, [True, False, True])
    np.testing.assert_array_equal(original, [True, False, True])
