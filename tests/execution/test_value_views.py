"""Selected value views: reading one invariant-coordinate block of a stored value.

A value view names a stored artifact by its original logical address and says
which canonical coordinates a consumer reads. A selected view is computed by
selecting first and moving only the selected block; an unsliced shared leaf is
declared as such. Neither is ever inferred from matching array lengths.
"""

import math
from collections.abc import Hashable, Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.abstract_program_inputs import abstract_program_inputs
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    MaterializedCoreProgram,
    ResolvedCoreProgram,
    ValueRead,
    resolve_core_program,
)
from _lcm.execution.donation import (
    DonatedBuffer,
    resolve_donations,
    unit_input_readers,
    withhold_shared_donations,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.output_layout import VALUE
from _lcm.execution.scheduler import BufferRegistry, PeriodTransferCache
from _lcm.execution.value_transfer import (
    CoordinateSelection,
    ResolvedValueTransfer,
    TransferStageKind,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    ValueViewDescriptor,
    ValueViewLeaf,
    _select_value_view,
    apply_value_transfer,
    apply_value_transfer_plan,
    transfer_result_key,
)
from _lcm.execution.value_views import (
    fail_if_value_transfer_exceeds_budget,
    plan_value_transfer_footprint,
)
from _lcm.typing import ArgumentTree, PytreeValue, ShapeDtypePytree
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import RegimeName, StateName

_STATES = ("pref_type", "assets", "health")
_SHAPE = (3, 4, 2)
_ARTIFACT = ValueArtifactAddress(
    kind=ValueArtifactKind.REGIME_VALUE, period=4, regime="retired"
)


def _compiled_selections() -> int:
    """Return how many executables the selection stage has compiled so far."""
    return _select_value_view._cache_size()  # ty: ignore[unresolved-attribute]


def _device() -> jax.Device:
    return jax.devices()[0]


def _single() -> jax.sharding.Sharding:
    return jax.sharding.SingleDeviceSharding(_device())


def _named() -> jax.sharding.Sharding:
    """A replicated one-device mesh layout, a different layout kind than `_single`."""
    mesh = jax.sharding.Mesh(np.asarray([_device()]), ("device",))
    return jax.sharding.NamedSharding(mesh, jax.P())


def _values(*, shape: tuple[int, ...] = _SHAPE, offset: float = 0.0) -> np.ndarray:
    """Distinct, exactly representable float32 entries, so a wrong block shows."""
    values = (np.arange(math.prod(shape), dtype=np.float32) * 1.5 - 7.0) + offset
    assert np.unique(values).size == values.size
    return values.reshape(shape)


def _stored(*, values: np.ndarray, sharding: jax.sharding.Sharding) -> jax.Array:
    stored = jax.device_put(values, sharding)
    assert stored.dtype == jnp.float32
    assert np.array_equal(np.asarray(stored), values)
    return stored


def _source(*, core_key: str = "main") -> ValueConsumerAddress:
    return ValueConsumerAddress(
        source_period=3,
        source_regime="working",
        core_key=core_key,
        channel=ValueInputChannel.NEXT_REGIME_VALUE,
        path=("retired",),
    )


def _selected_view(
    *,
    code: int,
    required: jax.sharding.Sharding,
    axis_names: tuple[str, ...] = _STATES,
    shape: tuple[int, ...] = _SHAPE,
    state: StateName = "pref_type",
    keep_axis: bool = False,
    width: int = 1,
) -> ValueViewDescriptor:
    axis = axis_names.index(state)
    consumer = list(shape)
    if keep_axis:
        consumer[axis] = width
    else:
        del consumer[axis]
    return ValueViewDescriptor(
        artifact=_ARTIFACT,
        leaf=ValueViewLeaf.SELECTED,
        stored_axis_names=axis_names,
        stored_shape=shape,
        dtype=jnp.float32,
        weak_type=False,
        consumer_shape=tuple(consumer),
        required_sharding=required,
        selections=(
            CoordinateSelection(
                state_name=state,
                start=code,
                width=width,
                codes=tuple(range(code, code + width)),
                keep_axis=keep_axis,
            ),
        ),
    )


def _shared_view(*, required: jax.sharding.Sharding) -> ValueViewDescriptor:
    return ValueViewDescriptor(
        artifact=_ARTIFACT,
        leaf=ValueViewLeaf.SHARED,
        stored_axis_names=_STATES,
        stored_shape=_SHAPE,
        dtype=jnp.float32,
        weak_type=False,
        consumer_shape=_SHAPE,
        required_sharding=required,
    )


def _transfer(
    *,
    stored: jax.Array,
    view: ValueViewDescriptor | None,
    required: jax.sharding.Sharding,
    kind: ValueTransferKind,
    core_key: str = "main",
    reused: bool = False,
) -> ResolvedValueTransfer:
    return ResolvedValueTransfer(
        target=_ARTIFACT,
        source=_source(core_key=core_key),
        kind=kind,
        stored_sharding=stored.sharding,
        source_sharding=required,
        expected_shape=stored.shape,
        expected_dtype=stored.dtype,
        reused_by_several_consumers=reused,
        view=view,
    )


def _selected_transfer(
    *,
    stored: jax.Array,
    code: int,
    copy: bool = False,
    core_key: str = "main",
    reused: bool = False,
) -> ResolvedValueTransfer:
    required = _named() if copy else _single()
    return _transfer(
        stored=stored,
        view=_selected_view(code=code, required=required),
        required=required,
        kind=(
            ValueTransferKind.COPY_TO_SOURCE_LAYOUT
            if copy
            else ValueTransferKind.ALIGNED_LOCAL
        ),
        core_key=core_key,
        reused=reused,
    )


@pytest.mark.parametrize("code", [0, 1, 2])
@pytest.mark.parametrize("copy", [False, True], ids=["aligned", "copy"])
def test_a_selected_view_delivers_exactly_the_selected_block(
    *, code: int, copy: bool
) -> None:
    """A type-`code` view delivers `V[code]` bit for bit, whatever operator follows."""
    values = _values()
    stored = _stored(values=values, sharding=_single())

    delivered = apply_value_transfer(
        value=stored,
        transfer=_selected_transfer(stored=stored, code=code, copy=copy),
    )

    assert np.asarray(delivered).tobytes() == values[code].tobytes()


def test_an_ordinary_read_of_the_same_artifact_delivers_every_type() -> None:
    """Without a view the same artifact is read whole, as every existing route does."""
    values = _values()
    stored = _stored(values=values, sharding=_single())

    delivered = apply_value_transfer(
        value=stored,
        transfer=_transfer(
            stored=stored,
            view=None,
            required=_single(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
        ),
    )

    assert delivered is stored


def test_a_selected_view_keeps_the_original_artifact_address() -> None:
    """A view is a representation of the stored artifact, not a new value function."""
    stored = _stored(values=_values(), sharding=_single())

    transfer = _selected_transfer(stored=stored, code=2)

    view = transfer.view
    assert view is not None
    assert (transfer.target, view.artifact) == (_ARTIFACT, _ARTIFACT)


def test_a_selected_view_keeps_original_codes_not_starting_at_zero() -> None:
    """A block holding code 2 records code 2; it is never renumbered to 0."""
    view = _selected_view(code=2, required=_single())

    assert view.selections[0].codes == (2,)


@pytest.mark.parametrize(
    "axis_names",
    [
        ("pref_type", "assets", "health"),
        ("assets", "pref_type", "health"),
        ("assets", "health", "pref_type"),
    ],
    ids=["leading", "middle", "trailing"],
)
def test_a_selection_follows_the_named_axis_under_any_axis_order(
    *, axis_names: tuple[str, ...]
) -> None:
    """The selected axis is found by its state name, wherever the layout puts it."""
    axis = axis_names.index("pref_type")
    extents = {"pref_type": 3, "assets": 4, "health": 2}
    shape = tuple(extents[name] for name in axis_names)
    values = _values(shape=shape)
    stored = _stored(values=values, sharding=_single())
    view = _selected_view(
        code=1, required=_single(), axis_names=axis_names, shape=shape
    )

    delivered = apply_value_transfer(
        value=stored,
        transfer=_transfer(
            stored=stored,
            view=view,
            required=_single(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
        ),
    )

    assert np.asarray(delivered).tobytes() == np.take(values, 1, axis=axis).tobytes()


def test_a_kept_axis_selection_delivers_the_block_with_its_axis() -> None:
    """`keep_axis=True` selects a width-two interval and keeps the axis."""
    values = _values()
    stored = _stored(values=values, sharding=_single())
    view = _selected_view(code=1, required=_single(), keep_axis=True, width=2)

    delivered = apply_value_transfer(
        value=stored,
        transfer=_transfer(
            stored=stored,
            view=view,
            required=_single(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
        ),
    )

    assert np.asarray(delivered).tobytes() == values[1:3].tobytes()


@pytest.mark.parametrize(
    ("consumer_shape", "match"),
    [
        (_SHAPE, "consumer shape"),
        ((4,), "consumer shape"),
        ((2, 4), "consumer shape"),
    ],
    ids=["stored-shape", "wrong-rank", "permuted"],
)
def test_a_selected_view_refuses_a_consumer_shape_its_selections_do_not_give(
    *, consumer_shape: tuple[int, ...], match: str
) -> None:
    """The consumer shape is checked against the selections, never inferred."""
    with pytest.raises(ValueError, match=match):
        ValueViewDescriptor(
            artifact=_ARTIFACT,
            leaf=ValueViewLeaf.SELECTED,
            stored_axis_names=_STATES,
            stored_shape=_SHAPE,
            dtype=jnp.float32,
            weak_type=False,
            consumer_shape=consumer_shape,
            required_sharding=_single(),
            selections=(
                CoordinateSelection(
                    state_name="pref_type", start=1, width=1, codes=(1,)
                ),
            ),
        )


def test_a_selected_view_requires_a_selection() -> None:
    """A selected leaf with nothing selected is refused, not read as shared."""
    with pytest.raises(ValueError, match="at least one selection"):
        ValueViewDescriptor(
            artifact=_ARTIFACT,
            leaf=ValueViewLeaf.SELECTED,
            stored_axis_names=_STATES,
            stored_shape=_SHAPE,
            dtype=jnp.float32,
            weak_type=False,
            consumer_shape=_SHAPE,
            required_sharding=_single(),
        )


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        (
            CoordinateSelection(state_name="wealth", start=0, width=1, codes=(0,)),
            "names no stored axis",
        ),
        (
            CoordinateSelection(state_name="pref_type", start=2, width=2, codes=(2, 3)),
            "outside",
        ),
        (
            CoordinateSelection(state_name="pref_type", start=0, width=2, codes=(0,)),
            "one code per selected position",
        ),
    ],
    ids=["unknown-state", "out-of-range", "code-count"],
)
def test_a_selection_outside_the_stored_axes_is_refused(
    *, selection: CoordinateSelection, match: str
) -> None:
    """A selection must name a stored axis and stay inside its extent."""
    with pytest.raises(ValueError, match=match):
        ValueViewDescriptor(
            artifact=_ARTIFACT,
            leaf=ValueViewLeaf.SELECTED,
            stored_axis_names=_STATES,
            stored_shape=_SHAPE,
            dtype=jnp.float32,
            weak_type=False,
            consumer_shape=(4, 2),
            required_sharding=_single(),
            selections=(selection,),
        )


def test_a_shared_leaf_carries_no_selection() -> None:
    """An unsliced shared leaf is declared as such and selects nothing."""
    with pytest.raises(ValueError, match="shared leaf"):
        ValueViewDescriptor(
            artifact=_ARTIFACT,
            leaf=ValueViewLeaf.SHARED,
            stored_axis_names=_STATES,
            stored_shape=_SHAPE,
            dtype=jnp.float32,
            weak_type=False,
            consumer_shape=(4, 2),
            required_sharding=_single(),
            selections=(
                CoordinateSelection(
                    state_name="pref_type", start=0, width=1, codes=(0,)
                ),
            ),
        )


def test_a_shared_leaf_delivers_the_stored_buffer_itself() -> None:
    """An aligned shared leaf aliases its owner: no copy is made."""
    stored = _stored(values=_values(), sharding=_single())

    delivered = apply_value_transfer(
        value=stored,
        transfer=_transfer(
            stored=stored,
            view=_shared_view(required=_single()),
            required=_single(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
        ),
    )

    assert delivered is stored


def test_a_full_width_selection_is_not_a_shared_leaf() -> None:
    """Selecting every code keeps the stored shape yet stays a distinct result."""
    stored = _stored(values=_values(), sharding=_single())
    full = _transfer(
        stored=stored,
        view=_selected_view(code=0, required=_single(), keep_axis=True, width=3),
        required=_single(),
        kind=ValueTransferKind.ALIGNED_LOCAL,
    )
    shared = _transfer(
        stored=stored,
        view=_shared_view(required=_single()),
        required=_single(),
        kind=ValueTransferKind.ALIGNED_LOCAL,
    )

    assert transfer_result_key(transfer=full) != transfer_result_key(transfer=shared)


def test_a_view_layout_must_equal_the_transfer_destination() -> None:
    """The view's required layout and the transfer's destination are one layout."""
    stored = _stored(values=_values(), sharding=_single())

    with pytest.raises(ValueError, match="required layout"):
        _transfer(
            stored=stored,
            view=_selected_view(code=0, required=_named()),
            required=_single(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
        )


def test_a_view_must_address_the_transfer_target() -> None:
    """A view of one artifact cannot be attached to a transfer of another."""
    stored = _stored(values=_values(), sharding=_single())
    other = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=4, regime="dead"
    )
    view = _selected_view(code=0, required=_single())
    foreign = ValueViewDescriptor(
        artifact=other,
        leaf=view.leaf,
        stored_axis_names=view.stored_axis_names,
        stored_shape=view.stored_shape,
        dtype=view.dtype,
        weak_type=view.weak_type,
        consumer_shape=view.consumer_shape,
        required_sharding=view.required_sharding,
        selections=view.selections,
    )

    with pytest.raises(ValueError, match="addresses"):
        _transfer(
            stored=stored,
            view=foreign,
            required=_single(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
        )


def test_a_selected_transfer_is_classified_from_the_selected_layout() -> None:
    """The operator after selection is named from the selected block's layout."""
    stored = _stored(values=_values(), sharding=_single())

    with pytest.raises(ValueError, match="copy_to_source_layout"):
        _transfer(
            stored=stored,
            view=_selected_view(code=0, required=_named()),
            required=_named(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
        )


def test_equal_shaped_views_of_different_types_have_distinct_result_keys() -> None:
    """Type 0 and type 1 views never share a transfer-result identity."""
    stored = _stored(values=_values(), sharding=_single())

    keys = {
        transfer_result_key(transfer=_selected_transfer(stored=stored, code=code))
        for code in (0, 1)
    }

    assert len(keys) == 2


def test_equal_shaped_views_of_different_types_share_one_lowering_identity() -> None:
    """The type code is a runtime operand: it is absent from the lowering key."""
    stored = _stored(values=_values(), sharding=_single())

    keys = {
        _selected_transfer(stored=stored, code=code).specialization_key
        for code in (0, 1)
    }

    assert len(keys) == 1


def test_a_view_changes_the_lowering_identity_of_an_ordinary_read() -> None:
    """A selected read and a whole read of one artifact never share a lowering."""
    stored = _stored(values=_values(), sharding=_single())
    whole = _transfer(
        stored=stored,
        view=None,
        required=_single(),
        kind=ValueTransferKind.ALIGNED_LOCAL,
    )

    assert (
        _selected_transfer(stored=stored, code=0).specialization_key
        != whole.specialization_key
    )


def test_an_ordinary_read_keeps_its_result_key() -> None:
    """A read without a view keeps the per-period cache key every route uses."""
    stored = _stored(values=_values(), sharding=_single())
    whole = _transfer(
        stored=stored,
        view=None,
        required=_single(),
        kind=ValueTransferKind.ALIGNED_LOCAL,
    )

    assert transfer_result_key(transfer=whole) == (_ARTIFACT, _single())


def test_a_type_code_change_reuses_the_selection_executable() -> None:
    """Selecting another type of the same shape compiles nothing new."""
    stored = _stored(values=_values(), sharding=_single())
    apply_value_transfer(
        value=stored, transfer=_selected_transfer(stored=stored, code=0)
    )
    before = _compiled_selections()

    apply_value_transfer(
        value=stored, transfer=_selected_transfer(stored=stored, code=1)
    )

    assert _compiled_selections() == before


def test_a_new_solution_generation_has_a_new_result_key() -> None:
    """The generation token is part of every selected transfer-result key."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _selected_transfer(stored=stored, code=1)

    assert transfer_result_key(transfer=transfer, generation=1) != (
        transfer_result_key(transfer=transfer, generation=2)
    )


def _cache(
    *,
    transfers: tuple[ResolvedValueTransfer, ...],
    counts: tuple[int, ...],
    generation: Hashable = None,
) -> PeriodTransferCache:
    return PeriodTransferCache(
        registry=BufferRegistry(),
        consumer_counts=MappingProxyType(
            {
                transfer_result_key(transfer=transfer, generation=generation): count
                for transfer, count in zip(transfers, counts, strict=True)
            }
        ),
        generation=generation,
    )


def _arguments(*, stored: jax.Array) -> MappingProxyType[str, ArgumentTree]:
    return MappingProxyType(
        {"next_regime_to_V_arr": MappingProxyType({"retired": stored})}
    )


def _delivered(*, arguments: Mapping[str, ArgumentTree]) -> np.ndarray:
    branch = arguments["next_regime_to_V_arr"]
    assert isinstance(branch, Mapping)
    return np.asarray(branch["retired"])


def test_a_shared_cache_serves_each_type_its_own_block() -> None:
    """Two equal-shaped type views through one cache never alias numerically."""
    values = _values()
    stored = _stored(values=values, sharding=_single())
    first, second = (
        _selected_transfer(
            stored=stored, code=code, copy=True, core_key=f"type_{code}", reused=True
        )
        for code in (0, 1)
    )
    cache = _cache(transfers=(first, second), counts=(1, 1))

    delivered = tuple(
        _delivered(
            arguments=apply_value_transfer_plan(
                arguments=_arguments(stored=stored), plan=(transfer,), cache=cache
            )
        ).tobytes()
        for transfer in (first, second)
    )

    assert delivered == (values[0].tobytes(), values[1].tobytes())


def test_a_parameter_change_is_never_served_a_stale_block() -> None:
    """A new generation's cache delivers the new values, with no recompile."""
    old_values = _values()
    new_values = _values(offset=100.0)
    old = _stored(values=old_values, sharding=_single())
    new = _stored(values=new_values, sharding=_single())
    transfer = _selected_transfer(stored=old, code=1, copy=True, reused=True)
    apply_value_transfer_plan(
        arguments=_arguments(stored=old),
        plan=(transfer,),
        cache=_cache(transfers=(transfer,), counts=(1,), generation="old"),
    )
    compiled_before = _compiled_selections()

    delivered = _delivered(
        arguments=apply_value_transfer_plan(
            arguments=_arguments(stored=new),
            plan=(transfer,),
            cache=_cache(transfers=(transfer,), counts=(1,), generation="new"),
        )
    )

    assert (delivered.tobytes(), _compiled_selections()) == (
        new_values[1].tobytes(),
        compiled_before,
    )


def test_a_selected_copy_is_released_only_after_its_last_consumer() -> None:
    """One consumer committing keeps a two-consumer copy; the second releases it."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _selected_transfer(stored=stored, code=1, copy=True, reused=True)
    key = transfer_result_key(transfer=transfer)
    cache = _cache(transfers=(transfer,), counts=(2,))
    copied = _single_copy(cache=cache, transfer=transfer, stored=stored)

    cache.commit_consumer(key=key)
    alive_after_first = not copied.is_deleted()
    cache.commit_consumer(key=key)

    assert (alive_after_first, copied.is_deleted()) == (True, True)


def test_type_copies_are_released_on_their_own_schedules() -> None:
    """The type-1 copy is released when its block finishes, not with type 0's."""
    stored = _stored(values=_values(), sharding=_single())
    first, second = (
        _selected_transfer(
            stored=stored, code=code, copy=True, core_key=f"type_{code}", reused=True
        )
        for code in (0, 1)
    )
    cache = _cache(transfers=(first, second), counts=(2, 1))
    first_copy = _single_copy(cache=cache, transfer=first, stored=stored)
    second_copy = _single_copy(cache=cache, transfer=second, stored=stored)

    cache.commit_consumer(key=transfer_result_key(transfer=second))

    assert (first_copy.is_deleted(), second_copy.is_deleted()) == (False, True)


def _single_copy(
    *,
    cache: PeriodTransferCache,
    transfer: ResolvedValueTransfer,
    stored: jax.Array,
) -> jax.Array:
    arguments = apply_value_transfer_plan(
        arguments=_arguments(stored=stored), plan=(transfer,), cache=cache
    )
    branch = arguments["next_regime_to_V_arr"]
    assert isinstance(branch, Mapping)
    copied = branch["retired"]
    assert isinstance(copied, jax.Array)
    assert copied is not stored
    return copied


def test_the_selection_and_the_copy_are_both_handed_to_the_completion_owner() -> None:
    """Every fresh buffer of a selected copy is observed, intermediate included."""
    stored = _stored(values=_values(), sharding=_single())
    observed: list[tuple[int, ...]] = []

    apply_value_transfer(
        value=stored,
        transfer=_selected_transfer(stored=stored, code=1, copy=True),
        on_materialized=lambda *, transfer, array: observed.append(array.shape),  # noqa: ARG005
    )

    assert observed == [(4, 2), (4, 2)]


def test_a_selected_copy_is_inspectable_as_select_then_communicate() -> None:
    """Selection runs on the stored layout and only the block is communicated."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _selected_transfer(stored=stored, code=1, copy=True)

    stages = tuple(
        (stage.kind, stage.input_shape, stage.output_shape, stage.operator)
        for stage in transfer.stages
    )

    assert stages == (
        (TransferStageKind.SELECT, _SHAPE, (4, 2), None),
        (
            TransferStageKind.COMMUNICATE,
            (4, 2),
            (4, 2),
            ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        ),
    )


def test_an_ordinary_read_is_one_communication_stage() -> None:
    """Without a view a transfer is its one catalogue operator, as before."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _transfer(
        stored=stored,
        view=None,
        required=_named(),
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
    )

    assert tuple(stage.kind for stage in transfer.stages) == (
        TransferStageKind.COMMUNICATE,
    )


_ITEM = 4
_FULL = math.prod(_SHAPE) * _ITEM
_BLOCK = 4 * 2 * _ITEM


@pytest.mark.parametrize(
    ("view", "copy", "expected"),
    [
        ("shared", False, (_FULL, 0, 0)),
        ("selected", False, (_FULL, _BLOCK, 0)),
        ("selected", True, (_FULL, _BLOCK, _BLOCK)),
    ],
    ids=["shared-alias", "selected-aligned", "selected-copy"],
)
def test_owner_selection_and_destination_bytes_are_accounted_separately(
    *, view: str, copy: bool, expected: tuple[int, int, int]
) -> None:
    """The stored owner stays charged; each fresh buffer is charged once."""
    stored = _stored(values=_values(), sharding=_single())
    required = _named() if copy else _single()
    transfer = _transfer(
        stored=stored,
        view=(
            _shared_view(required=required)
            if view == "shared"
            else _selected_view(code=1, required=required)
        ),
        required=required,
        kind=(
            ValueTransferKind.COPY_TO_SOURCE_LAYOUT
            if copy
            else ValueTransferKind.ALIGNED_LOCAL
        ),
    )
    device = _device().id

    footprint = plan_value_transfer_footprint(transfer=transfer)

    assert (
        footprint.owner_bytes[device],
        footprint.selected_bytes.get(device, 0),
        footprint.destination_bytes.get(device, 0),
    ) == expected


def test_a_selected_view_reduces_the_consumer_footprint_by_the_type_count() -> None:
    """Selecting one of three types delivers a third of the stored bytes."""
    stored = _stored(values=_values(), sharding=_single())

    cost = _selected_transfer(stored=stored, code=1, copy=True).cost

    assert (cost.logical_bytes, cost.per_device_bytes) == (_FULL // 3, _FULL // 3)


@pytest.mark.parametrize(
    ("view", "expected"),
    [("shared", True), ("selected", False)],
)
def test_only_an_aligned_shared_leaf_delivers_the_stored_buffer(
    *, view: str, expected: bool
) -> None:
    """A selected block is always a fresh buffer, even on an aligned layout."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _transfer(
        stored=stored,
        view=(
            _shared_view(required=_single())
            if view == "shared"
            else _selected_view(code=1, required=_single())
        ),
        required=_single(),
        kind=ValueTransferKind.ALIGNED_LOCAL,
    )

    assert transfer.delivers_stored_buffer is expected


def test_a_transfer_beyond_the_budget_is_refused_before_anything_runs() -> None:
    """Admission names the device and its need, and nothing is dispatched."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _selected_transfer(stored=stored, code=1, copy=True)
    before = _compiled_selections()

    with pytest.raises(ExecutionPlanningError, match=r"needs \d+ bytes"):
        fail_if_value_transfer_exceeds_budget(
            transfer=transfer,
            budget_bytes=_FULL + _BLOCK,
            other_resident_bytes=MappingProxyType({}),
        )
    assert _compiled_selections() == before


def test_a_transfer_exactly_at_the_budget_is_admitted() -> None:
    """The owner plus the selected block plus its copy fit an exact budget."""
    stored = _stored(values=_values(), sharding=_single())

    fail_if_value_transfer_exceeds_budget(
        transfer=_selected_transfer(stored=stored, code=1, copy=True),
        budget_bytes=_FULL + 2 * _BLOCK,
        other_resident_bytes=MappingProxyType({}),
    )


def _program_reading(
    *,
    stored: jax.Array,
    read: ValueRead,
    argument_branch: PytreeValue | ShapeDtypePytree,
) -> MaterializedCoreProgram:
    del stored
    return MaterializedCoreProgram(
        name="main",
        function=_consume,
        arguments=MappingProxyType({"next_regime_to_V_arr": argument_branch}),
        requirements=CoreExecutionRequirements(value_reads=(read,)),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _consume(
    *, next_regime_to_V_arr: MappingProxyType[RegimeName, jax.Array]
) -> jax.Array:
    return next_regime_to_V_arr["retired"] + 1.0


def test_a_resolved_program_receives_the_selected_block() -> None:
    """A planned program declaring a view read is handed the selected block."""
    values = _values()
    stored = _stored(values=values, sharding=_single())
    transfer = _selected_transfer(stored=stored, code=2)
    read = ValueRead(target=_ARTIFACT, source=_source(), view=transfer.view)

    resolved = resolve_core_program(
        program=_program_reading(
            stored=stored, read=read, argument_branch={"retired": stored}
        ),
        input_transfer_plan=(transfer,),
    )

    branch = resolved.arguments["next_regime_to_V_arr"]
    assert isinstance(branch, Mapping)
    assert np.asarray(branch["retired"]).tobytes() == values[2].tobytes()


def test_a_read_and_its_transfer_must_declare_the_same_view() -> None:
    """A consumer declaring one type cannot be planned with another type's view."""
    stored = _stored(values=_values(), sharding=_single())
    read = ValueRead(
        target=_ARTIFACT,
        source=_source(),
        view=_selected_view(code=0, required=_single()),
    )

    with pytest.raises(ValueError, match="view"):
        resolve_core_program(
            program=_program_reading(
                stored=stored, read=read, argument_branch={"retired": stored}
            ),
            input_transfer_plan=(_selected_transfer(stored=stored, code=1),),
        )


def test_a_plain_read_cannot_be_planned_with_a_view() -> None:
    """A read declaring no view is never silently narrowed to a block."""
    stored = _stored(values=_values(), sharding=_single())
    read = ValueRead(target=_ARTIFACT, source=_source())

    with pytest.raises(ValueError, match="view"):
        resolve_core_program(
            program=_program_reading(
                stored=stored, read=read, argument_branch={"retired": stored}
            ),
            input_transfer_plan=(_selected_transfer(stored=stored, code=1),),
        )


def test_a_read_view_must_address_its_target() -> None:
    """A declared read's view names the artifact the read targets."""
    other = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=4, regime="dead"
    )

    with pytest.raises(ValueError, match="addresses"):
        ValueRead(
            target=other,
            source=ValueConsumerAddress(
                source_period=3,
                source_regime="working",
                core_key="main",
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                path=("dead",),
            ),
            view=_selected_view(code=0, required=_single()),
        )


def test_an_abstract_view_input_has_the_consumer_shape() -> None:
    """Abstract lowering describes the selected block, not the stored array."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _selected_transfer(stored=stored, code=1)
    read = ValueRead(target=_ARTIFACT, source=_source(), view=transfer.view)

    described = abstract_program_inputs(
        program=_program_reading(
            stored=stored, read=read, argument_branch={"retired": stored}
        ),
        transfers=(transfer,),
        execution_sharding=_single(),
    )

    branch = described.arguments["next_regime_to_V_arr"]
    assert isinstance(branch, Mapping)
    leaf = branch["retired"]
    assert isinstance(leaf, jax.ShapeDtypeStruct)
    assert (leaf.shape, leaf.sharding) == ((4, 2), _single())


def test_an_abstract_view_input_with_the_stored_shape_is_refused() -> None:
    """An abstract consumer leaf must already have the view's consumer shape."""
    stored = _stored(values=_values(), sharding=_single())
    transfer = _selected_transfer(stored=stored, code=1)
    read = ValueRead(target=_ARTIFACT, source=_source(), view=transfer.view)
    stale = jax.ShapeDtypeStruct(_SHAPE, jnp.float32, sharding=_single())

    with pytest.raises(ValueError, match="shape mismatch"):
        resolve_core_program(
            program=_program_reading(
                stored=stored, read=read, argument_branch={"retired": stale}
            ),
            input_transfer_plan=(transfer,),
            abstract_inputs=True,
        )


def _resolved_program(
    *, transfer: ResolvedValueTransfer, read: ValueRead, argument: str
) -> ResolvedCoreProgram:
    return ResolvedCoreProgram(
        name=read.source.core_key,
        function=_consume,
        arguments=MappingProxyType({argument: jnp.zeros(3)}),
        static_kwargs=MappingProxyType({}),
        requirements=CoreExecutionRequirements(value_reads=(read,)),
        output_roles=None,
        disposition=CoreExecutionDisposition.DENSE,
        donation_candidates=(argument,),
        tile_widths=MappingProxyType({}),
        specialization_key=("test",),
        input_transfer_plan=(transfer,),
        disposition_reason="test",
    )


def _direct_source(*, core_key: str, argument: str) -> ValueConsumerAddress:
    return ValueConsumerAddress(
        source_period=3,
        source_regime="working",
        core_key=core_key,
        channel=ValueInputChannel.NEXT_REGIME_VALUE,
        argument=argument,
        path=(),
    )


def _direct_transfer(
    *, stored: jax.Array, source: ValueConsumerAddress, view: ValueViewDescriptor | None
) -> ResolvedValueTransfer:
    return ResolvedValueTransfer(
        target=_ARTIFACT,
        source=source,
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_sharding=stored.sharding,
        source_sharding=_single(),
        expected_shape=stored.shape,
        expected_dtype=stored.dtype,
        view=view,
    )


def _sole_reader_ledger() -> PlannedInputLiveness:
    return PlannedInputLiveness(dispatch_accesses={(3, "working"): (_ARTIFACT,)})


def test_a_selected_view_never_donates_its_owner() -> None:
    """The executable receives a block, so the stored owner is never handed over."""
    stored = _stored(values=_values(), sharding=_single())
    source = _direct_source(core_key="main", argument="V")
    view = _selected_view(code=1, required=_single())
    program = _resolved_program(
        transfer=_direct_transfer(stored=stored, source=source, view=view),
        read=ValueRead(target=_ARTIFACT, source=source, view=view),
        argument="V",
    )

    (donation,) = resolve_donations(
        program=program,
        dispatch=(3, "working"),
        ledger=_sole_reader_ledger(),
        n_periods=6,
    )

    assert donation.buffer is DonatedBuffer.TRANSFERRED_COPY


def test_an_overlapping_live_view_blocks_donation_of_its_owner() -> None:
    """A whole read is not donated while a view of the same owner is read too."""
    stored = _stored(values=_values(), sharding=_single())
    whole_source = _direct_source(core_key="whole", argument="V")
    view_source = _direct_source(core_key="type_1", argument="V")
    view = _selected_view(code=1, required=_single())
    whole = _resolved_program(
        transfer=_direct_transfer(stored=stored, source=whole_source, view=None),
        read=ValueRead(target=_ARTIFACT, source=whole_source),
        argument="V",
    )
    selected = _resolved_program(
        transfer=_direct_transfer(stored=stored, source=view_source, view=view),
        read=ValueRead(target=_ARTIFACT, source=view_source, view=view),
        argument="V",
    )
    nominated = resolve_donations(
        program=whole,
        dispatch=(3, "working"),
        ledger=_sole_reader_ledger(),
        n_periods=6,
    )

    (donation,) = withhold_shared_donations(
        program=whole,
        donations=nominated,
        unit_readers=unit_input_readers(programs=(whole, selected)),
    )

    assert donation.withheld_by == view_source


def test_an_owner_read_through_two_views_outlives_the_first_reader() -> None:
    """Views keep the artifact address, so each reading dispatch counts once."""
    view_reads = tuple(
        ValueRead(
            target=_ARTIFACT,
            source=_source(core_key=f"type_{code}"),
            view=_selected_view(code=code, required=_single()),
        )
        for code in (0, 1)
    )
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            (3, read.source.core_key): (read.target,) for read in view_reads
        }
    )

    ledger.commit_successful_dispatch(dispatch=(3, "type_0"))

    assert ledger.is_release_eligible(artifact=_ARTIFACT) is False
