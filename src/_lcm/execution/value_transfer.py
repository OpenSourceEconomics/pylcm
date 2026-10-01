"""Planner-owned transfers of stored values into solve-core inputs.

An economic dependency points from a source regime to a target regime, while the
stored value moves in the opposite direction during backward induction.  This module
names both ends independently: a target artifact says which stored array is read, and
a source consumer says exactly where that array enters a core.  The transfer
catalogue is a total function from a stored layout and a required layout to one
operator, and fails closed on the single pair no single collective can serve.
"""

import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from _lcm.execution.footprint import (
    ArtifactFootprint,
    layout_footprint,
    sharding_device_ids,
)
from _lcm.execution.runtime_sharding import runtime_shardings_match
from _lcm.typing import RegimeName, StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import ArtifactKey
from lcm.typing import ValueND

_VALUE_TRANSFER_VERSION = 2


class ValueArtifactKind(StrEnum):
    """Kind of stored solve-time value consumed by a source core."""

    REGIME_VALUE = "regime_value"
    GATED_CONTINUATION = "gated_continuation"
    CONTINUATION_LEAF = "continuation_leaf"
    REPLAY_ARTIFACT_LEAF = "replay_artifact_leaf"


class ValueInputChannel(StrEnum):
    """Core argument channel through which a stored value reaches a program."""

    NEXT_REGIME_VALUE = "next_regime_to_V_arr"
    SAME_PERIOD_VALUE = "same_period_regime_to_V_arr"
    EDGE_REFERENCE_VALUE = "edge_reference_regime_to_V_arr"
    CONTINUATION_LEAF = "next_regime_to_continuation"
    CURRENT_REPLAY_ARTIFACT = "current_replay_artifact"
    NEXT_REPLAY_ARTIFACT = "next_replay_artifact"


class ValueTransferKind(StrEnum):
    """Supported target-to-source representation changes."""

    ALIGNED_LOCAL = "aligned_local"
    COPY_TO_SOURCE_LAYOUT = "copy_to_source_layout"
    ALL_GATHER = "all_gather"
    LOCAL_SLICE = "local_slice"
    RESHARD = "reshard"
    CROSS_MESH_COPY = "cross_mesh_copy"


class TransferOperationClass(StrEnum):
    """What a transfer operator does to reach its required layout."""

    LOCAL = "local"
    DEVICE_COPY = "device_copy"
    COLLECTIVE = "collective"


_OPERATION_CLASS_BY_KIND = MappingProxyType(
    {
        ValueTransferKind.ALIGNED_LOCAL: TransferOperationClass.LOCAL,
        ValueTransferKind.COPY_TO_SOURCE_LAYOUT: TransferOperationClass.DEVICE_COPY,
        ValueTransferKind.CROSS_MESH_COPY: TransferOperationClass.DEVICE_COPY,
        ValueTransferKind.ALL_GATHER: TransferOperationClass.COLLECTIVE,
        ValueTransferKind.LOCAL_SLICE: TransferOperationClass.COLLECTIVE,
        ValueTransferKind.RESHARD: TransferOperationClass.COLLECTIVE,
    }
)


@dataclass(frozen=True, kw_only=True)
class TransferCost:
    """What one planned transfer occupies while it runs."""

    operation_class: TransferOperationClass
    """Whether the operator is local, a device copy, or a collective."""

    logical_bytes: int
    """Size of the whole value, independent of how it is laid out."""

    per_device_bytes: int
    """Bytes the required layout holds on each participating device.

    Planned shardings divide evenly, so every participant holds the same shard.
    """

    temporary_bytes: int
    """Bytes the operator itself holds beyond the result, per device."""

    devices: tuple[int, ...]
    """Ids of every device the operator touches, ascending."""

    reused_by_several_consumers: bool
    """Whether more than one source core of the period reads this result."""


class ValueViewLeaf(StrEnum):
    """How a consumer reads a stored artifact under a value view.

    - `SHARED`: the unsliced stored value, declared shared by every reader.
    - `SELECTED`: a block of canonical coordinates selected out of it.
    """

    SHARED = "shared"
    SELECTED = "selected"


class TransferStageKind(StrEnum):
    """One inspectable stage of a planned transfer.

    - `SELECT`: select coordinates on the stored layout; nothing is communicated.
    - `COMMUNICATE`: the one catalogue operator onto the required layout.
    """

    SELECT = "select"
    COMMUNICATE = "communicate"


@dataclass(frozen=True, kw_only=True)
class CoordinateSelection:
    """One contiguous interval of one named stored axis.

    `start` and `width` locate the interval among the axis positions; `codes`
    are the canonical codes those positions hold, in order. A block holding
    code 2 records code 2: selection never renumbers a coordinate. Whether the
    interval fits the axis is checked by the descriptor that owns the axis.
    """

    state_name: StateName
    """Name of the stored axis the interval selects on."""
    start: int
    """First selected position along that axis."""
    width: int
    """Number of consecutive selected positions."""
    codes: tuple[int, ...]
    """Canonical code held at each selected position."""
    keep_axis: bool = False
    """Whether the consumer keeps the axis; otherwise a width-1 axis is removed."""

    def __post_init__(self) -> None:
        """Snapshot the code sequence."""
        object.__setattr__(self, "codes", tuple(self.codes))


@dataclass(frozen=True, kw_only=True)
class ValueViewDescriptor:
    """The representation in which one consumer reads one stored artifact.

    A view keeps the artifact's original logical address: it is a
    representation of that value, not a different value function. A shared
    leaf reads the stored value unsliced; a selected leaf reads only its
    `selections`, located by axis name. Both are declared explicitly, so a
    selection is never inferred from array lengths that happen to match, and
    the consumer shape is checked against the selections rather than derived.
    """

    artifact: ValueArtifactAddress
    """Original logical address of the stored artifact."""
    leaf: ValueViewLeaf
    """Whether the consumer reads the stored value whole or a selected block."""
    stored_axis_names: tuple[StateName, ...]
    """Name of every stored axis, in stored order."""
    stored_shape: tuple[int, ...]
    """Shape of the stored artifact."""
    dtype: object
    """Element type of both the stored artifact and the consumer leaf."""
    weak_type: bool
    """Weak typing of the consumer leaf."""
    consumer_shape: tuple[int, ...]
    """Shape the consumer receives."""
    required_sharding: jax.sharding.Sharding
    """Layout the consumer requires."""
    selections: tuple[CoordinateSelection, ...] = ()
    """Selected intervals, one per selected axis; empty for a shared leaf."""

    def __post_init__(self) -> None:
        """Validate the declaration against the stored axes and the selections."""
        if not isinstance(self.artifact, ValueArtifactAddress):
            msg = "A value view must address a ValueArtifactAddress."
            raise TypeError(msg)
        _require_enum(value=self.leaf, enum_type=ValueViewLeaf, label="view leaf")
        names = tuple(self.stored_axis_names)
        stored_shape = _normalize_shape(shape=tuple(self.stored_shape))
        consumer_shape = _normalize_shape(shape=tuple(self.consumer_shape))
        selections = tuple(self.selections)
        object.__setattr__(self, "stored_axis_names", names)
        object.__setattr__(self, "stored_shape", stored_shape)
        object.__setattr__(self, "consumer_shape", consumer_shape)
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))
        for name in names:
            _require_name(name=name, label="stored axis name")
        if len(set(names)) != len(names) or len(names) != len(stored_shape):
            msg = (
                f"A value view needs one distinct name per stored axis; got {names!r} "
                f"for shape {stored_shape!r}."
            )
            raise ValueError(msg)
        if type(self.weak_type) is not bool:
            msg = f"A value view's weak_type must be a bool, got {self.weak_type!r}."
            raise TypeError(msg)
        _require_sharding(sharding=self.required_sharding, label="view required")
        if any(not isinstance(item, CoordinateSelection) for item in selections):
            msg = "A value view's selections must be CoordinateSelection entries."
            raise TypeError(msg)
        if self.leaf is ValueViewLeaf.SHARED:
            if selections or consumer_shape != stored_shape:
                msg = (
                    "A shared leaf reads the stored value unsliced: it carries no "
                    f"selection and keeps shape {stored_shape!r}."
                )
                raise ValueError(msg)
        else:
            _fail_if_selections_invalid(view=self)
        _check_sharding_shape(
            sharding=self.required_sharding, shape=consumer_shape, label="view"
        )

    @property
    def selected_axes(self) -> tuple[int, ...]:
        """Return the stored position of each selected axis, in selection order."""
        return tuple(
            self.stored_axis_names.index(item.state_name) for item in self.selections
        )

    @property
    def structure_key(self) -> Hashable:
        """Return what changes compiled code: shapes, axes, widths, never codes."""
        return (
            "value-view",
            self.leaf,
            self.stored_shape,
            self.dtype,
            self.weak_type,
            self.consumer_shape,
            tuple(
                (axis, item.width, item.keep_axis)
                for axis, item in zip(self.selected_axes, self.selections, strict=True)
            ),
        )

    @property
    def identity_key(self) -> Hashable:
        """Return what changes the delivered numbers: the structure and the codes."""
        return (
            self.structure_key,
            self.stored_axis_names,
            tuple(
                (item.state_name, item.start, item.codes) for item in self.selections
            ),
        )


@dataclass(frozen=True, kw_only=True)
class TransferStage:
    """One stage of a planned transfer, with its real shapes and layouts.

    A transfer's stages run in tuple order; each consumes the previous stage's
    output, and the first consumes the stored artifact.
    """

    kind: TransferStageKind
    """Whether this stage selects or communicates."""
    input_shape: tuple[int, ...]
    """Shape the stage reads."""
    output_shape: tuple[int, ...]
    """Shape the stage produces."""
    input_sharding: jax.sharding.Sharding
    """Layout the stage reads."""
    output_sharding: jax.sharding.Sharding
    """Layout the stage produces."""
    operator: ValueTransferKind | None
    """Catalogue operator of a communication stage; `None` for a selection."""
    item_bytes: int
    """Bytes per element."""

    @property
    def allocates(self) -> bool:
        """Whether the stage's output is a fresh buffer rather than its input."""
        return self.operator is not ValueTransferKind.ALIGNED_LOCAL

    @property
    def output_footprint(self) -> ArtifactFootprint:
        """Return the per-device bytes and devices of the stage's output."""
        return layout_footprint(
            sharding=self.output_sharding,
            shape=self.output_shape,
            item_bytes=self.item_bytes,
        )


def transfer_result_key(
    *, transfer: ResolvedValueTransfer, generation: Hashable = None
) -> tuple[Hashable, Hashable]:
    """Return the `(delivered value, required layout)` identity of a transfer.

    A read without a view keeps the key every per-period cache uses, the
    artifact and the required layout, with the artifact paired with
    `generation` only when one is given. A view's value identity always
    carries the generation, the original artifact address and the view's
    identity — its selected coordinates with their codes, shapes, element type
    and weak typing — so equal-shaped blocks of two types never share a result,
    nor do the blocks of two solution generations.
    """
    if transfer.view is None:
        artifact: Hashable = (
            transfer.target if generation is None else (generation, transfer.target)
        )
    else:
        artifact = (
            "value-view",
            generation,
            transfer.target,
            transfer.view.identity_key,
        )
    return (artifact, transfer.source_sharding)


def _select_view_blocks(
    *,
    value: jax.Array,
    starts: jax.Array,
    axes: tuple[int, ...],
    widths: tuple[int, ...],
    kept: tuple[bool, ...],
    out_sharding: jax.sharding.Sharding,
) -> jax.Array:
    """Slice each selected interval on the stored layout, then drop removed axes.

    `starts` is a runtime operand, so selecting another code of the same shape
    reuses the executable.
    """
    selected = value
    for index, (axis, width) in enumerate(zip(axes, widths, strict=True)):
        selected = jax.lax.dynamic_slice_in_dim(
            selected, starts[index], width, axis=axis
        )
    removed = tuple(axis for axis, keep in zip(axes, kept, strict=True) if not keep)
    if removed:
        selected = jnp.squeeze(selected, axis=removed)
    return jax.lax.with_sharding_constraint(selected, out_sharding)


_select_value_view = jax.jit(
    _select_view_blocks, static_argnames=("axes", "widths", "kept", "out_sharding")
)


def _selection_operands(*, transfer: ResolvedValueTransfer) -> dict[str, object]:
    """Return every argument of the selection executable except the stored value."""
    view = transfer.view
    if view is None or view.leaf is not ValueViewLeaf.SELECTED:
        msg = "Only a selected value view has a selection stage."
        raise ValueError(msg)
    return {
        "starts": jnp.asarray(
            tuple(item.start for item in view.selections), dtype=jnp.int32
        ),
        "axes": view.selected_axes,
        "widths": tuple(item.width for item in view.selections),
        "kept": tuple(item.keep_axis for item in view.selections),
        "out_sharding": _selection_sharding(view=view, layout=transfer.stored_sharding),
    }


@runtime_checkable
class TransferCache(Protocol):
    """A per-period store of transferred copies several consumers share."""

    def get(self, *, transfer: ResolvedValueTransfer) -> jax.Array | None:
        """Return the copy made for this transfer's artifact and layout, if any."""
        ...

    def put(
        self, *, transfer: ResolvedValueTransfer, array: jax.Array, stored: jax.Array
    ) -> None:
        """Record the copy made for this transfer's artifact and layout.

        `stored` is the pre-transfer value, so an implementation can tell a
        genuinely new buffer from one a `device_put` returned unchanged.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class ValueArtifactAddress:
    """Logical address of one stored target value, gated continuation, or leaf.

    ``period`` is the value's solved period for :attr:`REGIME_VALUE` and the
    target/fold period for :attr:`GATED_CONTINUATION`.  A gated continuation is
    owned by the economic source regime and edge target together, which prevents
    two distinct ``Wbar`` objects with the same shape from sharing an identity.
    A :attr:`CONTINUATION_LEAF` is one pytree leaf of the keyed continuation the
    target regime stored for its period, addressed as ``(period, regime,
    artifact_key, leaf_path)``.
    """

    kind: ValueArtifactKind
    """Which stored solve-time value this address names."""
    period: int
    """Solved period of a regime value, fold period of a gated continuation."""
    regime: RegimeName
    """Regime owning the stored value."""
    target_regime: RegimeName | None = None
    """Edge target of a gated continuation; `None` for every other kind."""
    artifact_key: ArtifactKey | None = None
    """Versioned key of the continuation whose leaf is addressed."""
    leaf_path: tuple[str, ...] = ()
    """Pytree path of the addressed leaf inside that continuation."""

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported artifact addresses."""
        _require_enum(
            value=self.kind, enum_type=ValueArtifactKind, label="artifact kind"
        )
        _require_period(period=self.period, label="artifact period")
        _require_name(name=self.regime, label="artifact regime")
        object.__setattr__(self, "leaf_path", tuple(self.leaf_path))
        if self.kind in {
            ValueArtifactKind.CONTINUATION_LEAF,
            ValueArtifactKind.REPLAY_ARTIFACT_LEAF,
        }:
            if self.target_regime is not None:
                msg = (
                    "A keyed artifact leaf cannot name an edge target regime, "
                    f"got {self.target_regime!r}."
                )
                raise ValueError(msg)
            if not isinstance(self.artifact_key, ArtifactKey):
                msg = (
                    "A keyed artifact leaf must name its ArtifactKey, got "
                    f"{self.artifact_key!r}."
                )
                raise TypeError(msg)
            if (
                self.kind is ValueArtifactKind.CONTINUATION_LEAF and not self.leaf_path
            ) or any(not isinstance(step, str) or not step for step in self.leaf_path):
                msg = (
                    "An artifact leaf_path requires supported leaf addressing with "
                    f"non-empty strings, got {self.leaf_path!r}."
                )
                raise ValueError(msg)
            return
        if self.artifact_key is not None or self.leaf_path:
            msg = (
                f"A {self.kind.value} artifact carries no artifact_key and no "
                f"leaf_path, got {self.artifact_key!r} and {self.leaf_path!r}."
            )
            raise ValueError(msg)
        if self.kind is ValueArtifactKind.REGIME_VALUE:
            if self.target_regime is not None:
                msg = "A regime-value artifact cannot name an edge target regime."
                raise ValueError(msg)
            return
        if self.kind is ValueArtifactKind.GATED_CONTINUATION:
            _require_name(name=self.target_regime, label="gated-edge target regime")
            return
        msg = f"Unsupported value artifact kind: {self.kind!r}."
        raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class ValueConsumerAddress:
    """Logical address of one value leaf consumed by a source core.

    `path` is relative to `argument` when the read names one, and to `channel`
    otherwise; for a channel-indexed read its first segment is the target or
    reference regime key.  Keeping the path separate from the artifact identity
    allows one stored value to feed several argument leaves without conflating
    their liveness events.
    """

    source_period: int
    source_regime: RegimeName
    core_key: str
    channel: ValueInputChannel
    path: tuple[str | int, ...]
    argument: str | None = None
    """Program argument holding the leaf, when it is not under `channel`."""

    def __post_init__(self) -> None:
        """Validate the complete core-input locator."""
        _require_period(period=self.source_period, label="source period")
        _require_name(name=self.source_regime, label="source regime")
        _require_name(name=self.core_key, label="core key")
        _require_enum(
            value=self.channel, enum_type=ValueInputChannel, label="input channel"
        )
        if self.argument is not None:
            _require_name(name=self.argument, label="consumer argument")
            if not isinstance(self.path, tuple):
                msg = "A value consumer path must be a tuple."
                raise TypeError(msg)
        elif not isinstance(self.path, tuple) or not self.path:
            msg = "A value consumer path must be a non-empty tuple."
            raise TypeError(msg)
        for segment in self.path:
            _validate_path_segment(segment=segment)


@dataclass(frozen=True, kw_only=True)
class ResolvedValueTransfer:
    """One validated target artifact transfer into one source-core leaf.

    The full object is hashable and retains exact logical coordinates for
    inspection and liveness.  ``specialization_key`` deliberately omits absolute
    periods and source-regime/core coordinates: those do not change compiled code.
    It retains the argument-tree role — the target mapping key in ``source.path``
    for a channel-indexed read, the argument name in ``source.argument`` for a
    direct one — plus the operator, concrete layouts, and leaf metadata, so
    behaviorally different transfers cannot share a lowering.
    ``reused_by_several_consumers`` stays outside that key: sharing one result
    between consumers is a scheduling fact and changes no generated code.

    ``expected_shape`` is the stored artifact's shape. Without a view the
    consumer receives that shape. A selected `view` is planned as
    `stored artifact -> select -> communicate`: the selection runs on the stored
    layout, so `kind` is the operator from the *selected* block's layout to the
    required one, and only the block is ever communicated. A view adds its
    structure, never its codes, to ``specialization_key``.
    """

    target: ValueArtifactAddress
    source: ValueConsumerAddress
    kind: ValueTransferKind
    stored_sharding: jax.sharding.Sharding
    source_sharding: jax.sharding.Sharding
    expected_shape: tuple[int, ...]
    expected_dtype: object
    reused_by_several_consumers: bool = False
    """Whether several source cores of one period read this transfer's result."""
    view: ValueViewDescriptor | None = None
    """Representation the consumer reads the target in; `None` reads it whole."""
    specialization_key: Hashable = field(init=False)

    def __post_init__(self) -> None:
        """Validate the resolved operator and derive its compilation identity."""
        if not isinstance(self.target, ValueArtifactAddress):
            msg = "target must be a ValueArtifactAddress."
            raise TypeError(msg)
        if not isinstance(self.source, ValueConsumerAddress):
            msg = "source must be a ValueConsumerAddress."
            raise TypeError(msg)
        _require_enum(
            value=self.kind, enum_type=ValueTransferKind, label="transfer kind"
        )
        _require_sharding(sharding=self.stored_sharding, label="stored")
        _require_sharding(sharding=self.source_sharding, label="source")
        shape = _normalize_shape(shape=self.expected_shape)
        dtype = jnp.dtype(self.expected_dtype)
        object.__setattr__(self, "expected_shape", shape)
        object.__setattr__(self, "expected_dtype", dtype)
        if self.view is not None:
            _fail_if_view_mismatches_transfer(transfer=self)
        _check_sharding_shape(
            sharding=self.stored_sharding,
            shape=shape,
            label="stored",
        )
        _check_sharding_shape(
            sharding=self.source_sharding,
            shape=self.consumer_shape,
            label="source",
        )
        _validate_edge_identity(target=self.target, source=self.source)
        delivered_sharding = self.stages[-1].input_sharding
        expected = classify_value_transfer(
            stored_sharding=delivered_sharding,
            required_sharding=self.source_sharding,
        )
        if self.kind is not expected:
            msg = (
                f"A transfer from {delivered_sharding} to {self.source_sharding} "
                f"is a {expected.value}, not a {self.kind.value}."
            )
            raise ValueError(msg)

        object.__setattr__(
            self,
            "specialization_key",
            (
                "value-transfer",
                _VALUE_TRANSFER_VERSION,
                self.target.kind,
                self.source.channel,
                self.source.path,
                self.source.argument,
                self.kind,
                self.stored_sharding,
                self.source_sharding,
                shape,
                dtype,
                *(() if self.view is None else (self.view.structure_key,)),
            ),
        )

    @property
    def consumer_shape(self) -> tuple[int, ...]:
        """Return the shape the consumer receives."""
        return self.expected_shape if self.view is None else self.view.consumer_shape

    @property
    def selects(self) -> bool:
        """Whether a selection stage precedes communication."""
        return self.view is not None and self.view.leaf is ValueViewLeaf.SELECTED

    @property
    def delivers_stored_buffer(self) -> bool:
        """Whether the consumer receives the stored artifact's own buffer."""
        return self.kind is ValueTransferKind.ALIGNED_LOCAL and not self.selects

    @property
    def stages(self) -> tuple[TransferStage, ...]:
        """Return the transfer's stages, selection first when there is one."""
        item_bytes = jnp.dtype(self.expected_dtype).itemsize
        selected_sharding = self.stored_sharding
        stages: list[TransferStage] = []
        if self.selects:
            selected_sharding = _selection_sharding(
                view=self.view,  # ty: ignore[invalid-argument-type]
                layout=self.stored_sharding,
            )
            stages.append(
                TransferStage(
                    kind=TransferStageKind.SELECT,
                    input_shape=self.expected_shape,
                    output_shape=self.consumer_shape,
                    input_sharding=self.stored_sharding,
                    output_sharding=selected_sharding,
                    operator=None,
                    item_bytes=item_bytes,
                )
            )
        stages.append(
            TransferStage(
                kind=TransferStageKind.COMMUNICATE,
                input_shape=self.consumer_shape,
                output_shape=self.consumer_shape,
                input_sharding=selected_sharding,
                output_sharding=self.source_sharding,
                operator=self.kind,
                item_bytes=item_bytes,
            )
        )
        return tuple(stages)

    @property
    def cost(self) -> TransferCost:
        """Return what this transfer occupies, from its two concrete layouts.

        A transfer's temporary bytes are the largest per-device sum over the
        fresh buffers its stages allocate. Without a selection that is the
        copy, or nothing for an aligned read; a selected block is always fresh,
        and a copy of it holds the block on both layouts until it completes.
        """
        item_bytes = jnp.dtype(self.expected_dtype).itemsize
        logical_bytes = item_bytes * math.prod(self.consumer_shape)
        stored_devices = sharding_device_ids(sharding=self.stored_sharding)
        required = layout_footprint(
            sharding=self.source_sharding,
            shape=self.consumer_shape,
            item_bytes=item_bytes,
        )
        per_device_bytes = required.bytes_per_device
        fresh: dict[int, int] = {}
        for stage in self.stages:
            if stage.allocates:
                footprint = stage.output_footprint
                for device in footprint.device_ids:
                    fresh[device] = fresh.get(device, 0) + footprint.bytes_per_device
        return TransferCost(
            operation_class=_OPERATION_CLASS_BY_KIND[self.kind],
            logical_bytes=logical_bytes,
            per_device_bytes=per_device_bytes,
            temporary_bytes=max(fresh.values(), default=0),
            devices=tuple(sorted(set(stored_devices) | set(required.device_ids))),
            reused_by_several_consumers=self.reused_by_several_consumers,
        )


def resolve_value_transfer(
    *,
    target: ValueArtifactAddress,
    source: ValueConsumerAddress,
    kind: ValueTransferKind,
    stored_template: object,
    source_sharding: jax.sharding.Sharding,
    view: ValueViewDescriptor | None = None,
) -> ResolvedValueTransfer:
    """Resolve one logical target-to-source edge from its stored template."""
    shape = getattr(stored_template, "shape", None)
    dtype = getattr(stored_template, "dtype", None)
    stored_sharding = getattr(stored_template, "sharding", None)
    if shape is None or dtype is None:
        msg = "A stored value template must expose an absolute shape and dtype."
        raise TypeError(msg)
    if not isinstance(stored_sharding, jax.sharding.Sharding):
        msg = "A stored value template must expose a concrete JAX sharding."
        raise TypeError(msg)
    return ResolvedValueTransfer(
        target=target,
        source=source,
        kind=kind,
        stored_sharding=stored_sharding,
        source_sharding=source_sharding,
        expected_shape=tuple(shape),
        expected_dtype=dtype,
        view=view,
    )


@runtime_checkable
class MaterializedTransferObserver(Protocol):
    """Observe a newly returned concrete copy before its metadata is checked."""

    def __call__(self, *, transfer: ResolvedValueTransfer, array: ValueND) -> None:
        """Receive one fresh operator result, never an aligned value or cache hit."""
        ...


def apply_value_transfer(
    *,
    value: object,
    transfer: ResolvedValueTransfer,
    on_materialized: MaterializedTransferObserver | None = None,
) -> jax.Array:
    """Apply one resolved adapter after validating the exact stored artifact.

    A selected view first selects its block on the stored layout, in one
    executable whose type codes are runtime operands. An `ALIGNED_LOCAL`
    transfer then hands the stored array, or the selected block, on unchanged.
    Every other operator is one recorded `jax.device_put` onto the required
    layout, so the collective XLA emits is the one the plan already names, and
    it moves only the block.
    """
    if not isinstance(transfer, ResolvedValueTransfer):
        msg = "transfer must be a ResolvedValueTransfer."
        raise TypeError(msg)
    stored = _assert_value_metadata(
        value=value,
        expected_shape=transfer.expected_shape,
        expected_dtype=transfer.expected_dtype,
        expected_sharding=transfer.stored_sharding,
        label="stored",
    )
    if transfer.selects:
        stored = _select_stored_block(
            stored=stored, transfer=transfer, on_materialized=on_materialized
        )
    if transfer.kind is ValueTransferKind.ALIGNED_LOCAL:
        return stored
    copied = jax.device_put(stored, transfer.source_sharding)
    if on_materialized is not None:
        on_materialized(transfer=transfer, array=copied)
    _assert_value_metadata(
        value=copied,
        expected_shape=transfer.consumer_shape,
        expected_dtype=transfer.expected_dtype,
        expected_sharding=transfer.source_sharding,
        label="transferred",
    )
    return copied


def apply_value_transfer_plan(
    *,
    arguments: Mapping[str, object],
    plan: Iterable[ResolvedValueTransfer],
    cache: TransferCache | None = None,
    on_materialized: MaterializedTransferObserver | None = None,
) -> Mapping[str, object]:
    """Apply a transfer plan to an immutable copy of a core-argument tree.

    With a `cache`, a transfer marked as reused by several consumers is
    executed once per cache lifetime and served from the cache afterwards.
    An `ALIGNED_LOCAL` transfer's result is the stored value's own buffer, so
    the cache serves it like any other but its `put` never registers it: the
    buffer it names already belongs to the stored artifact.

    A source locator is the read's named argument, or ``channel.value`` when it
    names none, followed by ``path``. Each locator may occur once in a plan.
    Mappings are rebuilt in their original iteration order and frozen; tuples
    remain tuples. Other containers are unsupported, so lowering and runtime
    dispatch cannot silently disagree about traversal.
    """
    if not isinstance(arguments, Mapping):
        msg = "Core arguments for a value-transfer plan must be a mapping."
        raise TypeError(msg)
    transfers = tuple(plan)
    seen: set[tuple[str, tuple[str | int, ...]]] = set()
    result: Mapping[str, object] = MappingProxyType(dict(arguments))
    for transfer in transfers:
        if not isinstance(transfer, ResolvedValueTransfer):
            msg = "A value-transfer plan may contain only ResolvedValueTransfer items."
            raise TypeError(msg)
        locator = (
            transfer.source.argument or transfer.source.channel.value,
            transfer.source.path,
        )
        if locator in seen:
            msg = f"Duplicate value-transfer consumer path: {locator!r}."
            raise ValueError(msg)
        seen.add(locator)
        root, path = locator
        if root not in result:
            msg = f"Value-transfer input argument {root!r} is missing."
            raise KeyError(msg)
        replaced = _replace_transfer_leaf(
            node=result[root],
            path=path,
            transfer=transfer,
            traversed=(root,),
            cache=cache,
            on_materialized=on_materialized,
        )
        updated = dict(result)
        updated[root] = replaced
        result = MappingProxyType(updated)
    return result


def classify_value_transfer(
    *,
    stored_sharding: jax.sharding.Sharding,
    required_sharding: jax.sharding.Sharding,
) -> ValueTransferKind:
    """Name the one operator that takes a stored layout to a required layout.

    The catalogue is total over the pairs the planner can produce:

    - equal layouts stay `ALIGNED_LOCAL`;
    - either layout not being a `NamedSharding` is a `COPY_TO_SOURCE_LAYOUT`,
      which covers a single-device value moved onto any other placement and a
      sharded value read onto one device;
    - on one mesh, sharded to replicated is an `ALL_GATHER`, replicated to
      sharded a `LOCAL_SLICE`, and one named axis to another a `RESHARD`;
    - two same-mesh layouts whose specs agree while some other attribute (a
      `memory_kind`) differs are a `RESHARD`, the conservative reading: a
      recorded representation change rather than a silent no-op;
    - a required mesh that is disjoint from the stored one, or nested inside it,
      or contains it, is a `CROSS_MESH_COPY`.

    Two meshes that share devices while neither contains the other are refused:
    no single collective serves them, and picking one silently would move the
    value through a placement the plan does not record.
    """
    _require_sharding(sharding=stored_sharding, label="stored")
    _require_sharding(sharding=required_sharding, label="required")
    if stored_sharding == required_sharding:
        return ValueTransferKind.ALIGNED_LOCAL
    stored_named = isinstance(stored_sharding, jax.NamedSharding)
    required_named = isinstance(required_sharding, jax.NamedSharding)
    if not stored_named or not required_named:
        return ValueTransferKind.COPY_TO_SOURCE_LAYOUT
    if stored_sharding.mesh == required_sharding.mesh:
        stored_axes = _named_axes(spec=stored_sharding.spec)
        required_axes = _named_axes(spec=required_sharding.spec)
        if stored_axes and not required_axes:
            return ValueTransferKind.ALL_GATHER
        if not stored_axes and required_axes:
            return ValueTransferKind.LOCAL_SLICE
        return ValueTransferKind.RESHARD
    stored_devices = frozenset(stored_sharding.mesh.devices.flat)
    required_devices = frozenset(required_sharding.mesh.devices.flat)
    if (
        not stored_devices & required_devices
        or stored_devices <= required_devices
        or required_devices <= stored_devices
    ):
        return ValueTransferKind.CROSS_MESH_COPY
    msg = (
        "Overlapping but unequal device meshes cannot be served by one planned "
        f"transfer: stored on {sorted(device.id for device in stored_devices)}, "
        f"required on {sorted(device.id for device in required_devices)}."
    )
    raise ExecutionPlanningError(msg)


def _named_axes(*, spec: jax.sharding.PartitionSpec) -> tuple[str, ...]:
    """Return the mesh axes one partition spec shards over, in spec order.

    A spec entry is a mesh-axis name, a tuple of such names, or `None`.  Any
    other entry — a sentinel such as `PartitionSpec.UNCONSTRAINED`, which leaves
    the axis for the compiler to choose — names no placement the plan can
    record, so it is refused rather than read as an axis group.
    """
    axes: list[str] = []
    for entry in spec:
        if entry is None:
            continue
        if isinstance(entry, str):
            axes.append(entry)
            continue
        if isinstance(entry, tuple) and all(isinstance(name, str) for name in entry):
            axes.extend(entry)
            continue
        msg = (
            "A planned partition spec entry must be a mesh-axis name, a tuple "
            f"of names, or None; {spec!r} carries {entry!r}."
        )
        raise ExecutionPlanningError(msg)
    return tuple(axes)


def _replace_transfer_leaf(
    *,
    node: object,
    path: tuple[str | int, ...],
    transfer: ResolvedValueTransfer,
    traversed: tuple[str | int, ...],
    cache: TransferCache | None,
    on_materialized: MaterializedTransferObserver | None,
) -> object:
    """Rebuild one supported argument branch and replace its selected leaf."""
    if not path:
        return _transferred_leaf(
            node=node, transfer=transfer, cache=cache, on_materialized=on_materialized
        )
    segment, *remaining = path
    rest = tuple(remaining)
    if isinstance(node, Mapping):
        if segment not in node:
            msg = f"Value-transfer mapping path {(*traversed, segment)!r} is missing."
            raise KeyError(msg)
        updated = dict(node)
        updated[segment] = _replace_transfer_leaf(
            node=node[segment],
            path=rest,
            transfer=transfer,
            traversed=(*traversed, segment),
            cache=cache,
            on_materialized=on_materialized,
        )
        return MappingProxyType(updated)
    if isinstance(node, tuple):
        if type(segment) is not int:
            msg = (
                "A value-transfer tuple path requires an integer index at "
                f"{traversed!r}, got {segment!r}."
            )
            raise TypeError(msg)
        if segment >= len(node):
            msg = (
                f"Value-transfer tuple index {segment} is out of range at "
                f"{traversed!r}."
            )
            raise IndexError(msg)
        updated = list(node)
        updated[segment] = _replace_transfer_leaf(
            node=node[segment],
            path=rest,
            transfer=transfer,
            traversed=(*traversed, segment),
            cache=cache,
            on_materialized=on_materialized,
        )
        return tuple(updated)
    if is_dataclass(node) and not isinstance(node, type):
        return _replace_dataclass_field(
            node=node,
            segment=segment,
            path=rest,
            transfer=transfer,
            traversed=traversed,
            cache=cache,
            on_materialized=on_materialized,
        )
    msg = (
        f"Value-transfer path {traversed!r} would rebuild a "
        f"{type(node).__name__}; only mapping, tuple, and dataclass containers "
        "are rebuilt."
    )
    raise TypeError(msg)


def _transferred_leaf(
    *,
    node: object,
    transfer: ResolvedValueTransfer,
    cache: TransferCache | None,
    on_materialized: MaterializedTransferObserver | None,
) -> object:
    """Apply one transfer to the selected leaf, sharing a copy where one is cached."""
    if cache is None or not transfer.reused_by_several_consumers:
        return apply_value_transfer(
            value=node, transfer=transfer, on_materialized=on_materialized
        )
    cached = cache.get(transfer=transfer)
    if cached is not None and not cached.is_deleted():
        return cached
    copied = apply_value_transfer(
        value=node, transfer=transfer, on_materialized=on_materialized
    )
    if not isinstance(node, jax.Array):
        msg = "A cached transfer's pre-transfer value must be a concrete JAX array."
        raise TypeError(msg)
    cache.put(transfer=transfer, array=copied, stored=node)
    return copied


def _replace_dataclass_field(
    *,
    node: object,
    segment: str | int,
    path: tuple[str | int, ...],
    transfer: ResolvedValueTransfer,
    traversed: tuple[str | int, ...],
    cache: TransferCache | None,
    on_materialized: MaterializedTransferObserver | None,
) -> object:
    """Rebuild one dataclass branch field by field around the replaced leaf.

    A solver's continuation payload is a frozen dataclass carrying arrays, so
    rebuilding it through its own constructor keeps its type and every field the
    transfer does not touch.
    """
    if type(segment) is not str:
        msg = (
            "A value-transfer dataclass path requires a field name at "
            f"{traversed!r}, got {segment!r}."
        )
        raise TypeError(msg)
    declared = {item.name for item in fields(node)}  # ty: ignore[invalid-argument-type]
    if segment not in declared:
        msg = (
            f"Value-transfer dataclass path {(*traversed, segment)!r} names "
            f"no field of {type(node).__name__}."
        )
        raise KeyError(msg)
    return replace(
        node,  # ty: ignore[invalid-argument-type]
        **{
            segment: _replace_transfer_leaf(
                node=getattr(node, segment),
                path=path,
                transfer=transfer,
                traversed=(*traversed, segment),
                cache=cache,
                on_materialized=on_materialized,
            )
        },
    )


def _validate_edge_identity(
    *, target: ValueArtifactAddress, source: ValueConsumerAddress
) -> None:
    """Match the source node and input leaf to the stored artifact."""
    if source.argument is None:
        expected_regime = (
            target.regime
            if target.kind is not ValueArtifactKind.GATED_CONTINUATION
            else target.target_regime
        )
        if source.path[0] != expected_regime:
            msg = (
                "The first value-consumer path segment must name the addressed "
                f"target: expected {expected_regime!r}, got {source.path[0]!r}."
            )
            raise ValueError(msg)

    if target.kind is ValueArtifactKind.CONTINUATION_LEAF:
        _validate_continuation_leaf_identity(target=target, source=source)
        return

    if _validate_replay_leaf_identity(target=target, source=source):
        return

    if target.kind is ValueArtifactKind.GATED_CONTINUATION:
        if target.regime != source.source_regime:
            msg = (
                "A gated continuation belongs to its economic source regime: "
                f"expected {target.regime!r}, got {source.source_regime!r}."
            )
            raise ValueError(msg)
        if source.channel is not ValueInputChannel.NEXT_REGIME_VALUE:
            msg = (
                "A gated continuation may enter a core only through "
                "next_regime_to_V_arr."
            )
            raise ValueError(msg)
        expected_period = source.source_period + 1
        if target.period != expected_period:
            msg = (
                "A gated continuation's fold period must be one after its source "
                f"period: expected {expected_period}, got {target.period}."
            )
            raise ValueError(msg)
        return

    expected_period = (
        source.source_period
        if source.channel is ValueInputChannel.SAME_PERIOD_VALUE
        else source.source_period + 1
    )
    if target.period != expected_period:
        relation = (
            "equal"
            if source.channel is ValueInputChannel.SAME_PERIOD_VALUE
            else "one after"
        )
        msg = f"A regime-value artifact period must be {relation} its source period."
        raise ValueError(msg)


def _validate_replay_leaf_identity(
    *, target: ValueArtifactAddress, source: ValueConsumerAddress
) -> bool:
    """Validate a keyed replay leaf, and reject replay channels on other artifacts."""
    offsets = {
        ValueInputChannel.CURRENT_REPLAY_ARTIFACT: 0,
        ValueInputChannel.NEXT_REPLAY_ARTIFACT: 1,
    }
    if target.kind is ValueArtifactKind.REPLAY_ARTIFACT_LEAF:
        if source.channel not in offsets:
            raise ValueError("A replay artifact requires a replay-artifact channel.")
        if target.period != source.source_period + offsets[source.channel]:
            raise ValueError("A replay artifact's period must match its read channel.")
        return True
    if source.channel in offsets:
        raise ValueError("A replay-artifact channel requires a replay artifact.")
    return False


def _validate_continuation_leaf_identity(
    *, target: ValueArtifactAddress, source: ValueConsumerAddress
) -> None:
    """Match one addressed continuation leaf to its channel and its period."""
    if source.channel is not ValueInputChannel.CONTINUATION_LEAF:
        msg = (
            "A continuation leaf may enter a core only through the "
            f"next_regime_to_continuation channel, got {source.channel!r}."
        )
        raise ValueError(msg)
    expected_period = source.source_period + 1
    if target.period != expected_period:
        msg = (
            "A continuation-leaf period must be one after its source period: "
            f"expected {expected_period}, got {target.period}."
        )
        raise ValueError(msg)


def _assert_value_metadata(
    *,
    value: object,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
    expected_sharding: jax.sharding.Sharding,
    label: str,
) -> jax.Array:
    """Validate shape, dtype, and exact layout at a transfer boundary."""
    if not isinstance(value, jax.Array):
        msg = f"The {label} transfer value must be a concrete JAX array."
        raise TypeError(msg)
    if value.shape != expected_shape:
        msg = (
            f"The {label} transfer value has shape {value.shape}; "
            f"expected {expected_shape}."
        )
        raise ValueError(msg)
    if value.dtype != expected_dtype:
        msg = (
            f"The {label} transfer value has dtype {value.dtype}; "
            f"expected {expected_dtype}."
        )
        raise TypeError(msg)
    if not runtime_shardings_match(
        actual=value.sharding, expected=expected_sharding, ndim=value.ndim
    ):
        msg = (
            f"The {label} transfer value has sharding {value.sharding}; "
            f"expected {expected_sharding}."
        )
        raise ValueError(msg)
    return value


def _normalize_shape(*, shape: object) -> tuple[int, ...]:
    """Return an immutable absolute shape, rejecting symbolic dimensions."""
    if not isinstance(shape, tuple):
        msg = "A resolved transfer shape must be a tuple."
        raise TypeError(msg)
    if any(type(size) is not int or size < 0 for size in shape):
        msg = (
            f"A resolved transfer requires a nonnegative absolute shape, got {shape!r}."
        )
        raise ValueError(msg)
    return shape


def _require_period(*, period: object, label: str) -> None:
    """Validate an absolute solve-period coordinate."""
    if type(period) is not int or period < 0:
        msg = f"{label} must be a nonnegative Python int, got {period!r}."
        raise ValueError(msg)


def _require_name(*, name: object, label: str) -> None:
    """Validate a nonempty logical name."""
    if not isinstance(name, str) or not name:
        msg = f"{label} must be a nonempty string, got {name!r}."
        raise ValueError(msg)


def _require_enum(*, value: object, enum_type: type[StrEnum], label: str) -> None:
    """Reject untyped strings and future unsupported enum members."""
    if not isinstance(value, enum_type):
        msg = f"{label} must be a {enum_type.__name__}, got {value!r}."
        raise TypeError(msg)


def _validate_path_segment(*, segment: object) -> None:
    """Accept only immutable mapping keys and sequence indices."""
    if isinstance(segment, str):
        if segment:
            return
        msg = "A value consumer path cannot contain an empty string."
        raise ValueError(msg)
    if type(segment) is int:
        if segment >= 0:
            return
        msg = "A value consumer path cannot contain a negative index."
        raise ValueError(msg)
    msg = f"Unsupported value consumer path segment: {segment!r}."
    raise TypeError(msg)


def _require_sharding(*, sharding: object, label: str) -> None:
    """Require a concrete JAX sharding at both transfer endpoints."""
    if not isinstance(sharding, jax.sharding.Sharding):
        msg = f"The {label} layout must be a concrete JAX sharding."
        raise TypeError(msg)


def _check_sharding_shape(
    *, sharding: jax.sharding.Sharding, shape: tuple[int, ...], label: str
) -> None:
    """Fail while planning when an endpoint cannot represent the value rank."""
    checker = getattr(sharding, "check_compatible_aval", None)
    if not callable(checker):
        msg = f"The {label} sharding does not expose shape compatibility checks."
        raise TypeError(msg)
    try:
        checker(shape)
    except ValueError as error:
        msg = f"The {label} sharding is incompatible with value shape {shape}."
        raise ValueError(msg) from error


def _select_stored_block(
    *,
    stored: jax.Array,
    transfer: ResolvedValueTransfer,
    on_materialized: MaterializedTransferObserver | None,
) -> jax.Array:
    """Run a selected view's selection stage and validate the block it returns.

    The block is a fresh buffer, so it is handed to `on_materialized` before
    any check can raise: a completion owner then holds it until the copy that
    reads it, or the consumer it is passed to, has finished.
    """
    operands = _selection_operands(transfer=transfer)
    selected = _select_value_view(value=stored, **operands)
    if on_materialized is not None:
        on_materialized(transfer=transfer, array=selected)
    _assert_value_metadata(
        value=selected,
        expected_shape=transfer.consumer_shape,
        expected_dtype=transfer.expected_dtype,
        expected_sharding=operands["out_sharding"],  # ty: ignore[invalid-argument-type]
        label="selected",
    )
    view = transfer.view
    if view is not None and selected.weak_type is not view.weak_type:
        msg = (
            f"The selected transfer value has weak_type={selected.weak_type}; the "
            "view declares the opposite."
        )
        raise TypeError(msg)
    return selected


def _fail_if_view_mismatches_transfer(*, transfer: ResolvedValueTransfer) -> None:
    """Require a view to describe exactly the artifact and layouts it is planned on."""
    view = transfer.view
    if not isinstance(view, ValueViewDescriptor):
        msg = f"A transfer view must be a ValueViewDescriptor, got {view!r}."
        raise TypeError(msg)
    if view.artifact != transfer.target:
        msg = (
            f"A transfer of {transfer.target!r} cannot carry a view that addresses "
            f"{view.artifact!r}."
        )
        raise ValueError(msg)
    if (view.stored_shape, view.dtype) != (
        transfer.expected_shape,
        transfer.expected_dtype,
    ):
        msg = (
            f"A view of a {view.dtype} array of shape {view.stored_shape} cannot read "
            f"a stored {transfer.expected_dtype} array of shape "
            f"{transfer.expected_shape}."
        )
        raise ValueError(msg)
    if view.required_sharding != transfer.source_sharding:
        msg = (
            f"A view's required layout {view.required_sharding} must equal the "
            f"transfer destination {transfer.source_sharding}."
        )
        raise ValueError(msg)


def _fail_if_selections_invalid(*, view: ValueViewDescriptor) -> None:
    """Check a selected leaf's intervals and the consumer shape they give."""
    if not view.selections:
        msg = "A selected leaf needs at least one selection."
        raise ValueError(msg)
    consumer = list(view.stored_shape)
    removed: list[int] = []
    for item in view.selections:
        if item.state_name not in view.stored_axis_names:
            msg = (
                f"A selection of {item.state_name!r} names no stored axis of "
                f"{view.stored_axis_names!r}."
            )
            raise ValueError(msg)
        axis = view.stored_axis_names.index(item.state_name)
        extent = view.stored_shape[axis]
        if (
            type(item.start) is not int
            or type(item.width) is not int
            or type(item.keep_axis) is not bool
            or item.start < 0
            or item.width < 1
            or item.start + item.width > extent
        ):
            msg = (
                f"The selection [{item.start!r}, {item.start!r} + {item.width!r}) of "
                f"{item.state_name!r} lies outside its {extent} positions."
            )
            raise ValueError(msg)
        if len(item.codes) != item.width or any(
            type(code) is not int for code in item.codes
        ):
            msg = (
                f"A selection of {item.state_name!r} needs one code per selected "
                f"position, got {item.codes!r} for width {item.width}."
            )
            raise ValueError(msg)
        if not item.keep_axis and item.width != 1:
            msg = (
                f"A selection of {item.state_name!r} removes its axis only at width "
                f"1, got width {item.width}."
            )
            raise ValueError(msg)
        consumer[axis] = item.width
        if not item.keep_axis:
            removed.append(axis)
    if len(set(view.selected_axes)) != len(view.selections):
        msg = f"A value view selects one axis at most once, got {view.selections!r}."
        raise ValueError(msg)
    expected = tuple(size for axis, size in enumerate(consumer) if axis not in removed)
    if view.consumer_shape != expected:
        msg = (
            f"A selected view's consumer shape {view.consumer_shape} is not the "
            f"shape {expected} its selections give."
        )
        raise ValueError(msg)


def _selection_sharding(
    *, view: ValueViewDescriptor, layout: jax.sharding.Sharding
) -> jax.sharding.Sharding:
    """Return the layout a block selected on the stored `layout` keeps.

    Selecting along an axis no device partitions is local to every device, so
    the block keeps the stored mesh and the stored partitioning of every other
    axis. Selecting along a partitioned axis would leave the block on only the
    devices holding it; that layout is not planned here and is refused.
    """
    if isinstance(layout, jax.sharding.SingleDeviceSharding):
        return layout
    if not isinstance(layout, jax.NamedSharding):
        msg = (
            "A selected value view requires a single-device or named stored "
            f"layout, got {layout}."
        )
        raise ExecutionPlanningError(msg)
    entries = list(layout.spec)
    entries += [None] * (len(view.stored_shape) - len(entries))
    removed = set()
    for axis, item in zip(view.selected_axes, view.selections, strict=True):
        if entries[axis] is not None:
            msg = (
                f"A selection of {item.state_name!r} runs along an axis partitioned "
                f"over {entries[axis]!r}; selected views of partitioned axes are not "
                "planned."
            )
            raise ExecutionPlanningError(msg)
        if not item.keep_axis:
            removed.add(axis)
    return jax.NamedSharding(
        layout.mesh,
        jax.P(*(entry for axis, entry in enumerate(entries) if axis not in removed)),
        memory_kind=layout.memory_kind,
    )
