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

from _lcm.execution.footprint import layout_footprint, sharding_device_ids
from _lcm.execution.runtime_sharding import runtime_shardings_match
from _lcm.typing import RegimeName
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import ArtifactKey

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
        _check_sharding_shape(
            sharding=self.stored_sharding,
            shape=shape,
            label="stored",
        )
        _check_sharding_shape(
            sharding=self.source_sharding,
            shape=shape,
            label="source",
        )
        _validate_edge_identity(target=self.target, source=self.source)
        expected = classify_value_transfer(
            stored_sharding=self.stored_sharding,
            required_sharding=self.source_sharding,
        )
        if self.kind is not expected:
            msg = (
                f"A transfer from {self.stored_sharding} to {self.source_sharding} "
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
            ),
        )

    @property
    def cost(self) -> TransferCost:
        """Return what this transfer occupies, from its two concrete layouts."""
        item_bytes = jnp.dtype(self.expected_dtype).itemsize
        logical_bytes = item_bytes * math.prod(self.expected_shape)
        stored_devices = sharding_device_ids(sharding=self.stored_sharding)
        required = layout_footprint(
            sharding=self.source_sharding,
            shape=self.expected_shape,
            item_bytes=item_bytes,
        )
        per_device_bytes = required.bytes_per_device
        return TransferCost(
            operation_class=_OPERATION_CLASS_BY_KIND[self.kind],
            logical_bytes=logical_bytes,
            per_device_bytes=per_device_bytes,
            temporary_bytes=(
                0 if self.kind is ValueTransferKind.ALIGNED_LOCAL else per_device_bytes
            ),
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
    )


def apply_value_transfer(
    *, value: object, transfer: ResolvedValueTransfer
) -> jax.Array:
    """Apply one resolved adapter after validating the exact stored artifact.

    An `ALIGNED_LOCAL` transfer hands the stored array on unchanged.  Every other
    operator is one recorded `jax.device_put` onto the required layout, so the
    collective XLA emits is the one the plan already names.
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
    if transfer.kind is ValueTransferKind.ALIGNED_LOCAL:
        return stored
    copied = jax.device_put(stored, transfer.source_sharding)
    _assert_value_metadata(
        value=copied,
        expected_shape=transfer.expected_shape,
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
) -> object:
    """Rebuild one supported argument branch and replace its selected leaf."""
    if not path:
        return _transferred_leaf(node=node, transfer=transfer, cache=cache)
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
) -> object:
    """Apply one transfer to the selected leaf, sharing a copy where one is cached."""
    if cache is None or not transfer.reused_by_several_consumers:
        return apply_value_transfer(value=node, transfer=transfer)
    cached = cache.get(transfer=transfer)
    if cached is not None and not cached.is_deleted():
        return cached
    copied = apply_value_transfer(value=node, transfer=transfer)
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
