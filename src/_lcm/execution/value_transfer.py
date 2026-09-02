"""Planner-owned transfers of stored values into solve-core inputs.

An economic dependency points from a source regime to a target regime, while the
stored value moves in the opposite direction during backward induction.  This module
names both ends independently: a target artifact says which stored array is read, and
a source consumer says exactly where that array enters a core.  Concrete transfer
operators remain deliberately small and fail closed until a production route needs a
larger catalogue.
"""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.typing import RegimeName

_VALUE_TRANSFER_VERSION = 1


class ValueArtifactKind(StrEnum):
    """Kind of stored solve-time value consumed by a source core."""

    REGIME_VALUE = "regime_value"
    GATED_CONTINUATION = "gated_continuation"


class ValueInputChannel(StrEnum):
    """GridSearch argument channel through which a value reaches a core."""

    NEXT_REGIME_VALUE = "next_regime_to_V_arr"
    SAME_PERIOD_VALUE = "same_period_regime_to_V_arr"
    EDGE_REFERENCE_VALUE = "edge_reference_regime_to_V_arr"


class ValueTransferKind(StrEnum):
    """Supported target-to-source representation changes."""

    ALIGNED_LOCAL = "aligned_local"
    COPY_TO_SOURCE_LAYOUT = "copy_to_source_layout"


@dataclass(frozen=True, kw_only=True)
class ValueArtifactAddress:
    """Logical address of one stored target value or gated continuation.

    ``period`` is the value's solved period for :attr:`REGIME_VALUE` and the
    target/fold period for :attr:`GATED_CONTINUATION`.  A gated continuation is
    owned by the economic source regime and edge target together, which prevents
    two distinct ``Wbar`` objects with the same shape from sharing an identity.
    """

    kind: ValueArtifactKind
    period: int
    regime: RegimeName
    target_regime: RegimeName | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported artifact addresses."""
        _require_enum(
            value=self.kind, enum_type=ValueArtifactKind, label="artifact kind"
        )
        _require_period(period=self.period, label="artifact period")
        _require_name(name=self.regime, label="artifact regime")
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

    ``path`` is relative to ``channel``.  For the currently supported mappings,
    its first segment is the target or reference regime key.  Keeping the path
    separate from the artifact identity allows one stored value to feed several
    argument leaves without conflating their liveness events.
    """

    source_period: int
    source_regime: RegimeName
    core_key: str
    channel: ValueInputChannel
    path: tuple[str | int, ...]

    def __post_init__(self) -> None:
        """Validate the complete core-input locator."""
        _require_period(period=self.source_period, label="source period")
        _require_name(name=self.source_regime, label="source regime")
        _require_name(name=self.core_key, label="core key")
        _require_enum(
            value=self.channel, enum_type=ValueInputChannel, label="input channel"
        )
        if not isinstance(self.path, tuple) or not self.path:
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
    It retains the argument-tree role, including the target mapping key in
    ``source.path``, plus the operator, concrete layouts, and leaf metadata, so
    behaviorally different transfers cannot share a lowering.
    """

    target: ValueArtifactAddress
    source: ValueConsumerAddress
    kind: ValueTransferKind
    stored_sharding: jax.sharding.Sharding
    source_sharding: jax.sharding.Sharding
    expected_shape: tuple[int, ...]
    expected_dtype: object
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
        if self.kind is ValueTransferKind.ALIGNED_LOCAL:
            if self.stored_sharding != self.source_sharding:
                msg = (
                    "ALIGNED_LOCAL requires identical stored and source shardings; "
                    "a representation change must use COPY_TO_SOURCE_LAYOUT."
                )
                raise ValueError(msg)
        elif self.kind is ValueTransferKind.COPY_TO_SOURCE_LAYOUT:
            if self.stored_sharding == self.source_sharding:
                msg = (
                    "COPY_TO_SOURCE_LAYOUT requires a distinct source sharding; "
                    "an unchanged representation must use ALIGNED_LOCAL."
                )
                raise ValueError(msg)
        else:
            msg = f"Unsupported value transfer kind: {self.kind!r}."
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
                self.kind,
                self.stored_sharding,
                self.source_sharding,
                shape,
                dtype,
            ),
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
    """Apply one resolved adapter after validating the exact stored artifact."""
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
    if transfer.kind is ValueTransferKind.COPY_TO_SOURCE_LAYOUT:
        copied = jax.device_put(stored, transfer.source_sharding)
        _assert_value_metadata(
            value=copied,
            expected_shape=transfer.expected_shape,
            expected_dtype=transfer.expected_dtype,
            expected_sharding=transfer.source_sharding,
            label="transferred",
        )
        return copied
    msg = f"Unsupported value transfer kind: {transfer.kind!r}."
    raise ValueError(msg)


def apply_value_transfer_plan(
    *,
    arguments: Mapping[str, object],
    plan: Iterable[ResolvedValueTransfer],
) -> Mapping[str, object]:
    """Apply a transfer plan to an immutable copy of a core-argument tree.

    A source locator is ``channel.value`` followed by ``path``. Each locator
    may occur once in a plan. Mappings are rebuilt in their original iteration
    order and frozen; tuples remain tuples. Other containers are unsupported,
    so lowering and runtime dispatch cannot silently disagree about traversal.
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
        locator = (transfer.source.channel.value, transfer.source.path)
        if locator in seen:
            msg = f"Duplicate value-transfer consumer path: {locator!r}."
            raise ValueError(msg)
        seen.add(locator)
        channel, path = locator
        if channel not in result:
            msg = f"Value-transfer input channel {channel!r} is missing."
            raise KeyError(msg)
        replaced = _replace_transfer_leaf(
            node=result[channel],
            path=path,
            transfer=transfer,
            traversed=(channel,),
        )
        updated = dict(result)
        updated[channel] = replaced
        result = MappingProxyType(updated)
    return result


def _replace_transfer_leaf(
    *,
    node: object,
    path: tuple[str | int, ...],
    transfer: ResolvedValueTransfer,
    traversed: tuple[str | int, ...],
) -> object:
    """Rebuild one supported argument branch and replace its selected leaf."""
    if not path:
        return apply_value_transfer(value=node, transfer=transfer)
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
        )
        return tuple(updated)
    msg = (
        f"Value-transfer path {traversed!r} reached unsupported container "
        f"{type(node).__name__}."
    )
    raise TypeError(msg)


def _validate_edge_identity(
    *, target: ValueArtifactAddress, source: ValueConsumerAddress
) -> None:
    """Match the source node and input leaf to the stored artifact."""
    expected_regime = (
        target.regime
        if target.kind is ValueArtifactKind.REGIME_VALUE
        else target.target_regime
    )
    if source.path[0] != expected_regime:
        msg = (
            "The first value-consumer path segment must name the addressed target: "
            f"expected {expected_regime!r}, got {source.path[0]!r}."
        )
        raise ValueError(msg)

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
    if value.sharding != expected_sharding:
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
