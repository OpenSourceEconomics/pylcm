"""Tests for planner-owned target-to-source value transfers."""

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    apply_value_transfer,
    apply_value_transfer_plan,
    resolve_value_transfer,
)


def _mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.asarray(jax.devices()), ("device",))


def _named_sharding() -> jax.NamedSharding:
    return jax.NamedSharding(mesh=_mesh(), spec=jax.P())


def _stored_value() -> jax.Array:
    return jax.device_put(jnp.arange(4, dtype=jnp.float32), _named_sharding())


def _target(
    *,
    kind: ValueArtifactKind = ValueArtifactKind.REGIME_VALUE,
    period: int = 3,
) -> ValueArtifactAddress:
    return ValueArtifactAddress(
        kind=kind,
        period=period,
        regime="working" if kind is ValueArtifactKind.REGIME_VALUE else "couple",
        target_regime=(
            None if kind is ValueArtifactKind.REGIME_VALUE else "continuing_couple"
        ),
    )


def _source(
    *,
    source_period: int = 2,
    source_regime: str = "working",
    channel: ValueInputChannel = ValueInputChannel.NEXT_REGIME_VALUE,
    path: tuple[str | int, ...] = ("working",),
) -> ValueConsumerAddress:
    return ValueConsumerAddress(
        source_period=source_period,
        source_regime=source_regime,
        core_key="main",
        channel=channel,
        path=path,
    )


def _resolve_aligned(*, value: jax.Array | None = None) -> ResolvedValueTransfer:
    stored = _stored_value() if value is None else value
    return resolve_value_transfer(
        target=_target(),
        source=_source(),
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=stored,
        source_sharding=stored.sharding,
    )


def test_aligned_local_validates_and_returns_the_identical_array() -> None:
    stored = _stored_value()
    transfer = _resolve_aligned(value=stored)

    result = apply_value_transfer(value=stored, transfer=transfer)

    assert result is stored
    assert hash(transfer)
    assert hash(transfer.specialization_key)


def test_copy_explicitly_places_value_in_the_source_layout() -> None:
    stored = _stored_value()
    source_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    transfer = resolve_value_transfer(
        target=_target(),
        source=_source(),
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_template=stored,
        source_sharding=source_sharding,
    )

    result = apply_value_transfer(value=stored, transfer=transfer)

    assert result is not stored
    assert result.sharding == source_sharding
    np.testing.assert_array_equal(result, stored)


def test_gated_continuation_identity_keeps_source_target_and_fold_period() -> None:
    target = _target(kind=ValueArtifactKind.GATED_CONTINUATION, period=7)
    source = _source(
        source_period=6, source_regime="couple", path=("continuing_couple",)
    )
    stored = _stored_value()

    transfer = resolve_value_transfer(
        target=target,
        source=source,
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=stored,
        source_sharding=stored.sharding,
    )

    assert transfer.target.regime == "couple"
    assert transfer.target.target_regime == "continuing_couple"
    assert transfer.target.period == 7


def test_specialization_key_omits_coordinates_but_tracks_behavior() -> None:
    stored = _stored_value()
    first = _resolve_aligned(value=stored)
    other_coordinates = resolve_value_transfer(
        target=_target(period=9),
        source=_source(source_period=8),
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=stored,
        source_sharding=stored.sharding,
    )
    other_channel = resolve_value_transfer(
        target=_target(),
        source=_source(
            source_period=3,
            channel=ValueInputChannel.SAME_PERIOD_VALUE,
        ),
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=stored,
        source_sharding=stored.sharding,
    )

    assert first != other_coordinates
    assert first.specialization_key == other_coordinates.specialization_key
    assert first.specialization_key != other_channel.specialization_key


def test_addresses_are_immutable_and_same_artifact_can_feed_multiple_paths() -> None:
    target = _target()
    first = _source(path=("working",))
    second = _source(path=("working", 0))

    assert first != second
    assert hash(target)
    with pytest.raises(FrozenInstanceError):
        target.period = 4  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"kind": "regime_value"}, BeartypeCallHintParamViolation, "parameter kind"),
        ({"period": -1}, ValueError, "artifact period"),
        ({"regime": ""}, ValueError, "artifact regime"),
        ({"target_regime": "working"}, ValueError, "cannot name"),
    ],
)
def test_regime_value_address_fails_closed(*, kwargs, error, message) -> None:
    values = {
        "kind": ValueArtifactKind.REGIME_VALUE,
        "period": 3,
        "regime": "working",
        "target_regime": None,
    }
    values.update(kwargs)

    with pytest.raises(error, match=message):
        ValueArtifactAddress(**values)


def test_gated_continuation_requires_an_edge_target() -> None:
    with pytest.raises(ValueError, match="target regime"):
        ValueArtifactAddress(
            kind=ValueArtifactKind.GATED_CONTINUATION,
            period=3,
            regime="couple",
        )


@pytest.mark.parametrize(
    ("replacement", "error", "message"),
    [
        (
            {"channel": "next_regime_to_V_arr"},
            BeartypeCallHintParamViolation,
            "parameter channel",
        ),
        ({"path": []}, BeartypeCallHintParamViolation, "parameter path"),
        ({"path": ()}, TypeError, "non-empty tuple"),
        ({"path": ("",)}, ValueError, "empty string"),
        ({"path": (-1,)}, ValueError, "negative index"),
        ({"path": (True,)}, TypeError, "Unsupported"),
        ({"source_period": True}, ValueError, "source period"),
    ],
)
def test_consumer_address_fails_closed(*, replacement, error, message) -> None:
    kwargs = {
        "source_period": 2,
        "source_regime": "working",
        "core_key": "main",
        "channel": ValueInputChannel.NEXT_REGIME_VALUE,
        "path": ("working",),
    }
    kwargs.update(replacement)

    with pytest.raises(error, match=message):
        ValueConsumerAddress(**kwargs)


def test_resolver_rejects_path_that_does_not_address_artifact() -> None:
    stored = _stored_value()

    with pytest.raises(ValueError, match="first value-consumer path"):
        resolve_value_transfer(
            target=_target(),
            source=_source(path=("retired",)),
            kind=ValueTransferKind.ALIGNED_LOCAL,
            stored_template=stored,
            source_sharding=stored.sharding,
        )


def test_resolver_requires_concrete_shape_dtype_and_sharding() -> None:
    with pytest.raises(TypeError, match="shape and dtype"):
        resolve_value_transfer(
            target=_target(),
            source=_source(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
            stored_template=object(),
            source_sharding=_named_sharding(),
        )
    with pytest.raises(TypeError, match="concrete JAX sharding"):
        resolve_value_transfer(
            target=_target(),
            source=_source(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
            stored_template=jax.ShapeDtypeStruct((4,), jnp.float32),
            source_sharding=_named_sharding(),
        )


def test_resolver_rejects_misclassified_or_incompatible_layouts() -> None:
    stored = _stored_value()
    different = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with pytest.raises(ValueError, match="ALIGNED_LOCAL requires identical"):
        resolve_value_transfer(
            target=_target(),
            source=_source(),
            kind=ValueTransferKind.ALIGNED_LOCAL,
            stored_template=stored,
            source_sharding=different,
        )
    with pytest.raises(ValueError, match="requires a distinct"):
        resolve_value_transfer(
            target=_target(),
            source=_source(),
            kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
            stored_template=stored,
            source_sharding=stored.sharding,
        )
    with pytest.raises(ValueError, match="incompatible with value shape"):
        resolve_value_transfer(
            target=ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=3,
                regime="scalar",
            ),
            source=_source(path=("scalar",)),
            kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
            stored_template=jax.device_put(jnp.asarray(1.0), different),
            source_sharding=jax.NamedSharding(mesh=_mesh(), spec=jax.P("device")),
        )


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (jnp.zeros((5,), dtype=jnp.float32), ValueError, "shape"),
        (jnp.zeros((4,), dtype=jnp.int32), TypeError, "dtype"),
        (object(), TypeError, "concrete JAX array"),
    ],
)
def test_apply_rejects_wrong_stored_metadata(*, value, error, message) -> None:
    transfer = _resolve_aligned()

    with pytest.raises(error, match=message):
        apply_value_transfer(value=value, transfer=transfer)


def test_apply_rejects_wrong_stored_sharding() -> None:
    stored = _stored_value()
    transfer = _resolve_aligned(value=stored)
    wrong = jax.device_put(
        jnp.arange(4, dtype=jnp.float32),
        jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )

    with pytest.raises(ValueError, match="stored transfer value has sharding"):
        apply_value_transfer(value=wrong, transfer=transfer)


def test_resolved_transfer_rejects_untyped_transfer_kind() -> None:
    stored = _stored_value()
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter kind"):
        ResolvedValueTransfer(
            target=_target(),
            source=_source(),
            kind="aligned_local",  # ty: ignore[invalid-argument-type]
            stored_sharding=stored.sharding,
            source_sharding=stored.sharding,
            expected_shape=stored.shape,
            expected_dtype=stored.dtype,
        )


@pytest.mark.parametrize(
    ("target", "source", "message"),
    [
        (
            _target(kind=ValueArtifactKind.GATED_CONTINUATION, period=7),
            _source(
                source_period=6,
                source_regime="other_couple",
                path=("continuing_couple",),
            ),
            "belongs to its economic source regime",
        ),
        (
            _target(kind=ValueArtifactKind.GATED_CONTINUATION, period=7),
            _source(
                source_period=7,
                source_regime="couple",
                channel=ValueInputChannel.SAME_PERIOD_VALUE,
                path=("continuing_couple",),
            ),
            "only through next_regime_to_V_arr",
        ),
        (
            _target(kind=ValueArtifactKind.GATED_CONTINUATION, period=7),
            _source(
                source_period=5,
                source_regime="couple",
                path=("continuing_couple",),
            ),
            "fold period",
        ),
        (
            _target(period=3),
            _source(
                source_period=2,
                channel=ValueInputChannel.SAME_PERIOD_VALUE,
            ),
            "regime-value artifact period",
        ),
        (
            _target(period=3),
            _source(source_period=3),
            "regime-value artifact period",
        ),
        (
            _target(period=3),
            _source(
                source_period=3,
                channel=ValueInputChannel.EDGE_REFERENCE_VALUE,
            ),
            "regime-value artifact period",
        ),
    ],
    ids=[
        "gated-source",
        "gated-channel",
        "gated-fold-period",
        "same-period-value",
        "next-period-value",
        "edge-reference-value",
    ],
)
def test_resolver_rejects_incoherent_source_target_coordinates(
    *,
    target: ValueArtifactAddress,
    source: ValueConsumerAddress,
    message: str,
) -> None:
    stored = _stored_value()

    with pytest.raises(ValueError, match=message):
        resolve_value_transfer(
            target=target,
            source=source,
            kind=ValueTransferKind.ALIGNED_LOCAL,
            stored_template=stored,
            source_sharding=stored.sharding,
        )


def test_plan_rebuilds_nested_mappings_and_tuples_without_mutation() -> None:
    stored = _stored_value()
    destination = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    working = (stored,)
    inner = MappingProxyType({"unused": object(), "working": working})
    arguments = MappingProxyType(
        {
            "before": object(),
            ValueInputChannel.NEXT_REGIME_VALUE.value: inner,
            "after": object(),
        }
    )
    transfer = resolve_value_transfer(
        target=_target(),
        source=_source(path=("working", 0)),
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_template=stored,
        source_sharding=destination,
    )

    result = apply_value_transfer_plan(arguments=arguments, plan=(transfer,))

    assert isinstance(result, MappingProxyType)
    assert tuple(result) == tuple(arguments)
    rebuilt_inner = result[ValueInputChannel.NEXT_REGIME_VALUE.value]
    assert isinstance(rebuilt_inner, MappingProxyType)
    assert tuple(rebuilt_inner) == tuple(inner)
    assert rebuilt_inner is not inner
    assert isinstance(rebuilt_inner["working"], tuple)
    assert rebuilt_inner["working"][0].sharding == destination
    assert working[0] is stored


def test_plan_rejects_duplicate_consumer_paths() -> None:
    stored = _stored_value()
    transfer = _resolve_aligned(value=stored)
    arguments = {
        ValueInputChannel.NEXT_REGIME_VALUE.value: {"working": stored},
    }

    with pytest.raises(ValueError, match="Duplicate value-transfer consumer path"):
        apply_value_transfer_plan(
            arguments=arguments,
            plan=(transfer, transfer),
        )


def test_plan_rejects_missing_channel_or_mapping_path() -> None:
    stored = _stored_value()
    transfer = _resolve_aligned(value=stored)

    with pytest.raises(KeyError, match="input channel"):
        apply_value_transfer_plan(arguments={}, plan=(transfer,))
    with pytest.raises(KeyError, match="mapping path"):
        apply_value_transfer_plan(
            arguments={
                ValueInputChannel.NEXT_REGIME_VALUE.value: {"retired": stored},
            },
            plan=(transfer,),
        )


@pytest.mark.parametrize(
    ("path", "branch", "error", "message"),
    [
        (("working", 0), [object()], TypeError, "unsupported container list"),
        (("working", "leaf"), (object(),), TypeError, "requires an integer index"),
        (("working", 1), (object(),), IndexError, "out of range"),
    ],
)
def test_plan_rejects_unsupported_or_invalid_tree_traversal(
    *,
    path: tuple[str | int, ...],
    branch: object,
    error: type[Exception],
    message: str,
) -> None:
    stored = _stored_value()
    transfer = resolve_value_transfer(
        target=_target(),
        source=_source(path=path),
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=stored,
        source_sharding=stored.sharding,
    )
    arguments = {
        ValueInputChannel.NEXT_REGIME_VALUE.value: {"working": branch},
    }

    with pytest.raises(error, match=message):
        apply_value_transfer_plan(arguments=arguments, plan=(transfer,))
