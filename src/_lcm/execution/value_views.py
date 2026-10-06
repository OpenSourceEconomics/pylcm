"""Ownership accounting and admission for transfers that read a value view.

A selected view is planned as `stored artifact -> select -> communicate`. While
it runs, three kinds of buffer are live on each device: the stored owner, which
stays charged for as long as any consumer may still read it; the selected block,
a fresh buffer on the stored layout; and the communicated copy on the required
layout. An aligned communication stage forwards its input and allocates
nothing, and an aligned shared leaf aliases its owner outright. This module
reports the three separately, from the transfer's own stages, and refuses a
transfer whose peak does not fit a budget before anything is dispatched.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax

from _lcm.execution.footprint import layout_footprint
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    TransferStageKind,
    _select_value_view,
    _selection_operands,
)
from lcm.exceptions import ExecutionPlanningError


@dataclass(frozen=True, kw_only=True)
class ValueTransferFootprint:
    """Per-device bytes one planned transfer holds, by owner."""

    owner_bytes: MappingProxyType[int, int]
    """The stored artifact, on every device holding a shard of it."""

    selected_bytes: MappingProxyType[int, int]
    """The selected block on the stored layout; empty without a selection."""

    destination_bytes: MappingProxyType[int, int]
    """A copy on the required layout; empty when the last stage is aligned."""

    @property
    def peak_bytes(self) -> MappingProxyType[int, int]:
        """Return every device's bytes with all three owners live at once."""
        devices = (
            self.owner_bytes.keys()
            | self.selected_bytes.keys()
            | self.destination_bytes.keys()
        )
        return MappingProxyType(
            {
                device: self.owner_bytes.get(device, 0)
                + self.selected_bytes.get(device, 0)
                + self.destination_bytes.get(device, 0)
                for device in sorted(devices)
            }
        )


def plan_value_transfer_footprint(
    *, transfer: ResolvedValueTransfer
) -> ValueTransferFootprint:
    """Return the owner, selected-block and destination bytes of one transfer.

    Every stage that allocates charges its output once on the devices that
    hold it; the intermediate block is conservatively charged alongside the
    copy that reads it.
    """
    owner = layout_footprint(
        sharding=transfer.stored_sharding,
        shape=transfer.expected_shape,
        item_bytes=transfer.expected_dtype.itemsize,  # ty: ignore[unresolved-attribute]
    )
    charged: dict[TransferStageKind, MappingProxyType[int, int]] = {}
    for stage in transfer.stages:
        if stage.allocates:
            footprint = stage.output_footprint
            charged[stage.kind] = MappingProxyType(
                dict.fromkeys(footprint.device_ids, footprint.bytes_per_device)
            )
    empty: MappingProxyType[int, int] = MappingProxyType({})
    return ValueTransferFootprint(
        owner_bytes=MappingProxyType(
            dict.fromkeys(owner.device_ids, owner.bytes_per_device)
        ),
        selected_bytes=charged.get(TransferStageKind.SELECT, empty),
        destination_bytes=charged.get(TransferStageKind.COMMUNICATE, empty),
    )


def fail_if_value_transfer_exceeds_budget(
    *,
    transfer: ResolvedValueTransfer,
    budget_bytes: int,
    other_resident_bytes: Mapping[int, int],
) -> None:
    """Refuse, before dispatch, a transfer whose peak exceeds a per-device budget.

    Args:
        transfer: The planned transfer.
        budget_bytes: Bytes admission allows on each device.
        other_resident_bytes: Bytes already live on each device besides the
            transfer's own stored owner, selected block and copy.

    Raises:
        ExecutionPlanningError: Naming every device whose need exceeds the
            budget, with its need and the owners that make it up.

    """
    if type(budget_bytes) is not int or budget_bytes < 1:
        msg = f"A transfer budget must be a positive int, got {budget_bytes!r}."
        raise ValueError(msg)
    footprint = plan_value_transfer_footprint(transfer=transfer)
    over = tuple(
        (device, other_resident_bytes.get(device, 0) + peak)
        for device, peak in footprint.peak_bytes.items()
        if other_resident_bytes.get(device, 0) + peak > budget_bytes
    )
    if over:
        details = "; ".join(
            f"device {device} needs {need} bytes (stored owner "
            f"{footprint.owner_bytes.get(device, 0)}, selected block "
            f"{footprint.selected_bytes.get(device, 0)}, copy "
            f"{footprint.destination_bytes.get(device, 0)}, other resident "
            f"{other_resident_bytes.get(device, 0)})"
            for device, need in over
        )
        msg = (
            f"The transfer of {transfer.target!r} into {transfer.source!r} does not "
            f"fit the budget of {budget_bytes} bytes per device: {details}. Raise "
            "the budget, or select a narrower block."
        )
        raise ExecutionPlanningError(msg)


def lower_value_view_selection(
    *, transfer: ResolvedValueTransfer
) -> jax.stages.Lowered:
    """Lower a selected view's selection stage exactly as dispatch runs it.

    The stored value is described abstractly on its stored layout, so the
    lowering and its compiled memory report describe the selection alone.
    """
    return _select_value_view.lower(
        value=jax.ShapeDtypeStruct(
            transfer.expected_shape,
            transfer.expected_dtype,
            sharding=transfer.stored_sharding,
        ),
        **_selection_operands(transfer=transfer),
    )
