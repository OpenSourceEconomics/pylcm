"""Solved values kept in pinned host memory, read back onto the device.

Under `ExecutionConfig.value_storage="host"` a solve keeps each period's values in
pinned host memory. Nothing outside the solve computes on those arrays: arithmetic
on a host-memory array runs as slow host compute and leaves its result in host
memory. Every reader instead receives a device copy, so results, reads and panels
are those of device storage.
"""

import weakref
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import jax
import numpy as np
from jax.sharding import SingleDeviceSharding

from _lcm.execution.footprint import per_device_footprint
from _lcm.typing import ArtifactPayload
from lcm._solver_api.entries import _LazyEntry
from lcm._solver_api.identity import LoadState
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND, RegimeName

# The memory kind a stored value lives in.
HOST_MEMORY_KIND: Literal["pinned_host"] = "pinned_host"

# Periods whose device copies a read-through view keeps at once: a simulation
# step reads its own period, the one its subjects land in and, replaying a
# same-period exit, the one after that.
_CACHED_PERIODS = 3


def copy_values_to_host(
    *, values: Mapping[RegimeName, jax.Array]
) -> MappingProxyType[RegimeName, jax.Array]:
    """Start copying one period's arrays to pinned host memory, without waiting.

    Each copy keeps its array's sharding, so every device copies its own slice,
    and runs while the device computes whatever is dispatched after it.
    """
    return MappingProxyType(
        {
            regime_name: jax.device_put(
                value, value.sharding.with_memory_kind(HOST_MEMORY_KIND)
            )
            for regime_name, value in values.items()
        }
    )


def is_host_value(value: jax.Array) -> bool:
    """Whether `value` is an array a solve stored in pinned host memory."""
    return value.sharding.memory_kind == HOST_MEMORY_KIND


def to_device(value: jax.Array) -> jax.Array:
    """Copy a stored array back to device memory on its own sharding."""
    return jax.device_put(value, value.sharding.with_memory_kind("device"))


def to_cpu_device(value: jax.Array) -> jax.Array:
    """Place a stored array on the CPU device, where the solve keeps its policies.

    An artifact store copies the arrays it hands out; on the CPU device that copy
    is an ordinary computation, while on host memory it would be slow host compute
    leaving its result in host memory.
    """
    # Naming the memory kind matters on a CPU backend, where the stored array's
    # device is this same CPU device and only the memory kind differs.
    return jax.device_put(
        value, SingleDeviceSharding(jax.devices("cpu")[0], memory_kind="device")
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class HostValueEntry(_LazyEntry):
    """One stored value in pinned host memory, copied to the device when read."""

    value: FloatND
    """The stored value, in pinned host memory."""

    budget_bytes: int | None = None
    """Per-device budget of the solve that stored the value; `None` unbudgeted."""

    @property
    def load_state(self) -> LoadState:
        """Nothing of a host-stored value is on a device until it is read."""
        return LoadState.UNLOADED

    def materialize(self, *, template: ArtifactPayload | None = None) -> FloatND:
        """Copy the value onto its devices."""
        del template
        return to_device(self.value)

    def fresh_value(self) -> FloatND:
        """Copy the value into a device buffer owned by this reader alone."""
        return to_device(self.value)

    def host_value(self) -> np.ndarray:
        """Return the value as a host array without placing it on a device."""
        return np.asarray(self.value)

    @classmethod
    def _admit_joint_materialization(cls, *, entries: tuple[_LazyEntry, ...]) -> None:
        """Refuse to place host-stored values on the devices together above budget."""
        admit_values_on_devices(
            layouts=tuple(
                (entry.value, entry.budget_bytes)
                for entry in entries
                if isinstance(entry, HostValueEntry)
            ),
            where="host-stored",
        )


def admit_values_on_devices(
    *,
    layouts: tuple[tuple[jax.Array | jax.ShapeDtypeStruct, int | None], ...],
    where: str,
) -> None:
    """Refuse to place stored values on their devices at once above the budget.

    Each layout is a value's array or shape, on the sharding it is read onto,
    with the per-device budget of the solve that stored it.

    Raises:
        ExecutionPlanningError: Naming each device whose need exceeds the
            budget, with the remedies.

    """
    budgets = {budget for _, budget in layouts if budget is not None}
    if not budgets:
        return
    budget = min(budgets)
    need: dict[int, int] = {}
    for layout, _ in layouts:
        footprint = per_device_footprint(array=layout)
        for device in footprint.device_ids:
            need[device] = need.get(device, 0) + footprint.bytes_per_device
    over = {device: count for device, count in need.items() if count > budget}
    if not over:
        return
    msg = (
        f"Materializing {len(layouts)} {where} values at once needs "
        f"{dict(sorted(over.items()))!r} bytes on devices whose budget is "
        f"{budget} bytes. The solve kept them off the devices with "
        "ExecutionConfig.value_storage because they do not fit together. Read "
        "them one at a time with SolutionResult.value(period=..., regime=...), "
        "save the result with SolutionResult.save, or raise "
        "ExecutionConfig.device_memory_bytes."
    )
    raise ExecutionPlanningError(msg)


class _RecentDeviceCopies:
    """Device copies of the stored arrays of the most recently read periods."""

    def __init__(
        self,
        *,
        stand_in: MappingProxyType[
            int, MappingProxyType[RegimeName, jax.ShapeDtypeStruct]
        ],
    ) -> None:
        self._copies: dict[int, dict[RegimeName, jax.Array]] = {}
        self.stand_in = stand_in
        """The view's values as shapes on their device placement, for planners."""

    def held(self) -> tuple[dict[RegimeName, jax.Array], ...]:
        """Return the device copies kept, one mapping per cached period."""
        return tuple(self._copies.values())

    def read(
        self, *, period: int, regime: RegimeName, stored: jax.Array | _LazyEntry
    ) -> jax.Array:
        """Return the device copy of one stored value, copying it on first read.

        The period read least recently is dropped first. A reader holds a value
        to the same object while its period is open, so a period still in use
        must outlive one that was merely copied earlier.
        """
        if period in self._copies:
            self._copies[period] = self._copies.pop(period)
        else:
            while len(self._copies) >= _CACHED_PERIODS:
                del self._copies[next(iter(self._copies))]
            self._copies[period] = {}
        copies = self._copies[period]
        if regime not in copies:
            copies[regime] = (
                cast("jax.Array", stored.materialize())
                if isinstance(stored, _LazyEntry)
                else to_device(stored)
            )
        return copies[regime]


class _DeviceReadPeriod(Mapping[RegimeName, jax.Array]):
    """One period's stored arrays, each handed out as a device copy."""

    def __init__(
        self,
        *,
        period: int,
        stored: Mapping[RegimeName, jax.Array | _LazyEntry],
        copies: _RecentDeviceCopies,
    ) -> None:
        self._period = period
        self._stored = stored
        self._copies = copies

    def __getitem__(self, regime: RegimeName) -> jax.Array:
        return self._copies.read(
            period=self._period, regime=regime, stored=self._stored[regime]
        )

    def __iter__(self) -> Iterator[RegimeName]:
        return iter(self._stored)

    def __len__(self) -> int:
        return len(self._stored)


def device_read_view(
    *, values: Mapping[int, Mapping[RegimeName, jax.Array | _LazyEntry]]
) -> MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]:
    """Return host-stored arrays as a mapping that hands out device copies.

    A value is copied when it is first read and kept while its period is one of
    the three most recently read, so a reader that walks the periods in order
    holds at most three periods on the device and copies each period once per
    walk.
    """
    copies = _RecentDeviceCopies(
        stand_in=MappingProxyType(
            {
                period: MappingProxyType(
                    {
                        regime: _device_stand_in(stored=value)
                        for regime, value in stored.items()
                    }
                )
                for period, stored in values.items()
            }
        )
    )
    view = MappingProxyType(
        {
            period: MappingProxyType(
                _DeviceReadPeriod(period=period, stored=stored, copies=copies)
            )
            for period, stored in values.items()
        }
    )
    # Only the view reaches `copies`, so this entry goes with the view and a
    # later object reusing its id finds nothing.
    _VIEW_COPIES[id(view)] = copies
    return view


if TYPE_CHECKING:
    type _StoredValues = Mapping[int, Mapping[RegimeName, jax.Array]]
else:
    # A runtime check samples one entry, and sampling a read-through view copies
    # that value onto the device.
    type _StoredValues = object


# The copy cache of every live read-through view, by the view's id.
_VIEW_COPIES: weakref.WeakValueDictionary[int, _RecentDeviceCopies] = (
    weakref.WeakValueDictionary()
)


def is_device_read_view(values: _StoredValues) -> bool:
    """Whether `values` is a read-through view of values kept off the device."""
    return id(values) in _VIEW_COPIES


def _device_stand_in(*, stored: jax.Array | _LazyEntry) -> jax.ShapeDtypeStruct:
    """Describe a stored value as the array a read would place on the device."""
    if isinstance(stored, jax.Array):
        return jax.ShapeDtypeStruct(
            stored.shape,
            stored.dtype,
            sharding=stored.sharding.with_memory_kind("device"),
        )
    # A disk entry records the shape, dtype and sharding it was solved on.
    shape, dtype, sharding = (
        cast("tuple[int, ...]", getattr(stored, "shape")),  # noqa: B009
        cast("np.dtype", getattr(stored, "dtype")),  # noqa: B009
        cast("jax.sharding.Sharding", getattr(stored, "sharding")),  # noqa: B009
    )
    return jax.ShapeDtypeStruct(
        shape, dtype, sharding=sharding.with_memory_kind("device")
    )


def planning_stand_in(
    values: _StoredValues,
) -> Mapping[int, Mapping[RegimeName, jax.Array | jax.ShapeDtypeStruct]]:
    """Return a read-through view as shapes, or any other values as they are.

    Planners need only the shape, dtype and placement of the values; reading a
    view for them would copy every value onto the device.
    """
    copies = _VIEW_COPIES.get(id(values))
    return values if copies is None else copies.stand_in


def device_copies_of_view(*, view_id: int) -> tuple[jax.Array, ...] | None:
    """Return the device copies the read-through view with `view_id` keeps.

    `None` when no live view has that id, so a caller can ask of any object.
    """
    copies = _VIEW_COPIES.get(view_id)
    if copies is None:
        return None
    return tuple(
        copy for regime_to_copy in copies.held() for copy in regime_to_copy.values()
    )
