"""Solved values kept in pinned host memory, read back onto the device.

Under `ExecutionConfig.value_storage="host"` a solve keeps each period's values in
pinned host memory. Nothing outside the solve computes on those arrays: arithmetic
on a host-memory array runs as slow host compute and leaves its result in host
memory. Every reader instead receives a device copy, so results, reads and panels
are those of device storage.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import jax
import numpy as np

from _lcm.typing import ArtifactPayload
from lcm._solver_api.entries import _LazyEntry
from lcm._solver_api.identity import LoadState
from lcm.typing import FloatND, RegimeName

# The memory kind a stored value lives in.
HOST_MEMORY_KIND: Literal["pinned_host"] = "pinned_host"

# Periods whose device copies a read-through view keeps at once: the period a
# simulation step reads and the period its subjects land in.
_CACHED_PERIODS = 2


def copy_values_to_host(
    *, values: Mapping[RegimeName, FloatND]
) -> MappingProxyType[RegimeName, FloatND]:
    """Start copying one period's values to pinned host memory, without waiting.

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


def is_host_value(value: FloatND) -> bool:
    """Whether `value` is an array a solve stored in pinned host memory."""
    return value.sharding.memory_kind == HOST_MEMORY_KIND


def to_device(value: FloatND) -> FloatND:
    """Copy a stored value back to device memory on its own sharding."""
    return jax.device_put(value, value.sharding.with_memory_kind("device"))


@dataclass(frozen=True, kw_only=True, eq=False)
class HostValueEntry(_LazyEntry):
    """One stored value in pinned host memory, copied to the device when read."""

    value: FloatND
    """The stored value, in pinned host memory."""

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


class _RecentDeviceCopies:
    """Device copies of the stored values of the most recently read periods."""

    def __init__(self) -> None:
        self._copies: dict[int, dict[RegimeName, FloatND]] = {}

    def read(self, *, period: int, regime: RegimeName, stored: FloatND) -> FloatND:
        """Return the device copy of one stored value, copying it on first read."""
        if period not in self._copies:
            while len(self._copies) >= _CACHED_PERIODS:
                del self._copies[next(iter(self._copies))]
            self._copies[period] = {}
        copies = self._copies[period]
        if regime not in copies:
            copies[regime] = to_device(stored)
        return copies[regime]


class _DeviceReadPeriod(Mapping[RegimeName, FloatND]):
    """One period's stored values, each handed out as a device copy."""

    def __init__(
        self,
        *,
        period: int,
        stored: Mapping[RegimeName, FloatND],
        copies: _RecentDeviceCopies,
    ) -> None:
        self._period = period
        self._stored = stored
        self._copies = copies

    def __getitem__(self, regime: RegimeName) -> FloatND:
        return self._copies.read(
            period=self._period, regime=regime, stored=self._stored[regime]
        )

    def __iter__(self) -> Iterator[RegimeName]:
        return iter(self._stored)

    def __len__(self) -> int:
        return len(self._stored)


def device_read_view(
    *, values: Mapping[int, Mapping[RegimeName, FloatND]]
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Return host-stored values as a mapping that hands out device copies.

    A value is copied when it is first read and kept while its period is one of
    the two most recently read, so a reader that walks the periods in order holds
    at most two periods on the device and copies each period once per walk.
    """
    copies = _RecentDeviceCopies()
    return MappingProxyType(
        {
            period: MappingProxyType(
                _DeviceReadPeriod(period=period, stored=stored, copies=copies)
            )
            for period, stored in values.items()
        }
    )
