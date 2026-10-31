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
from _lcm.typing import ArtifactPayload, SimulationPolicy
from lcm._solver_api.entries import _LazyEntry
from lcm._solver_api.identity import LoadState
from lcm.exceptions import ExecutionPlanningError, ValueStorageError
from lcm.typing import FloatND, RegimeName

# The memory kind a stored value lives in.
HOST_MEMORY_KIND: Literal["pinned_host"] = "pinned_host"

# Periods whose device copies a read-through view keeps at once: a simulation
# step reads its own period, the one its subjects land in and, replaying a
# same-period exit, the one after that; the next period's reads, copied early
# while this period runs, add one more.
_CACHED_PERIODS = 4

# What a read-through view reads from, hands out, and describes to planners: a
# value array or a simulation policy, kept in host memory or as a lazy entry.
type _Stored = jax.Array | SimulationPolicy | _LazyEntry
type _OnDevice = jax.Array | SimulationPolicy
type _StandIn = jax.ShapeDtypeStruct | SimulationPolicy


# Where a host copy goes, as a failed copy names it.
_HOST_DESTINATION = "pinned host memory"


def copy_values_to_host(
    *, period: int, values: Mapping[RegimeName, jax.Array]
) -> MappingProxyType[RegimeName, jax.Array]:
    """Start copying one period's arrays to pinned host memory, without waiting.

    Each copy keeps its array's sharding, so every device copies its own slice,
    and runs while the device computes whatever is dispatched after it.

    Raises:
        ValueStorageError: A copy could not be started, naming its period and
            regime.

    """
    copies: dict[RegimeName, jax.Array] = {}
    for regime_name, value in values.items():
        try:
            copies[regime_name] = _to_pinned_host(value)
        except RuntimeError as error:
            raise value_storage_error(
                action="copy",
                period=period,
                regime=regime_name,
                destination=_HOST_DESTINATION,
            ) from error
    return MappingProxyType(copies)


def copy_policies_to_host(
    *, period: int, policies: Mapping[RegimeName, SimulationPolicy]
) -> MappingProxyType[RegimeName, SimulationPolicy]:
    """Start copying one period's simulation policies to pinned host memory.

    Each array leaf travels as a value does, on its own sharding and without
    waiting; the policy's structure and any non-array leaf are kept.

    Raises:
        ValueStorageError: A copy could not be started, naming its period and
            regime.

    """
    copies: dict[RegimeName, SimulationPolicy] = {}
    for regime_name, policy in policies.items():
        try:
            copies[regime_name] = jax.tree.map(
                lambda leaf: (
                    _to_pinned_host(leaf) if isinstance(leaf, jax.Array) else leaf
                ),
                policy,
            )
        except RuntimeError as error:
            raise value_storage_error(
                action="copy",
                period=period,
                regime=regime_name,
                destination=_HOST_DESTINATION,
            ) from error
    return MappingProxyType(copies)


def _to_pinned_host(value: jax.Array) -> jax.Array:
    """Dispatch one array's copy to pinned host memory on its own sharding."""
    return jax.device_put(value, value.sharding.with_memory_kind(HOST_MEMORY_KIND))


def wait_for_host_copies(
    *, copies: tuple[tuple[int, RegimeName, jax.Array], ...]
) -> None:
    """Wait until every `(period, regime, copy)` has arrived in pinned host memory.

    Raises:
        ValueStorageError: A copy failed, naming its period and regime.

    """
    for period, regime_name, copy in copies:
        try:
            jax.block_until_ready(copy)
        except RuntimeError as error:
            raise value_storage_error(
                action="copy",
                period=period,
                regime=regime_name,
                destination=_HOST_DESTINATION,
            ) from error


def value_storage_error(
    *,
    action: Literal["copy", "write"],
    period: int,
    regime: RegimeName,
    destination: str,
    code: int | None = None,
) -> ValueStorageError:
    """Return the error for a value that did not reach its value storage."""
    of_code = "" if code is None else f"code {code} of "
    return ValueStorageError(
        f"Could not {action} {of_code}period {period}, regime {regime!r} to "
        f"{destination}."
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
        stand_in: MappingProxyType[int, MappingProxyType[RegimeName, _StandIn]],
        stored: Mapping[int, Mapping[RegimeName, _Stored]],
    ) -> None:
        self._copies: dict[int, dict[RegimeName, _OnDevice]] = {}
        self.stand_in = stand_in
        """The view's values as shapes on their device placement, for planners."""
        self.stored = stored
        """What the view reads from: host-memory arrays or lazy entries."""

    def held(self) -> tuple[dict[RegimeName, _OnDevice], ...]:
        """Return the device copies kept, one mapping per cached period."""
        return tuple(self._copies.values())

    def read(self, *, period: int, regime: RegimeName, stored: _Stored) -> _OnDevice:
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
                cast("_OnDevice", stored.materialize())
                if isinstance(stored, _LazyEntry)
                else jax.tree.map(
                    lambda leaf: (
                        to_device(leaf) if isinstance(leaf, jax.Array) else leaf
                    ),
                    stored,
                )
            )
        return copies[regime]


class _DeviceReadPeriod(Mapping[RegimeName, _OnDevice]):
    """One period's stored arrays or policies, each handed out as a device copy."""

    def __init__(
        self,
        *,
        period: int,
        stored: Mapping[RegimeName, _Stored],
        copies: _RecentDeviceCopies,
    ) -> None:
        self._period = period
        self._stored = stored
        self._copies = copies

    def __getitem__(self, regime: RegimeName) -> _OnDevice:
        return self._copies.read(
            period=self._period, regime=regime, stored=self._stored[regime]
        )

    def __contains__(self, regime: object) -> bool:
        # Asking whether a regime is stored reads nothing: the mixin's default
        # would look the value up, copying it onto the device.
        return regime in self._stored

    def __iter__(self) -> Iterator[RegimeName]:
        return iter(self._stored)

    def __len__(self) -> int:
        return len(self._stored)


def policy_read_view(
    *, policies: Mapping[int, Mapping[RegimeName, SimulationPolicy | _LazyEntry]]
) -> MappingProxyType[int, MappingProxyType[RegimeName, SimulationPolicy]]:
    """Return stored simulation policies as a mapping that hands out device copies.

    It caches as `device_read_view` does, so the chunks of a period share one
    copy of each policy they read.
    """
    return cast(
        "MappingProxyType[int, MappingProxyType[RegimeName, SimulationPolicy]]",
        device_read_view(values=policies),
    )


def stored_form[T](
    values: Mapping[int, Mapping[RegimeName, T]],
) -> Mapping[int, Mapping[RegimeName, T]]:
    """Return a read-through view's stored contents for a host-side check.

    Host-memory arrays come as they are; a lazy entry is read onto the CPU device
    (`host_payload`). Anything that is not a view comes back as it is.
    """
    copies = _VIEW_COPIES.get(id(values))
    if copies is None:
        return values
    return MappingProxyType(
        {
            period: MappingProxyType(
                {
                    regime: cast(
                        "T",
                        getattr(stored, "host_payload")()  # noqa: B009
                        if isinstance(stored, _LazyEntry)
                        else stored,
                    )
                    for regime, stored in regime_to_stored.items()
                }
            )
            for period, regime_to_stored in copies.stored.items()
        }
    )


def device_read_view(
    *, values: Mapping[int, Mapping[RegimeName, _Stored]]
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
        ),
        stored=values,
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
    # A view of values hands out arrays; `policy_read_view` recasts its own.
    return cast("MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]", view)


class _LazyReadMapping[K, V](Mapping[K, V]):
    """A read-only mapping that reads each lazy entry when it is looked up."""

    def __init__(self, entries: Mapping[K, V | _LazyEntry]) -> None:
        self.entries: MappingProxyType[K, V | _LazyEntry] = MappingProxyType(
            dict(entries)
        )

    def __getitem__(self, key: K) -> V:
        entry = self.entries[key]
        return (
            cast("V", entry.materialize()) if isinstance(entry, _LazyEntry) else entry
        )

    def __contains__(self, key: object) -> bool:
        # Membership reads nothing; the mixin's default would read the entry.
        return key in self.entries

    def __iter__(self) -> Iterator[K]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def lazy_read_view[K, V](
    *, entries: Mapping[K, V | _LazyEntry]
) -> MappingProxyType[K, V]:
    """Return a mapping that reads each lazy entry, such as a policy on disk, on lookup.

    Nothing read is kept: each lookup reads the entry again, so the view holds
    no device memory.
    """
    lazy = _LazyReadMapping[K, V](entries)
    view = MappingProxyType(lazy)
    # Only the view reaches `lazy`, so this entry goes with the view.
    _LAZY_VIEWS[id(view)] = lazy
    return view


def lazy_entries_of_view(
    *, view_id: int
) -> tuple[ArtifactPayload | _LazyEntry, ...] | None:
    """The stored entries behind a lazy read view, by the view's id; else `None`."""
    lazy = _LAZY_VIEWS.get(view_id)
    return None if lazy is None else tuple(lazy.entries.values())


if TYPE_CHECKING:
    type _StoredValues = Mapping[int, Mapping[RegimeName, jax.Array]]
else:
    # A runtime check samples one entry, and sampling a read-through view copies
    # that value onto the device.
    type _StoredValues = object


# The mapping behind every live lazy read view, by the view's id.
_LAZY_VIEWS: weakref.WeakValueDictionary[int, _LazyReadMapping] = (
    weakref.WeakValueDictionary()
)

# The copy cache of every live read-through view, by the view's id.
_VIEW_COPIES: weakref.WeakValueDictionary[int, _RecentDeviceCopies] = (
    weakref.WeakValueDictionary()
)


def is_device_read_view(values: _StoredValues) -> bool:
    """Whether `values` is a read-through view of values kept off the device."""
    return id(values) in _VIEW_COPIES


def _device_stand_in(*, stored: _Stored) -> _StandIn:
    """Describe a stored value as what a read would place on the device."""
    if isinstance(stored, _LazyEntry):
        # A disk entry records the shapes, dtypes and shardings it was solved on.
        return cast("_StandIn", getattr(stored, "device_stand_in")())  # noqa: B009
    return jax.tree.map(
        lambda leaf: (
            jax.ShapeDtypeStruct(
                leaf.shape,
                leaf.dtype,
                sharding=leaf.sharding.with_memory_kind("device"),
            )
            if isinstance(leaf, jax.Array)
            else leaf
        ),
        stored,
    )


def planning_stand_in[T](values: T) -> T:
    """Return a read-through view as shapes, or any other values as they are.

    Planners need only the shape, dtype and placement of stored values or
    policies; reading a view for them would copy each onto the device. A stand-in
    keeps a policy's structure with a `jax.ShapeDtypeStruct` for each array.
    """
    copies = _VIEW_COPIES.get(id(values))
    return values if copies is None else cast("T", copies.stand_in)


def device_copies_of_view(*, view_id: int) -> tuple[jax.Array, ...] | None:
    """Return the device copies the read-through view with `view_id` keeps.

    `None` when no live view has that id, so a caller can ask of any object.
    """
    copies = _VIEW_COPIES.get(view_id)
    if copies is None:
        return None
    return tuple(
        leaf
        for regime_to_copy in copies.held()
        for copy in regime_to_copy.values()
        for leaf in jax.tree.leaves(copy)
        if isinstance(leaf, jax.Array)
    )
