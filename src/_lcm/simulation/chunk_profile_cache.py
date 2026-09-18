"""Runtime-owned bounded cache of abstract multi-stage chunk profiles.

A cache entry holds only the immutable `SimulationChunkProfile` returned by
`profile_simulation_chunk` (compiled executables, compiler reports, and byte
counts) never caller arrays, results, masks, probabilities, grids, or
closures over one call's arguments. A hit never skips admission: the caller
still runs the unchanged `_required_bytes` formula against the returned
profile and the call-local resident-bytes ledger.

The cache is bounded in aggregate across every `SimulationRuntime` a model
builds for its distinct `_simulate_runtime_regimes` shapes, not merely within
one runtime: entries carry each runtime's private `ProfileCacheToken`, and
eviction (LRU by entry count, then by an aggregate metadata-byte budget)
operates over one process-wide table.

Entries do not outlive the runtime that produced them. A cached profile holds
compiled executables, so the registry watches each token with a weak
reference and releases that token's entries as soon as the owning runtime
becomes unreachable, instead of holding executables until LRU eviction. The
release is deferred rather than performed inside the weak-reference callback,
which can run at an arbitrary point during garbage collection and must
therefore not take this registry's lock; every entry point drains the pending
releases first, so a dropped runtime's entries are gone before any lookup can
observe them.
"""

import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import cast

from _lcm.simulation.chunk_planning import SimulationChunkProfile

# Capacity policy: documented, fixed bounds shared by every model runtime in
# this process. Both bounds are enforced together; either one being exceeded
# evicts the least-recently-used entry first.
PROFILE_CACHE_MAX_ENTRIES = 64
PROFILE_CACHE_MAX_METADATA_BYTES = 33_554_432  # 32 MiB of abstract metadata


class ProfileCacheToken:
    """Weak-referenceable identity of one `SimulationRuntime`'s cached profiles.

    A bare marker carrying no state: never a caller array, a compiled
    executable, or a closure. It exists as its own class rather than a plain
    `object()` because the registry takes a weak reference to it, so that the
    runtime's profiles are released when the runtime itself is dropped.
    """

    __slots__ = ("__weakref__",)


def _estimate_metadata_bytes(*, profile: object) -> int:
    """Approximate this profile's own bookkeeping footprint, not device bytes.

    This sizes the cached Python/metadata object graph (stage count, device
    tuples, mapping entries) so the aggregate cache is bounded even when a
    heterogeneous population of runtimes each contributes many stages. It is
    deliberately not the device-resident byte totals the profile reserves;
    those remain part of the unchanged admission formula, not cache policy.
    Non-`SimulationChunkProfile` values (only ever exercised by this module's
    own unit tests, never by production dispatch) get a fixed minimal size.
    """
    if not isinstance(profile, SimulationChunkProfile):
        return 128
    per_stage = 256
    per_mapping_entry = 64
    return (
        128
        + per_stage * len(profile.stages)
        + per_mapping_entry
        * (len(profile.fixed_reservation) + len(profile.output_reservation))
    )


@dataclass(frozen=True, kw_only=True)
class _ProfileCacheEntry:
    profile: object
    metadata_bytes: int


class ChunkProfileCacheRegistry:
    """Process-wide, aggregate-bounded LRU shared by every `SimulationRuntime`.

    Keys are `(token_id, canonical_key)` pairs, where `token_id` is the
    identity of the owning runtime's `ProfileCacheToken` (never a caller
    array), so profiles built under different runtimes (different execution
    configs, devices, or JIT dispositions) never collide even if their
    canonical keys happen to coincide.

    Reusing a token's identity is safe because a dead token's entries are
    always released before the next lookup runs: the identity of a live token
    cannot equal that of a token which died before the live one was built.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[int, Hashable], _ProfileCacheEntry] = (
            OrderedDict()
        )
        self._metadata_bytes = 0
        self._in_flight: dict[tuple[int, Hashable], threading.Event] = {}
        self._watched_token_ids: set[int] = set()
        self._released_token_ids: list[int] = []

    def clear(self) -> None:
        """Drop every cached profile; used only by tests and explicit resets."""
        with self._lock:
            self._entries.clear()
            self._metadata_bytes = 0
            self._watched_token_ids.clear()
            self._released_token_ids.clear()

    def __len__(self) -> int:
        self._release_dropped_runtimes()
        with self._lock:
            return len(self._entries)

    @property
    def metadata_bytes(self) -> int:
        """Aggregate estimated metadata footprint of every live entry."""
        self._release_dropped_runtimes()
        with self._lock:
            return self._metadata_bytes

    def _release_dropped_runtimes(self) -> None:
        """Drop the entries of every runtime whose token has become unreachable.

        `list.pop` under the GIL is the only mutation a weak-reference
        callback performs, so this can take the registry lock safely while a
        collection is in progress.
        """
        pending: list[int] = []
        while True:
            try:
                pending.append(self._released_token_ids.pop())
            except IndexError:
                break
        if not pending:
            return
        dropped = set(pending)
        with self._lock:
            self._watched_token_ids -= dropped
            for full_key in [k for k in self._entries if k[0] in dropped]:
                self._metadata_bytes -= self._entries.pop(full_key).metadata_bytes

    def get_or_build[ProfileT](
        self,
        *,
        runtime_token: object,
        key: Hashable,
        build: Callable[[], ProfileT],
    ) -> ProfileT:
        """Resolve one immutable profile under single-flight, keyed by identity.

        Concurrent callers requesting the same runtime and `key` share one
        build and observe the same immutable profile object. A failed build
        publishes nothing at all, so a waiter never adopts a partial entry; it
        retries, which either finds a profile another caller published or
        makes the waiter the next builder and surfaces the failure to it.

        Entries are stored untyped because one registry serves every caller.
        A published entry was produced by a builder invoked under the same
        `(runtime_token, key)`, and the key carries the runtime's own identity
        plus the caller's canonical argument metadata, so a hit returns what
        this caller's own builder would have returned.
        """
        self._release_dropped_runtimes()
        full_key = (id(runtime_token), key)
        while True:
            with self._lock:
                cached = self._entries.get(full_key)
                if cached is not None:
                    self._entries.move_to_end(full_key)
                    return cast("ProfileT", cached.profile)
                event = self._in_flight.get(full_key)
                if event is None:
                    event = threading.Event()
                    self._in_flight[full_key] = event
                    owns_build = True
                else:
                    owns_build = False
            if not owns_build:
                event.wait()
                # Either the entry is now published, or the builder failed and
                # published nothing; retry resolves both.
                continue
            try:
                profile = build()
            except BaseException:
                with self._lock:
                    del self._in_flight[full_key]
                event.set()
                raise
            self._watch(runtime_token=runtime_token)
            with self._lock:
                self._publish(full_key=full_key, profile=profile)
                del self._in_flight[full_key]
            event.set()
            return profile

    def _watch(self, *, runtime_token: object) -> None:
        """Arrange for this token's entries to be released when it dies.

        A token that does not support weak references (only plain stand-in
        values in this module's own unit tests; every production token is a
        `ProfileCacheToken`) is simply left to the aggregate LRU bounds.
        """
        token_id = id(runtime_token)
        with self._lock:
            if token_id in self._watched_token_ids:
                return
            self._watched_token_ids.add(token_id)
        try:
            weakref.finalize(runtime_token, self._released_token_ids.append, token_id)
        except TypeError:
            with self._lock:
                self._watched_token_ids.discard(token_id)

    def _publish(self, *, full_key: tuple[int, Hashable], profile: object) -> None:
        """Insert one entry and evict LRU entries until both bounds hold."""
        metadata_bytes = _estimate_metadata_bytes(profile=profile)
        self._entries[full_key] = _ProfileCacheEntry(
            profile=profile, metadata_bytes=metadata_bytes
        )
        self._entries.move_to_end(full_key)
        self._metadata_bytes += metadata_bytes
        while self._entries and (
            len(self._entries) > PROFILE_CACHE_MAX_ENTRIES
            or self._metadata_bytes > PROFILE_CACHE_MAX_METADATA_BYTES
        ):
            _, evicted = self._entries.popitem(last=False)
            self._metadata_bytes -= evicted.metadata_bytes


_REGISTRY = ChunkProfileCacheRegistry()


def profile_cache_registry() -> ChunkProfileCacheRegistry:
    """Return the one process-wide registry shared by every model runtime."""
    return _REGISTRY
