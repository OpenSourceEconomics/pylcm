"""Physical buffer lifetime after the ledger closes a count, and the wave plan.

The ledger (`liveness.py`) is logical: it counts declared consumers per artifact
key. This module is physical: it knows which buffer an array occupies, which
keys share that buffer, when every output an asynchronous dispatch produced is
ready, and how to delete a buffer without leaving a deleted array inside the
rolling input mappings.

It also turns one period's nodes into a ready list: `plan_period_waves` groups
the period's dispatch units into waves such that a unit is ready only once
every same-period reference it reads has already dispatched, and units join
one wave only while their device sets stay pairwise disjoint — the boundary
that lets independent nodes dispatch back to back instead of one at a time.
"""

import dataclasses
import logging
import weakref
from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import MappingProxyType

import jax
from jaxtyping import PyTree

from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.value_transfer import ResolvedValueTransfer
from _lcm.typing import RegimeName
from lcm.exceptions import ExecutionPlanningError

_logger = logging.getLogger(__name__)

type BufferIdentity = tuple[tuple[int, int], ...]
type ShardIdentity = tuple[int, int]
# Weak references to the arrays that made one declaration, keeping none of them
# alive: the declaration holds while one of them is alive and undeleted.
type _DeclaringArrays = list[weakref.ref[jax.Array]]


def buffer_identity(*, array: jax.Array) -> BufferIdentity:
    """Return the device buffers an array occupies, as `(device id, pointer)` pairs.

    Two arrays with one identity share memory: releasing one releases the other.
    """
    if array.is_deleted():
        msg = "A deleted array has no buffer identity."
        raise ValueError(msg)
    return tuple(
        sorted(
            (shard.device.id, shard.data.unsafe_buffer_pointer())
            for shard in array.addressable_shards
        )
    )


def shard_identities(*, array: jax.Array) -> frozenset[ShardIdentity]:
    """Return the array's device buffers as a set, one entry per addressable shard.

    The membership form of `buffer_identity`. Sharing is a per-shard question:
    a `device_put` broadcasting a value onto a wider, replicated sharding can
    reuse the source's own buffer for one shard while allocating fresh buffers
    for the rest, so the two arrays' identities differ as whole tuples even
    though releasing one would still release a buffer the other still occupies.
    """
    return frozenset(buffer_identity(array=array))


def shares_a_buffer(*, first: jax.Array, second: jax.Array) -> bool:
    """Report whether any device buffer of `first` is also occupied by `second`."""
    return not shard_identities(array=first).isdisjoint(shard_identities(array=second))


class BufferRegistry:
    """Record which artifact keys name which device buffer.

    A buffer is registered under every key that reaches it, shard by shard; a
    release consults the registry so a buffer two keys share is deleted only
    when both keys may go. Forgetting a buffer drops every key on every shard
    of it, which is terminal: nothing names that buffer afterwards, so it is
    never donated or released again.

    Physical release is only ever of a buffer a compiled executable produced.
    A solver may hand an array it was given straight back out — an eager
    `jnp.broadcast_to` onto the shape an array already has returns that array —
    and freeing such a buffer destroys something the solve does not own. Three
    guards enforce the rule, and the caller installs all three:

    - an eager solve releases nothing at all, because an eager dispatch's
      outputs can be any object its inputs contained;
    - a compiled dispatch declares, through `declare_passed_through`, every
      output leaf whose buffer one of its inputs already occupied;
    - the arrays the model holds for its whole life are declared once through
      `declare_not_produced`, before the first dispatch.

    The second guard binds per dispatch unit, and a solve has more than one kind
    of them: for every unit — kernel, fold, terminal — `declare_passed_through`
    runs with that unit's own inputs and outputs before that unit's own release.
    A later unit's release is not held back for it, so a buffer is covered by
    the declaration of the unit that touched it, not by the ordering of the
    releases. `declare_not_produced` takes the rest: the model's arrays, and the
    payloads the result retains but has not yet copied, declared before the
    period's first release.

    The guards overlap deliberately: each is sound alone for the cases it sees,
    and none is trusted to see every case.

    All three answer per shard, never per whole array: an array is a buffer no
    dispatch produced as soon as ANY of its shards is one, because deleting it
    would free that shard along with the rest. A `device_put` onto a wider
    replicated sharding is the case that makes the difference — it keeps the
    source's own buffer for the shard the source already held and allocates the
    others, so the two arrays' whole-array identities differ while one buffer
    is common to both.

    A declaration and a registration hold for exactly as long as an array that
    made them is alive and undeleted. Both an identity and a key are recorded
    against weak references to those arrays, and both expire when the last of
    them goes; nothing here keeps an array, or the buffer behind it, alive.
    The rule is what makes an identity trustworthy: identities are device
    pointers, so one that outlived every array that declared it would name
    whatever allocation lands on that pointer next. `declared_shards` is the
    live declared set, pruned as it is read.

    The registry and the per-period transfer cache (`PeriodTransferCache`) are
    the two mutable engine-internal objects: every other execution-side
    structure the solve loop threads is immutable and replaced, never written
    into in place.
    """

    __slots__ = ("_keys_by_shard", "_unproduced_shards")

    def __init__(self) -> None:
        """Start with no registered buffer and no declared foreign buffer."""
        self._keys_by_shard: dict[ShardIdentity, dict[Hashable, _DeclaringArrays]] = {}
        self._unproduced_shards: dict[ShardIdentity, _DeclaringArrays] = {}

    def declare_not_produced(self, *, tree: PyTree) -> None:
        """Mark every array leaf of `tree` as a buffer no dispatch produced.

        Takes the arrays the model holds for its whole life — its materialized
        grids, its per-period state axes and its parameter vector — and the
        payloads the solve result retains. Every shard of every leaf is marked,
        so an array holding any one of them is covered too, and each leaf is
        kept as the weak owner of the shards it declares.

        Every declaration first drops the shards no live array declares any
        more, so a solve that declares once per period accumulates no entry for
        a buffer that is long gone.
        """
        self._prune_dead_declarations()
        for leaf in jax.tree.leaves(tree):
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
                for shard in shard_identities(array=leaf):
                    _add_declaring_array(
                        declaring=self._unproduced_shards.setdefault(shard, []),
                        array=leaf,
                    )

    def declare_passed_through(self, *, inputs: PyTree, outputs: PyTree) -> None:
        """Mark every output leaf whose buffer an input leaf already occupied.

        One dispatch's inputs and its outputs. An output holding any shard an
        input already held was passed through rather than computed there, so
        this dispatch did not produce it and no release may free it. The shards
        it shares are the ones marked; a shard the dispatch really did allocate
        stays its own.

        Both sides own the shards they share: the shard is a buffer no release
        may free while either the input that already held it or the output that
        handed it on is still alive.
        """
        self._prune_dead_declarations()
        holders: dict[ShardIdentity, list[jax.Array]] = {}
        for leaf in jax.tree.leaves(inputs):
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
                for shard in shard_identities(array=leaf):
                    holders.setdefault(shard, []).append(leaf)
        for leaf in jax.tree.leaves(outputs):
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
                for shard in shard_identities(array=leaf) & holders.keys():
                    declaring = self._unproduced_shards.setdefault(shard, [])
                    _add_declaring_array(declaring=declaring, array=leaf)
                    for holder in holders[shard]:
                        _add_declaring_array(declaring=declaring, array=holder)

    @property
    def declared_shards(self) -> frozenset[ShardIdentity]:
        """Return every shard identity a live declaring array still declares."""
        self._prune_dead_declarations()
        return frozenset(self._unproduced_shards)

    def is_not_produced(self, *, array: jax.Array) -> bool:
        """Report whether any shard of this array is a buffer no dispatch produced.

        A live declaration the array holds takes the array as one more of its
        weak owners. The declaration is a fact about a buffer, so it must last
        as long as any array that occupies it could reach a release — which is
        what makes the rule the registry's rather than each caller's.
        """
        answer = False
        for shard in shard_identities(array=array):
            declaring = self._unproduced_shards.get(shard)
            if declaring is None:
                continue
            if _keep_live_declaring_arrays(declaring=declaring):
                _add_declaring_array(declaring=declaring, array=array)
                answer = True
            else:
                del self._unproduced_shards[shard]
        return answer

    def register(self, *, array: jax.Array, artifact: Hashable) -> None:
        """Record that `artifact` names the buffer `array` occupies.

        Every shard the array holds records the key, because deleting the array
        frees all of them: a key on one shard is a key on the whole buffer.
        `array` is kept weakly, as the array the key names the buffer of.
        """
        for shard in shard_identities(array=array):
            keys = self._keys_by_shard.setdefault(shard, {})
            _add_declaring_array(declaring=keys.setdefault(artifact, []), array=array)

    def artifacts_sharing(self, *, array: jax.Array) -> frozenset[Hashable]:
        """Return every key whose registered array is alive on `array`'s buffer.

        Answered per shard, like every other question here: a `device_put` onto
        a wider replicated sharding, and an executable that reuses a donated
        input's buffer for one output shard, both leave two arrays whose
        whole-array identities differ while one buffer is common to both.
        Releasing either frees that buffer, so both keys are partners.
        """
        sharing: set[Hashable] = set()
        for shard in shard_identities(array=array):
            keys = self._keys_by_shard.get(shard)
            if keys is None:
                continue
            for artifact in tuple(keys):
                if _keep_live_declaring_arrays(declaring=keys[artifact]):
                    sharing.add(artifact)
                else:
                    del keys[artifact]
            if not keys:
                del self._keys_by_shard[shard]
        return frozenset(sharing)

    def forget(self, *, array: jax.Array) -> None:
        """Drop every key on every shard the buffer `array` occupies."""
        self.forget_identity(identity=buffer_identity(array=array))

    def forget_identity(self, *, identity: BufferIdentity) -> None:
        """Drop every key on the shards of the buffer with this identity.

        The form for a buffer that is already deleted — a donated input after
        its dispatch — and so cannot report its own identity any more.
        Forgetting is terminal for those keys: nothing names the buffer
        afterwards, so it is never donated or released again.
        """
        for shard in identity:
            self._keys_by_shard.pop(shard, None)

    def _prune_dead_declarations(self) -> None:
        """Drop every declared shard no live, undeleted array declares any more."""
        for shard in tuple(self._unproduced_shards):
            if not _keep_live_declaring_arrays(
                declaring=self._unproduced_shards[shard]
            ):
                del self._unproduced_shards[shard]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReleaseRecord:
    """One artifact key whose buffer was deleted, and the dispatch that closed it."""

    artifact: Hashable
    """The released artifact key."""

    closing_dispatch: Hashable
    """The dispatch whose commit closed the last count on the buffer."""


def release_closed_artifacts(
    *,
    ledger: PlannedInputLiveness,
    registry: BufferRegistry,
    artifacts: Iterable[Hashable],
    arrays_by_artifact: Mapping[Hashable, jax.Array],
    pending_outputs: Sequence[jax.Array],
    closing_dispatch: Hashable,
    logger: logging.Logger,
) -> tuple[ReleaseRecord, ...]:
    """Delete the buffers of closed artifacts once every pending output is ready.

    Each artifact must be release eligible in the ledger — a remaining consumer
    is an `ExecutionPlanningError`, never a warning. A buffer is deleted only when
    every key the registry holds on it is eligible too, so a leaf that is also a
    retained value survives. Arrays whose buffers overlap in any shard are one
    release: the array holding the group's whole shard set is deleted, and every
    key of the group is named on its record, so the shards freed are exactly the
    union of the eligible arrays' and each is freed once, whatever order the
    keys arrive in. A buffer no dispatch
    produced is never deleted, whatever the ledger says about the key that
    reached it. Nothing is deleted
    before one `block_until_ready` over `pending_outputs`, the outputs of every
    dispatch of the period so far, so an asynchronous computation never reads a
    freed buffer. Every deleted key is logged at debug level with the artifact
    key and the closing dispatch, and so is every key kept because no dispatch
    produced its buffer.
    """
    eligible: list[_ReleaseCandidate] = []
    for artifact in artifacts:
        if not ledger.is_release_eligible(artifact=artifact):
            msg = (
                f"Releasing {artifact!r} after dispatch {closing_dispatch!r} would "
                "drop a remaining consumer: it is still read, pinned or retained."
            )
            raise ExecutionPlanningError(msg)
        array = arrays_by_artifact[artifact]
        if array.is_deleted():
            continue
        if registry.is_not_produced(array=array):
            logger.debug(
                "kept %r after dispatch %r: no dispatch produced its buffer",
                artifact,
                closing_dispatch,
                extra={
                    "kept_artifact_key": artifact,
                    "closing_dispatch": closing_dispatch,
                },
            )
            continue
        partners = registry.artifacts_sharing(array=array) | {artifact}
        if not all(
            ledger.is_known(artifact=partner)
            and ledger.is_release_eligible(artifact=partner)
            for partner in partners
        ):
            continue
        eligible.append(
            _ReleaseCandidate(
                array=array,
                shards=shard_identities(array=array),
                keys=tuple(sorted(partners, key=repr)),
            )
        )
    to_delete = _one_delete_per_shared_buffer(candidates=eligible)
    if not to_delete:
        return ()
    jax.block_until_ready(tuple(pending_outputs))
    records: list[ReleaseRecord] = []
    for candidate in to_delete:
        array = candidate.array
        registry.forget(array=array)
        array.delete()
        for key in candidate.keys:
            logger.debug(
                "released %r after dispatch %r",
                key,
                closing_dispatch,
                extra={"artifact_key": key, "closing_dispatch": closing_dispatch},
            )
            records.append(
                ReleaseRecord(artifact=key, closing_dispatch=closing_dispatch)
            )
    return tuple(records)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ReleaseCandidate:
    """One array a release may delete, and the keys deleting it would free."""

    array: jax.Array
    """The array whose buffer the release would hand back."""

    shards: frozenset[ShardIdentity]
    """Every device buffer the array occupies."""

    keys: tuple[Hashable, ...]
    """Every registered key on those buffers, the candidate's own included."""


def _one_delete_per_shared_buffer(
    *, candidates: Sequence[_ReleaseCandidate]
) -> tuple[_ReleaseCandidate, ...]:
    """Pick, per group of buffer-sharing candidates, the one that frees them all.

    Deleting an array frees every shard it holds, so a group whose members
    overlap is freed by exactly one delete — and that one must hold the group's
    whole shard set, or the shards it lacks would stay allocated with no live
    array left to reach them, and the result would turn on the order the keys
    arrived in. A group no member covers is refused: no sequence of deletes
    frees such a group exactly once.
    """
    groups: list[tuple[set[ShardIdentity], list[_ReleaseCandidate]]] = []
    for candidate in candidates:
        shards = set(candidate.shards)
        members = [candidate]
        for group in [group for group in groups if group[0] & candidate.shards]:
            shards |= group[0]
            members.extend(group[1])
            groups.remove(group)
        groups.append((shards, members))
    chosen: list[_ReleaseCandidate] = []
    for shards, members in groups:
        covering = [member for member in members if member.shards == shards]
        if not covering:
            named = sorted({key for member in members for key in member.keys}, key=repr)
            msg = (
                "No single buffer of the release group covers every shard its "
                f"members occupy, so {named!r} cannot be freed exactly once."
            )
            raise ExecutionPlanningError(msg)
        chosen.append(covering[0])
    return tuple(chosen)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScheduledNode:
    """One program of one regime at one period: the unit of readiness."""

    period: int
    """The period the node solves."""

    regime: RegimeName
    """The regime whose kernel dispatches the program."""

    program: str
    """The program's graph key inside the kernel."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class DispatchUnit:
    """One period kernel's programs, dispatched as a whole in topological order."""

    period: int
    """The period the unit solves."""

    regime: RegimeName
    """The regime whose kernel is dispatched."""

    programs: tuple[str, ...]
    """The kernel's programs, producers before consumers."""


def plan_period_waves(
    *,
    nodes: Sequence[ScheduledNode],
    same_period_dependencies: Mapping[RegimeName, Sequence[RegimeName]],
    device_sets: Mapping[RegimeName, frozenset[int]],
) -> tuple[tuple[DispatchUnit, ...], ...]:
    """Group one period's nodes into waves of concurrently dispatchable units.

    The programs of one regime form one unit, in the order given, since a kernel
    dispatches its own programs. A unit is ready when every regime it reads at
    this period has been dispatched in an earlier wave. Ready units join one
    wave while their device sets are pairwise disjoint; a unit whose devices a
    wave already uses starts the next wave. Order within a wave, and among
    waves, follows the order of `nodes`, so declaration order breaks ties.
    """
    periods = {node.period for node in nodes}
    if len(periods) > 1:
        msg = (
            f"A wave plan covers one period; got nodes of periods {sorted(periods)!r}."
        )
        raise ValueError(msg)
    if not nodes:
        return ()
    period = next(iter(periods))
    programs_by_regime: dict[RegimeName, list[str]] = {}
    for node in nodes:
        programs_by_regime.setdefault(node.regime, []).append(node.program)
    units = {
        regime: DispatchUnit(period=period, regime=regime, programs=tuple(programs))
        for regime, programs in programs_by_regime.items()
    }
    remaining = list(units)
    dispatched: set[RegimeName] = set()
    waves: list[tuple[DispatchUnit, ...]] = []
    while remaining:
        wave: list[DispatchUnit] = []
        used_devices: set[int] = set()
        for regime in remaining:
            references = [
                reference
                for reference in same_period_dependencies.get(regime, ())
                if reference in units
            ]
            if any(reference not in dispatched for reference in references):
                continue
            if regime not in device_sets:
                msg = f"No device set was declared for regime {regime!r}."
                raise ExecutionPlanningError(msg)
            devices = device_sets[regime]
            if devices & used_devices:
                continue
            wave.append(units[regime])
            used_devices |= devices
        if not wave:
            msg = (
                "Same-period reads form a cycle among regimes "
                f"{tuple(remaining)!r}; no unit is ready."
            )
            raise ExecutionPlanningError(msg)
        dispatched.update(unit.regime for unit in wave)
        remaining = [regime for regime in remaining if regime not in dispatched]
        waves.append(tuple(wave))
    return tuple(waves)


def replace_leaf_by_identity(*, tree: object, old: object, new: object) -> object:
    """Return `tree` with the leaf that is `old` replaced by `new`.

    Identity, not equality, selects the leaf, so an equal-valued neighbour is
    untouched. The tree's structure is preserved, which keeps every compiled
    program's pytree calling convention intact. A tree without the leaf is
    refused: substituting nothing would hide a released buffer.
    """
    leaves, treedef = jax.tree.flatten(tree)
    if not any(leaf is old for leaf in leaves):
        msg = "The array to replace is not a leaf of the tree."
        raise ValueError(msg)
    return jax.tree.unflatten(
        treedef, [new if leaf is old else leaf for leaf in leaves]
    )


class PeriodTransferCache:
    """Copies of shared value transfers, held for one period and then dropped.

    Keyed by the artifact and the required layout, so several sources reading
    one stored value onto one layout share the one copy made for it. A copy
    that occupies no buffer of the stored artifact's is one the engine
    produced, not one the model owns: it is registered with the buffer
    registry under the exact key its declared consumers commit, and released
    through `release_closed_artifacts` — the same barrier and guards every
    other rolling solve input goes through — once every declared consumer has
    committed. Routing through that one implementation means a registered
    copy is deleted only after `pending_outputs` (the period's dispatched
    outputs so far) is confirmed ready, never while any of its shards is
    declared a buffer no dispatch produced, and every release, or every
    kept-buffer decision, is logged exactly like `release_closed_artifacts`'s
    other callers.

    A cache built with releasing off frees nothing at all, however many
    consumers commit: an eager dispatch is an ordinary Python call whose result
    may be any object its arguments contained, so no buffer it touched is known
    to be one the engine produced. That is the gate the solve's other release
    paths answer to as well.

    A copy that shares at least one buffer with the stored artifact — a
    `device_put` the required layout already matched, or one that reused the
    stored buffer for a shard of a wider replicated layout — is served from
    the cache but never registered or released here: releasing it would
    release a buffer the stored artifact still occupies. `shares_a_buffer`
    answers per shard, so a copy sharing even one shard with its source is
    treated as entirely unproduced; its other, genuinely fresh shards are
    freed only when the garbage collector reclaims the whole array, not by
    this cache.
    """

    __slots__ = (
        "_arrays",
        "_ledger",
        "_logger",
        "_next_dispatch_index",
        "_pending_outputs",
        "_registered_keys",
        "_registry",
        "_release_enabled",
    )

    def __init__(
        self,
        *,
        registry: BufferRegistry,
        consumer_counts: Mapping[tuple[Hashable, Hashable], int],
        pending_outputs: Sequence[jax.Array] = (),
        release_enabled: bool = True,
        logger: logging.Logger = _logger,
    ) -> None:
        """Start with no cached copy and the period's declared consumer counts.

        `pending_outputs` is read fresh at every `commit_consumer` call, so
        passing the same growing sequence the solve loop appends to keeps the
        release barrier current without threading it through every call.
        `release_enabled` is the solve's own release gate: with it false the
        cache counts consumers and refuses an over-commit as ever, and frees
        nothing.
        """
        # Keyed by Hashable, not the exact tuple shape, so the mapping widens
        # cleanly to `release_closed_artifacts`'s `Mapping[Hashable, jax.Array]`
        # parameter — `Mapping`'s key type parameter is invariant.
        self._arrays: dict[Hashable, jax.Array] = {}
        self._registry = registry
        self._pending_outputs = pending_outputs
        self._release_enabled = release_enabled
        self._logger = logger
        self._registered_keys: set[tuple[Hashable, Hashable]] = set()
        self._next_dispatch_index: dict[tuple[Hashable, Hashable], int] = dict.fromkeys(
            consumer_counts, 0
        )
        self._ledger: PlannedInputLiveness[
            tuple[tuple[Hashable, Hashable], int], tuple[Hashable, Hashable]
        ] = PlannedInputLiveness(
            dispatch_accesses={
                (key, index): (key,)
                for key, count in consumer_counts.items()
                for index in range(count)
            }
        )

    def get(self, *, transfer: ResolvedValueTransfer) -> jax.Array | None:
        """Return the copy made for the transfer's artifact and layout, if any."""
        return self._arrays.get((transfer.target, transfer.source_sharding))

    def put(
        self, *, transfer: ResolvedValueTransfer, array: jax.Array, stored: jax.Array
    ) -> None:
        """Record the copy made for the transfer's artifact and layout.

        Registers `array` with the buffer registry only when it occupies no
        buffer of `stored`'s: a `device_put` that reused a stored buffer,
        wholly or for one shard of a wider replicated layout, names no new
        buffer for this cache to release.
        """
        key = (transfer.target, transfer.source_sharding)
        self._arrays[key] = array
        if shares_a_buffer(first=array, second=stored):
            return
        if key not in self._next_dispatch_index:
            msg = (
                "A shared transfer copy was made for a key the period's "
                f"declared consumer count never named: {key!r}."
            )
            raise ExecutionPlanningError(msg)
        self._registry.register(array=array, artifact=key)
        self._registered_keys.add(key)

    def commit_consumer(
        self, *, key: tuple[Hashable, Hashable]
    ) -> tuple[ReleaseRecord, ...]:
        """Commit one of `key`'s declared consuming dispatches.

        A key this period declared no consumers for is a no-op, while one
        committed more often than the period declared consumers for it is an
        `ExecutionPlanningError` naming that key. Once every
        declared consumer has committed, a genuinely registered copy is
        released through `release_closed_artifacts` — blocked on this
        period's pending outputs, refused if any shard is declared not
        produced, logged like any other release. A copy that was never
        registered (served from a shared buffer), and every copy of a cache
        whose solve does not release, is never a release candidate, so it
        survives this call regardless of the count.
        """
        if key not in self._next_dispatch_index:
            return ()
        index = self._next_dispatch_index[key]
        self._next_dispatch_index[key] = index + 1
        dispatch = (key, index)
        try:
            newly_eligible = self._ledger.commit_successful_dispatch(dispatch=dispatch)
        except KeyError as error:
            msg = (
                "A shared transfer copy was committed by more dispatches than the "
                f"period's declared consumer count allows: {key!r}."
            )
            raise ExecutionPlanningError(msg) from error
        registered = newly_eligible & self._registered_keys
        if not registered or not self._release_enabled:
            return ()
        return release_closed_artifacts(
            ledger=self._ledger,
            registry=self._registry,
            artifacts=registered,
            arrays_by_artifact=MappingProxyType(dict(self._arrays)),
            pending_outputs=tuple(self._pending_outputs),
            closing_dispatch=dispatch,
            logger=self._logger,
        )

    def __len__(self) -> int:
        """Return the number of cached copies."""
        return len(self._arrays)


def _add_declaring_array(*, declaring: _DeclaringArrays, array: jax.Array) -> None:
    """Add `array` to the arrays a declaration is held by, weakly and once."""
    if not any(reference() is array for reference in declaring):
        declaring.append(weakref.ref(array))


def _keep_live_declaring_arrays(*, declaring: _DeclaringArrays) -> bool:
    """Drop the arrays that are gone or deleted; report whether one is left.

    A buffer identity is a device pointer, so an identity outliving every
    array that declared it would name whatever allocation lands on that
    pointer next. Dropping the reference here is what ends the declaration.
    """
    declaring[:] = [
        reference
        for reference in declaring
        if (array := reference()) is not None and not array.is_deleted()
    ]
    return bool(declaring)
