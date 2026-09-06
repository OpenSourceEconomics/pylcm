"""Physical lifetime of solve-time buffers after the ledger says a count closed.

The ledger (`liveness.py`) is logical: it counts declared consumers per artifact
key. This module is physical: it knows which buffer an array occupies, which
keys share that buffer, when every output an asynchronous dispatch produced is
ready, and how to delete a buffer without leaving a deleted array inside the
rolling input mappings.
"""

import dataclasses
import logging
from collections.abc import Hashable, Iterable, Mapping, Sequence

import jax
from jaxtyping import PyTree

from _lcm.execution.liveness import PlannedInputLiveness
from lcm.exceptions import ExecutionPlanningError

type BufferIdentity = tuple[tuple[int, int], ...]


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


class BufferRegistry:
    """Record which artifact keys name which device buffer.

    A buffer is registered under every key that reaches it; a release consults
    the registry so a buffer two keys share is deleted only when both keys may
    go. Forgetting a buffer drops every key on it.

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
    """

    __slots__ = ("_keys_by_buffer", "_unproduced_buffers")

    def __init__(self) -> None:
        """Start with no registered buffer and no declared foreign buffer."""
        self._keys_by_buffer: dict[BufferIdentity, set[Hashable]] = {}
        self._unproduced_buffers: set[BufferIdentity] = set()

    def declare_not_produced(self, *, tree: PyTree) -> None:
        """Mark every array leaf of `tree` as a buffer no dispatch produced.

        Takes the arrays the model holds for its whole life — its materialized
        grids, its per-period state axes and its parameter vector — and the
        payloads the solve result retains.
        """
        for leaf in jax.tree.leaves(tree):
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
                self._unproduced_buffers.add(buffer_identity(array=leaf))

    def declare_passed_through(self, *, inputs: PyTree, outputs: PyTree) -> None:
        """Mark every output leaf whose buffer an input leaf already occupied.

        One dispatch's inputs and its outputs. An output on an input's buffer
        was passed through rather than computed, so this dispatch did not
        produce it and no release may free it.
        """
        occupied = {
            buffer_identity(array=leaf)
            for leaf in jax.tree.leaves(inputs)
            if isinstance(leaf, jax.Array) and not leaf.is_deleted()
        }
        for leaf in jax.tree.leaves(outputs):
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
                identity = buffer_identity(array=leaf)
                if identity in occupied:
                    self._unproduced_buffers.add(identity)

    def is_not_produced(self, *, array: jax.Array) -> bool:
        """Report whether no dispatch produced this array's buffer."""
        return buffer_identity(array=array) in self._unproduced_buffers

    def register(self, *, array: jax.Array, artifact: Hashable) -> None:
        """Record that `artifact` names the buffer `array` occupies."""
        self._keys_by_buffer.setdefault(buffer_identity(array=array), set()).add(
            artifact
        )

    def artifacts_sharing(self, *, array: jax.Array) -> frozenset[Hashable]:
        """Return every key registered on the buffer `array` occupies."""
        return frozenset(self._keys_by_buffer.get(buffer_identity(array=array), ()))

    def forget(self, *, array: jax.Array) -> None:
        """Drop every key on the buffer `array` occupies."""
        self.forget_identity(identity=buffer_identity(array=array))

    def forget_identity(self, *, identity: BufferIdentity) -> None:
        """Drop every key on the buffer with this identity.

        The form for a buffer that is already deleted — a donated input after
        its dispatch — and so cannot report its own identity any more.
        """
        self._keys_by_buffer.pop(identity, None)


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
    retained value survives. A buffer no dispatch produced is never deleted,
    whatever the ledger says about the key that reached it. Nothing is deleted
    before one `block_until_ready` over `pending_outputs`, the outputs of every
    dispatch of the period so far, so an asynchronous computation never reads a
    freed buffer. Every deleted key is logged at debug level with the artifact
    key and the closing dispatch, and so is every key kept because no dispatch
    produced its buffer.
    """
    to_delete: dict[BufferIdentity, tuple[jax.Array, tuple[Hashable, ...]]] = {}
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
        to_delete[buffer_identity(array=array)] = (
            array,
            tuple(sorted(partners, key=repr)),
        )
    if not to_delete:
        return ()
    jax.block_until_ready(tuple(pending_outputs))
    records: list[ReleaseRecord] = []
    for array, keys in to_delete.values():
        registry.forget(array=array)
        array.delete()
        for key in keys:
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
class ScheduledNode:
    """One program of one regime at one period: the unit of readiness."""

    period: int
    """The period the node solves."""

    regime: str
    """The regime whose kernel dispatches the program."""

    program: str
    """The program's graph key inside the kernel."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class DispatchUnit:
    """One period kernel's programs, dispatched as a whole in topological order."""

    period: int
    """The period the unit solves."""

    regime: str
    """The regime whose kernel is dispatched."""

    programs: tuple[str, ...]
    """The kernel's programs, producers before consumers."""


def plan_period_waves(
    *,
    nodes: Sequence[ScheduledNode],
    same_period_dependencies: Mapping[str, Sequence[str]],
    device_sets: Mapping[str, frozenset[int]],
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
    programs_by_regime: dict[str, list[str]] = {}
    for node in nodes:
        programs_by_regime.setdefault(node.regime, []).append(node.program)
    period = next(iter(periods)) if periods else 0
    units = {
        regime: DispatchUnit(period=period, regime=regime, programs=tuple(programs))
        for regime, programs in programs_by_regime.items()
    }
    remaining = list(units)
    dispatched: set[str] = set()
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
