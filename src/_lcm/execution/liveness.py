"""Conservative remaining-consumer accounting for planned solve inputs.

The execution planner declares each dispatch's logical artifact accesses before a
solve starts.  The runtime commits those accesses only after the corresponding core
returns successfully.  Reaching zero on every key of a buffer's alias group makes an
unpinned, unretained buffer *eligible* for a later release decision; this module
never releases, donates, or offloads an array.
"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType


class PlannedInputLiveness[DispatchKey, ArtifactKey]:
    """Track the planned consumers, retention and aliases of solve-time artifacts.

    ``dispatch_accesses`` maps each immutable dispatch ID to its logical artifact
    accesses. An ID names one unit of work the runtime commits as a whole:

    - a ``(period, regime)`` core dispatch — one regime's compiled cores at one
      period;
    - a ``(period, source, target)`` gated-edge fold — the engine's own fold of one
      declared edge onto the target's grid at the period it folds.

    Three further facts qualify a count of zero:

    - ``pinned_artifacts`` have consumers no declaration names, so zero is not
      permission to release them;
    - ``retained_artifacts`` are kept on device by the solve result, so they are
      never released however their counts move;
    - ``aliases`` map an artifact key to the key of the same buffer one period
      later — a rolled entry whose producer does not run at its own period. The
      keys of one buffer form an alias group, and the group is release eligible
      only when every key in it has a count of zero, none is pinned and none is
      retained.

    An artifact is counted once per dispatch; duplicate declarations inside one
    dispatch are errors. A dispatch stays pending until that exact ID commits, so
    repeating a peer with the same access set cannot mask a skipped node.

    The ledger is separate from physical memory management. Call
    :meth:`commit_successful_dispatch` only after a core has returned
    successfully; a failed dispatch therefore has no liveness side effect. The
    scheduler that deletes buffers reads :meth:`is_release_eligible` and
    :meth:`has_sole_remaining_consumer`; this module never releases, donates or
    offloads an array.
    """

    __slots__ = (
        "_alias_of",
        "_dispatch_accesses",
        "_pending_dispatches",
        "_pinned_artifacts",
        "_remaining_by_artifact",
        "_retained_artifacts",
    )

    def __init__(
        self,
        *,
        dispatch_accesses: Mapping[DispatchKey, Iterable[ArtifactKey]],
        pinned_artifacts: Iterable[ArtifactKey] = (),
        retained_artifacts: Iterable[ArtifactKey] = (),
        aliases: Mapping[ArtifactKey, ArtifactKey] = MappingProxyType({}),
    ) -> None:
        """Build remaining-consumer counts from the declared dispatch accesses."""
        if not isinstance(dispatch_accesses, Mapping):
            msg = "dispatch_accesses must map immutable dispatch IDs to accesses."
            raise TypeError(msg)
        if not isinstance(aliases, Mapping):
            msg = "aliases must map an artifact key to the key it aliases."
            raise TypeError(msg)

        planned_dispatches: dict[DispatchKey, tuple[ArtifactKey, ...]] = {}

        remaining_by_artifact: dict[ArtifactKey, int] = {}
        for dispatch, accesses in dispatch_accesses.items():
            _require_hashable(value=dispatch, label="planned dispatch ID")
            snapshot = _snapshot_unique_hashable(
                values=accesses,
                label=f"planned dispatch {dispatch!r} accesses",
            )
            for artifact in snapshot:
                remaining_by_artifact[artifact] = (
                    remaining_by_artifact.get(artifact, 0) + 1
                )
            planned_dispatches[dispatch] = snapshot

        pinned = frozenset(
            _snapshot_unique_hashable(
                values=pinned_artifacts,
                label="pinned artifacts",
            )
        )
        retained = frozenset(
            _snapshot_unique_hashable(
                values=retained_artifacts,
                label="retained artifacts",
            )
        )
        for artifact in pinned | retained:
            remaining_by_artifact.setdefault(artifact, 0)

        alias_of: dict[ArtifactKey, ArtifactKey] = {}
        for artifact, target in aliases.items():
            _require_hashable(value=artifact, label="alias artifact")
            _require_hashable(value=target, label="alias target")
            if artifact == target:
                msg = f"An artifact cannot alias itself: {artifact!r}."
                raise ValueError(msg)
            remaining_by_artifact.setdefault(artifact, 0)
            remaining_by_artifact.setdefault(target, 0)
            alias_of[artifact] = target

        self._remaining_by_artifact = remaining_by_artifact
        self._pinned_artifacts = pinned
        self._retained_artifacts = retained
        self._alias_of = MappingProxyType(alias_of)
        self._dispatch_accesses = MappingProxyType(planned_dispatches)
        self._pending_dispatches = set(planned_dispatches)
        for artifact in alias_of:
            # A cycle would make every group query loop; reject it at construction.
            self.alias_group(artifact=artifact)

    @property
    def pending_dispatches(self) -> frozenset[DispatchKey]:
        """Return the exact dispatch IDs not yet committed."""
        return frozenset(self._pending_dispatches)

    @property
    def remaining_counts(self) -> Mapping[ArtifactKey, int]:
        """Return an immutable snapshot of the planned counts for inspection."""
        return MappingProxyType(dict(self._remaining_by_artifact))

    @property
    def retained_artifacts(self) -> frozenset[ArtifactKey]:
        """Return the artifacts the solve result keeps on device."""
        return self._retained_artifacts

    @property
    def aliases(self) -> Mapping[ArtifactKey, ArtifactKey]:
        """Return the declared alias map, artifact to the key one period later."""
        return self._alias_of

    def is_known(self, *, artifact: ArtifactKey) -> bool:
        """Report whether the artifact is part of the immutable logical plan."""
        _require_hashable(value=artifact, label="artifact")
        return artifact in self._remaining_by_artifact

    def remaining_consumers(self, *, artifact: ArtifactKey) -> int:
        """Return the finite planned consumers remaining for ``artifact``."""
        self._require_known(artifact=artifact)
        return self._remaining_by_artifact[artifact]

    def is_retained(self, *, artifact: ArtifactKey) -> bool:
        """Report whether the solve result keeps the artifact on device."""
        self._require_known(artifact=artifact)
        return artifact in self._retained_artifacts

    def is_pinned(self, *, artifact: ArtifactKey) -> bool:
        """Report whether consumers no declaration names keep the artifact."""
        self._require_known(artifact=artifact)
        return artifact in self._pinned_artifacts

    def alias_group(self, *, artifact: ArtifactKey) -> frozenset[ArtifactKey]:
        """Return every key that shares the artifact's buffer, itself included."""
        self._require_known(artifact=artifact)
        root = artifact
        seen: list[ArtifactKey] = [root]
        while root in self._alias_of:
            root = self._alias_of[root]
            if root in seen:
                msg = f"Alias declarations form a cycle through {root!r}."
                raise ValueError(msg)
            seen.append(root)
        members = {root}
        changed = True
        while changed:
            changed = False
            for source, target in self._alias_of.items():
                if target in members and source not in members:
                    members.add(source)
                    changed = True
        return frozenset(members)

    def is_release_eligible(self, *, artifact: ArtifactKey) -> bool:
        """Report whether the artifact's whole alias group may be released.

        ``True`` is permission for a separate scheduler to consider releasing the
        buffer: every key on it has zero planned consumers, none is pinned and
        none is retained. It is not a release operation.
        """
        return all(
            self._remaining_by_artifact[member] == 0
            and member not in self._pinned_artifacts
            and member not in self._retained_artifacts
            for member in self.alias_group(artifact=artifact)
        )

    def has_sole_remaining_consumer(
        self, *, artifact: ArtifactKey, dispatch: DispatchKey
    ) -> bool:
        """Report whether ``dispatch`` is the one planned consumer still pending.

        An artifact outside the immutable logical plan trivially has no planned
        consumer, so this answers ``False`` for it rather than raising — unlike
        the other per-artifact queries, which reject an unknown key.
        """
        _require_hashable(value=artifact, label="artifact")
        _require_hashable(value=dispatch, label="dispatch ID")
        if dispatch not in self._pending_dispatches:
            return False
        if artifact not in self._remaining_by_artifact:
            return False
        return (
            self._remaining_by_artifact[artifact] == 1
            and artifact in self._dispatch_accesses[dispatch]
        )

    def commit_successful_dispatch(
        self,
        *,
        dispatch: DispatchKey,
    ) -> frozenset[ArtifactKey]:
        """Atomically consume the exact identified dispatch's planned accesses.

        Unknown and already-committed IDs are rejected before any count changes. The
        return value names every key of every alias group that became release
        eligible at this commit; no physical release occurs here. The caller
        invokes this only after the named core dispatch returned successfully.
        """
        _require_hashable(value=dispatch, label="successful dispatch ID")
        if dispatch not in self._dispatch_accesses:
            msg = f"Successful dispatch named unknown planned ID: {dispatch!r}."
            raise KeyError(msg)
        if dispatch not in self._pending_dispatches:
            msg = f"Planned dispatch was already committed: {dispatch!r}."
            raise RuntimeError(msg)

        snapshot = self._dispatch_accesses[dispatch]

        exhausted = [
            artifact
            for artifact in snapshot
            if self._remaining_by_artifact[artifact] == 0
        ]
        if exhausted:
            msg = (
                "Successful dispatch would underflow remaining-consumer counts for "
                f"artifacts: {exhausted!r}."
            )
            raise RuntimeError(msg)

        for artifact in snapshot:
            self._remaining_by_artifact[artifact] -= 1
        self._pending_dispatches.remove(dispatch)
        newly_eligible: set[ArtifactKey] = set()
        for artifact in snapshot:
            if self._remaining_by_artifact[artifact] == 0 and (
                self.is_release_eligible(artifact=artifact)
            ):
                newly_eligible.update(self.alias_group(artifact=artifact))
        return frozenset(newly_eligible)

    def assert_solve_complete(self) -> None:
        """Reject a successful solve that skipped any exact planned dispatch."""
        if self._pending_dispatches:
            pending = tuple(sorted(self._pending_dispatches, key=repr))
            msg = f"Successful solve left planned dispatches uncommitted: {pending!r}."
            raise RuntimeError(msg)

        unfinished = {
            artifact: remaining
            for artifact, remaining in self._remaining_by_artifact.items()
            if remaining != 0
        }
        if unfinished:
            msg = (
                "Every planned dispatch committed, but input counts remain: "
                f"{unfinished!r}."
            )
            raise RuntimeError(msg)

    def _require_known(self, *, artifact: ArtifactKey) -> None:
        """Reject queries for artifacts outside the immutable logical plan."""
        _require_hashable(value=artifact, label="artifact")
        if artifact not in self._remaining_by_artifact:
            msg = f"Unknown planned input artifact: {artifact!r}."
            raise KeyError(msg)


def _snapshot_unique_hashable[Value](
    *,
    values: Iterable[Value],
    label: str,
) -> tuple[Value, ...]:
    """Snapshot one declaration while rejecting unhashable or duplicate keys."""
    snapshot: list[Value] = []
    seen: set[Value] = set()
    for value in values:
        _require_hashable(value=value, label=label)
        if value in seen:
            msg = f"{label.capitalize()} contain duplicate artifact {value!r}."
            raise ValueError(msg)
        seen.add(value)
        snapshot.append(value)
    return tuple(snapshot)


def _require_hashable(*, value: object, label: str) -> None:
    """Give invalid logical artifact keys a local, actionable error."""
    try:
        hash(value)
    except TypeError as error:
        msg = f"{label.capitalize()} must use hashable logical artifact keys."
        raise TypeError(msg) from error
