"""Conservative remaining-consumer accounting for planned solve inputs.

The execution planner declares each dispatch's logical artifact accesses before a
solve starts.  The runtime commits those accesses only after the corresponding core
returns successfully.  Reaching zero makes an unpinned artifact *eligible* for a
later release decision; this module never releases, donates, or offloads an array.
"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType


class PlannedInputLiveness[DispatchKey, ArtifactKey]:
    """Track the finite planned consumers of logical solve-time artifacts.

    ``dispatch_accesses`` maps each immutable dispatch ID to its logical artifact
    accesses. An artifact is counted once per dispatch; duplicate declarations inside
    one dispatch are errors. A dispatch stays pending until that exact ID commits, so
    repeating a peer with the same access set cannot mask a skipped node. Artifacts
    with dense or otherwise unplanned consumers must also be listed in
    ``pinned_artifacts``. Pinning preserves finite planned counts while preventing zero
    from becoming release eligibility.

    The mutable ledger is intentionally separate from physical memory management.
    Call :meth:`commit_successful_dispatch` only after a core has returned
    successfully.  A failed dispatch therefore has no liveness side effect.
    """

    __slots__ = (
        "_dispatch_accesses",
        "_pending_dispatches",
        "_pinned_artifacts",
        "_remaining_by_artifact",
    )

    def __init__(
        self,
        *,
        dispatch_accesses: Mapping[DispatchKey, Iterable[ArtifactKey]],
        pinned_artifacts: Iterable[ArtifactKey] = (),
    ) -> None:
        """Build remaining-consumer counts from the declared dispatch accesses."""
        if not isinstance(dispatch_accesses, Mapping):
            msg = "dispatch_accesses must map immutable dispatch IDs to accesses."
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
        for artifact in pinned:
            remaining_by_artifact.setdefault(artifact, 0)

        self._remaining_by_artifact = remaining_by_artifact
        self._pinned_artifacts = pinned
        self._dispatch_accesses = MappingProxyType(planned_dispatches)
        self._pending_dispatches = set(planned_dispatches)

    @property
    def pending_dispatches(self) -> frozenset[DispatchKey]:
        """Return the exact dispatch IDs not yet committed."""
        return frozenset(self._pending_dispatches)

    @property
    def remaining_counts(self) -> Mapping[ArtifactKey, int]:
        """Return an immutable snapshot of the planned counts for inspection."""
        return MappingProxyType(dict(self._remaining_by_artifact))

    def remaining_consumers(self, *, artifact: ArtifactKey) -> int:
        """Return the finite planned consumers remaining for ``artifact``."""
        self._require_known(artifact=artifact)
        return self._remaining_by_artifact[artifact]

    def is_release_eligible(self, *, artifact: ArtifactKey) -> bool:
        """Report zero planned consumers for an artifact without an unplanned pin.

        ``True`` is permission for a separate scheduler to consider releasing the
        artifact.  It is not a release operation and says nothing about result
        retention or compatible donation.
        """
        self._require_known(artifact=artifact)
        return (
            self._remaining_by_artifact[artifact] == 0
            and artifact not in self._pinned_artifacts
        )

    def commit_successful_dispatch(
        self,
        *,
        dispatch: DispatchKey,
    ) -> frozenset[ArtifactKey]:
        """Atomically consume the exact identified dispatch's planned accesses.

        Unknown and already-committed IDs are rejected before any count changes. The
        return value names artifacts that became release eligible at this commit; no
        physical release occurs here. The caller invokes this only after the named
        core dispatch returned successfully.
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

        newly_eligible: set[ArtifactKey] = set()
        for artifact in snapshot:
            remaining = self._remaining_by_artifact[artifact] - 1
            self._remaining_by_artifact[artifact] = remaining
            if remaining == 0 and artifact not in self._pinned_artifacts:
                newly_eligible.add(artifact)
        self._pending_dispatches.remove(dispatch)
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
