"""Certify joint coordinate reads using non-executing primitive analysis."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

type CoordinateName = str
type FibreId = str
type RelationId = str


class PrimitiveReads(Protocol):
    """Conservatively read original primitives with only supplied statuses fixed."""

    def __call__(
        self, *, fixed_statuses: Mapping[CoordinateName, int]
    ) -> frozenset[CoordinateName]: ...


@dataclass(frozen=True)
class JointReadSet:
    """Record a jointly certified coordinate set and its deterministic selection."""

    live_coordinates: frozenset[CoordinateName]
    """Coordinates sufficient to determine all primitive values at the fibre."""
    removal_order: tuple[CoordinateName, ...]
    """Declared status order used when attempting coordinate removal."""
    removed_statuses: tuple[CoordinateName, ...]
    """Statuses removed by successful joint checks, in selection order."""


@dataclass(frozen=True)
class CoordinateRead:
    """Record the source coordinates read by one successor-coordinate update."""

    coordinate: CoordinateName
    """Updated coordinate in the target fibre."""
    reads: JointReadSet
    """Joint read certificate fixed on the original source fibre."""


@dataclass(frozen=True)
class FibreLink:
    """Connect a source to one possible transition or same-period value read."""

    target: FibreId
    """Target fibre, whose live coordinates determine which updates are needed."""
    updates: tuple[CoordinateRead, ...]
    """Explicit updates for every target coordinate, including constant updates."""
    phase: Literal["solve", "simulation", "both"] = "both"
    """Original primitive channel; every channel contributes to the joint closure."""
    source_relations: tuple[RelationId, ...] = ()
    """Original incidences whose producers define this source-target connection."""


@dataclass(frozen=True)
class FibreReadFacts:
    """Hold fixed primitive read certificates before any layout is grouped."""

    identifier: FibreId
    """Unique identifier of the original fibre."""
    coordinates: frozenset[CoordinateName]
    """Original source-state coordinates; actions and runtime parameters excluded."""
    direct: JointReadSet
    """Joint reads of payoff, feasibility, outputs and retained law weights."""
    links: tuple[FibreLink, ...]
    """All possible transition and same-period value-read dependencies."""


@dataclass(frozen=True)
class FibreLiveness:
    """Record the least live set relative to the supplied primitive certificates."""

    identifier: FibreId
    """Original fibre to which the live set belongs."""
    live_coordinates: frozenset[CoordinateName]
    """Least fixed point of direct reads and successor-live update reads."""


def get_joint_read_set(
    *,
    coordinates: frozenset[CoordinateName],
    statuses: Mapping[CoordinateName, int],
    removal_order: tuple[CoordinateName, ...],
    read_primitives: PrimitiveReads,
) -> JointReadSet:
    """Choose a joint read set by removing statuses in declared order.

    The reader analyses metadata; it must not execute user callbacks. It must
    overestimate inputs for unsupported primitives. Every status outside the
    candidate read set remains free at the same time. Ordinary coordinates and
    action or parameter inputs are never specialised by this function.
    """
    if not set(statuses) <= coordinates:
        raise ValueError("Statuses must name declared coordinates")
    if any(type(value) is not int for value in statuses.values()):
        raise ValueError("Status assignments must contain integer category codes")
    if len(removal_order) != len(set(removal_order)) or set(removal_order) != set(
        statuses
    ):
        raise ValueError("Removal order must contain every status exactly once")

    def get_reads(kept: frozenset[CoordinateName]) -> frozenset[CoordinateName]:
        fixed_statuses = MappingProxyType(
            {name: statuses[name] for name in removal_order if name in kept}
        )
        return read_primitives(fixed_statuses=fixed_statuses) & coordinates

    kept = frozenset(statuses) | get_reads(frozenset())
    removed = []
    for name in removal_order:
        candidate = kept - {name}
        if get_reads(candidate) <= candidate:
            kept = candidate
            removed.append(name)
    live_coordinates = (kept & statuses.keys()) | get_reads(kept)
    if not live_coordinates <= kept:
        raise ValueError("Primitive reader changed its joint dependency certificate")
    return JointReadSet(
        live_coordinates=frozenset(live_coordinates),
        removal_order=removal_order,
        removed_statuses=tuple(removed),
    )


def calculate_fibre_liveness(
    facts: tuple[FibreReadFacts, ...],
) -> tuple[FibreLiveness, ...]:
    """Close fixed primitive read sets through all supplied fibre dependencies.

    Read sets are certified once before this closure. They must not be reselected
    as successor liveness grows: different valid joint sets can be incomparable.
    The result is least for these certificates, not for every possible read-set
    selection. Possible edges and same-period value reads must already be supplied;
    this operation neither infers support nor authenticates the original source.
    """
    by_identifier = {fibre.identifier: fibre for fibre in facts}
    if len(by_identifier) != len(facts) or any(
        not isinstance(fibre.identifier, str) or not fibre.identifier for fibre in facts
    ):
        raise ValueError("Fibre identifiers must be unique nonempty strings")
    _validate_fibre_reads(facts=facts, by_identifier=by_identifier)

    live = {fibre.identifier: fibre.direct.live_coordinates for fibre in facts}
    while True:
        updated = {}
        for fibre in facts:
            required = set(fibre.direct.live_coordinates)
            for link in fibre.links:
                for update in link.updates:
                    if update.coordinate in live[link.target]:
                        required.update(update.reads.live_coordinates)
            updated[fibre.identifier] = frozenset(required)
        if updated == live:
            return tuple(
                FibreLiveness(
                    identifier=fibre.identifier,
                    live_coordinates=live[fibre.identifier],
                )
                for fibre in facts
            )
        live = updated


def _validate_fibre_reads(
    *,
    facts: tuple[FibreReadFacts, ...],
    by_identifier: Mapping[FibreId, FibreReadFacts],
) -> None:
    for fibre in facts:
        if not fibre.direct.live_coordinates <= fibre.coordinates:
            raise ValueError(
                f"Direct reads exceed source coordinates: {fibre.identifier}"
            )
        for link in fibre.links:
            if link.target not in by_identifier:
                raise ValueError(f"Unknown target fibre: {link.target}")
            target_coordinates = by_identifier[link.target].coordinates
            updated_coordinates = frozenset(
                update.coordinate for update in link.updates
            )
            if (
                len(updated_coordinates) != len(link.updates)
                or updated_coordinates != target_coordinates
            ):
                raise ValueError(
                    f"Updates must cover every target coordinate exactly once: "
                    f"{fibre.identifier} -> {link.target}"
                )
            if any(
                not update.reads.live_coordinates <= fibre.coordinates
                for update in link.updates
            ):
                raise ValueError(
                    f"Update reads exceed source coordinates: {fibre.identifier}"
                )
