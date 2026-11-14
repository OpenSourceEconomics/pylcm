"""Certify joint coordinate reads using non-executing primitive analysis."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Literal, Protocol

type CoordinateName = str
type FibreId = str
type RelationId = str
type RegimeId = str
type CellId = str
type LocalAxis = tuple[RegimeId, CoordinateName]
type StatusAssignment = tuple[tuple[CoordinateName, int], ...]
type PremiseStatus = Literal["proved", "refuted", "unknown"]


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
class ScreenLink:
    """Keep the original channel and incidence when projecting a support link."""

    target: FibreId
    """Original target fibre identifier."""
    phase: Literal["solve", "simulation", "both"]
    """Original primitive channel, never pooled across phases."""
    marked: bool
    """Whether this is an original within-date value dependency."""
    source_relations: tuple[RelationId, ...]
    """Original producer incidences, preserved through projection."""


@dataclass(frozen=True)
class ScreenFibre:
    """Hold fixed facts on one point of the full original status rectangle."""

    identifier: FibreId
    """Original fibre identifier, including the original cell."""
    cell_id: CellId
    """Original regime-period cell, never merged by this screen."""
    regime: RegimeId
    """Original regime owning this fibre."""
    status_domains: tuple[tuple[CoordinateName, tuple[int, ...]], ...]
    """Complete original code domains, including codes in unkept fibres."""
    statuses: StatusAssignment
    """One code assignment in the original declared status order."""
    reads: tuple[tuple[str, JointReadSet], ...]
    """Fixed read certificates for each original primitive or update."""
    forced: frozenset[CoordinateName]
    """Fixed forced-action facts; empty retains every action coordinate."""
    excluded: tuple[tuple[CoordinateName, frozenset[int]], ...]
    """Certified infeasible categorical values on the entire original fibre."""
    links: tuple[ScreenLink, ...]
    """Original possible support with phase, marks and producer identity."""


@dataclass(frozen=True)
class ScreenPremise:
    """Record a finite premise check without claiming source authentication."""

    status: PremiseStatus
    """Whether this supplied-fact premise passes, fails or needs evidence."""
    witness: tuple[FibreId, ...]
    """Original parts that demonstrate a failed check, if any."""
    reason: str
    """The checked condition or missing evidence."""


@dataclass(frozen=True)
class InertProjection:
    """Describe a status-to-ordinary projection, never a Model state deletion."""

    projection: Mapping[FibreId, FibreId]
    """Map every original part to an original representative of its group."""
    premises: Mapping[str, ScreenPremise]
    """Separate I1-I5 and fewer-status premises a-c."""
    scope: Literal["identity", "inert-literal", "inert-joint"]
    """Proof scope, independent of whether the finite premises pass."""

    @property
    def conditions_pass(self) -> bool:
        """Require every finite premise; unknown evidence cannot authorize a count."""
        return self.premises.keys() == {
            "I1",
            "I2",
            "I3",
            "I4",
            "I5",
            "a",
            "b",
            "c",
        } and all(premise.status == "proved" for premise in self.premises.values())


def check_inert_projection(
    *,
    fibres: tuple[ScreenFibre, ...],
    kept: frozenset[FibreId],
    removed: frozenset[LocalAxis],
) -> InertProjection:
    """Check one joint projection against fixed full-grid supplied facts.

    Removed axes become ordinary coordinates; they are not removed from states.
    This does not select axes, authenticate source facts, infer economic support,
    or establish a cost optimum. Original primitive and full-hull exclusion facts
    must already be certified. Empty forcing retains all actions, so restricting
    the regime's original shock vector establishes A2(a) on every block. Other
    forcing choices retain unknown shock-law obligations.
    """
    by_id = _validate_screen_fibres(fibres=fibres, kept=kept, removed=removed)
    groups: dict[tuple[CellId, StatusAssignment], list[ScreenFibre]] = {}
    for fibre in fibres:
        assignment = tuple(
            (name, code)
            for name, code in fibre.statuses
            if (fibre.regime, name) not in removed
        )
        groups.setdefault((fibre.cell_id, assignment), []).append(fibre)
    projection = {
        part.identifier: parts[0].identifier
        for parts in groups.values()
        for part in parts
    }
    premises = {
        name: ScreenPremise("proved", (), reason)
        for name, reason in (
            ("I1", "All original parts of each kept projected fibre are kept"),
            (
                "I2",
                "Fixed reads/forcing agree on kept parts; exclusions agree on all parts",
            ),
            (
                "I3",
                "Every kept part has the same jointly projected marked phase support",
            ),
            ("I4", "Every marked projected link occurs inside each fixed-value slice"),
            (
                "I5",
                "Empty forcing retains all actions; use the original shock-law marginal",
            ),
            ("a", "Projected links, marks and initials are their literal images"),
            ("b", "Projected facts are equal to their parts, without averaging"),
            ("c", "Every lifted profile retains all original action coordinates"),
        )
    }
    for parts in groups.values():
        live_parts = tuple(part for part in parts if part.identifier in kept)
        if live_parts and len(live_parts) != len(parts):
            _refute_screen_premise(premises=premises, name="I1", parts=parts)
        for left, right in zip(live_parts, live_parts[1:]):
            if (
                dict((name, read.live_coordinates) for name, read in left.reads)
                != dict((name, read.live_coordinates) for name, read in right.reads)
                or left.forced != right.forced
            ):
                _refute_screen_premise(
                    premises=premises, name="I2", parts=(left, right)
                )
            if _projected_screen_links(
                fibre=left, projection=projection
            ) != _projected_screen_links(fibre=right, projection=projection):
                _refute_screen_premise(
                    premises=premises, name="I3", parts=(left, right)
                )
        for left, right in zip(parts, parts[1:]):
            if dict(left.excluded) != dict(right.excluded):
                _refute_screen_premise(
                    premises=premises, name="I2", parts=(left, right)
                )
    premises["b"] = ScreenPremise(
        premises["I2"].status, premises["I2"].witness, premises["b"].reason
    )
    coarse_marked = {
        (projection[fibre.identifier], projection[link.target])
        for fibre in fibres
        if fibre.identifier in kept
        for link in fibre.links
        if link.marked
        and link.target in kept
        and fibre.regime == by_id[link.target].regime
    }
    fine_marked = {
        (fibre.identifier, link.target)
        for fibre in fibres
        for link in fibre.links
        if link.marked
    }
    for source_group, target_group in coarse_marked:
        sources = tuple(
            part for part in fibres if projection[part.identifier] == source_group
        )
        targets = tuple(
            part for part in fibres if projection[part.identifier] == target_group
        )
        for left in sources:
            slice_value = tuple(
                sorted(
                    (name, code)
                    for name, code in left.statuses
                    if (left.regime, name) in removed
                )
            )
            for right in targets:
                if (
                    tuple(
                        sorted(
                            (name, code)
                            for name, code in right.statuses
                            if (right.regime, name) in removed
                        )
                    )
                    == slice_value
                    and (left.identifier, right.identifier) not in fine_marked
                ):
                    _refute_screen_premise(
                        premises=premises, name="I4", parts=(left, right)
                    )
    if any(fibre.forced for fibre in fibres if fibre.identifier in kept):
        for name in ("I4", "I5", "c"):
            if premises[name].status != "refuted":
                premises[name] = ScreenPremise(
                    "unknown",
                    (),
                    "Nonempty forcing needs a shock-law/admission certificate",
                )
    regimes = {fibre.regime for fibre in fibres}
    names = {name for _, name in removed}
    literal = (
        len(names) == 1
        and removed == frozenset(product(regimes, names))
        and all(names <= dict(fibre.status_domains).keys() for fibre in fibres)
    )
    return InertProjection(
        MappingProxyType(projection),
        MappingProxyType(premises),
        "identity" if not removed else "inert-literal" if literal else "inert-joint",
    )


def _projected_screen_links(
    *, fibre: ScreenFibre, projection: Mapping[FibreId, FibreId]
) -> frozenset[tuple[FibreId, str, bool, tuple[RelationId, ...]]]:
    return frozenset(
        (projection[link.target], link.phase, link.marked, link.source_relations)
        for link in fibre.links
    )


def _refute_screen_premise(
    *,
    premises: dict[str, ScreenPremise],
    name: str,
    parts: tuple[ScreenFibre, ...] | list[ScreenFibre],
) -> None:
    if premises[name].status == "proved":
        premises[name] = ScreenPremise(
            "refuted", tuple(part.identifier for part in parts), premises[name].reason
        )


def _validate_screen_fibres(
    *,
    fibres: tuple[ScreenFibre, ...],
    kept: frozenset[FibreId],
    removed: frozenset[LocalAxis],
) -> dict[FibreId, ScreenFibre]:
    by_id = {fibre.identifier: fibre for fibre in fibres}
    if len(by_id) != len(fibres) or any(not fibre.identifier for fibre in fibres):
        raise ValueError("Screen fibre identifiers must be unique and nonempty")
    if not kept <= by_id.keys():
        raise ValueError("Kept fibres must belong to the original rectangle")
    local_axes = {
        (fibre.regime, name) for fibre in fibres for name, _ in fibre.status_domains
    }
    if not removed <= local_axes:
        raise ValueError(
            "Removed axes must belong to original regimes and status domains"
        )
    cells: dict[CellId, list[ScreenFibre]] = {}
    regime_domains: dict[RegimeId, dict[CoordinateName, frozenset[int]]] = {}
    for fibre in fibres:
        cells.setdefault(fibre.cell_id, []).append(fibre)
        domains = {name: frozenset(codes) for name, codes in fibre.status_domains}
        if regime_domains.setdefault(fibre.regime, domains) != domains:
            raise ValueError(
                "Original status domains must agree across a regime's cells"
            )
        if len(dict(fibre.reads)) != len(fibre.reads) or len(
            dict(fibre.excluded)
        ) != len(fibre.excluded):
            raise ValueError("Primitive and action fact names must be unique")
        if any(link.target not in by_id for link in fibre.links):
            raise ValueError("Screen links must name original fibres")
    for parts in cells.values():
        original = parts[0]
        domains = original.status_domains
        if len(dict(domains)) != len(domains) or any(
            not codes
            or len(set(codes)) != len(codes)
            or any(type(code) is not int for code in codes)
            for _, codes in domains
        ):
            raise ValueError("Original rectangle needs distinct integer code domains")
        expected = set(product(*(codes for _, codes in domains)))
        actual = {tuple(code for _, code in part.statuses) for part in parts}
        names = tuple(name for name, _ in domains)
        if (
            any(
                part.status_domains != domains
                or part.regime != original.regime
                or tuple(name for name, _ in part.statuses) != names
                or any(type(code) is not int for _, code in part.statuses)
                for part in parts
            )
            or actual != expected
            or len(parts) != len(expected)
        ):
            raise ValueError("Screen requires the complete original status rectangle")
    return by_id


class PrimitiveReads(Protocol):
    """Conservatively read original primitives with only supplied statuses fixed."""

    def __call__(
        self, *, fixed_statuses: Mapping[CoordinateName, int]
    ) -> frozenset[CoordinateName]: ...


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
