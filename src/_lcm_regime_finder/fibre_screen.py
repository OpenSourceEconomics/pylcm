"""Build a finite status-projection check from authenticated original primitives."""

from collections import deque
from collections.abc import Mapping
from itertools import product

from .fibre_facts import (
    CoordinateName,
    JointReadSet,
    RegimeId,
    ScreenFibre,
    ScreenLink,
    check_inert_projection,
)
from .fibre_source import FibreSource, Wire, _mapping, _text


def build_source_screen(
    *,
    source: FibreSource,
    status_axes: Mapping[RegimeId, tuple[CoordinateName, ...]],
) -> Wire:
    """Check the joint demotion of supplied statuses, retaining original state grids.

    Status classification is an explicit input, never guessed from category labels.
    This checks the supplied proposal, not every subset or a preferred layout.
    Unknown support retains every possible target. Source qualifications remain
    separate from the finite premise check, and no native change is authorized.
    """
    regimes = {_text(cell["regime"]) for cell in source._cells.values()}
    if not status_axes.keys() <= regimes:
        raise ValueError("Status axes must name original admitted regimes")
    if any(len(set(axes)) != len(axes) for axes in status_axes.values()):
        raise ValueError("Status axes must not repeat coordinates")
    states = {
        cell_id: _mapping(cell["states"]) for cell_id, cell in source._cells.items()
    }
    assignments: dict[str, tuple[tuple[int, ...], ...]] = {}
    domains: dict[str, tuple[tuple[CoordinateName, tuple[int, ...]], ...]] = {}
    identifiers: dict[str, tuple[str, ...]] = {}
    for cell_id, cell in source._cells.items():
        requested = status_axes.get(_text(cell["regime"]), ())
        if not set(requested) <= states[cell_id].keys():
            raise ValueError("Status axes must name original categorical states")
        domains[cell_id] = tuple(
            (name, tuple(source._category(source._domains[_text(identity)]).by_code))
            for name, identity in states[cell_id].items()
            if name in requested
        )
        assignments[cell_id] = tuple(product(*(codes for _, codes in domains[cell_id])))
        identifiers[cell_id] = tuple(
            f"{cell_id}:{index}" for index in range(len(assignments[cell_id]))
        )
    templates = {row.identifier: row.links for row in source.get_cell_facts()}
    nodes = []
    initial = set()
    query_guard = any(
        relation.get("kind") == "policy_query"
        for relation in source._relations.values()
    )
    for cell_id, cell in source._cells.items():
        axes = tuple(name for name, _ in domains[cell_id])
        coordinates = frozenset(states[cell_id])
        for identifier, codes in zip(
            identifiers[cell_id], assignments[cell_id], strict=True
        ):
            statuses = dict(zip(axes, codes, strict=True))
            direct = (
                JointReadSet(coordinates, (), ())
                if query_guard
                else source.get_direct_reads(
                    cell_id=cell_id,
                    statuses=statuses,
                    removal_order=axes,
                )
            )
            reads = {"direct": direct}
            links = []
            for template in templates[cell_id]:
                target = template.target
                supports: dict[CoordinateName, frozenset[int]] = {}
                for coordinate in states[target]:
                    owners = tuple(
                        rid
                        for rid in template.source_relations
                        if coordinate
                        in _mapping(source._relations[rid].get("state_updates", {}))
                    )
                    if len(owners) > 1:
                        raise ValueError("Multiple original update producers")
                    key = (
                        "update:"
                        + ":".join(template.source_relations)
                        + ":"
                        + coordinate
                    )
                    reads[key] = (
                        source.get_update_reads(
                            relation_id=owners[0],
                            target_coordinate=coordinate,
                            statuses=statuses,
                            removal_order=axes,
                        )
                        if owners
                        else JointReadSet(coordinates, (), ())
                    )
                    if owners and coordinate in dict(domains[target]):
                        support = source.get_target_support(
                            relation_id=owners[0],
                            target_coordinate=coordinate,
                            statuses=statuses,
                        )
                        if support.proved:
                            supports[coordinate] = support.possible_codes
                target_axes = tuple(name for name, _ in domains[target])
                for target_id, target_codes in zip(
                    identifiers[target], assignments[target], strict=True
                ):
                    if all(
                        code
                        in supports.get(name, frozenset(dict(domains[target])[name]))
                        for name, code in zip(target_axes, target_codes, strict=True)
                    ):
                        links.append(
                            ScreenLink(
                                target=target_id,
                                phase=template.phase,
                                marked=cell["period"]
                                == source._cells[target]["period"],
                                source_relations=template.source_relations,
                            )
                        )
            excluded = tuple(
                (
                    name,
                    source.get_excluded_action_values(
                        cell_id=cell_id,
                        action=name,
                        status_values={
                            coordinate: (code,) for coordinate, code in statuses.items()
                        },
                    ),
                )
                for name, identity in _mapping(cell["actions"]).items()
                if source._domains[_text(identity)].get("kind") == "categorical"
            )
            nodes.append(
                ScreenFibre(
                    identifier=identifier,
                    cell_id=cell_id,
                    regime=_text(cell["regime"]),
                    status_domains=domains[cell_id],
                    statuses=tuple(statuses.items()),
                    reads=tuple(reads.items()),
                    forced=frozenset(),
                    excluded=excluded,
                    links=tuple(links),
                )
            )
            if cell["admissible_start"]:
                initial.add(identifier)
    fibres = tuple(nodes)
    by_id = {node.identifier: node for node in fibres}
    kept = set(initial)
    queue = deque(initial)
    while queue:
        for link in by_id[queue.popleft()].links:
            if link.target not in kept:
                kept.add(link.target)
                queue.append(link.target)
    removed = frozenset(
        (regime, name) for regime, axes in status_axes.items() for name in axes
    )
    screen = check_inert_projection(
        fibres=fibres, kept=frozenset(kept), removed=removed
    )
    raw_lanes = {
        (node.regime, tuple(sorted(node.statuses)))
        for node in fibres
        if node.identifier in kept
    }
    projected_lanes = {
        (
            node.regime,
            tuple(
                sorted(
                    (name, code)
                    for name, code in node.statuses
                    if (node.regime, name) not in removed
                )
            ),
        )
        for node in fibres
        if node.identifier in kept
    }
    return {
        "schema": "source-bound-inert-screen/v1",
        "conditions_pass": screen.conditions_pass,
        "transformation_authorized": False,
        "proof_scope": screen.scope,
        "status_axes": {name: list(axes) for name, axes in status_axes.items()},
        "forcing": "no forced-action facts",
        "forced_action_savings": "not measured",
        "raw_lanes": len(raw_lanes),
        "projected_lanes": len(projected_lanes) if screen.conditions_pass else None,
        "source_obligations": list(source.source_obligations),
        "premises": {
            name: {
                "status": premise.status,
                "witness": list(premise.witness),
                "reason": premise.reason,
            }
            for name, premise in screen.premises.items()
        },
        "projection": dict(screen.projection),
        "initial_fibres": sorted(initial),
        "possible_support_kept": sorted(kept),
        "qualification": "Explicit status classification; one joint proposal. Removed statuses become ordinary coordinates, not deleted states. Coordinate-marginal possible support and initial-node overapproximations are retained. Source qualifications are not cleared. No preferred layout, native adoption or optimum is certified.",
    }
