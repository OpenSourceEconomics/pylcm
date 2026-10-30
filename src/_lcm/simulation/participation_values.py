"""Call-local original-value operands for replaying a dated participation price."""

from collections.abc import Mapping
from dataclasses import fields, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from _lcm.execution.core_program import CoreBuildContext, ValueRead
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueInputChannel,
)
from _lcm.typing import ValueTemplate
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import RegimeName

if TYPE_CHECKING:
    from _lcm.policy_queries import ScalarPolicyQueries
else:
    # Simulation declarations are imported while canonical query modules initialize.
    type ScalarPolicyQueries = object


# Retained stored participation surfaces: period, publishing regime, payload leaf name.
type ParticipationSurfaces = Mapping[
    int, Mapping[RegimeName, Mapping[str, ValueTemplate]]
]


def is_participation_surface(*, address: ValueArtifactAddress) -> bool:
    """Whether `address` names a stored participation surface leaf."""
    from _lcm.participation_queries import _is_surface  # noqa: PLC0415

    return _is_surface(address=address, period=address.period)


def stored_participation_surface(
    *, address: ValueArtifactAddress, surfaces: ParticipationSurfaces
) -> ValueTemplate:
    """The retained surface `address` names.

    Raises:
        ExecutionPlanningError: If the solve retained no such surface.

    """
    try:
        return surfaces[address.period][address.regime][address.leaf_path[-1]]
    except KeyError as error:
        raise ExecutionPlanningError(
            "Participation replay is missing its stored participation surface for "
            f"{address.leaf_path[-1]!r} at period {address.period}, published by "
            f"{address.regime!r}. Simulate from a solution of this model, as "
            "solved or as loaded from its saved archive; a copy that dropped its "
            "participation surfaces cannot replay."
        ) from error


def _declared_participation_origins(
    *, queries: ScalarPolicyQueries
) -> Mapping[ValueArtifactAddress, ValueArtifactAddress]:
    """Recover only model-owned aliases reachable from this actual query closure."""
    from _lcm.participation_queries import _transport_alias  # noqa: PLC0415

    origins = {}
    pending = [queries]
    visited = set()
    while pending:
        graph = pending.pop()
        if id(graph) in visited:
            continue
        visited.add(id(graph))
        for node in graph.definitions.values():
            for plans in node.context.participation_queries.values():
                for plan in plans.values():
                    for name, origin in plan.transport_origins.items():
                        alias = _transport_alias(plan=plan, name=name)
                        previous = origins.setdefault(alias, origin)
                        if previous != origin:
                            raise ExecutionPlanningError(
                                "A participation replay alias has conflicting "
                                "original owners."
                            )
                    pending.append(plan.queries)
    return MappingProxyType(origins)


def _original_value(
    *,
    address: ValueArtifactAddress,
    aliases: Mapping[ValueArtifactAddress, ValueArtifactAddress],
) -> ValueArtifactAddress:
    visited = set()
    while address.kind is ValueArtifactKind.CONTINUATION_LEAF:
        if address in visited or address not in aliases:
            raise ExecutionPlanningError(
                "Participation replay requested an undeclared or cyclic operand origin."
            )
        visited.add(address)
        address = aliases[address]
    if address.kind is not ValueArtifactKind.REGIME_VALUE:
        raise ExecutionPlanningError(
            "Participation replay needs an original stored value."
        )
    return address


def participation_replay_value_reads(
    *, queries: ScalarPolicyQueries, core_key: str
) -> tuple[ValueRead, ...]:
    """Declare absolute original dates instead of requiring persisted solve carries."""
    aliases = _declared_participation_origins(queries=queries)
    reads = []
    for read in queries.value_reads(core_key=core_key):
        if read.target.kind is not ValueArtifactKind.CONTINUATION_LEAF:
            reads.append(read)
            continue
        if is_participation_surface(address=read.target):
            # A stored surface is its own retained array, not an alias of one.
            reads.append(
                replace(
                    read,
                    source=replace(
                        read.source,
                        channel=ValueInputChannel.PARTICIPATION_REPLAY_VALUE,
                    ),
                )
            )
            continue
        original = _original_value(address=read.target, aliases=aliases)
        reads.append(
            replace(
                read,
                target=original,
                source=replace(
                    read.source, channel=ValueInputChannel.PARTICIPATION_REPLAY_VALUE
                ),
            )
        )
    return tuple(reads)


def participation_replay_context(
    *,
    queries: ScalarPolicyQueries,
    context: CoreBuildContext,
    values: Mapping[int, Mapping[RegimeName, ValueTemplate]],
    surfaces: ParticipationSurfaces = MappingProxyType({}),
) -> CoreBuildContext:
    """Borrow demanded numeric operands without capturing the result or its history.

    A transported operand is the original stored value at its absolute date; a
    stored participation surface is the solve's retained surface at its date.
    """
    from _lcm.participation_queries import _ParticipationBuildContext  # noqa: PLC0415

    reads = queries.value_reads(core_key="participation_replay_context")
    addressed = {
        read.target
        for read in reads
        if read.target.kind is ValueArtifactKind.CONTINUATION_LEAF
    }
    if not addressed:
        return context
    aliases = _declared_participation_origins(queries=queries)
    operands = {}
    for alias in addressed:
        if is_participation_surface(address=alias):
            operands[alias] = stored_participation_surface(
                address=alias, surfaces=surfaces
            )
            continue
        original = _original_value(address=alias, aliases=aliases)
        try:
            operands[alias] = values[original.period][original.regime]
        except KeyError as error:
            raise ExecutionPlanningError(
                "Participation replay is missing its declared original-date value."
            ) from error
    return _ParticipationBuildContext(
        **{
            declaration.name: getattr(context, declaration.name)
            for declaration in fields(CoreBuildContext)
        },
        participation_values=MappingProxyType(operands),
    )
