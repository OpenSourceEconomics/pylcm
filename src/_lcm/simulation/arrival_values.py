"""Call-local original-value operands for replaying a dated arrival price."""

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


def _declared_arrival_origins(
    *, queries: ScalarPolicyQueries
) -> Mapping[ValueArtifactAddress, ValueArtifactAddress]:
    """Recover only model-owned aliases reachable from this actual query closure."""
    from _lcm.arrival_queries import _transport_alias  # noqa: PLC0415

    origins = {}
    pending = [queries]
    visited = set()
    while pending:
        graph = pending.pop()
        if id(graph) in visited:
            continue
        visited.add(id(graph))
        for node in graph.definitions.values():
            for plans in node.context.arrival_queries.values():
                for plan in plans.values():
                    for name, origin in plan.transport_origins.items():
                        alias = _transport_alias(plan=plan, name=name)
                        previous = origins.setdefault(alias, origin)
                        if previous != origin:
                            raise ExecutionPlanningError(
                                "An arrival replay alias has conflicting "
                                "original owners."
                            )
                    pending.append(plan.queries)
    return MappingProxyType(origins)


def _declared_projected_addresses(
    *, queries: ScalarPolicyQueries
) -> frozenset[ValueArtifactAddress]:
    """Separate genuine projected dates from arrival-only transport aliases."""
    from _lcm.regime_building.Q_and_F import projected_forward_readers  # noqa: PLC0415

    return frozenset(
        read.target
        for node in queries.definitions.values()
        if (function := node.context.Q_and_F_functions.get(node.read.period))
        is not None
        for reader in projected_forward_readers(function=function).values()
        for read in reader.reads(
            source=node.read.target, period=node.read.period, core="projected_query"
        )
    )


def _original_value(
    *,
    address: ValueArtifactAddress,
    aliases: Mapping[ValueArtifactAddress, ValueArtifactAddress],
) -> ValueArtifactAddress:
    visited = set()
    while address.kind is ValueArtifactKind.CONTINUATION_LEAF:
        if address in visited or address not in aliases:
            raise ExecutionPlanningError(
                "Arrival replay requested an undeclared or cyclic operand origin."
            )
        visited.add(address)
        address = aliases[address]
    if address.kind is not ValueArtifactKind.REGIME_VALUE:
        raise ExecutionPlanningError("Arrival replay needs an original stored value.")
    return address


def arrival_replay_value_reads(
    *, queries: ScalarPolicyQueries, core_key: str
) -> tuple[ValueRead, ...]:
    """Declare absolute original dates instead of requiring persisted solve carries."""
    aliases = _declared_arrival_origins(queries=queries)
    projected = _declared_projected_addresses(queries=queries)
    if projected & aliases.keys():
        raise ExecutionPlanningError(
            "A projected source cannot be an arrival transport alias."
        )
    reads = []
    for read in queries.value_reads(core_key=core_key):
        if (
            read.target.kind is not ValueArtifactKind.CONTINUATION_LEAF
            or read.target in projected
        ):
            reads.append(read)
            continue
        original = _original_value(address=read.target, aliases=aliases)
        reads.append(
            replace(
                read,
                target=original,
                source=replace(
                    read.source, channel=ValueInputChannel.ARRIVAL_REPLAY_VALUE
                ),
            )
        )
    return tuple(reads)


def arrival_replay_context(
    *,
    queries: ScalarPolicyQueries,
    context: CoreBuildContext,
    values: Mapping[int, Mapping[RegimeName, ValueTemplate]],
) -> CoreBuildContext:
    """Borrow demanded numeric operands without capturing the result or its history."""
    from _lcm.arrival_queries import _ArrivalBuildContext  # noqa: PLC0415

    reads = queries.value_reads(core_key="arrival_replay_context")
    addressed = {
        read.target
        for read in reads
        if read.target.kind is ValueArtifactKind.CONTINUATION_LEAF
    }
    if not addressed:
        return context
    aliases = _declared_arrival_origins(queries=queries)
    projected = _declared_projected_addresses(queries=queries)
    if addressed & projected:
        raise ExecutionPlanningError(
            "Projected exit/arrival queries require their separately acquired "
            "continuation context."
        )
    operands = {}
    for alias in addressed:
        original = _original_value(address=alias, aliases=aliases)
        try:
            operands[alias] = values[original.period][original.regime]
        except KeyError as error:
            raise ExecutionPlanningError(
                "Arrival replay is missing its declared original-date value."
            ) from error
    return _ArrivalBuildContext(
        **{
            declaration.name: getattr(context, declaration.name)
            for declaration in fields(CoreBuildContext)
        },
        arrival_values=MappingProxyType(operands),
    )
