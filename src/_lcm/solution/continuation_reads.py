"""Value reads a solver declares for the continuation leaves it consumes.

An EGM-family core reads its target's published rows, so the leaves are named
here once and every family declares them the same way: through the rolling
continuation mapping when the builder passes the whole payload, and by argument
name when the builder flattens the rows into named arguments.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from _lcm.continuation import ContinuationSpec
from _lcm.execution.core_program import CoreProgram, ValueRead
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.typing import RegimeName
from lcm.solver_api import ArtifactKey, ContinuationReader


def continuation_leaf_reads(
    *,
    template: ContinuationReader,
    artifact_key: ArtifactKey,
    target: RegimeName,
    source_regime: RegimeName,
    source_period: int,
    core_key: str,
    argument_by_leaf: Mapping[tuple[str, ...], str] | None = None,
) -> tuple[ValueRead, ...]:
    """Declare one read per published leaf of one target's continuation.

    Without `argument_by_leaf` the leaves are addressed inside the rolling
    `next_regime_to_continuation` mapping, keyed by the target regime. With it,
    only the named leaves are declared, each addressed as the program argument
    its builder puts it in.
    """
    reads: list[ValueRead] = []
    for leaf_path in template.leaves():
        argument = None if argument_by_leaf is None else argument_by_leaf.get(leaf_path)
        if argument_by_leaf is not None and argument is None:
            continue
        reads.append(
            ValueRead(
                target=ValueArtifactAddress(
                    kind=ValueArtifactKind.CONTINUATION_LEAF,
                    period=source_period + 1,
                    regime=target,
                    artifact_key=artifact_key,
                    leaf_path=leaf_path,
                ),
                source=ValueConsumerAddress(
                    source_period=source_period,
                    source_regime=source_regime,
                    core_key=core_key,
                    channel=ValueInputChannel.CONTINUATION_LEAF,
                    argument=argument,
                    path=() if argument else (target, *leaf_path),
                ),
            )
        )
    return tuple(reads)


def published_continuation_template(
    *,
    continuation_specs: Mapping[RegimeName, ContinuationSpec],
    target: RegimeName,
) -> ContinuationReader | None:
    """Return the payload `target` publishes, when it names its own leaves.

    `None` when the target publishes no continuation — which is every target on
    the build pass that establishes the templates — and when it publishes one
    that answers no leaf query, so a reader declares nothing rather than
    assuming a shape.
    """
    spec = continuation_specs.get(target)
    if spec is None or not isinstance(spec.template, ContinuationReader):
        return None
    return spec.template


def published_continuation_templates(
    *,
    continuation_specs: Mapping[RegimeName, ContinuationSpec],
    targets: Iterable[RegimeName],
) -> MappingProxyType[RegimeName, ContinuationReader]:
    """Return the published payload of each target that has one, in name order.

    Which leaves a target publishes is a fact about the target alone, so a
    solver resolves the templates once and re-addresses them per period.
    """
    resolved: dict[RegimeName, ContinuationReader] = {}
    for target in sorted(targets):
        template = published_continuation_template(
            continuation_specs=continuation_specs, target=target
        )
        if template is not None:
            resolved[target] = template
    return MappingProxyType(resolved)


def rekeyed_value_reads(
    *, reads: tuple[ValueRead, ...], core_key: str
) -> tuple[ValueRead, ...]:
    """Re-address declared reads to the core that actually runs them.

    A kernel that composes another kernel's program under a different name owns
    the reads that program declared, so the locator names the composing core.
    """
    return tuple(
        dataclasses.replace(
            read, source=dataclasses.replace(read.source, core_key=core_key)
        )
        for read in reads
    )


def with_continuation_leaf_reads(
    *,
    programs: Mapping[str, CoreProgram],
    reads_by_core_key: Mapping[str, tuple[ValueRead, ...]],
) -> MappingProxyType[str, CoreProgram]:
    """Attach one period's declared reads to a shared program graph.

    Solvers that reuse one numerical core across periods build the programs
    once; the reads are a fact about the period, so they are attached per period
    around the shared core rather than baked into it.
    """
    return MappingProxyType(
        {
            core_key: dataclasses.replace(
                program,
                requirements=dataclasses.replace(
                    program.requirements,
                    value_reads=reads_by_core_key.get(core_key, ()),
                ),
            )
            for core_key, program in programs.items()
        }
    )
