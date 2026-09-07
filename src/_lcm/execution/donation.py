"""Static resolution of which arguments a dispatch donates to its executable.

A donated argument's buffer is handed to XLA for reuse and is unreadable after
the call. The decision is made before lowering, because `donate_argnames` is
part of the executable, and it is made from the ledger: an argument is donated
when every artifact it carries has this dispatch as its sole remaining
consumer, is neither retained by the result nor pinned by an undeclared reader,
shares its buffer with no other key, and reaches the program on its stored
layout rather than as a transferred copy.

The ledger counts a dispatch once however many of its programs and arguments
read one artifact, so it answers which dispatch is the last consumer but not
how many executable inputs that dispatch feeds the buffer to. Donation is an
argument-level act, so the resolver asks the second question of the unit's
read census instead: an argument is donated only when it is the one declared
input locator of the whole unit aimed at every artifact it carries.
"""

import dataclasses
from collections.abc import Hashable, Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType

from _lcm.execution.core_program import ResolvedCoreProgram
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueConsumerAddress,
    ValueTransferKind,
)


class DonatedBuffer(StrEnum):
    """Which buffer a donation would hand to the executable."""

    STORED_ARTIFACT = "stored_artifact"
    """The artifact's own buffer; donating it frees what the ledger tracks."""

    TRANSFERRED_COPY = "transferred_copy"
    """A copy the transfer plan makes; donating it frees nothing tracked."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedDonation:
    """One candidate argument's donation decision for one dispatch."""

    argument: str
    """The program argument named in `donation_candidates`."""

    artifacts: tuple[ValueArtifactAddress, ...]
    """The artifacts the argument carries, by the reads addressed to it."""

    buffer: DonatedBuffer
    """Which buffer the argument would hand over."""

    withheld_by: ValueConsumerAddress | None = None
    """A second declared locator of the unit reading one of `artifacts`."""

    @property
    def donated(self) -> bool:
        """Whether the executable is lowered with this argument donated."""
        return self.buffer is DonatedBuffer.STORED_ARTIFACT and self.withheld_by is None


def unit_input_readers(
    *, programs: Iterable[ResolvedCoreProgram]
) -> MappingProxyType[ValueArtifactAddress, frozenset[ValueConsumerAddress]]:
    """Census the distinct declared input locators one unit aims at each artifact.

    The census answers the question the ledger's per-dispatch count cannot: how
    many executable inputs of this unit the buffer reaches. A locator names the
    core, the input channel, the argument and the path together, so two
    arguments of one core and two cores reading one artifact each contribute an
    entry, while the same core's width alternatives repeat one locator and
    contribute one.

    Args:
        programs: The resolved programs of one dispatch unit, one per core.

    Returns:
        Immutable mapping of artifact address to the distinct locators of that
        unit reading it.

    """
    readers: dict[ValueArtifactAddress, set[ValueConsumerAddress]] = {}
    for program in programs:
        for read in program.requirements.value_reads:
            readers.setdefault(read.target, set()).add(read.source)
    return MappingProxyType(
        {artifact: frozenset(sources) for artifact, sources in readers.items()}
    )


def resolve_donations(
    *,
    program: ResolvedCoreProgram,
    dispatch: Hashable,
    unit_readers: Mapping[ValueArtifactAddress, frozenset[ValueConsumerAddress]],
    ledger: PlannedInputLiveness,
    n_periods: int,
) -> tuple[ResolvedDonation, ...]:
    """Decide, per candidate argument, whether this dispatch donates it.

    A candidate no declared read addresses by name is left alone: the engine
    cannot say what buffer it carries. A read whose artifact lies beyond the
    last period names the solve-lifetime template, which is never donated, and
    one the ledger never registered is outside the plan, where only membership
    and sole-consumer questions have an answer at all. An artifact a second
    locator of the unit also reads is recorded with that locator and left
    undonated, because the executable still reads the buffer through it.

    Args:
        program: The resolved program whose candidates are decided.
        dispatch: The `(period, regime)` unit this program belongs to.
        unit_readers: `unit_input_readers` over every program of that unit.
        ledger: The planned remaining-consumer accounting of the solve.
        n_periods: Number of periods the solve runs.

    Returns:
        Tuple of one decision per candidate argument a declared read addresses.

    """
    donations: list[ResolvedDonation] = []
    for argument in program.donation_candidates:
        reads = tuple(
            read
            for read in program.requirements.value_reads
            if read.source.argument == argument and read.source.path == ()
        )
        if not reads:
            continue
        artifacts = tuple(read.target for read in reads)
        transfers = tuple(
            transfer
            for transfer in program.input_transfer_plan
            if transfer.source.argument == argument and transfer.source.path == ()
        )
        if any(
            transfer.kind is not ValueTransferKind.ALIGNED_LOCAL
            for transfer in transfers
        ):
            donations.append(
                ResolvedDonation(
                    argument=argument,
                    artifacts=artifacts,
                    buffer=DonatedBuffer.TRANSFERRED_COPY,
                )
            )
            continue
        if all(
            _is_donatable(
                artifact=artifact,
                dispatch=dispatch,
                ledger=ledger,
                n_periods=n_periods,
            )
            for artifact in artifacts
        ):
            donations.append(
                ResolvedDonation(
                    argument=argument,
                    artifacts=artifacts,
                    buffer=DonatedBuffer.STORED_ARTIFACT,
                    withheld_by=_second_locator(
                        artifacts=artifacts,
                        declared=frozenset(read.source for read in reads),
                        unit_readers=unit_readers,
                    ),
                )
            )
    return tuple(donations)


def _is_donatable(
    *,
    artifact: ValueArtifactAddress,
    dispatch: Hashable,
    ledger: PlannedInputLiveness,
    n_periods: int,
) -> bool:
    """Report whether this dispatch may hand one artifact's own buffer over.

    Membership is asked first: `is_known` and `has_sole_remaining_consumer` are
    the only questions an unplanned artifact answers, and every question after
    them presumes the ledger registered the key.
    """
    return (
        artifact.period < n_periods
        and ledger.is_known(artifact=artifact)
        and ledger.has_sole_remaining_consumer(artifact=artifact, dispatch=dispatch)
        and not ledger.is_retained(artifact=artifact)
        and not ledger.is_pinned(artifact=artifact)
        and ledger.alias_group(artifact=artifact) == frozenset({artifact})
    )


def _second_locator(
    *,
    artifacts: tuple[ValueArtifactAddress, ...],
    declared: frozenset[ValueConsumerAddress],
    unit_readers: Mapping[ValueArtifactAddress, frozenset[ValueConsumerAddress]],
) -> ValueConsumerAddress | None:
    """Return a locator other than this argument's own reading one artifact.

    `None` says the argument is the unit's one declared input locator for every
    artifact it carries, which is what makes handing the buffer over safe. The
    locators are ordered by their representation so a plan names the same one
    whichever order the programs of the unit were resolved in.
    """
    others = sorted(
        {
            locator
            for artifact in artifacts
            for locator in unit_readers.get(artifact, frozenset())
            if locator not in declared
        },
        key=repr,
    )
    return others[0] if others else None
