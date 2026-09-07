"""Static resolution of which arguments a dispatch donates to its executable.

A donated argument's buffer is handed to XLA for reuse and is unreadable after
the call. The decision is made before lowering, because `donate_argnames` is
part of the executable, and it is made from the ledger: an argument is donated
when every artifact it carries has this dispatch as its sole remaining
consumer, is neither retained by the result nor pinned by an undeclared reader,
shares its buffer with no other key, and reaches the program on its stored
layout rather than as a transferred copy.
"""

import dataclasses
from collections.abc import Hashable
from enum import StrEnum

from _lcm.execution.core_program import ResolvedCoreProgram
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueTransferKind


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

    @property
    def donated(self) -> bool:
        """Whether the executable is lowered with this argument donated."""
        return self.buffer is DonatedBuffer.STORED_ARTIFACT


def resolve_donations(
    *,
    program: ResolvedCoreProgram,
    dispatch: Hashable,
    ledger: PlannedInputLiveness,
    n_periods: int,
) -> tuple[ResolvedDonation, ...]:
    """Decide, per candidate argument, whether this dispatch donates it.

    A candidate no declared read addresses by name is left alone: the engine
    cannot say what buffer it carries. A read whose artifact lies beyond the
    last period names the solve-lifetime template, which is never donated, and
    one the ledger never registered is outside the plan, where only membership
    and sole-consumer questions have an answer at all.
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
