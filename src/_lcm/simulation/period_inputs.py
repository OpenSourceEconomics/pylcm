"""The concrete value-reading adapters executed by one forward regime unit."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax

from _lcm.egm.current_iid_replay_inputs import (
    CurrentIIDReplayInput,
)
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.engine import Regime
from _lcm.execution.core_program import ValueRead
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.regime_building.gated_edges import edge_may_fold_at_period
from _lcm.regime_building.projected_forward import ProjectedForwardInputs
from _lcm.simulation.replay_inputs import PreparedReplayReader, replay_payload_reads
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.solution.continuation_reads import rekeyed_value_reads
from _lcm.solution.host_values import DeclaredValues
from _lcm.typing import QAndFArg, QAndFKwargs, SimulationPolicy
from lcm.solver_api import DISSOLUTION_FLAG, SIMULATION_POLICY, ReplayReader
from lcm.typing import FloatND, ReferenceName, RegimeName

GATE_FOLD = "simulation_gate_fold"
GATE_ROUTE = "simulation_gate_route"
POLICY_SCORE = "simulation_policy_score"


def decision_reads(
    *,
    regime: Regime,
    period: int,
    policy: SimulationPolicy | None,
    reader: PreparedReplayReader | ReplayReader | None,
) -> tuple[ValueRead, ...]:
    """Name the actual decision adapter selected for this unit."""
    if (
        regime.simulation.case_r_replay_spec is not None
        or regime.simulation.uses_case_u_reconstruction
    ):
        return ()
    reads = regime.simulation.programs.decision[period].requirements.value_reads
    if reader is not None:
        return rekeyed_value_reads(
            reads=reads, core_key="simulation_external_replay_score"
        )
    if regime.simulation.replay_route.consumer_route == "nnbegm_finite" and isinstance(
        policy, NNBEGMSimPolicy
    ):
        # Policy payloads are acquired through their separate addressed owner.
        return tuple(
            read
            for read in reads
            if read.target.kind is not ValueArtifactKind.REPLAY_ARTIFACT_LEAF
        )
    return reads


def unit_value_reads(
    *,
    regime: Regime,
    name: str,
    period: int,
    values: DeclaredValues,
    flags: DeclaredValues,
    policy: SimulationPolicy | None,
    reader: PreparedReplayReader | ReplayReader | None,
    current_iid_replay_input: CurrentIIDReplayInput | None = None,
) -> tuple[ValueRead, ...]:
    """Declare every occurrence, while the owner counts distinct regime units."""
    reads = list(
        decision_reads(regime=regime, period=period, policy=policy, reader=reader)
    )
    if current_iid_replay_input is not None:
        from _lcm.egm.current_iid_replay_inputs import (  # noqa: PLC0415
            native_replay_program,
        )

        reads.extend(
            current_iid_replay_input.reads(
                core=native_replay_program(replay=current_iid_replay_input)
            )
        )
    for family in (
        regime.simulation.programs.exit_full,
        regime.simulation.programs.exit_participating,
        *regime.simulation.programs.arrival_families,
    ):
        if period in family:
            reads.extend(family[period].requirements.value_reads)
    reads.extend(
        gate_reads(regime=regime, name=name, period=period, values=values, flags=flags)
    )
    if policy is not None and reader is None:
        for family in (
            regime.simulation.programs.policy_prepare,
            regime.simulation.programs.policy_rank,
        ):
            if period in family:
                reads.extend(
                    read
                    for read in family[period].requirements.value_reads
                    if read.target.kind is ValueArtifactKind.REPLAY_ARTIFACT_LEAF
                )
        reads.extend(
            replay_payload_reads(
                payload=policy,
                key=SIMULATION_POLICY,
                period=period,
                regime=name,
                core="simulation_policy_replay",
            )
        )
        # Legacy policy interpolation/redecision scores its emitted actions with Q.
        # A finite NNBEGM reader already replaces the ordinary decision completely.
        if not (
            regime.simulation.replay_route.consumer_route == "nnbegm_finite"
            and isinstance(policy, NNBEGMSimPolicy)
        ):
            reads.extend(
                rekeyed_value_reads(
                    reads=regime.simulation.programs.decision[
                        period
                    ].requirements.value_reads,
                    core_key=POLICY_SCORE,
                )
            )
    if isinstance(reader, PreparedReplayReader):
        reads.extend(reader.reads())
    return tuple(reads)


def period_read_targets(
    *,
    active: Mapping[RegimeName, Regime],
    period: int,
    values: DeclaredValues,
    flags: DeclaredValues,
) -> tuple[frozenset[tuple[int, RegimeName]], frozenset[tuple[int, RegimeName]]]:
    """Return the stored values and dissolution flags `period` declares it reads.

    Every regime a subject can occupy at the period counts, by `(period,
    regime)`. Which policy or replay reader a regime uses does not change these
    targets, so none is needed; nothing is read to find them.
    """
    value_targets: set[tuple[int, RegimeName]] = set()
    flag_targets: set[tuple[int, RegimeName]] = set()
    for name, regime in active.items():
        for read in unit_value_reads(
            regime=regime,
            name=name,
            period=period,
            values=values,
            flags=flags,
            policy=None,
            reader=None,
        ):
            target = read.target
            if target.kind is ValueArtifactKind.REGIME_VALUE:
                value_targets.add((target.period, target.regime))
            elif target.artifact_key == DISSOLUTION_FLAG:
                flag_targets.add((target.period, target.regime))
    return frozenset(value_targets), frozenset(flag_targets)


def gate_reads(
    *,
    regime: Regime,
    name: str,
    period: int,
    values: DeclaredValues,
    flags: DeclaredValues,
) -> tuple[ValueRead, ...]:
    """Declare raw V and stored Boolean D for the fold and realized gate.

    Wbar belongs to the decision input. The realized gate instead consumes the
    raw target/reference values and D, with D converted to the interpolation
    dtype only after its Boolean stored artifact has been transferred.
    """
    landing_values = values.get(period + 1, {})
    landing_flags = flags.get(period + 1, {})
    reads: list[ValueRead] = []
    for target, edge in regime.gated_edges.items():
        if not edge_may_fold_at_period(
            edge=edge,
            source_name=name,
            fold_period=period + 1,
            solved_regimes=landing_values,
            source_reads_wbar=True,
        ):
            continue
        for core in (GATE_FOLD, GATE_ROUTE):
            reads.extend(
                ValueRead(
                    target=ValueArtifactAddress(
                        kind=ValueArtifactKind.REGIME_VALUE,
                        period=period + 1,
                        regime=reference,
                    ),
                    source=ValueConsumerAddress(
                        source_period=period,
                        source_regime=name,
                        core_key=core,
                        channel=ValueInputChannel.NEXT_REGIME_VALUE,
                        argument="edge_values",
                        path=(target, reference),
                    ),
                )
                for reference in dict.fromkeys((target, *edge.reference_regimes))
            )
            if target in landing_flags:
                reads.append(
                    ValueRead(
                        target=ValueArtifactAddress(
                            kind=ValueArtifactKind.REPLAY_ARTIFACT_LEAF,
                            period=period + 1,
                            regime=target,
                            artifact_key=DISSOLUTION_FLAG,
                        ),
                        source=ValueConsumerAddress(
                            source_period=period,
                            source_regime=name,
                            core_key=core,
                            channel=ValueInputChannel.NEXT_REPLAY_ARTIFACT,
                            argument="edge_flags",
                            path=(target,),
                        ),
                    )
                )
    return tuple(reads)


def acquire_gate_inputs(
    *,
    reads: tuple[ValueRead, ...],
    owner: PeriodSimulationReads,
    name: str,
    values: Mapping[int, Mapping[str, jax.Array]],
    flags: Mapping[int, Mapping[str, jax.Array]],
) -> tuple[Mapping[str, Mapping[str, jax.Array]], Mapping[str, jax.Array]]:
    """Acquire raw leaves into the exact argument paths the adapters consume.

    Every occurrence reads the original stored artifact. The owner shares its
    destination copy across edge references, fold/route families, and units.
    """
    placed_values: dict[str, dict[str, jax.Array]] = {}
    placed_flags: dict[str, jax.Array] = {}
    for read in reads:
        target = read.target
        is_flag = target.kind is ValueArtifactKind.REPLAY_ARTIFACT_LEAF
        source = flags if is_flag else values
        array = owner.read(
            unit=name, read=read, value=source[target.period][target.regime]
        )
        edge_target = cast("str", read.source.path[0])
        if is_flag:
            placed_flags[edge_target] = array
        else:
            reference = cast("str", read.source.path[1])
            placed_values.setdefault(edge_target, {})[reference] = array
    return (
        MappingProxyType(
            {target: MappingProxyType(row) for target, row in placed_values.items()}
        ),
        MappingProxyType(placed_flags),
    )


def acquire_decision_inputs(
    *,
    reads: tuple[ValueRead, ...],
    owner: PeriodSimulationReads,
    name: str,
    next_values: Mapping[str, jax.Array],
    referenced: QAndFKwargs,
    stored_values: Mapping[int, Mapping[str, jax.Array]],
    projected_forward_inputs: ProjectedForwardInputs = MappingProxyType({}),
) -> tuple[MappingProxyType[RegimeName, FloatND], dict[ReferenceName, QAndFArg]]:
    """Acquire the selected decision's V/Wbar leaves at their exact channels."""
    projected_targets = {
        read.target.regime
        for read in reads
        if read.target.kind is ValueArtifactKind.CONTINUATION_LEAF
    }
    channels: dict[ReferenceName, QAndFArg] = {
        "next_regime_to_V_arr": {
            target: value
            for target, value in next_values.items()
            if target not in projected_targets
        },
        **referenced,
    }
    for read in reads:
        channel = read.source.argument or read.source.channel.value
        target = read.target
        if target.kind is ValueArtifactKind.CONTINUATION_LEAF:
            from lcm.solver_api import ArtifactRef  # noqa: PLC0415

            if target.artifact_key is None:
                raise TypeError("Projected forward leaf has no original artifact key.")
            ref = ArtifactRef(
                period=target.period, regime=target.regime, key=target.artifact_key
            )
            if (
                ref.regime not in projected_forward_inputs.get(ref.period, {})
                or read.source.path
                or not read.source.argument
            ):
                raise TypeError(
                    "Projected forward requested an absent or undeclared flat leaf."
                )
            stored = projected_forward_inputs[ref.period][ref.regime].leaves()[
                target.leaf_path
            ]
            if not isinstance(stored, jax.Array):
                raise TypeError(
                    "Projected forward owner requires its retained numerical array."
                )
            channels[channel] = owner.read(unit=name, read=read, value=stored)
            continue
        # Stored originals are authoritative even if another adapter already read
        # this value. A derived Wbar is created once inside this regime unit.
        stored = (
            next_values[cast("str", target.target_regime)]
            if target.kind is ValueArtifactKind.GATED_CONTINUATION
            else stored_values[target.period][target.regime]
        )
        array = owner.read(unit=name, read=read, value=stored)
        entries = dict(cast("Mapping[str, jax.Array]", channels[channel]))
        entries[cast("str", read.source.path[0])] = array
        channels[channel] = MappingProxyType(entries)
    return (
        MappingProxyType(
            cast("Mapping[str, jax.Array]", channels.pop("next_regime_to_V_arr"))
        ),
        channels,
    )
