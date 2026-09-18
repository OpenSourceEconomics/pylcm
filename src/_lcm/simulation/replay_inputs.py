"""Addressed, period-local materialization of stored simulation replay payloads."""

import dataclasses
from types import MappingProxyType

import jax
import numpy as np

from _lcm.execution.core_program import ValueRead
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.value_reads import PeriodSimulationReads
from lcm.exceptions import InvalidSimulationInputError
from lcm.solver_api import (
    ArtifactKey,
    ExecutableReplayRoute,
    ReplayReader,
    ReplayRouteSnapshot,
    SimulationBuildContext,
)


def replay_payload_reads(
    *, payload: object, key: ArtifactKey, period: int, regime: str, core: str
) -> tuple[ValueRead, ...]:
    """Name every concrete array leaf passed to one replay adapter.

    The artifact locator uses typed JAX pytree path steps, so a dictionary key
    and a sequence index cannot acquire the same identity. The consumer path
    follows the actual payload container. A root-array payload has an empty path.
    """
    return tuple(
        _payload_read(key=key, period=period, regime=regime, core=core, path=path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(payload)[0]
        if isinstance(leaf, jax.Array | np.ndarray)
    )


def place_replay_payload[T](
    *,
    payload: T,
    key: ArtifactKey,
    period: int,
    regime: str,
    core: str,
    owner: PeriodSimulationReads,
) -> T:
    """Acquire the declared payload leaves and preserve its exact pytree shape."""

    # keyword-only-exempt: library-callback=jax.tree_util.tree_map_with_path
    def place(path: tuple, leaf: object) -> object:
        if not isinstance(leaf, jax.Array | np.ndarray):
            return leaf
        return owner.read(
            unit=regime,
            read=_payload_read(
                key=key, period=period, regime=regime, core=core, path=path
            ),
            value=leaf,
        )

    return jax.tree_util.tree_map_with_path(place, payload)


def _payload_read(
    *, key: ArtifactKey, period: int, regime: str, core: str, path: tuple
) -> ValueRead:
    """Build the same leaf address for declaration and acquisition."""
    return ValueRead(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REPLAY_ARTIFACT_LEAF,
            period=period,
            regime=regime,
            artifact_key=key,
            leaf_path=tuple(f"{type(step).__name__}:{step}" for step in path),
        ),
        source=ValueConsumerAddress(
            source_period=period,
            source_regime=regime,
            core_key=core,
            channel=ValueInputChannel.CURRENT_REPLAY_ARTIFACT,
            argument="payload",
            path=tuple(_consumer_step(step) for step in path),
        ),
    )


def _consumer_step(step: object) -> str | int:
    """Use the public container selector represented by a JAX path step."""
    if isinstance(step, jax.tree_util.GetAttrKey):
        return step.name
    if isinstance(step, jax.tree_util.DictKey):
        key = step.key
        if type(key) not in {str, int}:
            raise TypeError("Replay array mappings require string or integer keys.")
        return key
    if isinstance(step, jax.tree_util.SequenceKey | jax.tree_util.FlattenedIndexKey):
        return step.idx if isinstance(step, jax.tree_util.SequenceKey) else step.key
    raise TypeError(f"Unsupported replay payload path step: {step!r}.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PreparedReplayReader:
    """A preflighted external route whose concrete reader is built in its period.

    Whole-result validation retains the original snapshot and model authority.
    The exact placed period snapshot is validated again before reader construction.
    No callable closes over copied payloads before the period owner acquires them.
    """

    route: ExecutableReplayRoute
    snapshot: ReplayRouteSnapshot
    context: SimulationBuildContext

    def reads(self) -> tuple[ValueRead, ...]:
        """Return the external reader's exact current-period payload reads."""
        return tuple(
            read
            for key, payload in self.snapshot.artifacts.items()
            for read in replay_payload_reads(
                payload=payload,
                key=key,
                period=self.context.period,
                regime=self.context.regime_name,
                core="simulation_external_replay",
            )
        )

    def build(
        self, *, owner: PeriodSimulationReads, devices: tuple[jax.Device, ...]
    ) -> ReplayReader:
        """Build the reader against this period's owned device copies."""
        payloads = {
            key: place_replay_payload(
                payload=payload,
                key=key,
                period=self.context.period,
                regime=self.context.regime_name,
                core="simulation_external_replay",
                owner=owner,
            )
            for key, payload in self.snapshot.artifacts.items()
        }
        nodes = place_simulation_arguments(
            arguments={
                "state_nodes": self.context.state_nodes,
                "action_nodes": self.context.action_nodes,
            },
            subject_arg_names=(),
            value_reads=(),
            devices=devices,
        )
        context = dataclasses.replace(
            self.context,
            state_nodes=nodes["state_nodes"],
            action_nodes=nodes["action_nodes"],
        )
        snapshot = dataclasses.replace(
            self.snapshot, artifacts=MappingProxyType(payloads)
        )
        try:
            self.route.validate(snapshot=snapshot, context=context)
        except InvalidSimulationInputError:
            raise
        except Exception as error:
            raise InvalidSimulationInputError(
                "External replay route rejected placed artifacts at "
                f"({context.period}, {context.regime_name!r}): {error}"
            ) from error
        try:
            reader = self.route.build_reader(
                snapshot=snapshot,
                context=context,
            )
        except Exception as error:
            raise InvalidSimulationInputError(
                "External replay route could not build its period reader at "
                f"({context.period}, {context.regime_name!r}): {error}"
            ) from error
        if not isinstance(reader, ReplayReader):
            raise InvalidSimulationInputError(
                "External replay route returned a non-callable reader at "
                f"({context.period}, {context.regime_name!r})."
            )
        return reader
