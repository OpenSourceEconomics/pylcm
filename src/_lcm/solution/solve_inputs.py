"""Where each artifact key lives in the rolling input mappings of a solve.

The loop threads three mappings through every period: next-period values by
regime, next-period continuation payloads by regime, and gated-edge
continuations by `(source, target)`. A release or a donation acts on one array
inside them; this module finds that array by its artifact key and rebuilds the
mapping with the solve-lifetime template leaf in its place.
"""

import dataclasses
from types import MappingProxyType

import jax

from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.scheduler import BufferRegistry, replace_leaf_by_identity
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.solution.contract import ContinuationPayload
from _lcm.typing import RegimeName
from lcm.solver_api import ArtifactKey, ContinuationReader
from lcm.typing import FloatND

type EdgeKey = tuple[RegimeName, RegimeName]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SolveInputMappings:
    """The three rolling input mappings the period loop threads."""

    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND]
    """Next-period value per regime; the template for regimes not yet solved."""

    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload]
    """Next-period continuation payload per publishing regime."""

    next_edge_to_V_arr: MappingProxyType[EdgeKey, FloatND]
    """Next-period gated continuation per `(source, target)` edge."""


def locate_artifact(
    *, inputs: SolveInputMappings, artifact: ValueArtifactAddress
) -> jax.Array | None:
    """Return the array an artifact key names in the mappings, or `None`."""
    if artifact.kind is ValueArtifactKind.REGIME_VALUE:
        return inputs.next_regime_to_V_arr.get(artifact.regime)
    if artifact.kind is ValueArtifactKind.GATED_CONTINUATION:
        if artifact.target_regime is None:
            return None
        return inputs.next_edge_to_V_arr.get((artifact.regime, artifact.target_regime))
    payload = inputs.next_regime_to_continuation.get(artifact.regime)
    if not isinstance(payload, ContinuationReader):
        return None
    return payload.leaves().get(artifact.leaf_path)


def substitute_artifact(
    *,
    inputs: SolveInputMappings,
    templates: SolveInputMappings,
    artifact: ValueArtifactAddress,
) -> SolveInputMappings:
    """Return the mappings with the artifact's array replaced by its template leaf.

    The template mappings live for the whole solve, so the substitution
    allocates nothing and every mapping keeps its pytree structure.
    """
    old = locate_artifact(inputs=inputs, artifact=artifact)
    new = locate_artifact(inputs=templates, artifact=artifact)
    if old is None or new is None:
        msg = f"Artifact {artifact!r} is not addressed by the solve input mappings."
        raise KeyError(msg)
    if artifact.kind is ValueArtifactKind.REGIME_VALUE:
        return dataclasses.replace(
            inputs,
            next_regime_to_V_arr=replace_leaf_by_identity(
                tree=inputs.next_regime_to_V_arr, old=old, new=new
            ),
        )
    if artifact.kind is ValueArtifactKind.GATED_CONTINUATION:
        return dataclasses.replace(
            inputs,
            next_edge_to_V_arr=replace_leaf_by_identity(
                tree=inputs.next_edge_to_V_arr, old=old, new=new
            ),
        )
    return dataclasses.replace(
        inputs,
        next_regime_to_continuation=replace_leaf_by_identity(
            tree=inputs.next_regime_to_continuation, old=old, new=new
        ),
    )


def register_rolled_inputs(
    *,
    inputs: SolveInputMappings,
    next_period: int,
    ledger: PlannedInputLiveness,
    registry: BufferRegistry,
) -> None:
    """Register every known artifact key of the mappings on its current buffer.

    Called once per period, after the roll and before the first dispatch, so a
    buffer the roll carried forward is known under its new key before any count
    on its old key can close.
    """
    for regime_name, value in inputs.next_regime_to_V_arr.items():
        _register_known(
            registry=registry,
            ledger=ledger,
            array=value,
            artifact=ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=next_period,
                regime=regime_name,
            ),
        )
    for (source_name, target_name), value in inputs.next_edge_to_V_arr.items():
        _register_known(
            registry=registry,
            ledger=ledger,
            array=value,
            artifact=ValueArtifactAddress(
                kind=ValueArtifactKind.GATED_CONTINUATION,
                period=next_period,
                regime=source_name,
                target_regime=target_name,
            ),
        )
    for regime_name, payload in inputs.next_regime_to_continuation.items():
        if not isinstance(payload, ContinuationReader):
            continue
        artifact_key = getattr(payload, "artifact_key", None)
        if not isinstance(artifact_key, ArtifactKey):
            continue
        for leaf_path, leaf in payload.leaves().items():
            _register_known(
                registry=registry,
                ledger=ledger,
                array=leaf,
                artifact=ValueArtifactAddress(
                    kind=ValueArtifactKind.CONTINUATION_LEAF,
                    period=next_period,
                    regime=regime_name,
                    artifact_key=artifact_key,
                    leaf_path=leaf_path,
                ),
            )


def _register_known(
    *,
    registry: BufferRegistry,
    ledger: PlannedInputLiveness,
    array: jax.Array,
    artifact: ValueArtifactAddress,
) -> None:
    """Register one array under one key when the ledger plans that key."""
    if ledger.is_known(artifact=artifact) and not array.is_deleted():
        registry.register(array=array, artifact=artifact)
