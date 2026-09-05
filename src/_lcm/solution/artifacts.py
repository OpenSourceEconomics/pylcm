"""Built-in artifact identities and the adapter to public ``SolutionResult``."""

import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from _lcm.engine import Regime
from _lcm.execution.core_program import ProgramScope, core_program_graph
from _lcm.params.mapping_leaf import MappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.solution.contract import BackwardInductionResult
from _lcm.solution.result_snapshot import snapshot_artifact_store
from _lcm.typing import FlatParams, RegimeName
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ArtifactChannel,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    OmissionReason,
    PersistencePolicy,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
    _replay_route_identity,
)

if TYPE_CHECKING:
    from _lcm.solution.model_authority import SolutionAuthority
else:
    # The beartype import claw resolves annotations at runtime. Importing the
    # concrete type here would close the artifacts -> authority -> artifacts cycle;
    # static checking keeps the precise type above while runtime checks the rest of
    # this private bridge's fully concrete signature.
    SolutionAuthority = Any


def build_solution_result(  # noqa: C901, PLR0912, PLR0915
    *,
    internal_result: BackwardInductionResult,
    retention: ResultRetention,
    regimes: Mapping[RegimeName, Regime],
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    n_periods: int,
    model_instance_id: str,
    params_fingerprint: str,
    model_fingerprint: str,
    authority: SolutionAuthority,
) -> SolutionResult:
    """Label existing engine outputs without changing their numerical meaning."""
    replay: dict[ArtifactRef, object] = dict(internal_result.replay_artifacts)
    retained_continuations: dict[ArtifactRef, object] = dict(
        internal_result.retained_continuations
    )
    auxiliary: dict[ArtifactRef, object] = dict(internal_result.auxiliary_artifacts)
    diagnostics: dict[ArtifactRef, object] = {}
    omissions: dict[ArtifactRef, OmissionReason] = {}

    if retention.retains_replay:
        declared_replay_policies = MappingProxyType(
            {
                period: MappingProxyType(
                    {
                        regime_name: payload
                        for regime_name, payload in regime_to_payload.items()
                        if regimes[regime_name].simulation.egm_policy_read is not None
                    }
                )
                for period, regime_to_payload in (
                    internal_result.simulation_policies.items()
                )
            }
        )
        _add_nested_artifacts(
            target=replay,
            nested=declared_replay_policies,
            key=SIMULATION_POLICY,
        )
        _add_nested_artifacts(
            target=replay,
            nested=internal_result.dissolution_flags,
            key=DISSOLUTION_FLAG,
        )

    if retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS:
        # A policy replayed against solve-generated adaptive axes is read through
        # facts the solving model instance holds beside the result, not inside
        # it; nothing persistable carries them, so the policy is not kept.
        for policy_ref in tuple(replay):
            if (
                policy_ref.key == SIMULATION_POLICY
                and authority.replay[policy_ref].adaptive_outer_nodes is not None
            ):
                del replay[policy_ref]
                omissions[policy_ref] = OmissionReason.NOT_PERSISTED

        for store in (retained_continuations, replay, auxiliary):
            for ref in tuple(store):
                generic_authority = authority.artifacts.get(ref)
                if (
                    generic_authority is None
                    or generic_authority.descriptor.persistence
                    is PersistencePolicy.NOT_PERSISTED
                ):
                    del store[ref]
                    omissions[ref] = OmissionReason.NOT_PERSISTED

    _add_nested_artifacts(
        target=diagnostics,
        nested=internal_result.diagnostics,
        key=SOLVER_DIAGNOSTICS,
    )

    for period, regime_to_value in internal_result.value_functions.items():
        for regime_name in regime_to_value:
            regime = regimes[regime_name]
            user_regime = user_regimes[regime_name]

            policy_ref = ArtifactRef(
                period=period,
                regime=regime_name,
                key=SIMULATION_POLICY,
            )
            policy_descriptor = authority.replay[policy_ref]
            policy_read = regime.simulation.egm_policy_read
            can_publish_policy = regime.simulation.external_replay_route is None and (
                policy_read is not None
                or user_regime.solver.publishes_simulation_policy
                or _graph_publishes_replay(regime=regime, period=period)
            )
            if (
                can_publish_policy
                and policy_ref not in replay
                and policy_ref not in omissions
            ):
                if not policy_descriptor.applicable:
                    omissions[policy_ref] = OmissionReason.NOT_APPLICABLE
                elif not retention.retains_replay:
                    omissions[policy_ref] = OmissionReason.NOT_REQUESTED
                elif (
                    retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
                    and authority.artifacts[policy_ref].descriptor.persistence
                    is PersistencePolicy.NOT_PERSISTED
                ):
                    omissions[policy_ref] = OmissionReason.NOT_PERSISTED
                elif policy_descriptor.required:
                    msg = (
                        "A model-authoritative replay route published no required "
                        f"artifact at ({period}, {regime_name!r}, "
                        f"{SIMULATION_POLICY.type_id!r})."
                    )
                    raise RuntimeError(msg)
                else:
                    omissions[policy_ref] = OmissionReason.UNSUPPORTED

            if regime.stakeholders is not None:
                dissolution_ref = ArtifactRef(
                    period=period,
                    regime=regime_name,
                    key=DISSOLUTION_FLAG,
                )
                if dissolution_ref not in replay:
                    dissolution_descriptor = authority.replay[dissolution_ref]
                    if not retention.retains_replay:
                        omissions[dissolution_ref] = OmissionReason.NOT_REQUESTED
                    elif dissolution_descriptor.required:
                        msg = (
                            "A model-authoritative replay route published no "
                            "required artifact at "
                            f"({period}, {regime_name!r}, "
                            f"{DISSOLUTION_FLAG.type_id!r})."
                        )
                        raise RuntimeError(msg)
                    else:
                        omissions[dissolution_ref] = OmissionReason.UNSUPPORTED

    present_refs = set(retained_continuations) | set(replay) | set(auxiliary)
    for ref, artifact_authority in authority.artifacts.items():
        if ref in present_refs or ref in omissions:
            continue
        descriptor = artifact_authority.descriptor
        selected = retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS or (
            retention is ResultRetention.VALUES_AND_REPLAY
            and descriptor.channel is ArtifactChannel.REPLAY
        )
        if not artifact_authority.applicable:
            omissions[ref] = OmissionReason.NOT_APPLICABLE
        elif not selected:
            omissions[ref] = OmissionReason.NOT_REQUESTED
        elif descriptor.persistence is PersistencePolicy.NOT_PERSISTED:
            omissions[ref] = OmissionReason.NOT_PERSISTED
        elif artifact_authority.required:
            raise RuntimeError(
                "A model-authoritative route published no required artifact at "
                f"({ref.period}, {ref.regime!r}, {ref.key.type_id!r})."
            )
        else:
            omissions[ref] = OmissionReason.UNSUPPORTED

    result = SolutionResult(
        values=internal_result.value_functions,
        retained_continuations=snapshot_artifact_store(
            store=ArtifactStore(retained_continuations),
            authorities=authority.artifacts,
        ),
        replay_artifacts=snapshot_artifact_store(
            store=ArtifactStore(replay),
            authorities=authority.artifacts,
        ),
        auxiliary_artifacts=snapshot_artifact_store(
            store=ArtifactStore(auxiliary),
            authorities=authority.artifacts,
        ),
        metadata=SolutionMetadata(
            retention=retention,
            n_periods=n_periods,
            regime_names=tuple(regimes),
            solver_types=MappingProxyType(
                {
                    regime_name: (
                        f"{type(user_regime.solver).__module__}."
                        f"{type(user_regime.solver).__qualname__}"
                    )
                    for regime_name, user_regime in user_regimes.items()
                }
            ),
            model_instance_id=model_instance_id,
            params_fingerprint=params_fingerprint,
            model_fingerprint=model_fingerprint,
            solver_identities=MappingProxyType(
                {
                    regime_name: user_regime.solver.identity
                    for regime_name, user_regime in user_regimes.items()
                }
            ),
            replay_routes=MappingProxyType(
                {
                    regime_name: _replay_route_identity(
                        regimes[regime_name].simulation.replay_route
                    )
                    for regime_name in regimes
                }
            ),
            artifact_descriptors=MappingProxyType(dict(authority.artifact_descriptors)),
            value_schemas=MappingProxyType(
                {
                    (period, regime_name): ValueArraySchema(
                        shape=authority.values[(period, regime_name)].shape,
                        dtype=authority.values[(period, regime_name)].dtype,
                        axis_names=(authority.values[(period, regime_name)].axis_names),
                    )
                    for period, regime_to_value in (
                        internal_result.value_functions.items()
                    )
                    for regime_name in regime_to_value
                }
            ),
        ),
        omissions=MappingProxyType(omissions),
        diagnostics=snapshot_artifact_store(
            store=ArtifactStore(diagnostics),
            authorities=authority.artifacts,
        ),
    )
    object.__setattr__(
        result,
        "_artifact_authority",
        MappingProxyType(dict(authority.artifacts)),
    )
    return result


def fingerprint_flat_params(flat_params: FlatParams) -> str:
    """Return a deterministic SHA-256 of canonical flat parameter leaves.

    The digest records every tree path, container boundary, array shape,
    dtype, and C-order byte representation. It binds an in-memory result to the
    exact canonical parameters used by a solve; it is not a model fingerprint.
    """
    digest = hashlib.sha256()
    for regime_name in sorted(flat_params):
        for param_name in sorted(flat_params[regime_name]):
            _update_digest_value(
                digest=digest,
                value=flat_params[regime_name][param_name],
                path=(regime_name, param_name),
            )
    return digest.hexdigest()


def _update_digest_token(*, digest: Any, chunk: str | bytes) -> None:  # noqa: ANN401
    """Feed one length-prefixed token into the digest."""
    payload = chunk.encode() if isinstance(chunk, str) else chunk
    digest.update(len(payload).to_bytes(8, byteorder="big"))
    digest.update(payload)


def _update_digest_value(
    *,
    digest: Any,  # noqa: ANN401
    value: Any,  # noqa: ANN401
    path: tuple[str, ...],
) -> None:
    """Feed one parameter leaf, with its tree path and container boundaries."""
    _update_digest_token(digest=digest, chunk="path")
    _update_digest_token(digest=digest, chunk=str(len(path)))
    for component in path:
        _update_digest_token(digest=digest, chunk=component)
    if isinstance(value, MappingLeaf):
        _update_digest_token(digest=digest, chunk="mapping")
        for key in sorted(value.data):
            _update_digest_value(
                digest=digest, value=value.data[key], path=(*path, key)
            )
        return
    if isinstance(value, SequenceLeaf):
        _update_digest_token(digest=digest, chunk="sequence")
        for index, child in enumerate(value.data):
            _update_digest_value(digest=digest, value=child, path=(*path, str(index)))
        return

    # ``asarray`` keeps the declared rank; a 0-d parameter and a length-one
    # vector are distinct leaves even when their bytes agree.
    array = np.asarray(value)
    _update_digest_token(digest=digest, chunk="array")
    _update_digest_token(
        digest=digest, chunk=",".join(str(size) for size in array.shape)
    )
    _update_digest_token(digest=digest, chunk=array.dtype.str)
    _update_digest_token(digest=digest, chunk=array.tobytes(order="C"))


def _canonical_value_axis_names(*, regime: Regime) -> tuple[str, ...]:
    """Return the public names of the axes in one canonical stored value."""
    state_axes = tuple(
        name
        for name in regime.solution.state_names
        if name not in regime.fold_state_names
    )
    return (
        (*state_axes, "stakeholder") if regime.stakeholders is not None else state_axes
    )


def _graph_publishes_replay(*, regime: Regime, period: int) -> bool:
    """Return whether the period's kernel declares a replay-scoped program."""
    graph = core_program_graph(kernel=regime.solution.period_kernels[period])
    return any(program.scope is ProgramScope.REPLAY for program in graph.values())


def _add_nested_artifacts(
    *,
    target: dict[ArtifactRef, object],
    nested: Mapping[int, Mapping[RegimeName, object]],
    key: ArtifactKey,
) -> None:
    """Flatten one existing period/regime mapping into addressed artifacts."""
    for period, regime_to_payload in nested.items():
        for regime, payload in regime_to_payload.items():
            target[ArtifactRef(period=period, regime=regime, key=key)] = payload


__all__ = [
    "DISSOLUTION_FLAG",
    "EGM_CONTINUATION",
    "SIMULATION_POLICY",
    "SOLVER_DIAGNOSTICS",
]
