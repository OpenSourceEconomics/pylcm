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
from _lcm.typing import FlatParams, RegimeName
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    OmissionReason,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
)

if TYPE_CHECKING:
    from _lcm.solution.model_authority import SolutionAuthority
else:
    # The beartype import claw resolves annotations at runtime. Importing the
    # concrete type here would close the artifacts -> authority -> artifacts cycle;
    # static checking keeps the precise type above while runtime checks the rest of
    # this private bridge's fully concrete signature.
    SolutionAuthority = Any


def build_solution_result(  # noqa: C901, PLR0912
    *,
    internal_result: BackwardInductionResult,
    retention: ResultRetention,
    regimes: Mapping[RegimeName, Regime],
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    n_periods: int,
    model_instance_id: str,
    params_fingerprint: str,
    authority: SolutionAuthority,
) -> SolutionResult:
    """Label existing engine outputs without changing their numerical meaning."""
    replay: dict[ArtifactRef, object] = {}
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

    _add_nested_artifacts(
        target=diagnostics,
        nested=internal_result.diagnostics,
        key=SOLVER_DIAGNOSTICS,
    )

    for period, regime_to_value in internal_result.value_functions.items():
        for regime_name in regime_to_value:
            regime = regimes[regime_name]
            user_regime = user_regimes[regime_name]

            if regime.solution.continuation_template is not None:
                omissions[
                    ArtifactRef(
                        period=period,
                        regime=regime_name,
                        key=EGM_CONTINUATION,
                    )
                ] = (
                    OmissionReason.UNSUPPORTED
                    if retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
                    else OmissionReason.NOT_REQUESTED
                )

            policy_ref = ArtifactRef(
                period=period,
                regime=regime_name,
                key=SIMULATION_POLICY,
            )
            policy_descriptor = authority.replay[policy_ref]
            did_publish_policy = (
                period,
                regime_name,
            ) in internal_result.published_simulation_policy_cells
            policy_read = regime.simulation.egm_policy_read
            can_publish_policy = (
                did_publish_policy
                or policy_read is not None
                or user_regime.solver.publishes_simulation_policy
                or _graph_publishes_replay(regime=regime, period=period)
            )
            if can_publish_policy and policy_ref not in replay:
                if not policy_descriptor.applicable:
                    omissions[policy_ref] = OmissionReason.NOT_APPLICABLE
                elif not retention.retains_replay:
                    omissions[policy_ref] = OmissionReason.NOT_REQUESTED
                elif policy_descriptor.required:
                    msg = (
                        "A model-authoritative replay route published no required "
                        f"artifact at ({period}, {regime_name!r}, "
                        f"{SIMULATION_POLICY.type_id!r})."
                    )
                    raise RuntimeError(msg)
                else:
                    omissions[policy_ref] = OmissionReason.NOT_APPLICABLE

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
                        omissions[dissolution_ref] = OmissionReason.NOT_APPLICABLE

    return SolutionResult(
        values=internal_result.value_functions,
        retained_continuations=ArtifactStore(),
        replay_artifacts=ArtifactStore(replay),
        auxiliary_artifacts=ArtifactStore(),
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
        diagnostics=ArtifactStore(diagnostics),
    )


def fingerprint_flat_params(flat_params: FlatParams) -> str:
    """Return a deterministic SHA-256 of canonical flat parameter leaves.

    The digest records every tree path, container boundary, array shape,
    dtype, and C-order byte representation. It binds an in-memory result to the
    exact canonical parameters used by a solve; it is not a model fingerprint.
    """
    digest = hashlib.sha256()

    def update_token(token: str | bytes) -> None:
        payload = token.encode() if isinstance(token, str) else token
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)

    def update_value(*, value: Any, path: tuple[str, ...]) -> None:  # noqa: ANN401
        update_token("path")
        update_token(str(len(path)))
        for component in path:
            update_token(component)
        if isinstance(value, MappingLeaf):
            update_token("mapping")
            for key in sorted(value.data):
                update_value(value=value.data[key], path=(*path, key))
            return
        if isinstance(value, SequenceLeaf):
            update_token("sequence")
            for index, child in enumerate(value.data):
                update_value(value=child, path=(*path, str(index)))
            return

        array = np.ascontiguousarray(np.asarray(value))
        update_token("array")
        update_token(",".join(str(size) for size in array.shape))
        update_token(array.dtype.str)
        update_token(array.tobytes(order="C"))

    for regime_name in sorted(flat_params):
        for param_name in sorted(flat_params[regime_name]):
            update_value(
                value=flat_params[regime_name][param_name],
                path=(regime_name, param_name),
            )
    return digest.hexdigest()


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
