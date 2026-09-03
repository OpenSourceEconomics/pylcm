"""The solve loop's consumer of one period kernel's public output.

Every artifact a kernel publishes on a channel of its `KernelOutput` is read
here by its declared key or refused: the continuation channel carries the
`EGMCarry` a parent interpolates, the replay channel the simulation policy, the
solve-time channel the collective dissolution flag, and the auxiliary channel
the solver diagnostics and the engine-private replay authority. An artifact
under any other key, or a known key with a payload of the wrong type, is an
immediate error naming the regime and period rather than a silent loss, so a
producer-side artifact without its engine reader cannot ship.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from _lcm.continuation import ContinuationPayload
from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.published_policy import EGMSimPolicy, NBEGMGridPolicy, NNBEGMSimPolicy
from _lcm.solution.contract import (
    GENERATED_REPLAY_AUTHORITY,
    GeneratedReplayAuthority,
    SimulationPolicy,
)
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.typing import RegimeName
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ArtifactKey,
    KernelOutput,
)
from lcm.typing import BoolND, FloatND

_SIMULATION_POLICY_TYPES = (
    EGMSimPolicy,
    NBEGMGridPolicy,
    NNBEGMSimPolicy,
    NestedEGMSimPolicy,
)


@dataclass(frozen=True, kw_only=True)
class ConsumedKernelOutput:
    """One regime-period output read by the solve loop, artifact by artifact.

    Every field is what the kernel published under the corresponding key, or
    `None` where it published nothing there. The loop stores each one where
    something reads it; nothing here is produced by a kernel.
    """

    value: FloatND
    """The regime's value-function array on its exogenous state grid."""

    continuation: ContinuationPayload | None
    """The declared continuation a continuation-based parent interpolates."""

    simulation_policy: SimulationPolicy | None
    """The replay-channel policy forward simulation can interpolate."""

    generated_replay_authority: GeneratedReplayAuthority | None
    """The engine-private replay facts emitted beside an adaptive policy."""

    dissolution: BoolND | None
    """A collective regime's empty-feasible-set flag on the state axes.

    `True` exactly where no action satisfies the combined constraints, so the
    household argmax was taken over an empty set. Distinct from a numeric
    `-inf` value, which occurs on-path; gates consume this flag, never test
    `V == -inf`.
    """

    diagnostics: SolverDiagnostics | None
    """The solver's numerical self-report, when it measures anything."""


def consume_kernel_output(
    *,
    output: object,
    continuation_key: ArtifactKey | None,
    regime_name: RegimeName,
    period: int,
) -> ConsumedKernelOutput:
    """Read one kernel's output by declared key; refuse anything without a reader.

    `continuation_key` names the continuation the regime's parents read, or
    is `None` for a regime that publishes none. A published continuation under
    that key's `type_id` with another `schema_version`, a missing required
    continuation, an artifact under a key with no reader, a known key with a
    payload of the wrong type, and a replay authority without the policy it
    describes are each refused with the cell's coordinates.
    """
    if not isinstance(output, KernelOutput):
        msg = (
            f"Regime '{regime_name}' in period {period} returned unsupported "
            f"kernel output type {type(output).__name__}; a period kernel returns "
            "KernelOutput."
        )
        raise TypeError(msg)

    continuations = dict(output.continuations)
    continuation: ContinuationPayload | None = None
    if continuation_key is not None:
        _fail_on_version_mismatch(
            keys=continuations,
            expected=continuation_key,
            regime_name=regime_name,
            period=period,
        )
        if continuation_key not in continuations:
            _fail_on_unconsumed(
                channel="continuations",
                artifacts=continuations,
                regime_name=regime_name,
                period=period,
            )
            msg = (
                f"Regime '{regime_name}' in period {period} is missing required "
                f"continuation artifact '{continuation_key.type_id}' version "
                f"{continuation_key.schema_version}."
            )
            raise RuntimeError(msg)
        continuation = _pop_typed_artifact(
            channel="continuations",
            artifacts=continuations,
            key=continuation_key,
            expected_types=(EGMCarry,),
            regime_name=regime_name,
            period=period,
        )
    _fail_on_unconsumed(
        channel="continuations",
        artifacts=continuations,
        regime_name=regime_name,
        period=period,
    )

    replay = dict(output.replay)
    simulation_policy = _pop_typed_artifact(
        channel="replay",
        artifacts=replay,
        key=SIMULATION_POLICY,
        expected_types=_SIMULATION_POLICY_TYPES,
        regime_name=regime_name,
        period=period,
    )
    solve_time_artifacts = dict(output.solve_time_artifacts)
    dissolution = _pop_typed_artifact(
        channel="solve_time_artifacts",
        artifacts=solve_time_artifacts,
        key=DISSOLUTION_FLAG,
        expected_types=(jnp.ndarray,),
        regime_name=regime_name,
        period=period,
    )
    if dissolution is not None and jnp.dtype(dissolution.dtype) != jnp.bool_:
        msg = (
            f"Regime '{regime_name}' in period {period} published artifact "
            f"'{DISSOLUTION_FLAG.type_id}' with dtype {dissolution.dtype}; expected "
            "bool."
        )
        raise RuntimeError(msg)
    auxiliary = dict(output.auxiliary)
    diagnostics = _pop_typed_artifact(
        channel="auxiliary",
        artifacts=auxiliary,
        key=SOLVER_DIAGNOSTICS,
        expected_types=(SolverDiagnostics,),
        regime_name=regime_name,
        period=period,
    )
    generated_replay_authority = _pop_typed_artifact(
        channel="auxiliary",
        artifacts=auxiliary,
        key=GENERATED_REPLAY_AUTHORITY,
        expected_types=(GeneratedReplayAuthority,),
        regime_name=regime_name,
        period=period,
    )
    if generated_replay_authority is not None and simulation_policy is None:
        msg = (
            f"Regime '{regime_name}' in period {period} published artifact "
            f"'{GENERATED_REPLAY_AUTHORITY.type_id}' with no matching "
            f"'{SIMULATION_POLICY.type_id}' policy on the replay channel."
        )
        raise RuntimeError(msg)
    for channel, artifacts in (
        ("solve_time_artifacts", solve_time_artifacts),
        ("replay", replay),
        ("auxiliary", auxiliary),
    ):
        _fail_on_unconsumed(
            channel=channel,
            artifacts=artifacts,
            regime_name=regime_name,
            period=period,
        )

    return ConsumedKernelOutput(
        value=jnp.asarray(output.value),
        continuation=continuation,
        simulation_policy=simulation_policy,
        generated_replay_authority=generated_replay_authority,
        dissolution=dissolution,
        diagnostics=diagnostics,
    )


def _pop_typed_artifact(
    *,
    channel: str,
    artifacts: dict[ArtifactKey, object],
    key: ArtifactKey,
    expected_types: tuple[type, ...],
    regime_name: RegimeName,
    period: int,
) -> Any:  # noqa: ANN401
    """Remove the artifact under `key` from `artifacts` after checking its type.

    Returns `None` when the key is absent; a present artifact of a type the
    engine cannot consume is refused with the cell's coordinates.
    """
    if key not in artifacts:
        return None
    payload = artifacts.pop(key)
    if not isinstance(payload, expected_types):
        expected = ", ".join(expected_type.__name__ for expected_type in expected_types)
        msg = (
            f"Regime '{regime_name}' in period {period} published {channel} "
            f"artifact '{key.type_id}' version {key.schema_version} with "
            f"unsupported payload type {type(payload).__name__}; expected "
            f"{expected}."
        )
        raise RuntimeError(msg)  # noqa: TRY004 - a contract violation, not bad input.
    return payload


def _fail_on_version_mismatch(
    *,
    keys: Mapping[ArtifactKey, object],
    expected: ArtifactKey,
    regime_name: RegimeName,
    period: int,
) -> None:
    mismatches = tuple(
        key
        for key in keys
        if key.type_id == expected.type_id
        and key.schema_version != expected.schema_version
    )
    if mismatches:
        versions = ", ".join(str(key.schema_version) for key in mismatches)
        msg = (
            f"Regime '{regime_name}' in period {period} published artifact "
            f"'{expected.type_id}' with version {versions}; expected version "
            f"{expected.schema_version}."
        )
        raise RuntimeError(msg)


def _fail_on_unconsumed(
    *,
    channel: str,
    artifacts: Mapping[ArtifactKey, object],
    regime_name: RegimeName,
    period: int,
) -> None:
    if not artifacts:
        return
    identities = ", ".join(
        f"'{key.type_id}' v{key.schema_version}" for key in sorted(artifacts)
    )
    msg = (
        f"Regime '{regime_name}' in period {period} published unconsumed "
        f"{channel} artifacts: {identities}."
    )
    raise RuntimeError(msg)


__all__ = ["ConsumedKernelOutput", "consume_kernel_output"]
