"""Fail-closed bridge from public kernel outputs to the legacy solve loop."""

from collections.abc import Mapping

import jax.numpy as jnp

from _lcm.continuation import ContinuationPayload
from _lcm.egm.carry import EGMCarry
from _lcm.solution.contract import KernelResult
from _lcm.typing import RegimeName
from lcm.solver_api import ArtifactKey, KernelOutput


def normalize_kernel_output(
    *,
    output: KernelOutput | KernelResult,
    continuation_key: ArtifactKey | None,
    regime_name: RegimeName,
    period: int,
) -> KernelResult:
    """Consume one public output into the engine's current result representation.

    Legacy results pass through by identity, preserving every existing optional
    field. A public output is accepted only when every artifact is consumed by
    this bridge. This deliberately makes adding a producer-side artifact without
    its engine consumer an immediate, cell-labelled error instead of silent loss.
    """
    if isinstance(output, KernelResult):
        return output
    if not isinstance(output, KernelOutput):
        msg = (
            f"Regime '{regime_name}' in period {period} returned unsupported "
            f"kernel output type {type(output).__name__}."
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
        raw_continuation = continuations.pop(continuation_key)
        if not isinstance(raw_continuation, EGMCarry):
            msg = (
                f"Regime '{regime_name}' in period {period} published artifact "
                f"'{continuation_key.type_id}' version "
                f"{continuation_key.schema_version} with unsupported payload type "
                f"{type(raw_continuation).__name__}; expected EGMCarry."
            )
            raise RuntimeError(msg)
        continuation = raw_continuation

    _fail_on_unconsumed(
        channel="continuations",
        artifacts=continuations,
        regime_name=regime_name,
        period=period,
    )
    for channel, artifacts in (
        ("solve_time_artifacts", output.solve_time_artifacts),
        ("replay", output.replay),
        ("auxiliary", output.auxiliary),
    ):
        _fail_on_unconsumed(
            channel=channel,
            artifacts=artifacts,
            regime_name=regime_name,
            period=period,
        )

    return KernelResult(V_arr=jnp.asarray(output.value), continuation=continuation)


def require_legacy_kernel_result(
    *,
    output: KernelOutput | KernelResult,
    consumer: str,
) -> KernelResult:
    """Require a legacy result at an as-yet-unmigrated composite boundary.

    A composite kernel must explicitly consume every public artifact before it
    can safely wrap a migrated child. Until that migration is complete, refuse
    a public output instead of attribute-crashing or silently dropping its
    artifact channels. Existing legacy children pass through by identity.
    """
    if isinstance(output, KernelResult):
        return output
    if isinstance(output, KernelOutput):
        msg = (
            f"{consumer} cannot yet consume KernelOutput; migrate this composite "
            "kernel's artifact handling before wrapping a migrated child kernel."
        )
        raise RuntimeError(msg)  # noqa: TRY004 - migration boundary, not bad input.
    msg = f"{consumer} received unsupported kernel output type {type(output).__name__}."
    raise TypeError(msg)


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


__all__ = ["normalize_kernel_output", "require_legacy_kernel_result"]
