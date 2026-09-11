"""The solve-time reads no program declares, pinned so no release frees them.

Two reads reach a target's buffers outside the declared `ValueRead`s:

- the EGM family's argument builder reads the target's `breakpoints` leaf on
  the host to size its boundary candidates, and never passes it to the
  program, so `continuation_leaf_reads` cannot address it;
- a program that declares no reads at all (a host-driven kernel) may read any
  value, continuation or leaf of any regime it can reach.

Both are pinned: their counts stay finite, and zero is never permission to
release them.
"""

from collections.abc import Mapping
from types import MappingProxyType

from _lcm.continuation import ContinuationSpec
from _lcm.engine import Regime
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.solution.continuation_reads import published_continuation_template
from _lcm.typing import RegimeName
from lcm.solver_api import EGM_CONTINUATION, ArtifactKey

# Leaf paths a shipped argument builder reads on the host, by artifact key.
HOST_READ_CONTINUATION_LEAVES: MappingProxyType[
    ArtifactKey, tuple[tuple[str, ...], ...]
] = MappingProxyType({EGM_CONTINUATION: (("breakpoints",),)})


def undeclared_read_pins(
    *,
    regimes: Mapping[RegimeName, Regime],
    regime_name: RegimeName,
    period: int,
    declares_no_reads: bool,
) -> tuple[ValueArtifactAddress, ...]:
    """Return the artifacts one regime-period may read without declaring them.

    With `declares_no_reads` every reachable value, gated continuation,
    reference value and continuation leaf of the next period is pinned, plus
    every same-period reference value. Without it only the host-read leaves of
    `HOST_READ_CONTINUATION_LEAVES` are pinned, for each reachable target whose
    continuation publishes them. The last period reads no next period.
    """
    regime = regimes[regime_name]
    artifacts: list[ValueArtifactAddress] = []
    if declares_no_reads:
        artifacts.extend(
            ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE, period=period, regime=reference
            )
            for reference in regime.same_period_ref_regimes
        )
    reachability = regime.solution.reachability
    if period >= reachability.n_periods - 1:
        return tuple(dict.fromkeys(artifacts))
    continuation_specs = _continuation_specs(regimes=regimes)
    for target in reachability.targets(period=period, source=regime_name):
        if declares_no_reads:
            artifacts.extend(
                _next_period_value_reads(
                    regime=regime,
                    regime_name=regime_name,
                    period=period,
                    target=target,
                )
            )
        artifacts.extend(
            _continuation_leaf_pins(
                continuation_specs=continuation_specs,
                target_name=target,
                period=period + 1,
                every_leaf=declares_no_reads,
            )
        )
    return tuple(dict.fromkeys(artifacts))


def _continuation_specs(
    *, regimes: Mapping[RegimeName, Regime]
) -> MappingProxyType[RegimeName, ContinuationSpec]:
    """Return the continuation spec of every regime that publishes one."""
    return MappingProxyType(
        {
            regime_name: regime.solution.continuation_spec
            for regime_name, regime in regimes.items()
            if regime.solution.continuation_spec is not None
        }
    )


def _next_period_value_reads(
    *, regime: Regime, regime_name: RegimeName, period: int, target: RegimeName
) -> tuple[ValueArtifactAddress, ...]:
    """Return the value and gated-continuation reads one target may receive."""
    edge = regime.gated_edges.get(target)
    if edge is None:
        return (
            ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE, period=period + 1, regime=target
            ),
        )
    return (
        ValueArtifactAddress(
            kind=ValueArtifactKind.GATED_CONTINUATION,
            period=period + 1,
            regime=regime_name,
            target_regime=target,
        ),
        *(
            ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=period + 1,
                regime=reference,
            )
            for reference in edge.reference_regimes
        ),
    )


def _continuation_leaf_pins(
    *,
    continuation_specs: Mapping[RegimeName, ContinuationSpec],
    target_name: RegimeName,
    period: int,
    every_leaf: bool,
) -> tuple[ValueArtifactAddress, ...]:
    """Return the continuation-leaf pins of one target at one period.

    A target without a continuation, or with one that answers no leaf query,
    publishes no addressable leaf and pins nothing.
    """
    spec = continuation_specs.get(target_name)
    template = published_continuation_template(
        continuation_specs=continuation_specs, target=target_name
    )
    if spec is None or template is None:
        return ()
    if every_leaf:
        paths: tuple[tuple[str, ...], ...] = tuple(template.leaves())
    else:
        published = frozenset(template.leaves())
        paths = tuple(
            path
            for path in HOST_READ_CONTINUATION_LEAVES.get(spec.artifact_key, ())
            if path in published
        )
    return tuple(
        ValueArtifactAddress(
            kind=ValueArtifactKind.CONTINUATION_LEAF,
            period=period,
            regime=target_name,
            artifact_key=spec.artifact_key,
            leaf_path=path,
        )
        for path in paths
    )
