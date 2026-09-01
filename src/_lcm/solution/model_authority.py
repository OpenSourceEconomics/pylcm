"""Model-owned descriptions of values and replay artifacts.

Public solution metadata describes transported arrays.  It is not an authority for
facts that belong to the model which produced and later consumes those arrays.  This
module derives those facts again from the canonical model and canonical parameters so
labelled-result preflight has one immutable source for shapes, dtypes, routes, and
applicability.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from jax import Array

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.engine import EGMPolicyRead, NNBEGMPolicyRead, Regime
from _lcm.regime_building.gated_edges import (
    edge_may_fold_at_period,
    gate_reads_dissolution_flag,
    source_reads_folded_wbar,
)
from _lcm.solution.artifacts import _canonical_value_axis_names
from _lcm.solution.contract import BackwardInductionResult
from _lcm.solution.nnbegm import derive_nnbegm_replay_capability
from _lcm.solution.v_topology import _get_regime_V_shapes_and_shardings
from _lcm.typing import FlatParams, RegimeName
from lcm.ages import AgeGrid
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    SIMULATION_POLICY,
    ArtifactKey,
    ArtifactRef,
)


@dataclass(frozen=True, kw_only=True)
class ValueCellDescriptor:
    """Canonical representation of one model value cell."""

    payload_type: type[object]
    shape: tuple[int, ...]
    dtype: str
    axis_names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ReplayCellDescriptor:
    """Canonical representation and route of one replay-artifact cell."""

    ref: ArtifactRef
    payload_type: type[object] | tuple[type[object], ...] | None
    route: EGMPolicyRead | NNBEGMPolicyRead | None
    shape: tuple[int, ...] | None
    dtype: str | None
    applicable: bool
    required: bool
    consumer_route: str | None
    expected_replay_capability: OuterReplayCapability | None = None
    egm_node_count: int | None = None
    adaptive_outer_nodes: tuple[float, ...] | None = None
    channel: str = "replay_artifacts"


@dataclass(frozen=True, kw_only=True)
class SolutionAuthority:
    """Immutable model-owned descriptions for every active solution cell."""

    values: MappingProxyType[tuple[int, RegimeName], ValueCellDescriptor]
    replay: MappingProxyType[ArtifactRef, ReplayCellDescriptor]

    def refs_for_key(self, key: ArtifactKey) -> tuple[ArtifactRef, ...]:
        """Return active cells for one exact, versioned artifact identity."""
        return tuple(ref for ref in self.replay if ref.key == key)


def build_solution_authority(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
) -> SolutionAuthority:
    """Derive active value and replay descriptions solely from the model.

    Runtime-supplied grid points are already present in ``flat_params``.  Resolving the
    state-action spaces here therefore gives the same exact axis lengths as the period
    kernels without consulting any returned value or replay payload.
    """
    canonical_float = str(np.dtype(canonical_float_dtype()))
    topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes, flat_params=flat_params
    )
    values: dict[tuple[int, RegimeName], ValueCellDescriptor] = {}
    replay: dict[ArtifactRef, ReplayCellDescriptor] = {}
    required_dissolution_cells = _required_dissolution_cells(regimes=regimes)

    for regime_name, regime in regimes.items():
        value_descriptor = ValueCellDescriptor(
            payload_type=Array,
            shape=topology[regime_name].shape,
            dtype=canonical_float,
            axis_names=_canonical_value_axis_names(regime=regime),
        )
        policy_read = regime.simulation.egm_policy_read
        policy_type: type[object] | None
        policy_applicable = False
        policy_required = False
        consumer_route: str | None = None
        if isinstance(policy_read, EGMPolicyRead):
            policy_type = EGMSimPolicy
            policy_applicable = True
            policy_required = True
            consumer_route = "egm_off_grid"
        elif isinstance(policy_read, NNBEGMPolicyRead):
            policy_type = (
                NestedEGMSimPolicy
                if policy_read.replay_policy_is_nested
                else NNBEGMSimPolicy
            )
            policy_applicable = policy_read.policy_applicable
            policy_required = policy_read.policy_required
            consumer_route = (
                "nnbegm_nested"
                if policy_read.replay_policy_is_nested
                else "nnbegm_finite"
            )
        else:
            policy_type = None

        flag_applicable = regime.stakeholders is not None
        flag_shape = value_descriptor.shape[:-1] if flag_applicable else None
        state_action_space = (
            regime.solution.state_action_space(regime_params=flat_params[regime_name])
            if isinstance(policy_read, NNBEGMPolicyRead)
            else None
        )
        for period in regime.active_periods:
            coordinate = (period, regime_name)
            values[coordinate] = value_descriptor
            policy_ref = ArtifactRef(
                period=period,
                regime=regime_name,
                key=SIMULATION_POLICY,
            )
            policy_shape, egm_node_count = _policy_shape_and_node_count(
                regime=regime,
                policy_read=policy_read,
                period=period,
            )
            expected_capability = (
                derive_nnbegm_replay_capability(
                    period_kernel=regime.solution.period_kernels[period],
                    state_action_space=state_action_space,
                    flat_params=flat_params,
                    period=period,
                    ages=ages,
                )
                if isinstance(policy_read, NNBEGMPolicyRead)
                and state_action_space is not None
                else None
            )
            replay[policy_ref] = ReplayCellDescriptor(
                ref=policy_ref,
                payload_type=policy_type,
                route=policy_read,
                shape=policy_shape,
                dtype=canonical_float if policy_type is not None else None,
                applicable=policy_applicable,
                required=policy_required,
                consumer_route=consumer_route,
                expected_replay_capability=expected_capability,
                egm_node_count=egm_node_count,
            )
            flag_ref = ArtifactRef(
                period=period,
                regime=regime_name,
                key=DISSOLUTION_FLAG,
            )
            replay[flag_ref] = ReplayCellDescriptor(
                ref=flag_ref,
                payload_type=Array if flag_applicable else None,
                route=None,
                shape=flag_shape,
                dtype="bool" if flag_applicable else None,
                applicable=flag_applicable,
                required=coordinate in required_dissolution_cells,
                consumer_route=("gated_edge_dissolution" if flag_applicable else None),
            )

    return SolutionAuthority(
        values=MappingProxyType(values),
        replay=MappingProxyType(replay),
    )


def bind_generated_solution_authority(
    *,
    authority: SolutionAuthority,
    internal_result: BackwardInductionResult,
) -> SolutionAuthority:
    """Bind data-dependent axis coordinates before a result leaves the model.

    Adaptive NNBEGM decides its final shared outer mesh from exact solves, so its
    candidate coordinates cannot be reconstructed from declarations alone without
    repeating the solve. The producing kernel emits those coordinates on a private
    sidecar beside the replay payload. The model records that sidecar in immutable
    authority; the returned payload and metadata do not carry the trusted copy.
    """
    replay = dict(authority.replay)
    bound_refs: set[ArtifactRef] = set()
    for (
        period,
        regime_to_authority,
    ) in internal_result.generated_replay_authorities.items():
        for regime_name, generated in regime_to_authority.items():
            ref = ArtifactRef(
                period=period,
                regime=regime_name,
                key=SIMULATION_POLICY,
            )
            descriptor = replay[ref]
            if (
                descriptor.payload_type is None
                or descriptor.consumer_route != "nnbegm_nested"
                or not descriptor.applicable
            ):
                msg = (
                    "Generated adaptive replay authority has no applicable nested "
                    f"model route at ({period}, {regime_name!r})."
                )
                raise TypeError(msg)
            replay[ref] = replace(
                descriptor,
                adaptive_outer_nodes=generated.adaptive_outer_nodes,
            )
            bound_refs.add(ref)
    expected_refs = {
        ArtifactRef(period=period, regime=regime_name, key=SIMULATION_POLICY)
        for period, regime_to_policy in internal_result.simulation_policies.items()
        for regime_name in regime_to_policy
        if authority.replay[
            ArtifactRef(
                period=period,
                regime=regime_name,
                key=SIMULATION_POLICY,
            )
        ].consumer_route
        == "nnbegm_nested"
    }
    if bound_refs != expected_refs:
        missing = tuple(sorted(expected_refs - bound_refs))
        unexpected = tuple(sorted(bound_refs - expected_refs))
        msg = (
            "Generated adaptive replay authority coverage differs from published "
            f"nested policies: missing={missing}, unexpected={unexpected}."
        )
        raise TypeError(msg)
    return replace(authority, replay=MappingProxyType(replay))


def _policy_shape_and_node_count(
    *,
    regime: Regime,
    policy_read: EGMPolicyRead | NNBEGMPolicyRead | None,
    period: int,
) -> tuple[tuple[int, ...] | None, int | None]:
    """Return model-owned primary-array shape and EGM trailing-node length."""
    if isinstance(policy_read, EGMPolicyRead):
        template = regime.solution.continuation_template
        if not isinstance(template, EGMCarry):
            msg = "An EGM replay route has no model-owned EGM carry template."
            raise TypeError(msg)
        node_count = int(template.value.shape[-1])
        return (*policy_read.row_axis_lengths_by_period[period], node_count), node_count
    if not isinstance(policy_read, NNBEGMPolicyRead):
        return None, None
    if not policy_read.replay_policy_is_nested:
        if policy_read.candidate_count is None:
            msg = "A finite NNBEGM replay route has no model-owned candidate count."
            raise TypeError(msg)
        return (
            policy_read.candidate_count,
            *policy_read.state_axis_lengths_by_period[period],
        ), None
    if policy_read.liquid_state_name is None:
        msg = "A nested NNBEGM replay route has no model-owned liquid-state role."
        raise TypeError(msg)
    liquid_position = policy_read.state_names.index(policy_read.liquid_state_name)
    node_count = policy_read.state_axis_lengths_by_period[period][liquid_position]
    return (*policy_read.row_axis_lengths_by_period[period], node_count), node_count


def _required_dissolution_cells(
    *, regimes: MappingProxyType[RegimeName, Regime]
) -> frozenset[tuple[int, RegimeName]]:
    """Return every target cell whose flag a canonical gated route consumes."""
    solved_by_period: dict[int, set[RegimeName]] = {}
    for regime_name, regime in regimes.items():
        for period in regime.active_periods:
            solved_by_period.setdefault(period, set()).add(regime_name)

    required: set[tuple[int, RegimeName]] = set()
    for source_name, source in regimes.items():
        for edge in source.gated_edges.values():
            if not gate_reads_dissolution_flag(edge=edge):
                continue
            for period, solved_regimes in solved_by_period.items():
                source_reads = source_reads_folded_wbar(
                    source_active_periods=source.active_periods,
                    fold_period=period,
                )
                if not source_reads:
                    continue
                if edge_may_fold_at_period(
                    edge=edge,
                    source_name=source_name,
                    fold_period=period,
                    solved_regimes=solved_regimes,
                    source_reads_wbar=True,
                ):
                    required.add((period, edge.target))
    return frozenset(required)


__all__ = [
    "ReplayCellDescriptor",
    "SolutionAuthority",
    "ValueCellDescriptor",
    "bind_generated_solution_authority",
    "build_solution_authority",
]
