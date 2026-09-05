"""Model-owned descriptions of values and replay artifacts.

Public solution metadata describes transported arrays.  It is not an authority for
facts that belong to the model which produced and later consumes those arrays.  This
module derives those facts again from the canonical model and canonical parameters so
labelled-result preflight has one immutable source for shapes, dtypes, routes, and
applicability.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from _lcm.continuation import ContinuationSpec, EGMContinuationSpec
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy, OuterPolicyBank
from _lcm.egm.outer_inversion import DeclaredOuterInverse
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.engine import EGMPolicyRead, NNBEGMPolicyRead, Regime, StateActionSpace
from _lcm.execution.core_program import (
    core_program_graph,
    retained_artifact_payload_type,
)
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.grids.discrete import DiscreteGrid
from _lcm.regime_building.gated_edges import (
    edge_may_fold_at_period,
    gate_reads_dissolution_flag,
    source_reads_folded_wbar,
)
from _lcm.solution.artifacts import _canonical_value_axis_names
from _lcm.solution.contract import BackwardInductionResult
from _lcm.solution.nnbegm import derive_nnbegm_replay_capability
from _lcm.solution.result_snapshot import (
    snapshot_artifact_authorities,
    snapshot_artifact_descriptor,
    snapshot_artifact_ref,
)
from _lcm.solution.v_topology import _get_regime_V_shapes_and_shardings
from _lcm.typing import (
    ContinuousState,
    DiscreteState,
    FlatParams,
    RegimeName,
    StateName,
)
from lcm.ages import AgeGrid
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    SIMULATION_POLICY,
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
    ArtifactRef,
    AxisAuthority,
    AxisRole,
    CategoryDomain,
    ExecutableReplayRoute,
    LeafAuthority,
    PersistencePolicy,
    ReplayModelContext,
    ReplayRouteIdentity,
    ReplayRouteRequirements,
    TreePath,
    _artifact_authority_from_template_snapshot,
    _artifact_authority_template_snapshot,
    _CanonicalArtifactTemplate,
    _canonicalize_artifact_payload,
    _normalize_jax_tree_path,
    _replay_route_identity,
    _same_exact_artifact_contract,
    _snapshot_artifact_template_once,
)

_EGM_CONTINUATION_ROUTE = ReplayRouteIdentity(
    route_id="pylcm.egm_continuation",
    route_version=1,
)
_DISSOLUTION_ROUTE = ReplayRouteIdentity(
    route_id="pylcm.gated_edge_dissolution",
    route_version=1,
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
    artifacts: MappingProxyType[ArtifactRef, ArtifactAuthority] = MappingProxyType({})
    """Generic model-built authorities for independently retained artifacts."""
    artifact_descriptors: MappingProxyType[ArtifactRef, ArtifactDescriptor] = (
        MappingProxyType({})
    )
    """Independently reconstructible public schemas, excluding private axes."""

    def refs_for_key(self, key: ArtifactKey) -> tuple[ArtifactRef, ...]:
        """Return active cells for one exact, versioned artifact identity."""
        return tuple(ref for ref in self.replay if ref.key == key)


def snapshot_solution_authority(authority: SolutionAuthority) -> SolutionAuthority:
    """Detach every authority wrapper reused after a lazy value callback."""
    if type(authority) is not SolutionAuthority:
        raise TypeError("Solution authority must be exact SolutionAuthority.")
    for name, mapping in (
        ("values", authority.values),
        ("replay", authority.replay),
        ("artifacts", authority.artifacts),
        ("artifact_descriptors", authority.artifact_descriptors),
    ):
        if type(mapping) is not MappingProxyType:
            raise TypeError(f"Solution authority {name} must be immutable and exact.")

    values: dict[tuple[int, RegimeName], ValueCellDescriptor] = {}
    for coordinate, descriptor in authority.values.items():
        copied_coordinate = _snapshot_authority_value_coordinate(coordinate)
        values[copied_coordinate] = _snapshot_value_cell_descriptor(descriptor)
    replay = {
        snapshot_artifact_ref(ref): _snapshot_replay_cell_descriptor(descriptor)
        for ref, descriptor in authority.replay.items()
    }
    artifacts = snapshot_artifact_authorities(authority.artifacts)
    artifact_descriptors = MappingProxyType(
        {
            snapshot_artifact_ref(ref): snapshot_artifact_descriptor(descriptor)
            for ref, descriptor in authority.artifact_descriptors.items()
        }
    )
    if (
        len(values) != len(authority.values)
        or len(replay) != len(authority.replay)
        or len(artifacts) != len(authority.artifacts)
        or len(artifact_descriptors) != len(authority.artifact_descriptors)
    ):
        raise ValueError("Solution authority addresses collide after reconstruction.")
    if any(
        not _same_exact_artifact_contract(actual=ref, expected=descriptor.ref)
        for ref, descriptor in replay.items()
    ):
        raise ValueError("Replay authority keys differ from descriptor addresses.")
    if any(
        not _same_exact_artifact_contract(actual=ref.key, expected=descriptor.key)
        for ref, descriptor in artifact_descriptors.items()
    ):
        raise ValueError("Artifact descriptor keys differ from authority addresses.")
    if any(
        not _same_exact_artifact_contract(
            actual=ref.key,
            expected=artifact.descriptor.key,
        )
        for ref, artifact in artifacts.items()
    ):
        raise ValueError("Artifact authority keys differ from descriptor identities.")
    return SolutionAuthority(
        values=MappingProxyType(values),
        replay=MappingProxyType(replay),
        artifacts=artifacts,
        artifact_descriptors=artifact_descriptors,
    )


def _snapshot_value_cell_descriptor(
    descriptor: ValueCellDescriptor,
) -> ValueCellDescriptor:
    if type(descriptor) is not ValueCellDescriptor:
        raise TypeError("Value authority must use exact ValueCellDescriptor objects.")
    if not isinstance(descriptor.payload_type, type):
        raise TypeError("Value authority payload_type must be a type.")
    _require_exact_shape(value=descriptor.shape, label="value authority shape")
    _require_nonempty_exact_str(value=descriptor.dtype, label="value authority dtype")
    _require_exact_names(
        value=descriptor.axis_names, label="value authority axis_names"
    )
    if len(descriptor.axis_names) != len(descriptor.shape):
        raise ValueError("Value authority axis_names must name every dimension.")
    return ValueCellDescriptor(
        payload_type=descriptor.payload_type,
        shape=tuple(descriptor.shape),
        dtype=descriptor.dtype,
        axis_names=tuple(descriptor.axis_names),
    )


def _snapshot_authority_value_coordinate(coordinate: object) -> tuple[int, RegimeName]:
    if type(coordinate) is not tuple or len(coordinate) != 2:  # noqa: PLR2004
        raise TypeError("Value authority coordinates must be exact pairs.")
    period, regime_name = coordinate
    _require_nonnegative_exact_int(value=period, label="value authority period")
    _require_nonempty_exact_str(value=regime_name, label="value authority regime")
    return cast("int", period), cast("RegimeName", regime_name)


def _snapshot_replay_cell_descriptor(
    descriptor: ReplayCellDescriptor,
) -> ReplayCellDescriptor:
    if type(descriptor) is not ReplayCellDescriptor:
        raise TypeError("Replay authority must use exact ReplayCellDescriptor objects.")
    _validate_replay_cell_descriptor(descriptor)

    route = descriptor.route
    if type(route) is EGMPolicyRead:
        copied_route: EGMPolicyRead | NNBEGMPolicyRead | None = (
            _snapshot_egm_policy_read(route)
        )
    elif type(route) is NNBEGMPolicyRead:
        copied_route = _snapshot_nnbegm_policy_read(route)
    else:
        copied_route = None

    capability = descriptor.expected_replay_capability
    copied_capability = (
        None if capability is None else _snapshot_outer_replay_capability(capability)
    )
    payload_type = descriptor.payload_type
    copied_payload_type = (
        tuple(payload_type) if type(payload_type) is tuple else payload_type
    )
    return ReplayCellDescriptor(
        ref=snapshot_artifact_ref(descriptor.ref),
        payload_type=copied_payload_type,
        route=copied_route,
        shape=None if descriptor.shape is None else tuple(descriptor.shape),
        dtype=descriptor.dtype,
        applicable=descriptor.applicable,
        required=descriptor.required,
        consumer_route=descriptor.consumer_route,
        expected_replay_capability=copied_capability,
        egm_node_count=descriptor.egm_node_count,
        adaptive_outer_nodes=(
            None
            if descriptor.adaptive_outer_nodes is None
            else tuple(descriptor.adaptive_outer_nodes)
        ),
        channel=descriptor.channel,
    )


def _validate_replay_cell_descriptor(  # noqa: C901
    descriptor: ReplayCellDescriptor,
) -> None:
    """Validate the unguarded replay-cell dataclass before reconstructing it."""
    if type(descriptor.ref) is not ArtifactRef:
        raise TypeError("Replay authority ref must be exact ArtifactRef.")
    payload_type = descriptor.payload_type
    if type(payload_type) is tuple:
        if any(not isinstance(runtime_type, type) for runtime_type in payload_type):
            raise TypeError("Replay payload type tuple must contain only types.")
    elif payload_type is not None and not isinstance(payload_type, type):
        raise TypeError("Replay payload_type must be a type, exact tuple, or None.")
    route_type = type(descriptor.route)
    if descriptor.route is not None and (
        route_type is not EGMPolicyRead and route_type is not NNBEGMPolicyRead
    ):
        raise TypeError("Replay authority has an unsupported route wrapper.")
    if descriptor.shape is not None:
        _require_exact_shape(value=descriptor.shape, label="replay authority shape")
    if descriptor.dtype is not None:
        _require_nonempty_exact_str(
            value=descriptor.dtype, label="replay authority dtype"
        )
    if type(descriptor.applicable) is not bool or type(descriptor.required) is not bool:
        raise TypeError("Replay applicability and requiredness must be exact bools.")
    if descriptor.consumer_route is not None:
        _require_nonempty_exact_str(
            value=descriptor.consumer_route,
            label="replay consumer_route",
        )
    if (
        descriptor.expected_replay_capability is not None
        and type(descriptor.expected_replay_capability) is not OuterReplayCapability
    ):
        raise TypeError("Expected replay capability must be exact.")
    if descriptor.egm_node_count is not None:
        _require_nonnegative_exact_int(
            value=descriptor.egm_node_count,
            label="replay egm_node_count",
        )
    if descriptor.adaptive_outer_nodes is not None:
        _require_exact_floats(
            value=descriptor.adaptive_outer_nodes,
            label="replay adaptive_outer_nodes",
        )
    _require_nonempty_exact_str(value=descriptor.channel, label="replay channel")


def _snapshot_egm_policy_read(route: EGMPolicyRead) -> EGMPolicyRead:
    """Copy an ordinary EGM route after validating every structural field."""
    _require_nonempty_exact_str(value=route.action_name, label="EGM action_name")
    _require_nonempty_exact_str(
        value=route.resources_target, label="EGM resources_target"
    )
    _require_exact_float(
        value=route.savings_lower_bound, label="EGM savings_lower_bound"
    )
    _require_exact_names(
        value=route.row_discrete_state_names,
        label="EGM row_discrete_state_names",
    )
    _require_exact_names(
        value=route.row_passive_state_names,
        label="EGM row_passive_state_names",
    )
    _require_exact_names(
        value=route.row_discrete_action_names,
        label="EGM row_discrete_action_names",
    )
    _require_exact_period_lengths(
        value=route.row_axis_lengths_by_period,
        label="EGM row_axis_lengths_by_period",
    )
    _require_nonempty_exact_str(value=route.float_dtype, label="EGM float_dtype")
    return replace(
        route,
        row_discrete_state_names=tuple(route.row_discrete_state_names),
        row_passive_state_names=tuple(route.row_passive_state_names),
        row_discrete_action_names=tuple(route.row_discrete_action_names),
        row_axis_lengths_by_period=MappingProxyType(
            {
                period: tuple(lengths)
                for period, lengths in route.row_axis_lengths_by_period.items()
            }
        ),
    )


def _snapshot_nnbegm_policy_read(route: NNBEGMPolicyRead) -> NNBEGMPolicyRead:
    """Copy NNBEGM's wrapper while preserving its model-owned callables."""
    _validate_nnbegm_policy_read(route)
    return replace(
        route,
        outer_target_function_by_period=MappingProxyType(
            dict(route.outer_target_function_by_period)
        ),
        state_names=tuple(route.state_names),
        state_axis_lengths_by_period=MappingProxyType(
            {
                period: tuple(lengths)
                for period, lengths in route.state_axis_lengths_by_period.items()
            }
        ),
        row_discrete_state_names=tuple(route.row_discrete_state_names),
        row_passive_state_names=tuple(route.row_passive_state_names),
        row_axis_lengths_by_period=MappingProxyType(
            {
                period: tuple(lengths)
                for period, lengths in route.row_axis_lengths_by_period.items()
            }
        ),
        discrete_action_names=tuple(route.discrete_action_names),
        discrete_action_code_domains=MappingProxyType(
            {
                name: tuple(codes)
                for name, codes in route.discrete_action_code_domains.items()
            }
        ),
        candidate_discrete_action_codes=tuple(
            tuple(codes) for codes in route.candidate_discrete_action_codes
        ),
        outer_grid_values=(
            None if route.outer_grid_values is None else tuple(route.outer_grid_values)
        ),
        outer_state_domain_by_period=MappingProxyType(
            {
                period: tuple(bounds)
                for period, bounds in route.outer_state_domain_by_period.items()
            }
        ),
    )


def _validate_nnbegm_policy_read(  # noqa: C901, PLR0912
    route: NNBEGMPolicyRead,
) -> None:
    """Validate NNBEGM route structure before copying mappings or tuples."""
    if type(route) is not NNBEGMPolicyRead:
        raise TypeError("NNBEGM replay route must be exact.")
    _require_exact_mapping(
        value=route.outer_target_function_by_period,
        label="NNBEGM outer_target_function_by_period",
    )
    for period, function in route.outer_target_function_by_period.items():
        _require_nonnegative_exact_int(value=period, label="NNBEGM target period")
        if not callable(function):
            raise TypeError("NNBEGM outer target entries must be callable.")

    for field_name in (
        "outer_post_decision",
        "outer_state_name",
        "inner_action_name",
        "outer_action_name",
        "float_dtype",
        "integer_dtype",
    ):
        _require_nonempty_exact_str(
            value=getattr(route, field_name),
            label=f"NNBEGM {field_name}",
        )
    for field_name in (
        "outer_no_adjustment_target",
        "liquid_state_name",
        "resources_target",
    ):
        value = getattr(route, field_name)
        if value is not None:
            _require_nonempty_exact_str(value=value, label=f"NNBEGM {field_name}")

    for field_name in (
        "state_names",
        "row_discrete_state_names",
        "row_passive_state_names",
        "discrete_action_names",
    ):
        _require_exact_names(
            value=getattr(route, field_name),
            label=f"NNBEGM {field_name}",
        )
    _require_exact_period_lengths(
        value=route.state_axis_lengths_by_period,
        label="NNBEGM state_axis_lengths_by_period",
    )
    _require_exact_period_lengths(
        value=route.row_axis_lengths_by_period,
        label="NNBEGM row_axis_lengths_by_period",
    )

    _require_exact_mapping(
        value=route.discrete_action_code_domains,
        label="NNBEGM discrete_action_code_domains",
    )
    for name, codes in route.discrete_action_code_domains.items():
        _require_nonempty_exact_str(value=name, label="NNBEGM discrete action name")
        _require_exact_ints(value=codes, label="NNBEGM discrete action codes")

    _require_exact_tuple(
        value=route.candidate_discrete_action_codes,
        label="NNBEGM candidate_discrete_action_codes",
    )
    for codes in route.candidate_discrete_action_codes:
        _require_exact_ints(value=codes, label="NNBEGM candidate action codes")

    for field_name in (
        "candidate_count",
        "n_keeper_candidates",
        "golden_iterations",
    ):
        value = getattr(route, field_name)
        if value is not None:
            _require_nonnegative_exact_int(value=value, label=f"NNBEGM {field_name}")
    for field_name in (
        "savings_lower_bound",
        "value_atol",
        "value_rtol",
    ):
        value = getattr(route, field_name)
        if value is not None:
            _require_exact_float(value=value, label=f"NNBEGM {field_name}")

    if route.outer_grid_values is not None:
        _require_exact_floats(
            value=route.outer_grid_values,
            label="NNBEGM outer_grid_values",
        )
    _require_exact_mapping(
        value=route.outer_state_domain_by_period,
        label="NNBEGM outer_state_domain_by_period",
    )
    for period, bounds in route.outer_state_domain_by_period.items():
        _require_nonnegative_exact_int(value=period, label="NNBEGM domain period")
        _require_exact_floats(value=bounds, label="NNBEGM outer state bounds")
        if len(bounds) != 2:  # noqa: PLR2004
            raise ValueError("NNBEGM outer state bounds must contain two endpoints.")

    for field_name in (
        "policy_applicable",
        "policy_required",
        "fixed_cost_simulation_unsupported",
        "replay_policy_is_nested",
    ):
        if type(getattr(route, field_name)) is not bool:
            raise TypeError(f"NNBEGM {field_name} must be an exact bool.")


def _snapshot_outer_replay_capability(
    capability: OuterReplayCapability,
) -> OuterReplayCapability:
    if type(capability) is not OuterReplayCapability:
        raise TypeError("Outer replay capability must be exact.")
    inverse = _snapshot_declared_outer_inverse(capability.inverse)
    _require_exact_names(
        value=capability.undeclared_functions,
        label="outer replay undeclared_functions",
    )
    _require_exact_tuple(
        value=capability.unbindable_functions,
        label="outer replay unbindable_functions",
    )
    unbindable: list[tuple[str, tuple[str, ...]]] = []
    for entry in capability.unbindable_functions:
        if type(entry) is not tuple or len(entry) != 2:  # noqa: PLR2004
            raise TypeError("Outer replay unbindable entries must be exact pairs.")
        name, arguments = entry
        _require_nonempty_exact_str(value=name, label="outer replay function name")
        _require_exact_names(value=arguments, label="outer replay unbindable arguments")
        unbindable.append((name, tuple(arguments)))
    _require_exact_names(
        value=capability.unavailable_keeper_states,
        label="outer replay unavailable_keeper_states",
    )
    _require_exact_names(
        value=capability.unaddressable_passive_states,
        label="outer replay unaddressable_passive_states",
    )
    _require_exact_names(
        value=capability.unaddressable_discrete_actions,
        label="outer replay unaddressable_discrete_actions",
    )
    return OuterReplayCapability(
        inverse=inverse,
        undeclared_functions=tuple(capability.undeclared_functions),
        unbindable_functions=tuple(unbindable),
        unavailable_keeper_states=tuple(capability.unavailable_keeper_states),
        unaddressable_passive_states=tuple(capability.unaddressable_passive_states),
        unaddressable_discrete_actions=tuple(capability.unaddressable_discrete_actions),
    )


def _snapshot_declared_outer_inverse(
    inverse: DeclaredOuterInverse,
) -> DeclaredOuterInverse:
    if type(inverse) is not DeclaredOuterInverse:
        raise TypeError("Declared outer inverse must be exact.")
    coefficient = inverse.coefficient
    if type(coefficient) is not Fraction:
        raise TypeError("Declared outer coefficient must be an exact Fraction.")
    numerator = coefficient.numerator
    denominator = coefficient.denominator
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise TypeError(
            "Declared outer coefficient fields must be canonical exact ints."
        )
    canonical = Fraction(numerator, denominator)
    if canonical.numerator != numerator or canonical.denominator != denominator:
        raise ValueError("Declared outer coefficient must be in canonical form.")
    _require_exact_float(value=inverse.low, label="declared outer lower bound")
    _require_exact_float(value=inverse.high, label="declared outer upper bound")
    return DeclaredOuterInverse(
        coefficient=canonical,
        low=inverse.low,
        high=inverse.high,
    )


def _require_exact_mapping(*, value: object, label: str) -> None:
    if type(value) is not MappingProxyType:
        raise TypeError(f"{label} must be an immutable exact mapping.")


def _require_exact_tuple(*, value: object, label: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple.")


def _require_nonempty_exact_str(*, value: object, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact str.")
    if not value:
        raise ValueError(f"{label} must not be empty.")


def _require_nonnegative_exact_int(*, value: object, label: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int.")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative.")


def _require_exact_float(*, value: object, label: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{label} must be an exact float.")


def _require_exact_names(*, value: object, label: str) -> None:
    _require_exact_tuple(value=value, label=label)
    names = cast("tuple[object, ...]", value)
    if any(type(name) is not str or not name for name in names):
        raise TypeError(f"{label} must contain nonempty exact strs.")


def _require_exact_shape(*, value: object, label: str) -> None:
    _require_exact_tuple(value=value, label=label)
    shape = cast("tuple[object, ...]", value)
    if any(type(size) is not int for size in shape):
        raise TypeError(f"{label} must contain exact ints.")
    if any(cast("int", size) < 0 for size in shape):
        raise ValueError(f"{label} must contain nonnegative sizes.")


def _require_exact_ints(*, value: object, label: str) -> None:
    _require_exact_tuple(value=value, label=label)
    items = cast("tuple[object, ...]", value)
    if any(type(item) is not int for item in items):
        raise TypeError(f"{label} must contain exact ints.")


def _require_exact_floats(*, value: object, label: str) -> None:
    _require_exact_tuple(value=value, label=label)
    items = cast("tuple[object, ...]", value)
    if any(type(item) is not float for item in items):
        raise TypeError(f"{label} must contain exact floats.")


def _require_exact_period_lengths(*, value: object, label: str) -> None:
    _require_exact_mapping(value=value, label=label)
    mapping = cast("MappingProxyType[object, object]", value)
    for period, lengths in mapping.items():
        _require_nonnegative_exact_int(value=period, label=f"{label} period")
        _require_exact_shape(value=lengths, label=f"{label} lengths")


@dataclass(frozen=True, kw_only=True)
class _ArtifactLayout:
    """Exact semantic layout supplied to the generic PyTree authority builder."""

    leaf_axis_names: dict[TreePath, tuple[str, ...]]
    axes: tuple[AxisAuthority, ...]
    state_roles: tuple[str, ...] = ()
    action_roles: tuple[str, ...] = ()


def build_solution_authority(  # noqa: PLR0915
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
    artifacts: dict[ArtifactRef, ArtifactAuthority] = {}
    required_dissolution_cells = _required_dissolution_cells(regimes=regimes)

    for regime_name, regime in regimes.items():
        value_descriptor = ValueCellDescriptor(
            payload_type=Array,
            shape=topology[regime_name].shape,
            dtype=canonical_float,
            axis_names=_canonical_value_axis_names(regime=regime),
        )
        # The regime's declared route is the sole source of what it publishes,
        # under which reader, and whether a solve owes it — never the concrete
        # payload class, which the authority describes rather than discovers.
        policy_read = regime.simulation.egm_policy_read
        route = regime.simulation.replay_route
        route_identity = _replay_route_identity(route)
        external_route = regime.simulation.external_replay_route
        policy_type = None if external_route is not None else route.payload_type
        policy_applicable = (
            False if external_route is not None else route.policy_applicable
        )
        policy_required = False if external_route is not None else route.policy_required
        consumer_route = None if external_route is not None else route.consumer_route

        flag_applicable = regime.stakeholders is not None
        flag_shape = value_descriptor.shape[:-1] if flag_applicable else None
        # Always resolve named state/action axes from canonical runtime parameters.
        # A plugin build context can carry placeholder nodes for runtime irregular
        # grids; those values are descriptive input, never model authority.
        base_state_action_space = regime.solution.state_action_space(
            regime_params=flat_params[regime_name]
        )
        for period in regime.active_periods:
            state_action_space = _state_action_space_for_period(
                regime=regime,
                base=base_state_action_space,
                period=period,
            )
            producer_payload_types = _period_artifact_payload_types(
                regime=regime,
                period=period,
            )
            custom_authorities = {
                key: _bind_model_owned_artifact_facts(
                    authority=custom_authority,
                    regime=regime,
                    state_action_space=state_action_space,
                )
                for key, custom_authority in (
                    regime.solution.artifact_authorities.items()
                )
            }
            _validate_custom_continuation_authorities(
                regime_name=regime_name,
                period=period,
                continuation_spec=regime.solution.continuation_spec,
                custom_authorities=custom_authorities,
            )
            _validate_artifact_producer_types(
                regime_name=regime_name,
                period=period,
                producer_payload_types=producer_payload_types,
                custom_authorities=custom_authorities,
                built_in_policy_type=policy_type,
                external_route=external_route,
            )
            _validate_external_route_authorities(
                external_route=external_route,
                authorities=custom_authorities,
                producer_payload_types=producer_payload_types,
                context=_replay_model_context_from_state_action_space(
                    regime_name=regime_name,
                    period=period,
                    state_action_space=state_action_space,
                ),
            )
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
            if policy_type is not None:
                policy_persistence, policy_template = _policy_persistence_and_template(
                    policy_read=policy_read,
                    policy_shape=policy_shape,
                    expected_replay_capability=expected_capability,
                )
                if policy_template is None:
                    policy_snapshot = None
                    policy_containers = {}
                else:
                    policy_snapshot, policy_containers = (
                        _snapshot_artifact_template_once(
                            template=policy_template,
                            payload_runtime_type=policy_type,
                        )
                    )
                policy_layout = _policy_artifact_layout(
                    policy_read=policy_read,
                    template_snapshot=policy_snapshot,
                    period=period,
                )
                policy_authority = _authority_from_observed_template(
                    key=SIMULATION_POLICY,
                    channel=ArtifactChannel.REPLAY,
                    persistence=policy_persistence,
                    payload_runtime_type=policy_type,
                    template_snapshot=policy_snapshot,
                    container_runtime_types=policy_containers,
                    leaf_axis_names=policy_layout.leaf_axis_names,
                    axes=policy_layout.axes,
                    state_roles=policy_layout.state_roles,
                    action_roles=policy_layout.action_roles,
                    consumer_route=route_identity,
                    applicable=policy_applicable,
                    required=policy_required,
                )
                artifacts[policy_ref] = _bind_model_owned_artifact_facts(
                    authority=policy_authority,
                    regime=regime,
                    state_action_space=state_action_space,
                )
            elif (
                dormant_policy_type := producer_payload_types.get(SIMULATION_POLICY)
            ) is not None:
                # Some solve kernels deliberately compute an off-grid policy while
                # their simulation route recomputes on the action grid. The payload
                # is therefore structurally identifiable but inapplicable, and the
                # result ledger records NOT_APPLICABLE. Give that omission the same
                # exact descriptor coverage as every other artifact cell.
                policy_layout = _policy_artifact_layout(
                    policy_read=policy_read,
                    template_snapshot=None,
                    period=period,
                )
                dormant_authority = _authority_from_template(
                    key=SIMULATION_POLICY,
                    channel=ArtifactChannel.REPLAY,
                    persistence=PersistencePolicy.NOT_PERSISTED,
                    payload_runtime_type=dormant_policy_type,
                    template=None,
                    leaf_axis_names=policy_layout.leaf_axis_names,
                    axes=policy_layout.axes,
                    state_roles=policy_layout.state_roles,
                    action_roles=policy_layout.action_roles,
                    consumer_route=None,
                    applicable=False,
                    required=False,
                )
                artifacts[policy_ref] = _bind_model_owned_artifact_facts(
                    authority=dormant_authority,
                    regime=regime,
                    state_action_space=state_action_space,
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
            if flag_applicable and flag_shape is not None:
                flag_axis_names = value_descriptor.axis_names[: len(flag_shape)]
                flag_axes = tuple(
                    AxisAuthority(
                        name=name,
                        length=length,
                        role=(
                            AxisRole.STATE
                            if name in state_action_space.states
                            else AxisRole.OTHER
                        ),
                        coordinates=(
                            ()
                            if name in state_action_space.states
                            else tuple(range(length))
                        ),
                    )
                    for name, length in zip(flag_axis_names, flag_shape, strict=True)
                )
                flag_authority = _authority_from_template(
                    key=DISSOLUTION_FLAG,
                    channel=ArtifactChannel.REPLAY,
                    persistence=PersistencePolicy.MODEL_VERIFIABLE,
                    payload_runtime_type=Array,
                    template=jnp.zeros(flag_shape, dtype=bool),
                    leaf_axis_names={(): flag_axis_names},
                    axes=flag_axes,
                    state_roles=tuple(
                        name
                        for name in flag_axis_names
                        if name in state_action_space.states
                    ),
                    consumer_route=_DISSOLUTION_ROUTE,
                    applicable=True,
                    required=coordinate in required_dissolution_cells,
                )
                artifacts[flag_ref] = _bind_model_owned_artifact_facts(
                    authority=flag_authority,
                    regime=regime,
                    state_action_space=state_action_space,
                )

            for key, custom_authority in custom_authorities.items():
                if key != custom_authority.descriptor.key:
                    raise TypeError(
                        "Artifact authority mapping key differs from its descriptor: "
                        f"{key.type_id!r}."
                    )
                custom_ref = ArtifactRef(
                    period=period,
                    regime=regime_name,
                    key=key,
                )
                artifacts[custom_ref] = custom_authority

            continuation_spec = regime.solution.continuation_spec
            if (
                continuation_spec is not None
                and continuation_spec.artifact_key not in custom_authorities
            ):
                continuation_ref = ArtifactRef(
                    period=period,
                    regime=regime_name,
                    key=continuation_spec.artifact_key,
                )
                continuation_template = continuation_spec.template
                continuation_persistence = (
                    PersistencePolicy.MODEL_VERIFIABLE
                    if isinstance(continuation_template, EGMCarry)
                    else PersistencePolicy.NOT_PERSISTED
                )
                continuation_snapshot, continuation_containers = (
                    _snapshot_artifact_template_once(
                        template=continuation_template,
                        payload_runtime_type=type(continuation_template),
                    )
                )
                continuation_layout = (
                    _egm_carry_artifact_layout(
                        regime=regime,
                        state_action_space=state_action_space,
                        spec=continuation_spec,
                        template_snapshot=continuation_snapshot,
                    )
                    if isinstance(continuation_spec, EGMContinuationSpec)
                    else None
                )
                continuation_authority = _authority_from_observed_template(
                    key=continuation_spec.artifact_key,
                    channel=ArtifactChannel.CONTINUATION,
                    persistence=continuation_persistence,
                    payload_runtime_type=type(continuation_template),
                    template_snapshot=continuation_snapshot,
                    container_runtime_types=continuation_containers,
                    leaf_axis_names=(
                        None
                        if continuation_layout is None
                        else continuation_layout.leaf_axis_names
                    ),
                    axes=(
                        None
                        if continuation_layout is None
                        else continuation_layout.axes
                    ),
                    state_roles=(
                        ()
                        if continuation_layout is None
                        else continuation_layout.state_roles
                    ),
                    action_roles=(
                        ()
                        if continuation_layout is None
                        else continuation_layout.action_roles
                    ),
                    consumer_route=(
                        _EGM_CONTINUATION_ROUTE
                        if continuation_layout is not None
                        else None
                    ),
                    applicable=True,
                    required=True,
                )
                artifacts[continuation_ref] = (
                    _bind_model_owned_artifact_facts(
                        authority=continuation_authority,
                        regime=regime,
                        state_action_space=state_action_space,
                    )
                    if continuation_layout is not None
                    else continuation_authority
                )

    return SolutionAuthority(
        values=MappingProxyType(values),
        replay=MappingProxyType(replay),
        artifacts=MappingProxyType(artifacts),
        artifact_descriptors=MappingProxyType(
            {ref: artifact.descriptor for ref, artifact in artifacts.items()}
        ),
    )


def _validate_custom_continuation_authorities(
    *,
    regime_name: RegimeName,
    period: int,
    continuation_spec: ContinuationSpec | None,
    custom_authorities: dict[ArtifactKey, ArtifactAuthority],
) -> None:
    """Require a custom continuation authority to agree with its producer spec.

    A custom authority under the spec's key replaces the generic authority the model
    would otherwise derive below. It may choose its persistence policy and add semantic
    axes, but it cannot change which continuation is produced or whether that payload is
    required. Structural validation compares container and leaf metadata only; template
    values are intentionally irrelevant.
    """
    declared_continuation_keys = {
        key
        for key, authority in custom_authorities.items()
        if authority.descriptor.channel is ArtifactChannel.CONTINUATION
    }
    if continuation_spec is None:
        if declared_continuation_keys:
            raise TypeError(
                "Custom CONTINUATION authorities require a matching ContinuationSpec "
                f"in regime {regime_name!r}, period {period}: "
                f"{tuple(sorted(declared_continuation_keys))!r}."
            )
        return

    continuation_key = continuation_spec.artifact_key
    unexpected = declared_continuation_keys - {continuation_key}
    if unexpected:
        raise TypeError(
            "Custom CONTINUATION authorities must use the ContinuationSpec key in "
            f"regime {regime_name!r}, period {period}; unexpected "
            f"{tuple(sorted(unexpected))!r}."
        )

    authority = custom_authorities.get(continuation_key)
    if authority is None:
        return
    defects = tuple(
        message
        for invalid, message in (
            (
                authority.descriptor.key != continuation_key,
                "descriptor key differs from the ContinuationSpec key",
            ),
            (
                authority.descriptor.channel is not ArtifactChannel.CONTINUATION,
                "descriptor channel is not CONTINUATION",
            ),
            (
                authority.payload_runtime_type is not type(continuation_spec.template),
                "payload runtime type differs from the ContinuationSpec template",
            ),
            (not authority.applicable, "authority is not applicable"),
            (not authority.required, "authority is not required"),
        )
        if invalid
    )
    if defects:
        raise TypeError(
            "Custom authority shadowing a ContinuationSpec is contradictory in regime "
            f"{regime_name!r}, period {period}: {'; '.join(defects)}."
        )

    try:
        _canonicalize_artifact_payload(
            payload=continuation_spec.template,
            authority=authority,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Custom authority shadowing a ContinuationSpec has incompatible template, "
            f"container, or leaf structure in regime {regime_name!r}, period "
            f"{period}: {error}."
        ) from error


def _validate_external_route_authorities(
    *,
    external_route: ExecutableReplayRoute | None,
    authorities: dict[ArtifactKey, ArtifactAuthority],
    producer_payload_types: MappingProxyType[ArtifactKey, type[object]],
    context: ReplayModelContext,
) -> None:
    """Bind an external route only to replay authorities built with the solver."""
    if external_route is None:
        return
    identity = external_route.identity
    if type(identity) is not ReplayRouteIdentity:
        raise TypeError("An external replay route must expose an exact route identity.")
    requirements = external_route.requirements(context=context)
    if type(requirements) is not ReplayRouteRequirements:
        raise TypeError(
            "An external replay route must return exact ReplayRouteRequirements."
        )
    if not requirements.required_artifacts:
        raise ValueError("An external replay route must require at least one artifact.")
    missing = requirements.required_artifacts - authorities.keys()
    if missing:
        raise TypeError(
            "An external replay route has no solver-built authority for required "
            f"artifacts {tuple(sorted(missing))!r}."
        )
    _validate_required_external_producers(
        required_artifacts=requirements.required_artifacts,
        producer_payload_types=producer_payload_types,
    )
    matching = tuple(authorities[key] for key in requirements.required_artifacts)
    if any(
        authority.descriptor.channel is not ArtifactChannel.REPLAY
        or authority.consumer_route != identity
        or not authority.required
        or not authority.applicable
        for authority in matching
    ):
        raise TypeError(
            "An external replay route's required artifacts must be required REPLAY "
            "authorities bound to its exact identity."
        )
    if not any(
        authority.payload_runtime_type is external_route.payload_type
        for authority in matching
    ):
        raise TypeError(
            "An external replay route's payload type differs from its solver-built "
            "replay authority."
        )
    if any(
        authority.required
        and authority.descriptor.required_for != frozenset({identity})
        for authority in matching
    ):
        raise TypeError(
            "A required external replay artifact does not name its consuming route."
        )
    route_bound = frozenset(
        key
        for key, authority in authorities.items()
        if authority.required and authority.consumer_route == identity
    )
    if route_bound != requirements.required_artifacts:
        raise TypeError(
            "An external replay route's requirements must exactly equal its "
            "required solver-built authority keys."
        )


def _validate_required_external_producers(
    *,
    required_artifacts: frozenset[ArtifactKey],
    producer_payload_types: MappingProxyType[ArtifactKey, type[object]],
) -> None:
    """Require each external-route input to be published by a core program."""
    missing = required_artifacts - producer_payload_types.keys()
    if missing:
        raise TypeError(
            "An external replay route has no CoreProgram producer for required "
            f"artifacts {tuple(sorted(missing))!r}."
        )


def _period_artifact_payload_types(
    *, regime: Regime, period: int
) -> MappingProxyType[ArtifactKey, type[object]]:
    """Return every exact retained payload type from one validated producer graph."""
    graph = core_program_graph(kernel=regime.solution.period_kernels[period])
    keys = sorted(
        {key for program in graph.values() for key in program.retained_artifact_keys}
    )
    payload_types: dict[ArtifactKey, type[object]] = {}
    for key in keys:
        payload_type = retained_artifact_payload_type(graph=graph, key=key)
        if payload_type is None:  # pragma: no cover - graph validation owns this case
            raise RuntimeError(
                "A validated producer graph lost an artifact payload type."
            )
        payload_types[key] = payload_type
    return MappingProxyType(payload_types)


def _validate_artifact_producer_types(
    *,
    regime_name: RegimeName,
    period: int,
    producer_payload_types: MappingProxyType[ArtifactKey, type[object]],
    custom_authorities: dict[ArtifactKey, ArtifactAuthority],
    built_in_policy_type: type[object] | None,
    external_route: ExecutableReplayRoute | None,
) -> None:
    """Require producer declarations to agree with every consuming authority."""
    declared_policy_type = producer_payload_types.get(SIMULATION_POLICY)
    if external_route is not None and declared_policy_type is not None:
        raise TypeError(
            "An external solver cannot retain the engine-owned SIMULATION_POLICY "
            f"artifact in regime {regime_name!r}, period {period}."
        )
    if built_in_policy_type is not None:
        if declared_policy_type is None:
            raise TypeError(
                "A built-in replay route has no CoreProgram producer for "
                f"SIMULATION_POLICY in regime {regime_name!r}, period {period}."
            )
        if declared_policy_type is not built_in_policy_type:
            raise TypeError(
                "The CoreProgram producer and built-in replay route disagree on the "
                f"SIMULATION_POLICY payload type in regime {regime_name!r}, period "
                f"{period}: producer={declared_policy_type.__name__!r}, "
                f"route={built_in_policy_type.__name__!r}."
            )

    custom_producer_types = {
        key: payload_type
        for key, payload_type in producer_payload_types.items()
        if key != SIMULATION_POLICY
    }
    missing_required_producers = sorted(
        key
        for key, authority in custom_authorities.items()
        if authority.required
        and authority.descriptor.channel is not ArtifactChannel.CONTINUATION
        and key not in custom_producer_types
    )
    if missing_required_producers:
        raise TypeError(
            "Custom artifact authorities have no CoreProgram producer for required "
            "non-continuation artifacts in regime "
            f"{regime_name!r}, period {period}: "
            f"{tuple(missing_required_producers)!r}."
        )
    missing_authorities = sorted(
        custom_producer_types.keys() - custom_authorities.keys()
    )
    if missing_authorities:
        raise TypeError(
            "CoreProgram producers have no solver-built artifact authority in regime "
            f"{regime_name!r}, period {period}: {tuple(missing_authorities)!r}."
        )
    disagreements = tuple(
        (key, producer_type, custom_authorities[key].payload_runtime_type)
        for key, producer_type in sorted(custom_producer_types.items())
        if producer_type is not custom_authorities[key].payload_runtime_type
    )
    if disagreements:
        raise TypeError(
            "CoreProgram producers disagree with solver-built artifact authorities "
            f"in regime {regime_name!r}, period {period}: {disagreements!r}."
        )


def _policy_artifact_layout(
    *,
    policy_read: EGMPolicyRead | NNBEGMPolicyRead | None,
    template_snapshot: _CanonicalArtifactTemplate | None,
    period: int,
) -> _ArtifactLayout:
    """Describe every built-in policy leaf with its mathematical axes."""
    if isinstance(policy_read, EGMPolicyRead):
        return _egm_policy_artifact_layout(
            policy_read=policy_read,
            template_snapshot=template_snapshot,
            period=period,
        )
    if isinstance(policy_read, NNBEGMPolicyRead):
        if policy_read.replay_policy_is_nested:
            return _nested_policy_artifact_layout(
                policy_read=policy_read,
                template_snapshot=template_snapshot,
                period=period,
                adaptive_outer_nodes=None,
            )
        return _finite_nnbegm_policy_artifact_layout(
            policy_read=policy_read,
            template_snapshot=template_snapshot,
            period=period,
        )
    return _ArtifactLayout(leaf_axis_names={}, axes=())


def _egm_policy_artifact_layout(
    *,
    policy_read: EGMPolicyRead,
    template_snapshot: _CanonicalArtifactTemplate | None,
    period: int,
) -> _ArtifactLayout:
    """Return the shared row layout of an ordinary EGM replay policy."""
    row_state_names = (
        *policy_read.row_discrete_state_names,
        *policy_read.row_passive_state_names,
    )
    row_action_names = policy_read.row_discrete_action_names
    row_names = (*row_state_names, *row_action_names)
    row_lengths = policy_read.row_axis_lengths_by_period[period]
    node_count = 0
    paths: tuple[TreePath, ...] = ()
    if template_snapshot is not None:
        paths = template_snapshot.leaf_paths
        shapes = tuple(tuple(leaf.shape) for leaf in template_snapshot.leaves)
        if not shapes or any(shape[:-1] != row_lengths for shape in shapes):
            raise TypeError("An EGM policy template has inconsistent row axes.")
        node_count = shapes[0][-1]
        if any(shape[-1] != node_count for shape in shapes):
            raise TypeError("An EGM policy template has inconsistent node axes.")
    axes = (
        *_model_role_axes(
            names=row_names,
            lengths=row_lengths,
            state_names=row_state_names,
            action_names=row_action_names,
        ),
        AxisAuthority(
            name="pylcm:egm:node",
            length=node_count,
            role=AxisRole.OTHER,
            coordinates=tuple(range(node_count)),
        ),
    )
    axis_names = (*row_names, "pylcm:egm:node")
    return _ArtifactLayout(
        leaf_axis_names=dict.fromkeys(paths, axis_names),
        axes=axes,
        state_roles=row_state_names,
        action_roles=_unique_names((policy_read.action_name, *row_action_names)),
    )


def _finite_nnbegm_policy_artifact_layout(
    *,
    policy_read: NNBEGMPolicyRead,
    template_snapshot: _CanonicalArtifactTemplate | None,
    period: int,
) -> _ArtifactLayout:
    """Return the exact finite joint-bank, outer-node, and code-column axes."""
    if template_snapshot is None or policy_read.candidate_count is None:
        raise TypeError("A finite NNBEGM policy requires a static template.")
    paths = template_snapshot.leaf_paths
    shapes = tuple(tuple(leaf.shape) for leaf in template_snapshot.leaves)
    expected_main_shape = (
        policy_read.candidate_count,
        *policy_read.state_axis_lengths_by_period[period],
    )
    if len(paths) not in {4, 5} or any(
        shape != expected_main_shape for shape in shapes[:3]
    ):
        raise TypeError("A finite NNBEGM policy has inconsistent candidate banks.")
    candidate_axis = "pylcm:nnbegm:joint_candidate"
    outer_axis = "pylcm:nnbegm:outer_node"
    field_axis = "pylcm:nnbegm:discrete_action_field"
    leaf_axes: dict[TreePath, tuple[str, ...]] = dict.fromkeys(
        paths[:3], (candidate_axis, *policy_read.state_names)
    )
    leaf_axes[paths[3]] = (outer_axis,)
    axes: list[AxisAuthority] = [
        AxisAuthority(
            name=candidate_axis,
            length=policy_read.candidate_count,
            role=AxisRole.CANDIDATE,
            coordinates=tuple(range(policy_read.candidate_count)),
        ),
        *_model_role_axes(
            names=policy_read.state_names,
            lengths=policy_read.state_axis_lengths_by_period[period],
            state_names=policy_read.state_names,
            action_names=(),
        ),
    ]
    outer_nodes = (
        () if policy_read.outer_grid_values is None else policy_read.outer_grid_values
    )
    if shapes[3] != (len(outer_nodes),):
        raise TypeError("A finite NNBEGM policy has a mismatched outer-node axis.")
    axes.append(
        AxisAuthority(
            name=outer_axis,
            length=len(outer_nodes),
            role=AxisRole.CANDIDATE,
            coordinates=tuple(outer_nodes),
        )
    )
    if len(paths) == 5:  # noqa: PLR2004
        expected_codes_shape = (
            policy_read.candidate_count,
            len(policy_read.discrete_action_names),
        )
        if shapes[4] != expected_codes_shape:
            raise TypeError("A finite NNBEGM policy has a mismatched code bank.")
        leaf_axes[paths[4]] = (candidate_axis, field_axis)
        axes.append(
            AxisAuthority(
                name=field_axis,
                length=len(policy_read.discrete_action_names),
                role=AxisRole.OTHER,
                coordinates=tuple(policy_read.discrete_action_names),
            )
        )
    return _ArtifactLayout(
        leaf_axis_names=leaf_axes,
        axes=tuple(axes),
        state_roles=policy_read.state_names,
        action_roles=_unique_names(
            (
                policy_read.inner_action_name,
                policy_read.outer_action_name,
                *policy_read.discrete_action_names,
            )
        ),
    )


def _nested_policy_artifact_layout(
    *,
    policy_read: NNBEGMPolicyRead,
    template_snapshot: _CanonicalArtifactTemplate | None,
    period: int,
    adaptive_outer_nodes: tuple[float, ...] | None,
) -> _ArtifactLayout:
    """Return static state roles and, when known, generated candidate authority."""
    if policy_read.liquid_state_name is None:
        raise TypeError("A nested NNBEGM policy has no liquid-state role.")
    row_state_names = _unique_names(
        (
            *policy_read.row_discrete_state_names,
            *policy_read.row_passive_state_names,
            policy_read.liquid_state_name,
        )
    )
    state_lengths = dict(
        zip(
            policy_read.state_names,
            policy_read.state_axis_lengths_by_period[period],
            strict=True,
        )
    )
    try:
        row_lengths = tuple(state_lengths[name] for name in row_state_names)
    except KeyError as error:
        raise TypeError("A nested NNBEGM row role is not a model state.") from error
    axes: list[AxisAuthority] = list(
        _model_role_axes(
            names=row_state_names,
            lengths=row_lengths,
            state_names=row_state_names,
            action_names=(),
        )
    )
    leaf_axes: dict[TreePath, tuple[str, ...]] = {}
    if template_snapshot is not None:
        if adaptive_outer_nodes is None:
            raise TypeError("A nested NNBEGM template has no generated outer nodes.")
        paths = template_snapshot.leaf_paths
        shapes = tuple(tuple(leaf.shape) for leaf in template_snapshot.leaves)
        if (
            len(paths) != 9  # noqa: PLR2004
            or any(shape != row_lengths for shape in shapes[:4])
        ):
            raise TypeError("A nested NNBEGM keeper has inconsistent row axes.")
        candidate_axis = "pylcm:nnbegm:outer_candidate"
        n_candidates = len(adaptive_outer_nodes)
        if shapes[4] != (n_candidates,) or any(
            shape != (n_candidates, *row_lengths) for shape in shapes[5:]
        ):
            raise TypeError("A nested NNBEGM adjuster has inconsistent axes.")
        leaf_axes.update(dict.fromkeys(paths[:4], row_state_names))
        leaf_axes[paths[4]] = (candidate_axis,)
        leaf_axes.update(dict.fromkeys(paths[5:], (candidate_axis, *row_state_names)))
        axes.insert(
            0,
            AxisAuthority(
                name=candidate_axis,
                length=n_candidates,
                role=AxisRole.CANDIDATE,
                coordinates=tuple(adaptive_outer_nodes),
            ),
        )
    return _ArtifactLayout(
        leaf_axis_names=leaf_axes,
        axes=tuple(axes),
        state_roles=_unique_names((*row_state_names, policy_read.outer_state_name)),
        action_roles=_unique_names(
            (policy_read.inner_action_name, policy_read.outer_action_name)
        ),
    )


def _egm_carry_artifact_layout(  # noqa: C901, PLR0912, PLR0915
    *,
    regime: Regime,
    state_action_space: StateActionSpace,
    spec: EGMContinuationSpec,
    template_snapshot: _CanonicalArtifactTemplate,
) -> _ArtifactLayout:
    """Describe a built-in EGM carry from producer declarations and model axes."""
    template = spec.template
    paths_and_leaves = tuple(
        zip(template_snapshot.leaf_paths, template_snapshot.leaves, strict=True)
    )
    row_leaf = next(
        leaf for path, leaf in paths_and_leaves if path[-1] == "attribute:endog_grid"
    )
    row_shape = tuple(row_leaf.shape)
    if not row_shape:
        raise TypeError("An EGM carry must expose a trailing grid axis.")
    row_state_names = _declared_carry_row_state_names(
        regime=regime,
        state_action_space=state_action_space,
    )
    row_action_names = (
        tuple(state_action_space.discrete_actions)
        if spec.layout.retains_discrete_action_rows
        else ()
    )
    candidate_count = spec.layout.n_stacked_candidates
    leading_names = [*row_state_names, *row_action_names]
    leading_lengths = list(row_shape[: len(leading_names)])
    expected_leading_ndim = len(leading_names) + bool(candidate_count)
    if len(row_shape) - 1 != expected_leading_ndim:
        raise TypeError(
            "An EGM carry's producer-declared leading roles differ from its template."
        )
    axes: list[AxisAuthority] = list(
        _model_role_axes(
            names=tuple(leading_names),
            lengths=tuple(leading_lengths),
            state_names=row_state_names,
            action_names=row_action_names,
        )
    )
    if candidate_count:
        if row_shape[-2] != candidate_count:
            raise TypeError("An EGM carry has a mismatched stacked-candidate axis.")
        candidate_name = "pylcm:egm:candidate"
        leading_names.append(candidate_name)
        axes.append(
            AxisAuthority(
                name=candidate_name,
                length=candidate_count,
                role=AxisRole.CANDIDATE,
                coordinates=tuple(range(candidate_count)),
            )
        )
    trailing_state = (
        _declared_carry_trailing_state(
            regime=regime,
            state_action_space=state_action_space,
            row_state_names=row_state_names,
        )
        if spec.layout.rows_share_state_grid
        else None
    )
    trailing_name = trailing_state or "pylcm:egm:node"
    axes.append(
        AxisAuthority(
            name=trailing_name,
            length=row_shape[-1],
            role=AxisRole.STATE if trailing_state is not None else AxisRole.OTHER,
            coordinates=(
                () if trailing_state is not None else tuple(range(row_shape[-1]))
            ),
        )
    )
    common_axis_names = (*leading_names, trailing_name)
    leaf_axes: dict[TreePath, tuple[str, ...]] = {}
    for path, leaf in paths_and_leaves:
        shape = tuple(leaf.shape)
        field = path[-1]
        if field == "attribute:taste_shock_scale":
            names: tuple[str, ...] = ()
        elif field == "attribute:breakpoints":
            breakpoint_name = "pylcm:egm:breakpoint"
            names = (*leading_names, breakpoint_name)
            if not any(axis.name == breakpoint_name for axis in axes):
                axes.append(
                    AxisAuthority(
                        name=breakpoint_name,
                        length=shape[-1],
                        role=AxisRole.OTHER,
                        coordinates=tuple(range(shape[-1])),
                    )
                )
        else:
            names = common_axis_names
        if tuple(axis.length for axis in axes if axis.name in names) != shape:
            by_name = {axis.name: axis for axis in axes}
            if tuple(by_name[name].length for name in names) != shape:
                raise TypeError(f"An EGM carry leaf {field!r} has inconsistent axes.")
        leaf_axes[path] = names
    state_roles = _unique_names(
        (*row_state_names, *((trailing_state,) if trailing_state else ()))
    )
    policy_read = regime.simulation.egm_policy_read
    if template.policy is None:
        policy_action_names: tuple[str, ...] = ()
    elif isinstance(policy_read, EGMPolicyRead):
        policy_action_names = (policy_read.action_name,)
    elif isinstance(policy_read, NNBEGMPolicyRead):
        policy_action_names = (policy_read.inner_action_name,)
    else:
        continuous_actions = tuple(state_action_space.continuous_actions)
        if len(continuous_actions) != 1:
            raise TypeError(
                "An EGM carry policy row does not declare exactly one action role."
            )
        policy_action_names = continuous_actions
    action_roles = _unique_names((*row_action_names, *policy_action_names))
    return _ArtifactLayout(
        leaf_axis_names=leaf_axes,
        axes=tuple(axes),
        state_roles=state_roles,
        action_roles=action_roles,
    )


def _declared_carry_row_state_names(
    *, regime: Regime, state_action_space: StateActionSpace
) -> tuple[str, ...]:
    """Read the row-state declaration from the core, with terminal convention."""
    for kernel in regime.solution.period_kernels.values():
        graph = core_program_graph(kernel=kernel)
        for program in graph.values():
            with_paths, _tree = jax.tree_util.tree_flatten_with_path(
                program.output_roles
            )
            for path, role in with_paths:
                normalized = _normalize_jax_tree_path(path)
                if (
                    normalized
                    and normalized[-1] == "attribute:endog_grid"
                    and isinstance(role, StateAxesLeading)
                ):
                    return tuple(role.state_names)
    discrete = tuple(
        name
        for name in state_action_space.states
        if name in regime.solution.discrete_state_names
    )
    continuous = tuple(
        name for name in state_action_space.states if name not in discrete
    )
    return (*discrete, *continuous[1:])


def _declared_carry_trailing_state(
    *,
    regime: Regime,
    state_action_space: StateActionSpace,
    row_state_names: tuple[str, ...],
) -> str | None:
    """Return the one model state used as a carry's shared abscissa."""
    if not state_action_space.states and not row_state_names and regime.terminal:
        # A stateless terminal publishes two benign padding nodes so an EGM parent
        # can carry its scalar value through the ordinary row reader. Those nodes
        # have no model-state identity: they are a synthetic interpolation axis.
        return None
    policy_read = regime.simulation.egm_policy_read
    if isinstance(policy_read, NNBEGMPolicyRead):
        if policy_read.liquid_state_name is None:
            raise TypeError("A state-grid NNBEGM carry has no liquid-state role.")
        return policy_read.liquid_state_name
    candidates = tuple(
        name
        for name in state_action_space.states
        if name not in row_state_names
        and name not in regime.solution.discrete_state_names
    )
    if len(candidates) != 1:
        raise TypeError(
            "A state-grid EGM carry does not declare exactly one trailing state."
        )
    return candidates[0]


def _model_role_axes(
    *,
    names: tuple[str, ...],
    lengths: tuple[int, ...],
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> tuple[AxisAuthority, ...]:
    """Build deferred-coordinate axes for exact model state/action roles."""
    if len(names) != len(lengths):
        raise TypeError("Artifact role names and axis lengths differ.")
    state_set = set(state_names)
    action_set = set(action_names)
    axes: list[AxisAuthority] = []
    for name, length in zip(names, lengths, strict=True):
        if name in state_set:
            role = AxisRole.STATE
        elif name in action_set:
            role = AxisRole.ACTION
        else:
            raise TypeError(f"Artifact axis {name!r} has no declared model role.")
        axes.append(AxisAuthority(name=name, length=length, role=role))
    return tuple(axes)


def _unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve declaration order while removing repeated semantic roles."""
    return tuple(dict.fromkeys(names))


def _authority_from_template(
    *,
    key: ArtifactKey,
    channel: ArtifactChannel,
    persistence: PersistencePolicy,
    payload_runtime_type: type[object],
    template: object | None,
    leaf_axis_names: dict[TreePath, tuple[str, ...]] | None = None,
    axes: tuple[AxisAuthority, ...] | None = None,
    state_roles: tuple[str, ...] = (),
    action_roles: tuple[str, ...] = (),
    categorical_domains: dict[str, CategoryDomain] | None = None,
    consumer_route: ReplayRouteIdentity | None = None,
    applicable: bool,
    required: bool,
) -> ArtifactAuthority:
    """Observe one engine template once, then build its exact authority."""
    if template is None:
        template_snapshot = None
        containers: dict[TreePath, type[object]] = {}
    else:
        template_snapshot, containers = _snapshot_artifact_template_once(
            template=template,
            payload_runtime_type=payload_runtime_type,
        )
    return _authority_from_observed_template(
        key=key,
        channel=channel,
        persistence=persistence,
        payload_runtime_type=payload_runtime_type,
        template_snapshot=template_snapshot,
        container_runtime_types=containers,
        leaf_axis_names=leaf_axis_names,
        axes=axes,
        state_roles=state_roles,
        action_roles=action_roles,
        categorical_domains=categorical_domains,
        consumer_route=consumer_route,
        applicable=applicable,
        required=required,
    )


def _authority_from_observed_template(
    *,
    key: ArtifactKey,
    channel: ArtifactChannel,
    persistence: PersistencePolicy,
    payload_runtime_type: type[object],
    template_snapshot: _CanonicalArtifactTemplate | None,
    container_runtime_types: dict[TreePath, type[object]],
    leaf_axis_names: dict[TreePath, tuple[str, ...]] | None = None,
    axes: tuple[AxisAuthority, ...] | None = None,
    state_roles: tuple[str, ...] = (),
    action_roles: tuple[str, ...] = (),
    categorical_domains: dict[str, CategoryDomain] | None = None,
    consumer_route: ReplayRouteIdentity | None = None,
    applicable: bool,
    required: bool,
) -> ArtifactAuthority:
    """Build an exact authority from one already observed engine template."""
    categories = {} if categorical_domains is None else categorical_domains
    if template_snapshot is None:
        if container_runtime_types:
            raise TypeError("A missing template snapshot cannot declare containers.")
        authority_axes = () if axes is None else tuple(axes)
        descriptor = ArtifactDescriptor(
            key=key,
            channel=channel,
            persistence=persistence,
            payload_type_id=_payload_type_id(payload_runtime_type),
            named_axes=tuple(axis.descriptor for axis in authority_axes),
            state_roles=state_roles,
            action_roles=action_roles,
            categorical_domains=categories,
            required_for=(
                frozenset({consumer_route})
                if required and consumer_route is not None
                else frozenset()
            ),
            required=required,
        )
        return ArtifactAuthority(
            descriptor=descriptor,
            payload_runtime_type=payload_runtime_type,
            template=None,
            axes=authority_axes,
            state_roles=state_roles,
            action_roles=action_roles,
            categorical_domains=categories,
            consumer_route=consumer_route,
            applicable=applicable,
            required=required,
        )

    supplied_axis_names = {} if leaf_axis_names is None else leaf_axis_names
    generated_axes: dict[str, AxisAuthority] = {}
    leaf_authorities: dict[TreePath, LeafAuthority] = {}
    for normalized, leaf in zip(
        template_snapshot.leaf_paths,
        template_snapshot.leaves,
        strict=True,
    ):
        shape = tuple(leaf.shape)
        dtype = str(np.dtype(leaf.dtype))
        names = supplied_axis_names.get(normalized)
        if names is None:
            stem = "/".join(normalized) if normalized else "root"
            names = tuple(f"{stem}:axis_{index}" for index in range(len(shape)))
            for name, length in zip(names, shape, strict=True):
                generated_axes[name] = AxisAuthority(
                    name=name,
                    length=length,
                    role=AxisRole.OTHER,
                    coordinates=tuple(range(length)),
                )
        leaf_authorities[normalized] = LeafAuthority(
            path=normalized,
            runtime_type=Array,
            shape=shape,
            dtype=dtype,
            axis_names=names,
        )
    authority_axes = tuple(generated_axes.values()) if axes is None else tuple(axes)
    descriptor = ArtifactDescriptor(
        key=key,
        channel=channel,
        persistence=persistence,
        payload_type_id=_payload_type_id(payload_runtime_type),
        leaf_descriptors=tuple(leaf.descriptor for leaf in leaf_authorities.values()),
        named_axes=tuple(axis.descriptor for axis in authority_axes),
        state_roles=state_roles,
        action_roles=action_roles,
        categorical_domains=categories,
        required_for=(
            frozenset({consumer_route})
            if required and consumer_route is not None
            else frozenset()
        ),
        required=required,
    )
    return _artifact_authority_from_template_snapshot(
        descriptor=descriptor,
        payload_runtime_type=payload_runtime_type,
        template_snapshot=template_snapshot,
        container_runtime_types=container_runtime_types,
        leaves=leaf_authorities,
        axes=authority_axes,
        state_roles=state_roles,
        action_roles=action_roles,
        categorical_domains=categories,
        consumer_route=consumer_route,
        applicable=applicable,
        required=required,
    )


def _payload_type_id(payload_runtime_type: type[object]) -> str:
    """Return one stable descriptive type id without importing plugin code later."""
    if payload_runtime_type is Array:
        return "jax.Array"
    return f"{payload_runtime_type.__module__}.{payload_runtime_type.__qualname__}"


def _bind_model_owned_artifact_facts(  # noqa: C901
    *,
    authority: ArtifactAuthority,
    regime: Regime,
    state_action_space: StateActionSpace,
) -> ArtifactAuthority:
    """Replace declared state/action coordinates with canonical model values."""
    template_snapshot = _artifact_authority_template_snapshot(authority)
    if authority.descriptor.channel is ArtifactChannel.DIAGNOSTIC:
        raise TypeError(
            "Custom DIAGNOSTIC authorities are unsupported because KernelOutput "
            "exposes no custom diagnostic channel."
        )
    missing_states = set(authority.state_roles) - state_action_space.states.keys()
    missing_actions = set(authority.action_roles) - state_action_space.actions.keys()
    if missing_states or missing_actions:
        raise TypeError(
            "Artifact semantic roles are absent from the canonical model: "
            f"states={tuple(sorted(missing_states))!r}, "
            f"actions={tuple(sorted(missing_actions))!r}."
        )
    bound_axes: list[AxisAuthority] = []
    for axis in authority.axes:
        if axis.name in state_action_space.states and axis.role is not AxisRole.STATE:
            raise TypeError(
                f"Artifact axis {axis.name!r} names a canonical model state but "
                "does not declare STATE role."
            )
        if axis.name in state_action_space.actions and axis.role is not AxisRole.ACTION:
            raise TypeError(
                f"Artifact axis {axis.name!r} names a canonical model action but "
                "does not declare ACTION role."
            )
        if axis.role is AxisRole.STATE:
            nodes = state_action_space.states.get(axis.name)
        elif axis.role is AxisRole.ACTION:
            nodes = state_action_space.actions.get(axis.name)
        else:
            nodes = None
        if axis.role in {AxisRole.STATE, AxisRole.ACTION}:
            if nodes is None:
                raise TypeError(
                    f"Artifact axis {axis.name!r} has no matching model "
                    f"{axis.role.value} role."
                )
            coordinates = _json_coordinates(nodes)
            if len(coordinates) != axis.length:
                raise ValueError(
                    f"Artifact axis {axis.name!r} has declared length {axis.length}, "
                    f"but the canonical model supplies {len(coordinates)} nodes."
                )
            bound_axes.append(replace(axis, coordinates=coordinates))
        else:
            bound_axes.append(axis)

    categories = {
        name: CategoryDomain(
            labels=grid.categories,
            codes=grid.codes,
            ordered=grid.ordered,
        )
        for name in (*authority.state_roles, *authority.action_roles)
        if isinstance((grid := regime.solution.grids.get(name)), DiscreteGrid)
    }
    descriptor = replace(
        authority.descriptor,
        named_axes=tuple(axis.descriptor for axis in bound_axes),
        categorical_domains=categories,
    )
    return _artifact_authority_from_template_snapshot(
        descriptor=descriptor,
        payload_runtime_type=authority.payload_runtime_type,
        template_snapshot=template_snapshot,
        container_runtime_types=authority.container_runtime_types,
        leaves=authority.leaves,
        axes=tuple(bound_axes),
        state_roles=authority.state_roles,
        action_roles=authority.action_roles,
        categorical_domains=categories,
        consumer_route=authority.consumer_route,
        applicable=authority.applicable,
        required=authority.required,
    )


def _json_coordinates(nodes: object) -> tuple[bool | int | float | str, ...]:
    """Convert model grid nodes to immutable, exact transport scalars."""
    array = np.asarray(nodes)
    if array.ndim != 1:
        raise TypeError(
            "Artifact state/action authority requires one-dimensional nodes."
        )
    coordinates: list[bool | int | float | str] = []
    for raw in array.tolist():
        value = raw.item() if isinstance(raw, np.generic) else raw
        if not any(type(value) is allowed for allowed in (bool, int, float, str)):
            raise TypeError("Artifact coordinates must be exact JSON scalar values.")
        if type(value) is float and not np.isfinite(value):
            raise ValueError("Artifact coordinates must be finite model-owned values.")
        coordinates.append(value)
    return tuple(coordinates)


def _state_action_space_for_period(
    *, regime: Regime, base: StateActionSpace, period: int
) -> StateActionSpace:
    """Overlay canonical age-specialized state nodes for one exact solution cell."""
    if regime.solution.period_state_axes is None:
        return base
    period_states = regime.solution.period_state_axes.get(period)
    if not period_states:
        return base
    states = cast(
        "MappingProxyType[StateName, ContinuousState | DiscreteState]",
        MappingProxyType(dict(base.states) | dict(period_states)),
    )
    return base.replace(states=states)


def _replay_model_context_from_state_action_space(
    *,
    regime_name: RegimeName,
    period: int,
    state_action_space: StateActionSpace,
) -> ReplayModelContext:
    """Build the canonical solve-grid view shared by replay route callbacks.

    Replay artifacts describe decisions over the solve state/action space. A state
    carried only during simulation therefore remains available to the eventual
    ``ReplayReader`` call, but is not a named artifact axis and is intentionally absent
    from this context and its node mappings.
    """
    return ReplayModelContext(
        regime_name=regime_name,
        period=period,
        state_names=tuple(state_action_space.state_names),
        action_names=tuple(state_action_space.action_names),
        state_nodes={
            name: jnp.asarray(nodes)
            for name, nodes in state_action_space.states.items()
        },
        action_nodes={
            name: jnp.asarray(nodes)
            for name, nodes in state_action_space.actions.items()
        },
    )


def bind_generated_solution_authority(
    *,
    authority: SolutionAuthority,
    internal_result: BackwardInductionResult,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> SolutionAuthority:
    """Bind data-dependent axis coordinates before a result leaves the model.

    Adaptive NNBEGM decides its final shared outer mesh from exact solves, so its
    candidate coordinates cannot be reconstructed from declarations alone without
    repeating the solve. The producing kernel emits those coordinates on a private
    sidecar beside the replay payload. The model records that sidecar in immutable
    authority; the returned payload and metadata do not carry the trusted copy.
    """
    replay = dict(authority.replay)
    artifacts = dict(authority.artifacts)
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
            policy_read = descriptor.route
            if not isinstance(policy_read, NNBEGMPolicyRead):
                raise TypeError(
                    "Generated adaptive replay authority has no NNBEGM route."
                )
            old_authority = artifacts[ref]
            template = _adaptive_policy_template(
                policy_read=policy_read,
                policy_shape=descriptor.shape,
                expected_replay_capability=descriptor.expected_replay_capability,
                adaptive_outer_nodes=generated.adaptive_outer_nodes,
            )
            template_snapshot, template_containers = _snapshot_artifact_template_once(
                template=template,
                payload_runtime_type=old_authority.payload_runtime_type,
            )
            policy_layout = _nested_policy_artifact_layout(
                policy_read=policy_read,
                template_snapshot=template_snapshot,
                period=period,
                adaptive_outer_nodes=generated.adaptive_outer_nodes,
            )
            generated_authority = _authority_from_observed_template(
                key=old_authority.descriptor.key,
                channel=old_authority.descriptor.channel,
                persistence=old_authority.descriptor.persistence,
                payload_runtime_type=old_authority.payload_runtime_type,
                template_snapshot=template_snapshot,
                container_runtime_types=template_containers,
                leaf_axis_names=policy_layout.leaf_axis_names,
                axes=policy_layout.axes,
                state_roles=policy_layout.state_roles,
                action_roles=policy_layout.action_roles,
                consumer_route=old_authority.consumer_route,
                applicable=old_authority.applicable,
                required=old_authority.required,
            )
            regime = regimes[regime_name]
            state_action_space = _state_action_space_for_period(
                regime=regime,
                base=regime.solution.state_action_space(
                    regime_params=flat_params[regime_name]
                ),
                period=period,
            )
            artifacts[ref] = _bind_model_owned_artifact_facts(
                authority=generated_authority,
                regime=regime,
                state_action_space=state_action_space,
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
    return replace(
        authority,
        replay=MappingProxyType(replay),
        artifacts=MappingProxyType(artifacts),
    )


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


def _policy_persistence_and_template(
    *,
    policy_read: EGMPolicyRead | NNBEGMPolicyRead | None,
    policy_shape: tuple[int, ...] | None,
    expected_replay_capability: OuterReplayCapability | None,
) -> tuple[PersistencePolicy, object | None]:
    """Return the route's persistence policy and model-built PyTree template."""
    if isinstance(policy_read, EGMPolicyRead):
        if policy_shape is None:
            raise TypeError("An EGM replay route has no model-owned policy shape.")
        row = jnp.zeros(policy_shape, dtype=policy_read.float_dtype)
        return PersistencePolicy.MODEL_VERIFIABLE, EGMSimPolicy(
            endog_grid=row,
            policy=row,
            value=row,
            marginal_utility=row,
            row_discrete_state_names=policy_read.row_discrete_state_names,
            row_passive_state_names=policy_read.row_passive_state_names,
            row_discrete_action_names=policy_read.row_discrete_action_names,
        )

    if isinstance(policy_read, NNBEGMPolicyRead):
        if policy_read.replay_policy_is_nested:
            # The final outer nodes are solve-generated private authority.  A
            # serialized copy cannot authenticate itself after restoration.
            return PersistencePolicy.NOT_PERSISTED, None
        if (
            policy_shape is None
            or policy_read.outer_grid_values is None
            or policy_read.n_keeper_candidates is None
            or expected_replay_capability is None
        ):
            raise TypeError(
                "A finite NNBEGM replay route lacks model-verifiable policy facts."
            )
        bank = jnp.zeros(policy_shape, dtype=policy_read.float_dtype)
        discrete_codes = (
            None
            if not policy_read.discrete_action_names
            else jnp.asarray(
                policy_read.candidate_discrete_action_codes,
                dtype=policy_read.integer_dtype,
            )
        )
        return PersistencePolicy.MODEL_VERIFIABLE, NNBEGMSimPolicy(
            candidate_inner_action=bank,
            candidate_outer_target=bank,
            candidate_value=bank,
            outer_grid_values=jnp.asarray(
                policy_read.outer_grid_values,
                dtype=policy_read.float_dtype,
            ),
            state_names=policy_read.state_names,
            inner_action_name=policy_read.inner_action_name,
            outer_action_name=policy_read.outer_action_name,
            n_keeper_candidates=policy_read.n_keeper_candidates,
            replay_capability=expected_replay_capability,
            candidate_discrete_actions=discrete_codes,
            discrete_action_names=policy_read.discrete_action_names,
        )

    return PersistencePolicy.NOT_PERSISTED, None


def _inner_policy_from_array(
    *, array: Array, policy_read: NNBEGMPolicyRead
) -> EGMSimPolicy:
    """Fill every numerical field of an inner policy template with one array."""
    return EGMSimPolicy(
        endog_grid=array,
        policy=array,
        value=array,
        marginal_utility=array,
        row_discrete_state_names=policy_read.row_discrete_state_names,
        row_passive_state_names=policy_read.row_passive_state_names,
        row_discrete_action_names=(),
    )


def _adaptive_policy_template(
    *,
    policy_read: NNBEGMPolicyRead,
    policy_shape: tuple[int, ...] | None,
    expected_replay_capability: OuterReplayCapability | None,
    adaptive_outer_nodes: tuple[float, ...],
) -> NestedEGMSimPolicy:
    """Build a private same-model template from declared facts and generated axes."""
    if (
        policy_shape is None
        or expected_replay_capability is None
        or policy_read.liquid_state_name is None
        or policy_read.resources_target is None
        or policy_read.savings_lower_bound is None
        or policy_read.golden_iterations is None
        or policy_read.value_atol is None
        or policy_read.value_rtol is None
    ):
        raise TypeError("An adaptive NNBEGM route lacks model-owned replay facts.")
    row = jnp.zeros(policy_shape, dtype=policy_read.float_dtype)
    candidate_row = jnp.zeros(
        (len(adaptive_outer_nodes), *policy_shape),
        dtype=policy_read.float_dtype,
    )

    return NestedEGMSimPolicy(
        keeper=_inner_policy_from_array(policy_read=policy_read, array=row),
        adjuster=OuterPolicyBank(
            outer_nodes=jnp.asarray(
                adaptive_outer_nodes,
                dtype=policy_read.float_dtype,
            ),
            policies=_inner_policy_from_array(
                policy_read=policy_read, array=candidate_row
            ),
        ),
        outer_action_name=policy_read.outer_action_name,
        outer_state_name=policy_read.outer_state_name,
        outer_post_decision_name=policy_read.outer_post_decision,
        inner_action_name=policy_read.inner_action_name,
        liquid_state_name=policy_read.liquid_state_name,
        outer_no_adjustment_name=policy_read.outer_no_adjustment_target,
        resources_target_name=policy_read.resources_target,
        savings_lower_bound=policy_read.savings_lower_bound,
        golden_iterations=policy_read.golden_iterations,
        replay_capability=expected_replay_capability,
        value_atol=policy_read.value_atol,
        value_rtol=policy_read.value_rtol,
    )


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
    "_replay_model_context_from_state_action_space",
    "_state_action_space_for_period",
    "bind_generated_solution_authority",
    "build_solution_authority",
]
