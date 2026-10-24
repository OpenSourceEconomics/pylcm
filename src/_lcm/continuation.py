"""Continuation specs: what a regime publishes for the previous period's kernels.

The engine treats a continuation as an opaque `ContinuationArtifact` identified by
its key. `ContinuationSpec` pairs the all-finite template the loop rolls and lowers
with that key; the EGM family publishes the concrete `EGMContinuationSpec`, which
adds every layout property a reading parent needs.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import jax.numpy as jnp
from dags import get_ancestors

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.post_decision_bound import PostDecisionBoundSpec
from lcm.solver_api import EGM_CONTINUATION, ArtifactKey, ContinuationArtifact
from lcm.typing import ActionName, FunctionName, ReferenceName, StateName, UserFunction

if TYPE_CHECKING:
    from _lcm.execution.value_transfer import ValueArtifactAddress

# Backward induction stores and forwards this channel without reading its fields.
type ContinuationPayload = ContinuationArtifact

_EXACT_ENVELOPE_SIDE_IDS = (-1, 0, 1)


@dataclass(frozen=True, kw_only=True)
class FiniteOuterQuerySpec:
    """Canonical keeper and finite adjuster identities in common resources."""

    outer_state: StateName
    """Actual fixed current stock, including at an off-grid query."""
    outer_action: ActionName
    """Chosen outer action, excluded from the fixed-current floor."""
    outer_post_decision: FunctionName
    """Canonical target internally bound to a keeper or grid-node identity."""
    keeper_target: FunctionName | None
    """Canonical keeper map; absent for the actual current-stock identity."""
    cost_target: FunctionName | None
    """Canonical adjustment cost, absent only for a cost-free declared margin."""
    outer_nodes: tuple[float, ...]
    """Declared adjuster grid in its retained order, excluding keeper slot zero."""


@dataclass(frozen=True, kw_only=True)
class EGMRowTopologySpec:
    """Compiler-owned connected runs of represented conditional ordinary rows.

    - "exact_monotone_run" retains the exact envelope's emitted run identities.
    - "nb_budget_interval_run" retains NB ordinary-row connectivity. Scalar
      sampled rows may connect across continuous budget kinks; interval-held,
      jump and one-sided rows retain their own declared cuts.

    An ordinary owner code establishes compatible row connectivity. Compact
    borrowing corners select budget intervals through "corner_interval_id";
    an ordinary owner code is not a compact corner's interval selector.
    """

    kind: Literal["exact_monotone_run", "nb_budget_interval_run"]
    """Connected-row family emitted by the canonical source producer."""
    owner_capacity: int
    """Exclusive upper bound for the source producer's ordinary-row owner codes."""
    side_ids: tuple[int, ...] = _EXACT_ENVELOPE_SIDE_IDS
    """Known outgoing, unsided and incoming side codes."""
    schema_version: int = 1
    """Static ownership semantics, independent of the numeric carry schema."""

    def __post_init__(self) -> None:
        if (
            self.kind not in ("exact_monotone_run", "nb_budget_interval_run")
            or type(self.owner_capacity) is not int
            or self.owner_capacity < 1
            or type(self.side_ids) is not tuple
            or len(self.side_ids) != len(_EXACT_ENVELOPE_SIDE_IDS)
            or any(type(side) is not int for side in self.side_ids)
            or set(self.side_ids) != set(_EXACT_ENVELOPE_SIDE_IDS)
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise TypeError("Query row topology has invalid run/side/schema semantics.")


@dataclass(frozen=True, kw_only=True)
class EGMQuerySpec:
    """Model-owned meaning of conditional query rows and their evaluator inputs."""

    bound: PostDecisionBoundSpec
    """Installed source/target bound identity; numerical floors never supply it."""
    state_names: tuple[StateName, ...]
    """Ordered discrete/process then passive fixed-state axes."""
    action_names: tuple[ActionName, ...]
    """Ordered conditional discrete-action axes."""
    coordinate_kind: Literal["resources", "liquid", "nested_common_resources"]
    """Native row units and parent chain-factor convention."""
    marginal_convention: Literal["ordinary_linear", "paired_value_derivative"]
    """Declared marginal interpolation contract."""
    corner_coordinate_kind: Literal["compact_interval", "liquid_row"]
    """Exactly one supported corner representation."""
    row_capacity: int
    """Existing padded row capacity."""
    interval_ids: tuple[int, ...]
    """Canonical interval identities in retained slot order."""
    interval_side_ids: tuple[int, ...]
    """One-sided equality owner for each interval slot."""
    evaluator_targets: tuple[tuple[FunctionName, FunctionName], ...]
    """Canonical resources/utility/CE/predicate targets, indexed by purpose."""
    evaluator_inputs: tuple[tuple[FunctionName, tuple[ReferenceName, ...]], ...]
    """Resolved transitive inputs for every retained evaluator target."""
    evaluator_identity: tuple[tuple[FunctionName, Hashable], ...]
    """Canonical evaluator/dependency/phase identity, excluding numerical floors."""
    target_periods: tuple[int, ...]
    """Canonical target-period bindings represented by this declaration."""
    outer_mode: Literal["none", "finite_max"] = "none"
    """Outer operator, independent of the incidental row-bank shape."""
    outer_capacity: int = 0
    """Finite keeper-plus-adjuster candidate count."""
    row_topology: EGMRowTopologySpec | None = dataclass_field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Optional compiler-owned row runs, independent of corner interval labels."""
    interval_source_identity: tuple[Hashable, ...] = dataclass_field(
        default=(), metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical NB source order, comparisons and threshold-table selectors."""
    finite_outer: FiniteOuterQuerySpec | None = dataclass_field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Finite nested identities, independent of incidental numerical row shape."""
    schema_version: int = EGM_CONTINUATION.schema_version
    """Owning shared continuation schema."""

    def __post_init__(self) -> None:
        _validate_row_topology(topology=self.row_topology)
        if self.schema_version != EGM_CONTINUATION.schema_version:
            raise TypeError("Query authority has an incompatible continuation schema.")
        if (
            self.row_capacity < 1
            or not self.interval_ids
            or len(self.interval_ids) != len(self.interval_side_ids)
        ):
            raise TypeError(
                "Query authority requires row capacity and matched interval labels."
            )
        if len({*self.state_names, *self.action_names}) != len(self.state_names) + len(
            self.action_names
        ):
            raise TypeError(
                "Query authority requires distinct ordered state/action roles."
            )
        if (self.outer_mode == "none") != (self.outer_capacity == 0):
            raise TypeError("Query outer operator and capacity disagree.")
        if (self.outer_mode == "finite_max") != (self.finite_outer is not None):
            raise TypeError("Finite nested queries require canonical outer identities.")
        if self.finite_outer is not None and (
            type(self.finite_outer) is not FiniteOuterQuerySpec
            or not (
                self.coordinate_kind == "nested_common_resources"
                or (
                    self.coordinate_kind == "liquid"
                    and self.marginal_convention == "ordinary_linear"
                    and self.corner_coordinate_kind == "compact_interval"
                    and self.row_topology is not None
                    and self.row_topology.kind == "nb_budget_interval_run"
                )
            )
            or self.outer_capacity != len(self.finite_outer.outer_nodes) + 1
        ):
            raise TypeError(
                "Finite nested coordinates and candidate authority disagree."
            )


def _validate_row_topology(*, topology: EGMRowTopologySpec | None) -> None:
    """Reject substituted or malformed static row ownership descriptors."""
    if topology is not None:
        if type(topology) is not EGMRowTopologySpec:
            raise TypeError("Query row topology requires an exact compiler record.")
        EGMRowTopologySpec.__post_init__(topology)


def _validate_row_topology_authority(
    *,
    left: EGMRowTopologySpec | None,
    right: EGMRowTopologySpec | None,
) -> None:
    """Compare every compiler topology field independently of record identity."""
    same = (left is None and right is None) or (
        left is not None
        and right is not None
        and all(
            _same_bound_identity(
                left=getattr(left, item.name), right=getattr(right, item.name)
            )
            for item in fields(EGMRowTopologySpec)
        )
    )
    if not same:
        raise TypeError("Query row topology differs from model authority.")


def validate_query_bound_authority(
    *, declared: PostDecisionBoundSpec, model_bound: PostDecisionBoundSpec
) -> None:
    """Reject a descriptor whose bound differs from the independently installed one."""
    if (
        type(declared) is not PostDecisionBoundSpec
        or type(model_bound) is not PostDecisionBoundSpec
    ):
        raise TypeError(
            "Query bound authority requires exact model-owned bound records."
        )
    fields = (
        "post_decision",
        "liquid_state",
        "action",
        "resources_target",
        "kind",
        "literal_value",
        "lower_target",
        "lower_inputs",
        "dependency_identity",
        "declaration_names",
    )
    if not all(
        _same_bound_identity(
            left=getattr(declared, name), right=getattr(model_bound, name)
        )
        for name in fields
    ):
        raise TypeError("Query bound differs from model-owned bound authority.")


def _same_bound_identity(*, left: Hashable, right: Hashable) -> bool:
    """Compare inert keys recursively and executable dependencies by identity."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        right = cast("tuple[Hashable, ...]", right)
        return len(left) == len(right) and all(
            _same_bound_identity(left=a, right=b)
            for a, b in zip(left, right, strict=True)
        )
    if type(left) in (str, int, float, bool, type(None)):
        return left == right
    return left is right


def validate_query_model_authority(
    *,
    spec: EGMQuerySpec,
    model_bound: PostDecisionBoundSpec,
    state_names: tuple[StateName, ...],
    action_names: tuple[ActionName, ...],
    model_spec: EGMQuerySpec | None = None,
) -> None:
    """Bind query axes and the floor to independent canonical model declarations."""
    if type(spec) is not EGMQuerySpec:
        raise TypeError("Query authority must be an exact model-owned record.")
    _validate_row_topology(topology=spec.row_topology)
    validate_query_bound_authority(declared=spec.bound, model_bound=model_bound)
    if spec.state_names != state_names or spec.action_names != action_names:
        raise TypeError("Query state/action roles differ from canonical model roles.")
    if model_spec is not None:
        if type(model_spec) is not EGMQuerySpec:
            raise TypeError("Query model authority must be an exact canonical record.")
        _validate_row_topology(topology=model_spec.row_topology)
        validate_query_bound_authority(
            declared=model_spec.bound, model_bound=model_bound
        )
        _validate_row_topology_authority(
            left=spec.row_topology, right=model_spec.row_topology
        )
        for field in fields(EGMQuerySpec):
            if field.name in ("bound", "row_topology"):
                continue
            left, right = getattr(spec, field.name), getattr(model_spec, field.name)
            if field.name == "finite_outer":
                if (left is None) != (right is None) or (
                    left is not None
                    and any(
                        not _same_bound_identity(
                            left=getattr(left, item.name),
                            right=getattr(right, item.name),
                        )
                        for item in fields(type(left))
                    )
                ):
                    raise TypeError(
                        "Query finite outer semantics differ from model authority."
                    )
            elif not _same_bound_identity(left=left, right=right):
                raise TypeError(f"Query {field.name} differs from model authority.")


def build_query_model_authority(
    *,
    bound: PostDecisionBoundSpec,
    state_names: tuple[StateName, ...],
    action_names: tuple[ActionName, ...],
    coordinate_kind: Literal["resources", "liquid", "nested_common_resources"],
    marginal_convention: Literal["ordinary_linear", "paired_value_derivative"],
    corner_coordinate_kind: Literal["compact_interval", "liquid_row"],
    row_capacity: int,
    interval_ids: tuple[int, ...],
    interval_side_ids: tuple[int, ...],
    evaluator_targets: tuple[tuple[FunctionName, FunctionName], ...],
    functions: Mapping[FunctionName, UserFunction],
    target_periods: tuple[int, ...],
    functions_by_period: Mapping[int, Mapping[FunctionName, UserFunction]]
    | None = None,
    outer_mode: Literal["none", "finite_max"] = "none",
    outer_capacity: int = 0,
    row_topology: EGMRowTopologySpec | None = None,
    interval_source_identity: tuple[Hashable, ...] = (),
    finite_outer: FiniteOuterQuerySpec | None = None,
) -> EGMQuerySpec:
    """Bind trusted route declarations to the canonical evaluator dependency pool.

    Route compilation supplies coordinate, interval and outer semantics before a
    payload exists. Canonical phase functions supply evaluator identities and leaf
    inputs. The installed bound and period bindings remain independent of arrays.
    """
    if not evaluator_targets or len(
        {purpose for purpose, _ in evaluator_targets}
    ) != len(evaluator_targets):
        raise TypeError("Query authority needs distinct canonical evaluator purposes.")
    if dict(evaluator_targets).get("resources") != bound.resources_target:
        raise TypeError(
            "Query resource evaluator differs from the installed liquid role."
        )
    evaluator_targets = _query_bound_evaluator_targets(
        bound=bound, targets=evaluator_targets
    )
    inputs, identity = _query_evaluator_contract(
        functions=functions,
        targets=evaluator_targets,
        functions_by_period=functions_by_period,
    )
    return EGMQuerySpec(
        bound=bound,
        state_names=state_names,
        action_names=action_names,
        coordinate_kind=coordinate_kind,
        marginal_convention=marginal_convention,
        corner_coordinate_kind=corner_coordinate_kind,
        row_capacity=row_capacity,
        interval_ids=interval_ids,
        interval_side_ids=interval_side_ids,
        evaluator_targets=evaluator_targets,
        evaluator_inputs=inputs,
        evaluator_identity=identity,
        target_periods=target_periods,
        outer_mode=outer_mode,
        outer_capacity=outer_capacity,
        row_topology=row_topology,
        interval_source_identity=interval_source_identity,
        finite_outer=finite_outer,
    )


def _query_bound_evaluator_targets(
    *,
    bound: PostDecisionBoundSpec,
    targets: tuple[tuple[FunctionName, FunctionName], ...],
) -> tuple[tuple[FunctionName, FunctionName], ...]:
    """Include the installed floor DAG independently of route evaluator choices."""
    if bound.kind != "dag":
        return targets
    if bound.lower_target is None:
        raise TypeError("Query DAG bound lacks its canonical lower target.")
    supplied = dict(targets).get("lower_bound")
    if supplied is not None:
        if supplied != bound.lower_target:
            raise TypeError("Query lower evaluator differs from the installed bound.")
        return targets
    return (*targets, ("lower_bound", bound.lower_target))


def _query_evaluator_contract(
    *,
    functions: Mapping[FunctionName, UserFunction],
    targets: tuple[tuple[FunctionName, FunctionName], ...],
    functions_by_period: Mapping[int, Mapping[FunctionName, UserFunction]]
    | None = None,
) -> tuple[
    tuple[tuple[FunctionName, tuple[ReferenceName, ...]], ...],
    tuple[tuple[FunctionName, Hashable], ...],
]:
    """Resolve every executable ancestor and external input of each evaluator."""
    inputs = []
    identities = []
    for purpose, target in targets:
        contexts = (
            (None, functions),
            *tuple(sorted((functions_by_period or {}).items())),
        )
        leaf_inputs: set[ReferenceName] = set()
        dependencies = []
        for period, pool in contexts:
            if target not in pool:
                raise TypeError(
                    f"Query evaluator {target!r} is absent from its period pool."
                )
            ancestors = get_ancestors(pool, targets=target, include_targets=True)
            leaf_inputs.update(ancestors - pool.keys())
            dependencies.append(
                (
                    period,
                    tuple(
                        (name, pool[name]) for name in sorted(ancestors & pool.keys())
                    ),
                )
            )
        inputs.append((purpose, tuple(sorted(leaf_inputs))))
        identities.append((purpose, tuple(dependencies)))
    return tuple(inputs), tuple(identities)


def validate_query_evaluator_authority(
    *,
    spec: EGMQuerySpec,
    functions: Mapping[FunctionName, UserFunction],
    target_periods: tuple[int, ...],
    functions_by_period: Mapping[int, Mapping[FunctionName, UserFunction]]
    | None = None,
) -> None:
    """Authenticate canonical phase dependencies and exact target-time binding."""
    inputs, identities = _query_evaluator_contract(
        functions=functions,
        targets=_query_bound_evaluator_targets(
            bound=spec.bound, targets=spec.evaluator_targets
        ),
        functions_by_period=functions_by_period,
    )
    if (
        not spec.evaluator_targets
        or not _same_bound_identity(left=spec.evaluator_inputs, right=inputs)
        or not _same_bound_identity(left=spec.evaluator_identity, right=identities)
    ):
        raise TypeError(
            "Query evaluator inputs/dependencies differ from canonical model authority."
        )
    if spec.target_periods != target_periods:
        raise TypeError("Query target periods differ from canonical model authority.")


@dataclass(frozen=True, kw_only=True)
class EGMContinuationLayout:
    """Static interpretation of the leading axes in one EGM carry."""

    retains_discrete_action_rows: bool = True
    """Whether a parent must aggregate child discrete-action rows."""

    rows_share_state_grid: bool = False
    """Whether every row uses the child's own state grid as abscissae."""

    n_stacked_candidates: int = 0
    """Length of a stacked outer-candidate axis; zero means no such axis."""

    query_spec: EGMQuerySpec | None = None
    """Optional model-owned actual-query authority, absent on the legacy row path."""


@dataclass(frozen=True, kw_only=True)
class ContinuationSpec:
    """Template and identity of the continuation a regime publishes."""

    template: ContinuationArtifact
    """All-finite payload with the exact shapes used for rolling and lowering."""

    artifact_key: ArtifactKey
    """Versioned identity under which the kernel publishes this payload."""

    templates_by_period: Mapping[int, ContinuationArtifact] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )
    """Actual producer-date schemas when demanded operand trees vary by date."""

    def __post_init__(self) -> None:
        if self.template.artifact_key != self.artifact_key:
            msg = (
                f"The continuation template carries key {self.template.artifact_key}, "
                f"not the declared {self.artifact_key}."
            )
            raise ValueError(msg)
        templates = dict(self.templates_by_period)
        for period, template in templates.items():
            if type(period) is not int or period < 0:
                raise ValueError("A continuation schema needs a nonnegative period.")
            if template.artifact_key != self.artifact_key:
                raise ValueError(
                    f"The continuation template at period {period} carries key "
                    f"{template.artifact_key}, not {self.artifact_key}."
                )
        object.__setattr__(self, "templates_by_period", MappingProxyType(templates))

    def template_at(self, *, period: int) -> ContinuationArtifact:
        """Return the schema of this exact producer, without borrowing a date."""
        if not self.templates_by_period:
            return self.template
        if period not in self.templates_by_period:
            raise ValueError(f"No continuation schema is declared at period {period}.")
        return self.templates_by_period[period]

    def stored_value_aliases(
        self,
    ) -> Mapping[ValueArtifactAddress, ValueArtifactAddress]:
        """Declare transported leaves that retain another stored artifact's buffer."""
        return MappingProxyType({})


@dataclass(frozen=True, kw_only=True)
class EGMContinuationSpec(ContinuationSpec):
    """Concrete EGM continuation template bundled with its static layout."""

    template: EGMCarry
    """All-finite payload with the exact shapes used for rolling and lowering."""

    artifact_key: ArtifactKey = EGM_CONTINUATION
    """Versioned identity under which an EGM kernel publishes this payload."""

    layout: EGMContinuationLayout = EGMContinuationLayout()
    """How a reading parent interprets the template and produced payloads."""

    def __post_init__(self) -> None:
        super().__post_init__()
        query = self.template.query_data
        spec = self.layout.query_spec
        if (query is None) != (spec is None):
            raise TypeError(
                "Feature query data and model-owned query authority must agree."
            )
        if spec is not None:
            if self.layout.n_stacked_candidates != spec.outer_capacity or (
                spec.action_names and not self.layout.retains_discrete_action_rows
            ):
                raise TypeError(
                    "Query authority disagrees with conditional layout flags."
                )
            validate_query_template(template=self.template, spec=spec)


def validate_query_template(  # noqa: C901, PLR0912
    *, template: EGMCarry, spec: EGMQuerySpec
) -> None:
    """Check feature leaves against their declared coordinate, axes, and dtypes."""
    query = template.query_data
    if query is None:
        raise TypeError("Query authority requires numeric query data.")
    from _lcm.egm.query_data import EGMQueryData  # noqa: PLC0415

    if type(query) is not EGMQueryData:
        raise TypeError("Query data must be an exact numeric record.")
    EGMQueryData.__post_init__(query)
    shape = template.endog_grid.shape
    if template.endog_grid.dtype != canonical_float_dtype():
        raise TypeError("Query coordinates require the canonical float dtype.")
    if (
        template.taste_shock_scale.shape != ()
        or template.taste_shock_scale.dtype != canonical_float_dtype()
    ):
        raise TypeError("Query taste scale requires a canonical floating scalar.")
    prefix = shape[:-1]
    if template.breakpoints is not None and (
        template.breakpoints.ndim < 1
        or template.breakpoints.shape[:-1] != prefix
        or template.breakpoints.dtype != canonical_float_dtype()
    ):
        raise TypeError("Query breakpoint rows differ from canonical axes/dtype.")
    expected_rank = (
        len(spec.state_names) + len(spec.action_names) + bool(spec.outer_capacity)
    )
    if (
        len(prefix) != expected_rank
        or shape[-1] != spec.row_capacity
        or (spec.outer_capacity and prefix[-1] != spec.outer_capacity)
    ):
        raise TypeError("Query authority leading row roles disagree with the row bank.")
    for field in ("value", "marginal_utility", "policy"):
        row = getattr(template, field)
        if row is not None and (
            row.shape != shape or row.dtype != canonical_float_dtype()
        ):
            raise TypeError(f"Query {field} bank differs from declared row axes/dtype.")
    if (query.liquid_corner_data is not None) != (
        spec.corner_coordinate_kind == "liquid_row"
    ):
        raise TypeError(
            "Query authority and corner coordinate representation disagree."
        )
    interval_capacity = len(spec.interval_ids)
    for path, leaf in query.leaves().items():
        name = path[-1]
        dtype = (
            jnp.bool_
            if name.endswith("link_valid")
            else jnp.int32
            if name
            in (
                "node_kind",
                "owner_id",
                "side_id",
                "interval_id",
                "corner_kind",
                "corner_interval_id",
                "corner_side_id",
            )
            else canonical_float_dtype()
        )
        if path[0] == "liquid_corner_data":
            expected = (
                *prefix,
                spec.row_capacity - 1 if name == "link_valid" else spec.row_capacity,
                interval_capacity,
            )
        elif name == "floor_used":
            expected = prefix[: len(spec.state_names)]
        elif name.startswith("corner_"):
            expected = (*prefix, interval_capacity)
        else:
            expected = (
                *prefix,
                spec.row_capacity - 1 if name == "link_valid" else spec.row_capacity,
            )
        if leaf.shape != expected or leaf.dtype != dtype:
            raise TypeError(
                f"Query leaf {'.'.join(path)} differs from model-owned axes/dtype."
            )
