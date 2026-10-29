"""Dated, action-valued replay of the original Case-R resources artifact."""

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.current_iid import AdmittedGroupParams, CurrentIIDDrawPlan
from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.current_iid_native import BoundResourcesMap, replay_case_r_candidate
from _lcm.egm.current_iid_resources_artifact import (
    OriginalResourcesRow,
    ResourcesRowLeaf,
    resources_row_leaf_specs,
    resources_row_payload,
    resources_row_reads,
    restore_resources_row,
)
from _lcm.egm.nbegm_resources import NBEGMResourcesBank
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.execution.core_program import ValueRead
from _lcm.shock_overrides import CurrentIIDOverride
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.typing import (
    ConstraintFunction,
    EconFunctionKwargs,
    PRNGKeyND,
    PytreeChild,
    QAndFArg,
)
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.shocks import BaseInnovations, PhysicalValues, ShockName
from lcm.solver_api import ArtifactKey, ReplayMode, ReplayRouteIdentity
from lcm.typing import ActionName, BoolND, FloatND, IntND, ReferenceName, RegimeName

CASE_R_REPLAY_PROGRAM = "simulate_case_r_replay"
CASE_R_REPLAY_IDENTITY = ReplayRouteIdentity(
    route_id="pylcm.engine.case_r_owned", route_version=1
)


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseRReplayPeriod:
    """Numerical callbacks and roles owned by the genuine dated producer."""

    period: int
    liquid_state: ReferenceName
    action_name: ActionName
    resources: Callable[..., FloatND]
    utility: Callable[..., FloatND]
    coefficient: Fraction
    owner_capacity: int
    row_reads: tuple[ValueRead, ...]
    row_tree: jax.tree_util.PyTreeDef
    """Original native container layout, retained only by the model declaration."""
    row_leaf_specs: tuple[jax.ShapeDtypeStruct, ...]
    """Exact original leaf count, shapes and dtypes before artifact placement."""
    envelope_arithmetic: ComparisonArithmetic = field(
        default="ordinary", metadata={"fingerprint_omit_if_default": True}
    )
    """The producer's envelope comparison arithmetic, used to read an NB bank."""


@dataclass(frozen=True, kw_only=True)
class CaseRReplaySpec:
    """Model-owned identity; numeric row and call parameters remain operands."""

    regime: RegimeName
    artifact_key: ArtifactKey
    periods: Mapping[int, CaseRReplayPeriod]

    def __post_init__(self) -> None:
        """Freeze the producer dates independently of mutable caller mappings."""
        periods = dict(self.periods)
        if not periods or any(date != item.period for date, item in periods.items()):
            raise ValueError("Case-R replay requires its actual producer dates.")
        object.__setattr__(self, "periods", MappingProxyType(periods))

    @property
    def replay_mode(self) -> ReplayMode:
        """The original candidate and borrowing corner determine the action."""
        return ReplayMode.EXACT_REPLAY

    @property
    def payload_type(self) -> None:
        """The original row is addressed by its own key, not SIMULATION_POLICY."""
        return None

    @property
    def identity(self) -> ReplayRouteIdentity:
        """Bind the original resources artifact to its engine-owned reader."""
        return CASE_R_REPLAY_IDENTITY

    @property
    def policy_applicable(self) -> bool:
        """No SIMULATION_POLICY artifact belongs to this resources-row route."""
        return False

    @property
    def policy_required(self) -> bool:
        """The required original row has its separate artifact authority."""
        return False

    @property
    def consumer_route(self) -> str:
        """Name the specific action-valued engine family."""
        return "case_r_owned"


@dataclass(frozen=True, kw_only=True)
class CaseRReplayInput:
    """Original artifact after model/parameter result authentication."""

    regime: RegimeName
    period: int
    key: ArtifactKey
    row: OriginalResourcesRow

    def require(self, *, spec: CaseRReplaySpec, period: int) -> OriginalResourcesRow:
        """Refuse a foreign owner/date/key before constructing a decision."""
        if (
            self.regime != spec.regime
            or self.period != period
            or self.key != spec.artifact_key
            or period not in spec.periods
            or type(self.row) not in (OwnedResourcesRow, NBEGMResourcesBank)
        ):
            raise InvalidSimulationInputError(
                "Case-R replay artifact does not match its model-owned producer."
            )
        declaration = spec.periods[period]
        if (
            jax.tree.structure(self.row) != declaration.row_tree
            or resources_row_leaf_specs(row=self.row) != declaration.row_leaf_specs
        ):
            raise InvalidSimulationInputError(
                "Case-R replay artifact does not match its model-owned row schema."
            )
        return self.row

    def reads(self, *, core: str) -> tuple[ValueRead, ...]:
        """Declare the actual original resources-row leaves under their own key."""
        return resources_row_reads(
            row=self.row,
            key=self.key,
            period=self.period,
            regime=self.regime,
            core=core,
        )

    def acquire(
        self, *, core: str, owner: PeriodSimulationReads, spec: CaseRReplaySpec
    ) -> OriginalResourcesRow:
        """Authenticate before acquiring declared buffers through the period owner."""
        self.require(spec=spec, period=self.period)
        leaves = tuple(
            owner.read(unit=self.regime, read=read, value=leaf)
            for read, leaf in zip(
                self.reads(core=core), resources_row_payload(row=self.row), strict=True
            )
        )
        return jax.tree.unflatten(jax.tree.structure(self.row), leaves)


type CaseRReplayInputs = Mapping[int, Mapping[RegimeName, CaseRReplayInput]]

jax.tree_util.register_dataclass(
    CaseRReplayInput, data_fields=("row",), meta_fields=("regime", "period", "key")
)


type _CaseRReplayChild = PytreeChild | Mapping[ReferenceName, PytreeChild]


class CaseRReplayOutput(NamedTuple):
    """Action-valued decision, same frame and its integrity summaries."""

    actions: Mapping[ActionName, FloatND]
    value: FloatND
    physical: Mapping[ShockName, FloatND | IntND]
    law_invalid: BoolND
    invalid: BoolND
    feasible: BoolND

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[_CaseRReplayChild]], CaseRReplayOutput],
        tuple[None, tuple[_CaseRReplayChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_case_r_replay_output, (None, tuple(self))


def _flatten_case_r_replay_output(
    record: CaseRReplayOutput,
) -> tuple[tuple[_CaseRReplayChild, ...], None]:
    return tuple(record), None


def _flatten_case_r_replay_output_with_keys(
    record: CaseRReplayOutput,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _CaseRReplayChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), child)
        for name, child in zip(CaseRReplayOutput._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_case_r_replay_output(
    metadata: None,
    children: Iterable[_CaseRReplayChild],
) -> CaseRReplayOutput:
    """Restore numeric, abstract or role children under the fixed field schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(CaseRReplayOutput._fields):
        raise TypeError("Case-R replay reconstruction requires its exact fields.")
    return cast("CaseRReplayOutput", tuple.__new__(CaseRReplayOutput, leaves))


jax.tree_util.register_pytree_with_keys(
    CaseRReplayOutput,
    _flatten_case_r_replay_output_with_keys,
    _unflatten_case_r_replay_output,
    _flatten_case_r_replay_output,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseRActionReplay:
    """One compiled family over explicit rows, keys, overrides and parameters."""

    declaration: CaseRReplayPeriod
    draw_plan: CurrentIIDDrawPlan
    feasibility: ConstraintFunction

    def __call__(
        self,
        *,
        row: tuple[ResourcesRowLeaf, ...],
        states: Mapping[ReferenceName, FloatND | IntND],
        keys: Mapping[ShockName, PRNGKeyND],
        overrides: Mapping[ShockName, CurrentIIDOverride],
        params: EconFunctionKwargs,
        group_params: AdmittedGroupParams,
        period: IntND,
        age: FloatND | IntND,
    ) -> CaseRReplayOutput:
        """Realize once, select the original candidate and retain its frame."""
        declaration = self.declaration
        native_row = restore_resources_row(
            payload=row,
            tree=declaration.row_tree,
            leaf_specs=declaration.row_leaf_specs,
        )
        pool = {**params, "period": period, "age": age}
        resource_names = tuple(inspect.signature(declaration.resources).parameters)

        # The declared subject tiler maps this scalar reconstruction.
        def cell(
            *,
            subject: Mapping[ReferenceName, FloatND | IntND],
            addressed: Mapping[ShockName, PRNGKeyND],
            supplied: Mapping[ShockName, CurrentIIDOverride],
        ) -> CaseRReplayOutput:
            arguments = cast("Mapping[ReferenceName, QAndFArg]", {**pool, **subject})
            overrides: dict[ShockName, BaseInnovations | PhysicalValues] = {}
            active: dict[ShockName, BoolND] = {}
            for name, item in supplied.items():
                values, mask = item.values, item.active
                if isinstance(values, jax.ShapeDtypeStruct) or isinstance(
                    mask, jax.ShapeDtypeStruct
                ):
                    raise ExecutionPlanningError(
                        "Case-R override descriptors must be traced before execution."
                    )
                overrides[name] = (
                    BaseInnovations(values=values)
                    if item.kind == "base"
                    else PhysicalValues(values=values)
                )
                active[name] = jnp.asarray(mask)
                if active[name].dtype != jnp.bool_:
                    raise ExecutionPlanningError(
                        "Case-R override activation must be boolean."
                    )
            frame = self.draw_plan.realise(
                keys=addressed,
                arguments=arguments,
                group_params=group_params,
                overrides=overrides,
                override_active=active,
            )
            selected = replay_case_r_candidate(
                row=native_row,
                liquid=subject[declaration.liquid_state],
                physical=frame.physical,
                frame_valid=frame.valid & (period == declaration.period),
                resources=BoundResourcesMap(
                    function=declaration.resources,
                    liquid_state=declaration.liquid_state,
                    pool={**pool, **subject},
                    names=resource_names,
                ),
                utility=BoundUtilityOfAction(
                    utility_func=declaration.utility,
                    action_name=declaration.action_name,
                    bound={**pool, **subject},
                ),
                coefficient=declaration.coefficient,
                owner_capacity=declaration.owner_capacity,
                arithmetic=declaration.envelope_arithmetic,
            )
            actual = {
                **pool,
                **subject,
                **frame.physical,
                declaration.action_name: selected.policy,
            }
            ordinary = self.feasibility(
                **{
                    name: actual[name]
                    for name in inspect.signature(self.feasibility).parameters
                }
            )
            return CaseRReplayOutput(
                actions={declaration.action_name: selected.policy},
                value=selected.value,
                physical=dict(frame.physical),
                law_invalid=~frame.valid,
                invalid=~selected.valid,
                feasible=selected.feasible & ordinary,
            )

        return cell(subject=states, addressed=keys, supplied=overrides)
