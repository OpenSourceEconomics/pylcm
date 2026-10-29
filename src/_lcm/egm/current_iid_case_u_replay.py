"""Original dated future inputs retained for private Case-U reconstruction.

This payload contains future continuations, never current-node policy statistics.
The private scalar route reconstructs actions with the original dated node solver.
"""

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import numpy as np

from _lcm.continuation import (
    ContinuationSpec,
    EGMContinuationSpec,
    ProjectedProducerDate,
)
from _lcm.egm.carry import EGMCarry
from _lcm.egm.step_core import _EgmKernelPieces
from _lcm.egm.upper_envelope.fues import QueryBracket
from _lcm.execution.core_program import ValueRead
from _lcm.simulation.replay_inputs import replay_payload_reads
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.typing import FlatParams, PytreeChild
from lcm.exceptions import InvalidSimulationInputError, UnsupportedOperationError
from lcm.solver_api import ArtifactKey, ReplayMode, ReplayRouteIdentity
from lcm.typing import ActionName, EGMAccuracy, EGMReplay, RegimeName

if TYPE_CHECKING:
    from _lcm.egm.continuation import _ChildRead
    from _lcm.egm.current_iid_nbegm import NBCaseUPoint, NBCaseURidePoint
    from _lcm.egm.current_iid_negm import NEGMCaseUPoint
    from _lcm.egm.current_iid_nnbegm import NNBEGMCaseUPoint
else:
    # This record is imported while the numerical engine is initialized.
    type _ChildRead = object
    type NBCaseUPoint = object
    type NBCaseURidePoint = object
    type NEGMCaseUPoint = object
    type NNBEGMCaseUPoint = object

CASE_U_FUTURE_REPLAY = ArtifactKey(
    type_id="pylcm.engine.case_u_future_replay", schema_version=1
)
CASE_U_REPLAY_IDENTITY = ReplayRouteIdentity(
    route_id="pylcm.engine.case_u_reconstruction", route_version=1
)
CASE_U_GRID_REPLAY_IDENTITY = ReplayRouteIdentity(
    route_id="pylcm.engine.case_u_grid_approximation", route_version=1
)


@dataclass(frozen=True, kw_only=True)
class CaseUReplayChildren:
    """Original shock-free children, borrowed until admitted retained export."""

    source: RegimeName
    period: int
    child_names: tuple[RegimeName, ...]
    child_values: tuple[EGMCarry, ...]

    @property
    def children(self) -> Mapping[RegimeName, EGMCarry]:
        """Restore the original names over the retained original carry buffers."""
        return MappingProxyType(
            dict(zip(self.child_names, self.child_values, strict=True))
        )

    def __post_init__(self) -> None:
        if not self.source or type(self.period) is not int or self.period < 0:
            raise TypeError("Case-U replay requires its actual source/date.")
        if (
            type(self.child_names) is not tuple
            or type(self.child_values) is not tuple
            or len(self.child_names) != len(self.child_values)
            or len(set(self.child_names)) != len(self.child_names)
            or any(type(name) is not str or not name for name in self.child_names)
        ):
            raise TypeError("Case-U replay lost its exact original child names.")
        if not self.child_values or any(
            not isinstance(child, EGMCarry) for child in self.child_values
        ):
            raise UnsupportedOperationError(
                "Case-U replay still requires original ordinary future carries."
            )


def _flatten_case_u_children(
    payload: CaseUReplayChildren,
) -> tuple[
    tuple[tuple[jax.tree_util.GetAttrKey, tuple[EGMCarry, ...]], ...],
    tuple[RegimeName, int, tuple[RegimeName, ...]],
]:
    return ((jax.tree_util.GetAttrKey("child_values"), payload.child_values),), (
        payload.source,
        payload.period,
        payload.child_names,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_case_u_children(
    metadata: tuple[RegimeName, int, tuple[RegimeName, ...]],
    children: Iterable[tuple[EGMCarry, ...]],
) -> CaseUReplayChildren:
    """Restore exact numeric/abstract leaves under the original static date."""
    (actual,) = tuple(children)
    result = object.__new__(CaseUReplayChildren)
    object.__setattr__(result, "source", metadata[0])
    object.__setattr__(result, "period", metadata[1])
    object.__setattr__(result, "child_names", metadata[2])
    object.__setattr__(result, "child_values", actual)
    return result


jax.tree_util.register_pytree_with_keys(
    CaseUReplayChildren, _flatten_case_u_children, _unflatten_case_u_children
)


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUReplayFutureDate:
    """Actual immutable future schema authority of one installed U producer."""

    source: RegimeName
    period: int
    specs: Mapping[RegimeName, ContinuationSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))
        if not self.source or type(self.period) is not int or self.period < 0:
            raise TypeError("Case-U future authority requires its actual source/date.")
        if not self.specs:
            raise UnsupportedOperationError(
                "Case-U replay has no actual future schema."
            )
        for spec in self.specs.values():
            if not isinstance(spec.template_at(period=self.period + 1), EGMCarry):
                raise UnsupportedOperationError(
                    "Case-U replay still requires ordinary typed future authority."
                )

    def template(self) -> CaseUReplayChildren:
        """Borrow original producer templates without executing array factories."""
        children = {}
        for target, spec in self.specs.items():
            child = spec.template_at(period=self.period + 1)
            if not isinstance(child, EGMCarry):
                raise TypeError("Case-U future template lost its original carry type.")
            children[target] = child
        return CaseUReplayChildren(
            source=self.source,
            period=self.period,
            child_names=tuple(children),
            child_values=tuple(children.values()),
        )

    def bind(
        self, *, children: Mapping[RegimeName, EGMCarry], period: int
    ) -> CaseUReplayChildren:
        """Package original buffers only after original date/schema agreement."""
        if period != self.period or tuple(children) != tuple(self.specs):
            raise InvalidSimulationInputError(
                "Case-U replay source/date/target roster differs from its producer."
            )
        for target, spec in self.specs.items():
            expected = spec.template_at(period=self.period + 1)
            actual = children[target]
            if (
                not isinstance(actual, EGMCarry)
                or jax.tree.structure(actual) != jax.tree.structure(expected)
                or tuple((x.shape, x.dtype) for x in jax.tree.leaves(actual))
                != tuple((x.shape, x.dtype) for x in jax.tree.leaves(expected))
            ):
                raise InvalidSimulationInputError(
                    "Case-U replay future payload differs from its dated schema."
                )
        return CaseUReplayChildren(
            source=self.source,
            period=self.period,
            child_names=tuple(children),
            child_values=tuple(children.values()),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUReplayPeriod:
    """Original dated constructor and numeric argument projection for one source."""

    future: CaseUReplayFutureDate
    date: ProjectedProducerDate
    pieces: _EgmKernelPieces | None
    fixed_flat_params: FlatParams
    transition_target_names: tuple[RegimeName, ...]
    savings_point_width: int | None
    stochastic_node_width: int | None
    envelope_cell_width: int
    nbegm_point: NBCaseUPoint | NBCaseURidePoint | None = None
    negm_point: NEGMCaseUPoint | NNBEGMCaseUPoint | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Genuine complete finite nested constructor, separate from NB ownership."""
    replay_refine_to_bracket: Callable[..., QueryBracket] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """The configured envelope's bracket finder without provenance, for replay."""

    @property
    def action_name(self) -> str:
        """Name the action from this date's original solver constructor."""
        if self.negm_point is not None:
            return self.negm_point.action_name
        if self.nbegm_point is not None:
            return self.nbegm_point.action_name
        if self.pieces is not None:
            return self.pieces.action_name
        raise TypeError("Case-U replay lacks its original action declaration.")

    @property
    def additional_continuous_action_names(self) -> tuple[ActionName, ...]:
        """Actual jointly selected outer action, when this source is nested."""
        return () if self.negm_point is None else (self.negm_point.outer_action,)

    @property
    def discrete_action_names(self) -> tuple[ActionName, ...]:
        """Declare the actual ride-cell branch codes returned with consumption."""
        if self.nbegm_point is not None:
            from _lcm.egm.current_iid_nbegm import NBCaseURidePoint  # noqa: PLC0415

            if isinstance(self.nbegm_point, NBCaseURidePoint):
                return self.nbegm_point.discrete_action_names
        return ()

    def __post_init__(self) -> None:  # noqa: C901, PLR0912 - exact independent source constructors
        if (
            sum(
                item is not None
                for item in (self.pieces, self.nbegm_point, self.negm_point)
            )
            != 1
        ):
            raise TypeError(
                "Case-U replay requires exactly one original solver constructor."
            )
        if self.nbegm_point is not None:
            from _lcm.egm.current_iid_nbegm import (  # noqa: PLC0415
                NBCaseUPoint,
                NBCaseURidePoint,
            )

            if type(self.nbegm_point) not in (NBCaseUPoint, NBCaseURidePoint):
                raise TypeError("Case-U replay requires its original NB point type.")
        if (
            self.date.authority.producer != self.future.source
            or self.date.authority.period != self.future.period
        ):
            raise TypeError("Case-U reconstruction lost its actual original date.")
        if self.negm_point is not None:
            from _lcm.egm.current_iid_negm import NEGMCaseUPoint  # noqa: PLC0415
            from _lcm.egm.current_iid_nnbegm import NNBEGMCaseUPoint  # noqa: PLC0415
            from _lcm.egm.step import _EGMStep  # noqa: PLC0415
            from _lcm.solution.dcegm import _DCEGMPeriodKernel  # noqa: PLC0415

            point = self.negm_point
            keeper, adjuster = point.owner.keeper_kernel, point.owner.adjuster_kernel
            if type(point) is NNBEGMCaseUPoint:
                from _lcm.solution.nnbegm import (  # noqa: PLC0415
                    _FiniteNNBEGMPeriodKernel,
                )

                if (
                    type(point.owner) is not _FiniteNNBEGMPeriodKernel
                    or point.period != self.future.period
                    or point.context.regime_name != self.future.source
                    or point.context.current_iid_plans.get(point.period)
                    is not point.plan
                    or tuple(self.future.specs) != (point.target,)
                    or point.keeper.kernel is not keeper
                    or point.adjuster.kernel is not adjuster
                ):
                    raise TypeError(
                        "Case-U replay lost its genuine finite NNBEGM source."
                    )
                for original in (point.keeper, point.adjuster):
                    original.__post_init__()
                    plan = original.continuation_plan
                    if (
                        plan.query_targets
                        or plan.scalar_targets
                        or tuple(plan.stateful_targets) != (point.target,)
                    ):
                        raise UnsupportedOperationError(
                            "Finite NNBEGM reconstruction requires original "
                            "ordinary rows."
                        )
                    for target, read in plan.child_reads.items():
                        _require_ordinary_passive_child_rows(
                            read=read,
                            spec=self.future.specs[target],
                            period=self.future.period + 1,
                        )
                return
            if (
                type(point) is not NEGMCaseUPoint
                or point.period != self.future.period
                or point.context.regime_name != self.future.source
                or point.context.current_iid_plans.get(point.period) is not point.plan
                or tuple(self.future.specs) != (point.target,)
                or not isinstance(keeper, _DCEGMPeriodKernel)
                or not isinstance(adjuster, _DCEGMPeriodKernel)
                or not isinstance(keeper.demand_step, _EGMStep)
                or not isinstance(adjuster.demand_step, _EGMStep)
                or point.keeper is not keeper.demand_step.pieces
                or point.adjuster is not adjuster.demand_step.pieces
            ):
                raise TypeError("Case-U replay lost its genuine finite NEGM source.")
            for original in (point.keeper, point.adjuster):
                if (
                    original.continuation_plan.query_targets
                    or original.continuation_plan.scalar_targets
                    or tuple(original.continuation_plan.stateful_targets)
                    != (point.target,)
                ):
                    raise UnsupportedOperationError(
                        "Finite NEGM reconstruction requires original "
                        "ordinary future rows."
                    )
                for target, read in original.continuation_plan.child_reads.items():
                    _require_ordinary_passive_child_rows(
                        read=read,
                        spec=self.future.specs[target],
                        period=self.future.period + 1,
                    )
            return
        if self.nbegm_point is not None:
            plan = self.nbegm_point.continuation_plan
        elif self.pieces is not None:
            plan = self.pieces.continuation_plan
        else:
            raise TypeError("Case-U replay has no original point constructor.")
        ride = (
            self.nbegm_point is not None and type(self.nbegm_point) is NBCaseURidePoint
        )
        if ride:
            if (
                plan.query_targets
                or plan.scalar_targets
                or tuple(plan.stateful_targets) != tuple(self.future.specs)
                or tuple(plan.child_reads) != tuple(self.future.specs)
            ):
                raise UnsupportedOperationError(
                    "NB ride reconstruction requires its original ordinary target rows."
                )
            for target, read in plan.child_reads.items():
                _require_ordinary_passive_child_rows(
                    read=read,
                    spec=self.future.specs[target],
                    period=self.future.period + 1,
                )
            return
        if any(
            (read.passive_grids and not ride)
            or read.row_values
            or read.projected_date is not None
            for read in plan.child_reads.values()
        ):
            raise UnsupportedOperationError(
                "Case-U reconstruction still requires scalar ordinary future rows."
            )


def _require_ordinary_passive_child_rows(
    *, read: _ChildRead, spec: ContinuationSpec, period: int
) -> None:
    """Keep genuine static passive row meshes under their original carry schema.

    Ordinary child reads precompute passive resource-row bindings even when
    their resources ignore those axes. These are generation-owned grid buffers,
    independent of projected dates and runtime stochastic node operands.
    """
    if not isinstance(spec, EGMContinuationSpec):
        raise UnsupportedOperationError(
            "Current-U reconstruction requires original ordinary child authority."
        )
    child = spec.template_at(period=period)
    if any(grid.ndim != 1 or not grid.size for grid in read.passive_grids):
        raise UnsupportedOperationError(
            "Current-U reconstruction lost its original ordinary passive-row schema."
        )
    shape = tuple(grid.shape[0] for grid in read.passive_grids)
    if (
        not isinstance(child, EGMCarry)
        or read.query is not None
        or read.projected_date is not None
        or read.query_target is not None
        or read.query_action_names
        or read.query_action_values
        or read.n_outer_candidates
        or read.discrete_state_names
        or read.stochastic_state_names
        or read.stochastic_node_values
        or read.process_grid_names
        or read.euler_draw_names
        or read.local_draw_names
        or read.co_map_state_names
        or read.has_taste_shocks
        or spec.layout.query_spec is not None
        or spec.layout.n_stacked_candidates
        or read.rows_share_state_grid != spec.layout.rows_share_state_grid
        or child.breakpoints is not None
        or child.query_data is not None
        or read.row_arg_names != read.passive_state_names
        or len(read.passive_state_names) != len(read.passive_grids)
        or len(read.row_values) != len(read.passive_grids)
        or read.row_block_shape != shape
        or any(
            values.shape != (math.prod(shape),) or values.dtype != grid.dtype
            for grid, values in zip(read.passive_grids, read.row_values, strict=True)
        )
        or child.value.shape[:-1] != shape
        or child.marginal_utility.shape != child.value.shape
        or child.endog_grid.shape not in (child.value.shape, child.value.shape[-1:])
    ):
        raise UnsupportedOperationError(
            "Current-U reconstruction lost its original ordinary passive-row schema."
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUReplaySpec:
    """Engine-owned original-constructor route; every numeric read stays an input."""

    regime: RegimeName
    periods: Mapping[int, CaseUReplayPeriod]
    artifact_key: ArtifactKey = CASE_U_FUTURE_REPLAY
    replay: EGMReplay = "candidate"
    accuracy: EGMAccuracy = "balanced"

    def __post_init__(self) -> None:
        if self.accuracy == "certified" and self.replay == "grid":
            raise UnsupportedOperationError(
                "The certified EGM profile refuses grid replay."
            )
        periods = dict(self.periods)
        if not periods or any(
            period != item.future.period or item.future.source != self.regime
            for period, item in periods.items()
        ):
            raise TypeError("Case-U reconstruction requires its exact dated producers.")
        object.__setattr__(self, "periods", MappingProxyType(periods))

    @property
    def reconstructs_candidates(self) -> bool:
        """Select original candidate reconstruction without erasing its authority."""
        return self.replay == "candidate"

    @property
    def replay_mode(self) -> ReplayMode:
        return (
            ReplayMode.EXACT_REPLAY
            if self.reconstructs_candidates
            else ReplayMode.VALID_RECOMPUTATION
        )

    @property
    def payload_type(self) -> None:
        return None

    @property
    def identity(self) -> ReplayRouteIdentity:
        return (
            CASE_U_REPLAY_IDENTITY
            if self.reconstructs_candidates
            else CASE_U_GRID_REPLAY_IDENTITY
        )

    @property
    def policy_applicable(self) -> bool:
        return False

    @property
    def policy_required(self) -> bool:
        return False

    @property
    def consumer_route(self) -> str:
        return (
            "case_u_reconstruction" if self.reconstructs_candidates else "case_u_grid"
        )


@dataclass(frozen=True, kw_only=True)
class CaseUReplayInput:
    """Future payload authenticated against its real Model/result/date authority."""

    payload: CaseUReplayChildren
    key: ArtifactKey = CASE_U_FUTURE_REPLAY

    def require(self, *, spec: CaseUReplaySpec, period: int) -> CaseUReplayChildren:
        if (
            not spec.reconstructs_candidates
            or self.key != spec.artifact_key
            or self.payload.source != spec.regime
            or self.payload.period != period
            or period not in spec.periods
        ):
            raise InvalidSimulationInputError(
                "Case-U reconstruction payload belongs to another producer/date/key."
            )
        spec.periods[period].future.bind(children=self.payload.children, period=period)
        return self.payload

    def reads(self, *, core: str) -> tuple[ValueRead, ...]:
        return case_u_future_reads(
            payload=self.payload,
            key=self.key,
            period=self.payload.period,
            regime=self.payload.source,
            core=core,
            argument="future",
        )

    def acquire(
        self, *, core: str, owner: PeriodSimulationReads
    ) -> CaseUReplayChildren:
        reads = iter(self.reads(core=core))

        def place(leaf: PytreeChild) -> PytreeChild:
            if not isinstance(leaf, jax.Array | np.ndarray):
                return leaf
            return owner.read(unit=self.payload.source, read=next(reads), value=leaf)

        return jax.tree.map(place, self.payload)


def case_u_future_reads(
    *,
    payload: CaseUReplayChildren,
    key: ArtifactKey,
    period: int,
    regime: RegimeName,
    core: str,
    argument: str = "future",
) -> tuple[ValueRead, ...]:
    """Address original stored leaves against the actual numeric tuple operand."""
    reads = replay_payload_reads(
        payload=payload,
        key=key,
        period=period,
        regime=regime,
        core=core,
        argument=argument,
    )
    if any(read.source.path[:1] != ("child_values",) for read in reads):
        raise TypeError("Case-U replay lost its original child-value read paths.")
    return tuple(
        replace(read, source=replace(read.source, path=read.source.path[1:]))
        for read in reads
    )


CASE_U_REPLAY_PROGRAM = "simulate_case_u_reconstruction"

jax.tree_util.register_dataclass(
    CaseUReplayInput, data_fields=("payload",), meta_fields=("key",)
)
