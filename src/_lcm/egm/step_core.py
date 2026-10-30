"""The textbook DC-EGM step: Euler inversion to the upper-envelope value row.

The single-post-state algorithm a reviewer can read top to bottom. For one
discrete-choice combo it Euler-inverts the continuation over the savings grid,
appends the closed-form credit-constrained candidates, refines the candidate
value correspondence to its upper envelope, and publishes the value function and
the next period's carry rows on the regime's exogenous state grid. It reads the
expected continuation through `continuation`'s per-target carry reader and knows
nothing about asset-row mode, the multi-target machinery itself, or the kernel
orchestration that maps it over the discrete-combo product.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.continuation import EGMQuerySpec
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import (
    ContinuationPlan,
    bind_continuation,
)
from _lcm.egm.euler import invert_euler
from _lcm.egm.interp import (
    interp_on_padded_grid,
)
from _lcm.egm.nested_floor import PreparedNestedFloor, nested_floor_at_cell
from _lcm.egm.preferences import (
    AnalyticInverseMarginalUtility,
    BoundUtilityOfAction,
    get_numeric_inverse_marginal_utility,
)
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.upper_envelope import OwnedQueryBracket
from _lcm.egm.upper_envelope.fues import QueryBracket
from _lcm.egm.upper_envelope.segment_envelope import (
    OwnedEnvelope,
    _NegmBindingEnvelope,
    _OwnedNegmBindingEnvelope,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    ContinuationReadCounts,
    QueryInputStatus,
    merge_continuation_read_counts,
    merge_query_input_status,
    reduce_continuation_read_counts,
    shift_savings_grid,
)
from _lcm.typing import (
    ActionName,
    EconFunctionArg,
    EconFunctionKwargs,
    PytreeChild,
    RegimeName,
    StateName,
)
from _lcm.utils.dispatchers import map_over_leading_axis
from lcm.typing import (
    Bool1D,
    BoolND,
    Float1D,
    FloatND,
    Int1D,
    IntND,
    ReferenceName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)

# Refined-row fields every envelope backend returns, ahead of the optional
# read-support verdict.
_REFINED_ROW_FIELDS: int = 4
_MIN_RESOURCES_DOMAIN_POINTS: int = 2

type _ComboRows = tuple[Float1D, Float1D, Float1D, Float1D, Float1D, ScalarBool]
type _NodeOutput = tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]
type _NodeRows = tuple[FloatND, FloatND, FloatND, FloatND]
type _ConditionalComboRows = tuple[
    Float1D,
    Float1D,
    Float1D,
    Float1D,
    Float1D,
    ScalarBool,
    EGMQueryData | None,
    BoolND,
]

# Smallest constrained-segment action as a fraction of the segment's span.
# The constrained candidates are geometrically spaced from this offset toward
# the borrowing limit (additive, so a non-positive limit stays well-defined).
CONSTRAINED_OFFSET_FRACTION = 1e-4


@dataclass(frozen=True, kw_only=True)
class _EgmKernelPieces:
    """Build-time statics shared by every per-combo EGM computation."""

    euler_state_name: StateName
    """Name of the regime's continuous (Euler) state."""

    action_name: ActionName
    """Name of the regime's continuous action."""

    savings_nodes: Float1D
    """The exogenous end-of-period savings grid."""

    borrowing_limit: ScalarFloat
    """Lower bound of the savings grid."""

    n_constrained: int
    """Number of closed-form credit-constrained candidate points."""

    constrained_ratio: float
    """Static geometric spacing ratio of the constrained candidates."""

    n_pad: int
    """Static length of the envelope-refinement workspace (FUES/RFC, overflow)."""

    n_carry_rows: int
    """Static length of the persisted carry rows.

    Equals `n_pad` in the single-post-state kernel (the carry is the refined
    envelope) and `n_euler_nodes` in asset-row mode (the carry is one published
    point per exogenous Euler node, with no envelope-workspace padding)."""

    combo_names: tuple[StateName | ActionName, ...]
    """Discrete-state, passive-state, then discrete-action names (carry-axis order)."""

    euler_axis_in_V: int
    """Canonical axis of the Euler state in the published value-function array."""

    utility_func: Callable[..., FloatND]
    """The regime's concatenated utility function."""

    inverse_marginal_utility_func: Callable[..., FloatND] | None
    """The regime's concatenated inverse-marginal-utility function.

    `None` when the regime supplies no analytic inverse: EGM then inverts `u'`
    numerically (the iEGM path), deriving the marginal utility from `utility_func`
    at the call site.
    """

    own_resources_func: Callable[..., FloatND]
    """The regime's concatenated resources function."""

    feasibility_func: Callable[..., ScalarBool] | None
    """Current predicate evaluated with the state names bound at its route site."""

    build_W_kwargs: Callable[[EconFunctionKwargs], dict[ReferenceName, EconFunctionArg]]
    """Closure assembling the Bellman aggregator's keyword arguments."""

    refine: Callable[
        ...,
        tuple[Float1D, Float1D, Float1D, ScalarInt]
        | tuple[Float1D, Float1D, Float1D, ScalarInt, ScalarBool]
        | OwnedEnvelope
        | _NegmBindingEnvelope
        | _OwnedNegmBindingEnvelope,
    ]
    """The configured upper-envelope backend (single-post-state carry)."""

    refine_to_bracket: Callable[..., QueryBracket | OwnedQueryBracket]
    """The single-query bracket finder for the asset-row publish (builds the full
    refined row and slices the bracketing node pair)."""

    continuation_plan: ContinuationPlan
    """Build-time statics of the per-savings-node continuation aggregation."""

    lower_func: Callable[..., ScalarFloat] | None = None
    """Canonical callable floor, evaluated independently of action alternatives."""

    floor_status: BoundInputStatus | None = None
    """Mandatory integrity of this cell's actual floor and shifted nodes."""

    query_spec: EGMQuerySpec | None = None
    """Installed model authority for this conditional source, when present."""

    check_query_construction: bool = False
    """The compiled source owns a mandatory query-construction summary slot."""
    collect_continuation_read_counts: bool = False
    """Python-static optional event reduction on the original continuation calls."""

    _negm_current_publication: bool = False
    """The literal single-post-state exact backend can classify current queries."""

    _retain_nested_feasibility: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Keep current eligibility per ordinary action for an enclosing outer choice."""

    _prepared_nested_floor: PreparedNestedFloor | None = None
    """Invocation-local floor bank from the enclosing nested graph."""

    _prepared_nested_axes: tuple[tuple[StateName, Float1D | Int1D], ...] = ()
    """The actual supplied fixed-current grids indexing that transient bank."""


def bind_cell_floor(
    *, pieces: _EgmKernelPieces, pool: EconFunctionKwargs
) -> _EgmKernelPieces:
    """Translate the supplied savings offsets once at their fixed-cell owner."""
    if pieces._prepared_nested_floor is not None:  # noqa: SLF001
        floor, nodes, status = nested_floor_at_cell(
            prepared=pieces._prepared_nested_floor,  # noqa: SLF001
            offsets=pieces.savings_nodes - pieces.savings_nodes[0],
            axes=pieces._prepared_nested_axes,  # noqa: SLF001
            pool=pool,
        )
        return replace(
            pieces,
            savings_nodes=nodes,
            borrowing_limit=floor,
            lower_func=None,
            floor_status=status,
        )
    if pieces.lower_func is None:
        if pieces.query_spec is not None and pieces.query_spec.bound.kind == "literal":
            nodes = pieces.savings_nodes
            return replace(
                pieces,
                floor_status=BoundInputStatus(
                    nonfinite_floor=~jnp.isfinite(pieces.borrowing_limit),
                    invalid_shifted_grid=(
                        ~jnp.all(jnp.isfinite(nodes)) | jnp.any(nodes[1:] <= nodes[:-1])
                    ),
                ),
            )
        return pieces
    floor = jnp.asarray(pieces.lower_func(**pool), dtype=pieces.savings_nodes.dtype)
    offsets = pieces.savings_nodes - pieces.savings_nodes[0]
    nodes, status = shift_savings_grid(floor=floor, offsets=offsets)
    invalid = status.nonfinite_floor | status.invalid_shifted_grid
    return replace(
        pieces,
        savings_nodes=jnp.where(
            invalid, jnp.arange(nodes.size, dtype=nodes.dtype), nodes
        ),
        borrowing_limit=jnp.where(invalid, 0.0, floor),
        lower_func=None,
        floor_status=status,
    )


def _get_solve_one_combo(
    *,
    pieces: _EgmKernelPieces,
    pool: EconFunctionKwargs,
    state_grid: Float1D,
    next_regime_to_continuation: MappingProxyType[
        RegimeName, EGMCarry | ProjectedEGMCarry
    ],
    euler_point_width: int | None,
    savings_point_width: int | None,
    stochastic_node_width: int | None,
    envelope_cell_width: int,
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> Callable[
    [tuple[ScalarInt | ScalarFloat, ...]],
    _ComboRows | BoundedResult[_ConditionalComboRows],
]:
    """Build the per-combo EGM computation for one kernel invocation.

    `euler_point_width` is accepted for a uniform builder signature but unused:
    the single-post-state kernel solves once per combo, with no per-asset-node
    axis to tile (only the asset-row kernel runs that loop).

    `resolved_process_grids` maps each runtime-resolved process state to its
    solve-time grid, threaded into the continuation so a resources function
    reading a process node integrates over the resolved nodes.
    """
    del euler_point_width
    return _SolveOneCombo(
        pieces=pieces,
        pool=pool,
        state_grid=state_grid,
        next_regime_to_continuation=next_regime_to_continuation,
        savings_point_width=savings_point_width,
        stochastic_node_width=stochastic_node_width,
        envelope_cell_width=envelope_cell_width,
        resolved_process_grids=resolved_process_grids,
    )


type _OwnedResourcesChild = (
    PytreeChild
    | OwnedEnvelope
    | BoundInputStatus
    | tuple[QueryInputStatus, ...]
    | ContinuationReadCounts
)


class OwnedResourcesRow(NamedTuple):
    """Transient canonical source row with original owners and mandatory integrity."""

    envelope: OwnedEnvelope
    marginal: Float1D
    node_kind: IntND
    borrowing_limit: ScalarFloat
    upper_savings: ScalarFloat
    expected_floor_value: ScalarFloat
    discount_factor: ScalarFloat
    discounted_floor_value: ScalarFloat
    resources_domain: Float1D
    construction_invalid: ScalarBool
    floor_status: BoundInputStatus | None
    child_status: tuple[QueryInputStatus, ...] | None
    continuation_read_counts: ContinuationReadCounts | None

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[_OwnedResourcesChild]], OwnedResourcesRow],
        tuple[None, tuple[_OwnedResourcesChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_owned_resources_row, (None, tuple(self))


@dataclass(frozen=True, kw_only=True)
class _ConstructedCombo:
    """Invocation-local producer state, consumed before numerical publication."""

    pieces: _EgmKernelPieces
    combo_pool: EconFunctionKwargs
    discount_factor: ScalarFloat
    utility_of_action: BoundUtilityOfAction
    expected_values: FloatND
    publish_resources: FloatND
    candidate_dead: Bool1D
    lowest_euler_point: ScalarFloat
    refined: (
        tuple[Float1D, Float1D, Float1D, ScalarInt]
        | tuple[Float1D, Float1D, Float1D, ScalarInt, ScalarBool]
        | OwnedEnvelope
        | _NegmBindingEnvelope
    )
    binding_query: BoolND | None
    child_status: tuple[QueryInputStatus, ...] | None
    read_counts: ContinuationReadCounts | None


@dataclass(frozen=True, kw_only=True, eq=False)
class _SolveOneCombo:
    """The single-post-state EGM step for one kernel invocation, per combo.

    Takes the combo's values (discrete codes and passive node values)
    positionally so `jax.vmap` can batch over flattened combo arrays.
    """

    pieces: _EgmKernelPieces
    """Build-time statics shared by every per-combo computation."""

    pool: EconFunctionKwargs
    """The kernel's flat params, `period`, and `age`."""

    state_grid: Float1D
    """The regime's exogenous Euler-state grid."""

    next_regime_to_continuation: MappingProxyType[
        RegimeName, EGMCarry | ProjectedEGMCarry
    ]
    """The next period's EGM carries."""

    savings_point_width: int | None
    """Tile width of the per-savings-node loop; `None` runs it in one tile."""

    stochastic_node_width: int | None
    """Block width of the streamed node expectation; `None` folds one block."""

    envelope_cell_width: int
    """Number of exact-envelope node cells resolved together."""

    resolved_process_grids: Mapping[StateName, FloatND]
    """Solve-time grids of runtime-resolved process states."""

    def __call__(
        self,
        combo_values: tuple[ScalarInt | ScalarFloat, ...],
    ) -> _ComboRows | BoundedResult[_ConditionalComboRows]:
        """Run the EGM step for one (discrete x passive-node) combo.

        Returns:
            Tuple of the combo's value row on the exogenous state grid, its
            refined endogenous grid, the published consumption policy on that
            grid, the value and marginal-utility carry rows, and the
            backend's read-support verdict (whether the row's linear span is
            free of compacted coverage gaps, so the off-grid simulation read
            may consume it). Conditional sources append original candidate data
            and exact node feasibility, with integrity carried as bounded auxiliary.

        """
        constructed = self._construct(combo_values=combo_values)
        dtype = self.state_grid.dtype
        pieces = constructed.pieces
        combo_pool = constructed.combo_pool
        discount_factor = constructed.discount_factor
        utility_of_action = constructed.utility_of_action
        expected_values = constructed.expected_values
        publish_resources = constructed.publish_resources
        candidate_dead = constructed.candidate_dead
        refined = constructed.refined
        binding_query = constructed.binding_query
        child_status = constructed.child_status
        read_counts = constructed.read_counts
        conditional = pieces.query_spec is not None or pieces.check_query_construction
        refined_grid, refined_policy, refined_value, n_kept, read_supported = (
            _refined_rows(
                refined=refined,
                query_spec=pieces.query_spec,
                check_query_construction=pieces.check_query_construction,
            )
        )

        V_row, value_row, marginal_utility_row = _publish_V_and_carry_rows(
            refined_grid=refined_grid,
            refined_policy=refined_policy,
            refined_value=refined_value,
            n_kept=n_kept,
            n_pad=pieces.n_pad,
            publish_resources=publish_resources,
            lowest_euler_point=constructed.lowest_euler_point,
            borrowing_limit=pieces.borrowing_limit,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
            _negm_current_publication=binding_query is not None,
            _binding_query=binding_query,
        )

        no_live_candidate, V_row, value_row, marginal_utility_row = (
            _mask_dead_combo_rows(
                candidate_dead=candidate_dead,
                V_row=V_row,
                value_row=value_row,
                marginal_utility_row=marginal_utility_row,
            )
        )

        construction = (
            _source_construction_status(
                refined=refined,
                marginal_row=marginal_utility_row,
                expected_at_floor=expected_values[0],
            )
            if conditional
            else None
        )
        if construction is not None and pieces._negm_current_publication:  # noqa: SLF001
            # Mandatory current-resource integrity survives node-feasibility masks.
            construction = (
                construction[0] | jnp.any(~jnp.isfinite(publish_resources)),
                construction[1],
            )

        # A row with no live candidate has finite placeholder geometry for
        # continuation reads and their JVPs. NaN-valued candidates stay live.
        refined_grid, refined_policy = (
            jnp.where(
                no_live_candidate,
                jnp.arange(refined_grid.size, dtype=dtype),
                refined_grid,
            ),
            jnp.where(no_live_candidate, 0.0, refined_policy),
        )

        feasible = jnp.asarray(True)  # noqa: FBT003
        if pieces.feasibility_func is not None:
            # Owned rows evaluate current-state predicates only for publication.
            # Combo-only rows suppress their value and marginal together so
            # an infeasible alternative contributes no choice probability.
            feasible = (
                current_node_feasibility(
                    pieces=pieces, pool=combo_pool, state_grid=self.state_grid
                )
                if pieces.query_spec is not None
                else pieces.feasibility_func(**combo_pool)
            )
            if pieces.query_spec is None:
                V_row = jnp.where(feasible, V_row, -jnp.inf)
                value_row = jnp.where(feasible, value_row, -jnp.inf)
                marginal_utility_row = jnp.where(feasible, marginal_utility_row, 0.0)
                # Dead rows need finite geometry for continuation reads and JVPs.
                refined_grid = jnp.where(
                    feasible, refined_grid, jnp.arange(refined_grid.size, dtype=dtype)
                )
                refined_policy = jnp.where(feasible, refined_policy, 0.0)

        if pieces._negm_current_publication and not conditional:  # noqa: SLF001
            # Current publication integrity takes precedence over economic masks.
            V_row = jnp.where(jnp.isfinite(publish_resources), V_row, jnp.nan)

        if conditional:
            if not isinstance(refined, OwnedEnvelope) or construction is None:
                raise TypeError("Conditional publication requires owned envelope rows.")
            node_feasible = publish_resources > pieces.borrowing_limit
            if pieces.feasibility_func is not None:
                node_feasible = node_feasible & feasible
            return _conditional_combo_output(
                pieces=pieces,
                refined=refined,
                V_row=jnp.where(node_feasible, V_row, -jnp.inf),
                value_row=value_row,
                marginal_row=marginal_utility_row,
                expected_at_floor=expected_values[0],
                node_feasible=node_feasible,
                read_supported=read_supported,
                child_status=child_status,
                grid_row=refined_grid,
                policy_row=refined_policy,
                construction_invalid=construction[0],
                kinds=construction[1],
                # Conditional banks require these reads even at rejected public states.
                continuation_read_counts=read_counts,
            )

        return _with_literal_child_status(
            output=(
                V_row,
                refined_grid.astype(dtype),
                refined_policy.astype(dtype),
                value_row,
                marginal_utility_row,
                read_supported,
            ),
            pieces=pieces,
            child_status=child_status,
            node_feasible=(publish_resources > pieces.borrowing_limit) & feasible,
            continuation_read_counts=reduce_continuation_read_counts(
                counts=read_counts, active=feasible
            ),
        )

    def owned_resources_row(
        self,
        *,
        combo_values: tuple[ScalarInt | ScalarFloat, ...],
        resources_domain: Float1D,
        refine: Callable[..., OwnedEnvelope],
    ) -> OwnedResourcesRow:
        """Construct one canonical transient row on its whole resources domain.

        The caller supplies the same common finite resources domain for every
        current draw that reuses this row. Original child expectation, Euler
        inversion, constrained candidates and envelope selection run once here.
        No query authority is manufactured and no current node bank is returned.
        """
        row = self._construct(
            combo_values=combo_values,
            resources_domain=resources_domain,
            owned_refine=refine,
        )
        if not isinstance(row.refined, OwnedEnvelope):
            raise TypeError("A projected source requires its native owned envelope.")
        marginal = jax.vmap(jax.grad(row.utility_of_action))(
            jnp.where(jnp.isnan(row.refined.policy), 1.0, row.refined.policy)
        )
        marginal = jnp.where(jnp.isnan(row.refined.policy), jnp.nan, marginal)
        marginal = jnp.where(jnp.isneginf(row.refined.value), 0.0, marginal)
        invalid, kinds = _source_construction_status(
            refined=row.refined,
            marginal_row=marginal,
            expected_at_floor=row.expected_values[0],
        )
        domain_invalid = (
            resources_domain.size < _MIN_RESOURCES_DOMAIN_POINTS
            or resources_domain.ndim != 1
        )
        invalid = (
            invalid
            | domain_invalid
            | jnp.any(~jnp.isfinite(resources_domain))
            | jnp.any(resources_domain[1:] <= resources_domain[:-1])
        )
        return OwnedResourcesRow(
            envelope=row.refined,
            marginal=marginal,
            node_kind=kinds,
            borrowing_limit=row.pieces.borrowing_limit,
            upper_savings=row.pieces.savings_nodes[-1],
            expected_floor_value=row.expected_values[0],
            discount_factor=row.discount_factor,
            discounted_floor_value=row.discount_factor * row.expected_values[0],
            resources_domain=resources_domain,
            construction_invalid=invalid,
            floor_status=row.pieces.floor_status,
            child_status=row.child_status,
            continuation_read_counts=row.read_counts,
        )

    def _construct(
        self,
        *,
        combo_values: tuple[ScalarInt | ScalarFloat, ...],
        resources_domain: Float1D | None = None,
        owned_refine: Callable[..., OwnedEnvelope] | None = None,
    ) -> _ConstructedCombo:
        """Run the shared candidate producer before publication chooses coordinates."""
        pieces = self.pieces
        dtype = self.state_grid.dtype
        combo_pool = {
            **self.pool,
            **dict(zip(pieces.combo_names, combo_values, strict=True)),
        }
        pieces = bind_cell_floor(pieces=pieces, pool=combo_pool)
        # Validation pins the default Bellman aggregator, whose single
        # non-(utility, CE) parameter is the discount factor.
        (discount_factor_arg,) = tuple(pieces.build_W_kwargs(combo_pool).values())
        discount_factor = cast("ScalarFloat", discount_factor_arg)
        utility_of_action = BoundUtilityOfAction(
            utility_func=pieces.utility_func,
            action_name=pieces.action_name,
            bound=combo_pool,
        )
        compute_node = _get_compute_node(
            pieces=pieces,
            combo_pool=combo_pool,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            next_regime_to_continuation=self.next_regime_to_continuation,
            dtype=dtype,
            stochastic_node_width=self.stochastic_node_width,
            resolved_process_grids=self.resolved_process_grids,
        )
        child_status = None
        conditional = (
            pieces.query_spec is not None
            or pieces.check_query_construction
            or owned_refine is not None
        )
        if conditional:
            candidates, child_status, read_counts = _compute_nodes_after_first(
                first=compute_node(pieces.savings_nodes[0]),
                compute_node=compute_node,
                savings_nodes=pieces.savings_nodes,
                savings_point_width=self.savings_point_width,
            )
        else:
            candidates, child_status, read_counts = (
                _legacy_candidates_with_child_status(
                    pieces=pieces,
                    compute_node=compute_node,
                    savings_point_width=self.savings_point_width,
                )
            )
        actions, endog_grid, values, expected_values = candidates
        own_resources_of_state = ResourcesOfState(
            resources_func=pieces.own_resources_func,
            euler_state_name=pieces.euler_state_name,
            bound=combo_pool,
        )
        publish_resources = (
            jax.vmap(own_resources_of_state)(self.state_grid)
            if resources_domain is None
            else resources_domain
        )

        constrained_actions, constrained_values = _compute_constrained_candidates(
            first_endogenous_point=endog_grid[0],
            publish_resources=(
                publish_resources if resources_domain is None else resources_domain
            ),
            borrowing_limit=pieces.borrowing_limit,
            n_constrained=pieces.n_constrained,
            constrained_ratio=pieces.constrained_ratio,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

        (
            candidate_grid,
            candidate_policy,
            candidate_value,
            candidate_savings,
            candidate_dead,
        ) = _assemble_combo_candidates(
            pieces=pieces,
            constrained_actions=constrained_actions,
            constrained_values=constrained_values,
            endog_grid=endog_grid,
            actions=actions,
            values=values,
        )
        candidate_marginal = _candidate_supgradient(
            policy=candidate_policy,
            dead=candidate_dead,
            utility_of_action=utility_of_action,
        )
        # No explicit `segment_id` is passed: the candidate chain is a single
        # connected branch (the constrained segment joins the unconstrained Euler
        # branch continuously at the borrowing limit). Within-period discrete
        # choices are maximized across combos in the V array, not stacked here, so
        # `refine` never sees two unrelated branches that could be bridged. A
        # future-kink-induced fold is data-emergent (the inverted endogenous grid
        # decreases) rather than known a priori, so it is the backend's geometry —
        # not an explicit label — that resolves it. Explicit topology is the
        # oracle/test path (LTM/query accept `segment_id`); production has none
        # to emit.
        # A backend that cannot certify off-grid read support returns the
        # four-element row and the read is treated as supported; only a backend
        # that computes the flag can withhold it.
        binding_request = (
            {
                "_binding_source_count": pieces.n_constrained + 1,
                "_binding_queries": publish_resources,
            }
            if pieces._negm_current_publication  # noqa: SLF001
            else {}
        )
        refine = pieces.refine if owned_refine is None else owned_refine
        refined = refine(
            cell_width=self.envelope_cell_width,
            endog_grid=jnp.where(candidate_dead, jnp.nan, candidate_grid),
            policy=jnp.where(candidate_dead, jnp.nan, candidate_policy),
            value=jnp.where(candidate_dead, jnp.nan, candidate_value),
            marginal_utility=candidate_marginal,
            savings=jnp.where(candidate_dead, jnp.nan, candidate_savings),
            **binding_request,
        )
        binding_query = (
            refined.binding_query
            if isinstance(refined, _NegmBindingEnvelope | _OwnedNegmBindingEnvelope)
            else None
        )
        if isinstance(refined, _OwnedNegmBindingEnvelope):
            refined = refined.owned
        return _ConstructedCombo(
            pieces=pieces,
            combo_pool=combo_pool,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            expected_values=expected_values,
            publish_resources=publish_resources,
            candidate_dead=candidate_dead,
            # A NaN Euler candidate keeps the minimum NaN, which disables the
            # closed-form publication rather than reading past the poison.
            lowest_euler_point=jnp.min(
                jnp.where(candidate_dead[pieces.n_constrained :], jnp.inf, endog_grid)
            ),
            refined=refined,
            binding_query=binding_query,
            child_status=child_status,
            read_counts=read_counts,
        )


def _assemble_combo_candidates(
    *,
    pieces: _EgmKernelPieces,
    constrained_actions: Float1D,
    constrained_values: Float1D,
    endog_grid: Float1D,
    actions: Float1D,
    values: Float1D,
) -> tuple[Float1D, Float1D, Float1D, Float1D, Bool1D]:
    """Append constrained and Euler candidates with their pristine savings sources."""
    candidate_grid = jnp.concatenate(
        [pieces.borrowing_limit + constrained_actions, endog_grid]
    )
    candidate_policy = jnp.concatenate([constrained_actions, actions])
    candidate_value = jnp.concatenate([constrained_values, values])
    # Exogenous source savings per candidate: the savings node for each Euler
    # candidate (`endog_grid = savings_node + action`, so the implied
    # `endog_grid - policy` equals it in exact arithmetic), the borrowing
    # limit for the constrained candidates (their savings is pinned there).
    # FUES compares these pristine sources instead of the lossy implied
    # difference, so a same-source tie is never dropped by rounding.
    candidate_savings = jnp.concatenate(
        [
            jnp.full_like(constrained_actions, pieces.borrowing_limit),
            pieces.savings_nodes,
        ]
    )
    # A `-inf`-valued candidate (e.g. a corner whose continuation is
    # `-inf`) is dominated by every finite candidate and would inject
    # `-inf - (-inf) = NaN` into the envelope scan's gradient arithmetic.
    # Mask the triple to NaN — the scan's absent form. NaN-valued
    # candidates stay as they are: genuine poison must keep propagating.
    candidate_dead = jnp.isneginf(candidate_value)
    return (
        candidate_grid,
        candidate_policy,
        candidate_value,
        candidate_savings,
        candidate_dead,
    )


def _mask_dead_combo_rows(
    *,
    candidate_dead: Bool1D,
    V_row: Float1D,
    value_row: Float1D,
    marginal_utility_row: Float1D,
) -> tuple[ScalarBool, Float1D, Float1D, Float1D]:
    """Publish zero marginal and absent values when every candidate is dead."""
    # A combo with no live candidate (its entire continuation is `-inf`)
    # is worth `-inf` everywhere, like an infeasible combo.
    no_live_candidate = jnp.all(candidate_dead)
    V_row = jnp.where(no_live_candidate, -jnp.inf, V_row)
    value_row = jnp.where(no_live_candidate, -jnp.inf, value_row)
    marginal_utility_row = jnp.where(no_live_candidate, 0.0, marginal_utility_row)

    return no_live_candidate, V_row, value_row, marginal_utility_row


def current_node_feasibility(
    *, pieces: _EgmKernelPieces, pool: EconFunctionKwargs, state_grid: Float1D
) -> BoolND:
    """Evaluate predicates on the actual publication grid after inversion."""
    if pieces.feasibility_func is None:
        return jnp.ones(state_grid.shape, dtype=jnp.bool_)
    return jax.vmap(_CurrentNodeFeasibility(pieces=pieces, pool=pool))(state_grid)


@dataclass(frozen=True, kw_only=True, eq=False)
class _CurrentNodeFeasibility:
    """Bind the actual current Euler node beside a fixed combo and its clock."""

    pieces: _EgmKernelPieces
    """The installed current-state predicate and liquid role."""
    pool: EconFunctionKwargs
    """Actual discrete/passive values and qualified parameters for this period."""

    def __call__(self, state: ScalarFloat) -> ScalarBool:
        predicate = self.pieces.feasibility_func
        if predicate is None:
            raise TypeError("A current-node predicate must be installed.")
        return predicate(**{**self.pool, self.pieces.euler_state_name: state})


def _with_literal_child_status(
    *,
    output: _ComboRows,
    pieces: _EgmKernelPieces,
    child_status: tuple[QueryInputStatus, ...] | None,
    node_feasible: BoolND,
    continuation_read_counts: ContinuationReadCounts | None = None,
) -> _ComboRows | BoundedResult[_ConditionalComboRows]:
    """Transport inherited child integrity beside unchanged legacy source rows."""
    if (
        not pieces.continuation_plan.query_targets
        and continuation_read_counts is None
        and not pieces._retain_nested_feasibility  # noqa: SLF001
    ):
        return output
    if pieces.continuation_plan.query_targets and child_status is None:
        raise TypeError("Installed child readers require reduced query summaries.")
    return BoundedResult(
        output=(*output, None, node_feasible),
        bound_status=(),
        query_status=child_status,
        continuation_read_counts=continuation_read_counts,
    )


def _legacy_candidates_with_child_status(
    *,
    pieces: _EgmKernelPieces,
    compute_node: Callable[[ScalarFloat], _NodeOutput | BoundedResult[_NodeOutput]],
    savings_point_width: int | None,
) -> tuple[
    _NodeRows, tuple[QueryInputStatus, ...] | None, ContinuationReadCounts | None
]:
    """Map the legacy candidate schedule while reducing installed child summaries."""
    candidates = _compute_nodes_over_savings(
        compute_node=compute_node,
        savings_nodes=pieces.savings_nodes,
        savings_point_width=savings_point_width,
        bound_summary=bool(pieces.continuation_plan.query_targets)
        or pieces.collect_continuation_read_counts,
    )
    return (
        (
            candidates.output,
            candidates.query_status,
            candidates.continuation_read_counts,
        )
        if isinstance(candidates, BoundedResult)
        else (candidates, None, None)
    )


def _refined_rows(
    *,
    refined: tuple[Float1D, Float1D, Float1D, ScalarInt]
    | tuple[Float1D, Float1D, Float1D, ScalarInt, ScalarBool]
    | OwnedEnvelope
    | _NegmBindingEnvelope,
    query_spec: EGMQuerySpec | None,
    check_query_construction: bool = False,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt, ScalarBool]:
    """Separate static owned publication from the legacy optional support flag."""
    if query_spec is not None or check_query_construction:
        if not isinstance(refined, OwnedEnvelope):
            raise TypeError("Conditional publication requires owned envelope rows.")
        edge = (jnp.arange(refined.link_valid.size) < refined.n_kept - 1) & (
            refined.grid[1:] > refined.grid[:-1]
        )
        return (
            refined.grid,
            refined.policy,
            refined.value,
            refined.n_kept,
            jnp.all(~edge | refined.link_valid),
        )
    if isinstance(refined, OwnedEnvelope):
        raise TypeError("Owned envelope rows require installed source authority.")
    if isinstance(refined, _NegmBindingEnvelope):
        return (
            refined.grid,
            refined.policy,
            refined.value,
            refined.n_kept,
            jnp.asarray(a=True),
        )
    grid, policy, value, n_kept = refined[:_REFINED_ROW_FIELDS]
    support = (
        refined[_REFINED_ROW_FIELDS]
        if len(refined) > _REFINED_ROW_FIELDS
        else jnp.asarray(True)  # noqa: FBT003
    )
    return grid, policy, value, n_kept, support


def _source_construction_status(
    *,
    refined: tuple[Float1D, Float1D, Float1D, ScalarInt]
    | tuple[Float1D, Float1D, Float1D, ScalarInt, ScalarBool]
    | OwnedEnvelope
    | _NegmBindingEnvelope,
    marginal_row: Float1D,
    expected_at_floor: ScalarFloat,
) -> tuple[ScalarBool, IntND]:
    """Inspect the real emitted prefix before economic masks change carry rows."""
    if not isinstance(refined, OwnedEnvelope):
        raise TypeError("Query construction requires owned envelope rows.")
    active = jnp.arange(refined.grid.size) < refined.n_kept
    represented = (
        jnp.isfinite(refined.grid)
        & jnp.isfinite(refined.policy)
        & jnp.isfinite(refined.savings)
        & (jnp.isfinite(refined.value) | jnp.isneginf(refined.value))
        & jnp.isfinite(marginal_row)
    )
    known_corner = jnp.isfinite(expected_at_floor) | jnp.isneginf(expected_at_floor)
    invalid = (
        (refined.n_kept > refined.grid.size)
        | jnp.any(active & ~represented)
        | ~known_corner
    )
    kinds = jnp.where(
        active & represented,
        jnp.where(
            jnp.isneginf(refined.value),
            QueryNodeKind.ECONOMIC_MINUS_INF,
            QueryNodeKind.FINITE,
        ),
        QueryNodeKind.PAD,
    ).astype(jnp.int32)
    return invalid, kinds


def _conditional_combo_output(
    *,
    pieces: _EgmKernelPieces,
    refined: OwnedEnvelope,
    V_row: Float1D,
    value_row: Float1D,
    marginal_row: Float1D,
    expected_at_floor: ScalarFloat,
    node_feasible: BoolND,
    read_supported: ScalarBool,
    child_status: tuple[QueryInputStatus, ...] | None,
    grid_row: Float1D,
    policy_row: Float1D,
    construction_invalid: ScalarBool,
    kinds: IntND,
    continuation_read_counts: ContinuationReadCounts | None = None,
) -> BoundedResult[_ConditionalComboRows]:
    """Publish original source records and mandatory prefix integrity separately."""
    spec = pieces.query_spec
    if spec is None and not pieces.check_query_construction:
        raise TypeError("Conditional publication requires installed source authority.")
    data = None
    if spec is not None:
        known_corner = jnp.isfinite(expected_at_floor) | jnp.isneginf(expected_at_floor)
        corner_kind = jnp.where(
            known_corner,
            jnp.where(
                jnp.isneginf(expected_at_floor),
                QueryNodeKind.ECONOMIC_MINUS_INF,
                QueryNodeKind.FINITE,
            ),
            QueryNodeKind.PAD,
        ).astype(jnp.int32)
        data = EGMQueryData(
            savings=refined.savings,
            node_kind=kinds,
            link_valid=refined.link_valid,
            owner_id=refined.owner_id,
            side_id=refined.side_id,
            floor_used=pieces.borrowing_limit,
            corner_continuation=expected_at_floor[None],
            corner_kind=corner_kind[None],
            corner_interval_id=jnp.asarray(spec.interval_ids, dtype=jnp.int32),
            corner_side_id=jnp.asarray(spec.interval_side_ids, dtype=jnp.int32),
        )
    return BoundedResult(
        output=(
            V_row,
            grid_row,
            policy_row,
            value_row,
            marginal_row,
            read_supported,
            data,
            node_feasible,
        ),
        bound_status=() if pieces.floor_status is None else (pieces.floor_status,),
        continuation_read_counts=continuation_read_counts,
        query_status=(
            *(child_status or ()),
            QueryInputStatus(
                nonfinite_floor=jnp.asarray(False),  # noqa: FBT003
                invalid_shifted_grid=jnp.asarray(False),  # noqa: FBT003
                coverage_missing=construction_invalid,
            ),
        ),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class ResourcesOfState:
    """The regime's resources as a function of its Euler state, all else bound."""

    resources_func: Callable[..., FloatND]
    """The regime's concatenated resources function."""

    euler_state_name: StateName
    """The regime's name for its continuous (Euler) state."""

    bound: EconFunctionKwargs
    """Every other argument of `resources_func`, by name."""

    def __call__(self, state_value: ScalarFloat) -> ScalarFloat:
        return self.resources_func(**{self.euler_state_name: state_value}, **self.bound)


def tile_block_size(*, width: int | None, extent: int) -> int:
    """Return the `lax.map` block a planner tile width asks for; `0` fuses.

    A width the axis cannot fill — `None` for an axis no program declares, or
    one at or above the extent — runs the loop as a single fused vmap, which is
    what `map_over_leading_axis` and `productmap` read a block of `0` as.
    """
    if width is None or width >= extent:
        return 0
    return width


def _compute_nodes_over_savings(
    *,
    compute_node: Callable[[ScalarFloat], _NodeOutput | BoundedResult[_NodeOutput]],
    savings_nodes: Float1D,
    savings_point_width: int | None,
    bound_summary: bool = False,
) -> _NodeRows | BoundedResult[_NodeRows]:
    """Run `compute_node` over every savings node, in tiles the plan sizes.

    A `savings_point_width` below the grid length runs the per-savings-node
    continuation — the dominant egm_step working buffer (savings nodes by the
    child stochastic mesh by the combo block) — in `lax.map` tiles, bounding
    peak memory; a width covering the whole grid keeps the fused vmap. The
    output is identical either way: the per-node `(action, endogenous
    resources, value, expected continuation)` candidates stacked along the
    savings axis, which the constrained-region assembly and the upper envelope
    then consume on the full grid. Installed child query slots use
    `bound_summary` to OR-reduce integrity inside each tile while retaining
    only the four ordinary candidate arrays on the savings axis.
    """
    return map_over_leading_axis(
        func=compute_node,
        xs=savings_nodes,
        batch_size=tile_block_size(
            width=savings_point_width, extent=int(savings_nodes.shape[0])
        ),
        bound_summary=bound_summary,
    )


def _compute_nodes_after_first(
    *,
    first: _NodeOutput | BoundedResult[_NodeOutput],
    compute_node: Callable[[ScalarFloat], _NodeOutput | BoundedResult[_NodeOutput]],
    savings_nodes: Float1D,
    savings_point_width: int | None,
) -> tuple[
    _NodeRows, tuple[QueryInputStatus, ...] | None, ContinuationReadCounts | None
]:
    """Reuse the first candidate and fold the remaining child slots within tiles."""
    first_output = first.output if isinstance(first, BoundedResult) else first
    status = first.query_status if isinstance(first, BoundedResult) else None
    counts = (
        first.continuation_read_counts if isinstance(first, BoundedResult) else None
    )
    if savings_nodes.size == 1:
        return (
            cast("_NodeRows", tuple(jnp.atleast_1d(value) for value in first_output)),
            status,
            counts,
        )
    tail = _compute_nodes_over_savings(
        compute_node=compute_node,
        savings_nodes=savings_nodes[1:],
        savings_point_width=savings_point_width,
        bound_summary=isinstance(first, BoundedResult),
    )
    tail_output = tail.output if isinstance(tail, BoundedResult) else tail
    if isinstance(tail, BoundedResult):
        status = merge_query_input_status(left=status, right=tail.query_status)
        counts = merge_continuation_read_counts(
            left=counts, right=tail.continuation_read_counts
        )
    candidates = tuple(
        jnp.concatenate((jnp.atleast_1d(value), tail_value))
        for value, tail_value in zip(first_output, tail_output, strict=True)
    )
    return cast("_NodeRows", candidates), status, counts


def _get_compute_node(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: EconFunctionKwargs,
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    next_regime_to_continuation: MappingProxyType[
        RegimeName, EGMCarry | ProjectedEGMCarry
    ],
    dtype: np.dtype,
    stochastic_node_width: int | None,
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
    economic_continuation: Callable[
        [ScalarFloat], BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ]
    | None = None,
) -> Callable[[ScalarFloat], _NodeOutput | BoundedResult[_NodeOutput]]:
    """Build the per-savings-node Euler inversion for one discrete combo."""
    continuation = (
        economic_continuation
        if economic_continuation is not None
        else bind_continuation(
            plan=pieces.continuation_plan,
            combo_pool=combo_pool,
            next_regime_to_continuation=next_regime_to_continuation,
            dtype=dtype,
            stochastic_node_width=stochastic_node_width,
            resolved_process_grids=resolved_process_grids,
            collect_continuation_read_counts=pieces.collect_continuation_read_counts,
        )
    )

    analytic_inverse = pieces.inverse_marginal_utility_func
    inverse_marginal_utility: Callable[[ScalarFloat], ScalarFloat]
    if analytic_inverse is not None:
        inverse_marginal_utility = AnalyticInverseMarginalUtility(
            analytic_inverse=analytic_inverse, bound=combo_pool
        )
    else:
        # iEGM: no analytic inverse supplied, so invert `u'` numerically. The
        # marginal utility is the action-derivative of the combo-bound utility;
        # the bracket spans a small floor to a generous multiple of the savings
        # grid's top node (interior optima lie within resources, the savings
        # scale; a clamped near-zero-marginal corner whose root exceeds the
        # bracket lands far to the right and is discarded by the envelope, exactly
        # as the analytic path's large value is).
        #
        # `action_upper` must dominate every feasible consumption. It is derived
        # from the savings-grid scale (the model's own resources scale) rather than
        # the resources bound directly, since the per-savings-node kernel has no
        # single current-state resources value here. A model whose optimal
        # consumption can genuinely exceed ~1000x the top savings node (income far
        # above the savings scale) is mis-scaled for this grid and would clamp a
        # real interior root to the bound (reported with `dc/dm = 0`, like a binding
        # corner); widen the savings grid or supply an analytic inverse in that case.
        action_upper = pieces.savings_nodes[-1] * 1000.0 + 1000.0
        action_lower = jnp.asarray(1e-8, dtype=action_upper.dtype)
        inverse_marginal_utility = get_numeric_inverse_marginal_utility(
            marginal_utility=jax.grad(utility_of_action),
            action_lower=action_lower,
            action_upper=action_upper,
        )
    return _ComputeNode(
        continuation=continuation,
        discount_factor=discount_factor,
        inverse_marginal_utility=inverse_marginal_utility,
        utility_of_action=utility_of_action,
        query_slots_required=bool(pieces.continuation_plan.query_targets),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _ComputeNode:
    """Euler-invert one savings node against the bound continuation."""

    continuation: Callable[
        [ScalarFloat],
        tuple[ScalarFloat, ScalarFloat]
        | BoundedResult[tuple[ScalarFloat, ScalarFloat]],
    ]
    """Expected continuation value and marginal at a savings node."""

    discount_factor: ScalarFloat
    """The combo's discount factor."""

    inverse_marginal_utility: Callable[[ScalarFloat], ScalarFloat]
    """The Euler inversion `(u')^{-1}`, parameters bound."""

    utility_of_action: Callable[[ScalarFloat], ScalarFloat]
    """Felicity of the consumption action, everything else bound."""
    query_slots_required: bool = False
    """Whether this installed continuation independently declares child slots."""

    def __call__(
        self,
        savings_value: ScalarFloat,
    ) -> _NodeOutput | BoundedResult[_NodeOutput]:
        continuation = self.continuation(savings_value)
        if type(continuation) is BoundedResult:
            continuation = cast(
                "BoundedResult[tuple[ScalarFloat, ScalarFloat]]", continuation
            )
            if continuation.bound_status or (
                continuation.query_status is None
                and (
                    self.query_slots_required
                    or continuation.continuation_read_counts is None
                )
            ):
                raise TypeError(
                    "Child continuation requires query slots and no own bound slots."
                )
            expected_value, expected_marginal = continuation.output
        else:
            expected_value, expected_marginal = cast(
                "tuple[ScalarFloat, ScalarFloat]", continuation
            )
        action = invert_euler(
            expected_marginal_continuation=expected_marginal,
            discount_factor=self.discount_factor,
            inverse_marginal_utility=self.inverse_marginal_utility,
        )
        endog_point = savings_value + action
        value = self.utility_of_action(action) + self.discount_factor * expected_value
        output: _NodeOutput = (action, endog_point, value, expected_value)
        if type(continuation) is BoundedResult:
            return BoundedResult[_NodeOutput](
                output=output,
                bound_status=(),
                query_status=continuation.query_status,
                continuation_read_counts=continuation.continuation_read_counts,
            )
        return output


def _candidate_supgradient(
    *,
    policy: Float1D,
    dead: BoolND,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
) -> Float1D:
    """Compute the candidate value-row supgradient $\\mu = \\partial v / \\partial R$.

    By the envelope theorem the value row's slope in resources is the marginal
    utility of consumption $u'(c)$ at the candidate's optimal action — the
    same exact slope the published carry row carries. The upper-envelope
    backend reads it as the tangent gradient at each candidate; FUES ignores
    it. Dead candidates get $0$ (they are dropped before the dominance test).

    Args:
        policy: Candidate policy (consumption) values.
        dead: Indicator of dead candidates (their continuation is `-inf`).
        utility_of_action: Utility with everything but the continuous action
            bound.

    Returns:
        Supgradient per candidate.

    """
    safe_policy = jnp.where(dead, 1.0, policy)
    marginal = jax.vmap(jax.grad(utility_of_action))(safe_policy)
    return jnp.where(dead, 0.0, marginal)


def _compute_constrained_candidates(
    *,
    first_endogenous_point: ScalarFloat,
    publish_resources: FloatND,
    borrowing_limit: ScalarFloat,
    n_constrained: int,
    constrained_ratio: float,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[FloatND, FloatND]:
    """Compute the closed-form credit-constrained candidate points.

    Below the first endogenous point the borrowing limit binds: the action is
    `R - borrowing_limit` and the continuation is the limit-node expectation.
    The candidate actions are geometrically spaced toward the limit; the span
    is capped at the publish range so a degenerate (huge) first endogenous
    point cannot stretch the sample beyond where queries land.

    The span is the width of the interval the segment covers, so it is only a
    segment at all when that width is a positive, finite number. It is not when

    - the first endogenous point sits at or below the borrowing limit,
    - the exogenous grid never rises above the limit, or
    - either bound is not a finite number.

    In each of those the constrained interval is empty and the segment
    contributes no candidates, so its actions and values are NaN — the
    candidate chain's absent form. Substituting a positive width instead would
    sample a segment that does not exist: at a width of one smallest normal the
    actions come out around the dtype's subnormal range, where a curved felicity
    is astronomically negative or infinite. Such candidates are never an
    optimum, and their magnitude defeats the exact upper envelope's certified
    sign, poisoning the row that a dead segment would have left intact.

    Args:
        first_endogenous_point: The endogenous resources point of the lowest
            savings node.
        publish_resources: Resources at the regime's exogenous state grid.
        borrowing_limit: Lower bound of the savings grid.
        n_constrained: Number of constrained candidate points.
        constrained_ratio: Static geometric spacing ratio of the candidates.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of constrained candidate actions and values.

    """
    dtype = publish_resources.dtype
    span = (
        jnp.minimum(first_endogenous_point, jnp.max(publish_resources))
        - borrowing_limit
    )
    empty = ~((span > 0.0) & jnp.isfinite(span))
    # A unit width only keeps the arithmetic below in range; the candidates it
    # produces are discarded wholesale by the `empty` mask, never published.
    safe_span = jnp.where(empty, jnp.ones_like(span), span)
    constrained_actions = (
        safe_span
        * CONSTRAINED_OFFSET_FRACTION
        * constrained_ratio ** jnp.arange(n_constrained, dtype=dtype)
    )
    constrained_values = (
        jax.vmap(utility_of_action)(constrained_actions)
        + discounted_expected_value_at_limit
    )
    return (
        jnp.where(empty, jnp.nan, constrained_actions),
        jnp.where(empty, jnp.nan, constrained_values),
    )


def _publish_V_and_carry_rows(
    *,
    refined_grid: Float1D,
    refined_policy: Float1D,
    refined_value: Float1D,
    n_kept: ScalarInt,
    n_pad: int,
    publish_resources: FloatND,
    lowest_euler_point: ScalarFloat,
    borrowing_limit: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
    _negm_current_publication: bool = False,
    _binding_query: BoolND | None = None,
) -> tuple[FloatND, Float1D, Float1D]:
    """Interpolate V onto the exogenous grid and finish the carry value rows.

    The published value is the closed-form constrained value — the exact value
    of saving exactly the borrowing limit — substituted outright

    - below the lowest refined envelope point, where the interpolant would
      continue the first bracket's secant rather than clamp, and
    - below the lowest live Euler candidate, where the constrained candidates
      are the only candidates, so the envelope there is the constrained branch
      itself. A cubic through that fixed handful of candidates would carry an
      interpolation error that no refinement of the savings or state grid
      removes.

    Elsewhere the refined envelope is read with the cubic Hermite interpolant
    (the marginal-utility row is the value row's exact slope by the envelope
    theorem), floored at the constrained value, which remains a feasible-policy
    lower bound everywhere.

    Envelope overflow is not silent: the outputs are NaN-poisoned so the
    solve loop's NaN diagnostics surface the offending (regime, period).

    Args:
        refined_grid: Refined endogenous grid row from the envelope backend.
        refined_policy: Refined policy row.
        refined_value: Refined value row.
        n_kept: Number of envelope points the backend kept.
        n_pad: Static length of the refined rows.
        publish_resources: Resources at the regime's exogenous state grid.
        lowest_euler_point: Resources of the lowest live Euler candidate; NaN
            disables the substitution below it.
        borrowing_limit: Lower bound of the savings grid.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of the value row on the exogenous state grid, the carry value
        row, and the carry marginal-utility row.

    """
    dtype = publish_resources.dtype
    overflowed = n_kept > n_pad

    marginal_utility = jax.vmap(jax.grad(utility_of_action))(
        jnp.where(jnp.isnan(refined_policy), 1.0, refined_policy)
    )
    marginal_utility = jnp.where(jnp.isnan(refined_policy), jnp.nan, marginal_utility)
    marginal_utility = jnp.where(jnp.isneginf(refined_value), 0.0, marginal_utility)

    value_interpolated = interp_on_padded_grid(
        x_query=publish_resources,
        xp=refined_grid,
        fp=refined_value,
        fp_slopes=marginal_utility,
    )
    closed_form_actions = publish_resources - borrowing_limit
    value_constrained = jnp.where(
        closed_form_actions > 0.0,
        jax.vmap(utility_of_action)(
            jnp.maximum(closed_form_actions, jnp.finfo(dtype).tiny)
        )
        + discounted_expected_value_at_limit,
        -jnp.inf,
    )
    # Below the lowest refined point, and below the lowest live Euler
    # candidate, the closed-form constrained value (the exact value of saving
    # exactly the borrowing limit) is published outright, replacing the
    # interpolant rather than agreeing with it. Everywhere else the constrained
    # value is a feasible-policy floor under the Hermite read of the refined
    # envelope. The cut-off is the lowest live Euler candidate, never the first
    # one: a degenerate inversion can push the first Euler point right (a
    # zero-ish marginal continuation makes it ~1/eps) past Euler candidates that
    # fold below it, and those must stay on the envelope read.
    refined_grid_start = refined_grid[0]
    V_row = jnp.where(
        (closed_form_actions > 0.0)
        & (
            (publish_resources <= refined_grid_start)
            | (publish_resources < lowest_euler_point)
        ),
        value_constrained,
        jnp.maximum(value_interpolated, value_constrained),
    )
    V_row = jnp.where(overflowed, jnp.nan, V_row).astype(dtype)

    if _negm_current_publication:
        if _binding_query is None:
            raise TypeError("Nested current publication requires native query flags.")
        live = jnp.arange(refined_grid.size) < n_kept
        refused = overflowed | jnp.any(
            live
            & ~(
                jnp.isfinite(refined_grid)
                & jnp.isfinite(refined_policy)
                & jnp.isfinite(refined_value)
            )
        )
        V_row = jnp.where(_binding_query, value_constrained, V_row)
        V_row = jnp.where(publish_resources <= borrowing_limit, -jnp.inf, V_row)
        V_row = jnp.where(~jnp.isfinite(publish_resources) | refused, jnp.nan, V_row)

    # The carry value row is the refined upper envelope itself — the correct
    # object for a parent to interpolate. The constrained floor applies only to
    # the published `V_row` on the exogenous grid; flooring the carry would lift
    # finite envelope nodes above the true envelope wherever the closed-form
    # constrained value exceeds them, overstating the parent's continuation.
    value_row = jnp.where(overflowed, jnp.nan, refined_value).astype(dtype)
    # The carry marginal-utility row is the parent's Hermite slope. On overflow
    # the backend still returns a finite truncated prefix, so — like the value
    # rows — it must be NaN-poisoned too, or an overflowed period would feed the
    # parent a wrong-but-finite slope past the NaN diagnostics.
    marginal_row = jnp.where(overflowed, jnp.nan, marginal_utility).astype(dtype)
    return V_row, value_row, marginal_row


def _flatten_owned_resources_row(
    record: OwnedResourcesRow,
) -> tuple[tuple[_OwnedResourcesChild, ...], None]:
    return tuple(record), None


def _flatten_owned_resources_row_with_keys(
    record: OwnedResourcesRow,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _OwnedResourcesChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(OwnedResourcesRow._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_owned_resources_row(
    metadata: None,
    children: Iterable[_OwnedResourcesChild],
) -> OwnedResourcesRow:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(OwnedResourcesRow._fields):
        raise TypeError(
            "Native OwnedResourcesRow reconstruction requires its exact fields."
        )
    return cast("OwnedResourcesRow", tuple.__new__(OwnedResourcesRow, leaves))


jax.tree_util.register_pytree_with_keys(
    OwnedResourcesRow,
    _flatten_owned_resources_row_with_keys,
    _unflatten_owned_resources_row,
    _flatten_owned_resources_row,
)
