"""Asset-row DC-EGM: the per-Euler-node solve for Euler-state-dependent stages.

When any savings-stage function reads the current Euler state — the state's own
law, the regime-transition probabilities, a transition weight, or a passive
state's law — the kernel solves once per exogenous asset node instead of once
per combo. Conditional on a node, every such read is a per-combo constant, so
`step_core`'s single-post-state pipeline is exact within the node's row, and the
row publishes only its own node (brute-force-equivalent by construction). This
module maps that core pipeline over the Euler grid, differentiates the
continuation in the Euler slot for the per-node marginal $dV/dR$, and publishes
each node's scalar value and optimal action at its single resources query. The
per-combo carry holds the per-node published points, not the envelope workspace.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import (
    bind_continuation,
)
from _lcm.egm.interp import _interp_between_nodes
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.query_data import EGMLiquidCornerData, EGMQueryData, QueryNodeKind
from _lcm.egm.step_core import (
    ResourcesOfState,
    _candidate_supgradient,
    _compute_constrained_candidates,
    _compute_nodes_after_first,
    _compute_nodes_over_savings,
    _EgmKernelPieces,
    _get_compute_node,
    _NodeOutput,
    _NodeRows,
    bind_cell_floor,
    current_node_feasibility,
    tile_block_size,
)
from _lcm.egm.upper_envelope import OwnedQueryBracket
from _lcm.egm.upper_envelope.fues import (
    QueryBracket,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    QueryInputStatus,
    merge_continuation_read_counts,
    merge_query_input_status,
    reduce_continuation_read_counts,
)
from _lcm.typing import (
    EconFunctionKwargs,
    RegimeName,
    StateName,
)
from _lcm.utils.dispatchers import map_over_leading_axis
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)

type _OwnedAssetNode = tuple[
    ScalarFloat,
    ScalarFloat,
    ScalarFloat,
    ScalarFloat,
    ScalarFloat,
    ScalarInt,
    ScalarInt,
    ScalarBool,
    ScalarFloat,
    ScalarFloat,
]
type _AssetComboRows = tuple[Float1D, Float1D, Float1D, Float1D, Float1D, ScalarBool]
type _AssetNode = tuple[ScalarFloat, ScalarFloat, ScalarFloat]
type _AssetNodeRows = tuple[Float1D, Float1D, Float1D]
_ASSET_NODE_CHANNEL_COUNT = 3
_OWNED_ASSET_NODE_CHANNEL_COUNT = 10
type _OwnedAssetRows = tuple[
    Float1D,
    Float1D,
    Float1D,
    Float1D,
    Float1D,
    IntND,
    IntND,
    BoolND,
    Float1D,
    Float1D,
]
type _ConditionalAssetRows = tuple[
    Float1D,
    Float1D,
    Float1D,
    Float1D,
    Float1D,
    ScalarBool,
    EGMQueryData | None,
    BoolND,
]


def _get_solve_one_combo_asset_rows(
    *,
    pieces: _EgmKernelPieces,
    pool: EconFunctionKwargs,
    state_grid: Float1D,
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    euler_point_width: int | None,
    savings_point_width: int | None,
    stochastic_node_width: int | None,
    envelope_cell_width: int,
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> Callable[
    [tuple[ScalarInt | ScalarFloat, ...]],
    _AssetComboRows | BoundedResult[_ConditionalAssetRows],
]:
    """Build the per-combo EGM computation solving per exogenous asset node.

    Used when any savings-stage function (the Euler state's law, the
    regime-transition probabilities, a stochastic transition weight, a
    passive state's law) reads the current Euler state: conditional on one
    node of the Euler grid every such read is a per-combo constant, so the
    single-post-state pipeline (Euler inversion over the savings nodes,
    constrained candidates, upper-envelope refinement) is exact within the
    node's row, and each row publishes only its own node — exactly where
    the brute-force oracle evaluates the same decision-time functions. The
    per-combo carry row holds the per-node published points: abscissa the
    node resources (strictly increasing by the resources monotonicity check),
    value the published V, and marginal the corrected
    $dV/dR = u'(c^*) + \\beta\\, (\\partial W/\\partial a)|_{A^*} / R'(a)$,
    NaN-padded to the carry length. The Euler-state gradient
    $\\partial W/\\partial a$ rebuilds the full continuation closure from
    the traced node value, so it carries every direct channel: the law's
    residual, $\\sum \\partial P/\\partial a \\cdot EV$ from the
    probabilities, the weights' and passive laws' derivatives.
    """
    return _SolveOneComboAssetRows(
        pieces=pieces,
        pool=pool,
        state_grid=state_grid,
        next_regime_to_continuation=next_regime_to_continuation,
        euler_point_width=euler_point_width,
        savings_point_width=savings_point_width,
        stochastic_node_width=stochastic_node_width,
        envelope_cell_width=envelope_cell_width,
        resolved_process_grids=resolved_process_grids,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _SolveOneComboAssetRows:
    """The per-asset-node EGM step for one kernel invocation, per combo.

    Takes the combo's values (discrete codes and passive node values)
    positionally so `jax.vmap` can batch over flattened combo arrays.
    """

    pieces: _EgmKernelPieces
    """Build-time statics shared by every per-combo computation."""

    pool: EconFunctionKwargs
    """The kernel's flat params, `period`, and `age`."""

    state_grid: Float1D
    """The regime's exogenous Euler-state grid."""

    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry]
    """The next period's EGM carries."""

    euler_point_width: int | None
    """Tile width of the per-node solve; `None` runs it in one tile."""

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
    ) -> _AssetComboRows | BoundedResult[_ConditionalAssetRows]:
        """Run the per-asset-node EGM step for one (discrete x passive) combo.

        Returns:
            Tuple of the combo's value row on the exogenous state grid and
            its per-node endogenous grid, the published consumption policy on
            that grid, the value and marginal-utility carry rows, and the
            read-support verdict (constant `False`: per-node rows never
            qualify for the off-grid read).

        """
        pieces = self.pieces
        dtype = self.state_grid.dtype
        n_state = int(self.state_grid.shape[0])
        combo_pool = {
            **self.pool,
            **dict(zip(pieces.combo_names, combo_values, strict=True)),
        }
        pieces = bind_cell_floor(pieces=pieces, pool=combo_pool)
        # Validation pins the default Bellman aggregator, whose single
        # non-(utility, CE) parameter is the discount factor.
        (discount_factor_arg,) = tuple(pieces.build_W_kwargs(combo_pool).values())
        discount_factor = cast("ScalarFloat", discount_factor_arg)
        own_resources_of_state = ResourcesOfState(
            resources_func=pieces.own_resources_func,
            euler_state_name=pieces.euler_state_name,
            bound=combo_pool,
        )
        continuation_of_euler_state = functools.partial(
            _continuation_of_euler_state,
            pieces=pieces,
            combo_pool=combo_pool,
            next_regime_to_continuation=self.next_regime_to_continuation,
            dtype=dtype,
            stochastic_node_width=self.stochastic_node_width,
            resolved_process_grids=self.resolved_process_grids,
        )
        solve_one_node = _SolveOneNode(
            envelope_cell_width=self.envelope_cell_width,
            pieces=pieces,
            combo_pool=combo_pool,
            discount_factor=discount_factor,
            next_regime_to_continuation=self.next_regime_to_continuation,
            dtype=dtype,
            stochastic_node_width=self.stochastic_node_width,
            resolved_process_grids=self.resolved_process_grids,
            savings_point_width=self.savings_point_width,
            own_resources_of_state=own_resources_of_state,
            continuation_of_euler_state=continuation_of_euler_state,
        )

        # Run the per-asset-node solve in `lax.map` tiles the plan sizes, which
        # bounds peak working-set memory; a width covering the whole grid keeps
        # the fused vmap. The tiles are concatenated, so the two schedules name
        # the same result.
        block = tile_block_size(width=self.euler_point_width, extent=n_state)
        if (
            pieces.query_spec is not None
            or pieces.check_query_construction
            or pieces.continuation_plan.query_targets
            or pieces.collect_continuation_read_counts
        ):
            nodes = map_over_leading_axis(
                func=solve_one_node,
                xs=self.state_grid,
                batch_size=block,
                bound_summary=True,
            )
            if not isinstance(nodes, BoundedResult):
                raise TypeError("Conditional asset rows require bounded node results.")
            if pieces.query_spec is None:
                if len(nodes.output) != _ASSET_NODE_CHANNEL_COUNT:
                    raise TypeError(
                        "Literal asset publication requires three channels."
                    )
                return _status_asset_rows(
                    nodes=cast("BoundedResult[_AssetNodeRows]", nodes),
                    pieces=pieces,
                    resources=jax.vmap(own_resources_of_state)(self.state_grid),
                    combo_pool=combo_pool,
                )
            if len(nodes.output) != _OWNED_ASSET_NODE_CHANNEL_COUNT:
                raise TypeError("Owned asset publication requires ten channels.")
            return _conditional_asset_rows(
                nodes=cast("BoundedResult[_OwnedAssetRows]", nodes),
                pieces=pieces,
                resources=jax.vmap(own_resources_of_state)(self.state_grid),
                combo_pool=combo_pool,
                state_grid=self.state_grid,
            )
        if block:
            mapped_nodes = jax.lax.map(
                solve_one_node, self.state_grid, batch_size=block
            )
        else:
            mapped_nodes = jax.vmap(solve_one_node)(self.state_grid)
        if isinstance(mapped_nodes, BoundedResult):
            raise TypeError("Child query slots require conditional asset publication.")
        V_vec, policy_vec, mu_vec = mapped_nodes
        publish_resources = jax.vmap(own_resources_of_state)(self.state_grid)

        pad = jnp.full((pieces.n_carry_rows - n_state,), jnp.nan, dtype=dtype)
        grid_row = jnp.concatenate([publish_resources.astype(dtype), pad])
        policy_row = jnp.concatenate([policy_vec.astype(dtype), pad])
        value_row = jnp.concatenate([V_vec.astype(dtype), pad])
        marginal_row = jnp.concatenate([mu_vec.astype(dtype), pad])

        if pieces.feasibility_func is not None:
            # Infeasible discrete combos: -inf value rows so they win no
            # maximum and carry zero choice probability; exactly-zero
            # marginal utility so probability-weighted sums stay finite.
            feasible = pieces.feasibility_func(**combo_pool)
            V_vec = jnp.where(feasible, V_vec, -jnp.inf)
            value_row = jnp.where(feasible, value_row, -jnp.inf)
            marginal_row = jnp.where(feasible, marginal_row, 0.0)

        # Asset-row regimes never qualify for the off-grid policy read (each
        # row publishes one point per exogenous node, not a crossing-complete
        # resources-space row), so the read-support verdict is the fail-closed
        # constant.
        return (
            V_vec.astype(dtype),
            grid_row,
            policy_row,
            value_row,
            marginal_row,
            jnp.asarray(False),  # noqa: FBT003
        )


@dataclass(frozen=True, kw_only=True)
class _ConstructedAssetNode:
    """Keep one original candidate construction local to its query."""

    pieces: _EgmKernelPieces
    """Actual bound floor and candidate construction statics."""
    utility_of_action: BoundUtilityOfAction
    """Current utility with this physical frame and state bound."""
    expected_values: FloatND
    """Original savings-stage expectations, including the borrowing floor."""
    candidate_dead: BoolND
    """Original economic absence classification before envelope refinement."""
    bracket: QueryBracket | OwnedQueryBracket
    """Original configured envelope's selected query bracket."""
    resources_at_node: ScalarFloat
    """Actual current resources at the selected physical state."""
    resources_gradient: ScalarFloat
    """Current resources derivative holding the physical frame fixed."""
    floor_direct: ScalarFloat
    """Existing conditional source's direct floor continuation derivative."""
    child_status: tuple[QueryInputStatus, ...] | None
    """Mandatory child summaries from the original construction."""
    read_counts: ContinuationReadCounts | None
    """Original independently eligible continuation read events."""
    discount_factor: ScalarFloat
    """Original discount operand used by the borrowing-corner selector."""
    first_endogenous_point: ScalarFloat
    """Actual first Euler point governing the original constrained span."""


@dataclass(frozen=True, kw_only=True, eq=False)
class _SolveOneNode:
    """The single-post-state pipeline conditional on one exogenous asset node."""

    pieces: _EgmKernelPieces
    """Build-time statics shared by every per-combo computation."""

    combo_pool: EconFunctionKwargs
    """The combo's pool: flat params, `period`, `age`, and the combo's values."""

    discount_factor: ScalarFloat
    """The combo's discount factor."""

    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry]
    """The next period's EGM carries."""

    dtype: np.dtype
    """The canonical float dtype of the state grid."""

    stochastic_node_width: int | None
    """Block width of the streamed node expectation; `None` folds one block."""

    envelope_cell_width: int
    """Number of exact-envelope node cells resolved together."""

    resolved_process_grids: Mapping[StateName, FloatND]
    """Solve-time grids of runtime-resolved process states."""

    savings_point_width: int | None
    """Tile width of the per-savings-node loop; `None` runs it in one tile."""

    own_resources_of_state: Callable[[ScalarFloat], ScalarFloat]
    """The regime's resources as a function of its Euler state, combo bound."""

    continuation_of_euler_state: Callable[
        [ScalarFloat, ScalarFloat], ScalarFloat | BoundedResult[ScalarFloat]
    ]
    """Expected continuation of `(state_value, savings_value)`, combo bound."""

    economic_continuation: (
        Callable[[ScalarFloat], BoundedResult[tuple[ScalarFloat, ScalarFloat]]] | None
    ) = None
    """Actual declared CE reader shared by candidate inversion and source envelope."""

    def __call__(
        self,
        node_value: ScalarFloat,
    ) -> _AssetNode | BoundedResult[_AssetNode] | BoundedResult[_OwnedAssetNode]:
        constructed = self.construct(node_value=node_value)
        pieces = constructed.pieces
        discount_factor = self.discount_factor
        utility_of_action = constructed.utility_of_action
        expected_values = constructed.expected_values
        candidate_dead = constructed.candidate_dead
        bracket = constructed.bracket
        resources_at_node = constructed.resources_at_node
        resources_gradient = constructed.resources_gradient
        floor_direct = constructed.floor_direct
        child_status = constructed.child_status
        read_counts = constructed.read_counts

        if pieces.query_spec is not None:
            if type(bracket) is not OwnedQueryBracket:
                raise TypeError("Conditional asset rows require source-owned brackets.")
            return _conditional_asset_node(
                node=self,
                pieces=pieces,
                bracket=bracket,
                node_value=node_value,
                resources=resources_at_node,
                resources_gradient=resources_gradient,
                utility_of_action=utility_of_action,
                expected_at_floor=expected_values[0],
                floor_direct=floor_direct,
                child_status=child_status,
                read_counts=read_counts,
            )

        construction_invalid = jnp.asarray(a=False)
        if pieces.check_query_construction:
            if type(bracket) is not OwnedQueryBracket:
                raise TypeError("Asset construction checks require owned brackets.")
            construction_invalid = bracket.construction_invalid
            bracket = bracket.bracket
        elif isinstance(bracket, OwnedQueryBracket):
            raise TypeError("Owned asset brackets require installed query authority.")
        V_node, policy_node = publish_node_from_bracket(
            bracket=bracket,
            n_pad=pieces.n_pad,
            resources_at_node=resources_at_node,
            borrowing_limit=pieces.borrowing_limit,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

        # The carry marginal at this node:
        # dV/dR = u'(c*) + discount_factor * (dW/da at fixed A*) / R'(a),
        # with A* = R(a) - c*(a). The envelope term u'(c*) covers the
        # savings channel; the second term is the law's direct
        # Euler-state channel through the continuation, mapped into
        # resources space by the resources slope.
        marginal_utility_node = jax.grad(utility_of_action)(
            jnp.where(jnp.isnan(policy_node), 1.0, policy_node)
        )
        savings_at_optimum = resources_at_node - policy_node
        if (
            pieces.check_query_construction
            or pieces.continuation_plan.query_targets
            or pieces.collect_continuation_read_counts
        ):
            continuation_gradient, direct_status, direct_counts = (
                _asset_direct_gradient(
                    node=self, node_value=node_value, saving=savings_at_optimum
                )
            )
            child_status = merge_query_input_status(
                left=child_status, right=direct_status
            )
            read_counts = merge_continuation_read_counts(
                left=read_counts, right=direct_counts
            )
        else:
            continuation_gradient = jax.grad(self.continuation_of_euler_state)(
                node_value, savings_at_optimum
            )
        mu_node = (
            marginal_utility_node
            + discount_factor * continuation_gradient / resources_gradient
        )
        V_node, mu_node = _finalize_asset_row_node(
            V_node=V_node,
            mu_node=mu_node,
            policy_node=policy_node,
            candidate_dead=candidate_dead,
            resources_gradient=resources_gradient,
        )
        if pieces.check_query_construction:
            corner_known = jnp.isfinite(expected_values[0]) | jnp.isneginf(
                expected_values[0]
            )
            invalid = (
                construction_invalid
                | ~corner_known
                | ~(jnp.isfinite(V_node) | jnp.isneginf(V_node))
                | ~jnp.isfinite(mu_node)
                | ~jnp.isfinite(resources_gradient)
                | (resources_gradient <= 0.0)
            )
            return BoundedResult(
                output=(V_node, policy_node, mu_node),
                bound_status=(),
                continuation_read_counts=read_counts,
                query_status=(
                    *(child_status or ()),
                    QueryInputStatus(
                        jnp.asarray(a=False), jnp.asarray(a=False), invalid
                    ),
                ),
            )
        return _with_asset_child_status(
            output=(V_node, policy_node, mu_node),
            installed=bool(pieces.continuation_plan.query_targets),
            status=child_status,
            read_counts=read_counts,
        )

    def construct(
        self,
        *,
        node_value: ScalarFloat,
    ) -> _ConstructedAssetNode:
        """Construct the original candidates once before selecting output channels."""
        pieces = bind_cell_floor(pieces=self.pieces, pool=self.combo_pool)
        discount_factor = self.discount_factor
        node_pool = {**self.combo_pool, pieces.euler_state_name: node_value}
        utility_of_action = BoundUtilityOfAction(
            utility_func=pieces.utility_func,
            action_name=pieces.action_name,
            bound=node_pool,
        )
        compute_node = _get_compute_node(
            pieces=pieces,
            combo_pool=node_pool,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            next_regime_to_continuation=self.next_regime_to_continuation,
            dtype=self.dtype,
            stochastic_node_width=self.stochastic_node_width,
            resolved_process_grids=self.resolved_process_grids,
            economic_continuation=self.economic_continuation,
        )
        child_status = None
        floor_direct = jnp.asarray(0.0, dtype=self.dtype)
        if pieces.query_spec is not None:
            candidates, floor_direct, child_status, read_counts = (
                _asset_candidates_with_direct(
                    node=self,
                    pieces=pieces,
                    node_value=node_value,
                    utility_of_action=utility_of_action,
                    compute_node=compute_node,
                )
            )
        else:
            candidates, child_status, read_counts = _asset_candidates_without_direct(
                pieces=pieces,
                compute_node=compute_node,
                savings_point_width=self.savings_point_width,
            )
        actions, endog_grid, values, expected_values = candidates

        resources_at_node, resources_gradient = jax.value_and_grad(
            self.own_resources_of_state
        )(node_value)

        constrained_actions, constrained_values = _compute_constrained_candidates(
            first_endogenous_point=endog_grid[0],
            publish_resources=resources_at_node,
            borrowing_limit=pieces.borrowing_limit,
            n_constrained=pieces.n_constrained,
            constrained_ratio=pieces.constrained_ratio,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

        candidate_grid = jnp.concatenate(
            [pieces.borrowing_limit + constrained_actions, endog_grid]
        )
        candidate_policy = jnp.concatenate([constrained_actions, actions])
        candidate_value = jnp.concatenate([constrained_values, values])
        # Exogenous source savings per candidate: the savings node for each
        # Euler candidate (`endog_grid = savings_node + action`, so the
        # implied `endog_grid - policy` equals it in exact arithmetic), the
        # borrowing limit for the constrained candidates (their savings is
        # pinned there). FUES compares these pristine sources exactly.
        candidate_savings = jnp.concatenate(
            [
                jnp.full_like(constrained_actions, pieces.borrowing_limit),
                pieces.savings_nodes,
            ]
        )
        # Same `-inf` masking as the default per-combo computation: dead
        # candidates become the envelope scan's absent form (NaN).
        candidate_dead = jnp.isneginf(candidate_value)
        candidate_marginal = _candidate_supgradient(
            policy=candidate_policy,
            dead=candidate_dead,
            utility_of_action=utility_of_action,
        )
        # The node reads its refined envelope at one query
        # (`resources_at_node`): every finder materializes the full refined
        # envelope row and locates the bracketing pair, so the published
        # `(V, policy)` is a full-envelope-then-interpolate. A sub-`n_pad`
        # streamed finder is future work for all backends.
        bracket = pieces.refine_to_bracket(
            cell_width=self.envelope_cell_width,
            endog_grid=jnp.where(candidate_dead, jnp.nan, candidate_grid),
            policy=jnp.where(candidate_dead, jnp.nan, candidate_policy),
            value=jnp.where(candidate_dead, jnp.nan, candidate_value),
            marginal_utility=candidate_marginal,
            savings=jnp.where(candidate_dead, jnp.nan, candidate_savings),
            x_query=resources_at_node,
        )

        return _ConstructedAssetNode(
            pieces=pieces,
            utility_of_action=utility_of_action,
            expected_values=expected_values,
            candidate_dead=candidate_dead,
            bracket=bracket,
            resources_at_node=resources_at_node,
            resources_gradient=resources_gradient,
            floor_direct=floor_direct,
            child_status=child_status,
            read_counts=read_counts,
            discount_factor=self.discount_factor,
            first_endogenous_point=endog_grid[0],
        )


def _with_asset_child_status(
    *,
    output: _AssetNode,
    installed: bool,
    status: tuple[QueryInputStatus, ...] | None,
    read_counts: ContinuationReadCounts | None,
) -> _AssetNode | BoundedResult[_AssetNode]:
    """Keep child integrity independent of source bracket ownership."""
    if not installed and read_counts is None:
        return output
    if installed and status is None:
        raise TypeError("Installed asset child readers require query summaries.")
    return BoundedResult(
        output=output,
        bound_status=(),
        query_status=status,
        continuation_read_counts=read_counts,
    )


def _asset_candidates_without_direct(
    *,
    pieces: _EgmKernelPieces,
    compute_node: Callable[[ScalarFloat], _NodeOutput | BoundedResult[_NodeOutput]],
    savings_point_width: int | None,
) -> tuple[
    _NodeRows, tuple[QueryInputStatus, ...] | None, ContinuationReadCounts | None
]:
    """Evaluate candidates once and summarize only statically declared child slots."""
    if pieces.check_query_construction:
        return _compute_nodes_after_first(
            first=compute_node(pieces.savings_nodes[0]),
            compute_node=compute_node,
            savings_nodes=pieces.savings_nodes,
            savings_point_width=savings_point_width,
        )
    candidates = _compute_nodes_over_savings(
        compute_node=compute_node,
        savings_nodes=pieces.savings_nodes,
        savings_point_width=savings_point_width,
        bound_summary=bool(pieces.continuation_plan.query_targets)
        or pieces.collect_continuation_read_counts,
    )
    if isinstance(candidates, BoundedResult):
        return (
            candidates.output,
            candidates.query_status,
            candidates.continuation_read_counts,
        )
    return candidates, None, None


def _asset_direct_gradient(
    *, node: _SolveOneNode, node_value: ScalarFloat, saving: ScalarFloat
) -> tuple[
    ScalarFloat, tuple[QueryInputStatus, ...] | None, ContinuationReadCounts | None
]:
    """Differentiate the numeric continuation while retaining its child slots."""

    def continuation(
        state_value: ScalarFloat,
    ) -> tuple[
        ScalarFloat,
        tuple[tuple[QueryInputStatus, ...] | None, ContinuationReadCounts | None],
    ]:
        result = node.continuation_of_euler_state(state_value, saving)
        if isinstance(result, BoundedResult):
            if result.bound_status or (
                result.query_status is None
                and (
                    node.pieces.continuation_plan.query_targets
                    or result.continuation_read_counts is None
                )
            ):
                raise TypeError("Direct child reads require declared query slots.")
            return result.output, (result.query_status, result.continuation_read_counts)
        return result, (None, None)

    (_, (status, counts)), gradient = jax.value_and_grad(continuation, has_aux=True)(
        node_value
    )
    return gradient, status, counts


def _asset_candidates_with_direct(
    *,
    node: _SolveOneNode,
    pieces: _EgmKernelPieces,
    node_value: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    compute_node: Callable[[ScalarFloat], _NodeOutput | BoundedResult[_NodeOutput]],
) -> tuple[
    _NodeRows,
    ScalarFloat,
    tuple[QueryInputStatus, ...] | None,
    ContinuationReadCounts | None,
]:
    """Differentiate first-node W and retain its numeric candidate as auxiliary."""

    def first(
        state_value: ScalarFloat,
    ) -> tuple[ScalarFloat, _NodeOutput | BoundedResult[_NodeOutput]]:
        first_compute = _get_compute_node(
            pieces=pieces,
            combo_pool={**node.combo_pool, pieces.euler_state_name: state_value},
            discount_factor=node.discount_factor,
            utility_of_action=utility_of_action,
            next_regime_to_continuation=node.next_regime_to_continuation,
            dtype=node.dtype,
            stochastic_node_width=node.stochastic_node_width,
            resolved_process_grids=node.resolved_process_grids,
        )
        result = first_compute(pieces.savings_nodes[0])
        output = result.output if isinstance(result, BoundedResult) else result
        return output[3], result

    (_, first_result), direct = jax.value_and_grad(first, has_aux=True)(node_value)
    candidates, child_status, read_counts = _compute_nodes_after_first(
        first=first_result,
        compute_node=compute_node,
        savings_nodes=pieces.savings_nodes,
        savings_point_width=node.savings_point_width,
    )
    return candidates, direct, child_status, read_counts


def _conditional_asset_node(
    *,
    node: _SolveOneNode,
    pieces: _EgmKernelPieces,
    bracket: OwnedQueryBracket,
    node_value: ScalarFloat,
    resources: ScalarFloat,
    resources_gradient: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    expected_at_floor: ScalarFloat,
    floor_direct: ScalarFloat,
    child_status: tuple[QueryInputStatus, ...] | None,
    read_counts: ContinuationReadCounts | None,
) -> BoundedResult[_OwnedAssetNode]:
    """Keep owned ordinary and independently evaluated corner candidates separate."""
    pair = bracket.bracket
    lower_slope = _node_value_slope(
        policy=pair.lower_policy,
        value=pair.lower_value,
        utility_of_action=utility_of_action,
    )
    upper_slope = _node_value_slope(
        policy=pair.upper_policy,
        value=pair.upper_value,
        utility_of_action=utility_of_action,
    )
    ordinary_value = _interp_between_nodes(
        x_query=resources,
        xp_lower=pair.lower_grid,
        xp_upper=pair.upper_grid,
        fp_lower=pair.lower_value,
        fp_upper=pair.upper_value,
        slope_lower=lower_slope,
        slope_upper=upper_slope,
    )
    saving = bracket.savings
    policy = resources - saving
    supported = (
        bracket.read_supported & (saving >= pieces.borrowing_limit) & (policy > 0.0)
    )
    safe_policy = jnp.where(supported, policy, 1.0)
    safe_saving = jnp.where(supported, saving, pieces.borrowing_limit)

    chosen_direct, direct_status, direct_counts = _asset_direct_gradient(
        node=node, node_value=node_value, saving=safe_saving
    )
    child_status = merge_query_input_status(left=child_status, right=direct_status)
    read_counts = merge_continuation_read_counts(
        left=read_counts,
        right=reduce_continuation_read_counts(counts=direct_counts, active=supported),
    )
    mu = (
        jax.grad(utility_of_action)(safe_policy)
        + node.discount_factor * chosen_direct / resources_gradient
    )
    economic = jnp.isneginf(ordinary_value)
    mu = jnp.where(economic, 0.0, mu)
    valid_value = jnp.isfinite(ordinary_value) | economic
    invalid_numeric = bracket.read_supported & (
        ~jnp.isfinite(saving) | ~valid_value | ~jnp.isfinite(mu)
    )
    supported &= valid_value & ~invalid_numeric
    corner_policy = resources - pieces.borrowing_limit
    corner_value = (
        utility_of_action(jnp.maximum(corner_policy, jnp.finfo(resources.dtype).tiny))
        + node.discount_factor * expected_at_floor
    )
    corner_value = jnp.where(corner_policy > 0.0, corner_value, -jnp.inf)
    published_value = jnp.where(
        supported, jnp.maximum(ordinary_value, corner_value), corner_value
    )
    native_floor_direct = floor_direct / resources_gradient
    corner_known = jnp.isfinite(expected_at_floor) | jnp.isneginf(expected_at_floor)
    construction_invalid = (
        bracket.construction_invalid
        | invalid_numeric
        | ~corner_known
        | ~jnp.isfinite(native_floor_direct)
        | ~jnp.isfinite(resources_gradient)
        | (resources_gradient <= 0.0)
    )
    return BoundedResult(
        output=(
            published_value,
            policy,
            ordinary_value,
            mu,
            saving,
            bracket.owner_id,
            bracket.side_id,
            supported,
            expected_at_floor,
            native_floor_direct,
        ),
        bound_status=(),
        continuation_read_counts=read_counts,
        query_status=(
            *(child_status or ()),
            QueryInputStatus(
                jnp.asarray(a=False),
                jnp.asarray(a=False),
                construction_invalid,
            ),
        ),
    )


def _status_asset_rows(
    *,
    nodes: BoundedResult[_AssetNodeRows],
    pieces: _EgmKernelPieces,
    resources: Float1D,
    combo_pool: EconFunctionKwargs,
) -> BoundedResult[_ConditionalAssetRows]:
    """Carry mandatory child integrity beside literal asset-row publication."""
    V, policy, mu = nodes.output
    n_padding = pieces.n_carry_rows - resources.shape[0]

    def floats(array: Float1D) -> Float1D:
        return jnp.pad(array, (0, n_padding), constant_values=jnp.nan)

    value = floats(V)
    marginal = floats(mu)
    feasible = resources > pieces.borrowing_limit
    if pieces.feasibility_func is not None:
        predicate = pieces.feasibility_func(**combo_pool)
        feasible &= predicate
        V = jnp.where(predicate, V, -jnp.inf)
        value = jnp.where(predicate, value, -jnp.inf)
        marginal = jnp.where(predicate, marginal, 0.0)
    return BoundedResult(
        output=(
            V,
            floats(resources),
            floats(policy),
            value,
            marginal,
            jnp.asarray(a=False),
            None,
            feasible,
        ),
        bound_status=() if pieces.floor_status is None else (pieces.floor_status,),
        query_status=nodes.query_status,
        continuation_read_counts=reduce_continuation_read_counts(
            counts=nodes.continuation_read_counts,
            active=jnp.asarray(a=True)
            if pieces.feasibility_func is None
            else predicate,
        ),
    )


def _conditional_asset_rows(
    *,
    nodes: BoundedResult[_OwnedAssetRows],
    pieces: _EgmKernelPieces,
    resources: Float1D,
    combo_pool: EconFunctionKwargs,
    state_grid: Float1D,
) -> BoundedResult[_ConditionalAssetRows]:
    """Publish real J coordinates once and retain the independent liquid corner bank."""
    spec = pieces.query_spec
    if spec is None or spec.corner_coordinate_kind != "liquid_row":
        raise TypeError(
            "Conditional asset rows require installed liquid-row authority."
        )
    V, policy, value, mu, saving, owner, side, supported, Wb, direct = nodes.output
    n_padding = pieces.n_carry_rows - resources.shape[0]

    def floats(array: Float1D) -> Float1D:
        return jnp.pad(array, (0, n_padding), constant_values=jnp.nan)

    def integers(*, array: IntND, pad: int = 0) -> IntND:
        return jnp.pad(array, (0, n_padding), constant_values=pad)

    represented = integers(array=supported.astype(jnp.int32)).astype(jnp.bool_)
    owners = integers(array=owner, pad=-1)
    kinds = jnp.where(
        represented,
        jnp.where(
            jnp.isneginf(floats(value)),
            int(QueryNodeKind.ECONOMIC_MINUS_INF),
            int(QueryNodeKind.FINITE),
        ),
        int(QueryNodeKind.PAD),
    ).astype(jnp.int32)
    corner_known = jnp.isfinite(Wb) | jnp.isneginf(Wb)
    corner_kind = integers(
        array=jnp.where(
            corner_known,
            jnp.where(
                jnp.isneginf(Wb),
                int(QueryNodeKind.ECONOMIC_MINUS_INF),
                int(QueryNodeKind.FINITE),
            ),
            int(QueryNodeKind.PAD),
        ).astype(jnp.int32)
    )
    corner = EGMLiquidCornerData(
        continuation=floats(Wb)[:, None],
        direct_marginal=floats(direct)[:, None],
        node_kind=corner_kind[:, None],
        link_valid=((corner_kind[:-1] != 0) & (corner_kind[1:] != 0))[:, None],
        owner_id=jnp.full(
            (pieces.n_carry_rows, 1), spec.interval_ids[0], dtype=jnp.int32
        ),
        side_id=jnp.full(
            (pieces.n_carry_rows, 1), spec.interval_side_ids[0], dtype=jnp.int32
        ),
        interval_id=jnp.full(
            (pieces.n_carry_rows, 1), spec.interval_ids[0], dtype=jnp.int32
        ),
    )
    data = EGMQueryData(
        savings=floats(saving),
        node_kind=kinds,
        link_valid=represented[:-1] & represented[1:] & (owners[:-1] == owners[1:]),
        owner_id=owners,
        side_id=integers(array=side),
        floor_used=pieces.borrowing_limit,
        liquid_corner_data=corner,
    )
    feasible = resources > pieces.borrowing_limit
    if pieces.feasibility_func is not None:
        feasible &= current_node_feasibility(
            pieces=pieces, pool=combo_pool, state_grid=state_grid
        )
    return BoundedResult(
        output=(
            jnp.where(feasible, V, -jnp.inf),
            floats(resources),
            floats(policy),
            floats(value),
            floats(mu),
            jnp.asarray(a=False),
            data,
            feasible,
        ),
        bound_status=() if pieces.floor_status is None else (pieces.floor_status,),
        query_status=nodes.query_status,
        # The retained conditional and liquid-corner records require these reads.
        continuation_read_counts=nodes.continuation_read_counts,
    )


# keyword-only-exempt: library-callback=jax.grad
def _continuation_of_euler_state(
    state_value: ScalarFloat,
    savings_value: ScalarFloat,
    *,
    pieces: _EgmKernelPieces,
    combo_pool: EconFunctionKwargs,
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    dtype: np.dtype,
    stochastic_node_width: int | None,
    resolved_process_grids: Mapping[StateName, FloatND],
) -> ScalarFloat | BoundedResult[ScalarFloat]:
    """Expected continuation with the Euler slot as the grad argument.

    Rebuilds the per-combo continuation with the Euler slot of the combo pool
    bound to `state_value` (positional, so `jax.grad` differentiates the law's
    direct Euler-state channel) and evaluates it at fixed savings.
    """
    node_pool = {**combo_pool, pieces.euler_state_name: state_value}
    expected_continuation = _get_expected_continuation_value(
        pieces=pieces,
        combo_pool=node_pool,
        next_regime_to_continuation=next_regime_to_continuation,
        dtype=dtype,
        stochastic_node_width=stochastic_node_width,
        resolved_process_grids=resolved_process_grids,
    )
    return expected_continuation(savings_value)


def _finalize_asset_row_node(
    *,
    V_node: ScalarFloat,
    mu_node: ScalarFloat,
    policy_node: ScalarFloat,
    candidate_dead: BoolND,
    resources_gradient: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Apply the no-live-candidate and resources-validity guards to one node.

    A node with no live candidate (its entire continuation is `-inf`) is worth
    `-inf`, like an infeasible combo, with an exactly-zero marginal so
    probability-weighted sums stay finite. Separately, the published marginal
    divides the direct continuation channel by the resources slope `dR/da`, which
    the method requires strictly positive (the carry abscissae are the node
    resources). The build-time strict-monotonicity check skips a resources map that
    reads a parameter, so a non-positive slope would otherwise publish a
    finite value/marginal with the wrong orientation. Fail loud: NaN the value
    *and* the marginal so the solve's NaN diagnostics (which scan the value array)
    name the offending regime/period rather than returning silently-wrong numbers.
    """
    mu_node = jnp.where(jnp.isnan(policy_node), jnp.nan, mu_node)
    no_live_candidate = jnp.all(candidate_dead)
    V_node = jnp.where(no_live_candidate, -jnp.inf, V_node)
    mu_node = jnp.where(jnp.isneginf(V_node) | no_live_candidate, 0.0, mu_node)
    invalid_resources = resources_gradient <= 0.0
    V_node = jnp.where(invalid_resources, jnp.nan, V_node)
    mu_node = jnp.where(invalid_resources, jnp.nan, mu_node)
    return V_node, mu_node


def _get_expected_continuation_value(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: EconFunctionKwargs,
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    dtype: np.dtype,
    stochastic_node_width: int | None,
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> Callable[[ScalarFloat], ScalarFloat | BoundedResult[ScalarFloat]]:
    """Build the expected-continuation map $W(A)$ for one combo pool.

    Mirrors the expected-value aggregation of the per-savings-node Euler
    inversion: per-target smoothed carry reads, regime-transition-probability
    weighted, plus the scalar targets' constant values. The asset-row mode
    differentiates this map in the Euler slot of the combo pool to obtain
    the direct Euler-state channel $\\partial W/\\partial a$. The
    probabilities, transition weights, and child next-state reads are all
    evaluated from the combo pool *inside* this builder, so when the pool's
    Euler slot is a traced value the gradient carries their first-order
    terms (e.g. $\\sum \\partial P/\\partial a \\cdot EV$) — precomputing
    them outside the differentiated closure would silently drop those
    terms (Danskin does not cancel them: the probabilities are not the
    softmax of the values they weight).
    """
    return _ExpectedContinuationValue(
        continuation=bind_continuation(
            plan=pieces.continuation_plan,
            combo_pool=combo_pool,
            next_regime_to_continuation=next_regime_to_continuation,
            dtype=dtype,
            stochastic_node_width=stochastic_node_width,
            resolved_process_grids=resolved_process_grids,
            collect_continuation_read_counts=pieces.collect_continuation_read_counts,
        ),
        query_slots_required=bool(pieces.continuation_plan.query_targets),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _ExpectedContinuationValue:
    """The expected continuation value $W(A)$ of one combo pool."""

    continuation: Callable[
        [ScalarFloat],
        tuple[ScalarFloat, ScalarFloat]
        | BoundedResult[tuple[ScalarFloat, ScalarFloat]],
    ]
    """The bound continuation, returning the value and marginal at a node."""
    query_slots_required: bool = False
    """Canonical child-slot presence, independently of optional counters."""

    def __call__(
        self, savings_value: ScalarFloat
    ) -> ScalarFloat | BoundedResult[ScalarFloat]:
        result = self.continuation(savings_value)
        if isinstance(result, BoundedResult):
            if result.bound_status or (
                result.query_status is None
                and (
                    self.query_slots_required or result.continuation_read_counts is None
                )
            ):
                raise TypeError(
                    "Child values require query slots and no own bound slots."
                )
            return BoundedResult(
                output=result.output[0],
                bound_status=(),
                query_status=result.query_status,
                continuation_read_counts=result.continuation_read_counts,
            )
        return result[0]


def publish_node_from_bracket(
    *,
    bracket: QueryBracket,
    n_pad: int,
    resources_at_node: ScalarFloat,
    borrowing_limit: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Publish one asset node's value and optimal action from its query bracket.

    Reads the two envelope nodes `refine_to_bracket` sliced around
    `resources_at_node` from the full refined row:

    - The value is the cubic-Hermite read of the envelope between the two
      bracket nodes (the value slope at each node is `grad(utility_of_action)`
      at that node's policy, the envelope-theorem marginal), floored at the
      closed-form constrained value, which is a feasible-policy lower bound.
    - Below the lowest envelope node the closed-form constrained value is
      published outright; the winning branch sets the action (the closed-form
      `R - borrowing_limit` where constrained wins, the interpolated policy
      otherwise).
    - Between the lowest envelope node and the first Euler point nothing needs
      substituting: the node's constrained candidates span up to
      `min(first Euler point, resources_at_node)`, so a binding query sits on
      the top constrained candidate and the Hermite read returns its exact
      closed-form value.
    - Envelope overflow (`n_kept > n_pad`) NaN-poisons both outputs, identical
      to the row path, so the solve loop's NaN diagnostics surface the offending
      (regime, period).

    Because the value and policy arithmetic is the shared `_interp_between_nodes`
    primitive — the same one the row path reaches through
    `interp_on_padded_grid`, reading the same refined row — the bracket publish
    cannot diverge from row-then-interpolate.

    Args:
        bracket: The query bracket from `refine_to_bracket`.
        n_pad: Static length of the envelope-refinement workspace (the overflow
            threshold).
        resources_at_node: Resources at this exogenous Euler node (the row's
            single publish query).
        borrowing_limit: Lower bound of the savings grid.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of the node's published value and published optimal action.

    """
    dtype = resources_at_node.dtype
    overflowed = bracket.n_kept > n_pad

    # The value Hermite slope is the envelope-theorem marginal `u'(c*)`, masked
    # exactly as the row path: NaN where the node's policy is NaN (a padded
    # slot), 0.0 where the node's value is `-inf` (an infeasible endpoint), so
    # `_interp_between_nodes` falls back to the linear rule on those brackets.
    slope_lower = _node_value_slope(
        policy=bracket.lower_policy,
        value=bracket.lower_value,
        utility_of_action=utility_of_action,
    )
    slope_upper = _node_value_slope(
        policy=bracket.upper_policy,
        value=bracket.upper_value,
        utility_of_action=utility_of_action,
    )

    value_interpolated = _interp_between_nodes(
        x_query=resources_at_node,
        xp_lower=bracket.lower_grid,
        xp_upper=bracket.upper_grid,
        fp_lower=bracket.lower_value,
        fp_upper=bracket.upper_value,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
    )
    policy_interpolated = _interp_between_nodes(
        x_query=resources_at_node,
        xp_lower=bracket.lower_grid,
        xp_upper=bracket.upper_grid,
        fp_lower=bracket.lower_policy,
        fp_upper=bracket.upper_policy,
    )
    # Degenerate envelopes mirror the row read's contract exactly
    # (`interp_on_prepared_grid`): a single kept node — always the bracket's
    # lower slot, the upper is a NaN pad — is a constant clamp on both sides,
    # and an empty envelope reads NaN.
    single_node = bracket.n_kept == 1
    empty_envelope = bracket.n_kept == 0
    value_interpolated = jnp.where(single_node, bracket.lower_value, value_interpolated)
    value_interpolated = jnp.where(empty_envelope, jnp.nan, value_interpolated)
    policy_interpolated = jnp.where(
        single_node, bracket.lower_policy, policy_interpolated
    )
    policy_interpolated = jnp.where(empty_envelope, jnp.nan, policy_interpolated)

    closed_form_action = resources_at_node - borrowing_limit
    value_constrained = jnp.where(
        closed_form_action > 0.0,
        utility_of_action(jnp.maximum(closed_form_action, jnp.finfo(dtype).tiny))
        + discounted_expected_value_at_limit,
        -jnp.inf,
    )
    below_refined = (closed_form_action > 0.0) & (
        resources_at_node <= bracket.first_grid
    )
    constrained_wins = below_refined | (value_constrained >= value_interpolated)
    V_node = jnp.where(
        below_refined,
        value_constrained,
        jnp.maximum(value_interpolated, value_constrained),
    )
    policy_node = jnp.where(constrained_wins, closed_form_action, policy_interpolated)
    V_node = jnp.where(overflowed, jnp.nan, V_node).astype(dtype)
    policy_node = jnp.where(overflowed, jnp.nan, policy_node).astype(dtype)
    return V_node, policy_node


def _node_value_slope(
    *,
    policy: ScalarFloat,
    value: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
) -> ScalarFloat:
    """Envelope-theorem value slope at one bracket node, masked like the row.

    The slope is `grad(utility_of_action)` at the node's policy — NaN where the
    policy is a padded NaN slot, 0.0 where the node value is `-inf` — matching
    the per-node masking the row path applies before interpolating.
    """
    slope = jax.grad(utility_of_action)(jnp.where(jnp.isnan(policy), 1.0, policy))
    slope = jnp.where(jnp.isnan(policy), jnp.nan, slope)
    return jnp.where(jnp.isneginf(value), 0.0, slope)
