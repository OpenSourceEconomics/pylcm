"""DC producers publish owned conditional records beside node feasibility."""

import functools
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import (
    ContinuationPayload,
    EGMContinuationSpec,
    EGMQuerySpec,
    EGMRowTopologySpec,
    validate_query_template,
)
from _lcm.egm import asset_row as asset_row_module
from _lcm.egm import step_core as step_core_module
from _lcm.egm.asset_row import _SolveOneComboAssetRows
from _lcm.egm.carry import EGMCarry, read_owned_row
from _lcm.egm.continuation import ContinuationPlan
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.egm.query_data import EGMQueryData
from _lcm.egm.step import _EGMStep
from _lcm.egm.step_core import _ComputeNode, _EgmKernelPieces, _SolveOneCombo
from _lcm.egm.upper_envelope import (
    OwnedQueryBracket,
    get_bracket_finder,
    get_upper_envelope,
)
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.dcegm import _dcegm_values_core, _DCEGMPeriodKernel
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    FeasibilityKernelOutput,
    QueryInputStatus,
    validate_query_input_status,
)
from _lcm.typing import EconFunctionArg, EconFunctionKwargs, FlatParams, PytreeValue
from lcm import IrregSpacedGrid
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import EGM_CONTINUATION, SIMULATION_POLICY
from lcm.solvers import DCEGM
from lcm.typing import Float1D, FloatND, RegimeName, ScalarBool, ScalarFloat, StateName
from tests.conftest import DECIMAL_PRECISION
from tests.egm.test_dynamic_dcegm_floor import (
    _build_producer,
    _direct_continuation,
    _lower,
)
from tests.solution.test_dynamic_dcegm_binding import _capture_processor


def _query_spec() -> EGMQuerySpec:
    bound = PostDecisionBoundSpec(
        post_decision="savings",
        liquid_state="wealth",
        action="consumption",
        resources_target="resources",
        kind="dag",
        literal_value=None,
        lower_target="borrowing_lower",
        lower_inputs=("z",),
        dependency_identity=(((), _lower),),
        declaration_names=("borrowing",),
    )
    return EGMQuerySpec(
        bound=bound,
        state_names=("z",),
        action_names=(),
        coordinate_kind="resources",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind="compact_interval",
        row_capacity=32,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(("resources", "resources"), ("utility", "utility")),
        evaluator_inputs=(),
        evaluator_identity=(),
        target_periods=(0, 1),
        row_topology=EGMRowTopologySpec(kind="exact_monotone_run", owner_capacity=24),
    )


def _feature_producer(*, monkeypatch: pytest.MonkeyPatch) -> _SolveOneCombo:
    producer = _build_producer(monkeypatch=monkeypatch)
    solver = DCEGM(
        savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0, 1.5])),
        n_constrained_points=4,
    )
    return replace(
        producer,
        state_grid=jnp.asarray([0.0]),
        pieces=replace(
            producer.pieces,
            query_spec=_query_spec(),
            refine=get_upper_envelope(
                solver=solver, n_refined=32, publish_provenance=True
            ),
            refine_to_bracket=get_bracket_finder(
                solver=solver, n_refined=32, publish_provenance=True
            ),
        ),
    )


@pytest.mark.parametrize("z", [0.0, 1.0])
def test_ordinary_producer_publishes_original_saving_and_unmasked_corner(
    *, monkeypatch: pytest.MonkeyPatch, z: float
) -> None:
    """Original saving and ownership accompany the first expected continuation."""
    producer = _feature_producer(monkeypatch=monkeypatch)
    result = producer((jnp.asarray(z),))
    assert isinstance(result, BoundedResult)
    output = result.output
    data = output[6]
    assert data is not None
    assert data.corner_continuation is not None
    active = data.node_kind != 0
    saving = np.asarray(data.savings)[np.asarray(active)]
    floor = z / 2.0
    np.testing.assert_allclose(
        saving[[0, -1]],
        [floor, floor + 1.5],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(data.floor_used, floor)
    np.testing.assert_allclose(
        data.corner_continuation,
        [np.log(3.0 * floor + 4.0)],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_array_equal(data.corner_kind, [1])
    np.testing.assert_array_equal(data.corner_interval_id, [0])
    np.testing.assert_array_equal(data.corner_side_id, [0])
    np.testing.assert_allclose(
        output[0],
        [np.log((1.0 - floor) * (3.0 * floor + 4.0))],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_array_equal(output[7], [True])
    assert result.query_status is not None
    assert len(result.query_status) == 1
    assert not bool(result.query_status[0].coverage_missing)
    assert len(result.bound_status) == 1
    assert not bool(result.bound_status[0].nonfinite_floor)
    assert not bool(result.bound_status[0].invalid_shifted_grid)


def _excluded(**kwargs: EconFunctionArg) -> ScalarBool:
    del kwargs
    return jnp.asarray(a=False)


def test_current_predicate_masks_nodes_without_erasing_conditional_source(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node feasibility leaves the independent row and lower continuation intact."""
    producer = _feature_producer(monkeypatch=monkeypatch)
    producer = replace(
        producer, pieces=replace(producer.pieces, feasibility_func=_excluded)
    )
    result = producer((jnp.asarray(0.0),))
    assert isinstance(result, BoundedResult)
    output = result.output
    data = output[6]
    assert data is not None
    assert data.corner_continuation is not None
    active = data.node_kind != 0
    np.testing.assert_array_equal(output[0], [-np.inf])
    np.testing.assert_array_equal(output[7], [False])
    np.testing.assert_allclose(
        data.corner_continuation, [np.log(4.0)], rtol=10 ** (-DECIMAL_PRECISION)
    )
    np.testing.assert_allclose(
        np.asarray(data.savings)[np.asarray(active)][[0, -1]], [0.0, 1.5]
    )
    assert np.isfinite(np.asarray(output[3])[np.asarray(active)]).all()
    assert result.query_status is not None
    assert not bool(result.query_status[0].coverage_missing)


def test_unselected_construction_failure_survives_excluded_source_nodes(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An excluded source still owns mandatory emitted-prefix construction rejection."""
    producer = _feature_producer(monkeypatch=monkeypatch)
    refine = producer.pieces.refine

    def corrupt(
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
        marginal_utility: Float1D,
        savings: Float1D,
        cell_width: int = 1,
    ) -> OwnedEnvelope:
        output = refine(
            endog_grid=endog_grid,
            policy=policy,
            value=value,
            marginal_utility=marginal_utility,
            savings=savings,
            cell_width=cell_width,
        )
        assert type(output) is OwnedEnvelope
        return output._replace(
            savings=output.savings.at[output.n_kept - 1].set(jnp.nan)
        )

    producer = replace(
        producer,
        pieces=replace(producer.pieces, refine=corrupt, feasibility_func=_excluded),
    )
    result = producer((jnp.asarray(0.0),))
    assert isinstance(result, BoundedResult)
    assert result.query_status is not None
    assert bool(result.query_status[0].coverage_missing)
    assert not bool(result.query_status[0].nonfinite_floor)
    assert not bool(result.query_status[0].invalid_shifted_grid)
    np.testing.assert_array_equal(result.output[0], [-np.inf])


def _direct_expected_continuation(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: EconFunctionKwargs,
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    dtype: np.dtype,
    stochastic_node_width: int | None,
    resolved_process_grids: Mapping[StateName, FloatND],
) -> Callable[[ScalarFloat], ScalarFloat]:
    del pieces, next_regime_to_continuation, dtype, stochastic_node_width
    del resolved_process_grids
    return functools.partial(
        _direct_continuation, cast("ScalarFloat", combo_pool["wealth"])
    )


def _asset_feature_producer(
    *, monkeypatch: pytest.MonkeyPatch, state_grid: Float1D
) -> _SolveOneComboAssetRows:
    base = _feature_producer(monkeypatch=monkeypatch)
    assert base.pieces.query_spec is not None
    pieces = replace(
        base.pieces,
        n_carry_rows=int(state_grid.shape[0]),
        query_spec=replace(
            base.pieces.query_spec,
            row_capacity=int(state_grid.shape[0]),
            corner_coordinate_kind="liquid_row",
        ),
    )
    monkeypatch.setattr(
        asset_row_module,
        "_get_expected_continuation_value",
        _direct_expected_continuation,
    )
    return _SolveOneComboAssetRows(
        pieces=pieces,
        pool=base.pool,
        state_grid=state_grid,
        next_regime_to_continuation=MappingProxyType({}),
        euler_point_width=None,
        savings_point_width=None,
        stochastic_node_width=None,
        envelope_cell_width=1,
        resolved_process_grids=MappingProxyType({}),
    )


def test_asset_rows_keep_floor_direct_derivative_distinct_from_selected_saving(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The liquid corner bank retains its own native derivative on the real grid."""
    producer = _asset_feature_producer(
        monkeypatch=monkeypatch, state_grid=jnp.asarray([0.0, 3.0])
    )
    result = producer((jnp.asarray(1.0),))
    assert isinstance(result, BoundedResult)
    output = result.output
    data = output[6]
    assert data is not None
    corner = data.liquid_corner_data
    assert corner is not None
    assert data.corner_continuation is None
    np.testing.assert_allclose(output[1], [1.0, 7.0])
    np.testing.assert_allclose(data.savings, [0.5, 2.0])
    np.testing.assert_allclose(data.floor_used, 0.5)
    np.testing.assert_allclose(output[2], [0.5, 5.0])
    np.testing.assert_allclose(
        output[0],
        [np.log(11.0 / 4.0), np.log(169.0 / 3.0)],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(
        output[4],
        [23.0 / 11.0, 1.0 / 5.0 + 1.0 / 26.0],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(
        corner.continuation[:, 0],
        np.log([5.5, 8.5]),
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(
        corner.direct_marginal[:, 0],
        [1.0 / 11.0, 1.0 / 17.0],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_array_equal(corner.node_kind[:, 0], [1, 1])
    np.testing.assert_array_equal(corner.interval_id[:, 0], [0, 0])
    np.testing.assert_array_equal(output[7], [True, True])
    assert result.query_status is not None
    assert not bool(result.query_status[-1].coverage_missing)


def test_asset_node_value_derivative_uses_the_published_liquid_row_piece(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conditional envelope's clamp does not select a foreign liquid-row germ."""
    producer = _asset_feature_producer(
        monkeypatch=monkeypatch, state_grid=jnp.asarray([0.0, 3.0, 6.0])
    )
    result = producer((jnp.asarray(1.0),))
    assert isinstance(result, BoundedResult)
    output = result.output
    data = output[6]
    assert data is not None
    np.testing.assert_allclose(output[1], [1.0, 7.0, 13.0])
    np.testing.assert_allclose(data.savings, [0.5, 2.0, 2.0])
    assert int(data.side_id[1]) == -1

    def represented_value(resources: ScalarFloat) -> ScalarFloat:
        return read_owned_row(
            query=resources,
            grid=output[1],
            value=output[3],
            marginal=output[4],
            data=data,
            owner_capacity=24,
            side_is_row_side=False,
        ).value

    def published_value(resources: ScalarFloat) -> ScalarFloat:
        return interp_on_padded_grid(
            x_query=resources,
            xp=output[1],
            fp=output[3],
            fp_slopes=output[4],
        )

    reference = jax.grad(published_value)(jnp.asarray(7.0))
    np.testing.assert_allclose(
        reference,
        0.5 * np.log(256.0 / 169.0),
        rtol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(
        jax.grad(represented_value)(jnp.asarray(7.0)),
        reference,
        rtol=10 ** (-DECIMAL_PRECISION),
    )


def _inverse_status_marginal(marginal: ScalarFloat) -> ScalarFloat:
    return 1.0 / marginal


def _install_bounded_child(
    *,
    monkeypatch: pytest.MonkeyPatch,
    evaluated: list[tuple[float, float]],
) -> None:
    """Bind Euler candidates and direct reads to the same bounded continuation."""

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record(wealth: ScalarFloat, saving: ScalarFloat) -> None:
        evaluated.append((float(wealth), float(saving)))

    def bind(
        *,
        plan: ContinuationPlan,
        combo_pool: EconFunctionKwargs,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        dtype: np.dtype,
        stochastic_node_width: int | None = None,
        resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
        co_map_state_names: tuple[StateName, ...] = (),
        collect_continuation_read_counts: bool = False,
    ) -> Callable[[ScalarFloat], BoundedResult[tuple[ScalarFloat, ScalarFloat]]]:
        del plan, next_regime_to_continuation, dtype, stochastic_node_width
        del resolved_process_grids, co_map_state_names
        del collect_continuation_read_counts
        wealth = cast("ScalarFloat", combo_pool.get("wealth", jnp.asarray(0.0)))

        def child(
            saving: ScalarFloat,
        ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
            jax.debug.callback(record, wealth, saving, ordered=True)
            terminal = 3.0 * saving + 4.0 + wealth
            return BoundedResult(
                output=(jnp.log(terminal), 3.0 / terminal),
                bound_status=(),
                query_status=(
                    QueryInputStatus(
                        saving == 0.5,
                        saving == 1.0,
                        (saving > 0.5) & (saving < 1.0),
                    ),
                ),
            )

        return child

    def factory(
        *,
        pieces: _EgmKernelPieces,
        combo_pool: EconFunctionKwargs,
        discount_factor: ScalarFloat,
        utility_of_action: Callable[[ScalarFloat], ScalarFloat],
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        dtype: np.dtype,
        stochastic_node_width: int | None,
        resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
        economic_continuation: Callable[
            [ScalarFloat], BoundedResult[tuple[ScalarFloat, ScalarFloat]]
        ]
        | None = None,
    ) -> _ComputeNode:
        assert economic_continuation is None
        return _ComputeNode(
            continuation=bind(
                plan=pieces.continuation_plan,
                combo_pool=combo_pool,
                next_regime_to_continuation=next_regime_to_continuation,
                dtype=dtype,
                stochastic_node_width=stochastic_node_width,
                resolved_process_grids=resolved_process_grids,
            ),
            discount_factor=discount_factor,
            inverse_marginal_utility=_inverse_status_marginal,
            utility_of_action=utility_of_action,
        )

    monkeypatch.setattr(asset_row_module, "_get_compute_node", factory)
    monkeypatch.setattr(step_core_module, "_get_compute_node", factory)
    monkeypatch.setattr(asset_row_module, "bind_continuation", bind)


def test_asset_direct_gradient_keeps_child_auxiliary_and_reuses_first_read(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An off-grid chosen-saving read contributes status without repeating floor AD."""
    expected_factory = asset_row_module._get_expected_continuation_value
    producer = _asset_feature_producer(
        monkeypatch=monkeypatch, state_grid=jnp.asarray([0.0, 1.0, 3.0])
    )
    evaluated: list[tuple[float, float]] = []
    _install_bounded_child(monkeypatch=monkeypatch, evaluated=evaluated)
    monkeypatch.setattr(
        asset_row_module, "_get_expected_continuation_value", expected_factory
    )
    result = producer((jnp.asarray(1.0),))
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    output = result.output
    tolerance = 10 ** (-DECIMAL_PRECISION)
    assert output[6] is not None
    np.testing.assert_allclose(output[6].savings, [0.5, 2.0 / 3.0, 2.0], rtol=tolerance)
    np.testing.assert_allclose(output[2], [0.5, 7.0 / 3.0, 5.0], rtol=tolerance)
    np.testing.assert_allclose(
        output[4], [23.0 / 11.0, 0.5, 31.0 / 130.0], rtol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(output[0])[[0, 2]], np.log([11.0 / 4.0, 169.0 / 3.0]), rtol=tolerance
    )
    corner = output[6].liquid_corner_data
    assert corner is not None
    np.testing.assert_allclose(
        corner.continuation[:, 0], np.log([5.5, 6.5, 8.5]), rtol=tolerance
    )
    np.testing.assert_allclose(
        corner.direct_marginal[:, 0],
        [1.0 / 11.0, 1.0 / 13.0, 1.0 / 17.0],
        rtol=tolerance,
    )
    assert result.query_status is not None
    assert len(result.query_status) == 2
    np.testing.assert_array_equal(result.query_status[0], [True, True, True])
    np.testing.assert_array_equal(result.query_status[1], [False, False, False])
    assert len(evaluated) == 15
    for wealth in (0.0, 1.0, 3.0):
        assert sum(state == wealth for state, _ in evaluated) == 5
    assert evaluated.count((0.0, 0.5)) == 2
    assert evaluated.count((3.0, 2.0)) == 2
    middle = sorted(saving for state, saving in evaluated if state == 1.0)
    np.testing.assert_allclose(middle, [0.5, 2.0 / 3.0, 1.0, 1.5, 2.0], rtol=tolerance)


def test_ordinary_producer_preserves_child_slots_without_repeating_floor_read(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Child integrity accompanies the ordinary envelope and its original first W."""
    producer = _feature_producer(monkeypatch=monkeypatch)
    evaluated: list[tuple[float, float]] = []
    _install_bounded_child(monkeypatch=monkeypatch, evaluated=evaluated)
    result = producer((jnp.asarray(1.0),))
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    output = result.output
    data = output[6]
    assert data is not None
    assert data.corner_continuation is not None
    tolerance = 10 ** (-DECIMAL_PRECISION)
    np.testing.assert_allclose(output[0], [np.log(11.0 / 4.0)], rtol=tolerance)
    np.testing.assert_allclose(data.corner_continuation, [np.log(5.5)], rtol=tolerance)
    np.testing.assert_allclose(data.floor_used, 0.5)
    active = np.asarray(data.node_kind) != 0
    np.testing.assert_allclose(
        np.asarray(data.savings)[active][[0, -1]], [0.5, 2.0], rtol=tolerance
    )
    assert result.query_status is not None
    assert len(result.query_status) == 2
    np.testing.assert_array_equal(result.query_status[0], [True, True, False])
    np.testing.assert_array_equal(result.query_status[1], [False, False, False])
    assert len(result.bound_status) == 1
    np.testing.assert_array_equal(result.bound_status[0], [False, False])
    assert len(evaluated) == 4
    np.testing.assert_allclose(
        sorted(saving for _, saving in evaluated), [0.5, 1.0, 1.5, 2.0]
    )


def test_literal_floor_parent_retains_child_status_without_query_payload(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal-floor source owns construction integrity beside its child slots."""
    producer = _feature_producer(monkeypatch=monkeypatch)
    producer = replace(
        producer,
        pieces=replace(
            producer.pieces,
            query_spec=None,
            lower_func=None,
            check_query_construction=True,
        ),
    )
    evaluated: list[tuple[float, float]] = []
    _install_bounded_child(monkeypatch=monkeypatch, evaluated=evaluated)
    result = producer((jnp.asarray(1.0),))
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    np.testing.assert_allclose(
        result.output[0], [np.log(4.0)], rtol=10 ** (-DECIMAL_PRECISION)
    )
    assert result.output[6] is None
    np.testing.assert_array_equal(result.output[7], [True])
    assert result.bound_status == ()
    assert result.query_status is not None
    assert len(result.query_status) == 2
    np.testing.assert_array_equal(result.query_status[0], [True, True, False])
    np.testing.assert_array_equal(result.query_status[1], [False, False, False])
    assert len(evaluated) == 4
    np.testing.assert_allclose(
        sorted(saving for _, saving in evaluated), [0.0, 0.5, 1.0, 1.5]
    )


def test_literal_asset_parent_retains_chosen_read_status_without_query_payload(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal asset-row source retains candidate and chosen-read child integrity."""
    expected_factory = asset_row_module._get_expected_continuation_value
    producer = _asset_feature_producer(
        monkeypatch=monkeypatch, state_grid=jnp.asarray([0.0, 1.0])
    )
    producer = replace(
        producer,
        pieces=replace(
            producer.pieces,
            query_spec=None,
            lower_func=None,
            borrowing_limit=jnp.asarray(0.5),
            savings_nodes=jnp.asarray([0.5, 1.0, 1.5, 2.0]),
            check_query_construction=True,
        ),
    )
    evaluated: list[tuple[float, float]] = []
    _install_bounded_child(monkeypatch=monkeypatch, evaluated=evaluated)
    monkeypatch.setattr(
        asset_row_module, "_get_expected_continuation_value", expected_factory
    )
    result = producer((jnp.asarray(1.0),))
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    tolerance = 10 ** (-DECIMAL_PRECISION)
    np.testing.assert_allclose(result.output[0][0], np.log(11.0 / 4.0), rtol=tolerance)
    np.testing.assert_allclose(result.output[1], [1.0, 3.0], rtol=tolerance)
    np.testing.assert_allclose(result.output[2], [0.5, 7.0 / 3.0], rtol=tolerance)
    np.testing.assert_allclose(result.output[4], [23.0 / 11.0, 0.5], rtol=tolerance)
    assert result.output[6] is None
    np.testing.assert_array_equal(result.output[7], [True, True])
    assert result.bound_status == ()
    assert result.query_status is not None
    assert len(result.query_status) == 2
    np.testing.assert_array_equal(result.query_status[0], [True, True, True])
    np.testing.assert_array_equal(result.query_status[1], [False, False, False])
    assert len(evaluated) == 10
    for wealth in (0.0, 1.0):
        assert sum(state == wealth for state, _ in evaluated) == 5
    middle = sorted(saving for state, saving in evaluated if state == 1.0)
    np.testing.assert_allclose(middle, [0.5, 2.0 / 3.0, 1.0, 1.5, 2.0], rtol=tolerance)


def _conditional_step(*, monkeypatch: pytest.MonkeyPatch) -> _EGMStep:
    """Build the actual step around the source-owned producer and bounded child."""
    producer = _feature_producer(monkeypatch=monkeypatch)
    return _EGMStep(
        pieces=replace(producer.pieces, combo_names=()),
        get_solve_one_combo=step_core_module._get_solve_one_combo,
        mask_gap_rows=False,
        has_taste_shocks=False,
        own_discrete_state_names=(),
        own_passive_state_names=(),
        own_discrete_action_values=MappingProxyType({}),
        own_runtime_process_names=(),
    )


def test_conditional_step_publishes_query_carry_counts_and_integrity(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full step separates node feasibility from unmasked conditional carry rows."""
    step = _conditional_step(monkeypatch=monkeypatch)
    evaluated: list[tuple[float, float]] = []
    _install_bounded_child(monkeypatch=monkeypatch, evaluated=evaluated)
    result = step(
        next_regime_to_continuation=MappingProxyType({}),
        wealth=jnp.asarray([-0.25, 0.0]),
        z=jnp.asarray(1.0),
        period=jnp.asarray(0),
        age=jnp.asarray(0.0),
        discount_factor=jnp.asarray(1.0),
    )
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    assert len(result.output) == 4
    value, carry, policy, counts = result.output
    tolerance = 10 ** (-DECIMAL_PRECISION)
    np.testing.assert_array_equal(value[:1], [-np.inf])
    np.testing.assert_allclose(value[1:], [np.log(11.0 / 4.0)], rtol=tolerance)
    np.testing.assert_array_equal(counts, [1, 0])
    assert carry.query_data is not None
    np.testing.assert_allclose(carry.query_data.floor_used, 0.5)
    active = np.asarray(carry.query_data.node_kind) != 0
    np.testing.assert_allclose(
        np.asarray(carry.query_data.savings)[active][[0, -1]], [0.5, 2.0]
    )
    assert carry.query_data.corner_continuation is not None
    np.testing.assert_allclose(
        carry.query_data.corner_continuation, [np.log(5.5)], rtol=tolerance
    )
    np.testing.assert_allclose(policy.endog_grid, carry.endog_grid, equal_nan=True)
    assert len(result.bound_status) == 1
    np.testing.assert_array_equal(result.bound_status[0], [False, False])
    assert result.query_status is not None
    assert len(result.query_status) == 2
    np.testing.assert_array_equal(result.query_status[0], [True, True, False])
    np.testing.assert_array_equal(result.query_status[1], [False, False, False])
    assert len(evaluated) == 4


def test_conditional_values_core_retains_counts_and_integrity_without_policy(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping policy output preserves conditional carry and acceptance data."""
    step = _conditional_step(monkeypatch=monkeypatch)
    evaluated: list[tuple[float, float]] = []
    _install_bounded_child(monkeypatch=monkeypatch, evaluated=evaluated)
    result = _dcegm_values_core(
        core=step,
        next_regime_to_continuation=MappingProxyType({}),
        wealth=jnp.asarray([-0.25, 0.0]),
        z=jnp.asarray(1.0),
        period=jnp.asarray(0),
        age=jnp.asarray(0.0),
        discount_factor=jnp.asarray(1.0),
    )
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    assert len(result.output) == 3
    value, carry, counts = result.output
    np.testing.assert_array_equal(value[:1], [-np.inf])
    np.testing.assert_allclose(
        value[1:], [np.log(11.0 / 4.0)], rtol=10 ** (-DECIMAL_PRECISION)
    )
    np.testing.assert_array_equal(counts, [1, 0])
    assert carry.query_data is not None
    np.testing.assert_allclose(carry.query_data.floor_used, 0.5)
    assert result.query_status is not None
    assert len(result.query_status) == 2
    np.testing.assert_array_equal(result.query_status[0], [True, True, False])
    np.testing.assert_array_equal(result.query_status[1], [False, False, False])
    assert len(result.bound_status) == 1
    np.testing.assert_array_equal(result.bound_status[0], [False, False])
    assert len(evaluated) == 4


@pytest.mark.parametrize("cell_width", [None, 1])
def test_mapped_conditional_step_keeps_floor_per_cell_across_discrete_choices(
    *, monkeypatch: pytest.MonkeyPatch, cell_width: int | None
) -> None:
    """Choice rows share a cell floor and preserve one reduced integrity bank."""
    step = _conditional_step(monkeypatch=monkeypatch)
    assert step.pieces.query_spec is not None
    floors: list[float] = []

    def record(z: np.ndarray) -> None:
        floors.append(float(z))

    def lower(*, z: ScalarFloat, **kwargs: EconFunctionArg) -> ScalarFloat:
        del kwargs
        jax.debug.callback(record, z)
        return z / 2.0

    step = replace(
        step,
        pieces=replace(
            step.pieces,
            combo_names=("z", "choice"),
            lower_func=lower,
            query_spec=replace(step.pieces.query_spec, action_names=("choice",)),
        ),
        own_passive_state_names=("z",),
        own_discrete_action_values=MappingProxyType(
            {"choice": jnp.asarray([0, 1, 2], dtype=jnp.int32)}
        ),
    )
    evaluated: list[tuple[float, float]] = []
    _install_bounded_child(monkeypatch=monkeypatch, evaluated=evaluated)
    result = jax.jit(step, static_argnames=("_lcm_cell_width",))(
        next_regime_to_continuation=MappingProxyType({}),
        _lcm_cell_width=cell_width,
        wealth=jnp.asarray([0.0]),
        z=jnp.asarray([0.0, 1.0]),
        period=jnp.asarray(0),
        age=jnp.asarray(0.0),
        discount_factor=jnp.asarray(1.0),
    )
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    value, carry, policy, counts = result.output
    tolerance = 10 ** (-DECIMAL_PRECISION)
    np.testing.assert_allclose(
        value, [[np.log(4.0), np.log(11.0 / 4.0)]], rtol=tolerance
    )
    np.testing.assert_array_equal(counts, [0, 0])
    assert carry.query_data is not None
    assert carry.query_data.floor_used.shape == (2,)
    np.testing.assert_allclose(carry.query_data.floor_used, [0.0, 0.5])
    assert sorted(floors) == [0.0, 1.0]
    assert carry.endog_grid.shape == (2, 3, 32)
    assert carry.query_data.savings.shape == (2, 3, 32)
    for cell, expected in enumerate(([0.0, 1.5], [0.5, 2.0])):
        for choice in range(3):
            active = np.asarray(carry.query_data.node_kind[cell, choice]) != 0
            np.testing.assert_allclose(
                np.asarray(carry.query_data.savings[cell, choice])[active][[0, -1]],
                expected,
            )
    np.testing.assert_allclose(policy.endog_grid, carry.endog_grid, equal_nan=True)
    assert result.query_status is not None
    assert len(result.query_status) == 2
    np.testing.assert_array_equal(result.query_status[0], [True, True, False])
    np.testing.assert_array_equal(result.query_status[1], [False, False, False])
    assert len(result.bound_status) == 1
    np.testing.assert_array_equal(result.bound_status[0], [False, False])


def _canonical_producer_steps(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool
) -> tuple[tuple[_EGMStep, ...], EGMQuerySpec, EGMContinuationSpec]:
    """Build both source periods through the genuine canonical processor."""
    processor = _capture_processor(monkeypatch=monkeypatch, asset=asset)
    real_build = DCEGM.build_period_kernels
    cores: list[_EGMStep] = []
    authorities: list[EGMQuerySpec] = []

    # keyword-only-exempt: library-callback=Python.descriptor_binding
    def build(solver: DCEGM, *, context: SolverBuildContext) -> SolutionKernels:
        built = real_build(solver, context=context)
        assert context.dcegm_query_models is not None
        model = context.dcegm_query_models[context.regime_name]
        authorities.append(model.authority)
        for kernel in built.period_kernels.values():
            assert isinstance(kernel, _DCEGMPeriodKernel)
            core = inspect.unwrap(kernel.core_programs()["replay"].function)
            assert isinstance(core, _EGMStep)
            cores.append(core)
        return built

    with monkeypatch.context() as patch:
        patch.setattr(DCEGM, "build_period_kernels", build)
        regimes = processor(gated_continuations_by_source=MappingProxyType({}))
    assert len(authorities) == 1
    assert regimes["saving"].solution.query_authority is authorities[0]
    continuation = regimes["saving"].solution.continuation_spec
    assert isinstance(continuation, EGMContinuationSpec)
    return tuple(cores), authorities[0], continuation


@pytest.mark.parametrize("asset", [False, True])
def test_canonical_dc_kernel_installs_its_source_query_authority(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool
) -> None:
    """Both canonical producer builders share the source model's exact authority."""
    cores, authority, _continuation = _canonical_producer_steps(
        monkeypatch=monkeypatch, asset=asset
    )
    assert len(cores) == 2
    assert cores[0].pieces.query_spec is authority
    assert cores[1].pieces.query_spec is authority
    assert authority.corner_coordinate_kind == (
        "liquid_row" if asset else "compact_interval"
    )


@pytest.mark.parametrize("asset", [False, True])
def test_canonical_dc_factory_emits_original_saving_with_owned_geometry(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool
) -> None:
    """Canonical feature factories transport supplied saving on their native read."""
    cores, _authority, _continuation = _canonical_producer_steps(
        monkeypatch=monkeypatch, asset=asset
    )
    pieces = cores[0].pieces
    grid = jnp.asarray([1.0, 2.0, 3.0])
    policy = jnp.asarray([0.5, 1.0, 1.5])
    value = jnp.asarray([0.0, 1.0, 1.5])
    marginal = jnp.asarray([1.0, 1.0, 1.0])
    savings = jnp.asarray([11.0, 13.0, 17.0])
    if asset:
        bracket = pieces.refine_to_bracket(
            x_query=jnp.asarray(1.5),
            endog_grid=grid,
            policy=policy,
            value=value,
            marginal_utility=marginal,
            savings=savings,
        )
        assert isinstance(bracket, OwnedQueryBracket)
        np.testing.assert_allclose(bracket.savings, 12.0)
        np.testing.assert_allclose(
            [bracket.lower_savings, bracket.upper_savings], [11.0, 13.0]
        )
        assert bool(bracket.read_supported)
        assert not bool(bracket.construction_invalid)
        assert int(bracket.owner_id) == 0
    else:
        envelope = pieces.refine(
            endog_grid=grid,
            policy=policy,
            value=value,
            marginal_utility=marginal,
            savings=savings,
        )
        assert isinstance(envelope, OwnedEnvelope)
        assert int(envelope.n_kept) == 3
        np.testing.assert_allclose(envelope.savings[:3], [11.0, 13.0, 17.0])
        np.testing.assert_array_equal(envelope.owner_id[:3], [0, 0, 0])


@pytest.mark.parametrize("asset", [False, True])
def test_canonical_dc_template_retains_exact_conditional_shapes_and_authority(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool
) -> None:
    """Rolling templates use the producer's declared conditional numeric layout."""
    _cores, authority, continuation = _canonical_producer_steps(
        monkeypatch=monkeypatch, asset=asset
    )
    assert continuation.layout.query_spec is authority
    data = continuation.template.query_data
    assert data is not None
    capacity = 5 if asset else 14
    assert continuation.template.endog_grid.shape == (2, 3, 2, 2, capacity)
    assert data.savings.shape == (2, 3, 2, 2, capacity)
    assert data.floor_used.shape == (2, 3, 2)
    np.testing.assert_array_equal(data.floor_used, np.zeros((2, 3, 2)))
    np.testing.assert_array_equal(data.node_kind, np.zeros(data.node_kind.shape))
    np.testing.assert_array_equal(data.link_valid, np.zeros(data.link_valid.shape))
    if asset:
        assert data.corner_continuation is None
        corner = data.liquid_corner_data
        assert corner is not None
        assert corner.continuation.shape == (2, 3, 2, 2, 5, 1)
        np.testing.assert_array_equal(
            corner.direct_marginal, np.zeros((2, 3, 2, 2, 5, 1))
        )
    else:
        assert data.liquid_corner_data is None
        assert data.corner_continuation is not None
        assert data.corner_continuation.shape == (2, 3, 2, 2, 1)
        np.testing.assert_array_equal(
            data.corner_continuation, np.zeros((2, 3, 2, 2, 1))
        )


@pytest.mark.parametrize("asset", [False, True])
def test_canonical_dc_kernel_declares_source_construction_slot_and_roles(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool
) -> None:
    """Source construction has a canonical slot and complete bounded output roles."""
    processor = _capture_processor(monkeypatch=monkeypatch, asset=asset)
    real_build = DCEGM.build_period_kernels
    seen: list[int] = []

    # keyword-only-exempt: library-callback=Python.descriptor_binding
    def build(solver: DCEGM, *, context: SolverBuildContext) -> SolutionKernels:
        built = real_build(solver, context=context)
        assert context.dcegm_query_models is not None
        model = context.dcegm_query_models[context.regime_name]
        slots = built.query_input_authorities
        assert slots is not None
        for period, kernel in built.period_kernels.items():
            assert isinstance(kernel, _DCEGMPeriodKernel)
            assert slots[period][-1] == (
                context.regime_name,
                period,
                model.authority.bound,
            )
            assert slots[period][-1][2] is model.authority.bound
            for name, program in kernel.core_programs().items():
                roles = program.output_roles
                assert isinstance(roles, BoundedResult)
                assert len(roles.output) == (3 if name == "main" else 4)
                assert roles.output[0] is VALUE
                assert len(roles.bound_status) == 1
                assert len(roles.bound_status[0]) == 2
                assert roles.query_status is not None
                assert len(roles.query_status) == len(slots[period])
                assert len(roles.query_status[-1]) == 3
                for flag in jax.tree.leaves(roles.query_status[-1]):
                    assert isinstance(flag, StateAxesLeading)
                    assert flag.state_names == ()
                    assert flag.shape == ()
                    assert flag.dtype == jnp.bool_
                carry = roles.output[1]
                assert isinstance(carry, EGMCarry)
                data = carry.query_data
                assert data is not None
                assert isinstance(data.floor_used, StateAxesLeading)
                assert data.floor_used.state_names == model.authority.state_names
                assert data.floor_used.shape == (2, 3, 2)
                assert isinstance(data.savings, StateAxesLeading)
                assert data.savings.shape == (2, 3, 2, 2, 5 if asset else 14)
                counts = roles.output[2 if name == "main" else 3]
                assert len(counts) == 2
                for count in counts:
                    assert isinstance(count, StateAxesLeading)
                    assert count.state_names == ()
                    assert count.shape == ()
                    assert count.dtype == jnp.int32
            seen.append(period)
        return built

    with monkeypatch.context() as patch:
        patch.setattr(DCEGM, "build_period_kernels", build)
        regimes = processor(gated_continuations_by_source=MappingProxyType({}))
    assert sorted(seen) == [0, 1]
    assert regimes["saving"].solution.query_input_authorities is not None


@pytest.mark.parametrize("asset", [False, True])
@pytest.mark.parametrize("variant", ["main", "replay"])
def test_dc_period_adapter_preserves_source_failure_before_outer_acceptance(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool, variant: Literal["main", "replay"]
) -> None:
    """A finite compiled value cannot discard its source-construction rejection."""
    processor = _capture_processor(monkeypatch=monkeypatch, asset=asset)
    real_build = DCEGM.build_period_kernels
    installed: list[tuple[SolutionKernels, SolverBuildContext]] = []

    # keyword-only-exempt: library-callback=Python.descriptor_binding
    def build(solver: DCEGM, *, context: SolverBuildContext) -> SolutionKernels:
        built = real_build(solver, context=context)
        installed.append((built, context))
        return built

    with monkeypatch.context() as patch:
        patch.setattr(DCEGM, "build_period_kernels", build)
        regimes = processor(gated_continuations_by_source=MappingProxyType({}))
    assert len(installed) == 1
    built, context = installed[0]
    assert isinstance(built.continuation_spec, EGMContinuationSpec)
    carry = built.continuation_spec.template
    policy = EGMSimPolicy(
        endog_grid=carry.endog_grid,
        policy=carry.endog_grid,
        value=carry.value,
        marginal_utility=carry.marginal_utility,
        row_discrete_state_names=("income",),
        row_passive_state_names=("passive_b", "passive_a"),
        row_discrete_action_names=("work",),
    )
    values = jnp.full((2, 3, 5, 2), 7.0)
    counts = (jnp.int32(0), jnp.int32(0))
    output = (
        (values, carry, counts)
        if variant == "main"
        else (values, carry, policy, counts)
    )
    failed = BoundedResult(
        output=output,
        bound_status=(BoundInputStatus(jnp.asarray(a=False), jnp.asarray(a=False)),),
        query_status=(
            QueryInputStatus(
                jnp.asarray(a=False), jnp.asarray(a=False), jnp.asarray(a=True)
            ),
        ),
    )

    def compiled(**kwargs: PytreeValue) -> PytreeValue:
        del kwargs
        return failed

    payloads: dict[RegimeName, ContinuationPayload] = {}
    for name, regime in regimes.items():
        template = regime.solution.continuation_template
        if template is not None:
            payloads[name] = template
    kernel = built.period_kernels[1]
    assert isinstance(kernel, _DCEGMPeriodKernel)
    result = kernel(
        compiled_cores={variant: compiled},
        state_action_space=context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=payloads,
        flat_params=MappingProxyType({}),
        period=1,
        ages=context.ages,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(result, FeasibilityKernelOutput)
    np.testing.assert_array_equal(result.value, np.full((2, 3, 5, 2), 7.0))
    np.testing.assert_array_equal(result.feasibility_counts, [0, 0])
    assert result.continuations[EGM_CONTINUATION] is carry
    assert (SIMULATION_POLICY in result.replay) == (variant == "replay")
    assert len(result.bound_status) == 1
    np.testing.assert_array_equal(result.bound_status[0], [False, False])
    assert result.query_status is not None
    np.testing.assert_array_equal(result.query_status[0], [False, False, True])
    assert built.query_input_authorities is not None
    with pytest.raises(
        InvalidValueFunctionError, match=r"query_coverage_missing.*saving.*period 1"
    ):
        validate_query_input_status(
            statuses=result.query_status, authorities=built.query_input_authorities[1]
        )


@pytest.mark.parametrize("asset", [False, True])
def test_canonical_native_period_publishes_floor_corner_and_current_values(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool
) -> None:
    """Actual compiled producers retain canonical floors before outer acceptance."""
    processor = _capture_processor(monkeypatch=monkeypatch, asset=asset)
    real_build = DCEGM.build_period_kernels
    installed: list[tuple[SolutionKernels, SolverBuildContext]] = []

    # keyword-only-exempt: library-callback=Python.descriptor_binding
    def build(solver: DCEGM, *, context: SolverBuildContext) -> SolutionKernels:
        built = real_build(solver, context=context)
        installed.append((built, context))
        return built

    with monkeypatch.context() as patch:
        patch.setattr(DCEGM, "build_period_kernels", build)
        regimes = processor(gated_continuations_by_source=MappingProxyType({}))
    assert len(installed) == 1
    built, context = installed[0]
    assert context.enable_jit
    source_params = MappingProxyType(
        {
            "koopmans_aggregator__discount_factor": jnp.asarray(0.0),
            "utility__utility_scale": jnp.asarray(0.0),
            "floor_helper__floor_scale": jnp.asarray(0.5),
            "eligible__eligibility": jnp.asarray(0.0),
        }
    )
    assert context.flat_param_names == frozenset(source_params)
    flat_params: FlatParams = MappingProxyType(
        {"saving": source_params, "done": MappingProxyType({})}
    )
    payloads: dict[RegimeName, ContinuationPayload] = {}
    for name, regime in regimes.items():
        template = regime.solution.continuation_template
        if template is not None:
            payloads[name] = template
    grid = (
        context.regime_to_v_interpolation_info["done"]
        .continuous_states["liquid"]
        .to_jax()
    )
    payloads["done"] = EGMCarry(
        endog_grid=grid,
        value=jnp.log(grid),
        marginal_utility=1.0 / grid,
        taste_shock_scale=jnp.asarray(0.0),
    )
    kernel = built.period_kernels[1]
    assert isinstance(kernel, _DCEGMPeriodKernel)
    core = kernel.core_programs()["main"].function
    assert hasattr(core, "lower")
    result = kernel(
        compiled_cores={"main": core},
        state_action_space=context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=payloads,
        flat_params=flat_params,
        period=1,
        ages=context.ages,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(result, FeasibilityKernelOutput)
    tolerance = 10 ** (-DECIMAL_PRECISION)
    np.testing.assert_allclose(
        result.value[:, :, -1, 0], np.full((2, 3), np.log(8.9)), rtol=tolerance
    )
    np.testing.assert_allclose(
        result.value[:, :, -1, 1], np.full((2, 3), np.log(8.4)), rtol=tolerance
    )
    np.testing.assert_array_equal(result.feasibility_counts, [0, 0])
    carry = result.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    assert context.dcegm_query_models is not None
    authority = context.dcegm_query_models["saving"].authority
    validate_query_template(template=carry, spec=authority)
    data = carry.query_data
    assert data is not None
    np.testing.assert_allclose(
        data.floor_used, np.broadcast_to([0.1, 0.6], (2, 3, 2)), rtol=tolerance
    )
    if asset:
        _assert_asset_corner(data=data, tolerance=tolerance)
    assert len(result.bound_status) == 1
    np.testing.assert_array_equal(result.bound_status[0], [False, False])
    assert result.query_status is not None
    assert len(result.query_status) == 1
    np.testing.assert_array_equal(result.query_status[0], [False, False, False])
    assert built.query_input_authorities is not None
    validate_query_input_status(
        statuses=result.query_status, authorities=built.query_input_authorities[1]
    )


def _assert_asset_corner(*, data: EGMQueryData, tolerance: float) -> None:
    """Check the retained liquid-corner continuation and direct marginal."""
    corner = data.liquid_corner_data
    assert corner is not None
    np.testing.assert_allclose(
        corner.continuation[..., -1, 0],
        np.full((2, 3, 2, 2), np.log(8.0)),
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        corner.direct_marginal[..., -1, 0], np.zeros((2, 3, 2, 2)), atol=tolerance
    )
