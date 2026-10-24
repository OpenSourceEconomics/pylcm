"""DC-EGM floors precede inversion and retain independently live corners."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TypedDict, Unpack, cast

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import asset_row as asset_row_module
from _lcm.egm import step as step_module
from _lcm.egm import step_core
from _lcm.egm.asset_row import _SolveOneNode
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan
from _lcm.egm.step_core import _SolveOneCombo
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution import backward_induction, feasibility
from _lcm.solution.contract import _BoundLiquidMargin
from _lcm.typing import (
    EconFunction,
    EconFunctionArg,
    EconFunctionKwargs,
    EconFunctionsMapping,
    RegimeTransitionFunction,
)
from _lcm.utils.dispatchers import map_over_leading_axis
from lcm import IrregSpacedGrid
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import KernelOutput
from lcm.solvers import DCEGM
from lcm.typing import Float1D, FloatND, RegimeName, ScalarFloat, StateName
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution.test_dynamic_bound_authority import _bound
from tests.solution.test_dynamic_bound_runtime import _floor_landing_model


def _utility(*, consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def _resources(*, wealth: FloatND) -> FloatND:
    return 2.0 * wealth + 1.0


def _lower(*, z: FloatND) -> FloatND:
    return z / 2.0


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _aggregator(*, utility: FloatND, CE: FloatND, discount_factor: float) -> FloatND:
    return utility + discount_factor * CE


def _probabilities(**kwargs: FloatND) -> dict[str, FloatND]:
    del kwargs
    return {}


def _build_producer(*, monkeypatch: pytest.MonkeyPatch) -> _SolveOneCombo:
    """Use an analytic child read to isolate current-cell producer economics."""
    solver = DCEGM(
        savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0, 1.5])),
        n_constrained_points=4,
    )._with_liquid_margin(
        _BoundLiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        )
    )
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
    solver = replace(solver, post_decision_bound=bound)
    plan = ContinuationPlan(
        stateful_targets=(),
        scalar_targets=(),
        child_reads=MappingProxyType({}),
        compute_regime_transition_probs=cast(
            "RegimeTransitionFunction", _probabilities
        ),
        post_decision_name="savings",
    )
    monkeypatch.setattr(step_module, "build_continuation_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(step_core, "_get_compute_node", _analytic_compute_node)
    monkeypatch.setattr(asset_row_module, "_get_compute_node", _analytic_compute_node)
    pieces = step_module._build_kernel_pieces(
        solver=solver,
        user_regimes=MappingProxyType({}),
        laws=MappingProxyType({}),
        functions=cast(
            "EconFunctionsMapping",
            MappingProxyType(
                {
                    "utility": _utility,
                    "resources": _resources,
                    "inverse_marginal_utility": _inverse,
                    "borrowing_lower": _lower,
                }
            ),
        ),
        koopmans_aggregator=cast("EconFunction", _aggregator),
        constraints=MappingProxyType({}),
        transitions=MappingProxyType({}),
        transition_plans=MappingProxyType({}),
        compute_regime_transition_probs=cast(
            "RegimeTransitionFunction", _probabilities
        ),
        stateful_targets=(),
        scalar_targets=(),
        n_pad=32,
        n_carry_rows=32,
        own_discrete_state_names=(),
        own_passive_state_names=("z",),
        own_discrete_action_values=MappingProxyType({}),
        euler_axis_in_V=0,
        regime_to_v_interpolation_info=MappingProxyType({}),
        age_values=jnp.asarray([0.0, 1.0]),
    )
    return _SolveOneCombo(
        pieces=pieces,
        pool={
            "period": jnp.asarray(0),
            "age": jnp.asarray(0.0),
            "discount_factor": jnp.asarray(1.0),
        },
        state_grid=jnp.asarray([-0.125, 0.0]),
        next_regime_to_continuation=MappingProxyType({}),
        savings_point_width=None,
        stochastic_node_width=None,
        envelope_cell_width=1,
        resolved_process_grids=MappingProxyType({}),
    )


def _analytic_compute_node(
    *,
    pieces: step_core._EgmKernelPieces,
    combo_pool: EconFunctionKwargs,
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    dtype: np.dtype,
    stochastic_node_width: int | None,
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> Callable[[ScalarFloat], tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]]:
    """Return the exact Euler candidate for terminal log(3 a + 4)."""
    del pieces, discount_factor, utility_of_action, next_regime_to_continuation
    del dtype, stochastic_node_width, resolved_process_grids
    wealth = cast("ScalarFloat", combo_pool.get("wealth", jnp.asarray(0.0)))

    def compute(
        savings: ScalarFloat,
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
        continuation = jnp.log(3.0 * savings + 4.0 + wealth)
        consumption = (3.0 * savings + 4.0 + wealth) / 3.0
        return (
            consumption,
            savings + consumption,
            jnp.log(consumption) + continuation,
            continuation,
        )

    return compute


@pytest.mark.parametrize(
    ("z", "expected"), [(0.0, np.log(4.0)), (1.0, np.log(11.0 / 4.0))]
)
@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_cell_floor_reaches_constrained_producer(
    *, monkeypatch: pytest.MonkeyPatch, z: float, expected: float
) -> None:
    """A fixed passive cell saves its own floor at resources one."""
    producer = _build_producer(monkeypatch=monkeypatch)
    output = producer((jnp.asarray(z),))
    assert not isinstance(output, feasibility.BoundedResult)
    np.testing.assert_array_almost_equal(
        output[0][1], expected, decimal=DECIMAL_PRECISION
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_prepared_source_offsets_keep_order_and_extent(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Translating a floor preserves the supplied spacing and node count."""
    producer = _build_producer(monkeypatch=monkeypatch)
    observed: list[np.ndarray] = []
    original = step_core._compute_nodes_over_savings

    def capture(
        *,
        compute_node: Callable[
            [ScalarFloat], tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]
        ],
        savings_nodes: Float1D,
        savings_point_width: int | None,
        bound_summary: bool = False,
    ) -> tuple[FloatND, ...]:
        observed.append(np.asarray(savings_nodes))
        output = original(
            compute_node=compute_node,
            savings_nodes=savings_nodes,
            savings_point_width=savings_point_width,
            bound_summary=bound_summary,
        )
        assert not isinstance(output, feasibility.BoundedResult)
        return output

    monkeypatch.setattr(step_core, "_compute_nodes_over_savings", capture)
    producer((jnp.asarray(1.0),))
    np.testing.assert_array_equal(observed[0], [0.5, 1.0, 1.5, 2.0])


def _node_resources(wealth: ScalarFloat) -> ScalarFloat:
    return 2.0 * wealth + 1.0


# keyword-only-exempt: library-callback=jax.grad
def _direct_continuation(wealth: ScalarFloat, savings: ScalarFloat) -> ScalarFloat:
    return jnp.log(3.0 * savings + 4.0 + wealth)


@pytest.mark.parametrize(
    ("z", "expected_value", "expected_consumption", "expected_marginal"),
    [
        (0.0, np.log(4.0), 1.0, 9.0 / 8.0),
        (1.0, np.log(11.0 / 4.0), 0.5, 23.0 / 11.0),
    ],
)
@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_asset_node_floor_and_direct_channel_have_native_units(
    *,
    monkeypatch: pytest.MonkeyPatch,
    z: float,
    expected_value: float,
    expected_consumption: float,
    expected_marginal: float,
) -> None:
    """The lower corner includes its fixed-saving direct continuation derivative."""
    producer = _build_producer(monkeypatch=monkeypatch)
    node = _SolveOneNode(
        pieces=producer.pieces,
        combo_pool={**producer.pool, "z": jnp.asarray(z)},
        discount_factor=jnp.asarray(1.0),
        next_regime_to_continuation=MappingProxyType({}),
        dtype=jnp.asarray(0.0).dtype,
        stochastic_node_width=None,
        envelope_cell_width=1,
        resolved_process_grids=MappingProxyType({}),
        savings_point_width=None,
        own_resources_of_state=_node_resources,
        continuation_of_euler_state=_direct_continuation,
    )
    output = node(jnp.asarray(0.0))
    assert not isinstance(output, feasibility.BoundedResult)
    value, consumption, marginal = output
    np.testing.assert_array_almost_equal(
        value, expected_value, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        consumption, expected_consumption, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        marginal, expected_marginal, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("width", [2, 5])
@pytest.mark.parametrize("active", [False, True])
def test_query_only_integrity_survives_partial_owned_windows(
    *, width: int, active: bool
) -> None:
    """A source without its own bound retains every active child's summary."""
    status_type = getattr(feasibility, "QueryInputStatus", None)
    assert status_type is not None, "Child query validity needs its own typed summary."

    def evaluate(value: ScalarFloat) -> feasibility.BoundedResult[FloatND]:
        return feasibility.BoundedResult(
            output=2.0 * value,
            bound_status=(),
            query_status=(
                status_type(
                    nonfinite_floor=value == 4.0,
                    invalid_shifted_grid=value == 3.0,
                    coverage_missing=value == 2.0,
                ),
            ),
        )

    values = jnp.arange(5, dtype=float)
    output = jax.jit(
        lambda xs: map_over_leading_axis(
            func=evaluate,
            xs=xs,
            batch_size=width,
            bound_summary=True,
            active=jnp.ones(5, dtype=bool) if active else xs < 2.0,
        )
    )(values)
    np.testing.assert_array_equal(output.output, [0.0, 2.0, 4.0, 6.0, 8.0])
    assert output.bound_status == ()
    assert len(output.query_status) == 1
    np.testing.assert_array_equal(jnp.stack(output.query_status[0]), [active] * 3)


def test_feature_free_bound_status_keeps_two_flag_schema() -> None:
    """The ordinary bound summary and absent optional query layout stay unchanged."""
    status = feasibility.BoundInputStatus(
        nonfinite_floor=jnp.asarray(a=False),
        invalid_shifted_grid=jnp.asarray(a=False),
    )
    output = feasibility.BoundedResult(output=jnp.asarray(7.0), bound_status=(status,))
    assert getattr(output, "query_status", None) is None
    assert len(jax.tree.leaves(output)) == 3
    with pytest.raises(TypeError):
        feasibility.BoundInputStatus(*([jnp.asarray(a=False)] * 3))


@pytest.mark.parametrize("missing", [False, True])
def test_query_acceptance_retains_target_slots_and_arithmetic_priority(
    *, missing: bool
) -> None:
    """A later child's invalid floor precedes an earlier child's missing coverage."""
    accept = getattr(feasibility, "validate_query_input_status", None)
    assert accept is not None, "Child summaries need model-owned host acceptance."
    authorities = (("early", 3, _bound()), ("late", 7, _bound()))
    records = (
        feasibility.QueryInputStatus(
            nonfinite_floor=jnp.asarray(a=False),
            invalid_shifted_grid=jnp.asarray(a=False),
            coverage_missing=jnp.asarray(a=True),
        ),
        feasibility.QueryInputStatus(
            nonfinite_floor=jnp.asarray(a=True),
            invalid_shifted_grid=jnp.asarray(a=False),
            coverage_missing=jnp.asarray(a=False),
        ),
    )
    if missing:
        with pytest.raises(TypeError, match="declaration slots"):
            accept(statuses=records[:1], authorities=authorities)
    else:
        with pytest.raises(
            InvalidValueFunctionError,
            match=r"borrowing.*nonfinite borrowing floor.*late.*period 7",
        ):
            accept(statuses=records, authorities=authorities)


class _KernelRunKwargs(TypedDict):
    compiled_cores: MappingProxyType[str, backward_induction.PlannedCore]
    capture_target: backward_induction.PeriodCaptureTarget | None


class _DispatchKwargs(_KernelRunKwargs):
    value_template: FloatND
    regime: backward_induction.Regime
    regime_name: RegimeName
    period: int
    run_kernel: Callable[..., KernelOutput]


def test_invariant_dispatch_keeps_distinct_child_records(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two invariant outputs OR each child's own record without merging targets."""
    dispatch = backward_induction._run_dispatch_unit
    observed: list[int] = []

    def instrument(**kwargs: Unpack[_DispatchKwargs]) -> tuple[KernelOutput, bool]:
        if kwargs["regime_name"] != "saving":
            return dispatch(**kwargs)
        run = kwargs["run_kernel"]

        def tagged(**run_kwargs: Unpack[_KernelRunKwargs]) -> KernelOutput:
            result = run(**run_kwargs)
            block = len(observed)
            observed.append(block)
            records = tuple(
                feasibility.QueryInputStatus(
                    nonfinite_floor=jnp.asarray(block == target),
                    invalid_shifted_grid=jnp.asarray(a=False),
                    coverage_missing=jnp.asarray(block != target),
                )
                for target in range(2)
            )
            object.__setattr__(result, "query_status", records)
            return result

        regime = kwargs["regime"]
        declared = replace(
            regime,
            solution=replace(
                regime.solution,
                query_input_authorities=MappingProxyType(
                    {kwargs["period"]: (("early", 1, _bound()), ("late", 1, _bound()))}
                ),
            ),
        )
        kwargs["regime"], kwargs["run_kernel"] = declared, tagged
        result, blocked = dispatch(**kwargs)
        assert blocked
        records = getattr(result, "query_status", None)
        assert records is not None, "Invariant dispatch discarded child query records."
        assert len(records) == 2
        np.testing.assert_array_equal(
            jnp.stack(tuple(jnp.stack(record) for record in records)),
            [[True, False, True], [True, False, True]],
        )
        # This fixture owns only the dispatch seam; its source installs no query.
        object.__setattr__(result, "query_status", None)
        return result, blocked

    monkeypatch.setattr(backward_induction, "_run_dispatch_unit", instrument)
    _floor_landing_model(
        handwritten=False,
        invalid=False,
        budgeted=True,
        value_aware=False,
        invariant=True,
        raised=False,
    ).solve(params={"discount_factor": 1.0}, log_level="off")
    assert observed == [0, 1]


@pytest.mark.parametrize("bare_first", [False, True])
def test_each_invariant_block_must_supply_declared_child_records(
    *, monkeypatch: pytest.MonkeyPatch, bare_first: bool
) -> None:
    """A bare output cannot borrow another block's mandatory child summaries."""
    dispatch = backward_induction._run_dispatch_unit

    def instrument(**kwargs: Unpack[_DispatchKwargs]) -> tuple[KernelOutput, bool]:
        if kwargs["regime_name"] != "saving":
            return dispatch(**kwargs)
        run = kwargs["run_kernel"]
        blocks: list[int] = []

        def tagged(**run_kwargs: Unpack[_KernelRunKwargs]) -> KernelOutput:
            output = run(**run_kwargs)
            block = len(blocks)
            blocks.append(block)
            if (block == 0) == bare_first:
                return KernelOutput(value=output.value)
            assert isinstance(output, feasibility.FeasibilityKernelOutput)
            return replace(
                output,
                query_status=(
                    feasibility.QueryInputStatus(
                        nonfinite_floor=jnp.asarray(a=False),
                        invalid_shifted_grid=jnp.asarray(a=False),
                        coverage_missing=jnp.asarray(a=False),
                    ),
                ),
            )

        regime = kwargs["regime"]
        declared = replace(
            regime,
            solution=replace(
                regime.solution,
                query_input_authorities=MappingProxyType(
                    {kwargs["period"]: (("done", 1, _bound()),)}
                ),
            ),
        )
        kwargs["regime"], kwargs["run_kernel"] = declared, tagged
        output, blocked = dispatch(**kwargs)
        # The fixture tests block completeness before the separate host acceptor.
        object.__setattr__(output, "query_status", None)
        return output, blocked

    monkeypatch.setattr(backward_induction, "_run_dispatch_unit", instrument)
    model = _floor_landing_model(
        handwritten=False,
        invalid=False,
        budgeted=True,
        value_aware=False,
        invariant=True,
        raised=False,
    )
    with pytest.raises(TypeError, match="declaration slots"):
        model.solve(params={"discount_factor": 1.0}, log_level="off")


@pytest.mark.parametrize("asset_rows", [False, True])
@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_cell_floor_is_evaluated_once_before_discrete_action_mapping(
    *, monkeypatch: pytest.MonkeyPatch, asset_rows: bool
) -> None:
    """Adding discrete alternatives does not repeat the fixed-cell floor DAG."""
    producer = _build_producer(monkeypatch=monkeypatch)
    seen: list[float] = []

    def record(z: np.ndarray) -> None:
        seen.append(float(z))

    def lower(*, z: ScalarFloat, **kwargs: EconFunctionArg) -> ScalarFloat:
        del kwargs
        jax.debug.callback(record, z)
        return z / 2.0

    pieces = replace(
        producer.pieces,
        lower_func=lower,
        combo_names=("z", "choice"),
    )
    kernel = step_module._EGMStep(
        pieces=pieces,
        get_solve_one_combo=(
            asset_row_module._get_solve_one_combo_asset_rows
            if asset_rows
            else step_core._get_solve_one_combo
        ),
        mask_gap_rows=False,
        has_taste_shocks=False,
        own_discrete_state_names=(),
        own_passive_state_names=("z",),
        own_discrete_action_values=MappingProxyType(
            {"choice": jnp.array([0, 1, 2], dtype=jnp.int32)}
        ),
        own_runtime_process_names=(),
    )
    value, _, _ = jax.jit(kernel)(
        next_regime_to_continuation=MappingProxyType({}),
        wealth=producer.state_grid,
        z=jnp.array([0.0, 1.0]),
        discount_factor=jnp.array(1.0),
        period=jnp.array(0, dtype=jnp.int32),
        age=jnp.array(0.0),
    )
    value.block_until_ready()
    np.testing.assert_allclose(
        value[-1],
        jnp.log(jnp.array([4.0, 11.0 / 4.0])),
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    assert sorted(seen) == [0.0, 1.0]


@pytest.mark.parametrize("query_aux", [False, True])
def test_model_program_status_roles_survive_pickle_reconstruction(
    *, query_aux: bool
) -> None:
    """Persisted program role trees preserve their exact bounded status record."""
    status = (
        feasibility.QueryInputStatus(
            nonfinite_floor=jnp.asarray(a=False),
            invalid_shifted_grid=jnp.asarray(a=False),
            coverage_missing=jnp.asarray(a=False),
        )
        if query_aux
        else feasibility.BoundInputStatus(
            nonfinite_floor=jnp.asarray(a=False),
            invalid_shifted_grid=jnp.asarray(a=False),
        )
    )
    roles = jax.tree.map(lambda _: StateAxesLeading(state_names=(), shape=()), status)
    restored = cloudpickle.loads(cloudpickle.dumps(roles))
    assert type(restored) is type(status)
    assert len(restored) == (3 if query_aux else 2)
    assert all(isinstance(role, StateAxesLeading) for role in restored)
    assert all(role.state_names == () and role.shape == () for role in restored)
