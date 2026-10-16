"""Value a reconstructed plain action with the installed DC child operator."""

import functools
import logging
import math
from pathlib import Path
from typing import NotRequired, TypedDict, Unpack, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.dcegm_query import DCEGMQuery, DCEGMQueryResult
from _lcm.egm.one_asset_egm_step import RetiredEGMResult
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution.egm import (
    BoundedOneRowOutput,
    OneRowOutput,
    _EGMCore,
    _EGMPeriodKernel,
    _RegimeMassGuardedCore,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    FeasibilityKernelOutput,
    validate_query_input_status,
)
from _lcm.solution.period_replay import _load_capture_payload
from _lcm.typing import EconFunctionKwargs, PytreeValue
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import EGM_CONTINUATION, KernelOutput
from lcm.solvers import DCEGM, EGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    FunctionName,
    ScalarFloat,
    ScalarInt,
    UserFunction,
    UserParams,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    RegimeId,
    _inverse,
    _ordinary_law,
    _parent_resources,
    _savings,
    _terminal,
    _utility,
)
from tests.egm.test_dynamic_dcegm_boundary_factory import _owned_reference
from tests.egm.test_dynamic_dcegm_consumers import _capture_with_checks_off
from tests.egm.test_plain_literal_dc_resource_transport import _model as _period_model
from tests.solution._nbegm_direct_oracle import OracleContext, interpolate_row

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.parametrize("literal", [True, False], ids=["literal", "feature"])
def test_plain_action_values_the_actual_landing_with_its_dc_child_operator(
    *, literal: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read target resources at the original interpolated saving and source law."""
    model, params = _model(literal=literal, active=True)
    kernel, context = _capture(
        model=model, params=params, directory=tmp_path, monkeypatch=monkeypatch
    )
    core, _ = _core(kernel=kernel)
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    _assert_child_bank(child=child, literal=literal)
    reference = _parent_reference(child=child, literal=literal, active=True)
    valuation_name: FunctionName = "_value_reconstructed_action"
    original_valuation = getattr(_EGMCore, valuation_name)
    with patch.object(
        _EGMCore,
        "_value_reconstructed_action",
        autospec=True,
        side_effect=original_valuation,
    ) as valuation_calls:
        with jax.disable_jit():
            eager = _run(kernel=kernel, context=context, compiled=False)
        steps = [
            call.kwargs["step"]
            for call in valuation_calls.call_args_list
            if call.args[0] is core
        ]
    _assert_numeric(output=eager, reference=reference)
    assert len(steps) == 1
    _assert_step(step=steps[0], reference=reference, active=True)
    compiled = _run(kernel=kernel, context=context, compiled=True)
    _assert_numeric(output=compiled, reference=reference)
    _assert_close(actual=eager.value, expected=np.asarray(compiled.value))
    _assert_status(model=model, output=eager, literal=literal)
    _assert_status(model=model, output=compiled, literal=literal)


def test_literal_dc_interior_rescore_uses_each_actual_target_time_and_parameters(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read each interior action with its own specialized next-period resource map."""
    model = _period_model(parent_liquid=(0.34, 0.38))
    params: UserParams = {
        "parent": {
            "utility": {"free_shift": 5.0},
            "koopmans_aggregator": {"discount_factor": 1.0},
        },
        "child": {
            "resources": {"free_shift": 0.25},
            "koopmans_aggregator": {"discount_factor": 1.0},
        },
        "done": {},
    }
    valuation_name: FunctionName = "_value_reconstructed_action"
    original_valuation = getattr(_EGMCore, valuation_name)
    for period in (0, 1):
        kernel, context = _capture_period(
            model=model,
            params=params,
            period=period,
            directory=tmp_path / str(period),
            monkeypatch=monkeypatch,
        )
        core, _ = _core(kernel=kernel)
        child = context["next_regime_to_continuation"]["child"]
        assert isinstance(child, EGMCarry)
        assert child.query_data_for_transport() is None
        reference = _period_reference(child=child, period=period)
        with patch.object(
            _EGMCore,
            valuation_name,
            autospec=True,
            side_effect=original_valuation,
        ) as valuation_calls:
            with jax.disable_jit():
                eager = _run(kernel=kernel, context=context, compiled=False)
            steps = [
                call.kwargs["step"]
                for call in valuation_calls.call_args_list
                if call.args[0] is core
            ]
        _assert_numeric(output=eager, reference=reference)
        assert len(steps) == 1
        _assert_step(step=steps[0], reference=reference, active=True)
        _assert_numeric(
            output=_run(kernel=kernel, context=context, compiled=True),
            reference=reference,
        )


def _capture_period(
    *,
    model: Model,
    params: UserParams,
    period: int,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_EGMPeriodKernel, OracleContext]:
    """Capture the actual rolling inputs at a requested source computational period."""
    with monkeypatch.context() as scoped:
        scoped.setenv("LCM_CAPTURE_PERIOD", f"parent@{period}")
        scoped.setenv("LCM_CAPTURE_DIR", str(directory))
        model.solve(params=params, runtime_checks=False, log_level="off")
    payload = _load_capture_payload(directory=directory / f"parent@{period}")
    inputs = payload["kernel_kwargs"]
    kernel = model._regimes["parent"].solution.period_kernels[period]
    assert isinstance(kernel, _EGMPeriodKernel)
    assert kernel.source_period == period
    assert kernel.continuation_target == "child"
    return kernel, {
        "state_action_space": inputs["state_action_space"],
        "next_regime_to_V_arr": inputs["next_regime_to_V_arr"],
        "next_regime_to_continuation": inputs["next_regime_to_continuation"],
        "flat_params": inputs["flat_params"],
        "period": inputs["period"],
        "ages": inputs["ages"],
    }


def _period_reference(
    *, child: EGMCarry, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rescore the same adjacent Euler action with that target's padded LINEAR row."""
    saving_nodes = np.asarray([0.2, 0.3, 0.4], dtype=np.asarray(child.value).dtype)
    physical_nodes = 5.0 * saving_nodes - 1.0 + 0.25 * period
    slope = 2.0 + period
    resource_nodes = slope * physical_nodes + 0.5 + 0.375 * (period + 1)
    pairs = np.asarray(
        [
            _child_pair(child=child, literal=True, resources=float(r))
            for r in resource_nodes
        ]
    )
    consumption_nodes = 1.0 / (5.0 * slope * pairs[:, 1])
    endog = saving_nodes + consumption_nodes
    assert np.all(np.diff(endog) > 0.0)
    value, marginal, consumption, saving = [], [], [], []
    eps = np.finfo(np.asarray(child.value).dtype).eps
    for current in (0.34, 0.38):
        upper = int(np.searchsorted(endog, current, side="right"))
        assert 0 < upper < len(endog)
        lower = upper - 1
        weight = (current - endog[lower]) / (endog[upper] - endog[lower])
        assert 0.0 < weight < 1.0
        c = consumption_nodes[lower] + weight * (
            consumption_nodes[upper] - consumption_nodes[lower]
        )
        a = saving_nodes[lower] + weight * (saving_nodes[upper] - saving_nodes[lower])
        physical = 5.0 * a - 1.0 + 0.25 * period
        resources = slope * physical + 0.5 + 0.375 * (period + 1)
        continuation = _child_pair(child=child, literal=True, resources=resources)[0]
        expected = math.log(c) + 22.0 + continuation
        if period == 1:
            wrong_resource = 2.0 * physical + 0.875
            wrong_continuation = _child_pair(
                child=child, literal=True, resources=wrong_resource
            )[0]
            wrong_value = math.log(c) + 22.0 + wrong_continuation
            assert np.isfinite([expected, wrong_value]).all()
            assert not np.isclose(
                wrong_value, expected, rtol=300.0 * eps, atol=300.0 * eps
            )
        value.append(expected)
        marginal.append(1.0 / c)
        consumption.append(c)
        saving.append(a)
    return (
        np.asarray(value),
        np.asarray(marginal),
        np.asarray(consumption),
        np.asarray(saving),
    )


def test_feature_dc_euler_and_valuation_share_one_actual_floor_preparation(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the same prepared target floor in both actual child-read sites."""
    model, params = _model(literal=False, active=True)
    kernel, context = _capture(
        model=model, params=params, directory=tmp_path, monkeypatch=monkeypatch
    )
    core, _ = _core(kernel=kernel)
    reader = core.next_dc_query
    assert reader is not None
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    reference = _parent_reference(child=child, literal=False, active=True)
    prepared: list[tuple[ScalarFloat, Float1D, BoundInputStatus]] = []
    reads: list[tuple[ScalarFloat, Float1D, BoundInputStatus]] = []
    original_prepare = DCEGMQuery.evaluate_floor
    original_read = DCEGMQuery._read_at_floor

    # keyword-only-exempt: primary-argument=self
    def observe_prepare(
        self: DCEGMQuery, *, pool: EconFunctionKwargs
    ) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
        result = original_prepare(self, pool=pool)
        if self is reader:
            prepared.append(result)
        return result

    def observe_read(
        self: DCEGMQuery, **arguments: Unpack[_ReadInputs]
    ) -> DCEGMQueryResult:
        result = original_read(self, **arguments)
        if self is reader:
            reads.append(arguments["floor_state"])
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(DCEGMQuery, "evaluate_floor", observe_prepare)
        scoped.setattr(DCEGMQuery, "_read_at_floor", observe_read)
        with jax.disable_jit():
            output = _run(kernel=kernel, context=context, compiled=False)
    _assert_numeric(output=output, reference=reference)
    assert len(prepared) == 1
    assert len(reads) == 2
    assert reads[0] is prepared[0]
    assert reads[1] is prepared[0]
    np.testing.assert_allclose(prepared[0][0], 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(prepared[0][1], [0.5, 2.0, 3.0], rtol=0.0, atol=0.0)
    assert not bool(prepared[0][2].nonfinite_floor)
    assert not bool(prepared[0][2].invalid_shifted_grid)
    _assert_status(model=model, output=output, literal=False)
    # Agreement concerns the compiled installed program; the Python observation
    # above counts eager data-flow sites rather than XLA runtime evaluations.
    _assert_numeric(
        output=_run(kernel=kernel, context=context, compiled=True), reference=reference
    )


@pytest.mark.parametrize("active", [True, False], ids=["interior", "mixed"])
def test_dc_valuation_coverage_stays_in_its_owned_masked_target_slot(
    *, active: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report active second-read coverage and discard inactive valuation faults."""
    model, params = _model(literal=False, active=active)
    kernel, context = _capture(
        model=model, params=params, directory=tmp_path, monkeypatch=monkeypatch
    )
    core, guard = _core(kernel=kernel)
    reader = core.next_dc_query
    assert reader is not None
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    reference = _parent_reference(child=child, literal=False, active=active)
    calls: list[
        tuple[
            tuple[ScalarFloat, Float1D, BoundInputStatus],
            DCEGMQueryResult,
            DCEGMQueryResult,
        ]
    ] = []
    observed: list[BoundedOneRowOutput] = []
    frames: list[tuple[int, np.ndarray]] = []
    original_read = DCEGMQuery._read_at_floor
    original_guard = _RegimeMassGuardedCore.__call__

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record(stage: np.ndarray, facts: np.ndarray) -> None:
        frames.append((int(stage.item()), np.array(facts, copy=True)))

    def second_read_fault(
        self: DCEGMQuery, **arguments: Unpack[_ReadInputs]
    ) -> DCEGMQueryResult:
        result = original_read(self, **arguments)
        if self is not reader:
            return result
        state = cast("ScalarFloat", arguments["pool"]["liquid"])
        inactive_landing = 4.0 * jnp.log1p(jnp.asarray(0.4, dtype=state.dtype)) - 0.3
        injected = (
            jnp.ones_like(result.coverage_missing)
            if active
            else state == inactive_landing
        )
        returned = (
            result._replace(coverage_missing=injected) if len(calls) == 1 else result
        )
        jax.debug.callback(
            record,
            jnp.asarray(len(calls)),
            jnp.stack(
                (
                    result.coverage_missing,
                    returned.coverage_missing,
                    result.value,
                    returned.value,
                    result.marginal,
                    returned.marginal,
                    result.savings,
                    returned.savings,
                    state,
                    arguments["row_queries"],
                )
            ),
        )
        calls.append((arguments["floor_state"], result, returned))
        return returned

    # keyword-only-exempt: primary-argument=self
    def observe_guard(
        self: _RegimeMassGuardedCore,
        *,
        retains_regime_mass: BoolND,
        **arguments: PytreeValue,
    ) -> OneRowOutput:
        result = original_guard(
            self, retains_regime_mass=retains_regime_mass, **arguments
        )
        if self is guard:
            assert isinstance(result, BoundedResult)
            observed.append(cast("BoundedOneRowOutput", result))
        return result

    valuation_name: FunctionName = "_value_reconstructed_action"
    original_valuation = getattr(_EGMCore, valuation_name)
    with (
        monkeypatch.context() as scoped,
        patch.object(
            _EGMCore,
            "_value_reconstructed_action",
            autospec=True,
            side_effect=original_valuation,
        ) as valuation_calls,
    ):
        scoped.setattr(DCEGMQuery, "_read_at_floor", second_read_fault)
        scoped.setattr(_RegimeMassGuardedCore, "__call__", observe_guard)
        if active:
            with pytest.raises(
                InvalidValueFunctionError,
                match=r"query_coverage_missing.*child.*period 1",
            ):
                model.solve(params=params, runtime_checks=False, log_level="off")
        else:
            solution = model.solve(params=params, runtime_checks=False, log_level="off")
            _assert_close(actual=solution.values[0]["parent"], expected=reference[0])
        steps = [
            call.kwargs["step"]
            for call in valuation_calls.call_args_list
            if call.args[0] is core
        ]
    jax.effects_barrier()
    _assert_fault_observations(
        model=model,
        active=active,
        reference=reference,
        steps=steps,
        calls=calls,
        frames=frames,
        observed=observed,
    )


def _assert_fault_observations(
    *,
    model: Model,
    active: bool,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    steps: list[RetiredEGMResult],
    calls: list[
        tuple[
            tuple[ScalarFloat, Float1D, BoundInputStatus],
            DCEGMQueryResult,
            DCEGMQueryResult,
        ]
    ],
    frames: list[tuple[int, np.ndarray]],
    observed: list[BoundedOneRowOutput],
) -> None:
    """Inspect the raw injected fault and its single correctly attributed slot."""
    for stage, count in ((0, 3), (1, 2)):
        selected = np.asarray([facts for index, facts in frames if index == stage])
        assert selected.shape == (count, 10)
        np.testing.assert_array_equal(selected[:, 0], np.zeros(count))
        if stage == 0 or active:
            np.testing.assert_array_equal(selected[:, 1], np.full(count, stage))
        else:
            dtype = selected.dtype
            inactive = float(4.0 * jnp.log1p(jnp.asarray(0.4, dtype=dtype)) - 0.3)
            fault = selected[:, 8] == inactive
            np.testing.assert_array_equal(np.sort(fault), [False, True])
            np.testing.assert_array_equal(selected[:, 1], fault)
        np.testing.assert_array_equal(selected[:, 2:8:2], selected[:, 3:8:2])
        _assert_close(actual=selected[:, 9], expected=2.0 * selected[:, 8] + 1.0)
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[0][1] is calls[0][2]
    original, seeded = calls[1][1:]
    assert seeded.value is original.value
    assert seeded.marginal is original.marginal
    assert seeded.savings is original.savings
    assert seeded.bound_status is original.bound_status
    assert len(steps) == 1
    _assert_step(step=steps[0], reference=reference, active=active)
    assert len(observed) == 1
    result = observed[0]
    assert result.query_status is not None
    assert len(result.query_status) == 1
    np.testing.assert_array_equal(result.query_status[0], [False, False, active])
    slots = model._regimes["parent"].solution.query_input_authorities
    assert slots is not None
    assert tuple((name, period) for name, period, _ in slots[0]) == (("child", 1),)
    if active:
        with pytest.raises(
            InvalidValueFunctionError, match=r"query_coverage_missing.*child.*period 1"
        ):
            validate_query_input_status(
                statuses=result.query_status, authorities=slots[0]
            )
    else:
        validate_query_input_status(statuses=result.query_status, authorities=slots[0])
    for status in result.bound_status:
        np.testing.assert_array_equal(status, [False, False])
    assert len(result.output) == 4
    value, carry, _, source_coverage = result.output
    _assert_close(actual=value, expected=reference[0])
    _assert_close(actual=carry.marginal_utility, expected=reference[1])
    assert carry.query_data_for_transport() is None
    assert not bool(source_coverage)


class _ReadInputs(TypedDict):
    grid: FloatND
    value: FloatND
    marginal: FloatND
    data: EGMQueryData
    row_queries: FloatND
    row_gradients: FloatND
    passive_values: tuple[ScalarFloat, ...]
    passive_grids: tuple[Float1D, ...]
    pool: EconFunctionKwargs
    resource_gradient: ScalarFloat
    floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus]
    paired_marginal_read: NotRequired[bool]


def _source_law(
    *, savings: FloatND, scale: float, shift: float, period: ScalarInt, age: float
) -> ContinuousState:
    return scale * jnp.log1p(savings) + shift + period / 8.0 + (age - 40.0) / 16.0


def _source_utility(*, consumption: ContinuousAction, shift: float) -> FloatND:
    return jnp.log(consumption) + shift


def _available_at_age(age: float) -> UserFunction:
    slope = 2.0 + 0.25 * (age - 41.0)

    def available_wealth(liquid: ContinuousState) -> FloatND:
        return slope * liquid

    return available_wealth


def _resources(
    *, available_wealth: FloatND, shift: float, period: ScalarInt, age: float
) -> FloatND:
    return available_wealth + shift + period / 8.0 + (age - 40.0) / 16.0


def _lower(*, floor: float, period: ScalarInt, age: float) -> FloatND:
    return floor + period / 32.0 + (age - 40.0) / 32.0


def _model(*, literal: bool, active: bool) -> tuple[Model, UserParams]:
    """Use distinct source and target namespaces with a concave source law."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    actions = {"consumption": IrregSpacedGrid(points=(0.01, 0.05, 0.1, 0.5, 1.0))}
    functions = {
        "savings": _savings,
        "inverse_marginal_utility": _inverse,
    }
    return Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(
                        points=(0.9, 1.0) if active else (0.41, 1.0)
                    )
                },
                actions=actions,
                functions={
                    **functions,
                    "utility": _source_utility,
                    "resources": _parent_resources,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.4)
                },
                state_transitions={"liquid": _source_law},
                solver=EGM(savings_grid=IrregSpacedGrid(points=(0.4, 0.7, 1.0))),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=(0.0, 1.0, 3.0))},
                actions=actions,
                functions={
                    **functions,
                    "utility": _utility,
                    "resources": _resources,
                    "available_wealth": AgeSpecializedFunction(
                        build=_available_at_age, signature=lambda age: age
                    ),
                },
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=margin, lower=0.5 if literal else _lower
                    )
                },
                state_transitions={"liquid": _ordinary_law},
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=(0.5, 2.0, 3.0)),
                    n_constrained_points=4,
                ),
            ),
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=(5.5, 10.0, 13.0))},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={"parent": {"child": 40}, "child": {"done": 41}},
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
        fixed_params={
            "parent": {"next_liquid": {"scale": 4.0}, "utility": {"shift": 0.25}}
        },
        enable_jit=False,
    ), {
        "parent": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "next_liquid": {"shift": -0.3},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "resources": {"shift": 0.8125},
            **({} if literal else {"borrowing_lower": {"floor": 0.4375}}),
        },
        "done": {},
    }


def _capture(
    *,
    model: Model,
    params: UserParams,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_EGMPeriodKernel, OracleContext]:
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=directory,
        monkeypatch=monkeypatch,
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    assert kernel.source_period == 0
    assert kernel.continuation_target == "child"
    return kernel, context


def _core(*, kernel: _EGMPeriodKernel) -> tuple[_EGMCore, _RegimeMassGuardedCore]:
    function = kernel.core_programs()["main"].function
    while isinstance(function, functools.partial):
        function = function.func
    function = getattr(function, "__wrapped__", function)
    assert isinstance(function, _RegimeMassGuardedCore)
    assert isinstance(function.core, _EGMCore)
    return function.core, function


def _run(
    *, kernel: _EGMPeriodKernel, context: OracleContext, compiled: bool
) -> KernelOutput:
    function = kernel.core_programs()["main"].function
    return kernel(
        compiled_cores={"main": jax.jit(function) if compiled else function},
        **context,
        logger=logging.getLogger(__name__),
    )


def _assert_child_bank(*, child: EGMCarry, literal: bool) -> None:
    """Check actual computed child support before using a represented-row oracle."""
    grid = np.asarray(child.endog_grid)
    live = np.isfinite(grid)
    assert live.sum() >= 2
    assert np.all(np.diff(grid[live]) >= 0.0)
    assert np.isfinite(np.asarray(child.value)[live]).all()
    assert np.isfinite(np.asarray(child.marginal_utility)[live]).all()
    data = child.query_data_for_transport()
    if literal:
        assert data is None
    else:
        assert data is not None
        _assert_close(actual=data.floor_used, expected=0.5)
        assert data.corner_continuation is not None
        _assert_close(actual=data.corner_continuation, expected=[math.log(5.5)])
        np.testing.assert_array_equal(data.corner_kind, [QueryNodeKind.FINITE])
        assert data.liquid_corner_data is None


def _child_pair(
    *, child: EGMCarry, literal: bool, resources: float
) -> tuple[float, float]:
    """Use scalar padded LINEAR or the independent owned tuple and corner rule."""
    grid = np.asarray(child.endog_grid)
    values = np.asarray(child.value)
    marginal = np.asarray(child.marginal_utility)
    if literal:
        return (
            interpolate_row(query=resources, xp=grid, fp=values, slopes=None),
            interpolate_row(query=resources, xp=grid, fp=marginal, slopes=None),
        )
    data = child.query_data_for_transport()
    assert data is not None
    assert resources > 0.5
    corner = (math.log(resources - 0.5) + math.log(5.5), 1.0 / (resources - 0.5))
    row = _owned_reference(
        grid=grid,
        values=values,
        marginal=marginal,
        savings=np.asarray(data.savings),
        data=data,
        query=resources,
    )
    if (
        row is not None
        and 0.5 <= row[2] <= 3.0
        and resources > row[2]
        and row[0] >= corner[0]
    ):
        return row[0], row[1]
    return corner


def _parent_reference(
    *, child: EGMCarry, literal: bool, active: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive original Euler actions and rescore only strict interior tuples."""
    saving_nodes = np.asarray([0.4, 0.7, 1.0], dtype=float)
    landing_nodes = 4.0 * np.log1p(saving_nodes) - 0.3
    resource_nodes = 2.0 * landing_nodes + 1.0
    assert np.all((landing_nodes > 0.0) & (landing_nodes < 3.0))
    pairs = np.asarray(
        [
            _child_pair(child=child, literal=literal, resources=float(resources))
            for resources in resource_nodes
        ]
    )
    euler_consumption = (1.0 + saving_nodes) / (8.0 * pairs[:, 1])
    endog = saving_nodes + euler_consumption
    assert np.all(np.diff(endog) > 0.0)
    current_nodes = (0.9, 1.0) if active else (0.41, 1.0)
    value, marginal, consumption, saving = [], [], [], []
    eps = np.finfo(np.asarray(child.value).dtype).eps
    for current in current_nodes:
        if current < endog[0]:
            c, a = current - 0.4, 0.4
            continuation = pairs[0, 0]
        else:
            upper = int(np.searchsorted(endog, current, side="right"))
            assert 0 < upper < len(endog)
            lower = upper - 1
            weight = (current - endog[lower]) / (endog[upper] - endog[lower])
            assert 0.0 < weight < 1.0
            c = euler_consumption[lower] + weight * (
                euler_consumption[upper] - euler_consumption[lower]
            )
            a = saving_nodes[lower] + weight * (
                saving_nodes[upper] - saving_nodes[lower]
            )
            assert abs(a + c - current) < 10.0 * np.finfo(float).eps
            actual_landing = 4.0 * math.log1p(a) - 0.3
            resources = 2.0 * actual_landing + 1.0
            continuation = _child_pair(
                child=child, literal=literal, resources=resources
            )[0]
            wrong_raw = _child_pair(
                child=child, literal=literal, resources=actual_landing
            )[0]
            cached = (1.0 - weight) * pairs[lower, 0] + weight * pairs[upper, 0]
            interpolated_landing = (1.0 - weight) * landing_nodes[lower] + weight * (
                landing_nodes[upper]
            )
            wrong_law = _child_pair(
                child=child, literal=literal, resources=2.0 * interpolated_landing + 1.0
            )[0]
            assert np.isfinite([continuation, wrong_raw, cached, wrong_law]).all()
            for alternative in (wrong_raw, cached, wrong_law):
                assert abs(continuation - alternative) > 200.0 * eps
                tolerance = 300.0 * eps
                assert not np.isclose(
                    math.log(c) + 0.25 + alternative,
                    math.log(c) + 0.25 + continuation,
                    rtol=tolerance,
                    atol=tolerance,
                )
        consumption.append(c)
        saving.append(a)
        value.append(math.log(c) + 0.25 + continuation)
        marginal.append(1.0 / c)
    return (
        np.asarray(value),
        np.asarray(marginal),
        np.asarray(consumption),
        np.asarray(saving),
    )


def _assert_close(
    *, actual: FloatND | np.ndarray, expected: float | list[float] | np.ndarray
) -> None:
    dtype = np.asarray(actual).dtype
    tolerance = 300.0 * np.finfo(dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _assert_step(
    *,
    step: RetiredEGMResult,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    active: bool,
) -> None:
    assert isinstance(step, RetiredEGMResult)
    assert step.savings is not None
    np.testing.assert_array_equal(
        step.rescore_interior, [True, True] if active else [False, True]
    )
    _assert_close(actual=step.consumption, expected=reference[2])
    _assert_close(actual=step.savings, expected=reference[3])
    _assert_close(actual=step.marginal, expected=reference[1])


def _assert_numeric(
    *,
    output: KernelOutput,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _assert_close(actual=output.value, expected=reference[0])
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    _assert_close(actual=carry.marginal_utility, expected=reference[1])
    assert carry.query_data_for_transport() is None


def _assert_status(*, model: Model, output: KernelOutput, literal: bool) -> None:
    if literal:
        assert not isinstance(output, FeasibilityKernelOutput)
        return
    assert isinstance(output, FeasibilityKernelOutput)
    slots = model._regimes["parent"].solution.query_input_authorities
    assert slots is not None
    assert tuple((name, period) for name, period, _ in slots[0]) == (("child", 1),)
    assert output.query_status is not None
    np.testing.assert_array_equal(output.query_status[0], [False, False, False])
    assert len(output.query_status) == 1
    validate_query_input_status(statuses=output.query_status, authorities=slots[0])
    for status in output.bound_status:
        np.testing.assert_array_equal(status, [False, False])
